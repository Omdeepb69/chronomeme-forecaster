import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, Tuple, Dict, Any

def plot_historical_data(
    df: pd.DataFrame,
    date_col: str = 'timestamp',
    mention_col: str = 'mention_count',
    sentiment_col: Optional[str] = 'sentiment_score',
    title: str = 'Historical Meme Mentions and Sentiment'
) -> go.Figure:
    """
    Visualizes historical mention counts and optionally sentiment scores over time.
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add Mention Count Trace
    fig.add_trace(
        go.Scatter(
            x=df[date_col],
            y=df[mention_col],
            mode='lines+markers',
            name='Mention Count',
            marker=dict(size=4),
            line=dict(width=2)
        ),
        secondary_y=False,
    )

    fig.update_yaxes(title_text="Mention Count", secondary_y=False)

    # Add Sentiment Score Trace if available
    if sentiment_col and sentiment_col in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df[date_col],
                y=df[sentiment_col],
                mode='lines',
                name='Sentiment Score',
                line=dict(width=2, dash='dot', color='rgba(255, 127, 14, 0.8)'),
                yaxis='y2'
            ),
            secondary_y=True,
        )
        fig.update_yaxes(title_text="Sentiment Score", secondary_y=True, range=[-1.1, 1.1])
    else:
        fig.update_layout(yaxis2=dict(showgrid=False, zeroline=False, showticklabels=False))


    fig.update_layout(
        title=title,
        xaxis_title="Date",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    fig.update_xaxes(rangeslider_visible=True)

    return fig


def plot_forecast(
    history_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    date_col: str = 'timestamp',
    history_val_col: str = 'mention_count',
    forecast_val_col: str = 'forecast',
    lower_bound_col: Optional[str] = 'forecast_lower',
    upper_bound_col: Optional[str] = 'forecast_upper',
    title: str = 'Meme Mention Forecast'
) -> go.Figure:
    """
    Visualizes historical data along with the forecast and confidence intervals.
    """
    fig = go.Figure()

    # Add Historical Data
    fig.add_trace(
        go.Scatter(
            x=history_df[date_col],
            y=history_df[history_val_col],
            mode='lines',
            name='Historical Mentions',
            line=dict(color='rgb(31, 119, 180)')
        )
    )

    # Add Forecast Data
    fig.add_trace(
        go.Scatter(
            x=forecast_df[date_col],
            y=forecast_df[forecast_val_col],
            mode='lines',
            name='Forecast',
            line=dict(color='rgb(255, 127, 14)', dash='dash')
        )
    )

    # Add Confidence Intervals if available
    if lower_bound_col and upper_bound_col and \
       lower_bound_col in forecast_df.columns and \
       upper_bound_col in forecast_df.columns:

        fig.add_trace(
            go.Scatter(
                x=forecast_df[date_col].tolist() + forecast_df[date_col].tolist()[::-1], # x, then x reversed
                y=forecast_df[upper_bound_col].tolist() + forecast_df[lower_bound_col].tolist()[::-1], # upper, then lower reversed
                fill='toself',
                fillcolor='rgba(255, 127, 14, 0.2)',
                line=dict(color='rgba(255, 255, 255, 0)'),
                hoverinfo="skip",
                showlegend=True,
                name='Confidence Interval'
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Mention Count / Trend Score",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    fig.update_xaxes(rangeslider_visible=True)

    return fig


def calculate_peak_virality_window(
    forecast_df: pd.DataFrame,
    date_col: str = 'timestamp',
    forecast_val_col: str = 'forecast',
    window_days: int = 3
) -> Optional[Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Calculates a simple peak virality window around the max forecast point.
    Returns (start_date, end_date) or None if forecast is empty.
    """
    if forecast_df.empty or forecast_val_col not in forecast_df.columns:
        return None

    peak_index = forecast_df[forecast_val_col].idxmax()
    peak_date = forecast_df.loc[peak_index, date_col]

    half_window = pd.Timedelta(days=window_days / 2)
    start_date = peak_date - half_window
    end_date = peak_date + half_window

    # Ensure window stays within forecast range if needed, or allow extrapolation
    # For simplicity, we just calculate around the peak date.
    # Clamping to actual forecast range:
    # start_date = max(start_date, forecast_df[date_col].min())
    # end_date = min(end_date, forecast_df[date_col].max())

    return start_date, end_date


def add_peak_virality_window_to_plot(
    fig: go.Figure,
    peak_window: Optional[Tuple[pd.Timestamp, pd.Timestamp]],
    fillcolor: str = 'rgba(0, 200, 83, 0.2)',
    linecolor: str = 'rgba(0, 200, 83, 1.0)',
    linewidth: int = 1
) -> go.Figure:
    """
    Adds a shaded region to an existing Plotly figure to indicate the peak virality window.
    """
    if peak_window:
        start_date, end_date = peak_window
        fig.add_vrect(
            x0=start_date, x1=end_date,
            fillcolor=fillcolor,
            opacity=0.5,
            layer="below",
            line_width=linewidth,
            line_color=linecolor,
            annotation_text="Predicted Peak",
            annotation_position="top left",
            annotation=dict(font_size=10, font_color=linecolor)
        )
    return fig


def plot_forecast_with_peak(
    history_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    date_col: str = 'timestamp',
    history_val_col: str = 'mention_count',
    forecast_val_col: str = 'forecast',
    lower_bound_col: Optional[str] = 'forecast_lower',
    upper_bound_col: Optional[str] = 'forecast_upper',
    peak_window_days: int = 5,
    title: str = 'Meme Mention Forecast with Predicted Peak Virality'
) -> go.Figure:
    """
    Combines forecast plotting and peak virality window visualization.
    """
    fig = plot_forecast(
        history_df=history_df,
        forecast_df=forecast_df,
        date_col=date_col,
        history_val_col=history_val_col,
        forecast_val_col=forecast_val_col,
        lower_bound_col=lower_bound_col,
        upper_bound_col=upper_bound_col,
        title=title # Set initial title
    )

    peak_window = calculate_peak_virality_window(
        forecast_df=forecast_df,
        date_col=date_col,
        forecast_val_col=forecast_val_col,
        window_days=peak_window_days
    )

    fig = add_peak_virality_window_to_plot(fig, peak_window)

    # Update title again if needed, or keep the one from plot_forecast
    fig.update_layout(title=title)

    return fig


def plot_model_performance(
    actual_values: pd.Series,
    predicted_values: pd.Series,
    residuals: Optional[pd.Series] = None,
    title: str = 'Model Performance Evaluation'
) -> go.Figure:
    """
    Visualizes model performance: Actual vs. Predicted and optionally Residuals.
    Assumes actual_values and predicted_values share the same index (time).
    """
    if residuals is None:
        residuals = actual_values - predicted_values

    fig = make_subplots(rows=2, cols=1,
                        shared_xaxes=True,
                        subplot_titles=('Actual vs. Predicted', 'Residuals Over Time'),
                        vertical_spacing=0.1)

    common_index = actual_values.index.intersection(predicted_values.index)
    if not isinstance(common_index, pd.DatetimeIndex):
         try:
             common_index = pd.to_datetime(common_index)
         except Exception:
             # Use numerical index if conversion fails
             common_index = np.arange(len(common_index))


    # Actual vs Predicted Plot
    fig.add_trace(
        go.Scatter(x=common_index, y=actual_values.loc[common_index], mode='lines', name='Actual',
                   line=dict(color='rgb(31, 119, 180)')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=common_index, y=predicted_values.loc[common_index], mode='lines', name='Predicted',
                   line=dict(color='rgb(255, 127, 14)', dash='dash')),
        row=1, col=1
    )

    # Residuals Plot
    fig.add_trace(
        go.Scatter(x=common_index, y=residuals.loc[common_index], mode='markers', name='Residuals',
                   marker=dict(color='rgba(44, 160, 44, 0.7)', size=5)),
        row=2, col=1
    )
    # Add zero line for residuals
    fig.add_hline(y=0, line_dash="dot", line_color="grey", row=2, col=1)


    fig.update_layout(
        title=title,
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=40, r=20, t=80, b=40) # Increased top margin for subplot titles
    )

    fig.update_yaxes(title_text="Value", row=1, col=1)
    fig.update_yaxes(title_text="Residual", row=2, col=1)
    fig.update_xaxes(title_text="Time / Index", row=2, col=1)

    return fig


def _create_mock_data(days=90, forecast_days=30) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Helper to create mock data for demonstration."""
    base_date = pd.to_datetime('2023-01-01')
    dates = pd.date_range(start=base_date, periods=days, freq='D')
    
    # Simulate historical data
    mention_counts = (
        np.sin(np.linspace(0, 3 * np.pi, days)) * 50 +
        np.random.normal(0, 15, days) +
        np.linspace(10, 50, days) # Upward trend
    )
    mention_counts = np.maximum(0, mention_counts).astype(int) # Ensure non-negative

    sentiment_scores = (
        np.cos(np.linspace(0, 2 * np.pi, days)) * 0.3 +
        np.random.normal(0, 0.1, days) +
        0.1 # Slightly positive bias
    )
    sentiment_scores = np.clip(sentiment_scores, -1, 1)

    history_df = pd.DataFrame({
        'timestamp': dates,
        'mention_count': mention_counts,
        'sentiment_score': sentiment_scores
    })

    # Simulate forecast data
    forecast_start_date = dates[-1] + pd.Timedelta(days=1)
    forecast_dates = pd.date_range(start=forecast_start_date, periods=forecast_days, freq='D')

    # Continue the sine wave and trend loosely
    last_hist_val = mention_counts[-1]
    forecast_values = (
        np.sin(np.linspace(3 * np.pi, 4 * np.pi, forecast_days)) * 40 +
        np.random.normal(0, 20, forecast_days) +
        np.linspace(last_hist_val, last_hist_val + 10, forecast_days) # Continue trend slightly
    )
    forecast_values = np.maximum(0, forecast_values)

    forecast_std_dev = np.linspace(15, 30, forecast_days) # Increasing uncertainty
    forecast_lower = np.maximum(0, forecast_values - 1.96 * forecast_std_dev)
    forecast_upper = forecast_values + 1.96 * forecast_std_dev

    forecast_df = pd.DataFrame({
        'timestamp': forecast_dates,
        'forecast': forecast_values,
        'forecast_lower': forecast_lower,
        'forecast_upper': forecast_upper
    })

    return history_df, forecast_df


if __name__ == '__main__':
    print("Generating mock data and running visualization examples...")

    # Create mock data
    hist_data, fc_data = _create_mock_data(days=120, forecast_days=45)

    # Example 1: Plot historical data
    print("Displaying historical data plot...")
    fig_hist = plot_historical_data(hist_data)
    # In a real application, you might use fig_hist.show() or save it
    # fig_hist.show() # Uncomment to display locally if running interactively

    # Example 2: Plot forecast with peak virality
    print("Displaying forecast plot with peak virality...")
    fig_fc_peak = plot_forecast_with_peak(
        history_df=hist_data,
        forecast_df=fc_data,
        peak_window_days=7
    )
    # fig_fc_peak.show() # Uncomment to display locally

    # Example 3: Plot model performance (using last 30 days of history as 'actual')
    print("Displaying model performance plot...")
    # Simulate some 'actual' values that correspond to a forecast period
    # For this demo, let's pretend the first 15 days of the forecast had actuals
    actual_test_period = hist_data['mention_count'].iloc[-15:]
    predicted_test_period = fc_data['forecast'].iloc[:15] # Match the length
    
    # Ensure indices match for plotting - use timestamp if possible, else reset
    actual_test_period.index = fc_data['timestamp'].iloc[:15]
    predicted_test_period.index = fc_data['timestamp'].iloc[:15]


    if not actual_test_period.empty and not predicted_test_period.empty:
         fig_perf = plot_model_performance(
             actual_values=actual_test_period,
             predicted_values=predicted_test_period
         )
         # fig_perf.show() # Uncomment to display locally
    else:
         print("Skipping performance plot due to insufficient overlapping data.")


    print("Visualization examples generated (plots not shown automatically in script).")
    print("If running interactively, uncomment the '.show()' lines to view plots.")