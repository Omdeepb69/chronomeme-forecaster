import pandas as pd
import numpy as np
import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import logging
import os
from datetime import timedelta, datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Ensure NLTK VADER lexicon is available
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except nltk.downloader.DownloadError:
    logging.info("Downloading NLTK VADER lexicon...")
    nltk.download('vader_lexicon')
except LookupError:
     logging.info("NLTK VADER lexicon lookup failed, attempting download...")
     nltk.download('vader_lexicon')


def generate_mock_data(num_records: int = 1000, start_date_str: str = "2023-01-01", end_date_str: str = "2024-01-01") -> pd.DataFrame:
    """Generates mock meme mention data."""
    logging.info("Generating mock data...")
    start_date = pd.to_datetime(start_date_str)
    end_date = pd.to_datetime(end_date_str)
    date_range = pd.date_range(start=start_date, end=end_date, periods=num_records)

    # Simulate varying mention frequency
    base_freq = np.random.rand(num_records) * 10
    seasonal_component = 5 * (1 + np.sin(np.linspace(0, 4 * np.pi, num_records)))
    noise = np.random.randn(num_records) * 2
    timestamps_indices = (base_freq + seasonal_component + noise).astype(int)
    timestamps_indices = np.clip(timestamps_indices, 0, len(date_range) - 1)
    timestamps = date_range[timestamps_indices]
    timestamps = sorted(timestamps) # Ensure chronological order for realism

    meme_keywords = ["doge", "stonks", "cat meme", "distracted boyfriend", "feels good man", "pepe", "harambe", "success kid"]
    sentiments = ["amazing!", "lol so true", "this is hilarious", "worst meme ever", "meh", "pretty funny", "I don't get it", "peak internet", "so relatable", "cringe"]

    texts = [f"Just saw the {np.random.choice(meme_keywords)} meme, {np.random.choice(sentiments)}" for _ in range(num_records)]

    df = pd.DataFrame({'timestamp': timestamps, 'text': texts})
    logging.info(f"Generated mock data with {len(df)} records.")
    return df

def load_data(file_path: str | None = None) -> pd.DataFrame:
    """Loads data from a CSV file or generates mock data."""
    if file_path and os.path.exists(file_path):
        logging.info(f"Loading data from {file_path}...")
        try:
            df = pd.read_csv(file_path)
            logging.info("Data loaded successfully.")
            return df
        except Exception as e:
            logging.error(f"Error loading data from {file_path}: {e}")
            raise
    else:
        if file_path:
            logging.warning(f"File not found at {file_path}. Generating mock data instead.")
        else:
            logging.info("No file path provided. Generating mock data.")
        return generate_mock_data()

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses the raw data."""
    logging.info("Preprocessing data...")
    if 'timestamp' not in df.columns or 'text' not in df.columns:
        raise ValueError("Input DataFrame must contain 'timestamp' and 'text' columns.")

    df_processed = df.copy()
    df_processed['timestamp'] = pd.to_datetime(df_processed['timestamp'], errors='coerce')
    df_processed = df_processed.dropna(subset=['timestamp', 'text'])
    df_processed['text'] = df_processed['text'].astype(str)
    df_processed = df_processed.sort_values(by='timestamp').reset_index(drop=True)
    logging.info("Data preprocessing complete.")
    return df_processed

def analyze_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """Applies VADER sentiment analysis."""
    logging.info("Analyzing sentiment...")
    analyzer = SentimentIntensityAnalyzer()
    df_sentiment = df.copy()

    def get_sentiment_score(text):
        try:
            return analyzer.polarity_scores(text)['compound']
        except Exception as e:
            logging.warning(f"Could not analyze sentiment for text: '{text[:50]}...'. Error: {e}")
            return 0.0 # Return neutral score on error

    df_sentiment['sentiment_score'] = df_sentiment['text'].apply(get_sentiment_score)
    logging.info("Sentiment analysis complete.")
    return df_sentiment

def aggregate_data(df: pd.DataFrame, freq: str = 'D') -> pd.DataFrame:
    """Aggregates data by time frequency."""
    logging.info(f"Aggregating data with frequency '{freq}'...")
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
         raise TypeError("Timestamp column must be of datetime type for aggregation.")

    df_agg = df.set_index('timestamp')
    aggregated = df_agg.resample(freq).agg(
        mention_count=('text', 'count'),
        avg_sentiment_score=('sentiment_score', 'mean')
    )
    aggregated['avg_sentiment_score'] = aggregated['avg_sentiment_score'].fillna(0) # Fill days with no mentions

    # Create a 'trend_score' combining count and sentiment
    # Normalize sentiment to be non-negative for multiplication
    aggregated['normalized_sentiment'] = (aggregated['avg_sentiment_score'] + 1) / 2
    aggregated['trend_score'] = aggregated['mention_count'] * aggregated['normalized_sentiment']

    aggregated = aggregated.reset_index()
    aggregated.rename(columns={'timestamp': 'ds'}, inplace=True) # Prophet requires 'ds' column
    logging.info("Data aggregation complete.")
    return aggregated

def split_data(df_agg: pd.DataFrame, test_size: float = 0.2):
    """Splits aggregated time series data chronologically."""
    logging.info(f"Splitting data with test size {test_size}...")
    if not isinstance(df_agg, pd.DataFrame) or 'ds' not in df_agg.columns:
        raise ValueError("Input must be a DataFrame with a 'ds' column.")
    if not pd.api.types.is_datetime64_any_dtype(df_agg['ds']):
        raise TypeError("'ds' column must be of datetime type.")

    df_agg_sorted = df_agg.sort_values(by='ds')
    split_index = int(len(df_agg_sorted) * (1 - test_size))
    df_train = df_agg_sorted.iloc[:split_index]
    df_test = df_agg_sorted.iloc[split_index:]

    logging.info(f"Train set size: {len(df_train)}, Test set size: {len(df_test)}")
    return df_train, df_test

def train_prophet_model(df_train: pd.DataFrame, target_column: str = 'mention_count') -> Prophet:
    """Trains a Prophet model."""
    logging.info(f"Training Prophet model on target '{target_column}'...")
    if target_column not in df_train.columns:
        raise ValueError(f"Target column '{target_column}' not found in training data.")

    prophet_df = df_train[['ds', target_column]].rename(columns={target_column: 'y'})

    # Handle potential zero/negative values if using log transform later
    if (prophet_df['y'] <= 0).any():
        logging.warning(f"Target column '{target_column}' contains non-positive values. Prophet works best with positive data.")
        # Optional: Add a small constant if planning log transform, or handle differently
        # prophet_df['y'] = prophet_df['y'] + 1e-6

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False, # Usually less relevant for daily aggregated data
        changepoint_prior_scale=0.05, # Default is 0.05
        seasonality_prior_scale=10.0 # Default is 10.0
    )
    # Add country holidays if relevant, e.g., model.add_country_holidays(country_name='US')

    model.fit(prophet_df)
    logging.info("Prophet model training complete.")
    return model

def make_forecast(model: Prophet, periods: int = 30, freq: str = 'D') -> pd.DataFrame:
    """Generates future forecast."""
    logging.info(f"Generating forecast for {periods} periods with frequency '{freq}'...")
    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)
    logging.info("Forecast generation complete.")
    return forecast

def evaluate_model(model: Prophet, df_test: pd.DataFrame, target_column: str = 'mention_count'):
    """Evaluates the model on the test set."""
    logging.info("Evaluating model on test set...")
    if target_column not in df_test.columns:
        raise ValueError(f"Target column '{target_column}' not found in test data.")

    test_prophet_df = df_test[['ds', target_column]].rename(columns={target_column: 'y'})
    forecast = model.predict(test_prophet_df[['ds']])

    # Align forecast with actuals
    comparison_df = pd.merge(test_prophet_df, forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], on='ds')

    mae = mean_absolute_error(comparison_df['y'], comparison_df['yhat'])
    rmse = np.sqrt(mean_squared_error(comparison_df['y'], comparison_df['yhat']))

    logging.info(f"Test Set Evaluation (Target: {target_column}):")
    logging.info(f"  MAE: {mae:.4f}")
    logging.info(f"  RMSE: {rmse:.4f}")

    return mae, rmse, comparison_df


def plot_forecast(model: Prophet, forecast: pd.DataFrame, historical_data: pd.DataFrame | None = None, target_column: str = 'mention_count', title: str = 'Meme Mention Forecast'):
    """Visualizes the forecast."""
    logging.info("Plotting forecast...")
    fig1 = model.plot(forecast)
    if historical_data is not None and target_column in historical_data.columns:
        plt.scatter(historical_data['ds'].dt.to_pydatetime(), historical_data[target_column], color='r', s=10, label='Actual Historical')
        plt.legend()
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel(target_column.replace('_', ' ').title())
    plt.tight_layout()
    plt.show()

    try:
        fig2 = model.plot_components(forecast)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        logging.warning(f"Could not plot Prophet components: {e}")

    logging.info("Plotting complete.")


def predict_peak_virality(forecast: pd.DataFrame, historical_end_date: pd.Timestamp, window_days: int = 7) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    """Predicts the peak virality window based on forecast."""
    logging.info("Predicting peak virality window...")
    future_forecast = forecast[forecast['ds'] > historical_end_date].copy()

    if future_forecast.empty:
        logging.warning("No future forecast data available to predict peak.")
        return None, None

    peak_index = future_forecast['yhat'].idxmax()
    peak_date = future_forecast.loc[peak_index, 'ds']
    peak_value = future_forecast.loc[peak_index, 'yhat']

    window_start = peak_date - timedelta(days=window_days // 2)
    window_end = peak_date + timedelta(days=window_days // 2)

    logging.info(f"Predicted peak date: {peak_date.strftime('%Y-%m-%d')} with forecasted value {peak_value:.2f}")
    logging.info(f"Predicted peak virality window ({window_days} days): {window_start.strftime('%Y-%m-%d')} to {window_end.strftime('%Y-%m-%d')}")

    return window_start, window_end


def run_pipeline(file_path: str | None = None,
                 target_column: str = 'mention_count',
                 forecast_periods: int = 90,
                 aggregation_freq: str = 'D',
                 test_split_size: float = 0.15,
                 peak_window_days: int = 14):
    """Runs the complete data loading, processing, forecasting, and visualization pipeline."""

    # 1. Load Data
    raw_df = load_data(file_path)

    # 2. Preprocess Data
    processed_df = preprocess_data(raw_df)

    # 3. Analyze Sentiment
    sentiment_df = analyze_sentiment(processed_df)

    # 4. Aggregate Data
    aggregated_df = aggregate_data(sentiment_df, freq=aggregation_freq)

    # Check if enough data for splitting and training
    if len(aggregated_df) < 20: # Need sufficient data points for Prophet
        logging.error(f"Insufficient aggregated data points ({len(aggregated_df)}) after processing. Cannot proceed with modeling.")
        return

    # 5. Split Data
    df_train, df_test = split_data(aggregated_df, test_size=test_split_size)

    if df_train.empty or len(df_train) < 2: # Prophet requires at least 2 data points
        logging.error(f"Insufficient training data points ({len(df_train)}) after split. Cannot train model.")
        return

    # 6. Train Model
    model = train_prophet_model(df_train, target_column=target_column)

    # 7. Make Forecast
    forecast = make_forecast(model, periods=forecast_periods, freq=aggregation_freq)

    # 8. Evaluate Model (Optional but recommended)
    if not df_test.empty:
        evaluate_model(model, df_test, target_column=target_column)
    else:
        logging.warning("Test set is empty, skipping evaluation.")

    # 9. Visualize Forecast
    plot_forecast(model, forecast, historical_data=aggregated_df, target_column=target_column, title=f'Forecast for {target_column}')

    # 10. Predict Peak Virality
    historical_end_date = df_train['ds'].max()
    peak_start, peak_end = predict_peak_virality(forecast, historical_end_date, window_days=peak_window_days)

    logging.info("ChronoMeme Forecaster pipeline finished.")


if __name__ == "__main__":
    # Example usage:
    # To use a CSV file (ensure it has 'timestamp' and 'text' columns):
    # run_pipeline(file_path='path/to/your/meme_data.csv')

    # To use generated mock data:
    run_pipeline(file_path=None,
                 target_column='mention_count', # Can also be 'trend_score' or 'avg_sentiment_score'
                 forecast_periods=90,
                 aggregation_freq='D',
                 test_split_size=0.15,
                 peak_window_days=14)