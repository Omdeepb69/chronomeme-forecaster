import pandas as pd
import numpy as np
import argparse
import os
import sys
import logging
from datetime import datetime, timedelta
import snscrape.modules.twitter as sntwitter

# Suppress Prophet logs and warnings
import prophet.utilities as prophet_utils
prophet_utils.logger.setLevel('ERROR')
import warnings
warnings.filterwarnings('ignore', category=FutureWarning) # Suppress pandas future warnings
warnings.filterwarnings('ignore', message='The behavior of DatetimeProperties.to_pydatetime is deprecated') # Suppress specific prophet/pandas warning

# Import core libraries after suppressing warnings
from prophet import Prophet
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
sns.set_theme(style="darkgrid")

# --- NLTK Data Download ---
def download_nltk_data():
    """Downloads the VADER lexicon if not already present."""
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
        logging.debug("VADER lexicon found.")
    except nltk.downloader.DownloadError as e:
        logging.error(f"NLTK Download Error: {e}. VADER lexicon might be missing.")
        print("Error: VADER lexicon not found. Please ensure NLTK data is correctly installed.", file=sys.stderr)
        sys.exit(1)
    except LookupError:
        logging.info("VADER lexicon not found. Downloading...")
        try:
            nltk.download('vader_lexicon', quiet=True)
            logging.info("VADER lexicon downloaded successfully.")
        except Exception as e:
            logging.error(f"Failed to download VADER lexicon: {e}")
            print(f"Error: Could not download VADER lexicon. Please check your internet connection or install manually. Details: {e}", file=sys.stderr)
            sys.exit(1)

# --- Twitter Data Scraping ---
def fetch_tweets(keyword, max_results=500):
    """Scrapes tweets for a given keyword using snscrape."""
    logging.info(f"Scraping tweets for keyword: {keyword} (max: {max_results})")
    tweets = []
    try:
        for i, tweet in enumerate(sntwitter.TwitterSearchScraper(f'{keyword} lang:en').get_items()):
            if i >= max_results:
                break
            tweets.append([tweet.date, tweet.content, keyword])
            if (i + 1) % 100 == 0:
                logging.info(f"Scraped {i + 1} tweets so far...")
        
        # Create DataFrame with needed columns
        df = pd.DataFrame(tweets, columns=['timestamp', 'text', 'meme_name'])
        logging.info(f"Successfully scraped {len(df)} tweets for '{keyword}'")
        return df
    except Exception as e:
        logging.error(f"Error scraping tweets: {e}")
        raise ValueError(f"Failed to scrape tweets: {e}")

# --- Data Ingestion ---
def load_data(file_path: str, meme_name: str) -> pd.DataFrame:
    """Loads and preprocesses data from a CSV file."""
    logging.info(f"Loading data for meme '{meme_name}' from {file_path}...")
    if not os.path.exists(file_path):
        logging.error(f"Input file not found: {file_path}")
        raise FileNotFoundError(f"Input file not found: {file_path}")

    try:
        df = pd.read_csv(file_path)
        logging.info(f"Loaded {len(df)} total records.")

        required_columns = ['timestamp', 'text', 'meme_name']
        if not all(col in df.columns for col in required_columns):
            missing = [col for col in required_columns if col not in df.columns]
            logging.error(f"CSV missing required columns: {missing}")
            raise ValueError(f"Input CSV must contain columns: {required_columns}")

        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df.dropna(subset=['timestamp', 'text', 'meme_name'], inplace=True) # Drop rows with critical missing data

        # Filter by meme name (case-insensitive)
        df_filtered = df[df['meme_name'].str.lower() == meme_name.lower()].copy()

        if df_filtered.empty:
            logging.warning(f"No data found for meme '{meme_name}' in the provided file.")
            # Return empty dataframe with correct columns to avoid downstream errors
            return pd.DataFrame(columns=['timestamp', 'text', 'meme_name'])

        logging.info(f"Found {len(df_filtered)} records for meme '{meme_name}'.")
        df_filtered.sort_values('timestamp', inplace=True)
        return df_filtered

    except pd.errors.ParserError as e:
        logging.error(f"Error parsing CSV file {file_path}: {e}")
        raise ValueError(f"Could not parse CSV file: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred during data loading: {e}")
        raise

# --- Sentiment Analysis ---
def analyze_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """Performs sentiment analysis and aggregates data daily."""
    logging.info("Performing sentiment analysis...")
    if df.empty or 'text' not in df.columns:
        logging.warning("Input DataFrame for sentiment analysis is empty or missing 'text' column.")
        # Return structure expected by downstream functions
        return pd.DataFrame({
            'ds': pd.Series(dtype='datetime64[ns]'),
            'mentions': pd.Series(dtype='int'),
            'avg_sentiment': pd.Series(dtype='float'),
            'trend_score': pd.Series(dtype='float')
        })

    try:
        analyzer = SentimentIntensityAnalyzer()
        df['sentiment_score'] = df['text'].apply(lambda text: analyzer.polarity_scores(str(text))['compound'])
        logging.info("Sentiment scores calculated.")

        # Aggregate daily
        df['date'] = df['timestamp'].dt.date
        daily_summary = df.groupby('date').agg(
            mentions=('text', 'count'),
            avg_sentiment=('sentiment_score', 'mean')
        ).reset_index()

        # Calculate trend score: mentions weighted by sentiment (scaled 0 to 2)
        daily_summary['trend_score'] = daily_summary['mentions'] * (1 + daily_summary['avg_sentiment'])
        daily_summary['trend_score'] = daily_summary['trend_score'].apply(lambda x: max(0, x)) # Ensure non-negative score

        # Prepare for Prophet: rename columns
        daily_summary.rename(columns={'date': 'ds', 'trend_score': 'y'}, inplace=True)
        daily_summary['ds'] = pd.to_datetime(daily_summary['ds']) # Ensure ds is datetime

        logging.info("Daily aggregation and trend score calculation complete.")
        return daily_summary[['ds', 'mentions', 'avg_sentiment', 'y']]

    except Exception as e:
        logging.error(f"An error occurred during sentiment analysis: {e}")
        raise

# --- Time Series Forecasting ---
def forecast_trend(daily_data: pd.DataFrame, periods: int) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Forecasts future trend score using Prophet."""
    logging.info(f"Starting trend forecasting for {periods} days...")

    if daily_data.empty or 'ds' not in daily_data.columns or 'y' not in daily_data.columns:
        logging.warning("Input DataFrame for forecasting is empty or missing 'ds'/'y' columns.")
        return None, None
    if len(daily_data) < 2:
        logging.warning("Not enough data points (< 2) to perform forecasting.")
        return None, None

    try:
        # Ensure ds is datetime and y is numeric
        daily_data['ds'] = pd.to_datetime(daily_data['ds'])
        daily_data['y'] = pd.to_numeric(daily_data['y'], errors='coerce')
        daily_data = daily_data.dropna(subset=['ds', 'y'])

        if len(daily_data) < 2:
             logging.warning("Not enough valid data points after cleaning for forecasting.")
             return None, None

        model = Prophet(
            yearly_seasonality=False,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.05 # Default, adjust if needed
        )
        model.fit(daily_data[['ds', 'y']])

        future = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)

        logging.info("Forecasting complete.")
        # Return only relevant columns
        forecast_relevant = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
        return model, forecast_relevant

    except Exception as e:
        logging.error(f"An error occurred during forecasting: {e}")
        # Don't raise here, allow visualization of historical data if possible
        return None, None

# --- Peak Virality Calculation ---
def predict_peak_virality(forecast: pd.DataFrame, historical_end_date: pd.Timestamp) -> tuple[pd.Timestamp | None, float | None]:
    """Estimates the peak virality window based on the forecast."""
    logging.info("Calculating predicted peak virality...")
    if forecast is None or forecast.empty:
        logging.warning("Forecast data is empty, cannot predict peak virality.")
        return None, None

    try:
        # Consider only future predictions
        future_forecast = forecast[forecast['ds'] > historical_end_date].copy()

        if future_forecast.empty:
            logging.info("No future forecast data available to predict peak.")
            return None, None

        # Find the date with the maximum predicted trend score (yhat)
        peak_row = future_forecast.loc[future_forecast['yhat'].idxmax()]
        peak_date = peak_row['ds']
        peak_score = peak_row['yhat']

        logging.info(f"Predicted peak virality date: {peak_date.strftime('%Y-%m-%d')} with score: {peak_score:.2f}")
        return peak_date, peak_score

    except Exception as e:
        logging.error(f"An error occurred during peak virality calculation: {e}")
        return None, None

# --- Visualization ---
def visualize_results(
    meme_name: str,
    historical_data: pd.DataFrame,
    forecast_data: pd.DataFrame | None,
    peak_date: pd.Timestamp | None,
    output_dir: str
):
    """Generates and saves plots for historical data, sentiment, and forecast."""
    logging.info(f"Generating visualizations in directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    fig_hist_path = os.path.join(output_dir, f"{meme_name}_historical_trend.png")
    fig_sentiment_path = os.path.join(output_dir, f"{meme_name}_sentiment_over_time.png")
    fig_forecast_path = os.path.join(output_dir, f"{meme_name}_forecast.png")

    try:
        # Plot 1: Historical Mentions and Trend Score
        plt.figure(figsize=(12, 6))
        ax1 = sns.lineplot(data=historical_data, x='ds', y='mentions', label='Daily Mentions', color='skyblue')
        ax1.set_ylabel('Daily Mentions')
        ax2 = ax1.twinx()
        sns.lineplot(data=historical_data, x='ds', y='y', label='Trend Score (y)', color='darkorange', ax=ax2)
        ax2.set_ylabel('Trend Score')
        plt.title(f"Historical Mentions and Trend Score for '{meme_name}'")
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')
        plt.xlabel("Date")
        plt.tight_layout()
        plt.savefig(fig_hist_path)
        plt.close()
        logging.info(f"Saved historical trend plot to {fig_hist_path}")

        # Plot 2: Sentiment Over Time
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=historical_data, x='ds', y='avg_sentiment', label='Average Daily Sentiment', color='lightgreen')
        plt.axhline(0, color='grey', linestyle='--', linewidth=0.8) # Add zero line
        plt.title(f"Average Daily Sentiment for '{meme_name}'")
        plt.xlabel("Date")
        plt.ylabel("Average Sentiment Score (VADER Compound)")
        plt.ylim(-1.1, 1.1) # VADER range is -1 to 1
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_sentiment_path)
        plt.close()
        logging.info(f"Saved sentiment plot to {fig_sentiment_path}")

        # Plot 3: Forecast
        if forecast_data is not None and not forecast_data.empty:
            plt.figure(figsize=(12, 6))
            # Plot historical points
            plt.scatter(historical_data['ds'], historical_data['y'], color='black', label='Historical Trend Score', s=10, alpha=0.7)
            # Plot forecast line
            sns.lineplot(data=forecast_data, x='ds', y='yhat', label='Forecasted Trend Score', color='red')
            # Plot uncertainty interval
            plt.fill_between(forecast_data['ds'], forecast_data['yhat_lower'], forecast_data['yhat_upper'], color='red', alpha=0.2, label='Uncertainty Interval')

            # Highlight predicted peak
            if peak_date:
                plt.axvline(peak_date, color='purple', linestyle='--', label=f'Predicted Peak ({peak_date.strftime("%Y-%m-%d")})')

            plt.title(f"Trend Score Forecast for '{meme_name}'")
            plt.xlabel("Date")
            plt.ylabel("Trend Score")
            plt.legend()
            plt.tight_layout()
            plt.savefig(fig_forecast_path)
            plt.close()
            logging.info(f"Saved forecast plot to {fig_forecast_path}")
        else:
            logging.warning("Skipping forecast plot as forecast data is unavailable.")

    except Exception as e:
        logging.error(f"An error occurred during visualization: {e}")
        # Don't crash the whole program, just log the error

# --- Data Export ---
def export_to_csv(df: pd.DataFrame, meme_name: str, output_dir: str):
    """Exports scraped and analyzed data to CSV."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{meme_name}_data.csv")
    df.to_csv(output_path, index=False)
    logging.info(f"Data exported to {output_path}")
    return output_path

# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(description="ChronoMeme Forecaster: Predict meme virality trends.")
    parser.add_argument("meme_name", help="The name of the meme/keyword to analyze (case-insensitive).")
    parser.add_argument("-i", "--input-file", help="Path to existing input CSV data file (columns: timestamp, text, meme_name).")
    parser.add_argument("-s", "--scrape", action="store_true", help="Scrape Twitter data instead of using input file.")
    parser.add_argument("-m", "--max-tweets", type=int, default=500, help="Maximum number of tweets to scrape (only used with --scrape).")
    parser.add_argument("-o", "--output-dir", default="meme_forecast_output", help="Directory to save plots and results.")
    parser.add_argument("-p", "--periods", type=int, default=30, help="Number of days to forecast into the future.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging.")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # 0. Setup (NLTK data)
        download_nltk_data()

        # 1. Data Acquisition
        if args.scrape:
            logging.info(f"Scraping mode activated for meme/keyword: {args.meme_name}")
            raw_data = fetch_tweets(args.meme_name, args.max_tweets)
            if raw_data.empty:
                print(f"No tweets found for '{args.meme_name}'. Exiting.", file=sys.stderr)
                sys.exit(0)
            
            # Export scraped data to CSV
            csv_path = export_to_csv(raw_data, args.meme_name, args.output_dir)
            print(f"Scraped data saved to: {csv_path}")
        elif args.input_file:
            # 1. Data Ingestion from file
            raw_data = load_data(args.input_file, args.meme_name)
            if raw_data.empty:
                print(f"No data found for meme '{args.meme_name}'. Exiting.", file=sys.stderr)
                sys.exit(0) # Not an error, just no data
        else:
            print("Error: Either --input-file or --scrape must be specified.", file=sys.stderr)
            sys.exit(1)

        # 2. Sentiment Analysis & Daily Aggregation
        daily_summary_data = analyze_sentiment(raw_data)
        if daily_summary_data.empty:
            print(f"Could not process data for sentiment analysis for meme '{args.meme_name}'. Exiting.", file=sys.stderr)
            sys.exit(1)

        historical_end_date = daily_summary_data['ds'].max()
        logging.info(f"Historical data available up to: {historical_end_date.strftime('%Y-%m-%d')}")

        # 3. Time Series Forecasting
        model, forecast_data = forecast_trend(daily_summary_data, args.periods)

        # 4. Peak Virality Prediction
        peak_date, peak_score = None, None
        if forecast_data is not None:
            peak_date, peak_score = predict_peak_virality(forecast_data, historical_end_date)
        else:
             logging.warning("Forecasting failed or produced no results. Skipping peak prediction.")

        # 5. Visualization
        visualize_results(args.meme_name, daily_summary_data, forecast_data, peak_date, args.output_dir)

        # 6. Output Summary
        print("\n--- ChronoMeme Forecaster Summary ---")
        print(f"Meme/Keyword Analyzed: {args.meme_name}")
        if args.scrape:
            print(f"Data Source: Twitter (scraped {len(raw_data)} tweets)")
        else:
            print(f"Data Source: {args.input_file}")
        print(f"Historical Data Range: {daily_summary_data['ds'].min().strftime('%Y-%m-%d')} to {historical_end_date.strftime('%Y-%m-%d')}")
        print(f"Forecast Period: {args.periods} days")
        if forecast_data is not None:
             print(f"Forecast generated successfully.")
             if peak_date and peak_score is not None:
                 print(f"Predicted Peak Virality Date: {peak_date.strftime('%Y-%m-%d')}")
                 print(f"Predicted Peak Trend Score: {peak_score:.2f}")
             else:
                 print("Could not determine a clear peak within the forecast period.")
        else:
            print("Forecasting could not be completed.")
        print(f"Visualizations saved to: {args.output_dir}")
        print("-------------------------------------\n")

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Data Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        logging.exception("An unexpected error occurred in the main execution flow.") # Log full traceback
        print(f"An unexpected error occurred: {e}. Check logs for details.", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    # Example Usage with existing CSV file:
    # python meme_virality_forecaster.py Doge -i mock_meme_data.csv -o doge_output -p 14 -v
    
    # Example Usage with Twitter scraping:
    # python meme_virality_forecaster.py "cat meme" -s -m 1000 -o cat_meme_output -p 30 -v
    
    # Check if running as main script or imported
    if len(sys.argv) > 1: # Basic check if arguments were passed
         main()
    else:
        # Provide guidance if run without arguments
        print("ChronoMeme Forecaster: A tool for meme virality prediction")
        print("\nUsage options:")
        print("1. With existing data:")
        print("   python meme_virality_forecaster.py <meme_name> -i <input_file.csv> [options]")
        print("   Example: python meme_virality_forecaster.py Doge -i meme_data.csv -o doge_output -p 30")
        print("\n2. With Twitter scraping:")
        print("   python meme_virality_forecaster.py <keyword> -s [options]")
        print("   Example: python meme_virality_forecaster.py \"cat meme\" -s -m 1000 -o cat_meme_output")
        print("\nRun with --help for more options.")