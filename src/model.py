```python
import pandas as pd
import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from prophet import Prophet
from prophet.serialize import model_to_json, model_from_json
from prophet.diagnostics import cross_validation, performance_metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import json
import os
from datetime import timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Download VADER lexicon if not already downloaded
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except nltk.downloader.DownloadError:
    logging.info("Downloading VADER lexicon...")
    nltk.download('vader_lexicon')
except LookupError:
     logging.info("VADER lexicon lookup failed, attempting download...")
     nltk.download('vader_lexicon')


class ChronoMemeForecaster:
    """
    Predicts the short-term 'virality' or trend score of internet memes
    based on social media mention frequency and sentiment analysis over time.
    """

    def __init__(self, prophet_params=None):
        """
        Initializes the ChronoMemeForecaster.

        Args:
            prophet_params (dict, optional): Parameters for the Prophet model.
                                             Defaults to None, using Prophet defaults.
        """
        self.model = None
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.prophet_params = prophet_params if prophet_params else {}
        self.processed_data = None
        self.forecast_df = None
        self.performance_metrics = None

    def _calculate_sentiment(self, text):
        """Calculates the compound sentiment score for a given text."""
        if pd.isna(text) or not isinstance(text, str):
            return 0.0
        return self.sentiment_analyzer.polarity_scores(text)['compound']

    def preprocess_data(self, df, date_col, text_col=None, count_col=None, date_format=None):
        """
        Preprocesses the input DataFrame for modeling.

        Args:
            df (pd.DataFrame): Input DataFrame with time-series data.
            date_col (str): Name of the column containing timestamps.
            text_col (str, optional): Name of the column containing mention text for sentiment analysis.
                                      Required if count_col is not provided.
            count_col (str, optional): Name of the column containing pre-aggregated mention counts.
                                       If provided, sentiment analysis is skipped for frequency calculation.
            date_format (str, optional): The format of the date column if it needs parsing.

        Returns:
            pd.DataFrame: Processed DataFrame with 'ds' (datetime) and 'y' (mention frequency) columns,
                          and optionally 'sentiment' column.
        """
        logging.info("Starting data preprocessing...")
        if date_col not in df.columns:
            raise ValueError(f"Date column '{date_col}' not found in DataFrame.")
        if text_col is None and count_col is None:
            raise ValueError("Either 'text_col' or 'count_col' must be provided.")
        if text_col and text_col not in df.columns:
             raise ValueError(f"Text column '{text_col}' not found in DataFrame.")
        if count_col and count_col not in df.columns:
             raise ValueError(f"Count column '{count_col}' not found in DataFrame.")

        df_processed = df.copy()

        # Convert date column to datetime
        try:
            if date_format:
                df_processed['ds'] = pd.to_datetime(df_processed[date_col], format=date_format)
            else:
                df_processed['ds'] = pd.to_datetime(df_processed[date_col])
        except Exception as e:
            raise ValueError(f"Error parsing date column '{date_col}': {e}")

        df_processed['ds'] = df_processed['ds'].dt.tz_localize(None) # Ensure timezone naive

        # Calculate sentiment if text column is provided
        if text_col:
            logging.info(f"Calculating sentiment for column '{text_col}'...")
            df_processed['sentiment'] = df_processed[text_col].apply(self._calculate_sentiment)

        # Aggregate data by day
        df_processed = df_processed.set_index('ds')
        aggregation_dict = {}

        if count_col:
            logging.info(f"Aggregating daily counts from column '{count_col}'...")
            aggregation_dict['y'] = pd.NamedAgg(column=count_col, aggfunc='sum')
        elif text_col:
             logging.info(f"Aggregating daily counts based on occurrences...")
             # Use size to count mentions if no explicit count column
             aggregation_dict['y'] = pd.NamedAgg(column=date_col, aggfunc='size') # Use any column for size
        else:
             # This case should technically be caught earlier, but added for safety
             raise ValueError("Cannot determine aggregation target ('count_col' or 'text_col' needed).")


        if 'sentiment' in df_processed.columns:
             logging.info("Aggregating daily average sentiment...")
             aggregation_dict['avg_sentiment'] = pd.NamedAgg(column='sentiment', aggfunc='mean')

        daily_data = df_processed.resample('D').agg(**aggregation_dict).reset_index()

        # Fill missing days with 0 mentions and neutral sentiment
        if 'y' in daily_data.columns:
            daily_data['y'] = daily_data['y'].fillna(0)
        if 'avg_sentiment' in daily_data.columns:
            daily_data['avg_sentiment'] = daily_data['avg_sentiment'].fillna(0.0)

        # Ensure 'ds' and 'y' columns exist for Prophet
        if 'y' not in daily_data.columns:
            raise ValueError("Failed to create 'y' column during aggregation.")
        if 'ds' not in daily_data.columns:
             raise ValueError("Failed to create 'ds' column during aggregation.")

        daily_data = daily_data.rename(columns={'avg_sentiment': 'sentiment'}) # Keep 'sentiment' name consistent

        self.processed_data = daily_data[['ds', 'y'] + (['sentiment'] if 'sentiment' in daily_data.columns else [])]
        logging.info("Data preprocessing finished.")
        logging.info(f"Processed data shape: {self.processed_data.shape}")
        logging.info(f"Processed data columns: {self.processed_data.columns.tolist()}")
        logging.info(f"Date range: {self.processed_data['ds'].min()} to {self.processed_data['ds'].max()}")

        # Ensure 'y' is numeric
        self.processed_data['y'] = pd.to_numeric(self.processed_data['y'], errors='coerce').fillna(0)

        return self.processed_data


    def train(self, train_df=None, validation_split=0.2, perform_cv=False, cv_initial='730 days', cv_period='180 days', cv_horizon='365 days'):
        """
        Trains the Prophet model on the processed data.

        Args:
            train_df (pd.DataFrame, optional): DataFrame with 'ds' and 'y' columns for training.
                                               If None, uses the data processed by preprocess_data.
            validation_split (float): Fraction of data to use for validation (0 to 1). Default is 0.2.
                                      Used only if perform_cv is False.
            perform_cv (bool): Whether to perform cross-validation using Prophet's diagnostics. Default is False.
            cv_initial (str): Initial training period for cross-validation (Prophet format).
            cv_period (str): Spacing between cutoff dates for cross-validation (Prophet format).
            cv_horizon (str): Forecast horizon for cross-validation (Prophet format).
        """
        if train_df is not None:
            if not all(col in train_df.columns for col in ['ds', 'y']):
                raise ValueError("Training DataFrame must contain 'ds' and 'y' columns.")
            data_to_train = train_df.copy()
        elif self.processed_data is not None:
            data_to_train = self.processed_data.copy()
        else:
            raise ValueError("No data available for training. Call preprocess_data first or provide train_df.")

        if data_to_train.empty or data_to_train['y'].sum() == 0:
            logging.warning("Training data is empty or contains only zero values. Model may not train effectively.")
            # Still attempt to train, Prophet might handle it, but results will be trivial.

        logging.info(f"Initializing Prophet model with params: {self.prophet_params}")
        self.model = Prophet(**self.prophet_params)

        # Add sentiment as a regressor if available
        if 'sentiment' in data_to_train.columns:
            logging.info("Adding 'sentiment' as a regressor.")
            self.model.add_regressor('sentiment')
            # Ensure regressor column doesn't have NaNs for training rows
            data_to_train['sentiment'] = data_to_train['sentiment'].fillna(0.0)


        if perform_cv:
            logging.info("Starting Prophet cross-validation...")
            try:
                df_cv = cross_validation(self.model, initial=cv_initial, period=cv_period, horizon=cv_horizon,
                                         parallel="processes") # Use multiprocessing if available
                self.performance_metrics = performance_metrics(df_cv)
                logging.info("Cross-validation finished.")
                logging.info(f"CV Performance Metrics:\n{self.performance_metrics.head()}")

                # Train final model on all data after CV
                logging.info("Training final model on all available data...")
                self.model = Prophet(**self.prophet_params) # Re-initialize
                if 'sentiment' in data_to_train.columns:
                     self.model.add_regressor('sentiment')
                self.model.fit(data_to_train[['ds', 'y'] + (['sentiment'] if 'sentiment' in data_to_train.columns else [])])

            except Exception as e:
                logging.error(f"Cross-validation failed: {e}. Training on split data instead.")
                perform_cv = False # Fallback to simple split

        if not perform_cv:
            if validation_split > 0 and len(data_to_train) > 1:
                split_index = int(len(data_to_train) * (1 - validation_split))
                train_set = data_to_train.iloc[:split_index]
                validation_set = data_to_train.iloc[split_index:]

                logging.info(f"Training on {len(train_set)} samples, validating on {len(validation_set)} samples.")
                self.model.fit(train_set[['ds', 'y'] + (['sentiment'] if 'sentiment' in train_set.columns else [])])

                # Evaluate on validation set
                future_val = validation_set[['ds']]
                if 'sentiment' in validation_set.columns:
                    future_val['sentiment'] = validation_set['sentiment'].fillna(0.0)

                forecast_val = self.model.predict(future_val)
                self.evaluate(validation_set, forecast_val) # Store metrics internally

                # Train final model on all data after evaluation
                logging.info("Re-training final model on all available data...")
                self.model = Prophet(**self.prophet_params) # Re-initialize
                if 'sentiment' in data_to_train.columns:
                     self.model.add_regressor('sentiment')
                self.model.fit(data_to_train[['ds', 'y'] + (['sentiment'] if 'sentiment' in data_to_train.columns else [])])

            else:
                logging.info("Training model on all available data (no validation split).")
                self.model.fit(data_to_train[['ds', 'y'] + (['sentiment'] if 'sentiment' in data_to_train.columns else [])])

        logging.info("Model training finished.")


    def predict(self, periods=30, future_df=None):
        """
        Makes future predictions.

        Args:
            periods (int): Number of future periods (days) to forecast. Used if future_df is None.
            future_df (pd.DataFrame, optional): DataFrame with 'ds' column for future dates.
                                                If it contains a 'sentiment' column, it will be used as a regressor.
                                                If None, future dates are generated automatically.

        Returns:
            pd.DataFrame: DataFrame containing the forecast ('yhat', 'yhat_lower', 'yhat_upper').
        """
        if not self.model:
            raise ValueError("Model has not been trained yet. Call train() first.")

        logging.info(f"Generating forecast for {periods} periods...")
        if future_df is None:
            future = self.model.make_future_dataframe(periods=periods)
        else:
            if 'ds' not in future_df.columns:
                raise ValueError("future_df must contain a 'ds' column.")
            future = future_df[['ds']].copy()
            future['ds'] = pd.to_datetime(future['ds']).dt.tz_localize(None)


        # Add regressor values to future dataframe if needed
        if 'sentiment' in self.model.regressors:
            logging.info("Adding sentiment regressor data to future dataframe...")
            if future_df is not None and 'sentiment' in future_df.columns:
                 future['sentiment'] = future_df['sentiment'].fillna(0.0)
            elif self.processed_data is not None and 'sentiment' in self.processed_data.columns:
                 # Use last known sentiment or mean sentiment as a simple forward fill strategy
                 last_sentiment = self.processed_data['sentiment'].iloc[-1] if not self.processed_data.empty else 0.0
                 logging.warning(f"Sentiment regressor required, but not provided in future_df. Using last known value: {last_sentiment}")
                 future['sentiment'] = last_sentiment
                 future['sentiment'] = future['sentiment'].fillna(0.0) # Ensure no NaNs introduced
            else:
                 logging.warning("Sentiment regressor required, but no sentiment data available. Using 0.0.")
                 future['sentiment'] = 0.0


        self.forecast_df = self.model.predict(future)
        logging.info("Forecast generation finished.")
        return self.forecast_df

    def evaluate(self, test_df, forecast_df=None):
        """
        Evaluates the model performance on test data.

        Args:
            test_df (pd.DataFrame): DataFrame with 'ds' and 'y' (actual values).
            forecast_df (pd.DataFrame, optional): DataFrame with forecast results ('ds', 'yhat').
                                                 If None, assumes forecast is stored internally from validation split.

        Returns:
            dict: Dictionary containing evaluation metrics (MAE, MSE, RMSE).
        """
        logging.info("Evaluating model performance...")
        if forecast_df is None and self.forecast_df is None:
             logging.warning("No forecast data provided or stored internally for evaluation.")
             # Attempt to use validation forecast if available from train()
             # This part is tricky as forecast_val was local to train(). Store it?
             # For now, rely on explicit forecast_df or CV metrics.
             # Let's store the validation forecast if generated.
             # Revisit: Need a better way to handle internal validation forecast storage.
             # For now, require forecast_df if not using CV.
             raise ValueError("Provide forecast_df for evaluation if not using CV or internal validation.")

        if forecast_df is None:
            # This case is currently unlikely based on the logic above, but for safety:
            logging.warning("Using internally stored forecast for evaluation (likely from validation split).")
            forecast_to_eval = self.forecast_df # Assumes predict was called matching test_df dates
        else:
            forecast_to_eval = forecast_df

        # Merge actual and predicted values
        results = pd.merge(test_df[['ds', 'y']], forecast_to_eval[['ds', 'yhat']], on='ds')

        if results.empty:
            logging.error("Evaluation failed: No matching 'ds' found between test_df and forecast_df.")
            return None

        y_true = results['y']
        y_pred = results['yhat']

        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)

        metrics = {
            'mae': mae,
            'mse': mse,
            'rmse': rmse
        }

        # Store metrics if evaluation was done during training's validation split
        if forecast_df is None: # Heuristic: if forecast_df wasn't passed, it was internal validation
             self.performance_metrics = pd.DataFrame([metrics]) # Store as DataFrame for consistency

        logging.info(f"Evaluation Metrics: MAE={mae:.4f}, MSE={mse:.4f}, RMSE={rmse:.4f}")
        return metrics

    def plot_forecast(self, include_history=True):
        """
        Plots the historical data and the forecast.

        Returns:
            matplotlib.figure.Figure: The figure object containing the plot.
        """
        if not self.model:
            raise ValueError("Model has not been trained yet.")
        if self.forecast_df is None:
            raise ValueError("No forecast available to plot. Call predict() first.")

        logging.info("Generating forecast plot...")
        fig = self.model.plot(self.forecast_df)

        if include_history and self.processed_data is not None:
            # Overlay actual data points for comparison if desired
            ax = fig.gca()
            ax.plot(self.processed_data['ds'], self.processed_data['y'], 'k.', label='Actual Mentions')
            ax.legend()

        ax = fig.gca()
        ax.set_title('Meme Virality Forecast')
        ax.set_xlabel('Date')
        ax.set_ylabel('Predicted Mention Frequency (yhat)')
        plt.tight_layout()
        logging.info("Plot generation finished.")
        return fig

    def plot_components(self):
        """
        Plots the components of the forecast (trend, seasonality).

        Returns:
            matplotlib.figure.Figure: The figure object containing the components plot.
        """
        if not self.model:
            raise ValueError("Model has not been trained yet.")
        if self.forecast_df is None:
            raise ValueError("No forecast available to plot components. Call predict() first.")

        logging.info("Generating forecast components plot...")
        fig = self.model.plot_components(self.forecast_df)
        plt.tight_layout()
        logging.info("Components plot generation finished.")
        return fig

    def predict_peak_virality(self, forecast_horizon_df=None):
        """
        Predicts the peak virality window within the forecast horizon.

        Args:
            forecast_horizon_df (pd.DataFrame, optional): The forecast DataFrame. If None, uses the
                                                          internally stored forecast.

        Returns:
            tuple: A tuple containing (peak_date, peak_score) or (None, None) if no forecast exists.
        """
        if forecast_horizon_df is None:
            forecast_horizon_df = self.forecast_df

        if forecast_horizon_df is None or 'yhat' not in forecast_horizon_df.columns:
            logging.warning("No forecast data available to predict peak virality.")
            return None, None

        # Find the index of the maximum predicted value ('yhat')
        peak_index = forecast_horizon_df['yhat'].idxmax()
        peak_data = forecast_horizon_df.loc[peak_index]

        peak_date = peak_data['ds']
        peak_score = peak_data['yhat']

        logging.info(f"Predicted peak virality: Score={peak_score:.2f} on Date={peak_date.strftime('%Y-%m-%d')}")
        return peak_date, peak_score


    def save_model(self, filepath):
        """
        Saves the trained Prophet model to a JSON file.

        Args:
            filepath (str): The path to save the model file (e.g., 'meme_model.json').
        """
        if not self.model:
            raise ValueError("No model trained to save.")

        logging.info(f"Saving model to {filepath}...")
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as fout:
            json.dump(model_to_json(self.model), fout)
        logging.info("Model saved successfully.")

    def load_model(self, filepath):
        """
        Loads a trained Prophet model from a JSON file.

        Args:
            filepath (str): The path to the saved model file.
        """
        logging.info(f"Loading model from {filepath}...")
        if not os.path.exists(filepath):
             raise FileNotFoundError(f"Model file not found at {filepath}")

        with open(filepath, 'r') as fin:
            model_json = json.load(fin)
        self.model = model_from_json(model_json)
        logging.info("Model loaded successfully.")
        # Note: Regressors are saved/loaded automatically with the model state


# Example Usage Block
if __name__ == "__main__":

    # 1. Generate Mock Data
    logging.info("Generating mock data...")
    np.random.seed(42)
    date_rng = pd.date_range(start='2023-01-01', end='2024-06-30', freq='H')
    mock_data = pd.DataFrame(date_rng, columns=['timestamp'])

    # Simulate meme mentions (cyclical + noise + trend)
    base_trend = np.linspace(0, 5, len(mock_data)) # Slow linear trend
    daily_cycle = 10 * (1 + np.sin(mock_data['timestamp'].dt.hour * 2 * np.pi / 24))
    weekly_cycle = 15 * (1 + np.sin(mock_data['timestamp'].dt.dayofweek * 2 * np.pi / 7))
    noise = np.random.poisson(5, len(mock_data)) # Poisson noise for counts
    mock_data['mentions'] = np.maximum(0, base_trend + daily_cycle + weekly_cycle + noise).astype(int)

    # Simulate sentiment (correlated loosely with mentions)
    sentiment_base = (mock_data['mentions'] / mock_data['mentions'].max()) * 0.6 - 0.3 # Scale mentions to approx [-0.3, 0.3]
    sentiment_noise = np.random.normal(0, 0.2, len(mock_data))
    mock_data['sentiment_score'] = np.clip(sentiment_base + sentiment_noise, -1.0, 1.0)

    # Add mock text (not really used by VADER here, just for structure)
    mock_data['text'] = "mock meme text " + mock_data['mentions'].astype(str)

    # Simulate using pre-aggregated counts and pre-calculated sentiment
    # We will aggregate this hourly data to daily in preprocessing
    logging.info(f"Generated {len(mock_data)} hourly mock data points.")


    # 2. Initialize Forecaster
    # Example Prophet parameters (optional)
    params = {
        'growth': 'linear',
        'seasonality_mode': 'additive',
        'yearly_seasonality': True,
        'weekly_seasonality': True,
        'daily_seasonality': False # Daily data, so daily seasonality within Prophet is less meaningful
    }
    forecaster = ChronoMemeForecaster(prophet_params=params)

    # 3. Preprocess Data
    # Use 'mentions' as the count_col and 'sentiment_score' directly
    # We need to aggregate the hourly mock data to daily first for preprocessing
    mock_data['date_col'] = mock_data['timestamp'].dt.date
    daily_mock = mock_data.groupby('date_col').agg(
        total_mentions=('mentions', 'sum'),
        avg_sentiment=('sentiment_score', 'mean')
    ).reset_index()
    daily_mock = daily_mock.rename(columns={'date_col': 'date'}) # Rename for clarity

    # Now preprocess the daily aggregated data
    try:
        # Pass the pre-aggregated daily counts and average sentiment
        processed_df = forecaster.preprocess_data(daily_mock, date_col='date', count_col='total_mentions')
        # Manually add the pre-calculated average sentiment back for regressor use
        processed_df = pd.merge(processed_df, daily_mock[['date', 'avg_sentiment']], left_on='ds', right_on='date', how='left')
        processed_df = processed_df.rename(columns={'avg_sentiment': 'sentiment'})
        processed_df['sentiment'] = processed_df['sentiment'].fillna(0.0)
        forecaster.processed_data = processed_df[['ds', 'y', 'sentiment']] # Update internal state

        logging.info("Mock data preprocessed successfully.")
        logging.info(f"\nProcessed Data Head:\n{forecaster.processed_data.head()}")

        # 4. Train Model (using internal processed data)
        # Split data for train/test demonstration (alternative to CV)
        split_date = forecaster.processed_data['ds'].max() - timedelta(days=60)
        train_data = forecaster.processed_data[forecaster.processed_data['ds'] <= split_date]
        test_data = forecaster.processed_data[forecaster.processed_data['ds'] > split_date]

        logging.info(f"Training on data up to {split_date}")
        forecaster.train(train_df=train_data, validation_split=0) # Train only on the explicit train split

        # 5. Predict Future
        logging.info("Predicting future...")
        # Create future dataframe including test period and future forecast
        future_periods = 90 # Forecast 90 days beyond the end of training data
        last_train_date = train_data['ds'].max()
        future_dates = pd.date_range(start=last_train_date + timedelta(days=1), periods=len(test_data) + future_periods)
        future_predict_df = pd.DataFrame({'ds': future_dates})

        # Add future sentiment (using last known value as placeholder)
        last_sentiment = train_data['sentiment'].iloc[-1] if not train_data.empty else 0.0
        future_predict_df['sentiment'] = last_sentiment

        forecast = forecaster.predict(future_df=future_predict_df)
        logging.info(f"\nForecast DataFrame Head:\n{forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head()}")

        # 6. Evaluate Model on Test Set
        logging.info("Evaluating model on test set...")
        # Merge forecast with actual test data for evaluation
        