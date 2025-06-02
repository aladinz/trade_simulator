"""
Stock Price Prediction Module with Advanced ML Models

This module provides functionality for predicting stock price movements
using various machine learning algorithms and technical indicators.
"""
import numpy as np
import pandas as pd
from datetime import timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
from indicators.rsi import calculate_rsi
from indicators.sma import add_sma_signals

class StockPredictor:
    """
    A class for predicting stock price movements using various ML algorithms
    """
    
    def __init__(self, model_type='random_forest'):
        """
        Initialize the predictor with the specified model type
        
        Parameters:
        -----------
        model_type : str
            Type of model to use ('random_forest', 'xgboost', 'ensemble')
        """
        self.model_type = model_type
        self.model = None
        self.feature_importance = None
        self.mse = None
        self.mae = None
        
    def _create_model(self):
        """Create the specified machine learning model"""
        if self.model_type == 'random_forest':
            return RandomForestRegressor(
                n_estimators=100, 
                max_depth=10,
                min_samples_split=5,
                random_state=42
            )
        elif self.model_type == 'xgboost':
            return XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
        elif self.model_type == 'ensemble':
            # For ensemble, we'll return None here and handle it in predict
            return None
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def _normalize_dataframe(self, df):
        """
        Normalize the DataFrame columns in case they have multi-index columns
        from yfinance or other sources.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input DataFrame that might have multi-index columns
            
        Returns:
        --------
        pandas.DataFrame
            Normalized DataFrame with single-level column names
        """
        df = df.copy()
        
        # Check if we have multi-index columns
        if isinstance(df.columns, pd.MultiIndex):
            # Flatten multi-index columns
            df.columns = [col[0] if col[1] == '' else f"{col[0]}_{col[1]}" for col in df.columns]
        
        # Ensure we have the required columns 
        required_cols = ['Date', 'Close', 'Open', 'High', 'Low', 'Volume']
        
        # Handle case where columns have ticker suffix
        for col in required_cols:
            # If the exact column doesn't exist, look for it with ticker suffix
            if col not in df.columns:
                # Find columns that start with the required name
                matching_cols = [c for c in df.columns if c.startswith(f"{col}_") or c == f"{col}"]
                
                if matching_cols:
                    # Use the first matching column and rename it
                    df[col] = df[matching_cols[0]]
        
        return df
            
    def engineer_features(self, df):
        """
        Create technical indicators and features for prediction
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Dataframe with Date, Open, High, Low, Close, Volume columns
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with additional engineered features
        """
        # Normalize the DataFrame to handle multi-index columns from yfinance
        df = self._normalize_dataframe(df)
        
        # Basic time features
        df['Days'] = (df['Date'] - df['Date'].min()).dt.days
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        df['Month'] = df['Date'].dt.month
        
        # Price features
        df['PriceChange'] = df['Close'].pct_change()
        df['PriceRange'] = (df['High'] - df['Low']) / df['Close']
        df['ClosePctRank'] = df['Close'].rank(pct=True)
        
        # Volume features
        df['VolumeChange'] = df['Volume'].pct_change()
        df['RelativeVolume'] = df['Volume'] / df['Volume'].rolling(20).mean()
        
        # Technical indicators (using existing modules)
        df = add_sma_signals(df)
        df = calculate_rsi(df)
        
        # Additional technical indicators
        df['SMA20_Ratio'] = df['Close'] / df['SMA20']
        df['SMA50_Ratio'] = df['Close'] / df['SMA50']
        df['SMA_Crossover'] = (df['SMA20'] > df['SMA50']).astype(int)
        
        # Bollinger Bands
        df['20MA'] = df['Close'].rolling(window=20).mean()
        df['20SD'] = df['Close'].rolling(window=20).std()
        df['UpperBand'] = df['20MA'] + (df['20SD'] * 2)
        df['LowerBand'] = df['20MA'] - (df['20SD'] * 2)
        df['BBPosition'] = (df['Close'] - df['LowerBand']) / (df['UpperBand'] - df['LowerBand'])
        
        # MACD
        df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = df['EMA12'] - df['EMA26']
        df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Histogram'] = df['MACD'] - df['Signal']
        
        # Momentum indicator
        df['Momentum'] = df['Close'] / df['Close'].shift(10)
        
        # Drop NaN values
        df = df.dropna()
        
        return df
        
    def train(self, df, target_days=1, test_size=0.2):
        """
        Train the prediction model
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Stock price dataframe with OHLCV data
        target_days : int
            Number of days in the future to predict
        test_size : float
            Fraction of data to use for testing
            
        Returns:
        --------
        dict
            Dictionary with training results
        """
        # Engineer features
        feature_df = self.engineer_features(df)
        
        # Create target variable (future price)
        feature_df['Target'] = feature_df['Close'].shift(-target_days)
        feature_df = feature_df.dropna()
        
        # Select features for the model
        features = [
            'Days', 'DayOfWeek', 'Month', 'PriceChange', 'PriceRange',
            'ClosePctRank', 'VolumeChange', 'RelativeVolume', 'RSI',
            'SMA20_Ratio', 'SMA50_Ratio', 'SMA_Crossover',
            'BBPosition', 'MACD', 'MACD_Histogram', 'Momentum'
        ]
        
        # Validate that all required features exist
        missing_features = [f for f in features if f not in feature_df.columns]
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        # Prepare data for training
        X = feature_df[features]
        y = feature_df['Target']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False
        )
        
        # Create and train model
        self.model = self._create_model()
        
        if self.model_type == 'ensemble':
            # For ensemble, train both models and average predictions
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
            xgb_model = XGBRegressor(n_estimators=100, random_state=42)
            
            rf_model.fit(X_train, y_train)
            xgb_model.fit(X_train, y_train)
            
            # Store both models
            self.model = {
                'random_forest': rf_model,
                'xgboost': xgb_model
            }
            
            # Make predictions with both models
            rf_preds = rf_model.predict(X_test)
            xgb_preds = xgb_model.predict(X_test)
            
            # Average predictions
            ensemble_preds = (rf_preds + xgb_preds) / 2
            
            # Calculate metrics
            self.mse = mean_squared_error(y_test, ensemble_preds)
            self.mae = mean_absolute_error(y_test, ensemble_preds)
            
            # Feature importance (average from both models)
            rf_importance = rf_model.feature_importances_
            xgb_importance = xgb_model.feature_importances_
            self.feature_importance = dict(zip(
                features, 
                [(rf + xgb) / 2 for rf, xgb in zip(rf_importance, xgb_importance)]
            ))
        else:
            # Standard single model training
            self.model.fit(X_train, y_train)
            
            # Make predictions
            predictions = self.model.predict(X_test)
            
            # Calculate metrics
            self.mse = mean_squared_error(y_test, predictions)
            self.mae = mean_absolute_error(y_test, predictions)
            
            # Get feature importance
            self.feature_importance = dict(zip(features, self.model.feature_importances_))
        
        return {
            'mse': self.mse,
            'mae': self.mae,
            'rmse': np.sqrt(self.mse),
            'feature_importance': self.feature_importance
        }
    
    def predict_price_movement(self, df, forecast_days=90):
        """
        Predict future price movement
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Stock price dataframe
        forecast_days : int
            Number of days to forecast
            
        Returns:
        --------
        dict
            Dictionary containing forecast results
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Engineer features
        feature_df = self.engineer_features(df)
        
        # Get the last day features for prediction starting point
        features = [
            'Days', 'DayOfWeek', 'Month', 'PriceChange', 'PriceRange',
            'ClosePctRank', 'VolumeChange', 'RelativeVolume', 'RSI',
            'SMA20_Ratio', 'SMA50_Ratio', 'SMA_Crossover',
            'BBPosition', 'MACD', 'MACD_Histogram', 'Momentum'
        ]
        
        # Generate forecast dates
        last_date = df['Date'].iloc[-1] if 'Date' in df.columns else pd.to_datetime(df.index[-1])
        forecast_dates = [last_date + timedelta(days=i) for i in range(1, forecast_days + 1)]
        
        # Initialize forecasts with confidence intervals
        forecasts = []
        confidence_intervals = []
        
        # Store minimum date once to avoid repetitive access
        min_date = df['Date'].min() if 'Date' in df.columns else pd.to_datetime(df.index.min())
        
        if self.model_type == 'ensemble':
            rf_model = self.model['random_forest']
            xgb_model = self.model['xgboost']
            
            # Simple rolling forecast - this is a simplified approach
            # For production, we'd need a more sophisticated method for updating features
            last_data = feature_df.iloc[-1:].copy()  # Get last row as a separate DataFrame
            
            for i, date in enumerate(forecast_dates):
                try:
                    # Work with a fresh copy of the last data for each iteration
                    forecast_row = last_data.copy()
                    
                    # Update time features directly with scalar values
                    days_value = (date - min_date).days
                    forecast_row.loc[forecast_row.index[0], 'Days'] = days_value
                    forecast_row.loc[forecast_row.index[0], 'DayOfWeek'] = date.dayofweek
                    forecast_row.loc[forecast_row.index[0], 'Month'] = date.month
                    
                    # Ensure we're using only the required features for prediction
                    X_pred = forecast_row[features]
                    
                    # Get predictions from both models
                    rf_pred = rf_model.predict(X_pred)[0]
                    xgb_pred = xgb_model.predict(X_pred)[0]
                    
                    # Average the predictions
                    ensemble_pred = (rf_pred + xgb_pred) / 2
                    
                    # Calculate confidence interval (simple approach)
                    diff = abs(rf_pred - xgb_pred)
                    lower_bound = ensemble_pred - diff/2
                    upper_bound = ensemble_pred + diff/2
                    
                    forecasts.append(ensemble_pred)
                    confidence_intervals.append((lower_bound, upper_bound))
                    
                except Exception as e:
                    raise ValueError(f"Error during forecast for date {date}: {str(e)}")
        else:
            # Simple rolling forecast for single model
            last_data = feature_df.iloc[-1:].copy()  # Get last row as a separate DataFrame
            
            for i, date in enumerate(forecast_dates):
                try:
                    # Work with a fresh copy of the last data for each iteration
                    forecast_row = last_data.copy()
                    
                    # Update time features directly with scalar values
                    days_value = (date - min_date).days
                    forecast_row.loc[forecast_row.index[0], 'Days'] = days_value
                    forecast_row.loc[forecast_row.index[0], 'DayOfWeek'] = date.dayofweek
                    forecast_row.loc[forecast_row.index[0], 'Month'] = date.month
                    
                    # Ensure we're using only the required features for prediction
                    X_pred = forecast_row[features]
                    
                    # Make prediction
                    next_price = self.model.predict(X_pred)[0]
                    
                    forecasts.append(next_price)
                    
                    # Simple confidence interval based on model error
                    confidence = np.sqrt(self.mse) * 1.96  # 95% confidence
                    confidence_intervals.append((next_price - confidence, next_price + confidence))
                    
                except Exception as e:
                    raise ValueError(f"Error during forecast for date {date}: {str(e)}")
        
        return {
            'dates': forecast_dates,
            'forecast': forecasts,
            'confidence_intervals': confidence_intervals,
            'model_metrics': {
                'mse': self.mse,
                'mae': self.mae,
                'rmse': np.sqrt(self.mse)
            }
        }

# Simple wrapper functions for easy usage from app.py

def predict_3_month_forecast(df, model_type='random_forest'):
    """
    Predict stock price 3 months into the future
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Stock price dataframe with OHLCV data
    model_type : str
        Type of model to use (random_forest, xgboost, ensemble)
        
    Returns:
    --------
    dict
        Dictionary containing forecast results
    """
    try:
        # Make sure the dataframe has the required columns
        if isinstance(df.columns, pd.MultiIndex):
            print("Warning: Found multi-index columns, normalizing...")
        
        # Make a clean copy for prediction
        df_copy = df.copy()
        
        # Ensure we have a Date column if using the DatetimeIndex
        if 'Date' not in df_copy.columns and isinstance(df_copy.index, pd.DatetimeIndex):
            df_copy = df_copy.reset_index()
        
        # Initialize and train the predictor
        predictor = StockPredictor(model_type=model_type)
        predictor.train(df_copy)
        
        # Generate forecast
        forecast_result = predictor.predict_price_movement(df_copy, forecast_days=90)
        
        # Verify forecast data is valid
        if len(forecast_result['forecast']) == 0:
            raise ValueError("Forecast produced empty results")
            
        return forecast_result
    except Exception as e:
        # Add more detailed error information with traceback
        import traceback
        error_details = traceback.format_exc()
        print(f"ML Forecast Error: {error_details}")
        raise ValueError(f"Failed to generate 3-month forecast: {str(e)}")

def get_feature_importance(df, model_type='random_forest'):
    """
    Get feature importance for the prediction model
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Stock price dataframe with OHLCV data
    model_type : str
        Type of model to use
        
    Returns:
    --------
    dict
        Dictionary of features and their importance scores
    """
    try:
        predictor = StockPredictor(model_type=model_type)
        training_results = predictor.train(df)
        return training_results['feature_importance']
    except Exception as e:
        # Add more detailed error information
        raise ValueError(f"Failed to get feature importance: {str(e)}")
