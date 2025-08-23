"""
A2AI Time Series Models Module

Handles temporal analysis for financial statement data across different enterprise lifecycles.
Supports variable-length time series due to company formations, extinctions, and restructuring.

Author: A2AI Development Team
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from abc import ABC, abstractmethod
import warnings
from dataclasses import dataclass
from enum import Enum

# ML/Stats imports
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Attention, MultiHeadAttention
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import torch
import torch.nn as nn
from scipy import stats
from scipy.signal import find_peaks

# Internal imports
from ..utils.statistical_utils import StatisticalUtils
from ..utils.lifecycle_utils import LifecycleUtils
from .base_model import BaseModel


class LifecycleStage(Enum):
    """Enterprise lifecycle stages for temporal analysis"""
    EMERGENCE = "emergence"      # 新設企業段階
    GROWTH = "growth"           # 成長段階
    MATURITY = "maturity"       # 成熟段階
    DECLINE = "decline"         # 衰退段階
    EXTINCTION = "extinction"    # 消滅段階
    RESTRUCTURE = "restructure"  # 再編段階


@dataclass
class TimeSeriesConfig:
    """Configuration for time series analysis"""
    sequence_length: int = 10        # LSTM/GRU sequence length
    forecast_horizon: int = 5        # Forecast periods ahead
    train_test_split: float = 0.8    # Train/test split ratio
    validation_split: float = 0.2    # Validation split
    batch_size: int = 32            # Training batch size
    epochs: int = 100               # Training epochs
    learning_rate: float = 0.001    # Learning rate
    dropout_rate: float = 0.2       # Dropout rate
    early_stopping_patience: int = 10  # Early stopping patience
    
    # ARIMA parameters
    arima_max_p: int = 5            # Max AR order
    arima_max_d: int = 2            # Max differencing
    arima_max_q: int = 5            # Max MA order
    
    # Seasonal parameters
    seasonal_periods: int = 4        # Quarterly seasonality
    detect_seasonality: bool = True  # Auto-detect seasonality


class TimeSeriesProcessor(BaseEstimator, TransformerMixin):
    """Processes variable-length time series data for different lifecycle stages"""
    
    def __init__(self, config: TimeSeriesConfig):
        self.config = config
        self.scalers = {}
        self.lifecycle_stats = {}
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fit scalers and compute lifecycle statistics"""
        # Fit scalers for each feature
        for column in X.columns:
            if X[column].dtype in ['float64', 'int64']:
                scaler = StandardScaler()
                valid_data = X[column].dropna()
                if len(valid_data) > 0:
                    self.scalers[column] = scaler.fit(valid_data.values.reshape(-1, 1))
        
        # Compute lifecycle stage statistics
        if 'lifecycle_stage' in X.columns:
            self.lifecycle_stats = self._compute_lifecycle_stats(X)
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data with appropriate scaling and lifecycle adjustments"""
        X_transformed = X.copy()
        
        # Apply scaling
        for column, scaler in self.scalers.items():
            if column in X_transformed.columns:
                non_null_mask = X_transformed[column].notna()
                if non_null_mask.sum() > 0:
                    X_transformed.loc[non_null_mask, column] = scaler.transform(
                        X_transformed.loc[non_null_mask, column].values.reshape(-1, 1)
                    ).flatten()
        
        return X_transformed
    
    def _compute_lifecycle_stats(self, X: pd.DataFrame) -> Dict:
        """Compute statistics for each lifecycle stage"""
        stats = {}
        for stage in LifecycleStage:
            stage_data = X[X['lifecycle_stage'] == stage.value]
            if len(stage_data) > 0:
                stats[stage.value] = {
                    'mean': stage_data.select_dtypes(include=[np.number]).mean(),
                    'std': stage_data.select_dtypes(include=[np.number]).std(),
                    'count': len(stage_data)
                }
        return stats


class BaseTimeSeriesModel(BaseModel, ABC):
    """Base class for time series models in A2AI"""
    
    def __init__(self, config: TimeSeriesConfig):
        super().__init__()
        self.config = config
        self.model = None
        self.processor = TimeSeriesProcessor(config)
        self.is_fitted = False
        
    @abstractmethod
    def _build_model(self, input_shape: Tuple[int, ...]) -> Any:
        """Build the underlying model"""
        pass
    
    @abstractmethod
    def _fit_model(self, X_train: np.ndarray, y_train: np.ndarray, 
                    X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> Any:
        """Fit the model to training data"""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray, forecast_horizon: Optional[int] = None) -> np.ndarray:
        """Make predictions"""
        pass
    
    def prepare_sequences(self, data: pd.DataFrame, target_col: str) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for time series modeling"""
        sequences = []
        targets = []
        
        # Group by company to handle variable-length series
        for company_id in data['company_id'].unique():
            company_data = data[data['company_id'] == company_id].sort_values('year')
            
            if len(company_data) >= self.config.sequence_length + 1:
                # Create sequences for this company
                for i in range(len(company_data) - self.config.sequence_length):
                    seq = company_data.iloc[i:i+self.config.sequence_length]
                    target = company_data.iloc[i+self.config.sequence_length][target_col]
                    
                    # Skip if target is NaN or if company was extinct
                    if not pd.isna(target):
                        sequences.append(seq.drop(['company_id', 'year', target_col], axis=1).values)
                        targets.append(target)
        
        return np.array(sequences), np.array(targets)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance"""
        predictions = self.predict(X_test)
        
        metrics = {
            'mse': mean_squared_error(y_test, predictions),
            'mae': mean_absolute_error(y_test, predictions),
            'rmse': np.sqrt(mean_squared_error(y_test, predictions))
        }
        
        # Add correlation if applicable
        if len(y_test) > 1:
            metrics['correlation'] = np.corrcoef(y_test, predictions)[0, 1]
        
        return metrics


class LSTMTimeSeriesModel(BaseTimeSeriesModel):
    """LSTM-based time series model for financial analysis"""
    
    def __init__(self, config: TimeSeriesConfig, lstm_units: List[int] = None):
        super().__init__(config)
        self.lstm_units = lstm_units or [50, 30]
        
    def _build_model(self, input_shape: Tuple[int, ...]) -> tf.keras.Model:
        """Build LSTM model architecture"""
        model = Sequential()
        
        # First LSTM layer
        model.add(LSTM(
            units=self.lstm_units[0],
            return_sequences=len(self.lstm_units) > 1,
            input_shape=input_shape,
            dropout=self.config.dropout_rate,
            recurrent_dropout=self.config.dropout_rate
        ))
        
        # Additional LSTM layers
        for i, units in enumerate(self.lstm_units[1:]):
            return_seq = i < len(self.lstm_units) - 2
            model.add(LSTM(
                units=units,
                return_sequences=return_seq,
                dropout=self.config.dropout_rate,
                recurrent_dropout=self.config.dropout_rate
            ))
        
        # Dense layers
        model.add(Dense(20, activation='relu'))
        model.add(Dropout(self.config.dropout_rate))
        model.add(Dense(1, activation='linear'))
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=self.config.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def _fit_model(self, X_train: np.ndarray, y_train: np.ndarray,
                    X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None):
        """Fit LSTM model"""
        callbacks = [
            EarlyStopping(patience=self.config.early_stopping_patience, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=5)
        ]
        
        validation_data = (X_val, y_val) if X_val is not None else None
        
        history = self.model.fit(
            X_train, y_train,
            batch_size=self.config.batch_size,
            epochs=self.config.epochs,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=0
        )
        
        return history
    
    def fit(self, data: pd.DataFrame, target_col: str):
        """Fit the LSTM model to data"""
        # Prepare data
        data_processed = self.processor.fit_transform(data)
        X, y = self.prepare_sequences(data_processed, target_col)
        
        if len(X) == 0:
            raise ValueError("No valid sequences found in data")
        
        # Build model
        self.model = self._build_model((X.shape[1], X.shape[2]))
        
        # Train/validation split
        split_idx = int(len(X) * self.config.train_test_split)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Train model
        history = self._fit_model(X_train, y_train, X_val, y_val)
        self.is_fitted = True
        
        return history
    
    def predict(self, X: np.ndarray, forecast_horizon: Optional[int] = None) -> np.ndarray:
        """Make predictions using LSTM model"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        return self.model.predict(X, verbose=0).flatten()
    
    def forecast_lifecycle_transition(self, company_data: pd.DataFrame, 
                                    target_col: str) -> Dict[str, float]:
        """Forecast when lifecycle transition might occur"""
        # This is a specialized method for lifecycle analysis
        if len(company_data) < self.config.sequence_length:
            return {'transition_probability': 0.0, 'periods_to_transition': np.inf}
        
        # Prepare recent data
        recent_data = company_data.tail(self.config.sequence_length)
        X = recent_data.drop(['company_id', 'year', target_col], axis=1, errors='ignore').values
        X = X.reshape(1, X.shape[0], X.shape[1])
        
        # Get prediction
        prediction = self.predict(X)[0]
        
        # Analyze trend
        recent_values = company_data[target_col].tail(5).values
        trend = np.polyfit(range(len(recent_values)), recent_values, 1)[0]
        
        # Estimate transition probability based on prediction and trend
        transition_prob = max(0, min(1, abs(trend) * 0.1 + abs(prediction - recent_values[-1]) * 0.05))
        
        return {
            'predicted_value': prediction,
            'trend_slope': trend,
            'transition_probability': transition_prob,
            'periods_to_transition': max(1, int(10 / (transition_prob + 0.01)))
        }


class ARIMATimeSeriesModel(BaseTimeSeriesModel):
    """ARIMA-based time series model for traditional econometric analysis"""
    
    def __init__(self, config: TimeSeriesConfig):
        super().__init__(config)
        self.arima_orders = {}  # Store best ARIMA orders for each series
        
    def _build_model(self, input_shape: Tuple[int, ...]) -> None:
        """ARIMA doesn't need to build a model beforehand"""
        pass
    
    def _fit_model(self, X_train: np.ndarray, y_train: np.ndarray,
                    X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None):
        """Fit ARIMA model - this is handled per series in fit method"""
        pass
    
    def _find_best_arima_order(self, series: pd.Series) -> Tuple[int, int, int]:
        """Find best ARIMA order using AIC"""
        best_aic = np.inf
        best_order = (1, 1, 1)
        
        # Test different combinations
        for p in range(self.config.arima_max_p + 1):
            for d in range(self.config.arima_max_d + 1):
                for q in range(self.config.arima_max_q + 1):
                    try:
                        model = ARIMA(series, order=(p, d, q))
                        fitted_model = model.fit()
                        if fitted_model.aic < best_aic:
                            best_aic = fitted_model.aic
                            best_order = (p, d, q)
                    except:
                        continue
        
        return best_order
    
    def _check_stationarity(self, series: pd.Series) -> Tuple[bool, Dict]:
        """Check stationarity using Augmented Dickey-Fuller test"""
        result = adfuller(series.dropna())
        return result[1] < 0.05, {
            'adf_statistic': result[0],
            'p_value': result[1],
            'critical_values': result[4]
        }
    
    def fit(self, data: pd.DataFrame, target_col: str):
        """Fit ARIMA models for each company"""
        self.models = {}
        
        # Fit separate ARIMA model for each company
        for company_id in data['company_id'].unique():
            company_data = data[data['company_id'] == company_id].sort_values('year')
            
            if len(company_data) >= 10:  # Minimum data points for ARIMA
                series = company_data[target_col].dropna()
                
                if len(series) >= 10:
                    try:
                        # Find best ARIMA order
                        best_order = self._find_best_arima_order(series)
                        
                        # Fit model
                        model = ARIMA(series, order=best_order)
                        fitted_model = model.fit()
                        
                        self.models[company_id] = fitted_model
                        self.arima_orders[company_id] = best_order
                        
                    except Exception as e:
                        warnings.warn(f"Failed to fit ARIMA for company {company_id}: {str(e)}")
        
        self.is_fitted = len(self.models) > 0
        return self
    
    def predict(self, X: np.ndarray, forecast_horizon: Optional[int] = None) -> Dict[str, np.ndarray]:
        """Make predictions for each company"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        horizon = forecast_horizon or self.config.forecast_horizon
        predictions = {}
        
        for company_id, model in self.models.items():
            try:
                forecast = model.forecast(steps=horizon)
                predictions[company_id] = forecast
            except Exception as e:
                warnings.warn(f"Prediction failed for company {company_id}: {str(e)}")
                predictions[company_id] = np.array([np.nan] * horizon)
        
        return predictions
    
    def analyze_seasonality(self, data: pd.DataFrame, target_col: str) -> Dict[str, Dict]:
        """Analyze seasonality patterns in the data"""
        seasonality_results = {}
        
        for company_id in data['company_id'].unique():
            company_data = data[data['company_id'] == company_id].sort_values('year')
            
            if len(company_data) >= self.config.seasonal_periods * 3:  # Minimum for seasonal analysis
                series = company_data[target_col].dropna()
                
                try:
                    # Perform seasonal decomposition
                    decomposition = seasonal_decompose(
                        series, 
                        model='additive',
                        period=self.config.seasonal_periods
                    )
                    
                    seasonality_results[company_id] = {
                        'trend': decomposition.trend.dropna(),
                        'seasonal': decomposition.seasonal.dropna(),
                        'residual': decomposition.resid.dropna(),
                        'seasonal_strength': np.var(decomposition.seasonal.dropna()) / np.var(series)
                    }
                    
                except Exception as e:
                    warnings.warn(f"Seasonality analysis failed for company {company_id}: {str(e)}")
        
        return seasonality_results


class TransformerTimeSeriesModel(BaseTimeSeriesModel):
    """Transformer-based time series model for complex temporal patterns"""
    
    def __init__(self, config: TimeSeriesConfig, 
                    d_model: int = 64, 
                    num_heads: int = 8, 
                    num_layers: int = 2):
        super().__init__(config)
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        
    def _build_model(self, input_shape: Tuple[int, ...]) -> tf.keras.Model:
        """Build Transformer model architecture"""
        inputs = tf.keras.layers.Input(shape=input_shape)
        
        # Input projection
        x = tf.keras.layers.Dense(self.d_model)(inputs)
        
        # Positional encoding
        x = self._add_positional_encoding(x)
        
        # Transformer layers
        for _ in range(self.num_layers):
            x = self._transformer_block(x)
        
        # Global average pooling
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        
        # Output layers
        x = tf.keras.layers.Dense(32, activation='relu')(x)
        x = tf.keras.layers.Dropout(self.config.dropout_rate)(x)
        outputs = tf.keras.layers.Dense(1, activation='linear')(x)
        
        model = tf.keras.Model(inputs, outputs)
        model.compile(
            optimizer=Adam(learning_rate=self.config.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def _add_positional_encoding(self, x):
        """Add positional encoding to input"""
        seq_len = tf.shape(x)[1]
        d_model = self.d_model
        
        pos = tf.cast(tf.range(seq_len)[:, tf.newaxis], tf.float32)
        i = tf.cast(tf.range(d_model)[tf.newaxis, :], tf.float32)
        
        angle_rates = 1 / tf.pow(10000.0, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        angle_rads = pos * angle_rates
        
        pos_encoding = tf.concat([
            tf.sin(angle_rads[:, 0::2]),
            tf.cos(angle_rads[:, 1::2])
        ], axis=-1)
        
        return x + pos_encoding
    
    def _transformer_block(self, x):
        """Transformer block with multi-head attention"""
        # Multi-head attention
        attention_output = MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.d_model // self.num_heads
        )(x, x)
        
        attention_output = tf.keras.layers.Dropout(self.config.dropout_rate)(attention_output)
        x1 = tf.keras.layers.LayerNormalization()(x + attention_output)
        
        # Feed forward
        ffn_output = tf.keras.layers.Dense(self.d_model * 4, activation='relu')(x1)
        ffn_output = tf.keras.layers.Dense(self.d_model)(ffn_output)
        ffn_output = tf.keras.layers.Dropout(self.config.dropout_rate)(ffn_output)
        
        return tf.keras.layers.LayerNormalization()(x1 + ffn_output)
    
    def _fit_model(self, X_train: np.ndarray, y_train: np.ndarray,
                    X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None):
        """Fit Transformer model"""
        callbacks = [
            EarlyStopping(patience=self.config.early_stopping_patience, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=5)
        ]
        
        validation_data = (X_val, y_val) if X_val is not None else None
        
        history = self.model.fit(
            X_train, y_train,
            batch_size=self.config.batch_size,
            epochs=self.config.epochs,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=0
        )
        
        return history
    
    def fit(self, data: pd.DataFrame, target_col: str):
        """Fit the Transformer model to data"""
        # Prepare data
        data_processed = self.processor.fit_transform(data)
        X, y = self.prepare_sequences(data_processed, target_col)
        
        if len(X) == 0:
            raise ValueError("No valid sequences found in data")
        
        # Build model
        self.model = self._build_model((X.shape[1], X.shape[2]))
        
        # Train/validation split
        split_idx = int(len(X) * self.config.train_test_split)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Train model
        history = self._fit_model(X_train, y_train, X_val, y_val)
        self.is_fitted = True
        
        return history
    
    def predict(self, X: np.ndarray, forecast_horizon: Optional[int] = None) -> np.ndarray:
        """Make predictions using Transformer model"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        return self.model.predict(X, verbose=0).flatten()


class EnsembleTimeSeriesModel(BaseTimeSeriesModel):
    """Ensemble of multiple time series models for robust predictions"""
    
    def __init__(self, config: TimeSeriesConfig):
        super().__init__(config)
        self.models = {
            'lstm': LSTMTimeSeriesModel(config),
            'transformer': TransformerTimeSeriesModel(config),
            'arima': ARIMATimeSeriesModel(config)
        }
        self.weights = {'lstm': 0.4, 'transformer': 0.4, 'arima': 0.2}
        
    def _build_model(self, input_shape: Tuple[int, ...]) -> None:
        """Ensemble doesn't need to build a single model"""
        pass
    
    def _fit_model(self, X_train: np.ndarray, y_train: np.ndarray,
                    X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None):
        """Fit all models in ensemble"""
        pass
    
    def fit(self, data: pd.DataFrame, target_col: str):
        """Fit all models in the ensemble"""
        results = {}
        
        # Fit each model
        for name, model in self.models.items():
            try:
                results[name] = model.fit(data, target_col)
            except Exception as e:
                warnings.warn(f"Failed to fit {name} model: {str(e)}")
                results[name] = None
        
        self.is_fitted = any(results.values())
        return results
    
    def predict(self, X: np.ndarray, forecast_horizon: Optional[int] = None) -> np.ndarray:
        """Make ensemble predictions"""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before making predictions")
        
        predictions = {}
        
        # Get predictions from each model
        for name, model in self.models.items():
            if model.is_fitted:
                try:
                    if name == 'arima':
                        # ARIMA returns dict of company predictions
                        pred_dict = model.predict(X, forecast_horizon)
                        # For ensemble, we need to align this with other models
                        # This is simplified - in practice, you'd need more sophisticated alignment
                        predictions[name] = np.array(list(pred_dict.values())).flatten()[:len(X)]
                    else:
                        predictions[name] = model.predict(X, forecast_horizon)
                except Exception as e:
                    warnings.warn(f"Prediction failed for {name}: {str(e)}")
        
        if not predictions:
            raise ValueError("No models available for prediction")
        
        # Weighted ensemble
        ensemble_pred = np.zeros(len(list(predictions.values())[0]))
        total_weight = 0
        
        for name, pred in predictions.items():
            weight = self.weights.get(name, 1.0)
            ensemble_pred += weight * pred
            total_weight += weight
        
        return ensemble_pred / total_weight
    
    def get_individual_predictions(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Get predictions from individual models"""
        predictions = {}
        
        for name, model in self.models.items():
            if model.is_fitted:
                try:
                    predictions[name] = model.predict(X)
                except Exception as e:
                    warnings.warn(f"Prediction failed for {name}: {str(e)}")
        
        return predictions


class TimeSeriesAnalyzer:
    """Main analyzer for comprehensive time series analysis"""
    
    def __init__(self, config: TimeSeriesConfig = None):
        self.config = config or TimeSeriesConfig()
        self.models = {}
        self.analysis_results = {}
        
    def analyze_lifecycle_patterns(self, data: pd.DataFrame, 
                                    evaluation_metrics: List[str]) -> Dict[str, Any]:
        """Analyze time series patterns across different lifecycle stages"""
        results = {}
        
        for metric in evaluation_metrics:
            results[metric] = {}
            
            # Analyze patterns by lifecycle stage
            for stage in LifecycleStage:
                stage_data = data[data['lifecycle_stage'] == stage.value]
                
                if len(stage_data) > 0:
                    results[metric][stage.value] = self._analyze_stage_patterns(
                        stage_data, metric
                    )
        
        return results
    
    def _analyze_stage_patterns(self, data: pd.DataFrame, metric: str) -> Dict[str, Any]:
        """Analyze patterns within a specific lifecycle stage"""
        patterns = {
            'trend_analysis': self._analyze_trends(data, metric),
            'volatility_analysis': self._analyze_volatility(data, metric),
            'cyclical_patterns': self._detect_cycles(data, metric),
            'structural_breaks': self._detect_structural_breaks(data, metric)
        }
        
        return patterns
    
    def _analyze_trends(self, data: pd.DataFrame, metric: str) -> Dict[str, float]:
        """Analyze trend characteristics"""
        trends = {}
        
        for company_id in data['company_id'].unique():
            company_data = data[data['company_id'] == company_id].sort_values('year')
            
            if len(company_data) >= 5:
                values = company_data[metric].dropna()
                if len(values) >= 5:
                    # Linear trend
                    x = np.arange(len(values))
                    trend_coef = np.polyfit(x, values, 1)[0]
                    trends[company_id] = trend_coef
        
        return {
            'mean_trend': np.mean(list(trends.values())) if trends else 0,
            'trend_std': np.std(list(trends.values())) if trends else 0,
            'positive_trends': sum(1 for t in trends.values() if t > 0),
            'negative_trends': sum(1 for t in trends.values() if t < 0)
        }
    
    def _analyze_volatility(self, data: pd.DataFrame, metric: str) -> Dict[str, float]:
        """Analyze volatility patterns"""
        volatilities = []
        
        for company_id in data['company_id'].unique():
            company_data = data[data['company_id'] == company_id].sort_values('year')
            
            if len(company_data) >= 5:
                values = company_data[metric].dropna()
                if len(values) >= 5:
                    # Calculate volatility as coefficient of variation
                    mean_val = np.mean(values)
                    if mean_val != 0:
                        volatility = np.std(values) / abs(mean_val)
                        volatilities.append(volatility)
        
        return {
            'mean_volatility': np.mean(volatilities) if volatilities else 0,
            'volatility_std': np.std(volatilities) if volatilities else 0,
            'max_volatility': np.max(volatilities) if volatilities else 0,
            'min_volatility': np.min(volatilities) if volatilities else 0
        }
    
    def _detect_cycles(self, data: pd.DataFrame, metric: str) -> Dict[str, Any]:
        """Detect cyclical patterns in time series"""
        cycles_detected = []
        
        for company_id in data['company_id'].unique():
            company_data = data[data['company_id'] == company_id].sort_values('year')
            
            if len(company_data) >= 10:
                values = company_data[metric].dropna().values
                if len(values) >= 10:
                    # Simple cycle detection using FFT
                    try:
                        fft = np.fft.fft(values)
                        freqs = np.fft.fftfreq(len(values))
                        
                        # Find dominant frequencies
                        magnitude = np.abs(fft)
                        dominant_freq_idx = np.argmax(magnitude[1:len(magnitude)//2]) + 1
                        cycle_length = 1 / abs(freqs[dominant_freq_idx]) if freqs[dominant_freq_idx] != 0 else np.inf
                        
                        if cycle_length < len(values) and cycle_length > 2:
                            cycles_detected.append(cycle_length)
                    except:
                        pass
        
        return {
            'cycles_found': len(cycles_detected),
            'average_cycle_length': np.mean(cycles_detected) if cycles_detected else 0,
            'cycle_lengths': cycles_detected
        }
    
    def _detect_structural_breaks(self, data: pd.DataFrame, metric: str) -> Dict[str, Any]:
        """Detect structural breaks in time series"""
        breaks_detected = {}
        
        for company_id in data['company_id'].unique():
            company_data = data[company_id == company_data['company_id']].sort_values('year')
            
            if len(company_data) >= 10:
                values = company_data[metric].dropna().values
                if len(values) >= 10:
                    # Simple structural break detection using CUSUM
                    try:
                        mean_val = np.mean(values)
                        cumsum = np.cumsum(values - mean_val)
                        
                        # Find maximum deviation from zero
                        max_dev_idx = np.argmax(np.abs(cumsum))
                        max_deviation = abs(cumsum[max_dev_idx])
                        
                        # Use threshold based on standard deviation
                        threshold = 2 * np.std(values) * np.sqrt(len(values))
                        
                        if max_deviation > threshold:
                            break_year = company_data.iloc[max_dev_idx]['year']
                            breaks_detected[company_id] = {
                                'break_year': break_year,
                                'deviation_magnitude': max_deviation
                            }
                    except:
                        pass
        
        return {
            'total_breaks': len(breaks_detected),
            'break_details': breaks_detected
        }
    
    def forecast_market_lifecycle(self, data: pd.DataFrame, 
                                market_category: str,
                                forecast_horizon: int = 5) -> Dict[str, Any]:
        """Forecast market lifecycle trajectories"""
        results = {}
        
        # Filter data by market category
        market_data = data[data['market_category'] == market_category]
        
        if len(market_data) == 0:
            return {'error': f'No data found for market category: {market_category}'}
        
        # Initialize ensemble model
        ensemble_model = EnsembleTimeSeriesModel(self.config)
        
        # Forecast key metrics
        key_metrics = ['sales_growth_rate', 'operating_profit_margin', 'roe']
        
        for metric in key_metrics:
            if metric in market_data.columns:
                try:
                    # Fit model
                    ensemble_model.fit(market_data, metric)
                    
                    # Prepare forecast data
                    forecast_companies = market_data['company_id'].unique()
                    company_forecasts = {}
                    
                    for company_id in forecast_companies:
                        company_data = market_data[market_data['company_id'] == company_id].sort_values('year')
                        
                        if len(company_data) >= self.config.sequence_length:
                            # Get recent sequence
                            recent_data = company_data.tail(self.config.sequence_length)
                            X = recent_data.drop(['company_id', 'year', metric], axis=1, errors='ignore').values
                            X = X.reshape(1, X.shape[0], X.shape[1])
                            
                            # Make prediction
                            prediction = ensemble_model.predict(X, forecast_horizon)
                            company_forecasts[company_id] = prediction
                    
                    results[metric] = {
                        'company_forecasts': company_forecasts,
                        'market_trend': self._analyze_market_trend(company_forecasts),
                        'extinction_risk': self._assess_extinction_risk(company_forecasts, market_data, metric)
                    }
                    
                except Exception as e:
                    results[metric] = {'error': str(e)}
        
        return results
    
    def _analyze_market_trend(self, company_forecasts: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Analyze overall market trend from company forecasts"""
        if not company_forecasts:
            return {'trend': 0, 'confidence': 0}
        
        # Calculate weighted average trend
        all_forecasts = np.array(list(company_forecasts.values()))
        mean_forecast = np.mean(all_forecasts, axis=0)
        
        # Calculate trend slope
        x = np.arange(len(mean_forecast))
        trend_slope = np.polyfit(x, mean_forecast, 1)[0]
        
        # Calculate confidence based on forecast agreement
        forecast_std = np.std([fc[0] if len(fc) > 0 else 0 for fc in company_forecasts.values()])
        confidence = max(0, 1 - forecast_std / (abs(np.mean([fc[0] if len(fc) > 0 else 0 for fc in company_forecasts.values()])) + 1e-6))
        
        return {
            'trend_slope': trend_slope,
            'confidence': confidence,
            'mean_forecast': mean_forecast.tolist()
        }
    
    def _assess_extinction_risk(self, company_forecasts: Dict[str, np.ndarray], 
                                market_data: pd.DataFrame, metric: str) -> Dict[str, float]:
        """Assess extinction risk for companies in market"""
        risk_scores = {}
        
        for company_id, forecast in company_forecasts.items():
            if len(forecast) > 0:
                company_data = market_data[market_data['company_id'] == company_id]
                recent_performance = company_data[metric].tail(5).mean()
                
                # Calculate risk based on forecast decline
                forecast_decline = max(0, recent_performance - forecast[0]) if recent_performance > 0 else 0
                relative_decline = forecast_decline / (abs(recent_performance) + 1e-6)
                
                # Additional risk factors
                volatility = company_data[metric].std() / (abs(company_data[metric].mean()) + 1e-6)
                
                # Combined risk score
                risk_score = min(1.0, relative_decline * 0.7 + volatility * 0.3)
                risk_scores[company_id] = risk_score
        
        return risk_scores
    
    def analyze_emergence_success_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze success patterns of newly emerged companies"""
        emergence_data = data[data['lifecycle_stage'] == LifecycleStage.EMERGENCE.value]
        
        if len(emergence_data) == 0:
            return {'error': 'No emergence stage data found'}
        
        results = {
            'success_indicators': self._identify_success_indicators(emergence_data),
            'growth_trajectories': self._analyze_growth_trajectories(emergence_data),
            'survival_factors': self._analyze_emergence_survival_factors(emergence_data)
        }
        
        return results
    
    def _identify_success_indicators(self, emergence_data: pd.DataFrame) -> Dict[str, Any]:
        """Identify indicators of success for emerging companies"""
        indicators = {}
        
        # Define success as companies that survived and grew
        successful_companies = []
        failed_companies = []
        
        for company_id in emergence_data['company_id'].unique():
            company_data = emergence_data[emergence_data['company_id'] == company_id].sort_values('year')
            
            if len(company_data) >= 3:
                initial_sales = company_data['sales'].iloc[0] if 'sales' in company_data.columns else 0
                final_sales = company_data['sales'].iloc[-1] if 'sales' in company_data.columns else 0
                
                growth_rate = (final_sales - initial_sales) / (initial_sales + 1e-6) if initial_sales > 0 else 0
                
                if growth_rate > 0.1:  # 10% growth threshold
                    successful_companies.append(company_id)
                else:
                    failed_companies.append(company_id)
        
        # Analyze differences between successful and failed companies
        if successful_companies and failed_companies:
            success_data = emergence_data[emergence_data['company_id'].isin(successful_companies)]
            failure_data = emergence_data[emergence_data['company_id'].isin(failed_companies)]
            
            numeric_columns = success_data.select_dtypes(include=[np.number]).columns
            
            for col in numeric_columns:
                success_mean = success_data[col].mean()
                failure_mean = failure_data[col].mean()
                
                if not (pd.isna(success_mean) or pd.isna(failure_mean)):
                    difference = success_mean - failure_mean
                    indicators[col] = {
                        'success_mean': success_mean,
                        'failure_mean': failure_mean,
                        'difference': difference,
                        'relative_difference': difference / (abs(failure_mean) + 1e-6)
                    }
        
        return indicators
    
    def _analyze_growth_trajectories(self, emergence_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze growth trajectories of emerging companies"""
        trajectories = {}
        
        for company_id in emergence_data['company_id'].unique():
            company_data = emergence_data[emergence_data['company_id'] == company_id].sort_values('year')
            
            if len(company_data) >= 3 and 'sales' in company_data.columns:
                sales_values = company_data['sales'].values
                years = np.arange(len(sales_values))
                
                # Fit different growth models
                try:
                    # Linear growth
                    linear_coef = np.polyfit(years, sales_values, 1)[0]
                    
                    # Exponential growth (log-linear)
                    log_sales = np.log(sales_values + 1e-6)
                    exp_coef = np.polyfit(years, log_sales, 1)[0]
                    
                    # Polynomial growth
                    if len(years) >= 4:
                        poly_coef = np.polyfit(years, sales_values, 2)
                        acceleration = poly_coef[0]  # Second-order coefficient
                    else:
                        acceleration = 0
                    
                    trajectories[company_id] = {
                        'linear_growth_rate': linear_coef,
                        'exponential_growth_rate': exp_coef,
                        'acceleration': acceleration,
                        'initial_value': sales_values[0],
                        'final_value': sales_values[-1],
                        'duration_years': len(sales_values)
                    }
                    
                except Exception as e:
                    trajectories[company_id] = {'error': str(e)}
        
        return trajectories
    
    def _analyze_emergence_survival_factors(self, emergence_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze factors that contribute to emergence stage survival"""
        survival_factors = {}
        
        # Define survival as lasting more than median emergence period
        emergence_durations = []
        for company_id in emergence_data['company_id'].unique():
            company_data = emergence_data[emergence_data['company_id'] == company_id]
            emergence_durations.append(len(company_data))
        
        median_duration = np.median(emergence_durations) if emergence_durations else 3
        
        survivors = []
        non_survivors = []
        
        for company_id in emergence_data['company_id'].unique():
            company_data = emergence_data[emergence_data['company_id'] == company_id]
            if len(company_data) > median_duration:
                survivors.append(company_id)
            else:
                non_survivors.append(company_id)
        
        # Compare survivor vs non-survivor characteristics
        if survivors and non_survivors:
            survivor_data = emergence_data[emergence_data['company_id'].isin(survivors)]
            non_survivor_data = emergence_data[emergence_data['company_id'].isin(non_survivors)]
            
            numeric_columns = survivor_data.select_dtypes(include=[np.number]).columns
            
            for col in numeric_columns:
                survivor_mean = survivor_data[col].mean()
                non_survivor_mean = non_survivor_data[col].mean()
                
                if not (pd.isna(survivor_mean) or pd.isna(non_survivor_mean)):
                    # Statistical test
                    try:
                        statistic, p_value = stats.ttest_ind(
                            survivor_data[col].dropna(),
                            non_survivor_data[col].dropna()
                        )
                    except:
                        statistic, p_value = np.nan, np.nan
                    
                    survival_factors[col] = {
                        'survivor_mean': survivor_mean,
                        'non_survivor_mean': non_survivor_mean,
                        'difference': survivor_mean - non_survivor_mean,
                        't_statistic': statistic,
                        'p_value': p_value,
                        'significant': p_value < 0.05 if not pd.isna(p_value) else False
                    }
        
        return survival_factors
    
    def generate_lifecycle_report(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive lifecycle analysis report"""
        report = {
            'executive_summary': {},
            'lifecycle_patterns': {},
            'market_analysis': {},
            'predictions': {},
            'recommendations': {}
        }
        
        # Executive Summary
        total_companies = data['company_id'].nunique()
        total_years = data['year'].max() - data['year'].min() + 1
        
        stage_distribution = data['lifecycle_stage'].value_counts().to_dict()
        
        report['executive_summary'] = {
            'total_companies_analyzed': total_companies,
            'analysis_period_years': total_years,
            'lifecycle_stage_distribution': stage_distribution,
            'data_completeness': (1 - data.isnull().sum().sum() / data.size) * 100
        }
        
        # Lifecycle Patterns
        evaluation_metrics = ['sales', 'sales_growth_rate', 'operating_profit_margin', 'roe']
        available_metrics = [m for m in evaluation_metrics if m in data.columns]
        
        if available_metrics:
            report['lifecycle_patterns'] = self.analyze_lifecycle_patterns(data, available_metrics)
        
        # Market Analysis by Category
        if 'market_category' in data.columns:
            market_categories = data['market_category'].unique()
            for category in market_categories:
                report['market_analysis'][category] = self.forecast_market_lifecycle(
                    data, category, forecast_horizon=3
                )
        
        # Emergence Analysis
        report['emergence_analysis'] = self.analyze_emergence_success_patterns(data)
        
        # Generate Recommendations
        report['recommendations'] = self._generate_strategic_recommendations(report)
        
        return report
    
    def _generate_strategic_recommendations(self, analysis_results: Dict[str, Any]) -> Dict[str, List[str]]:
        """Generate strategic recommendations based on analysis results"""
        recommendations = {
            'high_share_markets': [],
            'declining_markets': [],
            'lost_markets': [],
            'emergence_strategies': [],
            'risk_management': []
        }
        
        # Analyze market trends and generate recommendations
        if 'market_analysis' in analysis_results:
            for market, analysis in analysis_results['market_analysis'].items():
                if 'error' not in analysis:
                    # Extract trend information
                    trends = {}
                    for metric, metric_analysis in analysis.items():
                        if isinstance(metric_analysis, dict) and 'market_trend' in metric_analysis:
                            trend_slope = metric_analysis['market_trend'].get('trend_slope', 0)
                            trends[metric] = trend_slope
                    
                    # Generate market-specific recommendations
                    if 'high' in market.lower():
                        recommendations['high_share_markets'].extend([
                            f"Maintain market leadership in {market} through continued R&D investment",
                            f"Monitor emerging competitors in {market} market",
                            f"Consider geographic expansion in {market} sector"
                        ])
                    elif 'declining' in market.lower():
                        recommendations['declining_markets'].extend([
                            f"Develop transition strategy for {market} market",
                            f"Identify adjacent markets for diversification from {market}",
                            f"Optimize operational efficiency in {market} sector"
                        ])
                    elif 'lost' in market.lower():
                        recommendations['lost_markets'].extend([
                            f"Evaluate re-entry opportunities in {market} market",
                            f"Analyze lessons learned from {market} market exit",
                            f"Focus resources on higher-potential markets than {market}"
                        ])
        
        # Emergence strategy recommendations
        if 'emergence_analysis' in analysis_results:
            emergence_analysis = analysis_results['emergence_analysis']
            if 'success_indicators' in emergence_analysis:
                recommendations['emergence_strategies'].extend([
                    "Focus on key success indicators identified in emergence analysis",
                    "Implement early-stage monitoring systems for new ventures",
                    "Develop standardized emergence stage evaluation criteria"
                ])
        
        # Risk management recommendations
        recommendations['risk_management'].extend([
            "Implement early warning systems for extinction risk",
            "Diversify across multiple market categories",
            "Monitor lifecycle stage transitions closely",
            "Develop contingency plans for market disruption"
        ])
        
        return recommendations


# Example usage and testing functions
def create_sample_config() -> TimeSeriesConfig:
    """Create sample configuration for testing"""
    return TimeSeriesConfig(
        sequence_length=8,
        forecast_horizon=3,
        train_test_split=0.8,
        batch_size=16,
        epochs=50,
        learning_rate=0.001
    )


def test_time_series_models():
    """Test function for time series models"""
    # Create sample data
    np.random.seed(42)
    companies = ['Company_A', 'Company_B', 'Company_C']
    years = range(1990, 2024)
    
    data = []
    for company in companies:
        for year in years:
            # Simulate lifecycle progression
            company_age = year - 1990
            if company_age < 10:
                stage = LifecycleStage.EMERGENCE.value
            elif company_age < 20:
                stage = LifecycleStage.GROWTH.value
            else:
                stage = LifecycleStage.MATURITY.value
            
            # Generate synthetic financial metrics
            base_sales = 1000 + company_age * 50 + np.random.normal(0, 100)
            growth_rate = max(-0.5, min(0.5, 0.1 - company_age * 0.005 + np.random.normal(0, 0.05)))
            profit_margin = max(0, min(0.3, 0.15 - company_age * 0.002 + np.random.normal(0, 0.02)))
            roe = max(0, min(0.4, 0.2 - company_age * 0.003 + np.random.normal(0, 0.03)))
            
            data.append({
                'company_id': company,
                'year': year,
                'sales': base_sales,
                'sales_growth_rate': growth_rate,
                'operating_profit_margin': profit_margin,
                'roe': roe,
                'lifecycle_stage': stage,
                'market_category': 'high_share' if company == 'Company_A' else 'declining'
            })
    
    df = pd.DataFrame(data)
    
    # Test models
    config = create_sample_config()
    analyzer = TimeSeriesAnalyzer(config)
    
    # Generate comprehensive report
    report = analyzer.generate_lifecycle_report(df)
    
    return report


if __name__ == "__main__":
    # Run tests
    test_results = test_time_series_models()
    print("Time series models test completed successfully")
    print(f"Report keys: {list(test_results.keys())}")