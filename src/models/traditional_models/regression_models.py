"""
A2AI (Advanced Financial Analysis AI) - Regression Models Module

This module implements comprehensive regression models for analyzing the relationship
between factor variables and evaluation metrics across different enterprise lifecycles.

Key Features:
- 9 evaluation metrics (6 traditional + 3 lifecycle-specific)
- 23 factor variables per evaluation metric
- Support for enterprise lifecycle stages (surviving, extinct, emerging)
- Robust handling of survivorship bias
- Market category analysis (high-share, declining, lost-share markets)
"""

import numpy as np
import pandas as pd
import warnings
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet, 
    HuberRegressor, TheilSenRegressor, RANSACRegressor
)
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import het_white
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.stattools import adfuller
from scipy import stats
from scipy.stats import jarque_bera, normaltest

warnings.filterwarnings('ignore', category=FutureWarning)

@dataclass
class RegressionResult:
    """Data class to store regression analysis results"""
    model_name: str
    target_variable: str
    coefficients: Dict[str, float]
    intercept: float
    r_squared: float
    adj_r_squared: float
    mse: float
    mae: float
    residuals: np.ndarray
    predictions: np.ndarray
    feature_importance: Dict[str, float]
    statistical_tests: Dict[str, Any]
    cross_validation_scores: np.ndarray
    lifecycle_analysis: Dict[str, Any]

class BaseRegressionModel(ABC, BaseEstimator, RegressorMixin):
    """Base class for all regression models in A2AI"""
    
    def __init__(self, 
                    handle_survivorship_bias: bool = True,
                    lifecycle_adjustment: bool = True,
                    market_category_weights: bool = True,
                    robust_preprocessing: bool = True):
        """
        Initialize base regression model
        
        Parameters:
        -----------
        handle_survivorship_bias : bool
            Whether to apply survivorship bias correction
        lifecycle_adjustment : bool
            Whether to adjust for enterprise lifecycle stages
        market_category_weights : bool
            Whether to apply market category-specific weights
        robust_preprocessing : bool
            Whether to use robust preprocessing methods
        """
        self.handle_survivorship_bias = handle_survivorship_bias
        self.lifecycle_adjustment = lifecycle_adjustment
        self.market_category_weights = market_category_weights
        self.robust_preprocessing = robust_preprocessing
        self.scaler = None
        self.feature_selector = None
        self.model = None
        self.is_fitted = False
        
    @abstractmethod
    def _create_model(self) -> Any:
        """Create the underlying model instance"""
        pass
    
    def _preprocess_data(self, X: pd.DataFrame, y: pd.Series = None) -> Tuple[pd.DataFrame, pd.Series]:
        """Preprocess data with lifecycle and bias considerations"""
        X_processed = X.copy()
        
        # Handle survivorship bias
        if self.handle_survivorship_bias and y is not None:
            X_processed, y = self._correct_survivorship_bias(X_processed, y)
        
        # Lifecycle adjustment
        if self.lifecycle_adjustment:
            X_processed = self._apply_lifecycle_adjustment(X_processed)
        
        # Market category weighting
        if self.market_category_weights:
            X_processed = self._apply_market_weights(X_processed)
        
        return X_processed, y
    
    def _correct_survivorship_bias(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Apply survivorship bias correction using inverse probability weighting"""
        # Create survival indicator (1 if company exists at end of period, 0 if extinct)
        if 'company_status' in X.columns:
            survival_prob = X.groupby(['market_category', 'company_age'])['company_status'].transform('mean')
            # Inverse probability weights
            weights = 1.0 / np.maximum(survival_prob, 0.1)  # Avoid division by zero
            # Apply weights to observations
            X_weighted = X.multiply(weights, axis=0)
            y_weighted = y * weights
            return X_weighted, y_weighted
        return X, y
    
    def _apply_lifecycle_adjustment(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply lifecycle stage adjustments"""
        X_adjusted = X.copy()
        
        if 'company_age' in X.columns:
            # Define lifecycle stages
            def get_lifecycle_stage(age):
                if age <= 5:
                    return 'startup'
                elif age <= 15:
                    return 'growth'
                elif age <= 30:
                    return 'mature'
                else:
                    return 'established'
            
            X_adjusted['lifecycle_stage'] = X_adjusted['company_age'].apply(get_lifecycle_stage)
            
            # Create lifecycle dummy variables
            lifecycle_dummies = pd.get_dummies(X_adjusted['lifecycle_stage'], prefix='lifecycle')
            X_adjusted = pd.concat([X_adjusted, lifecycle_dummies], axis=1)
            X_adjusted.drop('lifecycle_stage', axis=1, inplace=True)
        
        return X_adjusted
    
    def _apply_market_weights(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply market category specific weights"""
        X_weighted = X.copy()
        
        if 'market_category' in X.columns:
            # Define market category weights based on strategic importance
            market_weights = {
                'high_share': 1.2,    # Higher weight for successful markets
                'declining': 1.0,     # Standard weight
                'lost_share': 0.8     # Lower weight but still important for learning
            }
            
            # Apply weights
            for category, weight in market_weights.items():
                mask = X_weighted['market_category'] == category
                X_weighted.loc[mask] = X_weighted.loc[mask] * weight
        
        return X_weighted
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'BaseRegressionModel':
        """Fit the regression model"""
        # Preprocess data
        X_processed, y_processed = self._preprocess_data(X, y)
        
        # Initialize scaler
        if self.robust_preprocessing:
            self.scaler = RobustScaler()
        else:
            self.scaler = StandardScaler()
        
        # Fit scaler and transform data
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_processed.select_dtypes(include=[np.number])),
            columns=X_processed.select_dtypes(include=[np.number]).columns,
            index=X_processed.index
        )
        
        # Add categorical variables back
        categorical_cols = X_processed.select_dtypes(exclude=[np.number]).columns
        if len(categorical_cols) > 0:
            X_scaled = pd.concat([X_scaled, X_processed[categorical_cols]], axis=1)
        
        # Feature selection
        if len(X_scaled.columns) > 50:  # Only if too many features
            self.feature_selector = SelectKBest(f_regression, k=min(20, len(X_scaled.columns)))
            X_scaled = pd.DataFrame(
                self.feature_selector.fit_transform(X_scaled, y_processed),
                columns=X_scaled.columns[self.feature_selector.get_support()],
                index=X_scaled.index
            )
        
        # Create and fit model
        self.model = self._create_model()
        self.model.fit(X_scaled, y_processed)
        
        self.is_fitted = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Preprocess data
        X_processed, _ = self._preprocess_data(X)
        
        # Scale data
        X_scaled = pd.DataFrame(
            self.scaler.transform(X_processed.select_dtypes(include=[np.number])),
            columns=X_processed.select_dtypes(include=[np.number]).columns,
            index=X_processed.index
        )
        
        # Add categorical variables back
        categorical_cols = X_processed.select_dtypes(exclude=[np.number]).columns
        if len(categorical_cols) > 0:
            X_scaled = pd.concat([X_scaled, X_processed[categorical_cols]], axis=1)
        
        # Apply feature selection if fitted
        if self.feature_selector:
            X_scaled = pd.DataFrame(
                self.feature_selector.transform(X_scaled),
                columns=X_scaled.columns[self.feature_selector.get_support()],
                index=X_scaled.index
            )
        
        return self.model.predict(X_scaled)

class LinearRegressionModel(BaseRegressionModel):
    """Linear regression model with enhanced diagnostics"""
    
    def __init__(self, use_statsmodels: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.use_statsmodels = use_statsmodels
        self.statsmodel_result = None
    
    def _create_model(self) -> Any:
        return LinearRegression()
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'LinearRegressionModel':
        """Fit linear regression with comprehensive diagnostics"""
        super().fit(X, y)
        
        # If using statsmodels for detailed statistics
        if self.use_statsmodels:
            # Preprocess data for statsmodels
            X_processed, y_processed = self._preprocess_data(X, y)
            
            # Add constant term
            X_with_const = sm.add_constant(X_processed.select_dtypes(include=[np.number]))
            
            # Fit statsmodels OLS
            self.statsmodel_result = sm.OLS(y_processed, X_with_const).fit()
        
        return self
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get comprehensive regression diagnostics"""
        if not self.statsmodel_result:
            return {}
        
        diagnostics = {
            'r_squared': self.statsmodel_result.rsquared,
            'adj_r_squared': self.statsmodel_result.rsquared_adj,
            'f_statistic': self.statsmodel_result.fvalue,
            'f_pvalue': self.statsmodel_result.f_pvalue,
            'aic': self.statsmodel_result.aic,
            'bic': self.statsmodel_result.bic,
            'durbin_watson': durbin_watson(self.statsmodel_result.resid),
            'jarque_bera': jarque_bera(self.statsmodel_result.resid),
        }
        
        # Heteroskedasticity test
        try:
            het_test = het_white(self.statsmodel_result.resid, self.statsmodel_result.model.exog)
            diagnostics['white_test'] = {
                'statistic': het_test[0],
                'pvalue': het_test[1]
            }
        except:
            diagnostics['white_test'] = None
        
        return diagnostics

class RidgeRegressionModel(BaseRegressionModel):
    """Ridge regression model with cross-validation for alpha selection"""
    
    def __init__(self, alpha: float = 1.0, cv_alpha_selection: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.cv_alpha_selection = cv_alpha_selection
        self.best_alpha = alpha
    
    def _create_model(self) -> Any:
        if self.cv_alpha_selection:
            from sklearn.linear_model import RidgeCV
            alphas = np.logspace(-6, 6, 13)
            return RidgeCV(alphas=alphas, cv=5)
        return Ridge(alpha=self.alpha)
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'RidgeRegressionModel':
        super().fit(X, y)
        if self.cv_alpha_selection:
            self.best_alpha = self.model.alpha_
        return self

class LassoRegressionModel(BaseRegressionModel):
    """Lasso regression model with automatic feature selection"""
    
    def __init__(self, alpha: float = 1.0, cv_alpha_selection: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.cv_alpha_selection = cv_alpha_selection
        self.best_alpha = alpha
        self.selected_features = None
    
    def _create_model(self) -> Any:
        if self.cv_alpha_selection:
            from sklearn.linear_model import LassoCV
            alphas = np.logspace(-6, 6, 13)
            return LassoCV(alphas=alphas, cv=5, max_iter=2000)
        return Lasso(alpha=self.alpha, max_iter=2000)
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'LassoRegressionModel':
        super().fit(X, y)
        if self.cv_alpha_selection:
            self.best_alpha = self.model.alpha_
        
        # Get selected features (non-zero coefficients)
        if hasattr(self.model, 'coef_'):
            feature_names = X.select_dtypes(include=[np.number]).columns
            if self.feature_selector:
                feature_names = feature_names[self.feature_selector.get_support()]
            self.selected_features = feature_names[self.model.coef_ != 0].tolist()
        
        return self

class ElasticNetModel(BaseRegressionModel):
    """ElasticNet regression combining Ridge and Lasso regularization"""
    
    def __init__(self, alpha: float = 1.0, l1_ratio: float = 0.5, 
                 cv_selection: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.cv_selection = cv_selection
        self.best_alpha = alpha
        self.best_l1_ratio = l1_ratio
    
    def _create_model(self) -> Any:
        if self.cv_selection:
            from sklearn.linear_model import ElasticNetCV
            alphas = np.logspace(-6, 6, 13)
            l1_ratios = np.linspace(0.1, 0.9, 9)
            return ElasticNetCV(alphas=alphas, l1_ratio=l1_ratios, cv=5, max_iter=2000)
        return ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio, max_iter=2000)
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'ElasticNetModel':
        super().fit(X, y)
        if self.cv_selection:
            self.best_alpha = self.model.alpha_
            self.best_l1_ratio = self.model.l1_ratio_
        return self

class RobustRegressionModel(BaseRegressionModel):
    """Robust regression models resistant to outliers"""
    
    def __init__(self, method: str = 'huber', **kwargs):
        super().__init__(**kwargs)
        self.method = method
        self.robust_preprocessing = True  # Force robust preprocessing
    
    def _create_model(self) -> Any:
        if self.method == 'huber':
            return HuberRegressor()
        elif self.method == 'theil_sen':
            return TheilSenRegressor(random_state=42)
        elif self.method == 'ransac':
            return RANSACRegressor(random_state=42)
        else:
            raise ValueError(f"Unknown robust method: {self.method}")

class PanelDataModel(BaseRegressionModel):
    """Panel data regression model for time series of cross-sections"""
    
    def __init__(self, entity_effects: bool = True, time_effects: bool = True, 
                 cluster_by: str = 'company_id', **kwargs):
        super().__init__(**kwargs)
        self.entity_effects = entity_effects
        self.time_effects = time_effects
        self.cluster_by = cluster_by
        self.panel_result = None
    
    def _create_model(self) -> Any:
        return LinearRegression()  # Fallback to sklearn if panel methods fail
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'PanelDataModel':
        """Fit panel data model using fixed effects"""
        try:
            # Try to use linearmodels for panel data if available
            from linearmodels import PanelOLS
            
            # Preprocess data
            X_processed, y_processed = self._preprocess_data(X, y)
            
            # Set up panel data structure
            if 'company_id' in X_processed.columns and 'year' in X_processed.columns:
                data = pd.concat([X_processed, y_processed], axis=1)
                data = data.set_index(['company_id', 'year'])
                
                # Fit panel model
                self.panel_result = PanelOLS(
                    dependent=y_processed,
                    exog=X_processed.select_dtypes(include=[np.number]),
                    entity_effects=self.entity_effects,
                    time_effects=self.time_effects
                ).fit(cov_type='clustered', cluster_entity=True)
                
                self.is_fitted = True
                return self
        except ImportError:
            pass  # Fall back to standard regression
        
        # Fallback to standard regression with manual fixed effects
        super().fit(X, y)
        return self

class LifecycleRegressionAnalyzer:
    """Main analyzer class for lifecycle-aware regression analysis"""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.evaluation_metrics = [
            'sales_revenue', 'sales_growth_rate', 'operating_margin',
            'net_profit_margin', 'roe', 'value_added_ratio',
            'survival_probability', 'emergence_success_rate', 'succession_success_rate'
        ]
    
    def add_model(self, name: str, model: BaseRegressionModel) -> None:
        """Add a regression model to the analyzer"""
        self.models[name] = model
    
    def fit_all_models(self, X: pd.DataFrame, target_metrics: Dict[str, pd.Series]) -> None:
        """Fit all models for all target metrics"""
        self.results = {}
        
        for metric_name, y in target_metrics.items():
            print(f"Fitting models for {metric_name}...")
            self.results[metric_name] = {}
            
            for model_name, model in self.models.items():
                try:
                    # Fit model
                    model.fit(X, y)
                    
                    # Make predictions
                    y_pred = model.predict(X)
                    residuals = y - y_pred
                    
                    # Calculate metrics
                    r2 = r2_score(y, y_pred)
                    mse = mean_squared_error(y, y_pred)
                    mae = mean_absolute_error(y, y_pred)
                    
                    # Cross-validation
                    cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
                    
                    # Feature importance (if available)
                    feature_importance = {}
                    if hasattr(model.model, 'coef_'):
                        feature_names = X.select_dtypes(include=[np.number]).columns
                        if model.feature_selector:
                            feature_names = feature_names[model.feature_selector.get_support()]
                        feature_importance = dict(zip(feature_names, model.model.coef_))
                    
                    # Statistical tests
                    statistical_tests = {}
                    if isinstance(model, LinearRegressionModel):
                        statistical_tests = model.get_diagnostics()
                    
                    # Lifecycle analysis
                    lifecycle_analysis = self._perform_lifecycle_analysis(X, y, y_pred)
                    
                    # Store results
                    result = RegressionResult(
                        model_name=model_name,
                        target_variable=metric_name,
                        coefficients=feature_importance,
                        intercept=getattr(model.model, 'intercept_', 0.0),
                        r_squared=r2,
                        adj_r_squared=self._calculate_adjusted_r2(r2, len(X), len(feature_importance)),
                        mse=mse,
                        mae=mae,
                        residuals=residuals,
                        predictions=y_pred,
                        feature_importance=feature_importance,
                        statistical_tests=statistical_tests,
                        cross_validation_scores=cv_scores,
                        lifecycle_analysis=lifecycle_analysis
                    )
                    
                    self.results[metric_name][model_name] = result
                    
                except Exception as e:
                    print(f"Error fitting {model_name} for {metric_name}: {str(e)}")
                    continue
    
    def _calculate_adjusted_r2(self, r2: float, n: int, p: int) -> float:
        """Calculate adjusted R-squared"""
        if n - p - 1 <= 0:
            return r2
        return 1 - (1 - r2) * (n - 1) / (n - p - 1)
    
    def _perform_lifecycle_analysis(self, X: pd.DataFrame, y_true: pd.Series, 
                                    y_pred: np.ndarray) -> Dict[str, Any]:
        """Perform lifecycle-specific analysis"""
        lifecycle_analysis = {}
        
        if 'company_age' in X.columns:
            # Performance by lifecycle stage
            def get_stage(age):
                if age <= 5: return 'startup'
                elif age <= 15: return 'growth'
                elif age <= 30: return 'mature'
                else: return 'established'
            
            stages = X['company_age'].apply(get_stage)
            
            for stage in stages.unique():
                mask = stages == stage
                if mask.sum() > 5:  # Minimum observations
                    stage_r2 = r2_score(y_true[mask], y_pred[mask])
                    stage_mse = mean_squared_error(y_true[mask], y_pred[mask])
                    lifecycle_analysis[stage] = {
                        'r2': stage_r2,
                        'mse': stage_mse,
                        'n_observations': mask.sum()
                    }
        
        # Market category analysis
        if 'market_category' in X.columns:
            for category in X['market_category'].unique():
                mask = X['market_category'] == category
                if mask.sum() > 5:
                    cat_r2 = r2_score(y_true[mask], y_pred[mask])
                    cat_mse = mean_squared_error(y_true[mask], y_pred[mask])
                    lifecycle_analysis[f'market_{category}'] = {
                        'r2': cat_r2,
                        'mse': cat_mse,
                        'n_observations': mask.sum()
                    }
        
        return lifecycle_analysis
    
    def get_best_models(self) -> Dict[str, str]:
        """Get the best performing model for each metric"""
        best_models = {}
        
        for metric_name, models in self.results.items():
            if not models:
                continue
            
            best_model = max(models.items(), key=lambda x: x[1].r_squared)
            best_models[metric_name] = best_model[0]
        
        return best_models
    
    def generate_summary_report(self) -> pd.DataFrame:
        """Generate a summary report of all model performances"""
        summary_data = []
        
        for metric_name, models in self.results.items():
            for model_name, result in models.items():
                summary_data.append({
                    'target_metric': metric_name,
                    'model': model_name,
                    'r_squared': result.r_squared,
                    'adj_r_squared': result.adj_r_squared,
                    'mse': result.mse,
                    'mae': result.mae,
                    'cv_mean': result.cross_validation_scores.mean(),
                    'cv_std': result.cross_validation_scores.std()
                })
        
        return pd.DataFrame(summary_data)

# Factory function to create pre-configured models
def create_regression_models() -> Dict[str, BaseRegressionModel]:
    """Create a dictionary of pre-configured regression models"""
    models = {
        'linear_ols': LinearRegressionModel(use_statsmodels=True),
        'ridge': RidgeRegressionModel(cv_alpha_selection=True),
        'lasso': LassoRegressionModel(cv_alpha_selection=True),
        'elastic_net': ElasticNetModel(cv_selection=True),
        'huber_robust': RobustRegressionModel(method='huber'),
        'theil_sen_robust': RobustRegressionModel(method='theil_sen'),
        'panel_data': PanelDataModel(entity_effects=True, time_effects=True)
    }
    
    return models

# Example usage function
def run_comprehensive_regression_analysis(X: pd.DataFrame, 
                                        target_metrics: Dict[str, pd.Series]) -> LifecycleRegressionAnalyzer:
    """Run comprehensive regression analysis on all target metrics"""
    # Create analyzer
    analyzer = LifecycleRegressionAnalyzer()
    
    # Add models
    models = create_regression_models()
    for name, model in models.items():
        analyzer.add_model(name, model)
    
    # Fit all models
    analyzer.fit_all_models(X, target_metrics)
    
    return analyzer