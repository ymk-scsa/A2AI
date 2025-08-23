"""
A2AI - Advanced Financial Analysis AI
Market Entry Timing Analysis Model

This module analyzes the optimal timing for market entry by new companies
and evaluates how entry timing affects success probability and growth trajectories.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import warnings
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import lightgbm as lgb
import scipy.stats as stats
from scipy.optimize import minimize
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MarketPhase(Enum):
    """Market lifecycle phases"""
    EMERGENCE = "emergence"
    GROWTH = "growth" 
    MATURITY = "maturity"
    DECLINE = "decline"
    DISRUPTION = "disruption"

class EntryStrategy(Enum):
    """Market entry strategies"""
    PIONEER = "pioneer"        # First mover
    EARLY_FOLLOWER = "early_follower"  # Enter in growth phase
    LATE_ENTRANT = "late_entrant"      # Enter in maturity
    DISRUPTOR = "disruptor"    # Enter with disruptive technology

@dataclass
class MarketEntryMetrics:
    """Metrics for market entry analysis"""
    company_id: str
    market_category: str
    entry_year: int
    entry_phase: MarketPhase
    entry_strategy: EntryStrategy
    market_size_at_entry: float
    market_growth_rate: float
    competition_intensity: float
    technology_readiness: float
    success_score: float  # Composite success metric
    survival_years: int
    peak_market_share: float
    time_to_profitability: Optional[int]

class MarketEntryTimingModel:
    """
    Advanced model for analyzing market entry timing and its impact on company success.
    
    This model evaluates:
    1. Optimal entry timing based on market lifecycle
    2. Success probability by entry phase
    3. Competition intensity effects
    4. Technology readiness impact
    5. Pioneer vs. follower advantages
    """
    
    def __init__(self, 
                    model_type: str = 'ensemble',
                    random_state: int = 42):
        """
        Initialize the Market Entry Timing Model.
        
        Args:
            model_type: Type of model ('linear', 'rf', 'gbm', 'lgb', 'ensemble')
            random_state: Random seed for reproducibility
        """
        self.model_type = model_type
        self.random_state = random_state
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_importance = {}
        self.is_fitted = False
        
        # Initialize models
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize different model types"""
        models_config = {
            'linear': LinearRegression(),
            'ridge': Ridge(alpha=1.0, random_state=self.random_state),
            'rf': RandomForestRegressor(
                n_estimators=100, 
                max_depth=10, 
                random_state=self.random_state
            ),
            'gbm': GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=self.random_state
            ),
            'lgb': lgb.LGBMRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=self.random_state,
                verbose=-1
            )
        }
        
        if self.model_type == 'ensemble':
            self.models = models_config
        else:
            self.models[self.model_type] = models_config[self.model_type]
            
    def _extract_market_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract market-related features for timing analysis.
        
        Args:
            df: Input dataframe with company and market data
            
        Returns:
            DataFrame with extracted features
        """
        features = pd.DataFrame()
        
        # Basic market timing features
        features['entry_year'] = df['entry_year']
        features['market_age_at_entry'] = df['entry_year'] - df['market_birth_year']
        features['years_since_entry'] = 2024 - df['entry_year']
        
        # Market lifecycle features
        phase_encoder = LabelEncoder()
        features['market_phase_encoded'] = phase_encoder.fit_transform(df['entry_phase'])
        self.encoders['market_phase'] = phase_encoder
        
        strategy_encoder = LabelEncoder()
        features['entry_strategy_encoded'] = strategy_encoder.fit_transform(df['entry_strategy'])
        self.encoders['entry_strategy'] = strategy_encoder
        
        # Market conditions at entry
        features['market_size_at_entry'] = df['market_size_at_entry']
        features['market_growth_rate'] = df['market_growth_rate']
        features['competition_intensity'] = df['competition_intensity']
        features['technology_readiness'] = df['technology_readiness']
        
        # Derived timing features
        features['early_entry_advantage'] = (features['market_age_at_entry'] < 5).astype(int)
        features['growth_phase_entry'] = (df['entry_phase'] == MarketPhase.GROWTH.value).astype(int)
        features['pioneer_entry'] = (df['entry_strategy'] == EntryStrategy.PIONEER.value).astype(int)
        
        # Competition and market dynamics
        features['market_concentration'] = df.get('market_concentration', 0.5)
        features['barrier_to_entry'] = df.get('barrier_to_entry', 0.5)
        features['network_effects'] = df.get('network_effects', 0.0)
        
        # Technology and innovation factors
        features['tech_disruption_risk'] = df.get('tech_disruption_risk', 0.3)
        features['patent_intensity'] = df.get('patent_intensity', 0.0)
        features['rd_investment_ratio'] = df.get('rd_investment_ratio', 0.05)
        
        return features
        
    def _calculate_success_metrics(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate composite success metrics for market entry.
        
        Args:
            df: Input dataframe with company performance data
            
        Returns:
            Series with success scores
        """
        # Normalize individual metrics
        survival_score = np.clip(df['survival_years'] / 20, 0, 1)  # Max 20 years
        market_share_score = np.clip(df['peak_market_share'] / 50, 0, 1)  # Max 50%
        
        # Profitability score (inverse of time to profitability)
        profitability_score = np.where(
            df['time_to_profitability'].notna(),
            np.clip(1 - (df['time_to_profitability'] / 10), 0, 1),  # Max 10 years
            0.1  # Default low score for companies that never reached profitability
        )
        
        # Composite success score
        success_score = (
            0.4 * survival_score + 
            0.3 * market_share_score + 
            0.3 * profitability_score
        )
        
        return success_score
        
    def _prepare_training_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare training data for the model.
        
        Args:
            df: Raw input data
            
        Returns:
            Tuple of (features, target)
        """
        # Extract features
        X = self._extract_market_features(df)
        
        # Calculate target variable (success score)
        if 'success_score' in df.columns:
            y = df['success_score']
        else:
            y = self._calculate_success_metrics(df)
            
        # Handle missing values
        X = X.fillna(X.mean())
        y = y.fillna(y.mean())
        
        return X, y
        
    def fit(self, df: pd.DataFrame, 
            target_column: str = 'success_score') -> 'MarketEntryTimingModel':
        """
        Fit the market entry timing model.
        
        Args:
            df: Training data with company and market information
            target_column: Name of target variable column
            
        Returns:
            Fitted model instance
        """
        logger.info(f"Fitting market entry timing model with {len(df)} samples")
        
        # Prepare data
        X, y = self._prepare_training_data(df)
        
        # Split data for validation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )
        
        # Scale features
        self.scalers['features'] = StandardScaler()
        X_train_scaled = self.scalers['features'].fit_transform(X_train)
        X_test_scaled = self.scalers['features'].transform(X_test)
        
        # Fit models
        self.fitted_models = {}
        self.model_scores = {}
        
        for name, model in self.models.items():
            try:
                # Fit model
                if name in ['linear', 'ridge']:
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    
                # Calculate performance metrics
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                self.fitted_models[name] = model
                self.model_scores[name] = {
                    'mse': mse,
                    'mae': mae,
                    'r2': r2,
                    'rmse': np.sqrt(mse)
                }
                
                # Extract feature importance
                if hasattr(model, 'feature_importances_'):
                    self.feature_importance[name] = dict(zip(
                        X.columns, model.feature_importances_
                    ))
                elif hasattr(model, 'coef_'):
                    self.feature_importance[name] = dict(zip(
                        X.columns, np.abs(model.coef_)
                    ))
                    
                logger.info(f"{name} model - R²: {r2:.4f}, RMSE: {np.sqrt(mse):.4f}")
                
            except Exception as e:
                logger.warning(f"Failed to fit {name} model: {e}")
                continue
                
        self.feature_columns = X.columns.tolist()
        self.is_fitted = True
        
        return self
        
    def predict_entry_success(self, 
                            market_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Predict success probability for a given market entry scenario.
        
        Args:
            market_data: Dictionary with market entry parameters
            
        Returns:
            Dictionary with success predictions from different models
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
            
        # Convert to DataFrame
        df = pd.DataFrame([market_data])
        
        # Extract features
        X = self._extract_market_features(df)
        
        # Ensure all required columns are present
        for col in self.feature_columns:
            if col not in X.columns:
                X[col] = 0  # Default value for missing features
                
        X = X[self.feature_columns]  # Reorder columns
        X = X.fillna(0)
        
        predictions = {}
        
        for name, model in self.fitted_models.items():
            try:
                if name in ['linear', 'ridge']:
                    X_scaled = self.scalers['features'].transform(X)
                    pred = model.predict(X_scaled)[0]
                else:
                    pred = model.predict(X)[0]
                    
                predictions[name] = max(0, min(1, pred))  # Clip to [0,1]
                
            except Exception as e:
                logger.warning(f"Prediction failed for {name} model: {e}")
                continue
                
        # Ensemble prediction (if multiple models)
        if len(predictions) > 1:
            predictions['ensemble'] = np.mean(list(predictions.values()))
            
        return predictions
        
    def analyze_entry_timing_window(self, 
                                    market_data: Dict[str, Any],
                                    years_ahead: int = 10) -> pd.DataFrame:
        """
        Analyze the optimal entry timing window for a market.
        
        Args:
            market_data: Base market characteristics
            years_ahead: Number of years to analyze ahead
            
        Returns:
            DataFrame with timing analysis results
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before analysis")
            
        current_year = 2024
        results = []
        
        for year_offset in range(years_ahead + 1):
            entry_year = current_year + year_offset
            
            # Create scenario for this entry year
            scenario = market_data.copy()
            scenario['entry_year'] = entry_year
            scenario['market_age_at_entry'] = entry_year - market_data.get('market_birth_year', 2020)
            
            # Adjust market conditions based on time
            # Assume market growth slows over time
            base_growth = market_data.get('market_growth_rate', 0.1)
            scenario['market_growth_rate'] = base_growth * np.exp(-0.1 * year_offset)
            
            # Competition intensity increases over time
            base_competition = market_data.get('competition_intensity', 0.5)
            scenario['competition_intensity'] = min(1.0, base_competition + 0.05 * year_offset)
            
            # Market size grows
            base_size = market_data.get('market_size_at_entry', 1000)
            scenario['market_size_at_entry'] = base_size * (1 + base_growth) ** year_offset
            
            # Get predictions
            predictions = self.predict_entry_success(scenario)
            
            # Determine market phase based on age
            market_age = scenario['market_age_at_entry']
            if market_age < 2:
                phase = MarketPhase.EMERGENCE
            elif market_age < 7:
                phase = MarketPhase.GROWTH
            elif market_age < 15:
                phase = MarketPhase.MATURITY
            else:
                phase = MarketPhase.DECLINE
                
            results.append({
                'entry_year': entry_year,
                'years_from_now': year_offset,
                'market_age': market_age,
                'market_phase': phase.value,
                'predicted_success': predictions.get('ensemble', 
                                                    list(predictions.values())[0] if predictions else 0),
                'market_size': scenario['market_size_at_entry'],
                'market_growth': scenario['market_growth_rate'],
                'competition_intensity': scenario['competition_intensity']
            })
            
        return pd.DataFrame(results)
        
    def get_feature_importance(self, model_name: str = None) -> Dict[str, float]:
        """
        Get feature importance scores.
        
        Args:
            model_name: Specific model name (if None, returns ensemble average)
            
        Returns:
            Dictionary of feature importance scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
            
        if model_name and model_name in self.feature_importance:
            return self.feature_importance[model_name]
        
        # Return average importance across models
        all_features = set()
        for importances in self.feature_importance.values():
            all_features.update(importances.keys())
            
        avg_importance = {}
        for feature in all_features:
            scores = [imp.get(feature, 0) for imp in self.feature_importance.values()]
            avg_importance[feature] = np.mean(scores)
            
        return dict(sorted(avg_importance.items(), key=lambda x: x[1], reverse=True))
        
    def optimize_entry_timing(self, 
                            market_data: Dict[str, Any],
                            constraints: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Optimize entry timing to maximize success probability.
        
        Args:
            market_data: Base market characteristics
            constraints: Optional constraints (min/max years, etc.)
            
        Returns:
            Dictionary with optimization results
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before optimization")
            
        constraints = constraints or {}
        min_years = constraints.get('min_years_ahead', 0)
        max_years = constraints.get('max_years_ahead', 10)
        
        # Analyze timing window
        timing_analysis = self.analyze_entry_timing_window(
            market_data, max_years
        )
        
        # Filter by constraints
        valid_entries = timing_analysis[
            (timing_analysis['years_from_now'] >= min_years) &
            (timing_analysis['years_from_now'] <= max_years)
        ]
        
        if len(valid_entries) == 0:
            raise ValueError("No valid entry points found with given constraints")
            
        # Find optimal timing
        optimal_idx = valid_entries['predicted_success'].idxmax()
        optimal_entry = valid_entries.loc[optimal_idx]
        
        # Calculate confidence intervals
        success_mean = valid_entries['predicted_success'].mean()
        success_std = valid_entries['predicted_success'].std()
        
        return {
            'optimal_entry_year': int(optimal_entry['entry_year']),
            'optimal_years_ahead': int(optimal_entry['years_from_now']),
            'predicted_success_rate': float(optimal_entry['predicted_success']),
            'market_phase_at_entry': optimal_entry['market_phase'],
            'confidence_interval': {
                'lower': float(success_mean - 1.96 * success_std),
                'upper': float(success_mean + 1.96 * success_std)
            },
            'alternative_timings': valid_entries.nlargest(3, 'predicted_success')[
                ['entry_year', 'predicted_success', 'market_phase']
            ].to_dict('records')
        }
        
    def evaluate_entry_strategies(self, 
                                market_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Evaluate different entry strategies for a given market.
        
        Args:
            market_data: Market characteristics
            
        Returns:
            DataFrame comparing different entry strategies
        """
        strategies = [
            EntryStrategy.PIONEER,
            EntryStrategy.EARLY_FOLLOWER,
            EntryStrategy.LATE_ENTRANT,
            EntryStrategy.DISRUPTOR
        ]
        
        results = []
        
        for strategy in strategies:
            scenario = market_data.copy()
            scenario['entry_strategy'] = strategy.value
            
            # Adjust market conditions based on strategy
            if strategy == EntryStrategy.PIONEER:
                scenario['competition_intensity'] = 0.1
                scenario['market_growth_rate'] *= 1.5  # Higher growth potential
                scenario['technology_readiness'] = 0.6  # Lower tech readiness
                
            elif strategy == EntryStrategy.EARLY_FOLLOWER:
                scenario['competition_intensity'] = 0.3
                scenario['market_growth_rate'] *= 1.2
                scenario['technology_readiness'] = 0.8
                
            elif strategy == EntryStrategy.LATE_ENTRANT:
                scenario['competition_intensity'] = 0.7
                scenario['market_growth_rate'] *= 0.8
                scenario['technology_readiness'] = 0.9
                
            else:  # DISRUPTOR
                scenario['competition_intensity'] = 0.2
                scenario['market_growth_rate'] *= 2.0
                scenario['technology_readiness'] = 0.4
                scenario['tech_disruption_risk'] = 0.8
                
            predictions = self.predict_entry_success(scenario)
            
            results.append({
                'strategy': strategy.value,
                'predicted_success': predictions.get('ensemble', 
                                                    list(predictions.values())[0] if predictions else 0),
                'competition_intensity': scenario['competition_intensity'],
                'market_growth_potential': scenario['market_growth_rate'],
                'technology_readiness': scenario['technology_readiness'],
                'risk_level': 1 - scenario['technology_readiness']
            })
            
        return pd.DataFrame(results).sort_values('predicted_success', ascending=False)
        
    def get_model_performance(self) -> pd.DataFrame:
        """
        Get performance metrics for all fitted models.
        
        Returns:
            DataFrame with model performance metrics
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
            
        performance_data = []
        for name, scores in self.model_scores.items():
            performance_data.append({
                'model': name,
                **scores
            })
            
        return pd.DataFrame(performance_data).sort_values('r2', ascending=False)

# Example usage and testing
def create_sample_data(n_samples: int = 200) -> pd.DataFrame:
    """Create sample data for testing the market entry timing model."""
    np.random.seed(42)
    
    # Market categories from the document
    market_categories = [
        'high_share_robotics', 'high_share_endoscopy', 'high_share_machine_tools',
        'declining_automotive', 'declining_steel', 'declining_home_appliances',
        'lost_consumer_electronics', 'lost_semiconductors', 'lost_smartphones'
    ]
    
    data = []
    for i in range(n_samples):
        # Basic info
        company_id = f"company_{i:03d}"
        market_category = np.random.choice(market_categories)
        entry_year = np.random.randint(1990, 2020)
        market_birth_year = np.random.randint(1980, entry_year)
        
        # Market phase based on age at entry
        market_age = entry_year - market_birth_year
        if market_age < 3:
            phase = MarketPhase.EMERGENCE.value
        elif market_age < 8:
            phase = MarketPhase.GROWTH.value
        elif market_age < 15:
            phase = MarketPhase.MATURITY.value
        else:
            phase = MarketPhase.DECLINE.value
            
        # Entry strategy (influenced by market phase)
        if phase == MarketPhase.EMERGENCE.value:
            strategy = np.random.choice([EntryStrategy.PIONEER.value, EntryStrategy.DISRUPTOR.value])
        elif phase == MarketPhase.GROWTH.value:
            strategy = EntryStrategy.EARLY_FOLLOWER.value
        else:
            strategy = EntryStrategy.LATE_ENTRANT.value
            
        # Market conditions
        market_size = np.random.lognormal(15, 1)  # Market size in millions
        growth_rate = np.random.normal(0.15, 0.1) if phase in ['emergence', 'growth'] else np.random.normal(0.05, 0.05)
        competition = np.random.beta(2, 5) if phase == 'emergence' else np.random.beta(5, 2)
        tech_readiness = np.random.beta(3, 2) if phase in ['maturity', 'decline'] else np.random.beta(2, 3)
        
        # Success metrics (influenced by timing and strategy)
        base_success = 0.5
        if strategy == EntryStrategy.PIONEER.value:
            base_success += 0.2 if phase == 'emergence' else -0.1
        elif strategy == EntryStrategy.EARLY_FOLLOWER.value:
            base_success += 0.15
        elif strategy == EntryStrategy.DISRUPTOR.value:
            base_success += np.random.choice([0.3, -0.2], p=[0.3, 0.7])  # High risk/reward
            
        # Add noise and constraints
        success_score = np.clip(base_success + np.random.normal(0, 0.15), 0, 1)
        survival_years = int(np.clip(success_score * 25 + np.random.normal(0, 3), 1, 30))
        peak_share = success_score * 20 + np.random.exponential(2)
        time_to_profit = int(np.clip(8 - success_score * 6 + np.random.exponential(2), 1, 15))
        
        data.append({
            'company_id': company_id,
            'market_category': market_category,
            'entry_year': entry_year,
            'market_birth_year': market_birth_year,
            'entry_phase': phase,
            'entry_strategy': strategy,
            'market_size_at_entry': market_size,
            'market_growth_rate': growth_rate,
            'competition_intensity': competition,
            'technology_readiness': tech_readiness,
            'success_score': success_score,
            'survival_years': survival_years,
            'peak_market_share': peak_share,
            'time_to_profitability': time_to_profit,
            'market_concentration': np.random.beta(2, 2),
            'barrier_to_entry': np.random.beta(3, 2),
            'network_effects': np.random.beta(1, 3),
            'tech_disruption_risk': np.random.beta(2, 3),
            'patent_intensity': np.random.gamma(2, 0.1),
            'rd_investment_ratio': np.random.gamma(2, 0.02)
        })
        
    return pd.DataFrame(data)

if __name__ == "__main__":
    # Example usage
    print("Testing Market Entry Timing Model")
    
    # Create sample data
    sample_data = create_sample_data(300)
    print(f"Created {len(sample_data)} sample entries")
    
    # Initialize and fit model
    model = MarketEntryTimingModel(model_type='ensemble')
    model.fit(sample_data)
    
    # Show model performance
    print("\nModel Performance:")
    print(model.get_model_performance())
    
    # Example market entry scenario
    market_scenario = {
        'market_birth_year': 2020,
        'market_size_at_entry': 500,  # Million USD
        'market_growth_rate': 0.25,
        'competition_intensity': 0.3,
        'technology_readiness': 0.7,
        'market_concentration': 0.4,
        'barrier_to_entry': 0.6,
        'network_effects': 0.2,
        'tech_disruption_risk': 0.4,
        'patent_intensity': 0.15,
        'rd_investment_ratio': 0.08
    }
    
    # Analyze optimal entry timing
    print("\nOptimal Entry Timing Analysis:")
    timing_results = model.optimize_entry_timing(market_scenario)
    for key, value in timing_results.items():
        if key != 'alternative_timings':
            print(f"{key}: {value}")
    
    # Evaluate different entry strategies
    print("\nEntry Strategy Comparison:")
    strategy_comparison = model.evaluate_entry_strategies(market_scenario)
    print(strategy_comparison[['strategy', 'predicted_success', 'risk_level']].round(3))
    
    # Feature importance
    print("\nTop Feature Importance:")
    importance = model.get_feature_importance()
    for feature, score in list(importance.items())[:8]:
        print(f"{feature}: {score:.3f}")