"""
A2AI - Advanced Financial Analysis AI
Causal Forest Implementation for Financial Statement Analysis

This module implements Causal Forest for estimating heterogeneous treatment effects
in financial analysis, specifically designed to analyze the causal impact of
factor variables on evaluation metrics across different market categories
(high-share, declining-share, lost-share markets).
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import logging
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CausalForestResults:
    """Results container for Causal Forest analysis"""
    treatment_effects: np.ndarray
    treatment_effects_std: np.ndarray
    feature_importance: Dict[str, float]
    heterogeneity_score: float
    model_performance: Dict[str, float]
    predictions: Dict[str, np.ndarray]
    variable_importance: pd.DataFrame
    confidence_intervals: Dict[str, Tuple[np.ndarray, np.ndarray]]


class CausalForest:
    """
    Causal Forest implementation for A2AI financial analysis
    
    This implementation follows the methodology from:
    - Wager & Athey (2018): "Estimation and Inference of Heterogeneous Treatment Effects using Random Forests"
    - Athey & Wager (2019): "Estimating Treatment Effects with Causal Forests"
    
    Adapted for financial statement analysis to estimate causal effects of
    factor variables (treatments) on evaluation metrics (outcomes).
    """
    
    def __init__(
        self,
        n_estimators: int = 1000,
        max_depth: Optional[int] = None,
        min_samples_split: int = 10,
        min_samples_leaf: int = 5,
        max_features: Union[str, int] = 'sqrt',
        bootstrap: bool = True,
        honest_splits: bool = True,
        subsample_ratio: float = 0.5,
        n_jobs: int = -1,
        random_state: Optional[int] = 42,
        **kwargs
    ):
        """
        Initialize Causal Forest
        
        Parameters:
        -----------
        n_estimators : int
            Number of trees in the forest
        max_depth : int, optional
            Maximum depth of trees
        min_samples_split : int
            Minimum samples required to split an internal node
        min_samples_leaf : int
            Minimum samples required to be at a leaf node
        max_features : str or int
            Number of features to consider when looking for the best split
        bootstrap : bool
            Whether to use bootstrap samples
        honest_splits : bool
            Whether to use honest splitting (separate samples for splitting and estimation)
        subsample_ratio : float
            Ratio of samples to use for each tree
        n_jobs : int
            Number of parallel jobs
        random_state : int
            Random state for reproducibility
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.honest_splits = honest_splits
        self.subsample_ratio = subsample_ratio
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.kwargs = kwargs
        
        # Model storage
        self.outcome_forests = {}
        self.treatment_forests = {}
        self.propensity_forests = {}
        self.is_fitted = False
        
        # Feature information
        self.feature_names = None
        self.treatment_names = None
        self.outcome_names = None
        
        # Preprocessing
        self.scaler = StandardScaler()
        self.fit_preprocessor = False
        
    def _create_forest(self) -> RandomForestRegressor:
        """Create a random forest with specified parameters"""
        return RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            bootstrap=self.bootstrap,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            **self.kwargs
        )
    
    def _honest_split(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        sample_weight: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform honest splitting for unbiased estimation
        
        Returns:
        --------
        X_split, X_est, y_split, y_est : arrays
            Split data for tree building and honest estimation
        """
        n_samples = X.shape[0]
        split_idx = np.random.choice(
            n_samples, 
            size=int(n_samples * 0.5), 
            replace=False
        )
        est_idx = np.setdiff1d(np.arange(n_samples), split_idx)
        
        return X[split_idx], X[est_idx], y[split_idx], y[est_idx]
    
    def _estimate_propensity_scores(
        self, 
        X: np.ndarray, 
        W: np.ndarray
    ) -> np.ndarray:
        """
        Estimate propensity scores for treatment assignment
        
        Parameters:
        -----------
        X : array-like
            Covariates
        W : array-like
            Treatment assignments
            
        Returns:
        --------
        propensity_scores : array
            Estimated propensity scores
        """
        logger.info("Estimating propensity scores...")
        
        propensity_scores = np.zeros((X.shape[0], len(np.unique(W))))
        
        for treatment in np.unique(W):
            # Binary classification for each treatment
            y_binary = (W == treatment).astype(int)
            
            forest = self._create_forest()
            forest.fit(X, y_binary)
            
            # Store forest for later use
            self.propensity_forests[treatment] = forest
            
            # Predict probabilities
            propensities = forest.predict(X)
            propensity_scores[:, int(treatment)] = np.clip(propensities, 0.01, 0.99)
            
        return propensity_scores
    
    def _estimate_outcome_models(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        W: np.ndarray
    ) -> Dict[int, RandomForestRegressor]:
        """
        Estimate outcome models for each treatment group
        
        Parameters:
        -----------
        X : array-like
            Covariates
        y : array-like
            Outcomes
        W : array-like
            Treatment assignments
            
        Returns:
        --------
        outcome_models : dict
            Fitted outcome models for each treatment
        """
        logger.info("Estimating outcome models...")
        
        outcome_models = {}
        
        for treatment in np.unique(W):
            # Select samples with this treatment
            treatment_mask = (W == treatment)
            X_treat = X[treatment_mask]
            y_treat = y[treatment_mask]
            
            if len(X_treat) > self.min_samples_leaf:
                forest = self._create_forest()
                forest.fit(X_treat, y_treat)
                outcome_models[treatment] = forest
            else:
                logger.warning(f"Insufficient samples for treatment {treatment}")
                
        return outcome_models
    
    def _compute_residuals(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        W: np.ndarray, 
        propensity_scores: np.ndarray,
        outcome_models: Dict[int, RandomForestRegressor]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute orthogonalized residuals for causal forest
        
        Parameters:
        -----------
        X : array-like
            Covariates
        y : array-like
            Outcomes
        W : array-like
            Treatment assignments
        propensity_scores : array
            Estimated propensity scores
        outcome_models : dict
            Fitted outcome models
            
        Returns:
        --------
        y_residuals, W_residuals : arrays
            Orthogonalized residuals
        """
        logger.info("Computing orthogonalized residuals...")
        
        n_samples = X.shape[0]
        y_residuals = np.zeros(n_samples)
        W_residuals = np.zeros(n_samples)
        
        for i in range(n_samples):
            # Outcome residual: y - E[y|x]
            if W[i] in outcome_models:
                y_pred = outcome_models[W[i]].predict(X[i:i+1])[0]
            else:
                y_pred = np.mean(y[W == W[i]]) if np.any(W == W[i]) else np.mean(y)
                
            y_residuals[i] = y[i] - y_pred
            
            # Treatment residual: W - E[W|x] (propensity score)
            W_residuals[i] = W[i] - propensity_scores[i, int(W[i])]
            
        return y_residuals, W_residuals
    
    def _build_causal_forest(
        self, 
        X: np.ndarray, 
        y_residuals: np.ndarray, 
        W_residuals: np.ndarray
    ) -> RandomForestRegressor:
        """
        Build the final causal forest using residuals
        
        Parameters:
        -----------
        X : array-like
            Covariates
        y_residuals : array
            Outcome residuals
        W_residuals : array
            Treatment residuals
            
        Returns:
        --------
        causal_forest : RandomForestRegressor
            Fitted causal forest
        """
        logger.info("Building causal forest...")
        
        # Target variable for causal forest
        # This estimates tau(x) = E[y_residual * W_residual | X] / E[W_residual^2 | X]
        target = y_residuals * W_residuals
        
        causal_forest = self._create_forest()
        causal_forest.fit(X, target)
        
        return causal_forest
    
    def fit(
        self, 
        X: Union[np.ndarray, pd.DataFrame], 
        y: Union[np.ndarray, pd.Series], 
        W: Union[np.ndarray, pd.Series],
        feature_names: Optional[List[str]] = None,
        treatment_names: Optional[List[str]] = None,
        outcome_names: Optional[List[str]] = None
    ) -> 'CausalForest':
        """
        Fit the Causal Forest model
        
        Parameters:
        -----------
        X : array-like or DataFrame
            Covariates (factor variables)
        y : array-like or Series
            Outcomes (evaluation metrics)
        W : array-like or Series
            Treatment assignments (e.g., market categories, time periods, interventions)
        feature_names : list, optional
            Names of features
        treatment_names : list, optional
            Names of treatments
        outcome_names : list, optional
            Names of outcomes
            
        Returns:
        --------
        self : CausalForest
            Fitted model
        """
        logger.info("Starting Causal Forest training...")
        
        # Convert inputs to numpy arrays
        if isinstance(X, pd.DataFrame):
            self.feature_names = feature_names or list(X.columns)
            X = X.values
        else:
            self.feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]
            
        if isinstance(y, pd.Series):
            self.outcome_names = outcome_names or [y.name] if y.name else ["outcome"]
            y = y.values
        else:
            self.outcome_names = outcome_names or ["outcome"]
            
        if isinstance(W, pd.Series):
            self.treatment_names = treatment_names or [W.name] if W.name else ["treatment"]
            W = W.values
        else:
            self.treatment_names = treatment_names or ["treatment"]
        
        # Preprocess data
        if not self.fit_preprocessor:
            X = self.scaler.fit_transform(X)
            self.fit_preprocessor = True
        else:
            X = self.scaler.transform(X)
        
        # Validate inputs
        assert X.shape[0] == len(y) == len(W), "Input dimensions must match"
        assert len(np.unique(W)) >= 2, "Need at least 2 treatment groups"
        
        # Step 1: Estimate propensity scores
        propensity_scores = self._estimate_propensity_scores(X, W)
        
        # Step 2: Estimate outcome models
        outcome_models = self._estimate_outcome_models(X, y, W)
        self.outcome_forests = outcome_models
        
        # Step 3: Compute orthogonalized residuals
        y_residuals, W_residuals = self._compute_residuals(
            X, y, W, propensity_scores, outcome_models
        )
        
        # Step 4: Build causal forest
        self.causal_forest = self._build_causal_forest(X, y_residuals, W_residuals)
        
        # Store training data info
        self.n_samples_train = X.shape[0]
        self.n_features = X.shape[1]
        self.treatments = np.unique(W)
        self.is_fitted = True
        
        logger.info("Causal Forest training completed successfully")
        return self
    
    def predict_treatment_effects(
        self, 
        X: Union[np.ndarray, pd.DataFrame],
        treatment_pairs: Optional[List[Tuple[int, int]]] = None
    ) -> np.ndarray:
        """
        Predict heterogeneous treatment effects
        
        Parameters:
        -----------
        X : array-like or DataFrame
            Test covariates
        treatment_pairs : list of tuples, optional
            Pairs of treatments to compare (treatment, control)
            If None, compares all treatments to the first one
            
        Returns:
        --------
        treatment_effects : array
            Predicted treatment effects for each sample
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Convert to numpy array
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Preprocess
        X = self.scaler.transform(X)
        
        # Default treatment pairs
        if treatment_pairs is None:
            control_treatment = self.treatments[0]
            treatment_pairs = [(t, control_treatment) for t in self.treatments[1:]]
        
        treatment_effects = []
        
        for treatment, control in treatment_pairs:
            # Predict outcomes under each treatment
            if treatment in self.outcome_forests and control in self.outcome_forests:
                y_treat = self.outcome_forests[treatment].predict(X)
                y_control = self.outcome_forests[control].predict(X)
                effect = y_treat - y_control
            else:
                # Use causal forest prediction
                effect = self.causal_forest.predict(X)
            
            treatment_effects.append(effect)
        
        return np.column_stack(treatment_effects)
    
    def compute_feature_importance(self) -> Dict[str, float]:
        """
        Compute feature importance for treatment effect heterogeneity
        
        Returns:
        --------
        importance : dict
            Feature importance scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before computing importance")
        
        # Get feature importance from causal forest
        importance_scores = self.causal_forest.feature_importances_
        
        # Create importance dictionary
        importance_dict = {
            name: score for name, score in zip(self.feature_names, importance_scores)
        }
        
        return importance_dict
    
    def estimate_heterogeneity(
        self, 
        X: Union[np.ndarray, pd.DataFrame]
    ) -> float:
        """
        Estimate the degree of treatment effect heterogeneity
        
        Parameters:
        -----------
        X : array-like or DataFrame
            Covariates
            
        Returns:
        --------
        heterogeneity_score : float
            Measure of treatment effect heterogeneity (higher = more heterogeneous)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before estimating heterogeneity")
        
        # Predict treatment effects
        effects = self.predict_treatment_effects(X)
        
        # Compute variance of treatment effects
        heterogeneity_score = np.var(effects.flatten())
        
        return heterogeneity_score
    
    def bootstrap_inference(
        self, 
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        W: Union[np.ndarray, pd.Series],
        n_bootstrap: int = 100,
        confidence_level: float = 0.95
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Perform bootstrap inference for confidence intervals
        
        Parameters:
        -----------
        X, y, W : array-like
            Training data
        n_bootstrap : int
            Number of bootstrap samples
        confidence_level : float
            Confidence level for intervals
            
        Returns:
        --------
        confidence_intervals : dict
            Lower and upper bounds for treatment effects
        """
        logger.info(f"Running bootstrap inference with {n_bootstrap} samples...")
        
        # Convert inputs
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        if isinstance(W, pd.Series):
            W = W.values
        
        n_samples = X.shape[0]
        bootstrap_effects = []
        
        # Parallel bootstrap
        def bootstrap_fit(seed):
            np.random.seed(seed)
            boot_idx = np.random.choice(n_samples, size=n_samples, replace=True)
            
            # Create bootstrap model
            boot_forest = CausalForest(
                n_estimators=max(100, self.n_estimators // 5),  # Smaller for speed
                random_state=seed,
                **{k: v for k, v in self.__dict__.items() 
                    if k not in ['causal_forest', 'outcome_forests', 'propensity_forests']}
            )
            
            # Fit on bootstrap sample
            boot_forest.fit(X[boot_idx], y[boot_idx], W[boot_idx])
            
            # Predict on original data
            effects = boot_forest.predict_treatment_effects(X)
            return effects
        
        # Run bootstrap
        bootstrap_effects = Parallel(n_jobs=self.n_jobs)(
            delayed(bootstrap_fit)(seed) for seed in range(n_bootstrap)
        )
        
        # Stack results
        bootstrap_effects = np.stack(bootstrap_effects, axis=0)
        
        # Compute confidence intervals
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower_bounds = np.percentile(bootstrap_effects, lower_percentile, axis=0)
        upper_bounds = np.percentile(bootstrap_effects, upper_percentile, axis=0)
        
        confidence_intervals = {
            'lower': lower_bounds,
            'upper': upper_bounds
        }
        
        logger.info("Bootstrap inference completed")
        return confidence_intervals
    
    def analyze_financial_factors(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        W: Union[np.ndarray, pd.Series],
        factor_categories: Optional[Dict[str, List[str]]] = None
    ) -> Dict[str, Any]:
        """
        Specialized analysis for financial factors in A2AI context
        
        Parameters:
        -----------
        X : array-like or DataFrame
            Financial factor variables (120 factors)
        y : array-like or Series  
            Evaluation metrics (9 metrics)
        W : array-like or Series
            Market categories (high-share, declining, lost)
        factor_categories : dict, optional
            Grouping of factors into categories
            
        Returns:
        --------
        analysis_results : dict
            Comprehensive financial factor analysis results
        """
        logger.info("Starting specialized financial factor analysis...")
        
        # Fit model
        self.fit(X, y, W)
        
        # Predict treatment effects
        treatment_effects = self.predict_treatment_effects(X)
        
        # Compute feature importance
        feature_importance = self.compute_feature_importance()
        
        # Estimate heterogeneity
        heterogeneity_score = self.estimate_heterogeneity(X)
        
        # Bootstrap confidence intervals
        confidence_intervals = self.bootstrap_inference(X, y, W, n_bootstrap=50)
        
        # Model performance metrics
        performance_metrics = self._evaluate_model_performance(X, y, W)
        
        # Create variable importance DataFrame
        importance_df = pd.DataFrame([
            {'Factor': factor, 'Importance': importance} 
            for factor, importance in feature_importance.items()
        ]).sort_values('Importance', ascending=False)
        
        # Analyze by factor categories if provided
        category_analysis = {}
        if factor_categories:
            for category, factors in factor_categories.items():
                category_importance = np.mean([
                    feature_importance.get(factor, 0) for factor in factors
                ])
                category_analysis[category] = category_importance
        
        # Compile results
        results = CausalForestResults(
            treatment_effects=treatment_effects,
            treatment_effects_std=np.std(treatment_effects, axis=0),
            feature_importance=feature_importance,
            heterogeneity_score=heterogeneity_score,
            model_performance=performance_metrics,
            predictions={'treatment_effects': treatment_effects},
            variable_importance=importance_df,
            confidence_intervals=confidence_intervals
        )
        
        logger.info("Financial factor analysis completed")
        return {
            'results': results,
            'category_analysis': category_analysis,
            'summary_statistics': {
                'n_significant_factors': len([f for f, imp in feature_importance.items() if imp > 0.01]),
                'max_effect_size': np.max(np.abs(treatment_effects)),
                'heterogeneity_level': 'High' if heterogeneity_score > 0.1 else 'Moderate' if heterogeneity_score > 0.05 else 'Low'
            }
        }
    
    def _evaluate_model_performance(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        W: np.ndarray
    ) -> Dict[str, float]:
        """Evaluate model performance using cross-validation"""
        
        kf = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
        mse_scores = []
        r2_scores = []
        
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            W_train, W_val = W[train_idx], W[val_idx]
            
            # Fit temporary model
            temp_model = CausalForest(
                n_estimators=100,  # Smaller for speed
                random_state=self.random_state
            )
            temp_model.fit(X_train, y_train, W_train)
            
            # Predict and evaluate
            y_pred = temp_model.predict_treatment_effects(X_val)
            if y_pred.ndim > 1:
                y_pred = y_pred[:, 0]  # Use first treatment effect
                
            mse = mean_squared_error(y_val, y_pred[:len(y_val)])
            r2 = r2_score(y_val, y_pred[:len(y_val)])
            
            mse_scores.append(mse)
            r2_scores.append(r2)
        
        return {
            'mse': np.mean(mse_scores),
            'mse_std': np.std(mse_scores),
            'r2': np.mean(r2_scores),
            'r2_std': np.std(r2_scores)
        }


# Example usage and testing functions
def create_sample_financial_data(
    n_samples: int = 1000,
    n_factors: int = 120,
    n_markets: int = 3
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create sample financial data for testing
    
    Returns:
    --------
    X : array
        Financial factors (120 dimensions)
    y : array  
        Evaluation metric (e.g., ROE)
    W : array
        Market categories (0: high-share, 1: declining, 2: lost)
    """
    np.random.seed(42)
    
    # Generate correlated financial factors
    X = np.random.multivariate_normal(
        mean=np.zeros(n_factors),
        cov=np.eye(n_factors) + 0.1 * np.ones((n_factors, n_factors)),
        size=n_samples
    )
    
    # Market assignment (treatment)
    W = np.random.choice(n_markets, size=n_samples)
    
    # Generate outcome with heterogeneous treatment effects
    # High-share markets (W=0): positive effects from R&D, innovation factors
    # Declining markets (W=1): mixed effects
    # Lost markets (W=2): negative effects from most factors
    
    treatment_effects = np.zeros(n_samples)
    for i in range(n_samples):
        if W[i] == 0:  # High-share market
            treatment_effects[i] = 0.3 * (X[i, :20].mean() + np.random.normal(0, 0.1))
        elif W[i] == 1:  # Declining market  
            treatment_effects[i] = 0.1 * (X[i, 20:40].mean() + np.random.normal(0, 0.1))
        else:  # Lost market
            treatment_effects[i] = -0.2 * (X[i, 40:60].mean() + np.random.normal(0, 0.1))
    
    # Base outcome + treatment effects + noise
    y = 0.5 + treatment_effects + np.random.normal(0, 0.2, n_samples)
    
    return X, y, W


def example_usage():
    """Example usage of CausalForest for A2AI financial analysis"""
    
    logger.info("Running CausalForest example for A2AI...")
    
    # Generate sample data
    X, y, W = create_sample_financial_data(n_samples=1000, n_factors=50)
    
    # Create feature names (mimicking A2AI factor structure)
    factor_categories = {
        'Investment_Asset': [f'factor_{i}' for i in range(10)],
        'Human_Resources': [f'factor_{i}' for i in range(10, 20)], 
        'Operational_Efficiency': [f'factor_{i}' for i in range(20, 30)],
        'Market_Expansion': [f'factor_{i}' for i in range(30, 40)],
        'Financial_Structure': [f'factor_{i}' for i in range(40, 50)]
    }
    
    feature_names = []
    for category, factors in factor_categories.items():
        feature_names.extend(factors)
    
    # Initialize and fit Causal Forest
    cf = CausalForest(
        n_estimators=500,
        max_depth=10,
        random_state=42
    )
    
    # Run financial factor analysis
    results = cf.analyze_financial_factors(
        X=X, 
        y=y, 
        W=W,
        factor_categories=factor_categories
    )
    
    # Display results
    logger.info(f"Analysis Results Summary:")
    logger.info(f"Heterogeneity Score: {results['results'].heterogeneity_score:.4f}")
    logger.info(f"Model R²: {results['results'].model_performance['r2']:.4f}")
    logger.info(f"Number of Significant Factors: {results['summary_statistics']['n_significant_factors']}")
    logger.info(f"Max Effect Size: {results['summary_statistics']['max_effect_size']:.4f}")
    
    # Top 10 most important factors
    logger.info("\nTop 10 Most Important Factors:")
    top_factors = results['results'].variable_importance.head(10)
    for _, row in top_factors.iterrows():
        logger.info(f"  {row['Factor']}: {row['Importance']:.4f}")
    
    # Category-level analysis
    logger.info("\nFactor Category Analysis:")
    for category, importance in results['category_analysis'].items():
        logger.info(f"  {category}: {importance:.4f}")
    
    return results


if __name__ == "__main__":
    # Run example
    example_results = example_usage()
    logger.info("CausalForest implementation for A2AI completed successfully!")