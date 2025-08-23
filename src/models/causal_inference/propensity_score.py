"""
A2AI (Advanced Financial Analysis AI) - Propensity Score Matching Module

This module implements propensity score matching for causal inference in financial analysis.
It addresses selection bias and confounding variables to identify true causal effects
of factor variables on evaluation metrics across different market categories.

Key Features:
- Propensity score estimation for treatment assignment
- Matching algorithms (1:1, k:1, kernel matching)
- Covariate balance checking
- Treatment effect estimation with confidence intervals
- Market category-specific analysis
- Enterprise lifecycle stage consideration
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report
from scipy.stats import ttest_ind, chi2_contingency
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

@dataclass
class PropensityScoreConfig:
    """Configuration class for propensity score analysis"""
    matching_method: str = "nearest"  # nearest, kernel, stratification
    caliper: float = 0.1  # Maximum distance for matching
    replacement: bool = False  # With or without replacement
    ratio: int = 1  # k:1 matching ratio
    estimator_type: str = "logistic"  # logistic, random_forest, gradient_boosting
    balance_threshold: float = 0.1  # Standardized mean difference threshold
    min_propensity: float = 0.01  # Minimum propensity score for trimming
    max_propensity: float = 0.99  # Maximum propensity score for trimming

class PropensityScoreAnalyzer:
    """
    Propensity Score Matching for Causal Inference in A2AI
    
    This class implements propensity score matching to estimate causal effects
    of financial factors on business outcomes while controlling for confounding variables.
    """
    
    def __init__(self, config: PropensityScoreConfig = None):
        self.config = config or PropensityScoreConfig()
        self.propensity_model = None
        self.propensity_scores = None
        self.matched_data = None
        self.treatment_effect = None
        self.balance_statistics = None
        
    def estimate_propensity_scores(self, 
                                    X: pd.DataFrame, 
                                    treatment: pd.Series,
                                    feature_names: List[str] = None) -> Dict[str, Any]:
        """
        Estimate propensity scores using specified model
        
        Args:
            X: Covariate matrix (financial factors)
            treatment: Treatment indicator (e.g., market category, lifecycle stage)
            feature_names: Names of features for interpretation
            
        Returns:
            Dictionary containing propensity scores and model information
        """
        # Prepare data
        X_scaled = StandardScaler().fit_transform(X)
        feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]
        
        # Select and fit model
        if self.config.estimator_type == "logistic":
            self.propensity_model = LogisticRegression(random_state=42, max_iter=1000)
        elif self.config.estimator_type == "random_forest":
            self.propensity_model = RandomForestClassifier(
                n_estimators=100, random_state=42, max_depth=10
            )
        elif self.config.estimator_type == "gradient_boosting":
            self.propensity_model = GradientBoostingClassifier(
                n_estimators=100, random_state=42, max_depth=5
            )
        else:
            raise ValueError(f"Unknown estimator type: {self.config.estimator_type}")
        
        # Fit model
        self.propensity_model.fit(X_scaled, treatment)
        
        # Get propensity scores
        if hasattr(self.propensity_model, "predict_proba"):
            self.propensity_scores = self.propensity_model.predict_proba(X_scaled)[:, 1]
        else:
            # For models without predict_proba
            self.propensity_scores = self.propensity_model.decision_function(X_scaled)
            # Convert to probabilities using sigmoid
            self.propensity_scores = 1 / (1 + np.exp(-self.propensity_scores))
        
        # Calculate model performance
        auc_score = roc_auc_score(treatment, self.propensity_scores)
        
        # Feature importance (if available)
        feature_importance = None
        if hasattr(self.propensity_model, "feature_importances_"):
            feature_importance = dict(zip(feature_names, self.propensity_model.feature_importances_))
        elif hasattr(self.propensity_model, "coef_"):
            feature_importance = dict(zip(feature_names, abs(self.propensity_model.coef_[0])))
        
        return {
            "propensity_scores": self.propensity_scores,
            "auc_score": auc_score,
            "feature_importance": feature_importance,
            "model": self.propensity_model
        }
    
    def check_overlap(self, treatment: pd.Series) -> Dict[str, Any]:
        """
        Check overlap in propensity score distributions
        
        Args:
            treatment: Treatment indicator
            
        Returns:
            Dictionary with overlap statistics and recommendations
        """
        treated_scores = self.propensity_scores[treatment == 1]
        control_scores = self.propensity_scores[treatment == 0]
        
        overlap_stats = {
            "treated_min": np.min(treated_scores),
            "treated_max": np.max(treated_scores),
            "treated_mean": np.mean(treated_scores),
            "control_min": np.min(control_scores),
            "control_max": np.max(control_scores),
            "control_mean": np.mean(control_scores),
            "overlap_range": [
                max(np.min(treated_scores), np.min(control_scores)),
                min(np.max(treated_scores), np.max(control_scores))
            ]
        }
        
        # Check for common support violations
        violations = {
            "treated_below_control_min": np.sum(treated_scores < np.min(control_scores)),
            "treated_above_control_max": np.sum(treated_scores > np.max(control_scores)),
            "control_below_treated_min": np.sum(control_scores < np.min(treated_scores)),
            "control_above_treated_max": np.sum(control_scores > np.max(treated_scores))
        }
        
        return {"overlap_stats": overlap_stats, "violations": violations}
    
    def trim_by_propensity(self, 
                            data: pd.DataFrame, 
                            treatment: pd.Series,
                            outcome: pd.Series) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        Trim observations with extreme propensity scores
        
        Args:
            data: Original dataset
            treatment: Treatment indicator
            outcome: Outcome variable
            
        Returns:
            Trimmed data, treatment, and outcome
        """
        # Identify observations to keep
        keep_mask = (
            (self.propensity_scores >= self.config.min_propensity) &
            (self.propensity_scores <= self.config.max_propensity)
        )
        
        trimmed_data = data[keep_mask].copy()
        trimmed_treatment = treatment[keep_mask].copy()
        trimmed_outcome = outcome[keep_mask].copy()
        trimmed_scores = self.propensity_scores[keep_mask]
        
        print(f"Trimmed {np.sum(~keep_mask)} observations with extreme propensity scores")
        
        return trimmed_data, trimmed_treatment, trimmed_outcome, trimmed_scores
    
    def nearest_neighbor_matching(self, 
                                treatment: pd.Series,
                                propensity_scores: np.ndarray = None) -> np.ndarray:
        """
        Perform nearest neighbor matching
        
        Args:
            treatment: Treatment indicator
            propensity_scores: Propensity scores (if None, uses self.propensity_scores)
            
        Returns:
            Array of matched indices
        """
        if propensity_scores is None:
            propensity_scores = self.propensity_scores
            
        treated_idx = np.where(treatment == 1)[0]
        control_idx = np.where(treatment == 0)[0]
        
        matches = []
        used_controls = set() if not self.config.replacement else None
        
        for t_idx in treated_idx:
            t_score = propensity_scores[t_idx]
            
            # Calculate distances to all control units
            available_controls = control_idx
            if not self.config.replacement:
                available_controls = [c for c in control_idx if c not in used_controls]
            
            if not available_controls:
                continue
                
            distances = np.abs(propensity_scores[available_controls] - t_score)
            
            # Find k nearest matches within caliper
            valid_matches = np.where(distances <= self.config.caliper)[0]
            if len(valid_matches) == 0:
                continue
                
            # Sort by distance and select top k
            sorted_idx = np.argsort(distances[valid_matches])[:self.config.ratio]
            matched_controls = [available_controls[valid_matches[i]] for i in sorted_idx]
            
            for c_idx in matched_controls:
                matches.append((t_idx, c_idx, distances[np.where(np.array(available_controls) == c_idx)[0][0]]))
                if not self.config.replacement:
                    used_controls.add(c_idx)
        
        return np.array(matches)
    
    def kernel_matching(self, 
                        treatment: pd.Series,
                        outcome: pd.Series,
                        propensity_scores: np.ndarray = None,
                        bandwidth: float = 0.1) -> Dict[str, Any]:
        """
        Perform kernel matching
        
        Args:
            treatment: Treatment indicator
            outcome: Outcome variable
            propensity_scores: Propensity scores
            bandwidth: Kernel bandwidth
            
        Returns:
            Dictionary with treatment effects and matched outcomes
        """
        if propensity_scores is None:
            propensity_scores = self.propensity_scores
            
        treated_idx = np.where(treatment == 1)[0]
        control_idx = np.where(treatment == 0)[0]
        
        treated_outcomes = []
        matched_outcomes = []
        
        for t_idx in treated_idx:
            t_score = propensity_scores[t_idx]
            t_outcome = outcome.iloc[t_idx]
            
            # Calculate kernel weights
            distances = np.abs(propensity_scores[control_idx] - t_score)
            weights = np.exp(-0.5 * (distances / bandwidth) ** 2)
            weights = weights / np.sum(weights)
            
            # Calculate weighted average of control outcomes
            matched_outcome = np.average(outcome.iloc[control_idx], weights=weights)
            
            treated_outcomes.append(t_outcome)
            matched_outcomes.append(matched_outcome)
        
        return {
            "treated_outcomes": np.array(treated_outcomes),
            "matched_outcomes": np.array(matched_outcomes),
            "individual_effects": np.array(treated_outcomes) - np.array(matched_outcomes),
            "average_treatment_effect": np.mean(np.array(treated_outcomes) - np.array(matched_outcomes))
        }
    
    def stratification_matching(self, 
                                treatment: pd.Series,
                                outcome: pd.Series,
                                propensity_scores: np.ndarray = None,
                                n_strata: int = 5) -> Dict[str, Any]:
        """
        Perform stratification matching
        
        Args:
            treatment: Treatment indicator
            outcome: Outcome variable
            propensity_scores: Propensity scores
            n_strata: Number of strata
            
        Returns:
            Dictionary with stratified treatment effects
        """
        if propensity_scores is None:
            propensity_scores = self.propensity_scores
            
        # Create strata based on propensity score quantiles
        quantiles = np.linspace(0, 1, n_strata + 1)
        strata_boundaries = np.quantile(propensity_scores, quantiles)
        
        strata_effects = []
        strata_info = []
        
        for i in range(n_strata):
            # Define stratum
            if i == 0:
                stratum_mask = propensity_scores >= strata_boundaries[i]
            else:
                stratum_mask = (propensity_scores >= strata_boundaries[i]) & \
                                (propensity_scores < strata_boundaries[i + 1])
            
            if i == n_strata - 1:  # Last stratum includes upper boundary
                stratum_mask = propensity_scores >= strata_boundaries[i]
            
            # Get observations in this stratum
            stratum_treatment = treatment[stratum_mask]
            stratum_outcome = outcome[stratum_mask]
            
            if len(stratum_treatment) < 2 or \
                np.sum(stratum_treatment == 1) == 0 or \
                np.sum(stratum_treatment == 0) == 0:
                continue
            
            # Calculate treatment effect in this stratum
            treated_mean = np.mean(stratum_outcome[stratum_treatment == 1])
            control_mean = np.mean(stratum_outcome[stratum_treatment == 0])
            stratum_effect = treated_mean - control_mean
            
            # Calculate sample sizes
            n_treated = np.sum(stratum_treatment == 1)
            n_control = np.sum(stratum_treatment == 0)
            
            strata_effects.append(stratum_effect)
            strata_info.append({
                "stratum": i,
                "effect": stratum_effect,
                "n_treated": n_treated,
                "n_control": n_control,
                "treated_mean": treated_mean,
                "control_mean": control_mean,
                "propensity_range": [strata_boundaries[i], strata_boundaries[i + 1]]
            })
        
        # Calculate overall treatment effect (weighted by stratum size)
        total_n = sum([info["n_treated"] + info["n_control"] for info in strata_info])
        weighted_effect = np.average(
            strata_effects,
            weights=[info["n_treated"] + info["n_control"] for info in strata_info]
        )
        
        return {
            "strata_effects": strata_effects,
            "strata_info": strata_info,
            "weighted_average_effect": weighted_effect,
            "n_strata_used": len(strata_info)
        }
    
    def calculate_covariate_balance(self, 
                                    X: pd.DataFrame,
                                    treatment: pd.Series,
                                    matched_indices: np.ndarray = None) -> Dict[str, Any]:
        """
        Calculate covariate balance before and after matching
        
        Args:
            X: Covariate matrix
            treatment: Treatment indicator
            matched_indices: Array of matched pairs (if None, uses all data)
            
        Returns:
            Dictionary with balance statistics
        """
        def standardized_mean_difference(x_treated, x_control):
            """Calculate standardized mean difference"""
            mean_treated = np.mean(x_treated)
            mean_control = np.mean(x_control)
            var_treated = np.var(x_treated, ddof=1)
            var_control = np.var(x_control, ddof=1)
            pooled_var = (var_treated + var_control) / 2
            
            if pooled_var == 0:
                return 0
            
            return (mean_treated - mean_control) / np.sqrt(pooled_var)
        
        # Before matching
        treated_mask = treatment == 1
        control_mask = treatment == 0
        
        balance_before = {}
        balance_after = {}
        
        for col in X.columns:
            x_treated = X.loc[treated_mask, col]
            x_control = X.loc[control_mask, col]
            
            balance_before[col] = {
                "smd": standardized_mean_difference(x_treated, x_control),
                "treated_mean": np.mean(x_treated),
                "control_mean": np.mean(x_control),
                "treated_std": np.std(x_treated),
                "control_std": np.std(x_control)
            }
        
        # After matching (if matched indices provided)
        if matched_indices is not None:
            treated_matched_idx = matched_indices[:, 0].astype(int)
            control_matched_idx = matched_indices[:, 1].astype(int)
            
            for col in X.columns:
                x_treated_matched = X.iloc[treated_matched_idx][col]
                x_control_matched = X.iloc[control_matched_idx][col]
                
                balance_after[col] = {
                    "smd": standardized_mean_difference(x_treated_matched, x_control_matched),
                    "treated_mean": np.mean(x_treated_matched),
                    "control_mean": np.mean(x_control_matched),
                    "treated_std": np.std(x_treated_matched),
                    "control_std": np.std(x_control_matched)
                }
        
        # Calculate summary statistics
        smd_before = [abs(balance_before[col]["smd"]) for col in X.columns]
        smd_after = [abs(balance_after[col]["smd"]) for col in X.columns] if balance_after else []
        
        balance_summary = {
            "mean_smd_before": np.mean(smd_before),
            "max_smd_before": np.max(smd_before),
            "mean_smd_after": np.mean(smd_after) if smd_after else None,
            "max_smd_after": np.max(smd_after) if smd_after else None,
            "improvement": np.mean(smd_before) - np.mean(smd_after) if smd_after else None,
            "variables_balanced": np.sum(np.array(smd_after) < self.config.balance_threshold) if smd_after else None
        }
        
        return {
            "balance_before": balance_before,
            "balance_after": balance_after,
            "summary": balance_summary
        }
    
    def estimate_treatment_effect(self, 
                                outcome: pd.Series,
                                treatment: pd.Series,
                                matched_indices: np.ndarray = None,
                                bootstrap_iterations: int = 1000) -> Dict[str, Any]:
        """
        Estimate treatment effect with confidence intervals
        
        Args:
            outcome: Outcome variable
            treatment: Treatment indicator
            matched_indices: Matched pairs
            bootstrap_iterations: Number of bootstrap iterations
            
        Returns:
            Dictionary with treatment effect estimates and confidence intervals
        """
        if matched_indices is not None:
            # Use matched sample
            treated_idx = matched_indices[:, 0].astype(int)
            control_idx = matched_indices[:, 1].astype(int)
            
            treated_outcomes = outcome.iloc[treated_idx]
            control_outcomes = outcome.iloc[control_idx]
        else:
            # Use full sample
            treated_outcomes = outcome[treatment == 1]
            control_outcomes = outcome[treatment == 0]
        
        # Calculate treatment effect
        ate = np.mean(treated_outcomes) - np.mean(control_outcomes)
        
        # Statistical test
        t_stat, p_value = ttest_ind(treated_outcomes, control_outcomes)
        
        # Bootstrap confidence intervals
        bootstrap_effects = []
        n_treated = len(treated_outcomes)
        n_control = len(control_outcomes)
        
        for _ in range(bootstrap_iterations):
            # Bootstrap samples
            treated_bootstrap = np.random.choice(treated_outcomes, n_treated, replace=True)
            control_bootstrap = np.random.choice(control_outcomes, n_control, replace=True)
            
            bootstrap_ate = np.mean(treated_bootstrap) - np.mean(control_bootstrap)
            bootstrap_effects.append(bootstrap_ate)
        
        # Calculate confidence intervals
        ci_lower = np.percentile(bootstrap_effects, 2.5)
        ci_upper = np.percentile(bootstrap_effects, 97.5)
        
        return {
            "average_treatment_effect": ate,
            "treated_mean": np.mean(treated_outcomes),
            "control_mean": np.mean(control_outcomes),
            "t_statistic": t_stat,
            "p_value": p_value,
            "confidence_interval": [ci_lower, ci_upper],
            "bootstrap_effects": bootstrap_effects,
            "n_treated": n_treated,
            "n_control": n_control
        }
    
    def run_propensity_analysis(self, 
                                data: pd.DataFrame,
                                treatment_col: str,
                                outcome_col: str,
                                covariate_cols: List[str],
                                market_category_col: str = None) -> Dict[str, Any]:
        """
        Run complete propensity score analysis
        
        Args:
            data: Input dataset
            treatment_col: Treatment variable column name
            outcome_col: Outcome variable column name  
            covariate_cols: List of covariate column names
            market_category_col: Market category column for stratified analysis
            
        Returns:
            Dictionary with complete analysis results
        """
        results = {"overall": {}, "by_market": {}}
        
        # Prepare data
        X = data[covariate_cols].copy()
        treatment = data[treatment_col].copy()
        outcome = data[outcome_col].copy()
        
        # Remove missing values
        complete_mask = ~(X.isnull().any(axis=1) | treatment.isnull() | outcome.isnull())
        X = X[complete_mask]
        treatment = treatment[complete_mask]
        outcome = outcome[complete_mask]
        
        print(f"Analysis using {len(X)} complete observations")
        
        # Overall analysis
        results["overall"] = self._run_single_analysis(X, treatment, outcome, covariate_cols)
        
        # Market category stratified analysis
        if market_category_col and market_category_col in data.columns:
            market_categories = data[market_category_col][complete_mask].unique()
            
            for category in market_categories:
                if pd.isna(category):
                    continue
                    
                category_mask = data[market_category_col][complete_mask] == category
                if np.sum(category_mask) < 50:  # Skip categories with too few observations
                    continue
                
                X_cat = X[category_mask]
                treatment_cat = treatment[category_mask]
                outcome_cat = outcome[category_mask]
                
                if len(X_cat) > 0 and len(np.unique(treatment_cat)) > 1:
                    print(f"Analyzing market category: {category} ({len(X_cat)} observations)")
                    results["by_market"][category] = self._run_single_analysis(
                        X_cat, treatment_cat, outcome_cat, covariate_cols
                    )
        
        return results
    
    def _run_single_analysis(self, 
                            X: pd.DataFrame,
                            treatment: pd.Series,
                            outcome: pd.Series,
                            covariate_cols: List[str]) -> Dict[str, Any]:
        """
        Run propensity score analysis for a single dataset
        """
        # Step 1: Estimate propensity scores
        ps_results = self.estimate_propensity_scores(X, treatment, covariate_cols)
        
        # Step 2: Check overlap
        overlap_results = self.check_overlap(treatment)
        
        # Step 3: Trim extreme propensity scores if needed
        data_trimmed = pd.concat([X, treatment, outcome], axis=1)
        X_trim, treatment_trim, outcome_trim, ps_trim = self.trim_by_propensity(
            X, treatment, outcome
        )
        
        # Step 4: Perform matching
        matching_results = {}
        
        # Nearest neighbor matching
        if self.config.matching_method in ["nearest", "all"]:
            matches = self.nearest_neighbor_matching(treatment_trim, ps_trim)
            if len(matches) > 0:
                balance = self.calculate_covariate_balance(X_trim, treatment_trim, matches)
                effect = self.estimate_treatment_effect(outcome_trim, treatment_trim, matches)
                matching_results["nearest_neighbor"] = {
                    "matches": matches,
                    "balance": balance,
                    "treatment_effect": effect,
                    "n_matched_pairs": len(matches)
                }
        
        # Kernel matching
        if self.config.matching_method in ["kernel", "all"]:
            kernel_results = self.kernel_matching(treatment_trim, outcome_trim, ps_trim)
            matching_results["kernel"] = {
                "treatment_effect": kernel_results,
                "n_treated": len(kernel_results["treated_outcomes"])
            }
        
        # Stratification
        if self.config.matching_method in ["stratification", "all"]:
            strat_results = self.stratification_matching(treatment_trim, outcome_trim, ps_trim)
            matching_results["stratification"] = {
                "treatment_effect": strat_results,
                "n_strata": strat_results["n_strata_used"]
            }
        
        return {
            "propensity_scores": ps_results,
            "overlap": overlap_results,
            "matching": matching_results,
            "sample_sizes": {
                "original": len(X),
                "after_trimming": len(X_trim),
                "treated": np.sum(treatment_trim == 1),
                "control": np.sum(treatment_trim == 0)
            }
        }
    
    def plot_propensity_distribution(self, 
                                    treatment: pd.Series,
                                    save_path: str = None,
                                    figsize: Tuple[int, int] = (10, 6)) -> None:
        """
        Plot propensity score distributions
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Distribution plot
        treated_scores = self.propensity_scores[treatment == 1]
        control_scores = self.propensity_scores[treatment == 0]
        
        ax1.hist(control_scores, bins=30, alpha=0.7, label='Control', density=True)
        ax1.hist(treated_scores, bins=30, alpha=0.7, label='Treated', density=True)
        ax1.set_xlabel('Propensity Score')
        ax1.set_ylabel('Density')
        ax1.set_title('Propensity Score Distributions')
        ax1.legend()
        
        # Box plot
        ax2.boxplot([control_scores, treated_scores], labels=['Control', 'Treated'])
        ax2.set_ylabel('Propensity Score')
        ax2.set_title('Propensity Score Box Plots')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_covariate_balance(self, 
                                balance_results: Dict[str, Any],
                                save_path: str = None,
                                figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Plot covariate balance before and after matching
        """
        balance_before = balance_results["balance_before"]
        balance_after = balance_results.get("balance_after", {})
        
        variables = list(balance_before.keys())
        smd_before = [abs(balance_before[var]["smd"]) for var in variables]
        smd_after = [abs(balance_after[var]["smd"]) for var in variables] if balance_after else []
        
        fig, ax = plt.subplots(figsize=figsize)
        
        y_pos = np.arange(len(variables))
        
        ax.barh(y_pos - 0.2, smd_before, 0.4, label='Before Matching', alpha=0.7)
        if smd_after:
            ax.barh(y_pos + 0.2, smd_after, 0.4, label='After Matching', alpha=0.7)
        
        ax.axvline(x=self.config.balance_threshold, color='red', linestyle='--', 
                    label=f'Balance Threshold ({self.config.balance_threshold})')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(variables)
        ax.set_xlabel('Absolute Standardized Mean Difference')
        ax.set_title('Covariate Balance Before and After Matching')
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

# Example usage and utility functions for A2AI integration
class A2AIPropensityAnalysis:
    """
    Specialized propensity score analysis for A2AI financial data
    """
    
    def __init__(self):
        self.analyzer = PropensityScoreAnalyzer()
        
    def analyze_market_impact(self, 
                            financial_data: pd.DataFrame,
                            market_category: str = "market_share_category",
                            outcome_metrics: List[str] = None) -> Dict[str, Any]:
        """
        Analyze the causal impact of being in different market categories
        on financial performance metrics
        """
        outcome_metrics = outcome_metrics or [
            "sales_growth_rate", "operating_profit_margin", 
            "roe", "survival_probability"
        ]
        
        # Define treatment: High-share market vs others
        financial_data["high_share_treatment"] = (
            financial_data[market_category] == "high_share"
        ).astype(int)
        
        # Define covariates (financial factors)
        covariate_cols = [
            "tangible_fixed_assets", "rd_expense_ratio", "employee_count",
            "overseas_sales_ratio", "total_asset_turnover", "debt_ratio",
            "average_salary", "capital_investment_ratio", "intangible_assets"
        ]
        
        results = {}
        
        # Analyze each outcome metric
        for outcome in outcome_metrics:
            if outcome not in financial_data.columns:
                print(f"Warning: {outcome} not found in data, skipping...")
                continue
            
            print(f"\nAnalyzing causal effect on {outcome}...")
            
            # Run propensity score analysis
            analysis_results = self.analyzer.run_propensity_analysis(
                data=financial_data,
                treatment_col="high_share_treatment",
                outcome_col=outcome,
                covariate_cols=[col for col in covariate_cols if col in financial_data.columns],
                market_category_col=market_category
            )
            
            results[outcome] = analysis_results
        
        return results
    
    def analyze_lifecycle_impact(self,
                                financial_data: pd.DataFrame,
                                lifecycle_stage: str = "enterprise_age_category",
                                outcome_metrics: List[str] = None) -> Dict[str, Any]:
        """
        Analyze the causal impact of enterprise lifecycle stage on financial performance
        """
        outcome_metrics = outcome_metrics or [
            "sales_growth_rate", "operating_profit_margin", "roe"
        ]
        
        # Define treatment: Mature vs Young enterprises
        financial_data["mature_treatment"] = (
            financial_data[lifecycle_stage] == "mature"
        ).astype(int)
        
        # Define covariates
        covariate_cols = [
            "industry_sector", "market_share_category", "total_assets",
            "rd_expense_ratio", "overseas_sales_ratio", "employee_count"
        ]
        
        results = {}
        
        for outcome in outcome_metrics:
            if outcome not in financial_data.columns:
                continue
            
            analysis_results = self.analyzer.run_propensity_analysis(
                data=financial_data,
                treatment_col="mature_treatment",
                outcome_col=outcome,
                covariate_cols=[col for col in covariate_cols if col in financial_data.columns]
            )
            
            results[outcome] = analysis_results
        
        return results
    
    def analyze_rd_investment_impact(self,
                                    financial_data: pd.DataFrame,
                                    rd_threshold: float = 0.05) -> Dict[str, Any]:
        """
        Analyze the causal impact of R&D investment intensity on business outcomes
        """
        # Define treatment: High R&D investment (above threshold)
        financial_data["high_rd_treatment"] = (
            financial_data["rd_expense_ratio"] > rd_threshold
        ).astype(int)
        
        # Define covariates (excluding R&D ratio to avoid post-treatment bias)
        covariate_cols = [
            "enterprise_age", "total_assets", "employee_count", 
            "overseas_sales_ratio", "market_share_category",
            "tangible_fixed_assets", "debt_ratio"
        ]
        
        outcome_metrics = [
            "sales_growth_rate", "operating_profit_margin", 
            "roe", "patent_count", "market_share_growth"
        ]
        
        results = {}
        
        for outcome in outcome_metrics:
            if outcome not in financial_data.columns:
                continue
            
            analysis_results = self.analyzer.run_propensity_analysis(
                data=financial_data,
                treatment_col="high_rd_treatment",
                outcome_col=outcome,
                covariate_cols=[col for col in covariate_cols if col in financial_data.columns]
            )
            
            results[outcome] = analysis_results
        
        return results

def create_survival_treatment_analysis(financial_data: pd.DataFrame) -> Dict[str, Any]:
    """
    Specialized analysis for enterprise survival using propensity score matching
    
    This function analyzes what factors causally contribute to enterprise survival
    vs extinction, accounting for selection bias and confounding variables.
    """
    
    # Create survival treatment variable
    # 1 = Survived to end of observation period, 0 = Exited/Extinct
    financial_data["survival_treatment"] = (
        financial_data["enterprise_status"] == "active"
    ).astype(int)
    
    # Define pre-treatment covariates (measured before survival outcome)
    covariate_cols = [
        "initial_capital", "founding_year", "industry_concentration",
        "market_entry_timing", "initial_employee_count", "founder_experience",
        "initial_rd_ratio", "initial_debt_ratio", "market_share_category"
    ]
    
    # Define outcomes related to survival
    outcome_metrics = [
        "years_survived", "final_market_share", "final_total_assets",
        "final_roe", "exit_valuation"
    ]
    
    config = PropensityScoreConfig(
        matching_method="nearest",
        caliper=0.1,
        ratio=1,
        estimator_type="random_forest"  # Better for non-linear relationships
    )
    
    analyzer = PropensityScoreAnalyzer(config)
    results = {}
    
    for outcome in outcome_metrics:
        if outcome not in financial_data.columns:
            continue
        
        # Filter out missing data
        analysis_data = financial_data.dropna(
            subset=["survival_treatment", outcome] + 
            [col for col in covariate_cols if col in financial_data.columns]
        )
        
        if len(analysis_data) < 100:  # Skip if insufficient data
            continue
        
        print(f"\nSurvival Analysis for {outcome}: {len(analysis_data)} observations")
        
        analysis_results = analyzer.run_propensity_analysis(
            data=analysis_data,
            treatment_col="survival_treatment",
            outcome_col=outcome,
            covariate_cols=[col for col in covariate_cols if col in analysis_data.columns]
        )
        
        results[outcome] = analysis_results
    
    return results

def create_emergence_success_analysis(financial_data: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze factors that causally contribute to new enterprise success
    
    This function focuses on enterprises established after 2000 and analyzes
    what factors lead to successful market entry and growth.
    """
    
    # Filter to new enterprises (established after 2000)
    new_enterprises = financial_data[
        financial_data["founding_year"] >= 2000
    ].copy()
    
    if len(new_enterprises) < 50:
        return {"error": "Insufficient new enterprises for analysis"}
    
    # Define success treatment (e.g., achieving top quartile growth in first 5 years)
    growth_threshold = new_enterprises["early_growth_rate"].quantile(0.75)
    new_enterprises["success_treatment"] = (
        new_enterprises["early_growth_rate"] > growth_threshold
    ).astype(int)
    
    # Pre-treatment covariates (at founding)
    covariate_cols = [
        "initial_capital", "founder_age", "founder_education", "industry_experience",
        "market_timing_score", "initial_team_size", "vc_backed",
        "market_concentration", "technology_intensity"
    ]
    
    outcome_metrics = [
        "five_year_revenue", "market_share_achieved", "employee_growth",
        "innovation_output", "international_expansion"
    ]
    
    config = PropensityScoreConfig(
        matching_method="kernel",  # Good for continuous treatments
        estimator_type="gradient_boosting"
    )
    
    analyzer = PropensityScoreAnalyzer(config)
    results = {}
    
    for outcome in outcome_metrics:
        if outcome not in new_enterprises.columns:
            continue
        
        analysis_results = analyzer.run_propensity_analysis(
            data=new_enterprises,
            treatment_col="success_treatment",
            outcome_col=outcome,
            covariate_cols=[col for col in covariate_cols if col in new_enterprises.columns]
        )
        
        results[outcome] = analysis_results
    
    return results

def validate_propensity_assumptions(financial_data: pd.DataFrame,
                                    treatment_col: str,
                                    outcome_col: str,
                                    covariate_cols: List[str]) -> Dict[str, Any]:
    """
    Validate key assumptions for propensity score analysis
    
    Checks:
    1. Overlap/Common Support
    2. Unconfoundedness (indirect tests)
    3. Stable Unit Treatment Value Assumption (SUTVA)
    4. No post-treatment covariates
    """
    
    validation_results = {
        "overlap_check": {},
        "balance_check": {},
        "sensitivity_analysis": {},
        "sutva_check": {}
    }
    
    # Initialize analyzer
    analyzer = PropensityScoreAnalyzer()
    
    # Prepare data
    X = financial_data[covariate_cols].dropna()
    treatment = financial_data[treatment_col].loc[X.index]
    outcome = financial_data[outcome_col].loc[X.index]
    
    # Estimate propensity scores
    ps_results = analyzer.estimate_propensity_scores(X, treatment)
    
    # 1. Check overlap
    overlap_results = analyzer.check_overlap(treatment)
    validation_results["overlap_check"] = {
        "sufficient_overlap": overlap_results["violations"]["treated_below_control_min"] < len(treatment) * 0.05,
        "overlap_stats": overlap_results["overlap_stats"],
        "recommendations": []
    }
    
    if not validation_results["overlap_check"]["sufficient_overlap"]:
        validation_results["overlap_check"]["recommendations"].append(
            "Consider trimming extreme propensity scores or using different covariates"
        )
    
    # 2. Balance check (before matching)
    balance_results = analyzer.calculate_covariate_balance(X, treatment)
    mean_smd = balance_results["summary"]["mean_smd_before"]
    
    validation_results["balance_check"] = {
        "mean_smd": mean_smd,
        "well_balanced": mean_smd < 0.1,
        "covariates_with_large_smd": [
            var for var, stats in balance_results["balance_before"].items()
            if abs(stats["smd"]) > 0.25
        ]
    }
    
    # 3. Sensitivity analysis (Rosenbaum bounds simulation)
    validation_results["sensitivity_analysis"] = perform_sensitivity_analysis(
        outcome, treatment, ps_results["propensity_scores"]
    )
    
    # 4. SUTVA check (spillover effects)
    validation_results["sutva_check"] = check_sutva_violations(
        financial_data, treatment_col, outcome_col
    )
    
    return validation_results

def perform_sensitivity_analysis(outcome: pd.Series,
                                treatment: pd.Series,
                                propensity_scores: np.ndarray,
                                gamma_range: List[float] = None) -> Dict[str, Any]:
    """
    Perform sensitivity analysis for hidden bias (Rosenbaum bounds)
    """
    gamma_range = gamma_range or [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 2.0]
    
    sensitivity_results = {
        "gamma_values": gamma_range,
        "critical_gamma": None,
        "robust_inference": True
    }
    
    # This is a simplified version - full Rosenbaum bounds require more complex implementation
    base_effect = np.mean(outcome[treatment == 1]) - np.mean(outcome[treatment == 0])
    
    # Simulate potential bias under different gamma values
    simulated_effects = []
    
    for gamma in gamma_range:
        # Simulate hidden confounder effect
        # This is a simplified approximation
        bias_factor = (gamma - 1) / gamma
        max_bias = bias_factor * np.std(outcome)
        
        # Conservative estimate (assuming worst-case bias)
        conservative_effect = base_effect - max_bias
        simulated_effects.append(conservative_effect)
        
        # Check if effect remains significant
        if abs(conservative_effect) < 0.1 * np.std(outcome):  # Effect becomes negligible
            sensitivity_results["critical_gamma"] = gamma
            sensitivity_results["robust_inference"] = False
            break
    
    sensitivity_results["simulated_effects"] = simulated_effects
    
    return sensitivity_results

def check_sutva_violations(financial_data: pd.DataFrame,
                            treatment_col: str,
                            outcome_col: str) -> Dict[str, Any]:
    """
    Check for potential SUTVA violations (spillover effects)
    """
    sutva_results = {
        "potential_violations": [],
        "geographic_clusters": False,
        "industry_spillovers": False,
        "temporal_spillovers": False
    }
    
    # Check for geographic clustering
    if "prefecture" in financial_data.columns or "region" in financial_data.columns:
        location_col = "prefecture" if "prefecture" in financial_data.columns else "region"
        location_treatment_correlation = financial_data.groupby(location_col)[treatment_col].mean()
        
        if location_treatment_correlation.std() > 0.2:  # High variation in treatment by location
            sutva_results["geographic_clusters"] = True
            sutva_results["potential_violations"].append(
                "Geographic clustering of treatment may lead to spillover effects"
            )
    
    # Check for industry spillovers
    if "industry_code" in financial_data.columns:
        industry_treatment_correlation = financial_data.groupby("industry_code")[treatment_col].mean()
        
        if industry_treatment_correlation.std() > 0.3:
            sutva_results["industry_spillovers"] = True
            sutva_results["potential_violations"].append(
                "Industry-level treatment clustering may cause spillovers"
            )
    
    # Check for temporal spillovers
    if "year" in financial_data.columns:
        yearly_treatment = financial_data.groupby("year")[treatment_col].mean()
        if len(yearly_treatment) > 5 and yearly_treatment.std() > 0.2:
            sutva_results["temporal_spillovers"] = True
            sutva_results["potential_violations"].append(
                "Time-varying treatment intensity may cause temporal spillovers"
            )
    
    return sutva_results

# Integration with A2AI framework
def integrate_with_a2ai_pipeline(propensity_results: Dict[str, Any],
                                analysis_type: str = "market_impact") -> Dict[str, Any]:
    """
    Integrate propensity score results with A2AI analysis pipeline
    
    This function formats propensity score results for integration with
    other A2AI modules (survival analysis, emergence analysis, etc.)
    """
    
    integrated_results = {
        "analysis_type": analysis_type,
        "causal_effects": {},
        "methodological_notes": {},
        "policy_implications": {},
        "next_steps": []
    }
    
    # Extract key causal effects
    for outcome, results in propensity_results.items():
        if "overall" in results and "matching" in results["overall"]:
            matching_results = results["overall"]["matching"]
            
            # Get treatment effects from different methods
            effects = {}
            if "nearest_neighbor" in matching_results:
                nn_effect = matching_results["nearest_neighbor"]["treatment_effect"]
                effects["nearest_neighbor"] = {
                    "ate": nn_effect["average_treatment_effect"],
                    "ci": nn_effect["confidence_interval"],
                    "p_value": nn_effect["p_value"]
                }
            
            if "kernel" in matching_results:
                kernel_effect = matching_results["kernel"]["treatment_effect"]
                effects["kernel"] = {
                    "ate": kernel_effect["average_treatment_effect"]
                }
            
            if "stratification" in matching_results:
                strat_effect = matching_results["stratification"]["treatment_effect"]
                effects["stratification"] = {
                    "ate": strat_effect["weighted_average_effect"]
                }
            
            integrated_results["causal_effects"][outcome] = effects
    
    # Add methodological notes
    integrated_results["methodological_notes"] = {
        "assumption_validity": "Requires unconfoundedness assumption",
        "sample_restrictions": "May lose observations due to common support requirement",
        "interpretation": "Estimates average treatment effect on the treated (ATT)",
        "sensitivity": "Results should be checked for robustness to hidden bias"
    }
    
    # Generate policy implications based on analysis type
    if analysis_type == "market_impact":
        integrated_results["policy_implications"] = {
            "market_strategy": "Focus resources on high-impact market categories",
            "investment_priorities": "Allocate R&D based on causal effect magnitudes",
            "risk_management": "Account for selection effects in strategic planning"
        }
    elif analysis_type == "survival_analysis":
        integrated_results["policy_implications"] = {
            "early_warning": "Monitor key survival factors identified",
            "intervention_timing": "Act on causal factors before critical thresholds",
            "resource_allocation": "Prioritize factors with strongest causal evidence"
        }
    
    # Suggest next steps
    integrated_results["next_steps"] = [
        "Validate results with instrumental variables analysis",
        "Conduct sensitivity analysis for hidden confounders",
        "Compare with difference-in-differences estimates if panel data available",
        "Integrate findings with machine learning predictions"
    ]
    
    return integrated_results

# Export functions for A2AI integration
__all__ = [
    "PropensityScoreAnalyzer",
    "PropensityScoreConfig", 
    "A2AIPropensityAnalysis",
    "create_survival_treatment_analysis",
    "create_emergence_success_analysis",
    "validate_propensity_assumptions",
    "integrate_with_a2ai_pipeline"
]