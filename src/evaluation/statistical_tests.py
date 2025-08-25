"""
A2AI (Advanced Financial Analysis AI) - Statistical Tests Module

This module provides comprehensive statistical testing capabilities for:
1. Traditional financial analysis hypothesis testing
2. Survival analysis statistical tests
3. Causal inference validation
4. Market comparison tests
5. Lifecycle stage analysis tests
6. Emergence analysis tests
7. Bias detection and correction validation

Author: A2AI Development Team
Version: 1.0.0
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import chi2_contingency, fisher_exact, mannwhitneyu, kruskal
from scipy.stats import shapiro, levene, bartlett, anderson, jarque_bera
from scipy.stats import pearsonr, spearmanr, kendalltau
from lifelines.statistics import logrank_test, multivariate_logrank_test
from lifelines.statistics import pairwise_logrank_test
from statsmodels.stats.proportion import proportions_ztest, proportion_confint
from statsmodels.stats.weightstats import ttest_ind, ztest
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.diagnostic import het_white, het_breuschpagan
from statsmodels.stats.stattools import durbin_watson, jarque_bera as jb_test
from statsmodels.tsa.stattools import adfuller, kpss, coint
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings
from dataclasses import dataclass
from enum import Enum


class TestType(Enum):
    """Statistical test types supported by A2AI"""
    NORMALITY = "normality"
    INDEPENDENCE = "independence"
    HOMOGENEITY = "homogeneity"
    CORRELATION = "correlation"
    SURVIVAL = "survival"
    CAUSAL = "causal"
    MARKET_COMPARISON = "market_comparison"
    LIFECYCLE = "lifecycle"
    EMERGENCE = "emergence"
    TIME_SERIES = "time_series"


@dataclass
class TestResult:
    """Container for statistical test results"""
    test_name: str
    test_type: TestType
    statistic: float
    p_value: float
    degrees_of_freedom: Optional[int] = None
    critical_value: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    effect_size: Optional[float] = None
    interpretation: Optional[str] = None
    recommendations: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


class StatisticalTests:
    """
    Comprehensive statistical testing suite for A2AI financial analysis
    
    This class provides statistical tests for:
    - Traditional financial ratio analysis
    - Survival analysis validation
    - Causal inference testing
    - Market category comparisons
    - Corporate lifecycle analysis
    - Emergence pattern testing
    """
    
    def __init__(self, alpha: float = 0.05, correction_method: str = 'bonferroni'):
        """
        Initialize the statistical testing framework
        
        Args:
            alpha: Significance level (default 0.05)
            correction_method: Multiple testing correction method
        """
        self.alpha = alpha
        self.correction_method = correction_method
        self.test_results = []
        
    def reset_results(self):
        """Clear all stored test results"""
        self.test_results = []
    
    # =====================================
    # NORMALITY TESTS
    # =====================================
    
    def test_normality_comprehensive(self, data: np.ndarray, 
                                    variable_name: str = "variable") -> List[TestResult]:
        """
        Comprehensive normality testing using multiple tests
        Critical for A2AI financial ratio analysis
        
        Args:
            data: Data to test for normality
            variable_name: Name of the variable being tested
            
        Returns:
            List of TestResult objects for different normality tests
        """
        results = []
        
        # Remove NaN values
        clean_data = data[~np.isnan(data)]
        
        if len(clean_data) < 3:
            warnings.warn(f"Insufficient data for normality testing: {len(clean_data)} observations")
            return results
        
        # 1. Shapiro-Wilk Test (most powerful for small samples)
        if len(clean_data) <= 5000:  # Shapiro-Wilk limitation
            stat, p_val = shapiro(clean_data)
            result = TestResult(
                test_name="Shapiro-Wilk Test",
                test_type=TestType.NORMALITY,
                statistic=stat,
                p_value=p_val,
                interpretation=self._interpret_normality_test(p_val, "Shapiro-Wilk"),
                recommendations=self._get_normality_recommendations(p_val),
                metadata={"variable": variable_name, "n_obs": len(clean_data)}
            )
            results.append(result)
        
        # 2. Anderson-Darling Test
        ad_result = anderson(clean_data)
        # Anderson-Darling uses different critical values
        ad_p_value = None  # Anderson test doesn't directly provide p-value
        result = TestResult(
            test_name="Anderson-Darling Test",
            test_type=TestType.NORMALITY,
            statistic=ad_result.statistic,
            p_value=ad_p_value,
            critical_value=ad_result.critical_values[2],  # 5% significance level
            interpretation=f"Statistic: {ad_result.statistic:.4f}, Critical(5%): {ad_result.critical_values[2]:.4f}",
            metadata={"variable": variable_name, "significance_levels": ad_result.significance_level}
        )
        results.append(result)
        
        # 3. Jarque-Bera Test (good for large samples)
        if len(clean_data) >= 20:
            stat, p_val = jarque_bera(clean_data)
            result = TestResult(
                test_name="Jarque-Bera Test",
                test_type=TestType.NORMALITY,
                statistic=stat,
                p_value=p_val,
                degrees_of_freedom=2,
                interpretation=self._interpret_normality_test(p_val, "Jarque-Bera"),
                recommendations=self._get_normality_recommendations(p_val),
                metadata={"variable": variable_name, "suitable_for": "large_samples"}
            )
            results.append(result)
        
        self.test_results.extend(results)
        return results
    
    def _interpret_normality_test(self, p_value: float, test_name: str) -> str:
        """Interpret normality test results"""
        if p_value is None:
            return "Cannot determine normality (no p-value available)"
        
        if p_value > self.alpha:
            return f"{test_name}: Fail to reject H0. Data appears normally distributed (p={p_value:.4f})"
        else:
            return f"{test_name}: Reject H0. Data significantly deviates from normal distribution (p={p_value:.4f})"
    
    def _get_normality_recommendations(self, p_value: float) -> List[str]:
        """Get recommendations based on normality test results"""
        if p_value is None:
            return ["Consider visual inspection with Q-Q plots"]
        
        if p_value > self.alpha:
            return [
                "Use parametric tests (t-tests, ANOVA, Pearson correlation)",
                "Linear regression assumptions likely satisfied",
                "Normal confidence intervals appropriate"
            ]
        else:
            return [
                "Consider non-parametric tests (Mann-Whitney U, Kruskal-Wallis)",
                "Apply data transformations (log, sqrt, Box-Cox)",
                "Use robust statistical methods",
                "Bootstrap confidence intervals recommended"
            ]
    
    # =====================================
    # SURVIVAL ANALYSIS TESTS
    # =====================================
    
    def test_logrank(self, durations1: np.ndarray, events1: np.ndarray,
                    durations2: np.ndarray, events2: np.ndarray,
                    group1_name: str = "Group 1", group2_name: str = "Group 2") -> TestResult:
        """
        Log-rank test for comparing survival curves between two groups
        Critical for A2AI market category comparisons (high-share vs lost-share)
        
        Args:
            durations1, durations2: Survival times for each group
            events1, events2: Event indicators (1=death/failure, 0=censored)
            group1_name, group2_name: Group labels
            
        Returns:
            TestResult object with log-rank test results
        """
        try:
            results = logrank_test(durations1, durations2, events1, events2, alpha=self.alpha)
            
            effect_size = self._calculate_survival_effect_size(
                durations1, events1, durations2, events2
            )
            
            interpretation = self._interpret_survival_test(
                results.p_value, group1_name, group2_name
            )
            
            recommendations = self._get_survival_recommendations(results.p_value, effect_size)
            
            result = TestResult(
                test_name="Log-rank Test",
                test_type=TestType.SURVIVAL,
                statistic=results.test_statistic,
                p_value=results.p_value,
                degrees_of_freedom=1,
                effect_size=effect_size,
                interpretation=interpretation,
                recommendations=recommendations,
                metadata={
                    "group1": group1_name,
                    "group2": group2_name,
                    "n1": len(durations1),
                    "n2": len(durations2),
                    "events1": np.sum(events1),
                    "events2": np.sum(events2)
                }
            )
            
            self.test_results.append(result)
            return result
            
        except Exception as e:
            warnings.warn(f"Log-rank test failed: {str(e)}")
            return None
    
    def test_multivariate_logrank(self, duration_data: pd.DataFrame, 
                                    group_col: str, duration_col: str, 
                                    event_col: str) -> TestResult:
        """
        Multivariate log-rank test for comparing multiple groups
        Essential for comparing all three market categories in A2AI
        
        Args:
            duration_data: DataFrame with survival data
            group_col: Column name for group labels
            duration_col: Column name for durations
            event_col: Column name for event indicators
            
        Returns:
            TestResult object with multivariate log-rank test results
        """
        try:
            # Prepare data for multivariate test
            groups = duration_data[group_col].unique()
            
            results = multivariate_logrank_test(
                duration_data[duration_col], 
                duration_data[group_col], 
                duration_data[event_col],
                alpha=self.alpha
            )
            
            interpretation = f"Multivariate log-rank test across {len(groups)} market categories"
            if results.p_value < self.alpha:
                interpretation += f" shows significant differences (p={results.p_value:.4f})"
            else:
                interpretation += f" shows no significant differences (p={results.p_value:.4f})"
            
            result = TestResult(
                test_name="Multivariate Log-rank Test",
                test_type=TestType.SURVIVAL,
                statistic=results.test_statistic,
                p_value=results.p_value,
                degrees_of_freedom=len(groups) - 1,
                interpretation=interpretation,
                recommendations=self._get_multivariate_survival_recommendations(
                    results.p_value, groups
                ),
                metadata={
                    "groups": list(groups),
                    "n_groups": len(groups),
                    "total_n": len(duration_data)
                }
            )
            
            self.test_results.append(result)
            return result
            
        except Exception as e:
            warnings.warn(f"Multivariate log-rank test failed: {str(e)}")
            return None
    
    def _calculate_survival_effect_size(self, durations1: np.ndarray, events1: np.ndarray,
                                        durations2: np.ndarray, events2: np.ndarray) -> float:
        """Calculate effect size for survival analysis"""
        # Simple effect size based on median survival difference
        try:
            from lifelines import KaplanMeierFitter
            
            kmf1 = KaplanMeierFitter()
            kmf1.fit(durations1, events1)
            median1 = kmf1.median_survival_time_
            
            kmf2 = KaplanMeierFitter()
            kmf2.fit(durations2, events2)
            median2 = kmf2.median_survival_time_
            
            if pd.isna(median1) or pd.isna(median2):
                return None
            
            # Normalized difference
            pooled_median = (median1 + median2) / 2
            effect_size = abs(median1 - median2) / pooled_median if pooled_median > 0 else None
            
            return effect_size
            
        except:
            return None
    
    def _interpret_survival_test(self, p_value: float, group1: str, group2: str) -> str:
        """Interpret survival test results"""
        if p_value < self.alpha:
            return f"Significant difference in survival between {group1} and {group2} (p={p_value:.4f})"
        else:
            return f"No significant difference in survival between {group1} and {group2} (p={p_value:.4f})"
    
    def _get_survival_recommendations(self, p_value: float, effect_size: float) -> List[str]:
        """Get recommendations based on survival test results"""
        recommendations = []
        
        if p_value < self.alpha:
            recommendations.append("Significant survival difference detected - investigate underlying factors")
            if effect_size and effect_size > 0.5:
                recommendations.append("Large effect size - substantial practical significance")
            recommendations.append("Consider stratified analysis by additional variables")
            recommendations.append("Examine hazard ratios for specific time periods")
        else:
            recommendations.append("No significant survival difference - groups may be similar")
            recommendations.append("Consider combining groups for increased statistical power")
            recommendations.append("Examine if larger sample size needed")
        
        return recommendations
    
    def _get_multivariate_survival_recommendations(self, p_value: float, groups: List) -> List[str]:
        """Get recommendations for multivariate survival tests"""
        recommendations = []
        
        if p_value < self.alpha:
            recommendations.append("Significant differences among market categories detected")
            recommendations.append("Conduct pairwise comparisons to identify specific differences")
            recommendations.append("Consider post-hoc analysis with correction for multiple comparisons")
            recommendations.append("Examine which financial factors drive survival differences")
        else:
            recommendations.append("No significant differences among market categories")
            recommendations.append("Market categories may have similar survival patterns")
            recommendations.append("Consider alternative grouping strategies")
        
        return recommendations
    
    # =====================================
    # MARKET COMPARISON TESTS
    # =====================================
    
    def test_market_category_differences(self, data: pd.DataFrame, 
                                        category_col: str, 
                                        value_col: str,
                                        test_type: str = 'auto') -> TestResult:
        """
        Test for differences in financial metrics across market categories
        Core functionality for A2AI market comparison analysis
        
        Args:
            data: DataFrame with financial data
            category_col: Column with market categories (high_share, declining, lost)
            value_col: Column with financial metric values
            test_type: Type of test ('auto', 'anova', 'kruskal', 'welch')
            
        Returns:
            TestResult object with market comparison results
        """
        # Prepare data
        categories = data[category_col].unique()
        groups = [data[data[category_col] == cat][value_col].dropna().values 
                    for cat in categories]
        
        # Remove empty groups
        groups = [g for g in groups if len(g) > 0]
        categories = categories[:len(groups)]
        
        if len(groups) < 2:
            warnings.warn("Insufficient groups for comparison")
            return None
        
        # Determine appropriate test
        if test_type == 'auto':
            test_type = self._select_market_test(groups)
        
        if test_type == 'anova':
            return self._perform_anova(groups, categories, value_col)
        elif test_type == 'kruskal':
            return self._perform_kruskal_wallis(groups, categories, value_col)
        elif test_type == 'welch':
            return self._perform_welch_anova(groups, categories, value_col)
        else:
            raise ValueError(f"Unknown test type: {test_type}")
    
    def _select_market_test(self, groups: List[np.ndarray]) -> str:
        """Select appropriate test for market comparison"""
        # Check sample sizes
        min_size = min(len(g) for g in groups)
        if min_size < 10:
            return 'kruskal'  # Non-parametric for small samples
        
        # Check normality for each group
        normal_count = 0
        for group in groups:
            if len(group) >= 3:
                _, p_val = shapiro(group)
                if p_val > self.alpha:
                    normal_count += 1
        
        # Check homogeneity of variance if normality is satisfied
        if normal_count == len(groups):
            stat, p_val = levene(*groups)
            if p_val > self.alpha:
                return 'anova'  # Equal variances
            else:
                return 'welch'  # Unequal variances
        else:
            return 'kruskal'  # Non-parametric
    
    def _perform_anova(self, groups: List[np.ndarray], categories: List[str], 
                        metric_name: str) -> TestResult:
        """Perform one-way ANOVA"""
        stat, p_val = stats.f_oneway(*groups)
        
        # Calculate effect size (eta-squared)
        ss_between = sum(len(g) * (np.mean(g) - np.mean(np.concatenate(groups)))**2 
                        for g in groups)
        ss_total = sum(np.sum((g - np.mean(np.concatenate(groups)))**2) for g in groups)
        eta_squared = ss_between / ss_total if ss_total > 0 else 0
        
        result = TestResult(
            test_name="One-way ANOVA",
            test_type=TestType.MARKET_COMPARISON,
            statistic=stat,
            p_value=p_val,
            degrees_of_freedom=len(groups) - 1,
            effect_size=eta_squared,
            interpretation=self._interpret_market_test(p_val, "ANOVA", categories),
            recommendations=self._get_market_test_recommendations(p_val, eta_squared, "parametric"),
            metadata={
                "metric": metric_name,
                "categories": categories,
                "group_sizes": [len(g) for g in groups],
                "group_means": [np.mean(g) for g in groups]
            }
        )
        
        self.test_results.append(result)
        return result
    
    def _perform_kruskal_wallis(self, groups: List[np.ndarray], categories: List[str],
                                metric_name: str) -> TestResult:
        """Perform Kruskal-Wallis test"""
        stat, p_val = kruskal(*groups)
        
        # Calculate effect size (epsilon-squared)
        n = sum(len(g) for g in groups)
        epsilon_squared = (stat - len(groups) + 1) / (n - len(groups)) if n > len(groups) else 0
        
        result = TestResult(
            test_name="Kruskal-Wallis Test",
            test_type=TestType.MARKET_COMPARISON,
            statistic=stat,
            p_value=p_val,
            degrees_of_freedom=len(groups) - 1,
            effect_size=epsilon_squared,
            interpretation=self._interpret_market_test(p_val, "Kruskal-Wallis", categories),
            recommendations=self._get_market_test_recommendations(p_val, epsilon_squared, "non_parametric"),
            metadata={
                "metric": metric_name,
                "categories": categories,
                "group_sizes": [len(g) for g in groups],
                "group_medians": [np.median(g) for g in groups]
            }
        )
        
        self.test_results.append(result)
        return result
    
    def _perform_welch_anova(self, groups: List[np.ndarray], categories: List[str],
                            metric_name: str) -> TestResult:
        """Perform Welch's ANOVA (unequal variances)"""
        # Note: scipy doesn't have built-in Welch ANOVA, using approximation
        from scipy.stats import f
        
        k = len(groups)
        n_total = sum(len(g) for g in groups)
        
        # Welch's F-statistic calculation
        means = [np.mean(g) for g in groups]
        vars = [np.var(g, ddof=1) for g in groups]
        ns = [len(g) for g in groups]
        
        # Weights
        weights = [n/v for n, v in zip(ns, vars)]
        weight_sum = sum(weights)
        
        # Weighted grand mean
        grand_mean = sum(w * m for w, m in zip(weights, means)) / weight_sum
        
        # Welch's F-statistic
        numerator = sum(w * (m - grand_mean)**2 for w, m in zip(weights, means)) / (k - 1)
        
        # Denominator calculation (more complex)
        denom_terms = [(1 - w/weight_sum)**2 / (n - 1) for w, n in zip(weights, ns)]
        denominator = 1 + (2 * (k - 2) / (k**2 - 1)) * sum(denom_terms)
        
        welch_f = numerator / denominator if denominator > 0 else 0
        
        # Approximate p-value
        df1 = k - 1
        df2 = 1 / (3 * sum(denom_terms) / (k**2 - 1)) if sum(denom_terms) > 0 else n_total - k
        
        p_val = 1 - f.cdf(welch_f, df1, df2) if welch_f > 0 else 1.0
        
        result = TestResult(
            test_name="Welch's ANOVA",
            test_type=TestType.MARKET_COMPARISON,
            statistic=welch_f,
            p_value=p_val,
            degrees_of_freedom=int(df2),
            interpretation=self._interpret_market_test(p_val, "Welch ANOVA", categories),
            recommendations=self._get_market_test_recommendations(p_val, None, "welch"),
            metadata={
                "metric": metric_name,
                "categories": categories,
                "group_sizes": ns,
                "group_means": means,
                "group_variances": vars
            }
        )
        
        self.test_results.append(result)
        return result
    
    def _interpret_market_test(self, p_value: float, test_name: str, 
                                categories: List[str]) -> str:
        """Interpret market comparison test results"""
        if p_value < self.alpha:
            return f"{test_name}: Significant differences among market categories {categories} (p={p_value:.4f})"
        else:
            return f"{test_name}: No significant differences among market categories {categories} (p={p_value:.4f})"
    
    def _get_market_test_recommendations(self, p_value: float, effect_size: Optional[float],
                                        test_type: str) -> List[str]:
        """Get recommendations based on market comparison results"""
        recommendations = []
        
        if p_value < self.alpha:
            recommendations.append("Significant differences detected among market categories")
            recommendations.append("Perform post-hoc pairwise comparisons")
            
            if effect_size and effect_size > 0.1:
                recommendations.append("Large effect size indicates practically significant differences")
            
            if test_type == "non_parametric":
                recommendations.append("Use Mann-Whitney U tests for pairwise comparisons")
            else:
                recommendations.append("Use Tukey's HSD or Bonferroni correction for pairwise tests")
                
            recommendations.append("Investigate financial factors driving category differences")
        else:
            recommendations.append("No significant differences among market categories")
            recommendations.append("Consider combining categories or examining subgroups")
            recommendations.append("Check if sample sizes are adequate for detecting differences")
        
        return recommendations
    
    # =====================================
    # CAUSAL INFERENCE TESTS
    # =====================================
    
    def test_difference_in_differences_assumptions(self, data: pd.DataFrame,
                                                    group_col: str, time_col: str, 
                                                    outcome_col: str) -> Dict[str, TestResult]:
        """
        Test assumptions for difference-in-differences analysis
        Critical for A2AI causal inference of corporate events (M&A, spinoffs)
        
        Args:
            data: Panel data with treatment and control groups
            group_col: Column indicating treatment group
            time_col: Time period column
            outcome_col: Outcome variable
            
        Returns:
            Dictionary of TestResult objects for each assumption test
        """
        results = {}
        
        # 1. Parallel Trends Test (pre-treatment)
        pre_treatment = data[data[time_col] < 0]  # Assuming 0 is treatment time
        
        if len(pre_treatment) > 0:
            results['parallel_trends'] = self._test_parallel_trends(
                pre_treatment, group_col, time_col, outcome_col
            )
        
        # 2. Balance Test (pre-treatment characteristics)
        results['balance_test'] = self._test_pretreatment_balance(
            pre_treatment, group_col, outcome_col
        )
        
        # 3. No Anticipation Test
        results['no_anticipation'] = self._test_no_anticipation(
            data, group_col, time_col, outcome_col
        )
        
        self.test_results.extend(results.values())
        return results
    
    def _test_parallel_trends(self, data: pd.DataFrame, group_col: str, 
                            time_col: str, outcome_col: str) -> TestResult:
        """Test parallel trends assumption"""
        # Simple approach: test if time trends differ between groups
        treatment_group = data[data[group_col] == 1][outcome_col].values
        control_group = data[data[group_col] == 0][outcome_col].values
        
        # Calculate trend differences (simplified)
        if len(treatment_group) > 1 and len(control_group) > 1:
            treatment_trend = np.polyfit(range(len(treatment_group)), treatment_group, 1)[0]
            control_trend = np.polyfit(range(len(control_group)), control_group, 1)[0]
            
            trend_diff = abs(treatment_trend - control_trend)
            
            # Use Mann-Whitney U test for trend comparison
            stat, p_val = mannwhitneyu(treatment_group, control_group, alternative='two-sided')
            
            interpretation = "Parallel trends assumption "
            if p_val > self.alpha:
                interpretation += f"appears satisfied (p={p_val:.4f})"
            else:
                interpretation += f"may be violated (p={p_val:.4f})"
            
            return TestResult(
                test_name="Parallel Trends Test",
                test_type=TestType.CAUSAL,
                statistic=stat,
                p_value=p_val,
                interpretation=interpretation,
                metadata={
                    "treatment_trend": treatment_trend,
                    "control_trend": control_trend,
                    "trend_difference": trend_diff
                }
            )
        
        return TestResult(
            test_name="Parallel Trends Test",
            test_type=TestType.CAUSAL,
            statistic=np.nan,
            p_value=np.nan,
            interpretation="Insufficient data for parallel trends test"
        )
    
    def _test_pretreatment_balance(self, data: pd.DataFrame, 
                                    group_col: str, outcome_col: str) -> TestResult:
        """Test balance in pre-treatment characteristics"""
        if len(data) == 0:
            return TestResult(
                test_name="Pre-treatment Balance Test",
                test_type=TestType.CAUSAL,
                statistic=np.nan,
                p_value=np.nan,
                interpretation="No pre-treatment data available"
            )
        
        treatment_group = data[data[group_col] == 1][outcome_col].values
        control_group = data[data[group_col] == 0][outcome_col].values
        
        if len(treatment_group) == 0 or len(control_group) == 0:
            return TestResult(
                test_name="Pre-treatment Balance Test",
                test_type=TestType.CAUSAL,
                statistic=np.nan,
                p_value=np.nan,
                interpretation="Insufficient group data for balance test"
            )
        
        # Use t-test for balance
        stat, p_val = ttest_ind(treatment_group, control_group)
        
        interpretation = "Pre-treatment balance "
        if p_val > self.alpha:
            interpretation += f"appears adequate (p={p_val:.4f})"
        else:
            interpretation += f"may be problematic - groups differ significantly (p={p_val:.4f})"
        
        return TestResult(
            test_name="Pre-treatment Balance Test",
            test_type=TestType.CAUSAL,
            statistic=stat,
            p_value=p_val,
            interpretation=interpretation,
            recommendations=self._get_balance_recommendations(p_val),
            metadata={
                "treatment_mean": np.mean(treatment_group),
                "control_mean": np.mean(control_group),
                "treatment_n": len(treatment_group),
                "control_n": len(control_group)
            }
        )
    
    def _test_no_anticipation(self, data: pd.DataFrame, group_col: str,
                            time_col: str, outcome_col: str) -> TestResult:
        """Test no anticipation assumption (no pre-treatment effects)"""
        # Look at immediate pre-treatment period
        pre_treatment = data[(data[time_col] >= -2) & (data[time_col] < 0)]
        
        if len(pre_treatment) == 0:
            return TestResult(
                test_name="No Anticipation Test",
                test_type=TestType.CAUSAL,
                statistic=np.nan,
                p_value=np.nan,
                interpretation="Insufficient pre-treatment data for anticipation test"
            )
        
        # Test for differences in immediate pre-treatment period
        treatment_group = pre_treatment[pre_treatment[group_col] == 1][outcome_col].values
        control_group = pre_treatment[pre_treatment[group_col] == 0][outcome_col].values
        
        if len(treatment_group) == 0 or len(control_group) == 0:
            return TestResult(
                test_name="No Anticipation Test",
                test_type=TestType.CAUSAL,
                statistic=np.nan,
                p_value=np.nan,
                interpretation="Insufficient group data in pre-treatment period"
            )
        
        stat, p_val = ttest_ind(treatment_group, control_group)
        
        interpretation = "No anticipation assumption "
        if p_val > self.alpha:
            interpretation += f"appears satisfied (p={p_val:.4f})"
        else:
            interpretation += f"may be violated - anticipation effects detected (p={p_val:.4f})"
        
        return TestResult(
            test_name="No Anticipation Test",
            test_type=TestType.CAUSAL,
            statistic=stat,
            p_value=p_val,
            interpretation=interpretation,
            recommendations=self._get_anticipation_recommendations(p_val)
        )
    
    def _get_balance_recommendations(self, p_value: float) -> List[str]:
        """Get recommendations based on balance test results"""
        if p_value > self.alpha:
            return [
                "Pre-treatment balance appears adequate",
                "Proceed with difference-in-differences analysis",
                "Consider additional covariate balance checks"
            ]
        else:
            return [
                "Pre-treatment imbalance detected",
                "Consider matching or weighting methods",
                "Add control variables to DID specification",
                "Examine robustness with alternative control groups"
            ]
    
    def _get_anticipation_recommendations(self, p_value: float) -> List[str]:
        """Get recommendations based on anticipation test results"""
        if p_value > self.alpha:
            return [
                "No anticipation effects detected",
                "Proceed with standard DID analysis",
                "Treatment timing appears exogenous"
            ]
        else:
            return [
                "Anticipation effects may be present",
                "Consider alternative treatment timing",
                "Examine lead indicators in DID specification",
                "Use instrumental variables if available"
            ]
    
    # =====================================
    # LIFECYCLE ANALYSIS TESTS
    # =====================================
    
    def test_lifecycle_stage_transitions(self, data: pd.DataFrame, 
                                        company_col: str, time_col: str,
                                        stage_col: str) -> TestResult:
        """
        Test for significant transitions between corporate lifecycle stages
        Essential for A2AI lifecycle dynamics analysis
        
        Args:
            data: Panel data with lifecycle stages
            company_col: Company identifier column
            time_col: Time period column
            stage_col: Lifecycle stage column (e.g., startup, growth, mature, decline)
            
        Returns:
            TestResult object with transition analysis
        """
        # Create transition matrix
        companies = data[company_col].unique()
        transitions = []
        
        for company in companies:
            company_data = data[data[company_col] == company].sort_values(time_col)
            stages = company_data[stage_col].values
            
            for i in range(len(stages) - 1):
                transitions.append((stages[i], stages[i + 1]))
        
        if len(transitions) == 0:
            return TestResult(
                test_name="Lifecycle Stage Transitions Test",
                test_type=TestType.LIFECYCLE,
                statistic=np.nan,
                p_value=np.nan,
                interpretation="No stage transitions observed in data"
            )
        
        # Convert to contingency table
        from collections import Counter
        transition_counts = Counter(transitions)
        
        # Get unique stages
        all_stages = sorted(set(data[stage_col].unique()))
        
        # Create contingency matrix
        transition_matrix = np.zeros((len(all_stages), len(all_stages)))
        
        for i, from_stage in enumerate(all_stages):
            for j, to_stage in enumerate(all_stages):
                transition_matrix[i, j] = transition_counts.get((from_stage, to_stage), 0)
        
        # Chi-square test for independence
        chi2, p_val, dof, expected = chi2_contingency(transition_matrix + 1e-8)  # Add small constant
        
        interpretation = f"Lifecycle stage transitions show "
        if p_val < self.alpha:
            interpretation += f"significant non-random patterns (χ²={chi2:.4f}, p={p_val:.4f})"
        else:
            interpretation += f"random transition patterns (χ²={chi2:.4f}, p={p_val:.4f})"
        
        result = TestResult(
            test_name="Lifecycle Stage Transitions Test",
            test_type=TestType.LIFECYCLE,
            statistic=chi2,
            p_value=p_val,
            degrees_of_freedom=dof,
            interpretation=interpretation,
            recommendations=self._get_lifecycle_recommendations(p_val, transition_matrix, all_stages),
            metadata={
                "stages": all_stages,
                "n_transitions": len(transitions),
                "n_companies": len(companies),
                "transition_matrix": transition_matrix.tolist()
            }
        )
        
        self.test_results.append(result)
        return result
    
    def _get_lifecycle_recommendations(self, p_value: float, 
                                        transition_matrix: np.ndarray,
                                        stages: List[str]) -> List[str]:
        """Get recommendations based on lifecycle analysis"""
        recommendations = []
        
        if p_value < self.alpha:
            recommendations.append("Non-random lifecycle patterns detected")
            recommendations.append("Examine specific transition probabilities")
            recommendations.append("Identify factors driving stage transitions")
            
            # Find most common transitions
            max_idx = np.unravel_index(np.argmax(transition_matrix), transition_matrix.shape)
            most_common = f"{stages[max_idx[0]]} → {stages[max_idx[1]]}"
            recommendations.append(f"Most common transition: {most_common}")
            
        else:
            recommendations.append("Lifecycle transitions appear random")
            recommendations.append("Consider alternative stage definitions")
            recommendations.append("Examine if external factors influence transitions")
        
        return recommendations
    
    # =====================================
    # EMERGENCE ANALYSIS TESTS
    # =====================================
    
    def test_emergence_success_factors(self, data: pd.DataFrame,
                                        success_col: str, factor_cols: List[str]) -> List[TestResult]:
        """
        Test which factors significantly predict emergence success
        Critical for A2AI new company analysis
        
        Args:
            data: DataFrame with emergence data
            success_col: Binary success indicator (1=success, 0=failure)
            factor_cols: List of potential success factor columns
            
        Returns:
            List of TestResult objects for each factor
        """
        results = []
        
        for factor in factor_cols:
            # Skip if factor has insufficient data
            factor_data = data[[success_col, factor]].dropna()
            if len(factor_data) < 10:
                continue
            
            # Separate successful and unsuccessful companies
            success_group = factor_data[factor_data[success_col] == 1][factor].values
            failure_group = factor_data[factor_data[success_col] == 0][factor].values
            
            if len(success_group) == 0 or len(failure_group) == 0:
                continue
            
            # Choose appropriate test
            if self._is_continuous(factor_data[factor]):
                result = self._test_continuous_emergence_factor(
                    success_group, failure_group, factor
                )
            else:
                result = self._test_categorical_emergence_factor(
                    factor_data, success_col, factor
                )
            
            if result:
                results.append(result)
        
        self.test_results.extend(results)
        return results
    
    def _is_continuous(self, series: pd.Series) -> bool:
        """Check if variable is continuous"""
        return series.dtype in ['float64', 'float32', 'int64', 'int32'] and series.nunique() > 10
    
    def _test_continuous_emergence_factor(self, success_group: np.ndarray,
                                        failure_group: np.ndarray, 
                                        factor_name: str) -> TestResult:
        """Test continuous factor for emergence success"""
        # Use Mann-Whitney U test (robust to non-normality)
        stat, p_val = mannwhitneyu(success_group, failure_group, alternative='two-sided')
        
        # Calculate effect size (rank-biserial correlation)
        n1, n2 = len(success_group), len(failure_group)
        effect_size = (stat - (n1 * n2 / 2)) / (n1 * n2) if (n1 * n2) > 0 else 0
        
        interpretation = f"Factor '{factor_name}' "
        if p_val < self.alpha:
            interpretation += f"significantly predicts emergence success (p={p_val:.4f})"
            if np.median(success_group) > np.median(failure_group):
                interpretation += " - higher values associated with success"
            else:
                interpretation += " - lower values associated with success"
        else:
            interpretation += f"does not significantly predict emergence success (p={p_val:.4f})"
        
        return TestResult(
            test_name="Emergence Success Factor Test (Continuous)",
            test_type=TestType.EMERGENCE,
            statistic=stat,
            p_value=p_val,
            effect_size=effect_size,
            interpretation=interpretation,
            recommendations=self._get_emergence_factor_recommendations(
                p_val, effect_size, factor_name, "continuous"
            ),
            metadata={
                "factor": factor_name,
                "success_median": np.median(success_group),
                "failure_median": np.median(failure_group),
                "success_n": n1,
                "failure_n": n2
            }
        )
    
    def _test_categorical_emergence_factor(self, data: pd.DataFrame,
                                            success_col: str, factor_name: str) -> TestResult:
        """Test categorical factor for emergence success"""
        # Create contingency table
        contingency_table = pd.crosstab(data[factor_name], data[success_col])
        
        if contingency_table.size < 4:  # 2x2 minimum
            return None
        
        # Chi-square test
        chi2, p_val, dof, expected = chi2_contingency(contingency_table)
        
        # Cramér's V for effect size
        n = contingency_table.sum().sum()
        cramers_v = np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))
        
        interpretation = f"Categorical factor '{factor_name}' "
        if p_val < self.alpha:
            interpretation += f"significantly associated with emergence success (p={p_val:.4f})"
        else:
            interpretation += f"not significantly associated with emergence success (p={p_val:.4f})"
        
        return TestResult(
            test_name="Emergence Success Factor Test (Categorical)",
            test_type=TestType.EMERGENCE,
            statistic=chi2,
            p_value=p_val,
            degrees_of_freedom=dof,
            effect_size=cramers_v,
            interpretation=interpretation,
            recommendations=self._get_emergence_factor_recommendations(
                p_val, cramers_v, factor_name, "categorical"
            ),
            metadata={
                "factor": factor_name,
                "contingency_table": contingency_table.to_dict(),
                "total_n": n
            }
        )
    
    def _get_emergence_factor_recommendations(self, p_value: float, effect_size: float,
                                            factor_name: str, factor_type: str) -> List[str]:
        """Get recommendations for emergence factor analysis"""
        recommendations = []
        
        if p_value < self.alpha:
            recommendations.append(f"'{factor_name}' is a significant predictor of emergence success")
            
            if effect_size and effect_size > 0.3:
                recommendations.append("Large effect size indicates strong practical significance")
            elif effect_size and effect_size > 0.1:
                recommendations.append("Medium effect size indicates moderate practical significance")
            
            recommendations.append("Include this factor in emergence prediction models")
            recommendations.append("Examine interaction effects with other significant factors")
            
        else:
            recommendations.append(f"'{factor_name}' is not a significant predictor")
            recommendations.append("Consider removing from prediction models")
            recommendations.append("May be useful in interaction with other factors")
        
        return recommendations
    
    # =====================================
    # TIME SERIES TESTS
    # =====================================
    
    def test_stationarity(self, series: pd.Series, variable_name: str = "series") -> List[TestResult]:
        """
        Test time series stationarity using multiple tests
        Essential for A2AI time series analysis of financial metrics
        
        Args:
            series: Time series data
            variable_name: Name of the variable
            
        Returns:
            List of TestResult objects for stationarity tests
        """
        results = []
        
        # Remove NaN values
        clean_series = series.dropna()
        
        if len(clean_series) < 10:
            warnings.warn(f"Insufficient data for stationarity tests: {len(clean_series)} observations")
            return results
        
        # 1. Augmented Dickey-Fuller Test
        adf_stat, adf_pval, adf_lags, adf_nobs, adf_crit, adf_icbest = adfuller(clean_series, autolag='AIC')
        
        adf_result = TestResult(
            test_name="Augmented Dickey-Fuller Test",
            test_type=TestType.TIME_SERIES,
            statistic=adf_stat,
            p_value=adf_pval,
            critical_value=adf_crit['5%'],
            interpretation=self._interpret_adf_test(adf_stat, adf_pval, adf_crit['5%']),
            recommendations=self._get_stationarity_recommendations(adf_pval, "ADF"),
            metadata={
                "variable": variable_name,
                "lags_used": adf_lags,
                "n_observations": adf_nobs,
                "critical_values": adf_crit
            }
        )
        results.append(adf_result)
        
        # 2. KPSS Test (complementary to ADF)
        kpss_stat, kpss_pval, kpss_lags, kpss_crit = kpss(clean_series, regression='c', nlags='auto')
        
        kpss_result = TestResult(
            test_name="KPSS Test",
            test_type=TestType.TIME_SERIES,
            statistic=kpss_stat,
            p_value=kpss_pval,
            critical_value=kpss_crit['5%'],
            interpretation=self._interpret_kpss_test(kpss_stat, kpss_pval, kpss_crit['5%']),
            recommendations=self._get_stationarity_recommendations(kpss_pval, "KPSS"),
            metadata={
                "variable": variable_name,
                "lags_used": kpss_lags,
                "critical_values": kpss_crit
            }
        )
        results.append(kpss_result)
        
        self.test_results.extend(results)
        return results
    
    def _interpret_adf_test(self, statistic: float, p_value: float, critical_value: float) -> str:
        """Interpret ADF test results"""
        if p_value < self.alpha:
            return f"ADF test rejects H0: Series appears stationary (stat={statistic:.4f}, p={p_value:.4f})"
        else:
            return f"ADF test fails to reject H0: Series appears non-stationary (stat={statistic:.4f}, p={p_value:.4f})"
    
    def _interpret_kpss_test(self, statistic: float, p_value: float, critical_value: float) -> str:
        """Interpret KPSS test results"""
        if p_value < self.alpha:
            return f"KPSS test rejects H0: Series appears non-stationary (stat={statistic:.4f}, p={p_value:.4f})"
        else:
            return f"KPSS test fails to reject H0: Series appears stationary (stat={statistic:.4f}, p={p_value:.4f})"
    
    def _get_stationarity_recommendations(self, p_value: float, test_type: str) -> List[str]:
        """Get recommendations based on stationarity tests"""
        recommendations = []
        
        if test_type == "ADF":
            if p_value < self.alpha:
                recommendations.append("Series appears stationary - suitable for standard time series analysis")
                recommendations.append("Can use levels in regression analysis")
            else:
                recommendations.append("Series appears non-stationary - consider differencing")
                recommendations.append("Use cointegration analysis for relationships")
                recommendations.append("Apply first differences or other transformations")
        
        elif test_type == "KPSS":
            if p_value < self.alpha:
                recommendations.append("KPSS indicates non-stationarity")
                recommendations.append("Consider trend stationarity vs difference stationarity")
            else:
                recommendations.append("KPSS supports stationarity")
                recommendations.append("Series suitable for standard analysis")
        
        return recommendations
    
    # =====================================
    # BIAS DETECTION TESTS
    # =====================================
    
    def test_survivorship_bias(self, full_data: pd.DataFrame, survivor_data: pd.DataFrame,
                                metric_cols: List[str]) -> List[TestResult]:
        """
        Test for survivorship bias in financial analysis
        Core functionality for A2AI bias correction validation
        
        Args:
            full_data: Complete dataset including extinct companies
            survivor_data: Dataset with only surviving companies
            metric_cols: Financial metrics to test for bias
            
        Returns:
            List of TestResult objects for survivorship bias tests
        """
        results = []
        
        for metric in metric_cols:
            if metric not in full_data.columns or metric not in survivor_data.columns:
                continue
            
            full_values = full_data[metric].dropna().values
            survivor_values = survivor_data[metric].dropna().values
            
            if len(full_values) < 10 or len(survivor_values) < 10:
                continue
            
            # Test for significant differences between full and survivor samples
            stat, p_val = mannwhitneyu(full_values, survivor_values, alternative='two-sided')
            
            # Calculate bias magnitude
            full_mean = np.mean(full_values)
            survivor_mean = np.mean(survivor_values)
            bias_magnitude = (survivor_mean - full_mean) / full_mean if full_mean != 0 else 0
            
            interpretation = f"Survivorship bias test for '{metric}': "
            if p_val < self.alpha:
                interpretation += f"Significant bias detected (p={p_val:.4f})"
                if bias_magnitude > 0:
                    interpretation += f" - survivor sample overestimates by {bias_magnitude*100:.1f}%"
                else:
                    interpretation += f" - survivor sample underestimates by {abs(bias_magnitude)*100:.1f}%"
            else:
                interpretation += f"No significant bias detected (p={p_val:.4f})"
            
            result = TestResult(
                test_name="Survivorship Bias Test",
                test_type=TestType.INDEPENDENCE,
                statistic=stat,
                p_value=p_val,
                effect_size=abs(bias_magnitude),
                interpretation=interpretation,
                recommendations=self._get_survivorship_bias_recommendations(p_val, bias_magnitude),
                metadata={
                    "metric": metric,
                    "full_sample_mean": full_mean,
                    "survivor_sample_mean": survivor_mean,
                    "bias_magnitude": bias_magnitude,
                    "full_sample_n": len(full_values),
                    "survivor_sample_n": len(survivor_values)
                }
            )
            
            results.append(result)
        
        self.test_results.extend(results)
        return results
    
    def _get_survivorship_bias_recommendations(self, p_value: float, 
                                                bias_magnitude: float) -> List[str]:
        """Get recommendations based on survivorship bias test"""
        recommendations = []
        
        if p_value < self.alpha:
            recommendations.append("Significant survivorship bias detected")
            
            if abs(bias_magnitude) > 0.1:
                recommendations.append("Large bias magnitude - correction strongly recommended")
            
            recommendations.append("Include extinct companies in analysis")
            recommendations.append("Use weighted analysis to correct for bias")
            recommendations.append("Apply inverse probability weighting if appropriate")
            recommendations.append("Report both biased and corrected estimates")
            
        else:
            recommendations.append("No significant survivorship bias detected")
            recommendations.append("Survivor-only analysis may be acceptable")
            recommendations.append("Consider robustness checks with full sample")
        
        return recommendations
    
    # =====================================
    # UTILITY METHODS
    # =====================================
    
    def apply_multiple_testing_correction(self, method: str = None) -> None:
        """
        Apply multiple testing corrections to stored results
        
        Args:
            method: Correction method ('bonferroni', 'holm', 'fdr_bh')
        """
        if method is None:
            method = self.correction_method
        
        p_values = [r.p_value for r in self.test_results if r.p_value is not None]
        
        if len(p_values) == 0:
            return
        
        if method == 'bonferroni':
            corrected_alpha = self.alpha / len(p_values)
            for result in self.test_results:
                if result.p_value is not None:
                    result.metadata = result.metadata or {}
                    result.metadata['bonferroni_significant'] = result.p_value < corrected_alpha
                    result.metadata['corrected_alpha'] = corrected_alpha
        
        elif method == 'fdr_bh':
            from statsmodels.stats.multitest import multipletests
            
            _, corrected_p, _, _ = multipletests(p_values, method='fdr_bh', alpha=self.alpha)
            
            p_index = 0
            for result in self.test_results:
                if result.p_value is not None:
                    result.metadata = result.metadata or {}
                    result.metadata['fdr_corrected_p'] = corrected_p[p_index]
                    result.metadata['fdr_significant'] = corrected_p[p_index] < self.alpha
                    p_index += 1
    
    def get_summary_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive summary report of all statistical tests
        
        Returns:
            Dictionary with summary statistics and recommendations
        """
        if not self.test_results:
            return {"message": "No tests performed yet"}
        
        # Group results by test type
        results_by_type = {}
        for result in self.test_results:
            test_type = result.test_type.value
            if test_type not in results_by_type:
                results_by_type[test_type] = []
            results_by_type[test_type].append(result)
        
        # Calculate summary statistics
        total_tests = len(self.test_results)
        significant_tests = sum(1 for r in self.test_results 
                                if r.p_value is not None and r.p_value < self.alpha)
        
        summary = {
            "total_tests_performed": total_tests,
            "significant_tests": significant_tests,
            "significance_rate": significant_tests / total_tests if total_tests > 0 else 0,
            "alpha_level": self.alpha,
            "correction_method": self.correction_method,
            "results_by_type": {},
            "key_findings": [],
            "overall_recommendations": []
        }
        
        # Summarize by test type
        for test_type, results in results_by_type.items():
            type_significant = sum(1 for r in results 
                                    if r.p_value is not None and r.p_value < self.alpha)
            
            summary["results_by_type"][test_type] = {
                "total_tests": len(results),
                "significant_tests": type_significant,
                "significance_rate": type_significant / len(results) if results else 0,
                "tests": [
                    {
                        "test_name": r.test_name,
                        "p_value": r.p_value,
                        "significant": r.p_value < self.alpha if r.p_value is not None else False,
                        "interpretation": r.interpretation
                    }
                    for r in results
                ]
            }
        
        # Generate key findings
        if significant_tests > 0:
            summary["key_findings"].append(f"Found {significant_tests} significant results out of {total_tests} tests")
            
            # Highlight important findings by test type
            for test_type in ["survival", "market_comparison", "causal"]:
                if test_type in results_by_type:
                    sig_in_type = sum(1 for r in results_by_type[test_type] 
                                    if r.p_value is not None and r.p_value < self.alpha)
                    if sig_in_type > 0:
                        summary["key_findings"].append(f"{test_type.title()} analysis: {sig_in_type} significant findings")
        
        # Overall recommendations
        if significant_tests / total_tests > 0.3:
            summary["overall_recommendations"].append("High proportion of significant results - strong evidence for hypotheses")
        elif significant_tests / total_tests < 0.05:
            summary["overall_recommendations"].append("Low significance rate - consider sample size or effect sizes")
        
        summary["overall_recommendations"].extend([
            "Review individual test recommendations for specific actions",
            "Consider multiple testing corrections if many tests performed",
            "Validate findings with additional data or methods"
        ])
        
        return summary
    
    def export_results(self, filepath: str, format: str = 'csv') -> None:
        """
        Export test results to file
        
        Args:
            filepath: Output file path
            format: Export format ('csv', 'json', 'xlsx')
        """
        if not self.test_results:
            warnings.warn("No test results to export")
            return
        
        # Convert results to DataFrame
        export_data = []
        for result in self.test_results:
            export_data.append({
                'test_name': result.test_name,
                'test_type': result.test_type.value,
                'statistic': result.statistic,
                'p_value': result.p_value,
                'degrees_of_freedom': result.degrees_of_freedom,
                'critical_value': result.critical_value,
                'effect_size': result.effect_size,
                'interpretation': result.interpretation,
                'significant': result.p_value < self.alpha if result.p_value is not None else None
            })
        
        df = pd.DataFrame(export_data)
        
        if format.lower() == 'csv':
            df.to_csv(filepath, index=False)
        elif format.lower() == 'json':
            df.to_json(filepath, orient='records', indent=2)
        elif format.lower() == 'xlsx':
            df.to_excel(filepath, index=False)
        else:
            raise ValueError(f"Unsupported export format: {format}")


# =====================================
# SPECIALIZED TEST COLLECTIONS
# =====================================

class A2AITestSuite:
    """
    Specialized test suite for A2AI comprehensive analysis
    Provides pre-configured test batteries for common A2AI scenarios
    """
    
    def __init__(self, alpha: float = 0.05):
        self.tester = StatisticalTests(alpha=alpha)
        self.alpha = alpha
    
    def run_market_category_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Complete statistical analysis comparing market categories
        
        Args:
            data: DataFrame with market categories and financial metrics
            
        Returns:
            Comprehensive analysis results
        """
        results = {
            "market_comparison_tests": [],
            "survival_analysis_tests": [],
            "bias_tests": [],
            "summary": {}
        }
        
        # Market category comparisons for key financial metrics
        financial_metrics = [
            'revenue', 'growth_rate', 'operating_margin', 'roe', 'value_added_ratio'
        ]
        
        for metric in financial_metrics:
            if metric in data.columns:
                test_result = self.tester.test_market_category_differences(
                    data, 'market_category', metric
                )
                if test_result:
                    results["market_comparison_tests"].append(test_result)
        
        # Survival analysis if survival data available
        if all(col in data.columns for col in ['duration', 'event', 'market_category']):
            # Test each pair of market categories
            categories = data['market_category'].unique()
            for i, cat1 in enumerate(categories):
                for cat2 in categories[i+1:]:
                    cat1_data = data[data['market_category'] == cat1]
                    cat2_data = data[data['market_category'] == cat2]
                    
                    survival_result = self.tester.test_logrank(
                        cat1_data['duration'].values, cat1_data['event'].values,
                        cat2_data['duration'].values, cat2_data['event'].values,
                        cat1, cat2
                    )
                    if survival_result:
                        results["survival_analysis_tests"].append(survival_result)
        
        # Generate summary
        results["summary"] = self.tester.get_summary_report()
        
        return results
    
    def run_emergence_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Complete statistical analysis for emergence patterns
        
        Args:
            data: DataFrame with emergence data and success indicators
            
        Returns:
            Comprehensive emergence analysis results
        """
        results = {
            "success_factor_tests": [],
            "lifecycle_tests": [],
            "time_series_tests": [],
            "summary": {}
        }
        
        # Test emergence success factors
        if 'emergence_success' in data.columns:
            factor_columns = [col for col in data.columns if col.startswith('factor_')]
            if not factor_columns:
                # Default financial factors for emergence analysis
                factor_columns = [
                    'initial_capital', 'r_and_d_ratio', 'employee_count', 
                    'market_entry_timing', 'technology_advantage'
                ]
            
            factor_columns = [col for col in factor_columns if col in data.columns]
            
            factor_results = self.tester.test_emergence_success_factors(
                data, 'emergence_success', factor_columns
            )
            results["success_factor_tests"] = factor_results
        
        # Lifecycle stage analysis
        if all(col in data.columns for col in ['company_id', 'time_period', 'lifecycle_stage']):
            lifecycle_result = self.tester.test_lifecycle_stage_transitions(
                data, 'company_id', 'time_period', 'lifecycle_stage'
            )
            if lifecycle_result:
                results["lifecycle_tests"].append(lifecycle_result)
        
        # Time series stationarity tests for key metrics
        time_series_metrics = ['revenue', 'growth_rate', 'market_share']
        for metric in time_series_metrics:
            if metric in data.columns:
                stationarity_results = self.tester.test_stationarity(
                    data[metric], metric
                )
                results["time_series_tests"].extend(stationarity_results)
        
        # Generate summary
        results["summary"] = self.tester.get_summary_report()
        
        return results
    
    def run_causal_inference_validation(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate causal inference assumptions for A2AI analysis
        
        Args:
            data: Panel data for causal analysis
            
        Returns:
            Causal inference validation results
        """
        results = {
            "did_assumption_tests": {},
            "balance_tests": [],
            "robustness_tests": [],
            "summary": {}
        }
        
        # Difference-in-differences assumption tests
        if all(col in data.columns for col in ['treatment_group', 'time_period', 'outcome']):
            did_results = self.tester.test_difference_in_differences_assumptions(
                data, 'treatment_group', 'time_period', 'outcome'
            )
            results["did_assumption_tests"] = did_results
        
        # Additional robustness tests
        # Placebo tests, alternative control groups, etc.
        
        results["summary"] = self.tester.get_summary_report()
        
        return results
    
    def run_comprehensive_bias_analysis(self, full_data: pd.DataFrame, 
                                        survivor_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Comprehensive bias detection and correction analysis
        
        Args:
            full_data: Complete dataset including extinct companies
            survivor_data: Dataset with surviving companies only
            
        Returns:
            Bias analysis results
        """
        results = {
            "survivorship_bias_tests": [],
            "selection_bias_tests": [],
            "correction_recommendations": [],
            "summary": {}
        }
        
        # Survivorship bias tests
        financial_metrics = [
            'revenue', 'growth_rate', 'operating_margin', 'roe', 
            'debt_ratio', 'current_ratio', 'asset_turnover'
        ]
        
        available_metrics = [m for m in financial_metrics if m in full_data.columns and m in survivor_data.columns]
        
        survivorship_results = self.tester.test_survivorship_bias(
            full_data, survivor_data, available_metrics
        )
        results["survivorship_bias_tests"] = survivorship_results
        
        # Selection bias analysis
        # Test if sample selection mechanism is random
        
        # Generate correction recommendations
        significant_bias_count = sum(1 for r in survivorship_results 
                                    if r.p_value < self.alpha)
        
        if significant_bias_count > 0:
            results["correction_recommendations"] = [
                f"Significant survivorship bias detected in {significant_bias_count} metrics",
                "Include extinct companies in all analyses",
                "Apply inverse probability weighting",
                "Use Heckman correction for selection bias",
                "Report both biased and corrected estimates"
            ]
        else:
            results["correction_recommendations"] = [
                "No significant survivorship bias detected",
                "Survivor-only analysis appears valid",
                "Consider robustness checks with full sample"
            ]
        
        results["summary"] = self.tester.get_summary_report()
        
        return results


# =====================================
# A2AI-SPECIFIC HYPOTHESIS TESTS
# =====================================

class A2AIHypothesisTests:
    """
    Specialized hypothesis tests for A2AI research questions
    """
    
    def __init__(self, alpha: float = 0.05):
        self.tester = StatisticalTests(alpha=alpha)
        self.alpha = alpha
    
    def test_world_share_decline_hypothesis(self, data: pd.DataFrame) -> TestResult:
        """
        Test the hypothesis that Japanese companies in declining markets
        show systematically different financial patterns
        
        H0: No difference in financial deterioration patterns between market categories
        H1: Declining markets show distinct financial deterioration patterns
        """
        if 'market_category' not in data.columns:
            raise ValueError("Market category column required")
        
        # Focus on declining vs high-share markets
        declining_data = data[data['market_category'] == 'declining']
        high_share_data = data[data['market_category'] == 'high_share']
        
        # Test multiple financial deterioration indicators
        deterioration_metrics = [
            'revenue_growth_decline', 'margin_compression', 'market_share_loss',
            'r_and_d_reduction', 'employment_reduction'
        ]
        
        available_metrics = [m for m in deterioration_metrics if m in data.columns]
        
        if not available_metrics:
            return TestResult(
                test_name="World Share Decline Hypothesis Test",
                test_type=TestType.MARKET_COMPARISON,
                statistic=np.nan,
                p_value=np.nan,
                interpretation="No deterioration metrics available for testing"
            )
        
        # Multivariate test using first available metric
        test_metric = available_metrics[0]
        
        return self.tester.test_market_category_differences(
            pd.concat([declining_data, high_share_data]),
            'market_category', test_metric
        )
    
    def test_emergence_innovation_hypothesis(self, data: pd.DataFrame) -> TestResult:
        """
        Test hypothesis that successful new companies in high-share markets
        demonstrate superior innovation metrics
        
        H0: Innovation metrics do not predict emergence success
        H1: Innovation metrics significantly predict emergence success
        """
        if not all(col in data.columns for col in ['emergence_success', 'innovation_score']):
            return TestResult(
                test_name="Emergence Innovation Hypothesis Test",
                test_type=TestType.EMERGENCE,
                statistic=np.nan,
                p_value=np.nan,
                interpretation="Required columns (emergence_success, innovation_score) not available"
            )
        
        # Test innovation as predictor of emergence success
        success_group = data[data['emergence_success'] == 1]['innovation_score'].dropna().values
        failure_group = data[data['emergence_success'] == 0]['innovation_score'].dropna().values
        
        if len(success_group) == 0 or len(failure_group) == 0:
            return TestResult(
                test_name="Emergence Innovation Hypothesis Test",
                test_type=TestType.EMERGENCE,
                statistic=np.nan,
                p_value=np.nan,
                interpretation="Insufficient data in success/failure groups"
            )
        
        from scipy.stats import mannwhitneyu
        stat, p_val = mannwhitneyu(success_group, failure_group, alternative='greater')
        
        # Effect size
        n1, n2 = len(success_group), len(failure_group)
        effect_size = (stat - (n1 * n2 / 2)) / (n1 * n2) if (n1 * n2) > 0 else 0
        
        interpretation = "Innovation-emergence hypothesis: "
        if p_val < self.alpha:
            interpretation += f"Supported - successful companies show higher innovation (p={p_val:.4f})"
        else:
            interpretation += f"Not supported - no significant innovation advantage (p={p_val:.4f})"
        
        return TestResult(
            test_name="Emergence Innovation Hypothesis Test",
            test_type=TestType.EMERGENCE,
            statistic=stat,
            p_value=p_val,
            effect_size=effect_size,
            interpretation=interpretation,
            metadata={
                "success_innovation_median": np.median(success_group),
                "failure_innovation_median": np.median(failure_group),
                "success_n": n1,
                "failure_n": n2
            }
        )
    
    def test_lifecycle_resilience_hypothesis(self, data: pd.DataFrame) -> TestResult:
        """
        Test hypothesis that companies in high-share markets demonstrate
        greater resilience across lifecycle stages
        
        H0: Lifecycle resilience is similar across market categories
        H1: High-share market companies show greater lifecycle resilience
        """
        if not all(col in data.columns for col in ['market_category', 'lifecycle_stage', 'resilience_score']):
            return TestResult(
                test_name="Lifecycle Resilience Hypothesis Test",
                test_type=TestType.LIFECYCLE,
                statistic=np.nan,
                p_value=np.nan,
                interpretation="Required columns not available"
            )
        
        # Compare resilience scores across market categories
        return self.tester.test_market_category_differences(
            data, 'market_category', 'resilience_score'
        )
    
    def test_financial_factor_causality_hypothesis(self, data: pd.DataFrame,
                                                    factor: str, outcome: str) -> TestResult:
        """
        Test causal relationship between financial factors and outcomes
        
        H0: Financial factor does not causally affect outcome
        H1: Financial factor has causal effect on outcome
        """
        if not all(col in data.columns for col in [factor, outcome, 'company_id', 'time_period']):
            return TestResult(
                test_name="Financial Factor Causality Test",
                test_type=TestType.CAUSAL,
                statistic=np.nan,
                p_value=np.nan,
                interpretation="Required columns for causal analysis not available"
            )
        
        # Simplified causality test using lead-lag correlation
        # More sophisticated methods would use instrumental variables, etc.
        
        companies = data['company_id'].unique()
        correlations = []
        
        for company in companies:
            company_data = data[data['company_id'] == company].sort_values('time_period')
            
            if len(company_data) < 3:
                continue
            
            factor_values = company_data[factor].values
            outcome_values = company_data[outcome].values
            
            # Lag factor by one period
            if len(factor_values) > 1 and len(outcome_values) > 1:
                lagged_factor = factor_values[:-1]
                future_outcome = outcome_values[1:]
                
                if len(lagged_factor) > 0 and len(future_outcome) > 0:
                    corr, _ = pearsonr(lagged_factor, future_outcome)
                    if not np.isnan(corr):
                        correlations.append(corr)
        
        if len(correlations) == 0:
            return TestResult(
                test_name="Financial Factor Causality Test",
                test_type=TestType.CAUSAL,
                statistic=np.nan,
                p_value=np.nan,
                interpretation="Insufficient data for causality analysis"
            )
        
        # Test if mean correlation is significantly different from zero
        mean_corr = np.mean(correlations)
        se_corr = np.std(correlations) / np.sqrt(len(correlations))
        t_stat = mean_corr / se_corr if se_corr > 0 else 0
        
        # Two-tailed t-test
        from scipy.stats import t
        p_val = 2 * (1 - t.cdf(abs(t_stat), len(correlations) - 1))
        
        interpretation = f"Causality test ({factor} → {outcome}): "
        if p_val < self.alpha:
            interpretation += f"Evidence of causal relationship (p={p_val:.4f}, mean_corr={mean_corr:.4f})"
        else:
            interpretation += f"No evidence of causal relationship (p={p_val:.4f}, mean_corr={mean_corr:.4f})"
        
        return TestResult(
            test_name="Financial Factor Causality Test",
            test_type=TestType.CAUSAL,
            statistic=t_stat,
            p_value=p_val,
            effect_size=abs(mean_corr),
            interpretation=interpretation,
            metadata={
                "factor": factor,
                "outcome": outcome,
                "mean_correlation": mean_corr,
                "n_companies": len(correlations),
                "correlation_se": se_corr
            }
        )


# =====================================
# EXAMPLE USAGE AND TESTING
# =====================================

def example_usage():
    """
    Demonstrate A2AI statistical testing capabilities
    """
    # Initialize the testing framework
    tester = StatisticalTests(alpha=0.05)
    
    # Example 1: Market category comparison
    print("=== A2AI Statistical Testing Example ===")
    
    # Simulated data for demonstration
    np.random.seed(42)
    n_companies = 150
    
    # Create sample data matching A2AI structure
    market_categories = ['high_share'] * 50 + ['declining'] * 50 + ['lost'] * 50
    
    sample_data = pd.DataFrame({
        'company_id': range(n_companies),
        'market_category': market_categories,
        'revenue': np.random.lognormal(10, 1, n_companies),
        'growth_rate': np.random.normal(0.05, 0.1, n_companies),
        'operating_margin': np.random.normal(0.1, 0.05, n_companies),
        'roe': np.random.normal(0.08, 0.06, n_companies),
        'r_and_d_ratio': np.random.normal(0.03, 0.02, n_companies),
        'duration': np.random.exponential(10, n_companies),
        'event': np.random.binomial(1, 0.3, n_companies),
    })
    
    # Add some realistic patterns
    # High-share companies tend to have better metrics
    high_share_mask = sample_data['market_category'] == 'high_share'
    sample_data.loc[high_share_mask, 'operating_margin'] += 0.02
    sample_data.loc[high_share_mask, 'roe'] += 0.01
    
    # Lost market companies tend to have worse survival
    lost_mask = sample_data['market_category'] == 'lost'
    sample_data.loc[lost_mask, 'duration'] *= 0.7
    sample_data.loc[lost_mask, 'event'] = np.random.binomial(1, 0.6, sum(lost_mask))
    
    print("\n1. Testing market category differences in operating margin:")
    market_test = tester.test_market_category_differences(
        sample_data, 'market_category', 'operating_margin'
    )
    print(f"Result: {market_test.interpretation}")
    
    print("\n2. Testing survival differences between high-share and lost markets:")
    high_share_data = sample_data[sample_data['market_category'] == 'high_share']
    lost_data = sample_data[sample_data['market_category'] == 'lost']
    
    survival_test = tester.test_logrank(
        high_share_data['duration'].values, high_share_data['event'].values,
        lost_data['duration'].values, lost_data['event'].values,
        "High Share", "Lost Market"
    )
    print(f"Result: {survival_test.interpretation}")
    
    print("\n3. Testing normality of financial ratios:")
    normality_tests = tester.test_normality_comprehensive(
        sample_data['roe'].values, "Return on Equity"
    )
    for test in normality_tests:
        print(f"  {test.test_name}: {test.interpretation}")
    
    print("\n4. Comprehensive test suite:")
    suite = A2AITestSuite(alpha=0.05)
    comprehensive_results = suite.run_market_category_analysis(sample_data)
    
    summary = comprehensive_results['summary']
    print(f"Total tests performed: {summary['total_tests_performed']}")
    print(f"Significant results: {summary['significant_tests']}")
    print(f"Significance rate: {summary['significance_rate']:.2%}")
    
    # Generate final summary report
    print("\n=== Final Summary Report ===")
    final_summary = tester.get_summary_report()
    
    for finding in final_summary['key_findings']:
        print(f"• {finding}")
    
    print("\nRecommendations:")
    for rec in final_summary['overall_recommendations']:
        print(f"• {rec}")


if __name__ == "__main__":
    # Run example if script is executed directly
    example_usage()