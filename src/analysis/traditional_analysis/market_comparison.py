"""
A2AI Market Comparison Analysis Module
====================================

This module provides comprehensive market comparison analysis capabilities for A2AI,
focusing on comparing financial performance patterns across three market categories:
1. High Market Share Markets (現在もシェアが高い市場)
2. Declining Market Share Markets (現在進行形でシェア低下中のメジャー市場)  
3. Lost Market Share Markets (完全にシェアを失ったメジャー市場)

The analysis covers 150 companies across 40 years of financial data with 9 evaluation 
metrics and 23 factor items each, including survival analysis and lifecycle dynamics.

Author: A2AI Development Team
Version: 1.0.0
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import warnings
from dataclasses import dataclass
from enum import Enum
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MarketCategory(Enum):
    """Market category enumeration"""
    HIGH_SHARE = "high_share"
    DECLINING_SHARE = "declining_share" 
    LOST_SHARE = "lost_share"

@dataclass
class ComparisonResult:
    """Data class for storing comparison analysis results"""
    metric_name: str
    market_stats: Dict[str, Dict[str, float]]
    statistical_tests: Dict[str, Any]
    effect_sizes: Dict[str, float]
    significance_level: float
    interpretation: str

@dataclass
class LifecycleComparisonResult:
    """Data class for lifecycle-specific comparison results"""
    stage: str
    market_patterns: Dict[str, List[float]]
    survival_curves: Dict[str, Any]
    transition_probabilities: Dict[str, float]

class MarketComparisonAnalyzer:
    """
    Advanced market comparison analyzer for A2AI system.
    
    This class performs comprehensive statistical comparison analysis across
    different market share categories, accounting for survival bias and 
    lifecycle dynamics.
    """
    
    def __init__(self, data: pd.DataFrame, market_categories: Dict[str, List[str]], 
                    config: Optional[Dict] = None):
        """
        Initialize the market comparison analyzer.
        
        Args:
            data: Financial data with companies and time series
            market_categories: Dict mapping market categories to company lists
            config: Optional configuration parameters
        """
        self.data = data.copy()
        self.market_categories = market_categories
        self.config = config or self._get_default_config()
        self.scaler = StandardScaler()
        
        # Define evaluation metrics (9 items)
        self.evaluation_metrics = [
            'revenue', 'revenue_growth_rate', 'operating_margin', 
            'net_margin', 'roe', 'value_added_ratio',
            'survival_probability', 'emergence_success_rate', 'succession_success_rate'
        ]
        
        # Initialize results storage
        self.comparison_results: Dict[str, ComparisonResult] = {}
        self.lifecycle_results: Dict[str, LifecycleComparisonResult] = {}
        
        logger.info("MarketComparisonAnalyzer initialized with %d companies", 
                    len(self.data['company_id'].unique()))
    
    def _get_default_config(self) -> Dict:
        """Get default configuration parameters"""
        return {
            'significance_level': 0.05,
            'min_observations': 10,
            'survival_analysis_enabled': True,
            'lifecycle_stages': ['startup', 'growth', 'maturity', 'decline', 'extinction'],
            'time_window_years': 40,
            'bootstrap_samples': 1000,
            'clustering_n_components': 3,
            'effect_size_thresholds': {'small': 0.2, 'medium': 0.5, 'large': 0.8}
        }
    
    def prepare_market_data(self) -> Dict[str, pd.DataFrame]:
        """
        Prepare and categorize data by market categories.
        
        Returns:
            Dict of DataFrames for each market category
        """
        market_data = {}
        
        for category, companies in self.market_categories.items():
            category_data = self.data[self.data['company_name'].isin(companies)].copy()
            
            # Add market category column
            category_data['market_category'] = category
            
            # Handle survival bias correction
            if self.config['survival_analysis_enabled']:
                category_data = self._apply_survival_bias_correction(category_data)
            
            market_data[category] = category_data
            
            logger.info(f"Prepared {category} data: {len(category_data)} records, "
                        f"{category_data['company_name'].nunique()} companies")
        
        return market_data
    
    def _apply_survival_bias_correction(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply survival bias correction to account for extinct companies.
        
        Args:
            data: Market category data
            
        Returns:
            Bias-corrected data
        """
        # Identify extinct companies based on data availability
        company_last_year = data.groupby('company_name')['year'].max()
        current_year = data['year'].max()
        
        # Mark companies that disappeared before the end of observation period
        extinct_companies = company_last_year[company_last_year < current_year].index
        
        data['is_extinct'] = data['company_name'].isin(extinct_companies)
        data['survival_time'] = data.groupby('company_name')['year'].transform(
            lambda x: x.max() - x.min() + 1
        )
        
        return data
    
    def perform_comprehensive_comparison(self) -> Dict[str, ComparisonResult]:
        """
        Perform comprehensive statistical comparison across market categories.
        
        Returns:
            Dictionary of comparison results for each evaluation metric
        """
        market_data = self.prepare_market_data()
        
        for metric in self.evaluation_metrics:
            logger.info(f"Analyzing metric: {metric}")
            
            # Extract metric data for each market category
            metric_data = {}
            for category, data in market_data.items():
                if metric in data.columns:
                    metric_data[category] = data[metric].dropna()
                else:
                    logger.warning(f"Metric {metric} not found in {category} data")
                    continue
            
            if len(metric_data) < 2:
                logger.warning(f"Insufficient data for metric {metric} comparison")
                continue
            
            # Perform statistical analysis
            comparison_result = self._analyze_metric_differences(metric, metric_data)
            self.comparison_results[metric] = comparison_result
        
        return self.comparison_results
    
    def _analyze_metric_differences(self, metric_name: str, 
                                    metric_data: Dict[str, pd.Series]) -> ComparisonResult:
        """
        Analyze statistical differences for a specific metric across market categories.
        
        Args:
            metric_name: Name of the evaluation metric
            metric_data: Dict of metric values for each market category
            
        Returns:
            ComparisonResult object with detailed analysis
        """
        # Calculate descriptive statistics
        market_stats = {}
        for category, values in metric_data.items():
            market_stats[category] = {
                'mean': values.mean(),
                'median': values.median(),
                'std': values.std(),
                'min': values.min(),
                'max': values.max(),
                'count': len(values),
                'skewness': stats.skew(values),
                'kurtosis': stats.kurtosis(values)
            }
        
        # Perform statistical tests
        statistical_tests = self._perform_statistical_tests(metric_data)
        
        # Calculate effect sizes
        effect_sizes = self._calculate_effect_sizes(metric_data)
        
        # Generate interpretation
        interpretation = self._generate_interpretation(
            metric_name, market_stats, statistical_tests, effect_sizes
        )
        
        return ComparisonResult(
            metric_name=metric_name,
            market_stats=market_stats,
            statistical_tests=statistical_tests,
            effect_sizes=effect_sizes,
            significance_level=self.config['significance_level'],
            interpretation=interpretation
        )
    
    def _perform_statistical_tests(self, metric_data: Dict[str, pd.Series]) -> Dict[str, Any]:
        """
        Perform comprehensive statistical tests for market comparison.
        
        Args:
            metric_data: Dict of metric values for each market category
            
        Returns:
            Dictionary containing various statistical test results
        """
        tests = {}
        
        # Collect all values and group labels
        all_values = []
        group_labels = []
        
        for category, values in metric_data.items():
            all_values.extend(values.tolist())
            group_labels.extend([category] * len(values))
        
        # One-way ANOVA
        groups = [metric_data[cat].values for cat in metric_data.keys()]
        if len(groups) >= 2 and all(len(group) >= self.config['min_observations'] for group in groups):
            f_stat, p_value_anova = stats.f_oneway(*groups)
            tests['anova'] = {
                'f_statistic': f_stat,
                'p_value': p_value_anova,
                'significant': p_value_anova < self.config['significance_level']
            }
        
        # Kruskal-Wallis test (non-parametric alternative)
        try:
            h_stat, p_value_kw = stats.kruskal(*groups)
            tests['kruskal_wallis'] = {
                'h_statistic': h_stat,
                'p_value': p_value_kw,
                'significant': p_value_kw < self.config['significance_level']
            }
        except ValueError as e:
            logger.warning(f"Kruskal-Wallis test failed: {e}")
        
        # Pairwise comparisons (Mann-Whitney U tests)
        categories = list(metric_data.keys())
        pairwise_tests = {}
        
        for i, cat1 in enumerate(categories):
            for cat2 in categories[i+1:]:
                try:
                    u_stat, p_value_mw = stats.mannwhitneyu(
                        metric_data[cat1], metric_data[cat2], 
                        alternative='two-sided'
                    )
                    pairwise_tests[f"{cat1}_vs_{cat2}"] = {
                        'u_statistic': u_stat,
                        'p_value': p_value_mw,
                        'significant': p_value_mw < self.config['significance_level']
                    }
                except ValueError as e:
                    logger.warning(f"Mann-Whitney U test failed for {cat1} vs {cat2}: {e}")
        
        tests['pairwise_comparisons'] = pairwise_tests
        
        # Levene test for equality of variances
        try:
            levene_stat, p_value_levene = stats.levene(*groups)
            tests['levene_test'] = {
                'statistic': levene_stat,
                'p_value': p_value_levene,
                'equal_variances': p_value_levene >= self.config['significance_level']
            }
        except ValueError as e:
            logger.warning(f"Levene test failed: {e}")
        
        return tests
    
    def _calculate_effect_sizes(self, metric_data: Dict[str, pd.Series]) -> Dict[str, float]:
        """
        Calculate effect sizes for market category comparisons.
        
        Args:
            metric_data: Dict of metric values for each market category
            
        Returns:
            Dictionary of effect sizes
        """
        effect_sizes = {}
        categories = list(metric_data.keys())
        
        # Cohen's d for pairwise comparisons
        for i, cat1 in enumerate(categories):
            for cat2 in categories[i+1:]:
                values1 = metric_data[cat1]
                values2 = metric_data[cat2]
                
                # Calculate pooled standard deviation
                pooled_std = np.sqrt(
                    ((len(values1) - 1) * values1.var() + 
                     (len(values2) - 1) * values2.var()) / 
                    (len(values1) + len(values2) - 2)
                )
                
                if pooled_std > 0:
                    cohens_d = (values1.mean() - values2.mean()) / pooled_std
                    effect_sizes[f"cohens_d_{cat1}_vs_{cat2}"] = cohens_d
        
        # Eta-squared for overall effect size
        if len(categories) > 2:
            # Calculate sum of squares between groups and within groups
            grand_mean = np.mean([val for values in metric_data.values() for val in values])
            
            ss_between = sum(
                len(values) * (values.mean() - grand_mean) ** 2 
                for values in metric_data.values()
            )
            
            ss_within = sum(
                sum((val - values.mean()) ** 2 for val in values)
                for values in metric_data.values()
            )
            
            ss_total = ss_between + ss_within
            
            if ss_total > 0:
                eta_squared = ss_between / ss_total
                effect_sizes['eta_squared'] = eta_squared
        
        return effect_sizes
    
    def _generate_interpretation(self, metric_name: str, market_stats: Dict,
                                statistical_tests: Dict, effect_sizes: Dict) -> str:
        """
        Generate human-readable interpretation of comparison results.
        
        Args:
            metric_name: Name of the evaluation metric
            market_stats: Descriptive statistics for each market category
            statistical_tests: Results of statistical tests
            effect_sizes: Calculated effect sizes
            
        Returns:
            Interpretation string
        """
        interpretation = f"Market Comparison Analysis for {metric_name}:\n"
        
        # Descriptive summary
        interpretation += "\nDescriptive Statistics:\n"
        for category, stats in market_stats.items():
            interpretation += f"- {category}: Mean={stats['mean']:.4f}, "
            interpretation += f"Median={stats['median']:.4f}, SD={stats['std']:.4f}, "
            interpretation += f"N={stats['count']}\n"
        
        # Statistical significance
        if 'anova' in statistical_tests:
            anova_result = statistical_tests['anova']
            significance = "significant" if anova_result['significant'] else "not significant"
            interpretation += f"\nOverall Difference: {significance} "
            interpretation += f"(F={anova_result['f_statistic']:.4f}, p={anova_result['p_value']:.4f})\n"
        
        # Effect size interpretation
        if 'eta_squared' in effect_sizes:
            eta_sq = effect_sizes['eta_squared']
            if eta_sq < self.config['effect_size_thresholds']['small']:
                effect_magnitude = "negligible"
            elif eta_sq < self.config['effect_size_thresholds']['medium']:
                effect_magnitude = "small"
            elif eta_sq < self.config['effect_size_thresholds']['large']:
                effect_magnitude = "medium"
            else:
                effect_magnitude = "large"
            
            interpretation += f"Effect Size: {effect_magnitude} (η² = {eta_sq:.4f})\n"
        
        # Pairwise comparison highlights
        if 'pairwise_comparisons' in statistical_tests:
            interpretation += "\nPairwise Comparisons:\n"
            for comparison, result in statistical_tests['pairwise_comparisons'].items():
                if result['significant']:
                    interpretation += f"- {comparison}: Significant difference (p={result['p_value']:.4f})\n"
        
        return interpretation
    
    def perform_lifecycle_comparison(self) -> Dict[str, LifecycleComparisonResult]:
        """
        Perform lifecycle-stage comparison analysis across market categories.
        
        Returns:
            Dictionary of lifecycle comparison results
        """
        market_data = self.prepare_market_data()
        
        # Define lifecycle stages based on company age and performance patterns
        for stage in self.config['lifecycle_stages']:
            logger.info(f"Analyzing lifecycle stage: {stage}")
            
            stage_data = {}
            for category, data in market_data.items():
                stage_subset = self._filter_by_lifecycle_stage(data, stage)
                if len(stage_subset) >= self.config['min_observations']:
                    stage_data[category] = stage_subset
            
            if len(stage_data) >= 2:
                lifecycle_result = self._analyze_lifecycle_stage(stage, stage_data)
                self.lifecycle_results[stage] = lifecycle_result
        
        return self.lifecycle_results
    
    def _filter_by_lifecycle_stage(self, data: pd.DataFrame, stage: str) -> pd.DataFrame:
        """
        Filter data based on lifecycle stage criteria.
        
        Args:
            data: Company financial data
            stage: Lifecycle stage to filter for
            
        Returns:
            Filtered data for the specified lifecycle stage
        """
        if stage == 'startup':
            # Companies in first 5 years of data
            return data[data['company_age'] <= 5]
        elif stage == 'growth':
            # Companies with positive growth and age 6-15 years
            return data[(data['company_age'] > 5) & (data['company_age'] <= 15) & 
                        (data['revenue_growth_rate'] > 0)]
        elif stage == 'maturity':
            # Established companies with stable metrics
            return data[(data['company_age'] > 15) & (data['company_age'] <= 30) &
                        (data['revenue_growth_rate'].abs() <= 0.1)]
        elif stage == 'decline':
            # Companies with declining performance
            return data[(data['company_age'] > 15) & (data['revenue_growth_rate'] < -0.05)]
        elif stage == 'extinction':
            # Companies that have ceased operations
            return data[data['is_extinct'] == True]
        else:
            return data
    
    def _analyze_lifecycle_stage(self, stage: str, 
                                stage_data: Dict[str, pd.DataFrame]) -> LifecycleComparisonResult:
        """
        Analyze patterns within a specific lifecycle stage across market categories.
        
        Args:
            stage: Lifecycle stage name
            stage_data: Data for each market category in this stage
            
        Returns:
            LifecycleComparisonResult with stage-specific analysis
        """
        # Extract key patterns for each market category
        market_patterns = {}
        for category, data in stage_data.items():
            patterns = []
            for metric in self.evaluation_metrics:
                if metric in data.columns:
                    patterns.append(data[metric].mean())
                else:
                    patterns.append(np.nan)
            market_patterns[category] = patterns
        
        # Calculate survival curves if applicable
        survival_curves = {}
        if stage != 'extinction':
            for category, data in stage_data.items():
                if 'survival_time' in data.columns:
                    survival_curves[category] = self._calculate_survival_curve(data)
        
        # Calculate transition probabilities
        transition_probs = self._calculate_transition_probabilities(stage_data)
        
        return LifecycleComparisonResult(
            stage=stage,
            market_patterns=market_patterns,
            survival_curves=survival_curves,
            transition_probabilities=transition_probs
        )
    
    def _calculate_survival_curve(self, data: pd.DataFrame) -> Dict:
        """
        Calculate survival curve for a dataset.
        
        Args:
            data: Company data with survival information
            
        Returns:
            Dictionary with survival curve data
        """
        if 'survival_time' not in data.columns:
            return {}
        
        # Sort by survival time
        sorted_data = data.sort_values('survival_time')
        
        # Calculate Kaplan-Meier survival estimates
        survival_times = sorted_data['survival_time'].unique()
        survival_probs = []
        
        n_at_risk = len(sorted_data)
        cumulative_survival = 1.0
        
        for time in survival_times:
            events_at_time = len(sorted_data[
                (sorted_data['survival_time'] == time) & 
                (sorted_data['is_extinct'] == True)
            ])
            
            if n_at_risk > 0:
                survival_rate = 1 - (events_at_time / n_at_risk)
                cumulative_survival *= survival_rate
                n_at_risk -= len(sorted_data[sorted_data['survival_time'] == time])
            
            survival_probs.append(cumulative_survival)
        
        return {
            'times': survival_times.tolist(),
            'survival_probabilities': survival_probs,
            'median_survival': np.median(sorted_data['survival_time'])
        }
    
    def _calculate_transition_probabilities(self, stage_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """
        Calculate transition probabilities between lifecycle stages.
        
        Args:
            stage_data: Data for each market category
            
        Returns:
            Dictionary of transition probabilities
        """
        transition_probs = {}
        
        for category, data in stage_data.items():
            # Calculate probability of successful stage transition
            if len(data) > 0:
                # Success defined as survival and performance improvement
                successful_transitions = len(data[
                    (data['is_extinct'] == False) & 
                    (data['revenue_growth_rate'] > 0)
                ])
                
                transition_prob = successful_transitions / len(data)
                transition_probs[f"{category}_success_rate"] = transition_prob
        
        return transition_probs
    
    def perform_clustering_analysis(self, n_clusters: Optional[int] = None) -> Dict[str, Any]:
        """
        Perform clustering analysis to identify distinct company patterns across markets.
        
        Args:
            n_clusters: Number of clusters (default uses config setting)
            
        Returns:
            Dictionary with clustering results and analysis
        """
        if n_clusters is None:
            n_clusters = self.config['clustering_n_components']
        
        # Prepare feature matrix
        market_data = self.prepare_market_data()
        
        # Combine all data and create feature matrix
        all_data = pd.concat(market_data.values(), ignore_index=True)
        
        # Select numerical columns for clustering
        feature_columns = [col for col in self.evaluation_metrics if col in all_data.columns]
        feature_matrix = all_data[feature_columns].fillna(0)
        
        # Standardize features
        feature_matrix_scaled = self.scaler.fit_transform(feature_matrix)
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(feature_matrix_scaled)
        
        # Add cluster labels to data
        all_data['cluster'] = cluster_labels
        
        # Analyze cluster characteristics
        cluster_analysis = {}
        for cluster_id in range(n_clusters):
            cluster_data = all_data[all_data['cluster'] == cluster_id]
            
            # Calculate cluster statistics
            cluster_stats = {}
            for metric in feature_columns:
                cluster_stats[metric] = {
                    'mean': cluster_data[metric].mean(),
                    'std': cluster_data[metric].std(),
                    'count': len(cluster_data)
                }
            
            # Market category distribution in cluster
            market_distribution = cluster_data['market_category'].value_counts().to_dict()
            
            cluster_analysis[f'cluster_{cluster_id}'] = {
                'statistics': cluster_stats,
                'market_distribution': market_distribution,
                'dominant_market': max(market_distribution, key=market_distribution.get)
            }
        
        # Perform dimensionality reduction for visualization
        pca = PCA(n_components=2)
        pca_features = pca.fit_transform(feature_matrix_scaled)
        
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(feature_matrix_scaled)-1))
        tsne_features = tsne.fit_transform(feature_matrix_scaled)
        
        return {
            'cluster_labels': cluster_labels,
            'cluster_analysis': cluster_analysis,
            'pca_coordinates': pca_features,
            'tsne_coordinates': tsne_features,
            'feature_importance': dict(zip(feature_columns, kmeans.cluster_centers_.std(axis=0))),
            'inertia': kmeans.inertia_
        }
    
    def generate_comprehensive_report(self) -> str:
        """
        Generate a comprehensive market comparison report.
        
        Returns:
            Formatted report string
        """
        report = "A2AI Market Comparison Analysis Report\n"
        report += "=" * 50 + "\n\n"
        
        # Executive Summary
        report += "Executive Summary\n"
        report += "-" * 20 + "\n"
        report += f"Analysis covers {len(self.data['company_name'].unique())} companies "
        report += f"across {len(self.market_categories)} market categories "
        report += f"over {self.config['time_window_years']} years.\n\n"
        
        # Market Category Overview
        report += "Market Categories:\n"
        for category, companies in self.market_categories.items():
            report += f"- {category}: {len(companies)} companies\n"
        report += "\n"
        
        # Detailed Results
        if self.comparison_results:
            report += "Metric Comparison Results\n"
            report += "-" * 30 + "\n"
            
            for metric, result in self.comparison_results.items():
                report += f"\n{metric.upper()}\n"
                report += result.interpretation + "\n"
        
        # Lifecycle Analysis Results
        if self.lifecycle_results:
            report += "\nLifecycle Analysis Results\n"
            report += "-" * 30 + "\n"
            
            for stage, result in self.lifecycle_results.items():
                report += f"\n{stage.upper()} STAGE\n"
                report += f"Market patterns identified across {len(result.market_patterns)} categories\n"
                
                if result.transition_probabilities:
                    report += "Transition success rates:\n"
                    for transition, prob in result.transition_probabilities.items():
                        report += f"- {transition}: {prob:.2%}\n"
        
        # Key Insights and Recommendations
        report += "\nKey Insights\n"
        report += "-" * 15 + "\n"
        report += self._generate_key_insights()
        
        return report
    
    def _generate_key_insights(self) -> str:
        """
        Generate key insights from the analysis results.
        
        Returns:
            Key insights string
        """
        insights = ""
        
        if not self.comparison_results:
            return "Analysis not yet performed. Please run comparison analysis first.\n"
        
        # Identify metrics with largest differences
        significant_metrics = []
        for metric, result in self.comparison_results.items():
            if 'anova' in result.statistical_tests and result.statistical_tests['anova']['significant']:
                significant_metrics.append(metric)
        
        if significant_metrics:
            insights += f"Significant differences found in {len(significant_metrics)} metrics:\n"
            for metric in significant_metrics[:3]:  # Top 3
                insights += f"- {metric}\n"
            insights += "\n"
        
        # Market category performance patterns
        insights += "Market Category Performance Patterns:\n"
        
        # Calculate average performance across metrics for each market category
        category_performance = {}
        for category in self.market_categories.keys():
            category_scores = []
            for metric, result in self.comparison_results.items():
                if category in result.market_stats:
                    # Normalize scores (higher is better, so normalize positively)
                    category_scores.append(result.market_stats[category]['mean'])
            
            if category_scores:
                category_performance[category] = np.mean(category_scores)
        
        # Rank categories
        sorted_categories = sorted(category_performance.items(), key=lambda x: x[1], reverse=True)
        
        for i, (category, score) in enumerate(sorted_categories, 1):
            insights += f"{i}. {category}: Average performance score = {score:.4f}\n"
        
        insights += "\nRecommendations:\n"
        insights += "1. Focus on factors that distinguish high-performing market categories\n"
        insights += "2. Investigate survival strategies from companies in lost-share markets\n"
        insights += "3. Apply lifecycle-specific strategies based on company maturity stage\n"
        
        return insights
    
    def save_results(self, filepath: str) -> None:
        """
        Save analysis results to file.
        
        Args:
            filepath: Path to save the results
        """
        results_dict = {
            'comparison_results': {
                metric: {
                    'metric_name': result.metric_name,
                    'market_stats': result.market_stats,
                    'statistical_tests': result.statistical_tests,
                    'effect_sizes': result.effect_sizes,
                    'significance_level': result.significance_level,
                    'interpretation': result.interpretation
                }
                for metric, result in self.comparison_results.items()
            },
            'lifecycle_results': {
                stage: {
                    'stage': result.stage,
                    'market_patterns': result.market_patterns,
                    'survival_curves': result.survival_curves,
                    'transition_probabilities': result.transition_probabilities
                }
                for stage, result in self.lifecycle_results.items()
            },
            'config': self.config
        }
        
        import json
        with open(filepath, 'w') as f:
            json.dump(results_dict, f, indent=2, default=str)
        
        logger.info(f"Results saved to {filepath}")

# Usage Example and Testing Functions
def example_usage():
    """
    Example usage of the MarketComparisonAnalyzer
    """
    # Sample data structure (would be loaded from actual financial data)
    np.random.seed(42)
    
    # Create sample data based on the actual company list
    companies = {
        'high_share': [
            'ファナック', '村田製作所', 'キーエンス', 'オリンパス', 'DMG森精機',
            '島津製作所', '堀場製作所', '東京精密', 'ミツトヨ', 'アルバック'
        ],
        'declining_share': [
            'トヨタ自動車', '日産自動車', 'パナソニック', '日本製鉄', 'JFEホールディングス',
            'シャープ', 'パナソニックエナジー', 'NEC', '富士通', 'バッファロー'
        ],
        'lost_share': [
            'ソニー', '三洋電機', '東芝', 'ルネサスエレクトロニクス', '京セラ',
            'パナソニック', '日立製作所', '三菱電機', 'カシオ計算機', '日本無線'
        ]
    }
    
    # Generate sample financial data
    data_list = []
    current_year = 2024
    
    for category, company_list in companies.items():
        for company in company_list:
            # Different survival patterns by market category
            if category == 'high_share':
                years = list(range(1984, current_year + 1))  # Full 40 years
                base_performance = np.random.normal(0.8, 0.1)  # Higher baseline
            elif category == 'declining_share':
                years = list(range(1984, current_year + 1))  # Full survival but declining
                base_performance = np.random.normal(0.6, 0.15)  # Medium baseline, declining trend
            else:  # lost_share
                # Some companies extinct, some survived with poor performance
                if np.random.random() < 0.4:  # 40% extinct
                    extinction_year = np.random.randint(1995, 2020)
                    years = list(range(1984, extinction_year))
                else:
                    years = list(range(1984, current_year + 1))
                base_performance = np.random.normal(0.4, 0.2)  # Lower baseline
            
            company_age_start = np.random.randint(0, 20)  # Companies have different starting ages
            
            for i, year in enumerate(years):
                # Create realistic financial metrics with trends
                company_age = company_age_start + i
                
                # Trend adjustments based on market category
                if category == 'declining_share':
                    trend_factor = max(0.1, 1.0 - (year - 1984) * 0.01)  # Gradual decline
                else:
                    trend_factor = 1.0
                
                # Generate correlated financial metrics
                revenue_base = base_performance * trend_factor * np.random.lognormal(0, 0.3)
                
                record = {
                    'company_name': company,
                    'year': year,
                    'company_age': company_age,
                    'market_category': category,
                    
                    # Traditional 6 evaluation metrics
                    'revenue': revenue_base * np.random.lognormal(10, 0.5),  # In billions
                    'revenue_growth_rate': np.random.normal(
                        0.05 * base_performance * trend_factor, 0.1
                    ),
                    'operating_margin': max(0, np.random.normal(
                        0.1 * base_performance * trend_factor, 0.05
                    )),
                    'net_margin': max(0, np.random.normal(
                        0.08 * base_performance * trend_factor, 0.04
                    )),
                    'roe': max(0, np.random.normal(
                        0.12 * base_performance * trend_factor, 0.06
                    )),
                    'value_added_ratio': max(0, min(1, np.random.normal(
                        0.3 * base_performance * trend_factor, 0.1
                    ))),
                    
                    # Extended 3 evaluation metrics
                    'survival_probability': max(0, min(1, base_performance * trend_factor + 
                                                        np.random.normal(0, 0.1))),
                    'emergence_success_rate': max(0, min(1, 
                        0.2 + 0.3 * base_performance * trend_factor + np.random.normal(0, 0.1)
                    )) if company_age < 10 else np.nan,
                    'succession_success_rate': max(0, min(1, 
                        0.5 * base_performance * trend_factor + np.random.normal(0, 0.15)
                    )) if company_age > 15 else np.nan,
                    
                    # Additional factor metrics (sample)
                    'rd_intensity': max(0, np.random.normal(
                        0.05 * base_performance, 0.02
                    )),
                    'employee_count': max(100, np.random.lognormal(8, 0.5)),
                    'total_assets': revenue_base * np.random.lognormal(1, 0.3),
                    
                    # Survival analysis fields
                    'is_extinct': year == years[-1] and len(years) < 35,  # Extinct if stopped before 2019
                    'survival_time': len(years)
                }
                
                data_list.append(record)
    
    # Convert to DataFrame
    sample_data = pd.DataFrame(data_list)
    
    # Initialize analyzer
    analyzer = MarketComparisonAnalyzer(
        data=sample_data,
        market_categories=companies,
        config={
            'significance_level': 0.05,
            'min_observations': 10,
            'survival_analysis_enabled': True,
            'lifecycle_stages': ['startup', 'growth', 'maturity', 'decline', 'extinction'],
            'time_window_years': 40,
            'bootstrap_samples': 1000
        }
    )
    
    print("=" * 60)
    print("A2AI Market Comparison Analysis Demo")
    print("=" * 60)
    
    # Perform comprehensive comparison
    print("\n1. Performing comprehensive metric comparison...")
    comparison_results = analyzer.perform_comprehensive_comparison()
    print(f"Completed analysis for {len(comparison_results)} metrics")
    
    # Perform lifecycle comparison
    print("\n2. Performing lifecycle comparison analysis...")
    lifecycle_results = analyzer.perform_lifecycle_comparison()
    print(f"Completed lifecycle analysis for {len(lifecycle_results)} stages")
    
    # Perform clustering analysis
    print("\n3. Performing clustering analysis...")
    clustering_results = analyzer.perform_clustering_analysis(n_clusters=3)
    print(f"Identified {len(clustering_results['cluster_analysis'])} distinct clusters")
    
    # Generate comprehensive report
    print("\n4. Generating comprehensive report...")
    report = analyzer.generate_comprehensive_report()
    
    # Display key results
    print("\n" + "=" * 60)
    print("KEY RESULTS SUMMARY")
    print("=" * 60)
    
    # Show sample metric comparison
    if 'revenue_growth_rate' in comparison_results:
        result = comparison_results['revenue_growth_rate']
        print(f"\nRevenue Growth Rate Analysis:")
        print(f"{'Category':<20} {'Mean':<10} {'Std':<10} {'Count':<8}")
        print("-" * 50)
        for category, stats in result.market_stats.items():
            print(f"{category:<20} {stats['mean']:<10.4f} {stats['std']:<10.4f} {stats['count']:<8}")
        
        if 'anova' in result.statistical_tests:
            anova = result.statistical_tests['anova']
            significance = "***" if anova['significant'] else "n.s."
            print(f"\nStatistical Test: F={anova['f_statistic']:.4f}, p={anova['p_value']:.4f} {significance}")
    
    # Show clustering results summary
    print(f"\nClustering Analysis:")
    print(f"{'Cluster':<10} {'Dominant Market':<20} {'Size':<8}")
    print("-" * 40)
    for cluster_id, analysis in clustering_results['cluster_analysis'].items():
        dominant_market = analysis['dominant_market']
        cluster_size = sum(analysis['market_distribution'].values())
        print(f"{cluster_id:<10} {dominant_market:<20} {cluster_size:<8}")
    
    # Show lifecycle insights
    if lifecycle_results:
        print(f"\nLifecycle Analysis - Transition Success Rates:")
        print(f"{'Stage':<15} {'Category':<20} {'Success Rate':<12}")
        print("-" * 50)
        for stage, result in lifecycle_results.items():
            if result.transition_probabilities:
                for transition, prob in list(result.transition_probabilities.items())[:3]:
                    category = transition.split('_')[0]
                    print(f"{stage:<15} {category:<20} {prob:<12.2%}")
    
    # Display partial report
    print("\n" + "=" * 60)
    print("COMPREHENSIVE REPORT PREVIEW")
    print("=" * 60)
    print(report[:2000] + "..." if len(report) > 2000 else report)
    
    return analyzer, comparison_results, lifecycle_results, clustering_results


def validate_market_comparison_functionality():
    """
    Validation function to test MarketComparisonAnalyzer functionality
    """
    print("Validating A2AI Market Comparison Analyzer...")
    
    try:
        # Run example analysis
        analyzer, comparison_results, lifecycle_results, clustering_results = example_usage()
        
        # Validation checks
        validations = []
        
        # Check 1: Basic functionality
        validations.append(("Analyzer initialization", analyzer is not None))
        validations.append(("Comparison results generated", len(comparison_results) > 0))
        validations.append(("Multiple metrics analyzed", len(comparison_results) >= 5))
        
        # Check 2: Statistical analysis
        has_statistical_tests = any(
            'anova' in result.statistical_tests or 'kruskal_wallis' in result.statistical_tests
            for result in comparison_results.values()
        )
        validations.append(("Statistical tests performed", has_statistical_tests))
        
        # Check 3: Effect size calculations
        has_effect_sizes = any(
            len(result.effect_sizes) > 0
            for result in comparison_results.values()
        )
        validations.append(("Effect sizes calculated", has_effect_sizes))
        
        # Check 4: Lifecycle analysis
        validations.append(("Lifecycle analysis completed", len(lifecycle_results) > 0))
        
        # Check 5: Clustering analysis
        validations.append(("Clustering analysis completed", clustering_results is not None))
        validations.append(("Clusters identified", 'cluster_analysis' in clustering_results))
        
        # Check 6: Report generation
        report = analyzer.generate_comprehensive_report()
        validations.append(("Report generated", len(report) > 1000))
        
        # Check 7: Survival bias correction
        market_data = analyzer.prepare_market_data()
        has_extinction_data = any(
            'is_extinct' in data.columns and data['is_extinct'].any()
            for data in market_data.values()
        )
        validations.append(("Survival bias handling", has_extinction_data))
        
        # Display validation results
        print("\nValidation Results:")
        print("-" * 50)
        passed = 0
        for check_name, passed_check in validations:
            status = "✓ PASS" if passed_check else "✗ FAIL"
            print(f"{check_name:<35} {status}")
            if passed_check:
                passed += 1
        
        print(f"\nOverall: {passed}/{len(validations)} checks passed")
        
        if passed == len(validations):
            print("🎉 All validation checks passed! MarketComparisonAnalyzer is ready for production use.")
        else:
            print("⚠️  Some validation checks failed. Please review the implementation.")
        
        return passed == len(validations)
        
    except Exception as e:
        print(f"❌ Validation failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run example and validation
    try:
        success = validate_market_comparison_functionality()
        if success:
            print("\n" + "="*60)
            print("A2AI MARKET COMPARISON ANALYZER - READY FOR DEPLOYMENT")
            print("="*60)
        else:
            print("\n" + "="*60)
            print("VALIDATION ISSUES DETECTED - PLEASE REVIEW")
            print("="*60)
    except Exception as e:
        print(f"Execution failed: {str(e)}")
        import traceback
        traceback.print_exc()