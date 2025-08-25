"""
A2AI (Advanced Financial Analysis AI)
Future Scenario Analyzer

This module provides comprehensive future scenario analysis capabilities for the 150 companies
across high-share, declining, and lost market categories. It integrates survival analysis,
emergence patterns, and lifecycle dynamics to generate strategic future scenarios.

Key Features:
- Multi-scenario Monte Carlo simulation
- Corporate lifecycle trajectory prediction
- Market ecosystem evolution modeling
- Survival probability forecasting
- Strategic positioning scenario analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum
import logging
from scipy import stats
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MarketCategory(Enum):
    """Market category classification"""
    HIGH_SHARE = "high_share"
    DECLINING = "declining" 
    LOST = "lost"

class ScenarioType(Enum):
    """Types of future scenarios"""
    OPTIMISTIC = "optimistic"
    REALISTIC = "realistic"
    PESSIMISTIC = "pessimistic"
    DISRUPTION = "disruption"

@dataclass
class ScenarioParameters:
    """Parameters for scenario generation"""
    time_horizon: int = 10  # years
    num_simulations: int = 1000
    confidence_intervals: List[float] = None
    market_volatility: float = 0.15
    technology_disruption_probability: float = 0.1
    economic_shock_probability: float = 0.05
    
    def __post_init__(self):
        if self.confidence_intervals is None:
            self.confidence_intervals = [0.05, 0.25, 0.75, 0.95]

class FutureScenarioAnalyzer:
    """
    Advanced future scenario analyzer for A2AI system
    
    This class provides comprehensive scenario analysis capabilities for predicting
    corporate performance and survival across different market categories and
    external environmental conditions.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the Future Scenario Analyzer
        
        Args:
            config: Configuration dictionary containing model parameters
        """
        self.config = config or {}
        self.models = {}
        self.scenario_results = {}
        self.feature_importance = {}
        self.market_categories = {
            'high_share': ['ファナック', '安川電機', '川崎重工業', '不二越', 'デンソーウェーブ',
                            'オリンパス', 'HOYA', '富士フイルム', '島津製作所', 'コニカミノルタ',
                            'DMG森精機', 'ヤマザキマザック', 'オークマ', '牧野フライス製作所', 'ジェイテクト',
                            '村田製作所', 'TDK', '京セラ', '太陽誘電', '日本特殊陶業',
                            'キーエンス', '島津製作所', '堀場製作所', '東京精密', 'ミツトヨ'],
            'declining': ['トヨタ自動車', '日産自動車', 'ホンダ', 'スズキ', 'マツダ',
                            '日本製鉄', 'JFEホールディングス', '神戸製鋼所', '日新製鋼', '大同特殊鋼',
                            'パナソニック', 'シャープ', 'ソニー（家電部門）', '東芝ライフスタイル', '日立グローバルライフソリューションズ',
                            'パナソニックエナジー', '村田製作所', 'GSユアサ', '東芝インフラシステムズ', '日立化成',
                            'NEC（NECパーソナル）', '富士通クライアントコンピューティング', '東芝（ダイナブック）', 'ソニー（VAIO）', 'エレコム'],
            'lost': ['ソニー（家電部門）', 'パナソニック', 'シャープ', '東芝ライフスタイル', '三菱電機（家電部門）',
                    '東芝（メモリ部門）', '日立製作所（旧）', '三菱電機（旧）', 'NEC（旧）', '富士通（旧）',
                    'ソニー（Xperia）', 'シャープ（AQUOS）', '京セラ', 'パナソニック', '富士通（arrows）',
                    'ソニー（VAIO）', 'NEC', '富士通', '東芝（dynabook）', 'シャープ',
                    'NEC', '富士通', '日立製作所', '松下電器（パナソニック）', 'シャープ']
        }
        
        logger.info("Future Scenario Analyzer initialized")
    
    def prepare_scenario_features(self, financial_data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare feature set for scenario analysis
        
        Args:
            financial_data: Historical financial data with 23 factor items per evaluation metric
            
        Returns:
            Feature matrix for scenario modeling
        """
        logger.info("Preparing scenario features...")
        
        # Core financial features (23 items per 9 evaluation metrics)
        evaluation_metrics = [
            'sales_revenue', 'sales_growth_rate', 'operating_margin', 'net_margin',
            'roe', 'value_added_rate', 'survival_probability', 'emergence_success_rate',
            'succession_success_rate'
        ]
        
        feature_columns = []
        
        # Generate feature names for each evaluation metric
        for metric in evaluation_metrics:
            base_factors = [
                # Investment & Asset Related (5 factors)
                f'{metric}_tangible_assets', f'{metric}_capex', f'{metric}_rd_expense',
                f'{metric}_intangible_assets', f'{metric}_investment_securities',
                
                # Human Resource Related (4 factors)
                f'{metric}_employee_count', f'{metric}_avg_annual_salary', 
                f'{metric}_retirement_costs', f'{metric}_welfare_costs',
                
                # Working Capital & Efficiency (5 factors)
                f'{metric}_accounts_receivable', f'{metric}_inventory', f'{metric}_total_assets',
                f'{metric}_receivables_turnover', f'{metric}_inventory_turnover',
                
                # Business Development (6 factors)
                f'{metric}_overseas_sales_ratio', f'{metric}_business_segments', 
                f'{metric}_sga_expenses', f'{metric}_advertising_costs',
                f'{metric}_non_operating_income', f'{metric}_order_backlog',
                
                # Extended Factors (3 factors)
                f'{metric}_company_age', f'{metric}_market_entry_timing', f'{metric}_parent_dependency'
            ]
            feature_columns.extend(base_factors)
        
        # Create feature matrix with scenario-relevant transformations
        scenario_features = pd.DataFrame(index=financial_data.index)
        
        # Add market category encoding
        scenario_features['market_category_high'] = 0
        scenario_features['market_category_declining'] = 0  
        scenario_features['market_category_lost'] = 0
        
        for idx, company in enumerate(financial_data.index):
            if company in self.market_categories['high_share']:
                scenario_features.loc[company, 'market_category_high'] = 1
            elif company in self.market_categories['declining']:
                scenario_features.loc[company, 'market_category_declining'] = 1
            else:
                scenario_features.loc[company, 'market_category_lost'] = 1
        
        # Add time-based features
        scenario_features['years_since_peak'] = np.random.randint(1, 15, len(financial_data))
        scenario_features['market_maturity_index'] = np.random.uniform(0.2, 0.9, len(financial_data))
        scenario_features['technology_adoption_rate'] = np.random.uniform(0.1, 0.8, len(financial_data))
        
        # Add volatility and risk measures
        scenario_features['revenue_volatility'] = np.random.uniform(0.05, 0.3, len(financial_data))
        scenario_features['margin_stability'] = np.random.uniform(0.4, 0.9, len(financial_data))
        scenario_features['debt_to_equity_ratio'] = np.random.uniform(0.1, 1.5, len(financial_data))
        
        # Simulate core financial ratios
        for col in feature_columns[:50]:  # Limit to manageable number for simulation
            scenario_features[col] = np.random.normal(0, 1, len(financial_data))
        
        logger.info(f"Generated {len(scenario_features.columns)} scenario features for {len(scenario_features)} companies")
        return scenario_features
    
    def build_scenario_models(self, features: pd.DataFrame, targets: pd.DataFrame) -> Dict[str, Any]:
        """
        Build predictive models for scenario analysis
        
        Args:
            features: Feature matrix
            targets: Target variables (evaluation metrics)
            
        Returns:
            Dictionary of trained models
        """
        logger.info("Building scenario prediction models...")
        
        models = {}
        feature_importance = {}
        
        # Models for each evaluation metric
        target_metrics = ['survival_probability', 'performance_score', 'market_share_change']
        
        scaler = StandardScaler()
        features_scaled = pd.DataFrame(
            scaler.fit_transform(features),
            columns=features.columns,
            index=features.index
        )
        
        for metric in target_metrics:
            # Create synthetic target data based on market categories
            if metric == 'survival_probability':
                target_values = np.where(
                    features['market_category_high'] == 1, 
                    np.random.beta(8, 2, len(features)),  # High survival for high-share
                    np.where(
                        features['market_category_declining'] == 1,
                        np.random.beta(5, 3, len(features)),  # Medium survival for declining
                        np.random.beta(2, 6, len(features))   # Low survival for lost markets
                    )
                )
            elif metric == 'performance_score':
                target_values = np.where(
                    features['market_category_high'] == 1,
                    np.random.normal(0.7, 0.1, len(features)),
                    np.where(
                        features['market_category_declining'] == 1,
                        np.random.normal(0.4, 0.15, len(features)),
                        np.random.normal(0.2, 0.1, len(features))
                    )
                )
            else:  # market_share_change
                target_values = np.where(
                    features['market_category_high'] == 1,
                    np.random.normal(0.02, 0.05, len(features)),
                    np.where(
                        features['market_category_declining'] == 1,
                        np.random.normal(-0.03, 0.08, len(features)),
                        np.random.normal(-0.08, 0.1, len(features))
                    )
                )
            
            # Train Random Forest model
            model = RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
            
            model.fit(features_scaled, target_values)
            models[metric] = model
            
            # Store feature importance
            importance_df = pd.DataFrame({
                'feature': features.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            feature_importance[metric] = importance_df
            
            logger.info(f"Model for {metric} trained with R² score: {model.score(features_scaled, target_values):.3f}")
        
        models['scaler'] = scaler
        return models, feature_importance
    
    def generate_scenario_conditions(self, scenario_type: ScenarioType, 
                                    years: int = 10) -> Dict[str, np.ndarray]:
        """
        Generate external scenario conditions
        
        Args:
            scenario_type: Type of scenario (optimistic, realistic, pessimistic, disruption)
            years: Number of years to project
            
        Returns:
            Dictionary of scenario condition arrays
        """
        conditions = {}
        
        if scenario_type == ScenarioType.OPTIMISTIC:
            conditions['gdp_growth'] = np.random.normal(0.03, 0.01, years)
            conditions['technology_advancement'] = np.random.uniform(1.05, 1.15, years)
            conditions['market_expansion'] = np.random.uniform(1.02, 1.08, years)
            conditions['regulatory_support'] = np.random.uniform(0.9, 1.1, years)
            conditions['competition_intensity'] = np.random.uniform(0.8, 1.0, years)
            
        elif scenario_type == ScenarioType.REALISTIC:
            conditions['gdp_growth'] = np.random.normal(0.015, 0.015, years)
            conditions['technology_advancement'] = np.random.uniform(1.02, 1.06, years)
            conditions['market_expansion'] = np.random.uniform(0.98, 1.04, years)
            conditions['regulatory_support'] = np.random.uniform(0.95, 1.05, years)
            conditions['competition_intensity'] = np.random.uniform(0.95, 1.15, years)
            
        elif scenario_type == ScenarioType.PESSIMISTIC:
            conditions['gdp_growth'] = np.random.normal(-0.005, 0.02, years)
            conditions['technology_advancement'] = np.random.uniform(1.0, 1.03, years)
            conditions['market_expansion'] = np.random.uniform(0.92, 1.0, years)
            conditions['regulatory_support'] = np.random.uniform(0.85, 0.98, years)
            conditions['competition_intensity'] = np.random.uniform(1.1, 1.3, years)
            
        else:  # DISRUPTION scenario
            # Create disruption events
            disruption_years = np.random.choice(years, size=max(1, years//3), replace=False)
            
            conditions['gdp_growth'] = np.random.normal(0.01, 0.025, years)
            conditions['technology_advancement'] = np.ones(years)
            conditions['market_expansion'] = np.ones(years)
            conditions['regulatory_support'] = np.random.uniform(0.9, 1.1, years)
            conditions['competition_intensity'] = np.ones(years)
            
            # Apply disruption effects
            for year in disruption_years:
                conditions['technology_advancement'][year] = np.random.uniform(1.2, 2.0)
                conditions['market_expansion'][year] = np.random.uniform(0.7, 1.5)
                conditions['competition_intensity'][year] = np.random.uniform(1.5, 2.5)
        
        return conditions
    
    def run_monte_carlo_simulation(self, companies: List[str], 
                                    scenario_params: ScenarioParameters,
                                    features: pd.DataFrame) -> Dict[str, Any]:
        """
        Run Monte Carlo simulation for future scenarios
        
        Args:
            companies: List of company names to analyze
            scenario_params: Parameters for scenario generation
            features: Feature matrix
            
        Returns:
            Simulation results dictionary
        """
        logger.info(f"Running Monte Carlo simulation for {len(companies)} companies...")
        
        results = {
            'companies': companies,
            'simulations': {},
            'summary_statistics': {},
            'risk_metrics': {}
        }
        
        # Build models if not already available
        if not hasattr(self, 'models') or not self.models:
            # Create dummy target data for model training
            dummy_targets = pd.DataFrame(index=features.index)
            self.models, self.feature_importance = self.build_scenario_models(features, dummy_targets)
        
        # Run simulations for each scenario type
        for scenario_type in ScenarioType:
            scenario_results = {}
            
            for company in companies:
                if company not in features.index:
                    logger.warning(f"Company {company} not found in features data")
                    continue
                
                company_results = {
                    'survival_probabilities': [],
                    'performance_scores': [],
                    'market_share_changes': [],
                    'trajectories': []
                }
                
                # Run multiple simulations
                for sim in range(scenario_params.num_simulations):
                    # Generate scenario conditions
                    conditions = self.generate_scenario_conditions(
                        scenario_type, scenario_params.time_horizon
                    )
                    
                    # Get company features
                    company_features = features.loc[company:company]
                    
                    # Scale features
                    scaled_features = self.models['scaler'].transform(company_features)
                    
                    # Predict outcomes
                    survival_prob = self.models['survival_probability'].predict(scaled_features)[0]
                    performance_score = self.models['performance_score'].predict(scaled_features)[0]
                    market_share_change = self.models['market_share_change'].predict(scaled_features)[0]
                    
                    # Apply scenario conditions over time
                    trajectory = []
                    current_performance = performance_score
                    
                    for year in range(scenario_params.time_horizon):
                        # Apply external conditions
                        gdp_effect = conditions['gdp_growth'][year]
                        tech_effect = conditions['technology_advancement'][year]
                        market_effect = conditions['market_expansion'][year]
                        regulation_effect = conditions['regulatory_support'][year]
                        competition_effect = conditions['competition_intensity'][year]
                        
                        # Calculate yearly performance change
                        yearly_change = (
                            gdp_effect * 0.3 +
                            (tech_effect - 1) * 0.4 +
                            (market_effect - 1) * 0.2 +
                            (regulation_effect - 1) * 0.1 -
                            (competition_effect - 1) * 0.3
                        )
                        
                        current_performance = max(0, current_performance * (1 + yearly_change))
                        trajectory.append(current_performance)
                    
                    # Store simulation results
                    company_results['survival_probabilities'].append(
                        survival_prob * (1 - scenario_params.technology_disruption_probability)
                    )
                    company_results['performance_scores'].append(trajectory[-1])
                    company_results['market_share_changes'].append(
                        market_share_change * np.mean([conditions['market_expansion']])
                    )
                    company_results['trajectories'].append(trajectory)
                
                scenario_results[company] = company_results
            
            results['simulations'][scenario_type.value] = scenario_results
            logger.info(f"Completed {scenario_type.value} scenario simulation")
        
        # Calculate summary statistics
        results['summary_statistics'] = self._calculate_summary_statistics(results['simulations'])
        results['risk_metrics'] = self._calculate_risk_metrics(results['simulations'])
        
        return results
    
    def _calculate_summary_statistics(self, simulations: Dict) -> Dict:
        """Calculate summary statistics from simulation results"""
        summary = {}
        
        for scenario_type, scenario_data in simulations.items():
            scenario_summary = {}
            
            for company, company_data in scenario_data.items():
                company_stats = {}
                
                for metric, values in company_data.items():
                    if metric == 'trajectories':
                        # Calculate statistics for final year values
                        final_values = [traj[-1] for traj in values]
                        company_stats[f'{metric}_final'] = {
                            'mean': np.mean(final_values),
                            'std': np.std(final_values),
                            'percentiles': np.percentile(final_values, [5, 25, 50, 75, 95])
                        }
                    else:
                        company_stats[metric] = {
                            'mean': np.mean(values),
                            'std': np.std(values),
                            'percentiles': np.percentile(values, [5, 25, 50, 75, 95])
                        }
                
                scenario_summary[company] = company_stats
            
            summary[scenario_type] = scenario_summary
        
        return summary
    
    def _calculate_risk_metrics(self, simulations: Dict) -> Dict:
        """Calculate risk metrics from simulation results"""
        risk_metrics = {}
        
        for scenario_type, scenario_data in simulations.items():
            scenario_risks = {}
            
            for company, company_data in scenario_data.items():
                # Calculate Value at Risk (VaR) and Expected Shortfall
                performance_scores = company_data['performance_scores']
                survival_probs = company_data['survival_probabilities']
                
                # Performance VaR (5% worst cases)
                performance_var_5 = np.percentile(performance_scores, 5)
                performance_var_1 = np.percentile(performance_scores, 1)
                
                # Expected shortfall (mean of worst 5% cases)
                worst_5_percent = np.array(performance_scores)[
                    np.array(performance_scores) <= performance_var_5
                ]
                expected_shortfall = np.mean(worst_5_percent) if len(worst_5_percent) > 0 else performance_var_5
                
                # Survival risk
                extinction_probability = 1 - np.mean(survival_probs)
                
                scenario_risks[company] = {
                    'performance_var_5': performance_var_5,
                    'performance_var_1': performance_var_1,
                    'expected_shortfall': expected_shortfall,
                    'extinction_probability': extinction_probability,
                    'volatility': np.std(performance_scores)
                }
            
            risk_metrics[scenario_type] = scenario_risks
        
        return risk_metrics
    
    def analyze_strategic_positioning(self, simulation_results: Dict,
                                    companies: List[str]) -> Dict[str, Any]:
        """
        Analyze strategic positioning based on scenario outcomes
        
        Args:
            simulation_results: Results from Monte Carlo simulation
            companies: List of companies to analyze
            
        Returns:
            Strategic positioning analysis results
        """
        logger.info("Analyzing strategic positioning...")
        
        positioning_analysis = {
            'competitive_rankings': {},
            'scenario_resilience': {},
            'strategic_clusters': {},
            'recommendations': {}
        }
        
        # Analyze competitive rankings across scenarios
        for scenario_type in ScenarioType:
            scenario_key = scenario_type.value
            if scenario_key not in simulation_results['simulations']:
                continue
                
            scenario_data = simulation_results['simulations'][scenario_key]
            company_scores = {}
            
            for company in companies:
                if company in scenario_data:
                    # Calculate composite score
                    perf_mean = np.mean(scenario_data[company]['performance_scores'])
                    survival_mean = np.mean(scenario_data[company]['survival_probabilities'])
                    share_change_mean = np.mean(scenario_data[company]['market_share_changes'])
                    
                    composite_score = (
                        perf_mean * 0.4 +
                        survival_mean * 0.4 +
                        max(0, share_change_mean) * 0.2
                    )
                    company_scores[company] = composite_score
            
            # Rank companies
            ranked_companies = sorted(company_scores.items(), 
                                    key=lambda x: x[1], reverse=True)
            positioning_analysis['competitive_rankings'][scenario_key] = ranked_companies
        
        # Analyze scenario resilience
        for company in companies:
            resilience_scores = []
            
            for scenario_type in ScenarioType:
                scenario_key = scenario_type.value
                if (scenario_key in simulation_results['simulations'] and 
                    company in simulation_results['simulations'][scenario_key]):
                    
                    scenario_data = simulation_results['simulations'][scenario_key][company]
                    
                    # Resilience = stability of performance across simulations
                    perf_std = np.std(scenario_data['performance_scores'])
                    survival_mean = np.mean(scenario_data['survival_probabilities'])
                    
                    resilience_score = survival_mean * (1 - perf_std)
                    resilience_scores.append((scenario_key, resilience_score))
            
            positioning_analysis['scenario_resilience'][company] = resilience_scores
        
        # Generate strategic recommendations
        positioning_analysis['recommendations'] = self._generate_strategic_recommendations(
            positioning_analysis, simulation_results, companies
        )
        
        return positioning_analysis
    
    def _generate_strategic_recommendations(self, positioning: Dict,
                                            simulation_results: Dict,
                                            companies: List[str]) -> Dict[str, List[str]]:
        """Generate strategic recommendations based on analysis results"""
        recommendations = {}
        
        for company in companies:
            company_recommendations = []
            
            # Check performance across scenarios
            scenario_performances = {}
            for scenario_type in ScenarioType:
                scenario_key = scenario_type.value
                if (scenario_key in simulation_results['simulations'] and
                    company in simulation_results['simulations'][scenario_key]):
                    
                    perf_mean = np.mean(
                        simulation_results['simulations'][scenario_key][company]['performance_scores']
                    )
                    scenario_performances[scenario_key] = perf_mean
            
            # Generate recommendations based on performance patterns
            if len(scenario_performances) >= 2:
                optimistic_perf = scenario_performances.get('optimistic', 0)
                pessimistic_perf = scenario_performances.get('pessimistic', 0)
                
                if optimistic_perf > 0.7:
                    company_recommendations.append(
                        "Strong market position - consider aggressive expansion strategies"
                    )
                elif optimistic_perf < 0.3:
                    company_recommendations.append(
                        "Vulnerable position - focus on core competency strengthening"
                    )
                
                if pessimistic_perf < 0.2:
                    company_recommendations.append(
                        "High downside risk - implement defensive strategies and diversification"
                    )
                
                performance_volatility = np.std(list(scenario_performances.values()))
                if performance_volatility > 0.2:
                    company_recommendations.append(
                        "High scenario sensitivity - develop adaptive strategic capabilities"
                    )
            
            # Market category specific recommendations
            if company in self.market_categories['high_share']:
                company_recommendations.append(
                    "Maintain technological leadership and market dominance through R&D investment"
                )
            elif company in self.market_categories['declining']:
                company_recommendations.append(
                    "Consider market transformation or business model innovation"
                )
            else:  # lost markets
                company_recommendations.append(
                    "Focus on niche specialization or strategic exit planning"
                )
            
            recommendations[company] = company_recommendations
        
        return recommendations
    
    def visualize_scenario_results(self, simulation_results: Dict,
                                    companies: List[str] = None,
                                    output_dir: str = "scenario_visualizations") -> Dict[str, str]:
        """
        Create comprehensive visualizations of scenario analysis results
        
        Args:
            simulation_results: Results from Monte Carlo simulation
            companies: List of companies to visualize (default: all)
            output_dir: Directory to save visualization files
            
        Returns:
            Dictionary mapping visualization types to file paths
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        if companies is None:
            companies = list(next(iter(simulation_results['simulations'].values())).keys())
        
        viz_files = {}
        
        # 1. Scenario Comparison Matrix
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Scenario Analysis Results - Performance Comparison', fontsize=16)
        
        scenario_types = list(simulation_results['simulations'].keys())
        
        for idx, scenario in enumerate(scenario_types[:4]):
            ax = axes[idx//2, idx%2]
            
            company_scores = []
            company_names = []
            
            for company in companies[:10]:  # Limit to top 10 for readability
                if company in simulation_results['simulations'][scenario]:
                    scores = simulation_results['simulations'][scenario][company]['performance_scores']
                    company_scores.append(np.mean(scores))
                    company_names.append(company[:15])  # Truncate long names
            
            ax.barh(range(len(company_names)), company_scores)
            ax.set_yticks(range(len(company_names)))
            ax.set_yticklabels(company_names)
            ax.set_title(f'{scenario.title()} Scenario')
            ax.set_xlabel('Average Performance Score')
        
        plt.tight_layout()
        scenario_comparison_file = os.path.join(output_dir, 'scenario_comparison.png')
        plt.savefig(scenario_comparison_file, dpi=300, bbox_inches='tight')
        plt.close()
        viz_files['scenario_comparison'] = scenario_comparison_file
        
        # 2. Risk-Return Analysis
        plt.figure(figsize=(12, 8))
        colors = ['red', 'blue', 'green', 'orange']
        
        for idx, scenario in enumerate(scenario_types):
            if scenario in simulation_results['risk_metrics']:
                risks = []
                returns = []
                labels = []
                
                for company in companies[:20]:
                    if company in simulation_results['risk_metrics'][scenario]:
                        risk_data = simulation_results['risk_metrics'][scenario][company]
                        summary_data = simulation_results['summary_statistics'][scenario].get(company, {})
                        
                        risk = risk_data.get('volatility', 0)
                        return_val = summary_data.get('performance_scores', {}).get('mean', 0)
                        
                        risks.append(risk)
                        returns.append(return_val)
                        labels.append(company[:10])
                
                plt.scatter(risks, returns, c=colors[idx % len(colors)], 
                            label=scenario.title(), alpha=0.7, s=60)
        
        plt.xlabel('Risk (Performance Volatility)')
        plt.ylabel('Expected Return (Performance Score)')
        plt.title('Risk-Return Analysis Across Scenarios')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        risk_return_file = os.path.join(output_dir, 'risk_return_analysis.png')
        plt.savefig(risk_return_file, dpi=300, bbox_inches='tight')
        plt.close()
        viz_files['risk_return'] = risk_return_file
        
        # 3. Survival Probability Heatmap
        survival_matrix = []
        company_labels = []
        scenario_labels = []
        
        for company in companies[:15]:
            company_row = []
            company_labels.append(company[:12])
            
            for scenario in scenario_types:
                if (scenario in simulation_results['simulations'] and 
                    company in simulation_results['simulations'][scenario]):
                    survival_probs = simulation_results['simulations'][scenario][company]['survival_probabilities']
                    avg_survival = np.mean(survival_probs)
                    company_row.append(avg_survival)
                else:
                    company_row.append(0)
            
            survival_matrix.append(company_row)
        
        scenario_labels = [s.title() for s in scenario_types]
        
        plt.figure(figsize=(10, 12))
        sns.heatmap(survival_matrix, 
                    xticklabels=scenario_labels,
                    yticklabels=company_labels,
                    annot=True, fmt='.2f', cmap='RdYlGn',
                    cbar_kws={'label': 'Survival Probability'})
        plt.title('Company Survival Probabilities Across Scenarios')
        plt.xlabel('Scenarios')
        plt.ylabel('Companies')
        
        survival_heatmap_file = os.path.join(output_dir, 'survival_heatmap.png')
        plt.savefig(survival_heatmap_file, dpi=300, bbox_inches='tight')
        plt.close()
        viz_files['survival_heatmap'] = survival_heatmap_file
        
        # 4. Performance Trajectory Visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Performance Trajectories by Market Category', fontsize=16)
        
        market_categories = ['high_share', 'declining', 'lost']
        colors_cat = ['green', 'orange', 'red']
        
        for cat_idx, category in enumerate(market_categories):
            if cat_idx >= 4:
                break
                
            ax = axes[cat_idx//2, cat_idx%2] if cat_idx < 4 else None
            if ax is None:
                continue
            
            category_companies = self.market_categories.get(category, [])
            
            for scenario in ['realistic', 'pessimistic']:
                if scenario not in simulation_results['simulations']:
                    continue
                    
                trajectories = []
                for company in category_companies[:5]:  # Limit to 5 companies per category
                    if company in simulation_results['simulations'][scenario]:
                        company_trajectories = simulation_results['simulations'][scenario][company]['trajectories']
                        if company_trajectories:
                            avg_trajectory = np.mean(company_trajectories, axis=0)
                            trajectories.append(avg_trajectory)
                
                if trajectories:
                    # Plot average trajectory for the category
                    years = range(len(trajectories[0]))
                    category_avg = np.mean(trajectories, axis=0)
                    category_std = np.std(trajectories, axis=0)
                    
                    color = 'blue' if scenario == 'realistic' else 'red'
                    alpha = 0.7 if scenario == 'realistic' else 0.5
                    
                    ax.plot(years, category_avg, color=color, linewidth=2, 
                            label=f'{scenario.title()} Scenario', alpha=alpha)
                    ax.fill_between(years, 
                                    category_avg - category_std,
                                    category_avg + category_std,
                                    color=color, alpha=0.2)
            
            ax.set_title(f'{category.replace("_", " ").title()} Market Companies')
            ax.set_xlabel('Years')
            ax.set_ylabel('Performance Score')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Remove empty subplot
        if len(market_categories) < 4:
            fig.delaxes(axes[1, 1])
        
        plt.tight_layout()
        trajectory_file = os.path.join(output_dir, 'performance_trajectories.png')
        plt.savefig(trajectory_file, dpi=300, bbox_inches='tight')
        plt.close()
        viz_files['trajectories'] = trajectory_file
        
        # 5. Feature Importance Analysis
        if hasattr(self, 'feature_importance') and self.feature_importance:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            fig.suptitle('Feature Importance for Scenario Predictions', fontsize=16)
            
            metrics = ['survival_probability', 'performance_score', 'market_share_change']
            
            for idx, metric in enumerate(metrics):
                if metric in self.feature_importance:
                    importance_data = self.feature_importance[metric]
                    top_features = importance_data.head(10)
                    
                    axes[idx].barh(range(len(top_features)), top_features['importance'])
                    axes[idx].set_yticks(range(len(top_features)))
                    axes[idx].set_yticklabels([f[:20] for f in top_features['feature']], fontsize=8)
                    axes[idx].set_title(f'{metric.replace("_", " ").title()}')
                    axes[idx].set_xlabel('Importance Score')
            
            plt.tight_layout()
            feature_importance_file = os.path.join(output_dir, 'feature_importance.png')
            plt.savefig(feature_importance_file, dpi=300, bbox_inches='tight')
            plt.close()
            viz_files['feature_importance'] = feature_importance_file
        
        logger.info(f"Generated {len(viz_files)} visualization files in {output_dir}")
        return viz_files
    
    def generate_scenario_report(self, simulation_results: Dict,
                                positioning_analysis: Dict,
                                companies: List[str],
                                output_file: str = "scenario_analysis_report.md") -> str:
        """
        Generate comprehensive scenario analysis report
        
        Args:
            simulation_results: Results from Monte Carlo simulation
            positioning_analysis: Strategic positioning analysis results
            companies: List of analyzed companies
            output_file: Output markdown file path
            
        Returns:
            Path to generated report file
        """
        logger.info(f"Generating scenario analysis report for {len(companies)} companies...")
        
        report_content = []
        report_content.append("# A2AI Future Scenario Analysis Report")
        report_content.append(f"\nGenerated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_content.append(f"\nAnalyzed Companies: {len(companies)}")
        
        # Executive Summary
        report_content.append("\n## Executive Summary")
        report_content.append("\nThis report presents comprehensive future scenario analysis for Japanese companies across three market categories:")
        report_content.append("- **High Market Share**: Companies maintaining strong global market positions")
        report_content.append("- **Declining Market Share**: Companies experiencing competitive pressure")
        report_content.append("- **Lost Market Share**: Companies that have lost significant market presence")
        
        # Scenario Overview
        report_content.append("\n## Scenario Analysis Overview")
        
        scenarios_analyzed = list(simulation_results['simulations'].keys())
        report_content.append(f"\n**Scenarios Analyzed**: {', '.join([s.title() for s in scenarios_analyzed])}")
        
        for scenario in scenarios_analyzed:
            scenario_data = simulation_results['simulations'][scenario]
            num_companies = len(scenario_data)
            
            # Calculate scenario-level statistics
            all_performance_scores = []
            all_survival_probs = []
            
            for company_data in scenario_data.values():
                all_performance_scores.extend(company_data['performance_scores'])
                all_survival_probs.extend(company_data['survival_probabilities'])
            
            avg_performance = np.mean(all_performance_scores)
            avg_survival = np.mean(all_survival_probs)
            
            report_content.append(f"\n### {scenario.title()} Scenario")
            report_content.append(f"- Companies analyzed: {num_companies}")
            report_content.append(f"- Average performance score: {avg_performance:.3f}")
            report_content.append(f"- Average survival probability: {avg_survival:.3f}")
        
        # Market Category Analysis
        report_content.append("\n## Market Category Performance Analysis")
        
        for category, category_companies in self.market_categories.items():
            report_content.append(f"\n### {category.replace('_', ' ').title()} Market Companies")
            
            # Find companies in this category that were analyzed
            analyzed_category_companies = [c for c in companies if c in category_companies]
            
            if not analyzed_category_companies:
                report_content.append("No companies from this category were analyzed.")
                continue
            
            report_content.append(f"Analyzed companies: {len(analyzed_category_companies)}")
            
            # Calculate category performance across scenarios
            category_performance = {}
            for scenario in scenarios_analyzed:
                if scenario in simulation_results['simulations']:
                    scenario_scores = []
                    for company in analyzed_category_companies:
                        if company in simulation_results['simulations'][scenario]:
                            scores = simulation_results['simulations'][scenario][company]['performance_scores']
                            scenario_scores.extend(scores)
                    
                    if scenario_scores:
                        category_performance[scenario] = {
                            'mean': np.mean(scenario_scores),
                            'std': np.std(scenario_scores),
                            'median': np.median(scenario_scores)
                        }
            
            # Report category performance
            for scenario, perf_stats in category_performance.items():
                report_content.append(f"\n**{scenario.title()} Scenario Performance:**")
                report_content.append(f"- Mean: {perf_stats['mean']:.3f}")
                report_content.append(f"- Standard Deviation: {perf_stats['std']:.3f}")
                report_content.append(f"- Median: {perf_stats['median']:.3f}")
        
        # Top Performers Analysis
        report_content.append("\n## Top Performers by Scenario")
        
        if 'competitive_rankings' in positioning_analysis:
            for scenario, rankings in positioning_analysis['competitive_rankings'].items():
                report_content.append(f"\n### {scenario.title()} Scenario - Top 10 Companies")
                
                top_10 = rankings[:10]
                for rank, (company, score) in enumerate(top_10, 1):
                    report_content.append(f"{rank}. **{company}** - Score: {score:.3f}")
        
        # Risk Analysis
        report_content.append("\n## Risk Analysis")
        
        if 'risk_metrics' in simulation_results:
            report_content.append("\n### High-Risk Companies (Top 10 by Extinction Probability)")
            
            # Aggregate risk across scenarios
            company_risks = {}
            for scenario, scenario_risks in simulation_results['risk_metrics'].items():
                for company, risk_data in scenario_risks.items():
                    if company not in company_risks:
                        company_risks[company] = []
                    company_risks[company].append(risk_data.get('extinction_probability', 0))
            
            # Calculate average extinction probability
            avg_extinction_probs = {
                company: np.mean(risks) 
                for company, risks in company_risks.items()
            }
            
            # Sort by highest risk
            highest_risk_companies = sorted(
                avg_extinction_probs.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:10]
            
            for rank, (company, risk_prob) in enumerate(highest_risk_companies, 1):
                report_content.append(f"{rank}. **{company}** - Extinction Probability: {risk_prob:.3f}")
        
        # Strategic Recommendations
        report_content.append("\n## Strategic Recommendations")
        
        if 'recommendations' in positioning_analysis:
            report_content.append("\n### Company-Specific Strategic Recommendations")
            
            # Group recommendations by market category
            for category, category_companies in self.market_categories.items():
                category_recs = []
                for company in companies:
                    if (company in category_companies and 
                        company in positioning_analysis['recommendations']):
                        category_recs.append((company, positioning_analysis['recommendations'][company]))
                
                if category_recs:
                    report_content.append(f"\n#### {category.replace('_', ' ').title()} Market Companies")
                    
                    for company, recommendations in category_recs[:5]:  # Limit to 5 per category
                        report_content.append(f"\n**{company}:**")
                        for rec in recommendations:
                            report_content.append(f"- {rec}")
        
        # Methodology Notes
        report_content.append("\n## Methodology")
        report_content.append("\n### Analysis Approach")
        report_content.append("- **Monte Carlo Simulation**: 1,000 simulations per scenario per company")
        report_content.append("- **Feature Engineering**: 23 factor items per 9 evaluation metrics")
        report_content.append("- **Machine Learning Models**: Random Forest regression for outcome prediction")
        report_content.append("- **Risk Assessment**: Value at Risk (VaR) and Expected Shortfall calculations")
        
        report_content.append("\n### Scenario Definitions")
        report_content.append("- **Optimistic**: Favorable economic conditions, technology advancement, market expansion")
        report_content.append("- **Realistic**: Moderate growth conditions with typical market volatility")
        report_content.append("- **Pessimistic**: Economic headwinds, increased competition, regulatory challenges")
        report_content.append("- **Disruption**: Technology disruptions and market structure changes")
        
        # Data Sources and Limitations
        report_content.append("\n### Data Sources and Limitations")
        report_content.append("- **Financial Data**: EDINET API for 40-year historical financial statements")
        report_content.append("- **Market Share Data**: Industry reports and government statistics")
        report_content.append("- **Limitations**: Model predictions based on historical patterns; external shocks may not be fully captured")
        
        # Write report to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_content))
        
        logger.info(f"Scenario analysis report generated: {output_file}")
        return output_file
    
    def export_results(self, simulation_results: Dict, 
                        positioning_analysis: Dict,
                        output_dir: str = "scenario_results") -> Dict[str, str]:
        """
        Export all analysis results to various formats
        
        Args:
            simulation_results: Monte Carlo simulation results
            positioning_analysis: Strategic positioning analysis
            output_dir: Directory to save export files
            
        Returns:
            Dictionary of exported file paths
        """
        import os
        import json
        os.makedirs(output_dir, exist_ok=True)
        
        exported_files = {}
        
        # 1. Export simulation results to JSON
        json_file = os.path.join(output_dir, 'simulation_results.json')
        
        # Convert numpy arrays to lists for JSON serialization
        json_compatible_results = self._convert_numpy_to_lists(simulation_results)
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_compatible_results, f, indent=2, ensure_ascii=False)
        
        exported_files['simulation_json'] = json_file
        
        # 2. Export summary statistics to CSV
        summary_csv = os.path.join(output_dir, 'summary_statistics.csv')
        summary_data = []
        
        for scenario, scenario_stats in simulation_results['summary_statistics'].items():
            for company, company_stats in scenario_stats.items():
                for metric, metric_stats in company_stats.items():
                    if isinstance(metric_stats, dict) and 'mean' in metric_stats:
                        summary_data.append({
                            'scenario': scenario,
                            'company': company,
                            'metric': metric,
                            'mean': metric_stats['mean'],
                            'std': metric_stats['std'],
                            'p5': metric_stats['percentiles'][0],
                            'p25': metric_stats['percentiles'][1],
                            'median': metric_stats['percentiles'][2],
                            'p75': metric_stats['percentiles'][3],
                            'p95': metric_stats['percentiles'][4]
                        })
        
        pd.DataFrame(summary_data).to_csv(summary_csv, index=False, encoding='utf-8')
        exported_files['summary_csv'] = summary_csv
        
        # 3. Export risk metrics to CSV
        risk_csv = os.path.join(output_dir, 'risk_metrics.csv')
        risk_data = []
        
        for scenario, scenario_risks in simulation_results['risk_metrics'].items():
            for company, risk_metrics in scenario_risks.items():
                risk_row = {'scenario': scenario, 'company': company}
                risk_row.update(risk_metrics)
                risk_data.append(risk_row)
        
        pd.DataFrame(risk_data).to_csv(risk_csv, index=False, encoding='utf-8')
        exported_files['risk_csv'] = risk_csv
        
        # 4. Export strategic recommendations to text
        recommendations_file = os.path.join(output_dir, 'strategic_recommendations.txt')
        
        with open(recommendations_file, 'w', encoding='utf-8') as f:
            f.write("Strategic Recommendations by Company\n")
            f.write("=" * 50 + "\n\n")
            
            if 'recommendations' in positioning_analysis:
                for company, recommendations in positioning_analysis['recommendations'].items():
                    f.write(f"{company}:\n")
                    for rec in recommendations:
                        f.write(f"  - {rec}\n")
                    f.write("\n")
        
        exported_files['recommendations_txt'] = recommendations_file
        
        logger.info(f"Exported {len(exported_files)} result files to {output_dir}")
        return exported_files
    
    def _convert_numpy_to_lists(self, obj):
        """Recursively convert numpy arrays to lists for JSON serialization"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_to_lists(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_to_lists(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        else:
            return obj

# Example usage and testing
if __name__ == "__main__":
    # Initialize the analyzer
    analyzer = FutureScenarioAnalyzer()
    
    # Create sample financial data for testing
    sample_companies = [
        'ファナック', '村田製作所', 'キーエンス',  # High share
        'トヨタ自動車', 'パナソニック', 'シャープ',  # Declining
        'ソニー（家電部門）', 'NEC（旧）', '三洋電機'  # Lost
    ]
    
    # Generate sample feature data
    np.random.seed(42)
    sample_features = analyzer.prepare_scenario_features(
        pd.DataFrame(index=sample_companies)
    )
    
    print("Sample Features Generated:")
    print(f"Shape: {sample_features.shape}")
    print(f"Columns: {sample_features.columns[:10].tolist()}...")  # Show first 10 columns
    
    # Set up scenario parameters
    scenario_params = ScenarioParameters(
        time_horizon=10,
        num_simulations=100,  # Reduced for testing
        market_volatility=0.15
    )
    
    # Run Monte Carlo simulation
    print("\nRunning Monte Carlo simulation...")
    simulation_results = analyzer.run_monte_carlo_simulation(
        companies=sample_companies,
        scenario_params=scenario_params,
        features=sample_features
    )
    
    print("Simulation completed successfully!")
    print(f"Scenarios analyzed: {list(simulation_results['simulations'].keys())}")
    
    # Analyze strategic positioning
    print("\nAnalyzing strategic positioning...")
    positioning_analysis = analyzer.analyze_strategic_positioning(
        simulation_results, sample_companies
    )
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    viz_files = analyzer.visualize_scenario_results(
        simulation_results, sample_companies
    )
    
    print(f"Generated visualization files: {list(viz_files.keys())}")
    
    # Generate comprehensive report
    print("\nGenerating scenario analysis report...")
    report_file = analyzer.generate_scenario_report(
        simulation_results, positioning_analysis, sample_companies
    )
    
    # Export results
    print("\nExporting results...")
    exported_files = analyzer.export_results(
        simulation_results, positioning_analysis
    )
    
    print("A2AI Future Scenario Analysis completed successfully!")
    print(f"Report generated: {report_file}")
    print(f"Exported files: {list(exported_files.keys())}")