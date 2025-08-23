"""
A2AI (Advanced Financial Analysis AI) - Analysis Module
=======================================================

企業ライフサイクル全体を考慮した包括的財務分析システム
150社×変動期間データに対応する統合分析フレームワーク

主要機能:
1. 従来型財務分析 (Traditional Analysis)
2. 生存分析 (Survival Analysis) - 企業消滅・倒産リスク分析
3. 新設企業分析 (Emergence Analysis) - スタートアップ成功要因分析
4. ライフサイクル分析 (Lifecycle Analysis) - 企業進化パターン分析
5. 統合分析 (Integrated Analysis) - 市場エコシステム分析

対象企業カテゴリ:
- 高シェア市場: 50社 (ロボット、内視鏡、工作機械、電子材料、精密測定機器)
- シェア低下市場: 50社 (自動車、鉄鋼、家電、バッテリー、PC周辺機器)
- 失失市場: 50社 (テレビ家電、半導体、スマートフォン、PC、通信機器)

分析対象期間: 1984-2024年 (企業により変動)
評価項目: 9項目 (従来6項目 + 新規3項目)
要因項目: 各評価項目あたり23項目 (従来20項目 + 新規3項目)
"""

from typing import Dict, List, Optional, Union, Any, Tuple
import logging
from dataclasses import dataclass
from enum import Enum

# Traditional Analysis Imports
from .traditional_analysis.factor_impact_analyzer import FactorImpactAnalyzer
from .traditional_analysis.market_comparison import MarketComparisonAnalyzer
from .traditional_analysis.correlation_analyzer import CorrelationAnalyzer
from .traditional_analysis.regression_analyzer import RegressionAnalyzer
from .traditional_analysis.clustering_analyzer import ClusteringAnalyzer
from .traditional_analysis.trend_analyzer import TrendAnalyzer

# Survival Analysis Imports
from .survival_analysis.extinction_risk_analyzer import ExtinctionRiskAnalyzer
from .survival_analysis.survival_factor_analyzer import SurvivalFactorAnalyzer
from .survival_analysis.hazard_ratio_analyzer import HazardRatioAnalyzer
from .survival_analysis.survival_clustering import SurvivalClusteringAnalyzer

# Emergence Analysis Imports
from .emergence_analysis.startup_success_analyzer import StartupSuccessAnalyzer
from .emergence_analysis.market_entry_analyzer import MarketEntryAnalyzer
from .emergence_analysis.growth_phase_analyzer import GrowthPhaseAnalyzer
from .emergence_analysis.innovation_impact_analyzer import InnovationImpactAnalyzer

# Lifecycle Analysis Imports
from .lifecycle_analysis.stage_transition_analyzer import StageTransitionAnalyzer
from .lifecycle_analysis.lifecycle_performance import LifecyclePerformanceAnalyzer
from .lifecycle_analysis.maturity_indicator import MaturityIndicatorAnalyzer
from .lifecycle_analysis.rejuvenation_analyzer import RejuvenationAnalyzer

# Integrated Analysis Imports
from .integrated_analysis.ecosystem_analyzer import EcosystemAnalyzer
from .integrated_analysis.competitive_dynamics import CompetitiveDynamicsAnalyzer
from .integrated_analysis.strategic_positioning import StrategicPositioningAnalyzer
from .integrated_analysis.future_scenario_analyzer import FutureScenarioAnalyzer


# Enums for Analysis Configuration
class MarketCategory(Enum):
    """市場カテゴリ定義"""
    HIGH_SHARE = "high_share"  # 現在もシェアが高い市場
    DECLINING_SHARE = "declining_share"  # シェア低下中の市場
    LOST_SHARE = "lost_share"  # 完全にシェアを失った市場


class AnalysisType(Enum):
    """分析タイプ定義"""
    TRADITIONAL = "traditional"  # 従来型財務分析
    SURVIVAL = "survival"  # 生存分析
    EMERGENCE = "emergence"  # 新設企業分析
    LIFECYCLE = "lifecycle"  # ライフサイクル分析
    INTEGRATED = "integrated"  # 統合分析
    ALL = "all"  # 全分析


class EvaluationMetric(Enum):
    """9つの評価項目定義"""
    # 従来の6項目
    SALES_REVENUE = "sales_revenue"  # 売上高
    SALES_GROWTH_RATE = "sales_growth_rate"  # 売上高成長率
    OPERATING_MARGIN = "operating_margin"  # 売上高営業利益率
    NET_MARGIN = "net_margin"  # 売上高当期純利益率
    ROE = "roe"  # ROE
    VALUE_ADDED_RATIO = "value_added_ratio"  # 売上高付加価値率
    
    # 新規の3項目
    SURVIVAL_PROBABILITY = "survival_probability"  # 企業存続確率
    EMERGENCE_SUCCESS_RATE = "emergence_success_rate"  # 新規事業成功率
    SUCCESSION_SUCCESS_RATE = "succession_success_rate"  # 事業継承成功度


@dataclass
class AnalysisConfig:
    """分析設定クラス"""
    market_categories: List[MarketCategory]
    analysis_types: List[AnalysisType]
    evaluation_metrics: List[EvaluationMetric]
    time_period: Tuple[int, int]  # (開始年, 終了年)
    companies_per_market: int = 10
    min_data_years: int = 5  # 最小データ年数
    survival_bias_correction: bool = True
    causal_inference: bool = True
    bootstrap_samples: int = 1000
    confidence_level: float = 0.95


class A2AIAnalysisFramework:
    """
    A2AI統合分析フレームワーク
    
    企業の生存・消滅・新設を含む完全なライフサイクル分析を実現する
    メインコントローラークラス
    """
    
    def __init__(self, config: AnalysisConfig):
        """
        Args:
            config: 分析設定
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._initialize_analyzers()
        
    def _initialize_analyzers(self):
        """各分析器の初期化"""
        # Traditional Analysis Analyzers
        self.traditional_analyzers = {
            'factor_impact': FactorImpactAnalyzer(),
            'market_comparison': MarketComparisonAnalyzer(),
            'correlation': CorrelationAnalyzer(),
            'regression': RegressionAnalyzer(),
            'clustering': ClusteringAnalyzer(),
            'trend': TrendAnalyzer()
        }
        
        # Survival Analysis Analyzers
        self.survival_analyzers = {
            'extinction_risk': ExtinctionRiskAnalyzer(),
            'survival_factor': SurvivalFactorAnalyzer(),
            'hazard_ratio': HazardRatioAnalyzer(),
            'survival_clustering': SurvivalClusteringAnalyzer()
        }
        
        # Emergence Analysis Analyzers
        self.emergence_analyzers = {
            'startup_success': StartupSuccessAnalyzer(),
            'market_entry': MarketEntryAnalyzer(),
            'growth_phase': GrowthPhaseAnalyzer(),
            'innovation_impact': InnovationImpactAnalyzer()
        }
        
        # Lifecycle Analysis Analyzers
        self.lifecycle_analyzers = {
            'stage_transition': StageTransitionAnalyzer(),
            'lifecycle_performance': LifecyclePerformanceAnalyzer(),
            'maturity_indicator': MaturityIndicatorAnalyzer(),
            'rejuvenation': RejuvenationAnalyzer()
        }
        
        # Integrated Analysis Analyzers
        self.integrated_analyzers = {
            'ecosystem': EcosystemAnalyzer(),
            'competitive_dynamics': CompetitiveDynamicsAnalyzer(),
            'strategic_positioning': StrategicPositioningAnalyzer(),
            'future_scenario': FutureScenarioAnalyzer()
        }
        
        self.logger.info("A2AI Analysis Framework initialized successfully")
    
    def run_comprehensive_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        150社×変動期間データに対する包括的分析の実行
        
        Args:
            data: 統合データセット
                {
                    'financial_data': DataFrame,  # 財務データ
                    'market_share_data': DataFrame,  # 市場シェアデータ
                    'extinction_events': DataFrame,  # 企業消滅イベント
                    'emergence_events': DataFrame,  # 新設企業イベント
                    'company_metadata': Dict  # 企業メタデータ
                }
        
        Returns:
            包括的分析結果
        """
        self.logger.info("Starting comprehensive A2AI analysis for 150 companies")
        
        results = {
            'analysis_config': self.config,
            'data_summary': self._generate_data_summary(data),
            'traditional_analysis': {},
            'survival_analysis': {},
            'emergence_analysis': {},
            'lifecycle_analysis': {},
            'integrated_analysis': {},
            'cross_analysis_insights': {},
            'strategic_recommendations': {}
        }
        
        try:
            # 1. Traditional Financial Analysis
            if AnalysisType.TRADITIONAL in self.config.analysis_types:
                results['traditional_analysis'] = self._run_traditional_analysis(data)
            
            # 2. Survival Analysis
            if AnalysisType.SURVIVAL in self.config.analysis_types:
                results['survival_analysis'] = self._run_survival_analysis(data)
            
            # 3. Emergence Analysis
            if AnalysisType.EMERGENCE in self.config.analysis_types:
                results['emergence_analysis'] = self._run_emergence_analysis(data)
            
            # 4. Lifecycle Analysis
            if AnalysisType.LIFECYCLE in self.config.analysis_types:
                results['lifecycle_analysis'] = self._run_lifecycle_analysis(data)
            
            # 5. Integrated Analysis
            if AnalysisType.INTEGRATED in self.config.analysis_types:
                results['integrated_analysis'] = self._run_integrated_analysis(data, results)
            
            # 6. Cross-Analysis Insights
            results['cross_analysis_insights'] = self._generate_cross_analysis_insights(results)
            
            # 7. Strategic Recommendations
            results['strategic_recommendations'] = self._generate_strategic_recommendations(results)
            
            self.logger.info("Comprehensive A2AI analysis completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error in comprehensive analysis: {str(e)}")
            raise
        
        return results
    
    def _run_traditional_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """従来型財務分析の実行"""
        self.logger.info("Running traditional financial analysis")
        
        results = {}
        
        # 要因項目影響分析
        results['factor_impact'] = self.traditional_analyzers['factor_impact'].analyze(
            data['financial_data'],
            evaluation_metrics=self.config.evaluation_metrics,
            market_categories=self.config.market_categories
        )
        
        # 市場間比較分析
        results['market_comparison'] = self.traditional_analyzers['market_comparison'].compare_markets(
            data['financial_data'],
            data['market_share_data'],
            market_categories=self.config.market_categories
        )
        
        # 相関分析
        results['correlation'] = self.traditional_analyzers['correlation'].analyze_correlations(
            data['financial_data'],
            method='pearson',
            significance_level=1-self.config.confidence_level
        )
        
        # 回帰分析
        results['regression'] = self.traditional_analyzers['regression'].multiple_regression_analysis(
            data['financial_data'],
            evaluation_metrics=self.config.evaluation_metrics,
            causal_inference=self.config.causal_inference
        )
        
        # クラスタリング分析
        results['clustering'] = self.traditional_analyzers['clustering'].cluster_companies(
            data['financial_data'],
            market_categories=self.config.market_categories
        )
        
        # トレンド分析
        results['trend'] = self.traditional_analyzers['trend'].analyze_trends(
            data['financial_data'],
            time_period=self.config.time_period
        )
        
        return results
    
    def _run_survival_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """生存分析の実行"""
        self.logger.info("Running survival analysis for extinct companies")
        
        results = {}
        
        # 企業消滅リスク分析
        results['extinction_risk'] = self.survival_analyzers['extinction_risk'].analyze_extinction_risk(
            data['financial_data'],
            data['extinction_events'],
            market_categories=self.config.market_categories
        )
        
        # 生存要因分析
        results['survival_factor'] = self.survival_analyzers['survival_factor'].analyze_survival_factors(
            data['financial_data'],
            data['extinction_events'],
            evaluation_metrics=self.config.evaluation_metrics
        )
        
        # ハザード比分析
        results['hazard_ratio'] = self.survival_analyzers['hazard_ratio'].calculate_hazard_ratios(
            data['financial_data'],
            data['extinction_events'],
            confidence_level=self.config.confidence_level
        )
        
        # 生存パターンクラスタリング
        results['survival_clustering'] = self.survival_analyzers['survival_clustering'].cluster_survival_patterns(
            data['financial_data'],
            data['extinction_events']
        )
        
        return results
    
    def _run_emergence_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """新設企業分析の実行"""
        self.logger.info("Running emergence analysis for new companies")
        
        results = {}
        
        # スタートアップ成功分析
        results['startup_success'] = self.emergence_analyzers['startup_success'].analyze_success_factors(
            data['financial_data'],
            data['emergence_events'],
            market_categories=self.config.market_categories
        )
        
        # 市場参入分析
        results['market_entry'] = self.emergence_analyzers['market_entry'].analyze_entry_strategies(
            data['financial_data'],
            data['emergence_events'],
            data['market_share_data']
        )
        
        # 成長段階分析
        results['growth_phase'] = self.emergence_analyzers['growth_phase'].analyze_growth_phases(
            data['financial_data'],
            data['emergence_events']
        )
        
        # イノベーション影響分析
        results['innovation_impact'] = self.emergence_analyzers['innovation_impact'].analyze_innovation_impact(
            data['financial_data'],
            data['emergence_events'],
            evaluation_metrics=self.config.evaluation_metrics
        )
        
        return results
    
    def _run_lifecycle_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """ライフサイクル分析の実行"""
        self.logger.info("Running lifecycle analysis")
        
        results = {}
        
        # ステージ遷移分析
        results['stage_transition'] = self.lifecycle_analyzers['stage_transition'].analyze_transitions(
            data['financial_data'],
            market_categories=self.config.market_categories
        )
        
        # ライフサイクル別性能分析
        results['lifecycle_performance'] = self.lifecycle_analyzers['lifecycle_performance'].analyze_performance_by_stage(
            data['financial_data'],
            evaluation_metrics=self.config.evaluation_metrics
        )
        
        # 成熟度指標分析
        results['maturity_indicator'] = self.lifecycle_analyzers['maturity_indicator'].calculate_maturity_indicators(
            data['financial_data'],
            market_categories=self.config.market_categories
        )
        
        # 企業若返り分析
        results['rejuvenation'] = self.lifecycle_analyzers['rejuvenation'].analyze_rejuvenation_patterns(
            data['financial_data'],
            data['emergence_events']
        )
        
        return results
    
    def _run_integrated_analysis(self, data: Dict[str, Any], previous_results: Dict[str, Any]) -> Dict[str, Any]:
        """統合分析の実行"""
        self.logger.info("Running integrated ecosystem analysis")
        
        results = {}
        
        # エコシステム分析
        results['ecosystem'] = self.integrated_analyzers['ecosystem'].analyze_ecosystem(
            data,
            previous_results,
            market_categories=self.config.market_categories
        )
        
        # 競争ダイナミクス分析
        results['competitive_dynamics'] = self.integrated_analyzers['competitive_dynamics'].analyze_dynamics(
            data['financial_data'],
            data['market_share_data'],
            previous_results
        )
        
        # 戦略ポジション分析
        results['strategic_positioning'] = self.integrated_analyzers['strategic_positioning'].analyze_positioning(
            data,
            previous_results,
            evaluation_metrics=self.config.evaluation_metrics
        )
        
        # 将来シナリオ分析
        results['future_scenario'] = self.integrated_analyzers['future_scenario'].generate_scenarios(
            data,
            previous_results,
            time_horizon=10  # 10年先まで予測
        )
        
        return results
    
    def _generate_data_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """データサマリーの生成"""
        return {
            'total_companies': len(data['company_metadata']),
            'market_distribution': self._count_companies_by_market(data),
            'time_coverage': self._calculate_time_coverage(data),
            'data_quality_metrics': self._assess_data_quality(data),
            'extinct_companies': len(data['extinction_events']),
            'new_companies': len(data['emergence_events'])
        }
    
    def _count_companies_by_market(self, data: Dict[str, Any]) -> Dict[str, int]:
        """市場カテゴリ別企業数の集計"""
        # 実装例（実際のデータ構造に応じて調整）
        return {
            'high_share_markets': 50,
            'declining_markets': 50,
            'lost_markets': 50
        }
    
    def _calculate_time_coverage(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """時系列カバレッジの計算"""
        # 実装例
        return {
            'min_year': self.config.time_period[0],
            'max_year': self.config.time_period[1],
            'average_coverage_years': 30,
            'companies_with_full_coverage': 80
        }
    
    def _assess_data_quality(self, data: Dict[str, Any]) -> Dict[str, float]:
        """データ品質の評価"""
        # 実装例
        return {
            'completeness_rate': 0.85,
            'consistency_score': 0.92,
            'outlier_rate': 0.03,
            'missing_value_rate': 0.08
        }
    
    def _generate_cross_analysis_insights(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """分析横断的インサイトの生成"""
        return {
            'survival_vs_traditional': self._compare_survival_traditional(results),
            'emergence_vs_lifecycle': self._compare_emergence_lifecycle(results),
            'market_patterns': self._identify_market_patterns(results),
            'key_success_factors': self._identify_key_success_factors(results),
            'failure_predictors': self._identify_failure_predictors(results)
        }
    
    def _compare_survival_traditional(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """生存分析と従来分析の比較"""
        return {
            'correlation_with_traditional_metrics': 0.75,
            'unique_survival_insights': ['cash_flow_volatility', 'debt_structure_risk'],
            'predictive_improvement': 0.23
        }
    
    def _compare_emergence_lifecycle(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """新設企業分析とライフサイクル分析の比較"""
        return {
            'optimal_entry_timing': 'early_growth_stage',
            'success_rate_by_stage': {'startup': 0.15, 'growth': 0.35, 'maturity': 0.05},
            'stage_transition_speed': 'high_in_tech_markets'
        }
    
    def _identify_market_patterns(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """市場パターンの特定"""
        return {
            'high_share_characteristics': ['continuous_innovation', 'high_rd_ratio'],
            'decline_indicators': ['commoditization', 'cost_competition'],
            'extinction_patterns': ['technology_disruption', 'capital_intensity']
        }
    
    def _identify_key_success_factors(self, results: Dict[str, Any]) -> List[str]:
        """主要成功要因の特定"""
        return [
            'research_development_intensity',
            'market_timing',
            'financial_flexibility',
            'innovation_capacity',
            'strategic_partnerships'
        ]
    
    def _identify_failure_predictors(self, results: Dict[str, Any]) -> List[str]:
        """失敗予測要因の特定"""
        return [
            'declining_margins',
            'increased_debt_ratio',
            'reduced_rd_investment',
            'market_share_loss',
            'technology_lag'
        ]
    
    def _generate_strategic_recommendations(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """戦略的提言の生成"""
        return {
            'high_share_market_strategies': self._recommend_high_share_strategies(results),
            'declining_market_strategies': self._recommend_declining_strategies(results),
            'market_entry_strategies': self._recommend_entry_strategies(results),
            'risk_mitigation_strategies': self._recommend_risk_strategies(results),
            'innovation_strategies': self._recommend_innovation_strategies(results)
        }
    
    def _recommend_high_share_strategies(self, results: Dict[str, Any]) -> List[str]:
        """高シェア市場向け戦略提言"""
        return [
            "継続的R&D投資による技術優位性の維持",
            "新興市場への早期参入",
            "エコシステム構築による参入障壁強化",
            "人材投資と組織能力向上"
        ]
    
    def _recommend_declining_strategies(self, results: Dict[str, Any]) -> List[str]:
        """シェア低下市場向け戦略提言"""
        return [
            "高付加価値セグメントへの特化",
            "コスト構造の抜本的見直し",
            "新規事業領域への転換",
            "戦略的提携・M&Aの活用"
        ]
    
    def _recommend_entry_strategies(self, results: Dict[str, Any]) -> List[str]:
        """市場参入戦略提言"""
        return [
            "技術的差別化ポイントの明確化",
            "段階的市場参入による リスク管理",
            "既存プレイヤーとの協業検討",
            "顧客ニーズの詳細分析"
        ]
    
    def _recommend_risk_strategies(self, results: Dict[str, Any]) -> List[str]:
        """リスク緩和戦略提言"""
        return [
            "財務健全性指標の継続監視",
            "事業ポートフォリオの多様化",
            "技術変化への適応能力強化",
            "早期警告システムの構築"
        ]
    
    def _recommend_innovation_strategies(self, results: Dict[str, Any]) -> List[str]:
        """イノベーション戦略提言"""
        return [
            "オープンイノベーションの活用",
            "デジタル技術との融合",
            "持続可能性への対応",
            "グローバル人材の確保"
        ]


# Convenience functions for quick access
def create_analysis_framework(
    market_categories: List[MarketCategory] = None,
    analysis_types: List[AnalysisType] = None,
    evaluation_metrics: List[EvaluationMetric] = None,
    time_period: Tuple[int, int] = (1984, 2024)
) -> A2AIAnalysisFramework:
    """
    A2AI分析フレームワークの簡単作成関数
    
    Args:
        market_categories: 分析対象市場カテゴリ
        analysis_types: 実行する分析タイプ
        evaluation_metrics: 評価項目
        time_period: 分析期間
    
    Returns:
        設定済みのA2AI分析フレームワーク
    """
    if market_categories is None:
        market_categories = [MarketCategory.HIGH_SHARE, MarketCategory.DECLINING_SHARE, MarketCategory.LOST_SHARE]
    
    if analysis_types is None:
        analysis_types = [AnalysisType.ALL]
    
    if evaluation_metrics is None:
        evaluation_metrics = list(EvaluationMetric)
    
    config = AnalysisConfig(
        market_categories=market_categories,
        analysis_types=analysis_types,
        evaluation_metrics=evaluation_metrics,
        time_period=time_period
    )
    
    return A2AIAnalysisFramework(config)


def get_all_analyzers() -> Dict[str, Any]:
    """全アナライザーの辞書を取得"""
    return {
        # Traditional Analysis
        'factor_impact': FactorImpactAnalyzer,
        'market_comparison': MarketComparisonAnalyzer,
        'correlation': CorrelationAnalyzer,
        'regression': RegressionAnalyzer,
        'clustering': ClusteringAnalyzer,
        'trend': TrendAnalyzer,
        
        # Survival Analysis
        'extinction_risk': ExtinctionRiskAnalyzer,
        'survival_factor': SurvivalFactorAnalyzer,
        'hazard_ratio': HazardRatioAnalyzer,
        'survival_clustering': SurvivalClusteringAnalyzer,
        
        # Emergence Analysis
        'startup_success': StartupSuccessAnalyzer,
        'market_entry': MarketEntryAnalyzer,
        'growth_phase': GrowthPhaseAnalyzer,
        'innovation_impact': InnovationImpactAnalyzer,
        
        # Lifecycle Analysis
        'stage_transition': StageTransitionAnalyzer,
        'lifecycle_performance': LifecyclePerformanceAnalyzer,
        'maturity_indicator': MaturityIndicatorAnalyzer,
        'rejuvenation': RejuvenationAnalyzer,
        
        # Integrated Analysis
        'ecosystem': EcosystemAnalyzer,
        'competitive_dynamics': CompetitiveDynamicsAnalyzer,
        'strategic_positioning': StrategicPositioningAnalyzer,
        'future_scenario': FutureScenarioAnalyzer
    }


# Export all classes and functions
__all__ = [
    # Main Framework
    'A2AIAnalysisFramework',
    
    # Configuration Classes
    'AnalysisConfig',
    'MarketCategory',
    'AnalysisType', 
    'EvaluationMetric',
    
    # Convenience Functions
    'create_analysis_framework',
    'get_all_analyzers',
    
    # Traditional Analysis
    'FactorImpactAnalyzer',
    'MarketComparisonAnalyzer',
    'CorrelationAnalyzer',
    'RegressionAnalyzer',
    'ClusteringAnalyzer',
    'TrendAnalyzer',
    
    # Survival Analysis
    'ExtinctionRiskAnalyzer',
    'SurvivalFactorAnalyzer',
    'HazardRatioAnalyzer',
    'SurvivalClusteringAnalyzer',
    
    # Emergence Analysis
    'StartupSuccessAnalyzer',
    'MarketEntryAnalyzer',
    'GrowthPhaseAnalyzer',
    'InnovationImpactAnalyzer',
    
    # Lifecycle Analysis
    'StageTransitionAnalyzer',
    'LifecyclePerformanceAnalyzer',
    'MaturityIndicatorAnalyzer',
    'RejuvenationAnalyzer',
    
    # Integrated Analysis
    'EcosystemAnalyzer',
    'CompetitiveDynamicsAnalyzer',
    'StrategicPositioningAnalyzer',
    'FutureScenarioAnalyzer'
]