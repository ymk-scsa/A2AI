# =============================================================================
# A2AI (Advanced Financial Analysis AI) - src/__init__.py
# 財務諸表分析AI：企業ライフサイクル全体を対象とした包括的分析システム
# =============================================================================

"""
A2AI: Advanced Financial Analysis AI

企業の生存・消滅・新設を含む完全なライフサイクル分析を可能にする
革新的な財務諸表分析システム

主要機能:
- 150社×変動期間の財務データ分析
- 9つの評価項目 × 各23要因項目による多角的分析  
- 生存分析による企業消滅リスク予測
- 新設企業の成功要因分析
- 因果推論による真の影響要因特定
- 生存バイアス完全対応
"""

__version__ = "1.0.0"
__author__ = "A2AI Development Team"
__email__ = "a2ai@example.com"
__description__ = "Advanced Financial Analysis AI for Corporate Lifecycle Analysis"

# Core modules imports
from . import data_collection
from . import preprocessing  
from . import feature_engineering
from . import models
from . import analysis
from . import evaluation
from . import visualization
from . import simulation
from . import prediction
from . import utils

# Key constants
EVALUATION_METRICS = [
    "売上高", "売上高成長率", "売上高営業利益率", 
    "売上高当期純利益率", "ROE", "売上高付加価値率",
    "企業存続確率", "新規事業成功率", "事業継承成功度"
]

MARKET_CATEGORIES = [
    "high_share_markets",    # 現在もシェアが高い市場
    "declining_markets",     # シェア低下中の市場  
    "lost_markets"           # 完全にシェアを失った市場
]

COMPANY_LIFECYCLE_STAGES = [
    "startup",      # 新設・創業期
    "growth",       # 成長期
    "maturity",     # 成熟期
    "decline",      # 衰退期
    "extinction"    # 消滅・撤退
]

TARGET_COMPANIES_COUNT = {
    "high_share_markets": 50,
    "declining_markets": 50, 
    "lost_markets": 50,
    "total": 150
}

# Data collection settings
DATA_COLLECTION_CONFIG = {
    "edinet_api_base_url": "https://disclosure.edinet-fsa.go.jp/api/v2/",
    "max_years_back": 40,  # 1984-2024
    "min_data_quality_threshold": 0.8,
    "include_extinct_companies": True,
    "include_spinoff_companies": True
}

# Analysis settings
ANALYSIS_CONFIG = {
    "factor_count_per_metric": 23,  # 各評価項目に対する要因項目数
    "survival_analysis_enabled": True,
    "causal_inference_enabled": True,
    "survivorship_bias_correction": True,
    "temporal_alignment": True
}

# Model settings
MODEL_CONFIG = {
    "cross_validation_folds": 5,
    "test_size_ratio": 0.2,
    "random_state": 42,
    "enable_ensemble": True,
    "enable_deep_learning": True,
    "enable_survival_models": True
}

# Logging configuration
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('a2ai.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
logger.info("A2AI (Advanced Financial Analysis AI) initialized successfully")

# =============================================================================
# src/data_collection/__init__.py
# =============================================================================

"""
Data Collection Module for A2AI

企業ライフサイクル全体のデータ収集を担当
- EDINET APIを使用した財務諸表データ収集
- 企業消滅・新設イベントの追跡
- 市場シェアデータの収集
- 生存バイアス対応データセット構築
"""

from .financial_scraper import FinancialDataScraper
from .market_share_collector import MarketShareCollector
from .industry_data_collector import IndustryDataCollector
from .lifecycle_data_collector import LifecycleDataCollector
from .extinction_event_tracker import ExtinctionEventTracker
from .spinoff_data_integrator import SpinoffDataIntegrator
from .emergence_data_tracker import EmergenceDataTracker
from .survival_data_generator import SurvivalDataGenerator
from .data_validator import DataValidator

__all__ = [
    'FinancialDataScraper',
    'MarketShareCollector', 
    'IndustryDataCollector',
    'LifecycleDataCollector',
    'ExtinctionEventTracker',
    'SpinoffDataIntegrator',
    'EmergenceDataTracker',
    'SurvivalDataGenerator',
    'DataValidator'
]

# Target companies by market category
HIGH_SHARE_COMPANIES = [
    # ロボット市場
    "ファナック", "安川電機", "川崎重工業", "不二越", "デンソーウェーブ",
    "三菱電機", "オムロン", "THK", "NSK", "IHI",
    # 内視鏡市場
    "オリンパス", "HOYA", "富士フイルム", "キヤノンメディカルシステムズ", "島津製作所",
    "コニカミノルタ", "ソニー", "トプコン", "エムスリー", "日立製作所",
    # 工作機械市場
    "DMG森精機", "ヤマザキマザック", "オークマ", "牧野フライス製作所", "ジェイテクト",
    "東芝機械", "アマダ", "ソディック", "三菱重工工作機械", "シギヤ精機製作所",
    # 電子材料市場
    "村田製作所", "TDK", "京セラ", "太陽誘電", "日本特殊陶業",
    "ローム", "プロテリアル", "住友電工", "日東電工", "日本碍子",
    # 精密測定機器市場
    "キーエンス", "島津製作所", "堀場製作所", "東京精密", "ミツトヨ",
    "オリンパス", "日本電産", "リオン", "アルバック", "ナブテスコ"
]

DECLINING_COMPANIES = [
    # 自動車市場
    "トヨタ自動車", "日産自動車", "ホンダ", "スズキ", "マツダ",
    "SUBARU", "いすゞ自動車", "三菱自動車", "ダイハツ工業", "日野自動車",
    # 鉄鋼市場
    "日本製鉄", "JFEホールディングス", "神戸製鋼所", "日新製鋼", "大同特殊鋼",
    "山陽特殊製鋼", "愛知製鋼", "中部鋼鈑", "淀川製鋼所", "日立金属",
    # スマート家電市場
    "パナソニック", "シャープ", "ソニー", "東芝ライフスタイル", "日立グローバルライフソリューションズ",
    "アイリスオーヤマ", "三菱電機", "象印マホービン", "タイガー魔法瓶", "山善",
    # バッテリー市場
    "パナソニックエナジー", "村田製作所", "GSユアサ", "東芝インフラシステムズ", "日立化成",
    "FDK", "NEC", "ENAX", "日本電産", "TDK",
    # PC・周辺機器市場
    "NEC", "富士通クライアントコンピューティング", "東芝", "ソニー", "エレコム",
    "バッファロー", "ロジテック", "プリンストン", "サンワサプライ", "アイ・オー・データ機器"
]

LOST_COMPANIES = [
    # 家電市場（完全撤退）
    "ソニー", "パナソニック", "シャープ", "東芝ライフスタイル", "三菱電機",
    "日立グローバルライフソリューションズ", "三洋電機", "ビクター", "アイワ", "船井電機",
    # 半導体市場（完全撤退）
    "東芝", "日立製作所", "三菱電機", "NEC", "富士通",
    "松下電器", "ソニー", "ルネサスエレクトロニクス", "シャープ", "ローム",
    # スマートフォン市場
    "ソニー", "シャープ", "京セラ", "パナソニック", "富士通",
    "NEC", "日立製作所", "三菱電機", "東芝", "カシオ計算機",
    # PC市場
    "ソニー", "NEC", "富士通", "東芝", "シャープ",
    "パナソニック", "日立製作所", "三菱電機", "カシオ計算機", "日本電気ホームエレクトロニクス",
    # 通信機器市場
    "NEC", "富士通", "日立製作所", "松下電器", "シャープ",
    "ソニー", "三菱電機", "京セラ", "カシオ計算機", "日本無線"
]

# Data collection configuration
EDINET_CONFIG = {
    "api_version": "v2",
    "timeout_seconds": 30,
    "retry_count": 3,
    "rate_limit_delay": 1.0
}

FINANCIAL_DATA_ITEMS = [
    # 貸借対照表項目
    "有形固定資産", "無形固定資産", "投資有価証券", "売上債権", "棚卸資産", "総資産",
    # 損益計算書項目  
    "売上高", "売上原価", "販売費及び一般管理費", "営業利益", "当期純利益",
    # キャッシュフロー項目
    "設備投資額", "営業キャッシュフロー", "投資キャッシュフロー",
    # 注記・その他
    "従業員数", "平均年間給与", "研究開発費", "海外売上高比率"
]

# =============================================================================
# src/preprocessing/__init__.py
# =============================================================================

"""
Preprocessing Module for A2AI

生存バイアス対応とライフサイクル整合を重視した前処理
- 企業消滅・新設を考慮したデータクリーニング
- ライフサイクル段階別正規化
- 時系列データの整合性確保
- 欠損値・外れ値の統計的処理
"""

from .data_cleaner import DataCleaner
from .feature_extractor import FeatureExtractor
from .factor_calculator import FactorCalculator
from .missing_value_handler import MissingValueHandler
from .outlier_detector import OutlierDetector
from .lifecycle_normalizer import LifecycleNormalizer
from .survivorship_bias_corrector import SurvivorshipBiasCorrector
from .extinction_feature_engineer import ExtinctionFeatureEngineer
from .emergence_success_analyzer import EmergenceSuccessAnalyzer
from .temporal_alignment_handler import TemporalAlignmentHandler

__all__ = [
    'DataCleaner',
    'FeatureExtractor', 
    'FactorCalculator',
    'MissingValueHandler',
    'OutlierDetector',
    'LifecycleNormalizer',
    'SurvivorshipBiasCorrector',
    'ExtinctionFeatureEngineer',
    'EmergenceSuccessAnalyzer',
    'TemporalAlignmentHandler'
]

# 前処理設定
PREPROCESSING_CONFIG = {
    "missing_value_threshold": 0.3,  # 30%以上欠損の変数は除外
    "outlier_detection_method": "iqr",  # IQR法による外れ値検出
    "outlier_threshold": 1.5,
    "normalize_method": "robust_scaler",  # ロバストスケーリング
    "lifecycle_alignment": True,
    "bias_correction_enabled": True
}

# 特徴量抽出対象の財務指標
FINANCIAL_RATIOS = [
    # 収益性指標
    "売上高営業利益率", "売上高当期純利益率", "ROE", "ROA",
    # 効率性指標
    "総資産回転率", "売上債権回転率", "棚卸資産回転率",
    # 安全性指標
    "自己資本比率", "流動比率", "固定比率",
    # 成長性指標
    "売上高成長率", "営業利益成長率", "総資産成長率"
]

# ライフサイクル段階判定基準
LIFECYCLE_CRITERIA = {
    "startup": {"age_max": 5, "growth_min": 0.2},
    "growth": {"age_range": (5, 15), "growth_min": 0.1},
    "maturity": {"age_range": (15, 30), "growth_range": (-0.05, 0.1)},
    "decline": {"age_min": 30, "growth_max": -0.05},
    "extinction": {"status": "delisted_or_bankrupt"}
}

# =============================================================================
# src/feature_engineering/__init__.py
# =============================================================================

"""
Feature Engineering Module for A2AI

9つの評価項目と拡張要因項目の計算・生成
- 従来6項目 + 新規3項目の評価指標
- 各評価項目に対する23要因項目
- 時系列特徴量とライフサイクル特徴量
- 生存分析・新設企業分析用特徴量
"""

from .evaluation_metrics import EvaluationMetrics
from .evaluation_metrics.traditional_metrics import TraditionalMetrics
from .evaluation_metrics.survival_metrics import SurvivalMetrics
from .evaluation_metrics.emergence_metrics import EmergenceMetrics
from .evaluation_metrics.succession_metrics import SuccessionMetrics
from .factor_metrics import FactorMetrics
from .time_series_features import TimeSeriesFeatures
from .interaction_features import InteractionFeatures
from .market_features import MarketFeatures
from .lifecycle_features import LifecycleFeatures
from .survival_features import SurvivalFeatures
from .emergence_features import EmergenceFeatures

__all__ = [
    'EvaluationMetrics',
    'TraditionalMetrics',
    'SurvivalMetrics', 
    'EmergenceMetrics',
    'SuccessionMetrics',
    'FactorMetrics',
    'TimeSeriesFeatures',
    'InteractionFeatures',
    'MarketFeatures',
    'LifecycleFeatures',
    'SurvivalFeatures',
    'EmergenceFeatures'
]

# 9つの評価項目定義
EVALUATION_METRICS_DEFINITION = {
    # 従来の6項目
    "売上高": {
        "description": "企業の事業規模を示す基本指標",
        "unit": "百万円",
        "calculation": "年間売上高"
    },
    "売上高成長率": {
        "description": "企業の成長性を示す指標", 
        "unit": "%",
        "calculation": "(当期売上高 - 前期売上高) / 前期売上高"
    },
    "売上高営業利益率": {
        "description": "本業での収益性を示す指標",
        "unit": "%", 
        "calculation": "営業利益 / 売上高"
    },
    "売上高当期純利益率": {
        "description": "最終的な収益性を示す指標",
        "unit": "%",
        "calculation": "当期純利益 / 売上高"
    },
    "ROE": {
        "description": "株主資本収益率",
        "unit": "%",
        "calculation": "当期純利益 / 自己資本"
    },
    "売上高付加価値率": {
        "description": "企業の付加価値創造能力",
        "unit": "%", 
        "calculation": "付加価値 / 売上高"
    },
    # 新規3項目
    "企業存続確率": {
        "description": "企業が将来も存続する確率",
        "unit": "確率(0-1)",
        "calculation": "生存分析モデルによる予測"
    },
    "新規事業成功率": {
        "description": "新規事業・新設企業の成功確率", 
        "unit": "確率(0-1)",
        "calculation": "成功予測モデルによる算出"
    },
    "事業継承成功度": {
        "description": "M&A・分社化の成功度",
        "unit": "スコア(0-100)",
        "calculation": "事業継承効果分析による評価"
    }
}

# 各評価項目に対する23要因項目のカテゴリ
FACTOR_CATEGORIES = [
    "投資・資産関連",      # 5項目
    "人的資源関連",        # 4項目  
    "運転資本・効率性関連", # 5項目
    "事業展開関連",        # 6項目
    "ライフサイクル関連"   # 3項目（新規追加）
]

# 特徴量エンジニアリング設定
FEATURE_CONFIG = {
    "enable_lag_features": True,
    "lag_periods": [1, 2, 3, 5],  # 1年、2年、3年、5年ラグ
    "enable_moving_averages": True,
    "ma_windows": [3, 5, 7],  # 3年、5年、7年移動平均
    "enable_trend_features": True,
    "enable_interaction_terms": True,
    "max_interaction_degree": 2
}

# =============================================================================
# src/models/__init__.py
# =============================================================================

"""
Models Module for A2AI

多様な分析手法による統合モデリング
- 従来の財務分析モデル
- 生存分析モデル群
- 新設企業分析モデル
- 因果推論モデル
- 統合分析モデル
"""

from .base_model import BaseModel
from .traditional_models import *
from .survival_models import *
from .emergence_models import *
from .causal_inference import *
from .integrated_models import *

__all__ = [
    'BaseModel',
    # Traditional models
    'RegressionModels',
    'EnsembleModels', 
    'DeepLearningModels',
    'TimeSeriesModels',
    # Survival models
    'CoxRegression',
    'KaplanMeier',
    'ParametricSurvival',
    'MachineLearninSurvival',
    # Emergence models
    'SuccessPrediction',
    'GrowthTrajectory',
    'MarketEntryTiming',
    # Causal inference
    'DifferenceInDifferences',
    'InstrumentalVariables',
    'PropensityScore',
    'CausalForest',
    # Integrated models
    'MultiStageAnalysis',
    'LifecycleTrajectory',
    'MarketEcosystem'
]

# モデル設定
MODEL_CONFIGS = {
    "traditional": {
        "test_size": 0.2,
        "cv_folds": 5,
        "random_state": 42
    },
    "survival": {
        "tie_method": "breslow",
        "alpha": 0.05,
        "bootstrap_samples": 1000
    },
    "emergence": {
        "success_threshold": 0.1,  # 10%成長を成功と定義
        "evaluation_period": 5    # 設立後5年での評価
    },
    "causal": {
        "bandwidth": "optimal",
        "kernel": "gaussian",
        "bootstrap_iterations": 500
    }
}

# =============================================================================
# src/analysis/__init__.py
# =============================================================================

"""
Analysis Module for A2AI

包括的な分析機能群
- 従来型財務分析
- 生存分析群  
- 新設企業分析群
- ライフサイクル分析群
- 統合分析群
"""

from .traditional_analysis import *
from .survival_analysis import *
from .emergence_analysis import *
from .lifecycle_analysis import *
from .integrated_analysis import *

__all__ = [
    # Traditional analysis
    'FactorImpactAnalyzer',
    'MarketComparison',
    'CorrelationAnalyzer',
    'RegressionAnalyzer',
    'ClusteringAnalyzer',
    'TrendAnalyzer',
    # Survival analysis
    'ExtinctionRiskAnalyzer',
    'SurvivalFactorAnalyzer', 
    'HazardRatioAnalyzer',
    'SurvivalClustering',
    # Emergence analysis
    'StartupSuccessAnalyzer',
    'MarketEntryAnalyzer',
    'GrowthPhaseAnalyzer',
    'InnovationImpactAnalyzer',
    # Lifecycle analysis
    'StageTransitionAnalyzer',
    'LifecyclePerformance',
    'MaturityIndicator',
    'RejuvenationAnalyzer',
    # Integrated analysis
    'EcosystemAnalyzer',
    'CompetitiveDynamics',
    'StrategicPositioning',
    'FutureScenarioAnalyzer'
]

# 分析設定
ANALYSIS_CONFIGS = {
    "correlation_threshold": 0.7,
    "significance_level": 0.05,
    "cluster_count_range": (2, 10),
    "trend_smoothing_window": 5,
    "survival_confidence_level": 0.95,
    "emergence_evaluation_years": 5
}

# =============================================================================
# src/evaluation/__init__.py
# =============================================================================

"""
Evaluation Module for A2AI

モデル評価と結果検証
- 統計的仮説検定
- 生存モデル評価
- 因果推論モデル評価  
- バイアス検出・評価
- 可視化による評価
"""

from .model_evaluator import ModelEvaluator
from .cross_validator import CrossValidator
from .statistical_tests import StatisticalTests
from .survival_model_evaluation import SurvivalModelEvaluation
from .causal_inference_evaluation import CausalInferenceEvaluation
from .bias_detection import BiasDetection
from .visualization_evaluator import VisualizationEvaluator

__all__ = [
    'ModelEvaluator',
    'CrossValidator',
    'StatisticalTests',
    'SurvivalModelEvaluation',
    'CausalInferenceEvaluation', 
    'BiasDetection',
    'VisualizationEvaluator'
]

# 評価指標設定
EVALUATION_METRICS = {
    "regression": ["mse", "mae", "r2", "rmse"],
    "classification": ["accuracy", "precision", "recall", "f1", "auc"],
    "survival": ["c_index", "log_likelihood", "brier_score"],
    "causal": ["ate", "att", "bias", "coverage"]
}

# =============================================================================
# src/visualization/__init__.py
# =============================================================================

"""
Visualization Module for A2AI

多角的可視化システム
- 従来型財務分析可視化
- 生存分析可視化
- ライフサイクル可視化
- 新設企業可視化
- 統合可視化
"""

from .traditional_viz import *
from .survival_viz import *
from .lifecycle_viz import *
from .emergence_viz import *
from .integrated_viz import *

__all__ = [
    # Traditional visualization
    'FactorVisualizer',
    'MarketVisualizer',
    'TimeSeriesPlots',
    'CorrelationHeatmaps',
    'PerformanceDashboards',
    # Survival visualization
    'SurvivalCurves',
    'HazardPlots',
    'RiskHeatmaps',
    'ExtinctionTimeline',
    # Lifecycle visualization
    'LifecycleTrajectories',
    'StageTransitions',
    'MaturityLandscape',
    'EvolutionAnimation',
    # Emergence visualization
    'StartupJourney',
    'SuccessFactors',
    'MarketEntryTiming',
    'InnovationDiffusion',
    # Integrated visualization
    'EcosystemNetworks',
    'CompetitiveLandscape',
    'StrategicPositioning',
    'InteractiveExplorer'
]

# 可視化設定
VIZ_CONFIG = {
    "default_figsize": (12, 8),
    "color_palette": "viridis",
    "dpi": 300,
    "save_format": "png",
    "interactive_backend": "plotly"
}

# =============================================================================
# src/simulation/__init__.py
# =============================================================================

"""
Simulation Module for A2AI

シミュレーション機能群
- モンテカルロシミュレーション
- シナリオ分析
- 市場ダイナミクス
- 企業ライフサイクル
- 政策影響評価
"""

from .monte_carlo import MonteCarlo
from .scenario_generator import ScenarioGenerator
from .market_dynamics import MarketDynamics
from .corporate_lifecycle import CorporateLifecycle
from .policy_impact_simulator import PolicyImpactSimulator

__all__ = [
    'MonteCarlo',
    'ScenarioGenerator',
    'MarketDynamics',
    'CorporateLifecycle',
    'PolicyImpactSimulator'
]

# =============================================================================
# src/prediction/__init__.py  
# =============================================================================

"""
Prediction Module for A2AI

予測機能群
- 生存確率予測
- 財務性能予測
- 市場シェア予測
- 新設企業成功予測
- シナリオベース予測
"""

from .survival_predictor import SurvivalPredictor
from .performance_predictor import PerformancePredictor
from .market_share_predictor import MarketSharePredictor
from .emergence_success_predictor import EmergenceSuccessPredictor
from .scenario_forecaster import ScenarioForecaster

__all__ = [
    'SurvivalPredictor',
    'PerformancePredictor', 
    'MarketSharePredictor',
    'EmergenceSuccessPredictor',
    'ScenarioForecaster'
]

# =============================================================================
# src/utils/__init__.py
# =============================================================================

"""
Utilities Module for A2AI

共通ユーティリティ機能群
- データ操作
- 数学的計算
- 生存分析ユーティリティ
- 因果推論ユーティリティ
- ライフサイクル分析ユーティリティ
- ログ管理
- データベース操作
- 統計分析
"""

from .data_utils import DataUtils
from .math_utils import MathUtils
from .survival_utils import SurvivalUtils
from .causal_utils import CausalUtils
from .lifecycle_utils import LifecycleUtils
from .file_utils import FileUtils
from .logging_utils import LoggingUtils
from .database_utils import DatabaseUtils
from .statistical_utils import StatisticalUtils

__all__ = [
    'DataUtils',
    'MathUtils',
    'SurvivalUtils',
    'CausalUtils',
    'LifecycleUtils',
    'FileUtils',
    'LoggingUtils',
    'DatabaseUtils',
    'StatisticalUtils'
]

# 共通設定
UTILS_CONFIG = {
    "default_encoding": "utf-8",
    "decimal_precision": 6,
    "date_format": "%Y-%m-%d",
    "currency_unit": "百万円"
}

logger.info("All A2AI modules initialized successfully")