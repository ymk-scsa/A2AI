"""
A2AI (Advanced Financial Analysis AI) - Evaluation Module
======================================================

企業ライフサイクル全体を考慮した財務諸表分析AIの評価システム

主な機能:
- 従来財務分析モデルの評価
- 生存分析モデルの評価（Cox回帰、Kaplan-Meier等）
- 新設企業分析モデルの評価
- 因果推論モデルの評価
- 統合分析モデルの評価
- 生存バイアス検出・補正の評価
- 予測精度・統計的有意性の検証

対象企業: 150社（高シェア50社 + シェア低下50社 + 完全失失50社）
分析期間: 最大40年間（1984-2024年）
評価項目: 9項目（従来6項目 + 生存3項目）
要因項目: 各23項目（従来20項目 + ライフサイクル3項目）
"""

from .model_evaluator import (
    ModelEvaluator,
    TraditionalModelEvaluator,
    RegressionEvaluator,
    ClassificationEvaluator,
    TimeSeriesEvaluator
)

from .survival_model_evaluation import (
    SurvivalModelEvaluator,
    CoxRegressionEvaluator,
    KaplanMeierEvaluator,
    ParametricSurvivalEvaluator,
    ConcordanceIndexCalculator,
    TimeToEventEvaluator
)

from .causal_inference_evaluation import (
    CausalInferenceEvaluator,
    DifferenceInDifferencesEvaluator,
    InstrumentalVariableEvaluator,
    PropensityScoreEvaluator,
    CausalForestEvaluator,
    TreatmentEffectEvaluator
)

from .cross_validator import (
    CrossValidator,
    TimeSeriesCrossValidator,
    LifecycleStageCrossValidator,
    MarketCategoryCrossValidator,
    SurvivalCrossValidator,
    BlockedCrossValidator
)

from .statistical_tests import (
    StatisticalTests,
    SurvivalAnalysisTests,
    CausalInferenceTests,
    MultipleComparisonTests,
    NonParametricTests,
    RobustnessTests,
    HypothesisTestSuite
)

from .bias_detection import (
    BiasDetector,
    SurvivorshipBiasDetector,
    SelectionBiasDetector,
    TemporalBiasDetector,
    DataLeakageDetector,
    OverfittingDetector,
    EndogeneityDetector
)

from .visualization_evaluator import (
    VisualizationEvaluator,
    PerformanceVisualizer,
    SurvivalCurveValidator,
    CausalEffectVisualizer,
    ModelComparisonVisualizer,
    ResidualAnalysisVisualizer,
    LifecyclePerformanceVisualizer
)

# 評価指標の定数定義
EVALUATION_METRICS = {
    # 従来の回帰・分類評価指標
    'regression': {
        'mse': 'mean_squared_error',
        'rmse': 'root_mean_squared_error', 
        'mae': 'mean_absolute_error',
        'r2': 'r_squared',
        'adjusted_r2': 'adjusted_r_squared',
        'mape': 'mean_absolute_percentage_error',
        'explained_variance': 'explained_variance_score'
    },
    
    'classification': {
        'accuracy': 'accuracy_score',
        'precision': 'precision_score',
        'recall': 'recall_score',
        'f1': 'f1_score',
        'auc_roc': 'roc_auc_score',
        'auc_pr': 'average_precision_score',
        'log_loss': 'log_loss'
    },
    
    # 生存分析特有の評価指標
    'survival': {
        'c_index': 'concordance_index',
        'harrels_c': 'harrels_concordance_index',
        'time_dependent_auc': 'time_dependent_roc_auc',
        'integrated_brier_score': 'integrated_brier_score',
        'log_likelihood': 'log_likelihood',
        'aic': 'akaike_information_criterion',
        'bic': 'bayesian_information_criterion'
    },
    
    # 因果推論評価指標
    'causal': {
        'ate': 'average_treatment_effect',
        'att': 'average_treatment_effect_on_treated',
        'cate': 'conditional_average_treatment_effect',
        'policy_risk': 'policy_risk',
        'tau_risk': 'tau_risk',
        'influence_function': 'influence_function_based_ci'
    },
    
    # 時系列分析評価指標
    'time_series': {
        'directional_accuracy': 'directional_accuracy',
        'theil_u': 'theil_u_statistic',
        'tracking_error': 'tracking_error',
        'information_ratio': 'information_ratio'
    }
}

# バイアス検出閾値
BIAS_DETECTION_THRESHOLDS = {
    'survivorship_bias': {
        'extinction_rate_threshold': 0.05,  # 5%以上の企業消滅で警告
        'missing_data_threshold': 0.10,    # 10%以上の欠損で警告
        'lifecycle_imbalance_threshold': 0.30  # ライフサイクル段階の30%以上偏りで警告
    },
    
    'selection_bias': {
        'market_representation_threshold': 0.20,  # 市場代表性20%以下で警告
        'size_bias_threshold': 0.15,             # 企業規模バイアス15%以上で警告
        'temporal_bias_threshold': 0.25          # 時系列バイアス25%以上で警告
    },
    
    'data_leakage': {
        'future_information_threshold': 0.01,    # 1%以上の未来情報混入で警告
        'target_leakage_threshold': 0.05,       # 5%以上のターゲットリークで警告
        'temporal_leakage_threshold': 0.03      # 3%以上の時系列リークで警告
    }
}

# 統計的検定の有意水準設定
STATISTICAL_SIGNIFICANCE_LEVELS = {
    'alpha': 0.05,           # 通常の有意水準
    'bonferroni_alpha': 0.001,  # 多重比較補正後
    'fdr_alpha': 0.1,        # False Discovery Rate制御
    'power': 0.8,            # 検出力
    'effect_size_threshold': {
        'small': 0.1,
        'medium': 0.3, 
        'large': 0.5
    }
}

# 企業ライフサイクル段階定義（評価時に使用）
LIFECYCLE_STAGES = {
    'startup': {
        'age_range': (0, 5),      # 設立0-5年
        'growth_threshold': 0.20,  # 年間成長率20%以上
        'volatility_threshold': 0.50  # 変動性50%以上
    },
    
    'growth': {
        'age_range': (6, 15),     # 設立6-15年  
        'growth_threshold': 0.10,  # 年間成長率10%以上
        'volatility_threshold': 0.30  # 変動性30%以上
    },
    
    'maturity': {
        'age_range': (16, 30),    # 設立16-30年
        'growth_threshold': 0.05,  # 年間成長率5%以上
        'volatility_threshold': 0.20  # 変動性20%以下
    },
    
    'decline_or_renewal': {
        'age_range': (31, float('inf')),  # 設立31年以上
        'growth_threshold': 0.00,         # 成長率基準なし
        'volatility_threshold': float('inf')  # 変動性基準なし
    }
}

# 市場カテゴリ評価基準
MARKET_CATEGORY_CRITERIA = {
    'high_share': {
        'min_companies': 45,      # 最低45社のデータが必要
        'min_years': 30,          # 最低30年のデータが必要
        'survival_rate': 0.90     # 90%以上の企業生存率期待
    },
    
    'declining_share': {
        'min_companies': 40,      # 最低40社のデータが必要
        'min_years': 35,          # 最低35年のデータが必要
        'survival_rate': 0.80     # 80%以上の企業生存率期待
    },
    
    'lost_share': {
        'min_companies': 25,      # 最低25社のデータが必要（多数が消滅予想）
        'min_years': 25,          # 最低25年のデータが必要
        'survival_rate': 0.50     # 50%以上の企業生存率期待
    }
}

# モデル性能比較基準
MODEL_PERFORMANCE_BENCHMARKS = {
    'traditional_financial_analysis': {
        'regression_r2_threshold': 0.60,      # R²値60%以上で良好
        'classification_auc_threshold': 0.75,  # AUC75%以上で良好
        'prediction_accuracy_threshold': 0.70  # 予測精度70%以上で良好
    },
    
    'survival_analysis': {
        'c_index_threshold': 0.65,            # C-index 65%以上で良好
        'time_auc_threshold': 0.70,           # Time-dependent AUC 70%以上
        'log_likelihood_improvement': 0.10    # 対数尤度10%以上改善
    },
    
    'causal_inference': {
        'treatment_effect_precision': 0.20,    # 処置効果推定精度20%以内
        'confidence_interval_coverage': 0.95,  # 信頼区間カバー率95%以上
        'robustness_test_pass_rate': 0.80     # 頑健性テスト80%以上通過
    }
}

# 評価レポート生成設定
EVALUATION_REPORT_CONFIG = {
    'output_formats': ['html', 'pdf', 'json'],
    'include_sections': [
        'executive_summary',
        'model_performance_overview',
        'statistical_significance_tests',
        'bias_detection_results',
        'cross_validation_results',
        'causal_inference_validation',
        'survival_analysis_validation',
        'robustness_analysis',
        'recommendations'
    ],
    
    'visualization_types': [
        'performance_comparison_charts',
        'survival_curves',
        'causal_effect_plots',
        'residual_analysis_plots',
        'bias_detection_heatmaps',
        'lifecycle_performance_trajectories'
    ],
    
    'detail_levels': {
        'executive': 'high_level_summary',
        'technical': 'detailed_analysis',
        'academic': 'full_statistical_details'
    }
}

def create_comprehensive_evaluator(
    models_dict: dict,
    data_config: dict,
    evaluation_config: dict = None
):
    """
    包括的な評価システムを作成
    
    Args:
        models_dict: 評価対象モデル辞書
        data_config: データ設定辞書
        evaluation_config: 評価設定辞書
    
    Returns:
        ComprehensiveEvaluator: 統合評価システム
    """
    from .comprehensive_evaluator import ComprehensiveEvaluator
    
    if evaluation_config is None:
        evaluation_config = {
            'metrics': EVALUATION_METRICS,
            'bias_thresholds': BIAS_DETECTION_THRESHOLDS,
            'significance_levels': STATISTICAL_SIGNIFICANCE_LEVELS,
            'performance_benchmarks': MODEL_PERFORMANCE_BENCHMARKS
        }
    
    return ComprehensiveEvaluator(
        models=models_dict,
        data_config=data_config,
        evaluation_config=evaluation_config
    )

def run_full_evaluation_suite(
    models_dict: dict,
    train_data: dict,
    test_data: dict,
    validation_data: dict = None,
    output_dir: str = "evaluation_results"
):
    """
    完全な評価スイートを実行
    
    Args:
        models_dict: 評価対象モデル辞書
        train_data: 訓練データ
        test_data: テストデータ  
        validation_data: 検証データ（オプション）
        output_dir: 結果出力ディレクトリ
    
    Returns:
        dict: 評価結果の包括的な辞書
    """
    from .evaluation_pipeline import run_evaluation_pipeline
    
    return run_evaluation_pipeline(
        models=models_dict,
        train_data=train_data,
        test_data=test_data,
        validation_data=validation_data,
        output_directory=output_dir,
        config=EVALUATION_REPORT_CONFIG
    )

# パッケージの公開API
__all__ = [
    # 主要評価クラス
    'ModelEvaluator',
    'SurvivalModelEvaluator', 
    'CausalInferenceEvaluator',
    'CrossValidator',
    'StatisticalTests',
    'BiasDetector',
    'VisualizationEvaluator',
    
    # 特殊化された評価クラス
    'TraditionalModelEvaluator',
    'CoxRegressionEvaluator',
    'KaplanMeierEvaluator',
    'DifferenceInDifferencesEvaluator',
    'PropensityScoreEvaluator',
    'TimeSeriesCrossValidator',
    'SurvivalCrossValidator',
    
    # バイアス検出クラス
    'SurvivorshipBiasDetector',
    'SelectionBiasDetector',
    'DataLeakageDetector',
    
    # 統計検定クラス
    'SurvivalAnalysisTests',
    'CausalInferenceTests',
    'MultipleComparisonTests',
    
    # 可視化クラス
    'PerformanceVisualizer',
    'SurvivalCurveValidator',
    'CausalEffectVisualizer',
    'LifecyclePerformanceVisualizer',
    
    # 設定・定数
    'EVALUATION_METRICS',
    'BIAS_DETECTION_THRESHOLDS',
    'STATISTICAL_SIGNIFICANCE_LEVELS',
    'LIFECYCLE_STAGES',
    'MARKET_CATEGORY_CRITERIA',
    'MODEL_PERFORMANCE_BENCHMARKS',
    'EVALUATION_REPORT_CONFIG',
    
    # ヘルパー関数
    'create_comprehensive_evaluator',
    'run_full_evaluation_suite'
]

# バージョン情報
__version__ = "1.0.0"
__author__ = "A2AI Development Team"
__description__ = "Advanced Financial Analysis AI - Comprehensive Model Evaluation System"