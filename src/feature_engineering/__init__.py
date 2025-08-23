"""
A2AI Feature Engineering Module

Advanced Financial Analysis AI (A2AI) の特徴量エンジニアリング機能を統合管理するモジュール。
企業ライフサイクル全体（設立-成長-成熟-衰退-消滅/再生）を考慮した特徴量生成を実現。

主要機能:
1. 従来6評価項目 + 新規3評価項目（計9項目）の計算
2. 各評価項目に対応する23拡張要因項目の生成
3. 企業ライフサイクル段階別特徴量の構築
4. 生存分析用特徴量の作成
5. 市場カテゴリ別（高シェア/低下/失失）特徴量の生成

評価項目:
- 従来6項目: 売上高、売上高成長率、売上高営業利益率、売上高当期純利益率、ROE、売上高付加価値率
- 新規3項目: 企業存続確率、新規事業成功率、事業継承成功度

要因項目: 各評価項目に対して23項目（20基本項目 + 3拡張項目）
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum

# 内部モジュールのインポート
from .evaluation_metrics import (
    TraditionalMetricsCalculator,
    SurvivalMetricsCalculator, 
    EmergenceMetricsCalculator,
    SuccessionMetricsCalculator
)
from .factor_metrics import FactorMetricsCalculator
from .time_series_features import TimeSeriesFeatureGenerator
from .interaction_features import InteractionFeatureGenerator
from .market_features import MarketFeatureGenerator
from .lifecycle_features import LifecycleFeatureGenerator
from .survival_features import SurvivalFeatureGenerator
from .emergence_features import EmergenceFeatureGenerator

# ユーティリティのインポート
from ..utils.lifecycle_utils import LifecycleStageClassifier
from ..utils.statistical_utils import StatisticalValidator


class MarketCategory(Enum):
    """市場カテゴリ定義"""
    HIGH_SHARE = "high_share"      # 高シェア市場
    DECLINING = "declining"        # シェア低下市場  
    LOST = "lost"                 # 完全失失市場


class LifecycleStage(Enum):
    """企業ライフサイクル段階定義"""
    STARTUP = "startup"           # 新設期（0-5年）
    GROWTH = "growth"            # 成長期（6-15年）
    MATURITY = "maturity"        # 成熟期（16-30年）
    DECLINE = "decline"          # 衰退期（31年-）
    EXTINCTION = "extinction"    # 消滅期
    SUCCESSION = "succession"    # 事業継承期（分社・統合）


@dataclass
class FeatureEngineringConfig:
    """特徴量エンジニアリング設定"""
    include_traditional_metrics: bool = True
    include_survival_metrics: bool = True
    include_emergence_metrics: bool = True
    include_succession_metrics: bool = True
    include_time_series_features: bool = True
    include_interaction_features: bool = True
    include_market_features: bool = True
    include_lifecycle_features: bool = True
    
    # 時系列関連設定
    time_window_years: int = 5
    min_data_points: int = 3
    
    # 生存分析関連設定
    survival_event_column: str = 'extinction_event'
    survival_time_column: str = 'survival_time'
    
    # 市場カテゴリ関連設定
    market_category_column: str = 'market_category'
    
    # ライフサイクル関連設定
    company_age_column: str = 'company_age'
    founding_date_column: str = 'founding_date'


class FeatureEngineering:
    """
    A2AI特徴量エンジニアリング統合クラス
    
    企業の生存・消滅・新設を含む完全なライフサイクルを考慮した
    財務諸表分析用特徴量を生成する。
    """
    
    def __init__(self, config: Optional[FeatureEngineringConfig] = None):
        """
        初期化
        
        Args:
            config: 特徴量エンジニアリング設定
        """
        self.config = config or FeatureEngineringConfig()
        self.logger = logging.getLogger(__name__)
        
        # 各計算機の初期化
        self._initialize_calculators()
        
        # バリデータの初期化
        self.validator = StatisticalValidator()
        self.lifecycle_classifier = LifecycleStageClassifier()
        
        self.logger.info("A2AI Feature Engineering initialized")
    
    def _initialize_calculators(self):
        """各種計算機の初期化"""
        # 評価項目計算機
        if self.config.include_traditional_metrics:
            self.traditional_calculator = TraditionalMetricsCalculator()
        if self.config.include_survival_metrics:
            self.survival_calculator = SurvivalMetricsCalculator()
        if self.config.include_emergence_metrics:
            self.emergence_calculator = EmergenceMetricsCalculator()
        if self.config.include_succession_metrics:
            self.succession_calculator = SuccessionMetricsCalculator()
        
        # 要因項目計算機
        self.factor_calculator = FactorMetricsCalculator()
        
        # 特徴量生成器
        if self.config.include_time_series_features:
            self.time_series_generator = TimeSeriesFeatureGenerator(
                window_years=self.config.time_window_years
            )
        if self.config.include_interaction_features:
            self.interaction_generator = InteractionFeatureGenerator()
        if self.config.include_market_features:
            self.market_generator = MarketFeatureGenerator()
        if self.config.include_lifecycle_features:
            self.lifecycle_generator = LifecycleFeatureGenerator()
            
        # 生存・新設分析用特徴量生成器
        self.survival_feature_generator = SurvivalFeatureGenerator()
        self.emergence_feature_generator = EmergenceFeatureGenerator()
    
    def generate_all_features(
        self, 
        financial_data: pd.DataFrame,
        company_metadata: Optional[pd.DataFrame] = None,
        market_data: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        全特徴量の統合生成
        
        Args:
            financial_data: 財務諸表データ
            company_metadata: 企業メタデータ（設立年、業界等）
            market_data: 市場データ（シェア情報等）
            
        Returns:
            pd.DataFrame: 全特徴量を含むデータフレーム
        """
        self.logger.info("Starting comprehensive feature generation")
        
        # データ検証
        self._validate_input_data(financial_data)
        
        features = financial_data.copy()
        
        # 1. 基本評価項目の計算
        evaluation_metrics = self.generate_evaluation_metrics(financial_data)
        features = pd.concat([features, evaluation_metrics], axis=1)
        
        # 2. 要因項目の計算
        factor_metrics = self.generate_factor_metrics(financial_data, evaluation_metrics)
        features = pd.concat([features, factor_metrics], axis=1)
        
        # 3. ライフサイクル特徴量
        if self.config.include_lifecycle_features and company_metadata is not None:
            lifecycle_features = self.generate_lifecycle_features(
                features, company_metadata
            )
            features = pd.concat([features, lifecycle_features], axis=1)
        
        # 4. 市場カテゴリ特徴量
        if self.config.include_market_features and market_data is not None:
            market_features = self.generate_market_features(features, market_data)
            features = pd.concat([features, market_features], axis=1)
        
        # 5. 時系列特徴量
        if self.config.include_time_series_features:
            time_series_features = self.generate_time_series_features(features)
            features = pd.concat([features, time_series_features], axis=1)
        
        # 6. 交互作用特徴量
        if self.config.include_interaction_features:
            interaction_features = self.generate_interaction_features(features)
            features = pd.concat([features, interaction_features], axis=1)
        
        # 7. 生存分析特徴量
        survival_features = self.generate_survival_features(
            features, company_metadata
        )
        features = pd.concat([features, survival_features], axis=1)
        
        # 8. 新設企業分析特徴量
        emergence_features = self.generate_emergence_features(
            features, company_metadata
        )
        features = pd.concat([features, emergence_features], axis=1)
        
        self.logger.info(f"Feature generation completed. Shape: {features.shape}")
        return features
    
    def generate_evaluation_metrics(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        9つの評価項目計算
        
        Args:
            data: 財務諸表データ
            
        Returns:
            pd.DataFrame: 評価項目データ
        """
        metrics = pd.DataFrame(index=data.index)
        
        # 従来6項目
        if self.config.include_traditional_metrics:
            traditional = self.traditional_calculator.calculate_all_metrics(data)
            metrics = pd.concat([metrics, traditional], axis=1)
        
        # 新規3項目
        if self.config.include_survival_metrics:
            survival = self.survival_calculator.calculate_survival_probability(data)
            metrics = pd.concat([metrics, survival], axis=1)
            
        if self.config.include_emergence_metrics:
            emergence = self.emergence_calculator.calculate_success_rate(data)
            metrics = pd.concat([metrics, emergence], axis=1)
            
        if self.config.include_succession_metrics:
            succession = self.succession_calculator.calculate_succession_success(data)
            metrics = pd.concat([metrics, succession], axis=1)
        
        return metrics
    
    def generate_factor_metrics(
        self, 
        financial_data: pd.DataFrame,
        evaluation_metrics: pd.DataFrame
    ) -> pd.DataFrame:
        """
        各評価項目に対応する23要因項目計算
        
        Args:
            financial_data: 財務諸表データ
            evaluation_metrics: 評価項目データ
            
        Returns:
            pd.DataFrame: 要因項目データ
        """
        return self.factor_calculator.calculate_all_factors(
            financial_data, evaluation_metrics
        )
    
    def generate_lifecycle_features(
        self, 
        data: pd.DataFrame,
        metadata: pd.DataFrame
    ) -> pd.DataFrame:
        """
        ライフサイクル特徴量生成
        
        Args:
            data: 財務データ
            metadata: 企業メタデータ
            
        Returns:
            pd.DataFrame: ライフサイクル特徴量
        """
        return self.lifecycle_generator.generate_features(data, metadata)
    
    def generate_market_features(
        self,
        data: pd.DataFrame,
        market_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        市場カテゴリ特徴量生成
        
        Args:
            data: 財務データ
            market_data: 市場データ
            
        Returns:
            pd.DataFrame: 市場特徴量
        """
        return self.market_generator.generate_features(data, market_data)
    
    def generate_time_series_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        時系列特徴量生成
        
        Args:
            data: 時系列財務データ
            
        Returns:
            pd.DataFrame: 時系列特徴量
        """
        return self.time_series_generator.generate_features(data)
    
    def generate_interaction_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        交互作用特徴量生成
        
        Args:
            data: 基本特徴量データ
            
        Returns:
            pd.DataFrame: 交互作用特徴量
        """
        return self.interaction_generator.generate_features(data)
    
    def generate_survival_features(
        self,
        data: pd.DataFrame,
        metadata: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        生存分析用特徴量生成
        
        Args:
            data: 財務データ
            metadata: 企業メタデータ
            
        Returns:
            pd.DataFrame: 生存分析特徴量
        """
        return self.survival_feature_generator.generate_features(data, metadata)
    
    def generate_emergence_features(
        self,
        data: pd.DataFrame,
        metadata: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        新設企業分析用特徴量生成
        
        Args:
            data: 財務データ
            metadata: 企業メタデータ
            
        Returns:
            pd.DataFrame: 新設企業分析特徴量
        """
        return self.emergence_feature_generator.generate_features(data, metadata)
    
    def get_feature_categories(self) -> Dict[str, List[str]]:
        """
        生成される特徴量のカテゴリ別リスト取得
        
        Returns:
            Dict[str, List[str]]: カテゴリ別特徴量名リスト
        """
        categories = {
            'evaluation_metrics': {
                'traditional': [
                    'revenue', 'revenue_growth_rate', 'operating_margin',
                    'net_margin', 'roe', 'value_added_ratio'
                ],
                'survival': ['survival_probability'],
                'emergence': ['business_success_rate'],
                'succession': ['succession_success_rate']
            },
            'factor_metrics': {
                'investment_assets': [
                    'tangible_fixed_assets', 'capex', 'rd_expenses',
                    'intangible_assets', 'investment_securities'
                ],
                'human_resources': [
                    'employee_count', 'average_salary', 'retirement_costs',
                    'welfare_expenses'
                ],
                'operational_efficiency': [
                    'accounts_receivable', 'inventory', 'total_assets',
                    'receivables_turnover', 'inventory_turnover'
                ],
                'business_expansion': [
                    'overseas_sales_ratio', 'business_segments',
                    'sga_expenses', 'advertising_expenses', 'other_income'
                ],
                'lifecycle_extended': [
                    'company_age', 'market_entry_timing', 'parent_dependency'
                ]
            },
            'time_series_features': [
                'trend_features', 'seasonality_features', 'volatility_features'
            ],
            'interaction_features': [
                'factor_interactions', 'market_interactions', 'lifecycle_interactions'
            ],
            'market_features': [
                'market_category_indicators', 'competitive_position',
                'market_share_dynamics'
            ],
            'lifecycle_features': [
                'stage_indicators', 'transition_probabilities', 'maturity_metrics'
            ],
            'survival_features': [
                'hazard_indicators', 'risk_factors', 'survival_predictors'
            ],
            'emergence_features': [
                'startup_indicators', 'growth_patterns', 'success_predictors'
            ]
        }
        return categories
    
    def get_feature_definitions(self) -> Dict[str, Dict[str, str]]:
        """
        特徴量の定義・説明取得
        
        Returns:
            Dict[str, Dict[str, str]]: 特徴量名: {定義, 計算式, 意義}
        """
        definitions = {}
        
        # 評価項目の定義
        definitions.update({
            'revenue': {
                'definition': '売上高',
                'formula': '損益計算書の売上高',
                'significance': '企業規模・市場ポジションの基本指標'
            },
            'revenue_growth_rate': {
                'definition': '売上高成長率',
                'formula': '(当期売上高 - 前期売上高) / 前期売上高',
                'significance': '企業の成長性・市場競争力の指標'
            },
            'survival_probability': {
                'definition': '企業存続確率',
                'formula': 'Cox回帰による生存確率推定',
                'significance': '企業の持続可能性・リスク評価指標'
            },
            'business_success_rate': {
                'definition': '新規事業成功率',
                'formula': '新規事業・製品の市場定着確率',
                'significance': 'イノベーション能力・成長戦略効果の指標'
            }
        })
        
        return definitions
    
    def _validate_input_data(self, data: pd.DataFrame) -> None:
        """
        入力データの検証
        
        Args:
            data: 検証対象データ
            
        Raises:
            ValueError: データが不正な場合
        """
        if data.empty:
            raise ValueError("Input data is empty")
        
        # 必須列の存在確認
        required_columns = ['company_id', 'fiscal_year']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # データ型確認
        if not pd.api.types.is_datetime64_any_dtype(data.index) and 'fiscal_year' in data.columns:
            if not pd.api.types.is_numeric_dtype(data['fiscal_year']):
                raise ValueError("fiscal_year must be numeric")
        
        # 最小データ数確認
        if len(data) < self.config.min_data_points:
            self.logger.warning(
                f"Data points ({len(data)}) less than minimum required "
                f"({self.config.min_data_points})"
            )
        
        self.logger.debug("Input data validation passed")


# パッケージレベルでのエクスポート
__all__ = [
    'FeatureEngineering',
    'FeatureEngineringConfig',
    'MarketCategory',
    'LifecycleStage',
    'TraditionalMetricsCalculator',
    'SurvivalMetricsCalculator',
    'EmergenceMetricsCalculator',
    'SuccessionMetricsCalculator',
    'FactorMetricsCalculator',
    'TimeSeriesFeatureGenerator',
    'InteractionFeatureGenerator',
    'MarketFeatureGenerator',
    'LifecycleFeatureGenerator',
    'SurvivalFeatureGenerator',
    'EmergenceFeatureGenerator'
]


# モジュール初期化時のバージョン情報
__version__ = "1.0.0"
__author__ = "A2AI Development Team"
__description__ = (
    "Advanced Financial Analysis AI - Feature Engineering Module. "
    "Comprehensive feature generation for corporate lifecycle analysis "
    "including survival, emergence, and succession dynamics."
)