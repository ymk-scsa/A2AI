"""
A2AI (Advanced Financial Analysis AI) - Preprocessing Module

企業ライフサイクル全体（存続・消滅・新設）に対応した財務諸表データ前処理システム

このモジュールは以下の機能を提供します：
- 150社×変動期間の財務データ前処理
- 企業消滅・新設イベント対応
- 生存バイアス補正機能
- ライフサイクル段階別正規化
- 9つの評価項目と拡張要因項目（各23項目）の抽出・計算

市場カテゴリ：
- 高シェア市場: ロボット、内視鏡、工作機械、電子材料、精密測定機器
- シェア低下市場: 自動車、鉄鋼、スマート家電、バッテリー、PC・周辺機器
- 完全失失市場: 家電、半導体、スマートフォン、PC、通信機器

著者: A2AI Development Team
バージョン: 1.0.0
作成日: 2024
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
from datetime import datetime, timedelta

# ========================================
# Core Preprocessing Modules
# ========================================

from .data_cleaner import (
    DataCleaner,
    clean_financial_data,
    standardize_column_names,
    handle_data_type_conversion
)

from .feature_extractor import (
    FeatureExtractor,
    extract_evaluation_metrics,
    extract_factor_metrics,
    calculate_financial_ratios
)

from .factor_calculator import (
    FactorCalculator,
    calculate_traditional_factors,
    calculate_extended_factors,
    compute_interaction_effects
)

from .missing_value_handler import (
    MissingValueHandler,
    handle_missing_values,
    impute_time_series_gaps,
    detect_systematic_missing_patterns
)

from .outlier_detector import (
    OutlierDetector,
    detect_statistical_outliers,
    handle_extreme_values,
    validate_financial_consistency
)

# ========================================
# Lifecycle-Specific Preprocessing Modules
# ========================================

from .lifecycle_normalizer import (
    LifecycleNormalizer,
    normalize_by_lifecycle_stage,
    adjust_for_company_age,
    standardize_growth_metrics
)

from .survivorship_bias_corrector import (
    SurvivorshipBiasCorrector,
    correct_survival_bias,
    weight_extinct_companies,
    adjust_sample_selection_bias
)

from .extinction_feature_engineer import (
    ExtinctionFeatureEngineer,
    extract_extinction_signals,
    calculate_distress_indicators,
    generate_bankruptcy_features
)

from .emergence_success_analyzer import (
    EmergenceSuccessAnalyzer,
    analyze_startup_patterns,
    calculate_early_stage_metrics,
    extract_growth_trajectory_features
)

from .temporal_alignment_handler import (
    TemporalAlignmentHandler,
    align_different_time_periods,
    synchronize_corporate_events,
    handle_variable_observation_windows
)

# ========================================
# Configuration and Constants
# ========================================

# 評価項目定義（9項目）
EVALUATION_METRICS = {
    'traditional': [
        'revenue',                    # 売上高
        'revenue_growth_rate',       # 売上高成長率
        'operating_margin',          # 売上高営業利益率
        'net_margin',               # 売上高当期純利益率
        'roe',                      # ROE
        'value_added_ratio'         # 売上高付加価値率
    ],
    'lifecycle': [
        'survival_probability',      # 企業存続確率
        'emergence_success_rate',    # 新規事業成功率
        'succession_success_rate'    # 事業継承成功度
    ]
}

# 要因項目定義（各評価項目につき23項目）
FACTOR_CATEGORIES = {
    'investment_assets': [
        'tangible_fixed_assets', 'capital_investment', 'rd_expenses',
        'intangible_assets', 'investment_securities'
    ],
    'human_resources': [
        'employee_count', 'average_salary', 'retirement_benefit_cost',
        'welfare_expenses'
    ],
    'operational_efficiency': [
        'accounts_receivable', 'inventory', 'total_assets',
        'receivable_turnover', 'inventory_turnover'
    ],
    'business_expansion': [
        'overseas_sales_ratio', 'business_segments', 'sga_expenses',
        'advertising_expenses', 'non_operating_income', 'order_backlog'
    ],
    'lifecycle_specific': [
        'company_age',              # 企業年齢
        'market_entry_timing',      # 市場参入時期
        'parent_dependency'         # 親会社依存度
    ]
}

# 市場カテゴリ定義
MARKET_CATEGORIES = {
    'high_share': {
        'robotics': ['ファナック', '安川電機', '川崎重工業', '不二越', 'デンソーウェーブ',
                    '三菱電機', 'オムロン', 'THK', 'NSK', 'IHI'],
        'endoscopy': ['オリンパス', 'HOYA', '富士フイルム', 'キヤノンメディカルシステムズ', '島津製作所',
                        'コニカミノルタ', 'ソニー', 'トプコン', 'エムスリー', '日立製作所'],
        'machine_tools': ['DMG森精機', 'ヤマザキマザック', 'オークマ', '牧野フライス製作所', 'ジェイテクト',
                            '東芝機械', 'アマダ', 'ソディック', '三菱重工工作機械', 'シギヤ精機製作所'],
        'electronic_materials': ['村田製作所', 'TDK', '京セラ', '太陽誘電', '日本特殊陶業',
                                'ローム', 'プロテリアル', '住友電工', '日東電工', '日本碍子'],
        'precision_instruments': ['キーエンス', '島津製作所', '堀場製作所', '東京精密', 'ミツトヨ',
                                    'オリンパス', '日本電産', 'リオン', 'アルバック', 'ナブテスコ']
    },
    'declining_share': {
        'automotive': ['トヨタ自動車', '日産自動車', 'ホンダ', 'スズキ', 'マツダ',
                        'SUBARU', 'いすゞ自動車', '三菱自動車', 'ダイハツ工業', '日野自動車'],
        'steel': ['日本製鉄', 'JFEホールディングス', '神戸製鋼所', '日新製鋼', '大同特殊鋼',
                    '山陽特殊製鋼', '愛知製鋼', '中部鋼鈑', '淀川製鋼所', '日立金属'],
        'smart_appliances': ['パナソニック', 'シャープ', 'ソニー', '東芝ライフスタイル',
                            '日立グローバルライフソリューションズ', 'アイリスオーヤマ',
                            '三菱電機', '象印マホービン', 'タイガー魔法瓶', '山善'],
        'battery': ['パナソニックエナジー', '村田製作所', 'GSユアサ', '東芝インフラシステムズ',
                    '日立化成', 'FDK', 'NEC', 'ENAX', '日本電産', 'TDK'],
        'pc_peripherals': ['NEC', '富士通クライアントコンピューティング', '東芝',
                            'ソニー', 'エレコム', 'バッファロー', 'ロジテック',
                            'プリンストン', 'サンワサプライ', 'アイ・オー・データ機器']
    },
    'lost_share': {
        'consumer_electronics': ['ソニー', 'パナソニック', 'シャープ', '東芝ライフスタイル',
                                '三菱電機', '日立グローバルライフソリューションズ',
                                '三洋電機', 'ビクター', 'アイワ', '船井電機'],
        'semiconductors': ['東芝', '日立製作所', '三菱電機', 'NEC', '富士通',
                            '松下電器', 'ソニー', 'ルネサスエレクトロニクス', 'シャープ', 'ローム'],
        'smartphones': ['ソニー', 'シャープ', '京セラ', 'パナソニック', '富士通',
                        'NEC', '日立製作所', '三菱電機', '東芝', 'カシオ計算機'],
        'pc_market': ['ソニー', 'NEC', '富士通', '東芝', 'シャープ', 'パナソニック',
                        '日立製作所', '三菱電機', 'カシオ計算機', '日本電気ホームエレクトロニクス'],
        'telecommunications': ['NEC', '富士通', '日立製作所', 'パナソニック', 'シャープ',
                                'ソニー', '三菱電機', '京セラ', 'カシオ計算機', '日本無線']
    }
}

# データ期間設定
DATA_PERIOD = {
    'start_year': 1984,
    'end_year': 2024,
    'total_years': 40,
    'min_required_years': 5  # 最低必要観測年数
}

# ========================================
# Main Preprocessing Pipeline Class
# ========================================

class A2AIPreprocessor:
    """
    A2AI財務諸表分析システムのメイン前処理クラス
    
    企業ライフサイクル全体（存続・消滅・新設）に対応した
    包括的な財務データ前処理を実行します。
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        前処理システムを初期化
        
        Parameters:
        -----------
        config : Dict, optional
            設定辞書（デフォルト設定を使用する場合はNone）
        """
        self.config = config or self._get_default_config()
        
        # コンポーネント初期化
        self.data_cleaner = DataCleaner(self.config.get('cleaning', {}))
        self.feature_extractor = FeatureExtractor(self.config.get('features', {}))
        self.factor_calculator = FactorCalculator(self.config.get('factors', {}))
        self.missing_handler = MissingValueHandler(self.config.get('missing', {}))
        self.outlier_detector = OutlierDetector(self.config.get('outliers', {}))
        
        # ライフサイクル対応コンポーネント
        self.lifecycle_normalizer = LifecycleNormalizer(self.config.get('lifecycle', {}))
        self.bias_corrector = SurvivorshipBiasCorrector(self.config.get('bias_correction', {}))
        self.extinction_engineer = ExtinctionFeatureEngineer(self.config.get('extinction', {}))
        self.emergence_analyzer = EmergenceSuccessAnalyzer(self.config.get('emergence', {}))
        self.temporal_aligner = TemporalAlignmentHandler(self.config.get('temporal', {}))
        
        # 処理統計
        self.processing_stats = {
            'companies_processed': 0,
            'extinction_events': 0,
            'emergence_events': 0,
            'data_quality_score': 0.0
        }
    
    def preprocess_full_dataset(self, 
                                raw_data: pd.DataFrame,
                                company_events: Optional[pd.DataFrame] = None) -> Dict[str, pd.DataFrame]:
        """
        150社×変動期間の完全データセット前処理
        
        Parameters:
        -----------
        raw_data : pd.DataFrame
            生の財務諸表データ
        company_events : pd.DataFrame, optional
            企業イベント情報（消滅・分社・統合など）
            
        Returns:
        --------
        Dict[str, pd.DataFrame]
            前処理済みデータセット辞書
        """
        print("A2AI前処理パイプライン開始...")
        print(f"対象企業数: {len(raw_data['company_name'].unique())}")
        
        results = {}
        
        # ステップ1: 基本データクリーニング
        print("ステップ1: データクリーニング実行中...")
        cleaned_data = self.data_cleaner.clean_full_dataset(raw_data)
        
        # ステップ2: 企業ライフサイクル情報の統合
        print("ステップ2: ライフサイクル情報統合中...")
        lifecycle_data = self._integrate_lifecycle_info(cleaned_data, company_events)
        
        # ステップ3: 評価項目・要因項目の抽出
        print("ステップ3: 評価項目・要因項目抽出中...")
        feature_data = self.feature_extractor.extract_all_metrics(lifecycle_data)
        
        # ステップ4: 欠損値処理（企業消滅・新設を考慮）
        print("ステップ4: 欠損値処理実行中...")
        imputed_data = self.missing_handler.handle_lifecycle_missing_values(feature_data)
        
        # ステップ5: 外れ値検出・処理
        print("ステップ5: 外れ値検出・処理中...")
        outlier_handled_data = self.outlier_detector.detect_and_handle_outliers(imputed_data)
        
        # ステップ6: ライフサイクル段階別正規化
        print("ステップ6: ライフサイクル正規化実行中...")
        normalized_data = self.lifecycle_normalizer.normalize_by_lifecycle(outlier_handled_data)
        
        # ステップ7: 生存バイアス補正
        print("ステップ7: 生存バイアス補正実行中...")
        bias_corrected_data = self.bias_corrector.correct_survivorship_bias(normalized_data)
        
        # ステップ8: 消滅予兆特徴量生成
        print("ステップ8: 消滅予兆特徴量生成中...")
        extinction_features = self.extinction_engineer.generate_extinction_features(bias_corrected_data)
        
        # ステップ9: 新設企業成功要因分析
        print("ステップ9: 新設企業特徴量生成中...")
        emergence_features = self.emergence_analyzer.analyze_emergence_patterns(bias_corrected_data)
        
        # ステップ10: 時系列整合
        print("ステップ10: 時系列整合実行中...")
        final_data = self.temporal_aligner.align_temporal_data(
            bias_corrected_data, extinction_features, emergence_features
        )
        
        # 結果の構造化
        results['main_dataset'] = final_data
        results['extinction_features'] = extinction_features
        results['emergence_features'] = emergence_features
        results['processing_metadata'] = self._generate_processing_metadata(final_data)
        
        print("A2AI前処理パイプライン完了!")
        print(f"処理済み企業数: {self.processing_stats['companies_processed']}")
        print(f"消滅イベント数: {self.processing_stats['extinction_events']}")
        print(f"新設イベント数: {self.processing_stats['emergence_events']}")
        
        return results
    
    def preprocess_by_market_category(self, 
                                    raw_data: pd.DataFrame,
                                    market_category: str) -> Dict[str, pd.DataFrame]:
        """
        市場カテゴリ別前処理
        
        Parameters:
        -----------
        raw_data : pd.DataFrame
            生データ
        market_category : str
            'high_share', 'declining_share', 'lost_share'のいずれか
            
        Returns:
        --------
        Dict[str, pd.DataFrame]
            市場カテゴリ別前処理済みデータ
        """
        if market_category not in MARKET_CATEGORIES:
            raise ValueError(f"無効な市場カテゴリ: {market_category}")
        
        print(f"{market_category}市場の前処理開始...")
        
        # 対象企業フィルタリング
        target_companies = []
        for market, companies in MARKET_CATEGORIES[market_category].items():
            target_companies.extend(companies)
        
        filtered_data = raw_data[raw_data['company_name'].isin(target_companies)]
        
        # 市場カテゴリ特化前処理
        processed_data = self.preprocess_full_dataset(filtered_data)
        processed_data['market_category'] = market_category
        
        return processed_data
    
    def _integrate_lifecycle_info(self, 
                                data: pd.DataFrame, 
                                events: Optional[pd.DataFrame]) -> pd.DataFrame:
        """企業ライフサイクル情報の統合"""
        lifecycle_data = data.copy()
        
        # 企業年齢計算
        lifecycle_data['company_age'] = self._calculate_company_age(lifecycle_data)
        
        # ライフサイクルステージ分類
        lifecycle_data['lifecycle_stage'] = self._classify_lifecycle_stage(lifecycle_data)
        
        # 企業イベント情報統合
        if events is not None:
            lifecycle_data = self._merge_company_events(lifecycle_data, events)
        
        return lifecycle_data
    
    def _calculate_company_age(self, data: pd.DataFrame) -> pd.Series:
        """企業年齢計算"""
        # 設立年情報から企業年齢を計算
        # 実装では各企業の設立年データベースとの照合が必要
        return data.groupby('company_name')['year'].transform(lambda x: x - x.min())
    
    def _classify_lifecycle_stage(self, data: pd.DataFrame) -> pd.Series:
        """ライフサイクルステージ分類"""
        def classify_stage(group):
            age = group['company_age'].iloc[0] if len(group) > 0 else 0
            if age < 5:
                return 'startup'
            elif age < 15:
                return 'growth'
            elif age < 30:
                return 'mature'
            else:
                return 'established'
        
        return data.groupby('company_name').apply(classify_stage).reindex(data.index, level=1)
    
    def _merge_company_events(self, 
                            data: pd.DataFrame, 
                            events: pd.DataFrame) -> pd.DataFrame:
        """企業イベント情報のマージ"""
        return data.merge(events, on=['company_name', 'year'], how='left')
    
    def _generate_processing_metadata(self, data: pd.DataFrame) -> Dict:
        """前処理メタデータ生成"""
        return {
            'processing_date': datetime.now().isoformat(),
            'total_companies': len(data['company_name'].unique()),
            'total_observations': len(data),
            'date_range': {
                'start': data['year'].min(),
                'end': data['year'].max()
            },
            'data_quality_metrics': self._calculate_data_quality_metrics(data),
            'market_distribution': self._get_market_distribution(data),
            'lifecycle_distribution': self._get_lifecycle_distribution(data)
        }
    
    def _calculate_data_quality_metrics(self, data: pd.DataFrame) -> Dict:
        """データ品質メトリクス計算"""
        return {
            'completeness_rate': (1 - data.isnull().sum().sum() / (len(data) * len(data.columns))),
            'consistency_score': self._calculate_consistency_score(data),
            'temporal_coverage': self._calculate_temporal_coverage(data)
        }
    
    def _calculate_consistency_score(self, data: pd.DataFrame) -> float:
        """データ一貫性スコア計算"""
        # 財務指標の論理的一貫性チェック
        consistency_checks = []
        
        # 例: 売上高 >= 営業利益のチェック
        if 'revenue' in data.columns and 'operating_profit' in data.columns:
            consistency_checks.append(
                (data['revenue'] >= data['operating_profit']).mean()
            )
        
        return np.mean(consistency_checks) if consistency_checks else 1.0
    
    def _calculate_temporal_coverage(self, data: pd.DataFrame) -> Dict:
        """時系列カバレッジ計算"""
        company_coverage = data.groupby('company_name')['year'].agg(['min', 'max', 'count'])
        
        return {
            'avg_observation_years': company_coverage['count'].mean(),
            'min_observation_years': company_coverage['count'].min(),
            'max_observation_years': company_coverage['count'].max(),
            'companies_with_full_coverage': (company_coverage['count'] >= 35).sum()
        }
    
    def _get_market_distribution(self, data: pd.DataFrame) -> Dict:
        """市場分布取得"""
        market_dist = {}
        
        for category, markets in MARKET_CATEGORIES.items():
            companies_in_category = []
            for market, companies in markets.items():
                companies_in_category.extend(companies)
            
            category_data = data[data['company_name'].isin(companies_in_category)]
            market_dist[category] = len(category_data['company_name'].unique())
        
        return market_dist
    
    def _get_lifecycle_distribution(self, data: pd.DataFrame) -> Dict:
        """ライフサイクル分布取得"""
        if 'lifecycle_stage' in data.columns:
            return data['lifecycle_stage'].value_counts().to_dict()
        return {}
    
    def _get_default_config(self) -> Dict:
        """デフォルト設定の取得"""
        return {
            'cleaning': {
                'remove_duplicates': True,
                'standardize_names': True,
                'validate_data_types': True
            },
            'features': {
                'calculate_ratios': True,
                'include_growth_rates': True,
                'generate_interaction_terms': False
            },
            'factors': {
                'include_traditional': True,
                'include_lifecycle': True,
                'calculate_interactions': False
            },
            'missing': {
                'imputation_method': 'time_series_aware',
                'forward_fill_limit': 2,
                'backward_fill_limit': 1
            },
            'outliers': {
                'detection_method': 'isolation_forest',
                'contamination_rate': 0.05,
                'handle_method': 'cap'
            },
            'lifecycle': {
                'normalize_by_age': True,
                'adjust_for_stage': True,
                'use_industry_benchmarks': True
            },
            'bias_correction': {
                'weight_extinct_companies': True,
                'adjust_sample_selection': True,
                'use_inverse_probability_weighting': True
            },
            'extinction': {
                'lookback_years': 3,
                'include_macro_factors': True,
                'generate_early_warning': True
            },
            'emergence': {
                'success_definition_years': 5,
                'include_founder_effects': False,
                'analyze_funding_patterns': False
            },
            'temporal': {
                'alignment_method': 'fiscal_year',
                'handle_different_periods': True,
                'interpolation_method': 'linear'
            }
        }

# ========================================
# Utility Functions
# ========================================

def load_and_preprocess(data_path: str, 
                        config_path: Optional[str] = None) -> Dict[str, pd.DataFrame]:
    """
    データファイルを読み込んで前処理を実行
    
    Parameters:
    -----------
    data_path : str
        データファイルパス
    config_path : str, optional
        設定ファイルパス
        
    Returns:
    --------
    Dict[str, pd.DataFrame]
        前処理済みデータ
    """
    # データ読み込み
    raw_data = pd.read_csv(data_path)
    
    # 設定読み込み
    config = None
    if config_path:
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    
    # 前処理実行
    preprocessor = A2AIPreprocessor(config)
    return preprocessor.preprocess_full_dataset(raw_data)

def validate_preprocessing_results(results: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """
    前処理結果の検証
    
    Parameters:
    -----------
    results : Dict[str, pd.DataFrame]
        前処理結果
        
    Returns:
    --------
    Dict[str, Any]
        検証結果
    """
    validation_results = {}
    
    main_data = results.get('main_dataset')
    if main_data is not None:
        validation_results.update({
            'data_shape': main_data.shape,
            'missing_values': main_data.isnull().sum().sum(),
            'duplicate_records': main_data.duplicated().sum(),
            'numeric_columns': len(main_data.select_dtypes(include=[np.number]).columns),
            'date_range': {
                'start': main_data['year'].min(),
                'end': main_data['year'].max()
            }
        })
    
    return validation_results

# ========================================
# Module Version and Metadata
# ========================================

__version__ = "1.0.0"
__author__ = "A2AI Development Team"
__description__ = "Advanced Financial Analysis AI - Preprocessing Module"

# モジュール公開関数・クラス
__all__ = [
    # メインクラス
    'A2AIPreprocessor',
    
    # 設定・定数
    'EVALUATION_METRICS',
    'FACTOR_CATEGORIES', 
    'MARKET_CATEGORIES',
    'DATA_PERIOD',
    
    # ユーティリティ関数
    'load_and_preprocess',
    'validate_preprocessing_results',
    
    # コアモジュールからの再エクスポート
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