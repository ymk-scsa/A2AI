"""
A2AI Time Series Features Generator
企業ライフサイクル全体を対象とした時系列特徴量生成システム

このモジュールは、存続・消滅・新設企業すべてを対象として、
40年間（1984-2024）の時系列特徴量を生成します。
異なる存続期間の企業間での比較分析を可能にする標準化された特徴量を提供します。
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from sklearn.preprocessing import StandardScaler, RobustScaler
import warnings
from datetime import datetime, timedelta
import logging
from abc import ABC, abstractmethod

class TimeSeriesFeatureError(Exception):
    """時系列特徴量生成に関するカスタム例外"""
    pass

class BaseTimeSeriesFeature(ABC):
    """時系列特徴量生成の基底クラス"""
    
    @abstractmethod
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """特徴量を計算する抽象メソッド"""
        pass

class TimeSeriesFeatureGenerator:
    """
    A2AI時系列特徴量生成クラス
    
    企業ライフサイクル全体（存続・消滅・新設）に対応した
    時系列特徴量を生成します。
    """
    
    def __init__(self, reference_period: int = 40, base_year: int = 1984):
        """
        Args:
            reference_period: 基準期間（年）
            base_year: 基準開始年
        """
        self.reference_period = reference_period
        self.base_year = base_year
        self.end_year = base_year + reference_period
        self.logger = logging.getLogger(__name__)
        
        # スケーラーの初期化
        self.robust_scaler = RobustScaler()
        self.standard_scaler = StandardScaler()
        
        # 特徴量計算クラスの初期化
        self._initialize_feature_calculators()
    
    def _initialize_feature_calculators(self):
        """特徴量計算クラスの初期化"""
        self.trend_calculator = TrendFeatureCalculator()
        self.volatility_calculator = VolatilityFeatureCalculator()
        self.growth_calculator = GrowthFeatureCalculator()
        self.cycle_calculator = CyclicalFeatureCalculator()
        self.lifecycle_calculator = LifecycleFeatureCalculator()
        self.survival_calculator = SurvivalFeatureCalculator()
        self.emergence_calculator = EmergenceFeatureCalculator()
    
    def generate_all_features(self, 
                            company_data: Dict[str, pd.DataFrame],
                            company_metadata: pd.DataFrame) -> pd.DataFrame:
        """
        全時系列特徴量を生成
        
        Args:
            company_data: 企業別財務データ {company_id: DataFrame}
            company_metadata: 企業メタデータ（設立年、消滅年等）
            
        Returns:
            時系列特徴量データフレーム
        """
        all_features = []
        
        for company_id, data in company_data.items():
            try:
                # 企業メタデータ取得
                metadata = company_metadata[
                    company_metadata['company_id'] == company_id
                ].iloc[0]
                
                # 基本時系列特徴量生成
                company_features = self._generate_company_features(
                    data, company_id, metadata
                )
                all_features.append(company_features)
                
            except Exception as e:
                self.logger.warning(f"企業 {company_id} の特徴量生成に失敗: {str(e)}")
                continue
        
        if not all_features:
            raise TimeSeriesFeatureError("特徴量生成に成功した企業が存在しません")
        
        # 全企業の特徴量を統合
        combined_features = pd.concat(all_features, ignore_index=True)
        
        # クロス企業正規化
        normalized_features = self._apply_cross_company_normalization(
            combined_features
        )
        
        return normalized_features
    
    def _generate_company_features(self, 
                                    data: pd.DataFrame,
                                    company_id: str,
                                    metadata: pd.Series) -> pd.DataFrame:
        """
        個別企業の時系列特徴量生成
        
        Args:
            data: 企業財務データ
            company_id: 企業ID
            metadata: 企業メタデータ
            
        Returns:
            企業別時系列特徴量
        """
        # データ前処理
        processed_data = self._preprocess_company_data(data, metadata)
        
        # 各種特徴量計算
        features = {}
        
        # 1. トレンド特徴量
        trend_features = self.trend_calculator.calculate(
            processed_data, metadata=metadata
        )
        features.update(trend_features)
        
        # 2. ボラティリティ特徴量
        volatility_features = self.volatility_calculator.calculate(
            processed_data, metadata=metadata
        )
        features.update(volatility_features)
        
        # 3. 成長特徴量
        growth_features = self.growth_calculator.calculate(
            processed_data, metadata=metadata
        )
        features.update(growth_features)
        
        # 4. 循環・季節性特徴量
        cycle_features = self.cycle_calculator.calculate(
            processed_data, metadata=metadata
        )
        features.update(cycle_features)
        
        # 5. ライフサイクル特徴量
        lifecycle_features = self.lifecycle_calculator.calculate(
            processed_data, metadata=metadata
        )
        features.update(lifecycle_features)
        
        # 6. 生存分析特徴量
        survival_features = self.survival_calculator.calculate(
            processed_data, metadata=metadata
        )
        features.update(survival_features)
        
        # 7. 新設企業特徴量（該当する場合）
        if self._is_emergence_company(metadata):
            emergence_features = self.emergence_calculator.calculate(
                processed_data, metadata=metadata
            )
            features.update(emergence_features)
        
        # 特徴量データフレーム作成
        feature_df = pd.DataFrame([features])
        feature_df['company_id'] = company_id
        feature_df['market_category'] = metadata['market_category']
        feature_df['data_start_year'] = processed_data['year'].min()
        feature_df['data_end_year'] = processed_data['year'].max()
        feature_df['data_length'] = len(processed_data)
        
        return feature_df
    
    def _preprocess_company_data(self, 
                                data: pd.DataFrame, 
                                metadata: pd.Series) -> pd.DataFrame:
        """
        企業データの前処理
        
        Args:
            data: 生データ
            metadata: メタデータ
            
        Returns:
            前処理済みデータ
        """
        processed_data = data.copy()
        
        # 年度カラムの確保
        if 'year' not in processed_data.columns:
            processed_data['year'] = processed_data.index
        
        # データ期間の調整
        start_year = max(metadata.get('establishment_year', self.base_year), 
                        self.base_year)
        end_year = min(metadata.get('extinction_year', self.end_year), 
                        self.end_year)
        
        processed_data = processed_data[
            (processed_data['year'] >= start_year) & 
            (processed_data['year'] <= end_year)
        ]
        
        # 基本的な欠損値処理
        processed_data = self._handle_missing_values(processed_data)
        
        # 外れ値処理
        processed_data = self._handle_outliers(processed_data)
        
        return processed_data
    
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """欠損値処理"""
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if col == 'year':
                continue
            
            # 前方・後方補間
            data[col] = data[col].fillna(method='ffill').fillna(method='bfill')
            
            # まだ欠損値がある場合は業界平均で補完（実装は別途）
            if data[col].isna().any():
                data[col] = data[col].fillna(data[col].median())
        
        return data
    
    def _handle_outliers(self, data: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
        """外れ値処理（IQR方式）"""
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if col == 'year':
                continue
            
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            # 外れ値をクリップ
            data[col] = data[col].clip(lower_bound, upper_bound)
        
        return data
    
    def _is_emergence_company(self, metadata: pd.Series) -> bool:
        """新設企業かどうかの判定"""
        establishment_year = metadata.get('establishment_year', self.base_year)
        return establishment_year > self.base_year
    
    def _apply_cross_company_normalization(self, 
                                            features: pd.DataFrame) -> pd.DataFrame:
        """企業間正規化処理"""
        normalized_features = features.copy()
        
        # 数値カラムの特定
        numeric_columns = features.select_dtypes(include=[np.number]).columns
        metadata_columns = ['company_id', 'market_category', 'data_start_year', 
                            'data_end_year', 'data_length']
        
        feature_columns = [col for col in numeric_columns 
                            if col not in metadata_columns]
        
        # 市場カテゴリ別に正規化
        for category in features['market_category'].unique():
            mask = features['market_category'] == category
            category_data = features.loc[mask, feature_columns]
            
            if len(category_data) > 1:
                # ロバスト正規化（外れ値に対して頑健）
                normalized_data = self.robust_scaler.fit_transform(category_data)
                normalized_features.loc[mask, feature_columns] = normalized_data
        
        return normalized_features

class TrendFeatureCalculator(BaseTimeSeriesFeature):
    """トレンド特徴量計算クラス"""
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> Dict:
        """トレンド特徴量の計算"""
        features = {}
        
        # 評価項目に対するトレンド計算
        evaluation_metrics = [
            '売上高', '売上高成長率', '売上高営業利益率', 
            '売上高当期純利益率', 'ROE', '売上高付加価値率'
        ]
        
        for metric in evaluation_metrics:
            if metric in data.columns:
                trend_stats = self._calculate_trend_statistics(data[metric])
                
                for stat_name, stat_value in trend_stats.items():
                    features[f'{metric}_trend_{stat_name}'] = stat_value
        
        return features
    
    def _calculate_trend_statistics(self, series: pd.Series) -> Dict:
        """個別指標のトレンド統計量計算"""
        values = series.dropna().values
        
        if len(values) < 3:
            return self._get_default_trend_stats()
        
        time_points = np.arange(len(values))
        
        # 線形回帰による傾き
        slope, intercept = np.polyfit(time_points, values, 1)
        
        # トレンドの統計量
        stats = {
            'linear_slope': slope,
            'linear_intercept': intercept,
            'trend_strength': self._calculate_trend_strength(values),
            'monotonic_ratio': self._calculate_monotonic_ratio(values),
            'trend_acceleration': self._calculate_trend_acceleration(values),
            'trend_volatility': self._calculate_trend_volatility(values, slope),
            'early_trend': self._calculate_period_trend(values[:len(values)//3]),
            'mid_trend': self._calculate_period_trend(
                values[len(values)//3:2*len(values)//3]
            ),
            'late_trend': self._calculate_period_trend(values[2*len(values)//3:]),
        }
        
        return stats
    
    def _calculate_trend_strength(self, values: np.ndarray) -> float:
        """トレンド強度の計算"""
        if len(values) < 2:
            return 0.0
        
        time_points = np.arange(len(values))
        correlation = np.corrcoef(time_points, values)[0, 1]
        return correlation**2 if not np.isnan(correlation) else 0.0
    
    def _calculate_monotonic_ratio(self, values: np.ndarray) -> float:
        """単調性比率の計算"""
        if len(values) < 2:
            return 0.0
        
        differences = np.diff(values)
        positive_changes = np.sum(differences > 0)
        negative_changes = np.sum(differences < 0)
        total_changes = len(differences)
        
        if total_changes == 0:
            return 0.0
        
        return max(positive_changes, negative_changes) / total_changes
    
    def _calculate_trend_acceleration(self, values: np.ndarray) -> float:
        """トレンド加速度の計算"""
        if len(values) < 3:
            return 0.0
        
        first_differences = np.diff(values)
        second_differences = np.diff(first_differences)
        
        return np.mean(second_differences) if len(second_differences) > 0 else 0.0
    
    def _calculate_trend_volatility(self, values: np.ndarray, slope: float) -> float:
        """トレンド周りのボラティリティ"""
        if len(values) < 2:
            return 0.0
        
        time_points = np.arange(len(values))
        trend_line = slope * time_points + values[0]
        residuals = values - trend_line
        
        return np.std(residuals)
    
    def _calculate_period_trend(self, period_values: np.ndarray) -> float:
        """期間別トレンド計算"""
        if len(period_values) < 2:
            return 0.0
        
        return (period_values[-1] - period_values[0]) / len(period_values)
    
    def _get_default_trend_stats(self) -> Dict:
        """デフォルトのトレンド統計量"""
        return {
            'linear_slope': 0.0,
            'linear_intercept': 0.0,
            'trend_strength': 0.0,
            'monotonic_ratio': 0.0,
            'trend_acceleration': 0.0,
            'trend_volatility': 0.0,
            'early_trend': 0.0,
            'mid_trend': 0.0,
            'late_trend': 0.0,
        }

class VolatilityFeatureCalculator(BaseTimeSeriesFeature):
    """ボラティリティ特徴量計算クラス"""
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> Dict:
        """ボラティリティ特徴量の計算"""
        features = {}
        
        evaluation_metrics = [
            '売上高', '売上高成長率', '売上高営業利益率', 
            '売上高当期純利益率', 'ROE', '売上高付加価値率'
        ]
        
        for metric in evaluation_metrics:
            if metric in data.columns:
                volatility_stats = self._calculate_volatility_statistics(
                    data[metric]
                )
                
                for stat_name, stat_value in volatility_stats.items():
                    features[f'{metric}_volatility_{stat_name}'] = stat_value
        
        return features
    
    def _calculate_volatility_statistics(self, series: pd.Series) -> Dict:
        """ボラティリティ統計量計算"""
        values = series.dropna().values
        
        if len(values) < 2:
            return self._get_default_volatility_stats()
        
        # 基本統計量
        std_dev = np.std(values)
        mean_val = np.mean(values)
        cv = std_dev / abs(mean_val) if mean_val != 0 else float('inf')
        
        # 時系列ボラティリティ
        returns = np.diff(values) / values[:-1]
        returns = returns[~np.isnan(returns)]
        
        stats = {
            'standard_deviation': std_dev,
            'coefficient_variation': cv,
            'range_volatility': np.max(values) - np.min(values),
            'interquartile_range': np.percentile(values, 75) - np.percentile(values, 25),
            'returns_volatility': np.std(returns) if len(returns) > 0 else 0.0,
            'downside_volatility': self._calculate_downside_volatility(returns),
            'volatility_clustering': self._calculate_volatility_clustering(returns),
            'volatility_persistence': self._calculate_volatility_persistence(returns),
        }
        
        return stats
    
    def _calculate_downside_volatility(self, returns: np.ndarray) -> float:
        """下方ボラティリティ計算"""
        if len(returns) == 0:
            return 0.0
        
        negative_returns = returns[returns < 0]
        return np.std(negative_returns) if len(negative_returns) > 0 else 0.0
    
    def _calculate_volatility_clustering(self, returns: np.ndarray) -> float:
        """ボラティリティクラスタリング計算"""
        if len(returns) < 3:
            return 0.0
        
        abs_returns = np.abs(returns)
        volatility_changes = np.diff(abs_returns)
        
        # 連続する高ボラティリティ期間の検出
        high_vol_threshold = np.percentile(abs_returns, 75)
        high_vol_periods = abs_returns > high_vol_threshold
        
        # クラスタリング指標
        clustering_score = 0.0
        current_cluster_length = 0
        total_clusters = 0
        
        for i, is_high_vol in enumerate(high_vol_periods):
            if is_high_vol:
                current_cluster_length += 1
            else:
                if current_cluster_length > 0:
                    clustering_score += current_cluster_length ** 2
                    total_clusters += 1
                current_cluster_length = 0
        
        if current_cluster_length > 0:
            clustering_score += current_cluster_length ** 2
            total_clusters += 1
        
        return clustering_score / len(returns) if len(returns) > 0 else 0.0
    
    def _calculate_volatility_persistence(self, returns: np.ndarray) -> float:
        """ボラティリティ持続性計算"""
        if len(returns) < 2:
            return 0.0
        
        abs_returns = np.abs(returns)
        
        if len(abs_returns) < 2:
            return 0.0
        
        # 自己相関計算
        correlation = np.corrcoef(abs_returns[:-1], abs_returns[1:])[0, 1]
        return correlation if not np.isnan(correlation) else 0.0
    
    def _get_default_volatility_stats(self) -> Dict:
        """デフォルトのボラティリティ統計量"""
        return {
            'standard_deviation': 0.0,
            'coefficient_variation': 0.0,
            'range_volatility': 0.0,
            'interquartile_range': 0.0,
            'returns_volatility': 0.0,
            'downside_volatility': 0.0,
            'volatility_clustering': 0.0,
            'volatility_persistence': 0.0,
        }

class GrowthFeatureCalculator(BaseTimeSeriesFeature):
    """成長特徴量計算クラス"""
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> Dict:
        """成長特徴量の計算"""
        features = {}
        
        # 成長率計算対象指標
        growth_metrics = ['売上高', '総資産', '純資産', '従業員数']
        
        for metric in growth_metrics:
            if metric in data.columns:
                growth_stats = self._calculate_growth_statistics(data[metric])
                
                for stat_name, stat_value in growth_stats.items():
                    features[f'{metric}_growth_{stat_name}'] = stat_value
        
        # 複合成長指標
        compound_growth = self._calculate_compound_growth_features(data)
        features.update(compound_growth)
        
        return features
    
    def _calculate_growth_statistics(self, series: pd.Series) -> Dict:
        """成長統計量計算"""
        values = series.dropna().values
        
        if len(values) < 2:
            return self._get_default_growth_stats()
        
        # 成長率計算
        growth_rates = []
        for i in range(1, len(values)):
            if values[i-1] != 0:
                growth_rate = (values[i] - values[i-1]) / abs(values[i-1])
                growth_rates.append(growth_rate)
        
        growth_rates = np.array(growth_rates)
        
        if len(growth_rates) == 0:
            return self._get_default_growth_stats()
        
        # CAGR (Compound Annual Growth Rate)
        n_years = len(values) - 1
        cagr = ((values[-1] / values[0]) ** (1/n_years) - 1) if values[0] > 0 and n_years > 0 else 0
        
        stats = {
            'cagr': cagr,
            'mean_growth_rate': np.mean(growth_rates),
            'median_growth_rate': np.median(growth_rates),
            'growth_volatility': np.std(growth_rates),
            'positive_growth_ratio': np.sum(growth_rates > 0) / len(growth_rates),
            'max_growth_rate': np.max(growth_rates),
            'min_growth_rate': np.min(growth_rates),
            'growth_acceleration': self._calculate_growth_acceleration(growth_rates),
            'growth_consistency': self._calculate_growth_consistency(growth_rates),
        }
        
        return stats
    
    def _calculate_growth_acceleration(self, growth_rates: np.ndarray) -> float:
        """成長加速度計算"""
        if len(growth_rates) < 2:
            return 0.0
        
        growth_changes = np.diff(growth_rates)
        return np.mean(growth_changes)
    
    def _calculate_growth_consistency(self, growth_rates: np.ndarray) -> float:
        """成長一貫性計算"""
        if len(growth_rates) < 2:
            return 0.0
        
        # 成長方向の一貫性
        positive_periods = np.sum(growth_rates > 0)
        negative_periods = np.sum(growth_rates < 0)
        total_periods = len(growth_rates)
        
        consistency = max(positive_periods, negative_periods) / total_periods
        return consistency
    
    def _calculate_compound_growth_features(self, data: pd.DataFrame) -> Dict:
        """複合成長特徴量計算"""
        features = {}
        
        # 売上高と利益の成長関係
        if '売上高' in data.columns and '売上高営業利益率' in data.columns:
            revenue_growth = self._calculate_simple_growth_rate(data['売上高'])
            profit_margin_change = self._calculate_simple_growth_rate(data['売上高営業利益率'])
            
            features['revenue_profit_growth_correlation'] = np.corrcoef(
                revenue_growth, profit_margin_change
            )[0, 1] if len(revenue_growth) > 1 and not np.isnan(np.corrcoef(revenue_growth, profit_margin_change)[0, 1]) else 0.0
        
        # 資産効率と成長の関係
        if '売上高' in data.columns and '総資産' in data.columns:
            revenue_growth = self._calculate_simple_growth_rate(data['売上高'])
            asset_growth = self._calculate_simple_growth_rate(data['総資産'])
            
            if len(revenue_growth) > 0 and len(asset_growth) > 0:
                features['asset_efficiency_growth'] = np.mean(revenue_growth) - np.mean(asset_growth)
        
        return features
    
    def _calculate_simple_growth_rate(self, series: pd.Series) -> np.ndarray:
        """単純成長率計算"""
        values = series.dropna().values
        if len(values) < 2:
            return np.array([])
        
        growth_rates = []
        for i in range(1, len(values)):
            if values[i-1] != 0:
                growth_rate = (values[i] - values[i-1]) / abs(values[i-1])
                growth_rates.append(growth_rate)
        
        return np.array(growth_rates)
    
    def _get_default_growth_stats(self) -> Dict:
        """デフォルトの成長統計量"""
        return {
            'cagr': 0.0,
            'mean_growth_rate': 0.0,
            'median_growth_rate': 0.0,
            'growth_volatility': 0.0,
            'positive_growth_ratio': 0.5,
            'max_growth_rate': 0.0,
            'min_growth_rate': 0.0,
            'growth_acceleration': 0.0,
            'growth_consistency': 0.5,
        }

class CyclicalFeatureCalculator(BaseTimeSeriesFeature):
    """循環・季節性特徴量計算クラス"""
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> Dict:
        """循環・季節性特徴量の計算"""
        features = {}
        
        evaluation_metrics = [
            '売上高', '売上高営業利益率', '売上高当期純利益率', 'ROE'
        ]
        
        for metric in evaluation_metrics:
            if metric in data.columns:
                cycle_stats = self._calculate_cyclical_statistics(data[metric])
                
                for stat_name, stat_value in cycle_stats.items():
                    features[f'{metric}_cycle_{stat_name}'] = stat_value
        
        return features
    
    def _calculate_cyclical_statistics(self, series: pd.Series) -> Dict:
        """循環性統計量計算"""
        values = series.dropna().values
        
        if len(values) < 4:
            return self._get_default_cycle_stats()
        
        # 周期性の検出
        cycle_features = self._detect_cycles(values)
        
        # トレンド除去後の循環性
        detrended_values = self._detrend_series(values)
        detrended_features = self._analyze_detrended_cycles(detrended_values)
        
        stats = {**cycle_features, **detrended_features}
        
        return stats
    
    def _detect_cycles(self, values: np.ndarray) -> Dict:
        """周期性検出"""
        features = {}
        
        # 自己相関による周期性検出
        max_lag = min(len(values) // 2, 10)
        autocorrelations = []
        
        for lag in range(1, max_lag + 1):
            if len(values) > lag:
                correlation = np.corrcoef(values[:-lag], values[lag:])[0, 1]
                autocorrelations.append(correlation if not np.isnan(correlation) else 0.0)
        
        if autocorrelations:
            features['max_autocorrelation'] = max(autocorrelations)
            features['autocorr_decay_rate'] = self._calculate_autocorr_decay(autocorrelations)
        else:
            features['max_autocorrelation'] = 0.0
            features['autocorr_decay_rate'] = 0.0
        
        # ピーク・トラフ分析
        peaks_troughs = self._analyze_peaks_troughs(values)
        features.update(peaks_troughs)
        
        return features
    
    def _detrend_series(self, values: np.ndarray) -> np.ndarray:
        """トレンド除去"""
        if len(values) < 3:
            return values
        
        # 線形トレンド除去
        time_points = np.arange(len(values))
        slope, intercept = np.polyfit(time_points, values, 1)
        trend = slope * time_points + intercept
        
        return values - trend
    
    def _analyze_detrended_cycles(self, detrended_values: np.ndarray) -> Dict:
        """トレンド除去後の循環分析"""
        if len(detrended_values) < 3:
            return {'cycle_regularity': 0.0, 'cycle_amplitude': 0.0}
        
        # 循環の規則性
        zero_crossings = self._count_zero_crossings(detrended_values)
        cycle_regularity = zero_crossings / len(detrended_values)
        
        # 循環の振幅
        cycle_amplitude = np.std(detrended_values)
        
        return {
            'cycle_regularity': cycle_regularity,
            'cycle_amplitude': cycle_amplitude
        }
    
    def _calculate_autocorr_decay(self, autocorrelations: List[float]) -> float:
        """自己相関減衰率計算"""
        if len(autocorrelations) < 2:
            return 0.0
        
        # 最初の有意な自己相関から減衰率を計算
        significant_autocorrs = [ac for ac in autocorrelations if abs(ac) > 0.1]
        
        if len(significant_autocorrs) < 2:
            return 1.0  # 急速な減衰
        
        # 指数的減衰の推定
        decay_rate = abs(significant_autocorrs[-1] / significant_autocorrs[0])
        return min(decay_rate, 1.0)
    
    def _analyze_peaks_troughs(self, values: np.ndarray) -> Dict:
        """ピーク・トラフ分析"""
        if len(values) < 5:
            return {'peak_frequency': 0.0, 'trough_frequency': 0.0, 'peak_trough_regularity': 0.0}
        
        # 極値検出（簡易版）
        peaks = []
        troughs = []
        
        for i in range(1, len(values) - 1):
            if values[i] > values[i-1] and values[i] > values[i+1]:
                peaks.append(i)
            elif values[i] < values[i-1] and values[i] < values[i+1]:
                troughs.append(i)
        
        # 頻度計算
        peak_frequency = len(peaks) / len(values)
        trough_frequency = len(troughs) / len(values)
        
        # 規則性計算
        regularity = 0.0
        if len(peaks) > 1 and len(troughs) > 1:
            peak_intervals = np.diff(peaks)
            trough_intervals = np.diff(troughs)
            
            peak_regularity = 1 - (np.std(peak_intervals) / np.mean(peak_intervals)) if np.mean(peak_intervals) > 0 else 0
            trough_regularity = 1 - (np.std(trough_intervals) / np.mean(trough_intervals)) if np.mean(trough_intervals) > 0 else 0
            
            regularity = (peak_regularity + trough_regularity) / 2
        
        return {
            'peak_frequency': peak_frequency,
            'trough_frequency': trough_frequency,
            'peak_trough_regularity': max(0, min(1, regularity))
        }
    
    def _count_zero_crossings(self, values: np.ndarray) -> int:
        """ゼロクロッシング数計算"""
        if len(values) < 2:
            return 0
        
        crossings = 0
        for i in range(1, len(values)):
            if values[i-1] * values[i] < 0:
                crossings += 1
        
        return crossings
    
    def _get_default_cycle_stats(self) -> Dict:
        """デフォルトの循環統計量"""
        return {
            'max_autocorrelation': 0.0,
            'autocorr_decay_rate': 0.0,
            'peak_frequency': 0.0,
            'trough_frequency': 0.0,
            'peak_trough_regularity': 0.0,
            'cycle_regularity': 0.0,
            'cycle_amplitude': 0.0,
        }

class LifecycleFeatureCalculator(BaseTimeSeriesFeature):
    """ライフサイクル特徴量計算クラス"""
    
    def calculate(self, data: pd.DataFrame, metadata: pd.Series = None, **kwargs) -> Dict:
        """ライフサイクル特徴量の計算"""
        features = {}
        
        if metadata is None:
            return {}
        
        # 基本ライフサイクル情報
        establishment_year = metadata.get('establishment_year', self.base_year)
        extinction_year = metadata.get('extinction_year', None)
        current_year = 2024
        
        # 企業年齢関連特徴量
        age_features = self._calculate_age_features(
            establishment_year, extinction_year, current_year, len(data)
        )
        features.update(age_features)
        
        # ライフサイクルステージ特徴量
        stage_features = self._calculate_lifecycle_stage_features(
            data, establishment_year, extinction_year
        )
        features.update(stage_features)
        
        # 成熟度指標
        maturity_features = self._calculate_maturity_features(data)
        features.update(maturity_features)
        
        return features
    
    def _calculate_age_features(self, establishment_year: int, 
                                extinction_year: Optional[int],
                                current_year: int, data_length: int) -> Dict:
        """企業年齢関連特徴量"""
        
        # 基本年齢計算
        if extinction_year:
            total_lifespan = extinction_year - establishment_year
            is_extinct = 1
            years_since_extinction = current_year - extinction_year
        else:
            total_lifespan = current_year - establishment_year
            is_extinct = 0
            years_since_extinction = 0
        
        # 相対年齢（業界内での相対的な年齢）
        base_year = 1984
        industry_age_reference = current_year - base_year
        relative_age = total_lifespan / industry_age_reference if industry_age_reference > 0 else 0
        
        return {
            'company_age_years': total_lifespan,
            'relative_industry_age': relative_age,
            'is_extinct': is_extinct,
            'years_since_extinction': years_since_extinction,
            'data_coverage_ratio': data_length / total_lifespan if total_lifespan > 0 else 0,
            'establishment_era': self._get_establishment_era(establishment_year),
        }
    
    def _calculate_lifecycle_stage_features(self, data: pd.DataFrame,
                                            establishment_year: int,
                                            extinction_year: Optional[int]) -> Dict:
        """ライフサイクルステージ特徴量"""
        
        data_with_age = data.copy()
        if 'year' in data.columns:
            data_with_age['company_age'] = data_with_age['year'] - establishment_year
        else:
            data_with_age['company_age'] = range(len(data))
        
        # ステージ分析対象指標
        key_metrics = ['売上高', '売上高成長率', '売上高営業利益率', 'ROE']
        
        stage_features = {}
        
        for metric in key_metrics:
            if metric in data.columns:
                metric_stages = self._analyze_metric_lifecycle_stages(
                    data_with_age[metric], data_with_age['company_age']
                )
                
                for stage_name, stage_value in metric_stages.items():
                    stage_features[f'{metric}_stage_{stage_name}'] = stage_value
        
        # 総合ライフサイクルステージ判定
        overall_stage = self._determine_overall_lifecycle_stage(
            data_with_age, establishment_year, extinction_year
        )
        stage_features.update(overall_stage)
        
        return stage_features
    
    def _analyze_metric_lifecycle_stages(self, metric_values: pd.Series, 
                                        ages: pd.Series) -> Dict:
        """指標別ライフサイクルステージ分析"""
        
        if len(metric_values.dropna()) < 3:
            return self._get_default_stage_features()
        
        # データを年齢でソート
        combined_data = pd.DataFrame({
            'metric': metric_values,
            'age': ages
        }).dropna().sort_values('age')
        
        if len(combined_data) < 3:
            return self._get_default_stage_features()
        
        # ステージ分割（初期・成長・成熟・衰退）
        n_periods = len(combined_data)
        stage_size = max(1, n_periods // 4)
        
        stages = {
            'startup': combined_data.iloc[:stage_size]['metric'],
            'growth': combined_data.iloc[stage_size:2*stage_size]['metric'],
            'maturity': combined_data.iloc[2*stage_size:3*stage_size]['metric'],
            'decline': combined_data.iloc[3*stage_size:]['metric']
        }
        
        stage_features = {}
        
        for stage_name, stage_data in stages.items():
            if len(stage_data) > 0:
                stage_features[f'{stage_name}_mean'] = stage_data.mean()
                stage_features[f'{stage_name}_growth'] = self._calculate_stage_growth_rate(stage_data)
                stage_features[f'{stage_name}_volatility'] = stage_data.std()
            else:
                stage_features[f'{stage_name}_mean'] = 0.0
                stage_features[f'{stage_name}_growth'] = 0.0
                stage_features[f'{stage_name}_volatility'] = 0.0
        
        return stage_features
    
    def _calculate_stage_growth_rate(self, stage_data: pd.Series) -> float:
        """ステージ内成長率計算"""
        if len(stage_data) < 2:
            return 0.0
        
        first_value = stage_data.iloc[0]
        last_value = stage_data.iloc[-1]
        
        if first_value == 0:
            return 0.0
        
        return (last_value - first_value) / abs(first_value)
    
    def _determine_overall_lifecycle_stage(self, data: pd.DataFrame,
                                            establishment_year: int,
                                            extinction_year: Optional[int]) -> Dict:
        """総合ライフサイクルステージ判定"""
        
        features = {}
        
        # 年齢ベースのステージ判定
        if 'company_age' in data.columns and len(data) > 0:
            max_age = data['company_age'].max()
            
            if max_age <= 5:
                primary_stage = 'startup'
            elif max_age <= 15:
                primary_stage = 'growth'
            elif max_age <= 30:
                primary_stage = 'maturity'
            else:
                primary_stage = 'mature_stable'
                
            if extinction_year:
                primary_stage = 'declined_extinct'
        else:
            primary_stage = 'unknown'
        
        features['primary_lifecycle_stage'] = primary_stage
        
        # ステージ遷移の分析
        if '売上高成長率' in data.columns:
            growth_trend = self._analyze_growth_stage_transitions(data['売上高成長率'])
            features.update(growth_trend)
        
        return features
    
    def _analyze_growth_stage_transitions(self, growth_rates: pd.Series) -> Dict:
        """成長ステージ遷移分析"""
        
        clean_growth = growth_rates.dropna()
        if len(clean_growth) < 5:
            return {'stage_transitions': 0, 'declining_phase_ratio': 0.0}
        
        # 高成長・安定成長・低成長・衰退の閾値
        high_growth_threshold = 0.15
        stable_growth_threshold = 0.05
        decline_threshold = 0.0
        
        # ステージ分類
        stages = []
        for growth in clean_growth:
            if growth > high_growth_threshold:
                stages.append('high_growth')
            elif growth > stable_growth_threshold:
                stages.append('stable_growth')
            elif growth > decline_threshold:
                stages.append('low_growth')
            else:
                stages.append('decline')
        
        # ステージ遷移回数
        transitions = sum(1 for i in range(1, len(stages)) if stages[i] != stages[i-1])
        
        # 衰退期間比率
        decline_periods = sum(1 for stage in stages if stage == 'decline')
        decline_ratio = decline_periods / len(stages)
        
        return {
            'stage_transitions': transitions,
            'declining_phase_ratio': decline_ratio
        }
    
    def _calculate_maturity_features(self, data: pd.DataFrame) -> Dict:
        """成熟度指標計算"""
        
        maturity_features = {}
        
        # 財務指標の安定性（成熟度の指標）
        stability_metrics = ['売上高成長率', '売上高営業利益率', 'ROE']
        
        for metric in stability_metrics:
            if metric in data.columns:
                stability_score = self._calculate_stability_score(data[metric])
                maturity_features[f'{metric}_stability_score'] = stability_score
        
        # 総合成熟度指標
        overall_maturity = self._calculate_overall_maturity_index(data)
        maturity_features['overall_maturity_index'] = overall_maturity
        
        return maturity_features
    
    def _calculate_stability_score(self, series: pd.Series) -> float:
        """安定性スコア計算"""
        
        clean_series = series.dropna()
        if len(clean_series) < 3:
            return 0.0
        
        # 変動係数の逆数を安定性の指標とする
        cv = clean_series.std() / abs(clean_series.mean()) if clean_series.mean() != 0 else float('inf')
        
        # 安定性スコア（0-1の範囲）
        stability_score = 1 / (1 + cv) if cv != float('inf') else 0.0
        
        return stability_score
    
    def _calculate_overall_maturity_index(self, data: pd.DataFrame) -> float:
        """総合成熟度指数計算"""
        
        maturity_indicators = []
        
        # 1. 成長率の安定性
        if '売上高成長率' in data.columns:
            growth_stability = self._calculate_stability_score(data['売上高成長率'])
            maturity_indicators.append(growth_stability)
        
        # 2. 収益性の安定性
        if '売上高営業利益率' in data.columns:
            profit_stability = self._calculate_stability_score(data['売上高営業利益率'])
            maturity_indicators.append(profit_stability)
        
        # 3. 財務健全性（ROEの一貫性）
        if 'ROE' in data.columns:
            roe_consistency = self._calculate_stability_score(data['ROE'])
            maturity_indicators.append(roe_consistency)
        
        # 総合成熟度指数
        if maturity_indicators:
            return sum(maturity_indicators) / len(maturity_indicators)
        else:
            return 0.0
    
    def _get_establishment_era(self, establishment_year: int) -> str:
        """設立時代の分類"""
        if establishment_year < 1945:
            return 'pre_war'
        elif establishment_year < 1970:
            return 'post_war_recovery'
        elif establishment_year < 1990:
            return 'high_growth_era'
        elif establishment_year < 2000:
            return 'bubble_collapse'
        elif establishment_year < 2010:
            return 'digital_transformation'
        else:
            return 'modern_era'
    
    def _get_default_stage_features(self) -> Dict:
        """デフォルトのステージ特徴量"""
        stages = ['startup', 'growth', 'maturity', 'decline']
        features = {}
        
        for stage in stages:
            features[f'{stage}_mean'] = 0.0
            features[f'{stage}_growth'] = 0.0
            features[f'{stage}_volatility'] = 0.0
        
        return features

class SurvivalFeatureCalculator(BaseTimeSeriesFeature):
    """生存分析特徴量計算クラス"""
    
    def calculate(self, data: pd.DataFrame, metadata: pd.Series = None, **kwargs) -> Dict:
        """生存分析特徴量の計算"""
        features = {}
        
        if metadata is None:
            return {}
        
        # 生存状態の判定
        extinction_year = metadata.get('extinction_year', None)
        is_extinct = extinction_year is not None
        
        # 基本生存特徴量
        survival_basics = self._calculate_basic_survival_features(
            data, metadata, is_extinct
        )
        features.update(survival_basics)
        
        # 生存リスク指標
        risk_features = self._calculate_survival_risk_features(data)
        features.update(risk_features)
        
        # 生存予測特徴量
        if not is_extinct:
            prediction_features = self._calculate_survival_prediction_features(data)
            features.update(prediction_features)
        
        return features
    
    def _calculate_basic_survival_features(self, data: pd.DataFrame, 
                                            metadata: pd.Series, 
                                            is_extinct: bool) -> Dict:
        """基本生存特徴量計算"""
        
        establishment_year = metadata.get('establishment_year', 1984)
        extinction_year = metadata.get('extinction_year', None)
        current_year = 2024
        
        if is_extinct:
            survival_time = extinction_year - establishment_year
            censoring_indicator = 1  # 事象発生（死亡）
        else:
            survival_time = current_year - establishment_year
            censoring_indicator = 0  # 右打ち切り（生存中）
        
        return {
            'survival_time_years': survival_time,
            'is_censored': 1 - censoring_indicator,
            'event_occurred': censoring_indicator,
            'survival_rate': 1 - censoring_indicator,
        }
    
    def _calculate_survival_risk_features(self, data: pd.DataFrame) -> Dict:
        """生存リスク特徴量計算"""
        
        risk_features = {}
        
        # 財務健全性リスク指標
        financial_risk_metrics = [
            '売上高営業利益率', '売上高当期純利益率', 'ROE'
        ]
        
        for metric in financial_risk_metrics:
            if metric in data.columns:
                metric_risk = self._calculate_financial_risk_score(data[metric])
                risk_features[f'{metric}_financial_risk'] = metric_risk
        
        # 成長性リスク指標
        if '売上高成長率' in data.columns:
            growth_risk = self._calculate_growth_risk_score(data['売上高成長率'])
            risk_features['growth_risk_score'] = growth_risk
        
        # 安定性リスク指標
        stability_risk = self._calculate_stability_risk_score(data)
        risk_features['stability_risk_score'] = stability_risk
        
        # 総合リスクスコア
        risk_scores = [v for v in risk_features.values() if isinstance(v, (int, float))]
        if risk_scores:
            risk_features['overall_survival_risk'] = sum(risk_scores) / len(risk_scores)
        else:
            risk_features['overall_survival_risk'] = 0.5
        
        return risk_features
    
    def _calculate_financial_risk_score(self, metric_series: pd.Series) -> float:
        """財務リスクスコア計算"""
        
        clean_series = metric_series.dropna()
        if len(clean_series) < 2:
            return 0.5
        
        # 負の値の比率
        negative_ratio = (clean_series < 0).sum() / len(clean_series)
        
        # 極端な変動の頻度
        volatility = clean_series.std()
        mean_val = clean_series.mean()
        
        extreme_volatility_ratio = volatility / abs(mean_val) if mean_val != 0 else 1.0
        
        # リスクスコア（0: 低リスク, 1: 高リスク）
        risk_score = min(1.0, (negative_ratio * 0.7 + 
                                min(extreme_volatility_ratio / 2.0, 0.3)))
        
        return risk_score
    
    def _calculate_growth_risk_score(self, growth_series: pd.Series) -> float:
        """成長リスクスコア計算"""
        
        clean_series = growth_series.dropna()
        if len(clean_series) < 3:
            return 0.5
        
        # 連続する負成長期間の検出
        negative_periods = []
        current_negative_streak = 0
        
        for growth in clean_series:
            if growth < 0:
                current_negative_streak += 1
            else:
                if current_negative_streak > 0:
                    negative_periods.append(current_negative_streak)
                current_negative_streak = 0
        
        if current_negative_streak > 0:
            negative_periods.append(current_negative_streak)
        
        # 最長連続負成長期間
        max_negative_streak = max(negative_periods) if negative_periods else 0
        
        # 成長リスクスコア
        risk_score = min(1.0, max_negative_streak / len(clean_series))
        
        return risk_score
    
    def _calculate_stability_risk_score(self, data: pd.DataFrame) -> float:
        """安定性リスクスコア計算"""
        
        stability_metrics = ['売上高', '売上高営業利益率', 'ROE']
        stability_scores = []
        
        for metric in stability_metrics:
            if metric in data.columns:
                metric_data = data[metric].dropna()
                if len(metric_data) > 2:
                    # 変動係数
                    cv = metric_data.std() / abs(metric_data.mean()) if metric_data.mean() != 0 else float('inf')
                    
                    # 安定性スコア（変動係数の逆数ベース）
                    stability_score = 1 / (1 + cv) if cv != float('inf') else 0
                    stability_scores.append(stability_score)
        
        if stability_scores:
            average_stability = sum(stability_scores) / len(stability_scores)
            risk_score = 1 - average_stability  # 安定性の逆がリスク
        else:
            risk_score = 0.5
        
        return risk_score
    
    def _calculate_survival_prediction_features(self, data: pd.DataFrame) -> Dict:
        """生存予測特徴量計算（生存企業向け）"""
        
        prediction_features = {}
        
        # 最近のトレンド（直近5年）
        recent_data = data.tail(5) if len(data) >= 5 else data
        
        # 最近の財務健全性トレンド
        if '売上高営業利益率' in recent_data.columns:
            profit_trend = self._calculate_recent_trend(recent_data['売上高営業利益率'])
            prediction_features['recent_profitability_trend'] = profit_trend
        
        # 最近の成長トレンド
        if '売上高成長率' in recent_data.columns:
            growth_trend = self._calculate_recent_trend(recent_data['売上高成長率'])
            prediction_features['recent_growth_trend'] = growth_trend
        
        # 最近の財務安定性
        recent_stability = self._calculate_recent_stability(recent_data)
        prediction_features['recent_stability_score'] = recent_stability
        
        # 予測生存確率（簡易モデル）
        survival_probability = self._estimate_survival_probability(
            prediction_features
        )
        prediction_features['estimated_survival_probability'] = survival_probability
        
        return prediction_features
    
    def _calculate_recent_trend(self, series: pd.Series) -> float:
        """最近のトレンド計算"""
        
        clean_series = series.dropna()
        if len(clean_series) < 2:
            return 0.0
        
        # 線形回帰による傾き
        x = np.arange(len(clean_series))
        y = clean_series.values
        
        if len(x) > 1:
            slope, _ = np.polyfit(x, y, 1)
            return slope
        else:
            return 0.0
    
    def _calculate_recent_stability(self, data: pd.DataFrame) -> float:
        """最近の安定性計算"""
        
        key_metrics = ['売上高営業利益率', 'ROE']
        stability_scores = []
        
        for metric in key_metrics:
            if metric in data.columns:
                metric_data = data[metric].dropna()
                if len(metric_data) > 1:
                    cv = metric_data.std() / abs(metric_data.mean()) if metric_data.mean() != 0 else float('inf')
                    stability = 1 / (1 + cv) if cv != float('inf') else 0
                    stability_scores.append(stability)
        
        return sum(stability_scores) / len(stability_scores) if stability_scores else 0.5
    
    def _estimate_survival_probability(self, features: Dict) -> float:
        """生存確率推定（簡易モデル）"""
        
        # 重み付き総合スコア
        profitability_weight = 0.4
        growth_weight = 0.3
        stability_weight = 0.3
        
        profitability_score = max(0, min(1, 0.5 + features.get('recent_profitability_trend', 0) * 10))
        growth_score = max(0, min(1, 0.5 + features.get('recent_growth_trend', 0) * 5))
        stability_score = features.get('recent_stability_score', 0.5)
        
        survival_probability = (
            profitability_weight * profitability_score +
            growth_weight * growth_score +
            stability_weight * stability_score
        )
        
        return survival_probability

class EmergenceFeatureCalculator(BaseTimeSeriesFeature):
    """新設企業特徴量計算クラス"""
    
    def calculate(self, data: pd.DataFrame, metadata: pd.Series = None, **kwargs) -> Dict:
        """新設企業特徴量の計算"""
        features = {}
        
        if metadata is None:
            return {}
        
        establishment_year = metadata.get('establishment_year', 1984)
        base_year = 1984
        
        # 新設企業でない場合は空の特徴量を返す
        if establishment_year <= base_year:
            return {}
        
        # 新設企業基本特徴量
        emergence_basics = self._calculate_emergence_basics(
            data, metadata, establishment_year
        )
        features.update(emergence_basics)
        
        # 初期成長特徴量
        initial_growth = self._calculate_initial_growth_features(data)
        features.update(initial_growth)
        
        # 市場参入戦略特徴量
        entry_strategy = self._calculate_market_entry_features(data, metadata)
        features.update(entry_strategy)
        
        # 成功予測特徴量
        success_prediction = self._calculate_success_prediction_features(data)
        features.update(success_prediction)
        
        return features
    
    def _calculate_emergence_basics(self, data: pd.DataFrame, 
                                    metadata: pd.Series, 
                                    establishment_year: int) -> Dict:
        """新設企業基本特徴量"""
        
        current_year = 2024
        company_age = current_year - establishment_year
        
        # 分社元情報
        parent_company = metadata.get('parent_company', None)
        is_spinoff = parent_company is not None
        
        # 設立時期の市場環境
        establishment_era = self._classify_establishment_era(establishment_year)
        
        return {
            'emergence_year': establishment_year,
            'years_since_establishment': company_age,
            'is_spinoff': 1 if is_spinoff else 0,
            'establishment_era_code': establishment_era,
            'data_years_available': len(data),
        }
    
    def _calculate_initial_growth_features(self, data: pd.DataFrame) -> Dict:
        """初期成長特徴量計算"""
        
        # 初期期間（設立後5年以内または利用可能データの前半）
        initial_period_length = min(5, len(data) // 2)
        initial_data = data.head(initial_period_length) if initial_period_length > 0 else data
        
        features = {}
        
        if len(initial_data) < 2:
            return self._get_default_emergence_features()
        
        # 初期成長率
        growth_metrics = ['売上高', '売上高成長率', '総資産']
        
        for metric in growth_metrics:
            if metric in initial_data.columns:
                initial_growth = self._calculate_initial_metric_growth(
                    initial_data[metric]
                )
                features[f'initial_{metric}_growth_rate'] = initial_growth
        
        # 初期収益性
        if '売上高営業利益率' in initial_data.columns:
            initial_profitability = initial_data['売上高営業利益率'].mean()
            features['initial_profitability_average'] = initial_profitability
        
        # 初期財務安定性
        initial_stability = self._calculate_initial_stability(initial_data)
        features['initial_financial_stability'] = initial_stability
        
        # 急成長期間の検出
        rapid_growth_period = self._detect_rapid_growth_period(data)
        features.update(rapid_growth_period)
        
        return features
    
    def _calculate_market_entry_features(self, data: pd.DataFrame, 
                                        metadata: pd.Series) -> Dict:
        """市場参入戦略特徴量"""
        
        features = {}
        
        # 市場参入時期の分析
        market_category = metadata.get('market_category', 'unknown')
        establishment_year = metadata.get('establishment_year', 2000)
        
        # 参入時期の特徴
        entry_timing = self._analyze_market_entry_timing(
            establishment_year, market_category
        )
        features.update(entry_timing)
        
        # 初期市場ポジション
        if len(data) >= 3:
            initial_position = self._analyze_initial_market_position(data)
            features.update(initial_position)
        
        # 差別化戦略の指標
        if '売上高付加価値率' in data.columns:
            differentiation_strategy = self._analyze_differentiation_strategy(
                data['売上高付加価値率']
            )
            features.update(differentiation_strategy)
        
        return features
    
    def _calculate_success_prediction_features(self, data: pd.DataFrame) -> Dict:
        """成功予測特徴量計算"""
        
        features = {}
        
        if len(data) < 3:
            return features
        
        # 成長軌道の分析
        growth_trajectory = self._analyze_growth_trajectory(data)
        features.update(growth_trajectory)
        
        # 持続可能性指標
        sustainability_indicators = self._calculate_sustainability_indicators(data)
        features.update(sustainability_indicators)
        
        # 競争力指標
        competitiveness_score = self._calculate_competitiveness_score(data)
        features['competitiveness_score'] = competitiveness_score
        
        # 総合成功予測スコア
        success_probability = self._estimate_success_probability(features)
        features['estimated_success_probability'] = success_probability
        
        return features
    
    def _calculate_initial_metric_growth(self, series: pd.Series) -> float:
        """初期指標成長率計算"""
        
        clean_series = series.dropna()
        if len(clean_series) < 2:
            return 0.0
        
        first_value = clean_series.iloc[0]
        last_value = clean_series.iloc[-1]
        
        if first_value == 0:
            return float('inf') if last_value > 0 else 0.0
        
        # 年平均成長率
        n_periods = len(clean_series) - 1
        if n_periods > 0:
            growth_rate = ((last_value / first_value) ** (1/n_periods) - 1)
            return growth_rate
        else:
            return 0.0
    
    def _calculate_initial_stability(self, data: pd.DataFrame) -> float:
        """初期財務安定性計算"""
        
        stability_metrics = ['売上高営業利益率', 'ROE']
        stability_scores = []
        
        for metric in stability_metrics:
            if metric in data.columns:
                metric_data = data[metric].dropna()
                if len(metric_data) > 1:
                    # 変動係数の逆数
                    cv = metric_data.std() / abs(metric_data.mean()) if metric_data.mean() != 0 else float('inf')
                    stability = 1 / (1 + cv) if cv != float('inf') else 0
                    stability_scores.append(stability)
        
        return sum(stability_scores) / len(stability_scores) if stability_scores else 0.5
    
    def _detect_rapid_growth_period(self, data: pd.DataFrame) -> Dict:
        """急成長期間検出"""
        
        features = {}
        
        if '売上高成長率' not in data.columns or len(data) < 3:
            return {'has_rapid_growth_period': 0, 'rapid_growth_duration': 0}
        
        growth_rates = data['売上高成長率'].dropna()
        
        # 急成長の閾値（年20%以上）
        rapid_growth_threshold = 0.20
        rapid_growth_periods = growth_rates > rapid_growth_threshold
        
        # 連続する急成長期間の検出
        rapid_periods = []
        current_period_length = 0
        
        for is_rapid in rapid_growth_periods:
            if is_rapid:
                current_period_length += 1
            else:
                if current_period_length > 0:
                    rapid_periods.append(current_period_length)
                current_period_length = 0
        
        if current_period_length > 0:
            rapid_periods.append(current_period_length)
        
        features['has_rapid_growth_period'] = 1 if rapid_periods else 0
        features['rapid_growth_duration'] = max(rapid_periods) if rapid_periods else 0
        features['rapid_growth_frequency'] = len(rapid_periods)
        
        return features
    
    def _analyze_market_entry_timing(self, establishment_year: int, 
                                    market_category: str) -> Dict:
        """市場参入タイミング分析"""
        
        features = {}
        
        # 時代区分による参入タイミング
        if establishment_year < 2000:
            timing_category = 'early_digital'
        elif establishment_year < 2010:
            timing_category = 'digital_transformation'
        elif establishment_year < 2020:
            timing_category = 'mobile_internet'
        else:
            timing_category = 'ai_iot_era'
        
        features['entry_timing_category'] = timing_category
        
        # 市場ライフサイクルでの参入タイミング推定
        market_maturity_at_entry = self._estimate_market_maturity_at_entry(
            establishment_year, market_category
        )
        features['market_maturity_at_entry'] = market_maturity_at_entry
        
        return features
    
    def _analyze_initial_market_position(self, data: pd.DataFrame) -> Dict:
        """初期市場ポジション分析"""
        
        features = {}
        
        # 初期3年間のデータ
        initial_data = data.head(3)
        
        # 初期規模
        if '売上高' in initial_data.columns:
            initial_revenue = initial_data['売上高'].mean()
            features['initial_revenue_scale'] = initial_revenue
        
        # 初期収益性
        if '売上高営業利益率' in initial_data.columns:
            initial_margin = initial_data['売上高営業利益率'].mean()
            features['initial_profit_margin'] = initial_margin
        
        # 初期効率性
        if 'ROE' in initial_data.columns:
            initial_roe = initial_data['ROE'].mean()
            features['initial_roe'] = initial_roe
        
        return features
    
    def _analyze_differentiation_strategy(self, value_added_ratio: pd.Series) -> Dict:
        """差別化戦略分析"""
        
        features = {}
        
        clean_ratio = value_added_ratio.dropna()
        if len(clean_ratio) < 2:
            return {'differentiation_strategy_strength': 0.0}
        
        # 付加価値率の水準と変化
        avg_value_added = clean_ratio.mean()
        value_added_trend = self._calculate_recent_trend(clean_ratio)
        
        # 差別化戦略強度（付加価値率の水準と向上トレンド）
        strategy_strength = min(1.0, avg_value_added + max(0, value_added_trend) * 2)
        
        features['differentiation_strategy_strength'] = strategy_strength
        features['average_value_added_ratio'] = avg_value_added
        features['value_added_improvement_trend'] = value_added_trend
        
        return features
    
    def _analyze_growth_trajectory(self, data: pd.DataFrame) -> Dict:
        """成長軌道分析"""
        
        features = {}
        
        if '売上高' not in data.columns or len(data) < 3:
            return features
        
        revenue_data = data['売上高'].dropna()
        
        # 成長軌道のパターン分析
        growth_pattern = self._classify_growth_pattern(revenue_data)
        features['growth_pattern'] = growth_pattern
        
        # 成長の持続性
        growth_sustainability = self._calculate_growth_sustainability(revenue_data)
        features['growth_sustainability'] = growth_sustainability
        
        # 成長の加速度
        if len(revenue_data) >= 3:
            growth_acceleration = self._calculate_growth_acceleration_trajectory(revenue_data)
            features['growth_acceleration'] = growth_acceleration
        
        return features
    
    def _calculate_sustainability_indicators(self, data: pd.DataFrame) -> Dict:
        """持続可能性指標計算"""
        
        features = {}
        
        # 財務持続可能性
        financial_sustainability = self._calculate_financial_sustainability(data)
        features['financial_sustainability'] = financial_sustainability
        
        # 成長持続可能性
        if '売上高成長率' in data.columns:
            growth_sustainability = self._calculate_growth_sustainability_score(
                data['売上高成長率']
            )
            features['growth_sustainability_score'] = growth_sustainability
        
        # 収益性持続可能性
        if '売上高営業利益率' in data.columns:
            profitability_sustainability = self._calculate_profitability_sustainability(
                data['売上高営業利益率']
            )
            features['profitability_sustainability'] = profitability_sustainability
        
        return features
    
    def _calculate_competitiveness_score(self, data: pd.DataFrame) -> float:
        """競争力スコア計算"""
        
        competitiveness_factors = []
        
        # 収益性競争力
        if '売上高営業利益率' in data.columns:
            profit_competitiveness = max(0, data['売上高営業利益率'].mean())
            competitiveness_factors.append(min(1.0, profit_competitiveness * 5))
        
        # 成長競争力
        if '売上高成長率' in data.columns:
            growth_competitiveness = max(0, data['売上高成長率'].mean())
            competitiveness_factors.append(min(1.0, growth_competitiveness * 2))
        
        # 効率性競争力
        if 'ROE' in data.columns:
            efficiency_competitiveness = max(0, data['ROE'].mean())
            competitiveness_factors.append(min(1.0, efficiency_competitiveness * 2))
        
        # 付加価値競争力
        if '売上高付加価値率' in data.columns:
            value_competitiveness = max(0, data['売上高付加価値率'].mean())
            competitiveness_factors.append(min(1.0, value_competitiveness))
        
        if competitiveness_factors:
            return sum(competitiveness_factors) / len(competitiveness_factors)
        else:
            return 0.5
    
    def _estimate_success_probability(self, features: Dict) -> float:
        """成功確率推定"""
        
        # 重み付き要因による成功確率計算
        success_factors = []
        
        # 成長要因
        if 'growth_sustainability' in features:
            success_factors.append(features['growth_sustainability'] * 0.3)
        
        # 競争力要因
        if 'competitiveness_score' in features:
            success_factors.append(features['competitiveness_score'] * 0.3)
        
        # 財務持続可能性要因
        if 'financial_sustainability' in features:
            success_factors.append(features['financial_sustainability'] * 0.2)
        
        # 差別化戦略要因
        if 'differentiation_strategy_strength' in features:
            success_factors.append(features['differentiation_strategy_strength'] * 0.2)
        
        if success_factors:
            return sum(success_factors)
        else:
            return 0.5
    
    def _classify_establishment_era(self, establishment_year: int) -> int:
        """設立時代の数値分類"""
        if establishment_year < 1990:
            return 1  # バブル期
        elif establishment_year < 2000:
            return 2  # バブル崩壊期
        elif establishment_year < 2010:
            return 3  # デジタル変革期
        elif establishment_year < 2020:
            return 4  # モバイル・インターネット期
        else:
            return 5  # AI・IoT期
    
    def _estimate_market_maturity_at_entry(self, establishment_year: int, 
                                            market_category: str) -> float:
        """参入時の市場成熟度推定"""
        
        # 市場カテゴリ別の成熟度推定（簡易版）
        if 'high_share' in market_category:
            # 高シェア市場：比較的成熟した市場への参入
            return 0.7
        elif 'declining' in market_category:
            # 低下市場：成熟期の市場への参入
            return 0.8
        elif 'lost' in market_category:
            # 失失市場：衰退期の市場での新規参入
            return 0.9
        else:
            return 0.5
    
    def _classify_growth_pattern(self, revenue_data: pd.Series) -> str:
        """成長パターン分類"""
        
        if len(revenue_data) < 3:
            return 'insufficient_data'
        
        # 成長率計算
        growth_rates = []
        for i in range(1, len(revenue_data)):
            if revenue_data.iloc[i-1] > 0:
                growth_rate = (revenue_data.iloc[i] - revenue_data.iloc[i-1]) / revenue_data.iloc[i-1]
                growth_rates.append(growth_rate)
        
        if not growth_rates:
            return 'no_growth_data'
        
        growth_rates = np.array(growth_rates)
        
        # パターン分類
        avg_growth = np.mean(growth_rates)
        growth_trend = np.polyfit(range(len(growth_rates)), growth_rates, 1)[0]
        
        if avg_growth > 0.2:
            if growth_trend > 0:
                return 'accelerating_high_growth'
            else:
                return 'decelerating_high_growth'
        elif avg_growth > 0.05:
            if growth_trend > 0:
                return 'accelerating_moderate_growth'
            else:
                return 'stable_moderate_growth'
        else:
            return 'low_or_negative_growth'
    
    def _calculate_growth_sustainability(self, revenue_data: pd.Series) -> float:
        """成長持続可能性計算"""
        
        if len(revenue_data) < 3:
            return 0.5
        
        # 成長率の変動係数（小さいほど持続可能）
        growth_rates = []
        for i in range(1, len(revenue_data)):
            if revenue_data.iloc[i-1] > 0:
                growth_rate = (revenue_data.iloc[i] - revenue_data.iloc[i-1]) / revenue_data.iloc[i-1]
                growth_rates.append(growth_rate)
        
        if not growth_rates:
            return 0.5
        
        growth_rates = np.array(growth_rates)
        
        # 持続可能性スコア
        avg_growth = np.mean(growth_rates)
        growth_volatility = np.std(growth_rates)
        
        if avg_growth > 0:
            sustainability = avg_growth / (1 + growth_volatility)
        else:
            sustainability = 0
        
        return min(1.0, max(0.0, sustainability))
    
    def _calculate_growth_acceleration_trajectory(self, revenue_data: pd.Series) -> float:
        """成長加速度軌道計算"""
        
        growth_rates = []
        for i in range(1, len(revenue_data)):
            if revenue_data.iloc[i-1] > 0:
                growth_rate = (revenue_data.iloc[i] - revenue_data.iloc[i-1]) / revenue_data.iloc[i-1]
                growth_rates.append(growth_rate)
        
        if len(growth_rates) < 2:
            return 0.0
        
        # 成長率の成長率（加速度）
        growth_acceleration = np.diff(growth_rates)
        return np.mean(growth_acceleration)
    
    def _calculate_financial_sustainability(self, data: pd.DataFrame) -> float:
        """財務持続可能性計算"""
        
        sustainability_scores = []
        
        # 収益性の持続性
        if '売上高営業利益率' in data.columns:
            profit_sustainability = self._calculate_metric_sustainability(
                data['売上高営業利益率']
            )
            sustainability_scores.append(profit_sustainability)
        
        # 効率性の持続性
        if 'ROE' in data.columns:
            roe_sustainability = self._calculate_metric_sustainability(data['ROE'])
            sustainability_scores.append(roe_sustainability)
        
        if sustainability_scores:
            return sum(sustainability_scores) / len(sustainability_scores)
        else:
            return 0.5
    
    def _calculate_growth_sustainability_score(self, growth_series: pd.Series) -> float:
        """成長持続可能性スコア"""
        
        clean_series = growth_series.dropna()
        if len(clean_series) < 2:
            return 0.5
        
        # 正の成長期間の比率
        positive_growth_ratio = (clean_series > 0).sum() / len(clean_series)
        
        # 成長率の安定性
        growth_stability = 1 / (1 + clean_series.std()) if clean_series.std() > 0 else 1
        
        return (positive_growth_ratio + growth_stability) / 2
    
    def _calculate_profitability_sustainability(self, profit_series: pd.Series) -> float:
        """収益性持続可能性計算"""
        
        return self._calculate_metric_sustainability(profit_series)
    
    def _calculate_metric_sustainability(self, series: pd.Series) -> float:
        """指標持続可能性計算（汎用）"""
        
        clean_series = series.dropna()
        if len(clean_series) < 2:
            return 0.5
        
        # 正の値の比率
        positive_ratio = (clean_series > 0).sum() / len(clean_series)
        
        # 安定性（変動係数の逆数）
        if clean_series.std() > 0 and clean_series.mean() != 0:
            cv = clean_series.std() / abs(clean_series.mean())
            stability = 1 / (1 + cv)
        else:
            stability = 1.0
        
        return (positive_ratio + stability) / 2
    
    def _get_default_emergence_features(self) -> Dict:
        """デフォルトの新設企業特徴量"""
        return {
            'initial_売上高_growth_rate': 0.0,
            'initial_売上高成長率_growth_rate': 0.0,
            'initial_総資産_growth_rate': 0.0,
            'initial_profitability_average': 0.0,
            'initial_financial_stability': 0.5,
        }


# ユーティリティ関数

def load_company_data(data_path: str) -> Dict[str, pd.DataFrame]:
    """
    企業データの読み込み
    
    Args:
        data_path: データディレクトリパス
        
    Returns:
        企業ID別データ辞書
    """
    # 実装は別途データ収集モジュールと連携
    pass

def load_company_metadata(metadata_path: str) -> pd.DataFrame:
    """
    企業メタデータの読み込み
    
    Args:
        metadata_path: メタデータファイルパス
        
    Returns:
        企業メタデータ
    """
    # 実装は別途データ収集モジュールと連携
    pass

def main():
    """メイン実行関数（テスト用）"""
    
    # 時系列特徴量生成器の初期化
    feature_generator = TimeSeriesFeatureGenerator(
        reference_period=40,
        base_year=1984
    )
    
    # サンプルデータでのテスト実行
    print("A2AI Time Series Features Generator initialized successfully")
    print(f"Reference period: {feature_generator.reference_period} years")
    print(f"Base year: {feature_generator.base_year}")
    
    # 特徴量計算クラスの確認
    calculators = [
        'trend_calculator',
        'volatility_calculator', 
        'growth_calculator',
        'cycle_calculator',
        'lifecycle_calculator',
        'survival_calculator',
        'emergence_calculator'
    ]
    
    print("\nInitialized feature calculators:")
    for calc_name in calculators:
        if hasattr(feature_generator, calc_name):
            print(f"  ✓ {calc_name}")
        else:
            print(f"  ✗ {calc_name}")

if __name__ == "__main__":
    main()