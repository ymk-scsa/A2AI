"""
A2AI Advanced Financial Analysis AI
外れ値検出モジュール (outlier_detector.py)

企業ライフサイクル全体（生存・消滅・新設）を考慮した高度な外れ値検出システム
150社×40年分の財務データに対応した包括的異常値検出
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import warnings
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import DBSCAN
from sklearn.covariance import EllipticEnvelope
import logging

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MarketCategory(Enum):
    """市場カテゴリー定義"""
    HIGH_SHARE = "high_share"  # 高シェア維持市場
    DECLINING = "declining"    # シェア低下中市場
    LOST = "lost"             # 完全失失市場

class LifecycleStage(Enum):
    """企業ライフサイクル段階"""
    EMERGING = "emerging"      # 新設・成長期
    MATURE = "mature"         # 成熟期
    DECLINING = "declining"   # 衰退期
    EXTINCTION = "extinction" # 消滅直前期
    RESTRUCTURING = "restructuring"  # 再編期

class OutlierType(Enum):
    """外れ値タイプ分類"""
    STATISTICAL = "statistical"        # 統計的外れ値
    CONTEXTUAL = "contextual"          # 文脈的異常値
    COLLECTIVE = "collective"          # 集合的異常値
    LIFECYCLE = "lifecycle"            # ライフサイクル関連
    ACCOUNTING_CHANGE = "accounting"   # 会計基準変更
    BUSINESS_EVENT = "business_event"  # 事業イベント関連

@dataclass
class OutlierResult:
    """外れ値検出結果"""
    company_id: str
    year: int
    metric_name: str
    value: float
    outlier_type: OutlierType
    severity: float  # 0-1, 1が最も重篤
    confidence: float  # 0-1, 1が最も確信度高い
    context: Dict[str, any]
    action_required: str  # "remove", "flag", "investigate", "transform"

class A2AIOutlierDetector:
    """
    A2AI専用外れ値検出システム
    
    特徴:
    - 企業ライフサイクル段階を考慮した検出
    - 市場カテゴリー別の正常範囲定義
    - 40年間の時系列トレンド考慮
    - 複数検出手法の組み合わせ
    - 事業イベント文脈での判定
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初期化
        
        Args:
            config: 設定辞書 {
                'statistical_methods': ['iqr', 'z_score', 'modified_z_score'],
                'ml_methods': ['isolation_forest', 'elliptic_envelope'],
                'time_series_methods': ['seasonal_decompose', 'change_point'],
                'thresholds': {'z_score': 3.0, 'iqr_factor': 1.5},
                'min_samples_per_group': 10
            }
        """
        default_config = {
            'statistical_methods': ['iqr', 'z_score', 'modified_z_score'],
            'ml_methods': ['isolation_forest', 'elliptic_envelope', 'dbscan'],
            'time_series_methods': ['rolling_stats', 'change_point_detection'],
            'thresholds': {
                'z_score': 3.0,
                'modified_z_score': 3.5,
                'iqr_factor': 1.5,
                'isolation_forest_contamination': 0.1,
                'elliptic_envelope_contamination': 0.1,
                'dbscan_eps': 0.5,
                'dbscan_min_samples': 5
            },
            'min_samples_per_group': 10,
            'lifecycle_adjustments': True,
            'market_category_normalization': True,
            'temporal_context_window': 5  # 前後5年の文脈を考慮
        }
        
        self.config = {**default_config, **(config or {})}
        
        # 市場カテゴリー別正常範囲定義
        self._initialize_market_category_ranges()
        
        # ライフサイクル段階別調整係数
        self._initialize_lifecycle_adjustments()
        
        # 会計基準変更年の定義
        self.accounting_change_years = {
            2010: "IFRS任意適用開始",
            2015: "IFRS強制適用範囲拡大", 
            2018: "収益認識基準変更",
            2021: "リース会計基準変更"
        }
        
        # 検出結果キャッシュ
        self.outlier_cache = {}
        
    def _initialize_market_category_ranges(self):
        """市場カテゴリー別正常範囲の初期化"""
        self.market_ranges = {
            MarketCategory.HIGH_SHARE: {
                'roe_range': (-0.05, 0.4),  # 高シェア企業は安定収益
                'growth_rate_range': (-0.1, 0.2),  # 安定成長
                'debt_ratio_range': (0.0, 0.6),    # 健全財務
                'rd_ratio_range': (0.0, 0.15)      # 高R&D投資
            },
            MarketCategory.DECLINING: {
                'roe_range': (-0.1, 0.3),
                'growth_rate_range': (-0.2, 0.15),
                'debt_ratio_range': (0.0, 0.7),
                'rd_ratio_range': (0.0, 0.1)
            },
            MarketCategory.LOST: {
                'roe_range': (-0.5, 0.2),  # 失失市場は赤字許容範囲広い
                'growth_rate_range': (-0.5, 0.1),  # マイナス成長も正常
                'debt_ratio_range': (0.0, 0.9),    # 財務悪化許容
                'rd_ratio_range': (0.0, 0.05)      # R&D削減傾向
            }
        }
    
    def _initialize_lifecycle_adjustments(self):
        """ライフサイクル段階別調整係数の初期化"""
        self.lifecycle_adjustments = {
            LifecycleStage.EMERGING: {
                'volatility_multiplier': 3.0,  # 新設企業は3倍の変動許容
                'growth_tolerance': 5.0,       # 高成長許容
                'loss_tolerance': 2.0          # 初期赤字許容
            },
            LifecycleStage.MATURE: {
                'volatility_multiplier': 1.0,  # 標準
                'growth_tolerance': 1.0,
                'loss_tolerance': 1.0
            },
            LifecycleStage.DECLINING: {
                'volatility_multiplier': 2.0,  # 衰退期は変動大
                'growth_tolerance': 0.5,       # マイナス成長が正常
                'loss_tolerance': 3.0          # 赤字許容度高い
            },
            LifecycleStage.EXTINCTION: {
                'volatility_multiplier': 5.0,  # 消滅前は大きく変動
                'growth_tolerance': 0.1,       # ほぼマイナス成長
                'loss_tolerance': 10.0         # 大幅赤字も正常
            },
            LifecycleStage.RESTRUCTURING: {
                'volatility_multiplier': 4.0,  # 再編期は特殊
                'growth_tolerance': 2.0,
                'loss_tolerance': 5.0
            }
        }
    
    def detect_outliers(
        self,
        df: pd.DataFrame,
        company_metadata: Optional[pd.DataFrame] = None,
        business_events: Optional[pd.DataFrame] = None
    ) -> List[OutlierResult]:
        """
        包括的外れ値検出の実行
        
        Args:
            df: 財務データ DataFrame
                必須列: ['company_id', 'year', 'metric_name', 'value']
            company_metadata: 企業メタデータ
                列: ['company_id', 'market_category', 'founded_year', 'extinction_year']
            business_events: 事業イベントデータ
                列: ['company_id', 'year', 'event_type', 'description']
        
        Returns:
            OutlierResult のリスト
        """
        logger.info("A2AI外れ値検出を開始します")
        
        # データ前処理
        df_processed = self._preprocess_data(df)
        
        # 企業ライフサイクル段階の特定
        lifecycle_data = self._identify_lifecycle_stages(df_processed, company_metadata)
        
        # 外れ値検出実行
        outliers = []
        
        # 1. 統計的外れ値検出
        statistical_outliers = self._detect_statistical_outliers(
            df_processed, lifecycle_data, company_metadata
        )
        outliers.extend(statistical_outliers)
        
        # 2. 機械学習ベース外れ値検出
        ml_outliers = self._detect_ml_outliers(
            df_processed, lifecycle_data, company_metadata
        )
        outliers.extend(ml_outliers)
        
        # 3. 時系列文脈外れ値検出
        temporal_outliers = self._detect_temporal_outliers(
            df_processed, lifecycle_data, company_metadata
        )
        outliers.extend(temporal_outliers)
        
        # 4. 事業イベント文脈外れ値検出
        if business_events is not None:
            event_outliers = self._detect_event_contextual_outliers(
                df_processed, business_events, lifecycle_data
            )
            outliers.extend(event_outliers)
        
        # 5. 会計基準変更影響検出
        accounting_outliers = self._detect_accounting_change_outliers(df_processed)
        outliers.extend(accounting_outliers)
        
        # 結果統合と重複除去
        consolidated_outliers = self._consolidate_outlier_results(outliers)
        
        logger.info(f"外れ値検出完了: {len(consolidated_outliers)}件の異常値を特定")
        
        return consolidated_outliers
    
    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """データ前処理"""
        df = df.copy()
        
        # 必須列の確認
        required_cols = ['company_id', 'year', 'metric_name', 'value']
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"必須列が不足しています: {missing_cols}")
        
        # 数値型変換
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        df['year'] = pd.to_numeric(df['year'], errors='coerce')
        
        # 無限値・NaNの除去
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=['value', 'year'])
        
        # 年代範囲の確認（1984-2024）
        df = df[(df['year'] >= 1984) & (df['year'] <= 2024)]
        
        return df
    
    def _identify_lifecycle_stages(
        self,
        df: pd.DataFrame,
        company_metadata: Optional[pd.DataFrame]
    ) -> pd.DataFrame:
        """企業ライフサイクル段階の特定"""
        lifecycle_data = []
        
        for company_id in df['company_id'].unique():
            company_data = df[df['company_id'] == company_id]
            years = sorted(company_data['year'].unique())
            
            # メタデータから設立年・消滅年を取得
            founded_year = None
            extinction_year = None
            if company_metadata is not None:
                meta = company_metadata[company_metadata['company_id'] == company_id]
                if not meta.empty:
                    founded_year = meta.iloc[0].get('founded_year')
                    extinction_year = meta.iloc[0].get('extinction_year')
            
            # データから推定（メタデータがない場合）
            if founded_year is None:
                founded_year = min(years) if years else 1984
            
            for year in years:
                # ライフサイクル段階の判定
                age = year - founded_year
                is_last_years = extinction_year and (year >= extinction_year - 3)
                
                if age <= 5:
                    stage = LifecycleStage.EMERGING
                elif is_last_years:
                    stage = LifecycleStage.EXTINCTION
                elif age > 30:
                    stage = LifecycleStage.MATURE
                else:
                    # 成長率・収益性から判定
                    company_metrics = company_data[company_data['year'] == year]
                    growth_metrics = company_metrics[
                        company_metrics['metric_name'].str.contains('growth|増加率', case=False)
                    ]
                    
                    if not growth_metrics.empty:
                        avg_growth = growth_metrics['value'].mean()
                        if avg_growth < -0.1:  # 10%以上のマイナス成長
                            stage = LifecycleStage.DECLINING
                        else:
                            stage = LifecycleStage.MATURE
                    else:
                        stage = LifecycleStage.MATURE
                
                lifecycle_data.append({
                    'company_id': company_id,
                    'year': year,
                    'lifecycle_stage': stage,
                    'company_age': age,
                    'founded_year': founded_year,
                    'extinction_year': extinction_year
                })
        
        return pd.DataFrame(lifecycle_data)
    
    def _detect_statistical_outliers(
        self,
        df: pd.DataFrame,
        lifecycle_data: pd.DataFrame,
        company_metadata: Optional[pd.DataFrame]
    ) -> List[OutlierResult]:
        """統計的手法による外れ値検出"""
        outliers = []
        
        for metric_name in df['metric_name'].unique():
            metric_data = df[df['metric_name'] == metric_name].copy()
            
            if len(metric_data) < self.config['min_samples_per_group']:
                continue
            
            # 市場カテゴリー別に検出
            if company_metadata is not None:
                metric_data = metric_data.merge(
                    company_metadata[['company_id', 'market_category']], 
                    on='company_id', 
                    how='left'
                )
                
                for category in metric_data['market_category'].dropna().unique():
                    category_data = metric_data[metric_data['market_category'] == category]
                    category_outliers = self._apply_statistical_methods(
                        category_data, metric_name, MarketCategory(category)
                    )
                    outliers.extend(category_outliers)
            else:
                # 全体で検出
                all_outliers = self._apply_statistical_methods(
                    metric_data, metric_name, None
                )
                outliers.extend(all_outliers)
        
        return outliers
    
    def _apply_statistical_methods(
        self,
        data: pd.DataFrame,
        metric_name: str,
        market_category: Optional[MarketCategory]
    ) -> List[OutlierResult]:
        """統計的手法の適用"""
        outliers = []
        values = data['value'].values
        
        # IQR法
        if 'iqr' in self.config['statistical_methods']:
            Q1 = np.percentile(values, 25)
            Q3 = np.percentile(values, 75)
            IQR = Q3 - Q1
            factor = self.config['thresholds']['iqr_factor']
            
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
            
            iqr_outliers = data[
                (data['value'] < lower_bound) | (data['value'] > upper_bound)
            ]
            
            for _, row in iqr_outliers.iterrows():
                severity = min(
                    abs(row['value'] - lower_bound) / IQR if row['value'] < lower_bound else 0,
                    abs(row['value'] - upper_bound) / IQR if row['value'] > upper_bound else 0
                ) / factor
                
                outliers.append(OutlierResult(
                    company_id=row['company_id'],
                    year=int(row['year']),
                    metric_name=metric_name,
                    value=row['value'],
                    outlier_type=OutlierType.STATISTICAL,
                    severity=min(severity, 1.0),
                    confidence=0.8,
                    context={'method': 'IQR', 'bounds': (lower_bound, upper_bound)},
                    action_required="investigate"
                ))
        
        # Z-Score法
        if 'z_score' in self.config['statistical_methods']:
            z_scores = np.abs(stats.zscore(values))
            threshold = self.config['thresholds']['z_score']
            
            z_outliers = data[z_scores > threshold]
            
            for idx, (_, row) in enumerate(z_outliers.iterrows()):
                z_score = z_scores[data.index.get_loc(row.name)]
                
                outliers.append(OutlierResult(
                    company_id=row['company_id'],
                    year=int(row['year']),
                    metric_name=metric_name,
                    value=row['value'],
                    outlier_type=OutlierType.STATISTICAL,
                    severity=min(z_score / threshold / 2, 1.0),
                    confidence=0.85,
                    context={'method': 'Z-Score', 'z_score': z_score},
                    action_required="flag" if z_score < threshold * 1.5 else "investigate"
                ))
        
        # Modified Z-Score法（中央値ベース）
        if 'modified_z_score' in self.config['statistical_methods']:
            median = np.median(values)
            mad = np.median(np.abs(values - median))  # Median Absolute Deviation
            
            if mad != 0:
                modified_z_scores = 0.6745 * (values - median) / mad
                threshold = self.config['thresholds']['modified_z_score']
                
                modified_outliers = data[np.abs(modified_z_scores) > threshold]
                
                for idx, (_, row) in enumerate(modified_outliers.iterrows()):
                    mod_z_score = abs(modified_z_scores[data.index.get_loc(row.name)])
                    
                    outliers.append(OutlierResult(
                        company_id=row['company_id'],
                        year=int(row['year']),
                        metric_name=metric_name,
                        value=row['value'],
                        outlier_type=OutlierType.STATISTICAL,
                        severity=min(mod_z_score / threshold / 2, 1.0),
                        confidence=0.9,  # Modified Z-Scoreは外れ値に対してロバスト
                        context={'method': 'Modified Z-Score', 'score': mod_z_score},
                        action_required="investigate" if mod_z_score > threshold * 1.2 else "flag"
                    ))
        
        return outliers
    
    def _detect_ml_outliers(
        self,
        df: pd.DataFrame,
        lifecycle_data: pd.DataFrame,
        company_metadata: Optional[pd.DataFrame]
    ) -> List[OutlierResult]:
        """機械学習ベース外れ値検出"""
        outliers = []
        
        # メトリック別にピボットテーブル作成
        pivot_data = df.pivot_table(
            index=['company_id', 'year'], 
            columns='metric_name', 
            values='value', 
            aggfunc='mean'
        ).fillna(method='ffill').fillna(method='bfill')
        
        if pivot_data.empty or len(pivot_data.columns) < 2:
            logger.warning("機械学習外れ値検出: データ不足のためスキップ")
            return outliers
        
        # 特徴量準備
        features = pivot_data.select_dtypes(include=[np.number]).values
        
        # 欠損値処理
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='median')
        features = imputer.fit_transform(features)
        
        # 標準化
        scaler = RobustScaler()  # 外れ値に強いスケーラー
        features_scaled = scaler.fit_transform(features)
        
        # Isolation Forest
        if 'isolation_forest' in self.config['ml_methods']:
            iso_forest = IsolationForest(
                contamination=self.config['thresholds']['isolation_forest_contamination'],
                random_state=42,
                n_estimators=200
            )
            
            iso_predictions = iso_forest.fit_predict(features_scaled)
            iso_scores = iso_forest.decision_function(features_scaled)
            
            # 外れ値の結果作成
            for idx, (prediction, score) in enumerate(zip(iso_predictions, iso_scores)):
                if prediction == -1:  # 外れ値
                    company_id, year = pivot_data.index[idx]
                    
                    # 最も異常な指標を特定
                    row_features = features_scaled[idx]
                    feature_importance = np.abs(row_features)
                    most_anomalous_idx = np.argmax(feature_importance)
                    most_anomalous_metric = pivot_data.columns[most_anomalous_idx]
                    
                    outliers.append(OutlierResult(
                        company_id=company_id,
                        year=int(year),
                        metric_name=most_anomalous_metric,
                        value=pivot_data.iloc[idx][most_anomalous_metric],
                        outlier_type=OutlierType.COLLECTIVE,
                        severity=min(abs(score) * 2, 1.0),  # scoreは負値なので絶対値
                        confidence=0.75,
                        context={
                            'method': 'Isolation Forest', 
                            'anomaly_score': score,
                            'feature_importance': feature_importance[most_anomalous_idx]
                        },
                        action_required="investigate"
                    ))
        
        # Elliptic Envelope (多変量ガウス分布ベース)
        if 'elliptic_envelope' in self.config['ml_methods']:
            try:
                elliptic = EllipticEnvelope(
                    contamination=self.config['thresholds']['elliptic_envelope_contamination'],
                    random_state=42
                )
                
                elliptic_predictions = elliptic.fit_predict(features_scaled)
                
                for idx, prediction in enumerate(elliptic_predictions):
                    if prediction == -1:  # 外れ値
                        company_id, year = pivot_data.index[idx]
                        
                        # マハラノビス距離的な異常度計算
                        decision_score = elliptic.decision_function([features_scaled[idx]])[0]
                        
                        outliers.append(OutlierResult(
                            company_id=company_id,
                            year=int(year),
                            metric_name="multivariate_anomaly",  # 多変量異常
                            value=decision_score,
                            outlier_type=OutlierType.COLLECTIVE,
                            severity=min(abs(decision_score) * 0.1, 1.0),
                            confidence=0.8,
                            context={'method': 'Elliptic Envelope', 'mahalanobis_like': decision_score},
                            action_required="investigate"
                        ))
                        
            except Exception as e:
                logger.warning(f"Elliptic Envelope 検出でエラー: {e}")
        
        # DBSCAN クラスタリングベース
        if 'dbscan' in self.config['ml_methods']:
            try:
                dbscan = DBSCAN(
                    eps=self.config['thresholds']['dbscan_eps'],
                    min_samples=self.config['thresholds']['dbscan_min_samples']
                )
                
                cluster_labels = dbscan.fit_predict(features_scaled)
                
                # ノイズポイント（cluster = -1）を外れ値とする
                noise_indices = np.where(cluster_labels == -1)[0]
                
                for idx in noise_indices:
                    company_id, year = pivot_data.index[idx]
                    
                    outliers.append(OutlierResult(
                        company_id=company_id,
                        year=int(year),
                        metric_name="cluster_anomaly",
                        value=0.0,  # クラスタリング異常では数値意味なし
                        outlier_type=OutlierType.COLLECTIVE,
                        severity=0.7,  # 中程度の重篤度
                        confidence=0.7,
                        context={'method': 'DBSCAN', 'cluster_label': -1},
                        action_required="flag"
                    ))
                    
            except Exception as e:
                logger.warning(f"DBSCAN 外れ値検出でエラー: {e}")
        
        return outliers
    
    def _detect_temporal_outliers(
        self,
        df: pd.DataFrame,
        lifecycle_data: pd.DataFrame,
        company_metadata: Optional[pd.DataFrame]
    ) -> List[OutlierResult]:
        """時系列文脈での外れ値検出"""
        outliers = []
        
        for company_id in df['company_id'].unique():
            company_data = df[df['company_id'] == company_id].copy()
            
            for metric_name in company_data['metric_name'].unique():
                metric_series = company_data[company_data['metric_name'] == metric_name].copy()
                metric_series = metric_series.sort_values('year')
                
                if len(metric_series) < 3:  # 最低3年分必要
                    continue
                
                # 移動統計による検出
                if 'rolling_stats' in self.config['time_series_methods']:
                    rolling_outliers = self._detect_rolling_outliers(
                        metric_series, company_id, metric_name
                    )
                    outliers.extend(rolling_outliers)
                
                # 変化点検出
                if 'change_point_detection' in self.config['time_series_methods']:
                    change_point_outliers = self._detect_change_points(
                        metric_series, company_id, metric_name, lifecycle_data
                    )
                    outliers.extend(change_point_outliers)
        
        return outliers
    
    def _detect_rolling_outliers(
        self,
        metric_series: pd.DataFrame,
        company_id: str,
        metric_name: str
    ) -> List[OutlierResult]:
        """移動統計による異常値検出"""
        outliers = []
        window = self.config['temporal_context_window']
        
        if len(metric_series) < window:
            return outliers
        
        # 移動平均・移動標準偏差計算
        metric_series = metric_series.set_index('year')['value']
        rolling_mean = metric_series.rolling(window=window, center=True).mean()
        rolling_std = metric_series.rolling(window=window, center=True).std()
        
        # 移動Z-score計算
        rolling_z_scores = (metric_series - rolling_mean) / rolling_std
        threshold = 2.5  # 時系列は少し緩い閾値
        
        anomalous_years = rolling_z_scores[rolling_z_scores.abs() > threshold].index
        
        for year in anomalous_years:
            if pd.notna(rolling_z_scores[year]):
                outliers.append(OutlierResult(
                    company_id=company_id,
                    year=int(year),
                    metric_name=metric_name,
                    value=metric_series[year],
                    outlier_type=OutlierType.CONTEXTUAL,
                    severity=min(abs(rolling_z_scores[year]) / threshold / 2, 1.0),
                    confidence=0.8,
                    context={
                        'method': 'Rolling Statistics',
                        'rolling_z_score': rolling_z_scores[year],
                        'rolling_mean': rolling_mean[year],
                        'rolling_std': rolling_std[year]
                    },
                    action_required="investigate"
                ))
        
        return outliers
    
    def _detect_change_points(
        self,
        metric_series: pd.DataFrame,
        company_id: str,
        metric_name: str,
        lifecycle_data: pd.DataFrame
    ) -> List[OutlierResult]:
        """変化点検出による構造変化の特定"""
        outliers = []
        
        # 単純な変化率ベース変化点検出
        values = metric_series.sort_values('year')['value'].values
        years = metric_series.sort_values('year')['year'].values
        
        if len(values) < 5:
            return outliers
        
        # 前年比変化率計算
        change_rates = np.diff(values) / (np.abs(values[:-1]) + 1e-6)  # ゼロ除算防止
        
        # 変化率の標準偏差
        change_std = np.std(change_rates)
        if change_std == 0:
            return outliers
        
        # 異常な変化率の検出（3σ以上）
        threshold = 3.0
        anomalous_changes = np.where(np.abs(change_rates) > threshold * change_std)[0]
        
        for idx in anomalous_changes:
            year = years[idx + 1]  # 変化後の年
            change_rate = change_rates[idx]
            
            # ライフサイクル文脈での調整
            lifecycle_info = lifecycle_data[
                (lifecycle_data['company_id'] == company_id) & 
                (lifecycle_data['year'] == year)
            ]
            
            # 調整係数適用
            severity_adjustment = 1.0
            if not lifecycle_info.empty:
                stage = lifecycle_info.iloc[0]['lifecycle_stage']
                if stage in self.lifecycle_adjustments:
                    volatility_mult = self.lifecycle_adjustments[stage]['volatility_multiplier']
                    severity_adjustment = 1.0 / volatility_mult  # 許容変動が大きいほど重篤度下げる
            
            severity = min(abs(change_rate) / (threshold * change_std) * severity_adjustment, 1.0)
            
            # 重篤度に応じてアクション決定
            if severity > 0.8:
                action = "remove"
            elif severity > 0.5:
                action = "investigate"
            else:
                action = "flag"
            
            outliers.append(OutlierResult(
                company_id=company_id,
                year=int(year),
                metric_name=metric_name,
                value=values[idx + 1],
                outlier_type=OutlierType.CONTEXTUAL,
                severity=severity,
                confidence=0.7,
                context={
                    'method': 'Change Point Detection',
                    'change_rate': change_rate,
                    'previous_value': values[idx],
                    'lifecycle_adjustment': severity_adjustment
                },
                action_required=action
            ))
        
        return outliers
    
    def _detect_event_contextual_outliers(
        self,
        df: pd.DataFrame,
        business_events: pd.DataFrame,
        lifecycle_data: pd.DataFrame
    ) -> List[OutlierResult]:
        """事業イベント文脈での外れ値検出"""
        outliers = []
        
        # 事業イベント前後の異常値を検出
        for _, event in business_events.iterrows():
            company_id = event['company_id']
            event_year = event['year']
            event_type = event['event_type']
            
            # イベント前後3年のデータを取得
            company_data = df[df['company_id'] == company_id].copy()
            event_window_data = company_data[
                (company_data['year'] >= event_year - 3) & 
                (company_data['year'] <= event_year + 3)
            ]
            
            if event_window_data.empty:
                continue
            
            # イベントタイプ別の期待される変動パターン
            expected_patterns = self._get_expected_event_patterns(event_type)
            
            for metric_name in event_window_data['metric_name'].unique():
                metric_data = event_window_data[event_window_data['metric_name'] == metric_name]
                
                if len(metric_data) < 3:  # 最低3年分必要
                    continue
                
                # イベント年の値と期待パターンの比較
                event_year_data = metric_data[metric_data['year'] == event_year]
                if event_year_data.empty:
                    continue
                
                event_value = event_year_data.iloc[0]['value']
                
                # ベースライン（イベント前平均）計算
                pre_event_data = metric_data[metric_data['year'] < event_year]
                if len(pre_event_data) < 2:
                    continue
                
                baseline = pre_event_data['value'].mean()
                baseline_std = pre_event_data['value'].std()
                
                if baseline_std == 0:
                    continue
                
                # 期待される変化とのズレを計算
                expected_change = expected_patterns.get(metric_name, 0.0)  # 期待変化率
                expected_value = baseline * (1 + expected_change)
                
                # 実際の値との乖離度
                deviation = abs(event_value - expected_value) / (baseline_std + 1e-6)
                
                if deviation > 2.0:  # 2σ以上の乖離で異常判定
                    severity = min(deviation / 5.0, 1.0)  # 5σで最大重篤度
                    
                    outliers.append(OutlierResult(
                        company_id=company_id,
                        year=int(event_year),
                        metric_name=metric_name,
                        value=event_value,
                        outlier_type=OutlierType.BUSINESS_EVENT,
                        severity=severity,
                        confidence=0.85,
                        context={
                            'method': 'Business Event Context',
                            'event_type': event_type,
                            'expected_value': expected_value,
                            'baseline': baseline,
                            'deviation_sigma': deviation
                        },
                        action_required="investigate" if severity > 0.6 else "flag"
                    ))
        
        return outliers
    
    def _get_expected_event_patterns(self, event_type: str) -> Dict[str, float]:
        """事業イベントタイプ別の期待変動パターン"""
        patterns = {
            'merger': {  # 合併
                'total_assets': 0.5,  # 50%増加期待
                'sales': 0.3,         # 30%増加期待
                'employees': 0.4,     # 40%増加期待
                'rd_expense': 0.2     # 20%増加期待
            },
            'acquisition': {  # 買収
                'total_assets': 0.3,
                'sales': 0.25,
                'debt_ratio': 0.1     # 負債比率上昇
            },
            'divestiture': {  # 売却・撤退
                'total_assets': -0.2,  # 20%減少期待
                'sales': -0.3,         # 30%減少期待
                'employees': -0.25     # 25%減少期待
            },
            'spinoff': {  # 分社化
                'total_assets': -0.15,
                'sales': -0.2,
                'employees': -0.3
            },
            'bankruptcy': {  # 倒産
                'equity_ratio': -0.5,  # 自己資本比率大幅悪化
                'current_ratio': -0.4, # 流動比率悪化
                'interest_coverage': -0.8  # 利払い能力悪化
            },
            'ipo': {  # 新規上場
                'equity_ratio': 0.3,   # 自己資本比率改善
                'cash_ratio': 0.5      # 現金比率向上
            },
            'delisting': {  # 上場廃止
                'market_value': -0.6,  # 時価総額大幅減
                'trading_volume': -0.8 # 売買高減少
            }
        }
        
        return patterns.get(event_type, {})
    
    def _detect_accounting_change_outliers(self, df: pd.DataFrame) -> List[OutlierResult]:
        """会計基準変更による外れ値検出"""
        outliers = []
        
        for change_year, description in self.accounting_change_years.items():
            # 変更年前後のデータを比較
            year_before = change_year - 1
            year_after = change_year
            
            for company_id in df['company_id'].unique():
                company_data = df[df['company_id'] == company_id]
                
                before_data = company_data[company_data['year'] == year_before]
                after_data = company_data[company_data['year'] == year_after]
                
                if before_data.empty or after_data.empty:
                    continue
                
                # 同一メトリックでの比較
                common_metrics = set(before_data['metric_name']) & set(after_data['metric_name'])
                
                for metric_name in common_metrics:
                    before_value = before_data[before_data['metric_name'] == metric_name]['value'].iloc[0]
                    after_value = after_data[after_data['metric_name'] == metric_name]['value'].iloc[0]
                    
                    # 極端な変化（50%以上）を会計基準変更影響として検出
                    if abs(before_value) > 1e-6:  # ゼロ除算防止
                        change_rate = abs((after_value - before_value) / before_value)
                        
                        if change_rate > 0.5:  # 50%以上の変化
                            severity = min(change_rate, 1.0)
                            
                            outliers.append(OutlierResult(
                                company_id=company_id,
                                year=change_year,
                                metric_name=metric_name,
                                value=after_value,
                                outlier_type=OutlierType.ACCOUNTING_CHANGE,
                                severity=severity,
                                confidence=0.9,  # 会計基準変更は確実
                                context={
                                    'method': 'Accounting Standards Change',
                                    'change_description': description,
                                    'before_value': before_value,
                                    'change_rate': change_rate
                                },
                                action_required="transform"  # 調整処理が必要
                            ))
        
        return outliers
    
    def _consolidate_outlier_results(self, outliers: List[OutlierResult]) -> List[OutlierResult]:
        """外れ値結果の統合と重複除去"""
        # 同一企業・年・メトリックの重複を統合
        consolidation_dict = {}
        
        for outlier in outliers:
            key = (outlier.company_id, outlier.year, outlier.metric_name)
            
            if key in consolidation_dict:
                # 既存結果と統合
                existing = consolidation_dict[key]
                
                # より高い重篤度を採用
                if outlier.severity > existing.severity:
                    # コンテキスト情報を統合
                    combined_context = {**existing.context, **outlier.context}
                    combined_context['detection_methods'] = [
                        existing.context.get('method', 'unknown'),
                        outlier.context.get('method', 'unknown')
                    ]
                    
                    consolidated = OutlierResult(
                        company_id=outlier.company_id,
                        year=outlier.year,
                        metric_name=outlier.metric_name,
                        value=outlier.value,
                        outlier_type=outlier.outlier_type,
                        severity=outlier.severity,
                        confidence=max(existing.confidence, outlier.confidence),
                        context=combined_context,
                        action_required=outlier.action_required
                    )
                    
                    consolidation_dict[key] = consolidated
            else:
                consolidation_dict[key] = outlier
        
        # 重篤度でソート
        consolidated_outliers = list(consolidation_dict.values())
        consolidated_outliers.sort(key=lambda x: x.severity, reverse=True)
        
        return consolidated_outliers
    
    def generate_outlier_report(
        self, 
        outliers: List[OutlierResult],
        output_path: Optional[str] = None
    ) -> Dict:
        """外れ値検出結果レポートの生成"""
        report = {
            'summary': {
                'total_outliers': len(outliers),
                'by_type': {},
                'by_severity': {'high': 0, 'medium': 0, 'low': 0},
                'by_action': {}
            },
            'details': [],
            'recommendations': []
        }
        
        # 統計サマリー
        for outlier in outliers:
            # タイプ別集計
            type_name = outlier.outlier_type.value
            report['summary']['by_type'][type_name] = report['summary']['by_type'].get(type_name, 0) + 1
            
            # 重篤度別集計
            if outlier.severity > 0.7:
                report['summary']['by_severity']['high'] += 1
            elif outlier.severity > 0.3:
                report['summary']['by_severity']['medium'] += 1
            else:
                report['summary']['by_severity']['low'] += 1
            
            # アクション別集計
            action = outlier.action_required
            report['summary']['by_action'][action] = report['summary']['by_action'].get(action, 0) + 1
            
            # 詳細情報
            report['details'].append({
                'company_id': outlier.company_id,
                'year': outlier.year,
                'metric_name': outlier.metric_name,
                'value': outlier.value,
                'outlier_type': type_name,
                'severity': round(outlier.severity, 3),
                'confidence': round(outlier.confidence, 3),
                'action_required': outlier.action_required,
                'context': outlier.context
            })
        
        # 推奨事項生成
        report['recommendations'] = self._generate_recommendations(outliers)
        
        # ファイル出力
        if output_path:
            import json
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2, default=str)
            
            logger.info(f"外れ値検出レポートを出力しました: {output_path}")
        
        return report
    
    def _generate_recommendations(self, outliers: List[OutlierResult]) -> List[str]:
        """推奨事項の生成"""
        recommendations = []
        
        # 高重篤度の外れ値に対する推奨
        high_severity = [o for o in outliers if o.severity > 0.8]
        if high_severity:
            recommendations.append(
                f"高重篤度の外れ値が{len(high_severity)}件検出されました。"
                "データの妥当性を詳細調査し、必要に応じて除外を検討してください。"
            )
        
        # 会計基準変更影響の推奨
        accounting_outliers = [o for o in outliers if o.outlier_type == OutlierType.ACCOUNTING_CHANGE]
        if accounting_outliers:
            recommendations.append(
                f"会計基準変更による影響が{len(accounting_outliers)}件検出されました。"
                "遡及修正またはプロフォーマ調整を行って時系列の連続性を確保してください。"
            )
        
        # 事業イベント関連の推奨
        business_event_outliers = [o for o in outliers if o.outlier_type == OutlierType.BUSINESS_EVENT]
        if business_event_outliers:
            recommendations.append(
                f"事業イベント関連の異常値が{len(business_event_outliers)}件検出されました。"
                "イベントの性質を考慮した分析手法の採用を検討してください。"
            )
        
        # 企業消滅関連の推奨
        extinction_companies = set([
            o.company_id for o in outliers 
            if any('extinction' in str(o.context).lower() for o in outliers if o.company_id == o.company_id)
        ])
        if extinction_companies:
            recommendations.append(
                f"消滅企業{len(extinction_companies)}社に関連する異常値が検出されました。"
                "生存バイアス回避のため、これらのデータを分析に含める価値を検討してください。"
            )
        
        return recommendations
    
    def apply_outlier_treatments(
        self,
        df: pd.DataFrame,
        outliers: List[OutlierResult],
        treatment_strategy: str = "conservative"
    ) -> pd.DataFrame:
        """
        検出された外れ値への処理適用
        
        Args:
            df: 元データ
            outliers: 検出された外れ値リスト
            treatment_strategy: 処理戦略
                - "conservative": 高重篤度のみ処理
                - "aggressive": 中重篤度以上を処理
                - "custom": action_required に従って処理
        
        Returns:
            処理済みデータフレーム
        """
        df_treated = df.copy()
        treatment_log = []
        
        for outlier in outliers:
            should_treat = False
            treatment_method = None
            
            # 処理戦略に基づく判定
            if treatment_strategy == "conservative" and outlier.severity > 0.8:
                should_treat = True
                treatment_method = "remove" if outlier.action_required == "remove" else "flag"
            elif treatment_strategy == "aggressive" and outlier.severity > 0.5:
                should_treat = True
                treatment_method = outlier.action_required
            elif treatment_strategy == "custom":
                should_treat = True
                treatment_method = outlier.action_required
            
            if not should_treat:
                continue
            
            # データ特定
            mask = (
                (df_treated['company_id'] == outlier.company_id) &
                (df_treated['year'] == outlier.year) &
                (df_treated['metric_name'] == outlier.metric_name)
            )
            
            if not mask.any():
                continue
            
            # 処理実行
            if treatment_method == "remove":
                df_treated = df_treated[~mask]
                treatment_log.append(f"Removed: {outlier.company_id}, {outlier.year}, {outlier.metric_name}")
                
            elif treatment_method == "flag":
                df_treated.loc[mask, 'outlier_flag'] = True
                df_treated.loc[mask, 'outlier_severity'] = outlier.severity
                treatment_log.append(f"Flagged: {outlier.company_id}, {outlier.year}, {outlier.metric_name}")
                
            elif treatment_method == "transform":
                # 会計基準変更等の場合の調整処理
                if outlier.outlier_type == OutlierType.ACCOUNTING_CHANGE:
                    # 前年値ベースの調整
                    before_value = outlier.context.get('before_value')
                    if before_value is not None:
                        # 簡単な線形補間調整（実際にはより精密な調整が必要）
                        adjusted_value = (before_value + outlier.value) / 2
                        df_treated.loc[mask, 'value'] = adjusted_value
                        df_treated.loc[mask, 'adjusted_flag'] = True
                        treatment_log.append(f"Adjusted: {outlier.company_id}, {outlier.year}, {outlier.metric_name}")
                
            elif treatment_method == "investigate":
                df_treated.loc[mask, 'investigate_flag'] = True
                treatment_log.append(f"Investigate: {outlier.company_id}, {outlier.year}, {outlier.metric_name}")
        
        logger.info(f"外れ値処理完了: {len(treatment_log)}件の処理を実行")
        
        # 処理ログを別途保存することも可能
        return df_treated


# 使用例・テストコード
if __name__ == "__main__":
    # サンプルデータでのテスト
    np.random.seed(42)
    
    # サンプル財務データ作成
    companies = ['company_A', 'company_B', 'company_C']
    years = list(range(2010, 2021))
    metrics = ['sales', 'profit', 'assets', 'roe']
    
    sample_data = []
    for company in companies:
        for year in years:
            for metric in metrics:
                # 正常値 + 一部外れ値
                base_value = np.random.normal(100, 10)
                if np.random.random() < 0.05:  # 5%の確率で外れ値
                    base_value += np.random.normal(0, 50)
                
                sample_data.append({
                    'company_id': company,
                    'year': year,
                    'metric_name': metric,
                    'value': base_value
                })
    
    df_sample = pd.DataFrame(sample_data)
    
    # 外れ値検出器の初期化と実行
    detector = A2AIOutlierDetector()
    outliers = detector.detect_outliers(df_sample)
    
    print(f"検出された外れ値数: {len(outliers)}")
    
    # レポート生成
    report = detector.generate_outlier_report(outliers)
    print(f"外れ値サマリー: {report['summary']}")
    
    # 処理適用
    df_treated = detector.apply_outlier_treatments(df_sample, outliers, "conservative")
    print(f"処理前データ数: {len(df_sample)}, 処理後データ数: {len(df_treated)}")