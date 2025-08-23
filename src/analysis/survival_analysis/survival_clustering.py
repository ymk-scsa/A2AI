"""
A2AI - Advanced Financial Analysis AI
Survival Clustering Analysis Module

企業の生存パターンに基づくクラスタリング分析
- 企業ライフサイクル全体での生存戦略類型化
- 市場カテゴリー別の生存パターン分析
- 倒産・分社・新設企業を含む包括的分析
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import warnings
from dataclasses import dataclass
from enum import Enum

# 科学計算ライブラリ
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.manifold import TSNE

# 生存分析ライブラリ
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test
from lifelines.plotting import plot_lifetimes

# 統計・数値計算
import scipy.stats as stats
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform

# 可視化
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# A2AI内部モジュール
from ...utils.survival_utils import SurvivalMetrics, LifetimeData
from ...utils.statistical_utils import StatisticalTests
from ...config.settings import Config

warnings.filterwarnings('ignore')


class MarketCategory(Enum):
    """市場カテゴリー定義"""
    HIGH_SHARE = "高シェア市場"
    DECLINING = "シェア低下市場" 
    LOST_SHARE = "シェア失失市場"


class SurvivalPattern(Enum):
    """生存パターン分類"""
    PERSISTENT_LEADER = "持続的リーダー"
    RESILIENT_SURVIVOR = "回復力ある生存者"
    DECLINING_INCUMBENT = "衰退中企業"
    RAPID_EXTINCTION = "急速消滅"
    EMERGING_WINNER = "新興勝者"
    STRUGGLING_NEWCOMER = "苦戦新規"


@dataclass
class ClusteringResult:
    """クラスタリング結果格納クラス"""
    cluster_labels: np.ndarray
    cluster_centers: np.ndarray
    silhouette_score: float
    calinski_harabasz: float
    davies_bouldin: float
    n_clusters: int
    method: str
    
    
@dataclass
class SurvivalCluster:
    """生存クラスター情報"""
    cluster_id: int
    pattern_type: SurvivalPattern
    companies: List[str]
    market_distribution: Dict[str, int]
    survival_characteristics: Dict[str, float]
    median_survival_time: Optional[float]
    hazard_ratio: Optional[float]
    key_financial_factors: List[str]


class SurvivalClustering:
    """
    企業生存パターンクラスタリング分析クラス
    
    150社の企業を生存パターンに基づいてクラスタリングし、
    各市場カテゴリーでの生存戦略を類型化する。
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        初期化
        
        Args:
            config: A2AI設定オブジェクト
        """
        self.config = config or Config()
        self.survival_metrics = SurvivalMetrics()
        self.statistical_tests = StatisticalTests()
        
        # クラスタリング結果格納
        self.clustering_results_: Dict[str, ClusteringResult] = {}
        self.optimal_clustering_: Optional[ClusteringResult] = None
        self.survival_clusters_: List[SurvivalCluster] = []
        
        # データ格納
        self.survival_data_: Optional[pd.DataFrame] = None
        self.feature_matrix_: Optional[np.ndarray] = None
        self.company_mapping_: Dict[int, str] = {}
        
        # 前処理器
        self.scaler_ = StandardScaler()
        self.pca_ = PCA(n_components=0.95)  # 95%の分散を保持
        
    def prepare_survival_features(
        self,
        financial_data: pd.DataFrame,
        market_categories: Dict[str, MarketCategory],
        survival_events: pd.DataFrame
    ) -> pd.DataFrame:
        """
        生存分析用特徴量の準備
        
        Args:
            financial_data: 財務データ (150社×40年)
            market_categories: 企業-市場カテゴリーマッピング
            survival_events: 生存イベントデータ (企業消滅・設立情報)
            
        Returns:
            生存分析用特徴量DataFrame
        """
        print("生存分析用特徴量を準備中...")
        
        features = []
        
        for company in financial_data['company_name'].unique():
            company_data = financial_data[
                financial_data['company_name'] == company
            ].sort_values('year')
            
            if len(company_data) == 0:
                continue
                
            # 基本生存情報
            survival_info = self._extract_survival_info(
                company, company_data, survival_events
            )
            
            # 財務特徴量計算
            financial_features = self._calculate_financial_features(company_data)
            
            # 市場・競争特徴量
            market_features = self._calculate_market_features(
                company, company_data, market_categories
            )
            
            # 生存特徴量統合
            company_features = {
                'company_name': company,
                'market_category': market_categories.get(company, 'Unknown'),
                **survival_info,
                **financial_features,
                **market_features
            }
            
            features.append(company_features)
        
        self.survival_data_ = pd.DataFrame(features)
        print(f"生存特徴量準備完了: {len(self.survival_data_)}社")
        
        return self.survival_data_
    
    def _extract_survival_info(
        self,
        company: str,
        company_data: pd.DataFrame,
        survival_events: pd.DataFrame
    ) -> Dict[str, Union[float, bool]]:
        """企業の生存情報抽出"""
        
        # 基本期間情報
        start_year = company_data['year'].min()
        end_year = company_data['year'].max()
        duration = end_year - start_year + 1
        
        # 生存イベント情報
        company_events = survival_events[
            survival_events['company_name'] == company
        ]
        
        # 企業消滅・倒産イベント
        extinction_event = company_events[
            company_events['event_type'].isin(['extinction', 'bankruptcy', 'merger'])
        ]
        
        is_extinct = len(extinction_event) > 0
        extinction_year = extinction_event['event_year'].iloc[0] if is_extinct else None
        
        # 分社・新設イベント
        spinoff_events = company_events[
            company_events['event_type'] == 'spinoff'
        ]
        
        emergence_events = company_events[
            company_events['event_type'] == 'establishment'
        ]
        
        return {
            'start_year': start_year,
            'end_year': end_year,
            'duration': duration,
            'is_extinct': is_extinct,
            'extinction_year': extinction_year,
            'survival_time': extinction_year - start_year if extinction_year else 2024 - start_year,
            'censored': not is_extinct,  # 打ち切りデータか
            'has_spinoff': len(spinoff_events) > 0,
            'is_newcomer': len(emergence_events) > 0,
            'establishment_year': emergence_events['event_year'].iloc[0] if len(emergence_events) > 0 else start_year
        }
    
    def _calculate_financial_features(self, company_data: pd.DataFrame) -> Dict[str, float]:
        """財務特徴量計算"""
        
        # 23の拡張要因項目から主要指標を計算
        features = {}
        
        # 収益性トレンド
        if 'revenue' in company_data.columns:
            revenue_trend = self._calculate_trend(company_data['revenue'])
            features['revenue_trend'] = revenue_trend
            features['revenue_volatility'] = company_data['revenue'].std() / company_data['revenue'].mean()
            features['revenue_growth_stability'] = self._calculate_growth_stability(company_data['revenue'])
        
        # 財務健全性
        if 'total_assets' in company_data.columns and 'total_liabilities' in company_data.columns:
            equity_ratio = (company_data['total_assets'] - company_data['total_liabilities']) / company_data['total_assets']
            features['avg_equity_ratio'] = equity_ratio.mean()
            features['equity_ratio_trend'] = self._calculate_trend(equity_ratio)
        
        # 投資・R&D活動
        if 'rd_expenses' in company_data.columns:
            rd_intensity = company_data['rd_expenses'] / company_data['revenue']
            features['avg_rd_intensity'] = rd_intensity.mean()
            features['rd_intensity_trend'] = self._calculate_trend(rd_intensity)
        
        # 効率性指標
        if 'total_assets' in company_data.columns:
            asset_turnover = company_data['revenue'] / company_data['total_assets']
            features['avg_asset_turnover'] = asset_turnover.mean()
            features['asset_turnover_trend'] = self._calculate_trend(asset_turnover)
        
        # 収益性指標
        if 'operating_income' in company_data.columns:
            operating_margin = company_data['operating_income'] / company_data['revenue']
            features['avg_operating_margin'] = operating_margin.mean()
            features['operating_margin_trend'] = self._calculate_trend(operating_margin)
            features['margin_volatility'] = operating_margin.std()
        
        # 成長性指標
        if len(company_data) > 1:
            revenue_growth = company_data['revenue'].pct_change()
            features['avg_revenue_growth'] = revenue_growth.mean()
            features['growth_consistency'] = 1 / (1 + revenue_growth.std())
        
        # 危険信号検出
        features['financial_distress_score'] = self._calculate_distress_score(company_data)
        
        return features
    
    def _calculate_market_features(
        self,
        company: str,
        company_data: pd.DataFrame,
        market_categories: Dict[str, MarketCategory]
    ) -> Dict[str, float]:
        """市場・競争特徴量計算"""
        
        features = {}
        market_cat = market_categories.get(company, MarketCategory.HIGH_SHARE)
        
        # 市場カテゴリー特徴
        features['market_category_numeric'] = {
            MarketCategory.HIGH_SHARE: 3,
            MarketCategory.DECLINING: 2,  
            MarketCategory.LOST_SHARE: 1
        }.get(market_cat, 2)
        
        # 市場地位の変化（プロキシ指標）
        if 'market_share' in company_data.columns:
            market_share_trend = self._calculate_trend(company_data['market_share'])
            features['market_share_trend'] = market_share_trend
        
        # グローバル展開度
        if 'overseas_revenue_ratio' in company_data.columns:
            features['avg_global_ratio'] = company_data['overseas_revenue_ratio'].mean()
            features['global_expansion_trend'] = self._calculate_trend(
                company_data['overseas_revenue_ratio']
            )
        
        return features
    
    def _calculate_trend(self, series: pd.Series) -> float:
        """時系列トレンド計算（最小二乗法）"""
        if len(series) < 2:
            return 0.0
        
        series_clean = series.dropna()
        if len(series_clean) < 2:
            return 0.0
        
        x = np.arange(len(series_clean))
        y = series_clean.values
        
        # 線形回帰の傾き
        slope, _, _, _, _ = stats.linregress(x, y)
        return slope
    
    def _calculate_growth_stability(self, series: pd.Series) -> float:
        """成長安定性指標"""
        if len(series) < 3:
            return 0.0
        
        growth_rates = series.pct_change().dropna()
        if len(growth_rates) == 0:
            return 0.0
        
        # 成長率の変動係数の逆数
        cv = growth_rates.std() / abs(growth_rates.mean()) if growth_rates.mean() != 0 else float('inf')
        return 1 / (1 + cv)
    
    def _calculate_distress_score(self, company_data: pd.DataFrame) -> float:
        """財務危機スコア計算（簡易版Altman Z-Score）"""
        
        if len(company_data) == 0:
            return 0.0
        
        latest_data = company_data.iloc[-1]
        score = 0.0
        
        # 各指標の計算（利用可能な場合）
        try:
            if 'total_assets' in latest_data and latest_data['total_assets'] > 0:
                # 流動比率代理
                if 'current_assets' in latest_data and 'current_liabilities' in latest_data:
                    current_ratio = latest_data['current_assets'] / latest_data['current_liabilities']
                    score += min(current_ratio, 3) * 0.3
                
                # 資産回転率
                if 'revenue' in latest_data:
                    asset_turnover = latest_data['revenue'] / latest_data['total_assets']
                    score += asset_turnover * 0.3
                
                # 利益率
                if 'net_income' in latest_data:
                    roe = latest_data['net_income'] / latest_data['total_assets']
                    score += roe * 10 * 0.4
            
        except (KeyError, ZeroDivisionError, TypeError):
            pass
        
        return max(0, score)
    
    def perform_clustering(
        self,
        n_clusters_range: Tuple[int, int] = (3, 15),
        methods: List[str] = ['kmeans', 'hierarchical', 'gaussian_mixture']
    ) -> Dict[str, ClusteringResult]:
        """
        複数手法による生存パターンクラスタリング実行
        
        Args:
            n_clusters_range: クラスタ数の範囲
            methods: 使用するクラスタリング手法
            
        Returns:
            手法別クラスタリング結果
        """
        print("生存パターンクラスタリング開始...")
        
        if self.survival_data_ is None:
            raise ValueError("生存特徴量データが準備されていません")
        
        # 特徴量行列準備
        self._prepare_feature_matrix()
        
        results = {}
        
        for method in methods:
            print(f"{method}による クラスタリング実行中...")
            
            method_results = {}
            best_score = -1
            best_result = None
            
            for n_clusters in range(n_clusters_range[0], n_clusters_range[1] + 1):
                try:
                    result = self._perform_single_clustering(method, n_clusters)
                    method_results[n_clusters] = result
                    
                    # シルエットスコアで最適解選択
                    if result.silhouette_score > best_score:
                        best_score = result.silhouette_score
                        best_result = result
                        
                except Exception as e:
                    print(f"クラスタ数{n_clusters}でエラー: {e}")
                    continue
            
            if best_result:
                results[method] = best_result
                print(f"{method}: 最適クラスタ数{best_result.n_clusters}, "
                        f"シルエットスコア{best_result.silhouette_score:.3f}")
        
        self.clustering_results_ = results
        
        # 総合的に最適な手法選択
        self._select_optimal_clustering()
        
        return results
    
    def _prepare_feature_matrix(self):
        """特徴量行列の準備と前処理"""
        
        # 数値特徴量のみ選択
        numeric_features = []
        feature_names = []
        
        for col in self.survival_data_.columns:
            if col in ['company_name', 'market_category']:
                continue
            
            values = pd.to_numeric(self.survival_data_[col], errors='coerce')
            if not values.isna().all():
                numeric_features.append(values.fillna(values.median()))
                feature_names.append(col)
        
        self.feature_matrix_ = np.column_stack(numeric_features)
        self.feature_names_ = feature_names
        
        # 企業名マッピング
        self.company_mapping_ = {
            i: name for i, name in enumerate(self.survival_data_['company_name'])
        }
        
        # 標準化
        self.feature_matrix_scaled_ = self.scaler_.fit_transform(self.feature_matrix_)
        
        # 次元削減（オプション）
        if self.feature_matrix_scaled_.shape[1] > 20:
            self.feature_matrix_pca_ = self.pca_.fit_transform(self.feature_matrix_scaled_)
            print(f"PCAにより{self.feature_matrix_scaled_.shape[1]}次元から"
                    f"{self.feature_matrix_pca_.shape[1]}次元に削減")
        else:
            self.feature_matrix_pca_ = self.feature_matrix_scaled_
    
    def _perform_single_clustering(
        self,
        method: str,
        n_clusters: int
    ) -> ClusteringResult:
        """単一手法でのクラスタリング実行"""
        
        feature_data = self.feature_matrix_pca_
        
        if method == 'kmeans':
            clusterer = KMeans(
                n_clusters=n_clusters,
                random_state=42,
                n_init=10
            )
            labels = clusterer.fit_predict(feature_data)
            centers = clusterer.cluster_centers_
            
        elif method == 'hierarchical':
            clusterer = AgglomerativeClustering(n_clusters=n_clusters)
            labels = clusterer.fit_predict(feature_data)
            centers = np.array([
                feature_data[labels == i].mean(axis=0)
                for i in range(n_clusters)
            ])
            
        elif method == 'gaussian_mixture':
            clusterer = GaussianMixture(
                n_components=n_clusters,
                random_state=42
            )
            clusterer.fit(feature_data)
            labels = clusterer.predict(feature_data)
            centers = clusterer.means_
            
        elif method == 'dbscan':
            clusterer = DBSCAN(eps=0.5, min_samples=5)
            labels = clusterer.fit_predict(feature_data)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            
            if n_clusters == 0:
                raise ValueError("DBSCANでクラスタが形成されませんでした")
            
            centers = np.array([
                feature_data[labels == i].mean(axis=0)
                for i in set(labels) if i != -1
            ])
        
        else:
            raise ValueError(f"未知のクラスタリング手法: {method}")
        
        # 評価指標計算
        if len(set(labels)) > 1:
            silhouette = silhouette_score(feature_data, labels)
            calinski_harabasz = calinski_harabasz_score(feature_data, labels)
            davies_bouldin = davies_bouldin_score(feature_data, labels)
        else:
            silhouette = -1
            calinski_harabasz = 0
            davies_bouldin = float('inf')
        
        return ClusteringResult(
            cluster_labels=labels,
            cluster_centers=centers,
            silhouette_score=silhouette,
            calinski_harabasz=calinski_harabasz,
            davies_bouldin=davies_bouldin,
            n_clusters=n_clusters,
            method=method
        )
    
    def _select_optimal_clustering(self):
        """最適なクラスタリング結果の選択"""
        
        if not self.clustering_results_:
            return
        
        # 複合スコアによる評価
        best_score = -float('inf')
        best_method = None
        
        for method, result in self.clustering_results_.items():
            # 複合評価スコア（シルエット重視）
            composite_score = (
                result.silhouette_score * 0.6 +
                min(result.calinski_harabasz / 1000, 1) * 0.3 +
                (1 / (1 + result.davies_bouldin)) * 0.1
            )
            
            if composite_score > best_score:
                best_score = composite_score
                best_method = method
        
        if best_method:
            self.optimal_clustering_ = self.clustering_results_[best_method]
            print(f"最適手法: {best_method} (複合スコア: {best_score:.3f})")
        
    def analyze_survival_clusters(self) -> List[SurvivalCluster]:
        """
        クラスター別生存特性分析
        
        Returns:
            生存クラスター分析結果
        """
        if self.optimal_clustering_ is None:
            raise ValueError("クラスタリングが実行されていません")
        
        print("生存クラスター特性分析開始...")
        
        clusters = []
        labels = self.optimal_clustering_.cluster_labels
        
        for cluster_id in range(self.optimal_clustering_.n_clusters):
            cluster_mask = labels == cluster_id
            cluster_companies = [
                self.company_mapping_[i] 
                for i in range(len(labels)) 
                if cluster_mask[i]
            ]
            
            # クラスター内企業のデータ
            cluster_data = self.survival_data_[
                self.survival_data_['company_name'].isin(cluster_companies)
            ]
            
            # 生存特性分析
            survival_chars = self._analyze_cluster_survival_characteristics(cluster_data)
            
            # 市場分布
            market_dist = cluster_data['market_category'].value_counts().to_dict()
            
            # パターン分類
            pattern_type = self._classify_survival_pattern(cluster_data, survival_chars)
            
            # 主要財務要因
            key_factors = self._identify_key_financial_factors(
                cluster_mask, cluster_data
            )
            
            cluster = SurvivalCluster(
                cluster_id=cluster_id,
                pattern_type=pattern_type,
                companies=cluster_companies,
                market_distribution=market_dist,
                survival_characteristics=survival_chars,
                median_survival_time=survival_chars.get('median_survival_time'),
                hazard_ratio=survival_chars.get('hazard_ratio'),
                key_financial_factors=key_factors
            )
            
            clusters.append(cluster)
        
        self.survival_clusters_ = clusters
        
        # クラスター間統計的検定
        self._perform_cluster_statistical_tests()
        
        return clusters
    
    def _analyze_cluster_survival_characteristics(
        self, 
        cluster_data: pd.DataFrame
    ) -> Dict[str, float]:
        """クラスター生存特性分析"""
        
        characteristics = {}
        
        # 基本生存統計
        characteristics['extinction_rate'] = cluster_data['is_extinct'].mean()
        characteristics['avg_survival_time'] = cluster_data['survival_time'].mean()
        characteristics['median_survival_time'] = cluster_data['survival_time'].median()
        
        # 新規・分社企業比率
        characteristics['newcomer_rate'] = cluster_data['is_newcomer'].mean()
        characteristics['spinoff_rate'] = cluster_data['has_spinoff'].mean()
        
        # 財務健全性
        if 'avg_equity_ratio' in cluster_data.columns:
            characteristics['avg_financial_health'] = cluster_data['avg_equity_ratio'].mean()
        
        # 成長性
        if 'avg_revenue_growth' in cluster_data.columns:
            characteristics['avg_growth_rate'] = cluster_data['avg_revenue_growth'].mean()
        
        # 収益性
        if 'avg_operating_margin' in cluster_data.columns:
            characteristics['avg_profitability'] = cluster_data['avg_operating_margin'].mean()
        
        # イノベーション活動
        if 'avg_rd_intensity' in cluster_data.columns:
            characteristics['avg_innovation_intensity'] = cluster_data['avg_rd_intensity'].mean()
        
        # 危機耐性
        if 'financial_distress_score' in cluster_data.columns:
            characteristics['avg_distress_resilience'] = cluster_data['financial_distress_score'].mean()
        
        return characteristics
    
    def _classify_survival_pattern(
        self,
        cluster_data: pd.DataFrame,
        survival_chars: Dict[str, float]
    ) -> SurvivalPattern:
        """生存パターン分類"""
        
        extinction_rate = survival_chars['extinction_rate']
        avg_growth = survival_chars.get('avg_growth_rate', 0)
        newcomer_rate = survival_chars['newcomer_rate']
        
        # パターン分類ロジック
        if extinction_rate < 0.1 and avg_growth > 0.05:
            return SurvivalPattern.PERSISTENT_LEADER
        elif extinction_rate < 0.2 and avg_growth > 0:
            return SurvivalPattern.RESILIENT_SURVIVOR
        elif extinction_rate > 0.5:
            return SurvivalPattern.RAPID_EXTINCTION
        elif extinction_rate > 0.3:
            return SurvivalPattern.DECLINING_INCUMBENT
        elif newcomer_rate > 0.5 and avg_growth > 0.1:
            return SurvivalPattern.EMERGING_WINNER
        elif newcomer_rate > 0.3:
            return SurvivalPattern.STRUGGLING_NEWCOMER
        else:
            return SurvivalPattern.RESILIENT_SURVIVOR
    
    def _identify_key_financial_factors(
        self,
        cluster_mask: np.ndarray,
        cluster_data: pd.DataFrame
    ) -> List[str]:
        """クラスター特徴的財務要因特定"""
        
        key_factors = []
        
        # 全体平均との比較で特徴的要因を特定
        for col in self.survival_data_.select_dtypes(include=[np.number]).columns:
            if col in ['duration', 'survival_time']:
                continue
                
            cluster_mean = cluster_data[col].mean()
            overall_mean = self.survival_data_[col].mean()
            
            # 統計的有意差検定
            cluster_values = cluster_data[col].dropna()
            other_values = self.survival_data_[
                ~self.survival_data_['company_name'].isin(cluster_data['company_name'])
            ][col].dropna()
            
            if len(cluster_values) > 5 and len(other_values) > 5:
                _, p_value = stats.ttest_ind(cluster_values, other_values)
                
                # 有意差があり、かつ実用的差異がある場合
                if p_value < 0.05 and abs(cluster_mean - overall_mean) > 0.1 * abs(overall_mean):
                    key_factors.append(col)
        
        # 上位5要因まで
        return key_factors[:5]
    
    def _perform_cluster_statistical_tests(self):
        """クラスター間統計的検定"""
        
        print("クラスター間統計的検定実行中...")
        
        # 生存時間の差検定（Log-rank test）
        survival_times = []
        cluster_groups = []
        event_indicators = []
        
        for cluster in self.survival_clusters_:
            cluster_data = self.survival_data_[
                self.survival_data_['company_name'].isin(cluster.companies)
            ]
            
            survival_times.extend(cluster_data['survival_time'].tolist())
            cluster_groups.extend([cluster.cluster_id] * len(cluster.companies))
            event_indicators.extend((~cluster_data['censored']).tolist())
        
        # 全クラスター間Log-rank検定
        if len(set(cluster_groups)) > 1:
            try:
                # lifelines multivariate_logrank_test
                survival_df = pd.DataFrame({
                    'duration': survival_times,
                    'event': event_indicators,
                    'group': cluster_groups
                })
                
                test_result = multivariate_logrank_test(
                    survival_df['duration'],
                    survival_df['group'],
                    survival_df['event']
                )
                
                print(f"クラスター間生存時間差検定: p-value = {test_result.p_value:.4f}")
                
            except Exception as e:
                print(f"Log-rank検定エラー: {e}")
    
    def generate_cluster_survival_curves(self) -> Dict[str, go.Figure]:
        """
        クラスター別Kaplan-Meier生存曲線生成
        
        Returns:
            生存曲線プロット辞書
        """
        if not self.survival_clusters_:
            raise ValueError("生存クラスター分析が実行されていません")
        
        print("クラスター別生存曲線生成中...")
        
        plots = {}
        
        # 全体生存曲線
        fig_overall = go.Figure()
        
        colors = px.colors.qualitative.Set1[:len(self.survival_clusters_)]
        
        for i, cluster in enumerate(self.survival_clusters_):
            cluster_data = self.survival_data_[
                self.survival_data_['company_name'].isin(cluster.companies)
            ]
            
            # Kaplan-Meier推定
            kmf = KaplanMeierFitter()
            durations = cluster_data['survival_time']
            event_observed = ~cluster_data['censored']
            
            kmf.fit(durations, event_observed, label=f'Cluster {i}: {cluster.pattern_type.value}')
            
            # プロット追加
            fig_overall.add_trace(go.Scatter(
                x=kmf.timeline,
                y=kmf.survival_function_.iloc[:, 0],
                mode='lines',
                name=f'Cluster {i}: {cluster.pattern_type.value}',
                line=dict(color=colors[i], width=2),
                hovertemplate='時間: %{x}<br>生存確率: %{y:.3f}<extra></extra>'
            ))
            
            # 個別クラスタープロット
            fig_individual = go.Figure()
            fig_individual.add_trace(go.Scatter(
                x=kmf.timeline,
                y=kmf.survival_function_.iloc[:, 0],
                mode='lines',
                name=f'Cluster {i}',
                line=dict(color=colors[i], width=3),
                hovertemplate='時間: %{x}<br>生存確率: %{y:.3f}<extra></extra>'
            ))
            
            # 信頼区間
            if hasattr(kmf, 'confidence_interval_'):
                ci = kmf.confidence_interval_
                fig_individual.add_trace(go.Scatter(
                    x=list(kmf.timeline) + list(kmf.timeline[::-1]),
                    y=list(ci.iloc[:, 0]) + list(ci.iloc[:, 1][::-1]),
                    fill='toself',
                    fillcolor=f'rgba({colors[i]}, 0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='95%信頼区間',
                    showlegend=False
                ))
            
            fig_individual.update_layout(
                title=f'生存曲線 - Cluster {i}: {cluster.pattern_type.value}',
                xaxis_title='時間（年）',
                yaxis_title='生存確率',
                template='plotly_white'
            )
            
            plots[f'cluster_{i}'] = fig_individual
        
        # 全体比較プロット
        fig_overall.update_layout(
            title='クラスター別生存曲線比較',
            xaxis_title='時間（年）',
            yaxis_title='生存確率',
            template='plotly_white',
            legend=dict(x=0.02, y=0.02)
        )
        
        plots['overall_comparison'] = fig_overall
        
        return plots
    
    def create_cluster_characterization_dashboard(self) -> go.Figure:
        """
        クラスター特性ダッシュボード作成
        
        Returns:
            インタラクティブダッシュボード
        """
        if not self.survival_clusters_:
            raise ValueError("生存クラスター分析が実行されていません")
        
        print("クラスター特性ダッシュボード作成中...")
        
        # サブプロット構成
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                'クラスター規模・市場分布',
                '消滅率・新規参入率',
                '平均収益性・成長率',
                'R&D投資強度・財務健全性',
                '生存時間分布',
                'ハザード比比較'
            ],
            specs=[
                [{"type": "bar"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "scatter"}],
                [{"type": "box"}, {"type": "bar"}]
            ]
        )
        
        cluster_ids = [c.cluster_id for c in self.survival_clusters_]
        pattern_names = [c.pattern_type.value for c in self.survival_clusters_]
        colors = px.colors.qualitative.Set1[:len(self.survival_clusters_)]
        
        # 1. クラスター規模
        cluster_sizes = [len(c.companies) for c in self.survival_clusters_]
        fig.add_trace(
            go.Bar(
                x=pattern_names,
                y=cluster_sizes,
                marker_color=colors,
                name='企業数',
                text=cluster_sizes,
                textposition='auto'
            ),
            row=1, col=1
        )
        
        # 2. 消滅率
        extinction_rates = [
            c.survival_characteristics.get('extinction_rate', 0) * 100 
            for c in self.survival_clusters_
        ]
        fig.add_trace(
            go.Bar(
                x=pattern_names,
                y=extinction_rates,
                marker_color=colors,
                name='消滅率(%)',
                text=[f'{rate:.1f}%' for rate in extinction_rates],
                textposition='auto'
            ),
            row=1, col=2
        )
        
        # 3. 収益性 vs 成長率
        profitability = [
            c.survival_characteristics.get('avg_profitability', 0) * 100 
            for c in self.survival_clusters_
        ]
        growth_rates = [
            c.survival_characteristics.get('avg_growth_rate', 0) * 100 
            for c in self.survival_clusters_
        ]
        
        fig.add_trace(
            go.Scatter(
                x=growth_rates,
                y=profitability,
                mode='markers+text',
                marker=dict(
                    size=[s/2 for s in cluster_sizes],
                    color=colors,
                    opacity=0.7
                ),
                text=pattern_names,
                textposition='top center',
                name='収益性 vs 成長率'
            ),
            row=2, col=1
        )
        
        # 4. R&D投資強度
        rd_intensity = [
            c.survival_characteristics.get('avg_innovation_intensity', 0) * 100 
            for c in self.survival_clusters_
        ]
        fig.add_trace(
            go.Bar(
                x=pattern_names,
                y=rd_intensity,
                marker_color=colors,
                name='R&D強度(%)',
                text=[f'{rd:.2f}%' for rd in rd_intensity],
                textposition='auto'
            ),
            row=2, col=2
        )
        
        # 5. 生存時間分布（Box Plot）
        for i, cluster in enumerate(self.survival_clusters_):
            cluster_data = self.survival_data_[
                self.survival_data_['company_name'].isin(cluster.companies)
            ]
            
            fig.add_trace(
                go.Box(
                    y=cluster_data['survival_time'],
                    name=f'C{i}',
                    marker_color=colors[i],
                    boxpoints='outliers'
                ),
                row=3, col=1
            )
        
        # 6. ハザード比（相対リスク）
        hazard_ratios = []
        for cluster in self.survival_clusters_:
            # 基準クラスター（最大生存時間）との比較でハザード比計算
            hr = cluster.survival_characteristics.get('hazard_ratio', 1.0)
            hazard_ratios.append(hr)
        
        fig.add_trace(
            go.Bar(
                x=pattern_names,
                y=hazard_ratios,
                marker_color=colors,
                name='ハザード比',
                text=[f'{hr:.2f}' for hr in hazard_ratios],
                textposition='auto'
            ),
            row=3, col=2
        )
        
        # レイアウト調整
        fig.update_layout(
            height=1200,
            title_text='生存クラスター特性ダッシュボード',
            showlegend=False,
            template='plotly_white'
        )
        
        # 各軸のタイトル設定
        fig.update_xaxes(title_text='クラスター', row=1, col=1)
        fig.update_xaxes(title_text='クラスター', row=1, col=2)
        fig.update_xaxes(title_text='平均成長率(%)', row=2, col=1)
        fig.update_xaxes(title_text='クラスター', row=2, col=2)
        fig.update_xaxes(title_text='クラスター', row=3, col=1)
        fig.update_xaxes(title_text='クラスター', row=3, col=2)
        
        fig.update_yaxes(title_text='企業数', row=1, col=1)
        fig.update_yaxes(title_text='消滅率(%)', row=1, col=2)
        fig.update_yaxes(title_text='平均営業利益率(%)', row=2, col=1)
        fig.update_yaxes(title_text='R&D投資強度(%)', row=2, col=2)
        fig.update_yaxes(title_text='生存時間(年)', row=3, col=1)
        fig.update_yaxes(title_text='ハザード比', row=3, col=2)
        
        return fig
    
    def export_cluster_analysis_results(self, output_dir: str = 'results/survival_analysis'):
        """
        クラスター分析結果エクスポート
        
        Args:
            output_dir: 出力ディレクトリ
        """
        import os
        import json
        
        if not self.survival_clusters_:
            raise ValueError("分析結果がありません")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. クラスター企業リスト
        cluster_companies = {}
        for cluster in self.survival_clusters_:
            cluster_companies[f'cluster_{cluster.cluster_id}'] = {
                'pattern_type': cluster.pattern_type.value,
                'companies': cluster.companies,
                'market_distribution': cluster.market_distribution,
                'characteristics': cluster.survival_characteristics
            }
        
        with open(f'{output_dir}/cluster_companies.json', 'w', encoding='utf-8') as f:
            json.dump(cluster_companies, f, ensure_ascii=False, indent=2)
        
        # 2. クラスター別財務特徴量
        cluster_features = []
        for cluster in self.survival_clusters_:
            cluster_data = self.survival_data_[
                self.survival_data_['company_name'].isin(cluster.companies)
            ].copy()
            cluster_data['cluster_id'] = cluster.cluster_id
            cluster_data['pattern_type'] = cluster.pattern_type.value
            cluster_features.append(cluster_data)
        
        combined_features = pd.concat(cluster_features, ignore_index=True)
        combined_features.to_csv(f'{output_dir}/cluster_features.csv', index=False)
        
        # 3. クラスター統計サマリー
        summary_stats = []
        for cluster in self.survival_clusters_:
            stats_row = {
                'cluster_id': cluster.cluster_id,
                'pattern_type': cluster.pattern_type.value,
                'company_count': len(cluster.companies),
                'extinction_rate': cluster.survival_characteristics.get('extinction_rate', 0),
                'avg_survival_time': cluster.survival_characteristics.get('avg_survival_time', 0),
                'median_survival_time': cluster.median_survival_time,
                'hazard_ratio': cluster.hazard_ratio,
                'key_factors': ', '.join(cluster.key_financial_factors)
            }
            
            # 市場分布追加
            for market, count in cluster.market_distribution.items():
                stats_row[f'market_{market.lower().replace(" ", "_")}'] = count
            
            summary_stats.append(stats_row)
        
        summary_df = pd.DataFrame(summary_stats)
        summary_df.to_csv(f'{output_dir}/cluster_summary.csv', index=False)
        
        # 4. クラスタリング評価結果
        if self.optimal_clustering_:
            clustering_eval = {
                'optimal_method': self.optimal_clustering_.method,
                'n_clusters': self.optimal_clustering_.n_clusters,
                'silhouette_score': self.optimal_clustering_.silhouette_score,
                'calinski_harabasz_score': self.optimal_clustering_.calinski_harabasz,
                'davies_bouldin_score': self.optimal_clustering_.davies_bouldin
            }
            
            with open(f'{output_dir}/clustering_evaluation.json', 'w') as f:
                json.dump(clustering_eval, f, indent=2)
        
        print(f"クラスター分析結果を{output_dir}にエクスポート完了")
    
    def generate_strategic_insights(self) -> Dict[str, any]:
        """
        戦略的インサイト生成
        
        Returns:
            戦略的洞察辞書
        """
        if not self.survival_clusters_:
            raise ValueError("クラスター分析が実行されていません")
        
        print("戦略的インサイト生成中...")
        
        insights = {
            'cluster_patterns': {},
            'market_insights': {},
            'survival_factors': {},
            'strategic_recommendations': []
        }
        
        # クラスターパターン分析
        for cluster in self.survival_clusters_:
            pattern = cluster.pattern_type.value
            
            insights['cluster_patterns'][pattern] = {
                'description': self._get_pattern_description(cluster.pattern_type),
                'company_count': len(cluster.companies),
                'representative_companies': cluster.companies[:3],
                'key_characteristics': cluster.survival_characteristics,
                'success_factors': cluster.key_financial_factors,
                'market_focus': max(cluster.market_distribution, key=cluster.market_distribution.get)
            }
        
        # 市場別インサイト
        market_patterns = {}
        for cluster in self.survival_clusters_:
            for market, count in cluster.market_distribution.items():
                if market not in market_patterns:
                    market_patterns[market] = []
                market_patterns[market].append({
                    'pattern': cluster.pattern_type.value,
                    'count': count,
                    'extinction_rate': cluster.survival_characteristics.get('extinction_rate', 0)
                })
        
        insights['market_insights'] = market_patterns
        
        # 生存要因分析
        all_factors = {}
        for cluster in self.survival_clusters_:
            for factor in cluster.key_financial_factors:
                if factor not in all_factors:
                    all_factors[factor] = {'clusters': [], 'importance': 0}
                all_factors[factor]['clusters'].append(cluster.pattern_type.value)
                all_factors[factor]['importance'] += 1
        
        # 重要度順にソート
        insights['survival_factors'] = dict(
            sorted(all_factors.items(), key=lambda x: x[1]['importance'], reverse=True)
        )
        
        # 戦略的推奨事項
        insights['strategic_recommendations'] = self._generate_strategic_recommendations()
        
        return insights
    
    def _get_pattern_description(self, pattern: SurvivalPattern) -> str:
        """生存パターンの説明文生成"""
        descriptions = {
            SurvivalPattern.PERSISTENT_LEADER: "継続的な市場リーダーシップを維持し、安定した成長を実現している企業群",
            SurvivalPattern.RESILIENT_SURVIVOR: "市場変動に対する高い適応力を持ち、困難な状況でも生存し続ける企業群",
            SurvivalPattern.DECLINING_INCUMBENT: "過去の成功に依存し、市場変化への対応が遅れている既存大企業群",
            SurvivalPattern.RAPID_EXTINCTION: "急激な環境変化に適応できず、短期間で市場から退出した企業群",
            SurvivalPattern.EMERGING_WINNER: "新規市場参入に成功し、急速な成長を遂げている新興企業群",
            SurvivalPattern.STRUGGLING_NEWCOMER: "市場参入は果たしたものの、競争優位の確立に苦戦している新興企業群"
        }
        return descriptions.get(pattern, "未分類パターン")
    
    def _generate_strategic_recommendations(self) -> List[str]:
        """戦略的推奨事項生成"""
        recommendations = []
        
        # クラスター分析結果から推奨事項を導出
        high_survival_clusters = [
            c for c in self.survival_clusters_ 
            if c.survival_characteristics.get('extinction_rate', 1) < 0.2
        ]
        
        if high_survival_clusters:
            # 高生存率クラスターの共通要因
            common_factors = set()
            for cluster in high_survival_clusters:
                common_factors.update(cluster.key_financial_factors)
            
            if common_factors:
                recommendations.append(
                    f"生存率の高い企業群では{', '.join(list(common_factors)[:3])}が重要な要因として特定されました。"
                    "これらの財務指標の改善が企業存続に寄与する可能性があります。"
                )
        
        # 市場別推奨事項
        market_extinction_rates = {}
        for cluster in self.survival_clusters_:
            for market, count in cluster.market_distribution.items():
                if market not in market_extinction_rates:
                    market_extinction_rates[market] = []
                market_extinction_rates[market].append(
                    cluster.survival_characteristics.get('extinction_rate', 0)
                )
        
        for market, rates in market_extinction_rates.items():
            avg_rate = sum(rates) / len(rates)
            if avg_rate > 0.4:
                recommendations.append(
                    f"{market}では企業消滅率が高く（{avg_rate:.1%}）、"
                    "業界再編や新たな競争優位の構築が急務です。"
                )
        
        # イノベーション投資の推奨
        high_rd_survivors = [
            c for c in self.survival_clusters_
            if (c.survival_characteristics.get('avg_innovation_intensity', 0) > 0.03 and
                c.survival_characteristics.get('extinction_rate', 1) < 0.3)
        ]
        
        if high_rd_survivors:
            recommendations.append(
                "R&D投資強度が3%以上の企業群では生存率が有意に高く、"
                "継続的なイノベーション投資が長期存続の鍵となることが示されました。"
            )
        
        return recommendations


# 使用例とテスト用のメイン関数
def main():
    """
    SurvivalClusteringの使用例
    """
    # サンプルデータ生成（実際の使用では実データを使用）
    np.random.seed(42)
    
    # 150社のサンプル企業データ
    companies = [f'Company_{i:03d}' for i in range(150)]
    
    # 市場カテゴリー
    market_categories = {}
    for i, company in enumerate(companies):
        if i < 50:
            market_categories[company] = MarketCategory.HIGH_SHARE
        elif i < 100:
            market_categories[company] = MarketCategory.DECLINING
        else:
            market_categories[company] = MarketCategory.LOST_SHARE
    
    # サンプル財務データ
    financial_data = []
    for company in companies:
        for year in range(1984, 2024):
            financial_data.append({
                'company_name': company,
                'year': year,
                'revenue': np.random.lognormal(10, 0.5),
                'operating_income': np.random.normal(0.1, 0.05),
                'total_assets': np.random.lognormal(11, 0.5),
                'rd_expenses': np.random.normal(0.03, 0.02),
            })
    
    financial_df = pd.DataFrame(financial_data)
    
    # サンプル生存イベント
    survival_events = []
    for i, company in enumerate(companies):
        # 一部企業に消滅イベント
        if i % 20 == 0:
            survival_events.append({
                'company_name': company,
                'event_type': 'extinction',
                'event_year': np.random.randint(2000, 2020)
            })
        # 新設企業
        elif i % 15 == 0:
            survival_events.append({
                'company_name': company,
                'event_type': 'establishment',
                'event_year': np.random.randint(1990, 2010)
            })
    
    survival_events_df = pd.DataFrame(survival_events)
    
    # SurvivalClustering実行
    sc = SurvivalClustering()
    
    # 1. 特徴量準備
    survival_features = sc.prepare_survival_features(
        financial_df, market_categories, survival_events_df
    )
    
    # 2. クラスタリング実行
    clustering_results = sc.perform_clustering()
    
    # 3. 生存クラスター分析
    survival_clusters = sc.analyze_survival_clusters()
    
    # 4. 可視化
    survival_curves = sc.generate_cluster_survival_curves()
    dashboard = sc.create_cluster_characterization_dashboard()
    
    # 5. 結果エクスポート
    sc.export_cluster_analysis_results()
    
    # 6. 戦略的インサイト
    insights = sc.generate_strategic_insights()
    
    print("=== 生存クラスタリング分析完了 ===")
    print(f"識別されたクラスター数: {len(survival_clusters)}")
    for cluster in survival_clusters:
        print(f"Cluster {cluster.cluster_id}: {cluster.pattern_type.value} "
                f"({len(cluster.companies)}社)")
    
    return sc, insights


if __name__ == "__main__":
    clustering_system, strategic_insights = main()