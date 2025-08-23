"""
A2AI Clustering Analyzer Module

企業の財務データと市場特性に基づく多次元クラスタリング分析を実行
企業ライフサイクル、生存パターン、市場シェア変動を考慮した
包括的なクラスタリング手法を提供

Author: A2AI Development Team
Date: 2024
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# Machine Learning Libraries
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import IsolationForest

# Specialized Clustering
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from scipy.stats import multivariate_normal

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Ellipse
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


class ClusteringMethod(Enum):
    """クラスタリング手法の列挙型"""
    KMEANS = "kmeans"
    DBSCAN = "dbscan"
    HIERARCHICAL = "hierarchical"
    GAUSSIAN_MIXTURE = "gaussian_mixture"
    LIFECYCLE_BASED = "lifecycle_based"
    SURVIVAL_BASED = "survival_based"
    PERFORMANCE_BASED = "performance_based"
    MARKET_BASED = "market_based"


class ScalingMethod(Enum):
    """スケーリング手法の列挙型"""
    STANDARD = "standard"
    ROBUST = "robust"
    MINMAX = "minmax"
    NONE = "none"


@dataclass
class ClusteringConfig:
    """クラスタリング設定クラス"""
    method: ClusteringMethod
    n_clusters: Optional[int] = None
    scaling_method: ScalingMethod = ScalingMethod.STANDARD
    features: Optional[List[str]] = None
    random_state: int = 42
    min_samples: int = 5
    eps: float = 0.5
    linkage_method: str = 'ward'
    
    # A2AI固有の設定
    consider_lifecycle: bool = True
    consider_survival: bool = True
    consider_market_category: bool = True
    time_window: Optional[Tuple[int, int]] = None


@dataclass
class ClusterResult:
    """クラスタリング結果クラス"""
    labels: np.ndarray
    n_clusters: int
    silhouette_score: float
    calinski_harabasz_score: float
    davies_bouldin_score: float
    cluster_centers: Optional[np.ndarray] = None
    feature_names: List[str] = None
    cluster_statistics: Dict[int, Dict] = None
    outliers: Optional[np.ndarray] = None


class LifecycleClusterAnalyzer:
    """企業ライフサイクル特化クラスタリング分析"""
    
    def __init__(self):
        self.lifecycle_stages = {
            'startup': (0, 5),      # 設立0-5年
            'growth': (6, 15),      # 成長期6-15年
            'maturity': (16, 30),   # 成熟期16-30年
            'decline_or_renewal': (31, float('inf'))  # 衰退期または再生期31年以上
        }
    
    def extract_lifecycle_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """ライフサイクル特徴量の抽出"""
        features = pd.DataFrame(index=df.index)
        
        # 企業年齢ベースの特徴量
        if 'company_age' in df.columns:
            features['lifecycle_stage'] = df['company_age'].apply(self._classify_lifecycle_stage)
            features['normalized_age'] = df['company_age'] / df['company_age'].max()
            features['age_squared'] = features['normalized_age'] ** 2
            features['age_log'] = np.log1p(df['company_age'])
        
        # 成長パターン特徴量
        growth_cols = [col for col in df.columns if 'growth' in col.lower()]
        if growth_cols:
            features['avg_growth_rate'] = df[growth_cols].mean(axis=1)
            features['growth_volatility'] = df[growth_cols].std(axis=1)
            features['growth_trend'] = df[growth_cols].apply(self._calculate_trend, axis=1)
        
        # 業績安定性特徴量
        performance_cols = [col for col in df.columns if any(metric in col.lower() 
                            for metric in ['roi', 'roe', 'profit', 'margin'])]
        if performance_cols:
            features['performance_stability'] = 1 / (1 + df[performance_cols].std(axis=1))
            features['performance_level'] = df[performance_cols].mean(axis=1)
        
        return features
    
    def _classify_lifecycle_stage(self, age: float) -> str:
        """企業年齢からライフサイクルステージを分類"""
        for stage, (min_age, max_age) in self.lifecycle_stages.items():
            if min_age <= age <= max_age:
                return stage
        return 'mature'
    
    def _calculate_trend(self, series: pd.Series) -> float:
        """時系列のトレンドを計算"""
        if len(series.dropna()) < 2:
            return 0.0
        x = np.arange(len(series))
        y = series.values
        valid_mask = ~np.isnan(y)
        if np.sum(valid_mask) < 2:
            return 0.0
        return np.polyfit(x[valid_mask], y[valid_mask], 1)[0]


class SurvivalClusterAnalyzer:
    """生存パターンベースクラスタリング分析"""
    
    def extract_survival_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """生存分析関連の特徴量抽出"""
        features = pd.DataFrame(index=df.index)
        
        # 企業存続状態
        if 'survival_status' in df.columns:
            features['is_active'] = (df['survival_status'] == 'active').astype(int)
            features['survival_years'] = df.get('survival_years', df.get('company_age', 0))
        
        # リスク指標特徴量
        risk_indicators = ['debt_ratio', 'current_ratio', 'interest_coverage', 'cash_flow']
        available_risk = [col for col in risk_indicators if col in df.columns]
        
        if available_risk:
            features['financial_health_score'] = df[available_risk].apply(
                self._calculate_financial_health, axis=1)
            features['risk_level'] = df[available_risk].apply(
                self._calculate_risk_level, axis=1)
        
        # 業績悪化パターン
        profit_cols = [col for col in df.columns if 'profit' in col.lower() or 'income' in col.lower()]
        if profit_cols:
            features['consecutive_losses'] = df[profit_cols].apply(
                self._count_consecutive_negatives, axis=1)
            features['profit_decline_rate'] = df[profit_cols].apply(
                self._calculate_decline_rate, axis=1)
        
        return features
    
    def _calculate_financial_health(self, row: pd.Series) -> float:
        """財務健全性スコア計算"""
        score = 0.0
        if 'debt_ratio' in row.index and not pd.isna(row['debt_ratio']):
            score += (1 - min(row['debt_ratio'], 1)) * 0.3
        if 'current_ratio' in row.index and not pd.isna(row['current_ratio']):
            score += min(row['current_ratio'] / 2, 1) * 0.3
        if 'interest_coverage' in row.index and not pd.isna(row['interest_coverage']):
            score += min(row['interest_coverage'] / 5, 1) * 0.2
        if 'cash_flow' in row.index and not pd.isna(row['cash_flow']):
            score += (1 if row['cash_flow'] > 0 else 0) * 0.2
        return score
    
    def _calculate_risk_level(self, row: pd.Series) -> float:
        """リスクレベル計算（0: 低リスク, 1: 高リスク）"""
        return 1 - self._calculate_financial_health(row)
    
    def _count_consecutive_negatives(self, series: pd.Series) -> int:
        """連続した負の値の数をカウント"""
        values = series.dropna().values
        if len(values) == 0:
            return 0
        
        max_consecutive = 0
        current_consecutive = 0
        for value in reversed(values):  # 最新から過去に向かって
            if value < 0:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                break
        return max_consecutive
    
    def _calculate_decline_rate(self, series: pd.Series) -> float:
        """業績悪化率の計算"""
        values = series.dropna().values
        if len(values) < 2:
            return 0.0
        
        recent = np.mean(values[-3:]) if len(values) >= 3 else values[-1]
        past = np.mean(values[:3]) if len(values) >= 3 else values[0]
        
        if past == 0:
            return 0.0
        return (past - recent) / abs(past)


class ClusteringAnalyzer:
    """メインクラスタリング分析クラス"""
    
    def __init__(self, config: ClusteringConfig):
        self.config = config
        self.logger = self._setup_logger()
        self.scaler = self._initialize_scaler()
        self.lifecycle_analyzer = LifecycleClusterAnalyzer()
        self.survival_analyzer = SurvivalClusterAnalyzer()
        
        # クラスタリング結果保存
        self.results: Dict[str, ClusterResult] = {}
        self.scaled_data: Optional[np.ndarray] = None
        self.feature_names: List[str] = []
    
    def _setup_logger(self) -> logging.Logger:
        """ロガーのセットアップ"""
        logger = logging.getLogger(f'clustering_analyzer_{id(self)}')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _initialize_scaler(self):
        """スケーラーの初期化"""
        if self.config.scaling_method == ScalingMethod.STANDARD:
            return StandardScaler()
        elif self.config.scaling_method == ScalingMethod.ROBUST:
            return RobustScaler()
        elif self.config.scaling_method == ScalingMethod.MINMAX:
            return MinMaxScaler()
        else:
            return None
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """特徴量の準備と前処理"""
        self.logger.info("特徴量の準備を開始")
        
        # 基本特徴量の選択
        if self.config.features:
            base_features = df[self.config.features].copy()
        else:
            # 数値型の列を自動選択
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            base_features = df[numeric_cols].copy()
        
        # A2AI拡張特徴量の追加
        extended_features = base_features.copy()
        
        if self.config.consider_lifecycle:
            lifecycle_features = self.lifecycle_analyzer.extract_lifecycle_features(df)
            extended_features = pd.concat([extended_features, lifecycle_features], axis=1)
            self.logger.info(f"ライフサイクル特徴量を追加: {lifecycle_features.columns.tolist()}")
        
        if self.config.consider_survival:
            survival_features = self.survival_analyzer.extract_survival_features(df)
            extended_features = pd.concat([extended_features, survival_features], axis=1)
            self.logger.info(f"生存分析特徴量を追加: {survival_features.columns.tolist()}")
        
        # 欠損値処理
        extended_features = extended_features.fillna(extended_features.median())
        
        # 無限値・NaN値の処理
        extended_features = extended_features.replace([np.inf, -np.inf], np.nan)
        extended_features = extended_features.fillna(0)
        
        # スケーリング
        if self.scaler:
            scaled_data = self.scaler.fit_transform(extended_features)
        else:
            scaled_data = extended_features.values
        
        self.scaled_data = scaled_data
        self.feature_names = extended_features.columns.tolist()
        
        self.logger.info(f"最終的な特徴量数: {scaled_data.shape[1]}")
        return scaled_data, self.feature_names
    
    def perform_clustering(self, data: np.ndarray, method: ClusteringMethod = None) -> ClusterResult:
        """指定された手法でクラスタリングを実行"""
        if method is None:
            method = self.config.method
        
        self.logger.info(f"クラスタリング実行: {method.value}")
        
        if method == ClusteringMethod.KMEANS:
            return self._kmeans_clustering(data)
        elif method == ClusteringMethod.DBSCAN:
            return self._dbscan_clustering(data)
        elif method == ClusteringMethod.HIERARCHICAL:
            return self._hierarchical_clustering(data)
        elif method == ClusteringMethod.GAUSSIAN_MIXTURE:
            return self._gaussian_mixture_clustering(data)
        elif method == ClusteringMethod.LIFECYCLE_BASED:
            return self._lifecycle_based_clustering(data)
        elif method == ClusteringMethod.SURVIVAL_BASED:
            return self._survival_based_clustering(data)
        else:
            raise ValueError(f"サポートされていないクラスタリング手法: {method}")
    
    def _kmeans_clustering(self, data: np.ndarray) -> ClusterResult:
        """K-meansクラスタリング"""
        n_clusters = self.config.n_clusters or self._estimate_optimal_clusters(data, 'kmeans')
        
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=self.config.random_state,
            n_init=10
        )
        
        labels = kmeans.fit_predict(data)
        
        return self._create_cluster_result(
            labels, kmeans.cluster_centers_, data
        )
    
    def _dbscan_clustering(self, data: np.ndarray) -> ClusterResult:
        """DBSCANクラスタリング"""
        dbscan = DBSCAN(
            eps=self.config.eps,
            min_samples=self.config.min_samples
        )
        
        labels = dbscan.fit_predict(data)
        outliers = (labels == -1)
        
        return self._create_cluster_result(
            labels, None, data, outliers
        )
    
    def _hierarchical_clustering(self, data: np.ndarray) -> ClusterResult:
        """階層クラスタリング"""
        n_clusters = self.config.n_clusters or self._estimate_optimal_clusters(data, 'hierarchical')
        
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=self.config.linkage_method
        )
        
        labels = clustering.fit_predict(data)
        
        return self._create_cluster_result(labels, None, data)
    
    def _gaussian_mixture_clustering(self, data: np.ndarray) -> ClusterResult:
        """ガウシアン混合モデルクラスタリング"""
        n_components = self.config.n_clusters or self._estimate_optimal_clusters(data, 'gmm')
        
        gmm = GaussianMixture(
            n_components=n_components,
            random_state=self.config.random_state
        )
        
        labels = gmm.fit_predict(data)
        
        return self._create_cluster_result(
            labels, gmm.means_, data
        )
    
    def _lifecycle_based_clustering(self, data: np.ndarray) -> ClusterResult:
        """ライフサイクル特化クラスタリング"""
        # ライフサイクル特徴量に重みを付けて処理
        lifecycle_features = ['lifecycle_stage', 'normalized_age', 'growth_trend', 
                            'performance_stability']
        
        # 利用可能なライフサイクル特徴量を特定
        available_lifecycle = [i for i, name in enumerate(self.feature_names) 
                                if any(lf in name for lf in lifecycle_features)]
        
        if available_lifecycle:
            # ライフサイクル特徴量に重みを適用
            weighted_data = data.copy()
            weighted_data[:, available_lifecycle] *= 2.0  # ライフサイクル特徴量を2倍に重み付け
        else:
            weighted_data = data
        
        return self._kmeans_clustering(weighted_data)
    
    def _survival_based_clustering(self, data: np.ndarray) -> ClusterResult:
        """生存パターン特化クラスタリング"""
        survival_features = ['financial_health_score', 'risk_level', 
                            'consecutive_losses', 'survival_years']
        
        # 利用可能な生存特徴量を特定
        available_survival = [i for i, name in enumerate(self.feature_names) 
                            if any(sf in name for sf in survival_features)]
        
        if available_survival:
            # 生存特徴量に重みを適用
            weighted_data = data.copy()
            weighted_data[:, available_survival] *= 2.5  # 生存特徴量を2.5倍に重み付け
        else:
            weighted_data = data
        
        return self._kmeans_clustering(weighted_data)
    
    def _estimate_optimal_clusters(self, data: np.ndarray, method: str) -> int:
        """最適クラスター数の推定"""
        max_clusters = min(10, data.shape[0] // 2)
        if max_clusters < 2:
            return 2
        
        silhouette_scores = []
        
        for k in range(2, max_clusters + 1):
            if method == 'kmeans':
                labels = KMeans(n_clusters=k, random_state=self.config.random_state).fit_predict(data)
            elif method == 'hierarchical':
                labels = AgglomerativeClustering(n_clusters=k).fit_predict(data)
            elif method == 'gmm':
                labels = GaussianMixture(n_components=k, random_state=self.config.random_state).fit_predict(data)
            else:
                labels = KMeans(n_clusters=k, random_state=self.config.random_state).fit_predict(data)
            
            if len(np.unique(labels)) > 1:
                score = silhouette_score(data, labels)
                silhouette_scores.append(score)
            else:
                silhouette_scores.append(-1)
        
        optimal_k = np.argmax(silhouette_scores) + 2
        self.logger.info(f"最適クラスター数を推定: {optimal_k}")
        return optimal_k
    
    def _create_cluster_result(self, labels: np.ndarray, centers: Optional[np.ndarray], 
                                data: np.ndarray, outliers: Optional[np.ndarray] = None) -> ClusterResult:
        """クラスタリング結果オブジェクトの作成"""
        n_clusters = len(np.unique(labels[labels >= 0]))  # -1(outlier)を除く
        
        # 評価指標の計算
        if n_clusters > 1 and len(np.unique(labels)) > 1:
            sil_score = silhouette_score(data, labels)
            ch_score = calinski_harabasz_score(data, labels)
            db_score = davies_bouldin_score(data, labels)
        else:
            sil_score = ch_score = db_score = 0.0
        
        # クラスター統計の計算
        cluster_stats = self._calculate_cluster_statistics(data, labels)
        
        return ClusterResult(
            labels=labels,
            n_clusters=n_clusters,
            silhouette_score=sil_score,
            calinski_harabasz_score=ch_score,
            davies_bouldin_score=db_score,
            cluster_centers=centers,
            feature_names=self.feature_names,
            cluster_statistics=cluster_stats,
            outliers=outliers
        )
    
    def _calculate_cluster_statistics(self, data: np.ndarray, labels: np.ndarray) -> Dict[int, Dict]:
        """クラスター統計の計算"""
        stats = {}
        unique_labels = np.unique(labels)
        
        for label in unique_labels:
            if label == -1:  # 外れ値は除外
                continue
            
            mask = labels == label
            cluster_data = data[mask]
            
            stats[label] = {
                'size': np.sum(mask),
                'mean': np.mean(cluster_data, axis=0),
                'std': np.std(cluster_data, axis=0),
                'median': np.median(cluster_data, axis=0),
                'min': np.min(cluster_data, axis=0),
                'max': np.max(cluster_data, axis=0)
            }
        
        return stats
    
    def compare_clustering_methods(self, data: np.ndarray) -> Dict[str, ClusterResult]:
        """複数のクラスタリング手法を比較"""
        methods = [
            ClusteringMethod.KMEANS,
            ClusteringMethod.HIERARCHICAL,
            ClusteringMethod.GAUSSIAN_MIXTURE,
            ClusteringMethod.DBSCAN
        ]
        
        if self.config.consider_lifecycle:
            methods.append(ClusteringMethod.LIFECYCLE_BASED)
        
        if self.config.consider_survival:
            methods.append(ClusteringMethod.SURVIVAL_BASED)
        
        results = {}
        
        for method in methods:
            try:
                result = self.perform_clustering(data, method)
                results[method.value] = result
                self.logger.info(f"{method.value} - シルエット係数: {result.silhouette_score:.3f}")
            except Exception as e:
                self.logger.warning(f"{method.value} でエラー発生: {str(e)}")
                continue
        
        return results
    
    def analyze_cluster_characteristics(self, df: pd.DataFrame, result: ClusterResult) -> Dict[int, Dict]:
        """クラスター特性の詳細分析"""
        characteristics = {}
        
        for cluster_id in range(result.n_clusters):
            mask = result.labels == cluster_id
            cluster_companies = df[mask]
            
            char = {
                'cluster_size': np.sum(mask),
                'company_list': cluster_companies.index.tolist(),
                'dominant_features': self._identify_dominant_features(
                    result.cluster_statistics[cluster_id], cluster_id
                ),
                'market_distribution': self._analyze_market_distribution(cluster_companies),
                'lifecycle_distribution': self._analyze_lifecycle_distribution(cluster_companies),
                'survival_characteristics': self._analyze_survival_characteristics(cluster_companies)
            }
            
            characteristics[cluster_id] = char
        
        return characteristics
    
    def _identify_dominant_features(self, stats: Dict, cluster_id: int) -> List[Tuple[str, float]]:
        """クラスターの支配的特徴量を特定"""
        if not stats or 'mean' not in stats:
            return []
        
        means = stats['mean']
        feature_importance = []
        
        for i, feature_name in enumerate(self.feature_names):
            if i < len(means):
                importance = abs(means[i])
                feature_importance.append((feature_name, importance))
        
        # 重要度でソート
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        return feature_importance[:5]  # 上位5つを返す
    
    def _analyze_market_distribution(self, cluster_companies: pd.DataFrame) -> Dict[str, int]:
        """クラスター内の市場分布分析"""
        if 'market_category' not in cluster_companies.columns:
            return {}
        
        return cluster_companies['market_category'].value_counts().to_dict()
    
    def _analyze_lifecycle_distribution(self, cluster_companies: pd.DataFrame) -> Dict[str, int]:
        """クラスター内のライフサイクル分布分析"""
        lifecycle_cols = [col for col in cluster_companies.columns 
                            if 'lifecycle' in col.lower() or 'age' in col.lower()]
        
        if not lifecycle_cols:
            return {}
        
        if 'company_age' in cluster_companies.columns:
            ages = cluster_companies['company_age']
            lifecycle_dist = {}
            for stage, (min_age, max_age) in self.lifecycle_analyzer.lifecycle_stages.items():
                count = ((ages >= min_age) & (ages <= max_age)).sum()
                if count > 0:
                    lifecycle_dist[stage] = count
            return lifecycle_dist
        
        return {}
    
    def _analyze_survival_characteristics(self, cluster_companies: pd.DataFrame) -> Dict[str, Any]:
        """クラスター内の生存特性分析"""
        survival_chars = {}
        
        if 'survival_status' in cluster_companies.columns:
            survival_chars['active_ratio'] = (
                cluster_companies['survival_status'] == 'active'
            ).mean()
        
        if 'survival_years' in cluster_companies.columns:
            survival_chars['avg_survival_years'] = cluster_companies['survival_years'].mean()
        
        # 財務健全性指標
        financial_cols = ['debt_ratio', 'current_ratio', 'roe', 'roa']
        available_financial = [col for col in financial_cols if col in cluster_companies.columns]
        
        if available_financial:
            survival_chars['financial_health'] = {
                col: cluster_companies[col].mean() for col in available_financial
            }
        
        return survival_chars
    
    def generate_clustering_report(self, df: pd.DataFrame, results: Dict[str, ClusterResult]) -> str:
        """クラスタリング分析レポートの生成"""
        report = []
        report.append("=" * 80)
        report.append("A2AI クラスタリング分析レポート")
        report.append("=" * 80)
        report.append(f"分析対象企業数: {len(df)}")
        report.append(f"使用特徴量数: {len(self.feature_names)}")
        report.append("")
        
        # 手法別結果比較
        report.append("手法別評価指標比較:")
        report.append("-" * 50)
        for method_name, result in results.items():
            report.append(f"{method_name:20s} | クラスター数: {result.n_clusters:2d} | "
                            f"シルエット: {result.silhouette_score:6.3f} | "
                            f"CH指数: {result.calinski_harabasz_score:8.1f} | "
                            f"DB指数: {result.davies_bouldin_score:6.3f}")
        
        # 最良手法の詳細分析
        best_method = max(results.keys(), key=lambda k: results[k].silhouette_score)
        best_result = results[best_method]
        
        report.append("")
        report.append(f"最良手法: {best_method}")
        report.append("=" * 50)
        
        # クラスター特性分析
        characteristics = self.analyze_cluster_characteristics(df, best_result)
        
        for cluster_id, char in characteristics.items():
            report.append(f"クラスター {cluster_id} (企業数: {char['cluster_size']})")
            report.append("-" * 30)
            
            # 支配的特徴量
            if char['dominant_features']:
                report.append("支配的特徴量:")
                for feature, importance in char['dominant_features'][:3]:
                    report.append(f"  {feature}: {importance:.3f}")
            
            # 市場分布
            if char['market_distribution']:
                report.append("市場カテゴリー分布:")
                for market, count in char['market_distribution'].items():
                    report.append(f"  {market}: {count}社")