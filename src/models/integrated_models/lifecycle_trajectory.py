"""
A2AI - Advanced Financial Analysis AI
企業ライフサイクル軌道分析モジュール (lifecycle_trajectory.py)

このモジュールは、150社の企業について以下の分析を実行：
1. 企業ライフサイクル全期間の9つの評価項目軌道分析
2. 市場カテゴリー別（高シェア/低下/失失）の軌道パターン比較
3. 生存・消滅・新設企業の軌道特徴抽出
4. ライフサイクル段階遷移分析
5. 将来軌道予測モデル
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

class LifecycleTrajectoryAnalyzer:
    """
    企業ライフサイクル軌道分析の主要クラス
    
    企業の全生存期間を通じた9つの評価項目の変化軌道を分析し、
    市場カテゴリー間での軌道パターンの差異を特定する。
    """
    
    def __init__(self, config: Dict = None):
        """
        初期化
        
        Args:
            config: 設定パラメータ辞書
        """
        self.config = config or self._default_config()
        self.evaluation_metrics = [
            'revenue',  # 売上高
            'revenue_growth_rate',  # 売上高成長率
            'operating_margin',  # 売上高営業利益率
            'net_margin',  # 売上高当期純利益率
            'roe',  # ROE
            'value_added_ratio',  # 売上高付加価値率
            'survival_probability',  # 企業存続確率
            'emergence_success_rate',  # 新規事業成功率
            'succession_success_degree'  # 事業継承成功度
        ]
        
        self.market_categories = ['high_share', 'declining', 'lost']
        self.lifecycle_stages = ['startup', 'growth', 'maturity', 'decline', 'extinction']
        
        # 分析結果格納用
        self.trajectory_data = {}
        self.cluster_results = {}
        self.transition_matrices = {}
        self.prediction_models = {}
        
    def _default_config(self) -> Dict:
        """デフォルト設定"""
        return {
            'min_data_points': 5,  # 軌道分析に必要な最小データポイント数
            'smoothing_window': 3,  # 移動平均ウィンドウサイズ
            'cluster_count': 5,  # クラスタリング数
            'prediction_horizon': 5,  # 予測期間（年）
            'significance_level': 0.05,  # 統計的有意水準
            'lifecycle_threshold': {  # ライフサイクル段階判定閾値
                'startup_years': 10,
                'growth_revenue_growth': 0.1,
                'maturity_growth_range': (-0.05, 0.05),
                'decline_growth_threshold': -0.1
            }
        }
    
    def load_data(self, financial_data: pd.DataFrame, market_data: pd.DataFrame, 
                    company_events: pd.DataFrame) -> None:
        """
        分析データの読み込み
        
        Args:
            financial_data: 財務データ（150社×40年分）
            market_data: 市場カテゴリーデータ
            company_events: 企業イベントデータ（設立・消滅・分社等）
        """
        self.financial_data = financial_data
        self.market_data = market_data
        self.company_events = company_events
        
        # 企業別データ組織化
        self._organize_company_data()
        
        print(f"データ読み込み完了: {len(self.company_data)}社")
        print(f"市場カテゴリー別企業数:")
        for category in self.market_categories:
            count = len([c for c in self.company_data.values() 
                        if c['market_category'] == category])
            print(f"  {category}: {count}社")
    
    def _organize_company_data(self) -> None:
        """企業別データの組織化"""
        self.company_data = {}
        
        for company_id in self.financial_data['company_id'].unique():
            company_financial = self.financial_data[
                self.financial_data['company_id'] == company_id
            ].copy()
            
            # 市場カテゴリー取得
            market_info = self.market_data[
                self.market_data['company_id'] == company_id
            ]
            
            # 企業イベント取得
            events = self.company_events[
                self.company_events['company_id'] == company_id
            ]
            
            self.company_data[company_id] = {
                'financial_data': company_financial.sort_values('year'),
                'market_category': market_info['category'].iloc[0] if not market_info.empty else 'unknown',
                'market_name': market_info['market_name'].iloc[0] if not market_info.empty else 'unknown',
                'events': events,
                'establishment_year': events[events['event_type'] == 'establishment']['year'].min() if not events.empty else None,
                'extinction_year': events[events['event_type'] == 'extinction']['year'].max() if not events.empty else None
            }
    
    def analyze_trajectories(self) -> Dict:
        """
        企業軌道分析の実行
        
        Returns:
            trajectory_analysis_results: 軌道分析結果
        """
        print("企業軌道分析を開始...")
        
        results = {
            'individual_trajectories': {},
            'category_patterns': {},
            'lifecycle_transitions': {},
            'trajectory_clusters': {}
        }
        
        # 1. 個別企業軌道分析
        results['individual_trajectories'] = self._analyze_individual_trajectories()
        
        # 2. 市場カテゴリー別パターン分析
        results['category_patterns'] = self._analyze_category_patterns()
        
        # 3. ライフサイクル遷移分析
        results['lifecycle_transitions'] = self._analyze_lifecycle_transitions()
        
        # 4. 軌道クラスタリング分析
        results['trajectory_clusters'] = self._perform_trajectory_clustering()
        
        self.trajectory_results = results
        print("企業軌道分析完了")
        
        return results
    
    def _analyze_individual_trajectories(self) -> Dict:
        """個別企業の軌道分析"""
        individual_results = {}
        
        for company_id, company_info in self.company_data.items():
            financial_data = company_info['financial_data']
            
            if len(financial_data) < self.config['min_data_points']:
                continue
            
            # 各評価項目の軌道計算
            trajectories = {}
            for metric in self.evaluation_metrics:
                if metric in financial_data.columns:
                    # データの平滑化
                    smoothed_data = self._smooth_trajectory(
                        financial_data[metric].values,
                        window=self.config['smoothing_window']
                    )
                    
                    # 軌道特徴量計算
                    trajectory_features = self._calculate_trajectory_features(
                        financial_data['year'].values,
                        smoothed_data
                    )
                    
                    trajectories[metric] = {
                        'raw_data': financial_data[metric].values,
                        'smoothed_data': smoothed_data,
                        'years': financial_data['year'].values,
                        'features': trajectory_features
                    }
            
            # ライフサイクル段階分析
            lifecycle_stages = self._determine_lifecycle_stages(
                company_info, trajectories
            )
            
            individual_results[company_id] = {
                'trajectories': trajectories,
                'lifecycle_stages': lifecycle_stages,
                'market_category': company_info['market_category'],
                'survival_years': len(financial_data),
                'establishment_year': company_info['establishment_year'],
                'extinction_year': company_info['extinction_year']
            }
        
        return individual_results
    
    def _smooth_trajectory(self, data: np.ndarray, window: int) -> np.ndarray:
        """軌道データの平滑化"""
        if len(data) < window:
            return data
        
        # 移動平均による平滑化
        smoothed = np.convolve(data, np.ones(window)/window, mode='same')
        
        # 端点処理
        smoothed[:window//2] = data[:window//2]
        smoothed[-window//2:] = data[-window//2:]
        
        return smoothed
    
    def _calculate_trajectory_features(self, years: np.ndarray, values: np.ndarray) -> Dict:
        """軌道特徴量の計算"""
        features = {}
        
        try:
            # 基本統計量
            features['mean'] = np.mean(values)
            features['std'] = np.std(values)
            features['min'] = np.min(values)
            features['max'] = np.max(values)
            features['range'] = features['max'] - features['min']
            
            # トレンド分析
            if len(years) > 1:
                slope, intercept, r_value, p_value, std_err = stats.linregress(years, values)
                features['trend_slope'] = slope
                features['trend_r_squared'] = r_value ** 2
                features['trend_p_value'] = p_value
                
                # 変化率分析
                growth_rates = np.diff(values) / values[:-1]
                growth_rates = growth_rates[np.isfinite(growth_rates)]
                
                if len(growth_rates) > 0:
                    features['avg_growth_rate'] = np.mean(growth_rates)
                    features['growth_volatility'] = np.std(growth_rates)
                    features['max_growth_rate'] = np.max(growth_rates)
                    features['min_growth_rate'] = np.min(growth_rates)
            
            # 変曲点検出
            if len(values) > 3:
                second_derivative = np.gradient(np.gradient(values))
                turning_points = np.where(np.abs(second_derivative) > np.std(second_derivative) * 2)[0]
                features['turning_points_count'] = len(turning_points)
                features['turning_points'] = turning_points.tolist()
            
        except Exception as e:
            print(f"軌道特徴量計算エラー: {e}")
            features = {'error': str(e)}
        
        return features
    
    def _determine_lifecycle_stages(self, company_info: Dict, trajectories: Dict) -> List[Dict]:
        """企業のライフサイクル段階判定"""
        financial_data = company_info['financial_data']
        stages = []
        
        if 'revenue_growth_rate' not in trajectories:
            return stages
        
        growth_rates = trajectories['revenue_growth_rate']['smoothed_data']
        years = trajectories['revenue_growth_rate']['years']
        
        # 各年のライフサイクル段階判定
        for i, (year, growth_rate) in enumerate(zip(years, growth_rates)):
            company_age = year - (company_info['establishment_year'] or year - len(years))
            
            stage = self._classify_lifecycle_stage(
                company_age, growth_rate, i, len(years),
                company_info['extinction_year'] is not None
            )
            
            stages.append({
                'year': year,
                'stage': stage,
                'company_age': company_age,
                'growth_rate': growth_rate
            })
        
        return stages
    
    def _classify_lifecycle_stage(self, age: int, growth_rate: float, 
                                position: int, total_years: int, is_extinct: bool) -> str:
        """ライフサイクル段階の分類"""
        thresholds = self.config['lifecycle_threshold']
        
        # 消滅企業の場合
        if is_extinct and position >= total_years * 0.8:
            return 'extinction'
        
        # 企業年齢に基づく初期判定
        if age <= thresholds['startup_years']:
            return 'startup'
        
        # 成長率に基づく判定
        if growth_rate >= thresholds['growth_revenue_growth']:
            return 'growth'
        elif growth_rate <= thresholds['decline_growth_threshold']:
            return 'decline'
        elif thresholds['maturity_growth_range'][0] <= growth_rate <= thresholds['maturity_growth_range'][1]:
            return 'maturity'
        else:
            return 'growth' if growth_rate > 0 else 'decline'
    
    def _analyze_category_patterns(self) -> Dict:
        """市場カテゴリー別軌道パターン分析"""
        category_results = {}
        
        for category in self.market_categories:
            category_companies = {
                cid: data for cid, data in self.trajectory_data.get('individual_trajectories', {}).items()
                if data['market_category'] == category
            }
            
            if not category_companies:
                continue
            
            # カテゴリー別統計分析
            category_stats = self._calculate_category_statistics(category_companies)
            
            # 代表的軌道パターン抽出
            representative_patterns = self._extract_representative_patterns(category_companies)
            
            # カテゴリー間比較
            comparison_results = self._compare_with_other_categories(
                category, category_companies
            )
            
            category_results[category] = {
                'statistics': category_stats,
                'representative_patterns': representative_patterns,
                'comparisons': comparison_results,
                'company_count': len(category_companies)
            }
        
        return category_results
    
    def _calculate_category_statistics(self, category_companies: Dict) -> Dict:
        """カテゴリー別統計量計算"""
        stats_results = {}
        
        for metric in self.evaluation_metrics:
            metric_data = []
            feature_data = []
            
            for company_data in category_companies.values():
                if metric in company_data['trajectories']:
                    traj = company_data['trajectories'][metric]
                    metric_data.extend(traj['smoothed_data'])
                    
                    # 軌道特徴量収集
                    features = traj['features']
                    feature_data.append(features)
            
            if metric_data:
                stats_results[metric] = {
                    'mean': np.mean(metric_data),
                    'std': np.std(metric_data),
                    'median': np.median(metric_data),
                    'q25': np.percentile(metric_data, 25),
                    'q75': np.percentile(metric_data, 75),
                    'count': len(metric_data)
                }
                
                # 特徴量統計
                if feature_data:
                    feature_stats = {}
                    for feature_key in feature_data[0].keys():
                        if feature_key != 'error' and feature_key not in ['turning_points']:
                            try:
                                feature_values = [f[feature_key] for f in feature_data 
                                                if feature_key in f and np.isfinite(f[feature_key])]
                                if feature_values:
                                    feature_stats[feature_key] = {
                                        'mean': np.mean(feature_values),
                                        'std': np.std(feature_values)
                                    }
                            except:
                                pass
                    
                    stats_results[metric]['features'] = feature_stats
        
        return stats_results
    
    def _extract_representative_patterns(self, category_companies: Dict) -> Dict:
        """代表的軌道パターンの抽出"""
        patterns = {}
        
        for metric in self.evaluation_metrics:
            metric_trajectories = []
            
            for company_data in category_companies.values():
                if metric in company_data['trajectories']:
                    traj_data = company_data['trajectories'][metric]['smoothed_data']
                    if len(traj_data) >= self.config['min_data_points']:
                        # 軌道を標準化して長さを統一
                        normalized_traj = self._normalize_trajectory_length(traj_data, 20)
                        metric_trajectories.append(normalized_traj)
            
            if len(metric_trajectories) >= 3:
                # K-meansクラスタリングによるパターン抽出
                trajectories_array = np.array(metric_trajectories)
                
                # クラスタリング実行
                n_clusters = min(3, len(metric_trajectories))
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                cluster_labels = kmeans.fit_predict(trajectories_array)
                
                # 各クラスターの代表軌道計算
                cluster_patterns = {}
                for i in range(n_clusters):
                    cluster_trajectories = trajectories_array[cluster_labels == i]
                    representative = np.mean(cluster_trajectories, axis=0)
                    confidence_interval = np.std(cluster_trajectories, axis=0)
                    
                    cluster_patterns[f'pattern_{i}'] = {
                        'representative': representative,
                        'confidence_interval': confidence_interval,
                        'count': np.sum(cluster_labels == i)
                    }
                
                patterns[metric] = cluster_patterns
        
        return patterns
    
    def _normalize_trajectory_length(self, trajectory: np.ndarray, target_length: int) -> np.ndarray:
        """軌道長を正規化"""
        if len(trajectory) == target_length:
            return trajectory
        
        # 線形補間により長さを統一
        original_indices = np.linspace(0, len(trajectory) - 1, len(trajectory))
        target_indices = np.linspace(0, len(trajectory) - 1, target_length)
        
        normalized = np.interp(target_indices, original_indices, trajectory)
        return normalized
    
    def _compare_with_other_categories(self, current_category: str, 
                                        current_companies: Dict) -> Dict:
        """他カテゴリーとの比較分析"""
        comparisons = {}
        
        for other_category in self.market_categories:
            if other_category == current_category:
                continue
            
            other_companies = {
                cid: data for cid, data in self.trajectory_data.get('individual_trajectories', {}).items()
                if data['market_category'] == other_category
            }
            
            if not other_companies:
                continue
            
            # 統計的比較テスト
            comparison_results = self._statistical_comparison(
                current_companies, other_companies
            )
            
            comparisons[other_category] = comparison_results
        
        return comparisons
    
    def _statistical_comparison(self, group1: Dict, group2: Dict) -> Dict:
        """統計的比較テスト"""
        results = {}
        
        for metric in self.evaluation_metrics:
            # 各グループのデータ収集
            data1 = []
            data2 = []
            
            for company_data in group1.values():
                if metric in company_data['trajectories']:
                    data1.extend(company_data['trajectories'][metric]['smoothed_data'])
            
            for company_data in group2.values():
                if metric in company_data['trajectories']:
                    data2.extend(company_data['trajectories'][metric]['smoothed_data'])
            
            if len(data1) >= 10 and len(data2) >= 10:
                # t検定実行
                t_stat, p_value = stats.ttest_ind(data1, data2)
                
                # 効果量計算 (Cohen's d)
                pooled_std = np.sqrt(((len(data1) - 1) * np.var(data1, ddof=1) + 
                                    (len(data2) - 1) * np.var(data2, ddof=1)) / 
                                    (len(data1) + len(data2) - 2))
                cohens_d = (np.mean(data1) - np.mean(data2)) / pooled_std
                
                results[metric] = {
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'cohens_d': cohens_d,
                    'significant': p_value < self.config['significance_level'],
                    'group1_mean': np.mean(data1),
                    'group2_mean': np.mean(data2),
                    'effect_size': 'small' if abs(cohens_d) < 0.5 else 
                                    'medium' if abs(cohens_d) < 0.8 else 'large'
                }
        
        return results
    
    def _analyze_lifecycle_transitions(self) -> Dict:
        """ライフサイクル遷移分析"""
        transition_results = {}
        
        # 市場カテゴリー別遷移行列計算
        for category in self.market_categories:
            category_transitions = []
            
            for company_data in self.trajectory_data.get('individual_trajectories', {}).values():
                if company_data['market_category'] == category:
                    stages = company_data['lifecycle_stages']
                    if len(stages) > 1:
                        stage_sequence = [s['stage'] for s in stages]
                        category_transitions.append(stage_sequence)
            
            if category_transitions:
                transition_matrix = self._calculate_transition_matrix(category_transitions)
                transition_results[category] = {
                    'transition_matrix': transition_matrix,
                    'average_duration': self._calculate_stage_durations(category_transitions),
                    'common_paths': self._identify_common_paths(category_transitions)
                }
        
        return transition_results
    
    def _calculate_transition_matrix(self, stage_sequences: List[List[str]]) -> Dict:
        """遷移行列の計算"""
        transitions = {}
        
        # 全ての遷移をカウント
        for sequence in stage_sequences:
            for i in range(len(sequence) - 1):
                current_stage = sequence[i]
                next_stage = sequence[i + 1]
                
                if current_stage not in transitions:
                    transitions[current_stage] = {}
                if next_stage not in transitions[current_stage]:
                    transitions[current_stage][next_stage] = 0
                
                transitions[current_stage][next_stage] += 1
        
        # 確率に変換
        transition_matrix = {}
        for current_stage, next_stages in transitions.items():
            total_transitions = sum(next_stages.values())
            transition_matrix[current_stage] = {
                next_stage: count / total_transitions
                for next_stage, count in next_stages.items()
            }
        
        return transition_matrix
    
    def _calculate_stage_durations(self, stage_sequences: List[List[str]]) -> Dict:
        """各段階の平均滞在期間計算"""
        stage_durations = {stage: [] for stage in self.lifecycle_stages}
        
        for sequence in stage_sequences:
            current_stage = sequence[0]
            duration = 1
            
            for i in range(1, len(sequence)):
                if sequence[i] == current_stage:
                    duration += 1
                else:
                    stage_durations[current_stage].append(duration)
                    current_stage = sequence[i]
                    duration = 1
            
            # 最後の段階
            stage_durations[current_stage].append(duration)
        
        # 平均期間計算
        average_durations = {}
        for stage, durations in stage_durations.items():
            if durations:
                average_durations[stage] = {
                    'mean': np.mean(durations),
                    'std': np.std(durations),
                    'count': len(durations)
                }
        
        return average_durations
    
    def _identify_common_paths(self, stage_sequences: List[List[str]]) -> List[Dict]:
        """一般的な遷移パスの特定"""
        # 共通するサブシーケンス（長さ3以上）を検出
        common_paths = {}
        
        for sequence in stage_sequences:
            for length in range(3, min(6, len(sequence) + 1)):
                for start in range(len(sequence) - length + 1):
                    subseq = tuple(sequence[start:start + length])
                    
                    if subseq not in common_paths:
                        common_paths[subseq] = 0
                    common_paths[subseq] += 1
        
        # 頻度順でソート
        sorted_paths = sorted(common_paths.items(), key=lambda x: x[1], reverse=True)
        
        # 上位パスを返す（最低3回は出現）
        return [
            {'path': list(path), 'frequency': freq, 'percentage': freq / len(stage_sequences)}
            for path, freq in sorted_paths[:10]
            if freq >= 3
        ]
    
    def _perform_trajectory_clustering(self) -> Dict:
        """軌道クラスタリング分析"""
        clustering_results = {}
        
        for metric in self.evaluation_metrics:
            # 全企業の軌道データ収集
            all_trajectories = []
            company_ids = []
            
            for company_id, company_data in self.trajectory_data.get('individual_trajectories', {}).items():
                if metric in company_data['trajectories']:
                    traj_data = company_data['trajectories'][metric]['smoothed_data']
                    if len(traj_data) >= self.config['min_data_points']:
                        normalized_traj = self._normalize_trajectory_length(traj_data, 20)
                        all_trajectories.append(normalized_traj)
                        company_ids.append(company_id)
            
            if len(all_trajectories) < self.config['cluster_count']:
                continue
            
            # クラスタリング実行
            trajectories_array = np.array(all_trajectories)
            
            # 主成分分析による次元削減
            pca = PCA(n_components=min(10, trajectories_array.shape[1]))
            trajectories_pca = pca.fit_transform(trajectories_array)
            
            # K-meansクラスタリング
            kmeans = KMeans(n_clusters=self.config['cluster_count'], random_state=42)
            cluster_labels = kmeans.fit_predict(trajectories_pca)
            
            # クラスター結果分析
            cluster_analysis = self._analyze_clusters(
                trajectories_array, cluster_labels, company_ids, metric
            )
            
            clustering_results[metric] = {
                'cluster_labels': cluster_labels,
                'cluster_centers': kmeans.cluster_centers_,
                'pca_explained_variance': pca.explained_variance_ratio_,
                'cluster_analysis': cluster_analysis
            }
        
        return clustering_results
    
    def _analyze_clusters(self, trajectories: np.ndarray, labels: np.ndarray, 
                            company_ids: List, metric: str) -> Dict:
        """クラスター分析"""
        cluster_analysis = {}
        
        for cluster_id in np.unique(labels):
            cluster_mask = labels == cluster_id
            cluster_trajectories = trajectories[cluster_mask]
            cluster_companies = [company_ids[i] for i in np.where(cluster_mask)[0]]
            
            # クラスターの統計分析
            representative_trajectory = np.mean(cluster_trajectories, axis=0)
            trajectory_std = np.std(cluster_trajectories, axis=0)
            
            # 市場カテゴリー分布
            category_distribution = {}
            for company_id in cluster_companies:
                company_data = self.trajectory_data.get('individual_trajectories', {}).get(company_id)
                if company_data:
                    category = company_data['market_category']
                    category_distribution[category] = category_distribution.get(category, 0) + 1
            
            cluster_analysis[cluster_id] = {
                'company_count': len(cluster_companies),
                'companies': cluster_companies,
                'representative_trajectory': representative_trajectory,
                'trajectory_std': trajectory_std,
                'category_distribution': category_distribution,
                'dominant_category': max(category_distribution, key=category_distribution.get) if category_distribution else None
            }
        
        return cluster_analysis
    
    def predict_future_trajectories(self, company_id: str, horizon: int = None) -> Dict:
        """将来軌道予測"""
        if horizon is None:
            horizon = self.config['prediction_horizon']
        
        if company_id not in self.trajectory_data.get('individual_trajectories', {}):
            raise ValueError(f"企業ID {company_id} のデータが見つかりません")
        
        company_data = self.trajectory_data['individual_trajectories'][company_id]
        predictions = {}
        
        for metric in self.evaluation_metrics:
            if metric in company_data['trajectories']:
                trajectory = company_data['trajectories'][metric]
                
                # 時系列予測モデル構築
                prediction_result = self._build_prediction_model(
                    trajectory['years'], trajectory['smoothed_data'], horizon
                )
                
                predictions[metric] = prediction_result
        
        return {
            'company_id': company_id,
            'predictions': predictions,
            'horizon_years': horizon,
            'base_data_years': len(company_data['trajectories'][list(company_data['trajectories'].keys())[0]]['years']),
            'market_category': company_data['market_category']
        }
    
    def _build_prediction_model(self, years: np.ndarray, values: np.ndarray, 
                                horizon: int) -> Dict:
        """予測モデル構築"""
        try:
            # 特徴量準備
            X = self._prepare_prediction_features(years, values)
            y = values[len(X) - len(values):]
            
            if len(X) < 5:  # 最小データ要件
                return {'error': 'データ不足'}
            
            # Random Forest回帰モデル
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)
            
            # 将来予測
            future_years = np.arange(years[-1] + 1, years[-1] + horizon + 1)
            future_predictions = []
            
            # 逐次予測
            current_data = values.copy()
            current_years = years.copy()
            
            for future_year in future_years:
                # 予測用特徴量作成
                pred_features = self._prepare_prediction_features(
                    current_years, current_data
                )[-1].reshape(1, -1)
                
                # 予測実行
                pred_value = model.predict(pred_features)[0]
                future_predictions.append(pred_value)
                
                # データ更新（次の予測のため）
                current_data = np.append(current_data, pred_value)
                current_years = np.append(current_years, future_year)
            
            # 予測区間計算（ブートストラップ法）
            confidence_intervals = self._calculate_prediction_intervals(
                model, X, y, horizon, confidence=0.95
            )
            
            return {
                'predicted_values': future_predictions,
                'predicted_years': future_years.tolist(),
                'confidence_intervals': confidence_intervals,
                'model_score': model.score(X, y),
                'feature_importance': dict(zip(
                    [f'feature_{i}' for i in range(X.shape[1])],
                    model.feature_importances_
                ))
            }
            
        except Exception as e:
            return {'error': f'予測モデル構築エラー: {str(e)}'}
    
    def _prepare_prediction_features(self, years: np.ndarray, values: np.ndarray) -> np.ndarray:
        """予測用特徴量準備"""
        features = []
        window_size = min(5, len(values) - 1)
        
        for i in range(window_size, len(values)):
            # ラグ特徴量
            lag_features = values[i-window_size:i]
            
            # トレンド特徴量
            trend_slope = (values[i-1] - values[i-window_size]) / window_size
            
            # 移動平均特徴量
            moving_avg = np.mean(values[i-window_size:i])
            
            # 変動性特徴量
            volatility = np.std(values[i-window_size:i])
            
            # 年次特徴量
            year_feature = years[i] - years[0]  # 経過年数
            
            # 特徴量統合
            feature_vector = np.concatenate([
                lag_features,
                [trend_slope, moving_avg, volatility, year_feature]
            ])
            
            features.append(feature_vector)
        
        return np.array(features)
    
    def _calculate_prediction_intervals(self, model: RandomForestRegressor, 
                                        X: np.ndarray, y: np.ndarray, 
                                        horizon: int, confidence: float = 0.95) -> List[Tuple]:
        """予測区間計算"""
        try:
            # ブートストラップサンプリング
            n_bootstrap = 100
            bootstrap_predictions = []
            
            for _ in range(n_bootstrap):
                # サンプリング
                sample_indices = np.random.choice(len(X), size=len(X), replace=True)
                X_sample = X[sample_indices]
                y_sample = y[sample_indices]
                
                # モデル学習
                bootstrap_model = RandomForestRegressor(n_estimators=50, random_state=None)
                bootstrap_model.fit(X_sample, y_sample)
                
                # 予測（単純化のため最後の特徴量を使用）
                last_features = X[-1].reshape(1, -1)
                pred = bootstrap_model.predict(last_features)[0]
                bootstrap_predictions.append(pred)
            
            # 信頼区間計算
            alpha = 1 - confidence
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100
            
            lower_bound = np.percentile(bootstrap_predictions, lower_percentile)
            upper_bound = np.percentile(bootstrap_predictions, upper_percentile)
            
            # 全ホライゾンに対して同じ区間を適用（簡略化）
            intervals = [(lower_bound, upper_bound) for _ in range(horizon)]
            
            return intervals
            
        except Exception as e:
            print(f"予測区間計算エラー: {e}")
            return [(None, None)] * horizon
    
    def generate_trajectory_report(self, output_path: str = None) -> Dict:
        """軌道分析レポート生成"""
        if not hasattr(self, 'trajectory_results'):
            raise ValueError("軌道分析が実行されていません")
        
        report = {
            'executive_summary': self._generate_executive_summary(),
            'market_category_analysis': self._generate_category_analysis_report(),
            'lifecycle_insights': self._generate_lifecycle_insights(),
            'trajectory_patterns': self._generate_trajectory_patterns_report(),
            'strategic_recommendations': self._generate_strategic_recommendations(),
            'technical_details': {
                'data_summary': self._generate_data_summary(),
                'methodology': self._generate_methodology_summary(),
                'limitations': self._generate_limitations()
            }
        }
        
        # レポート出力
        if output_path:
            self._save_report(report, output_path)
        
        return report
    
    def _generate_executive_summary(self) -> Dict:
        """エグゼクティブサマリー生成"""
        individual_results = self.trajectory_results.get('individual_trajectories', {})
        
        # 基本統計
        total_companies = len(individual_results)
        category_counts = {}
        total_years_analyzed = 0
        
        for company_data in individual_results.values():
            category = company_data['market_category']
            category_counts[category] = category_counts.get(category, 0) + 1
            total_years_analyzed += company_data['survival_years']
        
        # 主要発見
        key_findings = []
        
        # カテゴリー別比較結果から主要な発見を抽出
        category_patterns = self.trajectory_results.get('category_patterns', {})
        for category, patterns in category_patterns.items():
            if 'comparisons' in patterns:
                for other_category, comparison in patterns['comparisons'].items():
                    # 統計的に有意な差異を検出
                    significant_metrics = [
                        metric for metric, results in comparison.items()
                        if results.get('significant', False)
                    ]
                    
                    if significant_metrics:
                        key_findings.append({
                            'finding': f'{category}市場企業は{other_category}市場企業と比較して{", ".join(significant_metrics)}で統計的に有意な差異',
                            'categories_compared': [category, other_category],
                            'significant_metrics': significant_metrics
                        })
        
        return {
            'analysis_scope': {
                'total_companies': total_companies,
                'category_distribution': category_counts,
                'total_company_years': total_years_analyzed,
                'analysis_period': '1984-2024'
            },
            'key_findings': key_findings[:5],  # 上位5つの発見
            'overall_insights': {
                'most_volatile_category': self._identify_most_volatile_category(),
                'most_stable_category': self._identify_most_stable_category(),
                'common_lifecycle_pattern': self._identify_common_lifecycle_pattern()
            }
        }
    
    def _identify_most_volatile_category(self) -> str:
        """最も変動の大きい市場カテゴリー特定"""
        category_patterns = self.trajectory_results.get('category_patterns', {})
        max_volatility = 0
        most_volatile = None
        
        for category, patterns in category_patterns.items():
            if 'statistics' in patterns:
                avg_volatility = 0
                metric_count = 0
                
                for metric, stats in patterns['statistics'].items():
                    if 'std' in stats:
                        avg_volatility += stats['std']
                        metric_count += 1
                
                if metric_count > 0:
                    avg_volatility /= metric_count
                    if avg_volatility > max_volatility:
                        max_volatility = avg_volatility
                        most_volatile = category
        
        return most_volatile or 'unknown'
    
    def _identify_most_stable_category(self) -> str:
        """最も安定した市場カテゴリー特定"""
        category_patterns = self.trajectory_results.get('category_patterns', {})
        min_volatility = float('inf')
        most_stable = None
        
        for category, patterns in category_patterns.items():
            if 'statistics' in patterns:
                avg_volatility = 0
                metric_count = 0
                
                for metric, stats in patterns['statistics'].items():
                    if 'std' in stats:
                        avg_volatility += stats['std']
                        metric_count += 1
                
                if metric_count > 0:
                    avg_volatility /= metric_count
                    if avg_volatility < min_volatility:
                        min_volatility = avg_volatility
                        most_stable = category
        
        return most_stable or 'unknown'
    
    def _identify_common_lifecycle_pattern(self) -> str:
        """共通ライフサイクルパターン特定"""
        lifecycle_results = self.trajectory_results.get('lifecycle_transitions', {})
        
        # 全カテゴリーから最も頻出する遷移パスを特定
        all_common_paths = []
        for category_data in lifecycle_results.values():
            if 'common_paths' in category_data:
                all_common_paths.extend(category_data['common_paths'])
        
        if all_common_paths:
            # 頻度順でソート
            sorted_paths = sorted(all_common_paths, key=lambda x: x['frequency'], reverse=True)
            if sorted_paths:
                return ' → '.join(sorted_paths[0]['path'])
        
        return 'startup → growth → maturity'
    
    def _generate_category_analysis_report(self) -> Dict:
        """市場カテゴリー分析レポート生成"""
        category_patterns = self.trajectory_results.get('category_patterns', {})
        report = {}
        
        for category, patterns in category_patterns.items():
            category_report = {
                'company_count': patterns.get('company_count', 0),
                'performance_characteristics': {},
                'trajectory_patterns': {},
                'competitive_position': {}
            }
            
            # 性能特性分析
            if 'statistics' in patterns:
                for metric, stats in patterns['statistics'].items():
                    if metric in self.evaluation_metrics:
                        category_report['performance_characteristics'][metric] = {
                            'average_performance': stats.get('mean', 0),
                            'volatility': stats.get('std', 0),
                            'performance_range': (stats.get('q25', 0), stats.get('q75', 0))
                        }
            
            # 軌道パターン
            if 'representative_patterns' in patterns:
                category_report['trajectory_patterns'] = patterns['representative_patterns']
            
            # 競合ポジション
            if 'comparisons' in patterns:
                significant_advantages = []
                significant_disadvantages = []
                
                for other_category, comparison in patterns['comparisons'].items():
                    for metric, result in comparison.items():
                        if result.get('significant', False):
                            if result['group1_mean'] > result['group2_mean']:
                                significant_advantages.append({
                                    'metric': metric,
                                    'vs_category': other_category,
                                    'advantage_magnitude': result['cohens_d']
                                })
                            else:
                                significant_disadvantages.append({
                                    'metric': metric,
                                    'vs_category': other_category,
                                    'disadvantage_magnitude': result['cohens_d']
                                })
                
                category_report['competitive_position'] = {
                    'significant_advantages': significant_advantages,
                    'significant_disadvantages': significant_disadvantages
                }
            
            report[category] = category_report
        
        return report
    
    def _generate_lifecycle_insights(self) -> Dict:
        """ライフサイクル洞察生成"""
        lifecycle_results = self.trajectory_results.get('lifecycle_transitions', {})
        insights = {}
        
        for category, transitions in lifecycle_results.items():
            category_insights = {}
            
            # 平均滞在期間分析
            if 'average_duration' in transitions:
                stage_durations = transitions['average_duration']
                longest_stage = max(stage_durations.keys(), 
                                    key=lambda x: stage_durations[x].get('mean', 0))
                shortest_stage = min(stage_durations.keys(), 
                                    key=lambda x: stage_durations[x].get('mean', float('inf')))
                
                category_insights['stage_duration_analysis'] = {
                    'longest_stage': longest_stage,
                    'shortest_stage': shortest_stage,
                    'stage_durations': {k: v['mean'] for k, v in stage_durations.items()}
                }
            
            # 遷移確率分析
            if 'transition_matrix' in transitions:
                transition_matrix = transitions['transition_matrix']
                high_probability_transitions = []
                
                for from_stage, to_stages in transition_matrix.items():
                    max_prob_stage = max(to_stages.keys(), key=lambda x: to_stages[x])
                    max_prob = to_stages[max_prob_stage]
                    
                    if max_prob > 0.5:  # 50%以上の確率
                        high_probability_transitions.append({
                            'from': from_stage,
                            'to': max_prob_stage,
                            'probability': max_prob
                        })
                
                category_insights['transition_patterns'] = {
                    'high_probability_transitions': high_probability_transitions,
                    'transition_matrix': transition_matrix
                }
            
            # 共通パス分析
            if 'common_paths' in transitions:
                category_insights['common_pathways'] = transitions['common_paths'][:3]
            
            insights[category] = category_insights
        
        return insights
    
    def _generate_trajectory_patterns_report(self) -> Dict:
        """軌道パターンレポート生成"""
        clustering_results = self.trajectory_results.get('trajectory_clusters', {})
        patterns_report = {}
        
        for metric, clustering in clustering_results.items():
            if 'cluster_analysis' in clustering:
                cluster_analysis = clustering['cluster_analysis']
                
                metric_patterns = []
                for cluster_id, cluster_info in cluster_analysis.items():
                    pattern_description = {
                        'cluster_id': cluster_id,
                        'company_count': cluster_info['company_count'],
                        'dominant_market_category': cluster_info.get('dominant_category'),
                        'category_distribution': cluster_info['category_distribution'],
                        'trajectory_characteristics': self._describe_trajectory_pattern(
                            cluster_info['representative_trajectory']
                        )
                    }
                    metric_patterns.append(pattern_description)
                
                patterns_report[metric] = {
                    'identified_patterns': len(metric_patterns),
                    'pattern_details': metric_patterns
                }
        
        return patterns_report
    
    def _describe_trajectory_pattern(self, trajectory: np.ndarray) -> Dict:
        """軌道パターンの特徴記述"""
        if len(trajectory) < 2:
            return {'description': 'insufficient_data'}
        
        # トレンド分析
        start_value = trajectory[0]
        end_value = trajectory[-1]
        overall_change = end_value - start_value
        
        # 変動パターン
        peaks = len([i for i in range(1, len(trajectory)-1) 
                    if trajectory[i] > trajectory[i-1] and trajectory[i] > trajectory[i+1]])
        troughs = len([i for i in range(1, len(trajectory)-1) 
                        if trajectory[i] < trajectory[i-1] and trajectory[i] < trajectory[i+1]])
        
        # パターン分類
        if abs(overall_change) < np.std(trajectory) * 0.5:
            trend_type = 'stable'
        elif overall_change > 0:
            trend_type = 'increasing'
        else:
            trend_type = 'decreasing'
        
        volatility_level = 'high' if np.std(trajectory) > np.mean(trajectory) * 0.3 else 'low'
        
        return {
            'trend_type': trend_type,
            'volatility_level': volatility_level,
            'overall_change_magnitude': abs(overall_change),
            'peaks_count': peaks,
            'troughs_count': troughs,
            'cyclical_behavior': 'yes' if peaks > 1 and troughs > 1 else 'no'
        }
    
    def _generate_strategic_recommendations(self) -> Dict:
        """戦略的推奨事項生成"""
        recommendations = {
            'high_share_markets': [],
            'declining_markets': [],
            'lost_markets': [],
            'general_insights': []
        }
        
        category_patterns = self.trajectory_results.get('category_patterns', {})
        
        # 各カテゴリーに対する推奨事項
        for category, patterns in category_patterns.items():
            category_key = f"{category}_markets"
            if category_key not in recommendations:
                continue
                
            if 'statistics' in patterns:
                # 高成長メトリクス特定
                high_performance_metrics = []
                for metric, stats in patterns['statistics'].items():
                    if stats.get('mean', 0) > 0.1:  # 閾値は調整可能
                        high_performance_metrics.append(metric)
                
                if high_performance_metrics:
                    recommendations[category_key].append({
                        'recommendation': f'{category}市場では{", ".join(high_performance_metrics)}の維持・向上が重要',
                        'basis': 'statistical_analysis',
                        'priority': 'high'
                    })
            
            # 比較分析から推奨事項抽出
            if 'comparisons' in patterns:
                for other_category, comparison in patterns['comparisons'].items():
                    disadvantage_metrics = []
                    for metric, result in comparison.items():
                        if (result.get('significant', False) and 
                            result['group1_mean'] < result['group2_mean']):
                            disadvantage_metrics.append(metric)
                    
                    if disadvantage_metrics:
                        recommendations[category_key].append({
                            'recommendation': f'{other_category}市場に学び、{", ".join(disadvantage_metrics)}の改善を検討',
                            'basis': 'comparative_analysis',
                            'priority': 'medium'
                        })
        
        # 一般的洞察
        recommendations['general_insights'] = [
            {
                'insight': 'ライフサイクル段階に応じた戦略的フォーカスの調整が重要',
                'supporting_evidence': '段階別平均滞在期間分析結果'
            },
            {
                'insight': '市場カテゴリー間での軌道パターンの違いは戦略的差別化機会を示唆',
                'supporting_evidence': 'クラスタリング分析結果'
            }
        ]
        
        return recommendations
    
    def _generate_data_summary(self) -> Dict:
        """データサマリー生成"""
        individual_results = self.trajectory_results.get('individual_trajectories', {})
        
        return {
            'total_companies_analyzed': len(individual_results),
            'metrics_analyzed': len(self.evaluation_metrics),
            'analysis_period': '1984-2024',
            'data_completeness': self._calculate_data_completeness(),
            'quality_indicators': self._calculate_quality_indicators()
        }
    
    def _calculate_data_completeness(self) -> Dict:
        """データ完全性計算"""
        individual_results = self.trajectory_results.get('individual_trajectories', {})
        completeness = {}
        
        for metric in self.evaluation_metrics:
            companies_with_metric = sum(
                1 for company_data in individual_results.values()
                if metric in company_data.get('trajectories', {})
            )
            completeness[metric] = {
                'companies_with_data': companies_with_metric,
                'completeness_ratio': companies_with_metric / len(individual_results) if individual_results else 0
            }
        
        return completeness
    
    def _calculate_quality_indicators(self) -> Dict:
        """品質指標計算"""
        individual_results = self.trajectory_results.get('individual_trajectories', {})
        
        total_data_points = 0
        companies_with_sufficient_data = 0
        
        for company_data in individual_results.values():
            company_data_points = sum(
                len(traj_data.get('raw_data', []))
                for traj_data in company_data.get('trajectories', {}).values()
            )
            total_data_points += company_data_points
            
            if company_data_points >= self.config['min_data_points'] * len(self.evaluation_metrics):
                companies_with_sufficient_data += 1
        
        return {
            'total_data_points': total_data_points,
            'average_data_points_per_company': total_data_points / len(individual_results) if individual_results else 0,
            'companies_with_sufficient_data': companies_with_sufficient_data,
            'sufficient_data_ratio': companies_with_sufficient_data / len(individual_results) if individual_results else 0
        }
    
    def _generate_methodology_summary(self) -> Dict:
        """方法論サマリー生成"""
        return {
            'trajectory_analysis_methods': [
                'Moving average smoothing',
                'Statistical feature extraction',
                'Lifecycle stage classification'
            ],
            'comparative_analysis_methods': [
                'Two-sample t-tests',
                'Effect size calculation (Cohen\'s d)',
                'Statistical significance testing'
            ],
            'clustering_methods': [
                'K-means clustering',
                'Principal Component Analysis (PCA)',
                'Trajectory normalization'
            ],
            'prediction_methods': [
                'Random Forest regression',
                'Bootstrap confidence intervals',
                'Sequential prediction approach'
            ],
            'parameters_used': {
                'smoothing_window': self.config['smoothing_window'],
                'cluster_count': self.config['cluster_count'],
                'prediction_horizon': self.config['prediction_horizon'],
                'significance_level': self.config['significance_level']
            }
        }
    
    def _generate_limitations(self) -> List[str]:
        """分析制限事項生成"""
        return [
            'データ可用性: 一部企業では完全な40年データが取得できない可能性',
            'サバイバーバイアス: 消滅企業データの不完全性が結果に影響する可能性',
            '外部要因: マクロ経済環境や業界固有要因の影響を完全には調整していない',
            '予測精度: 将来予測は過去パターンに基づくため、構造変化を予見できない可能性',
            'データ品質: 会計基準変更や企業再編による時系列データの不連続性',
            '統計的仮定: 正規分布を仮定した統計検定の適用限界',
            'クラスタリング: K-meansの球状クラスター仮定が実際のパターンと異なる可能性'
        ]
    
    def _save_report(self, report: Dict, output_path: str) -> None:
        """レポート保存"""
        import json
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"レポートが保存されました: {output_path}")


# 使用例とテスト関数
def example_usage():
    """使用例"""
    
    # 設定例
    config = {
        'min_data_points': 10,
        'smoothing_window': 5,
        'cluster_count': 4,
        'prediction_horizon': 3,
        'significance_level': 0.05
    }
    
    # アナライザー初期化
    analyzer = LifecycleTrajectoryAnalyzer(config)
    
    # サンプルデータ作成（実際の使用では実データを読み込み）
    sample_financial_data = pd.DataFrame({
        'company_id': ['FANUC'] * 40 + ['OLYMPUS'] * 35,
        'year': list(range(1984, 2024)) + list(range(1989, 2024)),
        'revenue': np.random.lognormal(10, 0.3, 75),
        'revenue_growth_rate': np.random.normal(0.05, 0.1, 75),
        'operating_margin': np.random.normal(0.15, 0.05, 75),
        'net_margin': np.random.normal(0.10, 0.04, 75),
        'roe': np.random.normal(0.12, 0.06, 75),
        'value_added_ratio': np.random.normal(0.35, 0.08, 75)
    })
    
    sample_market_data = pd.DataFrame({
        'company_id': ['FANUC', 'OLYMPUS'],
        'category': ['high_share', 'high_share'],
        'market_name': ['robot', 'endoscope']
    })
    
    sample_events = pd.DataFrame({
        'company_id': ['FANUC', 'OLYMPUS'],
        'event_type': ['establishment', 'establishment'],
        'year': [1972, 1919]
    })
    
    # データ読み込み
    analyzer.load_data(sample_financial_data, sample_market_data, sample_events)
    
    # 軌道分析実行
    results = analyzer.analyze_trajectories()
    
    # レポート生成
    report = analyzer.generate_trajectory_report('trajectory_analysis_report.json')
    
    print("軌道分析完了")
    print(f"分析企業数: {len(results['individual_trajectories'])}")
    
    return analyzer, results, report


if __name__ == "__main__":
    # 使用例実行
    analyzer, results, report = example_usage()