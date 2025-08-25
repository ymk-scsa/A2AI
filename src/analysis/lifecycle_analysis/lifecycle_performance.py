"""
A2AI - Advanced Financial Analysis AI
ライフサイクル別性能分析モジュール

企業のライフサイクル段階（設立・成長・成熟・衰退・再生・消滅）別に
財務性能を分析し、各段階での特徴的なパフォーマンスパターンを特定する。

Author: A2AI Development Team
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

from ...utils.lifecycle_utils import LifecycleUtilities
from ...utils.statistical_utils import StatisticalUtilities
from ...feature_engineering.lifecycle_features import LifecycleFeatureExtractor

class LifecyclePerformanceAnalyzer:
    """
    企業ライフサイクル別性能分析クラス
    
    各企業のライフサイクル段階を特定し、段階別の財務性能パターンを分析する。
    9つの評価項目×各23要因項目について、ライフサイクル段階別の特徴を抽出。
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初期化
        
        Args:
            config: 設定辞書（ライフサイクル段階定義、閾値等）
        """
        self.config = config or self._default_config()
        self.lifecycle_utils = LifecycleUtilities()
        self.stat_utils = StatisticalUtilities()
        self.feature_extractor = LifecycleFeatureExtractor()
        
        # ライフサイクル段階定義
        self.lifecycle_stages = {
            'emergence': 'エマージェンス期（0-5年）',
            'growth': '成長期（6-15年）', 
            'maturity': '成熟期（16-30年）',
            'decline': '衰退期（31年以上）',
            'renewal': '再生期（再編・復活）',
            'extinction': '消滅期（倒産・撤退）'
        }
        
        # 9つの評価項目定義
        self.evaluation_metrics = [
            'sales_revenue',           # 売上高
            'sales_growth_rate',       # 売上高成長率
            'operating_margin',        # 売上高営業利益率
            'net_margin',             # 売上高当期純利益率
            'roe',                    # ROE
            'value_added_ratio',      # 売上高付加価値率
            'survival_probability',    # 企業存続確率
            'emergence_success_rate',  # 新規事業成功率
            'succession_success_rate'  # 事業継承成功度
        ]
        
        # 結果格納用
        self.performance_results = {}
        self.stage_characteristics = {}
        self.transition_patterns = {}
    
    def _default_config(self) -> Dict[str, Any]:
        """デフォルト設定を返す"""
        return {
            'age_thresholds': {
                'emergence': (0, 5),
                'growth': (6, 15),
                'maturity': (16, 30),
                'decline': (31, np.inf)
            },
            'performance_windows': [1, 3, 5, 10],  # 年数
            'min_observations': 20,  # 統計分析に必要な最小観測数
            'significance_level': 0.05,
            'clustering_params': {
                'n_clusters_range': (2, 8),
                'random_state': 42
            }
        }
    
    def analyze_lifecycle_performance(self, 
                                    financial_data: pd.DataFrame,
                                    company_info: pd.DataFrame,
                                    market_categories: pd.DataFrame) -> Dict[str, Any]:
        """
        ライフサイクル別性能分析のメイン実行関数
        
        Args:
            financial_data: 財務データ（150社×40年分）
            company_info: 企業情報（設立年、業界、現状等）
            market_categories: 市場カテゴリ情報（高シェア/低下/失失）
        
        Returns:
            分析結果辞書
        """
        print("ライフサイクル別性能分析を開始...")
        
        # 1. データ前処理とライフサイクル段階特定
        processed_data = self._preprocess_data(financial_data, company_info, market_categories)
        
        # 2. 企業年齢とライフサイクル段階の計算
        stage_data = self._calculate_lifecycle_stages(processed_data)
        
        # 3. 段階別性能パターン分析
        stage_patterns = self._analyze_stage_patterns(stage_data)
        
        # 4. 段階間遷移分析
        transition_analysis = self._analyze_stage_transitions(stage_data)
        
        # 5. 市場カテゴリ別比較分析
        market_comparison = self._analyze_market_category_differences(stage_data)
        
        # 6. 要因項目影響度分析
        factor_impact = self._analyze_factor_impact_by_stage(stage_data)
        
        # 7. クラスタリングによるパターン発見
        clustering_results = self._discover_performance_clusters(stage_data)
        
        # 8. 統計的有意性検定
        significance_tests = self._perform_significance_tests(stage_data)
        
        # 結果統合
        results = {
            'stage_patterns': stage_patterns,
            'transition_analysis': transition_analysis,
            'market_comparison': market_comparison,
            'factor_impact': factor_impact,
            'clustering_results': clustering_results,
            'significance_tests': significance_tests,
            'summary_statistics': self._generate_summary_statistics(stage_data),
            'metadata': self._generate_metadata(processed_data)
        }
        
        self.performance_results = results
        print("ライフサイクル別性能分析が完了しました。")
        
        return results
    
    def _preprocess_data(self, financial_data: pd.DataFrame, 
                        company_info: pd.DataFrame,
                        market_categories: pd.DataFrame) -> pd.DataFrame:
        """データ前処理"""
        print("データ前処理中...")
        
        # データマージ
        merged_data = financial_data.merge(company_info, on='company_id', how='left')
        merged_data = merged_data.merge(market_categories, on='company_id', how='left')
        
        # 企業年齢計算
        merged_data['company_age'] = merged_data['year'] - merged_data['establishment_year']
        
        # 欠損値処理
        merged_data = self._handle_missing_values(merged_data)
        
        # 外れ値処理
        merged_data = self._handle_outliers(merged_data)
        
        return merged_data
    
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """欠損値処理"""
        # 評価項目の欠損値補完
        for metric in self.evaluation_metrics:
            if metric in data.columns:
                # 業界中央値で補完
                data[metric] = data.groupby('industry')[metric].transform(
                    lambda x: x.fillna(x.median())
                )
        
        return data
    
    def _handle_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """外れ値処理（IQRベース）"""
        for metric in self.evaluation_metrics:
            if metric in data.columns:
                Q1 = data[metric].quantile(0.25)
                Q3 = data[metric].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # 外れ値をキャップ処理
                data[metric] = np.clip(data[metric], lower_bound, upper_bound)
        
        return data
    
    def _calculate_lifecycle_stages(self, data: pd.DataFrame) -> pd.DataFrame:
        """企業年齢からライフサイクル段階を計算"""
        print("ライフサイクル段階を計算中...")
        
        def determine_stage(row):
            age = row['company_age']
            status = row.get('company_status', 'active')
            
            # 特殊ケース
            if status == 'extinct' or status == 'bankrupt':
                return 'extinction'
            elif row.get('recent_merger', False) or row.get('recent_spinoff', False):
                return 'renewal'
            
            # 年齢ベース判定
            if age <= 5:
                return 'emergence'
            elif age <= 15:
                return 'growth'
            elif age <= 30:
                return 'maturity'
            else:
                return 'decline'
        
        data['lifecycle_stage'] = data.apply(determine_stage, axis=1)
        
        return data
    
    def _analyze_stage_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """段階別性能パターン分析"""
        print("段階別性能パターンを分析中...")
        
        patterns = {}
        
        for stage in self.lifecycle_stages.keys():
            stage_data = data[data['lifecycle_stage'] == stage]
            
            if len(stage_data) < self.config['min_observations']:
                continue
            
            stage_analysis = {}
            
            # 各評価項目の統計量計算
            for metric in self.evaluation_metrics:
                if metric in stage_data.columns:
                    metric_stats = {
                        'mean': stage_data[metric].mean(),
                        'median': stage_data[metric].median(),
                        'std': stage_data[metric].std(),
                        'min': stage_data[metric].min(),
                        'max': stage_data[metric].max(),
                        'quartiles': [
                            stage_data[metric].quantile(0.25),
                            stage_data[metric].quantile(0.5),
                            stage_data[metric].quantile(0.75)
                        ]
                    }
                    stage_analysis[metric] = metric_stats
            
            # 市場カテゴリ別分布
            market_dist = stage_data['market_category'].value_counts(normalize=True).to_dict()
            stage_analysis['market_distribution'] = market_dist
            
            # 業界別分布
            industry_dist = stage_data['industry'].value_counts(normalize=True).to_dict()
            stage_analysis['industry_distribution'] = industry_dist
            
            # サンプルサイズ
            stage_analysis['sample_size'] = len(stage_data)
            stage_analysis['years_covered'] = stage_data['year'].nunique()
            stage_analysis['companies_covered'] = stage_data['company_id'].nunique()
            
            patterns[stage] = stage_analysis
        
        return patterns
    
    def _analyze_stage_transitions(self, data: pd.DataFrame) -> Dict[str, Any]:
        """段階間遷移分析"""
        print("段階間遷移を分析中...")
        
        # 企業別時系列データで遷移パターンを分析
        company_transitions = {}
        
        for company_id in data['company_id'].unique():
            company_data = data[data['company_id'] == company_id].sort_values('year')
            
            if len(company_data) < 3:  # 最低3年分のデータが必要
                continue
            
            # 段階変化の検出
            stage_sequence = company_data['lifecycle_stage'].tolist()
            transitions = []
            
            for i in range(1, len(stage_sequence)):
                if stage_sequence[i] != stage_sequence[i-1]:
                    transitions.append({
                        'from_stage': stage_sequence[i-1],
                        'to_stage': stage_sequence[i],
                        'year': company_data.iloc[i]['year'],
                        'company_age': company_data.iloc[i]['company_age']
                    })
            
            if transitions:
                company_transitions[company_id] = transitions
        
        # 遷移パターン集計
        transition_matrix = self._build_transition_matrix(company_transitions)
        transition_timing = self._analyze_transition_timing(company_transitions)
        transition_performance = self._analyze_transition_performance(data, company_transitions)
        
        return {
            'transition_matrix': transition_matrix,
            'transition_timing': transition_timing,
            'transition_performance': transition_performance,
            'company_transitions': company_transitions
        }
    
    def _build_transition_matrix(self, company_transitions: Dict) -> pd.DataFrame:
        """遷移行列の構築"""
        stages = list(self.lifecycle_stages.keys())
        matrix = pd.DataFrame(0, index=stages, columns=stages)
        
        for transitions in company_transitions.values():
            for trans in transitions:
                from_stage = trans['from_stage']
                to_stage = trans['to_stage']
                if from_stage in stages and to_stage in stages:
                    matrix.loc[from_stage, to_stage] += 1
        
        # 正規化（行ごとの確率に変換）
        row_sums = matrix.sum(axis=1)
        normalized_matrix = matrix.div(row_sums, axis=0).fillna(0)
        
        return normalized_matrix
    
    def _analyze_transition_timing(self, company_transitions: Dict) -> Dict[str, Any]:
        """遷移タイミング分析"""
        timing_analysis = {}
        
        for from_stage in self.lifecycle_stages.keys():
            for to_stage in self.lifecycle_stages.keys():
                if from_stage == to_stage:
                    continue
                
                transition_ages = []
                transition_years = []
                
                for transitions in company_transitions.values():
                    for trans in transitions:
                        if trans['from_stage'] == from_stage and trans['to_stage'] == to_stage:
                            transition_ages.append(trans['company_age'])
                            transition_years.append(trans['year'])
                
                if transition_ages:
                    timing_analysis[f"{from_stage}_to_{to_stage}"] = {
                        'mean_age': np.mean(transition_ages),
                        'median_age': np.median(transition_ages),
                        'std_age': np.std(transition_ages),
                        'count': len(transition_ages),
                        'years_range': (min(transition_years), max(transition_years))
                    }
        
        return timing_analysis
    
    def _analyze_transition_performance(self, data: pd.DataFrame, 
                                        company_transitions: Dict) -> Dict[str, Any]:
        """遷移前後の性能変化分析"""
        performance_changes = {}
        
        for metric in self.evaluation_metrics:
            if metric not in data.columns:
                continue
            
            metric_changes = []
            
            for company_id, transitions in company_transitions.items():
                company_data = data[data['company_id'] == company_id].sort_values('year')
                
                for trans in transitions:
                    trans_year = trans['year']
                    
                    # 遷移前後のデータを取得
                    before_data = company_data[company_data['year'] < trans_year]
                    after_data = company_data[company_data['year'] >= trans_year]
                    
                    if len(before_data) >= 1 and len(after_data) >= 1:
                        before_value = before_data[metric].iloc[-1]  # 直前の値
                        after_value = after_data[metric].iloc[0]     # 直後の値
                        
                        if not (np.isnan(before_value) or np.isnan(after_value)):
                            change_rate = (after_value - before_value) / before_value if before_value != 0 else 0
                            metric_changes.append({
                                'from_stage': trans['from_stage'],
                                'to_stage': trans['to_stage'],
                                'before_value': before_value,
                                'after_value': after_value,
                                'change_rate': change_rate,
                                'company_id': company_id
                            })
            
            performance_changes[metric] = metric_changes
        
        return performance_changes
    
    def _analyze_market_category_differences(self, data: pd.DataFrame) -> Dict[str, Any]:
        """市場カテゴリ別比較分析"""
        print("市場カテゴリ別比較を分析中...")
        
        market_analysis = {}
        
        for market_cat in ['high_share', 'declining_share', 'lost_share']:
            market_data = data[data['market_category'] == market_cat]
            
            if len(market_data) == 0:
                continue
            
            category_analysis = {}
            
            # 各ライフサイクル段階での性能分析
            for stage in self.lifecycle_stages.keys():
                stage_market_data = market_data[market_data['lifecycle_stage'] == stage]
                
                if len(stage_market_data) < self.config['min_observations']:
                    continue
                
                stage_metrics = {}
                for metric in self.evaluation_metrics:
                    if metric in stage_market_data.columns:
                        stage_metrics[metric] = {
                            'mean': stage_market_data[metric].mean(),
                            'median': stage_market_data[metric].median(),
                            'std': stage_market_data[metric].std()
                        }
                
                category_analysis[stage] = stage_metrics
            
            # ライフサイクル段階分布
            stage_distribution = market_data['lifecycle_stage'].value_counts(normalize=True).to_dict()
            category_analysis['stage_distribution'] = stage_distribution
            
            market_analysis[market_cat] = category_analysis
        
        return market_analysis
    
    def _analyze_factor_impact_by_stage(self, data: pd.DataFrame) -> Dict[str, Any]:
        """段階別要因項目影響度分析"""
        print("段階別要因項目影響度を分析中...")
        
        # 要因項目列を特定（evaluation_metricsではない列）
        factor_columns = [col for col in data.columns 
                            if col not in self.evaluation_metrics 
                            and col not in ['company_id', 'year', 'lifecycle_stage', 'market_category', 'industry']
                            and data[col].dtype in ['int64', 'float64']]
        
        factor_impact = {}
        
        for stage in self.lifecycle_stages.keys():
            stage_data = data[data['lifecycle_stage'] == stage]
            
            if len(stage_data) < self.config['min_observations']:
                continue
            
            stage_impact = {}
            
            for eval_metric in self.evaluation_metrics:
                if eval_metric not in stage_data.columns:
                    continue
                
                metric_impact = {}
                
                for factor in factor_columns[:23]:  # 各評価項目23要因項目
                    if factor in stage_data.columns:
                        # 相関係数計算
                        correlation = stage_data[eval_metric].corr(stage_data[factor])
                        
                        if not np.isnan(correlation):
                            # 統計的有意性検定
                            stat, p_value = stats.pearsonr(stage_data[eval_metric].dropna(), 
                                                            stage_data[factor].dropna())
                            
                            metric_impact[factor] = {
                                'correlation': correlation,
                                'p_value': p_value,
                                'significant': p_value < self.config['significance_level']
                            }
                
                stage_impact[eval_metric] = metric_impact
            
            factor_impact[stage] = stage_impact
        
        return factor_impact
    
    def _discover_performance_clusters(self, data: pd.DataFrame) -> Dict[str, Any]:
        """クラスタリングによるパフォーマンスパターン発見"""
        print("パフォーマンスクラスターを発見中...")
        
        clustering_results = {}
        
        # 数値データのみ選択
        numeric_cols = [col for col in self.evaluation_metrics if col in data.columns]
        clustering_data = data[numeric_cols].dropna()
        
        if len(clustering_data) < self.config['min_observations']:
            return {'error': 'Insufficient data for clustering'}
        
        # データ標準化
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(clustering_data)
        
        # 最適クラスター数決定
        silhouette_scores = []
        k_range = range(self.config['clustering_params']['n_clusters_range'][0], 
                        self.config['clustering_params']['n_clusters_range'][1] + 1)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, 
                            random_state=self.config['clustering_params']['random_state'])
            cluster_labels = kmeans.fit_predict(scaled_data)
            silhouette_avg = silhouette_score(scaled_data, cluster_labels)
            silhouette_scores.append(silhouette_avg)
        
        optimal_k = k_range[np.argmax(silhouette_scores)]
        
        # 最適クラスター数でクラスタリング実行
        kmeans_final = KMeans(n_clusters=optimal_k,
                                random_state=self.config['clustering_params']['random_state'])
        final_clusters = kmeans_final.fit_predict(scaled_data)
        
        # クラスター特徴分析
        cluster_analysis = {}
        for cluster_id in range(optimal_k):
            cluster_mask = final_clusters == cluster_id
            cluster_data = clustering_data[cluster_mask]
            
            cluster_stats = {}
            for metric in numeric_cols:
                cluster_stats[metric] = {
                    'mean': cluster_data[metric].mean(),
                    'std': cluster_data[metric].std(),
                    'size': len(cluster_data)
                }
            
            # クラスター内のライフサイクル段階分布
            cluster_lifecycle_dist = data[data.index.isin(cluster_data.index)]['lifecycle_stage'].value_counts(normalize=True).to_dict()
            cluster_stats['lifecycle_distribution'] = cluster_lifecycle_dist
            
            cluster_analysis[f'cluster_{cluster_id}'] = cluster_stats
        
        clustering_results = {
            'optimal_k': optimal_k,
            'silhouette_scores': dict(zip(k_range, silhouette_scores)),
            'cluster_centers': kmeans_final.cluster_centers_,
            'cluster_analysis': cluster_analysis,
            'cluster_labels': final_clusters
        }
        
        return clustering_results
    
    def _perform_significance_tests(self, data: pd.DataFrame) -> Dict[str, Any]:
        """統計的有意性検定"""
        print("統計的有意性検定を実行中...")
        
        significance_results = {}
        
        # 段階間の平均差検定
        stage_comparisons = {}
        stages = [stage for stage in self.lifecycle_stages.keys() 
                    if len(data[data['lifecycle_stage'] == stage]) >= self.config['min_observations']]
        
        for i, stage1 in enumerate(stages):
            for j, stage2 in enumerate(stages[i+1:], i+1):
                comparison_key = f"{stage1}_vs_{stage2}"
                stage1_data = data[data['lifecycle_stage'] == stage1]
                stage2_data = data[data['lifecycle_stage'] == stage2]
                
                metric_tests = {}
                for metric in self.evaluation_metrics:
                    if metric in data.columns:
                        values1 = stage1_data[metric].dropna()
                        values2 = stage2_data[metric].dropna()
                        
                        if len(values1) > 0 and len(values2) > 0:
                            # t検定
                            t_stat, t_p = stats.ttest_ind(values1, values2)
                            
                            # Mann-Whitney U検定（ノンパラメトリック）
                            u_stat, u_p = stats.mannwhitneyu(values1, values2, alternative='two-sided')
                            
                            metric_tests[metric] = {
                                't_test': {'statistic': t_stat, 'p_value': t_p},
                                'mannwhitney_test': {'statistic': u_stat, 'p_value': u_p},
                                'significant': min(t_p, u_p) < self.config['significance_level']
                            }
                
                stage_comparisons[comparison_key] = metric_tests
        
        significance_results['stage_comparisons'] = stage_comparisons
        
        # 市場カテゴリ間の比較
        market_comparisons = self._test_market_category_differences(data)
        significance_results['market_comparisons'] = market_comparisons
        
        return significance_results
    
    def _test_market_category_differences(self, data: pd.DataFrame) -> Dict[str, Any]:
        """市場カテゴリ間の統計的差異検定"""
        market_comparisons = {}
        
        markets = ['high_share', 'declining_share', 'lost_share']
        market_data = {market: data[data['market_category'] == market] for market in markets}
        
        for i, market1 in enumerate(markets):
            for j, market2 in enumerate(markets[i+1:], i+1):
                if len(market_data[market1]) < self.config['min_observations'] or \
                    len(market_data[market2]) < self.config['min_observations']:
                    continue
                
                comparison_key = f"{market1}_vs_{market2}"
                metric_tests = {}
                
                for metric in self.evaluation_metrics:
                    if metric in data.columns:
                        values1 = market_data[market1][metric].dropna()
                        values2 = market_data[market2][metric].dropna()
                        
                        if len(values1) > 0 and len(values2) > 0:
                            t_stat, t_p = stats.ttest_ind(values1, values2)
                            u_stat, u_p = stats.mannwhitneyu(values1, values2, alternative='two-sided')
                            
                            metric_tests[metric] = {
                                't_test': {'statistic': t_stat, 'p_value': t_p},
                                'mannwhitney_test': {'statistic': u_stat, 'p_value': u_p},
                                'significant': min(t_p, u_p) < self.config['significance_level']
                            }
                
                market_comparisons[comparison_key] = metric_tests
        
        return market_comparisons
    
    def _generate_summary_statistics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """サマリー統計生成"""
        summary = {
            'total_observations': len(data),
            'unique_companies': data['company_id'].nunique(),
            'years_covered': data['year'].nunique(),
            'year_range': (data['year'].min(), data['year'].max()),
            'lifecycle_stage_distribution': data['lifecycle_stage'].value_counts().to_dict(),
            'market_category_distribution': data['market_category'].value_counts().to_dict(),
            'industry_distribution': data['industry'].value_counts().to_dict()
        }
        
        return summary
    
    def _generate_metadata(self, data: pd.DataFrame) -> Dict[str, Any]:
        """メタデータ生成"""
        return {
            'analysis_date': pd.Timestamp.now().isoformat(),
            'data_shape': data.shape,
            'columns': list(data.columns),
            'config': self.config,
            'lifecycle_stages': self.lifecycle_stages,
            'evaluation_metrics': self.evaluation_metrics
        }
    
    def get_stage_performance_summary(self, stage: str) -> Dict[str, Any]:
        """特定段階の性能サマリー取得"""
        if stage not in self.performance_results.get('stage_patterns', {}):
            return {'error': f'Stage {stage} not found in analysis results'}
        
        return self.performance_results['stage_patterns'][stage]
    
    def get_top_performing_companies_by_stage(self, 
                                            stage: str, 
                                            metric: str, 
                                            top_n: int = 10) -> List[Dict]:
        """段階別トップパフォーマンス企業取得"""
        # 実装は具体的なデータ構造に依存するためスケルトン
        return []
    
    def export_analysis_results(self, output_path: str) -> bool:
        """分析結果をエクスポート"""
        try:
            if self.performance_results:
                # JSON形式で保存
                import json
                with open(f"{output_path}/lifecycle_performance_results.json", 'w', encoding='utf-8') as f:
                    json.dump(self.performance_results, f, ensure_ascii=False, indent=2, default=str)
                
                return True
        except Exception as e:
            print(f"Export error: {e}")
            return False
    
    def generate_performance_insights(self) -> List[str]:
        """
        分析結果から重要なインサイトを生成
        
        Returns:
            インサイトのリスト
        """
        insights = []
        
        if not self.performance_results:
            return ["分析が実行されていません。先にanalyze_lifecycle_performance()を実行してください。"]
        
        stage_patterns = self.performance_results.get('stage_patterns', {})
        transition_analysis = self.performance_results.get('transition_analysis', {})
        market_comparison = self.performance_results.get('market_comparison', {})
        
        # 1. 段階別性能パターンのインサイト
        if stage_patterns:
            # 最高パフォーマンス段階の特定
            stage_performance_scores = {}
            for stage, patterns in stage_patterns.items():
                if 'roe' in patterns:
                    stage_performance_scores[stage] = patterns['roe']['mean']
            
            if stage_performance_scores:
                best_stage = max(stage_performance_scores, key=stage_performance_scores.get)
                worst_stage = min(stage_performance_scores, key=stage_performance_scores.get)
                
                insights.append(f"ROEが最も高いライフサイクル段階は{self.lifecycle_stages[best_stage]}（平均{stage_performance_scores[best_stage]:.2%}）です。")
                insights.append(f"ROEが最も低いライフサイクル段階は{self.lifecycle_stages[worst_stage]}（平均{stage_performance_scores[worst_stage]:.2%}）です。")
        
        # 2. 遷移パターンのインサイト
        if transition_analysis and 'transition_matrix' in transition_analysis:
            transition_matrix = transition_analysis['transition_matrix']
            
            # 最も一般的な遷移パターン
            max_transition_prob = 0
            most_common_transition = None
            
            for from_stage in transition_matrix.index:
                for to_stage in transition_matrix.columns:
                    if from_stage != to_stage and transition_matrix.loc[from_stage, to_stage] > max_transition_prob:
                        max_transition_prob = transition_matrix.loc[from_stage, to_stage]
                        most_common_transition = (from_stage, to_stage)
            
            if most_common_transition:
                insights.append(f"最も一般的な段階遷移は{self.lifecycle_stages[most_common_transition[0]]}→{self.lifecycle_stages[most_common_transition[1]]}（確率{max_transition_prob:.2%}）です。")
        
        # 3. 市場カテゴリ比較のインサイト
        if market_comparison:
            # 高シェア市場の特徴
            high_share_data = market_comparison.get('high_share', {})
            lost_share_data = market_comparison.get('lost_share', {})
            
            if high_share_data and lost_share_data:
                # 成長期でのROE比較
                high_growth_roe = high_share_data.get('growth', {}).get('roe', {}).get('mean', 0)
                lost_growth_roe = lost_share_data.get('growth', {}).get('roe', {}).get('mean', 0)
                
                if high_growth_roe > 0 and lost_growth_roe > 0:
                    roe_diff = high_growth_roe - lost_growth_roe
                    insights.append(f"成長期において、高シェア市場企業のROEは失失市場企業より平均{roe_diff:.2%}ポイント高くなっています。")
        
        # 4. 要因項目影響度のインサイト
        factor_impact = self.performance_results.get('factor_impact', {})
        if factor_impact:
            # 段階別で最も影響度の高い要因項目を特定
            for stage, stage_factors in factor_impact.items():
                if 'sales_growth_rate' in stage_factors:
                    growth_factors = stage_factors['sales_growth_rate']
                    
                    # 有意な正の相関が最も高い要因項目
                    significant_positive_factors = [
                        (factor, data['correlation']) 
                        for factor, data in growth_factors.items()
                        if data.get('significant', False) and data['correlation'] > 0
                    ]
                    
                    if significant_positive_factors:
                        top_factor = max(significant_positive_factors, key=lambda x: x[1])
                        insights.append(f"{self.lifecycle_stages[stage]}において、売上成長率に最も正の影響を与える要因は{top_factor[0]}（相関係数{top_factor[1]:.3f}）です。")
        
        # 5. クラスタリング結果のインサイト
        clustering_results = self.performance_results.get('clustering_results', {})
        if clustering_results and 'optimal_k' in clustering_results:
            optimal_k = clustering_results['optimal_k']
            insights.append(f"財務性能パターンによって企業は{optimal_k}つの明確なクラスターに分類されます。")
            
            # 最高性能クラスターの特定
            cluster_analysis = clustering_results.get('cluster_analysis', {})
            if cluster_analysis:
                cluster_roe_means = {}
                for cluster_id, cluster_data in cluster_analysis.items():
                    if 'roe' in cluster_data:
                        cluster_roe_means[cluster_id] = cluster_data['roe']['mean']
                
                if cluster_roe_means:
                    best_cluster = max(cluster_roe_means, key=cluster_roe_means.get)
                    best_cluster_roe = cluster_roe_means[best_cluster]
                    insights.append(f"最高性能クラスター（{best_cluster}）の平均ROEは{best_cluster_roe:.2%}です。")
        
        # 6. 統計的有意性のインサイト
        significance_tests = self.performance_results.get('significance_tests', {})
        if significance_tests and 'stage_comparisons' in significance_tests:
            stage_comparisons = significance_tests['stage_comparisons']
            
            significant_comparisons = []
            for comparison, metrics in stage_comparisons.items():
                for metric, tests in metrics.items():
                    if tests.get('significant', False):
                        significant_comparisons.append((comparison, metric))
            
            if significant_comparisons:
                insights.append(f"ライフサイクル段階間で統計的に有意な差異が見られる指標は{len(significant_comparisons)}項目です。")
        
        # 7. データ品質・カバレッジのインサイト
        summary_stats = self.performance_results.get('summary_statistics', {})
        if summary_stats:
            total_obs = summary_stats.get('total_observations', 0)
            unique_companies = summary_stats.get('unique_companies', 0)
            years_covered = summary_stats.get('years_covered', 0)
            
            insights.append(f"分析対象は{unique_companies}社、{years_covered}年間、総観測数{total_obs}件のデータに基づいています。")
        
        return insights
    
    def visualize_lifecycle_performance(self, save_path: str = None) -> Dict[str, Any]:
        """
        ライフサイクル別性能の可視化
        
        Args:
            save_path: 図表の保存パス（オプション）
        
        Returns:
            生成された図表の情報
        """
        if not self.performance_results:
            return {'error': '分析結果がありません。先に分析を実行してください。'}
        
        visualizations = {}
        
        try:
            # 1. 段階別性能比較（箱ひげ図）
            fig1, axes1 = plt.subplots(2, 2, figsize=(15, 10))
            fig1.suptitle('ライフサイクル段階別財務性能比較', fontsize=16, fontweight='bold')
            
            stage_patterns = self.performance_results.get('stage_patterns', {})
            
            # ROE比較
            if stage_patterns:
                stages = list(stage_patterns.keys())
                roe_means = [stage_patterns[stage].get('roe', {}).get('mean', 0) for stage in stages]
                roe_stds = [stage_patterns[stage].get('roe', {}).get('std', 0) for stage in stages]
                
                axes1[0, 0].bar(stages, roe_means, yerr=roe_stds, capsize=5, alpha=0.7, color='skyblue')
                axes1[0, 0].set_title('ROE（自己資本利益率）')
                axes1[0, 0].set_ylabel('ROE (%)')
                axes1[0, 0].tick_params(axis='x', rotation=45)
                
                # 売上成長率比較
                growth_means = [stage_patterns[stage].get('sales_growth_rate', {}).get('mean', 0) for stage in stages]
                growth_stds = [stage_patterns[stage].get('sales_growth_rate', {}).get('std', 0) for stage in stages]
                
                axes1[0, 1].bar(stages, growth_means, yerr=growth_stds, capsize=5, alpha=0.7, color='lightgreen')
                axes1[0, 1].set_title('売上成長率')
                axes1[0, 1].set_ylabel('成長率 (%)')
                axes1[0, 1].tick_params(axis='x', rotation=45)
                
                # 営業利益率比較
                margin_means = [stage_patterns[stage].get('operating_margin', {}).get('mean', 0) for stage in stages]
                margin_stds = [stage_patterns[stage].get('operating_margin', {}).get('std', 0) for stage in stages]
                
                axes1[1, 0].bar(stages, margin_means, yerr=margin_stds, capsize=5, alpha=0.7, color='salmon')
                axes1[1, 0].set_title('営業利益率')
                axes1[1, 0].set_ylabel('利益率 (%)')
                axes1[1, 0].tick_params(axis='x', rotation=45)
                
                # 付加価値率比較
                value_means = [stage_patterns[stage].get('value_added_ratio', {}).get('mean', 0) for stage in stages]
                value_stds = [stage_patterns[stage].get('value_added_ratio', {}).get('std', 0) for stage in stages]
                
                axes1[1, 1].bar(stages, value_means, yerr=value_stds, capsize=5, alpha=0.7, color='gold')
                axes1[1, 1].set_title('付加価値率')
                axes1[1, 1].set_ylabel('付加価値率 (%)')
                axes1[1, 1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            if save_path:
                fig1.savefig(f"{save_path}/lifecycle_performance_comparison.png", dpi=300, bbox_inches='tight')
            
            visualizations['performance_comparison'] = fig1
            
            # 2. 遷移確率行列ヒートマップ
            transition_analysis = self.performance_results.get('transition_analysis', {})
            if 'transition_matrix' in transition_analysis:
                fig2, ax2 = plt.subplots(figsize=(10, 8))
                transition_matrix = transition_analysis['transition_matrix']
                
                sns.heatmap(transition_matrix, annot=True, cmap='Blues', fmt='.3f', ax=ax2)
                ax2.set_title('ライフサイクル段階遷移確率行列', fontsize=14, fontweight='bold')
                ax2.set_xlabel('遷移先段階')
                ax2.set_ylabel('遷移元段階')
                
                if save_path:
                    fig2.savefig(f"{save_path}/lifecycle_transition_matrix.png", dpi=300, bbox_inches='tight')
                
                visualizations['transition_matrix'] = fig2
            
            # 3. 市場カテゴリ別比較
            market_comparison = self.performance_results.get('market_comparison', {})
            if market_comparison:
                fig3, ax3 = plt.subplots(figsize=(12, 6))
                
                market_categories = list(market_comparison.keys())
                stages_for_market = ['emergence', 'growth', 'maturity', 'decline']
                
                x = np.arange(len(stages_for_market))
                width = 0.25
                
                for i, market in enumerate(market_categories):
                    market_data = market_comparison[market]
                    roe_values = []
                    
                    for stage in stages_for_market:
                        if stage in market_data and 'roe' in market_data[stage]:
                            roe_values.append(market_data[stage]['roe']['mean'])
                        else:
                            roe_values.append(0)
                    
                    ax3.bar(x + i * width, roe_values, width, label=market, alpha=0.8)
                
                ax3.set_xlabel('ライフサイクル段階')
                ax3.set_ylabel('平均ROE (%)')
                ax3.set_title('市場カテゴリ別・ライフサイクル段階別ROE比較')
                ax3.set_xticks(x + width)
                ax3.set_xticklabels(stages_for_market)
                ax3.legend()
                
                if save_path:
                    fig3.savefig(f"{save_path}/market_lifecycle_roe_comparison.png", dpi=300, bbox_inches='tight')
                
                visualizations['market_comparison'] = fig3
            
            return visualizations
            
        except Exception as e:
            return {'error': f'可視化中にエラーが発生しました: {str(e)}'}
    
    def predict_lifecycle_transition(self, 
                                    company_data: pd.DataFrame,
                                    prediction_horizon: int = 3) -> Dict[str, Any]:
        """
        企業のライフサイクル段階遷移予測
        
        Args:
            company_data: 予測対象企業の財務データ
            prediction_horizon: 予測期間（年）
        
        Returns:
            遷移予測結果
        """
        if not self.performance_results:
            return {'error': '分析結果がありません。先に分析を実行してください。'}
        
        transition_analysis = self.performance_results.get('transition_analysis', {})
        
        if 'transition_matrix' not in transition_analysis:
            return {'error': '遷移分析結果がありません。'}
        
        transition_matrix = transition_analysis['transition_matrix']
        
        # 現在の段階を特定
        current_age = company_data['company_age'].iloc[-1] if 'company_age' in company_data else 0
        current_stage = self._determine_current_stage(company_data, current_age)
        
        # 遷移確率に基づく予測
        predictions = {}
        current_probs = {stage: 1.0 if stage == current_stage else 0.0 
                        for stage in self.lifecycle_stages.keys()}
        
        for year in range(1, prediction_horizon + 1):
            next_probs = {}
            
            for to_stage in self.lifecycle_stages.keys():
                prob = 0
                for from_stage, current_prob in current_probs.items():
                    if from_stage in transition_matrix.index and to_stage in transition_matrix.columns:
                        prob += current_prob * transition_matrix.loc[from_stage, to_stage]
                next_probs[to_stage] = prob
            
            predictions[f'year_{year}'] = next_probs.copy()
            current_probs = next_probs
        
        # 最可能性段階の特定
        most_likely_stages = {}
        for year, probs in predictions.items():
            most_likely_stage = max(probs, key=probs.get)
            most_likely_stages[year] = {
                'stage': most_likely_stage,
                'probability': probs[most_likely_stage]
            }
        
        return {
            'current_stage': current_stage,
            'predictions': predictions,
            'most_likely_stages': most_likely_stages,
            'prediction_horizon': prediction_horizon
        }
    
    def _determine_current_stage(self, company_data: pd.DataFrame, age: int) -> str:
        """現在のライフサイクル段階を判定"""
        # 最新の財務データを基に総合判定
        latest_data = company_data.iloc[-1]
        
        # 基本的な年齢ベース判定
        if age <= 5:
            base_stage = 'emergence'
        elif age <= 15:
            base_stage = 'growth'
        elif age <= 30:
            base_stage = 'maturity'
        else:
            base_stage = 'decline'
        
        # 財務指標による調整
        if 'sales_growth_rate' in latest_data:
            growth_rate = latest_data['sales_growth_rate']
            if growth_rate < -0.1 and base_stage != 'emergence':  # 10%以上の減収
                return 'decline'
            elif growth_rate > 0.2 and age > 5:  # 20%以上の成長
                return 'growth'
        
        return base_stage