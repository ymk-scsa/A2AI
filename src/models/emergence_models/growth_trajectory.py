"""
A2AI - Advanced Financial Analysis AI
新設企業成長軌道分析モジュール

このモジュールは新設企業（分社・スピンオフ・独立創業）の成長軌道を分析し、
成功パターンと失敗パターンを特定します。

対象企業例：
- キオクシア（2018年設立、東芝メモリから独立）
- プロテリアル（2023年設立、日立金属から独立）
- デンソーウェーブ（2001年設立、デンソーから分社）

Author: A2AI Development Team
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from scipy import stats
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')

class GrowthTrajectoryAnalyzer:
    """
    新設企業の成長軌道分析クラス
    
    主な機能：
    1. 成長フェーズ分類（導入期、成長期、成熟期、衰退期）
    2. 成長パターン類型化（指数成長、線形成長、S字カーブ等）
    3. 成長要因項目の影響度分析
    4. 将来成長軌道予測
    5. 市場カテゴリ別成長パターン比較
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初期化
        
        Args:
            config (Dict, optional): 分析設定パラメータ
        """
        self.config = config or self._default_config()
        self.scaler = StandardScaler()
        self.growth_phases = ['introduction', 'growth', 'maturity', 'decline']
        self.growth_patterns = ['exponential', 'linear', 's_curve', 'logarithmic', 'decline']
        
        # 分析結果保存用
        self.trajectory_data = None
        self.growth_clusters = None
        self.pattern_models = {}
        self.phase_transitions = None
        
    def _default_config(self) -> Dict:
        """デフォルト設定"""
        return {
            'min_years_for_analysis': 3,  # 分析に必要な最小年数
            'growth_phase_threshold': {
                'introduction_growth_rate': 0.15,  # 導入期成長率閾値
                'maturity_growth_rate': 0.05,     # 成熟期成長率閾値
                'decline_growth_rate': -0.05      # 衰退期成長率閾値
            },
            'smoothing_window': 3,  # 移動平均ウィンドウサイズ
            'outlier_threshold': 3.0,  # 外れ値検出閾値（標準偏差）
            'clustering_method': 'kmeans',  # クラスタリング手法
            'n_clusters': 5,  # クラスタ数
        }
    
    def fit(self, emergence_data: pd.DataFrame, factor_data: pd.DataFrame) -> 'GrowthTrajectoryAnalyzer':
        """
        新設企業データで成長軌道モデルを学習
        
        Args:
            emergence_data (pd.DataFrame): 新設企業基本情報
                Columns: ['company_id', 'company_name', 'establishment_date', 
                            'parent_company', 'market_category', 'industry']
            factor_data (pd.DataFrame): 要因項目時系列データ
                Columns: ['company_id', 'year', 'sales', 'sales_growth_rate', 
                            'operating_margin', 'roe', 'employees', 'rd_expense', ...]
        
        Returns:
            self: 学習済みモデル
        """
        # データ前処理
        self.trajectory_data = self._prepare_trajectory_data(emergence_data, factor_data)
        
        # 成長フェーズ分類
        self._classify_growth_phases()
        
        # 成長パターン分析
        self._analyze_growth_patterns()
        
        # クラスタリング分析
        self._perform_clustering()
        
        # フェーズ遷移分析
        self._analyze_phase_transitions()
        
        return self
    
    def _prepare_trajectory_data(self, emergence_data: pd.DataFrame, 
                                factor_data: pd.DataFrame) -> pd.DataFrame:
        """
        成長軌道分析用データを準備
        
        Args:
            emergence_data: 新設企業基本情報
            factor_data: 要因項目時系列データ
        
        Returns:
            準備済み軌道データ
        """
        # 新設企業のみ抽出
        emergence_companies = set(emergence_data['company_id'].unique())
        trajectory_data = factor_data[factor_data['company_id'].isin(emergence_companies)].copy()
        
        # 基本情報結合
        trajectory_data = trajectory_data.merge(
            emergence_data[['company_id', 'establishment_date', 'market_category']], 
            on='company_id'
        )
        
        # 設立からの経過年数計算
        trajectory_data['establishment_date'] = pd.to_datetime(trajectory_data['establishment_date'])
        trajectory_data['years_since_establishment'] = (
            trajectory_data['year'] - trajectory_data['establishment_date'].dt.year
        )
        
        # 最小分析年数フィルタ
        company_years = trajectory_data.groupby('company_id')['years_since_establishment'].max()
        valid_companies = company_years[company_years >= self.config['min_years_for_analysis']].index
        trajectory_data = trajectory_data[trajectory_data['company_id'].isin(valid_companies)]
        
        # 欠損値処理
        trajectory_data = self._handle_missing_values(trajectory_data)
        
        # 外れ値処理
        trajectory_data = self._handle_outliers(trajectory_data)
        
        return trajectory_data
    
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """欠損値処理"""
        # 数値列の欠損値を前後の値で補間
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col not in ['company_id', 'year']:
                data[col] = data.groupby('company_id')[col].fillna(method='ffill').fillna(method='bfill')
        
        return data
    
    def _handle_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """外れ値処理"""
        numeric_columns = ['sales_growth_rate', 'operating_margin', 'roe']
        threshold = self.config['outlier_threshold']
        
        for col in numeric_columns:
            if col in data.columns:
                mean_val = data[col].mean()
                std_val = data[col].std()
                
                # 外れ値をクリップ
                lower_bound = mean_val - threshold * std_val
                upper_bound = mean_val + threshold * std_val
                data[col] = np.clip(data[col], lower_bound, upper_bound)
        
        return data
    
    def _classify_growth_phases(self):
        """成長フェーズ分類"""
        # 移動平均による平滑化
        window = self.config['smoothing_window']
        
        for company_id in self.trajectory_data['company_id'].unique():
            company_data = self.trajectory_data[
                self.trajectory_data['company_id'] == company_id
            ].sort_values('years_since_establishment')
            
            if len(company_data) < window:
                continue
            
            # 成長率の移動平均
            company_data['sales_growth_smooth'] = (
                company_data['sales_growth_rate'].rolling(window=window, center=True).mean()
            )
            
            # フェーズ分類
            thresholds = self.config['growth_phase_threshold']
            phases = []
            
            for _, row in company_data.iterrows():
                growth_rate = row['sales_growth_smooth']
                
                if pd.isna(growth_rate):
                    phases.append('unknown')
                elif growth_rate >= thresholds['introduction_growth_rate']:
                    phases.append('growth')
                elif growth_rate >= thresholds['maturity_growth_rate']:
                    phases.append('maturity')
                elif growth_rate >= thresholds['decline_growth_rate']:
                    phases.append('introduction')
                else:
                    phases.append('decline')
            
            # 結果を元データに反映
            indices = company_data.index
            self.trajectory_data.loc[indices, 'growth_phase'] = phases
    
    def _analyze_growth_patterns(self):
        """成長パターン分析"""
        self.pattern_models = {}
        
        for company_id in self.trajectory_data['company_id'].unique():
            company_data = self.trajectory_data[
                self.trajectory_data['company_id'] == company_id
            ].sort_values('years_since_establishment')
            
            if len(company_data) < 4:  # 最小データポイント数
                continue
            
            x = company_data['years_since_establishment'].values
            y = company_data['sales'].values
            
            # 各成長パターンでのフィッティング
            patterns = self._fit_growth_patterns(x, y)
            self.pattern_models[company_id] = patterns
            
            # 最適パターン選択（R²基準）
            best_pattern = max(patterns.items(), key=lambda item: item[1]['r_squared'])
            
            # 結果を元データに反映
            indices = company_data.index
            self.trajectory_data.loc[indices, 'growth_pattern'] = best_pattern[0]
            self.trajectory_data.loc[indices, 'pattern_r_squared'] = best_pattern[1]['r_squared']
    
    def _fit_growth_patterns(self, x: np.ndarray, y: np.ndarray) -> Dict:
        """
        各成長パターンでフィッティング
        
        Args:
            x: 設立からの経過年数
            y: 売上高
        
        Returns:
            各パターンのフィッティング結果
        """
        patterns = {}
        
        # 正規化
        x_norm = (x - x.min()) / (x.max() - x.min() + 1e-8)
        y_norm = (y - y.min()) / (y.max() - y.min() + 1e-8)
        
        # 1. 線形成長パターン
        try:
            linear_params = np.polyfit(x_norm, y_norm, 1)
            linear_pred = np.poly1d(linear_params)(x_norm)
            patterns['linear'] = {
                'params': linear_params,
                'r_squared': self._calculate_r_squared(y_norm, linear_pred)
            }
        except:
            patterns['linear'] = {'params': None, 'r_squared': 0}
        
        # 2. 指数成長パターン
        try:
            def exponential_func(x, a, b, c):
                return a * np.exp(b * x) + c
            
            popt, _ = curve_fit(exponential_func, x_norm, y_norm, maxfev=1000)
            exp_pred = exponential_func(x_norm, *popt)
            patterns['exponential'] = {
                'params': popt,
                'r_squared': self._calculate_r_squared(y_norm, exp_pred)
            }
        except:
            patterns['exponential'] = {'params': None, 'r_squared': 0}
        
        # 3. S字カーブ（ロジスティック）パターン
        try:
            def logistic_func(x, L, k, x0, b):
                return L / (1 + np.exp(-k * (x - x0))) + b
            
            popt, _ = curve_fit(logistic_func, x_norm, y_norm, maxfev=1000)
            logistic_pred = logistic_func(x_norm, *popt)
            patterns['s_curve'] = {
                'params': popt,
                'r_squared': self._calculate_r_squared(y_norm, logistic_pred)
            }
        except:
            patterns['s_curve'] = {'params': None, 'r_squared': 0}
        
        # 4. 対数成長パターン
        try:
            log_x = np.log(x_norm + 1)
            log_params = np.polyfit(log_x, y_norm, 1)
            log_pred = np.poly1d(log_params)(log_x)
            patterns['logarithmic'] = {
                'params': log_params,
                'r_squared': self._calculate_r_squared(y_norm, log_pred)
            }
        except:
            patterns['logarithmic'] = {'params': None, 'r_squared': 0}
        
        # 5. 衰退パターン（負の指数）
        try:
            def decline_func(x, a, b, c):
                return a * np.exp(-b * x) + c
            
            popt, _ = curve_fit(decline_func, x_norm, y_norm, maxfev=1000)
            decline_pred = decline_func(x_norm, *popt)
            patterns['decline'] = {
                'params': popt,
                'r_squared': self._calculate_r_squared(y_norm, decline_pred)
            }
        except:
            patterns['decline'] = {'params': None, 'r_squared': 0}
        
        return patterns
    
    def _calculate_r_squared(self, y_actual: np.ndarray, y_pred: np.ndarray) -> float:
        """R²計算"""
        if len(y_actual) != len(y_pred):
            return 0
        
        ss_res = np.sum((y_actual - y_pred) ** 2)
        ss_tot = np.sum((y_actual - np.mean(y_actual)) ** 2)
        
        if ss_tot == 0:
            return 0
        
        return 1 - (ss_res / ss_tot)
    
    def _perform_clustering(self):
        """成長軌道クラスタリング"""
        # クラスタリング用特徴量作成
        features = self._create_clustering_features()
        
        if len(features) == 0:
            return
        
        # 正規化
        features_scaled = self.scaler.fit_transform(features)
        
        # クラスタリング実行
        if self.config['clustering_method'] == 'kmeans':
            clusterer = KMeans(n_clusters=self.config['n_clusters'], random_state=42)
        else:
            clusterer = GaussianMixture(n_components=self.config['n_clusters'], random_state=42)
        
        cluster_labels = clusterer.fit_predict(features_scaled)
        
        # 結果保存
        self.growth_clusters = {
            'features': features,
            'labels': cluster_labels,
            'model': clusterer,
            'feature_names': features.columns.tolist()
        }
        
        # 元データに反映
        company_ids = features.index
        for i, company_id in enumerate(company_ids):
            mask = self.trajectory_data['company_id'] == company_id
            self.trajectory_data.loc[mask, 'growth_cluster'] = cluster_labels[i]
    
    def _create_clustering_features(self) -> pd.DataFrame:
        """クラスタリング用特徴量作成"""
        features_list = []
        
        for company_id in self.trajectory_data['company_id'].unique():
            company_data = self.trajectory_data[
                self.trajectory_data['company_id'] == company_id
            ].sort_values('years_since_establishment')
            
            if len(company_data) < 3:
                continue
            
            # 基本統計量
            feature_dict = {
                'company_id': company_id,
                'avg_sales_growth': company_data['sales_growth_rate'].mean(),
                'std_sales_growth': company_data['sales_growth_rate'].std(),
                'avg_operating_margin': company_data['operating_margin'].mean(),
                'avg_roe': company_data['roe'].mean(),
                'max_employees': company_data['employees'].max(),
                'total_rd_investment': company_data['rd_expense'].sum(),
                'years_in_operation': company_data['years_since_establishment'].max(),
            }
            
            # 成長パターン特徴量
            growth_counts = company_data['growth_phase'].value_counts()
            for phase in self.growth_phases:
                feature_dict[f'phase_{phase}_ratio'] = growth_counts.get(phase, 0) / len(company_data)
            
            # 最適成長パターン
            if company_id in self.pattern_models:
                best_pattern = max(
                    self.pattern_models[company_id].items(), 
                    key=lambda x: x[1]['r_squared']
                )
                feature_dict['best_pattern_r_squared'] = best_pattern[1]['r_squared']
                
                for i, pattern in enumerate(self.growth_patterns):
                    feature_dict[f'pattern_{pattern}'] = 1 if pattern == best_pattern[0] else 0
            
            features_list.append(feature_dict)
        
        if not features_list:
            return pd.DataFrame()
        
        features_df = pd.DataFrame(features_list)
        features_df = features_df.set_index('company_id')
        
        # 数値列のみ選択
        numeric_features = features_df.select_dtypes(include=[np.number])
        
        # 欠損値処理
        numeric_features = numeric_features.fillna(numeric_features.mean())
        
        return numeric_features
    
    def _analyze_phase_transitions(self):
        """フェーズ遷移分析"""
        transitions = []
        
        for company_id in self.trajectory_data['company_id'].unique():
            company_data = self.trajectory_data[
                self.trajectory_data['company_id'] == company_id
            ].sort_values('years_since_establishment')
            
            phases = company_data['growth_phase'].tolist()
            
            # 遷移パターン抽出
            for i in range(len(phases) - 1):
                from_phase = phases[i]
                to_phase = phases[i + 1]
                
                if from_phase != to_phase and pd.notna(from_phase) and pd.notna(to_phase):
                    transitions.append({
                        'company_id': company_id,
                        'from_phase': from_phase,
                        'to_phase': to_phase,
                        'transition_year': company_data.iloc[i]['years_since_establishment'],
                        'market_category': company_data.iloc[i]['market_category']
                    })
        
        self.phase_transitions = pd.DataFrame(transitions) if transitions else pd.DataFrame()
    
    def predict_growth_trajectory(self, company_id: str, future_years: int = 5) -> Dict:
        """
        成長軌道予測
        
        Args:
            company_id: 対象企業ID
            future_years: 予測年数
        
        Returns:
            予測結果辞書
        """
        if company_id not in self.pattern_models:
            return {'error': f'Company {company_id} not found in models'}
        
        company_data = self.trajectory_data[
            self.trajectory_data['company_id'] == company_id
        ].sort_values('years_since_establishment')
        
        if len(company_data) == 0:
            return {'error': f'No data found for company {company_id}'}
        
        # 最適パターンで予測
        patterns = self.pattern_models[company_id]
        best_pattern = max(patterns.items(), key=lambda x: x[1]['r_squared'])
        pattern_name, pattern_info = best_pattern
        
        # 現在までの年数
        current_years = company_data['years_since_establishment'].max()
        
        # 予測年数範囲
        future_x = np.arange(current_years + 1, current_years + future_years + 1)
        
        # 正規化パラメータ
        historical_x = company_data['years_since_establishment'].values
        historical_y = company_data['sales'].values
        
        x_min, x_max = historical_x.min(), historical_x.max()
        y_min, y_max = historical_y.min(), historical_y.max()
        
        # 予測実行
        future_x_norm = (future_x - x_min) / (x_max - x_min + 1e-8)
        
        if pattern_name == 'linear' and pattern_info['params'] is not None:
            future_y_norm = np.poly1d(pattern_info['params'])(future_x_norm)
        elif pattern_name == 'exponential' and pattern_info['params'] is not None:
            def exponential_func(x, a, b, c):
                return a * np.exp(b * x) + c
            future_y_norm = exponential_func(future_x_norm, *pattern_info['params'])
        # 他のパターンも同様に実装...
        else:
            # デフォルトは線形外挿
            trend = np.polyfit(historical_x, historical_y, 1)
            future_y_norm = np.poly1d(trend)(future_x)
            future_y = future_y_norm
        
        if 'future_y' not in locals():
            # 逆正規化
            future_y = future_y_norm * (y_max - y_min) + y_min
        
        return {
            'company_id': company_id,
            'current_year': current_years,
            'prediction_years': future_x.tolist(),
            'predicted_sales': future_y.tolist(),
            'growth_pattern': pattern_name,
            'model_r_squared': pattern_info['r_squared'],
            'confidence': self._calculate_prediction_confidence(pattern_info['r_squared'])
        }
    
    def _calculate_prediction_confidence(self, r_squared: float) -> str:
        """予測信頼度計算"""
        if r_squared >= 0.8:
            return 'high'
        elif r_squared >= 0.6:
            return 'medium'
        elif r_squared >= 0.4:
            return 'low'
        else:
            return 'very_low'
    
    def get_market_growth_comparison(self) -> pd.DataFrame:
        """市場カテゴリ別成長パターン比較"""
        if self.trajectory_data is None:
            return pd.DataFrame()
        
        # 市場カテゴリ別統計
        market_stats = []
        
        for market in self.trajectory_data['market_category'].unique():
            market_data = self.trajectory_data[self.trajectory_data['market_category'] == market]
            
            stats = {
                'market_category': market,
                'num_companies': market_data['company_id'].nunique(),
                'avg_sales_growth': market_data['sales_growth_rate'].mean(),
                'median_sales_growth': market_data['sales_growth_rate'].median(),
                'std_sales_growth': market_data['sales_growth_rate'].std(),
                'avg_operating_margin': market_data['operating_margin'].mean(),
                'avg_roe': market_data['roe'].mean(),
            }
            
            # 成長パターン分布
            pattern_counts = market_data['growth_pattern'].value_counts()
            for pattern in self.growth_patterns:
                stats[f'pattern_{pattern}_ratio'] = pattern_counts.get(pattern, 0) / len(market_data)
            
            # 成長フェーズ分布
            phase_counts = market_data['growth_phase'].value_counts()
            for phase in self.growth_phases:
                stats[f'phase_{phase}_ratio'] = phase_counts.get(phase, 0) / len(market_data)
            
            market_stats.append(stats)
        
        return pd.DataFrame(market_stats)
    
    def get_success_factors(self) -> pd.DataFrame:
        """成功要因分析結果取得"""
        if self.growth_clusters is None:
            return pd.DataFrame()
        
        features = self.growth_clusters['features']
        labels = self.growth_clusters['labels']
        
        # クラスタ別統計
        cluster_stats = []
        
        for cluster_id in np.unique(labels):
            cluster_mask = labels == cluster_id
            cluster_features = features[cluster_mask]
            
            stats = {
                'cluster_id': cluster_id,
                'num_companies': len(cluster_features),
                'avg_sales_growth': cluster_features['avg_sales_growth'].mean(),
                'avg_operating_margin': cluster_features['avg_operating_margin'].mean(),
                'avg_roe': cluster_features['avg_roe'].mean(),
                'avg_years_operation': cluster_features['years_in_operation'].mean(),
            }
            
            # 各特徴量の平均値
            for col in features.columns:
                if col not in stats:
                    stats[f'avg_{col}'] = cluster_features[col].mean()
            
            cluster_stats.append(stats)
        
        return pd.DataFrame(cluster_stats)
    
    def export_analysis_results(self) -> Dict:
        """分析結果エクスポート"""
        return {
            'trajectory_data': self.trajectory_data,
            'growth_clusters': self.growth_clusters,
            'pattern_models': self.pattern_models,
            'phase_transitions': self.phase_transitions,
            'market_comparison': self.get_market_growth_comparison(),
            'success_factors': self.get_success_factors(),
        }


# 使用例とテスト用のヘルパー関数
def create_sample_data():
    """サンプルデータ作成（テスト用）"""
    # 新設企業データ
    emergence_data = pd.DataFrame({
        'company_id': ['KIOXIA', 'PROTERIAL', 'DENSO_WAVE', 'ENAX'],
        'company_name': ['キオクシア', 'プロテリアル', 'デンソーウェーブ', 'ENAX'],
        'establishment_date': ['2018-06-01', '2023-04-01', '2001-07-01', '2007-03-01'],
        'parent_company': ['東芝', '日立金属', 'デンソー', ''],
        'market_category': ['lost_markets', 'high_share_markets', 'high_share_markets', 'declining_markets'],
        'industry': ['半導体', '電子材料', 'ロボット', 'バッテリー']
    })
    
    # 要因項目データ（サンプル）
    factor_data = []
    for company in emergence_data['company_id']:
        start_year = pd.to_datetime(emergence_data[emergence_data['company_id'] == company]['establishment_date'].iloc[0]).year
        for year in range(start_year, 2025):
            # 成長パターンを模擬
            years_since = year - start_year
            if company == 'KIOXIA':
                # S字カーブ成長
                base_sales = 1000 * (1 / (1 + np.exp(-0.5 * (years_since - 3))))
                growth_rate = 0.3 * np.exp(-0.2 * years_since) if years_since > 0 else 0.4
            elif company == 'PROTERIAL':
                # 立ち上がり期
                base_sales = 500 * years_since if years_since > 0 else 100
                growth_rate = 0.5 if years_since > 0 else 0
            else:
                # 線形成長
                base_sales = 300 + 50 * years_since
                growth_rate = 0.15 + np.random.normal(0, 0.05)
            
            factor_data.append({
                'company_id': company,
                'year': year,
                'sales': base_sales + np.random.normal(0, base_sales * 0.1),
                'sales_growth_rate': growth_rate + np.random.normal(0, 0.02),
                'operating_margin': 0.08 + np.random.normal(0, 0.02),
                'roe': 0.12 + np.random.normal(0, 0.03),
                'employees': int(100 + years_since * 20 + np.random.normal(0, 10)),
                'rd_expense': base_sales * 0.05 + np.random.normal(0, base_sales * 0.01),
            })
    
    factor_df = pd.DataFrame(factor_data)
    
    return emergence_data, factor_df


# 実行例
if __name__ == "__main__":
    # サンプルデータで動作確認
    emergence_data, factor_data = create_sample_data()
    
    # 分析実行
    analyzer = GrowthTrajectoryAnalyzer()
    analyzer.fit(emergence_data, factor_data)
    
    print("=== A2AI 成長軌道分析結果 ===")
    
    # 市場別成長パターン比較
    market_comparison = analyzer.get_market_growth_comparison()
    print("\n【市場カテゴリ別成長パターン】")
    print(market_comparison[['market_category', 'num_companies', 'avg_sales_growth', 'avg_operating_margin']])
    
    # 成功要因分析
    success_factors = analyzer.get_success_factors()
    print("\n【成長クラスタ別成功要因】")
    print(success_factors[['cluster_id', 'num_companies', 'avg_sales_growth', 'avg_roe']])
    
    # 個別企業予測
    print("\n【個別企業成長予測】")
    for company in emergence_data['company_id']:
        try:
            prediction = analyzer.predict_growth_trajectory(company, future_years=3)
            if 'error' not in prediction:
                print(f"{company}: パターン={prediction['growth_pattern']}, "
                        f"R²={prediction['model_r_squared']:.3f}, "
                        f"信頼度={prediction['confidence']}")
        except Exception as e:
            print(f"{company}: 予測エラー - {str(e)}")


class GrowthTrajectoryVisualizer:
    """
    成長軌道可視化クラス
    
    GrowthTrajectoryAnalyzerの分析結果を可視化するためのクラス
    """
    
    def __init__(self, analyzer: GrowthTrajectoryAnalyzer):
        """
        初期化
        
        Args:
            analyzer: 学習済みGrowthTrajectoryAnalyzer
        """
        self.analyzer = analyzer
        self.trajectory_data = analyzer.trajectory_data
    
    def plot_company_trajectory(self, company_id: str, save_path: Optional[str] = None):
        """
        個別企業の成長軌道プロット
        
        Args:
            company_id: 対象企業ID
            save_path: 保存パス（Noneの場合は表示のみ）
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        if self.trajectory_data is None:
            print("分析が実行されていません")
            return
        
        company_data = self.trajectory_data[
            self.trajectory_data['company_id'] == company_id
        ].sort_values('years_since_establishment')
        
        if len(company_data) == 0:
            print(f"企業 {company_id} のデータが見つかりません")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'成長軌道分析: {company_id}', fontsize=16, fontweight='bold')
        
        # 1. 売上高推移
        ax1 = axes[0, 0]
        ax1.plot(company_data['years_since_establishment'], 
                    company_data['sales'], 'b-o', linewidth=2, markersize=6)
        ax1.set_title('売上高推移')
        ax1.set_xlabel('設立からの経過年数')
        ax1.set_ylabel('売上高')
        ax1.grid(True, alpha=0.3)
        
        # 2. 成長率推移
        ax2 = axes[0, 1]
        ax2.plot(company_data['years_since_establishment'], 
                    company_data['sales_growth_rate'], 'g-o', linewidth=2, markersize=6)
        ax2.axhline(y=0, color='r', linestyle='--', alpha=0.7)
        ax2.set_title('売上成長率推移')
        ax2.set_xlabel('設立からの経過年数')
        ax2.set_ylabel('成長率')
        ax2.grid(True, alpha=0.3)
        
        # 3. 成長フェーズ
        ax3 = axes[1, 0]
        if 'growth_phase' in company_data.columns:
            phases = company_data['growth_phase'].value_counts()
            colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
            ax3.pie(phases.values, labels=phases.index, autopct='%1.1f%%', colors=colors)
            ax3.set_title('成長フェーズ分布')
        
        # 4. 財務指標推移
        ax4 = axes[1, 1]
        ax4.plot(company_data['years_since_establishment'], 
                    company_data['operating_margin'], 'r-o', label='営業利益率', linewidth=2)
        ax4.plot(company_data['years_since_establishment'], 
                    company_data['roe'], 'm-s', label='ROE', linewidth=2)
        ax4.set_title('財務指標推移')
        ax4.set_xlabel('設立からの経過年数')
        ax4.set_ylabel('比率')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
    
    def plot_growth_patterns_comparison(self, save_path: Optional[str] = None):
        """
        成長パターン比較プロット
        
        Args:
            save_path: 保存パス（Noneの場合は表示のみ）
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        if self.trajectory_data is None:
            print("分析が実行されていません")
            return
        
        # 成長パターン別の統計
        pattern_stats = []
        for pattern in self.analyzer.growth_patterns:
            pattern_data = self.trajectory_data[self.trajectory_data['growth_pattern'] == pattern]
            if len(pattern_data) > 0:
                pattern_stats.append({
                    'pattern': pattern,
                    'count': len(pattern_data),
                    'avg_growth': pattern_data['sales_growth_rate'].mean(),
                    'avg_margin': pattern_data['operating_margin'].mean(),
                    'avg_roe': pattern_data['roe'].mean()
                })
        
        if not pattern_stats:
            print("成長パターンデータがありません")
            return
        
        stats_df = pd.DataFrame(pattern_stats)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('成長パターン別比較分析', fontsize=16, fontweight='bold')
        
        # 1. パターン分布
        ax1 = axes[0, 0]
        ax1.bar(stats_df['pattern'], stats_df['count'], color='skyblue')
        ax1.set_title('成長パターン分布')
        ax1.set_xlabel('成長パターン')
        ax1.set_ylabel('企業数')
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. パターン別平均成長率
        ax2 = axes[0, 1]
        bars = ax2.bar(stats_df['pattern'], stats_df['avg_growth'], color='lightgreen')
        ax2.set_title('パターン別平均成長率')
        ax2.set_xlabel('成長パターン')
        ax2.set_ylabel('平均成長率')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. パターン別営業利益率
        ax3 = axes[1, 0]
        ax3.bar(stats_df['pattern'], stats_df['avg_margin'], color='coral')
        ax3.set_title('パターン別営業利益率')
        ax3.set_xlabel('成長パターン')
        ax3.set_ylabel('営業利益率')
        ax3.tick_params(axis='x', rotation=45)
        
        # 4. パターン別ROE
        ax4 = axes[1, 1]
        ax4.bar(stats_df['pattern'], stats_df['avg_roe'], color='gold')
        ax4.set_title('パターン別ROE')
        ax4.set_xlabel('成長パターン')
        ax4.set_ylabel('ROE')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
    
    def plot_market_category_comparison(self, save_path: Optional[str] = None):
        """
        市場カテゴリ別比較プロット
        
        Args:
            save_path: 保存パス（Noneの場合は表示のみ）
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        market_comparison = self.analyzer.get_market_growth_comparison()
        
        if len(market_comparison) == 0:
            print("市場比較データがありません")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('市場カテゴリ別成長パターン比較', fontsize=16, fontweight='bold')
        
        # 1. 市場別企業数
        ax1 = axes[0, 0]
        ax1.bar(market_comparison['market_category'], 
                market_comparison['num_companies'], color='lightblue')
        ax1.set_title('市場別新設企業数')
        ax1.set_xlabel('市場カテゴリ')
        ax1.set_ylabel('企業数')
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. 市場別平均成長率
        ax2 = axes[0, 1]
        bars = ax2.bar(market_comparison['market_category'], 
                        market_comparison['avg_sales_growth'], color='lightgreen')
        ax2.set_title('市場別平均成長率')
        ax2.set_xlabel('市場カテゴリ')
        ax2.set_ylabel('平均成長率')
        ax2.tick_params(axis='x', rotation=45)
        
        # 成長率に応じて色分け
        for i, bar in enumerate(bars):
            growth_rate = market_comparison.iloc[i]['avg_sales_growth']
            if growth_rate > 0.1:
                bar.set_color('green')
            elif growth_rate > 0:
                bar.set_color('yellow')
            else:
                bar.set_color('red')
        
        # 3. 市場別営業利益率
        ax3 = axes[1, 0]
        ax3.bar(market_comparison['market_category'], 
                market_comparison['avg_operating_margin'], color='coral')
        ax3.set_title('市場別営業利益率')
        ax3.set_xlabel('市場カテゴリ')
        ax3.set_ylabel('営業利益率')
        ax3.tick_params(axis='x', rotation=45)
        
        # 4. 市場別ROE
        ax4 = axes[1, 1]
        ax4.bar(market_comparison['market_category'], 
                market_comparison['avg_roe'], color='gold')
        ax4.set_title('市場別ROE')
        ax4.set_xlabel('市場カテゴリ')
        ax4.set_ylabel('ROE')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
    
    def plot_growth_prediction(self, company_id: str, future_years: int = 5, 
                                save_path: Optional[str] = None):
        """
        成長予測プロット
        
        Args:
            company_id: 対象企業ID
            future_years: 予測年数
            save_path: 保存パス（Noneの場合は表示のみ）
        """
        import matplotlib.pyplot as plt
        
        # 予測実行
        prediction = self.analyzer.predict_growth_trajectory(company_id, future_years)
        
        if 'error' in prediction:
            print(f"予測エラー: {prediction['error']}")
            return
        
        # 現在データ
        company_data = self.trajectory_data[
            self.trajectory_data['company_id'] == company_id
        ].sort_values('years_since_establishment')
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # 実績値プロット
        ax.plot(company_data['years_since_establishment'], 
                company_data['sales'], 'b-o', linewidth=2, markersize=8, 
                label='実績値', alpha=0.8)
        
        # 予測値プロット
        ax.plot(prediction['prediction_years'], 
                prediction['predicted_sales'], 'r--s', linewidth=2, markersize=8,
                label='予測値', alpha=0.8)
        
        # 境界線
        current_year = prediction['current_year']
        ax.axvline(x=current_year, color='gray', linestyle=':', alpha=0.7, 
                    label='現在年')
        
        ax.set_title(f'{company_id} 成長軌道予測\n'
                    f'パターン: {prediction["growth_pattern"]}, '
                    f'R²: {prediction["model_r_squared"]:.3f}, '
                    f'信頼度: {prediction["confidence"]}', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('設立からの経過年数', fontsize=12)
        ax.set_ylabel('売上高', fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # 信頼区間の表示（簡易版）
        confidence_level = {'high': 0.1, 'medium': 0.2, 'low': 0.3, 'very_low': 0.4}
        error_margin = confidence_level.get(prediction['confidence'], 0.4)
        
        pred_upper = [y * (1 + error_margin) for y in prediction['predicted_sales']]
        pred_lower = [y * (1 - error_margin) for y in prediction['predicted_sales']]
        
        ax.fill_between(prediction['prediction_years'], pred_lower, pred_upper, 
                        alpha=0.2, color='red', label='予測信頼区間')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()


# 使用例の拡張
def demo_growth_trajectory_analysis():
    """成長軌道分析のデモンストレーション"""
    
    print("=== A2AI 成長軌道分析システム デモンストレーション ===\n")
    
    # サンプルデータ作成
    emergence_data, factor_data = create_sample_data()
    print("✅ サンプルデータ作成完了")
    print(f"   - 新設企業数: {len(emergence_data)}")
    print(f"   - データポイント数: {len(factor_data)}")
    
    # 分析実行
    print("\n🔄 成長軌道分析実行中...")
    analyzer = GrowthTrajectoryAnalyzer()
    analyzer.fit(emergence_data, factor_data)
    print("✅ 成長軌道分析完了")
    
    # 結果表示
    print("\n📊 分析結果:")
    
    # 1. 市場別比較
    market_comparison = analyzer.get_market_growth_comparison()
    print("\n【市場カテゴリ別成長パターン】")
    for _, row in market_comparison.iterrows():
        print(f"  {row['market_category']}: "
                f"企業数={row['num_companies']}, "
                f"平均成長率={row['avg_sales_growth']:.1%}, "
                f"営業利益率={row['avg_operating_margin']:.1%}")
    
    # 2. 成功要因分析
    success_factors = analyzer.get_success_factors()
    print(f"\n【成長クラスタ分析】")
    print(f"  識別されたクラスタ数: {len(success_factors)}")
    
    # 3. 個別企業予測
    print(f"\n【個別企業成長予測】")
    for company in emergence_data['company_id']:
        try:
            prediction = analyzer.predict_growth_trajectory(company, future_years=3)
            if 'error' not in prediction:
                print(f"  {company}:")
                print(f"    成長パターン: {prediction['growth_pattern']}")
                print(f"    モデル精度(R²): {prediction['model_r_squared']:.3f}")
                print(f"    予測信頼度: {prediction['confidence']}")
                print(f"    3年後予測売上: {prediction['predicted_sales'][-1]:.0f}")
        except Exception as e:
            print(f"  {company}: 予測エラー - {str(e)}")
    
    # 4. フェーズ遷移分析
    if analyzer.phase_transitions is not None and len(analyzer.phase_transitions) > 0:
        print(f"\n【成長フェーズ遷移】")
        print(f"  検出された遷移パターン: {len(analyzer.phase_transitions)}件")
        transition_summary = analyzer.phase_transitions.groupby(['from_phase', 'to_phase']).size()
        for (from_p, to_p), count in transition_summary.items():
            print(f"    {from_p} → {to_p}: {count}件")
    
    print("\n🎯 成長軌道分析完了")
    print("   この分析により、新設企業の成長パターンと成功要因が明らかになりました。")
    
    return analyzer


# メイン実行部分の拡張
if __name__ == "__main__":
    # デモ実行
    analyzer = demo_growth_trajectory_analysis()
    
    print("\n" + "="*60)
    print("A2AI growth_trajectory.py の主要機能:")
    print("1. 新設企業の成長パターン分類（指数、線形、S字等）")
    print("2. 成長フェーズ分析（導入期、成長期、成熟期、衰退期）")
    print("3. 市場カテゴリ別成長パターン比較")
    print("4. クラスタリングによる成功要因分析")
    print("5. 将来成長軌道予測")
    print("6. フェーズ遷移分析")
    print("="*60)