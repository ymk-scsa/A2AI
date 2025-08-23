"""
A2AI (Advanced Financial Analysis AI)
新設企業成功分析モジュール

企業の新設・分社化から成功に至るまでの要因を分析するモジュール
対象企業例：
- デンソーウェーブ（2001年設立、デンソーから分社）
- キオクシア（2018年設立、東芝メモリから独立）
- プロテリアル（2023年設立、日立金属から独立）
- パナソニックエナジー（2022年分社）
など
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class StartupMetrics:
    """新設企業の成功指標を格納するデータクラス"""
    company_name: str
    establishment_year: int
    parent_company: Optional[str]
    market_category: str  # 'high_share', 'declining', 'lost'
    years_since_establishment: int
    
    # 成功指標
    revenue_growth_rate: float
    profit_margin_growth: float
    market_share_growth: float
    employee_growth_rate: float
    rd_investment_ratio: float
    
    # 財務健全性
    debt_to_equity_ratio: float
    current_ratio: float
    roe_trend: float
    cash_flow_stability: float
    
    # イノベーション指標
    patent_applications: int
    new_product_ratio: float
    technology_licensing_revenue: float
    
    # 市場適応性
    overseas_revenue_ratio: float
    segment_diversification: int
    customer_concentration_risk: float

class StartupSuccessAnalyzer:
    """新設企業成功分析クラス"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初期化
        
        Parameters:
        config (dict): 分析設定パラメータ
        """
        self.config = config or self._get_default_config()
        self.data: Optional[pd.DataFrame] = None
        self.models: Dict[str, Any] = {}
        self.scaler = StandardScaler()
        self.success_metrics_weights = self._initialize_success_weights()
        self.analysis_results: Dict[str, Any] = {}
        
    def _get_default_config(self) -> Dict[str, Any]:
        """デフォルト設定を取得"""
        return {
            'success_threshold_years': 5,  # 成功判定までの年数
            'min_revenue_growth': 0.1,     # 最小売上成長率
            'min_market_share_growth': 0.05, # 最小市場シェア成長率
            'profitability_weight': 0.3,   # 収益性の重み
            'growth_weight': 0.4,          # 成長性の重み
            'stability_weight': 0.3,       # 安定性の重み
            'random_state': 42,
            'cv_folds': 5
        }
    
    def _initialize_success_weights(self) -> Dict[str, float]:
        """成功指標の重み付けを初期化"""
        return {
            'revenue_growth': 0.25,
            'profit_margin': 0.20,
            'market_share': 0.20,
            'innovation': 0.15,
            'financial_health': 0.20
        }
    
    def load_startup_data(self, data_path: str = None, 
                            startup_data: pd.DataFrame = None) -> pd.DataFrame:
        """
        新設企業データを読み込み
        
        Parameters:
        data_path (str): データファイルのパス
        startup_data (DataFrame): 直接渡されるデータフレーム
        
        Returns:
        DataFrame: 処理済み新設企業データ
        """
        try:
            if startup_data is not None:
                self.data = startup_data.copy()
            elif data_path:
                self.data = pd.read_csv(data_path)
            else:
                # サンプルデータを生成
                self.data = self._generate_sample_data()
            
            # データ前処理
            self.data = self._preprocess_startup_data(self.data)
            
            logger.info(f"新設企業データを読み込みました: {len(self.data)} 社")
            return self.data
            
        except Exception as e:
            logger.error(f"データ読み込みエラー: {e}")
            raise
    
    def _generate_sample_data(self) -> pd.DataFrame:
        """分析用サンプルデータを生成（実際の企業を参考）"""
        np.random.seed(self.config['random_state'])
        
        # 実際の企業例を基にしたサンプルデータ
        companies = [
            # 高シェア市場の新設企業
            {'name': 'デンソーウェーブ', 'establishment_year': 2001, 'parent': 'デンソー', 
                'market_category': 'high_share', 'main_business': 'ロボット'},
            {'name': 'プロテリアル', 'establishment_year': 2023, 'parent': '日立金属', 
                'market_category': 'high_share', 'main_business': '電子材料'},
            
            # シェア低下市場の新設企業
            {'name': 'パナソニックエナジー', 'establishment_year': 2022, 'parent': 'パナソニック', 
                'market_category': 'declining', 'main_business': 'バッテリー'},
            {'name': 'トヨタコネクティッド', 'establishment_year': 2016, 'parent': 'トヨタ', 
                'market_category': 'declining', 'main_business': '自動車IT'},
            
            # 失失市場の新設企業
            {'name': 'キオクシア', 'establishment_year': 2018, 'parent': '東芝', 
                'market_category': 'lost', 'main_business': '半導体'},
            {'name': 'ソニーセミコンダクタ', 'establishment_year': 2016, 'parent': 'ソニー', 
                'market_category': 'lost', 'main_business': '半導体'}
        ]
        
        sample_data = []
        current_year = 2024
        
        for company in companies:
            years_active = current_year - company['establishment_year']
            
            # 市場カテゴリ別の成功傾向を反映
            if company['market_category'] == 'high_share':
                base_growth = np.random.normal(0.15, 0.05)
                innovation_factor = np.random.normal(1.2, 0.2)
            elif company['market_category'] == 'declining':
                base_growth = np.random.normal(0.08, 0.04)
                innovation_factor = np.random.normal(1.0, 0.15)
            else:  # lost market
                base_growth = np.random.normal(0.03, 0.06)
                innovation_factor = np.random.normal(0.8, 0.2)
            
            # 年度別データ生成
            for year_offset in range(max(1, years_active)):
                year = company['establishment_year'] + year_offset
                age = year_offset + 1
                
                # 企業年齢による成長曲線効果
                age_factor = 1.0 / (1.0 + 0.1 * age)  # 新しいほど高成長
                
                sample_data.append({
                    'company_name': company['name'],
                    'year': year,
                    'establishment_year': company['establishment_year'],
                    'parent_company': company.get('parent'),
                    'market_category': company['market_category'],
                    'years_since_establishment': age,
                    'main_business': company['main_business'],
                    
                    # 成長指標
                    'revenue_growth_rate': max(0, base_growth * age_factor + np.random.normal(0, 0.03)),
                    'profit_margin_growth': max(-0.1, np.random.normal(0.05, 0.02) * innovation_factor),
                    'market_share_growth': max(0, np.random.normal(0.03, 0.01) * age_factor),
                    'employee_growth_rate': max(0, np.random.normal(0.1, 0.03) * age_factor),
                    
                    # 投資指標
                    'rd_investment_ratio': max(0.01, np.random.normal(0.08, 0.02) * innovation_factor),
                    'capex_to_revenue_ratio': max(0.02, np.random.normal(0.12, 0.03)),
                    
                    # 財務健全性
                    'debt_to_equity_ratio': max(0.1, np.random.normal(0.4, 0.15)),
                    'current_ratio': max(1.0, np.random.normal(1.8, 0.3)),
                    'roe_trend': np.random.normal(0.12, 0.04),
                    'cash_flow_stability': max(0.1, np.random.normal(0.15, 0.05)),
                    
                    # イノベーション
                    'patent_applications': max(0, int(np.random.poisson(5) * innovation_factor)),
                    'new_product_ratio': max(0, np.random.normal(0.3, 0.1) * innovation_factor),
                    'technology_licensing_revenue': max(0, np.random.normal(0.02, 0.01)),
                    
                    # 市場適応性
                    'overseas_revenue_ratio': max(0, min(1, np.random.normal(0.4, 0.2))),
                    'segment_diversification': max(1, int(np.random.poisson(3))),
                    'customer_concentration_risk': max(0.1, min(0.8, np.random.normal(0.3, 0.1)))
                })
        
        return pd.DataFrame(sample_data)
    
    def _preprocess_startup_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """新設企業データの前処理"""
        try:
            # 欠損値処理
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
            
            # カテゴリ変数のエンコーディング
            if 'market_category' in df.columns:
                category_mapping = {'high_share': 2, 'declining': 1, 'lost': 0}
                df['market_category_encoded'] = df['market_category'].map(category_mapping)
            
            # 成功スコア計算
            df['success_score'] = self._calculate_success_score(df)
            
            # 成功フラグ（閾値以上を成功とする）
            success_threshold = df['success_score'].quantile(0.7)
            df['is_successful'] = (df['success_score'] >= success_threshold).astype(int)
            
            logger.info("データ前処理完了")
            return df
            
        except Exception as e:
            logger.error(f"前処理エラー: {e}")
            raise
    
    def _calculate_success_score(self, df: pd.DataFrame) -> pd.Series:
        """
        新設企業の総合成功スコアを計算
        
        Parameters:
        df (DataFrame): 企業データ
        
        Returns:
        Series: 成功スコア
        """
        # 各指標を0-1の範囲に正規化
        def normalize_metric(series):
            return (series - series.min()) / (series.max() - series.min() + 1e-8)
        
        # 成長性スコア
        growth_score = (
            normalize_metric(df['revenue_growth_rate']) * 0.4 +
            normalize_metric(df['market_share_growth']) * 0.3 +
            normalize_metric(df['employee_growth_rate']) * 0.3
        )
        
        # 収益性スコア
        profitability_score = (
            normalize_metric(df['profit_margin_growth']) * 0.6 +
            normalize_metric(df['roe_trend']) * 0.4
        )
        
        # イノベーションスコア
        innovation_score = (
            normalize_metric(df['rd_investment_ratio']) * 0.4 +
            normalize_metric(df['patent_applications']) * 0.3 +
            normalize_metric(df['new_product_ratio']) * 0.3
        )
        
        # 安定性スコア（負の指標は逆転）
        stability_score = (
            normalize_metric(df['current_ratio']) * 0.4 +
            normalize_metric(1 / (df['debt_to_equity_ratio'] + 1e-8)) * 0.3 +
            normalize_metric(df['cash_flow_stability']) * 0.3
        )
        
        # 総合スコア算出
        total_score = (
            growth_score * self.success_metrics_weights['revenue_growth'] +
            profitability_score * self.success_metrics_weights['profit_margin'] +
            innovation_score * self.success_metrics_weights['innovation'] +
            stability_score * self.success_metrics_weights['financial_health']
        )
        
        return total_score
    
    def analyze_startup_success_factors(self) -> Dict[str, Any]:
        """
        新設企業の成功要因分析を実行
        
        Returns:
        Dict: 分析結果
        """
        if self.data is None:
            raise ValueError("データが読み込まれていません")
        
        logger.info("新設企業成功要因分析を開始")
        
        results = {}
        
        # 1. 基本統計分析
        results['basic_stats'] = self._analyze_basic_statistics()
        
        # 2. 市場カテゴリ別分析
        results['market_category_analysis'] = self._analyze_by_market_category()
        
        # 3. 企業年齢別分析
        results['age_analysis'] = self._analyze_by_company_age()
        
        # 4. 成功要因の機械学習分析
        results['ml_analysis'] = self._perform_ml_analysis()
        
        # 5. 親会社依存度分析
        results['parent_dependency_analysis'] = self._analyze_parent_dependency()
        
        # 6. 時系列トレンド分析
        results['temporal_analysis'] = self._analyze_temporal_trends()
        
        self.analysis_results = results
        logger.info("分析完了")
        
        return results
    
    def _analyze_basic_statistics(self) -> Dict[str, Any]:
        """基本統計分析"""
        stats_data = {}
        
        # 成功企業と非成功企業の比較
        successful = self.data[self.data['is_successful'] == 1]
        unsuccessful = self.data[self.data['is_successful'] == 0]
        
        key_metrics = [
            'revenue_growth_rate', 'profit_margin_growth', 'rd_investment_ratio',
            'patent_applications', 'overseas_revenue_ratio', 'years_since_establishment'
        ]
        
        for metric in key_metrics:
            stats_data[metric] = {
                'successful_mean': successful[metric].mean(),
                'unsuccessful_mean': unsuccessful[metric].mean(),
                'difference': successful[metric].mean() - unsuccessful[metric].mean(),
                't_stat': stats.ttest_ind(successful[metric], unsuccessful[metric])[0],
                'p_value': stats.ttest_ind(successful[metric], unsuccessful[metric])[1]
            }
        
        return stats_data
    
    def _analyze_by_market_category(self) -> Dict[str, Any]:
        """市場カテゴリ別の成功率分析"""
        category_analysis = {}
        
        for category in self.data['market_category'].unique():
            if pd.isna(category):
                continue
                
            category_data = self.data[self.data['market_category'] == category]
            
            category_analysis[category] = {
                'company_count': len(category_data),
                'success_rate': category_data['is_successful'].mean(),
                'avg_success_score': category_data['success_score'].mean(),
                'avg_revenue_growth': category_data['revenue_growth_rate'].mean(),
                'avg_rd_ratio': category_data['rd_investment_ratio'].mean(),
                'avg_years_to_establish': category_data['years_since_establishment'].mean(),
                
                # 成功企業の特徴
                'successful_companies': category_data[
                    category_data['is_successful'] == 1
                ]['company_name'].unique().tolist() if len(category_data[category_data['is_successful'] == 1]) > 0 else []
            }
        
        return category_analysis
    
    def _analyze_by_company_age(self) -> Dict[str, Any]:
        """企業年齢別の成功パターン分析"""
        age_analysis = {}
        
        # 年齢グループ分け
        self.data['age_group'] = pd.cut(
            self.data['years_since_establishment'], 
            bins=[0, 3, 7, 15, float('inf')], 
            labels=['初期段階(1-3年)', '成長期(4-7年)', '成熟期(8-15年)', '安定期(15年以上)']
        )
        
        for age_group in self.data['age_group'].cat.categories:
            group_data = self.data[self.data['age_group'] == age_group]
            
            if len(group_data) == 0:
                continue
            
            age_analysis[age_group] = {
                'company_count': len(group_data),
                'success_rate': group_data['is_successful'].mean(),
                'avg_revenue_growth': group_data['revenue_growth_rate'].mean(),
                'avg_profit_margin': group_data['profit_margin_growth'].mean(),
                'avg_rd_investment': group_data['rd_investment_ratio'].mean(),
                'innovation_intensity': group_data['patent_applications'].mean(),
                
                # 主要成功要因（相関分析）
                'key_success_factors': self._identify_key_factors_for_group(group_data)
            }
        
        return age_analysis
    
    def _identify_key_factors_for_group(self, group_data: pd.DataFrame) -> Dict[str, float]:
        """特定グループの主要成功要因を特定"""
        factor_columns = [
            'revenue_growth_rate', 'profit_margin_growth', 'rd_investment_ratio',
            'patent_applications', 'overseas_revenue_ratio', 'current_ratio',
            'capex_to_revenue_ratio', 'new_product_ratio'
        ]
        
        correlations = {}
        for factor in factor_columns:
            if factor in group_data.columns:
                corr = group_data[factor].corr(group_data['success_score'])
                if not pd.isna(corr):
                    correlations[factor] = abs(corr)
        
        # 上位5つの要因を返す
        sorted_factors = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_factors[:5])
    
    def _perform_ml_analysis(self) -> Dict[str, Any]:
        """機械学習による成功要因分析"""
        # 特徴量準備
        feature_columns = [
            'revenue_growth_rate', 'profit_margin_growth', 'rd_investment_ratio',
            'patent_applications', 'overseas_revenue_ratio', 'current_ratio',
            'debt_to_equity_ratio', 'capex_to_revenue_ratio', 'new_product_ratio',
            'years_since_establishment', 'market_category_encoded'
        ]
        
        available_features = [col for col in feature_columns if col in self.data.columns]
        X = self.data[available_features].fillna(0)
        y = self.data['success_score']
        
        # データ分割
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=self.config['random_state']
        )
        
        # 標準化
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 複数モデルで学習
        models = {
            'RandomForest': RandomForestRegressor(
                n_estimators=100, 
                random_state=self.config['random_state']
            ),
            'GradientBoosting': GradientBoostingRegressor(
                n_estimators=100, 
                random_state=self.config['random_state']
            ),
            'LinearRegression': LinearRegression()
        }
        
        ml_results = {}
        
        for model_name, model in models.items():
            # 学習
            if model_name == 'LinearRegression':
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            # 評価
            ml_results[model_name] = {
                'mse': mean_squared_error(y_test, y_pred),
                'r2_score': r2_score(y_test, y_pred),
                'mae': mean_absolute_error(y_test, y_pred),
            }
            
            # 特徴量重要度（RandomForest, GradientBoosting）
            if hasattr(model, 'feature_importances_'):
                feature_importance = dict(zip(available_features, model.feature_importances_))
                ml_results[model_name]['feature_importance'] = dict(
                    sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
                )
            
            # 保存
            self.models[model_name] = model
        
        return ml_results
    
    def _analyze_parent_dependency(self) -> Dict[str, Any]:
        """親会社依存度による成功への影響分析"""
        dependency_analysis = {}
        
        # 親会社がある企業とない企業で比較
        has_parent = self.data[self.data['parent_company'].notna()]
        no_parent = self.data[self.data['parent_company'].isna()]
        
        dependency_analysis['comparison'] = {
            'with_parent': {
                'count': len(has_parent),
                'success_rate': has_parent['is_successful'].mean(),
                'avg_success_score': has_parent['success_score'].mean(),
                'avg_revenue_growth': has_parent['revenue_growth_rate'].mean(),
            },
            'without_parent': {
                'count': len(no_parent),
                'success_rate': no_parent['is_successful'].mean() if len(no_parent) > 0 else 0,
                'avg_success_score': no_parent['success_score'].mean() if len(no_parent) > 0 else 0,
                'avg_revenue_growth': no_parent['revenue_growth_rate'].mean() if len(no_parent) > 0 else 0,
            }
        }
        
        # 統計的有意性検定
        if len(has_parent) > 0 and len(no_parent) > 0:
            t_stat, p_value = stats.ttest_ind(
                has_parent['success_score'], 
                no_parent['success_score']
            )
            dependency_analysis['statistical_test'] = {
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
        
        return dependency_analysis
    
    def _analyze_temporal_trends(self) -> Dict[str, Any]:
        """時系列でのトレンド分析"""
        temporal_analysis = {}
        
        if 'year' in self.data.columns:
            # 年度別成功率トレンド
            yearly_stats = self.data.groupby('year').agg({
                'is_successful': 'mean',
                'success_score': 'mean',
                'revenue_growth_rate': 'mean',
                'rd_investment_ratio': 'mean',
                'company_name': 'nunique'
            }).rename(columns={'company_name': 'active_companies'})
            
            temporal_analysis['yearly_trends'] = yearly_stats.to_dict('index')
            
            # 設立年代別分析
            establishment_decade = (self.data['establishment_year'] // 10) * 10
            self.data['establishment_decade'] = establishment_decade
            
            decade_stats = self.data.groupby('establishment_decade').agg({
                'is_successful': 'mean',
                'success_score': 'mean',
                'years_since_establishment': 'mean',
                'company_name': 'nunique'
            }).rename(columns={'company_name': 'company_count'})
            
            temporal_analysis['establishment_decade_analysis'] = decade_stats.to_dict('index')
        
        return temporal_analysis
    
    def predict_startup_success(self, company_features: Dict[str, float]) -> Dict[str, Any]:
        """
        新設企業の成功確率を予測
        
        Parameters:
        company_features (dict): 企業の特徴量
        
        Returns:
        Dict: 予測結果
        """
        if not self.models:
            raise ValueError("モデルが学習されていません。先にanalyze_startup_success_factors()を実行してください。")
        
        # 特徴量をDataFrameに変換
        features_df = pd.DataFrame([company_features])
        
        predictions = {}
        
        for model_name, model in self.models.items():
            try:
                if model_name == 'LinearRegression':
                    # 標準化が必要
                    features_scaled = self.scaler.transform(features_df)
                    pred = model.predict(features_scaled)[0]
                else:
                    pred = model.predict(features_df)[0]
                
                predictions[model_name] = {
                    'success_score_prediction': pred,
                    'success_probability': min(1.0, max(0.0, pred))  # 0-1に正規化
                }
                
            except Exception as e:
                logger.warning(f"{model_name}での予測でエラー: {e}")
                continue
        
        # アンサンブル予測（平均）
        if predictions:
            ensemble_score = np.mean([p['success_score_prediction'] for p in predictions.values()])
            predictions['ensemble'] = {
                'success_score_prediction': ensemble_score,
                'success_probability': min(1.0, max(0.0, ensemble_score))
            }
        
        return predictions
    
    def generate_success_recommendations(self, company_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        新設企業向けの成功戦略提言を生成
        
        Parameters:
        company_data (dict): 特定企業のデータ（オプション）
        
        Returns:
        Dict: 戦略提言
        """
        recommendations = {}
        
        if not self.analysis_results:
            raise ValueError("分析が実行されていません。先にanalyze_startup_success_factors()を実行してください。")
        
        # 市場カテゴリ別推奨戦略
        market_recommendations = {}
        
        market_analysis = self.analysis_results.get('market_category_analysis', {})
        for category, data in market_analysis.items():
            success_rate = data.get('success_rate', 0)
            
            if category == 'high_share':
                market_recommendations[category] = {
                    'strategy_focus': ['技術革新', 'グローバル展開', 'プラットフォーム化'],
                    'key_success_factors': ['R&D投資比率', '特許申請数', '海外売上比率'],
                    'recommended_rd_ratio': '8%以上',
                    'target_growth_rate': '年15%以上',
                    'market_outlook': '高い成功確率が期待できる市場'
                }
            elif category == 'declining':
                market_recommendations[category] = {
                    'strategy_focus': ['事業転換', 'デジタル化', 'サービス化'],
                    'key_success_factors': ['新規事業比率', 'デジタル投資', '収益多様化'],
                    'recommended_rd_ratio': '10%以上',
                    'target_growth_rate': '年8%以上',
                    'market_outlook': '戦略的転換が成功のカギ'
                }
            else:  # lost market
                market_recommendations[category] = {
                    'strategy_focus': ['ニッチ特化', '高付加価値化', '新市場開拓'],
                    'key_success_factors': ['専門技術力', '顧客密着', 'コスト効率'],
                    'recommended_rd_ratio': '12%以上',
                    'target_growth_rate': '年3%以上',
                    'market_outlook': '困難だが差別化により成功可能'
                }
        
        recommendations['market_strategy'] = market_recommendations
        
        # 企業年齢別推奨戦略
        age_recommendations = {
            '初期段階(1-3年)': {
                'priority': ['製品開発', '市場検証', '資金調達'],
                'key_metrics': ['技術特許', '顧客獲得', 'キャッシュフロー'],
                'risk_factors': ['資金不足', '市場適応', '人材確保']
            },
            '成長期(4-7年)': {
                'priority': ['事業拡大', '組織構築', '品質向上'],
                'key_metrics': ['売上成長率', '市場シェア', '従業員定着率'],
                'risk_factors': ['スケール課題', '競合参入', '品質管理']
            },
            '成熟期(8-15年)': {
                'priority': ['事業多様化', 'グローバル展開', 'イノベーション'],
                'key_metrics': ['ROE', '海外売上比率', '新製品比率'],
                'risk_factors': ['成長鈍化', '組織硬直化', '技術陳腐化']
            },
            '安定期(15年以上)': {
                'priority': ['事業再構築', 'デジタル変革', '次世代育成'],
                'key_metrics': ['事業転換率', 'デジタル投資', '後継者育成'],
                'risk_factors': ['既存事業依存', '変化適応力', '世代交代']
            }
        }
        
        recommendations['lifecycle_strategy'] = age_recommendations
        
        # 成功要因ランキング（機械学習分析結果から）
        ml_analysis = self.analysis_results.get('ml_analysis', {})
        best_model_name = None
        best_r2 = -1
        
        for model_name, results in ml_analysis.items():
            if results.get('r2_score', 0) > best_r2:
                best_r2 = results['r2_score']
                best_model_name = model_name
        
        if best_model_name and 'feature_importance' in ml_analysis[best_model_name]:
            feature_importance = ml_analysis[best_model_name]['feature_importance']
            
            recommendations['critical_success_factors'] = {
                'top_5_factors': list(feature_importance.items())[:5],
                'factor_analysis': {
                    factor: {
                        'importance_score': importance,
                        'recommended_action': self._get_factor_recommendation(factor)
                    }
                    for factor, importance in list(feature_importance.items())[:5]
                }
            }
        
        # 特定企業向けカスタム提言
        if company_data:
            recommendations['custom_recommendations'] = self._generate_custom_recommendations(company_data)
        
        return recommendations
    
    def _get_factor_recommendation(self, factor: str) -> str:
        """成功要因に対する具体的推奨アクション"""
        recommendations = {
            'revenue_growth_rate': 'マーケット拡大とプロダクトライン拡充による売上成長の継続',
            'rd_investment_ratio': 'R&D投資を売上の8-12%に維持し、技術的差別化を強化',
            'patent_applications': '年間5-10件以上の特許申請で知的財産ポートフォリオを構築',
            'overseas_revenue_ratio': '海外売上比率40%以上を目標にグローバル市場開拓',
            'profit_margin_growth': '高付加価値製品・サービスによる利益率向上',
            'current_ratio': '流動比率1.5以上維持で財務安定性確保',
            'new_product_ratio': '売上の30%以上を新製品で確保し、イノベーション継続',
            'capex_to_revenue_ratio': '設備投資を売上の10-15%に最適化し、生産性向上',
            'market_category_encoded': '市場ポジショニング戦略の見直しと適応'
        }
        
        return recommendations.get(factor, 'データ分析に基づく戦略最適化を推奨')
    
    def _generate_custom_recommendations(self, company_data: Dict[str, Any]) -> Dict[str, Any]:
        """特定企業向けカスタム提言"""
        custom_recs = {}
        
        # 予測スコア取得
        try:
            predictions = self.predict_startup_success(company_data)
            ensemble_prob = predictions.get('ensemble', {}).get('success_probability', 0)
            
            custom_recs['current_status'] = {
                'success_probability': ensemble_prob,
                'performance_level': self._classify_performance_level(ensemble_prob)
            }
            
            # 弱点分析と改善提案
            weaknesses = self._identify_weaknesses(company_data)
            custom_recs['improvement_areas'] = weaknesses
            
            # 目標設定
            custom_recs['recommended_targets'] = self._generate_targets(company_data, ensemble_prob)
            
        except Exception as e:
            logger.warning(f"カスタム提言生成でエラー: {e}")
            custom_recs['status'] = 'データ不足により詳細分析が困難'
        
        return custom_recs
    
    def _classify_performance_level(self, probability: float) -> str:
        """成功確率に基づくパフォーマンスレベル分類"""
        if probability >= 0.8:
            return 'エクセレント（高成功確率）'
        elif probability >= 0.6:
            return 'グッド（中程度成功確率）'
        elif probability >= 0.4:
            return 'アベレージ（平均的）'
        else:
            return 'チャレンジング（要改善）'
    
    def _identify_weaknesses(self, company_data: Dict[str, Any]) -> List[Dict[str, str]]:
        """企業の弱点特定"""
        weaknesses = []
        
        # ベンチマーク値
        benchmarks = {
            'revenue_growth_rate': 0.1,
            'rd_investment_ratio': 0.08,
            'profit_margin_growth': 0.05,
            'current_ratio': 1.5,
            'overseas_revenue_ratio': 0.3
        }
        
        for metric, benchmark in benchmarks.items():
            if metric in company_data:
                value = company_data[metric]
                if value < benchmark:
                    weakness_level = 'critical' if value < benchmark * 0.5 else 'moderate'
                    weaknesses.append({
                        'metric': metric,
                        'current_value': value,
                        'benchmark': benchmark,
                        'gap': benchmark - value,
                        'priority': weakness_level,
                        'improvement_action': self._get_improvement_action(metric, value, benchmark)
                    })
        
        return sorted(weaknesses, key=lambda x: x['gap'], reverse=True)
    
    def _get_improvement_action(self, metric: str, current: float, benchmark: float) -> str:
        """改善アクション提案"""
        actions = {
            'revenue_growth_rate': f'売上成長率を{benchmark:.1%}以上に向上（新市場開拓、製品ライン拡充）',
            'rd_investment_ratio': f'R&D投資比率を{benchmark:.1%}以上に増強（技術革新、特許取得）',
            'profit_margin_growth': f'利益率を{benchmark:.1%}以上に改善（高付加価値化、コスト最適化）',
            'current_ratio': f'流動比率を{benchmark:.1f}以上に改善（財務体質強化、運転資本管理）',
            'overseas_revenue_ratio': f'海外売上比率を{benchmark:.1%}以上に拡大（グローバル戦略、現地パートナー）'
        }
        
        return actions.get(metric, '戦略的改善が必要')
    
    def _generate_targets(self, company_data: Dict[str, Any], success_prob: float) -> Dict[str, Any]:
        """目標設定生成"""
        targets = {}
        
        # 現在のパフォーマンスレベルに基づく目標設定
        if success_prob >= 0.6:
            # 高パフォーマンス企業向け
            targets['strategic_focus'] = 'リーダーシップ拡大'
            targets['growth_target'] = '年20%以上の売上成長'
            targets['innovation_target'] = 'R&D投資比率12%以上'
            targets['global_target'] = '海外売上比率50%以上'
        else:
            # 改善必要企業向け
            targets['strategic_focus'] = '基盤強化と成長加速'
            targets['growth_target'] = '年10%以上の売上成長'
            targets['innovation_target'] = 'R&D投資比率8%以上'
            targets['global_target'] = '海外売上比率30%以上'
        
        # 企業年齢に応じた調整
        if 'years_since_establishment' in company_data:
            age = company_data['years_since_establishment']
            if age <= 3:
                targets['immediate_priority'] = '市場確立と基盤構築'
            elif age <= 7:
                targets['immediate_priority'] = '事業拡大と効率化'
            else:
                targets['immediate_priority'] = '変革と次世代準備'
        
        return targets
    
    def export_analysis_report(self, output_path: str = None) -> str:
        """分析レポートを出力"""
        if not self.analysis_results:
            raise ValueError("分析が実行されていません")
        
        report_sections = []
        
        # エグゼクティブサマリー
        report_sections.append("# 新設企業成功分析レポート\n")
        report_sections.append("## エグゼクティブサマリー\n")
        
        # 基本統計
        basic_stats = self.analysis_results.get('basic_stats', {})
        if basic_stats:
            report_sections.append("### 主要発見事項\n")
            for metric, stats in list(basic_stats.items())[:3]:
                diff = stats.get('difference', 0)
                p_val = stats.get('p_value', 1)
                significance = "統計的に有意" if p_val < 0.05 else "統計的に非有意"
                
                report_sections.append(
                    f"- {metric}: 成功企業は平均{diff:.3f}高い ({significance})\n"
                )
        
        # 市場カテゴリ分析
        market_analysis = self.analysis_results.get('market_category_analysis', {})
        if market_analysis:
            report_sections.append("\n## 市場カテゴリ別分析\n")
            for category, data in market_analysis.items():
                success_rate = data.get('success_rate', 0)
                company_count = data.get('company_count', 0)
                
                report_sections.append(f"### {category.upper()}市場\n")
                report_sections.append(f"- 企業数: {company_count}社\n")
                report_sections.append(f"- 成功率: {success_rate:.1%}\n")
                report_sections.append(f"- 平均売上成長率: {data.get('avg_revenue_growth', 0):.1%}\n")
                
                successful_companies = data.get('successful_companies', [])
                if successful_companies:
                    report_sections.append(f"- 成功企業例: {', '.join(successful_companies[:3])}\n")
                
                report_sections.append("\n")
        
        # 機械学習分析結果
        ml_analysis = self.analysis_results.get('ml_analysis', {})
        if ml_analysis:
            report_sections.append("## AI分析による成功要因\n")
            
            # 最高性能モデル特定
            best_model = max(ml_analysis.items(), 
                            key=lambda x: x[1].get('r2_score', 0))
            model_name, model_results = best_model
            
            report_sections.append(f"### 最適モデル: {model_name}\n")
            report_sections.append(f"- 予測精度 (R²): {model_results.get('r2_score', 0):.3f}\n")
            
            # 重要要因
            feature_importance = model_results.get('feature_importance', {})
            if feature_importance:
                report_sections.append("\n### 成功要因ランキング\n")
                for i, (factor, importance) in enumerate(list(feature_importance.items())[:5], 1):
                    report_sections.append(f"{i}. {factor}: {importance:.3f}\n")
        
        # 提言
        try:
            recommendations = self.generate_success_recommendations()
            report_sections.append("\n## 戦略提言\n")
            
            market_strategy = recommendations.get('market_strategy', {})
            for category, strategy in market_strategy.items():
                report_sections.append(f"### {category.upper()}市場戦略\n")
                focus_areas = strategy.get('strategy_focus', [])
                report_sections.append(f"- 重点領域: {', '.join(focus_areas)}\n")
                rd_ratio = strategy.get('recommended_rd_ratio', 'N/A')
                report_sections.append(f"- 推奨R&D投資比率: {rd_ratio}\n")
                
        except Exception as e:
            logger.warning(f"提言生成でエラー: {e}")
        
        # レポート結合
        full_report = "".join(report_sections)
        
        # ファイル出力
        if output_path:
            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(full_report)
                logger.info(f"レポートを出力しました: {output_path}")
            except Exception as e:
                logger.error(f"ファイル出力エラー: {e}")
        
        return full_report
    
    def visualize_analysis_results(self, save_path: str = None) -> plt.Figure:
        """分析結果の可視化"""
        if not self.analysis_results:
            raise ValueError("分析が実行されていません")
        
        # 図のセットアップ
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('新設企業成功分析結果', fontsize=16, fontweight='bold')
        
        # 日本語フォント設定（可能な場合）
        try:
            plt.rcParams['font.family'] = ['DejaVu Sans', 'Hiragino Sans', 'Yu Gothic', 'Meiryo']
        except:
            pass
        
        # 1. 市場カテゴリ別成功率
        market_analysis = self.analysis_results.get('market_category_analysis', {})
        if market_analysis:
            categories = list(market_analysis.keys())
            success_rates = [data.get('success_rate', 0) for data in market_analysis.values()]
            
            axes[0, 0].bar(categories, success_rates, color=['green', 'orange', 'red'])
            axes[0, 0].set_title('Market Category Success Rate')
            axes[0, 0].set_ylabel('Success Rate')
            axes[0, 0].set_ylim(0, 1)
            
            for i, v in enumerate(success_rates):
                axes[0, 0].text(i, v + 0.01, f'{v:.1%}', ha='center')
        
        # 2. 企業年齢別成功パターン
        if self.data is not None and 'age_group' in self.data.columns:
            age_success = self.data.groupby('age_group')['is_successful'].mean()
            
            axes[0, 1].bar(range(len(age_success)), age_success.values, 
                            color=['lightblue', 'blue', 'navy', 'darkblue'])
            axes[0, 1].set_title('Success Rate by Company Age')
            axes[0, 1].set_ylabel('Success Rate')
            axes[0, 1].set_xticks(range(len(age_success)))
            axes[0, 1].set_xticklabels(age_success.index, rotation=45)
        
        # 3. 成功要因重要度（機械学習結果）
        ml_analysis = self.analysis_results.get('ml_analysis', {})
        if ml_analysis:
            best_model = max(ml_analysis.items(), 
                            key=lambda x: x[1].get('r2_score', 0))
            
            feature_importance = best_model[1].get('feature_importance', {})
            if feature_importance:
                top_features = list(feature_importance.items())[:8]
                factors, importances = zip(*top_features)
                
                axes[0, 2].barh(range(len(factors)), importances, color='skyblue')
                axes[0, 2].set_title(f'Feature Importance ({best_model[0]})')
                axes[0, 2].set_xlabel('Importance')
                axes[0, 2].set_yticks(range(len(factors)))
                axes[0, 2].set_yticklabels(factors)
        
        # 4. 成功スコア分布
        if self.data is not None:
            axes[1, 0].hist(self.data['success_score'], bins=20, alpha=0.7, color='green', 
                            label='All Companies')
            
            successful = self.data[self.data['is_successful'] == 1]
            if len(successful) > 0:
                axes[1, 0].hist(successful['success_score'], bins=20, alpha=0.7, 
                                color='darkgreen', label='Successful Companies')
            
            axes[1, 0].set_title('Success Score Distribution')
            axes[1, 0].set_xlabel('Success Score')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].legend()
        
        # 5. R&D投資と成功の関係
        if self.data is not None and 'rd_investment_ratio' in self.data.columns:
            successful = self.data[self.data['is_successful'] == 1]
            unsuccessful = self.data[self.data['is_successful'] == 0]
            
            axes[1, 1].scatter(unsuccessful['rd_investment_ratio'], 
                                unsuccessful['success_score'], 
                                alpha=0.6, color='red', label='Unsuccessful')
            axes[1, 1].scatter(successful['rd_investment_ratio'], 
                                successful['success_score'], 
                                alpha=0.6, color='green', label='Successful')
            
            axes[1, 1].set_title('R&D Investment vs Success')
            axes[1, 1].set_xlabel('R&D Investment Ratio')
            axes[1, 1].set_ylabel('Success Score')
            axes[1, 1].legend()
        
        # 6. 時系列トレンド
        temporal_analysis = self.analysis_results.get('temporal_analysis', {})
        if temporal_analysis and 'yearly_trends' in temporal_analysis:
            yearly_data = temporal_analysis['yearly_trends']
            years = sorted(yearly_data.keys())
            success_rates = [yearly_data[year]['is_successful'] for year in years]
            
            axes[1, 2].plot(years, success_rates, marker='o', linewidth=2, color='blue')
            axes[1, 2].set_title('Success Rate Trend Over Time')
            axes[1, 2].set_xlabel('Year')
            axes[1, 2].set_ylabel('Success Rate')
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存
        if save_path:
            try:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"可視化結果を保存しました: {save_path}")
            except Exception as e:
                logger.error(f"可視化保存エラー: {e}")
        
        return fig

# 使用例とテストコード
if __name__ == "__main__":
    # アナライザーの初期化
    analyzer = StartupSuccessAnalyzer()
    
    # サンプルデータで分析実行
    print("新設企業データを読み込んでいます...")
    data = analyzer.load_startup_data()
    
    print("成功要因分析を実行しています...")
    results = analyzer.analyze_startup_success_factors()
    
    # 結果表示
    print("\n=== 分析結果サマリー ===")
    
    # 市場カテゴリ別結果
    market_analysis = results.get('market_category_analysis', {})
    print(f"\n市場カテゴリ別成功率:")
    for category, data in market_analysis.items():
        print(f"  {category}: {data.get('success_rate', 0):.1%} ({data.get('company_count', 0)}社)")
    
    # 成功要因ランキング
    ml_results = results.get('ml_analysis', {})
    if ml_results:
        best_model = max(ml_results.items(), key=lambda x: x[1].get('r2_score', 0))
        print(f"\n主要成功要因 (by {best_model[0]}):")
        
        feature_importance = best_model[1].get('feature_importance', {})
        for i, (factor, importance) in enumerate(list(feature_importance.items())[:5], 1):
            print(f"  {i}. {factor}: {importance:.3f}")
    
    # 戦略提言生成
    print("\n戦略提言を生成しています...")
    recommendations = analyzer.generate_success_recommendations()
    
    market_strategy = recommendations.get('market_strategy', {})
    for category, strategy in market_strategy.items():
        print(f"\n{category}市場戦略:")
        focus_areas = strategy.get('strategy_focus', [])
        print(f"  重点領域: {', '.join(focus_areas)}")
    
    # 予測テスト
    print("\n予測テストを実行しています...")
    test_company = {
        'revenue_growth_rate': 0.12,
        'profit_margin_growth': 0.06,
        'rd_investment_ratio': 0.10,
        'patent_applications': 8,
        'overseas_revenue_ratio': 0.35,
        'current_ratio': 1.8,
        'debt_to_equity_ratio': 0.3,
        'capex_to_revenue_ratio': 0.15,
        'new_product_ratio': 0.4,
        'years_since_establishment': 5,
        'market_category_encoded': 2
    }
    
    predictions = analyzer.predict_startup_success(test_company)
    ensemble_pred = predictions.get('ensemble', {})
    print(f"予測成功確率: {ensemble_pred.get('success_probability', 0):.1%}")
    
    # レポート出力
    print("\nレポートを生成しています...")
    report = analyzer.export_analysis_report()
    print(f"レポート長: {len(report)} 文字")
    
    print("\n分析完了!")