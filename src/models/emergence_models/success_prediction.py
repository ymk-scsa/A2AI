"""
A2AI - Advanced Financial Analysis AI
Emergence Models: Success Prediction Module

新設企業の成功予測モデル
企業設立後の成長軌道と成功要因を分析し、将来の成功確率を予測する

Author: A2AI Development Team
Created: 2024-08
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score
import xgboost as xgb
import lightgbm as lgb
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class EmergenceSuccessPredictor:
    """
    新設企業成功予測クラス
    
    機能：
    1. 新設企業の成功パターン学習
    2. 成功確率予測
    3. 成功要因の特定
    4. 市場別成功戦略の提言
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or self._get_default_config()
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.success_thresholds = {}
        self.market_patterns = {}
        
    def _get_default_config(self) -> Dict:
        """デフォルト設定"""
        return {
            'success_metrics': [
                'revenue_growth_5yr',      # 5年間売上成長率
                'market_share_achievement', # 市場シェア獲得度
                'profitability_timeline',   # 収益化達成時期
                'survival_probability'      # 生存確率
            ],
            'success_thresholds': {
                'revenue_growth_5yr': 0.5,        # 50%以上の成長
                'market_share_achievement': 0.05,  # 5%以上のシェア
                'profitability_timeline': 5,       # 5年以内の黒字化
                'survival_probability': 0.8        # 80%以上の生存確率
            },
            'prediction_horizon': [1, 3, 5, 10],  # 予測年数
            'model_types': ['logistic', 'rf', 'xgb', 'lgb'],
            'feature_groups': [
                'founder_characteristics',    # 創業者特性
                'initial_resources',         # 初期リソース
                'market_conditions',         # 市場環境
                'business_model',           # ビジネスモデル
                'financial_metrics',        # 財務指標
                'innovation_factors',       # イノベーション要因
                'timing_factors',           # タイミング要因
                'parent_company_support'    # 親会社サポート（分社企業）
            ]
        }
    
    def prepare_success_features(self, emergence_data: pd.DataFrame, 
                                financial_data: pd.DataFrame,
                                market_data: pd.DataFrame) -> pd.DataFrame:
        """
        成功予測用特徴量の準備
        
        Parameters:
        -----------
        emergence_data : pd.DataFrame
            新設企業基本データ
        financial_data : pd.DataFrame
            財務データ
        market_data : pd.DataFrame
            市場データ
            
        Returns:
        --------
        pd.DataFrame : 成功予測用特徴量データセット
        """
        features = pd.DataFrame()
        
        # 1. 創業者・企業特性
        features['company_age'] = emergence_data['years_since_establishment']
        features['initial_capital'] = emergence_data['initial_capital_million_yen']
        features['founder_experience'] = emergence_data['founder_prior_experience_years']
        features['parent_company_size'] = emergence_data.get('parent_company_assets', 0)
        features['spinoff_flag'] = (emergence_data['establishment_type'] == 'spinoff').astype(int)
        
        # 2. 初期財務指標（設立後1-2年）
        early_financial = financial_data[financial_data['years_since_establishment'] <= 2]
        if not early_financial.empty:
            features['initial_revenue'] = early_financial.groupby('company_id')['revenue'].first()
            features['initial_rnd_intensity'] = early_financial.groupby('company_id')['rnd_expenses_rate'].first()
            features['initial_asset_efficiency'] = early_financial.groupby('company_id')['total_asset_turnover'].first()
            features['initial_leverage'] = early_financial.groupby('company_id')['debt_to_equity_ratio'].first()
        
        # 3. 市場環境要因
        features['market_growth_rate'] = market_data['annual_growth_rate']
        features['market_concentration'] = market_data['hhi_index']
        features['market_maturity'] = market_data['market_age_years']
        features['competitive_intensity'] = market_data['competitor_count']
        features['technology_intensity'] = market_data['rnd_intensity_avg']
        
        # 4. イノベーション・技術要因
        features['patent_count_early'] = emergence_data['patents_first_3years']
        features['rnd_investment_ratio'] = financial_data.groupby('company_id')['rnd_expenses_rate'].mean()
        features['technology_novelty_score'] = emergence_data['technology_novelty_index']
        features['digital_transformation_score'] = emergence_data['digital_readiness_index']
        
        # 5. ビジネスモデル要因
        features['business_model_innovation'] = emergence_data['business_model_innovation_score']
        features['platform_model_flag'] = (emergence_data['business_type'] == 'platform').astype(int)
        features['b2b_focus_ratio'] = emergence_data['b2b_revenue_ratio']
        features['recurring_revenue_ratio'] = emergence_data['recurring_revenue_ratio']
        
        # 6. 市場参入タイミング
        features['first_mover_advantage'] = emergence_data['market_entry_timing_score']
        features['economic_cycle_phase'] = emergence_data['economic_cycle_at_establishment']
        features['industry_lifecycle_stage'] = market_data['industry_lifecycle_stage']
        
        # 7. リソース・ネットワーク
        features['venture_capital_backed'] = emergence_data['vc_investment_flag'].astype(int)
        features['strategic_partnerships'] = emergence_data['partnership_count_early']
        features['talent_quality_score'] = emergence_data['employee_quality_index']
        features['network_centrality'] = emergence_data['industry_network_centrality']
        
        # 8. 分社企業特有の要因
        if 'parent_company_support' in emergence_data.columns:
            features['parent_support_score'] = emergence_data['parent_company_support']
            features['technology_transfer_value'] = emergence_data['inherited_technology_value']
            features['customer_base_transfer'] = emergence_data['inherited_customer_ratio']
        
        return features.fillna(0)
    
    def define_success_labels(self, financial_data: pd.DataFrame,
                                market_data: pd.DataFrame,
                                horizon: int = 5) -> pd.Series:
        """
        成功ラベルの定義
        
        Parameters:
        -----------
        financial_data : pd.DataFrame
            財務データ
        market_data : pd.DataFrame
            市場データ  
        horizon : int
            評価期間（年）
            
        Returns:
        --------
        pd.Series : 成功フラグ（1: 成功, 0: 非成功）
        """
        success_criteria = {}
        
        # 1. 売上成長率基準
        revenue_growth = financial_data.groupby('company_id').apply(
            lambda x: self._calculate_cagr(x['revenue'], horizon)
        )
        success_criteria['revenue_growth'] = (
            revenue_growth >= self.config['success_thresholds']['revenue_growth_5yr']
        ).astype(int)
        
        # 2. 市場シェア獲得基準
        market_share = financial_data.groupby('company_id')['market_share'].max()
        success_criteria['market_share'] = (
            market_share >= self.config['success_thresholds']['market_share_achievement']
        ).astype(int)
        
        # 3. 収益性達成基準
        profitability_timeline = financial_data.groupby('company_id').apply(
            lambda x: self._years_to_profitability(x)
        )
        success_criteria['profitability'] = (
            profitability_timeline <= self.config['success_thresholds']['profitability_timeline']
        ).astype(int)
        
        # 4. 生存基準
        survival_data = financial_data.groupby('company_id')['years_since_establishment'].max()
        success_criteria['survival'] = (survival_data >= horizon).astype(int)
        
        # 総合成功判定（複数基準のAND条件）
        success_df = pd.DataFrame(success_criteria)
        
        # 重み付き成功スコア計算
        weights = {'revenue_growth': 0.3, 'market_share': 0.25, 
                    'profitability': 0.25, 'survival': 0.2}
        
        weighted_success = sum(success_df[criterion] * weight 
                                for criterion, weight in weights.items())
        
        # 閾値以上を成功とする
        success_threshold = 0.6
        return (weighted_success >= success_threshold).astype(int)
    
    def train_success_models(self, features: pd.DataFrame, 
                            labels: pd.Series,
                            market_categories: pd.Series = None) -> Dict:
        """
        成功予測モデルの学習
        
        Parameters:
        -----------
        features : pd.DataFrame
            特徴量データ
        labels : pd.Series
            成功ラベル
        market_categories : pd.Series
            市場カテゴリ（high_share, declining, lost）
            
        Returns:
        --------
        Dict : 学習済みモデル群
        """
        results = {'overall': {}, 'by_market': {}}
        
        # データ分割（時系列考慮）
        tscv = TimeSeriesSplit(n_splits=5)
        
        # 特徴量正規化
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(features)
        self.scalers['main'] = scaler
        
        # 1. 全体モデル学習
        for model_name in self.config['model_types']:
            model = self._get_model(model_name)
            
            # 交差検証
            cv_scores = cross_val_score(
                model, X_scaled, labels, 
                cv=tscv, scoring='roc_auc'
            )
            
            # 最終モデル学習
            model.fit(X_scaled, labels)
            
            # モデル保存
            self.models[model_name] = model
            
            # 特徴量重要度
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[model_name] = dict(
                    zip(features.columns, model.feature_importances_)
                )
            
            results['overall'][model_name] = {
                'cv_auc_mean': cv_scores.mean(),
                'cv_auc_std': cv_scores.std(),
                'feature_importance': self.feature_importance.get(model_name, {})
            }
        
        # 2. 市場別モデル学習
        if market_categories is not None:
            for market_cat in market_categories.unique():
                if market_cat not in ['high_share', 'declining', 'lost']:
                    continue
                    
                mask = market_categories == market_cat
                X_market = X_scaled[mask]
                y_market = labels[mask]
                
                if len(y_market) < 20:  # サンプル数不足の場合はスキップ
                    continue
                
                results['by_market'][market_cat] = {}
                self.models[f'{market_cat}_models'] = {}
                
                for model_name in ['rf', 'xgb']:  # 市場別は主要モデルのみ
                    model = self._get_model(model_name)
                    
                    if len(np.unique(y_market)) > 1:  # 成功・失敗両方のケースが存在
                        model.fit(X_market, y_market)
                        self.models[f'{market_cat}_models'][model_name] = model
                        
                        # 特徴量重要度
                        if hasattr(model, 'feature_importances_'):
                            importance_key = f'{market_cat}_{model_name}'
                            self.feature_importance[importance_key] = dict(
                                zip(features.columns, model.feature_importances_)
                            )
                        
                        results['by_market'][market_cat][model_name] = {
                            'samples': len(y_market),
                            'success_rate': y_market.mean(),
                            'feature_importance': self.feature_importance.get(importance_key, {})
                        }
        
        return results
    
    def predict_success_probability(self, features: pd.DataFrame,
                                    horizon: int = 5,
                                    market_category: str = None,
                                    ensemble: bool = True) -> Dict:
        """
        成功確率予測
        
        Parameters:
        -----------
        features : pd.DataFrame
            予測対象の特徴量
        horizon : int
            予測期間（年）
        market_category : str
            市場カテゴリ
        ensemble : bool
            アンサンブル予測使用フラグ
            
        Returns:
        --------
        Dict : 予測結果
        """
        # 特徴量正規化
        X_scaled = self.scalers['main'].transform(features)
        
        predictions = {}
        
        if ensemble and len(self.models) > 1:
            # アンサンブル予測
            proba_list = []
            weights = {'logistic': 0.2, 'rf': 0.3, 'xgb': 0.3, 'lgb': 0.2}
            
            for model_name, model in self.models.items():
                if model_name in weights:
                    proba = model.predict_proba(X_scaled)[:, 1]
                    proba_list.append(proba * weights[model_name])
            
            if proba_list:
                ensemble_proba = np.sum(proba_list, axis=0)
                predictions['ensemble'] = {
                    'success_probability': ensemble_proba,
                    'success_prediction': (ensemble_proba >= 0.5).astype(int)
                }
        
        # 個別モデル予測
        for model_name, model in self.models.items():
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X_scaled)[:, 1]
                predictions[model_name] = {
                    'success_probability': proba,
                    'success_prediction': (proba >= 0.5).astype(int)
                }
        
        # 市場別予測
        if market_category and f'{market_category}_models' in self.models:
            market_models = self.models[f'{market_category}_models']
            market_predictions = {}
            
            for model_name, model in market_models.items():
                proba = model.predict_proba(X_scaled)[:, 1]
                market_predictions[model_name] = {
                    'success_probability': proba,
                    'success_prediction': (proba >= 0.5).astype(int)
                }
            
            predictions[f'{market_category}_specific'] = market_predictions
        
        return predictions
    
    def analyze_success_factors(self, features: pd.DataFrame,
                                market_category: str = None) -> Dict:
        """
        成功要因分析
        
        Parameters:
        -----------
        features : pd.DataFrame
            特徴量データ
        market_category : str
            市場カテゴリ
            
        Returns:
        --------
        Dict : 成功要因分析結果
        """
        analysis_results = {}
        
        # 1. 特徴量重要度分析
        if market_category:
            importance_key = f'{market_category}_rf'
            if importance_key in self.feature_importance:
                importance = self.feature_importance[importance_key]
        else:
            importance = self.feature_importance.get('rf', {})
        
        if importance:
            # 上位要因
            top_factors = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
            analysis_results['top_success_factors'] = top_factors
            
            # 要因グループ別分析
            factor_groups = self._group_factors_by_category(dict(top_factors))
            analysis_results['factor_groups'] = factor_groups
        
        # 2. 成功パターン分析
        success_patterns = self._identify_success_patterns(features, market_category)
        analysis_results['success_patterns'] = success_patterns
        
        # 3. ベンチマーク分析
        benchmarks = self._calculate_success_benchmarks(features, market_category)
        analysis_results['benchmarks'] = benchmarks
        
        return analysis_results
    
    def generate_strategic_recommendations(self, company_features: pd.Series,
                                            market_category: str,
                                            prediction_results: Dict) -> Dict:
        """
        戦略的提言生成
        
        Parameters:
        -----------
        company_features : pd.Series
            対象企業の特徴量
        market_category : str
            市場カテゴリ
        prediction_results : Dict
            予測結果
            
        Returns:
        --------
        Dict : 戦略的提言
        """
        recommendations = {
            'overall_assessment': {},
            'priority_actions': [],
            'risk_factors': [],
            'success_levers': [],
            'market_specific_advice': {}
        }
        
        # 成功確率評価
        success_prob = prediction_results.get('ensemble', {}).get('success_probability', [0])[0]
        
        if success_prob >= 0.7:
            risk_level = 'Low'
            assessment = 'High potential for success'
        elif success_prob >= 0.4:
            risk_level = 'Medium'
            assessment = 'Moderate success potential with strategic improvements needed'
        else:
            risk_level = 'High'
            assessment = 'Significant challenges requiring major strategic pivot'
        
        recommendations['overall_assessment'] = {
            'success_probability': success_prob,
            'risk_level': risk_level,
            'assessment': assessment
        }
        
        # 市場別特化アドバイス
        market_advice = self._get_market_specific_advice(market_category, company_features)
        recommendations['market_specific_advice'] = market_advice
        
        # 優先アクション
        priority_actions = self._generate_priority_actions(
            company_features, market_category, success_prob
        )
        recommendations['priority_actions'] = priority_actions
        
        return recommendations
    
    # ===== Private Methods =====
    
    def _get_model(self, model_name: str):
        """モデルインスタンス取得"""
        if model_name == 'logistic':
            return LogisticRegression(random_state=42, max_iter=1000)
        elif model_name == 'rf':
            return RandomForestClassifier(
                n_estimators=100, random_state=42, max_depth=10
            )
        elif model_name == 'xgb':
            return xgb.XGBClassifier(
                n_estimators=100, random_state=42, max_depth=6
            )
        elif model_name == 'lgb':
            return lgb.LGBMClassifier(
                n_estimators=100, random_state=42, max_depth=6, verbose=-1
            )
        else:
            raise ValueError(f"Unknown model type: {model_name}")
    
    def _calculate_cagr(self, revenue_series: pd.Series, years: int) -> float:
        """年平均成長率計算"""
        if len(revenue_series) < 2:
            return 0
        
        start_value = revenue_series.iloc[0]
        end_value = revenue_series.iloc[-1]
        
        if start_value <= 0:
            return 0
        
        actual_years = len(revenue_series) - 1
        if actual_years == 0:
            return 0
            
        cagr = (end_value / start_value) ** (1 / actual_years) - 1
        return cagr
    
    def _years_to_profitability(self, financial_data: pd.DataFrame) -> int:
        """収益化までの年数計算"""
        profitable_years = financial_data[
            financial_data['operating_profit_margin'] > 0
        ]['years_since_establishment']
        
        if len(profitable_years) == 0:
            return 999  # 未達成
        
        return profitable_years.min()
    
    def _group_factors_by_category(self, importance_dict: Dict) -> Dict:
        """要因のカテゴリ別グループ化"""
        categories = {
            'financial': ['revenue', 'profit', 'asset', 'leverage', 'rnd'],
            'market': ['market', 'competitive', 'share', 'growth'],
            'innovation': ['patent', 'technology', 'digital', 'innovation'],
            'resources': ['capital', 'talent', 'partnership', 'network'],
            'timing': ['timing', 'cycle', 'first_mover', 'lifecycle']
        }
        
        grouped = {cat: {} for cat in categories}
        
        for factor, importance in importance_dict.items():
            for category, keywords in categories.items():
                if any(keyword in factor.lower() for keyword in keywords):
                    grouped[category][factor] = importance
                    break
        
        return grouped
    
    def _identify_success_patterns(self, features: pd.DataFrame, 
                                    market_category: str = None) -> Dict:
        """成功パターン特定"""
        # 基本的な成功パターン
        patterns = {
            'high_rnd_innovation': 'R&D集約型イノベーション戦略',
            'fast_market_entry': '迅速な市場参入戦略', 
            'platform_business': 'プラットフォーム型ビジネスモデル',
            'parent_support_leverage': '親会社リソース活用戦略'
        }
        
        return patterns
    
    def _calculate_success_benchmarks(self, features: pd.DataFrame,
                                    market_category: str = None) -> Dict:
        """成功ベンチマーク計算"""
        benchmarks = {}
        
        # 主要指標のベンチマーク
        key_metrics = [
            'initial_rnd_intensity', 'initial_revenue', 'patent_count_early',
            'market_growth_rate', 'competitive_intensity'
        ]
        
        for metric in key_metrics:
            if metric in features.columns:
                benchmarks[metric] = {
                    'median': features[metric].median(),
                    'top_quartile': features[metric].quantile(0.75),
                    'top_decile': features[metric].quantile(0.9)
                }
        
        return benchmarks
    
    def _get_market_specific_advice(self, market_category: str, 
                                    features: pd.Series) -> Dict:
        """市場別特化アドバイス"""
        advice = {}
        
        if market_category == 'high_share':
            advice = {
                'key_strategy': 'Differentiation and niche focus',
                'critical_factors': ['technology_innovation', 'quality_excellence'],
                'success_examples': 'Focus on specialized high-value segments'
            }
        elif market_category == 'declining':
            advice = {
                'key_strategy': 'Digital transformation and efficiency',
                'critical_factors': ['cost_optimization', 'digital_capabilities'],
                'success_examples': 'Transform traditional business with digital technology'
            }
        elif market_category == 'lost':
            advice = {
                'key_strategy': 'New market creation or pivot',
                'critical_factors': ['business_model_innovation', 'market_timing'],
                'success_examples': 'Create entirely new market categories'
            }
        
        return advice
    
    def _generate_priority_actions(self, features: pd.Series,
                                    market_category: str,
                                    success_prob: float) -> List[str]:
        """優先アクション生成"""
        actions = []
        
        # R&D投資レベルチェック
        rnd_intensity = features.get('rnd_investment_ratio', 0)
        if rnd_intensity < 0.05:  # 5%未満
            actions.append('Increase R&D investment to industry competitive level')
        
        # 市場参入タイミング
        if features.get('first_mover_advantage', 0) < 0.5:
            actions.append('Accelerate market entry to capture first-mover advantage')
        
        # 成功確率が低い場合の緊急対策
        if success_prob < 0.4:
            actions.extend([
                'Consider strategic pivot or business model transformation',
                'Seek strategic partnerships or external investment',
                'Focus on core competencies and reduce scope'
            ])
        
        return actions[:5]  # 上位5つのアクション


# 使用例とテスト用のヘルパー関数
def create_sample_emergence_data() -> pd.DataFrame:
    """サンプル新設企業データ作成"""
    np.random.seed(42)
    n_companies = 100
    
    data = {
        'company_id': range(n_companies),
        'years_since_establishment': np.random.randint(1, 20, n_companies),
        'initial_capital_million_yen': np.random.lognormal(3, 1, n_companies),
        'founder_prior_experience_years': np.random.randint(0, 30, n_companies),
        'establishment_type': np.random.choice(['spinoff', 'startup'], n_companies),
        'patents_first_3years': np.random.poisson(2, n_companies),
        'technology_novelty_index': np.random.uniform(0, 1, n_companies),
        'business_model_innovation_score': np.random.uniform(0, 1, n_companies),
        'vc_investment_flag': np.random.choice([0, 1], n_companies, p=[0.7, 0.3]),
        'market_category': np.random.choice(['high_share', 'declining', 'lost'], n_companies)
    }
    
    return pd.DataFrame(data)


if __name__ == "__main__":
    # デモンストレーション
    print("A2AI - Emergence Success Prediction Model")
    print("=" * 50)
    
    # サンプルデータ作成
    emergence_data = create_sample_emergence_data()
    
    # 予測モデル初期化
    predictor = EmergenceSuccessPredictor()
    
    print(f"✓ Sample data created: {len(emergence_data)} companies")
    print("✓ Success prediction model initialized")
    print("\nModel components:")
    for component in predictor.config['feature_groups']:
        print(f"  - {component}")
    
    print(f"\nSuccess metrics: {len(predictor.config['success_metrics'])} indicators")
    print("  Ready for training and prediction...")