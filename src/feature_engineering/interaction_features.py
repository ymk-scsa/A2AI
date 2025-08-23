"""
A2AI - Advanced Financial Analysis AI
interaction_features.py

要因項目間の交互作用特徴量生成モジュール

このモジュールは、各評価項目（9項目）に対する要因項目（各23項目）間の
交互作用特徴量を生成し、企業の生存・成功・失敗パターンをより深く分析する。
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from itertools import combinations, product
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
import warnings
from dataclasses import dataclass
from enum import Enum

class InteractionType(Enum):
    """交互作用の種類を定義"""
    MULTIPLICATIVE = "multiplicative"  # 乗算型: X1 * X2
    ADDITIVE = "additive"  # 加算型: X1 + X2
    RATIO = "ratio"  # 比率型: X1 / X2
    DIFFERENCE = "difference"  # 差分型: X1 - X2
    POLYNOMIAL = "polynomial"  # 多項式型: X1^2, X1*X2, X2^2
    CONDITIONAL = "conditional"  # 条件付き: X1 if condition else X2
    THRESHOLD = "threshold"  # 閾値型: X1 * (X2 > threshold)

class MarketCategory(Enum):
    """市場カテゴリー"""
    HIGH_SHARE = "high_share"  # 世界シェア高市場
    DECLINING = "declining"    # シェア低下市場
    LOST = "lost"             # シェア失失市場

@dataclass
class InteractionConfig:
    """交互作用設定クラス"""
    interaction_types: List[InteractionType]
    max_interactions: int = 100
    min_correlation: float = 0.1
    feature_selection: bool = True
    market_specific: bool = True
    lifecycle_aware: bool = True
    survival_focused: bool = True

class InteractionFeatureGenerator:
    """要因項目間交互作用特徴量生成クラス"""
    
    def __init__(self, config: Optional[InteractionConfig] = None):
        """
        初期化
        
        Args:
            config: 交互作用設定
        """
        self.config = config or InteractionConfig(
            interaction_types=[
                InteractionType.MULTIPLICATIVE,
                InteractionType.RATIO,
                InteractionType.POLYNOMIAL,
                InteractionType.THRESHOLD
            ]
        )
        self.scaler = RobustScaler()
        self.feature_selector = None
        self.interaction_importance_ = {}
        self.generated_features_ = {}
        
        # 評価項目定義
        self.evaluation_metrics = {
            'revenue': '売上高',
            'revenue_growth': '売上高成長率', 
            'operating_margin': '売上高営業利益率',
            'net_margin': '売上高当期純利益率',
            'roe': 'ROE',
            'value_added_ratio': '売上高付加価値率',
            'survival_probability': '企業存続確率',
            'emergence_success': '新規事業成功率',
            'succession_success': '事業継承成功度'
        }
        
        # 要因項目カテゴリー（各評価項目に23項目）
        self.factor_categories = {
            'investment_assets': '投資・資産関連',
            'human_resources': '人的資源関連', 
            'operational_efficiency': '運転資本・効率性関連',
            'business_expansion': '事業展開関連',
            'cost_structure': 'コスト構造関連',
            'financial_structure': '財務構造関連',
            'market_position': '市場ポジション関連',
            'innovation': 'イノベーション関連',
            'lifecycle': 'ライフサイクル関連'
        }
    
    def generate_interactions(
        self,
        data: pd.DataFrame,
        target_metric: str,
        market_category: Optional[MarketCategory] = None
    ) -> pd.DataFrame:
        """
        交互作用特徴量生成
        
        Args:
            data: 入力データ（企業×年×要因項目）
            target_metric: 対象評価項目
            market_category: 市場カテゴリー
            
        Returns:
            交互作用特徴量を含むDataFrame
        """
        if target_metric not in self.evaluation_metrics:
            raise ValueError(f"未知の評価項目: {target_metric}")
        
        # データ前処理
        processed_data = self._preprocess_data(data, market_category)
        
        # 要因項目抽出
        factor_columns = self._get_factor_columns(processed_data, target_metric)
        
        # 交互作用特徴量生成
        interaction_features = pd.DataFrame(index=processed_data.index)
        
        for interaction_type in self.config.interaction_types:
            features = self._generate_interaction_by_type(
                processed_data[factor_columns], 
                interaction_type,
                target_metric
            )
            interaction_features = pd.concat([interaction_features, features], axis=1)
        
        # 特徴選択
        if self.config.feature_selection:
            interaction_features = self._select_features(
                interaction_features, 
                processed_data[target_metric]
            )
        
        # 重要度計算
        self._calculate_importance(
            interaction_features, 
            processed_data[target_metric],
            target_metric
        )
        
        return interaction_features
    
    def _preprocess_data(
        self, 
        data: pd.DataFrame, 
        market_category: Optional[MarketCategory]
    ) -> pd.DataFrame:
        """データ前処理"""
        processed = data.copy()
        
        # 市場カテゴリーフィルタリング
        if market_category and 'market_category' in processed.columns:
            processed = processed[
                processed['market_category'] == market_category.value
            ]
        
        # 欠損値処理
        processed = processed.fillna(processed.median())
        
        # 外れ値処理（IQR法）
        for col in processed.select_dtypes(include=[np.number]).columns:
            Q1 = processed[col].quantile(0.25)
            Q3 = processed[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            processed[col] = processed[col].clip(lower_bound, upper_bound)
        
        return processed
    
    def _get_factor_columns(self, data: pd.DataFrame, target_metric: str) -> List[str]:
        """対象評価項目の要因項目カラム取得"""
        # 評価項目別の要因項目パターン
        factor_patterns = {
            'revenue': [
                'tangible_assets', 'capex', 'rd_expense', 'intangible_assets',
                'investment_securities', 'total_return_ratio', 'employees',
                'avg_salary', 'retirement_cost', 'welfare_cost', 'receivables',
                'inventory', 'total_assets', 'receivables_turnover', 'inventory_turnover',
                'overseas_ratio', 'segment_count', 'sga_expense', 'advertising',
                'non_operating_income', 'order_backlog', 'company_age', 'market_entry_timing'
            ],
            'revenue_growth': [
                'capex_growth', 'rd_growth', 'tangible_growth', 'intangible_growth',
                'assets_growth', 'goodwill_growth', 'employees_growth', 'salary_growth',
                'personnel_cost_growth', 'retirement_growth', 'overseas_ratio_change',
                'segment_revenue_growth', 'sga_growth', 'advertising_growth',
                'non_operating_growth', 'receivables_growth', 'inventory_growth',
                'receivables_turnover_change', 'inventory_turnover_change',
                'asset_turnover_change', 'backlog_growth', 'company_age', 'market_entry_timing'
            ],
            # 他の評価項目も同様に定義...
        }
        
        # 共通要因項目（全評価項目で使用）
        common_factors = ['company_age', 'market_entry_timing', 'parent_dependency']
        
        target_factors = factor_patterns.get(target_metric, [])
        target_factors.extend(common_factors)
        
        # 実際に存在するカラムのみ返す
        available_factors = [col for col in target_factors if col in data.columns]
        
        return available_factors
    
    def _generate_interaction_by_type(
        self,
        factors_data: pd.DataFrame,
        interaction_type: InteractionType,
        target_metric: str
    ) -> pd.DataFrame:
        """交互作用タイプ別特徴量生成"""
        features = pd.DataFrame(index=factors_data.index)
        
        if interaction_type == InteractionType.MULTIPLICATIVE:
            features = self._multiplicative_interactions(factors_data)
            
        elif interaction_type == InteractionType.RATIO:
            features = self._ratio_interactions(factors_data)
            
        elif interaction_type == InteractionType.POLYNOMIAL:
            features = self._polynomial_interactions(factors_data)
            
        elif interaction_type == InteractionType.THRESHOLD:
            features = self._threshold_interactions(factors_data, target_metric)
            
        elif interaction_type == InteractionType.CONDITIONAL:
            features = self._conditional_interactions(factors_data, target_metric)
        
        return features
    
    def _multiplicative_interactions(self, data: pd.DataFrame) -> pd.DataFrame:
        """乗算型交互作用特徴量"""
        features = pd.DataFrame(index=data.index)
        
        # 重要な要因項目ペアの組み合わせ
        important_pairs = [
            ('rd_expense', 'intangible_assets'),  # R&D×無形資産
            ('capex', 'tangible_assets'),         # 設備投資×有形資産
            ('employees', 'avg_salary'),          # 従業員数×平均給与
            ('overseas_ratio', 'segment_count'),  # 海外比率×事業セグメント数
            ('receivables_turnover', 'inventory_turnover'),  # 債権×棚卸回転率
        ]
        
        for col1, col2 in important_pairs:
            if col1 in data.columns and col2 in data.columns:
                feature_name = f"{col1}_x_{col2}"
                features[feature_name] = data[col1] * data[col2]
        
        # カテゴリー内交互作用
        categories = {
            'efficiency': ['receivables_turnover', 'inventory_turnover', 'asset_turnover'],
            'investment': ['capex', 'rd_expense', 'investment_securities'],
            'human': ['employees', 'avg_salary', 'retirement_cost']
        }
        
        for category, cols in categories.items():
            available_cols = [col for col in cols if col in data.columns]
            for col1, col2 in combinations(available_cols, 2):
                feature_name = f"{category}_{col1}_x_{col2}"
                features[feature_name] = data[col1] * data[col2]
        
        return features
    
    def _ratio_interactions(self, data: pd.DataFrame) -> pd.DataFrame:
        """比率型交互作用特徴量"""
        features = pd.DataFrame(index=data.index)
        
        # 重要な比率特徴量
        ratio_pairs = [
            ('rd_expense', 'capex', 'rd_capex_ratio'),
            ('intangible_assets', 'tangible_assets', 'intangible_tangible_ratio'),
            ('avg_salary', 'employees', 'salary_per_employee'),
            ('overseas_ratio', 'total_assets', 'overseas_asset_intensity'),
            ('order_backlog', 'receivables', 'backlog_receivables_ratio'),
        ]
        
        for numerator, denominator, feature_name in ratio_pairs:
            if numerator in data.columns and denominator in data.columns:
                # ゼロ除算回避
                denominator_safe = data[denominator].replace(0, np.finfo(float).eps)
                features[feature_name] = data[numerator] / denominator_safe
        
        # 効率性指標の複合比率
        efficiency_ratios = [
            ('receivables_turnover', 'inventory_turnover', 'turnover_efficiency'),
            ('capex', 'total_assets', 'capex_intensity'),
            ('rd_expense', 'total_assets', 'rd_intensity')
        ]
        
        for col1, col2, feature_name in efficiency_ratios:
            if col1 in data.columns and col2 in data.columns:
                denominator_safe = data[col2].replace(0, np.finfo(float).eps)
                features[feature_name] = data[col1] / denominator_safe
        
        return features
    
    def _polynomial_interactions(self, data: pd.DataFrame) -> pd.DataFrame:
        """多項式交互作用特徴量"""
        features = pd.DataFrame(index=data.index)
        
        # 重要な非線形関係
        polynomial_features = [
            ('rd_expense', 2, 'rd_expense_sq'),
            ('capex', 2, 'capex_sq'),
            ('company_age', 2, 'company_age_sq'),
            ('employees', 0.5, 'employees_sqrt'),
            ('total_assets', 0.5, 'total_assets_sqrt'),
        ]
        
        for col, power, feature_name in polynomial_features:
            if col in data.columns:
                if power == 0.5:
                    features[feature_name] = np.sqrt(np.abs(data[col]))
                else:
                    features[feature_name] = np.power(data[col], power)
        
        # 3次交互作用（選択的）
        triple_interactions = [
            ('rd_expense', 'intangible_assets', 'company_age', 'innovation_maturity'),
            ('capex', 'tangible_assets', 'employees', 'capital_labor_scale'),
        ]
        
        for col1, col2, col3, feature_name in triple_interactions:
            if all(col in data.columns for col in [col1, col2, col3]):
                features[feature_name] = data[col1] * data[col2] * data[col3]
        
        return features
    
    def _threshold_interactions(self, data: pd.DataFrame, target_metric: str) -> pd.DataFrame:
        """閾値型交互作用特徴量"""
        features = pd.DataFrame(index=data.index)
        
        # 市場カテゴリー別閾値定義
        thresholds = {
            'high_performance': {
                'rd_expense': 0.05,    # R&D費率5%以上
                'overseas_ratio': 0.3,  # 海外売上比率30%以上
                'company_age': 20,      # 設立20年以上
            },
            'growth_phase': {
                'capex_growth': 0.1,    # 設備投資成長率10%以上
                'employees_growth': 0.05, # 従業員数成長率5%以上
                'rd_growth': 0.1,       # R&D成長率10%以上
            }
        }
        
        # 高性能企業特徴量
        for metric, threshold in thresholds['high_performance'].items():
            if metric in data.columns:
                high_perf_flag = (data[metric] > threshold).astype(int)
                
                # 他の要因項目との条件付き交互作用
                for col in data.columns:
                    if col != metric and col in data.columns:
                        feature_name = f"{col}_high_{metric}"
                        features[feature_name] = data[col] * high_perf_flag
        
        # 成長段階特徴量
        if target_metric in ['revenue_growth', 'emergence_success']:
            for metric, threshold in thresholds['growth_phase'].items():
                if metric in data.columns:
                    growth_flag = (data[metric] > threshold).astype(int)
                    
                    for col in data.columns:
                        if col != metric and col in data.columns:
                            feature_name = f"{col}_growth_{metric}"
                            features[feature_name] = data[col] * growth_flag
        
        return features
    
    def _conditional_interactions(self, data: pd.DataFrame, target_metric: str) -> pd.DataFrame:
        """条件付き交互作用特徴量"""
        features = pd.DataFrame(index=data.index)
        
        # 企業ライフサイクル条件
        if 'company_age' in data.columns:
            young_flag = (data['company_age'] < 10).astype(int)
            mature_flag = (data['company_age'] >= 20).astype(int)
            
            lifecycle_factors = ['rd_expense', 'capex', 'employees_growth']
            
            for factor in lifecycle_factors:
                if factor in data.columns:
                    features[f"{factor}_young"] = data[factor] * young_flag
                    features[f"{factor}_mature"] = data[factor] * mature_flag
        
        # 市場ポジション条件
        if 'overseas_ratio' in data.columns:
            global_flag = (data['overseas_ratio'] > 0.5).astype(int)
            domestic_flag = (data['overseas_ratio'] <= 0.2).astype(int)
            
            position_factors = ['rd_expense', 'advertising', 'segment_count']
            
            for factor in position_factors:
                if factor in data.columns:
                    features[f"{factor}_global"] = data[factor] * global_flag
                    features[f"{factor}_domestic"] = data[factor] * domestic_flag
        
        return features
    
    def _select_features(
        self, 
        features: pd.DataFrame, 
        target: pd.Series
    ) -> pd.DataFrame:
        """特徴選択"""
        if len(features.columns) <= self.config.max_interactions:
            return features
        
        # 相関による初期フィルタリング
        correlations = []
        for col in features.columns:
            try:
                corr = abs(features[col].corr(target))
                if not np.isnan(corr):
                    correlations.append((col, corr))
            except:
                continue
        
        # 相関上位選択
        correlations.sort(key=lambda x: x[1], reverse=True)
        selected_features = [col for col, _ in correlations[:self.config.max_interactions]]
        
        # 統計的特徴選択
        if len(selected_features) > self.config.max_interactions // 2:
            try:
                selector = SelectKBest(
                    score_func=f_regression,
                    k=min(self.config.max_interactions, len(selected_features))
                )
                selected_data = features[selected_features]
                selector.fit(selected_data, target)
                selected_features = selected_data.columns[selector.get_support()].tolist()
                
                self.feature_selector = selector
            except:
                # フォールバック：相関ベース選択
                selected_features = selected_features[:self.config.max_interactions]
        
        return features[selected_features]
    
    def _calculate_importance(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        target_metric: str
    ):
        """交互作用重要度計算"""
        importance_scores = {}
        
        for col in features.columns:
            try:
                # 相関係数
                correlation = abs(features[col].corr(target))
                
                # 相互情報量
                mutual_info = mutual_info_regression(
                    features[[col]], target, random_state=42
                )[0]
                
                # 統合スコア
                combined_score = 0.7 * correlation + 0.3 * mutual_info
                importance_scores[col] = {
                    'correlation': correlation,
                    'mutual_info': mutual_info,
                    'combined_score': combined_score
                }
            except:
                importance_scores[col] = {
                    'correlation': 0.0,
                    'mutual_info': 0.0,
                    'combined_score': 0.0
                }
        
        self.interaction_importance_[target_metric] = importance_scores
    
    def get_top_interactions(
        self, 
        target_metric: str, 
        top_k: int = 10
    ) -> List[Tuple[str, Dict[str, float]]]:
        """上位交互作用特徴量取得"""
        if target_metric not in self.interaction_importance_:
            return []
        
        importance_data = self.interaction_importance_[target_metric]
        sorted_interactions = sorted(
            importance_data.items(),
            key=lambda x: x[1]['combined_score'],
            reverse=True
        )
        
        return sorted_interactions[:top_k]
    
    def generate_interaction_report(self, target_metric: str) -> Dict[str, Any]:
        """交互作用分析レポート生成"""
        if target_metric not in self.interaction_importance_:
            return {}
        
        top_interactions = self.get_top_interactions(target_metric, 20)
        
        # カテゴリー別分析
        category_analysis = {}
        for interaction, scores in top_interactions:
            # 交互作用タイプ推定
            if '_x_' in interaction:
                interaction_type = 'multiplicative'
            elif '_ratio' in interaction:
                interaction_type = 'ratio'
            elif '_sq' in interaction or '_sqrt' in interaction:
                interaction_type = 'polynomial'
            elif '_high_' in interaction or '_growth_' in interaction:
                interaction_type = 'threshold'
            else:
                interaction_type = 'other'
            
            if interaction_type not in category_analysis:
                category_analysis[interaction_type] = []
            
            category_analysis[interaction_type].append({
                'feature': interaction,
                'importance': scores['combined_score']
            })
        
        return {
            'target_metric': target_metric,
            'top_interactions': top_interactions,
            'category_analysis': category_analysis,
            'total_features': len(self.interaction_importance_[target_metric]),
            'avg_importance': np.mean([
                scores['combined_score'] 
                for scores in self.interaction_importance_[target_metric].values()
            ])
        }

def create_interaction_config(
    focus_area: str = "comprehensive",
    max_features: int = 100
) -> InteractionConfig:
    """
    用途別交互作用設定作成
    
    Args:
        focus_area: 焦点領域 ("survival", "growth", "efficiency", "comprehensive")
        max_features: 最大特徴量数
    
    Returns:
        InteractionConfig
    """
    if focus_area == "survival":
        return InteractionConfig(
            interaction_types=[
                InteractionType.THRESHOLD,
                InteractionType.CONDITIONAL,
                InteractionType.RATIO
            ],
            max_interactions=max_features,
            survival_focused=True
        )
    
    elif focus_area == "growth":
        return InteractionConfig(
            interaction_types=[
                InteractionType.MULTIPLICATIVE,
                InteractionType.POLYNOMIAL,
                InteractionType.THRESHOLD
            ],
            max_interactions=max_features,
            lifecycle_aware=True
        )
    
    elif focus_area == "efficiency":
        return InteractionConfig(
            interaction_types=[
                InteractionType.RATIO,
                InteractionType.MULTIPLICATIVE
            ],
            max_interactions=max_features,
            market_specific=True
        )
    
    else:  # comprehensive
        return InteractionConfig(
            interaction_types=[
                InteractionType.MULTIPLICATIVE,
                InteractionType.RATIO,
                InteractionType.POLYNOMIAL,
                InteractionType.THRESHOLD,
                InteractionType.CONDITIONAL
            ],
            max_interactions=max_features,
            feature_selection=True,
            market_specific=True,
            lifecycle_aware=True,
            survival_focused=True
        )

# 使用例とテスト用コード
if __name__ == "__main__":
    # テスト用データ生成
    np.random.seed(42)
    n_companies = 150
    n_years = 40
    n_samples = n_companies * n_years
    
    test_data = pd.DataFrame({
        'company_id': np.repeat(range(n_companies), n_years),
        'year': np.tile(range(1984, 2024), n_companies),
        'market_category': np.random.choice(['high_share', 'declining', 'lost'], n_samples),
        
        # 評価項目
        'revenue': np.random.lognormal(10, 1, n_samples),
        'revenue_growth': np.random.normal(0.05, 0.2, n_samples),
        'operating_margin': np.random.beta(2, 8, n_samples),
        'roe': np.random.normal(0.1, 0.15, n_samples),
        
        # 要因項目サンプル
        'rd_expense': np.random.beta(1, 20, n_samples),
        'capex': np.random.lognormal(8, 1, n_samples),
        'intangible_assets': np.random.lognormal(7, 1.5, n_samples),
        'tangible_assets': np.random.lognormal(9, 1, n_samples),
        'employees': np.random.lognormal(6, 1, n_samples),
        'avg_salary': np.random.lognormal(6, 0.3, n_samples),
        'overseas_ratio': np.random.beta(2, 3, n_samples),
        'company_age': np.random.uniform(1, 50, n_samples),
        'receivables_turnover': np.random.gamma(3, 2, n_samples),
        'inventory_turnover': np.random.gamma(2, 3, n_samples),
    })
    
    # 交互作用特徴量生成器初期化
    config = create_interaction_config("comprehensive", max_features=50)
    generator = InteractionFeatureGenerator(config)
    
    # 交互作用特徴量生成
    interaction_features = generator.generate_interactions(
        test_data, 
        'revenue',
        MarketCategory.HIGH_SHARE
    )
    
    print(f"生成された交互作用特徴量数: {len(interaction_features.columns)}")
    print(f"サンプル特徴量: {list(interaction_features.columns[:5])}")
    
    # 重要度レポート
    report = generator.generate_interaction_report('revenue')
    print(f"\n上位5つの交互作用特徴量:")
    for feature, scores in report['top_interactions'][:5]:
        print(f"  {feature}: {scores['combined_score']:.3f}")