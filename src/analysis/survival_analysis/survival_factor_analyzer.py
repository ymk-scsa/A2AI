"""
A2AI - Advanced Financial Analysis AI
Survival Factor Analyzer

企業生存に影響を与える要因項目を特定・分析するモジュール
150社の企業ライフサイクルデータから生存要因を定量的に解析
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import logging
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# 統計・機械学習ライブラリ
from scipy import stats
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score

# 生存分析ライブラリ（lifelines）
try:
    from lifelines import CoxPHFitter, KaplanMeierFitter
    from lifelines.statistics import logrank_test, multivariate_logrank_test
    from lifelines.utils import concordance_index
    LIFELINES_AVAILABLE = True
except ImportError:
    LIFELINES_AVAILABLE = False
    logging.warning("lifelines not available. Some survival analysis features will be limited.")

# 内部モジュール
from ...utils.statistical_utils import StatisticalUtils
from ...utils.survival_utils import SurvivalUtils
from ...config.settings import ANALYSIS_CONFIG


class MarketCategory(Enum):
    """市場カテゴリ列挙型"""
    HIGH_SHARE = "high_share"      # 世界シェア高市場
    DECLINING = "declining"        # シェア低下市場
    LOST_SHARE = "lost_share"      # シェア失失市場


class SurvivalStage(Enum):
    """企業生存ステージ"""
    STARTUP = "startup"            # 新設期（設立～5年）
    GROWTH = "growth"              # 成長期（6～15年）
    MATURITY = "maturity"          # 成熟期（16～30年）
    DECLINE = "decline"            # 衰退期（30年～）
    EXTINCT = "extinct"            # 消滅


@dataclass
class SurvivalFactorResult:
    """生存要因分析結果データクラス"""
    factor_name: str
    hazard_ratio: float
    p_value: float
    confidence_interval: Tuple[float, float]
    importance_score: float
    market_specific_effect: Dict[str, float]
    stage_specific_effect: Dict[str, float]


class SurvivalFactorAnalyzer:
    """
    企業生存要因分析クラス
    
    企業の生存期間と要因項目の関係を多角的に分析し、
    市場カテゴリー別・企業ステージ別の生存要因を特定する
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初期化
        
        Args:
            config: 分析設定辞書
        """
        self.config = config or ANALYSIS_CONFIG
        self.logger = logging.getLogger(__name__)
        
        # 分析結果格納
        self.survival_data: Optional[pd.DataFrame] = None
        self.factor_results: Dict[str, SurvivalFactorResult] = {}
        self.market_comparison: Dict[str, Dict] = {}
        self.stage_analysis: Dict[str, Dict] = {}
        
        # 統計ユーティリティ
        self.stat_utils = StatisticalUtils()
        if LIFELINES_AVAILABLE:
            self.survival_utils = SurvivalUtils()
        
        # 要因項目定義（各評価項目23項目ずつ）
        self.factor_categories = {
            'investment_asset': [
                'tangible_fixed_assets', 'capital_investment', 'rd_expenses', 
                'intangible_assets', 'investment_securities', 'total_return_ratio'
            ],
            'human_resources': [
                'employee_count', 'average_salary', 'retirement_costs', 
                'welfare_expenses', 'employee_growth_rate', 'salary_growth_rate'
            ],
            'operational_efficiency': [
                'accounts_receivable', 'inventory', 'total_assets', 
                'receivables_turnover', 'inventory_turnover', 'asset_turnover'
            ],
            'business_expansion': [
                'overseas_sales_ratio', 'segment_count', 'sga_expenses', 
                'advertising_expenses', 'non_operating_income', 'order_backlog'
            ],
            'lifecycle_factors': [
                'company_age', 'market_entry_timing', 'parent_dependency'
            ]
        }
    
    def load_survival_data(self, data_path: str) -> None:
        """
        生存分析用データの読み込み
        
        Args:
            data_path: データファイルパス
        """
        try:
            self.survival_data = pd.read_csv(data_path)
            self.logger.info(f"Loaded survival data: {self.survival_data.shape}")
            
            # 必要カラムの存在確認
            required_cols = ['company_id', 'survival_time', 'event_occurred', 
                            'market_category', 'survival_stage']
            missing_cols = [col for col in required_cols if col not in self.survival_data.columns]
            
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
                
            # データ型の調整
            self.survival_data['survival_time'] = pd.to_numeric(self.survival_data['survival_time'])
            self.survival_data['event_occurred'] = self.survival_data['event_occurred'].astype(bool)
            
        except Exception as e:
            self.logger.error(f"Error loading survival data: {str(e)}")
            raise
    
    def prepare_factor_data(self) -> pd.DataFrame:
        """
        要因項目データの準備と前処理
        
        Returns:
            前処理済み要因項目データフレーム
        """
        if self.survival_data is None:
            raise ValueError("Survival data not loaded. Call load_survival_data() first.")
        
        # 全要因項目の取得
        all_factors = []
        for category_factors in self.factor_categories.values():
            all_factors.extend(category_factors)
        
        # 要因項目データの抽出
        factor_cols = [col for col in all_factors if col in self.survival_data.columns]
        missing_factors = [col for col in all_factors if col not in self.survival_data.columns]
        
        if missing_factors:
            self.logger.warning(f"Missing factor columns: {missing_factors}")
        
        factor_data = self.survival_data[['company_id', 'survival_time', 'event_occurred', 
                                        'market_category', 'survival_stage'] + factor_cols].copy()
        
        # 欠損値処理
        factor_data = self._handle_missing_values(factor_data)
        
        # 外れ値処理
        factor_data = self._handle_outliers(factor_data, factor_cols)
        
        # 正規化
        scaler = StandardScaler()
        factor_data[factor_cols] = scaler.fit_transform(factor_data[factor_cols])
        
        self.logger.info(f"Prepared factor data: {factor_data.shape}")
        return factor_data
    
    def analyze_cox_regression(self, factor_data: pd.DataFrame) -> Dict[str, SurvivalFactorResult]:
        """
        Cox回帰による生存要因分析
        
        Args:
            factor_data: 前処理済み要因項目データ
            
        Returns:
            要因項目別のハザード比・有意性結果
        """
        if not LIFELINES_AVAILABLE:
            self.logger.warning("lifelines not available. Skipping Cox regression analysis.")
            return {}
        
        results = {}
        
        # Cox回帰モデルの構築
        cox_data = factor_data.copy()
        cox_data = cox_data.rename(columns={'survival_time': 'T', 'event_occurred': 'E'})
        
        try:
            # 全要因項目での Cox回帰
            cph = CoxPHFitter()
            factor_cols = [col for col in cox_data.columns 
                            if col not in ['company_id', 'T', 'E', 'market_category', 'survival_stage']]
            
            cph.fit(cox_data[['T', 'E'] + factor_cols], duration_col='T', event_col='E')
            
            # 各要因項目の結果を抽出
            for factor in factor_cols:
                if factor in cph.summary.index:
                    summary_row = cph.summary.loc[factor]
                    
                    # 市場カテゴリー別の効果を分析
                    market_effects = self._analyze_market_specific_effects(
                        factor_data, factor, 'cox'
                    )
                    
                    # ステージ別の効果を分析
                    stage_effects = self._analyze_stage_specific_effects(
                        factor_data, factor, 'cox'
                    )
                    
                    results[factor] = SurvivalFactorResult(
                        factor_name=factor,
                        hazard_ratio=np.exp(summary_row['coef']),
                        p_value=summary_row['p'],
                        confidence_interval=(
                            np.exp(summary_row['coef lower 95%']),
                            np.exp(summary_row['coef upper 95%'])
                        ),
                        importance_score=abs(summary_row['coef']),
                        market_specific_effect=market_effects,
                        stage_specific_effect=stage_effects
                    )
            
            self.logger.info(f"Completed Cox regression analysis for {len(results)} factors")
            
        except Exception as e:
            self.logger.error(f"Error in Cox regression analysis: {str(e)}")
        
        return results
    
    def analyze_machine_learning_importance(self, factor_data: pd.DataFrame) -> Dict[str, float]:
        """
        機械学習による要因重要度分析
        
        Args:
            factor_data: 前処理済み要因項目データ
            
        Returns:
            要因項目別重要度スコア
        """
        factor_cols = [col for col in factor_data.columns 
                        if col not in ['company_id', 'survival_time', 'event_occurred', 
                                    'market_category', 'survival_stage']]
        
        X = factor_data[factor_cols]
        y = factor_data['survival_time']
        
        importance_results = {}
        
        # Random Forest による重要度分析
        try:
            rf_model = RandomForestRegressor(
                n_estimators=100, 
                random_state=42, 
                n_jobs=-1
            )
            rf_model.fit(X, y)
            
            rf_importance = dict(zip(factor_cols, rf_model.feature_importances_))
            importance_results['random_forest'] = rf_importance
            
        except Exception as e:
            self.logger.error(f"Error in Random Forest importance: {str(e)}")
        
        # Gradient Boosting による重要度分析
        try:
            gb_model = GradientBoostingRegressor(
                n_estimators=100, 
                random_state=42
            )
            gb_model.fit(X, y)
            
            gb_importance = dict(zip(factor_cols, gb_model.feature_importances_))
            importance_results['gradient_boosting'] = gb_importance
            
        except Exception as e:
            self.logger.error(f"Error in Gradient Boosting importance: {str(e)}")
        
        # 平均重要度の計算
        if importance_results:
            avg_importance = {}
            for factor in factor_cols:
                scores = [results[factor] for results in importance_results.values() 
                            if factor in results]
                if scores:
                    avg_importance[factor] = np.mean(scores)
        else:
            avg_importance = {}
        
        self.logger.info(f"Completed ML importance analysis for {len(avg_importance)} factors")
        return avg_importance
    
    def analyze_market_comparison(self, factor_data: pd.DataFrame) -> Dict[str, Dict]:
        """
        市場カテゴリー間での生存要因比較分析
        
        Args:
            factor_data: 前処理済み要因項目データ
            
        Returns:
            市場カテゴリー別生存要因比較結果
        """
        market_results = {}
        
        for market_cat in MarketCategory:
            market_data = factor_data[
                factor_data['market_category'] == market_cat.value
            ].copy()
            
            if len(market_data) < 10:  # 最小サンプルサイズチェック
                self.logger.warning(f"Insufficient data for market {market_cat.value}")
                continue
            
            # 市場別の生存統計
            survival_stats = self._calculate_survival_statistics(market_data)
            
            # 市場別の要因重要度
            factor_importance = self._calculate_market_factor_importance(market_data)
            
            market_results[market_cat.value] = {
                'survival_statistics': survival_stats,
                'factor_importance': factor_importance,
                'sample_size': len(market_data)
            }
        
        # 市場間統計的比較
        market_comparison = self._perform_market_statistical_comparison(
            factor_data, market_results
        )
        
        self.market_comparison = market_results
        self.logger.info(f"Completed market comparison analysis for {len(market_results)} markets")
        
        return market_results
    
    def analyze_survival_stages(self, factor_data: pd.DataFrame) -> Dict[str, Dict]:
        """
        企業ライフサイクルステージ別生存要因分析
        
        Args:
            factor_data: 前処理済み要因項目データ
            
        Returns:
            ステージ別生存要因分析結果
        """
        stage_results = {}
        
        for stage in SurvivalStage:
            if stage == SurvivalStage.EXTINCT:  # 消滅企業は別途処理
                continue
                
            stage_data = factor_data[
                factor_data['survival_stage'] == stage.value
            ].copy()
            
            if len(stage_data) < 5:  # 最小サンプルサイズチェック
                continue
            
            # ステージ別生存統計
            survival_stats = self._calculate_survival_statistics(stage_data)
            
            # ステージ別重要要因
            important_factors = self._identify_stage_critical_factors(stage_data)
            
            # 次ステージへの遷移要因
            transition_factors = self._analyze_stage_transition_factors(
                factor_data, stage.value
            )
            
            stage_results[stage.value] = {
                'survival_statistics': survival_stats,
                'critical_factors': important_factors,
                'transition_factors': transition_factors,
                'sample_size': len(stage_data)
            }
        
        self.stage_analysis = stage_results
        self.logger.info(f"Completed stage analysis for {len(stage_results)} stages")
        
        return stage_results
    
    def generate_factor_ranking(self, 
                                factor_data: pd.DataFrame,
                                ranking_method: str = 'composite') -> List[Tuple[str, float]]:
        """
        総合的な要因重要度ランキング生成
        
        Args:
            factor_data: 前処理済み要因項目データ
            ranking_method: ランキング手法 ('composite', 'cox', 'ml')
            
        Returns:
            要因項目の重要度ランキング（降順）
        """
        if ranking_method == 'composite':
            # Cox回帰結果の取得
            cox_results = self.analyze_cox_regression(factor_data)
            
            # ML重要度の取得
            ml_importance = self.analyze_machine_learning_importance(factor_data)
            
            # 統計的有意性の取得
            statistical_significance = self._calculate_statistical_significance(factor_data)
            
            # 複合スコアの計算
            composite_scores = {}
            all_factors = set(cox_results.keys()) | set(ml_importance.keys())
            
            for factor in all_factors:
                score = 0.0
                weight_sum = 0.0
                
                # Cox回帰重要度 (40%)
                if factor in cox_results:
                    cox_score = cox_results[factor].importance_score
                    score += cox_score * 0.4
                    weight_sum += 0.4
                
                # ML重要度 (40%)
                if factor in ml_importance:
                    ml_score = ml_importance[factor]
                    score += ml_score * 0.4
                    weight_sum += 0.4
                
                # 統計的有意性 (20%)
                if factor in statistical_significance:
                    stat_score = statistical_significance[factor]
                    score += stat_score * 0.2
                    weight_sum += 0.2
                
                if weight_sum > 0:
                    composite_scores[factor] = score / weight_sum
            
            # ランキング生成
            ranking = sorted(composite_scores.items(), key=lambda x: x[1], reverse=True)
            
        elif ranking_method == 'cox':
            cox_results = self.analyze_cox_regression(factor_data)
            ranking = sorted(
                [(name, result.importance_score) for name, result in cox_results.items()],
                key=lambda x: x[1], reverse=True
            )
            
        elif ranking_method == 'ml':
            ml_importance = self.analyze_machine_learning_importance(factor_data)
            ranking = sorted(ml_importance.items(), key=lambda x: x[1], reverse=True)
            
        else:
            raise ValueError(f"Unknown ranking method: {ranking_method}")
        
        self.logger.info(f"Generated factor ranking with {len(ranking)} factors")
        return ranking
    
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """欠損値処理"""
        # 数値列の欠損値を中央値で補完
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in ['company_id', 'survival_time', 'event_occurred']:
                data[col] = data[col].fillna(data[col].median())
        
        return data
    
    def _handle_outliers(self, data: pd.DataFrame, factor_cols: List[str]) -> pd.DataFrame:
        """外れ値処理（IQR法）"""
        for col in factor_cols:
            if col in data.columns:
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                data[col] = data[col].clip(lower=lower_bound, upper=upper_bound)
        
        return data
    
    def _analyze_market_specific_effects(self, 
                                        data: pd.DataFrame, 
                                        factor: str, 
                                        method: str) -> Dict[str, float]:
        """市場カテゴリー別の要因効果分析"""
        effects = {}
        
        for market_cat in MarketCategory:
            market_data = data[data['market_category'] == market_cat.value]
            if len(market_data) > 5:
                correlation = market_data[factor].corr(market_data['survival_time'])
                effects[market_cat.value] = correlation if not np.isnan(correlation) else 0.0
        
        return effects
    
    def _analyze_stage_specific_effects(self, 
                                        data: pd.DataFrame, 
                                        factor: str, 
                                        method: str) -> Dict[str, float]:
        """ライフサイクルステージ別の要因効果分析"""
        effects = {}
        
        for stage in SurvivalStage:
            if stage == SurvivalStage.EXTINCT:
                continue
                
            stage_data = data[data['survival_stage'] == stage.value]
            if len(stage_data) > 5:
                correlation = stage_data[factor].corr(stage_data['survival_time'])
                effects[stage.value] = correlation if not np.isnan(correlation) else 0.0
        
        return effects
    
    def _calculate_survival_statistics(self, data: pd.DataFrame) -> Dict[str, float]:
        """生存統計計算"""
        stats_dict = {
            'mean_survival_time': data['survival_time'].mean(),
            'median_survival_time': data['survival_time'].median(),
            'survival_rate': (~data['event_occurred']).mean(),
            'extinction_rate': data['event_occurred'].mean()
        }
        
        return stats_dict
    
    def _calculate_market_factor_importance(self, data: pd.DataFrame) -> Dict[str, float]:
        """市場別要因重要度計算"""
        factor_cols = [col for col in data.columns 
                        if col not in ['company_id', 'survival_time', 'event_occurred', 
                                    'market_category', 'survival_stage']]
        
        importance = {}
        for factor in factor_cols:
            correlation = abs(data[factor].corr(data['survival_time']))
            importance[factor] = correlation if not np.isnan(correlation) else 0.0
        
        return importance
    
    def _perform_market_statistical_comparison(self, 
                                                data: pd.DataFrame, 
                                                market_results: Dict) -> Dict:
        """市場間統計的比較"""
        # ログランク検定などの実装
        comparison_results = {}
        
        # 簡単な比較統計を計算
        market_survival_times = {}
        for market_cat in MarketCategory:
            market_data = data[data['market_category'] == market_cat.value]
            if len(market_data) > 0:
                market_survival_times[market_cat.value] = market_data['survival_time'].values
        
        # 市場間での生存時間分布の比較
        for i, (market1, times1) in enumerate(market_survival_times.items()):
            for market2, times2 in list(market_survival_times.items())[i+1:]:
                try:
                    statistic, p_value = stats.mannwhitneyu(times1, times2, alternative='two-sided')
                    comparison_results[f"{market1}_vs_{market2}"] = {
                        'statistic': statistic,
                        'p_value': p_value
                    }
                except:
                    pass
        
        return comparison_results
    
    def _identify_stage_critical_factors(self, data: pd.DataFrame) -> List[Tuple[str, float]]:
        """ステージ別重要要因特定"""
        factor_cols = [col for col in data.columns 
                        if col not in ['company_id', 'survival_time', 'event_occurred', 
                                    'market_category', 'survival_stage']]
        
        critical_factors = []
        for factor in factor_cols:
            correlation = abs(data[factor].corr(data['survival_time']))
            if not np.isnan(correlation):
                critical_factors.append((factor, correlation))
        
        return sorted(critical_factors, key=lambda x: x[1], reverse=True)[:10]
    
    def _analyze_stage_transition_factors(self, 
                                        data: pd.DataFrame, 
                                        current_stage: str) -> Dict[str, float]:
        """ステージ遷移要因分析"""
        # 簡略化された遷移要因分析
        transition_factors = {}
        
        current_stage_data = data[data['survival_stage'] == current_stage]
        if len(current_stage_data) > 5:
            factor_cols = [col for col in data.columns 
                            if col not in ['company_id', 'survival_time', 'event_occurred', 
                                        'market_category', 'survival_stage']]
            
            for factor in factor_cols:
                correlation = current_stage_data[factor].corr(current_stage_data['survival_time'])
                if not np.isnan(correlation):
                    transition_factors[factor] = correlation
        
        return transition_factors
    
    def _calculate_statistical_significance(self, data: pd.DataFrame) -> Dict[str, float]:
        """統計的有意性計算"""
        factor_cols = [col for col in data.columns 
                        if col not in ['company_id', 'survival_time', 'event_occurred', 
                                    'market_category', 'survival_stage']]
        
        significance = {}
        for factor in factor_cols:
            try:
                correlation, p_value = stats.pearsonr(data[factor], data['survival_time'])
                # p値を重要度スコアに変換（p値が小さいほど高スコア）
                significance[factor] = 1 - p_value if not np.isnan(p_value) else 0.0
            except:
                significance[factor] = 0.0
        
        return significance
    
    def get_analysis_summary(self) -> Dict:
        """分析結果サマリーの取得"""
        return {
            'total_factors_analyzed': len(self.factor_results),
            'market_categories_analyzed': len(self.market_comparison),
            'lifecycle_stages_analyzed': len(self.stage_analysis),
            'top_survival_factors': list(self.factor_results.keys())[:10],
            'analysis_config': self.config
        }