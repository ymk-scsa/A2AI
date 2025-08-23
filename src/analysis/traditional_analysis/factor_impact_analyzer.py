"""
A2AI - Advanced Financial Analysis AI
要因項目影響分析モジュール (factor_impact_analyzer.py)

このモジュールは120の要因項目が6つの評価項目に与える影響を定量化し、
世界シェア別市場（高シェア/低下/失失）間での差異を分析する。

主要機能:
1. 要因項目の重要度スコア算出
2. 市場カテゴリー別要因項目影響差異分析  
3. 時系列での要因項目影響変化分析
4. 統計的有意性検定
5. 要因項目間の交互作用分析
"""

import pandas as pd
import numpy as np
import warnings
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from datetime import datetime

warnings.filterwarnings('ignore')

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FactorImpactResult:
    """要因項目影響分析結果を格納するデータクラス"""
    factor_name: str
    evaluation_metric: str
    market_category: str
    importance_score: float
    correlation: float
    p_value: float
    coefficient: float
    confidence_interval: Tuple[float, float]
    model_type: str
    r2_score: float
    temporal_trend: Optional[List[float]] = None


class FactorImpactAnalyzer:
    """要因項目影響分析クラス"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初期化
        
        Args:
            config: 設定辞書（モデルパラメータ、閾値等）
        """
        self.config = config or self._get_default_config()
        self.results = []
        self.models = {}
        self.scaler = StandardScaler()
        
        # 評価項目定義
        self.evaluation_metrics = [
            'sales', 'sales_growth_rate', 'operating_margin', 
            'net_margin', 'roe', 'value_added_ratio'
        ]
        
        # 市場カテゴリー定義
        self.market_categories = [
            'high_share_markets', 'declining_markets', 'lost_markets'
        ]
        
        # 要因項目カテゴリー定義（各評価項目20項目）
        self.factor_categories = {
            'sales': self._get_sales_factors(),
            'sales_growth_rate': self._get_growth_factors(), 
            'operating_margin': self._get_operating_margin_factors(),
            'net_margin': self._get_net_margin_factors(),
            'roe': self._get_roe_factors(),
            'value_added_ratio': self._get_value_added_factors()
        }
        
    def _get_default_config(self) -> Dict:
        """デフォルト設定を返す"""
        return {
            'significance_level': 0.05,
            'min_importance_threshold': 0.01,
            'cross_validation_folds': 5,
            'random_state': 42,
            'n_estimators': 100,
            'max_features': 'sqrt',
            'test_size': 0.2,
            'models_to_use': ['linear', 'ridge', 'random_forest', 'gradient_boosting']
        }
    
    def _get_sales_factors(self) -> List[str]:
        """売上高の要因項目リスト"""
        return [
            'tangible_fixed_assets', 'capital_investment', 'rd_expenses', 'intangible_assets',
            'investment_securities', 'total_return_ratio', 'employee_count', 'average_salary',
            'retirement_costs', 'welfare_costs', 'accounts_receivable', 'inventory',
            'total_assets', 'receivables_turnover', 'inventory_turnover', 'overseas_sales_ratio',
            'business_segments', 'sga_expenses', 'advertising_costs', 'non_operating_income'
        ]
    
    def _get_growth_factors(self) -> List[str]:
        """売上高成長率の要因項目リスト"""  
        return [
            'capex_growth', 'rd_growth', 'tangible_assets_growth', 'intangible_growth',
            'total_assets_growth', 'goodwill_growth', 'employee_growth', 'salary_growth',
            'personnel_cost_growth', 'retirement_cost_growth', 'overseas_ratio_change',
            'segment_sales_growth', 'sga_growth', 'advertising_growth', 'non_op_income_growth',
            'receivables_growth', 'inventory_growth', 'receivables_turnover_change',
            'inventory_turnover_change', 'asset_turnover_change'
        ]
    
    def _get_operating_margin_factors(self) -> List[str]:
        """売上高営業利益率の要因項目リスト"""
        return [
            'material_cost_ratio', 'labor_cost_ratio', 'expense_ratio', 'outsourcing_ratio',
            'depreciation_ratio_manufacturing', 'sga_ratio', 'personnel_cost_ratio_sga',
            'advertising_ratio', 'rd_ratio', 'depreciation_ratio_sga', 'value_added_ratio',
            'labor_productivity', 'asset_efficiency', 'total_asset_turnover',
            'inventory_turnover', 'sales_scale', 'fixed_cost_ratio', 'variable_cost_ratio',
            'overseas_sales_ratio', 'business_concentration'
        ]
    
    def _get_net_margin_factors(self) -> List[str]:
        """売上高当期純利益率の要因項目リスト"""
        return [
            'operating_margin', 'sga_ratio', 'cogs_ratio', 'rd_ratio', 'depreciation_ratio',
            'interest_income_ratio', 'interest_expense_ratio', 'fx_gain_loss_ratio',
            'equity_income_ratio', 'non_operating_income_ratio', 'extraordinary_gain_ratio',
            'extraordinary_loss_ratio', 'effective_tax_rate', 'tax_adjustment_ratio',
            'pretax_margin', 'debt_ratio', 'equity_ratio', 'investment_gain_loss',
            'asset_sale_gain_loss', 'impairment_loss_ratio'
        ]
    
    def _get_roe_factors(self) -> List[str]:
        """ROEの要因項目リスト"""
        return [
            'net_margin', 'total_asset_turnover', 'operating_margin', 'cogs_ratio',
            'sga_ratio', 'equity_ratio', 'asset_equity_ratio', 'debt_equity_ratio',
            'current_ratio', 'fixed_ratio', 'receivables_turnover', 'inventory_turnover',
            'tangible_asset_turnover', 'cash_asset_ratio', 'investment_asset_ratio',
            'dividend_payout', 'retention_ratio', 'non_operating_income_ratio',
            'extraordinary_income_ratio', 'effective_tax_rate'
        ]
    
    def _get_value_added_factors(self) -> List[str]:
        """売上高付加価値率の要因項目リスト"""
        return [
            'rd_ratio', 'intangible_asset_ratio', 'patent_costs', 'software_ratio',
            'license_income', 'salary_industry_ratio', 'personnel_cost_ratio',
            'employee_sales_ratio', 'retirement_cost_ratio', 'welfare_ratio',
            'cogs_ratio_inverse', 'material_ratio_inverse', 'outsourcing_ratio_inverse',
            'labor_productivity', 'asset_productivity', 'overseas_sales_ratio',
            'high_value_segment_ratio', 'service_income_ratio', 'operating_margin',
            'brand_asset_ratio'
        ]
    
    def analyze_factor_impact(
        self, 
        data: pd.DataFrame,
        target_metric: str,
        market_category: Optional[str] = None,
        time_period: Optional[Tuple[int, int]] = None
    ) -> List[FactorImpactResult]:
        """
        要因項目影響分析を実行
        
        Args:
            data: 分析対象データ
            target_metric: 対象評価項目
            market_category: 市場カテゴリー（None の場合全市場）
            time_period: 分析期間（開始年、終了年）
            
        Returns:
            分析結果のリスト
        """
        logger.info(f"要因項目影響分析開始: {target_metric}, {market_category}")
        
        # データ前処理
        processed_data = self._preprocess_data(data, market_category, time_period)
        
        if processed_data.empty:
            logger.warning("分析対象データが空です")
            return []
        
        # 対象要因項目取得
        factor_columns = self.factor_categories.get(target_metric, [])
        if not factor_columns:
            logger.error(f"評価項目 {target_metric} の要因項目が定義されていません")
            return []
        
        # 利用可能な要因項目のフィルタリング
        available_factors = [col for col in factor_columns if col in processed_data.columns]
        if not available_factors:
            logger.warning(f"利用可能な要因項目がありません: {target_metric}")
            return []
        
        logger.info(f"利用可能な要因項目数: {len(available_factors)}")
        
        results = []
        
        # 各要因項目について影響分析実行
        for factor in available_factors:
            try:
                result = self._analyze_single_factor(
                    processed_data, factor, target_metric, market_category
                )
                if result:
                    results.append(result)
            except Exception as e:
                logger.error(f"要因項目 {factor} の分析中にエラー: {e}")
                continue
        
        # 結果の重要度順ソート
        results.sort(key=lambda x: abs(x.importance_score), reverse=True)
        
        logger.info(f"影響分析完了: {len(results)} 項目")
        return results
    
    def _preprocess_data(
        self, 
        data: pd.DataFrame, 
        market_category: Optional[str] = None,
        time_period: Optional[Tuple[int, int]] = None
    ) -> pd.DataFrame:
        """データ前処理"""
        
        processed = data.copy()
        
        # 市場カテゴリーフィルタ
        if market_category and 'market_category' in processed.columns:
            processed = processed[processed['market_category'] == market_category]
        
        # 期間フィルタ  
        if time_period and 'year' in processed.columns:
            start_year, end_year = time_period
            processed = processed[
                (processed['year'] >= start_year) & 
                (processed['year'] <= end_year)
            ]
        
        # 欠損値・無限値処理
        processed = processed.replace([np.inf, -np.inf], np.nan)
        processed = processed.dropna()
        
        return processed
    
    def _analyze_single_factor(
        self,
        data: pd.DataFrame,
        factor_name: str, 
        target_metric: str,
        market_category: Optional[str] = None
    ) -> Optional[FactorImpactResult]:
        """単一要因項目の影響分析"""
        
        if factor_name not in data.columns or target_metric not in data.columns:
            return None
            
        # データ抽出
        X = data[[factor_name]].values
        y = data[target_metric].values
        
        if len(X) < 10:  # 最低サンプル数チェック
            return None
        
        # 相関分析
        correlation, p_value = pearsonr(X.flatten(), y)
        
        # 統計的有意性チェック
        if p_value > self.config['significance_level']:
            correlation = 0.0  # 有意でない場合は0に設定
        
        # 複数モデルによる重要度算出
        importance_scores = []
        coefficients = []
        r2_scores = []
        
        # 線形回帰
        if 'linear' in self.config['models_to_use']:
            try:
                model = LinearRegression()
                model.fit(X, y)
                y_pred = model.predict(X)
                r2 = r2_score(y, y_pred)
                
                importance_scores.append(abs(correlation))
                coefficients.append(model.coef_[0])
                r2_scores.append(r2)
            except Exception as e:
                logger.debug(f"線形回帰エラー {factor_name}: {e}")
        
        # Ridge回帰
        if 'ridge' in self.config['models_to_use']:
            try:
                model = Ridge(random_state=self.config['random_state'])
                model.fit(X, y)
                y_pred = model.predict(X)
                r2 = r2_score(y, y_pred)
                
                importance_scores.append(abs(model.coef_[0]))
                coefficients.append(model.coef_[0])
                r2_scores.append(r2)
            except Exception as e:
                logger.debug(f"Ridge回帰エラー {factor_name}: {e}")
        
        # ランダムフォレスト
        if 'random_forest' in self.config['models_to_use']:
            try:
                model = RandomForestRegressor(
                    n_estimators=self.config['n_estimators'],
                    random_state=self.config['random_state']
                )
                model.fit(X, y)
                y_pred = model.predict(X)
                r2 = r2_score(y, y_pred)
                
                importance_scores.append(model.feature_importances_[0])
                coefficients.append(0.0)  # RF は係数を持たない
                r2_scores.append(r2)
            except Exception as e:
                logger.debug(f"ランダムフォレストエラー {factor_name}: {e}")
        
        # 勾配ブースティング
        if 'gradient_boosting' in self.config['models_to_use']:
            try:
                model = GradientBoostingRegressor(
                    n_estimators=self.config['n_estimators'],
                    random_state=self.config['random_state']
                )
                model.fit(X, y)
                y_pred = model.predict(X)
                r2 = r2_score(y, y_pred)
                
                importance_scores.append(model.feature_importances_[0])
                coefficients.append(0.0)  # GB は係数を持たない
                r2_scores.append(r2)
            except Exception as e:
                logger.debug(f"勾配ブースティングエラー {factor_name}: {e}")
        
        if not importance_scores:
            return None
        
        # 平均重要度スコア算出
        avg_importance = np.mean(importance_scores)
        avg_coefficient = np.mean([c for c in coefficients if c != 0.0]) if any(c != 0.0 for c in coefficients) else 0.0
        avg_r2 = np.mean(r2_scores)
        
        # 信頼区間計算（簡易版）
        std_importance = np.std(importance_scores) if len(importance_scores) > 1 else 0.0
        confidence_interval = (
            avg_importance - 1.96 * std_importance,
            avg_importance + 1.96 * std_importance
        )
        
        # 重要度閾値チェック
        if avg_importance < self.config['min_importance_threshold']:
            return None
        
        return FactorImpactResult(
            factor_name=factor_name,
            evaluation_metric=target_metric,
            market_category=market_category or 'all',
            importance_score=avg_importance,
            correlation=correlation,
            p_value=p_value,
            coefficient=avg_coefficient,
            confidence_interval=confidence_interval,
            model_type='ensemble',
            r2_score=avg_r2
        )
    
    def compare_market_categories(
        self,
        data: pd.DataFrame,
        target_metric: str
    ) -> Dict[str, List[FactorImpactResult]]:
        """
        市場カテゴリー間での要因項目影響差異分析
        
        Args:
            data: 分析対象データ
            target_metric: 対象評価項目
            
        Returns:
            市場カテゴリー別分析結果辞書
        """
        logger.info(f"市場カテゴリー比較分析開始: {target_metric}")
        
        results = {}
        
        for category in self.market_categories:
            if category in data['market_category'].unique():
                category_results = self.analyze_factor_impact(
                    data, target_metric, category
                )
                results[category] = category_results
                logger.info(f"{category}: {len(category_results)} 要因項目分析完了")
        
        return results
    
    def analyze_temporal_trends(
        self,
        data: pd.DataFrame,
        target_metric: str,
        factor_name: str,
        window_size: int = 5
    ) -> Dict[str, List[float]]:
        """
        要因項目影響の時系列変化分析
        
        Args:
            data: 分析対象データ 
            target_metric: 対象評価項目
            factor_name: 対象要因項目
            window_size: 移動窓サイズ（年）
            
        Returns:
            時系列影響変化データ
        """
        logger.info(f"時系列影響分析開始: {factor_name} -> {target_metric}")
        
        if 'year' not in data.columns:
            logger.error("年度列が存在しません")
            return {}
        
        years = sorted(data['year'].unique())
        correlations = []
        importance_scores = []
        
        for i in range(len(years) - window_size + 1):
            start_year = years[i]
            end_year = years[i + window_size - 1]
            
            period_data = data[
                (data['year'] >= start_year) & (data['year'] <= end_year)
            ]
            
            if len(period_data) < 10:
                correlations.append(np.nan)
                importance_scores.append(np.nan)
                continue
                
            try:
                result = self._analyze_single_factor(
                    period_data, factor_name, target_metric
                )
                
                if result:
                    correlations.append(result.correlation)
                    importance_scores.append(result.importance_score)
                else:
                    correlations.append(np.nan)
                    importance_scores.append(np.nan)
                    
            except Exception as e:
                logger.debug(f"期間 {start_year}-{end_year} 分析エラー: {e}")
                correlations.append(np.nan)
                importance_scores.append(np.nan)
        
        return {
            'years': years[window_size-1:],
            'correlations': correlations,
            'importance_scores': importance_scores
        }
    
    def get_top_factors(
        self,
        results: List[FactorImpactResult],
        n_top: int = 10,
        sort_by: str = 'importance_score'
    ) -> List[FactorImpactResult]:
        """
        上位要因項目取得
        
        Args:
            results: 分析結果リスト
            n_top: 上位何項目取得するか
            sort_by: ソート基準（'importance_score', 'correlation', 'r2_score'）
            
        Returns:
            上位要因項目リスト
        """
        if sort_by not in ['importance_score', 'correlation', 'r2_score']:
            sort_by = 'importance_score'
        
        # 絶対値でソート
        if sort_by == 'correlation':
            sorted_results = sorted(results, key=lambda x: abs(x.correlation), reverse=True)
        else:
            sorted_results = sorted(results, key=lambda x: abs(getattr(x, sort_by)), reverse=True)
        
        return sorted_results[:n_top]
    
    def calculate_interaction_effects(
        self,
        data: pd.DataFrame,
        target_metric: str,
        factor_pairs: List[Tuple[str, str]]
    ) -> List[Dict]:
        """
        要因項目間の交互作用効果分析
        
        Args:
            data: 分析対象データ
            target_metric: 対象評価項目  
            factor_pairs: 交互作用を調べる要因項目ペアのリスト
            
        Returns:
            交互作用効果分析結果
        """
        logger.info(f"交互作用分析開始: {len(factor_pairs)} ペア")
        
        interaction_results = []
        
        for factor1, factor2 in factor_pairs:
            if factor1 not in data.columns or factor2 not in data.columns:
                continue
                
            if target_metric not in data.columns:
                continue
            
            try:
                # 交互作用項作成
                interaction_term = data[factor1] * data[factor2]
                
                # 重回帰分析
                X = np.column_stack([
                    data[factor1].values,
                    data[factor2].values, 
                    interaction_term.values
                ])
                y = data[target_metric].values
                
                # NaN値除去
                mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
                X_clean = X[mask]
                y_clean = y[mask]
                
                if len(X_clean) < 10:
                    continue
                
                model = LinearRegression()
                model.fit(X_clean, y_clean)
                y_pred = model.predict(X_clean)
                r2 = r2_score(y_clean, y_pred)
                
                # 交互作用項の係数と有意性
                interaction_coef = model.coef_[2]
                
                # F検定による有意性テスト（簡易版）
                mse = mean_squared_error(y_clean, y_pred)
                interaction_t_stat = abs(interaction_coef) / (mse ** 0.5)
                
                interaction_results.append({
                    'factor1': factor1,
                    'factor2': factor2,
                    'interaction_coefficient': interaction_coef,
                    'interaction_t_statistic': interaction_t_stat,
                    'model_r2': r2,
                    'individual_coef1': model.coef_[0],
                    'individual_coef2': model.coef_[1],
                    'n_samples': len(X_clean)
                })
                
            except Exception as e:
                logger.debug(f"交互作用分析エラー {factor1}-{factor2}: {e}")
                continue
        
        # 交互作用効果の大きさでソート
        interaction_results.sort(
            key=lambda x: abs(x['interaction_coefficient']), 
            reverse=True
        )
        
        logger.info(f"交互作用分析完了: {len(interaction_results)} 結果")
        return interaction_results
    
    def generate_impact_summary(
        self,
        results: List[FactorImpactResult],
        market_category: str = 'all'
    ) -> Dict:
        """
        影響分析結果のサマリー生成
        
        Args:
            results: 分析結果リスト
            market_category: 市場カテゴリー
            
        Returns:
            サマリー辞書
        """
        if not results:
            return {}
        
        # 統計サマリー
        importance_scores = [r.importance_score for r in results]
        correlations = [r.correlation for r in results]
        r2_scores = [r.r2_score for r in results]
        
        summary = {
            'market_category': market_category,
            'total_factors_analyzed': len(results),
            'significant_factors': len([r for r in results if r.p_value <= 0.05]),
            'average_importance': np.mean(importance_scores),
            'max_importance': max(importance_scores),
            'min_importance': min(importance_scores),
            'average_correlation': np.mean(np.abs(correlations)),
            'max_correlation': max(np.abs(correlations)),
            'average_r2': np.mean(r2_scores),
            'top_positive_factors': [
                {'name': r.factor_name, 'score': r.importance_score, 'correlation': r.correlation}
                for r in results if r.correlation > 0
            ][:5],
            'top_negative_factors': [
                {'name': r.factor_name, 'score': r.importance_score, 'correlation': r.correlation}
                for r in results if r.correlation < 0
            ][:5]
        }
        
        return summary
    
    def save_results(
        self, 
        results: List[FactorImpactResult], 
        output_path: Union[str, Path],
        format: str = 'csv'
    ) -> None:
        """
        分析結果の保存
        
        Args:
            results: 分析結果リスト
            output_path: 出力パス
            format: 出力フォーマット ('csv', 'json', 'excel')
        """
        if not results:
            logger.warning("保存する結果がありません")
            return
        
        # DataFrame変換
        df_data = []
        for result in results:
            df_data.append({
                'factor_name': result.factor_name,
                'evaluation_metric': result.evaluation_metric,
                'market_category': result.market_category,
                'importance_score': result.importance_score,
                'correlation': result.correlation,
                'p_value': result.p_value,
                'coefficient': result.coefficient,
                'confidence_interval_lower': result.confidence_interval[0],
                'confidence_interval_upper': result.confidence_interval[1],
                'model_type': result.model_type,
                'r2_score': result.r2_score
            })
        
        df = pd.DataFrame(df_data)
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'csv':
            df.to_csv(output_path.with_suffix('.csv'), index=False)
        elif format == 'json':
            df.to_json(output_path.with_suffix('.json'), orient='records', indent=2)
        elif format == 'excel':
            df.to_excel(output_path.with_suffix('.xlsx'), index=False)
        else:
            raise ValueError(f"サポートされていないフォーマット: {format}")
        
        logger.info(f"分析結果を保存しました: {output_path}")


def main():
    """使用例・テスト用メイン関数"""
    
    # サンプルデータ生成
    np.random.seed(42)
    n_samples = 1000
    n_years = 10
    
    sample_data = {
        'company_id': np.repeat(range(100), n_years),
        'year': np.tile(range(2015, 2025), 100),
        'market_category': np.random.choice(['high_share_markets', 'declining_markets', 'lost_markets'], n_samples),
        'sales': np.random.lognormal(10, 1, n_samples),
        'tangible_fixed_assets': np.random.lognormal(8, 0.8, n_samples),
        'rd_expenses': np.random.lognormal(6, 1.2, n_samples),
        'employee_count': np.random.lognormal(5, 0.5, n_samples),
        'overseas_sales_ratio': np.random.uniform(0, 1, n_samples)
    }
    
    df = pd.DataFrame(sample_data)
    
    # 分析器初期化
    analyzer = FactorImpactAnalyzer()
    
    # 単一評価項目分析
    print("=== 売上高要因分析 ===")
    results = analyzer.analyze_factor_impact(df, 'sales')
    for result in results[:3]:
        print(f"{result.factor_name}: 重要度={result.importance_score:.4f}, 相関={result.correlation:.4f}")
    
    # 市場カテゴリー比較分析
    print("\n=== 市場カテゴリー比較分析 ===")
    market_results = analyzer.compare_market_categories(df, 'sales')
    for category, category_results in market_results.items():
        print(f"\n{category}:")
        for result in category_results[:2]:
            print(f"  {result.factor_name}: 重要度={result.importance_score:.4f}")
    
    # 時系列影響分析
    print("\n=== 時系列影響分析 ===")
    temporal_results = analyzer.analyze_temporal_trends(
        df, 'sales', 'tangible_fixed_assets', window_size=3
    )
    if temporal_results.get('correlations'):
        print(f"時系列相関変化: {len(temporal_results['correlations'])} データポイント")
    
    # 交互作用分析
    print("\n=== 交互作用分析 ===")
    interaction_pairs = [
        ('tangible_fixed_assets', 'rd_expenses'),
        ('employee_count', 'overseas_sales_ratio')
    ]
    interaction_results = analyzer.calculate_interaction_effects(
        df, 'sales', interaction_pairs
    )
    for interaction in interaction_results[:2]:
        print(f"{interaction['factor1']} × {interaction['factor2']}: "
                f"係数={interaction['interaction_coefficient']:.4f}")
    
    # サマリー生成
    print("\n=== 分析サマリー ===")
    summary = analyzer.generate_impact_summary(results, 'all')
    print(f"分析要因数: {summary.get('total_factors_analyzed', 0)}")
    print(f"有意要因数: {summary.get('significant_factors', 0)}")
    print(f"平均重要度: {summary.get('average_importance', 0):.4f}")
    
    # 結果保存（例）
    # analyzer.save_results(results, 'output/factor_analysis_results', 'csv')
    
    print("\n要因項目影響分析完了")


if __name__ == "__main__":
    main()