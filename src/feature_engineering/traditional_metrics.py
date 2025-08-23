"""
A2AI (Advanced Financial Analysis AI) - Traditional Metrics Module

従来の6つの財務評価項目を計算するクラス群:
1. 売上高 (Revenue)
2. 売上高成長率 (Revenue Growth Rate) 
3. 売上高営業利益率 (Operating Margin)
4. 売上高当期純利益率 (Net Profit Margin)
5. ROE (Return on Equity)
6. 売上高付加価値率 (Value Added Ratio)

各評価項目は対応する20の要因項目との関係分析に使用される
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
import warnings
from datetime import datetime
import logging

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TraditionalMetricsCalculator:
    """
    従来の財務評価項目を計算するメインクラス
    150社×40年分の財務データに対応
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        初期化
        
        Args:
            data: 財務諸表データ（企業×年度のパネルデータ）
        """
        self.data = data.copy()
        self.validate_data()
        
    def validate_data(self):
        """データ整合性検証"""
        required_columns = [
            'company_id', 'year', 'revenue', 'operating_income', 
            'net_income', 'total_assets', 'equity', 'cost_of_goods_sold',
            'employees', 'tangible_fixed_assets'
        ]
        
        missing_cols = [col for col in required_columns if col not in self.data.columns]
        if missing_cols:
            logger.warning(f"Missing columns detected: {missing_cols}")
            
    def calculate_all_metrics(self) -> pd.DataFrame:
        """
        全ての従来評価項目を一括計算
        
        Returns:
            計算済み評価項目を含むDataFrame
        """
        result_df = self.data.copy()
        
        # 各評価項目を計算
        result_df['revenue_metric'] = self.calculate_revenue()
        result_df['revenue_growth_rate'] = self.calculate_revenue_growth_rate()
        result_df['operating_margin'] = self.calculate_operating_margin()
        result_df['net_profit_margin'] = self.calculate_net_profit_margin()
        result_df['roe'] = self.calculate_roe()
        result_df['value_added_ratio'] = self.calculate_value_added_ratio()
        
        return result_df


class RevenueMetrics:
    """売上高評価項目の計算クラス"""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        
    def calculate_revenue(self) -> pd.Series:
        """
        売上高計算（基準値として使用）
        
        Returns:
            売上高系列
        """
        return self.data['revenue'].fillna(0)
    
    def calculate_log_revenue(self) -> pd.Series:
        """
        対数売上高（規模効果分析用）
        
        Returns:
            対数売上高系列
        """
        revenue = self.data['revenue'].replace(0, np.nan)
        return np.log(revenue)
    
    def calculate_revenue_per_employee(self) -> pd.Series:
        """
        従業員一人当たり売上高
        
        Returns:
            従業員一人当たり売上高
        """
        revenue = self.data['revenue']
        employees = self.data['employees'].replace(0, np.nan)
        return revenue / employees


class RevenueGrowthRateMetrics:
    """売上高成長率評価項目の計算クラス"""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data.sort_values(['company_id', 'year'])
        
    def calculate_revenue_growth_rate(self, periods: int = 1) -> pd.Series:
        """
        売上高成長率計算
        
        Args:
            periods: 成長率計算期間（デフォルト：1年）
            
        Returns:
            売上高成長率系列
        """
        revenue = self.data['revenue']
        revenue_lag = self.data.groupby('company_id')['revenue'].shift(periods)
        
        # ゼロ除算回避
        revenue_lag = revenue_lag.replace(0, np.nan)
        growth_rate = (revenue - revenue_lag) / revenue_lag
        
        return growth_rate
    
    def calculate_cagr(self, periods: int = 3) -> pd.Series:
        """
        年平均成長率（CAGR）計算
        
        Args:
            periods: CAGR計算期間
            
        Returns:
            CAGR系列
        """
        revenue_current = self.data['revenue']
        revenue_base = self.data.groupby('company_id')['revenue'].shift(periods)
        
        # ゼロ・負値回避
        revenue_base = revenue_base.replace(0, np.nan)
        revenue_current = revenue_current.replace(0, np.nan)
        
        cagr = (revenue_current / revenue_base) ** (1/periods) - 1
        return cagr
    
    def calculate_growth_volatility(self, window: int = 5) -> pd.Series:
        """
        売上高成長率のボラティリティ
        
        Args:
            window: 計算ウィンドウ
            
        Returns:
            成長率ボラティリティ系列
        """
        growth_rate = self.calculate_revenue_growth_rate()
        volatility = growth_rate.groupby(self.data['company_id']).rolling(window=window).std()
        return volatility.reset_index(level=0, drop=True)


class OperatingMarginMetrics:
    """売上高営業利益率評価項目の計算クラス"""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        
    def calculate_operating_margin(self) -> pd.Series:
        """
        売上高営業利益率計算
        
        Returns:
            売上高営業利益率系列
        """
        operating_income = self.data['operating_income']
        revenue = self.data['revenue'].replace(0, np.nan)
        
        return operating_income / revenue
    
    def calculate_operating_leverage(self) -> pd.Series:
        """
        営業レバレッジ（営業利益の売上高弾力性）
        
        Returns:
            営業レバレッジ系列
        """
        # 営業利益変化率 / 売上高変化率
        data_sorted = self.data.sort_values(['company_id', 'year'])
        
        operating_income_growth = data_sorted.groupby('company_id')['operating_income'].pct_change()
        revenue_growth = data_sorted.groupby('company_id')['revenue'].pct_change()
        
        # ゼロ除算回避
        revenue_growth = revenue_growth.replace(0, np.nan)
        leverage = operating_income_growth / revenue_growth
        
        return leverage
    
    def calculate_margin_stability(self, window: int = 5) -> pd.Series:
        """
        営業利益率の安定性（標準偏差）
        
        Args:
            window: 計算ウィンドウ
            
        Returns:
            営業利益率安定性系列
        """
        operating_margin = self.calculate_operating_margin()
        stability = operating_margin.groupby(self.data['company_id']).rolling(window=window).std()
        return stability.reset_index(level=0, drop=True)


class NetProfitMarginMetrics:
    """売上高当期純利益率評価項目の計算クラス"""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        
    def calculate_net_profit_margin(self) -> pd.Series:
        """
        売上高当期純利益率計算
        
        Returns:
            売上高当期純利益率系列
        """
        net_income = self.data['net_income']
        revenue = self.data['revenue'].replace(0, np.nan)
        
        return net_income / revenue
    
    def calculate_profit_margin_trend(self, window: int = 3) -> pd.Series:
        """
        純利益率のトレンド（移動平均）
        
        Args:
            window: 移動平均ウィンドウ
            
        Returns:
            純利益率トレンド系列
        """
        net_profit_margin = self.calculate_net_profit_margin()
        trend = net_profit_margin.groupby(self.data['company_id']).rolling(window=window).mean()
        return trend.reset_index(level=0, drop=True)
    
    def calculate_profit_quality(self) -> pd.Series:
        """
        利益の質（営業利益に対する当期純利益の比率）
        
        Returns:
            利益の質指標
        """
        net_income = self.data['net_income']
        operating_income = self.data['operating_income'].replace(0, np.nan)
        
        return net_income / operating_income


class ROEMetrics:
    """ROE（自己資本利益率）評価項目の計算クラス"""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        
    def calculate_roe(self) -> pd.Series:
        """
        ROE計算
        
        Returns:
            ROE系列
        """
        net_income = self.data['net_income']
        equity = self.data['equity'].replace(0, np.nan)
        
        return net_income / equity
    
    def calculate_dupont_components(self) -> pd.DataFrame:
        """
        デュポン分析による ROE = 純利益率 × 総資産回転率 × 財務レバレッジ
        
        Returns:
            デュポン分析構成要素のDataFrame
        """
        net_profit_margin = self.data['net_income'] / self.data['revenue'].replace(0, np.nan)
        asset_turnover = self.data['revenue'] / self.data['total_assets'].replace(0, np.nan)
        equity_multiplier = self.data['total_assets'] / self.data['equity'].replace(0, np.nan)
        
        roe_dupont = net_profit_margin * asset_turnover * equity_multiplier
        
        return pd.DataFrame({
            'net_profit_margin': net_profit_margin,
            'asset_turnover': asset_turnover,
            'equity_multiplier': equity_multiplier,
            'roe_dupont': roe_dupont
        })
    
    def calculate_sustainable_growth_rate(self) -> pd.Series:
        """
        持続可能成長率 = ROE × (1 - 配当性向)
        
        Returns:
            持続可能成長率系列
        """
        roe = self.calculate_roe()
        
        # 配当性向が利用可能な場合
        if 'dividend_payout_ratio' in self.data.columns:
            payout_ratio = self.data['dividend_payout_ratio'].fillna(0)
            retention_ratio = 1 - payout_ratio
        else:
            # 配当性向データがない場合は内部留保率を推定
            retention_ratio = 0.7  # 仮定値
            
        return roe * retention_ratio


class ValueAddedRatioMetrics:
    """売上高付加価値率評価項目の計算クラス"""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        
    def calculate_value_added_ratio(self) -> pd.Series:
        """
        売上高付加価値率計算
        付加価値 = 売上高 - 売上原価（材料費・外注費等）
        
        Returns:
            売上高付加価値率系列
        """
        revenue = self.data['revenue']
        cost_of_goods_sold = self.data['cost_of_goods_sold'].fillna(0)
        
        value_added = revenue - cost_of_goods_sold
        revenue_nonzero = revenue.replace(0, np.nan)
        
        return value_added / revenue_nonzero
    
    def calculate_labor_productivity(self) -> pd.Series:
        """
        労働生産性（従業員一人当たり付加価値）
        
        Returns:
            労働生産性系列
        """
        value_added = self._calculate_value_added()
        employees = self.data['employees'].replace(0, np.nan)
        
        return value_added / employees
    
    def calculate_capital_productivity(self) -> pd.Series:
        """
        資本生産性（有形固定資産当たり付加価値）
        
        Returns:
            資本生産性系列
        """
        value_added = self._calculate_value_added()
        tangible_assets = self.data['tangible_fixed_assets'].replace(0, np.nan)
        
        return value_added / tangible_assets
    
    def _calculate_value_added(self) -> pd.Series:
        """付加価値計算（内部使用）"""
        revenue = self.data['revenue']
        cost_of_goods_sold = self.data['cost_of_goods_sold'].fillna(0)
        return revenue - cost_of_goods_sold
    
    def calculate_value_added_components(self) -> pd.DataFrame:
        """
        付加価値構成要素分析
        
        Returns:
            付加価値構成要素のDataFrame
        """
        value_added = self._calculate_value_added()
        
        # 付加価値の主要構成要素
        labor_costs = self.data.get('labor_costs', 0)
        depreciation = self.data.get('depreciation', 0)
        operating_income = self.data['operating_income']
        
        return pd.DataFrame({
            'value_added': value_added,
            'labor_share': labor_costs / value_added.replace(0, np.nan),
            'depreciation_share': depreciation / value_added.replace(0, np.nan),
            'profit_share': operating_income / value_added.replace(0, np.nan)
        })


class TraditionalMetricsIntegrator:
    """
    全ての従来評価項目を統合計算するクラス
    150社×40年のデータセット全体に対応
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        初期化
        
        Args:
            data: 財務諸表データ（company_id, year, 各種財務データを含む）
        """
        self.data = data.copy()
        self.revenue_calculator = RevenueMetrics(data)
        self.growth_calculator = RevenueGrowthRateMetrics(data)
        self.operating_margin_calculator = OperatingMarginMetrics(data)
        self.net_margin_calculator = NetProfitMarginMetrics(data)
        self.roe_calculator = ROEMetrics(data)
        self.value_added_calculator = ValueAddedRatioMetrics(data)
        
    def calculate_all_traditional_metrics(self) -> pd.DataFrame:
        """
        全ての従来評価項目を一括計算
        
        Returns:
            全評価項目を含むDataFrame
        """
        result_df = self.data.copy()
        
        logger.info("Computing traditional financial metrics...")
        
        # 1. 売上高関連指標
        result_df['revenue'] = self.revenue_calculator.calculate_revenue()
        result_df['log_revenue'] = self.revenue_calculator.calculate_log_revenue()
        result_df['revenue_per_employee'] = self.revenue_calculator.calculate_revenue_per_employee()
        
        # 2. 売上高成長率関連指標
        result_df['revenue_growth_rate'] = self.growth_calculator.calculate_revenue_growth_rate()
        result_df['revenue_cagr_3y'] = self.growth_calculator.calculate_cagr(3)
        result_df['growth_volatility'] = self.growth_calculator.calculate_growth_volatility()
        
        # 3. 営業利益率関連指標
        result_df['operating_margin'] = self.operating_margin_calculator.calculate_operating_margin()
        result_df['operating_leverage'] = self.operating_margin_calculator.calculate_operating_leverage()
        result_df['margin_stability'] = self.operating_margin_calculator.calculate_margin_stability()
        
        # 4. 純利益率関連指標
        result_df['net_profit_margin'] = self.net_margin_calculator.calculate_net_profit_margin()
        result_df['profit_margin_trend'] = self.net_margin_calculator.calculate_profit_margin_trend()
        result_df['profit_quality'] = self.net_margin_calculator.calculate_profit_quality()
        
        # 5. ROE関連指標
        result_df['roe'] = self.roe_calculator.calculate_roe()
        dupont_components = self.roe_calculator.calculate_dupont_components()
        result_df = pd.concat([result_df, dupont_components], axis=1)
        result_df['sustainable_growth_rate'] = self.roe_calculator.calculate_sustainable_growth_rate()
        
        # 6. 付加価値率関連指標
        result_df['value_added_ratio'] = self.value_added_calculator.calculate_value_added_ratio()
        result_df['labor_productivity'] = self.value_added_calculator.calculate_labor_productivity()
        result_df['capital_productivity'] = self.value_added_calculator.calculate_capital_productivity()
        
        logger.info("Traditional metrics computation completed.")
        
        return result_df
    
    def get_metrics_summary(self) -> pd.DataFrame:
        """
        評価項目のサマリー統計
        
        Returns:
            各評価項目の基本統計量
        """
        metrics_df = self.calculate_all_traditional_metrics()
        
        metric_columns = [
            'revenue', 'revenue_growth_rate', 'operating_margin', 
            'net_profit_margin', 'roe', 'value_added_ratio'
        ]
        
        summary = metrics_df[metric_columns].describe()
        return summary
    
    def detect_outliers(self, method: str = 'iqr', threshold: float = 1.5) -> pd.DataFrame:
        """
        外れ値検出
        
        Args:
            method: 検出手法（'iqr' or 'zscore'）
            threshold: 閾値
            
        Returns:
            外れ値フラグを含むDataFrame
        """
        metrics_df = self.calculate_all_traditional_metrics()
        
        metric_columns = [
            'revenue_growth_rate', 'operating_margin', 
            'net_profit_margin', 'roe', 'value_added_ratio'
        ]
        
        outlier_flags = pd.DataFrame()
        
        for col in metric_columns:
            if method == 'iqr':
                Q1 = metrics_df[col].quantile(0.25)
                Q3 = metrics_df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                outlier_flags[f'{col}_outlier'] = (
                    (metrics_df[col] < lower_bound) | 
                    (metrics_df[col] > upper_bound)
                )
            elif method == 'zscore':
                z_scores = np.abs((metrics_df[col] - metrics_df[col].mean()) / metrics_df[col].std())
                outlier_flags[f'{col}_outlier'] = z_scores > threshold
                
        return pd.concat([metrics_df, outlier_flags], axis=1)


# 使用例とテスト用関数
def example_usage():
    """使用例"""
    # サンプルデータ作成（実際のEDINETデータ構造を想定）
    np.random.seed(42)
    n_companies = 10
    n_years = 5
    
    sample_data = []
    for company_id in range(1, n_companies + 1):
        for year in range(2020, 2020 + n_years):
            sample_data.append({
                'company_id': company_id,
                'year': year,
                'revenue': np.random.uniform(100000, 1000000),
                'operating_income': np.random.uniform(5000, 50000),
                'net_income': np.random.uniform(3000, 30000),
                'total_assets': np.random.uniform(200000, 2000000),
                'equity': np.random.uniform(100000, 1000000),
                'cost_of_goods_sold': np.random.uniform(50000, 500000),
                'employees': np.random.randint(100, 10000),
                'tangible_fixed_assets': np.random.uniform(50000, 500000)
            })
    
    df = pd.DataFrame(sample_data)
    
    # 評価項目計算
    integrator = TraditionalMetricsIntegrator(df)
    result = integrator.calculate_all_traditional_metrics()
    
    print("Sample of calculated metrics:")
    print(result[['company_id', 'year', 'revenue_growth_rate', 'operating_margin', 'roe']].head(10))
    
    print("\nMetrics summary:")
    print(integrator.get_metrics_summary())


if __name__ == "__main__":
    example_usage()