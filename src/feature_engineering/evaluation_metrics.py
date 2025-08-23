"""
A2AI (Advanced Financial Analysis AI) - 評価項目計算クラス
財務諸表分析における9つの評価項目を計算するためのメインクラス

評価項目:
1. 売上高 (Revenue)
2. 売上高成長率 (Revenue Growth Rate) 
3. 売上高営業利益率 (Operating Profit Margin)
4. 売上高当期純利益率 (Net Profit Margin)
5. ROE (Return on Equity)
6. 売上高付加価値率 (Value Added Ratio)
7. 企業存続確率 (Survival Probability) - 新規
8. 新規事業成功率 (Emergence Success Rate) - 新規
9. 事業継承成功度 (Business Succession Score) - 新規
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import warnings
from datetime import datetime, timedelta
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class CompanyInfo:
    """企業基本情報"""
    company_id: str
    company_name: str
    market_category: str  # 'high_share', 'declining', 'lost'
    establishment_date: Optional[datetime] = None
    extinction_date: Optional[datetime] = None
    parent_company: Optional[str] = None
    is_spinoff: bool = False


class BaseEvaluationMetric(ABC):
    """評価項目計算の基底クラス"""
    
    def __init__(self, metric_name: str):
        self.metric_name = metric_name
        self.required_columns = []
    
    @abstractmethod
    def calculate(self, data: pd.DataFrame, company_info: CompanyInfo) -> pd.Series:
        """評価項目を計算する抽象メソッド"""
        pass
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """データの妥当性をチェック"""
        missing_cols = set(self.required_columns) - set(data.columns)
        if missing_cols:
            warnings.warn(f"{self.metric_name}の計算に必要な列が不足: {missing_cols}")
            return False
        return True


class RevenueMetric(BaseEvaluationMetric):
    """売上高計算クラス"""
    
    def __init__(self):
        super().__init__("売上高")
        self.required_columns = ['revenue', 'fiscal_year']
    
    def calculate(self, data: pd.DataFrame, company_info: CompanyInfo) -> pd.Series:
        """売上高を計算"""
        if not self.validate_data(data):
            return pd.Series(dtype=float)
        
        # 基本的には売上高データをそのまま返すが、品質チェックを実施
        revenue_data = data['revenue'].copy()
        
        # 異常値チェック（前年比10倍以上の変化は要確認）
        if len(revenue_data) > 1:
            growth_rates = revenue_data.pct_change()
            extreme_changes = abs(growth_rates) > 10
            if extreme_changes.any():
                warnings.warn(f"{company_info.company_name}: 売上高に異常な変化を検出")
        
        # 企業消滅情報を考慮した処理
        if company_info.extinction_date:
            extinction_year = company_info.extinction_date.year
            revenue_data = revenue_data[data['fiscal_year'] <= extinction_year]
        
        return revenue_data


class RevenueGrowthRateMetric(BaseEvaluationMetric):
    """売上高成長率計算クラス"""
    
    def __init__(self):
        super().__init__("売上高成長率")
        self.required_columns = ['revenue', 'fiscal_year']
    
    def calculate(self, data: pd.DataFrame, company_info: CompanyInfo) -> pd.Series:
        """売上高成長率を計算"""
        if not self.validate_data(data):
            return pd.Series(dtype=float)
        
        # 売上高を時系列順にソート
        sorted_data = data.sort_values('fiscal_year')
        revenue = sorted_data['revenue']
        
        # 前年同期比成長率を計算
        growth_rate = revenue.pct_change()
        
        # 新設企業の場合、設立初年度は成長率計算不可
        if company_info.establishment_date:
            establishment_year = company_info.establishment_date.year
            first_year_mask = sorted_data['fiscal_year'] == establishment_year
            growth_rate.loc[first_year_mask] = np.nan
        
        # 分社企業の場合、特別な処理
        if company_info.is_spinoff and company_info.parent_company:
            # 分社年度の成長率は親会社データとの継続性を考慮
            # ここでは一旦NaNとして後続処理で補完
            spinoff_year_mask = sorted_data['fiscal_year'] <= (
                company_info.establishment_date.year if company_info.establishment_date else 1984
            )
            growth_rate.loc[spinoff_year_mask] = np.nan
        
        return growth_rate


class OperatingProfitMarginMetric(BaseEvaluationMetric):
    """売上高営業利益率計算クラス"""
    
    def __init__(self):
        super().__init__("売上高営業利益率")
        self.required_columns = ['revenue', 'operating_profit', 'fiscal_year']
    
    def calculate(self, data: pd.DataFrame, company_info: CompanyInfo) -> pd.Series:
        """売上高営業利益率を計算"""
        if not self.validate_data(data):
            return pd.Series(dtype=float)
        
        revenue = data['revenue']
        operating_profit = data['operating_profit']
        
        # 売上高営業利益率 = 営業利益 / 売上高 × 100
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            margin = (operating_profit / revenue) * 100
        
        # 異常値処理（-100%未満、100%超過は異常値として扱う）
        margin = margin.clip(-100, 100)
        
        # 企業消滅年度までのデータに制限
        if company_info.extinction_date:
            extinction_year = company_info.extinction_date.year
            mask = data['fiscal_year'] <= extinction_year
            margin = margin[mask]
        
        return margin


class NetProfitMarginMetric(BaseEvaluationMetric):
    """売上高当期純利益率計算クラス"""
    
    def __init__(self):
        super().__init__("売上高当期純利益率")
        self.required_columns = ['revenue', 'net_income', 'fiscal_year']
    
    def calculate(self, data: pd.DataFrame, company_info: CompanyInfo) -> pd.Series:
        """売上高当期純利益率を計算"""
        if not self.validate_data(data):
            return pd.Series(dtype=float)
        
        revenue = data['revenue']
        net_income = data['net_income']
        
        # 売上高当期純利益率 = 当期純利益 / 売上高 × 100
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            margin = (net_income / revenue) * 100
        
        # 異常値処理
        margin = margin.clip(-200, 200)
        
        # 企業消滅年度までのデータに制限
        if company_info.extinction_date:
            extinction_year = company_info.extinction_date.year
            mask = data['fiscal_year'] <= extinction_year
            margin = margin[mask]
        
        return margin


class ROEMetric(BaseEvaluationMetric):
    """ROE (Return on Equity) 計算クラス"""
    
    def __init__(self):
        super().__init__("ROE")
        self.required_columns = ['net_income', 'shareholders_equity', 'fiscal_year']
    
    def calculate(self, data: pd.DataFrame, company_info: CompanyInfo) -> pd.Series:
        """ROEを計算"""
        if not self.validate_data(data):
            return pd.Series(dtype=float)
        
        net_income = data['net_income']
        equity = data['shareholders_equity']
        
        # ROE = 当期純利益 / 自己資本 × 100
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            roe = (net_income / equity) * 100
        
        # 異常値処理（-500%～500%の範囲内に制限）
        roe = roe.clip(-500, 500)
        
        # 企業消滅年度までのデータに制限
        if company_info.extinction_date:
            extinction_year = company_info.extinction_date.year
            mask = data['fiscal_year'] <= extinction_year
            roe = roe[mask]
        
        return roe


class ValueAddedRatioMetric(BaseEvaluationMetric):
    """売上高付加価値率計算クラス"""
    
    def __init__(self):
        super().__init__("売上高付加価値率")
        self.required_columns = [
            'revenue', 'cost_of_goods_sold', 'material_costs', 
            'subcontractor_costs', 'fiscal_year'
        ]
    
    def calculate(self, data: pd.DataFrame, company_info: CompanyInfo) -> pd.Series:
        """売上高付加価値率を計算"""
        if not self.validate_data(data):
            return pd.Series(dtype=float)
        
        revenue = data['revenue']
        cogs = data['cost_of_goods_sold']
        
        # 材料費・外注費が利用可能な場合はより精密に計算
        if all(col in data.columns for col in ['material_costs', 'subcontractor_costs']):
            external_costs = data['material_costs'] + data['subcontractor_costs']
            value_added = revenue - external_costs
        else:
            # 簡易計算：売上高 - 売上原価
            value_added = revenue - cogs
        
        # 売上高付加価値率 = 付加価値 / 売上高 × 100
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            ratio = (value_added / revenue) * 100
        
        # 付加価値率は通常0-100%の範囲
        ratio = ratio.clip(0, 100)
        
        # 企業消滅年度までのデータに制限
        if company_info.extinction_date:
            extinction_year = company_info.extinction_date.year
            mask = data['fiscal_year'] <= extinction_year
            ratio = ratio[mask]
        
        return ratio


class SurvivalProbabilityMetric(BaseEvaluationMetric):
    """企業存続確率計算クラス（新規評価項目）"""
    
    def __init__(self):
        super().__init__("企業存続確率")
        self.required_columns = [
            'fiscal_year', 'revenue', 'operating_profit', 'net_income',
            'total_assets', 'shareholders_equity', 'cash_and_deposits'
        ]
    
    def calculate(self, data: pd.DataFrame, company_info: CompanyInfo) -> pd.Series:
        """企業存続確率を計算"""
        if not self.validate_data(data):
            return pd.Series(dtype=float)
        
        # Altman Z-Scoreベースの生存確率計算
        survival_scores = self._calculate_altman_zscore(data)
        
        # Z-Scoreを確率に変換（ロジスティック変換）
        survival_prob = 1 / (1 + np.exp(-survival_scores))
        
        # 企業が既に消滅している場合、消滅年度以降は0
        if company_info.extinction_date:
            extinction_year = company_info.extinction_date.year
            survival_prob.loc[data['fiscal_year'] > extinction_year] = 0.0
        
        return survival_prob * 100  # パーセンテージで返す
    
    def _calculate_altman_zscore(self, data: pd.DataFrame) -> pd.Series:
        """修正Altman Z-Scoreを計算"""
        # 必要な財務比率を計算
        working_capital = data.get('current_assets', data['total_assets'] * 0.4) - \
                         data.get('current_liabilities', data['total_assets'] * 0.3)
        
        wc_to_ta = working_capital / data['total_assets']
        retained_earnings_to_ta = data.get('retained_earnings', data['net_income']) / data['total_assets']
        ebit_to_ta = data['operating_profit'] / data['total_assets']
        equity_to_debt = data['shareholders_equity'] / (data['total_assets'] - data['shareholders_equity'])
        sales_to_ta = data['revenue'] / data['total_assets']
        
        # 修正Altman Z-Score計算
        z_score = (1.2 * wc_to_ta + 
                  1.4 * retained_earnings_to_ta +
                  3.3 * ebit_to_ta +
                  0.6 * equity_to_debt +
                  1.0 * sales_to_ta)
        
        return z_score


class EmergenceSuccessRateMetric(BaseEvaluationMetric):
    """新規事業成功率計算クラス（新規評価項目）"""
    
    def __init__(self):
        super().__init__("新規事業成功率")
        self.required_columns = [
            'fiscal_year', 'revenue', 'operating_profit', 'rd_expenses',
            'number_of_segments'
        ]
    
    def calculate(self, data: pd.DataFrame, company_info: CompanyInfo) -> pd.Series:
        """新規事業成功率を計算"""
        if not self.validate_data(data):
            return pd.Series(dtype=float)
        
        # 新設企業の場合のみ計算
        if not company_info.establishment_date:
            return pd.Series([50.0] * len(data), index=data.index)  # デフォルト値
        
        establishment_year = company_info.establishment_date.year
        
        # 設立からの経過年数を計算
        years_since_establishment = data['fiscal_year'] - establishment_year
        
        # 成功指標を複合的に計算
        success_indicators = []
        
        # 1. 売上成長率（設立2年目以降）
        revenue_growth = data['revenue'].pct_change()
        growth_score = np.where(revenue_growth > 0.1, 1.0, 
                                np.where(revenue_growth > 0, 0.5, 0.0))
        
        # 2. 収益性（営業利益率）
        operating_margin = data['operating_profit'] / data['revenue']
        profitability_score = np.where(operating_margin > 0.05, 1.0,
                                        np.where(operating_margin > 0, 0.5, 0.0))
        
        # 3. 研究開発投資積極性
        rd_intensity = data['rd_expenses'] / data['revenue']
        innovation_score = np.where(rd_intensity > 0.03, 1.0,
                                    np.where(rd_intensity > 0.01, 0.5, 0.0))
        
        # 4. 事業多角化（セグメント数の変化）
        diversification_score = np.where(data['number_of_segments'] > 1, 0.5, 0.0)
        
        # 総合成功率を計算（時間経過とともに重み調整）
        time_weight = np.minimum(years_since_establishment / 5, 1.0)  # 5年で最大重み
        
        success_rate = (growth_score * 0.3 + 
                       profitability_score * 0.3 +
                       innovation_score * 0.2 + 
                       diversification_score * 0.2) * time_weight
        
        return pd.Series(success_rate * 100, index=data.index)


class BusinessSuccessionScoreMetric(BaseEvaluationMetric):
    """事業継承成功度計算クラス（新規評価項目）"""
    
    def __init__(self):
        super().__init__("事業継承成功度")
        self.required_columns = [
            'fiscal_year', 'revenue', 'operating_profit', 'total_assets'
        ]
    
    def calculate(self, data: pd.DataFrame, company_info: CompanyInfo) -> pd.Series:
        """事業継承成功度を計算"""
        if not self.validate_data(data):
            return pd.Series(dtype=float)
        
        # 分社・統合企業の場合のみ意味のある指標
        if not company_info.is_spinoff:
            return pd.Series([50.0] * len(data), index=data.index)  # 中立値
        
        # 継承前後の性能比較が可能な場合の処理
        establishment_year = company_info.establishment_date.year if company_info.establishment_date else 2000
        
        # 事業継承後の性能指標
        succession_indicators = []
        
        # 1. 収益性の維持・向上
        operating_margin = (data['operating_profit'] / data['revenue']) * 100
        margin_trend = operating_margin.rolling(window=3, min_periods=1).mean().diff()
        profitability_score = np.where(margin_trend > 0, 1.0,
                                        np.where(margin_trend >= -1, 0.5, 0.0))
        
        # 2. 資産効率性
        asset_turnover = data['revenue'] / data['total_assets']
        efficiency_trend = asset_turnover.rolling(window=3, min_periods=1).mean().diff()
        efficiency_score = np.where(efficiency_trend > 0, 1.0,
                                    np.where(efficiency_trend >= -0.05, 0.5, 0.0))
        
        # 3. 成長性
        revenue_growth = data['revenue'].pct_change().rolling(window=3, min_periods=1).mean()
        growth_score = np.where(revenue_growth > 0.05, 1.0,
                                np.where(revenue_growth > 0, 0.5, 0.0))
        
        # 4. 安定性（収益の変動性）
        profit_volatility = data['operating_profit'].rolling(window=5, min_periods=2).std()
        stability_score = np.where(profit_volatility < data['operating_profit'].mean() * 0.2, 1.0, 0.5)
        
        # 継承からの経過年数による重み調整
        years_since_succession = data['fiscal_year'] - establishment_year
        time_weight = np.minimum(years_since_succession / 3, 1.0)  # 3年で安定
        
        succession_score = (profitability_score * 0.3 +
                          efficiency_score * 0.25 +
                          growth_score * 0.25 +
                          stability_score * 0.2) * time_weight
        
        return pd.Series(succession_score * 100, index=data.index)


class EvaluationMetricsCalculator:
    """評価項目計算の統合クラス"""
    
    def __init__(self):
        self.metrics = {
            'revenue': RevenueMetric(),
            'revenue_growth_rate': RevenueGrowthRateMetric(),
            'operating_profit_margin': OperatingProfitMarginMetric(),
            'net_profit_margin': NetProfitMarginMetric(),
            'roe': ROEMetric(),
            'value_added_ratio': ValueAddedRatioMetric(),
            'survival_probability': SurvivalProbabilityMetric(),
            'emergence_success_rate': EmergenceSuccessRateMetric(),
            'business_succession_score': BusinessSuccessionScoreMetric()
        }
    
    def calculate_all_metrics(self, 
                            data: pd.DataFrame, 
                            company_info: CompanyInfo) -> pd.DataFrame:
        """全ての評価項目を計算"""
        results = pd.DataFrame(index=data.index)
        
        for metric_name, metric_calculator in self.metrics.items():
            try:
                results[metric_name] = metric_calculator.calculate(data, company_info)
            except Exception as e:
                warnings.warn(f"{metric_name}の計算でエラーが発生: {e}")
                results[metric_name] = np.nan
        
        return results
    
    def calculate_single_metric(self, 
                                metric_name: str,
                                data: pd.DataFrame, 
                                company_info: CompanyInfo) -> pd.Series:
        """単一の評価項目を計算"""
        if metric_name not in self.metrics:
            raise ValueError(f"未知の評価項目: {metric_name}")
        
        return self.metrics[metric_name].calculate(data, company_info)
    
    def get_metric_names(self) -> List[str]:
        """利用可能な評価項目名を取得"""
        return list(self.metrics.keys())
    
    def get_required_columns(self, metric_name: str = None) -> Union[List[str], Dict[str, List[str]]]:
        """評価項目計算に必要な列名を取得"""
        if metric_name:
            if metric_name not in self.metrics:
                raise ValueError(f"未知の評価項目: {metric_name}")
            return self.metrics[metric_name].required_columns
        else:
            return {name: metric.required_columns for name, metric in self.metrics.items()}


# 使用例とテスト用のサンプル関数
def create_sample_company_info() -> CompanyInfo:
    """サンプル企業情報を作成"""
    return CompanyInfo(
        company_id="1001",
        company_name="ファナック",
        market_category="high_share",
        establishment_date=datetime(1972, 5, 1),
        extinction_date=None,
        parent_company=None,
        is_spinoff=False
    )


def create_sample_financial_data() -> pd.DataFrame:
    """サンプル財務データを作成"""
    np.random.seed(42)
    years = range(2020, 2025)
    
    data = pd.DataFrame({
        'fiscal_year': years,
        'revenue': [800000, 820000, 850000, 880000, 900000],
        'operating_profit': [120000, 125000, 130000, 135000, 140000],
        'net_income': [80000, 82000, 85000, 88000, 90000],
        'total_assets': [1500000, 1520000, 1550000, 1580000, 1600000],
        'shareholders_equity': [800000, 820000, 850000, 880000, 900000],
        'cost_of_goods_sold': [480000, 492000, 510000, 528000, 540000],
        'cash_and_deposits': [200000, 210000, 220000, 230000, 240000],
        'rd_expenses': [40000, 42000, 44000, 46000, 48000],
        'number_of_segments': [3, 3, 4, 4, 4],
        'current_assets': [600000, 610000, 620000, 630000, 640000],
        'current_liabilities': [300000, 300000, 300000, 300000, 300000],
        'retained_earnings': [400000, 420000, 440000, 460000, 480000],
        'material_costs': [200000, 205000, 210000, 215000, 220000],
        'subcontractor_costs': [150000, 152000, 155000, 158000, 160000]
    })
    
    return data


if __name__ == "__main__":
    # テスト実行
    calculator = EvaluationMetricsCalculator()
    sample_data = create_sample_financial_data()
    sample_company = create_sample_company_info()
    
    # 全評価項目計算のテスト
    results = calculator.calculate_all_metrics(sample_data, sample_company)
    print("計算された評価項目:")
    print(results.head())
    
    # 特定評価項目のテスト
    roe_values = calculator.calculate_single_metric('roe', sample_data, sample_company)
    print(f"\nROE値: {roe_values.values}")