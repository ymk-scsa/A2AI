"""
A2AI (Advanced Financial Analysis AI) - Factor Metrics Calculator
企業ライフサイクル全体を考慮した拡張要因項目計算クラス

このモジュールは、9つの評価項目それぞれに対する23の要因項目を計算します。
- 従来の20項目 + 3つの拡張項目（企業年齢、市場参入時期、親会社依存度）
- 生存バイアス対応、企業消滅・新設企業も含む分析対象
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple
import logging
from abc import ABC, abstractmethod

class FactorMetricsCalculator(ABC):
    """要因項目計算の基底クラス"""
    
    def __init__(self, financial_data: pd.DataFrame, market_data: pd.DataFrame = None):
        """
        Parameters:
        -----------
        financial_data : pd.DataFrame
            財務諸表データ（EDINET等から取得）
        market_data : pd.DataFrame, optional
            市場・業界データ（設立年、市場参入時期、親会社情報等）
        """
        self.financial_data = financial_data.copy()
        self.market_data = market_data.copy() if market_data is not None else pd.DataFrame()
        self.logger = logging.getLogger(__name__)
        
        # データ前処理
        self._preprocess_data()
        
    def _preprocess_data(self):
        """データの前処理"""
        # 日付列の処理
        if 'date' in self.financial_data.columns:
            self.financial_data['date'] = pd.to_datetime(self.financial_data['date'])
        
        # 企業識別子の統一
        if 'company_id' not in self.financial_data.columns and 'company_name' in self.financial_data.columns:
            self.financial_data['company_id'] = self.financial_data['company_name']
            
        # 欠損値の初期処理
        self.financial_data = self.financial_data.fillna(0)
        
    def calculate_all_factors(self, evaluation_target: str) -> pd.DataFrame:
        """指定された評価項目に対するすべての要因項目を計算"""
        method_name = f"calculate_{evaluation_target}_factors"
        if hasattr(self, method_name):
            return getattr(self, method_name)()
        else:
            raise ValueError(f"評価項目 {evaluation_target} に対応する計算メソッドが見つかりません")


class SalesFactorCalculator(FactorMetricsCalculator):
    """売上高の要因項目計算クラス（23項目）"""
    
    def calculate_sales_factors(self) -> pd.DataFrame:
        """売上高の23要因項目を計算"""
        factors = pd.DataFrame()
        factors['company_id'] = self.financial_data['company_id']
        factors['date'] = self.financial_data['date']
        
        # 投資・資産関連（5項目）
        factors['tangible_fixed_assets'] = self.financial_data.get('tangible_fixed_assets', 0)
        factors['capital_investment'] = self.financial_data.get('capital_expenditure', 0)
        factors['rd_expenses'] = self.financial_data.get('research_development_expenses', 0)
        factors['intangible_assets'] = self.financial_data.get('intangible_fixed_assets', 0)
        factors['investment_securities'] = self.financial_data.get('investment_securities', 0)
        
        # 総還元性向の計算
        dividends = self.financial_data.get('dividends_paid', 0)
        share_buybacks = self.financial_data.get('treasury_stock_acquired', 0)
        net_income = self.financial_data.get('net_income', 1)  # ゼロ除算回避
        factors['total_return_ratio'] = (dividends + share_buybacks) / net_income.replace(0, np.nan)
        
        # 人的資源関連（4項目）
        factors['employee_count'] = self.financial_data.get('number_of_employees', 0)
        factors['average_annual_salary'] = self.financial_data.get('average_annual_salary', 0)
        factors['retirement_benefit_expenses'] = self.financial_data.get('retirement_benefit_expenses', 0)
        factors['welfare_expenses'] = self.financial_data.get('welfare_expenses', 0)
        
        # 運転資本・効率性関連（5項目）
        factors['accounts_receivable'] = self.financial_data.get('notes_accounts_receivable', 0)
        factors['inventory'] = self.financial_data.get('inventories', 0)
        factors['total_assets'] = self.financial_data.get('total_assets', 1)
        
        # 回転率の計算
        sales = self.financial_data.get('net_sales', 1)
        factors['receivables_turnover'] = sales / factors['accounts_receivable'].replace(0, np.nan)
        factors['inventory_turnover'] = sales / factors['inventory'].replace(0, np.nan)
        
        # 事業展開関連（5項目）
        factors['overseas_sales_ratio'] = self.financial_data.get('overseas_sales_ratio', 0)
        factors['business_segment_count'] = self.financial_data.get('number_of_segments', 1)
        factors['selling_admin_expenses'] = self.financial_data.get('selling_general_admin_expenses', 0)
        factors['advertising_expenses'] = self.financial_data.get('advertising_expenses', 0)
        factors['non_operating_income'] = self.financial_data.get('non_operating_income', 0)
        
        # 受注残高（注記情報から）
        factors['order_backlog'] = self.financial_data.get('order_backlog', 0)
        
        # 拡張項目（3項目）
        factors = self._add_extended_factors(factors)
        
        return factors
    
    def _add_extended_factors(self, factors: pd.DataFrame) -> pd.DataFrame:
        """拡張要因項目（企業年齢、市場参入時期、親会社依存度）を追加"""
        # 企業年齢の計算
        if 'establishment_date' in self.market_data.columns:
            establishment_dates = self.market_data.set_index('company_id')['establishment_date']
            factors['company_age'] = factors.apply(
                lambda row: self._calculate_company_age(
                    row['company_id'], 
                    row['date'], 
                    establishment_dates
                ), axis=1
            )
        else:
            factors['company_age'] = 0
        
        # 市場参入時期（先発/後発効果）
        if 'market_entry_date' in self.market_data.columns:
            entry_dates = self.market_data.set_index('company_id')['market_entry_date']
            factors['market_entry_timing'] = factors.apply(
                lambda row: self._calculate_market_entry_timing(
                    row['company_id'],
                    entry_dates
                ), axis=1
            )
        else:
            factors['market_entry_timing'] = 0
        
        # 親会社依存度（分社企業の場合）
        if 'parent_company_sales' in self.financial_data.columns:
            total_sales = self.financial_data.get('net_sales', 1)
            parent_sales = self.financial_data.get('parent_company_sales', 0)
            factors['parent_dependency_ratio'] = parent_sales / total_sales.replace(0, np.nan)
        else:
            factors['parent_dependency_ratio'] = 0
        
        return factors
    
    def _calculate_company_age(self, company_id: str, current_date: pd.Timestamp, 
                              establishment_dates: pd.Series) -> float:
        """企業年齢を計算（年単位）"""
        try:
            if company_id in establishment_dates.index:
                establishment_date = pd.to_datetime(establishment_dates[company_id])
                age = (current_date - establishment_date).days / 365.25
                return max(0, age)
        except Exception as e:
            self.logger.warning(f"企業年齢計算エラー {company_id}: {e}")
        return 0
    
    def _calculate_market_entry_timing(self, company_id: str, entry_dates: pd.Series) -> int:
        """市場参入時期を計算（0=後発、1=先発）"""
        try:
            if company_id in entry_dates.index and len(entry_dates.dropna()) > 1:
                entry_date = pd.to_datetime(entry_dates[company_id])
                median_entry = entry_dates.dropna().median()
                return 1 if entry_date <= median_entry else 0
        except Exception as e:
            self.logger.warning(f"市場参入時期計算エラー {company_id}: {e}")
        return 0


class SalesGrowthFactorCalculator(FactorMetricsCalculator):
    """売上高成長率の要因項目計算クラス（23項目）"""
    
    def calculate_sales_growth_factors(self) -> pd.DataFrame:
        """売上高成長率の23要因項目を計算"""
        factors = pd.DataFrame()
        factors['company_id'] = self.financial_data['company_id']
        factors['date'] = self.financial_data['date']
        
        # 前年データを結合
        prev_data = self._get_previous_year_data()
        
        # 投資・拡張関連（6項目）
        factors['capex_growth_rate'] = self._calculate_growth_rate(
            self.financial_data.get('capital_expenditure', 0),
            prev_data.get('capital_expenditure', 0)
        )
        factors['rd_growth_rate'] = self._calculate_growth_rate(
            self.financial_data.get('research_development_expenses', 0),
            prev_data.get('research_development_expenses', 0)
        )
        factors['tangible_assets_growth'] = self._calculate_growth_rate(
            self.financial_data.get('tangible_fixed_assets', 0),
            prev_data.get('tangible_fixed_assets', 0)
        )
        factors['intangible_assets_growth'] = self._calculate_growth_rate(
            self.financial_data.get('intangible_fixed_assets', 0),
            prev_data.get('intangible_fixed_assets', 0)
        )
        factors['total_assets_growth'] = self._calculate_growth_rate(
            self.financial_data.get('total_assets', 0),
            prev_data.get('total_assets', 0)
        )
        
        # のれんの増加率・減損
        factors['goodwill_growth'] = self._calculate_growth_rate(
            self.financial_data.get('goodwill', 0),
            prev_data.get('goodwill', 0)
        )
        
        # 人的資源拡張（4項目）
        factors['employee_growth_rate'] = self._calculate_growth_rate(
            self.financial_data.get('number_of_employees', 0),
            prev_data.get('number_of_employees', 0)
        )
        factors['salary_growth_rate'] = self._calculate_growth_rate(
            self.financial_data.get('average_annual_salary', 0),
            prev_data.get('average_annual_salary', 0)
        )
        factors['personnel_cost_growth'] = self._calculate_growth_rate(
            self.financial_data.get('personnel_expenses', 0),
            prev_data.get('personnel_expenses', 0)
        )
        factors['retirement_cost_growth'] = self._calculate_growth_rate(
            self.financial_data.get('retirement_benefit_expenses', 0),
            prev_data.get('retirement_benefit_expenses', 0)
        )
        
        # 市場・事業拡大（5項目）
        factors['overseas_ratio_change'] = (
            self.financial_data.get('overseas_sales_ratio', 0) - 
            prev_data.get('overseas_sales_ratio', 0)
        )
        factors['segment_sales_growth'] = self._calculate_segment_growth()
        factors['sga_growth_rate'] = self._calculate_growth_rate(
            self.financial_data.get('selling_general_admin_expenses', 0),
            prev_data.get('selling_general_admin_expenses', 0)
        )
        factors['advertising_growth_rate'] = self._calculate_growth_rate(
            self.financial_data.get('advertising_expenses', 0),
            prev_data.get('advertising_expenses', 0)
        )
        factors['non_operating_income_growth'] = self._calculate_growth_rate(
            self.financial_data.get('non_operating_income', 0),
            prev_data.get('non_operating_income', 0)
        )
        
        # 効率性・能力関連（5項目）
        factors['receivables_growth_rate'] = self._calculate_growth_rate(
            self.financial_data.get('notes_accounts_receivable', 0),
            prev_data.get('notes_accounts_receivable', 0)
        )
        factors['inventory_growth_rate'] = self._calculate_growth_rate(
            self.financial_data.get('inventories', 0),
            prev_data.get('inventories', 0)
        )
        
        # 回転率変化の計算
        factors['receivables_turnover_change'] = self._calculate_turnover_change('receivables')
        factors['inventory_turnover_change'] = self._calculate_turnover_change('inventory')
        factors['total_asset_turnover_change'] = self._calculate_turnover_change('total_assets')
        
        # 受注残高増加率
        factors['order_backlog_growth'] = self._calculate_growth_rate(
            self.financial_data.get('order_backlog', 0),
            prev_data.get('order_backlog', 0)
        )
        
        # 拡張項目（3項目）
        factors = self._add_extended_factors(factors)
        
        return factors
    
    def _get_previous_year_data(self) -> pd.DataFrame:
        """前年同期のデータを取得"""
        prev_data = self.financial_data.copy()
        prev_data['date'] = prev_data['date'] + timedelta(days=365)
        return prev_data.set_index(['company_id', 'date']).reindex(
            self.financial_data.set_index(['company_id', 'date']).index
        ).fillna(0)
    
    def _calculate_growth_rate(self, current: pd.Series, previous: pd.Series) -> pd.Series:
        """成長率を計算"""
        return (current - previous) / previous.replace(0, np.nan)
    
    def _calculate_segment_growth(self) -> pd.Series:
        """セグメント別売上高増加率の加重平均を計算"""
        # 簡略化：全体売上高増加率で代用
        current_sales = self.financial_data.get('net_sales', 0)
        prev_sales = self._get_previous_year_data().get('net_sales', 0)
        return self._calculate_growth_rate(current_sales, prev_sales)
    
    def _calculate_turnover_change(self, asset_type: str) -> pd.Series:
        """回転率変化を計算"""
        if asset_type == 'receivables':
            asset_col = 'notes_accounts_receivable'
        elif asset_type == 'inventory':
            asset_col = 'inventories'
        elif asset_type == 'total_assets':
            asset_col = 'total_assets'
        else:
            return pd.Series(0, index=self.financial_data.index)
        
        sales = self.financial_data.get('net_sales', 1)
        assets = self.financial_data.get(asset_col, 1)
        current_turnover = sales / assets.replace(0, np.nan)
        
        prev_data = self._get_previous_year_data()
        prev_sales = prev_data.get('net_sales', 1)
        prev_assets = prev_data.get(asset_col, 1)
        prev_turnover = prev_sales / prev_assets.replace(0, np.nan)
        
        return current_turnover - prev_turnover


class OperatingMarginFactorCalculator(FactorMetricsCalculator):
    """売上高営業利益率の要因項目計算クラス（23項目）"""
    
    def calculate_operating_margin_factors(self) -> pd.DataFrame:
        """売上高営業利益率の23要因項目を計算"""
        factors = pd.DataFrame()
        factors['company_id'] = self.financial_data['company_id']
        factors['date'] = self.financial_data['date']
        
        sales = self.financial_data.get('net_sales', 1)
        
        # 売上原価構成（5項目）
        manufacturing_cost = self.financial_data.get('cost_of_sales', 0)
        factors['material_cost_ratio'] = self.financial_data.get('material_costs', 0) / sales
        factors['labor_cost_ratio'] = self.financial_data.get('labor_costs', 0) / sales
        factors['overhead_cost_ratio'] = self.financial_data.get('overhead_costs', 0) / sales
        factors['outsourcing_cost_ratio'] = self.financial_data.get('outsourcing_costs', 0) / sales
        factors['manufacturing_depreciation_ratio'] = self.financial_data.get('manufacturing_depreciation', 0) / sales
        
        # 販管費構成（5項目）
        sga_expenses = self.financial_data.get('selling_general_admin_expenses', 0)
        factors['sga_ratio'] = sga_expenses / sales
        factors['sga_personnel_ratio'] = self.financial_data.get('sga_personnel_expenses', 0) / sales
        factors['advertising_ratio'] = self.financial_data.get('advertising_expenses', 0) / sales
        factors['rd_ratio'] = self.financial_data.get('research_development_expenses', 0) / sales
        factors['sga_depreciation_ratio'] = self.financial_data.get('sga_depreciation', 0) / sales
        
        # 効率性指標（5項目）
        value_added = sales - manufacturing_cost
        factors['value_added_ratio'] = value_added / sales
        factors['labor_productivity'] = sales / self.financial_data.get('number_of_employees', 1)
        factors['asset_efficiency'] = sales / self.financial_data.get('tangible_fixed_assets', 1)
        factors['total_asset_turnover'] = sales / self.financial_data.get('total_assets', 1)
        factors['inventory_turnover'] = sales / self.financial_data.get('inventories', 1)
        
        # 規模・構造要因（5項目）
        factors['sales_scale'] = sales  # 規模効果
        total_costs = manufacturing_cost + sga_expenses
        factors['fixed_cost_ratio'] = self.financial_data.get('fixed_costs', total_costs * 0.3) / sales  # 推定
        factors['variable_cost_ratio'] = self.financial_data.get('variable_costs', total_costs * 0.7) / sales  # 推定
        factors['overseas_sales_ratio'] = self.financial_data.get('overseas_sales_ratio', 0)
        factors['business_concentration'] = self._calculate_business_concentration()
        
        # 拡張項目（3項目）
        factors = self._add_extended_factors(factors)
        
        return factors
    
    def _calculate_business_concentration(self) -> pd.Series:
        """事業セグメント集中度を計算（ハーフィンダール指数）"""
        # 簡略化：セグメント数の逆数で近似
        segment_count = self.financial_data.get('number_of_segments', 1)
        return 1 / segment_count


class NetMarginFactorCalculator(FactorMetricsCalculator):
    """売上高当期純利益率の要因項目計算クラス（23項目）"""
    
    def calculate_net_margin_factors(self) -> pd.DataFrame:
        """売上高当期純利益率の23要因項目を計算"""
        factors = pd.DataFrame()
        factors['company_id'] = self.financial_data['company_id']
        factors['date'] = self.financial_data['date']
        
        sales = self.financial_data.get('net_sales', 1)
        
        # 営業損益要因（5項目）
        operating_income = self.financial_data.get('operating_income', 0)
        factors['operating_margin'] = operating_income / sales
        factors['sga_ratio'] = self.financial_data.get('selling_general_admin_expenses', 0) / sales
        factors['cogs_ratio'] = self.financial_data.get('cost_of_sales', 0) / sales
        factors['rd_ratio'] = self.financial_data.get('research_development_expenses', 0) / sales
        factors['depreciation_ratio'] = self.financial_data.get('depreciation_amortization', 0) / sales
        
        # 営業外損益（5項目）
        factors['interest_income_ratio'] = self.financial_data.get('interest_income', 0) / sales
        factors['interest_expense_ratio'] = self.financial_data.get('interest_expense', 0) / sales
        factors['fx_gain_loss_ratio'] = self.financial_data.get('foreign_exchange_gain_loss', 0) / sales
        factors['equity_income_ratio'] = self.financial_data.get('equity_in_earnings', 0) / sales
        factors['non_operating_income_ratio'] = self.financial_data.get('non_operating_income', 0) / sales
        
        # 特別損益・税金（5項目）
        factors['extraordinary_gain_ratio'] = self.financial_data.get('extraordinary_gains', 0) / sales
        factors['extraordinary_loss_ratio'] = self.financial_data.get('extraordinary_losses', 0) / sales
        
        pretax_income = self.financial_data.get('income_before_taxes', 1)
        tax_expense = self.financial_data.get('income_tax_expense', 0)
        factors['effective_tax_rate'] = tax_expense / pretax_income.replace(0, np.nan)
        factors['tax_adjustment_ratio'] = self.financial_data.get('deferred_tax_adjustments', 0) / sales
        factors['pretax_margin'] = pretax_income / sales
        
        # 財務構造要因（5項目）
        total_assets = self.financial_data.get('total_assets', 1)
        interest_bearing_debt = self.financial_data.get('interest_bearing_debt', 0)
        equity = self.financial_data.get('total_equity', 1)
        
        factors['debt_ratio'] = interest_bearing_debt / total_assets
        factors['equity_ratio'] = equity / total_assets
        factors['investment_securities_gain_loss'] = self.financial_data.get('securities_valuation_gain_loss', 0) / sales
        factors['asset_disposal_gain_loss'] = self.financial_data.get('gain_loss_on_disposal_of_assets', 0) / sales
        factors['impairment_loss_ratio'] = self.financial_data.get('impairment_losses', 0) / sales
        
        # 拡張項目（3項目）
        factors = self._add_extended_factors(factors)
        
        return factors


class ROEFactorCalculator(FactorMetricsCalculator):
    """ROEの要因項目計算クラス（23項目）"""
    
    def calculate_roe_factors(self) -> pd.DataFrame:
        """ROEの23要因項目を計算"""
        factors = pd.DataFrame()
        factors['company_id'] = self.financial_data['company_id']
        factors['date'] = self.financial_data['date']
        
        sales = self.financial_data.get('net_sales', 1)
        total_assets = self.financial_data.get('total_assets', 1)
        equity = self.financial_data.get('total_equity', 1)
        net_income = self.financial_data.get('net_income', 0)
        
        # 収益性要因（ROA構成）（5項目）
        factors['net_margin'] = net_income / sales
        factors['total_asset_turnover'] = sales / total_assets
        factors['operating_margin'] = self.financial_data.get('operating_income', 0) / sales
        factors['cogs_ratio'] = self.financial_data.get('cost_of_sales', 0) / sales
        factors['sga_ratio'] = self.financial_data.get('selling_general_admin_expenses', 0) / sales
        
        # 財務レバレッジ（5項目）
        factors['equity_ratio'] = equity / total_assets
        factors['asset_equity_multiplier'] = total_assets / equity.replace(0, np.nan)
        factors['debt_equity_ratio'] = self.financial_data.get('interest_bearing_debt', 0) / equity.replace(0, np.nan)
        factors['current_ratio'] = (self.financial_data.get('current_assets', 0) / 
                                   self.financial_data.get('current_liabilities', 1))
        factors['fixed_ratio'] = (self.financial_data.get('fixed_assets', 0) / equity.replace(0, np.nan))
        
        # 資産効率性（5項目）
        factors['receivables_turnover'] = sales / self.financial_data.get('notes_accounts_receivable', 1)
        factors['inventory_turnover'] = sales / self.financial_data.get('inventories', 1)
        factors['fixed_asset_turnover'] = sales / self.financial_data.get('tangible_fixed_assets', 1)
        factors['cash_asset_ratio'] = (self.financial_data.get('cash_and_deposits', 0) / total_assets)
        factors['investment_securities_ratio'] = (self.financial_data.get('investment_securities', 0) / total_assets)
        
        # 収益・配当政策（5項目）
        factors['dividend_payout_ratio'] = (self.financial_data.get('dividends_paid', 0) / 
                                          net_income.replace(0, np.nan))
        factors['retained_earnings_ratio'] = ((net_income - self.financial_data.get('dividends_paid', 0)) / 
                                            net_income.replace(0, np.nan))
        factors['non_operating_income_ratio'] = self.financial_data.get('non_operating_income', 0) / sales
        factors['extraordinary_net_ratio'] = ((self.financial_data.get('extraordinary_gains', 0) - 
                                             self.financial_data.get('extraordinary_losses', 0)) / 
                                            net_income.replace(0, np.nan))
        factors['effective_tax_rate'] = (self.financial_data.get('income_tax_expense', 0) / 
                                       self.financial_data.get('income_before_taxes', 1))
        
        # 総還元性向
        share_buybacks = self.financial_data.get('treasury_stock_acquired', 0)
        dividends = self.financial_data.get('dividends_paid', 0)
        factors['total_return_ratio'] = ((dividends + share_buybacks) / net_income.replace(0, np.nan))
        
        # 拡張項目（3項目）
        factors = self._add_extended_factors(factors)
        
        return factors


class ValueAddedRatioFactorCalculator(FactorMetricsCalculator):
    """売上高付加価値率の要因項目計算クラス（23項目）"""
    
    def calculate_value_added_ratio_factors(self) -> pd.DataFrame:
        """売上高付加価値率の23要因項目を計算"""
        factors = pd.DataFrame()
        factors['company_id'] = self.financial_data['company_id']
        factors['date'] = self.financial_data['date']
        
        sales = self.financial_data.get('net_sales', 1)
        
        # 技術・研究開発（5項目）
        factors['rd_ratio'] = self.financial_data.get('research_development_expenses', 0) / sales
        factors['intangible_asset_ratio'] = self.financial_data.get('intangible_fixed_assets', 0) / sales
        factors['patent_expenses'] = self.financial_data.get('patent_related_expenses', 0) / sales
        factors['software_ratio'] = self.financial_data.get('software_assets', 0) / sales
        factors['technology_license_income'] = self.financial_data.get('technology_license_income', 0) / sales
        
        # 人的付加価値（5項目）
        industry_avg_salary = self.financial_data.get('industry_average_salary', 1)
        company_salary = self.financial_data.get('average_annual_salary', 1)
        factors['salary_premium_ratio'] = company_salary / industry_avg_salary
        factors['personnel_cost_ratio'] = self.financial_data.get('personnel_expenses', 0) / sales
        factors['employee_per_sales'] = (self.financial_data.get('number_of_employees', 0) / sales) * 1000000  # 百万円あたり
        factors['retirement_cost_ratio'] = self.financial_data.get('retirement_benefit_expenses', 0) / sales
        factors['welfare_cost_ratio'] = self.financial_data.get('welfare_expenses', 0) / sales
        
        # コスト構造・効率性（5項目）（逆数で付加価値への貢献度を計算）
        cost_of_sales = self.financial_data.get('cost_of_sales', 1)
        factors['inverse_cogs_ratio'] = 1 - (cost_of_sales / sales)  # 売上原価率の逆数効果
        factors['inverse_material_ratio'] = 1 - (self.financial_data.get('material_costs', 0) / sales)
        factors['inverse_outsourcing_ratio'] = 1 - (self.financial_data.get('outsourcing_costs', 0) / sales)
        factors['labor_productivity'] = sales / self.financial_data.get('number_of_employees', 1)
        factors['asset_productivity'] = sales / self.financial_data.get('tangible_fixed_assets', 1)
        
        # 事業構造・差別化（5項目）
        factors['overseas_sales_ratio'] = self.financial_data.get('overseas_sales_ratio', 0)
        factors['high_value_segment_ratio'] = self.financial_data.get('high_value_business_ratio', 0)
        factors['service_revenue_ratio'] = self.financial_data.get('service_maintenance_revenue', 0) / sales
        factors['operating_margin'] = self.financial_data.get('operating_income', 0) / sales  # 付加価値実現度
        factors['brand_intangible_ratio'] = self.financial_data.get('brand_trademark_assets', 0) / sales
        
        # 拡張項目（3項目）
        factors = self._add_extended_factors(factors)
        
        return factors


class SurvivalProbabilityFactorCalculator(FactorMetricsCalculator):
    """企業存続確率の要因項目計算クラス（23項目）- A2AI拡張評価項目"""
    
    def calculate_survival_probability_factors(self) -> pd.DataFrame:
        """企業存続確率の23要因項目を計算"""
        factors = pd.DataFrame()
        factors['company_id'] = self.financial_data['company_id']
        factors['date'] = self.financial_data['date']
        
        # 財務健全性指標（5項目）
        total_assets = self.financial_data.get('total_assets', 1)
        current_assets = self.financial_data.get('current_assets', 0)
        current_liabilities = self.financial_data.get('current_liabilities', 1)
        equity = self.financial_data.get('total_equity', 1)
        interest_bearing_debt = self.financial_data.get('interest_bearing_debt', 0)
        
        factors['current_ratio'] = current_assets / current_liabilities
        factors['equity_ratio'] = equity / total_assets
        factors['debt_ratio'] = interest_bearing_debt / total_assets
        factors['interest_coverage'] = (self.financial_data.get('operating_income', 0) / 
                                       self.financial_data.get('interest_expense', 1))
        factors['cash_ratio'] = self.financial_data.get('cash_and_deposits', 0) / current_liabilities
        
        # 収益性・成長性指標（5項目）
        sales = self.financial_data.get('net_sales', 1)
        net_income = self.financial_data.get('net_income', 0)
        factors['roa'] = net_income / total_assets
        factors['operating_margin'] = self.financial_data.get('operating_income', 0) / sales
        factors['sales_growth'] = self._calculate_sales_growth()
        factors['profit_stability'] = self._calculate_profit_volatility()  # 利益の安定性
        factors['market_share_trend'] = self._calculate_market_share_trend()
        
        # 効率性・競争力（5項目）
        factors['asset_turnover'] = sales / total_assets
        factors['inventory_turnover'] = sales / self.financial_data.get('inventories', 1)
        factors['rd_intensity'] = self.financial_data.get('research_development_expenses', 0) / sales
        factors['capex_intensity'] = self.financial_data.get('capital_expenditure', 0) / sales
        factors['employee_productivity'] = sales / self.financial_data.get('number_of_employees', 1)
        
        # 事業基盤・ガバナンス（5項目）
        factors['business_diversification'] = self.financial_data.get('number_of_segments', 1)
        factors['overseas_exposure'] = self.financial_data.get('overseas_sales_ratio', 0)
        factors['customer_concentration'] = self.financial_data.get('major_customer_dependency', 0)
        factors['supplier_diversity'] = 1 - self.financial_data.get('major_supplier_dependency', 0)
        factors['corporate_governance_score'] = self.financial_data.get('governance_score', 0.5)
        
        # 市場・業界要因（5項目）
        factors['market_growth_rate'] = self.financial_data.get('market_growth_rate', 0)
        factors['industry_concentration'] = self.financial_data.get('industry_hhi', 0.5)
        factors['regulatory_risk'] = self.financial_data.get('regulatory_risk_score', 0.5)
        factors['technology_disruption_risk'] = self.financial_data.get('tech_disruption_score', 0.5)
        factors['economic_sensitivity'] = self._calculate_economic_sensitivity()
        
        # 拡張項目（3項目）
        factors = self._add_extended_factors(factors)
        
        return factors
    
    def _calculate_sales_growth(self) -> pd.Series:
        """売上高成長率を計算"""
        current_sales = self.financial_data.get('net_sales', 0)
        prev_data = self._get_previous_year_data()
        prev_sales = prev_data.get('net_sales', 0)
        return (current_sales - prev_sales) / prev_sales.replace(0, np.nan)
    
    def _calculate_profit_volatility(self) -> pd.Series:
        """利益の変動性を計算（過去3年間の標準偏差）"""
        # 簡略化：現在の営業利益率で代用
        return 1 / (1 + abs(self.financial_data.get('operating_income', 0) / 
                           self.financial_data.get('net_sales', 1)))
    
    def _calculate_market_share_trend(self) -> pd.Series:
        """市場シェアトレンドを計算"""
        return self.financial_data.get('market_share_change', 0)
    
    def _calculate_economic_sensitivity(self) -> pd.Series:
        """経済感応度を計算"""
        return self.financial_data.get('beta_coefficient', 1.0)
    
    def _get_previous_year_data(self) -> pd.DataFrame:
        """前年データを取得（簡略化）"""
        return self.financial_data.copy()


class EmergenceSuccessFactorCalculator(FactorMetricsCalculator):
    """新規事業成功率の要因項目計算クラス（23項目）- A2AI拡張評価項目"""
    
    def calculate_emergence_success_factors(self) -> pd.DataFrame:
        """新規事業成功率の23要因項目を計算"""
        factors = pd.DataFrame()
        factors['company_id'] = self.financial_data['company_id']
        factors['date'] = self.financial_data['date']
        
        sales = self.financial_data.get('net_sales', 1)
        
        # イノベーション・技術要因（5項目）
        factors['rd_intensity'] = self.financial_data.get('research_development_expenses', 0) / sales
        factors['patent_portfolio'] = self.financial_data.get('patent_count', 0)
        factors['technology_investment'] = self.financial_data.get('technology_investment', 0) / sales
        factors['digital_transformation_score'] = self.financial_data.get('dx_score', 0)
        factors['innovation_pipeline'] = self.financial_data.get('new_product_ratio', 0)
        
        # 人的資本・組織要因（5項目）
        factors['talent_density'] = self.financial_data.get('phd_engineer_ratio', 0)
        factors['employee_engagement'] = self.financial_data.get('engagement_score', 0.5)
        factors['leadership_experience'] = self.financial_data.get('management_experience_years', 0)
        factors['organizational_agility'] = self.financial_data.get('agility_score', 0.5)
        factors['learning_investment'] = self.financial_data.get('training_expenses', 0) / sales
        
        # 市場・顧客要因（5項目）
        factors['market_timing'] = self.financial_data.get('market_timing_score', 0.5)
        factors['customer_validation'] = self.financial_data.get('customer_validation_score', 0.5)
        factors['market_size_potential'] = self.financial_data.get('addressable_market_size', 0)
        factors['competitive_differentiation'] = self.financial_data.get('differentiation_score', 0.5)
        factors['network_effects'] = self.financial_data.get('network_effect_score', 0)
        
        # 資源・戦略要因（5項目）
        factors['financial_resources'] = self.financial_data.get('cash_and_deposits', 0) / sales
        factors['strategic_partnerships'] = self.financial_data.get('partnership_count', 0)
        factors['ecosystem_positioning'] = self.financial_data.get('ecosystem_score', 0.5)
        factors['scalability_potential'] = self.financial_data.get('scalability_score', 0.5)
        factors['execution_capability'] = self.financial_data.get('execution_score', 0.5)
        
        # リスク・環境要因（5項目）
        factors['regulatory_support'] = self.financial_data.get('regulatory_support_score', 0.5)
        factors['macro_environment'] = self.financial_data.get('macro_environment_score', 0.5)
        factors['technology_maturity'] = self.financial_data.get('tech_maturity_score', 0.5)
        factors['funding_accessibility'] = self.financial_data.get('funding_access_score', 0.5)
        factors['risk_tolerance'] = self.financial_data.get('risk_tolerance_score', 0.5)
        
        # 拡張項目（3項目）
        factors = self._add_extended_factors(factors)
        
        return factors


class SuccessionSuccessFactorCalculator(FactorMetricsCalculator):
    """事業継承成功度の要因項目計算クラス（23項目）- A2AI拡張評価項目"""
    
    def calculate_succession_success_factors(self) -> pd.DataFrame:
        """事業継承成功度の23要因項目を計算"""
        factors = pd.DataFrame()
        factors['company_id'] = self.financial_data['company_id']
        factors['date'] = self.financial_data['date']
        
        # 統合前後の財務指標比較（5項目）
        factors['revenue_synergy'] = self.financial_data.get('post_merger_revenue_growth', 0)
        factors['cost_synergy'] = self.financial_data.get('cost_reduction_achieved', 0)
        factors['profitability_improvement'] = self.financial_data.get('margin_improvement', 0)
        factors['asset_utilization_gain'] = self.financial_data.get('asset_efficiency_gain', 0)
        factors['market_share_expansion'] = self.financial_data.get('market_share_gain', 0)
        
        # 組織統合・文化要因（5項目）
        factors['cultural_compatibility'] = self.financial_data.get('cultural_fit_score', 0.5)
        factors['talent_retention'] = self.financial_data.get('key_talent_retention_rate', 0.5)
        factors['integration_speed'] = self.financial_data.get('integration_completion_rate', 0)
        factors['communication_effectiveness'] = self.financial_data.get('communication_score', 0.5)
        factors['change_management'] = self.financial_data.get('change_management_score', 0.5)
        
        # 戦略・事業要因（5項目）
        factors['strategic_fit'] = self.financial_data.get('strategic_alignment_score', 0.5)
        factors['complementary_capabilities'] = self.financial_data.get('capability_complementarity', 0.5)
        factors['market_position_strength'] = self.financial_data.get('combined_market_position', 0.5)
        factors['innovation_acceleration'] = self.financial_data.get('innovation_acceleration_score', 0)
        factors['geographic_expansion'] = self.financial_data.get('geographic_coverage_gain', 0)
        
        # リスク・ガバナンス要因（5項目）
        factors['integration_risk_management'] = self.financial_data.get('integration_risk_score', 0.5)
        factors['regulatory_approval_smooth'] = self.financial_data.get('regulatory_smoothness', 0.5)
        factors['stakeholder_support'] = self.financial_data.get('stakeholder_support_score', 0.5)
        factors['governance_structure'] = self.financial_data.get('governance_effectiveness', 0.5)
        factors['transparency_level'] = self.financial_data.get('transparency_score', 0.5)
        
        # 価値創造・成果要因（5項目）
        factors['valuation_realization'] = self.financial_data.get('valuation_target_achievement', 0)
        factors['customer_satisfaction'] = self.financial_data.get('customer_satisfaction_score', 0.5)
        factors['employee_satisfaction'] = self.financial_data.get('employee_satisfaction_score', 0.5)
        factors['innovation_output'] = self.financial_data.get('new_product_development_rate', 0)
        factors['long_term_sustainability'] = self.financial_data.get('sustainability_score', 0.5)
        
        # 拡張項目（3項目）
        factors = self._add_extended_factors(factors)
        
        return factors


class FactorMetricsManager:
    """要因項目計算の統合管理クラス"""
    
    def __init__(self, financial_data: pd.DataFrame, market_data: pd.DataFrame = None):
        self.financial_data = financial_data
        self.market_data = market_data
        
        # 各評価項目の計算器を初期化
        self.calculators = {
            'sales': SalesFactorCalculator(financial_data, market_data),
            'sales_growth': SalesGrowthFactorCalculator(financial_data, market_data),
            'operating_margin': OperatingMarginFactorCalculator(financial_data, market_data),
            'net_margin': NetMarginFactorCalculator(financial_data, market_data),
            'roe': ROEFactorCalculator(financial_data, market_data),
            'value_added_ratio': ValueAddedRatioFactorCalculator(financial_data, market_data),
            'survival_probability': SurvivalProbabilityFactorCalculator(financial_data, market_data),
            'emergence_success': EmergenceSuccessFactorCalculator(financial_data, market_data),
            'succession_success': SuccessionSuccessFactorCalculator(financial_data, market_data)
        }
    
    def calculate_all_evaluation_factors(self) -> Dict[str, pd.DataFrame]:
        """すべての評価項目に対する要因項目を計算"""
        results = {}
        
        for evaluation_name, calculator in self.calculators.items():
            try:
                results[evaluation_name] = calculator.calculate_all_factors(evaluation_name)
                print(f"✓ {evaluation_name} の要因項目計算完了")
            except Exception as e:
                print(f"✗ {evaluation_name} の要因項目計算エラー: {e}")
                results[evaluation_name] = pd.DataFrame()
        
        return results
    
    def calculate_specific_factors(self, evaluation_targets: List[str]) -> Dict[str, pd.DataFrame]:
        """指定された評価項目の要因項目のみを計算"""
        results = {}
        
        for target in evaluation_targets:
            if target in self.calculators:
                try:
                    results[target] = self.calculators[target].calculate_all_factors(target)
                    print(f"✓ {target} の要因項目計算完了")
                except Exception as e:
                    print(f"✗ {target} の要因項目計算エラー: {e}")
                    results[target] = pd.DataFrame()
            else:
                print(f"⚠ 評価項目 {target} は対応していません")
        
        return results
    
    def export_factor_definitions(self) -> pd.DataFrame:
        """すべての要因項目の定義一覧を出力"""
        definitions = []
        
        factor_definitions = {
            'sales': [
                '有形固定資産', '設備投資額', '研究開発費', '無形固定資産', '投資有価証券',
                '総還元性向', '従業員数', '平均年間給与', '退職給付費用', '福利厚生費',
                '売上債権', '棚卸資産', '総資産', '売上債権回転率', '棚卸資産回転率',
                '海外売上高比率', '事業セグメント数', '販管費', '広告宣伝費', '営業外収益',
                '受注残高', '企業年齢', '市場参入時期', '親会社依存度'
            ],
            # 他の評価項目の定義も同様に追加可能
        }
        
        for eval_name, factors in factor_definitions.items():
            for i, factor in enumerate(factors, 1):
                definitions.append({
                    'evaluation_item': eval_name,
                    'factor_number': i,
                    'factor_name': factor,
                    'category': 'traditional' if i <= 20 else 'extended'
                })
        
        return pd.DataFrame(definitions)


# 使用例とテスト用のサンプルコード
if __name__ == "__main__":
    # サンプルデータの作成
    sample_financial_data = pd.DataFrame({
        'company_id': ['A001', 'A002', 'A003'] * 3,
        'date': pd.to_datetime(['2022-01-01', '2022-01-01', '2022-01-01', 
                                '2023-01-01', '2023-01-01', '2023-01-01',
                                '2024-01-01', '2024-01-01', '2024-01-01']),
        'net_sales': [1000000, 2000000, 1500000, 1100000, 2200000, 1600000, 1200000, 2400000, 1700000],
        'operating_income': [100000, 200000, 150000, 110000, 220000, 160000, 120000, 240000, 170000],
        'net_income': [80000, 160000, 120000, 88000, 176000, 128000, 96000, 192000, 136000],
        'total_assets': [2000000, 4000000, 3000000, 2200000, 4400000, 3300000, 2400000, 4800000, 3600000],
        'total_equity': [1000000, 2000000, 1500000, 1100000, 2200000, 1650000, 1200000, 2400000, 1800000],
        'research_development_expenses': [50000, 100000, 75000, 55000, 110000, 82500, 60000, 120000, 90000],
        'number_of_employees': [1000, 2000, 1500, 1050, 2100, 1575, 1100, 2200, 1650]
    })
    
    sample_market_data = pd.DataFrame({
        'company_id': ['A001', 'A002', 'A003'],
        'establishment_date': pd.to_datetime(['1990-01-01', '1985-01-01', '1995-01-01']),
        'market_entry_date': pd.to_datetime(['1992-01-01', '1987-01-01', '1997-01-01'])
    })
    
    # Factor Metrics Manager の初期化
    manager = FactorMetricsManager(sample_financial_data, sample_market_data)
    
    # 特定の評価項目の要因項目を計算
    sales_factors = manager.calculate_specific_factors(['sales'])
    print("\n=== 売上高の要因項目計算結果 ===")
    if 'sales' in sales_factors:
        print(sales_factors['sales'].head())
    
    # すべての評価項目の要因項目を計算（デモ用）
    print("\n=== 全評価項目の要因項目計算開始 ===")
    all_factors = manager.calculate_all_evaluation_factors()
    
    # 要因項目定義の出力
    definitions = manager.export_factor_definitions()
    print("\n=== 要因項目定義一覧 ===")
    print(definitions.head(10))