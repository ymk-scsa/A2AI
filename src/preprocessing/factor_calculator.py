"""
A2AI Factor Calculator
財務諸表から9つの評価項目に対応する各23の要因項目を計算するモジュール

評価項目:
1. 売上高
2. 売上高成長率  
3. 売上高営業利益率
4. 売上高当期純利益率
5. ROE
6. 売上高付加価値率
7. 企業存続確率（新規）
8. 新規事業成功率（新規）
9. 事業継承成功度（新規）
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import warnings
from dataclasses import dataclass
import logging

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FinancialData:
    """財務データ構造体"""
    company_id: str
    year: int
    revenue: Optional[float] = None
    operating_profit: Optional[float] = None
    net_profit: Optional[float] = None
    total_assets: Optional[float] = None
    shareholders_equity: Optional[float] = None
    tangible_fixed_assets: Optional[float] = None
    intangible_fixed_assets: Optional[float] = None
    investment_securities: Optional[float] = None
    accounts_receivable: Optional[float] = None
    inventories: Optional[float] = None
    cash_and_deposits: Optional[float] = None
    cost_of_sales: Optional[float] = None
    sg_a_expenses: Optional[float] = None
    rd_expenses: Optional[float] = None
    advertising_expenses: Optional[float] = None
    capex: Optional[float] = None
    depreciation: Optional[float] = None
    employees: Optional[int] = None
    average_salary: Optional[float] = None
    overseas_revenue_ratio: Optional[float] = None
    segments_count: Optional[int] = None
    interest_income: Optional[float] = None
    interest_expenses: Optional[float] = None
    fx_gain_loss: Optional[float] = None
    extraordinary_profit: Optional[float] = None
    extraordinary_loss: Optional[float] = None
    tax_rate: Optional[float] = None
    dividend_payout_ratio: Optional[float] = None
    
class FactorCalculator:
    """要因項目計算クラス"""
    
    def __init__(self):
        self.risk_free_rate = 0.02  # リスクフリーレート（デフォルト2%）
        
    def calculate_all_factors(self, financial_data: List[FinancialData], 
                            market_data: Optional[Dict] = None,
                            lifecycle_data: Optional[Dict] = None) -> pd.DataFrame:
        """
        全ての要因項目を計算
        
        Args:
            financial_data: 財務データリスト
            market_data: 市場データ（シェア情報等）
            lifecycle_data: ライフサイクルデータ（設立年、分社情報等）
            
        Returns:
            計算済み要因項目のDataFrame
        """
        results = []
        
        # 企業ごとにソート
        df = pd.DataFrame([vars(fd) for fd in financial_data])
        df = df.sort_values(['company_id', 'year'])
        
        for company_id in df['company_id'].unique():
            company_data = df[df['company_id'] == company_id].copy()
            company_lifecycle = lifecycle_data.get(company_id, {}) if lifecycle_data else {}
            
            for _, row in company_data.iterrows():
                factors = self._calculate_single_year_factors(
                    row, company_data, company_lifecycle, market_data
                )
                factors['company_id'] = company_id
                factors['year'] = row['year']
                results.append(factors)
        
        return pd.DataFrame(results)
    
    def _calculate_single_year_factors(self, current_row: pd.Series, 
                                        company_data: pd.DataFrame,
                                        lifecycle_data: Dict,
                                        market_data: Optional[Dict]) -> Dict:
        """単年度の要因項目計算"""
        
        factors = {}
        
        # 1. 売上高の要因項目（23項目）
        factors.update(self._calculate_revenue_factors(current_row, company_data))
        
        # 2. 売上高成長率の要因項目（23項目）
        factors.update(self._calculate_growth_factors(current_row, company_data))
        
        # 3. 売上高営業利益率の要因項目（23項目）
        factors.update(self._calculate_operating_margin_factors(current_row, company_data))
        
        # 4. 売上高当期純利益率の要因項目（23項目）
        factors.update(self._calculate_net_margin_factors(current_row, company_data))
        
        # 5. ROEの要因項目（23項目）
        factors.update(self._calculate_roe_factors(current_row, company_data))
        
        # 6. 売上高付加価値率の要因項目（23項目）
        factors.update(self._calculate_value_added_factors(current_row, company_data))
        
        # 7. 企業存続確率の要因項目（23項目）
        factors.update(self._calculate_survival_factors(current_row, company_data, lifecycle_data))
        
        # 8. 新規事業成功率の要因項目（23項目）
        factors.update(self._calculate_emergence_factors(current_row, company_data, lifecycle_data))
        
        # 9. 事業継承成功度の要因項目（23項目）
        factors.update(self._calculate_succession_factors(current_row, company_data, lifecycle_data))
        
        return factors
    
    def _calculate_revenue_factors(self, row: pd.Series, company_data: pd.DataFrame) -> Dict:
        """売上高の要因項目（23項目）計算"""
        factors = {}
        revenue = row.get('revenue', 0)
        
        if revenue == 0:
            return {f'rev_{i:02d}': None for i in range(1, 24)}
        
        # 投資・資産関連 (1-5)
        factors['rev_01'] = row.get('tangible_fixed_assets', 0)  # 有形固定資産
        factors['rev_02'] = row.get('capex', 0)  # 設備投資額
        factors['rev_03'] = row.get('rd_expenses', 0)  # 研究開発費
        factors['rev_04'] = row.get('intangible_fixed_assets', 0)  # 無形固定資産
        factors['rev_05'] = row.get('investment_securities', 0)  # 投資有価証券
        
        # 人的資源関連 (6-10)
        factors['rev_06'] = row.get('employees', 0)  # 従業員数
        factors['rev_07'] = row.get('average_salary', 0)  # 平均年間給与
        factors['rev_08'] = self._estimate_retirement_benefit_cost(row)  # 退職給付費用
        factors['rev_09'] = self._estimate_welfare_cost(row)  # 福利厚生費
        factors['rev_10'] = row.get('dividend_payout_ratio', 0)  # 総還元性向
        
        # 運転資本・効率性関連 (11-15)
        factors['rev_11'] = row.get('accounts_receivable', 0)  # 売上債権
        factors['rev_12'] = row.get('inventories', 0)  # 棚卸資産
        factors['rev_13'] = row.get('total_assets', 0)  # 総資産
        factors['rev_14'] = self._calculate_receivables_turnover(row)  # 売上債権回転率
        factors['rev_15'] = self._calculate_inventory_turnover(row)  # 棚卸資産回転率
        
        # 事業展開関連 (16-20)
        factors['rev_16'] = row.get('overseas_revenue_ratio', 0)  # 海外売上高比率
        factors['rev_17'] = row.get('segments_count', 1)  # 事業セグメント数
        factors['rev_18'] = row.get('sg_a_expenses', 0)  # 販売費及び一般管理費
        factors['rev_19'] = row.get('advertising_expenses', 0)  # 広告宣伝費
        factors['rev_20'] = self._calculate_non_operating_income(row)  # 営業外収益
        
        # 拡張項目 (21-23)
        factors['rev_21'] = self._calculate_company_age(row, company_data)  # 企業年齢
        factors['rev_22'] = self._calculate_market_entry_timing(row, company_data)  # 市場参入時期
        factors['rev_23'] = self._calculate_parent_dependency(row, company_data)  # 親会社依存度
        
        return factors
    
    def _calculate_growth_factors(self, row: pd.Series, company_data: pd.DataFrame) -> Dict:
        """売上高成長率の要因項目（23項目）計算"""
        factors = {}
        
        # 前年データ取得
        prev_year_data = self._get_previous_year_data(row, company_data)
        if prev_year_data is None:
            return {f'growth_{i:02d}': None for i in range(1, 24)}
        
        # 投資・拡張関連 (1-6)
        factors['growth_01'] = self._calculate_growth_rate(row.get('capex'), prev_year_data.get('capex'))
        factors['growth_02'] = self._calculate_growth_rate(row.get('rd_expenses'), prev_year_data.get('rd_expenses'))
        factors['growth_03'] = self._calculate_growth_rate(row.get('tangible_fixed_assets'), prev_year_data.get('tangible_fixed_assets'))
        factors['growth_04'] = self._calculate_growth_rate(row.get('intangible_fixed_assets'), prev_year_data.get('intangible_fixed_assets'))
        factors['growth_05'] = self._calculate_growth_rate(row.get('total_assets'), prev_year_data.get('total_assets'))
        factors['growth_06'] = self._calculate_goodwill_growth(row, prev_year_data)  # のれん増加率
        
        # 人的資源拡張 (7-10)
        factors['growth_07'] = self._calculate_growth_rate(row.get('employees'), prev_year_data.get('employees'))
        factors['growth_08'] = self._calculate_growth_rate(row.get('average_salary'), prev_year_data.get('average_salary'))
        factors['growth_09'] = self._calculate_personnel_cost_growth(row, prev_year_data)
        factors['growth_10'] = self._calculate_retirement_cost_growth(row, prev_year_data)
        
        # 市場・事業拡大 (11-15)
        factors['growth_11'] = self._calculate_overseas_ratio_change(row, prev_year_data)
        factors['growth_12'] = self._calculate_segment_revenue_growth(row, prev_year_data)
        factors['growth_13'] = self._calculate_growth_rate(row.get('sg_a_expenses'), prev_year_data.get('sg_a_expenses'))
        factors['growth_14'] = self._calculate_growth_rate(row.get('advertising_expenses'), prev_year_data.get('advertising_expenses'))
        factors['growth_15'] = self._calculate_non_operating_income_growth(row, prev_year_data)
        
        # 効率性・能力関連 (16-20)
        factors['growth_16'] = self._calculate_growth_rate(row.get('accounts_receivable'), prev_year_data.get('accounts_receivable'))
        factors['growth_17'] = self._calculate_growth_rate(row.get('inventories'), prev_year_data.get('inventories'))
        factors['growth_18'] = self._calculate_turnover_change(row, prev_year_data, 'receivables')
        factors['growth_19'] = self._calculate_turnover_change(row, prev_year_data, 'inventory')
        factors['growth_20'] = self._calculate_turnover_change(row, prev_year_data, 'total_assets')
        
        # 拡張項目 (21-23)
        factors['growth_21'] = self._calculate_maturity_impact(row, company_data)  # 成熟度影響
        factors['growth_22'] = self._calculate_innovation_cycle(row, company_data)  # イノベーションサイクル
        factors['growth_23'] = self._calculate_market_expansion_rate(row, company_data)  # 市場拡大率
        
        return factors
    
    def _calculate_operating_margin_factors(self, row: pd.Series, company_data: pd.DataFrame) -> Dict:
        """売上高営業利益率の要因項目（23項目）計算"""
        factors = {}
        revenue = row.get('revenue', 0)
        
        if revenue == 0:
            return {f'op_margin_{i:02d}': None for i in range(1, 24)}
        
        # 売上原価構成 (1-5)
        factors['op_margin_01'] = self._calculate_material_cost_ratio(row)  # 材料費率
        factors['op_margin_02'] = self._calculate_labor_cost_ratio(row)  # 労務費率
        factors['op_margin_03'] = self._calculate_overhead_ratio(row)  # 経費率
        factors['op_margin_04'] = self._calculate_outsourcing_ratio(row)  # 外注加工費率
        factors['op_margin_05'] = self._safe_divide(row.get('depreciation', 0), row.get('cost_of_sales', 1))  # 減価償却費率
        
        # 販管費構成 (6-10)
        factors['op_margin_06'] = self._safe_divide(row.get('sg_a_expenses', 0), revenue)  # 販管費率
        factors['op_margin_07'] = self._calculate_sga_personnel_ratio(row)  # 人件費率（販管費）
        factors['op_margin_08'] = self._safe_divide(row.get('advertising_expenses', 0), revenue)  # 広告宣伝費率
        factors['op_margin_09'] = self._safe_divide(row.get('rd_expenses', 0), revenue)  # 研究開発費率
        factors['op_margin_10'] = self._calculate_sga_depreciation_ratio(row)  # 減価償却費率（販管費）
        
        # 効率性指標 (11-15)
        factors['op_margin_11'] = self._calculate_value_added_ratio(row)  # 売上高付加価値率
        factors['op_margin_12'] = self._calculate_labor_productivity(row)  # 労働生産性
        factors['op_margin_13'] = self._calculate_asset_efficiency(row)  # 設備効率性
        factors['op_margin_14'] = self._calculate_total_asset_turnover(row)  # 総資産回転率
        factors['op_margin_15'] = self._calculate_inventory_turnover(row)  # 棚卸資産回転率
        
        # 規模・構造要因 (16-20)
        factors['op_margin_16'] = revenue  # 売上高（規模効果）
        factors['op_margin_17'] = self._calculate_fixed_cost_ratio(row)  # 固定費率
        factors['op_margin_18'] = self._calculate_variable_cost_ratio(row)  # 変動費率
        factors['op_margin_19'] = row.get('overseas_revenue_ratio', 0)  # 海外売上高比率
        factors['op_margin_20'] = self._calculate_business_concentration(row)  # 事業セグメント集中度
        
        # 拡張項目 (21-23)
        factors['op_margin_21'] = self._calculate_competitive_position(row, company_data)  # 競争ポジション
        factors['op_margin_22'] = self._calculate_operational_leverage(row, company_data)  # オペレーティングレバレッジ
        factors['op_margin_23'] = self._calculate_cost_structure_flexibility(row, company_data)  # コスト構造柔軟性
        
        return factors
    
    def _calculate_net_margin_factors(self, row: pd.Series, company_data: pd.DataFrame) -> Dict:
        """売上高当期純利益率の要因項目（23項目）計算"""
        factors = {}
        revenue = row.get('revenue', 0)
        
        if revenue == 0:
            return {f'net_margin_{i:02d}': None for i in range(1, 24)}
        
        # 営業損益要因 (1-5)
        factors['net_margin_01'] = self._safe_divide(row.get('operating_profit', 0), revenue)  # 売上高営業利益率
        factors['net_margin_02'] = self._safe_divide(row.get('sg_a_expenses', 0), revenue)  # 販管費率
        factors['net_margin_03'] = self._safe_divide(row.get('cost_of_sales', 0), revenue)  # 売上原価率
        factors['net_margin_04'] = self._safe_divide(row.get('rd_expenses', 0), revenue)  # 研究開発費率
        factors['net_margin_05'] = self._safe_divide(row.get('depreciation', 0), revenue)  # 減価償却費率
        
        # 営業外損益 (6-10)
        factors['net_margin_06'] = self._safe_divide(row.get('interest_income', 0), revenue)  # 受取利息・配当金
        factors['net_margin_07'] = self._safe_divide(row.get('interest_expenses', 0), revenue)  # 支払利息
        factors['net_margin_08'] = self._safe_divide(row.get('fx_gain_loss', 0), revenue)  # 為替差損益
        factors['net_margin_09'] = self._calculate_equity_method_income(row)  # 持分法投資損益
        factors['net_margin_10'] = self._calculate_non_operating_income_ratio(row)  # 営業外収益率
        
        # 特別損益・税金 (11-15)
        factors['net_margin_11'] = self._safe_divide(row.get('extraordinary_profit', 0), revenue)  # 特別利益
        factors['net_margin_12'] = self._safe_divide(row.get('extraordinary_loss', 0), revenue)  # 特別損失
        factors['net_margin_13'] = row.get('tax_rate', 0)  # 法人税等実効税率
        factors['net_margin_14'] = self._calculate_tax_adjustment(row)  # 法人税等調整額
        factors['net_margin_15'] = self._calculate_pretax_margin(row)  # 税引前当期純利益率
        
        # 財務構造要因 (16-20)
        factors['net_margin_16'] = self._calculate_interest_bearing_debt_ratio(row)  # 有利子負債比率
        factors['net_margin_17'] = self._calculate_equity_ratio(row)  # 自己資本比率
        factors['net_margin_18'] = self._calculate_investment_securities_gain_loss(row)  # 投資有価証券評価損益
        factors['net_margin_19'] = self._calculate_fixed_asset_disposal_gain_loss(row)  # 固定資産売却損益
        factors['net_margin_20'] = self._calculate_impairment_loss_ratio(row)  # 減損損失率
        
        # 拡張項目 (21-23)
        factors['net_margin_21'] = self._calculate_financial_leverage_impact(row, company_data)  # 財務レバレッジ影響
        factors['net_margin_22'] = self._calculate_risk_management_efficiency(row, company_data)  # リスク管理効率性
        factors['net_margin_23'] = self._calculate_capital_allocation_efficiency(row, company_data)  # 資本配分効率性
        
        return factors
    
    def _calculate_roe_factors(self, row: pd.Series, company_data: pd.DataFrame) -> Dict:
        """ROEの要因項目（23項目）計算"""
        factors = {}
        equity = row.get('shareholders_equity', 0)
        
        if equity == 0:
            return {f'roe_{i:02d}': None for i in range(1, 24)}
        
        # 収益性要因（ROA構成） (1-5)
        factors['roe_01'] = self._safe_divide(row.get('net_profit', 0), row.get('revenue', 1))  # 売上高当期純利益率
        factors['roe_02'] = self._calculate_total_asset_turnover(row)  # 総資産回転率
        factors['roe_03'] = self._safe_divide(row.get('operating_profit', 0), row.get('revenue', 1))  # 売上高営業利益率
        factors['roe_04'] = self._safe_divide(row.get('cost_of_sales', 0), row.get('revenue', 1))  # 売上原価率
        factors['roe_05'] = self._safe_divide(row.get('sg_a_expenses', 0), row.get('revenue', 1))  # 販管費率
        
        # 財務レバレッジ (6-10)
        factors['roe_06'] = self._calculate_equity_ratio(row)  # 自己資本比率
        factors['roe_07'] = self._safe_divide(row.get('total_assets', 1), equity)  # 総資産/自己資本倍率
        factors['roe_08'] = self._calculate_debt_equity_ratio(row)  # 有利子負債/自己資本比率
        factors['roe_09'] = self._calculate_current_ratio(row)  # 流動比率
        factors['roe_10'] = self._calculate_fixed_ratio(row)  # 固定比率
        
        # 資産効率性 (11-15)
        factors['roe_11'] = self._calculate_receivables_turnover(row)  # 売上債権回転率
        factors['roe_12'] = self._calculate_inventory_turnover(row)  # 棚卸資産回転率
        factors['roe_13'] = self._calculate_fixed_asset_turnover(row)  # 有形固定資産回転率
        factors['roe_14'] = self._safe_divide(row.get('cash_and_deposits', 0), row.get('total_assets', 1))  # 現預金比率
        factors['roe_15'] = self._safe_divide(row.get('investment_securities', 0), row.get('total_assets', 1))  # 投資有価証券比率
        
        # 収益・配当政策 (16-20)
        factors['roe_16'] = row.get('dividend_payout_ratio', 0)  # 配当性向
        factors['roe_17'] = self._calculate_internal_retention_ratio(row)  # 内部留保率
        factors['roe_18'] = self._calculate_non_operating_income_ratio(row)  # 営業外収益率
        factors['roe_19'] = self._calculate_extraordinary_impact_ratio(row)  # 特別損益影響率
        factors['roe_20'] = row.get('tax_rate', 0)  # 実効税率
        
        # 拡張項目 (21-23)
        factors['roe_21'] = self._calculate_shareholder_return_efficiency(row, company_data)  # 株主還元効率性
        factors['roe_22'] = self._calculate_growth_sustainability(row, company_data)  # 成長持続可能性
        factors['roe_23'] = self._calculate_capital_efficiency_trend(row, company_data)  # 資本効率性トレンド
        
        return factors
    
    def _calculate_value_added_factors(self, row: pd.Series, company_data: pd.DataFrame) -> Dict:
        """売上高付加価値率の要因項目（23項目）計算"""
        factors = {}
        revenue = row.get('revenue', 0)
        
        if revenue == 0:
            return {f'va_{i:02d}': None for i in range(1, 24)}
        
        # 技術・研究開発 (1-5)
        factors['va_01'] = self._safe_divide(row.get('rd_expenses', 0), revenue)  # 研究開発費率
        factors['va_02'] = self._safe_divide(row.get('intangible_fixed_assets', 0), revenue)  # 無形固定資産比率
        factors['va_03'] = self._calculate_patent_related_cost(row)  # 特許関連費用
        factors['va_04'] = self._calculate_software_ratio(row)  # ソフトウェア比率
        factors['va_05'] = self._calculate_technology_license_income(row)  # 技術ライセンス収入
        
        # 人的付加価値 (6-10)
        factors['va_06'] = self._calculate_salary_industry_ratio(row)  # 平均年間給与/業界平均比率
        factors['va_07'] = self._calculate_personnel_cost_ratio(row)  # 人件費率
        factors['va_08'] = self._calculate_employee_productivity_ratio(row)  # 従業員数/売上高比率
        factors['va_09'] = self._calculate_retirement_benefit_ratio(row)  # 退職給付費用率
        factors['va_10'] = self._calculate_welfare_cost_ratio(row)  # 福利厚生費率
        
        # コスト構造・効率性 (11-15)
        factors['va_11'] = 1 - self._safe_divide(row.get('cost_of_sales', 0), revenue)  # 売上原価率（逆数）
        factors['va_12'] = 1 - self._calculate_material_cost_ratio(row)  # 材料費率（逆数）
        factors['va_13'] = 1 - self._calculate_outsourcing_ratio(row)  # 外注加工費率（逆数）
        factors['va_14'] = self._calculate_labor_productivity(row)  # 労働生産性
        factors['va_15'] = self._calculate_asset_efficiency(row)  # 設備生産性
        
        # 事業構造・差別化 (16-20)
        factors['va_16'] = row.get('overseas_revenue_ratio', 0)  # 海外売上高比率
        factors['va_17'] = self._calculate_high_value_segment_ratio(row)  # 高付加価値事業セグメント比率
        factors['va_18'] = self._calculate_service_revenue_ratio(row)  # サービス・保守収入比率
        factors['va_19'] = self._safe_divide(row.get('operating_profit', 0), revenue)  # 営業利益率
        factors['va_20'] = self._calculate_intangible_asset_ratio(row)  # ブランド・商標等無形資産比率
        
        # 拡張項目 (21-23)
        factors['va_21'] = self._calculate_innovation_intensity(row, company_data)  # イノベーション強度
        factors['va_22'] = self._calculate_differentiation_degree(row, company_data)  # 差別化度
        factors['va_23'] = self._calculate_value_chain_integration(row, company_data)  # バリューチェーン統合度
        
        return factors
    
    def _calculate_survival_factors(self, row: pd.Series, company_data: pd.DataFrame, 
                                    lifecycle_data: Dict) -> Dict:
        """企業存続確率の要因項目（23項目）計算"""
        factors = {}
        
        # 財務健全性指標 (1-5)
        factors['survival_01'] = self._calculate_equity_ratio(row)  # 自己資本比率
        factors['survival_02'] = self._calculate_current_ratio(row)  # 流動比率
        factors['survival_03'] = self._calculate_debt_service_coverage(row)  # 債務返済能力
        factors['survival_04'] = self._calculate_interest_coverage(row)  # インタレストカバレッジ
        factors['survival_05'] = self._calculate_cash_flow_stability(row, company_data)  # キャッシュフロー安定性
        
        # 収益性・成長性 (6-10)
        factors['survival_06'] = self._safe_divide(row.get('operating_profit', 0), row.get('revenue', 1))  # 営業利益率
        factors['survival_07'] = self._calculate_roe(row)  # ROE
        factors['survival_08'] = self._calculate_revenue_growth_stability(row, company_data)  # 売上成長安定性
        factors['survival_09'] = self._calculate_profit_growth_stability(row, company_data)  # 利益成長安定性
        factors['survival_10'] = self._calculate_market_share_trend(row, company_data)  # 市場シェアトレンド
        
        # 適応能力・イノベーション (11-15)
        factors['survival_11'] = self._safe_divide(row.get('rd_expenses', 0), row.get('revenue', 1))  # R&D投資率
        factors['survival_12'] = self._calculate_business_model_flexibility(row, company_data)  # ビジネスモデル柔軟性
        factors['survival_13'] = self._calculate_technology_adaptation_rate(row, company_data)  # 技術適応率
        factors['survival_14'] = self._calculate_market_diversification(row)  # 市場多様化度
        factors['survival_15'] = self._calculate_innovation_frequency(row, company_data)  # イノベーション頻度
        
        # 競争優位性 (16-20)
        factors['survival_16'] = self._calculate_competitive_moat_strength(row, company_data)  # 競争優位性強度
        factors['survival_17'] = self._calculate_brand_value_proxy(row, company_data)  # ブランド価値代理指標
        factors['survival_18'] = self._calculate_customer_loyalty_proxy(row, company_data)  # 顧客ロイヤルティ代理指標
        factors['survival_19'] = self._calculate_supply_chain_resilience(row, company_data)  # サプライチェーン強靭性
        factors['survival_20'] = self._calculate_regulatory_compliance_score(row, company_data)  # 規制適応スコア
        
        # 拡張項目 (21-23)
        factors['survival_21'] = self._calculate_company_age_survival_curve(row, lifecycle_data)  # 企業年齢×生存曲線
        factors['survival_22'] = self._calculate_crisis_resistance_capacity(row, company_data)  # 危機耐性能力
        factors['survival_23'] = self._calculate_strategic_agility(row, company_data)  # 戦略的俊敏性
        
        return factors
    
    def _calculate_emergence_factors(self, row: pd.Series, company_data: pd.DataFrame, 
                                    lifecycle_data: Dict) -> Dict:
        """新規事業成功率の要因項目（23項目）計算"""
        factors = {}
        
        # スタートアップ基盤 (1-5)
        factors['emergence_01'] = self._calculate_initial_capital_adequacy(row, lifecycle_data)  # 初期資本充足度
        factors['emergence_02'] = self._calculate_founder_experience_proxy(row, lifecycle_data)  # 創業者経験代理指標
        factors['emergence_03'] = self._calculate_market_timing_score(row, lifecycle_data)  # 市場タイミングスコア
        factors['emergence_04'] = self._calculate_technology_novelty_degree(row, company_data)  # 技術新規性度
        factors['emergence_05'] = self._calculate_market_size_potential(row, lifecycle_data)  # 市場規模ポテンシャル
        
        # 成長エンジン (6-10)
        factors['emergence_06'] = self._calculate_revenue_scaling_rate(row, company_data)  # 売上拡大率
        factors['emergence_07'] = self._calculate_customer_acquisition_efficiency(row, company_data)  # 顧客獲得効率性
        factors['emergence_08'] = self._calculate_product_market_fit_proxy(row, company_data)  # プロダクトマーケットフィット代理指標
        factors['emergence_09'] = self._calculate_unit_economics_health(row, company_data)  # ユニットエコノミクス健全性
        factors['emergence_10'] = self._calculate_viral_coefficient_proxy(row, company_data)  # バイラル係数代理指標
        
        # 資源調達・投資 (11-15)
        factors['emergence_11'] = self._calculate_funding_efficiency(row, company_data)  # 資金調達効率性
        factors['emergence_12'] = self._calculate_talent_acquisition_rate(row, company_data)  # 人材獲得率
        factors['emergence_13'] = self._calculate_rd_investment_intensity(row, company_data)  # R&D投資強度
        factors['emergence_14'] = self._calculate_infrastructure_investment_ratio(row, company_data)  # インフラ投資比率
        factors['emergence_15'] = self._calculate_partnership_leverage(row, company_data)  # パートナーシップ活用度
        
        # 市場参入戦略 (16-20)
        factors['emergence_16'] = self._calculate_market_penetration_speed(row, company_data)  # 市場浸透速度
        factors['emergence_17'] = self._calculate_competitive_differentiation(row, company_data)  # 競争差別化度
        factors['emergence_18'] = self._calculate_pricing_strategy_effectiveness(row, company_data)  # 価格戦略効果性
        factors['emergence_19'] = self._calculate_channel_development_rate(row, company_data)  # チャネル開発率
        factors['emergence_20'] = self._calculate_brand_recognition_speed(row, company_data)  # ブランド認知速度
        
        # 拡張項目 (21-23)
        factors['emergence_21'] = self._calculate_ecosystem_integration_rate(row, company_data)  # エコシステム統合率
        factors['emergence_22'] = self._calculate_scalability_potential(row, company_data)  # スケーラビリティポテンシャル
        factors['emergence_23'] = self._calculate_resilience_building_rate(row, company_data)  # 強靭性構築率
        
        return factors
    
    def _calculate_succession_factors(self, row: pd.Series, company_data: pd.DataFrame, 
                                    lifecycle_data: Dict) -> Dict:
        """事業継承成功度の要因項目（23項目）計算"""
        factors = {}
        
        # 継承準備・計画 (1-5)
        factors['succession_01'] = self._calculate_succession_planning_maturity(row, lifecycle_data)  # 継承計画成熟度
        factors['succession_02'] = self._calculate_knowledge_transfer_efficiency(row, company_data)  # 知識移転効率性
        factors['succession_03'] = self._calculate_organizational_continuity(row, company_data)  # 組織継続性
        factors['succession_04'] = self._calculate_culture_preservation_score(row, company_data)  # 企業文化保存スコア
        factors['succession_05'] = self._calculate_stakeholder_alignment(row, company_data)  # ステークホルダー整合性
        
        # 財務継承効果 (6-10)
        factors['succession_06'] = self._calculate_synergy_realization_rate(row, company_data)  # シナジー実現率
        factors['succession_07'] = self._calculate_cost_integration_efficiency(row, company_data)  # コスト統合効率性
        factors['succession_08'] = self._calculate_revenue_retention_rate(row, company_data)  # 売上維持率
        factors['succession_09'] = self._calculate_customer_retention_during_transition(row, company_data)  # 移行期顧客維持率
        factors['succession_10'] = self._calculate_operational_disruption_minimization(row, company_data)  # 業務中断最小化
        
        # 統合・再編効果 (11-15)
        factors['succession_11'] = self._calculate_integration_speed(row, company_data)  # 統合速度
        factors['succession_12'] = self._calculate_duplicate_elimination_efficiency(row, company_data)  # 重複排除効率性
        factors['succession_13'] = self._calculate_best_practice_diffusion(row, company_data)  # ベストプラクティス拡散
        factors['succession_14'] = self._calculate_scale_economy_realization(row, company_data)  # 規模経済実現度
        factors['succession_15'] = self._calculate_scope_economy_realization(row, company_data)  # 範囲経済実現度
        
        # 戦略継承・発展 (16-20)
        factors['succession_16'] = self._calculate_strategic_vision_continuity(row, company_data)  # 戦略ビジョン継続性
        factors['succession_17'] = self._calculate_innovation_capability_enhancement(row, company_data)  # イノベーション能力向上
        factors['succession_18'] = self._calculate_market_position_strengthening(row, company_data)  # 市場ポジション強化
        factors['succession_19'] = self._calculate_competitive_advantage_amplification(row, company_data)  # 競争優位性増幅
        factors['succession_20'] = self._calculate_growth_opportunity_expansion(row, company_data)  # 成長機会拡大
        
        # 拡張項目 (21-23)
        factors['succession_21'] = self._calculate_next_generation_readiness(row, lifecycle_data)  # 次世代準備度
        factors['succession_22'] = self._calculate_legacy_value_preservation(row, company_data)  # レガシー価値保存
        factors['succession_23'] = self._calculate_transformation_success_rate(row, company_data)  # 変革成功率
        
        return factors
    
    # ユーティリティメソッド群
    
    def _safe_divide(self, numerator: Union[float, int, None], 
                    denominator: Union[float, int, None]) -> Optional[float]:
        """安全な除算（ゼロ除算対応）"""
        if numerator is None or denominator is None or denominator == 0:
            return None
        return float(numerator) / float(denominator)
    
    def _get_previous_year_data(self, current_row: pd.Series, 
                                company_data: pd.DataFrame) -> Optional[pd.Series]:
        """前年データ取得"""
        prev_year = current_row['year'] - 1
        prev_data = company_data[company_data['year'] == prev_year]
        return prev_data.iloc[0] if not prev_data.empty else None
    
    def _calculate_growth_rate(self, current_value: Optional[float], 
                                previous_value: Optional[float]) -> Optional[float]:
        """成長率計算"""
        if current_value is None or previous_value is None or previous_value == 0:
            return None
        return (current_value - previous_value) / previous_value
    
    def _calculate_receivables_turnover(self, row: pd.Series) -> Optional[float]:
        """売上債権回転率"""
        return self._safe_divide(row.get('revenue'), row.get('accounts_receivable'))
    
    def _calculate_inventory_turnover(self, row: pd.Series) -> Optional[float]:
        """棚卸資産回転率"""
        return self._safe_divide(row.get('cost_of_sales'), row.get('inventories'))
    
    def _calculate_total_asset_turnover(self, row: pd.Series) -> Optional[float]:
        """総資産回転率"""
        return self._safe_divide(row.get('revenue'), row.get('total_assets'))
    
    def _calculate_equity_ratio(self, row: pd.Series) -> Optional[float]:
        """自己資本比率"""
        return self._safe_divide(row.get('shareholders_equity'), row.get('total_assets'))
    
    def _calculate_roe(self, row: pd.Series) -> Optional[float]:
        """ROE計算"""
        return self._safe_divide(row.get('net_profit'), row.get('shareholders_equity'))
    
    def _calculate_labor_productivity(self, row: pd.Series) -> Optional[float]:
        """労働生産性（売上高/従業員数）"""
        return self._safe_divide(row.get('revenue'), row.get('employees'))
    
    def _calculate_asset_efficiency(self, row: pd.Series) -> Optional[float]:
        """設備効率性（売上高/有形固定資産）"""
        return self._safe_divide(row.get('revenue'), row.get('tangible_fixed_assets'))
    
    def _calculate_current_ratio(self, row: pd.Series) -> Optional[float]:
        """流動比率（推定）"""
        # 流動資産 = 現金 + 売上債権 + 棚卸資産（簡易推定）
        current_assets = (row.get('cash_and_deposits', 0) + 
                            row.get('accounts_receivable', 0) + 
                            row.get('inventories', 0))
        # 流動負債 = 総資産 - 自己資本 - 固定負債（簡易推定）
        total_liabilities = row.get('total_assets', 0) - row.get('shareholders_equity', 0)
        current_liabilities = total_liabilities * 0.6  # 経験的比率
        return self._safe_divide(current_assets, current_liabilities)
    
    def _calculate_debt_equity_ratio(self, row: pd.Series) -> Optional[float]:
        """有利子負債/自己資本比率（推定）"""
        # 有利子負債 = 支払利息から逆算（簡易推定）
        interest_expense = row.get('interest_expenses', 0)
        estimated_debt = interest_expense / 0.02 if interest_expense > 0 else 0  # 2%金利と仮定
        return self._safe_divide(estimated_debt, row.get('shareholders_equity'))
    
    # 推定・代理指標計算メソッド群
    
    def _estimate_retirement_benefit_cost(self, row: pd.Series) -> Optional[float]:
        """退職給付費用推定"""
        employees = row.get('employees', 0)
        avg_salary = row.get('average_salary', 0)
        if employees > 0 and avg_salary > 0:
            return employees * avg_salary * 0.05  # 経験的比率5%
        return None
    
    def _estimate_welfare_cost(self, row: pd.Series) -> Optional[float]:
        """福利厚生費推定"""
        employees = row.get('employees', 0)
        avg_salary = row.get('average_salary', 0)
        if employees > 0 and avg_salary > 0:
            return employees * avg_salary * 0.15  # 経験的比率15%
        return None
    
    def _calculate_material_cost_ratio(self, row: pd.Series) -> Optional[float]:
        """材料費率推定"""
        cost_of_sales = row.get('cost_of_sales', 0)
        revenue = row.get('revenue', 1)
        # 製造業の経験的比率：売上原価の60%が材料費
        return (cost_of_sales * 0.6) / revenue if revenue > 0 else None
    
    def _calculate_labor_cost_ratio(self, row: pd.Series) -> Optional[float]:
        """労務費率推定"""
        cost_of_sales = row.get('cost_of_sales', 0)
        revenue = row.get('revenue', 1)
        # 製造業の経験的比率：売上原価の25%が労務費
        return (cost_of_sales * 0.25) / revenue if revenue > 0 else None
    
    def _calculate_overhead_ratio(self, row: pd.Series) -> Optional[float]:
        """経費率推定"""
        cost_of_sales = row.get('cost_of_sales', 0)
        revenue = row.get('revenue', 1)
        # 製造業の経験的比率：売上原価の15%が経費
        return (cost_of_sales * 0.15) / revenue if revenue > 0 else None
    
    def _calculate_outsourcing_ratio(self, row: pd.Series) -> Optional[float]:
        """外注加工費率推定"""
        cost_of_sales = row.get('cost_of_sales', 0)
        revenue = row.get('revenue', 1)
        overseas_ratio = row.get('overseas_revenue_ratio', 0)
        # 海外売上比率が高いほど外注比率も高いと仮定
        outsourcing_rate = 0.1 + (overseas_ratio * 0.2)
        return (cost_of_sales * outsourcing_rate) / revenue if revenue > 0 else None
    
    def _calculate_non_operating_income(self, row: pd.Series) -> Optional[float]:
        """営業外収益計算"""
        interest_income = row.get('interest_income', 0)
        fx_gain = max(row.get('fx_gain_loss', 0), 0)  # プラスの場合のみ
        return interest_income + fx_gain
    
    def _calculate_company_age(self, row: pd.Series, company_data: pd.DataFrame) -> Optional[float]:
        """企業年齢計算"""
        current_year = row.get('year', 2024)
        min_year = company_data['year'].min()
        # データ開始年を設立年と仮定（実際には設立年データが望ましい）
        return current_year - min_year
    
    def _calculate_market_entry_timing(self, row: pd.Series, company_data: pd.DataFrame) -> Optional[float]:
        """市場参入時期指標（先発=1、後発=0に近い値）"""
        company_age = self._calculate_company_age(row, company_data)
        # 40年以上=先発、10年未満=後発として正規化
        if company_age is None:
            return None
        return min(company_age / 40, 1.0)
    
    def _calculate_parent_dependency(self, row: pd.Series, company_data: pd.DataFrame) -> Optional[float]:
        """親会社依存度（分社企業の場合）"""
        # 簡易推定：海外売上比率が低く、事業セグメントが少ない場合は依存度高
        overseas_ratio = row.get('overseas_revenue_ratio', 0)
        segments = row.get('segments_count', 1)
        
        if overseas_ratio < 0.2 and segments <= 2:
            return 0.8  # 高依存
        elif overseas_ratio < 0.5 and segments <= 3:
            return 0.5  # 中依存
        else:
            return 0.2  # 低依存
    
    def _calculate_value_added_ratio(self, row: pd.Series) -> Optional[float]:
        """売上高付加価値率"""
        revenue = row.get('revenue', 0)
        cost_of_sales = row.get('cost_of_sales', 0)
        material_cost = cost_of_sales * 0.6 if cost_of_sales else 0  # 材料費推定
        
        value_added = revenue - material_cost
        return self._safe_divide(value_added, revenue)
    
    def _calculate_fixed_cost_ratio(self, row: pd.Series) -> Optional[float]:
        """固定費率推定"""
        sg_a = row.get('sg_a_expenses', 0)
        depreciation = row.get('depreciation', 0)
        revenue = row.get('revenue', 1)
        
        # 固定費 = 販管費 + 減価償却費
        fixed_costs = sg_a + depreciation
        return self._safe_divide(fixed_costs, revenue)
    
    def _calculate_variable_cost_ratio(self, row: pd.Series) -> Optional[float]:
        """変動費率推定"""
        material_cost_ratio = self._calculate_material_cost_ratio(row) or 0
        outsourcing_ratio = self._calculate_outsourcing_ratio(row) or 0
        return material_cost_ratio + outsourcing_ratio
    
    def _calculate_business_concentration(self, row: pd.Series) -> Optional[float]:
        """事業セグメント集中度（HHI的指標）"""
        segments = row.get('segments_count', 1)
        if segments <= 1:
            return 1.0  # 完全集中
        else:
            # セグメント数が多いほど分散（集中度低下）
            return 1.0 / segments
    
    # 生存分析関連の高度な計算メソッド
    
    def _calculate_cash_flow_stability(self, row: pd.Series, company_data: pd.DataFrame) -> Optional[float]:
        """キャッシュフロー安定性"""
        # 過去5年間の営業CF変動係数を計算
        current_year = row.get('year')
        past_years = company_data[company_data['year'].between(current_year-5, current_year-1)]
        
        if len(past_years) < 3:
            return None
        
        # 営業CFを売上高で正規化
        normalized_cf = []
        for _, past_row in past_years.iterrows():
            operating_cf = self._estimate_operating_cf(past_row)
            revenue = past_row.get('revenue', 1)
            if operating_cf is not None and revenue > 0:
                normalized_cf.append(operating_cf / revenue)
        
        if len(normalized_cf) < 3:
            return None
        
        # 変動係数（標準偏差/平均）の逆数（安定性指標）
        mean_cf = np.mean(normalized_cf)
        std_cf = np.std(normalized_cf)
        
        if mean_cf > 0 and std_cf > 0:
            cv = std_cf / mean_cf
            return 1 / (1 + cv)  # 0-1の範囲で安定性を表現
        
        return None
    
    def _estimate_operating_cf(self, row: pd.Series) -> Optional[float]:
        """営業CF推定"""
        net_profit = row.get('net_profit', 0)
        depreciation = row.get('depreciation', 0)
        # 簡易推定：純利益 + 減価償却費
        return net_profit + depreciation
    
    def _calculate_revenue_growth_stability(self, row: pd.Series, company_data: pd.DataFrame) -> Optional[float]:
        """売上成長安定性"""
        current_year = row.get('year')
        past_years = company_data[company_data['year'].between(current_year-5, current_year)]
        
        if len(past_years) < 4:
            return None
        
        # 年次成長率の計算
        growth_rates = []
        for i in range(1, len(past_years)):
            current_rev = past_years.iloc[i]['revenue']
            prev_rev = past_years.iloc[i-1]['revenue']
            if current_rev and prev_rev and prev_rev > 0:
                growth_rate = (current_rev - prev_rev) / prev_rev
                growth_rates.append(growth_rate)
        
        if len(growth_rates) < 3:
            return None
        
        # 成長率の安定性（標準偏差の逆数）
        std_growth = np.std(growth_rates)
        return 1 / (1 + abs(std_growth)) if std_growth is not None else None
    
    def _calculate_market_share_trend(self, row: pd.Series, company_data: pd.DataFrame) -> Optional[float]:
        """市場シェアトレンド（売上成長率で代理）"""
        # 実際の市場シェアデータがない場合、売上成長率で代理
        prev_data = self._get_previous_year_data(row, company_data)
        if prev_data is None:
            return None
        
        return self._calculate_growth_rate(row.get('revenue'), prev_data.get('revenue'))
    
    def _calculate_debt_service_coverage(self, row: pd.Series) -> Optional[float]:
        """債務返済能力"""
        operating_cf = self._estimate_operating_cf(row)
        interest_expense = row.get('interest_expenses', 0)
        
        if operating_cf is None or interest_expense == 0:
            return None
        
        return operating_cf / interest_expense
    
    def _calculate_interest_coverage(self, row: pd.Series) -> Optional[float]:
        """インタレストカバレッジレシオ"""
        operating_profit = row.get('operating_profit', 0)
        interest_expense = row.get('interest_expenses', 1)
        return self._safe_divide(operating_profit, interest_expense)
    
    # 新設企業分析用の高度なメソッド
    
    def _calculate_initial_capital_adequacy(self, row: pd.Series, lifecycle_data: Dict) -> Optional[float]:
        """初期資本充足度"""
        current_assets = row.get('total_assets', 0)
        founding_year = lifecycle_data.get('founding_year')
        current_year = row.get('year')
        
        if founding_year and current_year:
            years_since_founding = current_year - founding_year
            if years_since_founding <= 5:  # 設立5年以内
                # 総資産の成長率で初期資本の妥当性を判断
                return min(current_assets / (years_since_founding * 1000000), 2.0)  # 正規化
        
        return None
    
    def _calculate_revenue_scaling_rate(self, row: pd.Series, company_data: pd.DataFrame) -> Optional[float]:
        """売上拡大率（スタートアップ用）"""
        # 過去3年間の売上成長率の加速度
        current_year = row.get('year')
        recent_years = company_data[company_data['year'].between(current_year-3, current_year)]
        
        if len(recent_years) < 3:
            return None
        
        growth_rates = []
        for i in range(1, len(recent_years)):
            current_rev = recent_years.iloc[i]['revenue']
            prev_rev = recent_years.iloc[i-1]['revenue']
            if current_rev and prev_rev and prev_rev > 0:
                growth_rates.append((current_rev - prev_rev) / prev_rev)
        
        if len(growth_rates) < 2:
            return None
        
        # 成長率の加速度（2次微分的概念）
        if len(growth_rates) >= 2:
            acceleration = growth_rates[-1] - growth_rates[-2]
            return acceleration
        
        return None
    
    # 事業継承分析用の高度なメソッド
    
    def _calculate_synergy_realization_rate(self, row: pd.Series, company_data: pd.DataFrame) -> Optional[float]:
        """シナジー実現率"""
        # M&A・統合後の収益性改善度で測定
        current_year = row.get('year')
        
        # 過去3年との比較で統合効果を測定
        past_data = company_data[company_data['year'] == current_year - 3]
        if past_data.empty:
            return None
        
        past_row = past_data.iloc[0]
        current_margin = self._safe_divide(row.get('operating_profit'), row.get('revenue'))
        past_margin = self._safe_divide(past_row.get('operating_profit'), past_row.get('revenue'))
        
        if current_margin is not None and past_margin is not None and past_margin > 0:
            return (current_margin - past_margin) / past_margin
        
        return None
    
    def _calculate_integration_speed(self, row: pd.Series, company_data: pd.DataFrame) -> Optional[float]:
        """統合速度（コスト統合の進捗度）"""
        # 販管費率の改善速度で測定
        prev_data = self._get_previous_year_data(row, company_data)
        if prev_data is None:
            return None
        
        current_sga_ratio = self._safe_divide(row.get('sg_a_expenses'), row.get('revenue'))
        prev_sga_ratio = self._safe_divide(prev_data.get('sg_a_expenses'), prev_data.get('revenue'))
        
        if current_sga_ratio is not None and prev_sga_ratio is not None and prev_sga_ratio > 0:
            improvement = (prev_sga_ratio - current_sga_ratio) / prev_sga_ratio
            return max(improvement, 0)  # マイナスの場合は0
        
        return None
    
    # その他の複雑な計算メソッド群
    
    def _calculate_competitive_moat_strength(self, row: pd.Series, company_data: pd.DataFrame) -> Optional[float]:
        """競争優位性強度"""
        # 複数指標の合成：R&D比率、営業利益率、市場シェア安定性
        rd_ratio = self._safe_divide(row.get('rd_expenses', 0), row.get('revenue', 1)) or 0
        operating_margin = self._safe_divide(row.get('operating_profit', 0), row.get('revenue', 1)) or 0
        revenue_stability = self._calculate_revenue_growth_stability(row, company_data) or 0
        
        # 重み付き平均
        moat_strength = (rd_ratio * 0.4 + operating_margin * 0.4 + revenue_stability * 0.2)
        return min(moat_strength, 1.0)  # 0-1に正規化
    
    def _calculate_business_model_flexibility(self, row: pd.Series, company_data: pd.DataFrame) -> Optional[float]:
        """ビジネスモデル柔軟性"""
        # 変動費比率が高く、固定費比率が低いほど柔軟性が高い
        variable_cost_ratio = self._calculate_variable_cost_ratio(row) or 0
        fixed_cost_ratio = self._calculate_fixed_cost_ratio(row) or 0
        
        # 柔軟性 = 変動費比率 - 固定費比率（正規化）
        flexibility = variable_cost_ratio - fixed_cost_ratio
        return (flexibility + 1) / 2  # -1〜1を0〜1に正規化
    
    def _calculate_technology_adaptation_rate(self, row: pd.Series, company_data: pd.DataFrame) -> Optional[float]:
        """技術適応率"""
        # R&D投資の増加傾向と無形固定資産の蓄積率
        current_year = row.get('year')
        past_3_years = company_data[company_data['year'].between(current_year-3, current_year-1)]
        
        if len(past_3_years) < 2:
            return None
        
        # R&D投資の平均成長率
        rd_growth_rates = []
        for i in range(1, len(past_3_years)):
            current_rd = past_3_years.iloc[i]['rd_expenses']
            prev_rd = past_3_years.iloc[i-1]['rd_expenses']
            if current_rd and prev_rd and prev_rd > 0:
                rd_growth_rates.append((current_rd - prev_rd) / prev_rd)
        
        if rd_growth_rates:
            avg_rd_growth = np.mean(rd_growth_rates)
            return max(0, min(avg_rd_growth, 1.0))  # 0-1に正規化
        
        return None
    
    def _calculate_market_diversification(self, row: pd.Series) -> Optional[float]:
        """市場多様化度"""
        overseas_ratio = row.get('overseas_revenue_ratio', 0)
        segments_count = row.get('segments_count', 1)
        
        # 海外比率とセグメント数から多様化度を計算
        geographic_diversification = min(overseas_ratio, 0.8) / 0.8  # 海外80%で最大
        business_diversification = min(segments_count / 5, 1.0)  # 5セグメントで最大
        
        return (geographic_diversification + business_diversification) / 2
    
    def _calculate_innovation_frequency(self, row: pd.Series, company_data: pd.DataFrame) -> Optional[float]:
        """イノベーション頻度"""
        # R&D投資の一貫性と無形資産の蓄積パターン
        current_year = row.get('year')
        past_5_years = company_data[company_data['year'].between(current_year-5, current_year)]
        
        if len(past_5_years) < 4:
            return None
        
        # R&D投資の継続性（年次投資率の標準偏差の逆数）
        rd_ratios = []
        for _, past_row in past_5_years.iterrows():
            revenue = past_row.get('revenue', 1)
            rd_expense = past_row.get('rd_expenses', 0)
            if revenue > 0:
                rd_ratios.append(rd_expense / revenue)
        
        if len(rd_ratios) >= 3:
            rd_consistency = 1 / (1 + np.std(rd_ratios))  # 一貫性が高いほど高スコア
            avg_rd_ratio = np.mean(rd_ratios)
            return (rd_consistency * 0.5) + (min(avg_rd_ratio * 10, 1.0) * 0.5)  # 正規化
        
        return None
    
    def _calculate_customer_acquisition_efficiency(self, row: pd.Series, company_data: pd.DataFrame) -> Optional[float]:
        """顧客獲得効率性"""
        # 広告費対売上高効率の改善トレンド
        advertising_expense = row.get('advertising_expenses', 0)
        revenue = row.get('revenue', 1)
        
        if advertising_expense == 0:
            return None
        
        # 広告費効率性（売上/広告費）
        ad_efficiency = self._safe_divide(revenue, advertising_expense)
        
        # 過去との比較でトレンドを評価
        prev_data = self._get_previous_year_data(row, company_data)
        if prev_data is not None:
            prev_ad_efficiency = self._safe_divide(prev_data.get('revenue'), prev_data.get('advertising_expenses'))
            if prev_ad_efficiency and prev_ad_efficiency > 0:
                efficiency_improvement = (ad_efficiency - prev_ad_efficiency) / prev_ad_efficiency
                return max(0, min(efficiency_improvement + 0.5, 1.0))  # 正規化
        
        return None
    
    def _calculate_unit_economics_health(self, row: pd.Series, company_data: pd.DataFrame) -> Optional[float]:
        """ユニットエコノミクス健全性"""
        # 従業員一人当たりの利益創出能力
        employees = row.get('employees', 1)
        operating_profit = row.get('operating_profit', 0)
        
        profit_per_employee = self._safe_divide(operating_profit, employees)
        
        if profit_per_employee is None:
            return None
        
        # 業界標準と比較（仮に500万円/人を標準とする）
        standard_profit_per_employee = 5000000
        health_score = profit_per_employee / standard_profit_per_employee
        
        return min(health_score, 2.0) / 2.0  # 0-1に正規化
    
    def _calculate_succession_planning_maturity(self, row: pd.Series, lifecycle_data: Dict) -> Optional[float]:
        """継承計画成熟度"""
        # 分社化・統合イベントからの経過年数と安定性
        succession_event_year = lifecycle_data.get('succession_event_year')
        current_year = row.get('year')
        
        if succession_event_year and current_year:
            years_since_succession = current_year - succession_event_year
            if years_since_succession >= 0:
                # 継承後の安定化度（5年で完全安定と仮定）
                return min(years_since_succession / 5, 1.0)
        
        return None
    
    def _calculate_knowledge_transfer_efficiency(self, row: pd.Series, company_data: pd.DataFrame) -> Optional[float]:
        """知識移転効率性"""
        # R&D継続性と人材維持率の複合指標
        rd_ratio = self._safe_divide(row.get('rd_expenses', 0), row.get('revenue', 1)) or 0
        
        # 従業員数の安定性（離職率の逆指標）
        prev_data = self._get_previous_year_data(row, company_data)
        if prev_data is not None:
            employee_retention = self._calculate_growth_rate(row.get('employees'), prev_data.get('employees'))
            if employee_retention is not None:
                # 適度な成長（-10%〜+20%）が望ましい
                retention_score = 1 - abs(employee_retention) if -0.1 <= employee_retention <= 0.2 else 0.5
                return (rd_ratio * 10 * 0.6) + (retention_score * 0.4)  # 重み付き合成
        
        return rd_ratio * 10  # R&D比率のみで代替
    
    # 全ての推定メソッドの基本実装
    
    def _calculate_goodwill_growth(self, row: pd.Series, prev_row: pd.Series) -> Optional[float]:
        """のれん増加率推定"""
        # 無形固定資産の増加でのれんを代理
        return self._calculate_growth_rate(row.get('intangible_fixed_assets'), prev_row.get('intangible_fixed_assets'))
    
    def _calculate_personnel_cost_growth(self, row: pd.Series, prev_row: pd.Series) -> Optional[float]:
        """人件費増加率推定"""
        current_personnel_cost = (row.get('employees', 0) * row.get('average_salary', 0))
        prev_personnel_cost = (prev_row.get('employees', 0) * prev_row.get('average_salary', 0))
        return self._calculate_growth_rate(current_personnel_cost, prev_personnel_cost)
    
    def _calculate_retirement_cost_growth(self, row: pd.Series, prev_row: pd.Series) -> Optional[float]:
        """退職給付費用増加率推定"""
        current_cost = self._estimate_retirement_benefit_cost(row)
        prev_cost = self._estimate_retirement_benefit_cost(prev_row)
        return self._calculate_growth_rate(current_cost, prev_cost)
    
    def _calculate_overseas_ratio_change(self, row: pd.Series, prev_row: pd.Series) -> Optional[float]:
        """海外売上高比率変化"""
        current_ratio = row.get('overseas_revenue_ratio', 0)
        prev_ratio = prev_row.get('overseas_revenue_ratio', 0)
        return current_ratio - prev_ratio  # 絶対変化量
    
    def _calculate_segment_revenue_growth(self, row: pd.Series, prev_row: pd.Series) -> Optional[float]:
        """セグメント別売上高増加率推定"""
        # セグメント数の変化と売上成長率の複合
        revenue_growth = self._calculate_growth_rate(row.get('revenue'), prev_row.get('revenue')) or 0
        segment_change = (row.get('segments_count', 1) - prev_row.get('segments_count', 1))
        
        # セグメント拡大による成長効果を加算
        segment_growth_effect = segment_change * 0.1  # セグメント1つ追加で10%効果と仮定
        return revenue_growth + segment_growth_effect
    
    def _calculate_non_operating_income_growth(self, row: pd.Series, prev_row: pd.Series) -> Optional[float]:
        """営業外収益増加率"""
        current_non_op = self._calculate_non_operating_income(row)
        prev_non_op = self._calculate_non_operating_income(prev_row)
        return self._calculate_growth_rate(current_non_op, prev_non_op)
    
    def _calculate_turnover_change(self, row: pd.Series, prev_row: pd.Series, turnover_type: str) -> Optional[float]:
        """回転率変化計算"""
        if turnover_type == 'receivables':
            current_turnover = self._calculate_receivables_turnover(row)
            prev_turnover = self._calculate_receivables_turnover(prev_row)
        elif turnover_type == 'inventory':
            current_turnover = self._calculate_inventory_turnover(row)
            prev_turnover = self._calculate_inventory_turnover(prev_row)
        elif turnover_type == 'total_assets':
            current_turnover = self._calculate_total_asset_turnover(row)
            prev_turnover = self._calculate_total_asset_turnover(prev_row)
        else:
            return None
        
        return self._calculate_growth_rate(current_turnover, prev_turnover)
    
    # 残りの推定メソッド群（簡略実装）
    
    def _calculate_sga_personnel_ratio(self, row: pd.Series) -> Optional[float]:
        """販管費人件費率推定"""
        sg_a = row.get('sg_a_expenses', 0)
        return sg_a * 0.4 if sg_a > 0 else None  # 販管費の40%が人件費と仮定
    
    def _calculate_sga_depreciation_ratio(self, row: pd.Series) -> Optional[float]:
        """販管費減価償却費率推定"""
        sg_a = row.get('sg_a_expenses', 0)
        return sg_a * 0.1 if sg_a > 0 else None  # 販管費の10%が減価償却費と仮定
    
    def _calculate_fixed_asset_turnover(self, row: pd.Series) -> Optional[float]:
        """有形固定資産回転率"""
        return self._safe_divide(row.get('revenue'), row.get('tangible_fixed_assets'))
    
    def _calculate_internal_retention_ratio(self, row: pd.Series) -> Optional[float]:
        """内部留保率"""
        payout_ratio = row.get('dividend_payout_ratio', 0)
        return 1 - payout_ratio if payout_ratio is not None else None
    
    def _calculate_non_operating_income_ratio(self, row: pd.Series) -> Optional[float]:
        """営業外収益率"""
        non_op_income = self._calculate_non_operating_income(row)
        return self._safe_divide(non_op_income, row.get('revenue'))
    
    def _calculate_extraordinary_impact_ratio(self, row: pd.Series) -> Optional[float]:
        """特別損益影響率"""
        extraordinary_net = (row.get('extraordinary_profit', 0) - row.get('extraordinary_loss', 0))
        net_profit = row.get('net_profit', 1)
        return self._safe_divide(extraordinary_net, net_profit)
    
    def _calculate_pretax_margin(self, row: pd.Series) -> Optional[float]:
        """税引前当期純利益率"""
        net_profit = row.get('net_profit', 0)
        tax_rate = row.get('tax_rate', 0.3)  # デフォルト30%
        pretax_profit = net_profit / (1 - tax_rate) if tax_rate < 1 else net_profit
        return self._safe_divide(pretax_profit, row.get('revenue'))
    
    def _calculate_interest_bearing_debt_ratio(self, row: pd.Series) -> Optional[float]:
        """有利子負債比率推定"""
        interest_expense = row.get('interest_expenses', 0)
        estimated_debt = interest_expense / 0.02 if interest_expense > 0 else 0
        return self._safe_divide(estimated_debt, row.get('total_assets'))
    
    def _calculate_investment_securities_gain_loss(self, row: pd.Series) -> Optional[float]:
        """投資有価証券評価損益推定"""
        # 営業外収益の一部として推定
        non_op_income = self._calculate_non_operating_income(row)
        interest_income = row.get('interest_income', 0)
        investment_gain = (non_op_income or 0) - interest_income
        return self._safe_divide(investment_gain, row.get('revenue'))
    
    def _calculate_fixed_asset_disposal_gain_loss(self, row: pd.Series) -> Optional[float]:
        """固定資産売却損益推定"""
        # 特別損益の一部として推定
        extraordinary_net = (row.get('extraordinary_profit', 0) - row.get('extraordinary_loss', 0))
        # 固定資産売却は特別損益の50%と仮定
        disposal_gain_loss = extraordinary_net * 0.5
        return self._safe_divide(disposal_gain_loss, row.get('revenue'))
    
    def _calculate_impairment_loss_ratio(self, row: pd.Series) -> Optional[float]:
        """減損損失率推定"""
        extraordinary_loss = row.get('extraordinary_loss', 0)
        # 特別損失の30%が減損と仮定
        impairment_loss = extraordinary_loss * 0.3
        return self._safe_divide(impairment_loss, row.get('revenue'))
    
    def _calculate_equity_method_income(self, row: pd.Series) -> Optional[float]:
        """持分法投資損益推定"""
        investment_securities = row.get('investment_securities', 0)
        # 投資有価証券の3%が年間収益と仮定
        estimated_equity_income = investment_securities * 0.03
        return self._safe_divide(estimated_equity_income, row.get('revenue'))
    
    def _calculate_tax_adjustment(self, row: pd.Series) -> Optional[float]:
        """法人税等調整額推定"""
        # 実効税率と標準税率の差分から推定
        actual_tax_rate = row.get('tax_rate', 0.3)
        standard_tax_rate = 0.3
        tax_adjustment_ratio = actual_tax_rate - standard_tax_rate
        return tax_adjustment_ratio
    
    # 高度な複合指標計算メソッド群
    
    def _calculate_maturity_impact(self, row: pd.Series, company_data: pd.DataFrame) -> Optional[float]:
        """成熟度影響指標"""
        company_age = self._calculate_company_age(row, company_data)
        if company_age is None:
            return None
        
        # S字カーブ：若い企業は高成長、成熟企業は安定成長
        maturity_factor = 1 / (1 + np.exp(-0.1 * (20 - company_age)))  # シグモイド関数
        return maturity_factor
    
    def _calculate_innovation_cycle(self, row: pd.Series, company_data: pd.DataFrame) -> Optional[float]:
        """イノベーションサイクル指標"""
        # R&D投資の周期性分析
        current_year = row.get('year')
        past_7_years = company_data[company_data['year'].between(current_year-7, current_year)]
        
        if len(past_7_years) < 5:
            return None
        
        rd_ratios = []
        for _, past_row in past_7_years.iterrows():
            rd_ratio = self._safe_divide(past_row.get('rd_expenses', 0), past_row.get('revenue', 1))
            if rd_ratio is not None:
                rd_ratios.append(rd_ratio)
        
        if len(rd_ratios) >= 5:
            # FFTによる周期性検出（簡易版）
            rd_series = np.array(rd_ratios)
            fft_result = np.fft.fft(rd_series)
            dominant_frequency = np.argmax(np.abs(fft_result[1:len(fft_result)//2])) + 1
            cycle_strength = np.abs(fft_result[dominant_frequency]) / len(rd_series)
            return min(cycle_strength, 1.0)
        
        return None
    
    def _calculate_market_expansion_rate(self, row: pd.Series, company_data: pd.DataFrame) -> Optional[float]:
        """市場拡大率指標"""
        # 海外売上比率の拡大速度
        current_year = row.get('year')
        past_3_years = company_data[company_data['year'].between(current_year-3, current_year)]
        
        if len(past_3_years) < 3:
            return None
        
        overseas_ratios = [r.get('overseas_revenue_ratio', 0) for _, r in past_3_years.iterrows()]
        
        # 線形回帰での傾き（拡大率）
        if len(overseas_ratios) >= 3:
            years = list(range(len(overseas_ratios)))
            slope = np.polyfit(years, overseas_ratios, 1)[0]  # 1次の係数（傾き）
            return max(0, min(slope * 10, 1.0))  # 正規化
        
        return None
    
    def _calculate_competitive_position(self, row: pd.Series, company_data: pd.DataFrame) -> Optional[float]:
        """競争ポジション指標"""
        # 営業利益率と規模の複合指標
        operating_margin = self._safe_divide(row.get('operating_profit', 0), row.get('revenue', 1)) or 0
        revenue_scale = row.get('revenue', 0)
        
        # 業界標準規模を1000億円と仮定
        scale_factor = min(revenue_scale / 100000000000, 1.0)  # 0-1正規化
        
        return (operating_margin * 5 * 0.7) + (scale_factor * 0.3)  # 重み付き合成
    
    def _calculate_operational_leverage(self, row: pd.Series, company_data: pd.DataFrame) -> Optional[float]:
        """オペレーティングレバレッジ"""
        # 売上変動に対する営業利益の変動倍率
        prev_data = self._get_previous_year_data(row, company_data)
        if prev_data is None:
            return None
        
        revenue_change = self._calculate_growth_rate(row.get('revenue'), prev_data.get('revenue'))
        operating_profit_change = self._calculate_growth_rate(row.get('operating_profit'), prev_data.get('operating_profit'))
        
        if revenue_change and revenue_change != 0 and operating_profit_change is not None:
            leverage = operating_profit_change / revenue_change
            return max(0, min(leverage / 5, 1.0))  # 正規化
        
        return None
    
    def _calculate_cost_structure_flexibility(self, row: pd.Series, company_data: pd.DataFrame) -> Optional[float]:
        """コスト構造柔軟性"""
        variable_ratio = self._calculate_variable_cost_ratio(row) or 0
        fixed_ratio = self._calculate_fixed_cost_ratio(row) or 0
        
        # 変動費比率が高いほど柔軟性が高い
        flexibility = variable_ratio / (variable_ratio + fixed_ratio) if (variable_ratio + fixed_ratio) > 0 else 0.5
        return flexibility
    
    # 残りの複雑な指標（簡易実装）
    
    def _calculate_financial_leverage_impact(self, row: pd.Series, company_data: pd.DataFrame) -> Optional[float]:
        """財務レバレッジ影響"""
        debt_equity_ratio = self._calculate_debt_equity_ratio(row) or 0
        return min(debt_equity_ratio, 2.0) / 2.0  # 0-1正規化
    
    def _calculate_risk_management_efficiency(self, row: pd.Series, company_data: pd.DataFrame) -> Optional[float]:
        """リスク管理効率性"""
        # キャッシュフロー安定性とデフォルトリスクの逆指標
        cf_stability = self._calculate_cash_flow_stability(row, company_data) or 0.5
        interest_coverage = self._calculate_interest_coverage(row) or 1
        risk_score = (cf_stability * 0.6) + (min(interest_coverage / 10, 1.0) * 0.4)
        return risk_score
    
    def _calculate_capital_allocation_efficiency(self, row: pd.Series, company_data: pd.DataFrame) -> Optional[float]:
        """資本配分効率性"""
        # ROEと資産回転率の複合指標
        roe = self._calculate_roe(row) or 0
        asset_turnover = self._calculate_total_asset_turnover(row) or 0
        return (min(roe * 5, 1.0) * 0.6) + (min(asset_turnover, 1.0) * 0.4)
    
    # さらに多くの推定メソッド（基本実装）
    
    def _calculate_shareholder_return_efficiency(self, row: pd.Series, company_data: pd.DataFrame) -> Optional[float]:
        """株主還元効率性"""
        dividend_payout = row.get('dividend_payout_ratio', 0)
        roe = self._calculate_roe(row) or 0
        # 適度な配当性向（30-60%）が効率的
        optimal_payout = 0.3 <= dividend_payout <= 0.6
        payout_score = 1.0 if optimal_payout else 0.5
        return (payout_score * 0.4) + (min(roe * 5, 1.0) * 0.6)
    
    def _calculate_growth_sustainability(self, row: pd.Series, company_data: pd.DataFrame) -> Optional[float]:
        """成長持続可能性"""
        internal_retention = self._calculate_internal_retention_ratio(row) or 0
        roe = self._calculate_roe(row) or 0
        sustainable_growth = internal_retention * roe
        return min(sustainable_growth * 10, 1.0)  # 正規化
    
    def _calculate_capital_efficiency_trend(self, row: pd.Series, company_data: pd.DataFrame) -> Optional[float]:
        """資本効率性トレンド"""
        current_year = row.get('year')
        past_3_years = company_data[company_data['year'].between(current_year-3, current_year)]
        
        if len(past_3_years) < 3:
            return None
        
        roe_values = []
        for _, past_row in past_3_years.iterrows():
            roe = self._calculate_roe(past_row)
            if roe is not None:
                roe_values.append(roe)
        
        if len(roe_values) >= 3:
            # 線形回帰の傾き
            years = list(range(len(roe_values)))
            slope = np.polyfit(years, roe_values, 1)[0]
            return max(0, min(slope * 20 + 0.5, 1.0))  # 正規化
        
        return None
    
    # 新設企業・イノベーション関連の複雑な指標
    
    def _calculate_innovation_intensity(self, row: pd.Series, company_data: pd.DataFrame) -> Optional[float]:
        """イノベーション強度"""
        rd_ratio = self._safe_divide(row.get('rd_expenses', 0), row.get('revenue', 1)) or 0
        intangible_ratio = self._safe_divide(row.get('intangible_fixed_assets', 0), row.get('total_assets', 1)) or 0
        return (rd_ratio * 10 * 0.6) + (intangible_ratio * 0.4)  # 重み付き合成
    
    def _calculate_differentiation_degree(self, row: pd.Series, company_data: pd.DataFrame) -> Optional[float]:
        """差別化度"""
        operating_margin = self._safe_divide(row.get('operating_profit', 0), row.get('revenue', 1)) or 0
        rd_ratio = self._safe_divide(row.get('rd_expenses', 0), row.get('revenue', 1)) or 0
        # 高い利益率とR&D投資の組み合わせで差別化度を測定
        return (operating_margin * 5 * 0.7) + (rd_ratio * 10 * 0.3)
    
    def _calculate_value_chain_integration(self, row: pd.Series, company_data: pd.DataFrame) -> Optional[float]:
        """バリューチェーン統合度"""
        # 外注比率の逆指標として統合度を測定
        outsourcing_ratio = self._calculate_outsourcing_ratio(row) or 0.2
        integration_degree = 1 - outsourcing_ratio
        return max(0, integration_degree)
    
    # 生存分析の高度な指標
    
    def _calculate_profit_growth_stability(self, row: pd.Series, company_data: pd.DataFrame) -> Optional[float]:
        """利益成長安定性"""
        current_year = row.get('year')
        past_5_years = company_data[company_data['year'].between(current_year-5, current_year)]
        
        if len(past_5_years) < 4:
            return None
        
        profit_growth_rates = []
        for i in range(1, len(past_5_years)):
            current_profit = past_5_years.iloc[i]['operating_profit']
            prev_profit = past_5_years.iloc[i-1]['operating_profit']
            if current_profit and prev_profit and prev_profit > 0:
                growth_rate = (current_profit - prev_profit) / prev_profit
                profit_growth_rates.append(growth_rate)
        
        if len(profit_growth_rates) >= 3:
            stability = 1 / (1 + np.std(profit_growth_rates))
            return stability
        
        return None
    
    # 残りのメソッド実装（基本的な推定ロジック）
    
    def _calculate_high_value_segment_ratio(self, row: pd.Series) -> Optional[float]:
        """高付加価値事業セグメント比率推定"""
        operating_margin = self._safe_divide(row.get('operating_profit', 0), row.get('revenue', 1)) or 0
        # 営業利益率15%以上を高付加価値と仮定
        if operating_margin >= 0.15:
            return 0.8  # 高付加価値比率80%
        elif operating_margin >= 0.1:
            return 0.5  # 中付加価値比率50%
        else:
            return 0.2  # 低付加価値比率20%
    
    def _calculate_service_revenue_ratio(self, row: pd.Series) -> Optional[float]:
        """サービス・保守収入比率推定"""
        # 営業外収益に含まれるサービス収入を推定
        non_op_income = self._calculate_non_operating_income(row) or 0
        revenue = row.get('revenue', 1)
        # 営業外収益の一部をサービス収入として推定
        service_ratio = min(non_op_income / revenue * 5, 0.5)  # 最大50%
        return service_ratio
    
    def _calculate_intangible_asset_ratio(self, row: pd.Series) -> Optional[float]:
        """ブランド・商標等無形資産比率"""
        intangible_assets = row.get('intangible_fixed_assets', 0)
        total_assets = row.get('total_assets', 1)
        return self._safe_divide(intangible_assets, total_assets)
    
    def _calculate_patent_related_cost(self, row: pd.Series) -> Optional[float]:
        """特許関連費用推定"""
        rd_expenses = row.get('rd_expenses', 0)
        # R&D費の20%が特許関連と推定
        return rd_expenses * 0.2 if rd_expenses > 0 else None
    
    def _calculate_software_ratio(self, row: pd.Series) -> Optional[float]:
        """ソフトウェア比率推定"""
        intangible_assets = row.get('intangible_fixed_assets', 0)
        total_assets = row.get('total_assets', 1)
        # 無形固定資産の70%がソフトウェアと推定
        software_assets = intangible_assets * 0.7
        return self._safe_divide(software_assets, total_assets)
    
    def _calculate_technology_license_income(self, row: pd.Series) -> Optional[float]:
        """技術ライセンス収入推定"""
        non_op_income = self._calculate_non_operating_income(row) or 0
        revenue = row.get('revenue', 1)
        # 営業外収益の30%がライセンス収入と推定
        license_income = non_op_income * 0.3
        return self._safe_divide(license_income, revenue)
    
    def _calculate_salary_industry_ratio(self, row: pd.Series) -> Optional[float]:
        """平均年間給与/業界平均比率推定"""
        avg_salary = row.get('average_salary', 0)
        # 業界平均を600万円と仮定
        industry_average = 6000000
        return self._safe_divide(avg_salary, industry_average)
    
    def _calculate_personnel_cost_ratio(self, row: pd.Series) -> Optional[float]:
        """人件費率推定"""
        employees = row.get('employees', 0)
        avg_salary = row.get('average_salary', 0)
        revenue = row.get('revenue', 1)
        
        if employees > 0 and avg_salary > 0 and revenue > 0:
            total_personnel_cost = employees * avg_salary
            return total_personnel_cost / revenue
        return None
    
    def _calculate_employee_productivity_ratio(self, row: pd.Series) -> Optional[float]:
        """従業員数/売上高比率（生産性の逆指標）"""
        employees = row.get('employees', 0)
        revenue = row.get('revenue', 1)
        return self._safe_divide(employees, revenue / 1000000)  # 売上高百万円当たり従業員数
    
    def _calculate_retirement_benefit_ratio(self, row: pd.Series) -> Optional[float]:
        """退職給付費用率"""
        retirement_cost = self._estimate_retirement_benefit_cost(row)
        revenue = row.get('revenue', 1)
        return self._safe_divide(retirement_cost, revenue)
    
    def _calculate_welfare_cost_ratio(self, row: pd.Series) -> Optional[float]:
        """福利厚生費率"""
        welfare_cost = self._estimate_welfare_cost(row)
        revenue = row.get('revenue', 1)
        return self._safe_divide(welfare_cost, revenue)
    
    def _calculate_fixed_ratio(self, row: pd.Series) -> Optional[float]:
        """固定比率"""
        fixed_assets = row.get('tangible_fixed_assets', 0) + row.get('intangible_fixed_assets', 0)
        equity = row.get('shareholders_equity', 1)
        return self._safe_divide(fixed_assets, equity)
    
    # 生存分析の複雑な指標群
    
    def _calculate_brand_value_proxy(self, row: pd.Series, company_data: pd.DataFrame) -> Optional[float]:
        """ブランド価値代理指標"""
        # 無形資産比率と営業利益率の複合
        intangible_ratio = self._calculate_intangible_asset_ratio(row) or 0
        operating_margin = self._safe_divide(row.get('operating_profit', 0), row.get('revenue', 1)) or 0
        advertising_ratio = self._safe_divide(row.get('advertising_expenses', 0), row.get('revenue', 1)) or 0
        
        # ブランド価値 = 無形資産 + 収益性 + 広告投資
        brand_proxy = (intangible_ratio * 0.4) + (operating_margin * 5 * 0.4) + (advertising_ratio * 10 * 0.2)
        return min(brand_proxy, 1.0)
    
    def _calculate_customer_loyalty_proxy(self, row: pd.Series, company_data: pd.DataFrame) -> Optional[float]:
        """顧客ロイヤルティ代理指標"""
        # 売上の安定性と収益性の複合
        revenue_stability = self._calculate_revenue_growth_stability(row, company_data) or 0.5
        operating_margin = self._safe_divide(row.get('operating_profit', 0), row.get('revenue', 1)) or 0
        
        # 高い利益率 + 安定売上 = 高いロイヤルティ
        loyalty_proxy = (revenue_stability * 0.6) + (min(operating_margin * 5, 1.0) * 0.4)
        return loyalty_proxy
    
    def _calculate_supply_chain_resilience(self, row: pd.Series, company_data: pd.DataFrame) -> Optional[float]:
        """サプライチェーン強靭性"""
        # 外注比率の逆指標と在庫回転率
        outsourcing_ratio = self._calculate_outsourcing_ratio(row) or 0.3
        inventory_turnover = self._calculate_inventory_turnover(row) or 5
        
        # 内製化率高 + 適度な在庫回転 = 強靭性高
        integration_score = 1 - outsourcing_ratio
        inventory_efficiency = min(inventory_turnover / 10, 1.0)
        
        resilience = (integration_score * 0.6) + (inventory_efficiency * 0.4)
        return resilience
    
    def _calculate_regulatory_compliance_score(self, row: pd.Series, company_data: pd.DataFrame) -> Optional[float]:
        """規制適応スコア"""
        # 特別損失の少なさ + 海外展開度（規制環境適応）
        extraordinary_loss_ratio = self._safe_divide(row.get('extraordinary_loss', 0), row.get('revenue', 1)) or 0
        overseas_ratio = row.get('overseas_revenue_ratio', 0)
        
        # 特別損失が少ない + 海外展開 = 規制適応力高
        compliance_score = (1 - min(extraordinary_loss_ratio * 10, 1.0)) * 0.7 + overseas_ratio * 0.3
        return compliance_score
    
    def _calculate_company_age_survival_curve(self, row: pd.Series, lifecycle_data: Dict) -> Optional[float]:
        """企業年齢×生存曲線"""
        founding_year = lifecycle_data.get('founding_year')
        current_year = row.get('year')
        
        if founding_year and current_year:
            age = current_year - founding_year
            # ワイブル分布による生存曲線（形状パラメータ=1.5、尺度パラメータ=50）
            survival_probability = np.exp(-((age / 50) ** 1.5))
            return survival_probability
        
        return None
    
    def _calculate_crisis_resistance_capacity(self, row: pd.Series, company_data: pd.DataFrame) -> Optional[float]:
        """危機耐性能力"""
        # 現金比率 + 収益性 + 財務安定性
        cash_ratio = self._safe_divide(row.get('cash_and_deposits', 0), row.get('total_assets', 1)) or 0
        operating_margin = self._safe_divide(row.get('operating_profit', 0), row.get('revenue', 1)) or 0
        equity_ratio = self._calculate_equity_ratio(row) or 0
        
        crisis_resistance = (cash_ratio * 0.4) + (operating_margin * 5 * 0.3) + (equity_ratio * 0.3)
        return min(crisis_resistance, 1.0)
    
    def _calculate_strategic_agility(self, row: pd.Series, company_data: pd.DataFrame) -> Optional[float]:
        """戦略的俊敏性"""
        # R&D比率 + 事業多様化 + コスト柔軟性
        rd_ratio = self._safe_divide(row.get('rd_expenses', 0), row.get('revenue', 1)) or 0
        diversification = self._calculate_market_diversification(row) or 0
        cost_flexibility = self._calculate_cost_structure_flexibility(row, company_data) or 0
        
        agility = (rd_ratio * 10 * 0.4) + (diversification * 0.3) + (cost_flexibility * 0.3)
        return min(agility, 1.0)
    
    # 新設企業分析の複雑な指標群
    
    def _calculate_market_timing_score(self, row: pd.Series, lifecycle_data: Dict) -> Optional[float]:
        """市場タイミングスコア"""
        founding_year = lifecycle_data.get('founding_year')
        current_year = row.get('year')
        
        if founding_year and current_year:
            years_since_founding = current_year - founding_year
            
            # 設立後の成長軌道評価
            if years_since_founding <= 5:
                # 設立初期は急成長が期待される
                revenue = row.get('revenue', 0)
                expected_revenue = years_since_founding * 500000000  # 年5億円成長を期待
                timing_score = min(revenue / expected_revenue, 2.0) / 2.0 if expected_revenue > 0 else 0
                return timing_score
        
        return None
    
    def _calculate_technology_novelty_degree(self, row: pd.Series, company_data: pd.DataFrame) -> Optional[float]:
        """技術新規性度"""
        # R&D集約度と特許投資比率
        rd_ratio = self._safe_divide(row.get('rd_expenses', 0), row.get('revenue', 1)) or 0
        patent_cost_ratio = self._safe_divide(self._calculate_patent_related_cost(row), row.get('revenue', 1)) or 0
        
        # 技術新規性 = 高R&D投資 + 特許投資
        novelty_degree = (rd_ratio * 5 * 0.7) + (patent_cost_ratio * 20 * 0.3)
        return min(novelty_degree, 1.0)
    
    def _calculate_market_size_potential(self, row: pd.Series, lifecycle_data: Dict) -> Optional[float]:
        """市場規模ポテンシャル"""
        # 海外展開度と事業拡張性
        overseas_ratio = row.get('overseas_revenue_ratio', 0)
        segments_count = row.get('segments_count', 1)
        
        # 大きな市場への参入可能性
        geographic_potential = overseas_ratio  # 海外=大市場
        business_expansion_potential = min(segments_count / 3, 1.0)  # 複数事業展開
        
        market_potential = (geographic_potential * 0.6) + (business_expansion_potential * 0.4)
        return market_potential
    
    def _calculate_product_market_fit_proxy(self, row: pd.Series, company_data: pd.DataFrame) -> Optional[float]:
        """プロダクトマーケットフィット代理指標"""
        # 売上成長と顧客獲得効率の組み合わせ
        revenue_growth = 0
        prev_data = self._get_previous_year_data(row, company_data)
        if prev_data is not None:
            revenue_growth = self._calculate_growth_rate(row.get('revenue'), prev_data.get('revenue')) or 0
        
        customer_efficiency = self._calculate_customer_acquisition_efficiency(row, company_data) or 0.5
        
        # 高成長 + 効率的顧客獲得 = PMF達成
        pmf_score = (min(revenue_growth * 2, 1.0) * 0.6) + (customer_efficiency * 0.4)
        return pmf_score
    
    def _calculate_viral_coefficient_proxy(self, row: pd.Series, company_data: pd.DataFrame) -> Optional[float]:
        """バイラル係数代理指標"""
        # 広告費効率と口コミ効果の代理指標
        advertising_efficiency = self._calculate_customer_acquisition_efficiency(row, company_data) or 0.5
        
        # 広告費が少ないのに成長している = バイラル効果
        ad_ratio = self._safe_divide(row.get('advertising_expenses', 0), row.get('revenue', 1)) or 0.05
        
        # 低広告費 + 高効率 = バイラル効果高
        viral_proxy = advertising_efficiency * (1 - min(ad_ratio * 20, 1.0))
        return viral_proxy
    
    def _calculate_funding_efficiency(self, row: pd.Series, company_data: pd.DataFrame) -> Optional[float]:
        """資金調達効率性"""
        # 総資産の成長に対する収益性の向上度
        prev_data = self._get_previous_year_data(row, company_data)
        if prev_data is None:
            return None
        
        asset_growth = self._calculate_growth_rate(row.get('total_assets'), prev_data.get('total_assets')) or 0
        roe_current = self._calculate_roe(row) or 0
        roe_prev = self._calculate_roe(prev_data) or 0
        
        if asset_growth > 0:
            funding_efficiency = (roe_current - roe_prev) / asset_growth
            return max(0, min(funding_efficiency + 0.5, 1.0))  # 正規化
        
        return None
    
    def _calculate_talent_acquisition_rate(self, row: pd.Series, company_data: pd.DataFrame) -> Optional[float]:
        """人材獲得率"""
        employee_growth = 0
        prev_data = self._get_previous_year_data(row, company_data)
        if prev_data is not None:
            employee_growth = self._calculate_growth_rate(row.get('employees'), prev_data.get('employees')) or 0
        
        # 従業員数成長率を人材獲得率とする
        return max(0, min(employee_growth, 1.0))
    
    def _calculate_rd_investment_intensity(self, row: pd.Series, company_data: pd.DataFrame) -> Optional[float]:
        """R&D投資強度"""
        rd_ratio = self._safe_divide(row.get('rd_expenses', 0), row.get('revenue', 1)) or 0
        
        # スタートアップは高R&D比率が期待される（15%以上で満点）
        intensity = min(rd_ratio / 0.15, 1.0)
        return intensity
    
    def _calculate_infrastructure_investment_ratio(self, row: pd.Series, company_data: pd.DataFrame) -> Optional[float]:
        """インフラ投資比率"""
        capex_ratio = self._safe_divide(row.get('capex', 0), row.get('revenue', 1)) or 0
        intangible_investment = self._safe_divide(row.get('intangible_fixed_assets', 0), row.get('total_assets', 1)) or 0
        
        # 設備投資 + 無形資産投資 = インフラ投資
        infrastructure_ratio = capex_ratio + intangible_investment
        return min(infrastructure_ratio, 1.0)
    
    def _calculate_partnership_leverage(self, row: pd.Series, company_data: pd.DataFrame) -> Optional[float]:
        """パートナーシップ活用度"""
        # 外注比率をパートナーシップ活用の代理指標とする
        outsourcing_ratio = self._calculate_outsourcing_ratio(row) or 0
        
        # 適度な外注（20-40%）が最適なパートナーシップ活用
        if 0.2 <= outsourcing_ratio <= 0.4:
            leverage_score = 1.0
        else:
            leverage_score = 1.0 - abs(outsourcing_ratio - 0.3) / 0.3
        
        return max(0, leverage_score)
    
    def _calculate_market_penetration_speed(self, row: pd.Series, company_data: pd.DataFrame) -> Optional[float]:
        """市場浸透速度"""
        # 売上成長率と市場シェア拡大の複合指標
        revenue_growth = 0
        prev_data = self._get_previous_year_data(row, company_data)
        if prev_data is not None:
            revenue_growth = self._calculate_growth_rate(row.get('revenue'), prev_data.get('revenue')) or 0
        
        overseas_expansion = self._calculate_overseas_ratio_change(row, prev_data) if prev_data else 0
        
        # 売上成長 + 地理的拡大 = 市場浸透速度
        penetration_speed = (min(revenue_growth, 1.0) * 0.7) + (min(abs(overseas_expansion) * 5, 1.0) * 0.3)
        return penetration_speed
    
    def _calculate_competitive_differentiation(self, row: pd.Series, company_data: pd.DataFrame) -> Optional[float]:
        """競争差別化度"""
        return self._calculate_differentiation_degree(row, company_data)  # 既存メソッドを流用
    
    def _calculate_pricing_strategy_effectiveness(self, row: pd.Series, company_data: pd.DataFrame) -> Optional[float]:
        """価格戦略効果性"""
        # 営業利益率と売上成長率のバランス
        operating_margin = self._safe_divide(row.get('operating_profit', 0), row.get('revenue', 1)) or 0
        
        revenue_growth = 0
        prev_data = self._get_previous_year_data(row, company_data)
        if prev_data is not None:
            revenue_growth = self._calculate_growth_rate(row.get('revenue'), prev_data.get('revenue')) or 0
        
        # 利益性と成長性のバランス
        effectiveness = (operating_margin * 5 * 0.5) + (min(revenue_growth, 1.0) * 0.5)
        return min(effectiveness, 1.0)
    
    def _calculate_channel_development_rate(self, row: pd.Series, company_data: pd.DataFrame) -> Optional[float]:
        """チャネル開発率"""
        # セグメント拡大と海外展開の複合
        segments_growth = 0
        prev_data = self._get_previous_year_data(row, company_data)
        if prev_data is not None:
            segments_growth = (row.get('segments_count', 1) - prev_data.get('segments_count', 1))
        
        overseas_expansion = self._calculate_overseas_ratio_change(row, prev_data) if prev_data else 0
        
        channel_development = (max(0, segments_growth) * 0.5) + (min(abs(overseas_expansion) * 5, 1.0) * 0.5)
        return min(channel_development, 1.0)
    
    def _calculate_brand_recognition_speed(self, row: pd.Series, company_data: pd.DataFrame) -> Optional[float]:
        """ブランド認知速度"""
        # 広告投資効率と売上成長の複合
        advertising_efficiency = self._calculate_customer_acquisition_efficiency(row, company_data) or 0.5
        
        brand_investment_ratio = self._safe_divide(row.get('advertising_expenses', 0), row.get('revenue', 1)) or 0
        
        # 適度な広告投資 + 高効率 = ブランド認知速度向上
        recognition_speed = (advertising_efficiency * 0.6) + (min(brand_investment_ratio * 20, 1.0) * 0.4)
        return recognition_speed
    
    def _calculate_ecosystem_integration_rate(self, row: pd.Series, company_data: pd.DataFrame) -> Optional[float]:
        """エコシステム統合率"""
        # パートナーシップ + 技術統合の複合指標
        partnership_leverage = self._calculate_partnership_leverage(row, company_data) or 0
        technology_integration = self._calculate_software_ratio(row) or 0
        
        integration_rate = (partnership_leverage * 0.6) + (technology_integration * 0.4)
        return integration_rate
    
    def _calculate_scalability_potential(self, row: pd.Series, company_data: pd.DataFrame) -> Optional[float]:
        """スケーラビリティポテンシャル"""
        # 変動費構造 + デジタル化度
        variable_cost_ratio = self._calculate_variable_cost_ratio(row) or 0.5
        digital_ratio = self._calculate_software_ratio(row) or 0
        
        # 変動費が高い + デジタル化 = スケーラブル
        scalability = (variable_cost_ratio * 0.5) + (digital_ratio * 0.5)
        return scalability
    
    def _calculate_resilience_building_rate(self, row: pd.Series, company_data: pd.DataFrame) -> Optional[float]:
        """強靭性構築率"""
        # 財務安定性の向上速度
        current_equity_ratio = self._calculate_equity_ratio(row) or 0
        
        prev_data = self._get_previous_year_data(row, company_data)
        if prev_data is not None:
            prev_equity_ratio = self._calculate_equity_ratio(prev_data) or 0
            resilience_improvement = current_equity_ratio - prev_equity_ratio
            return max(0, min(resilience_improvement * 10, 1.0))  # 正規化
        
        return current_equity_ratio  # 前年データなしの場合は現在値
    
    # 事業継承分析の複雑な指標群（残り）
    
    def _calculate_organizational_continuity(self, row: pd.Series, company_data: pd.DataFrame) -> Optional[float]:
        """組織継続性"""
        # 従業員維持率と給与安定性
        employee_retention = 1.0  # デフォルト
        salary_stability = 1.0
        
        prev_data = self._get_previous_year_data(row, company_data)
        if prev_data is not None:
            employee_change = self._calculate_growth_rate(row.get('employees'), prev_data.get('employees')) or 0
            salary_change = self._calculate_growth_rate(row.get('average_salary'), prev_data.get('average_salary')) or 0
            
            # 適度な変化（-5%〜+15%）が継続性を示す
            employee_retention = 1.0 if -0.05 <= employee_change <= 0.15 else 0.7
            salary_stability = 1.0 if -0.02 <= salary_change <= 0.10 else 0.7
        
        continuity = (employee_retention * 0.6) + (salary_stability * 0.4)
        return continuity
    
    def _calculate_culture_preservation_score(self, row: pd.Series, company_data: pd.DataFrame) -> Optional[float]:
        """企業文化保存スコア"""
        # R&D投資継続性と従業員定着の複合
        rd_consistency = self._calculate_innovation_frequency(row, company_data) or 0.5
        org_continuity = self._calculate_organizational_continuity(row, company_data) or 0.5
        
        culture_score = (rd_consistency * 0.5) + (org_continuity * 0.5)
        return culture_score
    
    def _calculate_stakeholder_alignment(self, row: pd.Series, company_data: pd.DataFrame) -> Optional[float]:
        """ステークホルダー整合性"""
        # 配当政策 + 従業員処遇 + 顧客満足（代理指標）
        dividend_consistency = 1.0
        employee_treatment = self._calculate_organizational_continuity(row, company_data) or 0.5
        customer_satisfaction = self._calculate_customer_loyalty_proxy(row, company_data) or 0.5
        
        # 配当性向の安定性チェック
        prev_data = self._get_previous_year_data(row, company_data)
        if prev_data is not None:
            dividend_change = abs((row.get('dividend_payout_ratio', 0) - prev_data.get('dividend_payout_ratio', 0)))
            dividend_consistency = 1.0 if dividend_change <= 0.1 else 0.7  # 10%以内の変化が安定
        
        alignment = (dividend_consistency * 0.3) + (employee_treatment * 0.4) + (customer_satisfaction * 0.3)
        return alignment
    
    def _calculate_cost_integration_efficiency(self, row: pd.Series, company_data: pd.DataFrame) -> Optional[float]:
        """コスト統合効率性"""
        return self._calculate_integration_speed(row, company_data)  # 既存メソッドを流用
    
    def _calculate_revenue_retention_rate(self, row: pd.Series, company_data: pd.DataFrame) -> Optional[float]:
        """売上維持率"""
        # 売上の安定性（変動の少なさ）
        revenue_stability = self._calculate_revenue_growth_stability(row, company_data) or 0.5
        
        # 売上成長が安定している = 維持率が高い
        return revenue_stability
    
    def _calculate_customer_retention_during_transition(self, row: pd.Series, company_data: pd.DataFrame) -> Optional[float]:
        """移行期顧客維持率"""
        # 売上債権の安定性で顧客維持を推定
        current_receivables_turnover = self._calculate_receivables_turnover(row)
        
        prev_data = self._get_previous_year_data(row, company_data)
        if prev_data is not None and current_receivables_turnover is not None:
            prev_receivables_turnover = self._calculate_receivables_turnover(prev_data)
            if prev_receivables_turnover is not None and prev_receivables_turnover > 0:
                turnover_stability = 1 - abs((current_receivables_turnover - prev_receivables_turnover) / prev_receivables_turnover)
                return max(0, turnover_stability)
        
        return 0.8  # デフォルト値
    
    def _calculate_operational_disruption_minimization(self, row: pd.Series, company_data: pd.DataFrame) -> Optional[float]:
        """業務中断最小化"""
        # 営業効率の維持度
        current_asset_turnover = self._calculate_total_asset_turnover(row)
        
        prev_data = self._get_previous_year_data(row, company_data)
        if prev_data is not None and current_asset_turnover is not None:
            prev_asset_turnover = self._calculate_total_asset_turnover(prev_data)
            if prev_asset_turnover is not None and prev_asset_turnover > 0:
                efficiency_maintenance = current_asset_turnover / prev_asset_turnover
                return min(efficiency_maintenance, 1.0)
        
        return 0.9  # デフォルト値
    
    def _calculate_duplicate_elimination_efficiency(self, row: pd.Series, company_data: pd.DataFrame) -> Optional[float]:
        """重複排除効率性"""
        # 販管費率の改善度
        current_sga_ratio = self._safe_divide(row.get('sg_a_expenses', 0), row.get('revenue', 1)) or 0
        
        prev_data = self._get_previous_year_data(row, company_data)
        if prev_data is not None:
            prev_sga_ratio = self._safe_divide(prev_data.get('sg_a_expenses', 0), prev_data.get('revenue', 1)) or 0
            if prev_sga_ratio > 0:
                sga_improvement = (prev_sga_ratio - current_sga_ratio) / prev_sga_ratio
                return max(0, min(sga_improvement, 1.0))
        
        return 0.5  # デフォルト値
    
    def _calculate_best_practice_diffusion(self, row: pd.Series, company_data: pd.DataFrame) -> Optional[float]:
        """ベストプラクティス拡散"""
        # 営業利益率の改善トレンド
        current_year = row.get('year')
        past_3_years = company_data[company_data['year'].between(current_year-3, current_year)]
        
        if len(past_3_years) >= 3:
            operating_margins = []
            for _, past_row in past_3_years.iterrows():
                margin = self._safe_divide(past_row.get('operating_profit', 0), past_row.get('revenue', 1))
                if margin is not None:
                    operating_margins.append(margin)
            
            if len(operating_margins) >= 3:
                # 線形回帰の傾き（改善トレンド）
                years = list(range(len(operating_margins)))
                slope = np.polyfit(years, operating_margins, 1)[0]
                diffusion_rate = max(0, min(slope * 20 + 0.5, 1.0))  # 正規化
                return diffusion_rate
        
        return 0.5  # デフォルト値
    
    def _calculate_scale_economy_realization(self, row: pd.Series, company_data: pd.DataFrame) -> Optional[float]:
        """規模経済実現度"""
        # 売上規模拡大に対する単位コスト削減効果
        revenue_scale = row.get('revenue', 0)
        cost_ratio = self._safe_divide(row.get('cost_of_sales', 0), row.get('revenue', 1)) or 0
        
        prev_data = self._get_previous_year_data(row, company_data)
        if prev_data is not None:
            prev_revenue = prev_data.get('revenue', 1)
            prev_cost_ratio = self._safe_divide(prev_data.get('cost_of_sales', 0), prev_data.get('revenue', 1)) or 0
            
            if prev_revenue > 0 and prev_cost_ratio > 0:
                revenue_growth = (revenue_scale - prev_revenue) / prev_revenue
                cost_reduction = (prev_cost_ratio - cost_ratio) / prev_cost_ratio
                
                # 売上成長 + コスト率削減 = 規模経済実現
                if revenue_growth > 0:
                    scale_economy = cost_reduction / revenue_growth
                    return max(0, min(scale_economy + 0.5, 1.0))  # 正規化
        
        return 0.5  # デフォルト値
    
    def _calculate_scope_economy_realization(self, row: pd.Series, company_data: pd.DataFrame) -> Optional[float]:
        """範囲経済実現度"""
        # 事業セグメント拡大に対する間接費効率化
        segments_count = row.get('segments_count', 1)
        sga_ratio = self._safe_divide(row.get('sg_a_expenses', 0), row.get('revenue', 1)) or 0
        
        prev_data = self._get_previous_year_data(row, company_data)
        if prev_data is not None:
            prev_segments = prev_data.get('segments_count', 1)
            prev_sga_ratio = self._safe_divide(prev_data.get('sg_a_expenses', 0), prev_data.get('revenue', 1)) or 0
            
            if prev_segments > 0 and prev_sga_ratio > 0:
                segment_expansion = (segments_count - prev_segments) / prev_segments
                sga_efficiency = (prev_sga_ratio - sga_ratio) / prev_sga_ratio
                
                # セグメント拡大 + 販管費効率化 = 範囲経済実現
                if segment_expansion > 0:
                    scope_economy = sga_efficiency / segment_expansion
                    return max(0, min(scope_economy + 0.5, 1.0))  # 正規化
        
        return 0.5  # デフォルト値
    
    def _calculate_strategic_vision_continuity(self, row: pd.Series, company_data: pd.DataFrame) -> Optional[float]:
        """戦略ビジョン継続性"""
        # R&D投資方針と事業展開の一貫性
        rd_consistency = self._calculate_innovation_frequency(row, company_data) or 0.5
        
        # セグメント戦略の安定性
        segments_stability = 1.0
        prev_data = self._get_previous_year_data(row, company_data)
        if prev_data is not None:
            segment_change = abs(row.get('segments_count', 1) - prev_data.get('segments_count', 1))
            segments_stability = 1.0 if segment_change <= 1 else 0.7  # 大幅変更は継続性に影響
        
        vision_continuity = (rd_consistency * 0.6) + (segments_stability * 0.4)
        return vision_continuity
    
    def _calculate_innovation_capability_enhancement(self, row: pd.Series, company_data: pd.DataFrame) -> Optional[float]:
        """イノベーション能力向上"""
        # R&D投資の拡大トレンド
        rd_ratio_current = self._safe_divide(row.get('rd_expenses', 0), row.get('revenue', 1)) or 0
        
        prev_data = self._get_previous_year_data(row, company_data)
        if prev_data is not None:
            rd_ratio_prev = self._safe_divide(prev_data.get('rd_expenses', 0), prev_data.get('revenue', 1)) or 0
            if rd_ratio_prev > 0:
                rd_enhancement = (rd_ratio_current - rd_ratio_prev) / rd_ratio_prev
                return max(0, min(rd_enhancement + 0.5, 1.0))  # 正規化
        
        return rd_ratio_current * 10  # 現在値で代替
    
    def _calculate_market_position_strengthening(self, row: pd.Series, company_data: pd.DataFrame) -> Optional[float]:
        """市場ポジション強化"""
        # 市場シェア（売上成長率で代理）と収益性の複合向上
        revenue_growth = 0
        margin_improvement = 0
        
        prev_data = self._get_previous_year_data(row, company_data)
        if prev_data is not None:
            revenue_growth = self._calculate_growth_rate(row.get('revenue'), prev_data.get('revenue')) or 0
            
            current_margin = self._safe_divide(row.get('operating_profit', 0), row.get('revenue', 1)) or 0
            prev_margin = self._safe_divide(prev_data.get('operating_profit', 0), prev_data.get('revenue', 1)) or 0
            if prev_margin > 0:
                margin_improvement = (current_margin - prev_margin) / prev_margin
        
        # シェア拡大 + 収益性向上 = ポジション強化
        position_strengthening = (min(revenue_growth, 1.0) * 0.5) + (max(0, min(margin_improvement + 0.5, 1.0)) * 0.5)
        return position_strengthening
    
    def _calculate_competitive_advantage_amplification(self, row: pd.Series, company_data: pd.DataFrame) -> Optional[float]:
        """競争優位性増幅"""
        # 差別化度の向上トレンド
        current_differentiation = self._calculate_differentiation_degree(row, company_data) or 0
        
        # 過去との比較（簡易版）
        current_year = row.get('year')
        past_2_years = company_data[company_data['year'].between(current_year-2, current_year-1)]
        
        if not past_2_years.empty:
            past_row = past_2_years.iloc[-1]  # 直近過去データ
            past_differentiation = self._calculate_differentiation_degree(past_row, company_data) or 0
            
            if past_differentiation > 0:
                amplification = (current_differentiation - past_differentiation) / past_differentiation
                return max(0, min(amplification + 0.5, 1.0))  # 正規化
        
        return current_differentiation  # 現在値で代替
    
    def _calculate_growth_opportunity_expansion(self, row: pd.Series, company_data: pd.DataFrame) -> Optional[float]:
        """成長機会拡大"""
        # 事業領域拡大と地理的拡大の複合
        market_expansion = self._calculate_market_expansion_rate(row, company_data) or 0
        business_expansion = 0
        
        prev_data = self._get_previous_year_data(row, company_data)
        if prev_data is not None:
            segment_growth = (row.get('segments_count', 1) - prev_data.get('segments_count', 1))
            business_expansion = max(0, min(segment_growth / 2, 1.0))  # 2セグメント増で満点
        
        opportunity_expansion = (market_expansion * 0.6) + (business_expansion * 0.4)
        return opportunity_expansion
    
    def _calculate_next_generation_readiness(self, row: pd.Series, lifecycle_data: Dict) -> Optional[float]:
        """次世代準備度"""
        # デジタル投資と人材投資の複合
        digital_readiness = self._calculate_software_ratio(row) or 0
        talent_investment = self._safe_divide(row.get('average_salary', 0), 8000000) or 0.5  # 800万円で1.0
        rd_investment = self._safe_divide(row.get('rd_expenses', 0), row.get('revenue', 1)) or 0
        
        next_gen_readiness = (digital_readiness * 0.4) + (min(talent_investment, 1.0) * 0.3) + (rd_investment * 10 * 0.3)
        return min(next_gen_readiness, 1.0)
    
    def _calculate_legacy_value_preservation(self, row: pd.Series, company_data: pd.DataFrame) -> Optional[float]:
        """レガシー価値保存"""
        # ブランド価値と技術資産の維持
        brand_value = self._calculate_brand_value_proxy(row, company_data) or 0
        intangible_preservation = self._calculate_intangible_asset_ratio(row) or 0
        
        legacy_preservation = (brand_value * 0.6) + (intangible_preservation * 0.4)
        return legacy_preservation
    
    def _calculate_transformation_success_rate(self, row: pd.Series, company_data: pd.DataFrame) -> Optional[float]:
        """変革成功率"""
        # 革新性と安定性のバランス
        innovation_rate = self._calculate_innovation_capability_enhancement(row, company_data) or 0.5
        stability = self._calculate_organizational_continuity(row, company_data) or 0.5
        
        # 適度な革新 + 組織安定性 = 変革成功
        transformation_success = (innovation_rate * 0.6) + (stability * 0.4)
        return transformation_success
    
    def get_factor_names(self) -> Dict[str, List[str]]:
        """
        各評価項目の要因項目名リストを取得
        
        Returns:
            要因項目名の辞書
        """
        factor_names = {
            'revenue': [
                '有形固定資産', '設備投資額', '研究開発費', '無形固定資産', '投資有価証券',
                '従業員数', '平均年間給与', '退職給付費用', '福利厚生費', '総還元性向',
                '売上債権', '棚卸資産', '総資産', '売上債権回転率', '棚卸資産回転率',
                '海外売上高比率', '事業セグメント数', '販売費及び一般管理費', '広告宣伝費', '営業外収益',
                '企業年齢', '市場参入時期', '親会社依存度'
            ],
            'growth': [
                '設備投資増加率', '研究開発費増加率', '有形固定資産増加率', '無形固定資産増加率', '総資産増加率', 'のれん増加率',
                '従業員数増加率', '平均年間給与増加率', '人件費増加率', '退職給付費用増加率',
                '海外売上高比率変化', 'セグメント別売上高増加率', '販管費増加率', '広告宣伝費増加率', '営業外収益増加率',
                '売上債権増加率', '棚卸資産増加率', '売上債権回転率変化', '棚卸資産回転率変化', '総資産回転率変化',
                '成熟度影響', 'イノベーションサイクル', '市場拡大率'
            ],
            'operating_margin': [
                '材料費率', '労務費率', '経費率', '外注加工費率', '減価償却費率（製造原価）',
                '販管費率', '人件費率（販管費）', '広告宣伝費率', '研究開発費率', '減価償却費率（販管費）',
                '売上高付加価値率', '労働生産性', '設備効率性', '総資産回転率', '棚卸資産回転率',
                '売上高（規模効果）', '固定費率', '変動費率', '海外売上高比率', '事業セグメント集中度',
                '競争ポジション', 'オペレーティングレバレッジ', 'コスト構造柔軟性'
            ],
            'net_margin': [
                '売上高営業利益率', '販管費率', '売上原価率', '研究開発費率', '減価償却費率',
                '受取利息・配当金', '支払利息', '為替差損益', '持分法投資損益', '営業外収益率',
                '特別利益', '特別損失', '法人税等実効税率', '法人税等調整額', '税引前当期純利益率',
                '有利子負債比率', '自己資本比率', '投資有価証券評価損益', '固定資産売却損益', '減損損失率',
                '財務レバレッジ影響', 'リスク管理効率性', '資本配分効率性'
            ],
            'roe': [
                '売上高当期純利益率', '総資産回転率', '売上高営業利益率', '売上原価率', '販管費率',
                '自己資本比率', '総資産/自己資本倍率', '有利子負債/自己資本比率', '流動比率', '固定比率',
                '売上債権回転率', '棚卸資産回転率', '有形固定資産回転率', '現預金比率', '投資有価証券比率',
                '配当性向', '内部留保率', '営業外収益率', '特別損益影響率', '実効税率',
                '株主還元効率性', '成長持続可能性', '資本効率性トレンド'
            ],
            'value_added': [
                '研究開発費率', '無形固定資産比率', '特許関連費用', 'ソフトウェア比率', '技術ライセンス収入',
                '平均年間給与/業界平均比率', '人件費率', '従業員数/売上高比率', '退職給付費用率', '福利厚生費率',
                '売上原価率（逆数）', '材料費率（逆数）', '外注加工費率（逆数）', '労働生産性', '設備生産性',
                '海外売上高比率', '高付加価値事業セグメント比率', 'サービス・保守収入比率', '営業利益率', 'ブランド・商標等無形資産比率',
                'イノベーション強度', '差別化度', 'バリューチェーン統合度'
            ],
            'survival': [
                '自己資本比率', '流動比率', '債務返済能力', 'インタレストカバレッジ', 'キャッシュフロー安定性',
                '営業利益率', 'ROE', '売上成長安定性', '利益成長安定性', '市場シェアトレンド',
                'R&D投資率', 'ビジネスモデル柔軟性', '技術適応率', '市場多様化度', 'イノベーション頻度',
                '競争優位性強度', 'ブランド価値代理指標', '顧客ロイヤルティ代理指標', 'サプライチェーン強靭性', '規制適応スコア',
                '企業年齢×生存曲線', '危機耐性能力', '戦略的俊敏性'
            ],
            'emergence': [
                '初期資本充足度', '創業者経験代理指標', '市場タイミングスコア', '技術新規性度', '市場規模ポテンシャル',
                '売上拡大率', '顧客獲得効率性', 'プロダクトマーケットフィット代理指標', 'ユニットエコノミクス健全性', 'バイラル係数代理指標',
                '資金調達効率性', '人材獲得率', 'R&D投資強度', 'インフラ投資比率', 'パートナーシップ活用度',
                '市場浸透速度', '競争差別化度', '価格戦略効果性', 'チャネル開発率', 'ブランド認知速度',
                'エコシステム統合率', 'スケーラビリティポテンシャル', '強靭性構築率'
            ],
            'succession': [
                '継承計画成熟度', '知識移転効率性', '組織継続性', '企業文化保存スコア', 'ステークホルダー整合性',
                'シナジー実現率', 'コスト統合効率性', '売上維持率', '移行期顧客維持率', '業務中断最小化',
                '統合速度', '重複排除効率性', 'ベストプラクティス拡散', '規模経済実現度', '範囲経済実現度',
                '戦略ビジョン継続性', 'イノベーション能力向上', '市場ポジション強化', '競争優位性増幅', '成長機会拡大',
                '次世代準備度', 'レガシー価値保存', '変革成功率'
            ]
        }
        
        return factor_names
    
    def validate_factors(self, factors_df: pd.DataFrame) -> Dict[str, any]:
        """
        計算された要因項目の妥当性検証
        
        Args:
            factors_df: 計算済み要因項目のDataFrame
            
        Returns:
            検証結果の辞書
        """
        validation_results = {
            'total_records': len(factors_df),
            'missing_values': {},
            'outliers': {},
            'data_quality_score': 0.0,
            'warnings': []
        }
        
        # 欠損値チェック
        for col in factors_df.columns:
            if col not in ['company_id', 'year']:
                missing_count = factors_df[col].isna().sum()
                missing_rate = missing_count / len(factors_df)
                validation_results['missing_values'][col] = {
                    'count': missing_count,
                    'rate': missing_rate
                }
                
                if missing_rate > 0.5:
                    validation_results['warnings'].append(f"{col}: 欠損率が50%を超えています ({missing_rate:.2%})")
        
        # 外れ値チェック（数値列のみ）
        numeric_columns = factors_df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col not in ['company_id', 'year']:
                Q1 = factors_df[col].quantile(0.25)
                Q3 = factors_df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = factors_df[(factors_df[col] < lower_bound) | (factors_df[col] > upper_bound)][col]
                outlier_rate = len(outliers) / len(factors_df)
                
                validation_results['outliers'][col] = {
                    'count': len(outliers),
                    'rate': outlier_rate,
                    'bounds': (lower_bound, upper_bound)
                }
                
                if outlier_rate > 0.1:
                    validation_results['warnings'].append(f"{col}: 外れ値が10%を超えています ({outlier_rate:.2%})")
        
        # データ品質スコア計算
        total_missing_rate = sum([v['rate'] for v in validation_results['missing_values'].values()]) / len(validation_results['missing_values'])
        total_outlier_rate = sum([v['rate'] for v in validation_results['outliers'].values()]) / len(validation_results['outliers'])
        
        validation_results['data_quality_score'] = max(0, 1.0 - (total_missing_rate * 0.6 + total_outlier_rate * 0.4))
        
        logger.info(f"データ品質検証完了: スコア {validation_results['data_quality_score']:.3f}, 警告 {len(validation_results['warnings'])} 件")
        
        return validation_results


# 使用例とテスト用のメインブロック
if __name__ == "__main__":
    # サンプルデータでのテスト
    sample_data = [
        FinancialData(
            company_id="test_company_1",
            year=2023,
            revenue=100000000000,  # 1000億円
            operating_profit=10000000000,  # 100億円
            net_profit=7000000000,  # 70億円
            total_assets=200000000000,  # 2000億円
            shareholders_equity=80000000000,  # 800億円
            tangible_fixed_assets=50000000000,  # 500億円
            intangible_fixed_assets=10000000000,  # 100億円
            rd_expenses=5000000000,  # 50億円
            employees=10000,
            average_salary=8000000,  # 800万円
            overseas_revenue_ratio=0.6,
            segments_count=3
        ),
        FinancialData(
            company_id="test_company_1",
            year=2022,
            revenue=95000000000,  # 950億円
            operating_profit=9000000000,  # 90億円
            net_profit=6500000000,  # 65億円
            total_assets=190000000000,  # 1900億円
            shareholders_equity=75000000000,  # 750億円
            employees=9500
        )
    ]
    
    # 計算器のインスタンス化とテスト実行
    calculator = FactorCalculator()
    
    try:
        # 要因項目計算
        factors_df = calculator.calculate_all_factors(sample_data)
        print("要因項目計算完了:")
        print(f"- 計算対象レコード数: {len(factors_df)}")
        print(f"- 計算項目数: {len(factors_df.columns) - 2}")  # company_id, yearを除く
        
        # 要因項目名取得
        factor_names = calculator.get_factor_names()
        print(f"- 評価項目数: {len(factor_names)}")
        
        # データ品質検証
        validation = calculator.validate_factors(factors_df)
        print(f"- データ品質スコア: {validation['data_quality_score']:.3f}")
        print(f"- 警告件数: {len(validation['warnings'])}")
        
    except Exception as e:
        logger.error(f"テスト実行中にエラーが発生しました: {e}")
        print(f"エラー: {e}")