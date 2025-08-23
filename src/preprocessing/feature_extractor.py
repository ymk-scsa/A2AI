"""
A2AI (Advanced Financial Analysis AI) - Feature Extractor
=========================================================

企業ライフサイクル全体（存続・消モジュール
滅・新設）から9つの評価項目と
各23の要因項目を体系的に抽出する
対象企業: 150社（高シェア市場50社、シェア低下市場50社、失失市場50社）
評価項目: 9項目（従来6項目 + 企業存続確率、新規事業成功率、事業継承成功度）
要因項目: 各評価項目23項目（従来20項目 + 企業年齢、市場参入時期、親会社依存度）
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from datetime import datetime, date
import warnings
from pathlib import Path

# A2AI内部モジュール
from ..utils.data_utils import (
    safe_divide, handle_missing_data, detect_outliers
)
from ..utils.lifecycle_utils import (
    determine_lifecycle_stage, calculate_company_age
)
from ..utils.math_utils import (
    calculate_growth_rate, calculate_ratio, rolling_average
)

@dataclass
class CompanyInfo:
    """企業基本情報"""
    company_id: str
    company_name: str
    market_category: str  # 'high_share', 'declining', 'lost'
    founded_date: Optional[date]
    extinction_date: Optional[date]  # 消滅日（存続企業はNone）
    spinoff_parent: Optional[str]  # 分社元企業（該当企業のみ）
    listing_status: str  # 'listed', 'delisted', 'subsidiary'

@dataclass
class ExtractionConfig:
    """特徴量抽出設定"""
    start_year: int = 1984
    end_year: int = 2024
    min_data_years: int = 3  # 最小データ年数
    outlier_threshold: float = 3.0  # 外れ値検出閾値（標準偏差）
    missing_threshold: float = 0.7  # 欠損率閾値（70%超で除外）
    smoothing_window: int = 3  # 平滑化ウィンドウ

class FeatureExtractor:
    """
    A2AI特徴量抽出器
    
    企業ライフサイクル全体から9つの評価項目と各23の要因項目を抽出
    """
    
    def __init__(self, config: ExtractionConfig = None):
        self.config = config or ExtractionConfig()
        self.logger = logging.getLogger(__name__)
        
        # 評価項目定義
        self.evaluation_items = [
            'sales_revenue',           # 1. 売上高
            'sales_growth_rate',       # 2. 売上高成長率
            'operating_margin',        # 3. 売上高営業利益率
            'net_margin',             # 4. 売上高当期純利益率
            'roe',                    # 5. ROE
            'value_added_ratio',      # 6. 売上高付加価値率
            'survival_probability',    # 7. 企業存続確率（新規）
            'emergence_success_rate',  # 8. 新規事業成功率（新規）
            'succession_success_rate'  # 9. 事業継承成功度（新規）
        ]
        
        # 要因項目カテゴリ定義
        self.factor_categories = {
            'investment_assets': [
                'tangible_fixed_assets', 'capital_expenditure', 'rd_expenses',
                'intangible_assets', 'investment_securities'
            ],
            'human_resources': [
                'employee_count', 'average_salary', 'retirement_benefit_cost',
                'welfare_expenses'
            ],
            'working_capital': [
                'trade_receivables', 'inventory', 'total_assets',
                'receivables_turnover', 'inventory_turnover'
            ],
            'business_expansion': [
                'overseas_sales_ratio', 'business_segments', 'sga_expenses',
                'advertising_expenses', 'non_operating_income', 'order_backlog'
            ],
            'lifecycle_factors': [  # 新規追加
                'company_age', 'market_entry_timing', 'parent_dependency'
            ]
        }
    
    def extract_all_features(
        self,
        financial_data: pd.DataFrame,
        company_info: CompanyInfo,
        market_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        全特徴量抽出メイン関数
        
        Args:
            financial_data: 財務諸表データ
            company_info: 企業基本情報
            market_data: 市場データ（オプション）
            
        Returns:
            Dict[str, pd.DataFrame]: 評価項目別の要因項目データフレーム
        """
        self.logger.info(f"特徴量抽出開始: {company_info.company_name}")
        
        # データ品質チェック
        if not self._validate_input_data(financial_data, company_info):
            raise ValueError(f"データ品質チェック失敗: {company_info.company_name}")
        
        # 前処理済みデータ作成
        processed_data = self._preprocess_financial_data(financial_data, company_info)
        
        # 各評価項目の要因項目抽出
        extracted_features = {}
        
        for eval_item in self.evaluation_items:
            self.logger.info(f"評価項目抽出: {eval_item}")
            
            try:
                features = self._extract_factors_for_evaluation(
                    eval_item, processed_data, company_info, market_data
                )
                extracted_features[eval_item] = features
                
            except Exception as e:
                self.logger.error(f"評価項目{eval_item}の抽出失敗: {e}")
                extracted_features[eval_item] = pd.DataFrame()
        
        # 品質検証
        validated_features = self._validate_extracted_features(extracted_features)
        
        self.logger.info(f"特徴量抽出完了: {company_info.company_name}")
        return validated_features
    
    def _validate_input_data(self, data: pd.DataFrame, info: CompanyInfo) -> bool:
        """入力データ品質チェック"""
        if data.empty:
            self.logger.error(f"財務データが空: {info.company_name}")
            return False
        
        # 必須カラムチェック
        required_columns = [
            'year', 'sales_revenue', 'operating_profit', 'net_income',
            'total_assets', 'shareholders_equity'
        ]
        
        missing_cols = [col for col in required_columns if col not in data.columns]
        if missing_cols:
            self.logger.error(f"必須カラム不足: {missing_cols}")
            return False
        
        # データ年数チェック
        data_years = len(data['year'].unique())
        if data_years < self.config.min_data_years:
            self.logger.warning(f"データ年数不足: {data_years}年 < {self.config.min_data_years}年")
        
        return True
    
    def _preprocess_financial_data(
        self, 
        data: pd.DataFrame, 
        info: CompanyInfo
    ) -> pd.DataFrame:
        """財務データ前処理"""
        processed = data.copy()
        
        # 年次データソート
        processed = processed.sort_values('year')
        
        # 企業ライフサイクル情報追加
        processed['company_age'] = processed['year'].apply(
            lambda year: calculate_company_age(year, info.founded_date)
        )
        
        processed['lifecycle_stage'] = processed.apply(
            lambda row: determine_lifecycle_stage(
                row['company_age'], 
                info.extinction_date,
                row['year']
            ), axis=1
        )
        
        # 消滅企業の場合、消滅フラグ追加
        if info.extinction_date:
            processed['years_to_extinction'] = processed['year'].apply(
                lambda year: max(0, info.extinction_date.year - year)
            )
            processed['is_extinct'] = processed['year'] >= info.extinction_date.year
        else:
            processed['years_to_extinction'] = np.nan
            processed['is_extinct'] = False
        
        # 分社企業の場合、独立年数計算
        processed['years_since_spinoff'] = 0
        if info.spinoff_parent:
            # 分社年推定（データから推定）
            spinoff_year = processed['year'].min()
            processed['years_since_spinoff'] = processed['year'] - spinoff_year
        
        # 基本財務比率計算
        processed = self._calculate_basic_ratios(processed)
        
        # 成長率計算
        processed = self._calculate_growth_rates(processed)
        
        # 外れ値処理
        processed = self._handle_outliers(processed)
        
        # 欠損値処理
        processed = handle_missing_data(processed, method='interpolate')
        
        return processed
    
    def _calculate_basic_ratios(self, data: pd.DataFrame) -> pd.DataFrame:
        """基本財務比率計算"""
        df = data.copy()
        
        # 基本比率計算
        df['operating_margin'] = safe_divide(df['operating_profit'], df['sales_revenue']) * 100
        df['net_margin'] = safe_divide(df['net_income'], df['sales_revenue']) * 100
        df['roe'] = safe_divide(df['net_income'], df['shareholders_equity']) * 100
        df['total_asset_turnover'] = safe_divide(df['sales_revenue'], df['total_assets'])
        
        # 回転率計算
        if 'trade_receivables' in df.columns:
            df['receivables_turnover'] = safe_divide(df['sales_revenue'], df['trade_receivables'])
        
        if 'inventory' in df.columns:
            df['inventory_turnover'] = safe_divide(df['cost_of_sales'], df['inventory'])
        
        # 付加価値率計算
        if 'value_added' in df.columns:
            df['value_added_ratio'] = safe_divide(df['value_added'], df['sales_revenue']) * 100
        else:
            # 付加価値 = 売上高 - 売上原価 + 人件費
            if 'cost_of_sales' in df.columns and 'personnel_expenses' in df.columns:
                df['value_added'] = (df['sales_revenue'] - df['cost_of_sales'] + 
                                    df.get('personnel_expenses', 0))
                df['value_added_ratio'] = safe_divide(df['value_added'], df['sales_revenue']) * 100
        
        return df
    
    def _calculate_growth_rates(self, data: pd.DataFrame) -> pd.DataFrame:
        """成長率計算"""
        df = data.copy()
        
        # 主要項目の成長率
        growth_items = [
            'sales_revenue', 'operating_profit', 'net_income', 'total_assets',
            'shareholders_equity', 'employee_count', 'rd_expenses'
        ]
        
        for item in growth_items:
            if item in df.columns:
                growth_col = f'{item}_growth_rate'
                df[growth_col] = calculate_growth_rate(df[item])
        
        return df
    
    def _handle_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """外れ値処理"""
        df = data.copy()
        
        # 数値カラムのみ対象
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col not in ['year', 'company_age']:  # 年度・年齢は除外
                outliers = detect_outliers(df[col], threshold=self.config.outlier_threshold)
                if outliers.any():
                    self.logger.info(f"外れ値検出: {col} - {outliers.sum()}件")
                    # 外れ値をNaNに置換（後で補間）
                    df.loc[outliers, col] = np.nan
        
        return df
    
    def _extract_factors_for_evaluation(
        self,
        eval_item: str,
        data: pd.DataFrame,
        info: CompanyInfo,
        market_data: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """評価項目別要因項目抽出"""
        
        if eval_item == 'sales_revenue':
            return self._extract_sales_factors(data, info)
        elif eval_item == 'sales_growth_rate':
            return self._extract_growth_factors(data, info)
        elif eval_item == 'operating_margin':
            return self._extract_margin_factors(data, info)
        elif eval_item == 'net_margin':
            return self._extract_net_margin_factors(data, info)
        elif eval_item == 'roe':
            return self._extract_roe_factors(data, info)
        elif eval_item == 'value_added_ratio':
            return self._extract_value_added_factors(data, info)
        elif eval_item == 'survival_probability':
            return self._extract_survival_factors(data, info)
        elif eval_item == 'emergence_success_rate':
            return self._extract_emergence_factors(data, info)
        elif eval_item == 'succession_success_rate':
            return self._extract_succession_factors(data, info)
        else:
            raise ValueError(f"未知の評価項目: {eval_item}")
    
    def _extract_sales_factors(self, data: pd.DataFrame, info: CompanyInfo) -> pd.DataFrame:
        """売上高の要因項目抽出（23項目）"""
        factors = pd.DataFrame(index=data.index)
        
        # 1-5: 投資・資産関連
        factors['tangible_fixed_assets'] = data.get('tangible_fixed_assets', np.nan)
        factors['capital_expenditure'] = data.get('capital_expenditure', np.nan)
        factors['rd_expenses'] = data.get('rd_expenses', np.nan)
        factors['intangible_assets'] = data.get('intangible_assets', np.nan)
        factors['investment_securities'] = data.get('investment_securities', np.nan)
        
        # 6-9: 人的資源関連
        factors['employee_count'] = data.get('employee_count', np.nan)
        factors['average_salary'] = data.get('average_salary', np.nan)
        factors['retirement_benefit_cost'] = data.get('retirement_benefit_cost', np.nan)
        factors['welfare_expenses'] = data.get('welfare_expenses', np.nan)
        
        # 10-14: 運転資本・効率性関連
        factors['trade_receivables'] = data.get('trade_receivables', np.nan)
        factors['inventory'] = data.get('inventory', np.nan)
        factors['total_assets'] = data.get('total_assets', np.nan)
        factors['receivables_turnover'] = data.get('receivables_turnover', np.nan)
        factors['inventory_turnover'] = data.get('inventory_turnover', np.nan)
        
        # 15-20: 事業展開関連
        factors['overseas_sales_ratio'] = data.get('overseas_sales_ratio', np.nan)
        factors['business_segments'] = data.get('business_segments', np.nan)
        factors['sga_expenses'] = data.get('sga_expenses', np.nan)
        factors['advertising_expenses'] = data.get('advertising_expenses', np.nan)
        factors['non_operating_income'] = data.get('non_operating_income', np.nan)
        factors['order_backlog'] = data.get('order_backlog', np.nan)
        
        # 21-23: ライフサイクル要因（新規）
        factors['company_age'] = data['company_age']
        factors['market_entry_timing'] = self._calculate_market_entry_timing(data, info)
        factors['parent_dependency'] = self._calculate_parent_dependency(data, info)
        
        return factors
    
    def _extract_growth_factors(self, data: pd.DataFrame, info: CompanyInfo) -> pd.DataFrame:
        """売上高成長率の要因項目抽出（23項目）"""
        factors = pd.DataFrame(index=data.index)
        
        # 1-6: 投資・拡張関連
        factors['capex_growth_rate'] = data.get('capital_expenditure_growth_rate', np.nan)
        factors['rd_growth_rate'] = data.get('rd_expenses_growth_rate', np.nan)
        factors['tangible_assets_growth_rate'] = data.get('tangible_fixed_assets_growth_rate', np.nan)
        factors['intangible_assets_growth_rate'] = data.get('intangible_assets_growth_rate', np.nan)
        factors['total_assets_growth_rate'] = data.get('total_assets_growth_rate', np.nan)
        factors['goodwill_impairment_rate'] = data.get('goodwill_impairment_rate', np.nan)
        
        # 7-10: 人的資源拡張
        factors['employee_growth_rate'] = data.get('employee_count_growth_rate', np.nan)
        factors['salary_growth_rate'] = data.get('average_salary_growth_rate', np.nan)
        factors['personnel_cost_growth_rate'] = data.get('personnel_expenses_growth_rate', np.nan)
        factors['retirement_cost_growth_rate'] = data.get('retirement_benefit_cost_growth_rate', np.nan)
        
        # 11-15: 市場・事業拡大
        factors['overseas_ratio_change'] = self._calculate_ratio_change(data, 'overseas_sales_ratio')
        factors['segment_sales_growth'] = data.get('segment_sales_growth_rate', np.nan)
        factors['sga_growth_rate'] = data.get('sga_expenses_growth_rate', np.nan)
        factors['advertising_growth_rate'] = data.get('advertising_expenses_growth_rate', np.nan)
        factors['non_operating_growth_rate'] = data.get('non_operating_income_growth_rate', np.nan)
        
        # 16-20: 効率性・能力関連
        factors['receivables_growth_rate'] = data.get('trade_receivables_growth_rate', np.nan)
        factors['inventory_growth_rate'] = data.get('inventory_growth_rate', np.nan)
        factors['receivables_turnover_change'] = self._calculate_ratio_change(data, 'receivables_turnover')
        factors['inventory_turnover_change'] = self._calculate_ratio_change(data, 'inventory_turnover')
        factors['asset_turnover_change'] = self._calculate_ratio_change(data, 'total_asset_turnover')
        
        # 21-23: ライフサイクル要因（新規）
        factors['company_age'] = data['company_age']
        factors['market_entry_timing'] = self._calculate_market_entry_timing(data, info)
        factors['parent_dependency'] = self._calculate_parent_dependency(data, info)
        
        return factors
    
    def _extract_margin_factors(self, data: pd.DataFrame, info: CompanyInfo) -> pd.DataFrame:
        """売上高営業利益率の要因項目抽出（23項目）"""
        factors = pd.DataFrame(index=data.index)
        
        # 1-5: 売上原価構成
        factors['material_cost_ratio'] = data.get('material_cost_ratio', np.nan)
        factors['labor_cost_ratio'] = data.get('labor_cost_ratio', np.nan)
        factors['overhead_cost_ratio'] = data.get('overhead_cost_ratio', np.nan)
        factors['outsourcing_cost_ratio'] = data.get('outsourcing_cost_ratio', np.nan)
        factors['manufacturing_depreciation_ratio'] = data.get('manufacturing_depreciation_ratio', np.nan)
        
        # 6-10: 販管費構成
        factors['sga_ratio'] = safe_divide(data.get('sga_expenses', 0), data['sales_revenue']) * 100
        factors['sga_personnel_ratio'] = data.get('sga_personnel_ratio', np.nan)
        factors['advertising_ratio'] = safe_divide(data.get('advertising_expenses', 0), data['sales_revenue']) * 100
        factors['rd_ratio'] = safe_divide(data.get('rd_expenses', 0), data['sales_revenue']) * 100
        factors['sga_depreciation_ratio'] = data.get('sga_depreciation_ratio', np.nan)
        
        # 11-15: 効率性指標
        factors['value_added_ratio'] = data.get('value_added_ratio', np.nan)
        factors['labor_productivity'] = safe_divide(data['sales_revenue'], data.get('employee_count', 1))
        factors['asset_productivity'] = safe_divide(data['sales_revenue'], data.get('tangible_fixed_assets', 1))
        factors['total_asset_turnover'] = data.get('total_asset_turnover', np.nan)
        factors['inventory_turnover'] = data.get('inventory_turnover', np.nan)
        
        # 16-20: 規模・構造要因
        factors['sales_scale'] = data['sales_revenue']  # 規模効果
        factors['fixed_cost_ratio'] = data.get('fixed_cost_ratio', np.nan)
        factors['variable_cost_ratio'] = data.get('variable_cost_ratio', np.nan)
        factors['overseas_sales_ratio'] = data.get('overseas_sales_ratio', np.nan)
        factors['business_concentration'] = data.get('business_concentration', np.nan)
        
        # 21-23: ライフサイクル要因（新規）
        factors['company_age'] = data['company_age']
        factors['market_entry_timing'] = self._calculate_market_entry_timing(data, info)
        factors['parent_dependency'] = self._calculate_parent_dependency(data, info)
        
        return factors
    
    def _extract_net_margin_factors(self, data: pd.DataFrame, info: CompanyInfo) -> pd.DataFrame:
        """売上高当期純利益率の要因項目抽出（23項目）"""
        factors = pd.DataFrame(index=data.index)
        
        # 1-5: 営業損益要因
        factors['operating_margin'] = data.get('operating_margin', np.nan)
        factors['sga_ratio'] = safe_divide(data.get('sga_expenses', 0), data['sales_revenue']) * 100
        factors['cogs_ratio'] = safe_divide(data.get('cost_of_sales', 0), data['sales_revenue']) * 100
        factors['rd_ratio'] = safe_divide(data.get('rd_expenses', 0), data['sales_revenue']) * 100
        factors['depreciation_ratio'] = safe_divide(data.get('depreciation', 0), data['sales_revenue']) * 100
        
        # 6-10: 営業外損益
        factors['interest_income_ratio'] = safe_divide(data.get('interest_income', 0), data['sales_revenue']) * 100
        factors['interest_expense_ratio'] = safe_divide(data.get('interest_expense', 0), data['sales_revenue']) * 100
        factors['fx_gain_loss_ratio'] = safe_divide(data.get('fx_gain_loss', 0), data['sales_revenue']) * 100
        factors['equity_method_income'] = data.get('equity_method_income', np.nan)
        factors['non_operating_income_ratio'] = safe_divide(data.get('non_operating_income', 0), data['sales_revenue']) * 100
        
        # 11-15: 特別損益・税金
        factors['extraordinary_gain_ratio'] = safe_divide(data.get('extraordinary_gains', 0), data['sales_revenue']) * 100
        factors['extraordinary_loss_ratio'] = safe_divide(data.get('extraordinary_losses', 0), data['sales_revenue']) * 100
        factors['effective_tax_rate'] = data.get('effective_tax_rate', np.nan)
        factors['tax_adjustment'] = data.get('tax_adjustment', np.nan)
        factors['pretax_margin'] = safe_divide(data.get('pretax_income', 0), data['sales_revenue']) * 100
        
        # 16-20: 財務構造要因
        factors['interest_bearing_debt_ratio'] = data.get('interest_bearing_debt_ratio', np.nan)
        factors['equity_ratio'] = safe_divide(data.get('shareholders_equity', 0), data.get('total_assets', 1)) * 100
        factors['investment_securities_gain_loss'] = data.get('investment_securities_gain_loss', np.nan)
        factors['fixed_asset_gain_loss'] = data.get('fixed_asset_gain_loss', np.nan)
        factors['impairment_loss_ratio'] = safe_divide(data.get('impairment_losses', 0), data['sales_revenue']) * 100
        
        # 21-23: ライフサイクル要因（新規）
        factors['company_age'] = data['company_age']
        factors['market_entry_timing'] = self._calculate_market_entry_timing(data, info)
        factors['parent_dependency'] = self._calculate_parent_dependency(data, info)
        
        return factors
    
    def _extract_roe_factors(self, data: pd.DataFrame, info: CompanyInfo) -> pd.DataFrame:
        """ROEの要因項目抽出（23項目）"""
        factors = pd.DataFrame(index=data.index)
        
        # 1-5: 収益性要因（ROA構成）
        factors['net_margin'] = data.get('net_margin', np.nan)
        factors['total_asset_turnover'] = data.get('total_asset_turnover', np.nan)
        factors['operating_margin'] = data.get('operating_margin', np.nan)
        factors['cogs_ratio'] = safe_divide(data.get('cost_of_sales', 0), data['sales_revenue']) * 100
        factors['sga_ratio'] = safe_divide(data.get('sga_expenses', 0), data['sales_revenue']) * 100
        
        # 6-10: 財務レバレッジ
        factors['equity_ratio'] = safe_divide(data.get('shareholders_equity', 0), data.get('total_assets', 1)) * 100
        factors['financial_leverage'] = safe_divide(data.get('total_assets', 1), data.get('shareholders_equity', 1))
        factors['debt_equity_ratio'] = data.get('debt_equity_ratio', np.nan)
        factors['current_ratio'] = data.get('current_ratio', np.nan)
        factors['fixed_ratio'] = data.get('fixed_ratio', np.nan)
        
        # 11-15: 資産効率性
        factors['receivables_turnover'] = data.get('receivables_turnover', np.nan)
        factors['inventory_turnover'] = data.get('inventory_turnover', np.nan)
        factors['fixed_asset_turnover'] = safe_divide(data['sales_revenue'], data.get('tangible_fixed_assets', 1))
        factors['cash_asset_ratio'] = safe_divide(data.get('cash_and_deposits', 0), data.get('total_assets', 1)) * 100
        factors['investment_securities_ratio'] = safe_divide(data.get('investment_securities', 0), data.get('total_assets', 1)) * 100
        
        # 16-20: 収益・配当政策
        factors['dividend_payout_ratio'] = data.get('dividend_payout_ratio', np.nan)
        factors['retained_earnings_ratio'] = data.get('retained_earnings_ratio', np.nan)
        factors['non_operating_income_ratio'] = safe_divide(data.get('non_operating_income', 0), data['sales_revenue']) * 100
        factors['extraordinary_net_income_ratio'] = data.get('extraordinary_net_income_ratio', np.nan)
        factors['effective_tax_rate'] = data.get('effective_tax_rate', np.nan)
        
        # 21-23: ライフサイクル要因（新規）
        factors['company_age'] = data['company_age']
        factors['market_entry_timing'] = self._calculate_market_entry_timing(data, info)
        factors['parent_dependency'] = self._calculate_parent_dependency(data, info)
        
        return factors
    
    def _extract_value_added_factors(self, data: pd.DataFrame, info: CompanyInfo) -> pd.DataFrame:
        """売上高付加価値率の要因項目抽出（23項目）"""
        factors = pd.DataFrame(index=data.index)
        
        # 1-5: 技術・研究開発
        factors['rd_ratio'] = safe_divide(data.get('rd_expenses', 0), data['sales_revenue']) * 100
        factors['intangible_assets_ratio'] = safe_divide(data.get('intangible_assets', 0), data['sales_revenue']) * 100
        factors['patent_expenses'] = data.get('patent_expenses', np.nan)
        factors['software_ratio'] = safe_divide(data.get('software_assets', 0), data['sales_revenue']) * 100
        factors['technology_license_income'] = data.get('technology_license_income', np.nan)
        
        # 6-10: 人的付加価値
        factors['salary_industry_ratio'] = data.get('salary_industry_ratio', np.nan)
        factors['personnel_cost_ratio'] = safe_divide(data.get('personnel_expenses', 0), data['sales_revenue']) * 100
        factors['employee_sales_ratio'] = safe_divide(data.get('employee_count', 1), data['sales_revenue'])
        factors['retirement_cost_ratio'] = safe_divide(data.get('retirement_benefit_cost', 0), data['sales_revenue']) * 100
        factors['welfare_cost_ratio'] = safe_divide(data.get('welfare_expenses', 0), data['sales_revenue']) * 100
        
        # 11-15: コスト構造・効率性
        factors['cogs_efficiency'] = 100 - safe_divide(data.get('cost_of_sales', 0), data['sales_revenue']) * 100
        factors['material_efficiency'] = 100 - data.get('material_cost_ratio', 0)
        factors['outsourcing_efficiency'] = 100 - data.get('outsourcing_cost_ratio', 0)
        factors['labor_productivity'] = safe_divide(data['sales_revenue'], data.get('employee_count', 1))
        factors['asset_productivity'] = safe_divide(data['sales_revenue'], data.get('tangible_fixed_assets', 1))
        
        # 16-20: 事業構造・差別化
        factors['overseas_sales_ratio'] = data.get('overseas_sales_ratio', np.nan)
        factors['high_value_segment_ratio'] = data.get('high_value_segment_ratio', np.nan)
        factors['service_revenue_ratio'] = data.get('service_revenue_ratio', np.nan)
        factors['operating_margin'] = data.get('operating_margin', np.nan)
        factors['brand_intangible_ratio'] = data.get('brand_intangible_ratio', np.nan)
        
        # 21-23: ライフサイクル要因（新規）
        factors['company_age'] = data['company_age']
        factors['market_entry_timing'] = self._calculate_market_entry_timing(data, info)
        factors['parent_dependency'] = self._calculate_parent_dependency(data, info)
        
        return factors
    
    def _extract_survival_factors(self, data: pd.DataFrame, info: CompanyInfo) -> pd.DataFrame:
        """企業存続確率の要因項目抽出（23項目）"""
        factors = pd.DataFrame(index=data.index)
        
        # 1-5: 財務健全性
        factors['equity_ratio'] = safe_divide(data.get('shareholders_equity', 0), data.get('total_assets', 1)) * 100
        factors['current_ratio'] = data.get('current_ratio', np.nan)
        factors['interest_coverage_ratio'] = data.get('interest_coverage_ratio', np.nan)
        factors['debt_service_coverage'] = data.get('debt_service_coverage', np.nan)
        factors['cash_flow_margin'] = safe_divide(data.get('operating_cash_flow', 0), data['sales_revenue']) * 100
        
        # 6-10: 収益安定性
        factors['earnings_volatility'] = data.get('earnings_volatility', np.nan)
        factors['revenue_stability'] = data.get('revenue_stability', np.nan)
        factors['margin_consistency'] = data.get('margin_consistency', np.nan)
        factors['cash_generation_ability'] = data.get('cash_generation_ability', np.nan)
        factors['dividend_sustainability'] = data.get('dividend_sustainability', np.nan)
        
        # 11-15: 競争優位性
        factors['market_share_stability'] = data.get('market_share_stability', np.nan)
        factors['customer_concentration'] = data.get('customer_concentration', np.nan)
        factors['technological_advantage'] = data.get('technological_advantage', np.nan)
        factors['brand_strength'] = data.get('brand_strength', np.nan)
        factors['switching_cost'] = data.get('switching_cost', np.nan)
        
        # 16-20: 適応能力
        factors['innovation_investment'] = safe_divide(data.get('rd_expenses', 0), data['sales_revenue']) * 100
        factors['organizational_flexibility'] = data.get('organizational_flexibility', np.nan)
        factors['market_diversification'] = data.get('market_diversification', np.nan)
        factors['digital_transformation'] = data.get('digital_transformation', np.nan)
        factors['crisis_response_ability'] = data.get('crisis_response_ability', np.nan)
        
        # 21-23: ライフサイクル要因（新規）
        factors['company_age'] = data['company_age']
        factors['market_entry_timing'] = self._calculate_market_entry_timing(data, info)
        factors['parent_dependency'] = self._calculate_parent_dependency(data, info)
        
        # 消滅企業の場合、消滅までの期間情報追加
        if info.extinction_date:
            factors['years_to_extinction'] = data.get('years_to_extinction', np.nan)
        
        return factors
    
    def _extract_emergence_factors(self, data: pd.DataFrame, info: CompanyInfo) -> pd.DataFrame:
        """新規事業成功率の要因項目抽出（23項目）"""
        factors = pd.DataFrame(index=data.index)
        
        # 新設企業・分社企業のみ対象
        if not (info.founded_date and info.founded_date.year >= 1990) and not info.spinoff_parent:
            # 既存企業の場合は全てNaNで返す
            for i in range(23):
                factors[f'emergence_factor_{i+1}'] = np.nan
            return factors
        
        # 1-5: 初期資本・投資
        factors['initial_capital_intensity'] = data.get('initial_capital_intensity', np.nan)
        factors['early_rd_intensity'] = self._calculate_early_rd_intensity(data)
        factors['startup_investment_ratio'] = data.get('startup_investment_ratio', np.nan)
        factors['founder_investment'] = data.get('founder_investment', np.nan)
        factors['external_funding_ratio'] = data.get('external_funding_ratio', np.nan)
        
        # 6-10: 成長軌道
        factors['early_growth_rate'] = self._calculate_early_growth_rate(data)
        factors['market_penetration_speed'] = data.get('market_penetration_speed', np.nan)
        factors['customer_acquisition_rate'] = data.get('customer_acquisition_rate', np.nan)
        factors['revenue_scaling_efficiency'] = data.get('revenue_scaling_efficiency', np.nan)
        factors['profitability_achievement_speed'] = self._calculate_profitability_speed(data)
        
        # 11-15: イノベーション・差別化
        factors['technology_uniqueness'] = data.get('technology_uniqueness', np.nan)
        factors['patent_portfolio_strength'] = data.get('patent_portfolio_strength', np.nan)
        factors['product_differentiation'] = data.get('product_differentiation', np.nan)
        factors['innovation_frequency'] = data.get('innovation_frequency', np.nan)
        factors['first_mover_advantage'] = self._calculate_first_mover_advantage(data, info)
        
        # 16-20: 組織・人材
        factors['founding_team_experience'] = data.get('founding_team_experience', np.nan)
        factors['talent_acquisition_ability'] = data.get('talent_acquisition_ability', np.nan)
        factors['organizational_learning_speed'] = data.get('organizational_learning_speed', np.nan)
        factors['culture_adaptability'] = data.get('culture_adaptability', np.nan)
        factors['leadership_stability'] = data.get('leadership_stability', np.nan)
        
        # 21-23: ライフサイクル要因（新規）
        factors['company_age'] = data['company_age']
        factors['market_entry_timing'] = self._calculate_market_entry_timing(data, info)
        factors['parent_dependency'] = self._calculate_parent_dependency(data, info)
        
        return factors
    
    def _extract_succession_factors(self, data: pd.DataFrame, info: CompanyInfo) -> pd.DataFrame:
        """事業継承成功度の要因項目抽出（23項目）"""
        factors = pd.DataFrame(index=data.index)
        
        # 1-5: 統合効果
        factors['synergy_realization'] = data.get('synergy_realization', np.nan)
        factors['cost_reduction_achievement'] = data.get('cost_reduction_achievement', np.nan)
        factors['revenue_enhancement'] = data.get('revenue_enhancement', np.nan)
        factors['market_expansion_success'] = data.get('market_expansion_success', np.nan)
        factors['technology_integration'] = data.get('technology_integration', np.nan)
        
        # 6-10: 組織統合
        factors['cultural_integration'] = data.get('cultural_integration', np.nan)
        factors['personnel_retention_rate'] = data.get('personnel_retention_rate', np.nan)
        factors['management_continuity'] = data.get('management_continuity', np.nan)
        factors['operational_integration'] = data.get('operational_integration', np.nan)
        factors['system_integration_success'] = data.get('system_integration_success', np.nan)
        
        # 11-15: 財務統合効果
        factors['working_capital_optimization'] = data.get('working_capital_optimization', np.nan)
        factors['financial_leverage_improvement'] = data.get('financial_leverage_improvement', np.nan)
        factors['cost_of_capital_reduction'] = data.get('cost_of_capital_reduction', np.nan)
        factors['cash_flow_improvement'] = data.get('cash_flow_improvement', np.nan)
        factors['profitability_enhancement'] = data.get('profitability_enhancement', np.nan)
        
        # 16-20: 戦略的価値実現
        factors['market_position_strengthening'] = data.get('market_position_strengthening', np.nan)
        factors['competitive_advantage_creation'] = data.get('competitive_advantage_creation', np.nan)
        factors['portfolio_optimization'] = data.get('portfolio_optimization', np.nan)
        factors['risk_diversification'] = data.get('risk_diversification', np.nan)
        factors['strategic_flexibility'] = data.get('strategic_flexibility', np.nan)
        
        # 21-23: ライフサイクル要因（新規）
        factors['company_age'] = data['company_age']
        factors['market_entry_timing'] = self._calculate_market_entry_timing(data, info)
        factors['parent_dependency'] = self._calculate_parent_dependency(data, info)
        
        return factors
    
    def _calculate_market_entry_timing(self, data: pd.DataFrame, info: CompanyInfo) -> pd.Series:
        """市場参入タイミング指標計算"""
        if not info.founded_date:
            return pd.Series(np.nan, index=data.index)
        
        # 市場成熟度に基づく参入タイミング評価
        # 1: 先発企業（1950年代以前）
        # 2: 成長期参入（1960-1980年代）
        # 3: 成熟期参入（1990-2000年代）
        # 4: 後発参入（2010年代以降）
        
        founded_year = info.founded_date.year
        
        if founded_year < 1960:
            timing_score = 1  # 先発
        elif founded_year < 1990:
            timing_score = 2  # 成長期
        elif founded_year < 2010:
            timing_score = 3  # 成熟期
        else:
            timing_score = 4  # 後発
        
        return pd.Series(timing_score, index=data.index)
    
    def _calculate_parent_dependency(self, data: pd.DataFrame, info: CompanyInfo) -> pd.Series:
        """親会社依存度計算"""
        if not info.spinoff_parent:
            return pd.Series(0, index=data.index)  # 独立企業は依存度0
        
        # 分社企業の場合、分社後経過年数から依存度を逆算
        # 新しい分社企業ほど依存度が高い
        years_since_spinoff = data.get('years_since_spinoff', 0)
        
        # 依存度計算（分社直後100%、10年で50%、20年で20%に減衰）
        dependency = 100 * np.exp(-0.1 * years_since_spinoff)
        
        return pd.Series(dependency, index=data.index)
    
    def _calculate_early_rd_intensity(self, data: pd.DataFrame) -> pd.Series:
        """初期R&D集約度計算（新設企業用）"""
        # 設立後5年間の平均R&D比率
        early_years = data.head(5)
        if len(early_years) == 0:
            return pd.Series(np.nan, index=data.index)
        
        early_rd_ratio = safe_divide(early_years.get('rd_expenses', 0), early_years['sales_revenue']).mean() * 100
        return pd.Series(early_rd_ratio, index=data.index)
    
    def _calculate_early_growth_rate(self, data: pd.DataFrame) -> pd.Series:
        """初期成長率計算（新設企業用）"""
        # 設立後5年間の平均成長率
        early_years = data.head(5)
        if len(early_years) < 2:
            return pd.Series(np.nan, index=data.index)
        
        early_growth = calculate_growth_rate(early_years['sales_revenue']).mean()
        return pd.Series(early_growth, index=data.index)
    
    def _calculate_profitability_speed(self, data: pd.DataFrame) -> pd.Series:
        """収益化達成速度計算（新設企業用）"""
        # 黒字化までの年数
        profitable_years = data[data.get('net_income', 0) > 0]
        
        if len(profitable_years) == 0:
            profitability_speed = np.nan
        else:
            first_profitable_year = profitable_years['year'].min()
            start_year = data['year'].min()
            profitability_speed = first_profitable_year - start_year
        
        return pd.Series(profitability_speed, index=data.index)
    
    def _calculate_first_mover_advantage(self, data: pd.DataFrame, info: CompanyInfo) -> pd.Series:
        """先発者優位指標計算"""
        # 市場参入タイミングに基づく先発者優位度
        market_entry = self._calculate_market_entry_timing(data, info)
        
        # 先発企業ほど高いスコア
        first_mover_score = 5 - market_entry.iloc[0] if len(market_entry) > 0 else 0
        
        return pd.Series(max(0, first_mover_score), index=data.index)
    
    def _calculate_ratio_change(self, data: pd.DataFrame, column: str) -> pd.Series:
        """比率変化計算"""
        if column not in data.columns:
            return pd.Series(np.nan, index=data.index)
        
        ratio_change = data[column].diff()
        return ratio_change
    
    def _validate_extracted_features(self, features: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """抽出特徴量品質検証"""
        validated_features = {}
        
        for eval_item, feature_df in features.items():
            if feature_df.empty:
                self.logger.warning(f"空の特徴量データ: {eval_item}")
                validated_features[eval_item] = feature_df
                continue
            
            # 欠損率チェック
            missing_ratio = feature_df.isnull().sum() / len(feature_df)
            high_missing_cols = missing_ratio[missing_ratio > self.config.missing_threshold].index
            
            if len(high_missing_cols) > 0:
                self.logger.warning(f"高欠損率カラム除外: {eval_item} - {list(high_missing_cols)}")
                feature_df = feature_df.drop(columns=high_missing_cols)
            
            # 要因項目数チェック
            expected_factors = 23
            actual_factors = len(feature_df.columns)
            
            if actual_factors < expected_factors:
                self.logger.warning(f"要因項目不足: {eval_item} - {actual_factors}/{expected_factors}")
            
            validated_features[eval_item] = feature_df
        
        return validated_features
    
    def extract_company_metadata(self, info: CompanyInfo) -> Dict[str, Union[str, int, float]]:
        """企業メタデータ抽出"""
        metadata = {
            'company_id': info.company_id,
            'company_name': info.company_name,
            'market_category': info.market_category,
            'founded_year': info.founded_date.year if info.founded_date else None,
            'extinction_year': info.extinction_date.year if info.extinction_date else None,
            'is_spinoff': 1 if info.spinoff_parent else 0,
            'spinoff_parent': info.spinoff_parent,
            'listing_status': info.listing_status,
            'company_lifespan': self._calculate_company_lifespan(info),
            'market_category_numeric': self._encode_market_category(info.market_category)
        }
        
        return metadata
    
    def _calculate_company_lifespan(self, info: CompanyInfo) -> Optional[int]:
        """企業存続期間計算"""
        if not info.founded_date:
            return None
        
        end_date = info.extinction_date or date(2024, 12, 31)
        lifespan = end_date.year - info.founded_date.year
        
        return lifespan
    
    def _encode_market_category(self, category: str) -> int:
        """市場カテゴリ数値化"""
        encoding = {
            'high_share': 3,    # 高シェア市場
            'declining': 2,     # シェア低下市場
            'lost': 1          # 失失市場
        }
        return encoding.get(category, 0)
    
    def extract_temporal_features(
        self, 
        data: pd.DataFrame, 
        window_size: int = 3
    ) -> pd.DataFrame:
        """時系列特徴量抽出"""
        temporal_features = pd.DataFrame(index=data.index)
        
        # 移動平均
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in ['year']:
                temporal_features[f'{col}_ma{window_size}'] = rolling_average(
                    data[col], window_size
                )
        
        # トレンド指標
        for col in numeric_cols:
            if col not in ['year'] and col in data.columns:
                temporal_features[f'{col}_trend'] = self._calculate_trend(data[col])
        
        # 変動性指標
        for col in numeric_cols:
            if col not in ['year'] and col in data.columns:
                temporal_features[f'{col}_volatility'] = self._calculate_volatility(
                    data[col], window_size
                )
        
        return temporal_features
    
    def _calculate_trend(self, series: pd.Series) -> pd.Series:
        """トレンド計算（回帰係数）"""
        if len(series) < 3:
            return pd.Series(np.nan, index=series.index)
        
        x = np.arange(len(series))
        valid_mask = ~series.isnull()
        
        if valid_mask.sum() < 3:
            return pd.Series(np.nan, index=series.index)
        
        # 線形回帰の傾き計算
        x_valid = x[valid_mask]
        y_valid = series[valid_mask]
        
        if len(x_valid) >= 2:
            slope = np.polyfit(x_valid, y_valid, 1)[0]
        else:
            slope = np.nan
        
        return pd.Series(slope, index=series.index)
    
    def _calculate_volatility(self, series: pd.Series, window: int) -> pd.Series:
        """変動性計算（移動標準偏差）"""
        return series.rolling(window=window, min_periods=2).std()
    
    def batch_extract_features(
        self,
        companies_data: Dict[str, Tuple[pd.DataFrame, CompanyInfo]],
        market_data: Optional[Dict[str, pd.DataFrame]] = None
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        バッチ特徴量抽出
        
        Args:
            companies_data: {company_id: (financial_data, company_info)}
            market_data: 市場データ（オプション）
            
        Returns:
            Dict[company_id, Dict[evaluation_item, factors_df]]
        """
        self.logger.info(f"バッチ特徴量抽出開始: {len(companies_data)}社")
        
        batch_results = {}
        successful_extractions = 0
        failed_extractions = 0
        
        for company_id, (financial_data, company_info) in companies_data.items():
            try:
                # 市場データ取得
                company_market_data = None
                if market_data and company_info.market_category in market_data:
                    company_market_data = market_data[company_info.market_category]
                
                # 特徴量抽出実行
                features = self.extract_all_features(
                    financial_data, company_info, company_market_data
                )
                
                batch_results[company_id] = features
                successful_extractions += 1
                
                if successful_extractions % 10 == 0:
                    self.logger.info(f"進捗: {successful_extractions}/{len(companies_data)}社完了")
                
            except Exception as e:
                self.logger.error(f"特徴量抽出失敗: {company_id} - {e}")
                failed_extractions += 1
                continue
        
        self.logger.info(
            f"バッチ特徴量抽出完了: 成功{successful_extractions}社、失敗{failed_extractions}社"
        )
        
        return batch_results
    
    def save_extracted_features(
        self,
        features: Dict[str, Dict[str, pd.DataFrame]],
        output_dir: Path
    ) -> None:
        """抽出特徴量保存"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 評価項目別に統合して保存
        for eval_item in self.evaluation_items:
            combined_data = []
            
            for company_id, company_features in features.items():
                if eval_item in company_features and not company_features[eval_item].empty:
                    company_df = company_features[eval_item].copy()
                    company_df['company_id'] = company_id
                    combined_data.append(company_df)
            
            if combined_data:
                combined_df = pd.concat(combined_data, ignore_index=True)
                output_path = output_dir / f"{eval_item}_factors.csv"
                combined_df.to_csv(output_path, index=False, encoding='utf-8-sig')
                self.logger.info(f"保存完了: {output_path}")
    
    def generate_extraction_report(
        self,
        features: Dict[str, Dict[str, pd.DataFrame]]
    ) -> Dict[str, Union[int, float, List]]:
        """特徴量抽出レポート生成"""
        report = {
            'total_companies': len(features),
            'evaluation_items_count': len(self.evaluation_items),
            'expected_factors_per_item': 23,
            'extraction_summary': {},
            'data_quality_summary': {},
            'lifecycle_distribution': {}
        }
        
        # 評価項目別サマリー
        for eval_item in self.evaluation_items:
            item_data = []
            for company_features in features.values():
                if eval_item in company_features:
                    item_data.append(company_features[eval_item])
            
            if item_data:
                combined = pd.concat(item_data, ignore_index=True)
                report['extraction_summary'][eval_item] = {
                    'extracted_factors': len(combined.columns),
                    'total_records': len(combined),
                    'missing_ratio': combined.isnull().sum().sum() / (len(combined) * len(combined.columns))
                }
        
        return report


class LifecycleAwareExtractor(FeatureExtractor):
    """
    ライフサイクル対応特徴量抽出器
    
    企業の存続・消滅・新設段階に応じた特徴量抽出を行う
    """
    
    def extract_lifecycle_specific_features(
        self,
        data: pd.DataFrame,
        info: CompanyInfo,
        stage: str  # 'startup', 'growth', 'maturity', 'decline', 'extinction'
    ) -> pd.DataFrame:
        """ライフサイクル段階別特徴量抽出"""
        
        if stage == 'startup':
            return self._extract_startup_features(data, info)
        elif stage == 'growth':
            return self._extract_growth_stage_features(data, info)
        elif stage == 'maturity':
            return self._extract_maturity_features(data, info)
        elif stage == 'decline':
            return self._extract_decline_features(data, info)
        elif stage == 'extinction':
            return self._extract_extinction_features(data, info)
        else:
            raise ValueError(f"未知のライフサイクル段階: {stage}")
    
    def _extract_startup_features(self, data: pd.DataFrame, info: CompanyInfo) -> pd.DataFrame:
        """スタートアップ段階特徴量"""
        features = pd.DataFrame(index=data.index)
        
        # スタートアップ特有指標
        features['burn_rate'] = self._calculate_burn_rate(data)
        features['runway_months'] = self._calculate_runway(data)
        features['initial_investment_efficiency'] = self._calculate_investment_efficiency(data)
        features['market_traction'] = self._calculate_market_traction(data)
        features['team_building_speed'] = calculate_growth_rate(data.get('employee_count', pd.Series()))
        
        return features
    
    def _extract_growth_stage_features(self, data: pd.DataFrame, info: CompanyInfo) -> pd.DataFrame:
        """成長段階特徴量"""
        features = pd.DataFrame(index=data.index)
        
        # 成長段階特有指標
        features['scaling_efficiency'] = self._calculate_scaling_efficiency(data)
        features['market_expansion_rate'] = data.get('market_expansion_rate', np.nan)
        features['operational_leverage'] = self._calculate_operational_leverage(data)
        features['competitive_positioning'] = data.get('competitive_positioning', np.nan)
        
        return features
    
    def _extract_maturity_features(self, data: pd.DataFrame, info: CompanyInfo) -> pd.DataFrame:
        """成熟段階特徴量"""
        features = pd.DataFrame(index=data.index)
        
        # 成熟段階特有指標
        features['market_defense_capability'] = data.get('market_defense_capability', np.nan)
        features['efficiency_optimization'] = data.get('efficiency_optimization', np.nan)
        features['dividend_stability'] = data.get('dividend_stability', np.nan)
        features['innovation_renewal'] = data.get('innovation_renewal', np.nan)
        
        return features
    
    def _extract_decline_features(self, data: pd.DataFrame, info: CompanyInfo) -> pd.DataFrame:
        """衰退段階特徴量"""
        features = pd.DataFrame(index=data.index)
        
        # 衰退段階特有指標
        features['restructuring_intensity'] = data.get('restructuring_intensity', np.nan)
        features['asset_disposal_rate'] = data.get('asset_disposal_rate', np.nan)
        features['cost_cutting_effectiveness'] = data.get('cost_cutting_effectiveness', np.nan)
        features['turnaround_potential'] = data.get('turnaround_potential', np.nan)
        
        return features
    
    def _extract_extinction_features(self, data: pd.DataFrame, info: CompanyInfo) -> pd.DataFrame:
        """消滅段階特徴量"""
        features = pd.DataFrame(index=data.index)
        
        # 消滅前段階特有指標
        features['distress_signals'] = self._calculate_distress_signals(data)
        features['asset_liquidation_rate'] = data.get('asset_liquidation_rate', np.nan)
        features['going_concern_risk'] = data.get('going_concern_risk', np.nan)
        features['bankruptcy_probability'] = self._calculate_bankruptcy_probability(data)
        
        return features
    
    def _calculate_burn_rate(self, data: pd.DataFrame) -> pd.Series:
        """バーンレート計算（スタートアップ用）"""
        # 営業キャッシュフローが負の場合のキャッシュ消費率
        operating_cf = data.get('operating_cash_flow', pd.Series())
        cash_balance = data.get('cash_and_deposits', pd.Series())
        
        burn_rate = -operating_cf / cash_balance * 12  # 月次バーンレート
        return burn_rate.fillna(0)
    
    def _calculate_runway(self, data: pd.DataFrame) -> pd.Series:
        """ランウェイ計算（資金枯渇までの月数）"""
        cash_balance = data.get('cash_and_deposits', pd.Series())
        burn_rate = self._calculate_burn_rate(data)
        
        # バーンレートが正（キャッシュ流出）の場合のみ計算
        runway = np.where(burn_rate > 0, cash_balance / burn_rate, np.inf)
        return pd.Series(runway, index=data.index)
    
    def _calculate_investment_efficiency(self, data: pd.DataFrame) -> pd.Series:
        """投資効率性計算"""
        capex = data.get('capital_expenditure', pd.Series())
        revenue_growth = data.get('sales_revenue_growth_rate', pd.Series())
        
        # 投資1単位あたりの売上成長率
        investment_efficiency = safe_divide(revenue_growth, capex / data['sales_revenue'] * 100)
        return investment_efficiency
    
    def _calculate_market_traction(self, data: pd.DataFrame) -> pd.Series:
        """市場牽引力計算"""
        # 売上成長率と顧客数増加率の複合指標
        revenue_growth = data.get('sales_revenue_growth_rate', pd.Series())
        customer_growth = data.get('customer_count_growth_rate', pd.Series())
        
        # 両方のデータがある場合は平均、片方のみの場合はその値
        traction = np.where(
            customer_growth.notna(),
            (revenue_growth + customer_growth) / 2,
            revenue_growth
        )
        
        return pd.Series(traction, index=data.index)
    
    def _calculate_scaling_efficiency(self, data: pd.DataFrame) -> pd.Series:
        """スケーリング効率性計算"""
        # 売上成長率 / 費用成長率
        revenue_growth = data.get('sales_revenue_growth_rate', pd.Series())
        cost_growth = data.get('total_cost_growth_rate', pd.Series())
        
        scaling_efficiency = safe_divide(revenue_growth, cost_growth)
        return scaling_efficiency
    
    def _calculate_operational_leverage(self, data: pd.DataFrame) -> pd.Series:
        """オペレーティングレバレッジ計算"""
        # 営業利益成長率 / 売上成長率
        operating_profit_growth = data.get('operating_profit_growth_rate', pd.Series())
        revenue_growth = data.get('sales_revenue_growth_rate', pd.Series())
        
        operational_leverage = safe_divide(operating_profit_growth, revenue_growth)
        return operational_leverage
    
    def _calculate_distress_signals(self, data: pd.DataFrame) -> pd.Series:
        """財務的危険信号計算"""
        # 複数の危険指標を統合
        signals = pd.DataFrame(index=data.index)
        
        # 1. 流動性危険
        current_ratio = data.get('current_ratio', pd.Series())
        signals['liquidity_risk'] = np.where(current_ratio < 1.0, 1, 0)
        
        # 2. 収益性危険
        operating_margin = data.get('operating_margin', pd.Series())
        signals['profitability_risk'] = np.where(operating_margin < 0, 1, 0)
        
        # 3. 財務レバレッジ危険
        debt_ratio = data.get('debt_ratio', pd.Series())
        signals['leverage_risk'] = np.where(debt_ratio > 70, 1, 0)
        
        # 4. キャッシュフロー危険
        operating_cf = data.get('operating_cash_flow', pd.Series())
        signals['cashflow_risk'] = np.where(operating_cf < 0, 1, 0)
        
        # 総合危険スコア
        distress_score = signals.sum(axis=1)
        return distress_score
    
    def _calculate_bankruptcy_probability(self, data: pd.DataFrame) -> pd.Series:
        """倒産確率計算（Altman Z-Score改良版）"""
        # 修正Z-Score計算
        working_capital = data.get('working_capital', pd.Series())
        total_assets = data.get('total_assets', pd.Series())
        retained_earnings = data.get('retained_earnings', pd.Series())
        ebit = data.get('operating_profit', pd.Series())
        sales = data.get('sales_revenue', pd.Series())
        
        # Z-Score構成要素
        z1 = safe_divide(working_capital, total_assets) * 1.2
        z2 = safe_divide(retained_earnings, total_assets) * 1.4
        z3 = safe_divide(ebit, total_assets) * 3.3
        z4 = safe_divide(sales, total_assets) * 1.0
        
        z_score = z1 + z2 + z3 + z4
        
        # Z-Scoreから倒産確率への変換
        # Z > 2.99: 安全 (確率 < 5%)
        # 1.81 < Z < 2.99: 注意 (確率 5-20%)
        # Z < 1.81: 危険 (確率 > 20%)
        
        bankruptcy_prob = np.where(
            z_score > 2.99, 5,
            np.where(z_score > 1.81, 20, 50)
        )
        
        return pd.Series(bankruptcy_prob, index=data.index)


class MarketCategorySpecificExtractor:
    """
    市場カテゴリー特化特徴量抽出器
    
    高シェア/シェア低下/失失市場それぞれに特化した特徴量抽出
    """
    
    def __init__(self, base_extractor: FeatureExtractor):
        self.base_extractor = base_extractor
        self.logger = logging.getLogger(__name__)
    
    def extract_high_share_features(self, data: pd.DataFrame, info: CompanyInfo) -> pd.DataFrame:
        """高シェア市場特有特徴量抽出"""
        features = pd.DataFrame(index=data.index)
        
        # 1. 市場リーダーシップ指標
        features['market_dominance'] = data.get('market_share', pd.Series())
        features['pricing_power'] = self._calculate_pricing_power(data)
        features['innovation_leadership'] = self._calculate_innovation_leadership(data)
        features['brand_premium'] = data.get('brand_premium', np.nan)
        
        # 2. 持続的競争優位
        features['moat_strength'] = data.get('moat_strength', np.nan)
        features['switching_cost'] = data.get('switching_cost', np.nan)
        features['network_effects'] = data.get('network_effects', np.nan)
        features['scale_advantages'] = self._calculate_scale_advantages(data)
        
        # 3. イノベーション維持
        features['rd_intensity'] = safe_divide(data.get('rd_expenses', 0), data['sales_revenue']) * 100
        features['patent_generation_rate'] = data.get('patent_generation_rate', np.nan)
        features['technology_renewal_cycle'] = data.get('technology_renewal_cycle', np.nan)
        
        return features
    
    def extract_declining_share_features(self, data: pd.DataFrame, info: CompanyInfo) -> pd.DataFrame:
        """シェア低下市場特有特徴量抽出"""
        features = pd.DataFrame(index=data.index)
        
        # 1. 競争激化対応
        features['competitive_pressure'] = data.get('competitive_pressure', np.nan)
        features['cost_reduction_rate'] = data.get('cost_reduction_rate', np.nan)
        features['efficiency_improvement'] = data.get('efficiency_improvement', np.nan)
        features['differentiation_strategy'] = data.get('differentiation_strategy', np.nan)
        
        # 2. 事業転換努力
        features['new_market_exploration'] = data.get('new_market_exploration', np.nan)
        features['business_model_innovation'] = data.get('business_model_innovation', np.nan)
        features['digital_transformation'] = data.get('digital_transformation', np.nan)
        features['strategic_alliance'] = data.get('strategic_alliance', np.nan)
        
        # 3. 防御戦略
        features['core_competency_focus'] = data.get('core_competency_focus', np.nan)
        features['niche_market_dominance'] = data.get('niche_market_dominance', np.nan)
        features['customer_loyalty'] = data.get('customer_loyalty', np.nan)
        
        return features
    
    def extract_lost_share_features(self, data: pd.DataFrame, info: CompanyInfo) -> pd.DataFrame:
        """失失市場特有特徴量抽出"""
        features = pd.DataFrame(index=data.index)
        
        # 1. 事業撤退・転換
        features['exit_strategy_execution'] = data.get('exit_strategy_execution', np.nan)
        features['asset_divestiture_rate'] = data.get('asset_divestiture_rate', np.nan)
        features['business_restructuring'] = data.get('business_restructuring', np.nan)
        features['core_business_pivot'] = data.get('core_business_pivot', np.nan)
        
        # 2. 生存戦略
        features['survival_strategy'] = data.get('survival_strategy', np.nan)
        features['cost_structure_transformation'] = data.get('cost_structure_transformation', np.nan)
        features['new_value_creation'] = data.get('new_value_creation', np.nan)
        features['strategic_partnership'] = data.get('strategic_partnership', np.nan)
        
        # 3. 再生・復活要因
        features['turnaround_capability'] = data.get('turnaround_capability', np.nan)
        features['management_renewal'] = data.get('management_renewal', np.nan)
        features['technology_disruption_adaptation'] = data.get('technology_disruption_adaptation', np.nan)
        
        return features
    
    def _calculate_pricing_power(self, data: pd.DataFrame) -> pd.Series:
        """価格決定力計算"""
        # 価格上昇に対する売上への影響度
        price_change = data.get('average_selling_price_change', pd.Series())
        volume_change = data.get('sales_volume_change', pd.Series())
        
        # 価格弾力性の逆数として価格決定力を定義
        price_elasticity = safe_divide(volume_change, price_change)
        pricing_power = safe_divide(1, abs(price_elasticity))
        
        return pricing_power
    
    def _calculate_innovation_leadership(self, data: pd.DataFrame) -> pd.Series:
        """イノベーションリーダーシップ計算"""
        # R&D効率性と特許生成率の複合指標
        rd_efficiency = safe_divide(data.get('patent_count', 0), data.get('rd_expenses', 1))
        innovation_impact = data.get('innovation_impact_score', pd.Series())
        
        leadership_score = (rd_efficiency * 0.6 + innovation_impact * 0.4)
        return leadership_score
    
    def _calculate_scale_advantages(self, data: pd.DataFrame) -> pd.Series:
        """規模の経済効果計算"""
        # 売上規模と収益率の関係
        sales_scale = data['sales_revenue']
        operating_margin = data.get('operating_margin', pd.Series())
        
        # 規模と収益率の相関から規模効果を推定
        scale_effect = sales_scale * operating_margin / 100
        return scale_effect


def create_feature_extraction_pipeline(config: ExtractionConfig = None) -> FeatureExtractor:
    """特徴量抽出パイプライン作成"""
    return FeatureExtractor(config)


def create_lifecycle_extractor(base_extractor: FeatureExtractor = None) -> LifecycleAwareExtractor:
    """ライフサイクル対応抽出器作成"""
    if base_extractor is None:
        base_extractor = FeatureExtractor()
    
    return LifecycleAwareExtractor(base_extractor)


# 使用例とテスト関数
def example_usage():
    """使用例"""
    # 設定作成
    config = ExtractionConfig(
        start_year=1984,
        end_year=2024,
        min_data_years=5,
        outlier_threshold=3.0
    )
    
    # 抽出器作成
    extractor = FeatureExtractor(config)
    lifecycle_extractor = LifecycleAwareExtractor(extractor)
    
    # サンプル企業情報
    company_info = CompanyInfo(
        company_id="JP001",
        company_name="ファナック",
        market_category="high_share",
        founded_date=date(1972, 5, 1),
        extinction_date=None,
        spinoff_parent=None,
        listing_status="listed"
    )
    
    # サンプル財務データ（実際の使用時はEDINETから取得）
    sample_data = pd.DataFrame({
        'year': range(1984, 2025),
        'sales_revenue': np.random.normal(500000, 50000, 41),  # サンプルデータ
        'operating_profit': np.random.normal(75000, 10000, 41),
        'net_income': np.random.normal(50000, 8000, 41),
        'total_assets': np.random.normal(800000, 80000, 41),
        'shareholders_equity': np.random.normal(400000, 40000, 41),
        'employee_count': np.random.normal(5000, 500, 41),
        'rd_expenses': np.random.normal(25000, 3000, 41)
    })
    
    try:
        # 特徴量抽出実行
        extracted_features = extractor.extract_all_features(sample_data, company_info)
        
        print("特徴量抽出成功:")
        for eval_item, factors in extracted_features.items():
            print(f"  {eval_item}: {len(factors.columns)}要因項目")
        
        # レポート生成
        report = extractor.generate_extraction_report({company_info.company_id: extracted_features})
        print(f"\n抽出レポート: {report['total_companies']}社処理完了")
        
    except Exception as e:
        print(f"エラー: {e}")


if __name__ == "__main__":
    # ログ設定
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 使用例実行
    example_usage()