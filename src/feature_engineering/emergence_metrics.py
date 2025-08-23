"""
A2AI - Advanced Financial Analysis AI
新設企業特徴量エンジニアリング

このモジュールは新設企業（スピンオフ、分社化、新規設立企業）に特化した
特徴量の計算・抽出を行います。

対象企業例：
- デンソーウェーブ（2001年設立、トヨタから分社）
- キオクシア（2018年設立、東芝メモリから独立）
- プロテリアル（2023年設立、日立金属から独立）
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import warnings
from dataclasses import dataclass
import logging

# ロギング設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EmergenceEvent:
    """新設企業設立イベントのデータクラス"""
    company_id: str
    company_name: str
    establishment_date: datetime
    emergence_type: str  # 'spinoff', 'merger', 'startup', 'carveout'
    parent_company_id: Optional[str] = None
    parent_company_name: Optional[str] = None
    initial_business_segment: Optional[str] = None
    initial_capital: Optional[float] = None
    initial_employees: Optional[int] = None


class EmergenceFeatureEngineering:
    """
    新設企業特徴量エンジニアリングクラス
    
    新設企業の成功要因、成長パターン、リスク要因を定量化する
    特徴量を生成・計算します。
    """
    
    def __init__(self, 
                    emergence_events: pd.DataFrame,
                    financial_data: pd.DataFrame,
                    market_data: Optional[pd.DataFrame] = None):
        """
        初期化
        
        Args:
            emergence_events: 新設企業設立イベントデータ
            financial_data: 財務諸表データ
            market_data: 市場データ（オプション）
        """
        self.emergence_events = emergence_events
        self.financial_data = financial_data
        self.market_data = market_data
        
        # 新設企業リストの作成
        self.emergence_companies = self._identify_emergence_companies()
        
        logger.info(f"新設企業特徴量エンジニアリング初期化完了: {len(self.emergence_companies)}社")
    
    def _identify_emergence_companies(self) -> List[str]:
        """新設企業を特定"""
        # 設立年が1990年以降の企業を新設企業として定義
        recent_threshold = datetime(1990, 1, 1)
        
        emergence_companies = []
        for _, event in self.emergence_events.iterrows():
            establishment_date = pd.to_datetime(event['establishment_date'])
            if establishment_date >= recent_threshold:
                emergence_companies.append(event['company_id'])
        
        return emergence_companies
    
    def calculate_emergence_timing_features(self, 
                                            company_id: str) -> Dict[str, float]:
        """
        新設企業の設立タイミング特徴量を計算
        
        Args:
            company_id: 企業ID
            
        Returns:
            タイミング関連特徴量辞書
        """
        try:
            company_event = self.emergence_events[
                self.emergence_events['company_id'] == company_id
            ].iloc[0]
            
            establishment_date = pd.to_datetime(company_event['establishment_date'])
            current_date = datetime.now()
            
            features = {}
            
            # 1. 企業年齢（設立からの経過年数）
            company_age = (current_date - establishment_date).days / 365.25
            features['company_age_years'] = company_age
            
            # 2. 設立年代の分類
            establishment_year = establishment_date.year
            if establishment_year < 2000:
                features['establishment_era_pre2000'] = 1
                features['establishment_era_2000s'] = 0
                features['establishment_era_2010s'] = 0
                features['establishment_era_2020s'] = 0
            elif establishment_year < 2010:
                features['establishment_era_pre2000'] = 0
                features['establishment_era_2000s'] = 1
                features['establishment_era_2010s'] = 0
                features['establishment_era_2020s'] = 0
            elif establishment_year < 2020:
                features['establishment_era_pre2000'] = 0
                features['establishment_era_2000s'] = 0
                features['establishment_era_2010s'] = 1
                features['establishment_era_2020s'] = 0
            else:
                features['establishment_era_pre2000'] = 0
                features['establishment_era_2000s'] = 0
                features['establishment_era_2010s'] = 0
                features['establishment_era_2020s'] = 1
            
            # 3. 市場参入時期（先発/後発効果）
            # 同一市場内での設立順位を計算
            market_segment = company_event.get('initial_business_segment', 'unknown')
            same_market_companies = self.emergence_events[
                self.emergence_events['initial_business_segment'] == market_segment
            ].sort_values('establishment_date')
            
            market_entry_rank = (same_market_companies['company_id'] == company_id).idxmax()
            total_market_entrants = len(same_market_companies)
            
            features['market_entry_rank'] = market_entry_rank + 1
            features['market_entry_rank_normalized'] = (market_entry_rank + 1) / total_market_entrants
            features['is_market_pioneer'] = 1 if market_entry_rank == 0 else 0
            features['is_early_follower'] = 1 if 0 < market_entry_rank <= 2 else 0
            
            return features
            
        except Exception as e:
            logger.error(f"タイミング特徴量計算エラー (企業ID: {company_id}): {str(e)}")
            return {}
    
    def calculate_parent_dependency_features(self, 
                                            company_id: str) -> Dict[str, float]:
        """
        親会社依存度特徴量を計算
        
        Args:
            company_id: 企業ID
            
        Returns:
            親会社依存度関連特徴量辞書
        """
        try:
            company_event = self.emergence_events[
                self.emergence_events['company_id'] == company_id
            ].iloc[0]
            
            features = {}
            
            # 1. 新設タイプ分類
            emergence_type = company_event.get('emergence_type', 'unknown')
            features['is_spinoff'] = 1 if emergence_type == 'spinoff' else 0
            features['is_carveout'] = 1 if emergence_type == 'carveout' else 0
            features['is_merger'] = 1 if emergence_type == 'merger' else 0
            features['is_startup'] = 1 if emergence_type == 'startup' else 0
            
            # 2. 親会社存在フラグ
            has_parent = pd.notna(company_event.get('parent_company_id'))
            features['has_parent_company'] = 1 if has_parent else 0
            
            if has_parent:
                parent_id = company_event['parent_company_id']
                
                # 3. 親会社の財務規模との比較
                try:
                    parent_financials = self.financial_data[
                        self.financial_data['company_id'] == parent_id
                    ].sort_values('fiscal_year').iloc[-1]
                    
                    company_financials = self.financial_data[
                        self.financial_data['company_id'] == company_id
                    ].sort_values('fiscal_year').iloc[-1]
                    
                    # 売上高比率
                    if parent_financials['revenue'] > 0:
                        features['revenue_ratio_to_parent'] = (
                            company_financials['revenue'] / parent_financials['revenue']
                        )
                    else:
                        features['revenue_ratio_to_parent'] = 0
                    
                    # 総資産比率
                    if parent_financials['total_assets'] > 0:
                        features['assets_ratio_to_parent'] = (
                            company_financials['total_assets'] / parent_financials['total_assets']
                        )
                    else:
                        features['assets_ratio_to_parent'] = 0
                    
                    # 従業員数比率
                    if parent_financials.get('employees', 0) > 0:
                        features['employees_ratio_to_parent'] = (
                            company_financials.get('employees', 0) / parent_financials['employees']
                        )
                    else:
                        features['employees_ratio_to_parent'] = 0
                        
                except Exception as e:
                    logger.warning(f"親会社比較計算エラー: {str(e)}")
                    features['revenue_ratio_to_parent'] = 0
                    features['assets_ratio_to_parent'] = 0
                    features['employees_ratio_to_parent'] = 0
            else:
                features['revenue_ratio_to_parent'] = 0
                features['assets_ratio_to_parent'] = 0
                features['employees_ratio_to_parent'] = 0
            
            # 4. 独立度スコア（複合指標）
            independence_score = (
                (1 - features['has_parent_company']) * 0.4 +
                features['is_startup'] * 0.3 +
                min(1.0, features['revenue_ratio_to_parent']) * 0.3
            )
            features['independence_score'] = independence_score
            
            return features
            
        except Exception as e:
            logger.error(f"親会社依存度計算エラー (企業ID: {company_id}): {str(e)}")
            return {}
    
    def calculate_initial_resource_features(self, 
                                            company_id: str) -> Dict[str, float]:
        """
        初期リソース特徴量を計算
        
        Args:
            company_id: 企業ID
            
        Returns:
            初期リソース関連特徴量辞書
        """
        try:
            company_event = self.emergence_events[
                self.emergence_events['company_id'] == company_id
            ].iloc[0]
            
            features = {}
            
            # 1. 初期資本金規模
            initial_capital = company_event.get('initial_capital', 0)
            features['initial_capital'] = initial_capital
            
            # 資本金規模カテゴリ
            if initial_capital < 1e8:  # 1億円未満
                features['capital_size_small'] = 1
                features['capital_size_medium'] = 0
                features['capital_size_large'] = 0
            elif initial_capital < 1e9:  # 10億円未満
                features['capital_size_small'] = 0
                features['capital_size_medium'] = 1
                features['capital_size_large'] = 0
            else:  # 10億円以上
                features['capital_size_small'] = 0
                features['capital_size_medium'] = 0
                features['capital_size_large'] = 1
            
            # 2. 初期従業員数
            initial_employees = company_event.get('initial_employees', 0)
            features['initial_employees'] = initial_employees
            
            # 従業員規模カテゴリ
            if initial_employees < 100:
                features['employee_size_startup'] = 1
                features['employee_size_small'] = 0
                features['employee_size_medium'] = 0
                features['employee_size_large'] = 0
            elif initial_employees < 500:
                features['employee_size_startup'] = 0
                features['employee_size_small'] = 1
                features['employee_size_medium'] = 0
                features['employee_size_large'] = 0
            elif initial_employees < 2000:
                features['employee_size_startup'] = 0
                features['employee_size_small'] = 0
                features['employee_size_medium'] = 1
                features['employee_size_large'] = 0
            else:
                features['employee_size_startup'] = 0
                features['employee_size_small'] = 0
                features['employee_size_medium'] = 0
                features['employee_size_large'] = 1
            
            # 3. 初期リソース充実度指標
            if initial_capital > 0 and initial_employees > 0:
                features['capital_per_employee'] = initial_capital / initial_employees
            else:
                features['capital_per_employee'] = 0
            
            # 4. リソース豊富度スコア
            capital_score = min(1.0, initial_capital / 1e9)  # 10億円を最大として正規化
            employee_score = min(1.0, initial_employees / 1000)  # 1000人を最大として正規化
            features['resource_abundance_score'] = (capital_score + employee_score) / 2
            
            return features
            
        except Exception as e:
            logger.error(f"初期リソース計算エラー (企業ID: {company_id}): {str(e)}")
            return {}
    
    def calculate_early_growth_features(self, 
                                        company_id: str, 
                                        analysis_period: int = 5) -> Dict[str, float]:
        """
        初期成長パフォーマンス特徴量を計算
        
        Args:
            company_id: 企業ID
            analysis_period: 分析期間（年）
            
        Returns:
            初期成長関連特徴量辞書
        """
        try:
            # 企業の財務データを取得
            company_financials = self.financial_data[
                self.financial_data['company_id'] == company_id
            ].sort_values('fiscal_year')
            
            if len(company_financials) < 2:
                logger.warning(f"財務データ不足 (企業ID: {company_id})")
                return {}
            
            # 設立年を取得
            company_event = self.emergence_events[
                self.emergence_events['company_id'] == company_id
            ].iloc[0]
            establishment_year = pd.to_datetime(company_event['establishment_date']).year
            
            # 初期期間のデータを抽出
            early_period_data = company_financials[
                company_financials['fiscal_year'] <= establishment_year + analysis_period
            ]
            
            if len(early_period_data) < 2:
                logger.warning(f"初期期間データ不足 (企業ID: {company_id})")
                return {}
            
            features = {}
            
            # 1. 初期売上成長率
            first_year = early_period_data.iloc[0]
            last_year = early_period_data.iloc[-1]
            years_diff = last_year['fiscal_year'] - first_year['fiscal_year']
            
            if first_year['revenue'] > 0 and years_diff > 0:
                # CAGR (年平均成長率) を計算
                cagr = ((last_year['revenue'] / first_year['revenue']) ** (1/years_diff)) - 1
                features['early_revenue_cagr'] = cagr
                features['early_revenue_growth_absolute'] = last_year['revenue'] - first_year['revenue']
            else:
                features['early_revenue_cagr'] = 0
                features['early_revenue_growth_absolute'] = 0
            
            # 2. 初期収益性の改善
            if first_year['revenue'] > 0 and last_year['revenue'] > 0:
                first_margin = first_year.get('operating_income', 0) / first_year['revenue']
                last_margin = last_year.get('operating_income', 0) / last_year['revenue']
                features['early_margin_improvement'] = last_margin - first_margin
            else:
                features['early_margin_improvement'] = 0
            
            # 3. 初期従業員増加率
            first_employees = first_year.get('employees', 0)
            last_employees = last_year.get('employees', 0)
            
            if first_employees > 0 and years_diff > 0:
                employee_cagr = ((last_employees / first_employees) ** (1/years_diff)) - 1
                features['early_employee_cagr'] = employee_cagr
            else:
                features['early_employee_cagr'] = 0
            
            # 4. 初期生産性指標
            if last_employees > 0:
                features['early_revenue_per_employee'] = last_year['revenue'] / last_employees
            else:
                features['early_revenue_per_employee'] = 0
            
            # 5. 初期資本効率
            if last_year.get('total_assets', 0) > 0:
                features['early_asset_turnover'] = last_year['revenue'] / last_year['total_assets']
            else:
                features['early_asset_turnover'] = 0
            
            # 6. 成長速度カテゴリ
            cagr = features['early_revenue_cagr']
            if cagr > 0.3:  # 30%以上
                features['growth_speed_high'] = 1
                features['growth_speed_medium'] = 0
                features['growth_speed_low'] = 0
            elif cagr > 0.1:  # 10%以上
                features['growth_speed_high'] = 0
                features['growth_speed_medium'] = 1
                features['growth_speed_low'] = 0
            else:
                features['growth_speed_high'] = 0
                features['growth_speed_medium'] = 0
                features['growth_speed_low'] = 1
            
            # 7. 総合初期成功指標
            revenue_growth_score = min(1.0, max(0, cagr))
            margin_score = min(1.0, max(0, features['early_margin_improvement'] + 0.1))
            productivity_score = min(1.0, features['early_revenue_per_employee'] / 10000000)  # 1000万円/人を基準
            
            features['early_success_composite_score'] = (
                revenue_growth_score * 0.4 +
                margin_score * 0.3 +
                productivity_score * 0.3
            )
            
            return features
            
        except Exception as e:
            logger.error(f"初期成長計算エラー (企業ID: {company_id}): {str(e)}")
            return {}
    
    def calculate_market_timing_features(self, 
                                        company_id: str) -> Dict[str, float]:
        """
        市場タイミング特徴量を計算
        
        Args:
            company_id: 企業ID
            
        Returns:
            市場タイミング関連特徴量辞書
        """
        try:
            company_event = self.emergence_events[
                self.emergence_events['company_id'] == company_id
            ].iloc[0]
            
            establishment_date = pd.to_datetime(company_event['establishment_date'])
            features = {}
            
            # 1. 経済環境での設立タイミング
            establishment_year = establishment_date.year
            
            # バブル経済期
            features['established_during_bubble'] = 1 if 1986 <= establishment_year <= 1991 else 0
            
            # バブル崩壊期
            features['established_during_recession'] = 1 if 1992 <= establishment_year <= 2002 else 0
            
            # ITブーム期
            features['established_during_it_boom'] = 1 if 1995 <= establishment_year <= 2000 else 0
            
            # リーマンショック期
            features['established_during_financial_crisis'] = 1 if 2008 <= establishment_year <= 2009 else 0
            
            # アベノミクス期
            features['established_during_abenomics'] = 1 if 2013 <= establishment_year <= 2020 else 0
            
            # COVID-19期
            features['established_during_covid'] = 1 if 2020 <= establishment_year <= 2022 else 0
            
            # 2. 技術革新期との一致
            # AI・IoT時代
            features['established_during_ai_era'] = 1 if establishment_year >= 2010 else 0
            
            # モバイル時代
            features['established_during_mobile_era'] = 1 if 2007 <= establishment_year <= 2020 else 0
            
            # クラウド時代
            features['established_during_cloud_era'] = 1 if establishment_year >= 2005 else 0
            
            # 3. 市場成熟度での参入タイミング
            market_segment = company_event.get('initial_business_segment', 'unknown')
            
            # 市場成熟度推定（簡易版）
            if market_segment in ['ロボット', 'AI', 'IoT', 'EV']:
                features['entered_emerging_market'] = 1
                features['entered_mature_market'] = 0
            elif market_segment in ['自動車', '鉄鋼', '家電']:
                features['entered_emerging_market'] = 0
                features['entered_mature_market'] = 1
            else:
                features['entered_emerging_market'] = 0.5
                features['entered_mature_market'] = 0.5
            
            # 4. 市場タイミング総合スコア
            favorable_timing_score = (
                features['established_during_it_boom'] * 0.2 +
                features['established_during_abenomics'] * 0.2 +
                features['established_during_ai_era'] * 0.3 +
                features['entered_emerging_market'] * 0.3
            )
            
            unfavorable_timing_score = (
                features['established_during_recession'] * 0.3 +
                features['established_during_financial_crisis'] * 0.4 +
                features['established_during_covid'] * 0.3
            )
            
            features['market_timing_favorability'] = favorable_timing_score - unfavorable_timing_score
            
            return features
            
        except Exception as e:
            logger.error(f"市場タイミング計算エラー (企業ID: {company_id}): {str(e)}")
            return {}
    
    def calculate_innovation_potential_features(self, 
                                                company_id: str) -> Dict[str, float]:
        """
        イノベーションポテンシャル特徴量を計算
        
        Args:
            company_id: 企業ID
            
        Returns:
            イノベーション関連特徴量辞書
        """
        try:
            # 企業の財務データを取得
            company_financials = self.financial_data[
                self.financial_data['company_id'] == company_id
            ].sort_values('fiscal_year')
            
            if len(company_financials) == 0:
                logger.warning(f"財務データなし (企業ID: {company_id})")
                return {}
            
            latest_data = company_financials.iloc[-1]
            features = {}
            
            # 1. R&D集約度
            if latest_data['revenue'] > 0:
                rd_ratio = latest_data.get('rd_expenses', 0) / latest_data['revenue']
                features['rd_intensity'] = rd_ratio
                
                # R&D集約度カテゴリ
                if rd_ratio > 0.1:  # 10%以上
                    features['rd_intensity_high'] = 1
                    features['rd_intensity_medium'] = 0
                    features['rd_intensity_low'] = 0
                elif rd_ratio > 0.03:  # 3%以上
                    features['rd_intensity_high'] = 0
                    features['rd_intensity_medium'] = 1
                    features['rd_intensity_low'] = 0
                else:
                    features['rd_intensity_high'] = 0
                    features['rd_intensity_medium'] = 0
                    features['rd_intensity_low'] = 1
            else:
                features['rd_intensity'] = 0
                features['rd_intensity_high'] = 0
                features['rd_intensity_medium'] = 0
                features['rd_intensity_low'] = 1
            
            # 2. 無形資産比率
            if latest_data.get('total_assets', 0) > 0:
                intangible_ratio = latest_data.get('intangible_assets', 0) / latest_data['total_assets']
                features['intangible_asset_ratio'] = intangible_ratio
            else:
                features['intangible_asset_ratio'] = 0
            
            # 3. 人的資本の質
            if latest_data.get('employees', 0) > 0:
                revenue_per_employee = latest_data['revenue'] / latest_data['employees']
                features['revenue_per_employee'] = revenue_per_employee
                
                # 高付加価値人材密度（仮定：売上/従業員が高い＝高スキル人材）
                if revenue_per_employee > 50000000:  # 5000万円/人以上
                    features['high_value_workforce'] = 1
                else:
                    features['high_value_workforce'] = 0
            else:
                features['revenue_per_employee'] = 0
                features['high_value_workforce'] = 0
            
            # 4. 技術集約業種フラグ
            company_event = self.emergence_events[
                self.emergence_events['company_id'] == company_id
            ].iloc[0]
            
            business_segment = company_event.get('initial_business_segment', '').lower()
            tech_keywords = ['ロボット', 'ai', 'iot', '半導体', 'バイオ', '医療機器', '精密機器']
            
            features['tech_intensive_sector'] = 1 if any(keyword in business_segment for keyword in tech_keywords) else 0
            
            # 5. イノベーション総合スコア
            innovation_score = (
                features['rd_intensity'] * 0.3 +
                features['intangible_asset_ratio'] * 0.2 +
                min(1.0, features['revenue_per_employee'] / 50000000) * 0.3 +
                features['tech_intensive_sector'] * 0.2
            )
            features['innovation_potential_score'] = innovation_score
            
            return features
            
        except Exception as e:
            logger.error(f"イノベーション計算エラー (企業ID: {company_id}): {str(e)}")
            return {}
    
    def generate_all_emergence_features(self, 
                                        company_id: str) -> Dict[str, float]:
        """
        指定企業の全新設企業特徴量を生成
        
        Args:
            company_id: 企業ID
            
        Returns:
            全特徴量を含む辞書
        """
        # 新設企業かどうかチェック
        if company_id not in self.emergence_companies:
            logger.warning(f"非新設企業 (企業ID: {company_id})")
            return {}
        
        all_features = {}
        
        # 各特徴量グループを計算
        feature_groups = [
            ('timing', self.calculate_emergence_timing_features),
            ('dependency', self.calculate_parent_dependency_features),
            ('resources', self.calculate_initial_resource_features),
            ('growth', self.calculate_early_growth_features),
            ('market_timing', self.calculate_market_timing_features),
            ('innovation', self.calculate_innovation_potential_features),
        ]
        
        for group_name, calc_func in feature_groups:
            try:
                group_features = calc_func(company_id)
                # プレフィックスを付けて特徴量名を区別
                for key, value in group_features.items():
                    all_features[f'emergence_{group_name}_{key}'] = value
                    
                logger.info(f"{group_name}特徴量計算完了 (企業ID: {company_id})")
                
            except Exception as e:
                logger.error(f"{group_name}特徴量計算失敗 (企業ID: {company_id}): {str(e)}")
        
        # メタ特徴量追加
        all_features['emergence_feature_count'] = len(all_features)
        all_features['is_emergence_company'] = 1
        
        return all_features
    
    def generate_emergence_features_batch(self, 
                                        company_ids: Optional[List[str]] = None) -> pd.DataFrame:
        """
        複数企業の新設企業特徴量を一括生成
        
        Args:
            company_ids: 対象企業IDリスト（Noneの場合は全新設企業）
            
        Returns:
            特徴量DataFrame
        """
        if company_ids is None:
            company_ids = self.emergence_companies
        
        results = []
        
        for company_id in company_ids:
            logger.info(f"新設企業特徴量生成開始: {company_id}")
            
            features = self.generate_all_emergence_features(company_id)
            if features:
                features['company_id'] = company_id
                results.append(features)
        
        if results:
            df = pd.DataFrame(results)
            logger.info(f"新設企業特徴量生成完了: {len(df)}社")
            return df
        else:
            logger.warning("新設企業特徴量生成結果なし")
            return pd.DataFrame()
    
    def calculate_emergence_success_probability(self, 
                                                company_id: str,
                                                success_metrics: Dict[str, float]) -> Dict[str, float]:
        """
        新設企業の成功確率を計算
        
        Args:
            company_id: 企業ID
            success_metrics: 成功指標の閾値辞書
                例: {'revenue_threshold': 1000000000,  # 10億円
                        'employee_threshold': 100,         # 100人
                        'years_threshold': 5}              # 5年生存
        
        Returns:
            成功確率関連指標辞書
        """
        try:
            # 企業の現在の状況を取得
            company_financials = self.financial_data[
                self.financial_data['company_id'] == company_id
            ].sort_values('fiscal_year')
            
            if len(company_financials) == 0:
                return {'success_probability': 0.0}
            
            latest_data = company_financials.iloc[-1]
            
            # 設立年を取得
            company_event = self.emergence_events[
                self.emergence_events['company_id'] == company_id
            ].iloc[0]
            establishment_year = pd.to_datetime(company_event['establishment_date']).year
            current_age = latest_data['fiscal_year'] - establishment_year
            
            results = {}
            
            # 1. 各成功指標の達成状況
            revenue_threshold = success_metrics.get('revenue_threshold', 1000000000)
            employee_threshold = success_metrics.get('employee_threshold', 100)
            years_threshold = success_metrics.get('years_threshold', 5)
            
            revenue_success = 1 if latest_data['revenue'] >= revenue_threshold else 0
            employee_success = 1 if latest_data.get('employees', 0) >= employee_threshold else 0
            survival_success = 1 if current_age >= years_threshold else 0
            
            results['revenue_success_achieved'] = revenue_success
            results['employee_success_achieved'] = employee_success
            results['survival_success_achieved'] = survival_success
            
            # 2. 成功達成度スコア
            revenue_score = min(1.0, latest_data['revenue'] / revenue_threshold)
            employee_score = min(1.0, latest_data.get('employees', 0) / employee_threshold)
            survival_score = min(1.0, current_age / years_threshold)
            
            results['revenue_achievement_ratio'] = revenue_score
            results['employee_achievement_ratio'] = employee_score
            results['survival_achievement_ratio'] = survival_score
            
            # 3. 総合成功スコア
            overall_success_score = (revenue_score * 0.4 + 
                                   employee_score * 0.3 + 
                                   survival_score * 0.3)
            results['overall_success_score'] = overall_success_score
            
            # 4. 成功確率推定（シンプルなロジスティック関数）
            # 実際の実装では機械学習モデルを使用
            success_probability = 1 / (1 + np.exp(-5 * (overall_success_score - 0.5)))
            results['success_probability'] = success_probability
            
            # 5. 成功カテゴリ分類
            if overall_success_score >= 0.8:
                results['success_category'] = 'high_success'
                results['success_category_high'] = 1
                results['success_category_medium'] = 0
                results['success_category_low'] = 0
            elif overall_success_score >= 0.5:
                results['success_category'] = 'medium_success'
                results['success_category_high'] = 0
                results['success_category_medium'] = 1
                results['success_category_low'] = 0
            else:
                results['success_category'] = 'low_success'
                results['success_category_high'] = 0
                results['success_category_medium'] = 0
                results['success_category_low'] = 1
            
            return results
            
        except Exception as e:
            logger.error(f"成功確率計算エラー (企業ID: {company_id}): {str(e)}")
            return {'success_probability': 0.0}
    
    def analyze_emergence_patterns(self) -> Dict[str, Union[pd.DataFrame, Dict]]:
        """
        新設企業の全体的なパターンを分析
        
        Returns:
            分析結果辞書
        """
        results = {}
        
        try:
            # 1. 新設企業の設立年代分析
            establishment_years = []
            emergence_types = []
            initial_capitals = []
            
            for _, event in self.emergence_events.iterrows():
                if event['company_id'] in self.emergence_companies:
                    establishment_years.append(pd.to_datetime(event['establishment_date']).year)
                    emergence_types.append(event.get('emergence_type', 'unknown'))
                    initial_capitals.append(event.get('initial_capital', 0))
            
            # 年代別分布
            decade_distribution = pd.Series(establishment_years).apply(
                lambda x: f"{(x//10)*10}s"
            ).value_counts().sort_index()
            
            results['decade_distribution'] = decade_distribution.to_dict()
            
            # 2. 新設タイプ別分析
            type_distribution = pd.Series(emergence_types).value_counts()
            results['emergence_type_distribution'] = type_distribution.to_dict()
            
            # 3. 初期資本金分析
            capital_stats = {
                'mean': np.mean([c for c in initial_capitals if c > 0]),
                'median': np.median([c for c in initial_capitals if c > 0]),
                'std': np.std([c for c in initial_capitals if c > 0]),
                'min': min([c for c in initial_capitals if c > 0]) if any(c > 0 for c in initial_capitals) else 0,
                'max': max(initial_capitals)
            }
            results['initial_capital_stats'] = capital_stats
            
            # 4. 成功率分析（簡易版）
            success_metrics = {
                'revenue_threshold': 1000000000,  # 10億円
                'employee_threshold': 100,        # 100人
                'years_threshold': 5              # 5年
            }
            
            success_rates = []
            for company_id in self.emergence_companies:
                success_prob = self.calculate_emergence_success_probability(
                    company_id, success_metrics
                )
                if success_prob.get('success_probability', 0) > 0:
                    success_rates.append(success_prob['success_probability'])
            
            if success_rates:
                results['success_rate_stats'] = {
                    'mean': np.mean(success_rates),
                    'median': np.median(success_rates),
                    'high_success_ratio': len([r for r in success_rates if r > 0.7]) / len(success_rates)
                }
            
            # 5. 業界別新設企業分析
            sector_analysis = {}
            for _, event in self.emergence_events.iterrows():
                if event['company_id'] in self.emergence_companies:
                    sector = event.get('initial_business_segment', 'unknown')
                    if sector not in sector_analysis:
                        sector_analysis[sector] = {'count': 0, 'avg_capital': 0, 'capitals': []}
                    
                    sector_analysis[sector]['count'] += 1
                    capital = event.get('initial_capital', 0)
                    if capital > 0:
                        sector_analysis[sector]['capitals'].append(capital)
            
            # 業界別平均資本金を計算
            for sector, data in sector_analysis.items():
                if data['capitals']:
                    data['avg_capital'] = np.mean(data['capitals'])
                del data['capitals']  # メモリ節約
            
            results['sector_analysis'] = sector_analysis
            
            logger.info("新設企業パターン分析完了")
            return results
            
        except Exception as e:
            logger.error(f"新設企業パターン分析エラー: {str(e)}")
            return {}
    
    def export_emergence_features_summary(self, 
                                        output_path: str = 'emergence_features_summary.csv') -> bool:
        """
        新設企業特徴量の要約をCSVファイルにエクスポート
        
        Args:
            output_path: 出力ファイルパス
            
        Returns:
            エクスポート成功フラグ
        """
        try:
            # 全新設企業の特徴量を生成
            features_df = self.generate_emergence_features_batch()
            
            if features_df.empty:
                logger.error("エクスポートするデータがありません")
                return False
            
            # 統計要約を追加
            summary_stats = features_df.describe()
            
            # メタデータを追加
            metadata = {
                'total_emergence_companies': len(self.emergence_companies),
                'features_generated': len(features_df.columns) - 1,  # company_idを除く
                'generation_timestamp': datetime.now().isoformat()
            }
            
            # CSVファイルに保存
            features_df.to_csv(output_path, index=False)
            
            # メタデータを別ファイルに保存
            metadata_path = output_path.replace('.csv', '_metadata.json')
            import json
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            # 統計要約も保存
            stats_path = output_path.replace('.csv', '_statistics.csv')
            summary_stats.to_csv(stats_path)
            
            logger.info(f"新設企業特徴量エクスポート完了: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"エクスポートエラー: {str(e)}")
            return False
    
    def get_feature_importance_analysis(self, 
                                        target_metric: str = 'success_probability') -> Dict[str, float]:
        """
        特徴量重要度分析（相関ベース）
        
        Args:
            target_metric: 目的変数となる指標名
            
        Returns:
            特徴量重要度辞書
        """
        try:
            # 全特徴量を生成
            features_df = self.generate_emergence_features_batch()
            
            if features_df.empty:
                logger.error("分析データなし")
                return {}
            
            # 成功確率を計算して結合
            success_metrics = {
                'revenue_threshold': 1000000000,
                'employee_threshold': 100,
                'years_threshold': 5
            }
            
            success_data = []
            for company_id in features_df['company_id']:
                success_prob = self.calculate_emergence_success_probability(
                    company_id, success_metrics
                )
                success_data.append(success_prob.get('success_probability', 0))
            
            features_df['success_probability'] = success_data
            
            # 数値列のみを抽出
            numeric_columns = features_df.select_dtypes(include=[np.number]).columns
            numeric_features = features_df[numeric_columns].drop(['company_id'], errors='ignore')
            
            if target_metric not in numeric_features.columns:
                logger.error(f"目的変数 {target_metric} が見つかりません")
                return {}
            
            # 相関係数を計算
            correlations = numeric_features.corr()[target_metric].abs()
            correlations = correlations.drop(target_metric, errors='ignore')
            correlations = correlations.sort_values(ascending=False)
            
            # 上位の重要特徴量を返す
            importance_dict = correlations.head(20).to_dict()
            
            logger.info(f"特徴量重要度分析完了: {len(importance_dict)}個の特徴量")
            return importance_dict
            
        except Exception as e:
            logger.error(f"重要度分析エラー: {str(e)}")
            return {}


# 使用例・テスト関数
def test_emergence_features():
    """
    EmergenceFeatureEngineeringクラスのテスト関数
    """
    # サンプルデータの作成
    sample_emergence_events = pd.DataFrame([
        {
            'company_id': 'DENSO_WAVE',
            'company_name': 'デンソーウェーブ',
            'establishment_date': '2001-04-01',
            'emergence_type': 'spinoff',
            'parent_company_id': 'DENSO',
            'parent_company_name': 'デンソー',
            'initial_business_segment': 'ロボット',
            'initial_capital': 500000000,  # 5億円
            'initial_employees': 150
        },
        {
            'company_id': 'KIOXIA',
            'company_name': 'キオクシア',
            'establishment_date': '2018-06-01',
            'emergence_type': 'carveout',
            'parent_company_id': 'TOSHIBA',
            'parent_company_name': '東芝',
            'initial_business_segment': '半導体',
            'initial_capital': 1000000000,  # 10億円
            'initial_employees': 300
        }
    ])
    
    sample_financial_data = pd.DataFrame([
        {
            'company_id': 'DENSO_WAVE',
            'fiscal_year': 2022,
            'revenue': 15000000000,  # 150億円
            'operating_income': 1500000000,  # 15億円
            'total_assets': 20000000000,  # 200億円
            'employees': 500,
            'rd_expenses': 1000000000,  # 10億円
            'intangible_assets': 2000000000  # 20億円
        },
        {
            'company_id': 'KIOXIA',
            'fiscal_year': 2022,
            'revenue': 800000000000,  # 8000億円
            'operating_income': 100000000000,  # 1000億円
            'total_assets': 1200000000000,  # 1.2兆円
            'employees': 8000,
            'rd_expenses': 80000000000,  # 800億円
            'intangible_assets': 150000000000  # 1500億円
        }
    ])
    
    # EmergenceFeatureEngineeringインスタンス作成
    emergence_fe = EmergenceFeatureEngineering(
        emergence_events=sample_emergence_events,
        financial_data=sample_financial_data
    )
    
    # 特徴量生成テスト
    print("=== 新設企業特徴量生成テスト ===")
    
    for company_id in ['DENSO_WAVE', 'KIOXIA']:
        print(f"\n企業ID: {company_id}")
        features = emergence_fe.generate_all_emergence_features(company_id)
        
        for feature_name, value in list(features.items())[:10]:  # 最初の10個のみ表示
            print(f"  {feature_name}: {value:.4f}")
        
        print(f"  総特徴量数: {len(features)}")
    
    # バッチ処理テスト
    print("\n=== バッチ処理テスト ===")
    batch_features = emergence_fe.generate_emergence_features_batch()
    print(f"バッチ処理結果: {batch_features.shape}")
    
    # パターン分析テスト
    print("\n=== パターン分析テスト ===")
    patterns = emergence_fe.analyze_emergence_patterns()
    print("分析結果キー:", list(patterns.keys()))


if __name__ == "__main__":
    test_emergence_features()