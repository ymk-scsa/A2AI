"""
A2AI - Advanced Financial Analysis AI
Lifecycle Normalizer Module

企業ライフサイクル段階別データ正規化処理
- 企業年齢・成熟度に基づく正規化
- 市場参入時期・競争環境の調整
- 分社・統合企業の継続性処理
- 消滅・新設企業のライフサイクル対応
"""

import pandas as pd
import numpy as np
import warnings
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy import stats
import logging

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LifecycleStage:
    """企業ライフサイクル段階定義"""
    startup: str = "startup"          # 設立期（0-5年）
    growth: str = "growth"            # 成長期（6-15年）
    maturity: str = "maturity"        # 成熟期（16-30年）
    mature: str = "mature"            # 成熟期（31-50年）
    decline: str = "decline"          # 衰退期（51年以上または消滅予兆）
    extinction: str = "extinction"    # 消滅期（倒産・吸収）
    revival: str = "revival"          # 復活期（再生・分社後）

@dataclass
class MarketEntryTiming:
    """市場参入時期分類"""
    pioneer: str = "pioneer"          # パイオニア（市場創造）
    early_adopter: str = "early"      # 早期参入（市場形成期）
    follower: str = "follower"        # フォロワー（成長期参入）
    late_entrant: str = "late"        # 後発参入（成熟期参入）

class LifecycleNormalizer:
    """
    企業ライフサイクル段階別正規化クラス
    
    機能:
    1. 企業年齢・成熟度に基づくデータ正規化
    2. 市場参入時期による調整係数適用
    3. 分社・統合企業の継続性処理
    4. 消滅・新設企業のライフサイクル対応
    5. ライフサイクル段階別ベンチマーク作成
    """
    
    def __init__(self, base_year: int = 1984):
        self.base_year = base_year
        self.current_year = datetime.now().year
        self.lifecycle_stages = LifecycleStage()
        self.market_entry_timing = MarketEntryTiming()
        self.normalization_cache = {}
        self.company_profiles = {}
        
        # ライフサイクル段階別正規化パラメータ
        self.stage_parameters = {
            self.lifecycle_stages.startup: {
                'volatility_factor': 2.0,      # 高ボラティリティ
                'growth_expectation': 0.5,     # 高成長期待
                'benchmark_window': 3,         # 短期ベンチマーク
                'risk_premium': 0.1           # 高リスクプレミアム
            },
            self.lifecycle_stages.growth: {
                'volatility_factor': 1.5,
                'growth_expectation': 0.3,
                'benchmark_window': 5,
                'risk_premium': 0.05
            },
            self.lifecycle_stages.maturity: {
                'volatility_factor': 1.0,
                'growth_expectation': 0.1,
                'benchmark_window': 7,
                'risk_premium': 0.02
            },
            self.lifecycle_stages.mature: {
                'volatility_factor': 0.8,
                'growth_expectation': 0.05,
                'benchmark_window': 10,
                'risk_premium': 0.01
            },
            self.lifecycle_stages.decline: {
                'volatility_factor': 1.8,
                'growth_expectation': -0.1,
                'benchmark_window': 3,
                'risk_premium': 0.15
            },
            self.lifecycle_stages.extinction: {
                'volatility_factor': 3.0,
                'growth_expectation': -0.5,
                'benchmark_window': 1,
                'risk_premium': 0.3
            },
            self.lifecycle_stages.revival: {
                'volatility_factor': 2.5,
                'growth_expectation': 0.4,
                'benchmark_window': 3,
                'risk_premium': 0.2
            }
        }
        
        # 評価項目のライフサイクル感度
        self.metric_sensitivity = {
            'sales': 0.8,                    # 売上高
            'sales_growth': 1.5,             # 売上高成長率
            'operating_margin': 1.2,         # 売上高営業利益率
            'net_margin': 1.3,               # 売上高当期純利益率
            'roe': 1.4,                      # ROE
            'value_added_ratio': 1.0,        # 売上高付加価値率
            'survival_probability': 2.0,     # 企業存続確率
            'emergence_success': 2.5,        # 新規事業成功率
            'succession_success': 1.8        # 事業継承成功度
        }
    
    def determine_lifecycle_stage(
        self, 
        company_name: str, 
        establishment_year: Optional[int] = None,
        extinction_year: Optional[int] = None,
        spinoff_year: Optional[int] = None,
        financial_indicators: Optional[Dict] = None
    ) -> str:
        """
        企業のライフサイクル段階を決定
        
        Args:
            company_name: 企業名
            establishment_year: 設立年
            extinction_year: 消滅年
            spinoff_year: 分社年
            financial_indicators: 財務指標（成長率、収益性等）
        
        Returns:
            ライフサイクル段階
        """
        try:
            # 消滅企業の場合
            if extinction_year is not None:
                return self.lifecycle_stages.extinction
            
            # 分社企業の場合（復活期として扱う）
            if spinoff_year is not None:
                years_since_spinoff = self.current_year - spinoff_year
                if years_since_spinoff <= 5:
                    return self.lifecycle_stages.revival
                else:
                    establishment_year = spinoff_year  # 分社年を基準年として使用
            
            # 設立年不明の場合のデフォルト推定
            if establishment_year is None:
                logger.warning(f"企業 {company_name} の設立年が不明です。成熟期として扱います。")
                return self.lifecycle_stages.maturity
            
            # 企業年齢計算
            company_age = self.current_year - establishment_year
            
            # 基本的なライフサイクル段階判定
            if company_age <= 5:
                base_stage = self.lifecycle_stages.startup
            elif company_age <= 15:
                base_stage = self.lifecycle_stages.growth
            elif company_age <= 30:
                base_stage = self.lifecycle_stages.maturity
            elif company_age <= 50:
                base_stage = self.lifecycle_stages.mature
            else:
                base_stage = self.lifecycle_stages.decline
            
            # 財務指標による調整
            if financial_indicators:
                adjusted_stage = self._adjust_stage_by_performance(
                    base_stage, financial_indicators, company_age
                )
                return adjusted_stage
            
            return base_stage
            
        except Exception as e:
            logger.error(f"ライフサイクル段階決定エラー ({company_name}): {e}")
            return self.lifecycle_stages.maturity  # デフォルト
    
    def _adjust_stage_by_performance(
        self, 
        base_stage: str, 
        financial_indicators: Dict, 
        company_age: int
    ) -> str:
        """
        財務パフォーマンスによるライフサイクル段階調整
        
        Args:
            base_stage: 基本ライフサイクル段階
            financial_indicators: 財務指標辞書
            company_age: 企業年齢
        
        Returns:
            調整後ライフサイクル段階
        """
        try:
            # 成長率指標の取得
            sales_growth = financial_indicators.get('sales_growth_rate', 0)
            profit_growth = financial_indicators.get('profit_growth_rate', 0)
            asset_growth = financial_indicators.get('asset_growth_rate', 0)
            
            # 収益性指標の取得
            operating_margin = financial_indicators.get('operating_margin', 0)
            roe = financial_indicators.get('roe', 0)
            
            # 効率性指標の取得
            asset_turnover = financial_indicators.get('asset_turnover', 0)
            
            # 総合成長スコア計算
            growth_score = (sales_growth * 0.4 + profit_growth * 0.3 + 
                          asset_growth * 0.3)
            
            # 収益性スコア計算
            profitability_score = (operating_margin * 0.6 + roe * 0.4)
            
            # ライフサイクル調整ロジック
            if company_age > 30:  # 成熟企業
                if growth_score > 0.15 and profitability_score > 0.1:
                    # 高成長・高収益性 → 復活期
                    return self.lifecycle_stages.revival
                elif growth_score < -0.1 or profitability_score < 0:
                    # 負成長・低収益 → 衰退期
                    return self.lifecycle_stages.decline
            
            elif company_age <= 15:  # 若い企業
                if growth_score < -0.05:
                    # 若い企業で負成長 → 要注意（成長期維持だが監視）
                    pass
            
            return base_stage
            
        except Exception as e:
            logger.error(f"財務指標による段階調整エラー: {e}")
            return base_stage
    
    def determine_market_entry_timing(
        self, 
        company_name: str,
        market_category: str,
        establishment_year: int,
        market_data: Optional[Dict] = None
    ) -> str:
        """
        市場参入時期の分類決定
        
        Args:
            company_name: 企業名
            market_category: 市場カテゴリ（high_share/declining/lost）
            establishment_year: 設立年
            market_data: 市場データ（オプション）
        
        Returns:
            市場参入時期分類
        """
        try:
            # 市場別の参入時期基準年
            market_reference_years = {
                'ロボット': 1970,
                '内視鏡': 1960,
                '工作機械': 1950,
                '電子材料': 1965,
                '精密測定機器': 1960,
                '自動車': 1900,
                '鉄鋼': 1900,
                'スマート家電': 1980,
                'バッテリー': 1990,
                'PC・周辺機器': 1980,
                '家電': 1950,
                '半導体': 1970,
                'スマートフォン': 1999,
                'PC': 1980,
                '通信機器': 1980
            }
            
            # 市場特定（企業名から推定またはmarket_dataから取得）
            market_name = self._identify_market_from_company(company_name, market_category)
            
            # 基準年の取得
            reference_year = market_reference_years.get(market_name, 1960)
            
            # 参入時期分類
            years_after_market_start = establishment_year - reference_year
            
            if years_after_market_start <= 10:
                return self.market_entry_timing.pioneer
            elif years_after_market_start <= 20:
                return self.market_entry_timing.early_adopter
            elif years_after_market_start <= 40:
                return self.market_entry_timing.follower
            else:
                return self.market_entry_timing.late_entrant
                
        except Exception as e:
            logger.error(f"市場参入時期決定エラー ({company_name}): {e}")
            return self.market_entry_timing.follower  # デフォルト
    
    def _identify_market_from_company(
        self, 
        company_name: str, 
        market_category: str
    ) -> str:
        """
        企業名から市場を特定（簡易版）
        
        Args:
            company_name: 企業名
            market_category: 市場カテゴリ
        
        Returns:
            市場名
        """
        # 企業名キーワードマッピング（簡易版）
        market_keywords = {
            'ロボット': ['ファナック', '安川電機', '川崎重工', '不二越', 'デンソーウェーブ'],
            '内視鏡': ['オリンパス', 'HOYA', '富士フイルム', 'キヤノンメディカル'],
            '工作機械': ['DMG森精機', 'ヤマザキマザック', 'オークマ', '牧野フライス'],
            '電子材料': ['村田製作所', 'TDK', '京セラ', '太陽誘電'],
            '精密測定機器': ['キーエンス', '島津製作所', '堀場製作所', '東京精密'],
            '自動車': ['トヨタ', '日産', 'ホンダ', 'スズキ', 'マツダ'],
            '鉄鋼': ['日本製鉄', 'JFE', '神戸製鋼', '日新製鋼'],
            'バッテリー': ['パナソニックエナジー', 'GSユアサ', '東芝インフラ']
        }
        
        for market, keywords in market_keywords.items():
            if any(keyword in company_name for keyword in keywords):
                return market
        
        return '汎用'  # デフォルト
    
    def normalize_by_lifecycle_stage(
        self, 
        data: pd.DataFrame,
        company_column: str = 'company_name',
        year_column: str = 'year',
        target_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        ライフサイクル段階別データ正規化
        
        Args:
            data: 財務データフレーム
            company_column: 企業名カラム
            year_column: 年度カラム
            target_columns: 正規化対象カラム（Noneの場合は数値カラム全て）
        
        Returns:
            正規化済みデータフレーム
        """
        try:
            normalized_data = data.copy()
            
            # 対象カラムの決定
            if target_columns is None:
                target_columns = data.select_dtypes(include=[np.number]).columns.tolist()
                if company_column in target_columns:
                    target_columns.remove(company_column)
                if year_column in target_columns:
                    target_columns.remove(year_column)
            
            # 企業ごとに処理
            for company in data[company_column].unique():
                company_data = data[data[company_column] == company].copy()
                
                # ライフサイクル段階決定
                lifecycle_stage = self._determine_company_lifecycle_stage(
                    company, company_data
                )
                
                # 市場参入時期決定
                entry_timing = self._determine_company_entry_timing(
                    company, company_data
                )
                
                # 段階別正規化パラメータ取得
                stage_params = self.stage_parameters.get(
                    lifecycle_stage, 
                    self.stage_parameters[self.lifecycle_stages.maturity]
                )
                
                # 企業データの正規化
                normalized_company_data = self._apply_lifecycle_normalization(
                    company_data, 
                    target_columns, 
                    stage_params,
                    lifecycle_stage,
                    entry_timing
                )
                
                # 結果をメインデータフレームに反映
                company_mask = normalized_data[company_column] == company
                normalized_data.loc[company_mask, target_columns] = \
                    normalized_company_data[target_columns]
                    
                # ライフサイクル情報を追加
                normalized_data.loc[company_mask, 'lifecycle_stage'] = lifecycle_stage
                normalized_data.loc[company_mask, 'entry_timing'] = entry_timing
            
            return normalized_data
            
        except Exception as e:
            logger.error(f"ライフサイクル正規化エラー: {e}")
            return data
    
    def _determine_company_lifecycle_stage(
        self, 
        company: str, 
        company_data: pd.DataFrame
    ) -> str:
        """
        企業のライフサイクル段階を決定（データベース）
        """
        # 設立年推定（データから）
        min_year = company_data['year'].min() if 'year' in company_data.columns else None
        establishment_year = min_year - 5 if min_year else None  # 推定設立年
        
        # 消滅年チェック（データの最終年が現在より古い場合）
        max_year = company_data['year'].max() if 'year' in company_data.columns else None
        extinction_year = max_year if max_year and max_year < self.current_year - 2 else None
        
        # 財務指標の計算
        financial_indicators = self._calculate_financial_indicators(company_data)
        
        return self.determine_lifecycle_stage(
            company, 
            establishment_year, 
            extinction_year,
            financial_indicators=financial_indicators
        )
    
    def _determine_company_entry_timing(
        self, 
        company: str, 
        company_data: pd.DataFrame
    ) -> str:
        """
        企業の市場参入時期を決定（データベース）
        """
        # 最初のデータ年を参入年として使用
        min_year = company_data['year'].min() if 'year' in company_data.columns else 1980
        
        # 市場カテゴリを推定（簡易版）
        market_category = "general"
        
        return self.determine_market_entry_timing(
            company, 
            market_category, 
            min_year
        )
    
    def _calculate_financial_indicators(
        self, 
        company_data: pd.DataFrame
    ) -> Dict:
        """
        企業データから財務指標を計算
        
        Args:
            company_data: 企業の財務データ
        
        Returns:
            財務指標辞書
        """
        try:
            indicators = {}
            
            # 成長率計算（可能な項目のみ）
            for column in ['sales', 'profit', 'assets']:
                if column in company_data.columns:
                    growth_rates = company_data[column].pct_change()
                    indicators[f'{column}_growth_rate'] = growth_rates.mean()
            
            # 収益性指標
            if 'operating_margin' in company_data.columns:
                indicators['operating_margin'] = company_data['operating_margin'].mean()
            
            if 'roe' in company_data.columns:
                indicators['roe'] = company_data['roe'].mean()
            
            # 効率性指標
            if 'asset_turnover' in company_data.columns:
                indicators['asset_turnover'] = company_data['asset_turnover'].mean()
            
            return indicators
            
        except Exception as e:
            logger.error(f"財務指標計算エラー: {e}")
            return {}
    
    def _apply_lifecycle_normalization(
        self,
        company_data: pd.DataFrame,
        target_columns: List[str],
        stage_params: Dict,
        lifecycle_stage: str,
        entry_timing: str
    ) -> pd.DataFrame:
        """
        ライフサイクル段階別正規化の適用
        
        Args:
            company_data: 企業データ
            target_columns: 正規化対象カラム
            stage_params: 段階別パラメータ
            lifecycle_stage: ライフサイクル段階
            entry_timing: 参入時期
        
        Returns:
            正規化済み企業データ
        """
        try:
            normalized_data = company_data.copy()
            
            for column in target_columns:
                if column not in company_data.columns:
                    continue
                
                # 元データの取得
                original_values = company_data[column].values
                
                # ライフサイクル感度取得
                sensitivity = self.metric_sensitivity.get(column, 1.0)
                
                # 正規化係数計算
                volatility_adjustment = stage_params['volatility_factor']
                growth_adjustment = stage_params['growth_expectation']
                
                # 参入時期による調整
                entry_adjustment = self._get_entry_timing_adjustment(entry_timing)
                
                # 総合調整係数
                total_adjustment = volatility_adjustment * sensitivity * entry_adjustment
                
                # 正規化の適用
                if lifecycle_stage == self.lifecycle_stages.extinction:
                    # 消滅企業: 最終年に向けて減衰
                    decay_factor = np.linspace(1.0, 0.1, len(original_values))
                    normalized_values = original_values * decay_factor
                
                elif lifecycle_stage == self.lifecycle_stages.revival:
                    # 復活期: 初期は低く、徐々に回復
                    recovery_factor = np.linspace(0.3, 1.2, len(original_values))
                    normalized_values = original_values * recovery_factor
                
                else:
                    # 通常の正規化
                    # Z-score正規化後、ライフサイクル調整
                    if len(original_values) > 1 and np.std(original_values) > 0:
                        z_scores = (original_values - np.mean(original_values)) / np.std(original_values)
                        normalized_values = z_scores / total_adjustment
                    else:
                        normalized_values = original_values
                
                normalized_data[column] = normalized_values
            
            return normalized_data
            
        except Exception as e:
            logger.error(f"正規化適用エラー: {e}")
            return company_data
    
    def _get_entry_timing_adjustment(self, entry_timing: str) -> float:
        """
        参入時期による調整係数を取得
        
        Args:
            entry_timing: 参入時期分類
        
        Returns:
            調整係数
        """
        adjustments = {
            self.market_entry_timing.pioneer: 1.5,      # パイオニア優位
            self.market_entry_timing.early_adopter: 1.2, # 早期参入優位
            self.market_entry_timing.follower: 1.0,     # ベースライン
            self.market_entry_timing.late_entrant: 0.8  # 後発不利
        }
        
        return adjustments.get(entry_timing, 1.0)
    
    def create_lifecycle_benchmarks(
        self, 
        data: pd.DataFrame,
        benchmark_metrics: Optional[List[str]] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        ライフサイクル段階別ベンチマークの作成
        
        Args:
            data: 正規化済みデータ
            benchmark_metrics: ベンチマーク対象指標
        
        Returns:
            段階別ベンチマーク辞書
        """
        try:
            if benchmark_metrics is None:
                benchmark_metrics = list(self.metric_sensitivity.keys())
            
            benchmarks = {}
            
            # ライフサイクル段階ごとにベンチマーク作成
            for stage in [
                self.lifecycle_stages.startup,
                self.lifecycle_stages.growth,
                self.lifecycle_stages.maturity,
                self.lifecycle_stages.mature,
                self.lifecycle_stages.decline,
                self.lifecycle_stages.revival
            ]:
                stage_data = data[data['lifecycle_stage'] == stage]
                
                if len(stage_data) == 0:
                    continue
                
                stage_benchmark = {}
                
                for metric in benchmark_metrics:
                    if metric in stage_data.columns:
                        metric_data = stage_data[metric].dropna()
                        
                        if len(metric_data) > 0:
                            stage_benchmark[metric] = {
                                'mean': metric_data.mean(),
                                'median': metric_data.median(),
                                'std': metric_data.std(),
                                'q25': metric_data.quantile(0.25),
                                'q75': metric_data.quantile(0.75),
                                'min': metric_data.min(),
                                'max': metric_data.max(),
                                'count': len(metric_data)
                            }
                
                benchmarks[stage] = pd.DataFrame(stage_benchmark).T
            
            return benchmarks
            
        except Exception as e:
            logger.error(f"ベンチマーク作成エラー: {e}")
            return {}
    
    def get_normalization_summary(self) -> Dict:
        """
        正規化処理サマリーの取得
        
        Returns:
            正規化サマリー辞書
        """
        return {
            'lifecycle_stages': list(self.stage_parameters.keys()),
            'market_entry_timings': [
                self.market_entry_timing.pioneer,
                self.market_entry_timing.early_adopter,
                self.market_entry_timing.follower,
                self.market_entry_timing.late_entrant
            ],
            'metric_sensitivities': self.metric_sensitivity,
            'stage_parameters': self.stage_parameters,
            'processed_companies': len(self.company_profiles),
            'normalization_cache_size': len(self.normalization_cache)
        }

# 使用例とテスト用関数
def example_usage():
    """
    LifecycleNormalizerの使用例
    """
    # サンプルデータ作成
    np.random.seed(42)
    companies = ['ファナック', 'トヨタ自動車', '三洋電機', 'キオクシア']
    years = range(1984, 2024)
    
    data_list = []
    for company in companies:
        for year in years:
            # 企業・年度によって異なるトレンドを設定
            if company == '三洋電機' and year > 2010:
                # 三洋電機は2012年に消滅
                break
            elif company == 'キオクシア' and year < 2018:
                # キオクシアは2018年設立
                continue
            
            data_list.append({
                'company_name': company,
                'year': year,
                'sales': np.random.normal(1000, 200) + year - 2000,
                'operating_margin': np.random.normal(0.1, 0.03),
                'roe': np.random.normal(0.12, 0.05),
                'sales_growth_rate': np.random.normal(0.05, 0.1)
            })
    
    sample_data = pd.DataFrame(data_list)
    
    # 正規化処理
    normalizer = LifecycleNormalizer()
    normalized_data = normalizer.normalize_by_lifecycle_stage(sample_data)
    
    # ベンチマーク作成
    benchmarks = normalizer.create_lifecycle_benchmarks(normalized_data)
    
    # サマリー取得
    summary = normalizer.get_normalization_summary()
    
    print("LifecycleNormalizer 実行完了")
    print(f"正規化データ形状: {normalized_data.shape}")
    print(f"ベンチマーク段階数: {len(benchmarks)}")
    print(f"処理サマリー: {summary}")
    
    return normalized_data, benchmarks, summary

if __name__ == "__main__":
    # テスト実行
    normalized_data, benchmarks, summary = example_usage()
    
    # 結果確認
    print("\n=== 正規化結果サンプル ===")
    print(normalized_data.head(10))
    
    print("\n=== ライフサイクル段階分布 ===")
    if 'lifecycle_stage' in normalized_data.columns:
        print(normalized_data['lifecycle_stage'].value_counts())
    
    print("\n=== 参入時期分布 ===")
    if 'entry_timing' in normalized_data.columns:
        print(normalized_data['entry_timing'].value_counts())
    
    # ベンチマーク結果表示
    print("\n=== ライフサイクル段階別ベンチマーク ===")
    for stage, benchmark_df in benchmarks.items():
        print(f"\n{stage}段階:")
        if not benchmark_df.empty:
            print(benchmark_df[['mean', 'median', 'std']].round(4))