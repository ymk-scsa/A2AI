"""
A2AI (Advanced Financial Analysis AI) - Hazard Ratio Analyzer
企業生存分析におけるハザード比（危険度比）分析モジュール

このモジュールは、150社の財務諸表データを用いて、各要因項目が企業消滅リスク
（ハザード）に与える影響を定量化し、市場カテゴリ別の差異を分析する。

主な機能:
1. Cox回帰によるハザード比計算
2. 市場カテゴリ別ハザード比比較
3. 時変係数ハザードモデル
4. 要因項目重要度ランキング
5. 生存確率予測
"""

import numpy as np
import pandas as pd
import warnings
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# 生存分析ライブラリ
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
from lifelines.statistics import logrank_test
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy import stats
import joblib

warnings.filterwarnings('ignore')

@dataclass
class HazardRatioResult:
    """ハザード比分析結果を格納するデータクラス"""
    hazard_ratios: pd.Series
    confidence_intervals: pd.DataFrame
    p_values: pd.Series
    concordance_index: float
    log_likelihood: float
    aic: float
    partial_hazard: pd.DataFrame
    survival_function: pd.DataFrame

@dataclass
class MarketComparisonResult:
    """市場カテゴリ別比較結果を格納するデータクラス"""
    high_share_hazards: HazardRatioResult
    declining_hazards: HazardRatioResult
    lost_hazards: HazardRatioResult
    comparison_stats: pd.DataFrame
    significant_factors: List[str]

class HazardRatioAnalyzer:
    """
    企業生存分析におけるハザード比分析クラス
    
    Cox比例ハザードモデルを用いて、23の拡張要因項目が企業消滅リスクに
    与える影響を定量化し、市場カテゴリ別の差異を分析する。
    """
    
    def __init__(self, 
                    penalizer: float = 0.005,
                    l1_ratio: float = 0.0,
                    alpha: float = 0.05):
        """
        Args:
            penalizer: 正則化パラメータ
            l1_ratio: L1正則化とL2正則化の比率 (0: L2, 1: L1)
            alpha: 有意水準
        """
        self.penalizer = penalizer
        self.l1_ratio = l1_ratio
        self.alpha = alpha
        self.scaler = StandardScaler()
        self.cox_models = {}
        self.hazard_results = {}
        
        # 23の拡張要因項目定義
        self.factor_columns = self._define_factor_columns()
        
        # 市場カテゴリ定義
        self.market_categories = {
            'high_share': ['ロボット', '内視鏡', '工作機械', '電子材料', '精密測定機器'],
            'declining': ['自動車', '鉄鋼', 'スマート家電', 'バッテリー', 'PC・周辺機器'],
            'lost': ['家電', '半導体', 'スマートフォン', 'PC', '通信機器']
        }
        
    def _define_factor_columns(self) -> List[str]:
        """23の拡張要因項目を定義"""
        return [
            # 従来20項目（売上高の要因項目例）
            'tangible_fixed_assets', 'equipment_investment', 'rd_expenses',
            'intangible_assets', 'investment_securities', 'total_payout_ratio',
            'employee_count', 'average_salary', 'retirement_benefit_cost',
            'welfare_expenses', 'accounts_receivable', 'inventory',
            'total_assets', 'receivable_turnover', 'inventory_turnover',
            'overseas_sales_ratio', 'business_segments', 'sg_expenses',
            'advertising_expenses', 'non_operating_income',
            
            # 拡張3項目
            'company_age',           # 企業年齢（設立からの経過年数）
            'market_entry_timing',   # 市場参入時期（先発/後発効果）
            'parent_dependency'      # 親会社依存度（分社企業の場合）
        ]
    
    def prepare_survival_data(self, 
                                financial_data: pd.DataFrame,
                                company_info: pd.DataFrame) -> pd.DataFrame:
        """
        生存分析用データセットの準備
        
        Args:
            financial_data: 財務諸表データ（150社×40年分）
            company_info: 企業情報（設立年、消滅年、市場カテゴリ等）
            
        Returns:
            生存分析用統合データセット
        """
        print("生存分析用データセットを準備中...")
        
        # 企業ごとの生存時間とイベント発生を計算
        survival_data = []
        
        for company_id in financial_data['company_id'].unique():
            company_financial = financial_data[
                financial_data['company_id'] == company_id
            ].copy()
            
            company_meta = company_info[
                company_info['company_id'] == company_id
            ].iloc[0]
            
            # 生存時間の計算
            if pd.isna(company_meta['extinction_year']):
                # 企業が存続している場合
                duration = 2024 - company_meta['establishment_year']
                event_occurred = 0  # 右打ち切り
                last_year = 2024
            else:
                # 企業が消滅した場合
                duration = company_meta['extinction_year'] - company_meta['establishment_year']
                event_occurred = 1  # イベント発生
                last_year = int(company_meta['extinction_year'])
            
            # 最新の財務指標を取得（消滅前年または2023年）
            target_year = min(last_year - 1, 2023)
            recent_data = company_financial[
                company_financial['year'] == target_year
            ]
            
            if len(recent_data) == 0:
                # データがない場合、利用可能な最新年を使用
                recent_data = company_financial.iloc[-1]
                financial_metrics = recent_data.to_dict()
            else:
                financial_metrics = recent_data.iloc[0].to_dict()
            
            # 拡張要因項目の計算
            extended_factors = self._calculate_extended_factors(
                financial_metrics, company_meta, target_year
            )
            
            # 生存データの構築
            survival_record = {
                'company_id': company_id,
                'company_name': company_meta['company_name'],
                'market_category': company_meta['market_category'],
                'market_type': company_meta['market_type'],
                'duration': duration,
                'event_occurred': event_occurred,
                'establishment_year': company_meta['establishment_year'],
                'extinction_year': company_meta.get('extinction_year', np.nan),
                **extended_factors
            }
            
            survival_data.append(survival_record)
        
        survival_df = pd.DataFrame(survival_data)
        
        # 欠損値処理
        survival_df = self._handle_missing_values(survival_df)
        
        # 外れ値処理
        survival_df = self._handle_outliers(survival_df)
        
        print(f"生存分析データセット準備完了: {len(survival_df)}社")
        print(f"イベント発生企業数: {survival_df['event_occurred'].sum()}社")
        print(f"右打ち切り企業数: {len(survival_df) - survival_df['event_occurred'].sum()}社")
        
        return survival_df
    
    def _calculate_extended_factors(self, 
                                    financial_data: dict, 
                                    company_meta: pd.Series,
                                    target_year: int) -> dict:
        """拡張要因項目の計算"""
        factors = {}
        
        # 従来の20項目は既に財務データに含まれていると仮定
        for factor in self.factor_columns[:20]:
            factors[factor] = financial_data.get(factor, np.nan)
        
        # 企業年齢
        factors['company_age'] = target_year - company_meta['establishment_year']
        
        # 市場参入時期（業界平均設立年との差）
        industry_avg_establishment = self._get_industry_average_establishment_year(
            company_meta['market_type']
        )
        factors['market_entry_timing'] = (
            company_meta['establishment_year'] - industry_avg_establishment
        )
        
        # 親会社依存度（分社企業の場合）
        if company_meta.get('is_spinoff', False):
            # 分社企業の場合、売上に占める親会社向け売上比率等で算出
            factors['parent_dependency'] = financial_data.get('parent_sales_ratio', 0.5)
        else:
            factors['parent_dependency'] = 0.0
        
        return factors
    
    def _get_industry_average_establishment_year(self, market_type: str) -> float:
        """業界平均設立年の取得（簡略化）"""
        # 実際の実装では、各市場の企業設立年の平均を計算
        industry_averages = {
            'ロボット': 1960, '内視鏡': 1950, '工作機械': 1945,
            '電子材料': 1955, '精密測定機器': 1950,
            '自動車': 1945, '鉄鋼': 1940, 'スマート家電': 1950,
            'バッテリー': 1960, 'PC・周辺機器': 1970,
            '家電': 1950, '半導体': 1965, 'スマートフォン': 1980,
            'PC': 1975, '通信機器': 1970
        }
        return industry_averages.get(market_type, 1955)
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """欠損値処理"""
        # 数値項目の欠損値を業界中央値で補完
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col not in ['duration', 'event_occurred']:
                # 市場カテゴリ別中央値で補完
                df[col] = df.groupby('market_category')[col].transform(
                    lambda x: x.fillna(x.median())
                )
        
        return df
    
    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """外れ値処理（Winsorization）"""
        numeric_cols = [col for col in self.factor_columns if col in df.columns]
        
        for col in numeric_cols:
            q1 = df[col].quantile(0.05)
            q99 = df[col].quantile(0.95)
            df[col] = np.clip(df[col], q1, q99)
        
        return df
    
    def fit_cox_model(self, 
                        survival_data: pd.DataFrame,
                        market_filter: Optional[str] = None) -> HazardRatioResult:
        """
        Cox比例ハザードモデルの学習
        
        Args:
            survival_data: 生存分析用データセット
            market_filter: 市場カテゴリフィルタ ('high_share', 'declining', 'lost')
            
        Returns:
            ハザード比分析結果
        """
        print(f"Cox回帰モデル学習開始 (市場フィルタ: {market_filter})")
        
        # データフィルタリング
        if market_filter:
            if market_filter == 'high_share':
                filtered_data = survival_data[
                    survival_data['market_type'].isin(self.market_categories['high_share'])
                ].copy()
            elif market_filter == 'declining':
                filtered_data = survival_data[
                    survival_data['market_type'].isin(self.market_categories['declining'])
                ].copy()
            elif market_filter == 'lost':
                filtered_data = survival_data[
                    survival_data['market_type'].isin(self.market_categories['lost'])
                ].copy()
            else:
                filtered_data = survival_data.copy()
        else:
            filtered_data = survival_data.copy()
        
        print(f"分析対象企業数: {len(filtered_data)}社")
        
        # 特徴量の標準化
        feature_cols = [col for col in self.factor_columns if col in filtered_data.columns]
        
        if len(filtered_data) < 10:
            print("警告: データ数が少なすぎます。分析を継続しますが結果の信頼性は低い可能性があります。")
        
        # 標準化
        filtered_data[feature_cols] = self.scaler.fit_transform(filtered_data[feature_cols])
        
        # Cox回帰の実行
        cph = CoxPHFitter(
            penalizer=self.penalizer,
            l1_ratio=self.l1_ratio,
            alpha=self.alpha
        )
        
        cox_data = filtered_data[feature_cols + ['duration', 'event_occurred']].copy()
        
        try:
            cph.fit(
                cox_data,
                duration_col='duration',
                event_col='event_occurred',
                show_progress=False
            )
            
            # 結果の抽出
            hazard_ratios = np.exp(cph.params_)  # exp(β) = hazard ratio
            confidence_intervals = np.exp(cph.confidence_intervals_)
            p_values = cph.summary['p']
            concordance_idx = cph.concordance_index_
            log_likelihood = cph.log_likelihood_
            aic = cph.AIC_partial_
            
            # 部分ハザード関数の計算
            partial_hazard = self._calculate_partial_hazard(cph, cox_data)
            
            # 生存関数の計算
            survival_function = self._calculate_survival_function(cph, cox_data)
            
            result = HazardRatioResult(
                hazard_ratios=hazard_ratios,
                confidence_intervals=confidence_intervals,
                p_values=p_values,
                concordance_index=concordance_idx,
                log_likelihood=log_likelihood,
                aic=aic,
                partial_hazard=partial_hazard,
                survival_function=survival_function
            )
            
            # モデル保存
            model_key = market_filter if market_filter else 'all_markets'
            self.cox_models[model_key] = cph
            self.hazard_results[model_key] = result
            
            print(f"Cox回帰完了 - C-index: {concordance_idx:.3f}, AIC: {aic:.2f}")
            
            return result
            
        except Exception as e:
            print(f"Cox回帰でエラーが発生: {e}")
            # エラー時はダミーの結果を返す
            dummy_result = HazardRatioResult(
                hazard_ratios=pd.Series(index=feature_cols, data=1.0),
                confidence_intervals=pd.DataFrame(index=feature_cols),
                p_values=pd.Series(index=feature_cols, data=1.0),
                concordance_index=0.5,
                log_likelihood=-1000.0,
                aic=2000.0,
                partial_hazard=pd.DataFrame(),
                survival_function=pd.DataFrame()
            )
            return dummy_result
    
    def _calculate_partial_hazard(self, 
                                    model: CoxPHFitter, 
                                    data: pd.DataFrame) -> pd.DataFrame:
        """部分ハザード関数の計算"""
        try:
            partial_hazard = model.predict_partial_hazard(data)
            return partial_hazard.reset_index()
        except:
            return pd.DataFrame()
    
    def _calculate_survival_function(self, 
                                    model: CoxPHFitter, 
                                    data: pd.DataFrame) -> pd.DataFrame:
        """生存関数の計算"""
        try:
            # 平均的な企業プロファイルでの生存関数
            mean_profile = data[model.params_.index].mean().to_frame().T
            survival_func = model.predict_survival_function(mean_profile)
            return survival_func.reset_index()
        except:
            return pd.DataFrame()
    
    def compare_market_hazards(self, survival_data: pd.DataFrame) -> MarketComparisonResult:
        """
        市場カテゴリ別ハザード比の比較分析
        
        Args:
            survival_data: 生存分析用データセット
            
        Returns:
            市場比較分析結果
        """
        print("市場カテゴリ別ハザード比比較分析開始...")
        
        # 各市場カテゴリでCox回帰実行
        high_share_result = self.fit_cox_model(survival_data, 'high_share')
        declining_result = self.fit_cox_model(survival_data, 'declining')
        lost_result = self.fit_cox_model(survival_data, 'lost')
        
        # 統計的比較
        comparison_stats = self._statistical_comparison_hazards(
            high_share_result, declining_result, lost_result
        )
        
        # 有意差のある要因項目を特定
        significant_factors = self._identify_significant_factors(
            high_share_result, declining_result, lost_result
        )
        
        result = MarketComparisonResult(
            high_share_hazards=high_share_result,
            declining_hazards=declining_result,
            lost_hazards=lost_result,
            comparison_stats=comparison_stats,
            significant_factors=significant_factors
        )
        
        print(f"市場比較完了 - 有意差要因数: {len(significant_factors)}")
        
        return result
    
    def _statistical_comparison_hazards(self, 
                                        high: HazardRatioResult,
                                        declining: HazardRatioResult, 
                                        lost: HazardRatioResult) -> pd.DataFrame:
        """ハザード比の統計的比較"""
        factors = high.hazard_ratios.index
        
        comparison_data = []
        for factor in factors:
            try:
                high_hr = high.hazard_ratios[factor]
                declining_hr = declining.hazard_ratios[factor]
                lost_hr = lost.hazard_ratios[factor]
                
                high_p = high.p_values[factor]
                declining_p = declining.p_values[factor]
                lost_p = lost.p_values[factor]
                
                # 効果サイズの計算
                high_vs_declining = abs(np.log(high_hr) - np.log(declining_hr))
                high_vs_lost = abs(np.log(high_hr) - np.log(lost_hr))
                declining_vs_lost = abs(np.log(declining_hr) - np.log(lost_hr))
                
                comparison_data.append({
                    'factor': factor,
                    'high_share_hr': high_hr,
                    'declining_hr': declining_hr,
                    'lost_hr': lost_hr,
                    'high_share_p': high_p,
                    'declining_p': declining_p,
                    'lost_p': lost_p,
                    'effect_size_high_vs_declining': high_vs_declining,
                    'effect_size_high_vs_lost': high_vs_lost,
                    'effect_size_declining_vs_lost': declining_vs_lost,
                    'max_effect_size': max(high_vs_declining, high_vs_lost, declining_vs_lost)
                })
            except:
                continue
        
        return pd.DataFrame(comparison_data)
    
    def _identify_significant_factors(self, 
                                    high: HazardRatioResult,
                                    declining: HazardRatioResult,
                                    lost: HazardRatioResult) -> List[str]:
        """有意差のある要因項目を特定"""
        significant = []
        
        factors = high.hazard_ratios.index
        
        for factor in factors:
            try:
                # 各市場で有意かつ、効果サイズが大きい要因を特定
                high_significant = high.p_values[factor] < self.alpha
                declining_significant = declining.p_values[factor] < self.alpha
                lost_significant = lost.p_values[factor] < self.alpha
                
                # 少なくとも2つの市場で有意
                if sum([high_significant, declining_significant, lost_significant]) >= 2:
                    # ハザード比の差が実質的に意味がある（1.5倍以上の差）
                    hrs = [
                        high.hazard_ratios[factor],
                        declining.hazard_ratios[factor],
                        lost.hazard_ratios[factor]
                    ]
                    
                    if max(hrs) / min(hrs) > 1.5:
                        significant.append(factor)
            except:
                continue
        
        return significant
    
    def analyze_factor_importance(self, 
                                survival_data: pd.DataFrame) -> pd.DataFrame:
        """
        要因項目重要度の分析
        
        Args:
            survival_data: 生存分析用データセット
            
        Returns:
            要因項目重要度ランキング
        """
        print("要因項目重要度分析開始...")
        
        # 全市場でのCox回帰
        overall_result = self.fit_cox_model(survival_data)
        
        # 重要度指標の計算
        importance_data = []
        
        for factor in overall_result.hazard_ratios.index:
            try:
                hr = overall_result.hazard_ratios[factor]
                p_value = overall_result.p_values[factor]
                
                # 重要度スコアの計算
                # |log(HR)| * (-log(p-value)) で効果サイズと有意性を組み合わせ
                log_hr = abs(np.log(hr))
                neg_log_p = -np.log(max(p_value, 1e-10))  # ゼロ除算回避
                
                importance_score = log_hr * neg_log_p
                
                # 信頼区間の幅（不確実性の指標）
                ci_lower = overall_result.confidence_intervals.loc[factor, 'coef lower 95%']
                ci_upper = overall_result.confidence_intervals.loc[factor, 'coef upper 95%']
                ci_width = ci_upper - ci_lower
                
                importance_data.append({
                    'factor': factor,
                    'hazard_ratio': hr,
                    'p_value': p_value,
                    'log_hazard_ratio': np.log(hr),
                    'abs_log_hazard_ratio': log_hr,
                    'neg_log_p_value': neg_log_p,
                    'importance_score': importance_score,
                    'ci_width': ci_width,
                    'significant': p_value < self.alpha,
                    'effect_direction': 'increase_risk' if hr > 1 else 'decrease_risk'
                })
            except:
                continue
        
        importance_df = pd.DataFrame(importance_data)
        
        # 重要度でソート
        importance_df = importance_df.sort_values('importance_score', ascending=False)
        
        print(f"要因項目重要度分析完了 - 有意要因数: {importance_df['significant'].sum()}")
        
        return importance_df
    
    def predict_survival_probability(self, 
                                    company_features: pd.DataFrame,
                                    time_points: List[int],
                                    market_category: str = 'all') -> pd.DataFrame:
        """
        企業の生存確率予測
        
        Args:
            company_features: 企業の特徴量データ
            time_points: 予測時点（年数）
            market_category: 使用する市場カテゴリモデル
            
        Returns:
            生存確率予測結果
        """
        model_key = market_category if market_category != 'all' else 'all_markets'
        
        if model_key not in self.cox_models:
            raise ValueError(f"モデル '{model_key}' が学習されていません。")
        
        model = self.cox_models[model_key]
        
        # 特徴量の標準化
        feature_cols = [col for col in self.factor_columns if col in company_features.columns]
        scaled_features = company_features[feature_cols].copy()
        scaled_features[feature_cols] = self.scaler.transform(scaled_features[feature_cols])
        
        # 生存確率の予測
        survival_functions = model.predict_survival_function(scaled_features)
        
        # 指定された時点での生存確率を抽出
        predictions = []
        
        for i, company_id in enumerate(company_features.index):
            company_survival = survival_functions.iloc[:, i]
            
            for time_point in time_points:
                # 最も近い時点の生存確率を取得
                closest_time = min(company_survival.index, 
                                    key=lambda x: abs(x - time_point))
                survival_prob = company_survival.loc[closest_time]
                
                predictions.append({
                    'company_id': company_id,
                    'time_point': time_point,
                    'survival_probability': survival_prob,
                    'hazard_risk': 1 - survival_prob
                })
        
        return pd.DataFrame(predictions)
    
    def generate_hazard_report(self, 
                                market_comparison: MarketComparisonResult,
                                factor_importance: pd.DataFrame) -> Dict[str, any]:
        """
        ハザード比分析レポートの生成
        
        Args:
            market_comparison: 市場比較結果
            factor_importance: 要因重要度結果
            
        Returns:
            分析レポート辞書
        """
        report = {
            'analysis_summary': {
                'total_factors_analyzed': len(factor_importance),
                'significant_factors_overall': factor_importance['significant'].sum(),
                'significant_factors_market_diff': len(market_comparison.significant_factors),
                'best_cox_model': self._find_best_model(),
            },
            
            'market_category_insights': {
                'high_share_markets': {
                    'concordance_index': market_comparison.high_share_hazards.concordance_index,
                    'top_risk_factors': self._get_top_risk_factors(
                        market_comparison.high_share_hazards
                    ),
                    'top_protective_factors': self._get_top_protective_factors(
                        market_comparison.high_share_hazards
                    )
                },
                'declining_markets': {
                    'concordance_index': market_comparison.declining_hazards.concordance_index,
                    'top_risk_factors': self._get_top_risk_factors(
                        market_comparison.declining_hazards
                    ),
                    'top_protective_factors': self._get_top_protective_factors(
                        market_comparison.declining_hazards
                    )
                },
                'lost_markets': {
                    'concordance_index': market_comparison.lost_hazards.concordance_index,
                    'top_risk_factors': self._get_top_risk_factors(
                        market_comparison.lost_hazards
                    ),
                    'top_protective_factors': self._get_top_protective_factors(
                        market_comparison.lost_hazards
                    )
                }
            },
            
            'factor_importance_ranking': factor_importance.head(10).to_dict('records'),
            
            'significant_market_differences': [
                {
                    'factor': factor,
                    'market_hazard_ratios': self._get_factor_market_hazards(
                        factor, market_comparison
                    )
                }
                for factor in market_comparison.significant_factors[:10]
            ],
            
            'strategic_insights': self._generate_strategic_insights(
                market_comparison, factor_importance
            )
        }
        
        return report
    
    def _find_best_model(self) -> str:
        """最適なモデルを特定（C-indexで評価）"""
        best_model = 'all_markets'
        best_concordance = 0.0
        
        for model_key, result in self.hazard_results.items():
            if result.concordance_index > best_concordance:
                best_concordance = result.concordance_index
                best_model = model_key
        
        return best_model
    
    def _get_top_risk_factors(self, hazard_result: HazardRatioResult, top_n: int = 5) -> List[Dict]:
        """リスクを高める上位要因を取得"""
        risk_factors = []
        
        for factor in hazard_result.hazard_ratios.index:
            hr = hazard_result.hazard_ratios[factor]
            p_val = hazard_result.p_values[factor]
            
            if hr > 1 and p_val < self.alpha:  # リスクを高め、有意
                risk_factors.append({
                    'factor': factor,
                    'hazard_ratio': hr,
                    'p_value': p_val,
                    'risk_increase_pct': (hr - 1) * 100
                })
        
        # ハザード比でソート
        risk_factors.sort(key=lambda x: x['hazard_ratio'], reverse=True)
        return risk_factors[:top_n]
    
    def _get_top_protective_factors(self, hazard_result: HazardRatioResult, top_n: int = 5) -> List[Dict]:
        """リスクを下げる上位要因を取得"""
        protective_factors = []
        
        for factor in hazard_result.hazard_ratios.index:
            hr = hazard_result.hazard_ratios[factor]
            p_val = hazard_result.p_values[factor]
            
            if hr < 1 and p_val < self.alpha:  # リスクを下げ、有意
                protective_factors.append({
                    'factor': factor,
                    'hazard_ratio': hr,
                    'p_value': p_val,
                    'risk_reduction_pct': (1 - hr) * 100
                })
        
        # ハザード比でソート（小さい順）
        protective_factors.sort(key=lambda x: x['hazard_ratio'])
        return protective_factors[:top_n]
    
    def _get_factor_market_hazards(self, 
                                    factor: str, 
                                    market_comparison: MarketComparisonResult) -> Dict:
        """特定要因の市場別ハザード比を取得"""
        try:
            return {
                'high_share': float(market_comparison.high_share_hazards.hazard_ratios[factor]),
                'declining': float(market_comparison.declining_hazards.hazard_ratios[factor]),
                'lost': float(market_comparison.lost_hazards.hazard_ratios[factor])
            }
        except:
            return {'high_share': 1.0, 'declining': 1.0, 'lost': 1.0}
    
    def _generate_strategic_insights(self, 
                                    market_comparison: MarketComparisonResult,
                                    factor_importance: pd.DataFrame) -> List[str]:
        """戦略的インサイトの生成"""
        insights = []
        
        # 1. 最も重要な生存要因
        top_factor = factor_importance.iloc[0]
        if top_factor['significant']:
            direction = "企業消滅リスクを増加" if top_factor['hazard_ratio'] > 1 else "企業生存確率を向上"
            insights.append(
                f"最も重要な生存要因は「{top_factor['factor']}」で、{direction}させる効果がある。"
            )
        
        # 2. 市場カテゴリ別の特徴
        concordances = {
            'high_share': market_comparison.high_share_hazards.concordance_index,
            'declining': market_comparison.declining_hazards.concordance_index,
            'lost': market_comparison.lost_hazards.concordance_index
        }
        
        best_predictable = max(concordances, key=concordances.get)
        market_names = {
            'high_share': '高シェア市場', 
            'declining': 'シェア低下市場',
            'lost': 'シェア失失市場'
        }
        
        insights.append(
            f"{market_names[best_predictable]}が最も予測しやすい生存パターンを示している "
            f"(C-index: {concordances[best_predictable]:.3f})。"
        )
        
        # 3. 新発見の要因項目効果
        extended_factors = ['company_age', 'market_entry_timing', 'parent_dependency']
        for factor in extended_factors:
            if factor in factor_importance['factor'].values:
                factor_data = factor_importance[
                    factor_importance['factor'] == factor
                ].iloc[0]
                
                if factor_data['significant']:
                    factor_names = {
                        'company_age': '企業年齢',
                        'market_entry_timing': '市場参入タイミング',
                        'parent_dependency': '親会社依存度'
                    }
                    
                    effect = "リスク要因" if factor_data['hazard_ratio'] > 1 else "保護要因"
                    insights.append(
                        f"拡張要因「{factor_names[factor]}」が有意な{effect}として特定された。"
                    )
        
        # 4. 市場間差異の重要な発見
        if len(market_comparison.significant_factors) > 0:
            top_diff_factor = market_comparison.significant_factors[0]
            market_hazards = self._get_factor_market_hazards(
                top_diff_factor, market_comparison
            )
            
            max_market = max(market_hazards, key=market_hazards.get)
            min_market = min(market_hazards, key=market_hazards.get)
            
            insights.append(
                f"「{top_diff_factor}」は市場間で最も大きな効果差を示し、"
                f"{market_names[max_market]}で最もリスクが高い。"
            )
        
        # 5. 生存戦略の提言
        protective_factors_count = factor_importance[
            (factor_importance['hazard_ratio'] < 1) & 
            (factor_importance['significant'] == True)
        ].shape[0]
        
        risk_factors_count = factor_importance[
            (factor_importance['hazard_ratio'] > 1) & 
            (factor_importance['significant'] == True)
        ].shape[0]
        
        if protective_factors_count > risk_factors_count:
            insights.append(
                f"分析された要因の多くが保護効果を持つため、"
                f"これらの要因を強化することで企業生存確率を向上できる。"
            )
        else:
            insights.append(
                f"多くの要因がリスク要因として作用するため、"
                f"慎重なリスク管理が企業生存には重要である。"
            )
        
        return insights
    
    def visualize_hazard_ratios(self, 
                                market_comparison: MarketComparisonResult,
                                save_path: Optional[str] = None) -> plt.Figure:
        """
        ハザード比の可視化
        
        Args:
            market_comparison: 市場比較結果
            save_path: 保存パス（指定時）
            
        Returns:
            matplotlib Figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('企業生存分析：市場カテゴリ別ハザード比比較', fontsize=16, fontweight='bold')
        
        # 1. 市場カテゴリ別ハザード比比較（上位要因）
        ax1 = axes[0, 0]
        
        # 有意な要因のみ抽出
        significant_factors = market_comparison.significant_factors[:10]
        if len(significant_factors) > 0:
            high_hrs = [market_comparison.high_share_hazards.hazard_ratios[f] 
                        for f in significant_factors]
            declining_hrs = [market_comparison.declining_hazards.hazard_ratios[f] 
                            for f in significant_factors]
            lost_hrs = [market_comparison.lost_hazards.hazard_ratios[f] 
                        for f in significant_factors]
            
            x = np.arange(len(significant_factors))
            width = 0.25
            
            ax1.bar(x - width, high_hrs, width, label='高シェア市場', alpha=0.8, color='green')
            ax1.bar(x, declining_hrs, width, label='シェア低下市場', alpha=0.8, color='orange')
            ax1.bar(x + width, lost_hrs, width, label='シェア失失市場', alpha=0.8, color='red')
            
            ax1.axhline(y=1, color='black', linestyle='--', alpha=0.5)
            ax1.set_xlabel('要因項目')
            ax1.set_ylabel('ハザード比')
            ax1.set_title('市場カテゴリ別ハザード比比較（有意要因）')
            ax1.set_xticks(x)
            ax1.set_xticklabels([f.replace('_', '\n') for f in significant_factors], 
                                rotation=45, ha='right')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # 2. 信頼区間付きハザード比（高シェア市場）
        ax2 = axes[0, 1]
        
        high_share_result = market_comparison.high_share_hazards
        factors = high_share_result.hazard_ratios.index[:15]  # 上位15要因
        
        hrs = high_share_result.hazard_ratios[factors]
        
        # 信頼区間の計算（対数スケールで）
        try:
            ci_lower = np.exp(high_share_result.confidence_intervals.loc[factors, 'coef lower 95%'])
            ci_upper = np.exp(high_share_result.confidence_intervals.loc[factors, 'coef upper 95%'])
            
            y_pos = np.arange(len(factors))
            
            ax2.barh(y_pos, hrs, alpha=0.7, 
                    color=['red' if hr > 1 else 'green' for hr in hrs])
            ax2.errorbar(hrs, y_pos, xerr=[hrs - ci_lower, ci_upper - hrs], 
                        fmt='none', ecolor='black', alpha=0.5)
        except:
            y_pos = np.arange(len(factors))
            ax2.barh(y_pos, hrs, alpha=0.7,
                    color=['red' if hr > 1 else 'green' for hr in hrs])
        
        ax2.axvline(x=1, color='black', linestyle='--', alpha=0.5)
        ax2.set_xlabel('ハザード比')
        ax2.set_ylabel('要因項目')
        ax2.set_title('高シェア市場：ハザード比と信頼区間')
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels([f.replace('_', ' ') for f in factors])
        ax2.set_xscale('log')
        ax2.grid(True, alpha=0.3)
        
        # 3. 市場別C-index比較
        ax3 = axes[1, 0]
        
        concordances = [
            market_comparison.high_share_hazards.concordance_index,
            market_comparison.declining_hazards.concordance_index,
            market_comparison.lost_hazards.concordance_index
        ]
        market_labels = ['高シェア市場', 'シェア低下市場', 'シェア失失市場']
        colors = ['green', 'orange', 'red']
        
        bars = ax3.bar(market_labels, concordances, color=colors, alpha=0.7)
        ax3.set_ylabel('Concordance Index')
        ax3.set_title('市場カテゴリ別予測精度（C-index）')
        ax3.set_ylim(0.4, max(concordances) + 0.1)
        
        # 値をバーの上に表示
        for bar, c_index in zip(bars, concordances):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{c_index:.3f}', ha='center', va='bottom')
        
        ax3.grid(True, alpha=0.3)
        
        # 4. 要因効果の散布図（効果サイズ vs 有意性）
        ax4 = axes[1, 1]
        
        # 全市場での結果を使用
        all_market_result = self.hazard_results.get('all_markets')
        if all_market_result:
            log_hrs = np.log(all_market_result.hazard_ratios).abs()
            neg_log_ps = -np.log(all_market_result.p_values.clip(lower=1e-10))
            
            # 有意性で色分け
            colors = ['red' if p < self.alpha else 'gray' 
                        for p in all_market_result.p_values]
            
            scatter = ax4.scatter(log_hrs, neg_log_ps, c=colors, alpha=0.6, s=50)
            
            # 有意性の閾値線
            ax4.axhline(y=-np.log(self.alpha), color='red', linestyle='--', alpha=0.5,
                        label=f'有意水準 (α={self.alpha})')
            
            ax4.set_xlabel('|log(Hazard Ratio)| (効果サイズ)')
            ax4.set_ylabel('-log(p-value) (有意性)')
            ax4.set_title('要因効果：効果サイズ vs 有意性')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def export_results(self, 
                        market_comparison: MarketComparisonResult,
                        factor_importance: pd.DataFrame,
                        output_dir: str) -> Dict[str, str]:
        """
        分析結果のエクスポート
        
        Args:
            market_comparison: 市場比較結果
            factor_importance: 要因重要度結果
            output_dir: 出力ディレクトリ
            
        Returns:
            エクスポートファイルパス辞書
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        exported_files = {}
        
        # 1. 市場比較統計の出力
        comparison_path = os.path.join(output_dir, 'market_hazard_comparison.csv')
        market_comparison.comparison_stats.to_csv(comparison_path, index=False, encoding='utf-8-sig')
        exported_files['market_comparison'] = comparison_path
        
        # 2. 要因重要度ランキングの出力
        importance_path = os.path.join(output_dir, 'factor_importance_ranking.csv')
        factor_importance.to_csv(importance_path, index=False, encoding='utf-8-sig')
        exported_files['factor_importance'] = importance_path
        
        # 3. 市場別ハザード比詳細
        for market_type, result in [
            ('high_share', market_comparison.high_share_hazards),
            ('declining', market_comparison.declining_hazards),
            ('lost', market_comparison.lost_hazards)
        ]:
            hazard_detail = pd.DataFrame({
                'factor': result.hazard_ratios.index,
                'hazard_ratio': result.hazard_ratios.values,
                'p_value': result.p_values.values,
                'significant': result.p_values.values < self.alpha
            })
            
            detail_path = os.path.join(output_dir, f'{market_type}_hazard_details.csv')
            hazard_detail.to_csv(detail_path, index=False, encoding='utf-8-sig')
            exported_files[f'{market_type}_details'] = detail_path
        
        # 4. 学習済みモデルの保存
        models_path = os.path.join(output_dir, 'cox_models.joblib')
        joblib.dump(self.cox_models, models_path)
        exported_files['models'] = models_path
        
        # 5. 分析レポートの保存
        report = self.generate_hazard_report(market_comparison, factor_importance)
        report_path = os.path.join(output_dir, 'hazard_analysis_report.json')
        
        import json
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        exported_files['report'] = report_path
        
        print(f"分析結果を {output_dir} にエクスポートしました。")
        
        return exported_files

# 使用例とメイン実行部分
def main():
    """
    ハザード比分析の実行例
    """
    # インスタンス作成
    analyzer = HazardRatioAnalyzer(penalizer=0.01, alpha=0.05)
    
    # サンプルデータの生成（実際の使用では実データを読み込み）
    print("サンプルデータ生成中...")
    
    # 財務諸表データのサンプル
    np.random.seed(42)
    n_companies = 150
    n_years = 40
    
    companies = [f"company_{i:03d}" for i in range(n_companies)]
    years = list(range(1984, 2024))
    
    # 市場カテゴリ別の企業分散
    market_types = (
        ['ロボット'] * 10 + ['内視鏡'] * 10 + ['工作機械'] * 10 + 
        ['電子材料'] * 10 + ['精密測定機器'] * 10 +  # 高シェア市場
        ['自動車'] * 10 + ['鉄鋼'] * 10 + ['スマート家電'] * 10 + 
        ['バッテリー'] * 10 + ['PC・周辺機器'] * 10 +  # シェア低下市場
        ['家電'] * 10 + ['半導体'] * 10 + ['スマートフォン'] * 10 + 
        ['PC'] * 10 + ['通信機器'] * 10  # シェア失失市場
    )
    
    financial_data_list = []
    company_info_list = []
    
    for i, company in enumerate(companies):
        market_type = market_types[i]
        
        # 市場カテゴリの決定
        if market_type in analyzer.market_categories['high_share']:
            market_category = 'high_share'
            extinction_prob = 0.1  # 低い消滅確率
        elif market_type in analyzer.market_categories['declining']:
            market_category = 'declining'
            extinction_prob = 0.3  # 中程度の消滅確率
        else:
            market_category = 'lost'
            extinction_prob = 0.6  # 高い消滅確率
        
        # 企業設立年（ランダム）
        establishment_year = np.random.randint(1950, 1990)
        
        # 消滅年の決定（確率的）
        if np.random.random() < extinction_prob:
            extinction_year = np.random.randint(establishment_year + 10, 2024)
            max_year = min(extinction_year, 2023)
        else:
            extinction_year = np.nan
            max_year = 2023
        
        # 企業情報
        company_info_list.append({
            'company_id': company,
            'company_name': f"企業{i+1}",
            'market_type': market_type,
            'market_category': market_category,
            'establishment_year': establishment_year,
            'extinction_year': extinction_year,
            'is_spinoff': np.random.random() < 0.2
        })
        
        # 財務データ生成（簡略化）
        active_years = [y for y in years 
                        if establishment_year <= y <= max_year]
        
        for year in active_years:
            # 基本的な財務指標をランダム生成
            financial_record = {
                'company_id': company,
                'year': year,
                'tangible_fixed_assets': np.random.lognormal(8, 1),
                'equipment_investment': np.random.lognormal(6, 1),
                'rd_expenses': np.random.lognormal(5, 1),
                'intangible_assets': np.random.lognormal(7, 1),
                'investment_securities': np.random.lognormal(6, 1),
                'total_payout_ratio': np.random.normal(0.3, 0.2),
                'employee_count': np.random.randint(100, 10000),
                'average_salary': np.random.normal(600, 100),
                'retirement_benefit_cost': np.random.lognormal(4, 1),
                'welfare_expenses': np.random.lognormal(4, 1),
                'accounts_receivable': np.random.lognormal(7, 1),
                'inventory': np.random.lognormal(7, 1),
                'total_assets': np.random.lognormal(9, 1),
                'receivable_turnover': np.random.normal(6, 2),
                'inventory_turnover': np.random.normal(4, 1),
                'overseas_sales_ratio': np.random.normal(0.4, 0.3),
                'business_segments': np.random.randint(1, 8),
                'sg_expenses': np.random.lognormal(7, 1),
                'advertising_expenses': np.random.lognormal(5, 1),
                'non_operating_income': np.random.lognormal(5, 1)
            }
            
            financial_data_list.append(financial_record)
    
    # DataFrameに変換
    financial_df = pd.DataFrame(financial_data_list)
    company_info_df = pd.DataFrame(company_info_list)
    
    print(f"データ生成完了 - 企業数: {len(company_info_df)}, 財務データ点数: {len(financial_df)}")
    
    # 1. 生存分析用データ準備
    print("\n=== 生存分析用データ準備 ===")
    survival_data = analyzer.prepare_survival_data(financial_df, company_info_df)
    
    # 2. 市場カテゴリ別ハザード比比較
    print("\n=== 市場カテゴリ別ハザード比分析 ===")
    market_comparison = analyzer.compare_market_hazards(survival_data)
    
    # 3. 要因項目重要度分析
    print("\n=== 要因項目重要度分析 ===")
    factor_importance = analyzer.analyze_factor_importance(survival_data)
    
    # 4. 結果表示
    print("\n=== 分析結果サマリー ===")
    print("\n上位5要因の重要度:")
    print(factor_importance[['factor', 'hazard_ratio', 'p_value', 'importance_score']].head())
    
    print(f"\n市場間で有意差のある要因数: {len(market_comparison.significant_factors)}")
    if len(market_comparison.significant_factors) > 0:
        print("市場間差異要因（上位5）:")
        for factor in market_comparison.significant_factors[:5]:
            hazards = analyzer._get_factor_market_hazards(factor, market_comparison)
            print(f"  {factor}: 高シェア={hazards['high_share']:.3f}, "
                    f"低下={hazards['declining']:.3f}, 失失={hazards['lost']:.3f}")
    
    # 5. レポート生成
    print("\n=== 分析レポート生成 ===")
    report = analyzer.generate_hazard_report(market_comparison, factor_importance)
    
    print("\n戦略的インサイト:")
    for insight in report['strategic_insights']:
        print(f"  • {insight}")
    
    # 6. 可視化
    print("\n=== 可視化生成 ===")
    fig = analyzer.visualize_hazard_ratios(market_comparison)
    plt.show()
    
    # 7. 生存確率予測例
    print("\n=== 生存確率予測例 ===")
    if len(survival_data) > 0:
        # サンプル企業での予測
        sample_features = survival_data[analyzer.factor_columns].iloc[:5]
        predictions = analyzer.predict_survival_probability(
            sample_features, 
            time_points=[5, 10, 20], 
            market_category='all_markets'
        )
        print("5年後、10年後、20年後の生存確率:")
        print(predictions[['company_id', 'time_point', 'survival_probability']].head(10))
    
    print("\n=== ハザード比分析完了 ===")

if __name__ == "__main__":
    main()