"""
Difference-in-Differences (DID) Analysis for A2AI Financial Analysis
===================================================================

差分差分法による因果推論分析モジュール

企業再編、政策変更、市場環境変化が財務指標に与える因果効果を測定。
150社×40年分の財務データを活用し、処置群・対照群の設定により真の因果関係を特定。

主要機能:
- 標準的差分差分法 (Standard DID)
- 時系列差分差分法 (Time-varying DID)
- 複数期間差分差分法 (Multi-period DID)
- 頑健性検定 (Robustness Tests)
- 平行トレンド仮定の検証 (Parallel Trends Test)

Author: A2AI Development Team
Date: 2024-08-22
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import scipy.stats as stats
from scipy.optimize import minimize
import warnings
import logging
from dataclasses import dataclass
from enum import Enum

# A2AI内部モジュールのインポート
from ..base_model import BaseModel
from ...utils.statistical_utils import robust_standard_errors, calculate_confidence_intervals
from ...utils.data_utils import validate_panel_data, create_balanced_panel


class TreatmentType(Enum):
    """処置タイプの定義"""
    MERGER_ACQUISITION = "merger_acquisition"       # M&A
    SPINOFF = "spinoff"                            # 分社化
    MARKET_ENTRY = "market_entry"                  # 新市場参入
    TECHNOLOGY_ADOPTION = "technology_adoption"     # 新技術導入
    REGULATORY_CHANGE = "regulatory_change"         # 規制変更
    MANAGEMENT_CHANGE = "management_change"         # 経営陣変更
    BUSINESS_MODEL_CHANGE = "business_model_change" # ビジネスモデル変更


@dataclass
class DIDResults:
    """差分差分法の分析結果を格納するデータクラス"""
    treatment_effect: float
    standard_error: float
    t_statistic: float
    p_value: float
    confidence_interval: Tuple[float, float]
    r_squared: float
    n_observations: int
    n_treated: int
    n_control: int
    pre_treatment_periods: int
    post_treatment_periods: int
    parallel_trends_test: Dict
    robustness_tests: Dict
    placebo_tests: Dict


@dataclass
class TreatmentEvent:
    """処置イベントの定義"""
    company_id: str
    treatment_date: pd.Timestamp
    treatment_type: TreatmentType
    description: str
    intensity: Optional[float] = None  # 処置の強度（0-1スケール）


class DifferenceInDifferences(BaseModel):
    """
    差分差分法による因果推論分析クラス
    
    企業レベルの処置効果を推定し、A2AIの9つの評価項目に対する
    各種イベントの因果効果を定量化する。
    """
    
    def __init__(self, 
                    outcome_vars: List[str] = None,
                    control_vars: List[str] = None,
                    cluster_var: str = 'company_id',
                    time_var: str = 'year',
                    robust_se: bool = True,
                    parallel_trends_periods: int = 3,
                    min_pre_periods: int = 2,
                    min_post_periods: int = 1):
        """
        Args:
            outcome_vars: 結果変数リスト（A2AIの9つの評価項目）
            control_vars: 統制変数リスト（23の要因項目から選択）
            cluster_var: クラスター変数（通常は企業ID）
            time_var: 時間変数
            robust_se: 頑健標準誤差の使用フラグ
            parallel_trends_periods: 平行トレンド検定の期間
            min_pre_periods: 最小事前期間
            min_post_periods: 最小事後期間
        """
        super().__init__()
        
        # デフォルトの結果変数（A2AIの9つの評価項目）
        self.outcome_vars = outcome_vars or [
            'sales_growth_rate',           # 売上高成長率
            'operating_profit_margin',     # 売上高営業利益率
            'net_profit_margin',           # 売上高当期純利益率
            'roe',                         # ROE
            'value_added_ratio',           # 売上高付加価値率
            'sales_level',                 # 売上高
            'survival_probability',        # 企業存続確率
            'emergence_success_rate',      # 新規事業成功率
            'succession_success_rate'      # 事業継承成功度
        ]
        
        # デフォルトの統制変数（23の要因項目から主要なものを選択）
        self.control_vars = control_vars or [
            'rd_intensity',                # 研究開発費率
            'capex_intensity',             # 設備投資率
            'employee_growth_rate',        # 従業員数増加率
            'total_asset_turnover',        # 総資産回転率
            'debt_to_equity_ratio',        # 負債資本比率
            'overseas_sales_ratio',        # 海外売上高比率
            'company_age',                 # 企業年齢
            'market_entry_timing'          # 市場参入時期
        ]
        
        self.cluster_var = cluster_var
        self.time_var = time_var
        self.robust_se = robust_se
        self.parallel_trends_periods = parallel_trends_periods
        self.min_pre_periods = min_pre_periods
        self.min_post_periods = min_post_periods
        
        self.logger = logging.getLogger(__name__)
        
    def prepare_did_data(self, 
                        df: pd.DataFrame, 
                        treatment_events: List[TreatmentEvent],
                        outcome_var: str) -> pd.DataFrame:
        """
        差分差分法のためのデータを準備
        
        Args:
            df: パネルデータ
            treatment_events: 処置イベントリスト  
            outcome_var: 分析対象の結果変数
            
        Returns:
            準備済みのデータフレーム
        """
        # データの検証
        if not validate_panel_data(df, self.cluster_var, self.time_var):
            raise ValueError("Invalid panel data format")
        
        # 処置グループと処置時期の特定
        df = df.copy()
        df['treated'] = 0
        df['post'] = 0
        df['treatment_intensity'] = 0.0
        
        for event in treatment_events:
            # 企業IDでフィルタ
            company_mask = df[self.cluster_var] == event.company_id
            
            # 処置時期以降をpost=1に設定
            if self.time_var == 'year':
                treatment_year = event.treatment_date.year
                time_mask = df[self.time_var] >= treatment_year
            else:
                time_mask = df[self.time_var] >= event.treatment_date
            
            # 処置グループの設定
            df.loc[company_mask, 'treated'] = 1
            df.loc[company_mask & time_mask, 'post'] = 1
            
            # 処置強度の設定
            if event.intensity is not None:
                df.loc[company_mask & time_mask, 'treatment_intensity'] = event.intensity
        
        # 処置効果の交互作用項を作成
        df['treated_post'] = df['treated'] * df['post']
        
        # 欠損値の処理
        required_vars = [outcome_var] + self.control_vars + ['treated', 'post', 'treated_post']
        df = df.dropna(subset=required_vars)
        
        # バランス型パネルの作成（必要に応じて）
        if len(df.groupby(self.cluster_var)[self.time_var].count().unique()) > 1:
            self.logger.warning("Unbalanced panel detected. Consider using balanced panel.")
        
        return df
    
    def estimate_standard_did(self, 
                            df: pd.DataFrame, 
                            outcome_var: str) -> DIDResults:
        """
        標準的な差分差分法の推定
        
        Y_it = α + β*Treated_i + γ*Post_t + δ*Treated_i*Post_t + X_it'θ + ε_it
        
        Args:
            df: 準備済みデータ
            outcome_var: 結果変数
            
        Returns:
            分析結果
        """
        # 回帰モデルの準備
        X_vars = ['treated', 'post', 'treated_post'] + self.control_vars
        
        # ダミー変数の作成（固定効果）
        company_dummies = pd.get_dummies(df[self.cluster_var], prefix='company')
        time_dummies = pd.get_dummies(df[self.time_var], prefix='time')
        
        # 説明変数の結合
        X = pd.concat([
            df[X_vars],
            company_dummies.iloc[:, 1:],  # 最初のダミーは除外（多重共線性回避）
            time_dummies.iloc[:, 1:]
        ], axis=1)
        
        y = df[outcome_var]
        
        # 線形回帰の実行
        model = LinearRegression()
        model.fit(X, y)
        
        # 予測値と残差
        y_pred = model.predict(X)
        residuals = y - y_pred
        
        # 処置効果の推定（treated_postの係数）
        treatment_effect = model.coef_[X_vars.index('treated_post')]
        
        # 標準誤差の計算
        if self.robust_se:
            se_matrix = robust_standard_errors(X.values, residuals.values, 
                                                cluster_var=df[self.cluster_var].values)
            standard_error = np.sqrt(se_matrix[X_vars.index('treated_post'), 
                                                X_vars.index('treated_post')])
        else:
            mse = np.mean(residuals**2)
            se_matrix = mse * np.linalg.inv(X.T @ X)
            standard_error = np.sqrt(se_matrix[X_vars.index('treated_post'), 
                                                X_vars.index('treated_post')])
        
        # 統計的検定
        t_statistic = treatment_effect / standard_error
        p_value = 2 * (1 - stats.t.cdf(np.abs(t_statistic), df=len(X) - X.shape[1]))
        
        # 信頼区間
        confidence_interval = calculate_confidence_intervals(
            treatment_effect, standard_error, alpha=0.05
        )
        
        # モデル適合度
        r_squared = r2_score(y, y_pred)
        
        # サンプルサイズ
        n_treated = (df['treated'] == 1).sum()
        n_control = (df['treated'] == 0).sum()
        n_observations = len(df)
        
        # 事前・事後期間数
        pre_periods = len(df[df['post'] == 0][self.time_var].unique())
        post_periods = len(df[df['post'] == 1][self.time_var].unique())
        
        # 平行トレンド検定
        parallel_trends_test = self._test_parallel_trends(df, outcome_var)
        
        # 頑健性検定
        robustness_tests = self._robustness_tests(df, outcome_var)
        
        # プラセボ検定
        placebo_tests = self._placebo_tests(df, outcome_var)
        
        return DIDResults(
            treatment_effect=treatment_effect,
            standard_error=standard_error,
            t_statistic=t_statistic,
            p_value=p_value,
            confidence_interval=confidence_interval,
            r_squared=r_squared,
            n_observations=n_observations,
            n_treated=n_treated,
            n_control=n_control,
            pre_treatment_periods=pre_periods,
            post_treatment_periods=post_periods,
            parallel_trends_test=parallel_trends_test,
            robustness_tests=robustness_tests,
            placebo_tests=placebo_tests
        )
    
    def estimate_time_varying_did(self, 
                                df: pd.DataFrame, 
                                outcome_var: str,
                                treatment_intensity_var: str = 'treatment_intensity') -> Dict:
        """
        時系列変動差分差分法の推定
        
        処置の効果が時間とともに変化する場合の分析
        
        Args:
            df: データフレーム
            outcome_var: 結果変数
            treatment_intensity_var: 処置強度変数
            
        Returns:
            時系列での処置効果
        """
        results = {}
        unique_periods = sorted(df[df['post'] == 1][self.time_var].unique())
        
        for period in unique_periods:
            # 各期間での処置効果を推定
            period_df = df[df[self.time_var] <= period].copy()
            period_df['post_period'] = (period_df[self.time_var] == period).astype(int)
            period_df['treated_post_period'] = (period_df['treated'] * 
                                                period_df['post_period'])
            
            # 回帰分析
            X_vars = ['treated', 'post_period', 'treated_post_period'] + self.control_vars
            X = period_df[X_vars]
            y = period_df[outcome_var]
            
            model = LinearRegression()
            model.fit(X, y)
            
            # 結果の保存
            treatment_effect = model.coef_[X_vars.index('treated_post_period')]
            results[period] = {
                'treatment_effect': treatment_effect,
                'period': period,
                'n_observations': len(period_df)
            }
        
        return results
    
    def estimate_multi_period_did(self, 
                                df: pd.DataFrame, 
                                treatment_events: List[TreatmentEvent],
                                outcome_var: str) -> Dict:
        """
        複数期間差分差分法の推定
        
        複数の処置時期を持つ企業群の分析
        
        Args:
            df: データフレーム
            treatment_events: 処置イベントリスト
            outcome_var: 結果変数
            
        Returns:
            処置時期別の効果推定結果
        """
        results = {}
        
        # 処置時期でグループ化
        treatment_groups = {}
        for event in treatment_events:
            if self.time_var == 'year':
                treatment_period = event.treatment_date.year
            else:
                treatment_period = event.treatment_date
            
            if treatment_period not in treatment_groups:
                treatment_groups[treatment_period] = []
            treatment_groups[treatment_period].append(event.company_id)
        
        # 各処置時期グループでDID推定
        for treatment_period, company_list in treatment_groups.items():
            # 当該グループのデータを準備
            group_events = [e for e in treatment_events 
                            if e.company_id in company_list]
            
            group_df = self.prepare_did_data(df, group_events, outcome_var)
            
            # DID推定
            group_results = self.estimate_standard_did(group_df, outcome_var)
            results[treatment_period] = group_results
        
        return results
    
    def _test_parallel_trends(self, df: pd.DataFrame, outcome_var: str) -> Dict:
        """
        平行トレンド仮定の検定
        
        処置前期間において処置群と対照群のトレンドが平行かを検定
        """
        # 処置前データのみを使用
        pre_treatment_df = df[df['post'] == 0].copy()
        
        if len(pre_treatment_df) == 0:
            return {'test_result': 'insufficient_data'}
        
        # 時間トレンドとの交互作用項を作成
        pre_treatment_df['time_trend'] = (pre_treatment_df[self.time_var] - 
                                        pre_treatment_df[self.time_var].min())
        pre_treatment_df['treated_time_trend'] = (pre_treatment_df['treated'] * 
                                                pre_treatment_df['time_trend'])
        
        # 回帰分析
        X_vars = ['treated', 'time_trend', 'treated_time_trend'] + self.control_vars
        X = pre_treatment_df[X_vars]
        y = pre_treatment_df[outcome_var]
        
        model = LinearRegression()
        model.fit(X, y)
        
        # treated_time_trendの係数が有意でないことを検定
        treatment_trend_coef = model.coef_[X_vars.index('treated_time_trend')]
        
        # 簡易的なt検定（より厳密には頑健標準誤差を使用）
        residuals = y - model.predict(X)
        mse = np.mean(residuals**2)
        se_matrix = mse * np.linalg.inv(X.T @ X)
        standard_error = np.sqrt(se_matrix[X_vars.index('treated_time_trend'), 
                                            X_vars.index('treated_time_trend')])
        
        t_statistic = treatment_trend_coef / standard_error
        p_value = 2 * (1 - stats.t.cdf(np.abs(t_statistic), df=len(X) - X.shape[1]))
        
        return {
            'coefficient': treatment_trend_coef,
            'standard_error': standard_error,
            't_statistic': t_statistic,
            'p_value': p_value,
            'test_result': 'passed' if p_value > 0.05 else 'failed'
        }
    
    def _robustness_tests(self, df: pd.DataFrame, outcome_var: str) -> Dict:
        """
        頑健性検定
        
        異なる仕様での推定結果の頑健性を確認
        """
        results = {}
        
        # 1. 統制変数なしの推定
        X_vars_minimal = ['treated', 'post', 'treated_post']
        X_minimal = df[X_vars_minimal]
        y = df[outcome_var]
        
        model_minimal = LinearRegression()
        model_minimal.fit(X_minimal, y)
        results['no_controls'] = model_minimal.coef_[X_vars_minimal.index('treated_post')]
        
        # 2. 異なる統制変数セットでの推定
        alternative_controls = ['rd_intensity', 'total_asset_turnover', 'company_age']
        X_vars_alt = ['treated', 'post', 'treated_post'] + alternative_controls
        
        if all(var in df.columns for var in alternative_controls):
            X_alt = df[X_vars_alt]
            model_alt = LinearRegression()
            model_alt.fit(X_alt, y)
            results['alternative_controls'] = model_alt.coef_[X_vars_alt.index('treated_post')]
        
        # 3. サブサンプルでの推定（高シェア市場のみなど）
        if 'market_category' in df.columns:
            for category in df['market_category'].unique():
                subset_df = df[df['market_category'] == category]
                if len(subset_df) > 50:  # 最小サンプルサイズチェック
                    subset_result = self.estimate_standard_did(subset_df, outcome_var)
                    results[f'subsample_{category}'] = subset_result.treatment_effect
        
        return results
    
    def _placebo_tests(self, df: pd.DataFrame, outcome_var: str) -> Dict:
        """
        プラセボ検定
        
        偽の処置時期での効果推定により、真の因果関係を検証
        """
        results = {}
        
        # 処置前期間でのプラセボ検定
        pre_treatment_df = df[df['post'] == 0].copy()
        
        if len(pre_treatment_df[self.time_var].unique()) >= 3:
            # 中間時点を偽の処置時期として設定
            time_periods = sorted(pre_treatment_df[self.time_var].unique())
            placebo_time = time_periods[len(time_periods)//2]
            
            pre_treatment_df['placebo_post'] = (pre_treatment_df[self.time_var] >= placebo_time).astype(int)
            pre_treatment_df['placebo_treated_post'] = (pre_treatment_df['treated'] * 
                                                        pre_treatment_df['placebo_post'])
            
            # プラセボ回帰
            X_vars_placebo = ['treated', 'placebo_post', 'placebo_treated_post'] + self.control_vars
            X_placebo = pre_treatment_df[X_vars_placebo]
            y_placebo = pre_treatment_df[outcome_var]
            
            model_placebo = LinearRegression()
            model_placebo.fit(X_placebo, y_placebo)
            
            placebo_effect = model_placebo.coef_[X_vars_placebo.index('placebo_treated_post')]
            results['pre_treatment_placebo'] = placebo_effect
        
        return results
    
    def visualize_treatment_effects(self, 
                                    results: Union[DIDResults, Dict],
                                    save_path: Optional[str] = None) -> plt.Figure:
        """
        処置効果の可視化
        
        Args:
            results: 分析結果
            save_path: 保存パス
            
        Returns:
            matplotlib Figure オブジェクト
        """
        if isinstance(results, DIDResults):
            # 単一結果の可視化
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            
            # 処置効果とエラーバー
            ax.errorbar([0], [results.treatment_effect], 
                        yerr=[results.confidence_interval[1] - results.treatment_effect],
                        fmt='o', capsize=10, capthick=2, markersize=8)
            
            ax.axhline(y=0, color='red', linestyle='--', alpha=0.7)
            ax.set_ylabel('Treatment Effect')
            ax.set_title('Difference-in-Differences Treatment Effect')
            ax.grid(True, alpha=0.3)
            
        else:
            # 複数期間結果の可視化
            fig, ax = plt.subplots(1, 1, figsize=(12, 6))
            
            periods = list(results.keys())
            effects = [results[p].treatment_effect if hasattr(results[p], 'treatment_effect') 
                        else results[p]['treatment_effect'] for p in periods]
            
            ax.plot(periods, effects, 'o-', linewidth=2, markersize=6)
            ax.axhline(y=0, color='red', linestyle='--', alpha=0.7)
            ax.set_xlabel('Time Period')
            ax.set_ylabel('Treatment Effect')
            ax.set_title('Time-Varying Treatment Effects')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def generate_did_report(self, 
                            results: DIDResults,
                            treatment_events: List[TreatmentEvent],
                            outcome_var: str) -> str:
        """
        差分差分法分析レポートの生成
        
        Args:
            results: 分析結果
            treatment_events: 処置イベント
            outcome_var: 結果変数
            
        Returns:
            分析レポート（markdown形式）
        """
        report = f"""
# Difference-in-Differences Analysis Report

## Analysis Overview
- **Outcome Variable**: {outcome_var}
- **Number of Treatment Events**: {len(treatment_events)}
- **Total Observations**: {results.n_observations:,}
- **Treated Units**: {results.n_treated:,}
- **Control Units**: {results.n_control:,}

## Treatment Effect Results

### Main Results
- **Treatment Effect**: {results.treatment_effect:.4f}
- **Standard Error**: {results.standard_error:.4f}
- **t-statistic**: {results.t_statistic:.4f}
- **p-value**: {results.p_value:.4f}
- **95% Confidence Interval**: [{results.confidence_interval[0]:.4f}, {results.confidence_interval[1]:.4f}]

### Model Diagnostics
- **R-squared**: {results.r_squared:.4f}
- **Pre-treatment Periods**: {results.pre_treatment_periods}
- **Post-treatment Periods**: {results.post_treatment_periods}

## Validity Tests

### Parallel Trends Test
- **Test Result**: {results.parallel_trends_test.get('test_result', 'N/A')}
- **p-value**: {results.parallel_trends_test.get('p_value', 'N/A')}

### Robustness Tests
"""
        
        for test_name, effect in results.robustness_tests.items():
            report += f"- **{test_name}**: {effect:.4f}\n"
        
        report += """
### Placebo Tests
"""
        
        for test_name, effect in results.placebo_tests.items():
            report += f"- **{test_name}**: {effect:.4f}\n"
        
        report += f"""

## Interpretation
The estimated treatment effect is {results.treatment_effect:.4f}, which is 
{'statistically significant' if results.p_value < 0.05 else 'not statistically significant'} 
at the 5% level (p-value = {results.p_value:.4f}).

This suggests that the treatment {'has a significant' if results.p_value < 0.05 else 'does not have a significant'} 
{'positive' if results.treatment_effect > 0 else 'negative'} effect on {outcome_var}.

## Treatment Events Analyzed
"""
        
        for i, event in enumerate(treatment_events, 1):
            report += f"""
### Event {i}
- **Company**: {event.company_id}
- **Date**: {event.treatment_date.strftime('%Y-%m-%d')}
- **Type**: {event.treatment_type.value}
- **Description**: {event.description}
"""
        
        return report
    
    def fit(self, 
            df: pd.DataFrame,
            treatment_events: List[TreatmentEvent],
            outcome_vars: Optional[List[str]] = None) -> Dict[str, DIDResults]:
        """
        複数の結果変数に対してDID分析を実行
        
        Args:
            df: パネルデータ
            treatment_events: 処置イベントリスト
            outcome_vars: 分析対象の結果変数リスト
            
        Returns:
            結果変数別の分析結果
        """
        if outcome_vars is None:
            outcome_vars = self.outcome_vars
        
        results = {}
        
        for outcome_var in outcome_vars:
            try:
                # データ準備
                did_df = self.prepare_did_data(df, treatment_events, outcome_var)
                
                # DID推定
                result = self.estimate_standard_did(did_df, outcome_var)
                results[outcome_var] = result
                
                self.logger.info(f"DID analysis completed for {outcome_var}")
                
            except Exception as e:
                self.logger.error(f"Error in DID analysis for {outcome_var}: {str(e)}")
                continue
        
        return results


# ユーティリティ関数
def create_treatment_events_from_data(df: pd.DataFrame,
                                    company_col: str = 'company_id',
                                    date_col: str = 'treatment_date',
                                    type_col: str = 'treatment_type',
                                    description_col: str = 'description') -> List[TreatmentEvent]:
    """
    データフレームから処置イベントリストを作成
    
    Args:
        df: 処置イベント情報を含むデータフレーム
        company_col: 企業ID列名
        date_col: 処置日列名
        type_col: 処置タイプ列名
        description_col: 説明列名
        
    Returns:
        処置イベントのリスト
    """
    events = []
    
    for _, row in df.iterrows():
        # 処置タイプの変換
        treatment_type_str = row[type_col]
        try:
            treatment_type = TreatmentType(treatment_type_str)
        except ValueError:
            # デフォルトとして経営変更を使用
            treatment_type = TreatmentType.MANAGEMENT_CHANGE
        
        # 処置日の変換
        treatment_date = pd.to_datetime(row[date_col])
        
        # 処置強度（存在する場合）
        intensity = row.get('intensity', None)
        
        event = TreatmentEvent(
            company_id=row[company_col],
            treatment_date=treatment_date,
            treatment_type=treatment_type,
            description=row[description_col],
            intensity=intensity
        )
        events.append(event)
    
    return events


def detect_treatment_events_automatically(df: pd.DataFrame,
                                        company_col: str = 'company_id',
                                        time_col: str = 'year',
                                        financial_vars: List[str] = None,
                                        threshold_multiplier: float = 2.0) -> List[TreatmentEvent]:
    """
    財務データから自動的に処置イベントを検出
    
    異常な財務指標の変化から企業再編やその他のイベントを推定
    
    Args:
        df: 財務データ
        company_col: 企業ID列名
        time_col: 時間列名
        financial_vars: 分析対象の財務変数
        threshold_multiplier: 異常検出の閾値倍数
        
    Returns:
        検出された処置イベントのリスト
    """
    if financial_vars is None:
        financial_vars = ['sales_growth_rate', 'total_assets', 'employee_count', 
                            'rd_intensity', 'overseas_sales_ratio']
    
    events = []
    
    for company in df[company_col].unique():
        company_df = df[df[company_col] == company].sort_values(time_col)
        
        for var in financial_vars:
            if var in company_df.columns:
                # 変化率の計算
                company_df[f'{var}_change'] = company_df[var].pct_change()
                
                # 異常値の検出（平均±閾値×標準偏差）
                mean_change = company_df[f'{var}_change'].mean()
                std_change = company_df[f'{var}_change'].std()
                threshold = threshold_multiplier * std_change
                
                # 異常な変化を検出
                anomalies = company_df[
                    abs(company_df[f'{var}_change'] - mean_change) > threshold
                ]
                
                for _, anomaly_row in anomalies.iterrows():
                    # 処置タイプの推定
                    if var == 'total_assets' and anomaly_row[f'{var}_change'] > 0.5:
                        treatment_type = TreatmentType.MERGER_ACQUISITION
                    elif var == 'employee_count' and anomaly_row[f'{var}_change'] < -0.3:
                        treatment_type = TreatmentType.SPINOFF
                    elif var == 'rd_intensity' and anomaly_row[f'{var}_change'] > 0.5:
                        treatment_type = TreatmentType.TECHNOLOGY_ADOPTION
                    else:
                        treatment_type = TreatmentType.BUSINESS_MODEL_CHANGE
                    
                    # 処置日の設定
                    if time_col == 'year':
                        treatment_date = pd.to_datetime(f"{anomaly_row[time_col]}-01-01")
                    else:
                        treatment_date = pd.to_datetime(anomaly_row[time_col])
                    
                    event = TreatmentEvent(
                        company_id=company,
                        treatment_date=treatment_date,
                        treatment_type=treatment_type,
                        description=f"Anomalous change in {var}: {anomaly_row[f'{var}_change']:.2%}",
                        intensity=min(1.0, abs(anomaly_row[f'{var}_change']))
                    )
                    events.append(event)
    
    # 重複イベントの除去（同一企業・同一年・同一タイプ）
    unique_events = []
    seen_combinations = set()
    
    for event in events:
        key = (event.company_id, event.treatment_date.year, event.treatment_type)
        if key not in seen_combinations:
            unique_events.append(event)
            seen_combinations.add(key)
    
    return unique_events


class SyntheticControlDID(DifferenceInDifferences):
    """
    合成統制法と差分差分法を組み合わせた分析クラス
    
    対照群が少ない場合に、複数の企業を組み合わせて
    合成的な対照群を作成し、より精密なDID分析を実行
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.synthetic_weights = {}
    
    def create_synthetic_control(self,
                                df: pd.DataFrame,
                                treated_company: str,
                                outcome_var: str,
                                pre_treatment_periods: int = 5) -> Dict[str, float]:
        """
        合成統制法による対照群の重み作成
        
        Args:
            df: データフレーム
            treated_company: 処置を受けた企業ID
            outcome_var: 結果変数
            pre_treatment_periods: 事前期間の数
            
        Returns:
            各対照企業の重み
        """
        # 処置企業と対照企業の分離
        treated_df = df[df[self.cluster_var] == treated_company]
        control_df = df[df[self.cluster_var] != treated_company]
        
        # 事前期間のデータのみ使用
        pre_treatment_data = df[df['post'] == 0]
        treated_pre = pre_treatment_data[pre_treatment_data[self.cluster_var] == treated_company]
        control_pre = pre_treatment_data[pre_treatment_data[self.cluster_var] != treated_company]
        
        # 処置企業の特徴ベクトル
        treated_features = treated_pre[self.control_vars + [outcome_var]].mean().values
        
        # 対照企業の特徴マトリックス
        control_companies = control_pre[self.cluster_var].unique()
        control_features_matrix = np.zeros((len(control_companies), len(treated_features)))
        
        for i, company in enumerate(control_companies):
            company_data = control_pre[control_pre[self.cluster_var] == company]
            control_features_matrix[i] = company_data[self.control_vars + [outcome_var]].mean().values
        
        # 最適化による重み計算
        def objective(weights):
            synthetic_features = weights @ control_features_matrix
            return np.sum((synthetic_features - treated_features) ** 2)
        
        # 制約条件：重みの合計=1、重み≥0
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = [(0, 1) for _ in range(len(control_companies))]
        initial_weights = np.ones(len(control_companies)) / len(control_companies)
        
        result = minimize(objective, initial_weights, 
                            method='SLSQP', bounds=bounds, constraints=constraints)
        
        # 重みの辞書作成
        weights_dict = {company: weight for company, weight in 
                        zip(control_companies, result.x)}
        
        # 0.01未満の重みを持つ企業を除外
        weights_dict = {k: v for k, v in weights_dict.items() if v >= 0.01}
        
        self.synthetic_weights[treated_company] = weights_dict
        return weights_dict
    
    def estimate_synthetic_did(self,
                                df: pd.DataFrame,
                                treatment_events: List[TreatmentEvent],
                                outcome_var: str) -> Dict[str, DIDResults]:
        """
        合成統制法を用いた差分差分法の推定
        
        Args:
            df: データフレーム
            treatment_events: 処置イベント
            outcome_var: 結果変数
            
        Returns:
            各処置企業に対する分析結果
        """
        results = {}
        
        for event in treatment_events:
            treated_company = event.company_id
            
            # 合成統制の重み作成
            weights = self.create_synthetic_control(df, treated_company, outcome_var)
            
            # 重み付き対照群データの作成
            control_companies = list(weights.keys())
            weighted_control_df = df[df[self.cluster_var].isin(control_companies)].copy()
            
            # 各対照企業のデータに重みを適用
            synthetic_control_data = []
            for time_period in weighted_control_df[self.time_var].unique():
                period_data = weighted_control_df[weighted_control_df[self.time_var] == time_period]
                
                # 重み付き平均の計算
                weighted_values = {}
                for var in [outcome_var] + self.control_vars:
                    if var in period_data.columns:
                        weighted_avg = sum(weights[company] * 
                                            period_data[period_data[self.cluster_var] == company][var].iloc[0]
                                            for company in control_companies 
                                            if company in period_data[self.cluster_var].values and 
                                            len(period_data[period_data[self.cluster_var] == company]) > 0)
                        weighted_values[var] = weighted_avg
                
                # 合成統制データポイントの作成
                synthetic_point = {
                    self.cluster_var: f'synthetic_control_{treated_company}',
                    self.time_var: time_period,
                    'treated': 0,  # 合成統制は対照群
                    **weighted_values
                }
                synthetic_control_data.append(synthetic_point)
            
            # 処置企業データと合成統制データの結合
            treated_df = df[df[self.cluster_var] == treated_company].copy()
            synthetic_df = pd.DataFrame(synthetic_control_data)
            
            combined_df = pd.concat([treated_df, synthetic_df], ignore_index=True)
            
            # DID分析の実行
            did_df = self.prepare_did_data(combined_df, [event], outcome_var)
            result = self.estimate_standard_did(did_df, outcome_var)
            
            results[treated_company] = result
        
        return results


class EventStudyDID(DifferenceInDifferences):
    """
    イベントスタディと差分差分法を組み合わせた分析クラス
    
    処置前後の複数期間にわたる動的な効果を分析
    """
    
    def estimate_dynamic_effects(self,
                                df: pd.DataFrame,
                                treatment_events: List[TreatmentEvent],
                                outcome_var: str,
                                leads: int = 3,
                                lags: int = 5) -> Dict[str, Dict]:
        """
        動的処置効果の推定（イベントスタディ）
        
        Args:
            df: データフレーム
            treatment_events: 処置イベント
            outcome_var: 結果変数
            leads: 事前期間数（anticipation effects）
            lags: 事後期間数
            
        Returns:
            期間別の処置効果
        """
        # 処置時期を基準とした相対時間の作成
        df = df.copy()
        df['relative_time'] = np.nan
        
        for event in treatment_events:
            company_mask = df[self.cluster_var] == event.company_id
            if self.time_var == 'year':
                treatment_year = event.treatment_date.year
                df.loc[company_mask, 'relative_time'] = df.loc[company_mask, self.time_var] - treatment_year
            else:
                # 日付型の場合の処理（簡略化）
                treatment_year = event.treatment_date.year
                df.loc[company_mask, 'relative_time'] = df.loc[company_mask, self.time_var] - treatment_year
        
        # 相対時間ダミーの作成
        for t in range(-leads, lags + 1):
            if t != -1:  # -1期間をベースライン（省略）として使用
                df[f'time_{t}'] = ((df['relative_time'] == t) & (df['treated'] == 1)).astype(int)
        
        # 回帰分析
        time_dummies = [f'time_{t}' for t in range(-leads, lags + 1) if t != -1]
        X_vars = ['treated'] + time_dummies + self.control_vars
        
        # 企業・時間固定効果の追加
        company_dummies = pd.get_dummies(df[self.cluster_var], prefix='company')
        time_fe_dummies = pd.get_dummies(df[self.time_var], prefix='time')
        
        X = pd.concat([
            df[X_vars],
            company_dummies.iloc[:, 1:],
            time_fe_dummies.iloc[:, 1:]
        ], axis=1)
        
        y = df[outcome_var]
        
        # 欠損値の除去
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[mask]
        y = y[mask]
        
        model = LinearRegression()
        model.fit(X, y)
        
        # 各期間の効果を抽出
        results = {}
        for t in range(-leads, lags + 1):
            if t != -1:
                dummy_name = f'time_{t}'
                if dummy_name in X_vars:
                    coef_idx = X_vars.index(dummy_name)
                    coefficient = model.coef_[coef_idx]
                    
                    # 簡易標準誤差（より厳密には頑健標準誤差を使用すべき）
                    residuals = y - model.predict(X)
                    mse = np.mean(residuals**2)
                    se_matrix = mse * np.linalg.inv(X.T @ X)
                    standard_error = np.sqrt(se_matrix[coef_idx, coef_idx])
                    
                    results[t] = {
                        'coefficient': coefficient,
                        'standard_error': standard_error,
                        't_statistic': coefficient / standard_error,
                        'period': t
                    }
        
        return results
    
    def plot_event_study(self,
                        dynamic_results: Dict[str, Dict],
                        outcome_var: str,
                        save_path: Optional[str] = None) -> plt.Figure:
        """
        イベントスタディ結果の可視化
        
        Args:
            dynamic_results: 動的効果の推定結果
            outcome_var: 結果変数名
            save_path: 保存パス
            
        Returns:
            matplotlib Figure
        """
        periods = sorted(dynamic_results.keys())
        coefficients = [dynamic_results[p]['coefficient'] for p in periods]
        standard_errors = [dynamic_results[p]['standard_error'] for p in periods]
        
        # 信頼区間の計算
        lower_ci = [coef - 1.96 * se for coef, se in zip(coefficients, standard_errors)]
        upper_ci = [coef + 1.96 * se for coef, se in zip(coefficients, standard_errors)]
        
        # プロット
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 係数とエラーバー
        ax.errorbar(periods, coefficients, yerr=[1.96 * se for se in standard_errors],
                    fmt='o-', capsize=5, capthick=2, linewidth=2, markersize=6)
        
        # 信頼区間
        ax.fill_between(periods, lower_ci, upper_ci, alpha=0.2)
        
        # 処置時点の縦線
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.7, 
                    label='Treatment Period')
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # 装飾
        ax.set_xlabel('Periods Relative to Treatment')
        ax.set_ylabel(f'Treatment Effect on {outcome_var}')
        ax.set_title(f'Dynamic Treatment Effects: {outcome_var}')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # x軸の目盛り設定
        ax.set_xticks(periods[::2])  # 2期間おきに目盛り
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


# A2AI特化のDID分析クラス
class A2AI_DifferenceInDifferences(DifferenceInDifferences):
    """
    A2AIプロジェクト特化の差分差分法分析クラス
    
    150社×40年のデータ構造と9つの評価項目、23の要因項目に最適化
    """
    
    def __init__(self):
        # A2AI固有の評価項目（9項目）
        outcome_vars = [
            'sales_growth_rate',           # 売上高成長率
            'operating_profit_margin',     # 売上高営業利益率  
            'net_profit_margin',           # 売上高当期純利益率
            'roe',                         # ROE
            'value_added_ratio',           # 売上高付加価値率
            'sales_level',                 # 売上高
            'survival_probability',        # 企業存続確率
            'emergence_success_rate',      # 新規事業成功率
            'succession_success_rate'      # 事業継承成功度
        ]
        
        # A2AI固有の要因項目（23項目）から主要な統制変数を選択
        control_vars = [
            'rd_intensity',                # 研究開発費率
            'capex_intensity',             # 設備投資率
            'tangible_asset_ratio',        # 有形固定資産比率
            'intangible_asset_ratio',      # 無形固定資産比率
            'employee_growth_rate',        # 従業員数増加率
            'average_salary_ratio',        # 平均年間給与比率
            'total_asset_turnover',        # 総資産回転率
            'inventory_turnover',          # 棚卸資産回転率
            'overseas_sales_ratio',        # 海外売上高比率
            'debt_to_equity_ratio',        # 負債資本比率
            'company_age',                 # 企業年齢
            'market_entry_timing',         # 市場参入時期
            'parent_dependency_ratio'      # 親会社依存度
        ]
        
        super().__init__(
            outcome_vars=outcome_vars,
            control_vars=control_vars,
            cluster_var='company_id',
            time_var='year',
            robust_se=True,
            parallel_trends_periods=3,
            min_pre_periods=2,
            min_post_periods=1
        )
    
    def analyze_market_category_effects(self,
                                        df: pd.DataFrame,
                                        treatment_events: List[TreatmentEvent]) -> Dict:
        """
        市場カテゴリー別（高シェア/低下/失失）の処置効果分析
        
        Args:
            df: 財務データ（market_category列を含む）
            treatment_events: 処置イベント
            
        Returns:
            市場カテゴリー別の分析結果
        """
        results = {}
        
        for category in ['high_share', 'declining', 'lost_share']:
            category_df = df[df['market_category'] == category]
            category_events = [e for e in treatment_events 
                                if e.company_id in category_df['company_id'].unique()]
            
            if len(category_events) > 0:
                category_results = self.fit(category_df, category_events)
                results[category] = category_results
        
        return results
    
    def compare_treatment_effectiveness(self,
                                        high_share_results: Dict[str, DIDResults],
                                        declining_results: Dict[str, DIDResults],
                                        lost_share_results: Dict[str, DIDResults]) -> Dict:
        """
        市場カテゴリー間での処置効果の比較分析
        
        Args:
            high_share_results: 高シェア市場の結果
            declining_results: シェア低下市場の結果
            lost_share_results: シェア失失市場の結果
            
        Returns:
            比較分析結果
        """
        comparison_results = {}
        
        # 共通の評価項目での比較
        common_outcomes = set(high_share_results.keys()) & \
                            set(declining_results.keys()) & \
                            set(lost_share_results.keys())
        
        for outcome in common_outcomes:
            high_effect = high_share_results[outcome].treatment_effect
            declining_effect = declining_results[outcome].treatment_effect
            lost_effect = lost_share_results[outcome].treatment_effect
            
            # 効果の大きさ比較
            comparison_results[outcome] = {
                'high_share_effect': high_effect,
                'declining_effect': declining_effect,
                'lost_share_effect': lost_effect,
                'high_vs_declining_diff': high_effect - declining_effect,
                'high_vs_lost_diff': high_effect - lost_effect,
                'declining_vs_lost_diff': declining_effect - lost_effect,
                'effect_ranking': sorted([
                    ('high_share', high_effect),
                    ('declining', declining_effect), 
                    ('lost_share', lost_effect)
                ], key=lambda x: x[1], reverse=True)
            }
        
        return comparison_results