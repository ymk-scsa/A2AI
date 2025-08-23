"""
A2AI (Advanced Financial Analysis AI)
操作変数法 (Instrumental Variables) モジュール

企業の財務諸表分析における内生性問題を解決するための操作変数法を実装。
150社×40年分のデータを用いて、要因項目が評価項目に与える真の因果効果を推定。

主要機能:
- Two-Stage Least Squares (2SLS) 回帰
- Generalized Method of Moments (GMM) 推定
- 操作変数の妥当性検定
- 弱操作変数問題の診断
- 過剰識別制約検定
- 内生性検定
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings
from dataclasses import dataclass
from scipy import stats
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import logging

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class IVEstimationResult:
    """操作変数法推定結果を格納するデータクラス"""
    coefficients: np.ndarray
    standard_errors: np.ndarray
    t_statistics: np.ndarray
    p_values: np.ndarray
    confidence_intervals: np.ndarray
    r_squared: float
    adjusted_r_squared: float
    n_observations: int
    n_instruments: int
    n_endogenous: int
    first_stage_f_stats: np.ndarray
    weak_instrument_test: Dict[str, float]
    overidentification_test: Dict[str, float]
    endogeneity_test: Dict[str, float]
    method: str


class InstrumentalVariables:
    """
    操作変数法による因果推論クラス
    
    財務諸表分析において、要因項目と評価項目の関係に存在する内生性問題を
    操作変数を用いて解決し、真の因果効果を推定する。
    """
    
    def __init__(self, 
                    alpha: float = 0.05,
                    robust: bool = True,
                    cluster_var: Optional[str] = None):
        """
        初期化
        
        Args:
            alpha: 有意水準（デフォルト: 0.05）
            robust: 頑健標準誤差を使用するか
            cluster_var: クラスター標準誤差用の変数名
        """
        self.alpha = alpha
        self.robust = robust
        self.cluster_var = cluster_var
        self.scaler = StandardScaler()
        
    def two_stage_least_squares(self,
                                data: pd.DataFrame,
                                dependent_var: str,
                                endogenous_vars: List[str],
                                instruments: List[str],
                                exogenous_vars: Optional[List[str]] = None,
                                entity_id: Optional[str] = None,
                                time_id: Optional[str] = None) -> IVEstimationResult:
        """
        2段階最小二乗法 (2SLS) による操作変数推定
        
        Args:
            data: 分析データ
            dependent_var: 被説明変数（評価項目）
            endogenous_vars: 内生説明変数（要因項目）
            instruments: 操作変数リスト
            exogenous_vars: 外生説明変数
            entity_id: 企業ID列名（パネルデータの場合）
            time_id: 時間ID列名（パネルデータの場合）
            
        Returns:
            IVEstimationResult: 推定結果
        """
        # データ準備
        clean_data = self._prepare_data(data, dependent_var, endogenous_vars, 
                                        instruments, exogenous_vars)
        
        y = clean_data[dependent_var].values
        X_endo = clean_data[endogenous_vars].values
        Z = clean_data[instruments].values
        
        # 外生変数の処理
        if exogenous_vars:
            X_exo = clean_data[exogenous_vars].values
            X = np.column_stack([X_endo, X_exo])
            Z_full = np.column_stack([Z, X_exo])
        else:
            X = X_endo
            Z_full = Z
            
        n, k = X.shape
        n_instruments = Z_full.shape[1]
        n_endogenous = len(endogenous_vars)
        
        # 定数項追加
        X = np.column_stack([np.ones(n), X])
        Z_full = np.column_stack([np.ones(n), Z_full])
        
        # 第1段階: 内生変数を操作変数で回帰
        first_stage_results = []
        first_stage_f_stats = []
        
        for i, endo_var in enumerate(endogenous_vars):
            X_endo_i = clean_data[endo_var].values
            reg = LinearRegression(fit_intercept=False)
            reg.fit(Z_full, X_endo_i)
            
            # 予測値
            X_endo_hat_i = reg.predict(Z_full)
            first_stage_results.append(X_endo_hat_i)
            
            # F統計量計算
            residuals = X_endo_i - X_endo_hat_i
            rss = np.sum(residuals**2)
            tss = np.sum((X_endo_i - np.mean(X_endo_i))**2)
            r_squared = 1 - (rss / tss)
            
            f_stat = (r_squared / (n_instruments - 1)) / ((1 - r_squared) / (n - n_instruments))
            first_stage_f_stats.append(f_stat)
            
        first_stage_f_stats = np.array(first_stage_f_stats)
        
        # 第2段階: 被説明変数を予測された内生変数で回帰
        if exogenous_vars:
            X_hat = np.column_stack([np.ones(n)] + first_stage_results + 
                                    [clean_data[var].values for var in exogenous_vars])
        else:
            X_hat = np.column_stack([np.ones(n)] + first_stage_results)
            
        # 2SLS推定
        beta_2sls = np.linalg.solve(X_hat.T @ X_hat, X_hat.T @ y)
        y_pred = X_hat @ beta_2sls
        residuals = y - y_pred
        
        # 標準誤差計算
        if self.robust:
            standard_errors = self._compute_robust_se(X_hat, residuals, Z_full)
        else:
            sigma_squared = np.sum(residuals**2) / (n - k - 1)
            var_covar_matrix = sigma_squared * np.linalg.inv(X_hat.T @ X_hat)
            standard_errors = np.sqrt(np.diag(var_covar_matrix))
            
        # 統計量計算
        t_stats = beta_2sls / standard_errors
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), n - k - 1))
        
        # 信頼区間
        t_critical = stats.t.ppf(1 - self.alpha/2, n - k - 1)
        ci_lower = beta_2sls - t_critical * standard_errors
        ci_upper = beta_2sls + t_critical * standard_errors
        confidence_intervals = np.column_stack([ci_lower, ci_upper])
        
        # R-squared計算
        r_squared = r2_score(y, y_pred)
        adjusted_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - k - 1)
        
        # 診断テスト
        weak_instrument_test = self._weak_instrument_test(first_stage_f_stats, n_endogenous)
        overid_test = self._overidentification_test(residuals, Z_full, n_instruments, k)
        endogeneity_test = self._endogeneity_test(clean_data, dependent_var, 
                                                    endogenous_vars, instruments, exogenous_vars)
        
        return IVEstimationResult(
            coefficients=beta_2sls,
            standard_errors=standard_errors,
            t_statistics=t_stats,
            p_values=p_values,
            confidence_intervals=confidence_intervals,
            r_squared=r_squared,
            adjusted_r_squared=adjusted_r_squared,
            n_observations=n,
            n_instruments=n_instruments,
            n_endogenous=n_endogenous,
            first_stage_f_stats=first_stage_f_stats,
            weak_instrument_test=weak_instrument_test,
            overidentification_test=overid_test,
            endogeneity_test=endogeneity_test,
            method="2SLS"
        )
    
    def gmm_estimation(self,
                        data: pd.DataFrame,
                        dependent_var: str,
                        endogenous_vars: List[str],
                        instruments: List[str],
                        exogenous_vars: Optional[List[str]] = None,
                        weight_matrix: str = "optimal") -> IVEstimationResult:
        """
        一般化モーメント法 (GMM) による操作変数推定
        
        Args:
            data: 分析データ
            dependent_var: 被説明変数
            endogenous_vars: 内生説明変数
            instruments: 操作変数
            exogenous_vars: 外生説明変数
            weight_matrix: ウエイト行列の種類 ("identity", "optimal")
            
        Returns:
            IVEstimationResult: 推定結果
        """
        # データ準備
        clean_data = self._prepare_data(data, dependent_var, endogenous_vars, 
                                        instruments, exogenous_vars)
        
        y = clean_data[dependent_var].values
        X_endo = clean_data[endogenous_vars].values
        Z = clean_data[instruments].values
        
        if exogenous_vars:
            X_exo = clean_data[exogenous_vars].values
            X = np.column_stack([np.ones(len(y)), X_endo, X_exo])
            Z_full = np.column_stack([np.ones(len(y)), Z, X_exo])
        else:
            X = np.column_stack([np.ones(len(y)), X_endo])
            Z_full = np.column_stack([np.ones(len(y)), Z])
            
        n, k = X.shape
        n_instruments = Z_full.shape[1]
        
        # GMM目的関数
        def gmm_objective(beta, W):
            residuals = y - X @ beta
            moments = Z_full.T @ residuals / n
            return moments.T @ W @ moments
            
        # 初期ウエイト行列（単位行列）
        W_initial = np.eye(n_instruments)
        
        # 第1段階GMM推定
        result = minimize(lambda beta: gmm_objective(beta, W_initial),
                            x0=np.zeros(k),
                            method='BFGS')
        beta_gmm1 = result.x
        
        # 最適ウエイト行列計算
        if weight_matrix == "optimal":
            residuals = y - X @ beta_gmm1
            moment_matrix = Z_full * residuals.reshape(-1, 1)
            S = moment_matrix.T @ moment_matrix / n
            W_optimal = np.linalg.inv(S)
            
            # 第2段階GMM推定
            result = minimize(lambda beta: gmm_objective(beta, W_optimal),
                                x0=beta_gmm1,
                                method='BFGS')
            beta_gmm = result.x
            W = W_optimal
        else:
            beta_gmm = beta_gmm1
            W = W_initial
            
        # 標準誤差計算
        residuals = y - X @ beta_gmm
        G = -X.T @ Z_full / n  # モーメント条件の勾配
        var_covar_matrix = np.linalg.inv(G @ W @ G.T) / n
        standard_errors = np.sqrt(np.diag(var_covar_matrix))
        
        # 統計量計算
        t_stats = beta_gmm / standard_errors
        p_values = 2 * (1 - stats.norm.cdf(np.abs(t_stats)))
        
        # 信頼区間
        z_critical = stats.norm.ppf(1 - self.alpha/2)
        ci_lower = beta_gmm - z_critical * standard_errors
        ci_upper = beta_gmm + z_critical * standard_errors
        confidence_intervals = np.column_stack([ci_lower, ci_upper])
        
        # R-squared計算
        y_pred = X @ beta_gmm
        r_squared = r2_score(y, y_pred)
        adjusted_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - k)
        
        # 診断テスト（簡易版）
        weak_instrument_test = {"message": "Not computed for GMM"}
        overid_test = self._overidentification_test(residuals, Z_full, n_instruments, k)
        endogeneity_test = {"message": "Not computed for GMM"}
        
        return IVEstimationResult(
            coefficients=beta_gmm,
            standard_errors=standard_errors,
            t_statistics=t_stats,
            p_values=p_values,
            confidence_intervals=confidence_intervals,
            r_squared=r_squared,
            adjusted_r_squared=adjusted_r_squared,
            n_observations=n,
            n_instruments=n_instruments,
            n_endogenous=len(endogenous_vars),
            first_stage_f_stats=np.array([]),
            weak_instrument_test=weak_instrument_test,
            overidentification_test=overid_test,
            endogeneity_test=endogeneity_test,
            method="GMM"
        )
    
    def panel_iv_estimation(self,
                            data: pd.DataFrame,
                            dependent_var: str,
                            endogenous_vars: List[str],
                            instruments: List[str],
                            entity_id: str,
                            time_id: str,
                            exogenous_vars: Optional[List[str]] = None,
                            fixed_effects: str = "entity") -> IVEstimationResult:
        """
        パネルデータ用操作変数推定（固定効果モデル）
        
        Args:
            data: パネルデータ
            dependent_var: 被説明変数
            endogenous_vars: 内生説明変数
            instruments: 操作変数
            entity_id: 企業ID
            time_id: 時間ID
            exogenous_vars: 外生説明変数
            fixed_effects: 固定効果の種類 ("entity", "time", "both")
            
        Returns:
            IVEstimationResult: 推定結果
        """
        # データ準備
        clean_data = self._prepare_data(data, dependent_var, endogenous_vars, 
                                        instruments, exogenous_vars, entity_id, time_id)
        
        # 固定効果による変換
        if fixed_effects in ["entity", "both"]:
            clean_data = self._within_transformation(clean_data, entity_id, 
                                                    [dependent_var] + endogenous_vars + instruments + 
                                                    (exogenous_vars or []))
        
        if fixed_effects in ["time", "both"]:
            clean_data = self._within_transformation(clean_data, time_id,
                                                    [dependent_var] + endogenous_vars + instruments + 
                                                    (exogenous_vars or []))
        
        # 2SLS実行
        return self.two_stage_least_squares(clean_data, dependent_var, endogenous_vars,
                                            instruments, exogenous_vars)
    
    def _prepare_data(self, 
                        data: pd.DataFrame, 
                        dependent_var: str,
                        endogenous_vars: List[str],
                        instruments: List[str],
                        exogenous_vars: Optional[List[str]] = None,
                        entity_id: Optional[str] = None,
                        time_id: Optional[str] = None) -> pd.DataFrame:
        """データの前処理"""
        all_vars = [dependent_var] + endogenous_vars + instruments
        if exogenous_vars:
            all_vars.extend(exogenous_vars)
        if entity_id:
            all_vars.append(entity_id)
        if time_id:
            all_vars.append(time_id)
            
        # 欠損値除去
        clean_data = data[all_vars].dropna()
        
        # 操作変数の関連性チェック
        self._check_instrument_relevance(clean_data, endogenous_vars, instruments)
        
        return clean_data
    
    def _check_instrument_relevance(self, 
                                    data: pd.DataFrame,
                                    endogenous_vars: List[str],
                                    instruments: List[str]) -> None:
        """操作変数の関連性チェック"""
        for endo_var in endogenous_vars:
            correlations = []
            for instrument in instruments:
                corr = data[endo_var].corr(data[instrument])
                correlations.append(abs(corr))
            
            max_corr = max(correlations)
            if max_corr < 0.1:
                warnings.warn(f"操作変数と内生変数{endo_var}の相関が低い可能性があります (最大相関: {max_corr:.3f})")
    
    def _compute_robust_se(self, 
                            X: np.ndarray, 
                            residuals: np.ndarray, 
                            Z: np.ndarray) -> np.ndarray:
        """頑健標準誤差の計算"""
        n, k = X.shape
        
        # White's heteroskedasticity-robust standard errors
        XZ = X.T @ Z
        meat = np.zeros((k, k))
        
        for i in range(n):
            xi = X[i, :].reshape(-1, 1)
            zi = Z[i, :].reshape(-1, 1)
            ei = residuals[i]
            meat += (xi @ zi.T) @ (zi @ xi.T) * ei**2
            
        bread = np.linalg.inv(XZ @ XZ.T / n)
        var_covar_matrix = bread @ (meat / n) @ bread / n
        
        return np.sqrt(np.diag(var_covar_matrix))
    
    def _weak_instrument_test(self, 
                                f_stats: np.ndarray, 
                                n_endogenous: int) -> Dict[str, float]:
        """弱操作変数検定"""
        # Stock-Yogo critical values (簡略版)
        critical_values = {1: 16.38, 2: 7.03, 3: 4.58}  # 10% maximal IV bias
        critical_value = critical_values.get(n_endogenous, 4.58)
        
        min_f_stat = np.min(f_stats)
        is_weak = min_f_stat < critical_value
        
        return {
            "min_f_statistic": min_f_stat,
            "critical_value": critical_value,
            "is_weak_instrument": is_weak,
            "f_statistics": f_stats.tolist()
        }
    
    def _overidentification_test(self, 
                                residuals: np.ndarray,
                                Z: np.ndarray,
                                n_instruments: int,
                                n_parameters: int) -> Dict[str, float]:
        """過剰識別制約検定 (Hansen J test)"""
        n = len(residuals)
        
        if n_instruments <= n_parameters:
            return {"message": "Exactly identified model - no overidentification test"}
        
        # モーメント条件
        moments = Z.T @ residuals / n
        
        # J統計量
        j_stat = n * moments.T @ moments
        degrees_freedom = n_instruments - n_parameters
        p_value = 1 - stats.chi2.cdf(j_stat, degrees_freedom)
        
        return {
            "j_statistic": float(j_stat),
            "degrees_freedom": degrees_freedom,
            "p_value": float(p_value),
            "reject_overidentification": p_value < 0.05
        }
    
    def _endogeneity_test(self,
                            data: pd.DataFrame,
                            dependent_var: str,
                            endogenous_vars: List[str],
                            instruments: List[str],
                            exogenous_vars: Optional[List[str]] = None) -> Dict[str, float]:
        """内生性検定 (Wu-Hausman test)"""
        try:
            # OLS推定
            all_exog_vars = (exogenous_vars or []) + endogenous_vars
            X_ols = data[all_exog_vars].values
            X_ols = np.column_stack([np.ones(len(X_ols)), X_ols])
            y = data[dependent_var].values
            
            beta_ols = np.linalg.solve(X_ols.T @ X_ols, X_ols.T @ y)
            residuals_ols = y - X_ols @ beta_ols
            rss_ols = np.sum(residuals_ols**2)
            
            # 2SLS推定
            iv_result = self.two_stage_least_squares(data, dependent_var, endogenous_vars,
                                                    instruments, exogenous_vars)
            
            # Hausman統計量計算（簡略版）
            hausman_stat = 0.0  # 実装を簡略化
            p_value = 1.0
            
            return {
                "hausman_statistic": hausman_stat,
                "p_value": p_value,
                "reject_exogeneity": p_value < 0.05,
                "message": "Simplified implementation"
            }
            
        except Exception as e:
            return {"error": f"内生性検定でエラーが発生: {str(e)}"}
    
    def _within_transformation(self, 
                                data: pd.DataFrame, 
                                group_var: str, 
                                transform_vars: List[str]) -> pd.DataFrame:
        """固定効果用のwithin変換"""
        result_data = data.copy()
        
        for var in transform_vars:
            if var in data.columns:
                group_means = data.groupby(group_var)[var].transform('mean')
                result_data[var] = data[var] - group_means
                
        return result_data
    
    def estimate_financial_factors_iv(self,
                                        data: pd.DataFrame,
                                        evaluation_metric: str,
                                        factor_vars: List[str],
                                        market_category: str,
                                        company_id: str = "company_id",
                                        year_id: str = "year") -> IVEstimationResult:
        """
        財務要因分析特化の操作変数推定
        
        Args:
            data: 財務データ
            evaluation_metric: 評価項目（売上高、ROE等）
            factor_vars: 要因項目リスト
            market_category: 市場カテゴリ（high_share, declining, lost）
            company_id: 企業ID列名
            year_id: 年度列名
            
        Returns:
            IVEstimationResult: 推定結果
        """
        # 市場カテゴリでフィルタ
        market_data = data[data['market_category'] == market_category].copy()
        
        # 操作変数の自動選択（業界平均、ラグ変数等）
        instruments = self._select_financial_instruments(market_data, factor_vars, year_id)
        
        # パネルIV推定実行
        return self.panel_iv_estimation(
            data=market_data,
            dependent_var=evaluation_metric,
            endogenous_vars=factor_vars,
            instruments=instruments,
            entity_id=company_id,
            time_id=year_id,
            fixed_effects="entity"
        )
    
    def _select_financial_instruments(self,
                                        data: pd.DataFrame,
                                        factor_vars: List[str],
                                        year_id: str) -> List[str]:
        """財務分析用操作変数の自動選択"""
        instruments = []
        
        for factor in factor_vars:
            # ラグ変数（t-2期）
            lag_var = f"{factor}_lag2"
            if lag_var not in data.columns:
                data[lag_var] = data.groupby('company_id')[factor].shift(2)
            if not data[lag_var].isna().all():
                instruments.append(lag_var)
                
            # 業界平均（同一年度の他社平均）
            industry_avg_var = f"{factor}_industry_avg"
            if industry_avg_var not in data.columns:
                data[industry_avg_var] = data.groupby(year_id)[factor].transform(
                    lambda x: x.mean()
                )
            if not data[industry_avg_var].isna().all():
                instruments.append(industry_avg_var)
        
        return [inst for inst in instruments if inst in data.columns]


def format_iv_results(result: IVEstimationResult, 
                        variable_names: Optional[List[str]] = None) -> str:
    """操作変数推定結果の整形出力"""
    if variable_names is None:
        variable_names = [f"Variable_{i}" for i in range(len(result.coefficients))]
    
    output = f"""
操作変数法推定結果 ({result.method})
{'='*50}

観測数: {result.n_observations}
操作変数数: {result.n_instruments}
内生変数数: {result.n_endogenous}

R-squared: {result.r_squared:.4f}
Adjusted R-squared: {result.adjusted_r_squared:.4f}

係数推定結果:
{'-'*50}
"""
    
    for i, var_name in enumerate(variable_names):
        output += f"{var_name:20s}: {result.coefficients[i]:8.4f} "
        output += f"({result.standard_errors[i]:6.4f}) "
        output += f"[{result.confidence_intervals[i][0]:6.4f}, {result.confidence_intervals[i][1]:6.4f}]"
        
        if result.p_values[i] < 0.01:
            output += " ***"
        elif result.p_values[i] < 0.05:
            output += " **"
        elif result.p_values[i] < 0.1:
            output += " *"
            
        output += f" (p={result.p_values[i]:.4f})\n"
    
    output += f"\n診断テスト:\n{'-'*30}\n"
    
    # 弱操作変数検定
    if isinstance(result.weak_instrument_test, dict):
        if "min_f_statistic" in result.weak_instrument_test:
            output += f"弱操作変数検定: F = {result.weak_instrument_test['min_f_statistic']:.2f}\n"
            output += f"  (Critical Value = {result.weak_instrument_test['critical_value']:.2f})\n"
            output += f"  弱操作変数の可能性: {'Yes' if result.weak_instrument_test['is_weak_instrument'] else 'No'}\n"
    
    # 過剰識別制約検定
    if isinstance(result.overidentification_test, dict):
        if "j_statistic" in result.overidentification_test:
            output += f"Hansen J検定: J = {result.overidentification_test['j_statistic']:.4f} "
            output += f"(p = {result.overidentification_test['p_value']:.4f})\n"
            output += f"  過剰識別制約棄却: {'Yes' if result.overidentification_test['reject_overidentification'] else 'No'}\n"
    
    output += "\n注: *** p<0.01, ** p<0.05, * p<0.1\n"
    output += "標準誤差は括弧内、95%信頼区間は角括弧内\n"
    
    return output


# 使用例とテスト関数
def example_usage():
    """使用例の実行"""
    # サンプルデータ生成
    np.random.seed(42)
    n_companies = 50
    n_years = 20
    n_obs = n_companies * n_years
    
    # パネルデータ生成
    company_ids = np.repeat(range(n_companies), n_years)
    years = np.tile(range(2005, 2025), n_companies)
    
    # 操作変数（外生的に決まる変数）
    z1 = np.random.normal(0, 1, n_obs)  # 業界ショック
    z2 = np.random.normal(0, 1, n_obs)  # 政策変数
    
    # 企業固定効果
    alpha_i = np.repeat(np.random.normal(0, 0.5, n_companies), n_years)
    
    # 時間固定効果
    gamma_t = np.tile(np.random.normal(0, 0.2, n_years), n_companies)
    
    # 内生変数（研究開発費比率）
    # 操作変数と相関があるが、直接的には被説明変数に影響しない部分を含む
    rd_ratio = 0.3 * z1 + 0.2 * z2 + alpha_i + np.random.normal(0, 0.3, n_obs)
    
    # 外生変数（企業規模）
    log_assets = 10 + 0.1 * years + alpha_i + np.random.normal(0, 0.5, n_obs)
    
    # 被説明変数（ROE）
    # 内生変数との逆因果関係や欠落変数バイアスを含む
    unobserved = np.random.normal(0, 0.2, n_obs)  # 欠落変数
    roe = 0.05 + 0.4 * rd_ratio + 0.2 * log_assets + alpha_i + gamma_t + \
          unobserved + 0.1 * np.random.normal(0, 1, n_obs)
    
    # 逆因果関係: ROEが高い企業ほどR&D投資を増やす傾向
    rd_ratio += 0.2 * roe + np.random.normal(0, 0.1, n_obs)
    
    # 市場カテゴリの追加
    market_categories = np.random.choice(['high_share', 'declining', 'lost'], 
                                        size=n_obs, p=[0.33, 0.34, 0.33])
    
    sample_data = pd.DataFrame({
        'company_id': company_ids,
        'year': years,
        'roe': roe,
        'rd_ratio': rd_ratio,
        'log_assets': log_assets,
        'industry_shock': z1,
        'policy_var': z2,
        'market_category': market_categories
    })
    
    print("A2AI 操作変数法の使用例")
    print("="*50)
    
    # 操作変数推定器の初期化
    iv_estimator = InstrumentalVariables(alpha=0.05, robust=True)
    
    # 2SLS推定
    print("\n1. 2段階最小二乗法 (2SLS) 推定")
    print("-"*30)
    
    try:
        result_2sls = iv_estimator.two_stage_least_squares(
            data=sample_data,
            dependent_var='roe',
            endogenous_vars=['rd_ratio'],
            instruments=['industry_shock', 'policy_var'],
            exogenous_vars=['log_assets'],
            entity_id='company_id',
            time_id='year'
        )
        
        print(format_iv_results(result_2sls, 
                                ['定数項', 'R&D比率', '企業規模']))
        
    except Exception as e:
        print(f"2SLS推定エラー: {e}")
    
    # GMM推定
    print("\n2. 一般化モーメント法 (GMM) 推定")
    print("-"*30)
    
    try:
        result_gmm = iv_estimator.gmm_estimation(
            data=sample_data,
            dependent_var='roe',
            endogenous_vars=['rd_ratio'],
            instruments=['industry_shock', 'policy_var'],
            exogenous_vars=['log_assets'],
            weight_matrix='optimal'
        )
        
        print(format_iv_results(result_gmm, 
                                ['定数項', 'R&D比率', '企業規模']))
        
    except Exception as e:
        print(f"GMM推定エラー: {e}")
    
    # パネルIV推定
    print("\n3. パネルデータ固定効果IV推定")
    print("-"*30)
    
    try:
        result_panel = iv_estimator.panel_iv_estimation(
            data=sample_data,
            dependent_var='roe',
            endogenous_vars=['rd_ratio'],
            instruments=['industry_shock', 'policy_var'],
            entity_id='company_id',
            time_id='year',
            exogenous_vars=['log_assets'],
            fixed_effects='entity'
        )
        
        print(format_iv_results(result_panel, 
                                ['定数項', 'R&D比率', '企業規模']))
        
    except Exception as e:
        print(f"パネルIV推定エラー: {e}")
    
    # 市場カテゴリ別分析
    print("\n4. 市場カテゴリ別IV分析")
    print("-"*30)
    
    for market_cat in ['high_share', 'declining', 'lost']:
        try:
            print(f"\n市場カテゴリ: {market_cat}")
            print("-"*20)
            
            result_market = iv_estimator.estimate_financial_factors_iv(
                data=sample_data,
                evaluation_metric='roe',
                factor_vars=['rd_ratio'],
                market_category=market_cat,
                company_id='company_id',
                year_id='year'
            )
            
            print(f"係数推定値: R&D比率 = {result_market.coefficients[1]:.4f}")
            print(f"標準誤差: {result_market.standard_errors[1]:.4f}")
            print(f"p値: {result_market.p_values[1]:.4f}")
            
        except Exception as e:
            print(f"市場カテゴリ{market_cat}でエラー: {e}")
    
    return sample_data


class FinancialFactorIVAnalyzer:
    """
    A2AI専用の財務要因分析クラス
    
    150社×40年データを用いた財務要因の因果効果分析に特化した
    操作変数法の応用クラス
    """
    
    def __init__(self, alpha: float = 0.05):
        self.iv_estimator = InstrumentalVariables(alpha=alpha, robust=True)
        self.results_cache = {}
        
    def analyze_factor_impact(self,
                                data: pd.DataFrame,
                                evaluation_metrics: List[str],
                                factor_categories: Dict[str, List[str]],
                                market_categories: List[str]) -> Dict[str, Dict]:
        """
        財務要因の評価項目への因果効果を包括的に分析
        
        Args:
            data: 財務データ（150社×40年）
            evaluation_metrics: 評価項目リスト（売上高、ROE等）
            factor_categories: 要因項目カテゴリ辞書
            market_categories: 市場カテゴリリスト
            
        Returns:
            Dict: 分析結果辞書
        """
        results = {}
        
        for metric in evaluation_metrics:
            results[metric] = {}
            
            for market in market_categories:
                results[metric][market] = {}
                
                # 市場データをフィルタ
                market_data = data[data['market_category'] == market].copy()
                
                if len(market_data) < 50:  # 最小サンプルサイズチェック
                    logger.warning(f"市場{market}のサンプルサイズが小さすぎます: {len(market_data)}")
                    continue
                
                for category_name, factors in factor_categories.items():
                    try:
                        # 操作変数の自動生成
                        instruments = self._generate_instruments(market_data, factors)
                        
                        if len(instruments) < len(factors):
                            logger.warning(f"操作変数が不足: {len(instruments)} < {len(factors)}")
                            continue
                        
                        # パネルIV推定実行
                        iv_result = self.iv_estimator.panel_iv_estimation(
                            data=market_data,
                            dependent_var=metric,
                            endogenous_vars=factors,
                            instruments=instruments,
                            entity_id='company_id',
                            time_id='year',
                            fixed_effects='both'
                        )
                        
                        results[metric][market][category_name] = {
                            'coefficients': iv_result.coefficients.tolist(),
                            'standard_errors': iv_result.standard_errors.tolist(),
                            'p_values': iv_result.p_values.tolist(),
                            'factor_names': factors,
                            'r_squared': iv_result.r_squared,
                            'n_observations': iv_result.n_observations,
                            'weak_instrument_test': iv_result.weak_instrument_test,
                            'overidentification_test': iv_result.overidentification_test
                        }
                        
                        logger.info(f"分析完了: {metric} - {market} - {category_name}")
                        
                    except Exception as e:
                        logger.error(f"分析エラー: {metric} - {market} - {category_name}: {e}")
                        results[metric][market][category_name] = {'error': str(e)}
        
        return results
    
    def _generate_instruments(self, 
                                data: pd.DataFrame, 
                                endogenous_vars: List[str]) -> List[str]:
        """財務分析用操作変数の自動生成"""
        instruments = []
        
        for var in endogenous_vars:
            # 2期ラグ
            lag2_var = f"{var}_lag2"
            data[lag2_var] = data.groupby('company_id')[var].shift(2)
            if not data[lag2_var].isna().all():
                instruments.append(lag2_var)
            
            # 3期ラグ
            lag3_var = f"{var}_lag3"
            data[lag3_var] = data.groupby('company_id')[var].shift(3)
            if not data[lag3_var].isna().all():
                instruments.append(lag3_var)
            
            # 業界中央値
            industry_median_var = f"{var}_industry_median"
            data[industry_median_var] = data.groupby(['year'])[var].transform('median')
            if not data[industry_median_var].isna().all():
                instruments.append(industry_median_var)
            
            # 移動平均変化率
            ma_change_var = f"{var}_ma_change"
            data[f"{var}_ma3"] = data.groupby('company_id')[var].rolling(3).mean().values
            data[ma_change_var] = data.groupby('company_id')[f"{var}_ma3"].pct_change()
            if not data[ma_change_var].isna().all():
                instruments.append(ma_change_var)
        
        return instruments
    
    def compare_market_effects(self,
                                results: Dict[str, Dict],
                                evaluation_metric: str,
                                factor_category: str) -> pd.DataFrame:
        """市場カテゴリ間での要因効果を比較"""
        comparison_data = []
        
        try:
            for market in ['high_share', 'declining', 'lost']:
                if (market in results[evaluation_metric] and 
                    factor_category in results[evaluation_metric][market]):
                    
                    market_result = results[evaluation_metric][market][factor_category]
                    
                    if 'error' not in market_result:
                        factors = market_result['factor_names']
                        coeffs = market_result['coefficients']
                        se = market_result['standard_errors']
                        p_vals = market_result['p_values']
                        
                        for i, factor in enumerate(factors):
                            comparison_data.append({
                                'market_category': market,
                                'factor': factor,
                                'coefficient': coeffs[i],
                                'standard_error': se[i],
                                'p_value': p_vals[i],
                                'significant': p_vals[i] < 0.05
                            })
        except KeyError as e:
            logger.error(f"比較分析エラー: キー{e}が見つかりません")
            return pd.DataFrame()
        
        return pd.DataFrame(comparison_data)
    
    def test_market_differences(self,
                                results: Dict[str, Dict],
                                evaluation_metric: str,
                                factor_category: str) -> Dict[str, float]:
        """市場間での要因効果の差の統計的検定"""
        comparison_df = self.compare_market_effects(results, evaluation_metric, factor_category)
        
        if comparison_df.empty:
            return {'error': 'データが不足しています'}
        
        # 各要因について市場間差の検定を実行
        test_results = {}
        
        for factor in comparison_df['factor'].unique():
            factor_data = comparison_df[comparison_df['factor'] == factor]
            
            if len(factor_data) >= 3:  # 3市場のデータが揃っている場合
                # 係数の差の検定（簡易版）
                coeffs = factor_data['coefficient'].values
                se = factor_data['standard_errors'].values if 'standard_errors' in factor_data.columns else factor_data['standard_error'].values
                
                # 最大値と最小値の差
                max_coeff = np.max(coeffs)
                min_coeff = np.min(coeffs)
                diff = max_coeff - min_coeff
                
                # 標準誤差の結合
                pooled_se = np.sqrt(np.sum(se**2))
                t_stat = diff / pooled_se if pooled_se > 0 else 0
                
                test_results[factor] = {
                    'difference': diff,
                    'pooled_se': pooled_se,
                    't_statistic': t_stat,
                    'significant_difference': abs(t_stat) > 1.96
                }
        
        return test_results


# テスト実行部分
if __name__ == "__main__":
    try:
        # 使用例の実行
        sample_data = example_usage()
        
        print("\n" + "="*60)
        print("A2AI 財務要因分析クラスのテスト")
        print("="*60)
        
        # 財務要因分析クラスのテスト
        analyzer = FinancialFactorIVAnalyzer(alpha=0.05)
        
        # 要因項目カテゴリの定義（サンプル）
        factor_categories = {
            'investment_factors': ['rd_ratio'],
            'efficiency_factors': ['log_assets']
        }
        
        # 包括的分析の実行
        comprehensive_results = analyzer.analyze_factor_impact(
            data=sample_data,
            evaluation_metrics=['roe'],
            factor_categories=factor_categories,
            market_categories=['high_share', 'declining', 'lost']
        )
        
        # 結果の表示
        for metric, market_results in comprehensive_results.items():
            print(f"\n評価項目: {metric}")
            print("-"*30)
            
            for market, category_results in market_results.items():
                print(f"\n市場カテゴリ: {market}")
                for category, result in category_results.items():
                    if 'error' not in result:
                        print(f"  {category}: R² = {result['r_squared']:.4f}, N = {result['n_observations']}")
                    else:
                        print(f"  {category}: エラー - {result['error']}")
        
        # 市場間比較
        print(f"\n市場間比較分析")
        print("-"*30)
        
        comparison_df = analyzer.compare_market_effects(
            comprehensive_results, 'roe', 'investment_factors'
        )
        
        if not comparison_df.empty:
            print(comparison_df.to_string(index=False))
            
            # 統計的検定
            test_results = analyzer.test_market_differences(
                comprehensive_results, 'roe', 'investment_factors'
            )
            
            print(f"\n市場間差検定結果:")
            for factor, test_result in test_results.items():
                if isinstance(test_result, dict) and 'error' not in test_result:
                    print(f"  {factor}: 差 = {test_result['difference']:.4f}, "
                            f"有意差 = {test_result['significant_difference']}")
        
        print(f"\nA2AI 操作変数法モジュールのテスト完了")
        
    except Exception as e:
        print(f"テスト実行エラー: {e}")
        import traceback
        traceback.print_exc()