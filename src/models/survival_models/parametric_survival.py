"""
A2AI: Advanced Financial Analysis AI
Parametric Survival Analysis Models

企業の生存時間分析のためのパラメトリック生存分析モデル群を実装。
ワイブル、指数、対数正規、ガンマ分布等を用いて企業の生存確率を推定。

主要機能:
- 複数の確率分布による生存モデル
- 企業の生存確率予測
- ハザード関数・生存関数の計算
- 要因項目（共変量）を考慮したモデリング
- モデル比較・選択機能
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import scipy.stats as stats
from scipy.optimize import minimize
from scipy.special import gamma as gamma_func
import matplotlib.pyplot as plt
import seaborn as sns
from abc import ABC, abstractmethod
import logging

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ParametricSurvivalBase(ABC):
    """
    パラメトリック生存分析の基底クラス
    """
    
    def __init__(self, distribution_name: str):
        self.distribution_name = distribution_name
        self.params = None
        self.fitted = False
        self.scaler = StandardScaler()
        self.feature_names = None
        self.n_features = None
        
    @abstractmethod
    def _log_likelihood(self, params: np.ndarray, X: np.ndarray, 
                        T: np.ndarray, E: np.ndarray) -> float:
        """対数尤度関数の計算"""
        pass
    
    @abstractmethod
    def survival_function(self, t: Union[float, np.ndarray], 
                            X: Optional[np.ndarray] = None) -> np.ndarray:
        """生存関数 S(t) = P(T > t)"""
        pass
    
    @abstractmethod
    def hazard_function(self, t: Union[float, np.ndarray], 
                        X: Optional[np.ndarray] = None) -> np.ndarray:
        """ハザード関数 h(t)"""
        pass
    
    def fit(self, X: np.ndarray, T: np.ndarray, E: np.ndarray) -> 'ParametricSurvivalBase':
        """
        モデルのフィッティング
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            共変量（要因項目）
        T : array-like, shape (n_samples,)
            観測時間（企業存続年数）
        E : array-like, shape (n_samples,)
            イベント指示子（1: 消滅, 0: 打ち切り）
        """
        X = np.asarray(X)
        T = np.asarray(T)
        E = np.asarray(E)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            
        self.n_features = X.shape[1]
        self.feature_names = [f'feature_{i}' for i in range(self.n_features)]
        
        # 特徴量の標準化
        X_scaled = self.scaler.fit_transform(X)
        
        # パラメータ初期化
        initial_params = self._initialize_params()
        
        # 最尤推定
        result = minimize(
            fun=lambda p: -self._log_likelihood(p, X_scaled, T, E),
            x0=initial_params,
            method='L-BFGS-B',
            bounds=self._get_param_bounds()
        )
        
        if not result.success:
            logger.warning(f"Optimization did not converge: {result.message}")
        
        self.params = result.x
        self.fitted = True
        self.log_likelihood_ = -result.fun
        self.aic_ = 2 * len(self.params) - 2 * self.log_likelihood_
        self.bic_ = len(self.params) * np.log(len(T)) - 2 * self.log_likelihood_
        
        logger.info(f"{self.distribution_name} model fitted. AIC: {self.aic_:.2f}, BIC: {self.bic_:.2f}")
        
        return self
    
    def predict_survival_probability(self, X: np.ndarray, t: Union[float, np.ndarray]) -> np.ndarray:
        """生存確率の予測"""
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        X_scaled = self.scaler.transform(X)
        return self.survival_function(t, X_scaled)
    
    def predict_hazard(self, X: np.ndarray, t: Union[float, np.ndarray]) -> np.ndarray:
        """ハザード関数の予測"""
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        X_scaled = self.scaler.transform(X)
        return self.hazard_function(t, X_scaled)
    
    def predict_median_survival_time(self, X: np.ndarray) -> np.ndarray:
        """中央生存時間の予測"""
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        # 生存確率が0.5となる時間を数値的に求める
        median_times = []
        for i in range(X.shape[0]):
            x_i = X[i:i+1]
            try:
                # 0.5に最も近い生存確率を持つ時間を探索
                t_range = np.linspace(1, 50, 1000)  # 1年から50年の範囲
                survival_probs = self.predict_survival_probability(x_i, t_range)
                median_idx = np.argmin(np.abs(survival_probs - 0.5))
                median_times.append(t_range[median_idx])
            except:
                median_times.append(np.nan)
        
        return np.array(median_times)
    
    @abstractmethod
    def _initialize_params(self) -> np.ndarray:
        """パラメータの初期値設定"""
        pass
    
    @abstractmethod
    def _get_param_bounds(self) -> List[Tuple[float, float]]:
        """パラメータの境界条件"""
        pass


class WeibullSurvival(ParametricSurvivalBase):
    """
    ワイブル分布による生存分析モデル
    
    企業の生存時間がワイブル分布に従うと仮定。
    故障率が時間とともに変化する場合に適用。
    """
    
    def __init__(self):
        super().__init__("Weibull")
    
    def _initialize_params(self) -> np.ndarray:
        """ワイブル分布パラメータ初期化: [λ, k, β₁, ..., βₚ]"""
        # λ (scale), k (shape), βs (covariate coefficients)
        return np.concatenate([
            [1.0, 1.0],  # λ, k
            np.zeros(self.n_features)  # βs
        ])
    
    def _get_param_bounds(self) -> List[Tuple[float, float]]:
        """パラメータ境界: λ > 0, k > 0, βs は自由"""
        bounds = [(1e-6, np.inf), (1e-6, np.inf)]  # λ, k
        bounds.extend([(-np.inf, np.inf)] * self.n_features)  # βs
        return bounds
    
    def _log_likelihood(self, params: np.ndarray, X: np.ndarray, 
                        T: np.ndarray, E: np.ndarray) -> float:
        """ワイブル分布の対数尤度"""
        lambda_param = params[0]
        k = params[1]
        betas = params[2:]
        
        # 線形予測子
        linear_predictor = X @ betas
        
        # 対数尤度計算
        log_lik = 0.0
        
        for i in range(len(T)):
            t_i = T[i]
            e_i = E[i]
            eta_i = np.exp(linear_predictor[i])
            
            # ワイブル分布のパラメータ調整
            scale_i = lambda_param * eta_i
            
            if e_i == 1:  # イベント発生（企業消滅）
                # 確率密度関数の対数
                log_lik += (np.log(k) - np.log(scale_i) + 
                           (k - 1) * (np.log(t_i) - np.log(scale_i)) -
                           (t_i / scale_i) ** k)
            else:  # 打ち切り
                # 生存関数の対数
                log_lik += -(t_i / scale_i) ** k
        
        return log_lik
    
    def survival_function(self, t: Union[float, np.ndarray], 
                            X: Optional[np.ndarray] = None) -> np.ndarray:
        """ワイブル生存関数"""
        if not self.fitted:
            raise ValueError("Model must be fitted")
        
        lambda_param = self.params[0]
        k = self.params[1]
        betas = self.params[2:]
        
        t = np.asarray(t)
        
        if X is None:
            # 基準生存関数
            scale = lambda_param
            return np.exp(-(t / scale) ** k)
        else:
            X = np.asarray(X)
            if X.ndim == 1:
                X = X.reshape(1, -1)
            
            linear_predictor = X @ betas
            eta = np.exp(linear_predictor)
            
            if t.ndim == 0:  # スカラー時間
                scales = lambda_param * eta
                return np.exp(-(t / scales) ** k)
            else:  # 時間配列
                survival_probs = np.zeros((len(X), len(t)))
                for i, eta_i in enumerate(eta):
                    scale_i = lambda_param * eta_i
                    survival_probs[i, :] = np.exp(-(t / scale_i) ** k)
                return survival_probs.squeeze()
    
    def hazard_function(self, t: Union[float, np.ndarray], 
                        X: Optional[np.ndarray] = None) -> np.ndarray:
        """ワイブルハザード関数"""
        if not self.fitted:
            raise ValueError("Model must be fitted")
        
        lambda_param = self.params[0]
        k = self.params[1]
        betas = self.params[2:]
        
        t = np.asarray(t)
        
        if X is None:
            scale = lambda_param
            return (k / scale) * (t / scale) ** (k - 1)
        else:
            X = np.asarray(X)
            if X.ndim == 1:
                X = X.reshape(1, -1)
            
            linear_predictor = X @ betas
            eta = np.exp(linear_predictor)
            
            if t.ndim == 0:
                scales = lambda_param * eta
                return (k / scales) * (t / scales) ** (k - 1)
            else:
                hazard_rates = np.zeros((len(X), len(t)))
                for i, eta_i in enumerate(eta):
                    scale_i = lambda_param * eta_i
                    hazard_rates[i, :] = (k / scale_i) * (t / scale_i) ** (k - 1)
                return hazard_rates.squeeze()


class ExponentialSurvival(ParametricSurvivalBase):
    """
    指数分布による生存分析モデル
    
    一定のハザード率を仮定。企業の「突然死」モデルに適用。
    """
    
    def __init__(self):
        super().__init__("Exponential")
    
    def _initialize_params(self) -> np.ndarray:
        """指数分布パラメータ初期化: [λ, β₁, ..., βₚ]"""
        return np.concatenate([
            [1.0],  # λ
            np.zeros(self.n_features)  # βs
        ])
    
    def _get_param_bounds(self) -> List[Tuple[float, float]]:
        bounds = [(1e-6, np.inf)]  # λ > 0
        bounds.extend([(-np.inf, np.inf)] * self.n_features)
        return bounds
    
    def _log_likelihood(self, params: np.ndarray, X: np.ndarray, 
                        T: np.ndarray, E: np.ndarray) -> float:
        """指数分布の対数尤度"""
        lambda_param = params[0]
        betas = params[1:]
        
        linear_predictor = X @ betas
        
        log_lik = 0.0
        for i in range(len(T)):
            t_i = T[i]
            e_i = E[i]
            rate_i = lambda_param * np.exp(linear_predictor[i])
            
            if e_i == 1:
                log_lik += np.log(rate_i) - rate_i * t_i
            else:
                log_lik += -rate_i * t_i
        
        return log_lik
    
    def survival_function(self, t: Union[float, np.ndarray], 
                            X: Optional[np.ndarray] = None) -> np.ndarray:
        """指数生存関数"""
        if not self.fitted:
            raise ValueError("Model must be fitted")
        
        lambda_param = self.params[0]
        betas = self.params[1:]
        
        t = np.asarray(t)
        
        if X is None:
            return np.exp(-lambda_param * t)
        else:
            X = np.asarray(X)
            if X.ndim == 1:
                X = X.reshape(1, -1)
            
            linear_predictor = X @ betas
            rates = lambda_param * np.exp(linear_predictor)
            
            if t.ndim == 0:
                return np.exp(-rates * t)
            else:
                survival_probs = np.zeros((len(X), len(t)))
                for i, rate_i in enumerate(rates):
                    survival_probs[i, :] = np.exp(-rate_i * t)
                return survival_probs.squeeze()
    
    def hazard_function(self, t: Union[float, np.ndarray], 
                        X: Optional[np.ndarray] = None) -> np.ndarray:
        """指数ハザード関数（一定）"""
        if not self.fitted:
            raise ValueError("Model must be fitted")
        
        lambda_param = self.params[0]
        betas = self.params[1:]
        
        t = np.asarray(t)
        
        if X is None:
            return np.full_like(t, lambda_param)
        else:
            X = np.asarray(X)
            if X.ndim == 1:
                X = X.reshape(1, -1)
            
            linear_predictor = X @ betas
            rates = lambda_param * np.exp(linear_predictor)
            
            if t.ndim == 0:
                return rates
            else:
                hazard_rates = np.zeros((len(X), len(t)))
                for i, rate_i in enumerate(rates):
                    hazard_rates[i, :] = np.full_like(t, rate_i)
                return hazard_rates.squeeze()


class LogNormalSurvival(ParametricSurvivalBase):
    """
    対数正規分布による生存分析モデル
    
    企業の成長・衰退が対数正規過程に従うと仮定。
    """
    
    def __init__(self):
        super().__init__("LogNormal")
    
    def _initialize_params(self) -> np.ndarray:
        """対数正規分布パラメータ初期化: [μ, σ, β₁, ..., βₚ]"""
        return np.concatenate([
            [0.0, 1.0],  # μ, σ
            np.zeros(self.n_features)  # βs
        ])
    
    def _get_param_bounds(self) -> List[Tuple[float, float]]:
        bounds = [(-np.inf, np.inf), (1e-6, np.inf)]  # μ free, σ > 0
        bounds.extend([(-np.inf, np.inf)] * self.n_features)
        return bounds
    
    def _log_likelihood(self, params: np.ndarray, X: np.ndarray, 
                        T: np.ndarray, E: np.ndarray) -> float:
        """対数正規分布の対数尤度"""
        mu = params[0]
        sigma = params[1]
        betas = params[2:]
        
        linear_predictor = X @ betas
        
        log_lik = 0.0
        for i in range(len(T)):
            t_i = T[i]
            e_i = E[i]
            mu_i = mu + linear_predictor[i]
            
            if e_i == 1:
                # 対数正規分布の確率密度関数の対数
                log_lik += (-np.log(t_i) - np.log(sigma) - 0.5 * np.log(2 * np.pi) -
                           0.5 * ((np.log(t_i) - mu_i) / sigma) ** 2)
            else:
                # 生存関数の対数（complementary CDF）
                z = (np.log(t_i) - mu_i) / sigma
                log_lik += np.log(1 - stats.norm.cdf(z))
        
        return log_lik
    
    def survival_function(self, t: Union[float, np.ndarray], 
                            X: Optional[np.ndarray] = None) -> np.ndarray:
        """対数正規生存関数"""
        if not self.fitted:
            raise ValueError("Model must be fitted")
        
        mu = self.params[0]
        sigma = self.params[1]
        betas = self.params[2:]
        
        t = np.asarray(t)
        
        if X is None:
            z = (np.log(t) - mu) / sigma
            return 1 - stats.norm.cdf(z)
        else:
            X = np.asarray(X)
            if X.ndim == 1:
                X = X.reshape(1, -1)
            
            linear_predictor = X @ betas
            mus = mu + linear_predictor
            
            if t.ndim == 0:
                z = (np.log(t) - mus) / sigma
                return 1 - stats.norm.cdf(z)
            else:
                survival_probs = np.zeros((len(X), len(t)))
                for i, mu_i in enumerate(mus):
                    z = (np.log(t) - mu_i) / sigma
                    survival_probs[i, :] = 1 - stats.norm.cdf(z)
                return survival_probs.squeeze()
    
    def hazard_function(self, t: Union[float, np.ndarray], 
                        X: Optional[np.ndarray] = None) -> np.ndarray:
        """対数正規ハザード関数"""
        if not self.fitted:
            raise ValueError("Model must be fitted")
        
        mu = self.params[0]
        sigma = self.params[1]
        betas = self.params[2:]
        
        t = np.asarray(t)
        
        if X is None:
            z = (np.log(t) - mu) / sigma
            pdf = stats.norm.pdf(z) / (t * sigma)
            survival = 1 - stats.norm.cdf(z)
            return pdf / survival
        else:
            X = np.asarray(X)
            if X.ndim == 1:
                X = X.reshape(1, -1)
            
            linear_predictor = X @ betas
            mus = mu + linear_predictor
            
            if t.ndim == 0:
                hazard_rates = []
                for mu_i in mus:
                    z = (np.log(t) - mu_i) / sigma
                    pdf = stats.norm.pdf(z) / (t * sigma)
                    survival = 1 - stats.norm.cdf(z)
                    hazard_rates.append(pdf / survival)
                return np.array(hazard_rates)
            else:
                hazard_rates = np.zeros((len(X), len(t)))
                for i, mu_i in enumerate(mus):
                    z = (np.log(t) - mu_i) / sigma
                    pdf = stats.norm.pdf(z) / (t * sigma)
                    survival = 1 - stats.norm.cdf(z)
                    hazard_rates[i, :] = pdf / survival
                return hazard_rates.squeeze()


class GammaSurvival(ParametricSurvivalBase):
    """
    ガンマ分布による生存分析モデル
    
    企業の生存時間がガンマ分布に従うと仮定。
    ワイブル分布と指数分布の中間的性質。
    """
    
    def __init__(self):
        super().__init__("Gamma")
    
    def _initialize_params(self) -> np.ndarray:
        """ガンマ分布パラメータ初期化: [α, β, γ₁, ..., γₚ]"""
        return np.concatenate([
            [1.0, 1.0],  # α (shape), β (scale)
            np.zeros(self.n_features)  # γs
        ])
    
    def _get_param_bounds(self) -> List[Tuple[float, float]]:
        bounds = [(1e-6, np.inf), (1e-6, np.inf)]  # α, β > 0
        bounds.extend([(-np.inf, np.inf)] * self.n_features)
        return bounds
    
    def _log_likelihood(self, params: np.ndarray, X: np.ndarray, 
                        T: np.ndarray, E: np.ndarray) -> float:
        """ガンマ分布の対数尤度"""
        alpha = params[0]
        beta = params[1]
        gammas = params[2:]
        
        linear_predictor = X @ gammas
        
        log_lik = 0.0
        for i in range(len(T)):
            t_i = T[i]
            e_i = E[i]
            beta_i = beta * np.exp(linear_predictor[i])
            
            if e_i == 1:
                # ガンマ分布の確率密度関数の対数
                log_lik += ((alpha - 1) * np.log(t_i) - t_i / beta_i -
                           alpha * np.log(beta_i) - np.log(gamma_func(alpha)))
            else:
                # 生存関数の対数
                log_lik += np.log(1 - stats.gamma.cdf(t_i, a=alpha, scale=beta_i))
        
        return log_lik
    
    def survival_function(self, t: Union[float, np.ndarray], 
                            X: Optional[np.ndarray] = None) -> np.ndarray:
        """ガンマ生存関数"""
        if not self.fitted:
            raise ValueError("Model must be fitted")
        
        alpha = self.params[0]
        beta = self.params[1]
        gammas = self.params[2:]
        
        t = np.asarray(t)
        
        if X is None:
            return 1 - stats.gamma.cdf(t, a=alpha, scale=beta)
        else:
            X = np.asarray(X)
            if X.ndim == 1:
                X = X.reshape(1, -1)
            
            linear_predictor = X @ gammas
            betas = beta * np.exp(linear_predictor)
            
            if t.ndim == 0:
                return 1 - stats.gamma.cdf(t, a=alpha, scale=betas)
            else:
                survival_probs = np.zeros((len(X), len(t)))
                for i, beta_i in enumerate(betas):
                    survival_probs[i, :] = 1 - stats.gamma.cdf(t, a=alpha, scale=beta_i)
                return survival_probs.squeeze()
    
    def hazard_function(self, t: Union[float, np.ndarray], 
                        X: Optional[np.ndarray] = None) -> np.ndarray:
        """ガンマハザード関数"""
        if not self.fitted:
            raise ValueError("Model must be fitted")
        
        alpha = self.params[0]
        beta = self.params[1]
        gammas = self.params[2:]
        
        t = np.asarray(t)
        
        if X is None:
            pdf = stats.gamma.pdf(t, a=alpha, scale=beta)
            survival = 1 - stats.gamma.cdf(t, a=alpha, scale=beta)
            return pdf / survival
        else:
            X = np.asarray(X)
            if X.ndim == 1:
                X = X.reshape(1, -1)
            
            linear_predictor = X @ gammas
            betas = beta * np.exp(linear_predictor)
            
            if t.ndim == 0:
                hazard_rates = []
                for beta_i in betas:
                    pdf = stats.gamma.pdf(t, a=alpha, scale=beta_i)
                    survival = 1 - stats.gamma.cdf(t, a=alpha, scale=beta_i)
                    hazard_rates.append(pdf / survival)
                return np.array(hazard_rates)
            else:
                hazard_rates = np.zeros((len(X), len(t)))
                for i, beta_i in enumerate(betas):
                    pdf = stats.gamma.pdf(t, a=alpha, scale=beta_i)
                    survival = 1 - stats.gamma.cdf(t, a=alpha, scale=beta_i)
                    hazard_rates[i, :] = pdf / survival
                return hazard_rates.squeeze()


class ParametricSurvivalAnalyzer:
    """
    複数のパラメトリック生存分析モデルを統合管理するクラス
    """
    
    def __init__(self):
        self.models = {
            'weibull': WeibullSurvival(),
            'exponential': ExponentialSurvival(),
            'lognormal': LogNormalSurvival(),
            'gamma': GammaSurvival()
        }
        self.fitted_models = {}
        self.model_comparison = None
    
    def fit_all_models(self, X: np.ndarray, T: np.ndarray, E: np.ndarray) -> Dict[str, Any]:
        """
        全てのパラメトリックモデルをフィッティング
        """
        results = {}
        
        for name, model in self.models.items():
            try:
                logger.info(f"Fitting {name} model...")
                fitted_model = model.fit(X, T, E)
                self.fitted_models[name] = fitted_model
                
                results[name] = {
                    'model': fitted_model,
                    'aic': fitted_model.aic_,
                    'bic': fitted_model.bic_,
                    'log_likelihood': fitted_model.log_likelihood_
                }
                
            except Exception as e:
                logger.error(f"Failed to fit {name} model: {str(e)}")
                results[name] = {'error': str(e)}
        
        # モデル比較
        self.model_comparison = self._compare_models(results)
        return results
    
    def _compare_models(self, results: Dict[str, Any]) -> pd.DataFrame:
        """モデル比較結果をDataFrameで作成"""
        comparison_data = []
        
        for name, result in results.items():
            if 'error' not in result:
                comparison_data.append({
                    'Model': name,
                    'AIC': result['aic'],
                    'BIC': result['bic'],
                    'Log_Likelihood': result['log_likelihood']
                })
        
        df = pd.DataFrame(comparison_data)
        if not df.empty:
            df = df.sort_values('AIC').reset_index(drop=True)
            df['AIC_rank'] = range(1, len(df) + 1)
            df['BIC_rank'] = df['BIC'].rank().astype(int)
        
        return df
    
    def get_best_model(self, criteria: str = 'aic') -> Tuple[str, ParametricSurvivalBase]:
        """
        最適モデルを選択
        
        Parameters:
        -----------
        criteria : str
            選択基準 ('aic', 'bic', 'log_likelihood')
        
        Returns:
        --------
        tuple : (model_name, model_instance)
        """
        if self.model_comparison is None or self.model_comparison.empty:
            raise ValueError("Models must be fitted first")
        
        if criteria == 'aic':
            best_model_name = self.model_comparison.loc[0, 'Model']
        elif criteria == 'bic':
            best_model_name = self.model_comparison.loc[
                self.model_comparison['BIC_rank'] == 1, 'Model'
            ].iloc[0]
        elif criteria == 'log_likelihood':
            best_model_name = self.model_comparison.loc[
                self.model_comparison['Log_Likelihood'].idxmax(), 'Model'
            ]
        else:
            raise ValueError("criteria must be 'aic', 'bic', or 'log_likelihood'")
        
        return best_model_name, self.fitted_models[best_model_name]
    
    def predict_survival_curves(self, X: np.ndarray, 
                                time_points: Optional[np.ndarray] = None,
                                model_name: Optional[str] = None) -> Dict[str, np.ndarray]:
        """
        生存曲線の予測
        
        Parameters:
        -----------
        X : array-like
            予測対象の特徴量
        time_points : array-like, optional
            予測時点（デフォルト: 1-40年）
        model_name : str, optional
            使用するモデル名（デフォルト: 最適モデル）
        
        Returns:
        --------
        dict : モデル別生存確率
        """
        if time_points is None:
            time_points = np.linspace(1, 40, 40)
        
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        if model_name is not None:
            if model_name not in self.fitted_models:
                raise ValueError(f"Model '{model_name}' not fitted")
            models_to_use = {model_name: self.fitted_models[model_name]}
        else:
            models_to_use = self.fitted_models
        
        predictions = {}
        for name, model in models_to_use.items():
            try:
                survival_probs = model.predict_survival_probability(X, time_points)
                predictions[name] = survival_probs
            except Exception as e:
                logger.warning(f"Failed to predict with {name}: {str(e)}")
        
        return predictions
    
    def predict_hazard_curves(self, X: np.ndarray,
                                time_points: Optional[np.ndarray] = None,
                                model_name: Optional[str] = None) -> Dict[str, np.ndarray]:
        """
        ハザード曲線の予測
        """
        if time_points is None:
            time_points = np.linspace(1, 40, 40)
        
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        if model_name is not None:
            if model_name not in self.fitted_models:
                raise ValueError(f"Model '{model_name}' not fitted")
            models_to_use = {model_name: self.fitted_models[model_name]}
        else:
            models_to_use = self.fitted_models
        
        predictions = {}
        for name, model in models_to_use.items():
            try:
                hazard_rates = model.predict_hazard(X, time_points)
                predictions[name] = hazard_rates
            except Exception as e:
                logger.warning(f"Failed to predict hazard with {name}: {str(e)}")
        
        return predictions
    
    def plot_survival_curves(self, X: np.ndarray, 
                            time_points: Optional[np.ndarray] = None,
                            model_names: Optional[List[str]] = None,
                            figsize: Tuple[int, int] = (12, 8),
                            save_path: Optional[str] = None) -> plt.Figure:
        """
        生存曲線の可視化
        """
        if time_points is None:
            time_points = np.linspace(1, 40, 40)
        
        if model_names is None:
            model_names = list(self.fitted_models.keys())
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Parametric Survival Analysis Results', fontsize=16)
        axes = axes.flatten()
        
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        for idx, model_name in enumerate(model_names[:4]):  # 最大4モデル
            if model_name in self.fitted_models:
                ax = axes[idx]
                
                survival_probs = self.predict_survival_curves(
                    X, time_points, model_name
                )[model_name]
                
                if survival_probs.ndim == 1:
                    ax.plot(time_points, survival_probs, 
                            color=colors[idx], linewidth=2, 
                            label=f'{model_name.capitalize()} Model')
                else:
                    for i in range(min(5, survival_probs.shape[0])):  # 最大5企業
                        ax.plot(time_points, survival_probs[i], 
                                color=colors[idx], alpha=0.7, 
                                label=f'{model_name.capitalize()} - Company {i+1}' if i == 0 else '')
                
                ax.set_xlabel('Time (Years)')
                ax.set_ylabel('Survival Probability')
                ax.set_title(f'{model_name.capitalize()} Distribution')
                ax.grid(True, alpha=0.3)
                ax.legend()
                ax.set_ylim(0, 1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_hazard_curves(self, X: np.ndarray,
                            time_points: Optional[np.ndarray] = None,
                            model_names: Optional[List[str]] = None,
                            figsize: Tuple[int, int] = (12, 8),
                            save_path: Optional[str] = None) -> plt.Figure:
        """
        ハザード曲線の可視化
        """
        if time_points is None:
            time_points = np.linspace(1, 40, 40)
        
        if model_names is None:
            model_names = list(self.fitted_models.keys())
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Parametric Hazard Function Analysis', fontsize=16)
        axes = axes.flatten()
        
        colors = ['blue', 'red', 'green', 'orange']
        
        for idx, model_name in enumerate(model_names[:4]):
            if model_name in self.fitted_models:
                ax = axes[idx]
                
                hazard_rates = self.predict_hazard_curves(
                    X, time_points, model_name
                )[model_name]
                
                if hazard_rates.ndim == 1:
                    ax.plot(time_points, hazard_rates, 
                            color=colors[idx], linewidth=2,
                            label=f'{model_name.capitalize()} Model')
                else:
                    for i in range(min(5, hazard_rates.shape[0])):
                        ax.plot(time_points, hazard_rates[i], 
                                color=colors[idx], alpha=0.7,
                                label=f'{model_name.capitalize()} - Company {i+1}' if i == 0 else '')
                
                ax.set_xlabel('Time (Years)')
                ax.set_ylabel('Hazard Rate')
                ax.set_title(f'{model_name.capitalize()} Hazard Function')
                ax.grid(True, alpha=0.3)
                ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def model_comparison_table(self) -> pd.DataFrame:
        """モデル比較表の取得"""
        if self.model_comparison is None:
            raise ValueError("Models must be fitted first")
        return self.model_comparison.copy()
    
    def export_predictions(self, X: np.ndarray, 
                            company_names: Optional[List[str]] = None,
                            time_points: Optional[np.ndarray] = None,
                            output_path: Optional[str] = None) -> pd.DataFrame:
        """
        予測結果をDataFrameで出力
        
        Parameters:
        -----------
        X : array-like
            企業の特徴量
        company_names : list, optional
            企業名リスト
        time_points : array-like, optional
            予測時点
        output_path : str, optional
            CSV出力パス
        
        Returns:
        --------
        pd.DataFrame : 予測結果
        """
        if time_points is None:
            time_points = np.array([5, 10, 15, 20, 25, 30])
        
        if company_names is None:
            company_names = [f'Company_{i+1}' for i in range(len(X))]
        
        results_data = []
        
        for i, company_name in enumerate(company_names):
            company_X = X[i:i+1]
            
            # 各モデルでの生存確率予測
            for model_name, model in self.fitted_models.items():
                survival_probs = model.predict_survival_probability(company_X, time_points)
                median_time = model.predict_median_survival_time(company_X)[0]
                
                for t_idx, t in enumerate(time_points):
                    results_data.append({
                        'Company': company_name,
                        'Model': model_name,
                        'Time_Years': t,
                        'Survival_Probability': survival_probs[t_idx] if survival_probs.ndim == 1 else survival_probs[0, t_idx],
                        'Median_Survival_Time': median_time
                    })
        
        results_df = pd.DataFrame(results_data)
        
        if output_path:
            results_df.to_csv(output_path, index=False)
            logger.info(f"Predictions exported to {output_path}")
        
        return results_df
    
    def validate_models(self, X_train: np.ndarray, T_train: np.ndarray, E_train: np.ndarray,
                        X_test: np.ndarray, T_test: np.ndarray, E_test: np.ndarray) -> Dict[str, float]:
        """
        モデルの交差検証
        
        Returns:
        --------
        dict : モデル別C-index（一致度指標）
        """
        validation_results = {}
        
        # 学習
        self.fit_all_models(X_train, T_train, E_train)
        
        # テストデータでの評価
        for model_name, model in self.fitted_models.items():
            try:
                # C-indexの計算
                c_index = self._calculate_c_index(model, X_test, T_test, E_test)
                validation_results[model_name] = c_index
            except Exception as e:
                logger.error(f"Validation failed for {model_name}: {str(e)}")
                validation_results[model_name] = np.nan
        
        return validation_results
    
    def _calculate_c_index(self, model: ParametricSurvivalBase, 
                            X: np.ndarray, T: np.ndarray, E: np.ndarray) -> float:
        """
        Concordance index (C-index) の計算
        
        C-index = P(predicted_risk_i > predicted_risk_j | T_i < T_j, E_i = 1)
        """
        n = len(T)
        
        # リスクスコアの予測（生存時間の逆数として近似）
        try:
            median_times = model.predict_median_survival_time(X)
            risk_scores = 1.0 / (median_times + 1e-8)  # 数値安定性のため小さい値を追加
        except:
            # 中央生存時間が計算できない場合、固定時点での生存確率を使用
            survival_probs = model.predict_survival_probability(X, 20)  # 20年時点
            risk_scores = 1.0 - survival_probs
        
        concordant_pairs = 0
        total_pairs = 0
        
        for i in range(n):
            for j in range(i + 1, n):
                # イベントが発生した方が先に起こった場合のみ評価
                if E[i] == 1 and T[i] < T[j]:
                    total_pairs += 1
                    if risk_scores[i] > risk_scores[j]:
                        concordant_pairs += 1
                elif E[j] == 1 and T[j] < T[i]:
                    total_pairs += 1
                    if risk_scores[j] > risk_scores[i]:
                        concordant_pairs += 1
        
        if total_pairs == 0:
            return np.nan
        
        return concordant_pairs / total_pairs


def create_sample_data(n_companies: int = 100, n_features: int = 10, 
                        random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    サンプルデータの生成（テスト用）
    
    Parameters:
    -----------
    n_companies : int
        企業数
    n_features : int
        特徴量数
    random_state : int
        乱数シード
    
    Returns:
    --------
    tuple : (X, T, E) - 特徴量、生存時間、イベント指示子
    """
    np.random.seed(random_state)
    
    # 特徴量生成
    X = np.random.randn(n_companies, n_features)
    
    # 生存時間生成（ワイブル分布ベース）
    scale_base = 20
    shape = 1.5
    
    # 特徴量の影響を組み込み
    risk_scores = np.exp(0.1 * X.sum(axis=1))
    scales = scale_base / risk_scores
    
    T = np.random.weibull(shape, n_companies) * scales.mean() + 1
    T = np.maximum(T, 0.5)  # 最小0.5年
    
    # イベント指示子（約70%がイベント発生）
    censoring_time = np.random.exponential(30, n_companies)
    E = (T <= censoring_time).astype(int)
    T = np.minimum(T, censoring_time)
    
    return X, T, E


# 使用例とテスト
if __name__ == "__main__":
    # サンプルデータ生成
    print("Generating sample data...")
    X, T, E = create_sample_data(n_companies=150, n_features=23)
    
    print(f"Data shape: X={X.shape}, T={T.shape}, E={E.shape}")
    print(f"Event rate: {E.mean():.2%}")
    print(f"Survival time stats: mean={T.mean():.1f}, std={T.std():.1f}")
    
    # 訓練・テスト分割
    X_train, X_test, T_train, T_test, E_train, E_test = train_test_split(
        X, T, E, test_size=0.3, random_state=42
    )
    
    # パラメトリック生存分析の実行
    print("\nFitting parametric survival models...")
    analyzer = ParametricSurvivalAnalyzer()
    
    # 全モデルのフィッティング
    results = analyzer.fit_all_models(X_train, T_train, E_train)
    
    # モデル比較結果の表示
    print("\nModel Comparison:")
    comparison_df = analyzer.model_comparison_table()
    print(comparison_df)
    
    # 最適モデルの取得
    best_model_name, best_model = analyzer.get_best_model('aic')
    print(f"\nBest model (AIC): {best_model_name}")
    
    # 予測例
    print("\nPrediction example for first 3 companies:")
    sample_companies = X_test[:3]
    time_points = np.array([5, 10, 15, 20])
    
    survival_predictions = analyzer.predict_survival_curves(
        sample_companies, time_points, best_model_name
    )
    
    for company_idx in range(3):
        print(f"\nCompany {company_idx + 1}:")
        for t_idx, t in enumerate(time_points):
            prob = survival_predictions[best_model_name][company_idx, t_idx]
            print(f"  P(T > {t} years) = {prob:.3f}")
    
    # 検証
    print("\nModel validation (C-index):")
    validation_results = analyzer.validate_models(
        X_train, T_train, E_train, X_test, T_test, E_test
    )
    
    for model_name, c_index in validation_results.items():
        print(f"{model_name}: {c_index:.3f}")
    
    # 可視化（オプション）
    try:
        print("\nGenerating survival curves plot...")
        fig = analyzer.plot_survival_curves(sample_companies[:1])
        plt.show()
    except Exception as e:
        print(f"Plotting failed: {e}")
    
    print("\nParametric survival analysis completed successfully!")