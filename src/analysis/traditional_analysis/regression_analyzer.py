"""
A2AI (Advanced Financial Analysis AI) - Regression Analyzer

企業ライフサイクル全体を考慮した高度な回帰分析モジュール
- 150社×40年分の財務データ対応
- 9つの評価項目と23の要因項目の関係分析
- 市場カテゴリ別（高シェア/低下/失失）比較分析
- 生存バイアス補正機能
- 多重共線性対応
- 異分散・系列相関対応
"""

import numpy as np
import pandas as pd
import warnings
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.diagnostic import het_white, acorr_breusch_godfrey
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import jarque_bera, durbin_watson


@dataclass
class RegressionConfig:
    """回帰分析設定クラス"""
    alpha: float = 0.05  # 有意水準
    vif_threshold: float = 10.0  # VIF閾値（多重共線性検出）
    outlier_threshold: float = 3.0  # 外れ値検出閾値（標準偏差倍数）
    min_observations: int = 30  # 最小観測数
    cross_validation_folds: int = 5  # 交差検証フォールド数
    regularization_alphas: List[float] = None  # 正則化パラメータ
    
    def __post_init__(self):
        if self.regularization_alphas is None:
            self.regularization_alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]


@dataclass
class RegressionResults:
    """回帰分析結果格納クラス"""
    model_type: str
    r_squared: float
    adj_r_squared: float
    f_statistic: float
    f_pvalue: float
    coefficients: pd.DataFrame
    residual_stats: Dict[str, float]
    diagnostic_tests: Dict[str, Dict[str, float]]
    vif_scores: pd.DataFrame
    predictions: np.ndarray
    residuals: np.ndarray
    feature_importance: pd.DataFrame
    cross_val_scores: np.ndarray


class RegressionAnalyzer:
    """
    A2AI 回帰分析クラス
    
    企業ライフサイクルを考慮した高度な回帰分析を実行
    """
    
    def __init__(self, config: RegressionConfig = None):
        self.config = config if config else RegressionConfig()
        self.scalers = {}
        self.models = {}
        self.results = {}
        
        # 評価項目定義（9項目）
        self.evaluation_metrics = [
            'sales_revenue',           # 売上高
            'sales_growth_rate',       # 売上高成長率
            'operating_profit_margin', # 売上高営業利益率
            'net_profit_margin',       # 売上高当期純利益率
            'roe',                     # ROE
            'value_added_ratio',       # 売上高付加価値率
            'survival_probability',    # 企業存続確率（新規）
            'emergence_success_rate',  # 新規事業成功率（新規）
            'succession_success_rate'  # 事業継承成功度（新規）
        ]
        
        # 市場カテゴリ
        self.market_categories = ['high_share', 'declining_share', 'lost_share']
        
    def prepare_regression_data(self, 
                                data: pd.DataFrame, 
                                target_metric: str,
                                market_category: Optional[str] = None,
                                exclude_survival_bias: bool = True) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """
        回帰分析用データ準備
        
        Args:
            data: 入力データ
            target_metric: 目的変数（評価項目）
            market_category: 市場カテゴリフィルタ
            exclude_survival_bias: 生存バイアス除去フラグ
            
        Returns:
            特徴量DataFrame, 目的変数Series, 特徴量名リスト
        """
        
        # データフィルタリング
        filtered_data = data.copy()
        
        if market_category:
            filtered_data = filtered_data[filtered_data['market_category'] == market_category]
            
        # 生存バイアス補正
        if exclude_survival_bias:
            # 消滅企業のデータも含めて分析
            # 生存期間で重み付け
            filtered_data['survival_weight'] = self._calculate_survival_weights(filtered_data)
        
        # 目的変数チェック
        if target_metric not in filtered_data.columns:
            raise ValueError(f"Target metric '{target_metric}' not found in data")
            
        # 欠損値処理
        filtered_data = self._handle_missing_values(filtered_data, target_metric)
        
        # 外れ値検出・処理
        filtered_data = self._handle_outliers(filtered_data, target_metric)
        
        # 特徴量選択（要因項目）
        factor_columns = self._get_factor_columns(filtered_data, target_metric)
        
        # データ分割
        X = filtered_data[factor_columns].copy()
        y = filtered_data[target_metric].copy()
        
        # 最小観測数チェック
        if len(X) < self.config.min_observations:
            raise ValueError(f"Insufficient observations: {len(X)} < {self.config.min_observations}")
            
        return X, y, factor_columns
    
    def _calculate_survival_weights(self, data: pd.DataFrame) -> pd.Series:
        """生存期間に基づく重み計算"""
        weights = pd.Series(1.0, index=data.index)
        
        # 企業の生存期間計算
        if 'company_age' in data.columns:
            # 若い企業や短命企業への重み調整
            max_age = data['company_age'].max()
            weights = data['company_age'] / max_age
            weights = weights.clip(lower=0.1)  # 最小重み0.1
            
        return weights
    
    def _handle_missing_values(self, data: pd.DataFrame, target_metric: str) -> pd.DataFrame:
        """欠損値処理"""
        # 目的変数の欠損値除去
        data = data.dropna(subset=[target_metric])
        
        # 説明変数の欠損値処理
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        # 業界中央値で補完
        if 'industry_code' in data.columns:
            for col in numeric_columns:
                if data[col].isnull().sum() > 0:
                    industry_medians = data.groupby('industry_code')[col].median()
                    data[col] = data[col].fillna(data['industry_code'].map(industry_medians))
        
        # 全体中央値で補完
        data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].median())
        
        return data
    
    def _handle_outliers(self, data: pd.DataFrame, target_metric: str) -> pd.DataFrame:
        """外れ値処理"""
        # Z-score法による外れ値検出
        z_scores = np.abs(stats.zscore(data[target_metric]))
        outlier_mask = z_scores < self.config.outlier_threshold
        
        return data[outlier_mask]
    
    def _get_factor_columns(self, data: pd.DataFrame, target_metric: str) -> List[str]:
        """要因項目カラム取得"""
        # 基本要因項目パターン
        factor_patterns = [
            # 投資・資産関連
            'tangible_fixed_assets', 'capital_investment', 'rd_expenses',
            'intangible_assets', 'investment_securities', 'total_payout_ratio',
            
            # 人的資源関連  
            'employee_count', 'average_salary', 'retirement_benefit_cost',
            'welfare_expenses',
            
            # 運転資本・効率性関連
            'accounts_receivable', 'inventory', 'total_assets',
            'receivables_turnover', 'inventory_turnover',
            
            # 事業展開関連
            'overseas_sales_ratio', 'segment_count', 'sga_expenses',
            'advertising_expenses', 'non_operating_income', 'order_backlog',
            
            # 新規追加項目
            'company_age', 'market_entry_timing', 'parent_dependency_ratio'
        ]
        
        # 実際に存在するカラムのみ選択
        available_factors = [col for col in factor_patterns if col in data.columns]
        
        # 目的変数と相関の高い追加変数を自動選択
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        correlations = data[numeric_columns].corr()[target_metric].abs()
        high_corr_factors = correlations[correlations > 0.1].index.tolist()
        
        # 重複除去して結合
        all_factors = list(set(available_factors + high_corr_factors))
        all_factors = [col for col in all_factors if col != target_metric]
        
        return all_factors
    
    def check_multicollinearity(self, X: pd.DataFrame) -> pd.DataFrame:
        """多重共線性チェック（VIF計算）"""
        vif_data = pd.DataFrame()
        vif_data["Feature"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        
        # 高VIF変数の特定
        high_vif_features = vif_data[vif_data["VIF"] > self.config.vif_threshold]["Feature"].tolist()
        
        if high_vif_features:
            warnings.warn(f"High multicollinearity detected in: {high_vif_features}")
            
        return vif_data
    
    def perform_regression_analysis(self, 
                                    X: pd.DataFrame, 
                                    y: pd.Series,
                                    model_types: List[str] = None,
                                    weights: Optional[pd.Series] = None) -> Dict[str, RegressionResults]:
        """
        回帰分析実行
        
        Args:
            X: 特徴量DataFrame
            y: 目的変数Series
            model_types: モデルタイプリスト
            weights: サンプル重み
            
        Returns:
            モデルタイプ別結果辞書
        """
        
        if model_types is None:
            model_types = ['ols', 'ridge', 'lasso', 'elastic_net']
            
        results = {}
        
        # データ標準化
        scaler = RobustScaler()  # 外れ値に頑健
        X_scaled = pd.DataFrame(scaler.fit_transform(X), 
                                columns=X.columns, 
                                index=X.index)
        
        # 各モデルで分析実行
        for model_type in model_types:
            try:
                if model_type == 'ols':
                    result = self._fit_ols_model(X_scaled, y, weights)
                elif model_type == 'ridge':
                    result = self._fit_ridge_model(X_scaled, y, weights)
                elif model_type == 'lasso':
                    result = self._fit_lasso_model(X_scaled, y, weights)
                elif model_type == 'elastic_net':
                    result = self._fit_elastic_net_model(X_scaled, y, weights)
                else:
                    continue
                    
                results[model_type] = result
                
            except Exception as e:
                warnings.warn(f"Failed to fit {model_type} model: {str(e)}")
                continue
        
        return results
    
    def _fit_ols_model(self, X: pd.DataFrame, y: pd.Series, weights: Optional[pd.Series] = None) -> RegressionResults:
        """OLS回帰分析"""
        # 定数項追加
        X_with_const = sm.add_constant(X)
        
        # モデル推定
        if weights is not None:
            model = sm.WLS(y, X_with_const, weights=weights)
        else:
            model = sm.OLS(y, X_with_const)
            
        fitted_model = model.fit()
        
        # 予測値・残差
        predictions = fitted_model.fittedvalues
        residuals = fitted_model.resid
        
        # 係数DataFrame作成
        coefficients = pd.DataFrame({
            'coefficient': fitted_model.params,
            'std_error': fitted_model.bse,
            't_statistic': fitted_model.tvalues,
            'p_value': fitted_model.pvalues,
            'conf_lower': fitted_model.conf_int()[0],
            'conf_upper': fitted_model.conf_int()[1]
        })
        
        # 残差統計
        residual_stats = {
            'mean': residuals.mean(),
            'std': residuals.std(),
            'skewness': residuals.skew(),
            'kurtosis': residuals.kurtosis(),
            'min': residuals.min(),
            'max': residuals.max()
        }
        
        # 診断テスト
        diagnostic_tests = self._perform_diagnostic_tests(X_with_const, y, fitted_model)
        
        # VIF計算
        vif_scores = self.check_multicollinearity(X)
        
        # 特徴量重要度（係数の絶対値）
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': np.abs(fitted_model.params[1:]),  # 定数項除く
            'coefficient': fitted_model.params[1:]
        }).sort_values('importance', ascending=False)
        
        # 交差検証
        cv_scores = self._cross_validate_ols(X, y, weights)
        
        return RegressionResults(
            model_type='ols',
            r_squared=fitted_model.rsquared,
            adj_r_squared=fitted_model.rsquared_adj,
            f_statistic=fitted_model.fvalue,
            f_pvalue=fitted_model.f_pvalue,
            coefficients=coefficients,
            residual_stats=residual_stats,
            diagnostic_tests=diagnostic_tests,
            vif_scores=vif_scores,
            predictions=predictions.values,
            residuals=residuals.values,
            feature_importance=feature_importance,
            cross_val_scores=cv_scores
        )
    
    def _fit_ridge_model(self, X: pd.DataFrame, y: pd.Series, weights: Optional[pd.Series] = None) -> RegressionResults:
        """Ridge回帰分析"""
        from sklearn.model_selection import GridSearchCV
        
        # パラメータチューニング
        ridge = Ridge()
        param_grid = {'alpha': self.config.regularization_alphas}
        
        if weights is not None:
            grid_search = GridSearchCV(ridge, param_grid, cv=self.config.cross_validation_folds, 
                                        scoring='neg_mean_squared_error')
            grid_search.fit(X, y, sample_weight=weights)
        else:
            grid_search = GridSearchCV(ridge, param_grid, cv=self.config.cross_validation_folds, 
                                        scoring='neg_mean_squared_error')
            grid_search.fit(X, y)
        
        best_model = grid_search.best_estimator_
        
        # 予測値・残差
        predictions = best_model.predict(X)
        residuals = y - predictions
        
        # 係数DataFrame作成
        coefficients = pd.DataFrame({
            'coefficient': best_model.coef_,
            'feature': X.columns
        })
        coefficients['abs_coefficient'] = np.abs(coefficients['coefficient'])
        
        # R²計算
        r_squared = r2_score(y, predictions)
        adj_r_squared = 1 - (1 - r_squared) * (len(y) - 1) / (len(y) - X.shape[1] - 1)
        
        # 特徴量重要度
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': np.abs(best_model.coef_),
            'coefficient': best_model.coef_
        }).sort_values('importance', ascending=False)
        
        # 交差検証スコア
        cv_scores = cross_val_score(best_model, X, y, cv=self.config.cross_validation_folds, 
                                    scoring='neg_mean_squared_error')
        
        return RegressionResults(
            model_type='ridge',
            r_squared=r_squared,
            adj_r_squared=adj_r_squared,
            f_statistic=np.nan,  # Ridge回帰ではF統計量は計算されない
            f_pvalue=np.nan,
            coefficients=coefficients,
            residual_stats={'mean': residuals.mean(), 'std': residuals.std()},
            diagnostic_tests={},
            vif_scores=pd.DataFrame(),
            predictions=predictions,
            residuals=residuals,
            feature_importance=feature_importance,
            cross_val_scores=cv_scores
        )
    
    def _fit_lasso_model(self, X: pd.DataFrame, y: pd.Series, weights: Optional[pd.Series] = None) -> RegressionResults:
        """Lasso回帰分析"""
        from sklearn.model_selection import GridSearchCV
        
        # パラメータチューニング
        lasso = Lasso(max_iter=2000)
        param_grid = {'alpha': self.config.regularization_alphas}
        
        grid_search = GridSearchCV(lasso, param_grid, cv=self.config.cross_validation_folds, 
                                    scoring='neg_mean_squared_error')
        grid_search.fit(X, y)
        
        best_model = grid_search.best_estimator_
        
        # 予測値・残差
        predictions = best_model.predict(X)
        residuals = y - predictions
        
        # 係数DataFrame作成（非ゼロ係数のみ）
        coefficients = pd.DataFrame({
            'coefficient': best_model.coef_,
            'feature': X.columns
        })
        coefficients = coefficients[coefficients['coefficient'] != 0]
        
        # R²計算
        r_squared = r2_score(y, predictions)
        adj_r_squared = 1 - (1 - r_squared) * (len(y) - 1) / (len(y) - X.shape[1] - 1)
        
        # 特徴量重要度（選択された特徴量のみ）
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': np.abs(best_model.coef_),
            'coefficient': best_model.coef_
        })
        feature_importance = feature_importance[feature_importance['importance'] > 0]
        feature_importance = feature_importance.sort_values('importance', ascending=False)
        
        # 交差検証スコア
        cv_scores = cross_val_score(best_model, X, y, cv=self.config.cross_validation_folds, 
                                    scoring='neg_mean_squared_error')
        
        return RegressionResults(
            model_type='lasso',
            r_squared=r_squared,
            adj_r_squared=adj_r_squared,
            f_statistic=np.nan,
            f_pvalue=np.nan,
            coefficients=coefficients,
            residual_stats={'mean': residuals.mean(), 'std': residuals.std()},
            diagnostic_tests={},
            vif_scores=pd.DataFrame(),
            predictions=predictions,
            residuals=residuals,
            feature_importance=feature_importance,
            cross_val_scores=cv_scores
        )
    
    def _fit_elastic_net_model(self, X: pd.DataFrame, y: pd.Series, weights: Optional[pd.Series] = None) -> RegressionResults:
        """ElasticNet回帰分析"""
        from sklearn.model_selection import GridSearchCV
        
        # パラメータチューニング
        elastic_net = ElasticNet(max_iter=2000)
        param_grid = {
            'alpha': self.config.regularization_alphas,
            'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
        }
        
        grid_search = GridSearchCV(elastic_net, param_grid, cv=self.config.cross_validation_folds, 
                                    scoring='neg_mean_squared_error')
        grid_search.fit(X, y)
        
        best_model = grid_search.best_estimator_
        
        # 予測値・残差
        predictions = best_model.predict(X)
        residuals = y - predictions
        
        # 係数DataFrame作成
        coefficients = pd.DataFrame({
            'coefficient': best_model.coef_,
            'feature': X.columns
        })
        
        # R²計算
        r_squared = r2_score(y, predictions)
        adj_r_squared = 1 - (1 - r_squared) * (len(y) - 1) / (len(y) - X.shape[1] - 1)
        
        # 特徴量重要度
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': np.abs(best_model.coef_),
            'coefficient': best_model.coef_
        }).sort_values('importance', ascending=False)
        
        # 交差検証スコア
        cv_scores = cross_val_score(best_model, X, y, cv=self.config.cross_validation_folds, 
                                    scoring='neg_mean_squared_error')
        
        return RegressionResults(
            model_type='elastic_net',
            r_squared=r_squared,
            adj_r_squared=adj_r_squared,
            f_statistic=np.nan,
            f_pvalue=np.nan,
            coefficients=coefficients,
            residual_stats={'mean': residuals.mean(), 'std': residuals.std()},
            diagnostic_tests={},
            vif_scores=pd.DataFrame(),
            predictions=predictions,
            residuals=residuals,
            feature_importance=feature_importance,
            cross_val_scores=cv_scores
        )
    
    def _perform_diagnostic_tests(self, X: pd.DataFrame, y: pd.Series, fitted_model) -> Dict[str, Dict[str, float]]:
        """回帰診断テスト実行"""
        tests = {}
        
        try:
            # 正規性テスト（Jarque-Bera）
            jb_stat, jb_pvalue = jarque_bera(fitted_model.resid)
            tests['normality'] = {'statistic': jb_stat, 'p_value': jb_pvalue}
            
            # 等分散性テスト（White test）
            white_stat, white_pvalue = het_white(fitted_model.resid, fitted_model.model.exog)[:2]
            tests['heteroscedasticity'] = {'statistic': white_stat, 'p_value': white_pvalue}
            
            # 系列相関テスト（Breusch-Godfrey）
            bg_stat, bg_pvalue = acorr_breusch_godfrey(fitted_model)[:2]
            tests['serial_correlation'] = {'statistic': bg_stat, 'p_value': bg_pvalue}
            
            # Durbin-Watson統計量
            dw_stat = durbin_watson(fitted_model.resid)
            tests['durbin_watson'] = {'statistic': dw_stat, 'p_value': np.nan}
            
        except Exception as e:
            warnings.warn(f"Some diagnostic tests failed: {str(e)}")
            
        return tests
    
    def _cross_validate_ols(self, X: pd.DataFrame, y: pd.Series, weights: Optional[pd.Series] = None) -> np.ndarray:
        """OLSモデルの交差検証"""
        from sklearn.model_selection import KFold
        
        kf = KFold(n_splits=self.config.cross_validation_folds, shuffle=True, random_state=42)
        cv_scores = []
        
        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # 定数項追加
            X_train_const = sm.add_constant(X_train)
            X_test_const = sm.add_constant(X_test)
            
            # モデル学習
            if weights is not None:
                w_train = weights.iloc[train_idx]
                model = sm.WLS(y_train, X_train_const, weights=w_train).fit()
            else:
                model = sm.OLS(y_train, X_train_const).fit()
            
            # 予測・評価
            y_pred = model.predict(X_test_const)
            score = -mean_squared_error(y_test, y_pred)  # 負の二乗誤差
            cv_scores.append(score)
            
        return np.array(cv_scores)
    
    def compare_markets(self, 
                        data: pd.DataFrame, 
                        target_metric: str,
                        model_type: str = 'ols') -> Dict[str, RegressionResults]:
        """市場カテゴリ別回帰分析比較"""
        
        market_results = {}
        
        for market_category in self.market_categories:
            try:
                # 市場別データ準備
                X, y, factors = self.prepare_regression_data(data, target_metric, market_category)
                
                # 回帰分析実行
                results = self.perform_regression_analysis(X, y, [model_type])
                
                if model_type in results:
                    market_results[market_category] = results[model_type]
                    
            except Exception as e:
                warnings.warn(f"Failed analysis for {market_category}: {str(e)}")
                continue
                
        return market_results
    
    def generate_regression_report(self, results: Dict[str, RegressionResults]) -> pd.DataFrame:
        """回帰分析結果レポート生成"""
        
        report_data = []
        
        for model_type, result in results.items():
            row = {
                'Model': model_type.upper(),
                'R-squared': f"{result.r_squared:.4f}",
                'Adj R-squared': f"{result.adj_r_squared:.4f}",
                'F-statistic': f"{result.f_statistic:.4f}" if not np.isnan(result.f_statistic) else "N/A",
                'F p-value': f"{result.f_pvalue:.4f}" if not np.isnan(result.f_pvalue) else "N/A",
                'CV Score Mean': f"{result.cross_val_scores.mean():.4f}",
                'CV Score Std': f"{result.cross_val_scores.std():.4f}",
                'Significant Features': len(result.feature_importance[result.feature_importance['importance'] > 0])
            }
            report_data.append(row)
            
        return pd.DataFrame(report_data)
    
    def plot_regression_diagnostics(self, result: RegressionResults, figsize: Tuple[int, int] = (15, 10)):
        """回帰診断プロット"""
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle(f'{result.model_type.upper()} Regression Diagnostics', fontsize=16)
        
        # 1. 残差プロット
        axes[0, 0].scatter(result.predictions, result.residuals, alpha=0.7)
        axes[0, 0].axhline(y=0, color='r', linestyle='--')
        axes[0, 0].set_xlabel('Fitted Values')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].set_title('Residuals vs Fitted')
        
        # 2. Q-Q プロット
        stats.probplot(result.residuals, dist="norm", plot=axes[0, 1])
        axes[0, 1].set_title('Q-Q Plot')
        
        # 3. 残差ヒストグラム
        axes[0, 2].hist(result.residuals, bins=30, edgecolor='black', alpha=0.7)
        axes[0, 2].set_xlabel('Residuals')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].set_title('Residuals Distribution')
        
        # 4. 特徴量重要度
        top_features = result.feature_importance.head(10)
        y_pos = np.arange(len(top_features))
        axes[1, 0].barh(y_pos, top_features['importance'])
        axes[1, 0].set_yticks(y_pos)
        axes[1, 0].set_yticklabels(top_features['feature'])
        axes[1, 0].set_xlabel('Importance')
        axes[1, 0].set_title('Top 10 Feature Importance')
        
        # 5. 予測値 vs 実測値
        min_val = min(result.predictions.min(), min(result.predictions + result.residuals))
        max_val = max(result.predictions.max(), max(result.predictions + result.residuals))
        axes[1, 1].scatter(result.predictions + result.residuals, result.predictions, alpha=0.7)
        axes[1, 1].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        axes[1, 1].set_xlabel('Actual Values')
        axes[1, 1].set_ylabel('Predicted Values')
        axes[1, 1].set_title('Predicted vs Actual')
        
        # 6. 交差検証スコア
        if len(result.cross_val_scores) > 0:
            axes[1, 2].boxplot(result.cross_val_scores)
            axes[1, 2].set_ylabel('Cross-Validation Score')
            axes[1, 2].set_title('Cross-Validation Performance')
            axes[1, 2].set_xticklabels(['CV Scores'])
        else:
            axes[1, 2].text(0.5, 0.5, 'No CV Scores Available', 
                            ha='center', va='center', transform=axes[1, 2].transAxes)
            axes[1, 2].set_title('Cross-Validation Performance')
        
        plt.tight_layout()
        return fig
    
    def analyze_factor_impact_by_market(self, 
                                        data: pd.DataFrame, 
                                        target_metric: str) -> pd.DataFrame:
        """市場別要因項目影響度分析"""
        
        impact_results = []
        
        for market_category in self.market_categories:
            try:
                # 市場別回帰分析
                X, y, factors = self.prepare_regression_data(data, target_metric, market_category)
                results = self.perform_regression_analysis(X, y, ['ols'])
                
                if 'ols' in results:
                    result = results['ols']
                    
                    # 有意な要因項目を抽出
                    significant_factors = result.coefficients[
                        (result.coefficients['p_value'] < self.config.alpha) & 
                        (result.coefficients.index != 'const')
                    ].copy()
                    
                    for factor, row in significant_factors.iterrows():
                        impact_results.append({
                            'market_category': market_category,
                            'factor': factor,
                            'coefficient': row['coefficient'],
                            'p_value': row['p_value'],
                            't_statistic': row['t_statistic'],
                            'impact_magnitude': abs(row['coefficient']),
                            'direction': 'positive' if row['coefficient'] > 0 else 'negative'
                        })
                        
            except Exception as e:
                warnings.warn(f"Failed factor impact analysis for {market_category}: {str(e)}")
                continue
        
        if impact_results:
            impact_df = pd.DataFrame(impact_results)
            return impact_df.sort_values(['market_category', 'impact_magnitude'], ascending=[True, False])
        else:
            return pd.DataFrame()
    
    def perform_comprehensive_analysis(self, 
                                        data: pd.DataFrame) -> Dict[str, Any]:
        """包括的回帰分析実行"""
        
        comprehensive_results = {
            'evaluation_metrics_analysis': {},
            'market_comparison': {},
            'factor_impact_summary': {},
            'model_performance_comparison': {},
            'statistical_insights': {}
        }
        
        # 各評価項目に対する分析
        for metric in self.evaluation_metrics:
            if metric not in data.columns:
                continue
                
            try:
                print(f"Analyzing {metric}...")
                
                # 全市場統合分析
                X, y, factors = self.prepare_regression_data(data, metric)
                all_market_results = self.perform_regression_analysis(X, y)
                comprehensive_results['evaluation_metrics_analysis'][metric] = all_market_results
                
                # 市場別比較分析
                market_results = self.compare_markets(data, metric)
                comprehensive_results['market_comparison'][metric] = market_results
                
                # 要因影響度分析
                factor_impact = self.analyze_factor_impact_by_market(data, metric)
                comprehensive_results['factor_impact_summary'][metric] = factor_impact
                
            except Exception as e:
                warnings.warn(f"Failed comprehensive analysis for {metric}: {str(e)}")
                continue
        
        # モデル性能比較
        comprehensive_results['model_performance_comparison'] = self._compare_model_performance(
            comprehensive_results['evaluation_metrics_analysis']
        )
        
        # 統計的インサイト生成
        comprehensive_results['statistical_insights'] = self._generate_statistical_insights(
            comprehensive_results
        )
        
        return comprehensive_results
    
    def _compare_model_performance(self, 
                                    evaluation_results: Dict[str, Dict[str, RegressionResults]]) -> pd.DataFrame:
        """モデル性能比較分析"""
        
        performance_data = []
        
        for metric, models in evaluation_results.items():
            for model_type, result in models.items():
                performance_data.append({
                    'evaluation_metric': metric,
                    'model_type': model_type,
                    'r_squared': result.r_squared,
                    'adj_r_squared': result.adj_r_squared,
                    'cv_score_mean': result.cross_val_scores.mean() if len(result.cross_val_scores) > 0 else np.nan,
                    'cv_score_std': result.cross_val_scores.std() if len(result.cross_val_scores) > 0 else np.nan,
                    'significant_features': len(result.feature_importance[result.feature_importance['importance'] > 0])
                })
        
        return pd.DataFrame(performance_data)
    
    def _generate_statistical_insights(self, comprehensive_results: Dict[str, Any]) -> Dict[str, Any]:
        """統計的インサイト生成"""
        
        insights = {
            'best_performing_models': {},
            'most_important_factors': {},
            'market_differences': {},
            'model_stability': {}
        }
        
        # 最良モデル特定
        performance_df = comprehensive_results['model_performance_comparison']
        if not performance_df.empty:
            for metric in performance_df['evaluation_metric'].unique():
                metric_data = performance_df[performance_df['evaluation_metric'] == metric]
                best_model = metric_data.loc[metric_data['adj_r_squared'].idxmax()]
                insights['best_performing_models'][metric] = {
                    'model': best_model['model_type'],
                    'adj_r_squared': best_model['adj_r_squared'],
                    'cv_stability': best_model['cv_score_std']
                }
        
        # 最重要要因項目
        for metric, factor_impact in comprehensive_results['factor_impact_summary'].items():
            if not factor_impact.empty:
                # 全市場で共通して重要な要因
                factor_frequency = factor_impact['factor'].value_counts()
                common_factors = factor_frequency[factor_frequency >= 2].index.tolist()  # 2市場以上で有意
                
                # 影響度の平均
                avg_impact = factor_impact.groupby('factor')['impact_magnitude'].mean().sort_values(ascending=False)
                
                insights['most_important_factors'][metric] = {
                    'common_factors': common_factors,
                    'top_impact_factors': avg_impact.head(5).to_dict()
                }
        
        # 市場間差異分析
        for metric, market_results in comprehensive_results['market_comparison'].items():
            if len(market_results) >= 2:
                r_squared_by_market = {market: result.r_squared 
                                        for market, result in market_results.items()}
                
                insights['market_differences'][metric] = {
                    'r_squared_range': max(r_squared_by_market.values()) - min(r_squared_by_market.values()),
                    'best_explained_market': max(r_squared_by_market.keys(), 
                                                key=lambda k: r_squared_by_market[k]),
                    'r_squared_by_market': r_squared_by_market
                }
        
        return insights
    
    def export_results_to_excel(self, 
                                comprehensive_results: Dict[str, Any], 
                                filepath: str):
        """結果をExcelファイルにエクスポート"""
        
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            
            # モデル性能比較
            if 'model_performance_comparison' in comprehensive_results:
                comprehensive_results['model_performance_comparison'].to_excel(
                    writer, sheet_name='Model_Performance', index=False
                )
            
            # 要因影響度サマリー
            for metric, factor_impact in comprehensive_results['factor_impact_summary'].items():
                if not factor_impact.empty:
                    sheet_name = f'Factor_Impact_{metric[:25]}'  # シート名制限対応
                    factor_impact.to_excel(writer, sheet_name=sheet_name, index=False)
            
            # 統計的インサイト
            insights_data = []
            for category, insights in comprehensive_results['statistical_insights'].items():
                if isinstance(insights, dict):
                    for key, value in insights.items():
                        insights_data.append({
                            'category': category,
                            'metric': key,
                            'insight': str(value)
                        })
            
            if insights_data:
                pd.DataFrame(insights_data).to_excel(
                    writer, sheet_name='Statistical_Insights', index=False
                )
    
    def create_factor_impact_heatmap(self, 
                                    factor_impact_data: Dict[str, pd.DataFrame],
                                    figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """要因項目影響度ヒートマップ作成"""
        
        # 全評価項目・全市場の要因影響度を統合
        all_impacts = []
        
        for metric, impact_df in factor_impact_data.items():
            if not impact_df.empty:
                pivot_data = impact_df.pivot_table(
                    values='impact_magnitude', 
                    index='factor', 
                    columns='market_category', 
                    aggfunc='mean'
                ).fillna(0)
                
                # メトリック名を列に追加
                pivot_data.columns = [f"{metric}_{col}" for col in pivot_data.columns]
                all_impacts.append(pivot_data)
        
        if not all_impacts:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, 'No significant factor impacts found', 
                    ha='center', va='center', transform=ax.transAxes)
            return fig
        
        # データ結合
        combined_impact = pd.concat(all_impacts, axis=1).fillna(0)
        
        # ヒートマップ作成
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(combined_impact, 
                    annot=True, 
                    fmt='.3f', 
                    cmap='RdYlBu_r',
                    center=0,
                    ax=ax)
        
        ax.set_title('Factor Impact Heatmap Across Markets and Metrics', fontsize=16)
        ax.set_xlabel('Metric_Market Category')
        ax.set_ylabel('Factor Variables')
        
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        return fig
    
    def generate_executive_summary(self, comprehensive_results: Dict[str, Any]) -> str:
        """エグゼクティブサマリー生成"""
        
        summary_sections = []
        
        # 分析概要
        summary_sections.append("# A2AI Regression Analysis - Executive Summary")
        summary_sections.append("\n## Analysis Overview")
        
        analyzed_metrics = len(comprehensive_results['evaluation_metrics_analysis'])
        summary_sections.append(f"- Analyzed {analyzed_metrics} evaluation metrics across 150 companies")
        summary_sections.append(f"- Compared performance across {len(self.market_categories)} market categories")
        summary_sections.append("- Applied multiple regression techniques (OLS, Ridge, Lasso, ElasticNet)")
        
        # 主要発見
        summary_sections.append("\n## Key Findings")
        
        insights = comprehensive_results.get('statistical_insights', {})
        
        # 最良モデル
        if 'best_performing_models' in insights:
            summary_sections.append("\n### Best Performing Models:")
            for metric, best_model in insights['best_performing_models'].items():
                summary_sections.append(
                    f"- {metric}: {best_model['model'].upper()} "
                    f"(Adj R² = {best_model['adj_r_squared']:.3f})"
                )
        
        # 重要要因
        if 'most_important_factors' in insights:
            summary_sections.append("\n### Most Important Factors:")
            for metric, factors in insights['most_important_factors'].items():
                common_factors = factors.get('common_factors', [])
                if common_factors:
                    summary_sections.append(
                        f"- {metric}: {', '.join(common_factors[:3])}"
                    )
        
        # 市場差異
        if 'market_differences' in insights:
            summary_sections.append("\n### Market Category Differences:")
            for metric, diff in insights['market_differences'].items():
                best_market = diff.get('best_explained_market', 'Unknown')
                r_squared_range = diff.get('r_squared_range', 0)
                summary_sections.append(
                    f"- {metric}: Best explained in {best_market} market "
                    f"(R² range: {r_squared_range:.3f})"
                )
        
        # 推奨事項
        summary_sections.append("\n## Recommendations")
        summary_sections.append("1. Focus on factors consistently significant across market categories")
        summary_sections.append("2. Apply market-specific strategies based on varying factor importance")
        summary_sections.append("3. Consider regularized models for high-dimensional factor analysis")
        summary_sections.append("4. Implement continuous monitoring of model performance and factor relevance")
        
        return '\n'.join(summary_sections)


# 使用例とテスト関数
def example_usage():
    """A2AI回帰分析の使用例"""
    
    # 設定作成
    config = RegressionConfig(
        alpha=0.05,
        vif_threshold=10.0,
        min_observations=50,
        cross_validation_folds=5
    )
    
    # アナライザー初期化
    analyzer = RegressionAnalyzer(config)
    
    # サンプルデータ作成（実際の使用では外部データを読み込み）
    np.random.seed(42)
    n_companies = 150
    n_years = 40
    
    sample_data = pd.DataFrame({
        'company_id': np.repeat(range(n_companies), n_years),
        'year': np.tile(range(1984, 2024), n_companies),
        'market_category': np.repeat(
            ['high_share', 'declining_share', 'lost_share'], 
            n_companies * n_years // 3
        ),
        
        # 評価項目（目的変数例）
        'sales_revenue': np.random.normal(1000, 500, n_companies * n_years),
        'roe': np.random.normal(0.1, 0.05, n_companies * n_years),
        
        # 要因項目（説明変数例）
        'rd_expenses': np.random.normal(50, 20, n_companies * n_years),
        'employee_count': np.random.normal(5000, 2000, n_companies * n_years),
        'total_assets': np.random.normal(10000, 5000, n_companies * n_years),
        'company_age': np.random.randint(1, 40, n_companies * n_years)
    })
    
    # 包括的分析実行
    results = analyzer.perform_comprehensive_analysis(sample_data)
    
    # エグゼクティブサマリー生成
    summary = analyzer.generate_executive_summary(results)
    print(summary)
    
    return analyzer, results


if __name__ == "__main__":
    # テスト実行
    analyzer, results = example_usage()