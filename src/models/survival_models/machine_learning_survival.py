"""
A2AI - Advanced Financial Analysis AI
Machine Learning-based Survival Models

このモジュールは機械学習手法を用いた生存分析モデルを実装します。
企業の倒産・消滅リスクを予測し、生存確率に影響する要因を特定します。

主要機能：
- Random Forest Survival Analysis
- Gradient Boosting Survival Models
- Neural Network-based Survival Models
- Deep Survival Learning
- Feature Importance Analysis for Survival
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import warnings

# 機械学習ライブラリ
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb

# 生存分析専用ライブラリ
try:
    from lifelines import CoxPHFitter
    from lifelines.utils import concordance_index
    from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
    from sksurv.linear_model import CoxPHSurvivalAnalysis
    from sksurv.metrics import concordance_index_censored, integrated_brier_score
    SURVIVAL_LIBS_AVAILABLE = True
except ImportError:
    SURVIVAL_LIBS_AVAILABLE = False
    warnings.warn("Survival analysis libraries not available. Installing lifelines, scikit-survival recommended.")

# ディープラーニング
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. Deep learning models will be disabled.")

# 可視化
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure

# ユーティリティ
from ..utils.survival_utils import SurvivalUtils
from ..utils.statistical_utils import StatisticalUtils


@dataclass
class SurvivalPrediction:
    """生存予測結果を格納するデータクラス"""
    company_id: str
    survival_probability: float
    risk_score: float
    predicted_survival_time: Optional[float]
    confidence_interval: Optional[Tuple[float, float]]
    feature_importance: Optional[Dict[str, float]]


@dataclass
class ModelPerformance:
    """モデル性能指標を格納するデータクラス"""
    model_name: str
    concordance_index: float
    brier_score: Optional[float]
    integrated_brier_score: Optional[float]
    auc: Optional[float]
    rmse: Optional[float]
    mae: Optional[float]
    r2: Optional[float]


class BaseSurvivalMLModel(ABC):
    """機械学習生存分析モデルの基底クラス"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.is_fitted = False
        self.feature_names = None
        self.scaler = None
        self.performance_metrics = None
        
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: np.ndarray, duration: np.ndarray) -> None:
        """モデルの学習"""
        pass
    
    @abstractmethod
    def predict_survival_probability(self, X: pd.DataFrame, time_points: np.ndarray) -> np.ndarray:
        """生存確率の予測"""
        pass
    
    @abstractmethod
    def predict_risk_score(self, X: pd.DataFrame) -> np.ndarray:
        """リスクスコアの予測"""
        pass
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """特徴量重要度の取得"""
        if hasattr(self.model, 'feature_importances_'):
            return dict(zip(self.feature_names, self.model.feature_importances_))
        return None


class RandomForestSurvival(BaseSurvivalMLModel):
    """Random Forest生存分析モデル"""
    
    def __init__(self, n_estimators: int = 100, max_depth: Optional[int] = None, 
                    min_samples_split: int = 2, random_state: int = 42):
        super().__init__("Random Forest Survival")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        
        if SURVIVAL_LIBS_AVAILABLE:
            self.model = RandomSurvivalForest(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                random_state=random_state,
                n_jobs=-1
            )
        else:
            # フォールバック：通常のRandom Forestで回帰
            self.model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                random_state=random_state,
                n_jobs=-1
            )
    
    def fit(self, X: pd.DataFrame, y: np.ndarray, duration: np.ndarray) -> None:
        """
        モデルの学習
        
        Args:
            X: 特徴量データ
            y: イベント発生フラグ (1: イベント発生, 0: 打ち切り)
            duration: 観測期間または生存時間
        """
        self.feature_names = X.columns.tolist()
        
        # データの標準化
        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=self.feature_names)
        
        if SURVIVAL_LIBS_AVAILABLE:
            # 構造化配列の作成（scikit-survival用）
            y_structured = np.array([(bool(event), time) for event, time in zip(y, duration)],
                                    dtype=[('event', bool), ('time', float)])
            self.model.fit(X_scaled, y_structured)
        else:
            # フォールバック：生存時間を直接予測
            self.model.fit(X_scaled, duration)
        
        self.is_fitted = True
    
    def predict_survival_probability(self, X: pd.DataFrame, time_points: np.ndarray) -> np.ndarray:
        """生存確率の予測"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_scaled = self.scaler.transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=self.feature_names)
        
        if SURVIVAL_LIBS_AVAILABLE:
            # 生存関数の予測
            survival_functions = self.model.predict_survival_function(X_scaled)
            predictions = np.array([sf(time_points) for sf in survival_functions])
            return predictions
        else:
            # フォールバック：生存時間予測から確率を推定
            predicted_times = self.model.predict(X_scaled)
            # 簡易的な生存確率計算（指数分布を仮定）
            predictions = np.exp(-time_points.reshape(1, -1) / predicted_times.reshape(-1, 1))
            return predictions
    
    def predict_risk_score(self, X: pd.DataFrame) -> np.ndarray:
        """リスクスコアの予測"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_scaled = self.scaler.transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=self.feature_names)
        
        if SURVIVAL_LIBS_AVAILABLE:
            # リスクスコアとして累積ハザード関数を使用
            risk_scores = -np.log(self.predict_survival_probability(X_scaled, np.array([1.0]))).flatten()
            return risk_scores
        else:
            # フォールバック：予測生存時間の逆数をリスクスコアとする
            predicted_times = self.model.predict(X_scaled)
            return 1.0 / (predicted_times + 1e-8)


class GradientBoostingSurvival(BaseSurvivalMLModel):
    """Gradient Boosting生存分析モデル"""
    
    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1,
                    max_depth: int = 3, random_state: int = 42):
        super().__init__("Gradient Boosting Survival")
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        
        if SURVIVAL_LIBS_AVAILABLE:
            self.model = GradientBoostingSurvivalAnalysis(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                random_state=random_state
            )
        else:
            self.model = GradientBoostingRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                random_state=random_state
            )
    
    def fit(self, X: pd.DataFrame, y: np.ndarray, duration: np.ndarray) -> None:
        """モデルの学習"""
        self.feature_names = X.columns.tolist()
        
        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=self.feature_names)
        
        if SURVIVAL_LIBS_AVAILABLE:
            y_structured = np.array([(bool(event), time) for event, time in zip(y, duration)],
                                    dtype=[('event', bool), ('time', float)])
            self.model.fit(X_scaled, y_structured)
        else:
            self.model.fit(X_scaled, duration)
        
        self.is_fitted = True
    
    def predict_survival_probability(self, X: pd.DataFrame, time_points: np.ndarray) -> np.ndarray:
        """生存確率の予測"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_scaled = self.scaler.transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=self.feature_names)
        
        if SURVIVAL_LIBS_AVAILABLE:
            survival_functions = self.model.predict_survival_function(X_scaled)
            predictions = np.array([sf(time_points) for sf in survival_functions])
            return predictions
        else:
            predicted_times = self.model.predict(X_scaled)
            predictions = np.exp(-time_points.reshape(1, -1) / predicted_times.reshape(-1, 1))
            return predictions
    
    def predict_risk_score(self, X: pd.DataFrame) -> np.ndarray:
        """リスクスコアの予測"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_scaled = self.scaler.transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=self.feature_names)
        
        if SURVIVAL_LIBS_AVAILABLE:
            risk_scores = -np.log(self.predict_survival_probability(X_scaled, np.array([1.0]))).flatten()
            return risk_scores
        else:
            predicted_times = self.model.predict(X_scaled)
            return 1.0 / (predicted_times + 1e-8)


class DeepSurvivalNetwork(nn.Module):
    """深層学習による生存分析ネットワーク"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 64, 32], 
                    dropout_rate: float = 0.2):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # 最終層：リスクスコア出力
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)


class DeepSurvival(BaseSurvivalMLModel):
    """深層学習生存分析モデル"""
    
    def __init__(self, hidden_dims: List[int] = [128, 64, 32], 
                    dropout_rate: float = 0.2, learning_rate: float = 0.001,
                    batch_size: int = 32, epochs: int = 100, random_state: int = 42):
        super().__init__("Deep Survival Network")
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.random_state = random_state
        
        if TORCH_AVAILABLE:
            torch.manual_seed(random_state)
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            raise ImportError("PyTorch is required for DeepSurvival model")
    
    def _partial_likelihood_loss(self, risk_scores, y, duration):
        """部分尤度損失関数（Cox比例ハザードモデルベース）"""
        # イベントが発生したサンプルのみを考慮
        event_mask = y.bool()
        
        if event_mask.sum() == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        event_risk_scores = risk_scores[event_mask]
        event_times = duration[event_mask]
        
        loss = 0.0
        for i, (risk_i, time_i) in enumerate(zip(event_risk_scores, event_times)):
            # time_i以降に観測されている全サンプル
            at_risk_mask = duration >= time_i
            at_risk_scores = risk_scores[at_risk_mask]
            
            if len(at_risk_scores) > 0:
                log_partial_likelihood = risk_i - torch.logsumexp(at_risk_scores, dim=0)
                loss -= log_partial_likelihood
        
        return loss / len(event_risk_scores)
    
    def fit(self, X: pd.DataFrame, y: np.ndarray, duration: np.ndarray) -> None:
        """モデルの学習"""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for DeepSurvival model")
        
        self.feature_names = X.columns.tolist()
        
        # データの標準化
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # PyTorchテンソルに変換
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        duration_tensor = torch.FloatTensor(duration).to(self.device)
        
        # データセットの作成
        dataset = TensorDataset(X_tensor, y_tensor, duration_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # ネットワークの初期化
        input_dim = X_scaled.shape[1]
        self.model = DeepSurvivalNetwork(input_dim, self.hidden_dims, self.dropout_rate).to(self.device)
        
        # オプティマイザーの設定
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # 学習ループ
        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            
            for batch_X, batch_y, batch_duration in dataloader:
                optimizer.zero_grad()
                
                risk_scores = self.model(batch_X).squeeze()
                loss = self._partial_likelihood_loss(risk_scores, batch_y, batch_duration)
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            if (epoch + 1) % 20 == 0:
                avg_loss = epoch_loss / len(dataloader)
                print(f"Epoch {epoch + 1}/{self.epochs}, Average Loss: {avg_loss:.4f}")
        
        self.model.eval()
        self.is_fitted = True
    
    def predict_survival_probability(self, X: pd.DataFrame, time_points: np.ndarray) -> np.ndarray:
        """生存確率の予測"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        risk_scores = self.predict_risk_score(X)
        
        # ベースライン生存関数を仮定（指数分布）
        # 実際の実装では学習データから推定する必要がある
        baseline_hazard = 0.1  # 簡略化
        
        predictions = []
        for risk_score in risk_scores:
            hazard = baseline_hazard * np.exp(risk_score)
            survival_prob = np.exp(-hazard * time_points)
            predictions.append(survival_prob)
        
        return np.array(predictions)
    
    def predict_risk_score(self, X: pd.DataFrame) -> np.ndarray:
        """リスクスコアの予測"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        with torch.no_grad():
            risk_scores = self.model(X_tensor).squeeze().cpu().numpy()
        
        return risk_scores


class MachineLearninSurvivalAnalyzer:
    """機械学習生存分析の統合クラス"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.model_performances = {}
        self.feature_importance_results = {}
        self.survival_utils = SurvivalUtils()
        self.statistical_utils = StatisticalUtils()
    
    def add_model(self, model: BaseSurvivalMLModel) -> None:
        """モデルの追加"""
        self.models[model.model_name] = model
    
    def fit_all_models(self, X: pd.DataFrame, y: np.ndarray, duration: np.ndarray,
                        test_size: float = 0.2) -> None:
        """全モデルの学習と評価"""
        # 訓練・テストセット分割
        X_train, X_test, y_train, y_test, duration_train, duration_test = train_test_split(
            X, y, duration, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        for model_name, model in self.models.items():
            print(f"Training {model_name}...")
            
            try:
                # モデルの学習
                model.fit(X_train, y_train, duration_train)
                
                # 性能評価
                performance = self._evaluate_model(model, X_test, y_test, duration_test)
                self.model_performances[model_name] = performance
                
                # 特徴量重要度の取得
                importance = model.get_feature_importance()
                if importance:
                    self.feature_importance_results[model_name] = importance
                
                print(f"Completed {model_name}: C-index = {performance.concordance_index:.4f}")
                
            except Exception as e:
                print(f"Error training {model_name}: {str(e)}")
                continue
    
    def _evaluate_model(self, model: BaseSurvivalMLModel, X_test: pd.DataFrame, 
                        y_test: np.ndarray, duration_test: np.ndarray) -> ModelPerformance:
        """モデル性能の評価"""
        try:
            # リスクスコアの予測
            risk_scores = model.predict_risk_score(X_test)
            
            # C-index（Concordance Index）の計算
            if SURVIVAL_LIBS_AVAILABLE:
                c_index = concordance_index_censored(y_test.astype(bool), duration_test, -risk_scores)[0]
            else:
                # フォールバック：相関係数を使用
                from scipy.stats import spearmanr
                c_index = abs(spearmanr(duration_test, risk_scores)[0])
            
            # その他の評価指標
            brier_score = None
            integrated_brier_score = None
            auc = None
            rmse = None
            mae = None
            r2 = None
            
            # 可能な場合、追加の評価指標を計算
            try:
                # 生存確率予測による評価
                time_points = np.array([1.0, 2.0, 3.0, 5.0])  # 1, 2, 3, 5年後
                survival_probs = model.predict_survival_probability(X_test, time_points)
                
                # MSE, MAE, R²（生存時間予測として評価）
                predicted_times = -1.0 / (risk_scores + 1e-8)  # リスクスコアから生存時間を逆算
                rmse = np.sqrt(mean_squared_error(duration_test, predicted_times))
                mae = mean_absolute_error(duration_test, predicted_times)
                r2 = r2_score(duration_test, predicted_times)
                
            except Exception:
                pass
            
            return ModelPerformance(
                model_name=model.model_name,
                concordance_index=c_index,
                brier_score=brier_score,
                integrated_brier_score=integrated_brier_score,
                auc=auc,
                rmse=rmse,
                mae=mae,
                r2=r2
            )
            
        except Exception as e:
            print(f"Error evaluating model {model.model_name}: {str(e)}")
            return ModelPerformance(
                model_name=model.model_name,
                concordance_index=0.5,  # ランダム予測レベル
                brier_score=None,
                integrated_brier_score=None,
                auc=None,
                rmse=None,
                mae=None,
                r2=None
            )
    
    def predict_company_survival(self, company_data: pd.DataFrame, 
                                time_points: np.ndarray,
                                model_name: Optional[str] = None) -> List[SurvivalPrediction]:
        """企業の生存確率予測"""
        if model_name and model_name in self.models:
            models_to_use = {model_name: self.models[model_name]}
        else:
            models_to_use = self.models
        
        predictions = []
        
        for idx, row in company_data.iterrows():
            company_id = str(row.get('company_id', f'company_{idx}'))
            X_company = row.drop('company_id', errors='ignore').to_frame().T
            
            ensemble_survival_probs = []
            ensemble_risk_scores = []
            feature_importance_ensemble = {}
            
            for model_name, model in models_to_use.items():
                if not model.is_fitted:
                    continue
                
                try:
                    # 生存確率予測
                    survival_probs = model.predict_survival_probability(X_company, time_points)
                    ensemble_survival_probs.append(survival_probs[0])
                    
                    # リスクスコア予測
                    risk_score = model.predict_risk_score(X_company)[0]
                    ensemble_risk_scores.append(risk_score)
                    
                    # 特徴量重要度（利用可能な場合）
                    importance = model.get_feature_importance()
                    if importance:
                        for feature, imp in importance.items():
                            if feature in feature_importance_ensemble:
                                feature_importance_ensemble[feature].append(imp)
                            else:
                                feature_importance_ensemble[feature] = [imp]
                
                except Exception as e:
                    print(f"Error predicting for {company_id} with {model_name}: {str(e)}")
                    continue
            
            if ensemble_survival_probs:
                # アンサンブル予測（平均）
                avg_survival_prob = np.mean(ensemble_survival_probs, axis=0)
                avg_risk_score = np.mean(ensemble_risk_scores)
                
                # 予測生存時間（中央値）
                median_survival_time = None
                try:
                    median_idx = np.where(avg_survival_prob <= 0.5)[0]
                    if len(median_idx) > 0:
                        median_survival_time = time_points[median_idx[0]]
                except:
                    pass
                
                # 特徴量重要度の平均
                avg_feature_importance = {
                    feature: np.mean(imps) for feature, imps in feature_importance_ensemble.items()
                }
                
                prediction = SurvivalPrediction(
                    company_id=company_id,
                    survival_probability=float(avg_survival_prob[-1]),  # 最長期間の生存確率
                    risk_score=avg_risk_score,
                    predicted_survival_time=median_survival_time,
                    confidence_interval=None,  # TODO: 信頼区間の計算
                    feature_importance=avg_feature_importance if avg_feature_importance else None
                )
                
                predictions.append(prediction)
        
        return predictions
    
    def get_model_comparison(self) -> pd.DataFrame:
        """モデル比較結果の取得"""
        if not self.model_performances:
            return pd.DataFrame()
        
        comparison_data = []
        for model_name, performance in self.model_performances.items():
            comparison_data.append({
                'Model': model_name,
                'Concordance_Index': performance.concordance_index,
                'RMSE': performance.rmse,
                'MAE': performance.mae,
                'R2': performance.r2
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        return comparison_df.sort_values('Concordance_Index', ascending=False)
    
    def plot_feature_importance(self, top_n: int = 20) -> Figure:
        """特徴量重要度の可視化"""
        if not self.feature_importance_results:
            return None
        
        fig, axes = plt.subplots(len(self.feature_importance_results), 1, 
                                figsize=(12, 6 * len(self.feature_importance_results)))
        
        if len(self.feature_importance_results) == 1:
            axes = [axes]
        
        for idx, (model_name, importance) in enumerate(self.feature_importance_results.items()):
            # 重要度でソートして上位N個を表示
            sorted_features = sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
            
            features, importances = zip(*sorted_features)
            
            ax = axes[idx]
            bars = ax.barh(range(len(features)), importances)
            ax.set_yticks(range(len(features)))
            ax.set_yticklabels(features)
            ax.set_xlabel('Feature Importance')
            ax.set_title(f'{model_name} - Top {top_n} Important Features')
            ax.invert_yaxis()
            
            # カラーマップ適用
            colors = plt.cm.RdYlBu(np.linspace(0, 1, len(bars)))
            for bar, color in zip(bars, colors):
                bar.set_color(color)
        
        plt.tight_layout()
        return fig
    
    def generate_survival_report(self) -> Dict[str, Any]:
        """生存分析レポートの生成"""
        report = {
            'model_performances': self.model_performances,
            'feature_importance': self.feature_importance_results,
            'model_comparison': self.get_model_comparison(),
            'best_model': None
        }
        
        # 最高性能モデルの特定
        if self.model_performances:
            best_model_name = max(self.model_performances.keys(), 
                                key=lambda x: self.model_performances[x].concordance_index)
            report['best_model'] = {
                'name': best_model_name,
                'performance': self.model_performances[best_model_name]
            }
        
        return report


def create_default_ml_survival_analyzer(random_state: int = 42) -> MachineLearninSurvivalAnalyzer:
    """デフォルトの機械学習生存分析器を作成"""
    analyzer = MachineLearninSurvivalAnalyzer(random_state=random_state)
    
    # デフォルトモデルの追加
    analyzer.add_model(RandomForestSurvival(n_estimators=100, random_state=random_state))
    analyzer.add_model(GradientBoostingSurvival(n_estimators=100, random_state=random_state))
    
    # PyTorchが利用可能な場合のみDeepSurvivalを追加
    if TORCH_AVAILABLE:
        analyzer.add_model(DeepSurvival(hidden_dims=[128, 64, 32], epochs=50, random_state=random_state))
    
    return analyzer


class XGBoostSurvival(BaseSurvivalMLModel):
    """XGBoost生存分析モデル"""
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 6, 
                    learning_rate: float = 0.1, random_state: int = 42):
        super().__init__("XGBoost Survival")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.model = None
    
    def fit(self, X: pd.DataFrame, y: np.ndarray, duration: np.ndarray) -> None:
        """モデルの学習"""
        self.feature_names = X.columns.tolist()
        
        # データの標準化
        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=self.feature_names)
        
        # XGBoostモデルの初期化
        self.model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            objective='survival:cox',  # Cox回帰目的関数
            eval_metric='cox-nloglik'
        )
        
        # 生存分析用のターゲット作成（負の持続時間でイベント発生を表現）
        y_survival = np.where(y == 1, -duration, duration)
        
        try:
            self.model.fit(X_scaled, y_survival)
        except Exception:
            # フォールバック：通常の回帰として学習
            self.model = xgb.XGBRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                random_state=self.random_state
            )
            self.model.fit(X_scaled, duration)
        
        self.is_fitted = True
    
    def predict_survival_probability(self, X: pd.DataFrame, time_points: np.ndarray) -> np.ndarray:
        """生存確率の予測"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_scaled = self.scaler.transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=self.feature_names)
        
        # リスクスコアまたは予測時間を取得
        predictions = self.model.predict(X_scaled)
        
        # 生存確率の計算（指数分布を仮定）
        survival_probs = []
        for pred in predictions:
            if hasattr(self.model, 'objective') and 'cox' in str(self.model.objective):
                # Cox回帰の場合：リスクスコア
                baseline_hazard = 0.1  # 簡略化
                hazard = baseline_hazard * np.exp(pred)
                survival_prob = np.exp(-hazard * time_points)
            else:
                # 回帰の場合：予測生存時間
                survival_prob = np.exp(-time_points / (abs(pred) + 1e-8))
            survival_probs.append(survival_prob)
        
        return np.array(survival_probs)
    
    def predict_risk_score(self, X: pd.DataFrame) -> np.ndarray:
        """リスクスコアの予測"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_scaled = self.scaler.transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=self.feature_names)
        
        predictions = self.model.predict(X_scaled)
        
        if hasattr(self.model, 'objective') and 'cox' in str(self.model.objective):
            return predictions  # 既にリスクスコア
        else:
            return 1.0 / (abs(predictions) + 1e-8)  # 予測時間の逆数


class LightGBMSurvival(BaseSurvivalMLModel):
    """LightGBM生存分析モデル"""
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 6,
                    learning_rate: float = 0.1, random_state: int = 42):
        super().__init__("LightGBM Survival")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.model = None
    
    def fit(self, X: pd.DataFrame, y: np.ndarray, duration: np.ndarray) -> None:
        """モデルの学習"""
        self.feature_names = X.columns.tolist()
        
        # データの標準化
        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=self.feature_names)
        
        # LightGBMモデルの初期化
        self.model = lgb.LGBMRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            verbose=-1
        )
        
        # 生存時間を直接予測する回帰問題として学習
        # イベント発生時は実際の時間、打ち切りの場合は観測時間を使用
        target = np.where(y == 1, duration, duration * 1.5)  # 打ち切りデータに重み付け
        
        self.model.fit(X_scaled, target)
        self.is_fitted = True
    
    def predict_survival_probability(self, X: pd.DataFrame, time_points: np.ndarray) -> np.ndarray:
        """生存確率の予測"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_scaled = self.scaler.transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=self.feature_names)
        
        predicted_times = self.model.predict(X_scaled)
        
        # 指数分布を仮定した生存確率計算
        survival_probs = []
        for pred_time in predicted_times:
            survival_prob = np.exp(-time_points / (pred_time + 1e-8))
            survival_probs.append(survival_prob)
        
        return np.array(survival_probs)
    
    def predict_risk_score(self, X: pd.DataFrame) -> np.ndarray:
        """リスクスコアの予測"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_scaled = self.scaler.transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=self.feature_names)
        
        predicted_times = self.model.predict(X_scaled)
        return 1.0 / (predicted_times + 1e-8)  # 予測時間の逆数をリスクスコアとする


class EnsembleSurvival(BaseSurvivalMLModel):
    """アンサンブル生存分析モデル"""
    
    def __init__(self, models: List[BaseSurvivalMLModel], weights: Optional[List[float]] = None):
        super().__init__("Ensemble Survival")
        self.base_models = models
        self.weights = weights or [1.0] * len(models)
        self.weights = np.array(self.weights) / np.sum(self.weights)  # 正規化
    
    def fit(self, X: pd.DataFrame, y: np.ndarray, duration: np.ndarray) -> None:
        """全ベースモデルの学習"""
        self.feature_names = X.columns.tolist()
        
        for model in self.base_models:
            print(f"Training base model: {model.model_name}")
            try:
                model.fit(X, y, duration)
            except Exception as e:
                print(f"Error training {model.model_name}: {str(e)}")
        
        self.is_fitted = True
    
    def predict_survival_probability(self, X: pd.DataFrame, time_points: np.ndarray) -> np.ndarray:
        """アンサンブル生存確率予測"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        ensemble_predictions = []
        
        for model, weight in zip(self.base_models, self.weights):
            if not model.is_fitted:
                continue
            
            try:
                pred = model.predict_survival_probability(X, time_points)
                ensemble_predictions.append(pred * weight)
            except Exception as e:
                print(f"Error predicting with {model.model_name}: {str(e)}")
                continue
        
        if ensemble_predictions:
            return np.sum(ensemble_predictions, axis=0)
        else:
            # フォールバック：ランダム予測
            return np.random.random((len(X), len(time_points))) * 0.5 + 0.25
    
    def predict_risk_score(self, X: pd.DataFrame) -> np.ndarray:
        """アンサンブルリスクスコア予測"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        ensemble_scores = []
        
        for model, weight in zip(self.base_models, self.weights):
            if not model.is_fitted:
                continue
            
            try:
                score = model.predict_risk_score(X)
                ensemble_scores.append(score * weight)
            except Exception as e:
                print(f"Error predicting risk with {model.model_name}: {str(e)}")
                continue
        
        if ensemble_scores:
            return np.sum(ensemble_scores, axis=0)
        else:
            return np.random.random(len(X))
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """アンサンブル特徴量重要度"""
        ensemble_importance = {}
        
        for model, weight in zip(self.base_models, self.weights):
            importance = model.get_feature_importance()
            if importance:
                for feature, imp in importance.items():
                    if feature in ensemble_importance:
                        ensemble_importance[feature] += imp * weight
                    else:
                        ensemble_importance[feature] = imp * weight
        
        return ensemble_importance if ensemble_importance else None


class SurvivalModelValidator:
    """生存分析モデルのクロスバリデーション"""
    
    def __init__(self, n_splits: int = 5, random_state: int = 42):
        self.n_splits = n_splits
        self.random_state = random_state
        self.cv_results = {}
    
    def cross_validate_model(self, model: BaseSurvivalMLModel, X: pd.DataFrame, 
                            y: np.ndarray, duration: np.ndarray) -> Dict[str, float]:
        """クロスバリデーション実行"""
        from sklearn.model_selection import StratifiedKFold
        
        # 層化K分割クロスバリデーション
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        
        cv_scores = {
            'concordance_index': [],
            'rmse': [],
            'mae': []
        }
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            print(f"Fold {fold + 1}/{self.n_splits}")
            
            # データ分割
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            duration_train, duration_val = duration[train_idx], duration[val_idx]
            
            try:
                # モデルのコピーを作成して学習
                model_copy = type(model)(**model.__dict__)
                model_copy.fit(X_train, y_train, duration_train)
                
                # 予測と評価
                risk_scores = model_copy.predict_risk_score(X_val)
                
                # C-index計算
                if SURVIVAL_LIBS_AVAILABLE:
                    c_index = concordance_index_censored(y_val.astype(bool), duration_val, -risk_scores)[0]
                else:
                    from scipy.stats import spearmanr
                    c_index = abs(spearmanr(duration_val, risk_scores)[0])
                
                cv_scores['concordance_index'].append(c_index)
                
                # RMSE, MAE計算（生存時間予測として）
                predicted_times = 1.0 / (risk_scores + 1e-8)
                rmse = np.sqrt(mean_squared_error(duration_val, predicted_times))
                mae = mean_absolute_error(duration_val, predicted_times)
                
                cv_scores['rmse'].append(rmse)
                cv_scores['mae'].append(mae)
                
            except Exception as e:
                print(f"Error in fold {fold + 1}: {str(e)}")
                continue
        
        # 平均スコア計算
        avg_scores = {}
        for metric, scores in cv_scores.items():
            if scores:
                avg_scores[f'{metric}_mean'] = np.mean(scores)
                avg_scores[f'{metric}_std'] = np.std(scores)
            else:
                avg_scores[f'{metric}_mean'] = 0.0
                avg_scores[f'{metric}_std'] = 0.0
        
        self.cv_results[model.model_name] = avg_scores
        return avg_scores
    
    def get_cv_results(self) -> pd.DataFrame:
        """クロスバリデーション結果の取得"""
        if not self.cv_results:
            return pd.DataFrame()
        
        results_df = pd.DataFrame(self.cv_results).T
        return results_df.sort_values('concordance_index_mean', ascending=False)


class SurvivalFeatureSelector:
    """生存分析用特徴選択"""
    
    def __init__(self, method: str = 'univariate', threshold: float = 0.05):
        self.method = method
        self.threshold = threshold
        self.selected_features = None
        self.feature_scores = {}
    
    def fit_transform(self, X: pd.DataFrame, y: np.ndarray, duration: np.ndarray) -> pd.DataFrame:
        """特徴選択の実行"""
        if self.method == 'univariate':
            return self._univariate_selection(X, y, duration)
        elif self.method == 'recursive':
            return self._recursive_selection(X, y, duration)
        elif self.method == 'lasso':
            return self._lasso_selection(X, y, duration)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _univariate_selection(self, X: pd.DataFrame, y: np.ndarray, duration: np.ndarray) -> pd.DataFrame:
        """単変量生存分析による特徴選択"""
        if not SURVIVAL_LIBS_AVAILABLE:
            print("Survival libraries not available. Using correlation-based selection.")
            return self._correlation_selection(X, duration)
        
        selected_features = []
        
        for feature in X.columns:
            try:
                # 単変量Cox回帰
                feature_data = pd.DataFrame({
                    'T': duration,
                    'E': y,
                    feature: X[feature]
                })
                
                cph = CoxPHFitter()
                cph.fit(feature_data, duration_col='T', event_col='E')
                
                # p値による選択
                p_value = cph.summary.loc[feature, 'p']
                self.feature_scores[feature] = p_value
                
                if p_value < self.threshold:
                    selected_features.append(feature)
                    
            except Exception as e:
                print(f"Error processing feature {feature}: {str(e)}")
                continue
        
        self.selected_features = selected_features
        return X[selected_features] if selected_features else X
    
    def _correlation_selection(self, X: pd.DataFrame, duration: np.ndarray) -> pd.DataFrame:
        """相関による特徴選択（フォールバック）"""
        from scipy.stats import spearmanr
        
        correlations = {}
        for feature in X.columns:
            try:
                corr, p_val = spearmanr(X[feature], duration)
                correlations[feature] = abs(corr)
                self.feature_scores[feature] = p_val
            except:
                correlations[feature] = 0.0
                self.feature_scores[feature] = 1.0
        
        # 相関の高い特徴を選択
        selected_features = [
            feature for feature, corr in correlations.items() 
            if corr > 0.1 and self.feature_scores[feature] < self.threshold
        ]
        
        self.selected_features = selected_features
        return X[selected_features] if selected_features else X
    
    def _recursive_selection(self, X: pd.DataFrame, y: np.ndarray, duration: np.ndarray) -> pd.DataFrame:
        """再帰的特徴選択"""
        from sklearn.feature_selection import RFE
        
        # ベースモデルとしてRandom Forestを使用
        base_model = RandomForestRegressor(n_estimators=50, random_state=42)
        
        # RFE実行
        n_features = max(5, len(X.columns) // 4)  # 1/4の特徴量を選択
        rfe = RFE(estimator=base_model, n_features_to_select=n_features)
        
        rfe.fit(X, duration)
        
        selected_features = X.columns[rfe.support_].tolist()
        self.selected_features = selected_features
        
        return X[selected_features]
    
    def _lasso_selection(self, X: pd.DataFrame, y: np.ndarray, duration: np.ndarray) -> pd.DataFrame:
        """Lasso回帰による特徴選択"""
        from sklearn.linear_model import LassoCV
        
        # Lasso回帰で特徴選択
        lasso = LassoCV(cv=5, random_state=42, max_iter=1000)
        lasso.fit(X, duration)
        
        # 係数が0でない特徴を選択
        selected_features = X.columns[lasso.coef_ != 0].tolist()
        self.selected_features = selected_features
        
        # 特徴量スコア（絶対値の係数）
        self.feature_scores = dict(zip(X.columns, np.abs(lasso.coef_)))
        
        return X[selected_features] if selected_features else X


def create_comprehensive_ml_survival_analyzer(random_state: int = 42) -> MachineLearninSurvivalAnalyzer:
    """包括的な機械学習生存分析器を作成"""
    analyzer = MachineLearninSurvivalAnalyzer(random_state=random_state)
    
    # 基本モデルの追加
    analyzer.add_model(RandomForestSurvival(n_estimators=200, random_state=random_state))
    analyzer.add_model(GradientBoostingSurvival(n_estimators=200, random_state=random_state))
    analyzer.add_model(XGBoostSurvival(n_estimators=200, random_state=random_state))
    analyzer.add_model(LightGBMSurvival(n_estimators=200, random_state=random_state))
    
    # ディープラーニングモデル（PyTorchが利用可能な場合）
    if TORCH_AVAILABLE:
        analyzer.add_model(DeepSurvival(
            hidden_dims=[256, 128, 64, 32], 
            epochs=100, 
            batch_size=64,
            random_state=random_state
        ))
    
    # アンサンブルモデル
    base_models = [
        RandomForestSurvival(n_estimators=100, random_state=random_state),
        GradientBoostingSurvival(n_estimators=100, random_state=random_state),
        XGBoostSurvival(n_estimators=100, random_state=random_state)
    ]
    
    analyzer.add_model(EnsembleSurvival(base_models, weights=[0.3, 0.3, 0.4]))
    
    return analyzer


# 使用例とテスト用の関数
def demo_ml_survival_analysis():
    """機械学習生存分析のデモンストレーション"""
    print("=== A2AI Machine Learning Survival Analysis Demo ===")
    
    # サンプルデータの生成
    np.random.seed(42)
    n_companies = 1000
    n_features = 50
    
    # 特徴量データ生成
    X = pd.DataFrame(
        np.random.randn(n_companies, n_features),
        columns=[f'feature_{i:02d}' for i in range(n_features)]
    )
    
    # 生存時間とイベントの生成
    # 特徴量に基づいた生存時間（指数分布）
    risk_factors = X.iloc[:, :5].mean(axis=1)  # 最初の5つの特徴量を使用
    true_survival_times = np.random.exponential(5 - risk_factors)  # リスクが高いほど短命
    
    # 観測期間（10年）での打ち切り
    observation_time = 10.0
    observed_times = np.minimum(true_survival_times, observation_time)
    events = (true_survival_times <= observation_time).astype(int)
    
    print(f"Generated data: {n_companies} companies, {n_features} features")
    print(f"Event rate: {events.mean():.2%}")
    print(f"Median survival time: {np.median(observed_times):.2f}")
    
    # 分析器の作成
    analyzer = create_comprehensive_ml_survival_analyzer(random_state=42)
    
    # 全モデルの学習と評価
    analyzer.fit_all_models(X, events, observed_times, test_size=0.2)
    
    # 結果の表示
    print("\n=== Model Performance Comparison ===")
    comparison = analyzer.get_model_comparison()
    print(comparison)
    
    # 予測例
    print("\n=== Survival Predictions for Sample Companies ===")
    sample_companies = X.head(5).copy()
    sample_companies['company_id'] = [f'Company_{i+1}' for i in range(5)]
    
    time_points = np.array([1, 2, 3, 5, 10])
    predictions = analyzer.predict_company_survival(sample_companies, time_points)
    
    for pred in predictions:
        print(f"\n{pred.company_id}:")
        print(f"  Risk Score: {pred.risk_score:.4f}")
        print(f"  10-year Survival Probability: {pred.survival_probability:.4f}")
        if pred.predicted_survival_time:
            print(f"  Predicted Median Survival: {pred.predicted_survival_time:.2f} years")
    
    # 特徴量重要度の可視化
    importance_fig = analyzer.plot_feature_importance(top_n=15)
    if importance_fig:
        print("\nFeature importance plot generated.")
    
    # 生存分析レポートの生成
    report = analyzer.generate_survival_report()
    print(f"\n=== Best Model ===")
    if report['best_model']:
        best = report['best_model']
        print(f"Model: {best['name']}")
        print(f"Concordance Index: {best['performance'].concordance_index:.4f}")
    
    print("\n=== ML Survival Analysis Demo Completed ===")


if __name__ == "__main__":
    demo_ml_survival_analysis()