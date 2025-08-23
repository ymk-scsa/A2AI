"""
A2AI Ensemble Models
企業ライフサイクル分析に特化したアンサンブル学習モデル群

このモジュールは、企業の生存・消滅・新設を含む完全なライフサイクル分析のための
アンサンブル学習モデルを実装します。
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from abc import ABC, abstractmethod
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier,
    ExtraTreesRegressor, ExtraTreesClassifier,
    AdaBoostRegressor, AdaBoostClassifier,
    VotingRegressor, VotingClassifier,
    BaggingRegressor, BaggingClassifier
)
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
import xgboost as xgb
import lightgbm as lgb
from scipy import stats
import logging
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseEnsembleModel(ABC):
    """
    アンサンブルモデルの基底クラス
    企業ライフサイクル分析に特化した共通機能を提供
    """
    
    def __init__(self, 
                    target_metric: str,
                    market_category: str = 'all',
                    lifecycle_stage: str = 'all',
                    use_survival_features: bool = True,
                    random_state: int = 42):
        """
        Parameters:
        -----------
        target_metric : str
            予測対象の評価指標 ('sales', 'growth_rate', 'operating_margin', etc.)
        market_category : str
            市場カテゴリ ('high_share', 'declining', 'lost', 'all')
        lifecycle_stage : str
            ライフサイクル段階 ('startup', 'growth', 'maturity', 'decline', 'all')
        use_survival_features : bool
            生存分析特徴量を使用するかどうか
        random_state : int
            乱数シード
        """
        self.target_metric = target_metric
        self.market_category = market_category
        self.lifecycle_stage = lifecycle_stage
        self.use_survival_features = use_survival_features
        self.random_state = random_state
        
        self.models = {}
        self.feature_importance = {}
        self.is_fitted = False
        self.scaler = StandardScaler()
        self.feature_names = None
        self.market_category_weights = None
        
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series, 
            sample_weight: Optional[np.ndarray] = None) -> 'BaseEnsembleModel':
        """モデルの学習"""
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """予測実行"""
        pass
    
    def _prepare_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        特徴量前処理
        企業ライフサイクル分析に特化した特徴量エンジニアリング
        """
        X_processed = X.copy()
        
        # 市場カテゴリフィルタリング
        if self.market_category != 'all' and 'market_category' in X_processed.columns:
            mask = X_processed['market_category'] == self.market_category
            X_processed = X_processed[mask]
        
        # ライフサイクル段階フィルタリング
        if self.lifecycle_stage != 'all' and 'lifecycle_stage' in X_processed.columns:
            mask = X_processed['lifecycle_stage'] == self.lifecycle_stage
            X_processed = X_processed[mask]
        
        # 生存分析特徴量の処理
        if not self.use_survival_features:
            survival_cols = [col for col in X_processed.columns if 'survival' in col.lower()]
            X_processed = X_processed.drop(columns=survival_cols, errors='ignore')
        
        # 時系列特徴量の生成
        X_processed = self._generate_time_features(X_processed)
        
        # 交互作用特徴量の生成
        X_processed = self._generate_interaction_features(X_processed)
        
        return X_processed
    
    def _generate_time_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """時系列特徴量生成"""
        X_processed = X.copy()
        
        # 企業年齢関連特徴量
        if 'company_age' in X_processed.columns:
            X_processed['company_age_log'] = np.log1p(X_processed['company_age'])
            X_processed['company_age_squared'] = X_processed['company_age'] ** 2
            X_processed['company_maturity'] = (X_processed['company_age'] > 20).astype(int)
        
        # トレンド特徴量（過去3年、5年の変化率）
        trend_cols = [col for col in X_processed.columns if '_growth_rate' in col]
        for col in trend_cols:
            if f'{col}_3yr_trend' in X_processed.columns:
                X_processed[f'{col}_acceleration'] = (
                    X_processed[f'{col}_3yr_trend'] - X_processed[col]
                )
        
        return X_processed
    
    def _generate_interaction_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """交互作用特徴量生成"""
        X_processed = X.copy()
        
        # 市場カテゴリと企業規模の交互作用
        if 'market_category' in X_processed.columns and 'total_assets' in X_processed.columns:
            market_dummies = pd.get_dummies(X_processed['market_category'], prefix='market')
            for col in market_dummies.columns:
                X_processed[f'{col}_x_assets'] = (
                    market_dummies[col] * X_processed['total_assets']
                )
        
        # 研究開発費と利益率の交互作用
        if 'rd_ratio' in X_processed.columns and 'operating_margin' in X_processed.columns:
            X_processed['rd_x_margin'] = (
                X_processed['rd_ratio'] * X_processed['operating_margin']
            )
        
        return X_processed
    
    def _calculate_market_weights(self, X: pd.DataFrame) -> np.ndarray:
        """
        市場カテゴリ別重み計算
        失失市場の企業データを重要視する重み付け
        """
        if 'market_category' not in X.columns:
            return np.ones(len(X))
        
        weights = np.ones(len(X))
        
        # 市場カテゴリ別重み設定
        market_weights = {
            'high_share': 1.0,    # 高シェア市場：標準重み
            'declining': 1.2,     # シェア低下市場：やや高い重み
            'lost': 1.5           # 失失市場：高い重み（失敗パターンを重要視）
        }
        
        for category, weight in market_weights.items():
            mask = X['market_category'] == category
            weights[mask] = weight
        
        return weights
    
    def get_feature_importance(self) -> Dict[str, float]:
        """特徴量重要度取得"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        return self.feature_importance
    
    def save_model(self, filepath: str):
        """モデル保存"""
        save_data = {
            'models': self.models,
            'feature_importance': self.feature_importance,
            'feature_names': self.feature_names,
            'scaler': self.scaler,
            'target_metric': self.target_metric,
            'market_category': self.market_category,
            'lifecycle_stage': self.lifecycle_stage,
            'use_survival_features': self.use_survival_features,
            'is_fitted': self.is_fitted
        }
        joblib.dump(save_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """モデル読み込み"""
        save_data = joblib.load(filepath)
        self.models = save_data['models']
        self.feature_importance = save_data['feature_importance']
        self.feature_names = save_data['feature_names']
        self.scaler = save_data['scaler']
        self.target_metric = save_data['target_metric']
        self.market_category = save_data['market_category']
        self.lifecycle_stage = save_data['lifecycle_stage']
        self.use_survival_features = save_data['use_survival_features']
        self.is_fitted = save_data['is_fitted']
        logger.info(f"Model loaded from {filepath}")


class FinancialEnsembleRegressor(BaseEnsembleModel, BaseEstimator, RegressorMixin):
    """
    財務指標予測用アンサンブル回帰モデル
    売上高、利益率、ROE等の連続値予測に使用
    """
    
    def __init__(self, 
                    target_metric: str,
                    market_category: str = 'all',
                    lifecycle_stage: str = 'all',
                    use_survival_features: bool = True,
                    ensemble_method: str = 'stacking',
                    n_estimators: int = 100,
                    random_state: int = 42):
        """
        Parameters:
        -----------
        ensemble_method : str
            アンサンブル手法 ('voting', 'stacking', 'blending')
        n_estimators : int
            各基底学習器の推定器数
        """
        super().__init__(target_metric, market_category, lifecycle_stage, 
                        use_survival_features, random_state)
        
        self.ensemble_method = ensemble_method
        self.n_estimators = n_estimators
        
        # 基底学習器の定義
        self.base_models = {
            'random_forest': RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=random_state,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=n_estimators,
                learning_rate=0.1,
                max_depth=6,
                random_state=random_state
            ),
            'extra_trees': ExtraTreesRegressor(
                n_estimators=n_estimators,
                max_depth=10,
                min_samples_split=5,
                random_state=random_state,
                n_jobs=-1
            ),
            'xgboost': xgb.XGBRegressor(
                n_estimators=n_estimators,
                learning_rate=0.1,
                max_depth=6,
                random_state=random_state,
                n_jobs=-1
            ),
            'lightgbm': lgb.LGBMRegressor(
                n_estimators=n_estimators,
                learning_rate=0.1,
                max_depth=6,
                random_state=random_state,
                n_jobs=-1,
                verbosity=-1
            )
        }
    
    def fit(self, X: pd.DataFrame, y: pd.Series, 
            sample_weight: Optional[np.ndarray] = None) -> 'FinancialEnsembleRegressor':
        """
        アンサンブルモデルの学習
        """
        logger.info(f"Training ensemble regressor for {self.target_metric}")
        
        # 特徴量前処理
        X_processed = self._prepare_features(X)
        self.feature_names = X_processed.columns.tolist()
        
        # 標準化
        X_scaled = self.scaler.fit_transform(X_processed)
        X_scaled = pd.DataFrame(X_scaled, columns=self.feature_names)
        
        # 市場カテゴリ別重み計算
        if sample_weight is None:
            sample_weight = self._calculate_market_weights(X)
        
        # アンサンブル手法に応じた学習
        if self.ensemble_method == 'voting':
            self._fit_voting_ensemble(X_scaled, y, sample_weight)
        elif self.ensemble_method == 'stacking':
            self._fit_stacking_ensemble(X_scaled, y, sample_weight)
        elif self.ensemble_method == 'blending':
            self._fit_blending_ensemble(X_scaled, y, sample_weight)
        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")
        
        # 特徴量重要度計算
        self._calculate_feature_importance(X_scaled, y)
        
        self.is_fitted = True
        logger.info("Ensemble regressor training completed")
        
        return self
    
    def _fit_voting_ensemble(self, X: pd.DataFrame, y: pd.Series, sample_weight: np.ndarray):
        """投票アンサンブルの学習"""
        # 各基底学習器の学習
        trained_models = []
        for name, model in self.base_models.items():
            try:
                model.fit(X, y, sample_weight=sample_weight)
                trained_models.append((name, model))
                logger.info(f"Trained {name} successfully")
            except Exception as e:
                logger.warning(f"Failed to train {name}: {e}")
        
        # 投票回帰器の構築
        self.models['voting'] = VotingRegressor(
            estimators=trained_models,
            n_jobs=-1
        )
        self.models['voting'].fit(X, y, sample_weight=sample_weight)
    
    def _fit_stacking_ensemble(self, X: pd.DataFrame, y: pd.Series, sample_weight: np.ndarray):
        """スタッキングアンサンブルの学習"""
        # レベル1モデルの学習とクロスバリデーション予測
        n_folds = 5
        tscv = TimeSeriesSplit(n_splits=n_folds)
        
        level1_features = np.zeros((len(X), len(self.base_models)))
        
        for i, (name, model) in enumerate(self.base_models.items()):
            fold_predictions = np.zeros(len(X))
            
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                weight_train = sample_weight[train_idx]
                
                try:
                    model_copy = self._clone_model(model)
                    model_copy.fit(X_train, y_train, sample_weight=weight_train)
                    fold_predictions[val_idx] = model_copy.predict(X_val)
                except Exception as e:
                    logger.warning(f"Fold training failed for {name}: {e}")
                    fold_predictions[val_idx] = y_val.mean()  # フォールバック
            
            level1_features[:, i] = fold_predictions
            
            # 全データでモデル再学習
            try:
                model.fit(X, y, sample_weight=sample_weight)
                self.models[f'level1_{name}'] = model
            except Exception as e:
                logger.warning(f"Final training failed for {name}: {e}")
        
        # レベル2モデル（メタ学習器）の学習
        self.models['meta_learner'] = GradientBoostingRegressor(
            n_estimators=50,
            learning_rate=0.1,
            random_state=self.random_state
        )
        self.models['meta_learner'].fit(level1_features, y, sample_weight=sample_weight)
    
    def _fit_blending_ensemble(self, X: pd.DataFrame, y: pd.Series, sample_weight: np.ndarray):
        """ブレンディングアンサンブルの学習"""
        # データを学習用とブレンド用に分割（80:20）
        split_idx = int(0.8 * len(X))
        
        X_train, X_blend = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_blend = y.iloc[:split_idx], y.iloc[split_idx:]
        weight_train = sample_weight[:split_idx]
        
        # レベル1モデルの学習
        blend_predictions = np.zeros((len(X_blend), len(self.base_models)))
        
        for i, (name, model) in enumerate(self.base_models.items()):
            try:
                model.fit(X_train, y_train, sample_weight=weight_train)
                blend_predictions[:, i] = model.predict(X_blend)
                self.models[f'level1_{name}'] = model
            except Exception as e:
                logger.warning(f"Blend training failed for {name}: {e}")
                blend_predictions[:, i] = y_blend.mean()
        
        # ブレンド重み学習
        from sklearn.linear_model import LinearRegression
        self.models['blender'] = LinearRegression()
        self.models['blender'].fit(blend_predictions, y_blend)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """予測実行"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # 特徴量前処理
        X_processed = self._prepare_features(X)
        X_scaled = self.scaler.transform(X_processed)
        X_scaled = pd.DataFrame(X_scaled, columns=self.feature_names)
        
        # アンサンブル手法に応じた予測
        if self.ensemble_method == 'voting':
            return self.models['voting'].predict(X_scaled)
        
        elif self.ensemble_method == 'stacking':
            # レベル1予測
            level1_preds = np.zeros((len(X_scaled), len(self.base_models)))
            for i, (name, _) in enumerate(self.base_models.items()):
                model = self.models.get(f'level1_{name}')
                if model:
                    level1_preds[:, i] = model.predict(X_scaled)
                else:
                    level1_preds[:, i] = 0  # フォールバック
            
            # レベル2予測
            return self.models['meta_learner'].predict(level1_preds)
        
        elif self.ensemble_method == 'blending':
            # レベル1予測
            blend_preds = np.zeros((len(X_scaled), len(self.base_models)))
            for i, (name, _) in enumerate(self.base_models.items()):
                model = self.models.get(f'level1_{name}')
                if model:
                    blend_preds[:, i] = model.predict(X_scaled)
                else:
                    blend_preds[:, i] = 0
            
            # ブレンド予測
            return self.models['blender'].predict(blend_preds)
        
        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")
    
    def _calculate_feature_importance(self, X: pd.DataFrame, y: pd.Series):
        """特徴量重要度計算"""
        importance_dict = {}
        
        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importance_dict[name] = model.feature_importances_
        
        # 平均重要度計算
        if importance_dict:
            importance_array = np.array(list(importance_dict.values()))
            avg_importance = np.mean(importance_array, axis=0)
            
            self.feature_importance = dict(zip(self.feature_names, avg_importance))
        else:
            # フォールバック：相関係数による重要度
            correlations = X.corrwith(y).abs()
            self.feature_importance = correlations.to_dict()
    
    def _clone_model(self, model):
        """モデルのクローン作成"""
        from copy import deepcopy
        return deepcopy(model)
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """モデル評価"""
        predictions = self.predict(X)
        
        return {
            'mse': mean_squared_error(y, predictions),
            'mae': mean_absolute_error(y, predictions),
            'r2': r2_score(y, predictions),
            'rmse': np.sqrt(mean_squared_error(y, predictions))
        }


class SurvivalEnsembleClassifier(BaseEnsembleModel, BaseEstimator, ClassifierMixin):
    """
    企業生存予測用アンサンブル分類モデル
    企業の生存・消滅予測に特化
    """
    
    def __init__(self, 
                    market_category: str = 'all',
                    lifecycle_stage: str = 'all',
                    prediction_horizon: int = 5,
                    n_estimators: int = 100,
                    random_state: int = 42):
        """
        Parameters:
        -----------
        prediction_horizon : int
            予測期間（年）
        """
        super().__init__('survival', market_category, lifecycle_stage, True, random_state)
        
        self.prediction_horizon = prediction_horizon
        self.n_estimators = n_estimators
        
        # 生存予測に特化した基底学習器
        self.base_models = {
            'survival_rf': RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=5,
                class_weight='balanced',
                random_state=random_state,
                n_jobs=-1
            ),
            'survival_gb': GradientBoostingClassifier(
                n_estimators=n_estimators,
                learning_rate=0.1,
                max_depth=8,
                random_state=random_state
            ),
            'survival_xgb': xgb.XGBClassifier(
                n_estimators=n_estimators,
                learning_rate=0.1,
                max_depth=8,
                random_state=random_state,
                n_jobs=-1,
                eval_metric='logloss'
            )
        }
    
    def fit(self, X: pd.DataFrame, y: pd.Series, 
            sample_weight: Optional[np.ndarray] = None) -> 'SurvivalEnsembleClassifier':
        """生存分類モデルの学習"""
        logger.info(f"Training survival ensemble classifier")
        
        # 特徴量前処理
        X_processed = self._prepare_features(X)
        self.feature_names = X_processed.columns.tolist()
        
        # 標準化
        X_scaled = self.scaler.fit_transform(X_processed)
        X_scaled = pd.DataFrame(X_scaled, columns=self.feature_names)
        
        # 消滅企業により高い重みを設定
        if sample_weight is None:
            # 消滅企業（y=0）により高い重みを設定
            sample_weight = np.where(y == 0, 2.0, 1.0)
        
        # 各基底学習器の学習
        trained_models = []
        for name, model in self.base_models.items():
            try:
                model.fit(X_scaled, y, sample_weight=sample_weight)
                trained_models.append((name, model))
                self.models[name] = model
                logger.info(f"Trained {name} successfully")
            except Exception as e:
                logger.warning(f"Failed to train {name}: {e}")
        
        # 投票分類器の構築
        if trained_models:
            self.models['voting'] = VotingClassifier(
                estimators=trained_models,
                voting='soft',
                n_jobs=-1
            )
            self.models['voting'].fit(X_scaled, y, sample_weight=sample_weight)
        
        # 特徴量重要度計算
        self._calculate_survival_feature_importance()
        
        self.is_fitted = True
        logger.info("Survival ensemble classifier training completed")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """生存確率予測"""
        return self.predict_proba(X)[:, 1]  # 生存確率を返す
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """生存確率予測（詳細）"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # 特徴量前処理
        X_processed = self._prepare_features(X)
        X_scaled = self.scaler.transform(X_processed)
        X_scaled = pd.DataFrame(X_scaled, columns=self.feature_names)
        
        # 投票分類器による予測
        if 'voting' in self.models:
            return self.models['voting'].predict_proba(X_scaled)
        else:
            # フォールバック：最初の利用可能なモデル
            for model in self.models.values():
                if hasattr(model, 'predict_proba'):
                    return model.predict_proba(X_scaled)
        
        raise RuntimeError("No trained model available for prediction")
    
    def _calculate_survival_feature_importance(self):
        """生存分析特化の特徴量重要度計算"""
        importance_dict = {}
        
        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_') and name != 'voting':
                importance_dict[name] = model.feature_importances_
        
        if importance_dict:
            # 各モデルの重要度を平均
            importance_array = np.array(list(importance_dict.values()))
            avg_importance = np.mean(importance_array, axis=0)
            
            self.feature_importance = dict(zip(self.feature_names, avg_importance))
        
        # 生存に特に重要な特徴量を特定
        self._identify_critical_survival_factors()
    
    def _identify_critical_survival_factors(self):
        """生存に重要な要因特定"""
        if not self.feature_importance:
            return
        
        # 重要度上位の特徴量を特定
        sorted_features = sorted(
            self.feature_importance.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        self.critical_survival_factors = {
            'top_10_factors': sorted_features[:10],
            'financial_factors': [(k, v) for k, v in sorted_features 
                                    if any(financial_term in k.lower() 
                                        for financial_term in ['ratio', 'margin', 'rate', 'turnover'])],
            'market_factors': [(k, v) for k, v in sorted_features 
                                if any(market_term in k.lower() 
                                    for market_term in ['market', 'share', 'competition'])],
            'operational_factors': [(k, v) for k, v in sorted_features 
                                    if any(ops_term in k.lower() 
                                            for ops_term in ['employee', 'rd', 'asset', 'investment'])]
        }
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """生存分類モデル評価"""
        predictions = self.predict_proba(X)[:, 1]
        predictions_binary = (predictions > 0.5).astype(int)
        
        return {
            'accuracy': accuracy_score(y, predictions_binary),
            'precision': precision_score(y, predictions_binary, average='weighted'),
            'recall': recall_score(y, predictions_binary, average='weighted'),
            'f1': f1_score(y, predictions_binary, average='weighted'),
            'auc_score': self._calculate_auc(y, predictions)
        }
    
    def _calculate_auc(self, y_true: pd.Series, y_scores: np.ndarray) -> float:
        """AUC計算"""
        try:
            from sklearn.metrics import roc_auc_score
            return roc_auc_score(y_true, y_scores)
        except Exception:
            return 0.5  # ランダム予測のAUC


class MultiTargetEnsembleModel(BaseEnsembleModel):
    """
    複数評価指標同時予測アンサンブルモデル
    売上高、利益率、ROE等を同時に予測
    """
    
    def __init__(self, 
                    target_metrics: List[str],
                    market_category: str = 'all',
                    lifecycle_stage: str = 'all',
                    use_survival_features: bool = True,
                    n_estimators: int = 100,
                    random_state: int = 42):
        """
        Parameters:
        -----------
        target_metrics : List[str]
            予測対象の評価指標リスト
        """
        super().__init__('multi_target', market_category, lifecycle_stage, 
                        use_survival_features, random_state)
        
        self.target_metrics = target_metrics
        self.n_estimators = n_estimators
        
        # 各評価指標用のモデル辞書
        self.metric_models = {}
        
        # マルチターゲット用基底学習器
        self.base_models = {
            'multi_rf': RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=12,
                min_samples_split=5,
                random_state=random_state,
                n_jobs=-1
            ),
            'multi_gb': GradientBoostingRegressor(
                n_estimators=n_estimators,
                learning_rate=0.1,
                max_depth=8,
                random_state=random_state
            ),
            'multi_xgb': xgb.XGBRegressor(
                n_estimators=n_estimators,
                learning_rate=0.1,
                max_depth=8,
                random_state=random_state,
                n_jobs=-1
            )
        }
    
    def fit(self, X: pd.DataFrame, y: pd.DataFrame, 
            sample_weight: Optional[np.ndarray] = None) -> 'MultiTargetEnsembleModel':
        """マルチターゲットモデルの学習"""
        logger.info("Training multi-target ensemble model")
        
        # 特徴量前処理
        X_processed = self._prepare_features(X)
        self.feature_names = X_processed.columns.tolist()
        
        # 標準化
        X_scaled = self.scaler.fit_transform(X_processed)
        X_scaled = pd.DataFrame(X_scaled, columns=self.feature_names)
        
        # 市場カテゴリ別重み計算
        if sample_weight is None:
            sample_weight = self._calculate_market_weights(X)
        
        # 各評価指標に対してモデル学習
        for metric in self.target_metrics:
            if metric not in y.columns:
                logger.warning(f"Target metric {metric} not found in y")
                continue
            
            logger.info(f"Training models for {metric}")
            self.metric_models[metric] = {}
            
            y_metric = y[metric].dropna()
            X_metric = X_scaled.loc[y_metric.index]
            weight_metric = sample_weight[y_metric.index] if len(sample_weight) > len(y_metric.index) else sample_weight
            
            # 各基底学習器の学習
            for name, base_model in self.base_models.items():
                try:
                    model_copy = self._clone_model(base_model)
                    model_copy.fit(X_metric, y_metric, sample_weight=weight_metric)
                    self.metric_models[metric][name] = model_copy
                except Exception as e:
                    logger.warning(f"Failed to train {name} for {metric}: {e}")
            
            # 投票回帰器の構築
            if self.metric_models[metric]:
                estimators = list(self.metric_models[metric].items())
                voting_model = VotingRegressor(estimators=estimators, n_jobs=-1)
                voting_model.fit(X_metric, y_metric, sample_weight=weight_metric)
                self.metric_models[metric]['voting'] = voting_model
        
        # 統合特徴量重要度計算
        self._calculate_multi_target_importance(X_scaled, y)
        
        self.is_fitted = True
        logger.info("Multi-target ensemble training completed")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """マルチターゲット予測"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # 特徴量前処理
        X_processed = self._prepare_features(X)
        X_scaled = self.scaler.transform(X_processed)
        X_scaled = pd.DataFrame(X_scaled, columns=self.feature_names)
        
        # 各評価指標の予測
        predictions = {}
        for metric in self.target_metrics:
            if metric in self.metric_models and 'voting' in self.metric_models[metric]:
                try:
                    predictions[metric] = self.metric_models[metric]['voting'].predict(X_scaled)
                except Exception as e:
                    logger.warning(f"Failed to predict {metric}: {e}")
                    predictions[metric] = np.zeros(len(X_scaled))
        
        return pd.DataFrame(predictions, index=X.index)
    
    def _calculate_multi_target_importance(self, X: pd.DataFrame, y: pd.DataFrame):
        """マルチターゲット特徴量重要度計算"""
        combined_importance = np.zeros(len(self.feature_names))
        
        for metric in self.target_metrics:
            if metric in self.metric_models:
                metric_importance = np.zeros(len(self.feature_names))
                model_count = 0
                
                for name, model in self.metric_models[metric].items():
                    if hasattr(model, 'feature_importances_') and name != 'voting':
                        metric_importance += model.feature_importances_
                        model_count += 1
                
                if model_count > 0:
                    metric_importance /= model_count
                    combined_importance += metric_importance
        
        # 平均重要度計算
        combined_importance /= len(self.target_metrics)
        self.feature_importance = dict(zip(self.feature_names, combined_importance))
    
    def _clone_model(self, model):
        """モデルのクローン作成"""
        from copy import deepcopy
        return deepcopy(model)
    
    def evaluate(self, X: pd.DataFrame, y: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """マルチターゲットモデル評価"""
        predictions = self.predict(X)
        evaluation_results = {}
        
        for metric in self.target_metrics:
            if metric in predictions.columns and metric in y.columns:
                y_true = y[metric].dropna()
                y_pred = predictions[metric].loc[y_true.index]
                
                evaluation_results[metric] = {
                    'mse': mean_squared_error(y_true, y_pred),
                    'mae': mean_absolute_error(y_true, y_pred),
                    'r2': r2_score(y_true, y_pred),
                    'rmse': np.sqrt(mean_squared_error(y_true, y_pred))
                }
        
        return evaluation_results


class AdaptiveEnsembleModel(BaseEnsembleModel):
    """
    適応的アンサンブルモデル
    市場環境・企業ライフサイクルに応じて動的にモデル重みを調整
    """
    
    def __init__(self, 
                    target_metric: str,
                    market_category: str = 'all',
                    lifecycle_stage: str = 'all',
                    adaptation_method: str = 'dynamic_weighting',
                    update_frequency: int = 12,  # 月単位
                    n_estimators: int = 100,
                    random_state: int = 42):
        """
        Parameters:
        -----------
        adaptation_method : str
            適応方法 ('dynamic_weighting', 'online_learning', 'concept_drift')
        update_frequency : int
            モデル更新頻度（月単位）
        """
        super().__init__(target_metric, market_category, lifecycle_stage, True, random_state)
        
        self.adaptation_method = adaptation_method
        self.update_frequency = update_frequency
        self.n_estimators = n_estimators
        
        # 適応的重み
        self.model_weights = {}
        self.performance_history = {}
        
        # 基底学習器（適応性を考慮した設計）
        self.base_models = {
            'adaptive_rf': RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=10,
                min_samples_split=5,
                warm_start=True,  # オンライン学習対応
                random_state=random_state,
                n_jobs=-1
            ),
            'adaptive_gb': GradientBoostingRegressor(
                n_estimators=n_estimators,
                learning_rate=0.1,
                max_depth=6,
                warm_start=True,
                random_state=random_state
            ),
            'adaptive_sgd': None  # SGDRegressorを後で初期化
        }
        
        # SGDRegressorの初期化
        from sklearn.linear_model import SGDRegressor
        self.base_models['adaptive_sgd'] = SGDRegressor(
            learning_rate='adaptive',
            eta0=0.01,
            random_state=random_state
        )
    
    def fit(self, X: pd.DataFrame, y: pd.Series, 
            sample_weight: Optional[np.ndarray] = None) -> 'AdaptiveEnsembleModel':
        """適応的アンサンブルモデルの学習"""
        logger.info("Training adaptive ensemble model")
        
        # 特徴量前処理
        X_processed = self._prepare_features(X)
        self.feature_names = X_processed.columns.tolist()
        
        # 標準化
        X_scaled = self.scaler.fit_transform(X_processed)
        X_scaled = pd.DataFrame(X_scaled, columns=self.feature_names)
        
        # 時系列分割による適応学習
        self._adaptive_fit(X_scaled, y, sample_weight)
        
        self.is_fitted = True
        logger.info("Adaptive ensemble training completed")
        
        return self
    
    def _adaptive_fit(self, X: pd.DataFrame, y: pd.Series, sample_weight: np.ndarray):
        """適応的学習プロセス"""
        if 'date' in X.columns:
            X_sorted = X.sort_values('date')
            y_sorted = y.loc[X_sorted.index]
        else:
            X_sorted, y_sorted = X, y
        
        # 時間窓でのローリング学習
        window_size = len(X_sorted) // 5  # 5分割
        
        for i in range(4):  # 最初の4窓で学習、最後の1窓で検証
            start_idx = i * window_size
            end_idx = (i + 2) * window_size  # オーバーラップあり
            
            X_window = X_sorted.iloc[start_idx:end_idx]
            y_window = y_sorted.iloc[start_idx:end_idx]
            
            # 各基底学習器の学習
            for name, model in self.base_models.items():
                try:
                    if sample_weight is not None:
                        weight_window = sample_weight[start_idx:end_idx]
                        model.fit(X_window, y_window, sample_weight=weight_window)
                    else:
                        model.fit(X_window, y_window)
                    
                    # 性能評価
                    val_start = end_idx
                    val_end = min(end_idx + window_size, len(X_sorted))
                    if val_end > val_start:
                        X_val = X_sorted.iloc[val_start:val_end]
                        y_val = y_sorted.iloc[val_start:val_end]
                        
                        pred_val = model.predict(X_val)
                        mse = mean_squared_error(y_val, pred_val)
                        
                        if name not in self.performance_history:
                            self.performance_history[name] = []
                        self.performance_history[name].append(mse)
                
                except Exception as e:
                    logger.warning(f"Failed to train {name} in window {i}: {e}")
        
        # 動的重み計算
        self._calculate_dynamic_weights()
        
        # 最終モデル構築
        self._build_final_adaptive_model(X_sorted, y_sorted, sample_weight)
    
    def _calculate_dynamic_weights(self):
        """動的重み計算"""
        self.model_weights = {}
        
        for name, history in self.performance_history.items():
            if history:
                # 最近の性能により高い重み
                recent_performance = np.mean(history[-2:]) if len(history) >= 2 else history[0]
                # 逆数重み（低いMSE = 高い重み）
                self.model_weights[name] = 1.0 / (recent_performance + 1e-6)
        
        # 重みの正規化
        total_weight = sum(self.model_weights.values())
        if total_weight > 0:
            self.model_weights = {k: v/total_weight for k, v in self.model_weights.items()}
    
    def _build_final_adaptive_model(self, X: pd.DataFrame, y: pd.Series, sample_weight: np.ndarray):
        """最終適応モデル構築"""
        trained_estimators = []
        
        for name, model in self.base_models.items():
            if name in self.model_weights and self.model_weights[name] > 0.01:  # 閾値以上の重みのみ
                trained_estimators.append((name, model))
        
        if trained_estimators:
            self.models['adaptive_voting'] = VotingRegressor(
                estimators=trained_estimators,
                n_jobs=-1
            )
            
            # 重みを考慮した学習
            if sample_weight is not None:
                self.models['adaptive_voting'].fit(X, y, sample_weight=sample_weight)
            else:
                self.models['adaptive_voting'].fit(X, y)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """適応的予測"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # 特徴量前処理
        X_processed = self._prepare_features(X)
        X_scaled = self.scaler.transform(X_processed)
        X_scaled = pd.DataFrame(X_scaled, columns=self.feature_names)
        
        if 'adaptive_voting' in self.models:
            return self.models['adaptive_voting'].predict(X_scaled)
        else:
            # フォールバック：重み付き平均予測
            predictions = []
            weights = []
            
            for name, model in self.base_models.items():
                if name in self.model_weights:
                    try:
                        pred = model.predict(X_scaled)
                        predictions.append(pred)
                        weights.append(self.model_weights[name])
                    except Exception:
                        continue
            
            if predictions:
                weighted_pred = np.average(predictions, axis=0, weights=weights)
                return weighted_pred
            else:
                return np.zeros(len(X_scaled))
    
    def update_model(self, X_new: pd.DataFrame, y_new: pd.Series):
        """オンライン更新"""
        logger.info("Updating adaptive ensemble model")
        
        # 特徴量前処理
        X_processed = self._prepare_features(X_new)
        X_scaled = self.scaler.transform(X_processed)
        X_scaled = pd.DataFrame(X_scaled, columns=self.feature_names)
        
        # 各基底学習器の更新
        for name, model in self.base_models.items():
            if hasattr(model, 'warm_start') or name == 'adaptive_sgd':
                try:
                    if name == 'adaptive_sgd':
                        # SGDは部分学習
                        model.partial_fit(X_scaled, y_new)
                    else:
                        # その他はwarm_start
                        model.warm_start = True
                        model.fit(X_scaled, y_new)
                    
                    # 性能更新
                    pred = model.predict(X_scaled)
                    mse = mean_squared_error(y_new, pred)
                    self.performance_history[name].append(mse)
                    
                except Exception as e:
                    logger.warning(f"Failed to update {name}: {e}")
        
        # 重み再計算
        self._calculate_dynamic_weights()
        
        logger.info("Model update completed")
    
    def get_model_weights(self) -> Dict[str, float]:
        """現在のモデル重み取得"""
        return self.model_weights.copy()
    
    def get_performance_history(self) -> Dict[str, List[float]]:
        """性能履歴取得"""
        return self.performance_history.copy()


def create_ensemble_model(model_type: str, 
                            target_metric: str,
                            market_category: str = 'all',
                         **kwargs) -> BaseEnsembleModel:
    """
    アンサンブルモデルファクトリ関数
    
    Parameters:
    -----------
    model_type : str
        モデルタイプ ('regression', 'survival', 'multi_target', 'adaptive')
    target_metric : str
        予測対象指標
    market_category : str
        市場カテゴリ
    **kwargs : dict
        その他のパラメータ
    
    Returns:
    --------
    BaseEnsembleModel
        初期化されたアンサンブルモデル
    """
    
    if model_type == 'regression':
        return FinancialEnsembleRegressor(
            target_metric=target_metric,
            market_category=market_category,
            **kwargs
        )
    
    elif model_type == 'survival':
        return SurvivalEnsembleClassifier(
            market_category=market_category,
            **kwargs
        )
    
    elif model_type == 'multi_target':
        target_metrics = kwargs.pop('target_metrics', [target_metric])
        return MultiTargetEnsembleModel(
            target_metrics=target_metrics,
            market_category=market_category,
            **kwargs
        )
    
    elif model_type == 'adaptive':
        return AdaptiveEnsembleModel(
            target_metric=target_metric,
            market_category=market_category,
            **kwargs
        )
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# 使用例とテスト用のヘルパー関数
def example_usage():
    """
    アンサンブルモデルの使用例
    """
    # サンプルデータ生成（実際の使用時は実データを使用）
    np.random.seed(42)
    n_samples = 1000
    n_features = 50
    
    # 特徴量データ
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    X['market_category'] = np.random.choice(['high_share', 'declining', 'lost'], n_samples)
    X['company_age'] = np.random.randint(1, 50, n_samples)
    X['rd_ratio'] = np.random.uniform(0, 0.2, n_samples)
    
    # 目的変数（売上高成長率）
    y = pd.Series(np.random.randn(n_samples) * 0.1 + 0.05)  # 平均5%の成長率
    
    # 1. 回帰モデルの例
    print("=== Financial Ensemble Regressor ===")
    regressor = create_ensemble_model(
        model_type='regression',
        target_metric='sales_growth_rate',
        market_category='all',
        ensemble_method='stacking'
    )
    
    # 学習・評価用データ分割
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # 学習
    regressor.fit(X_train, y_train)
    
    # 予測・評価
    predictions = regressor.predict(X_test)
    evaluation = regressor.evaluate(X_test, y_test)
    print(f"Regression Evaluation: {evaluation}")
    
    # 特徴量重要度
    importance = regressor.get_feature_importance()
    top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
    print(f"Top 10 Important Features: {top_features}")
    
    # 2. 生存分析モデルの例
    print("\n=== Survival Ensemble Classifier ===")
    # 生存データ（1=生存, 0=消滅）
    y_survival = pd.Series(np.random.choice([0, 1], n_samples, p=[0.2, 0.8]))
    
    survival_model = create_ensemble_model(
        model_type='survival',
        target_metric='survival',
        market_category='all'
    )
    
    y_survival_train = y_survival[:train_size]
    y_survival_test = y_survival[train_size:]
    
    survival_model.fit(X_train, y_survival_train)
    survival_predictions = survival_model.predict(X_test)
    survival_evaluation = survival_model.evaluate(X_test, y_survival_test)
    print(f"Survival Evaluation: {survival_evaluation}")
    
    # 3. マルチターゲットモデルの例
    print("\n=== Multi-Target Ensemble Model ===")
    # 複数目的変数
    y_multi = pd.DataFrame({
        'sales_growth_rate': y,
        'operating_margin': np.random.randn(n_samples) * 0.05 + 0.1,
        'roe': np.random.randn(n_samples) * 0.1 + 0.15
    })
    
    multi_model = create_ensemble_model(
        model_type='multi_target',
        target_metric='multi',
        target_metrics=['sales_growth_rate', 'operating_margin', 'roe'],
        market_category='all'
    )
    
    y_multi_train = y_multi[:train_size]
    y_multi_test = y_multi[train_size:]
    
    multi_model.fit(X_train, y_multi_train)
    multi_predictions = multi_model.predict(X_test)
    multi_evaluation = multi_model.evaluate(X_test, y_multi_test)
    print(f"Multi-Target Evaluation: {multi_evaluation}")


if __name__ == "__main__":
    # 使用例実行
    example_usage()