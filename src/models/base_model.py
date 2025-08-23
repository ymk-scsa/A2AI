"""
A2AI (Advanced Financial Analysis AI) Base Model

企業ライフサイクル全体を考慮した財務諸表分析AIの基底モデルクラス
- 生存分析、新設企業分析、因果推論を統合
- 生存バイアス完全対応
- 150社×40年分の企業データ分析対応
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from abc import ABC, abstractmethod
from datetime import datetime
import logging
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, RobustScaler
import warnings

# 設定とユーティリティのインポート
from ..utils.data_utils import DataUtils
from ..utils.math_utils import MathUtils
from ..utils.survival_utils import SurvivalUtils
from ..utils.causal_utils import CausalUtils
from ..utils.lifecycle_utils import LifecycleUtils
from ..utils.statistical_utils import StatisticalUtils


class ModelType:
    """モデルタイプの定義"""
    TRADITIONAL = "traditional"  # 従来型財務分析
    SURVIVAL = "survival"       # 生存分析
    EMERGENCE = "emergence"     # 新設企業分析
    CAUSAL = "causal"          # 因果推論
    INTEGRATED = "integrated"   # 統合分析


class LifecycleStage:
    """企業ライフサイクルステージ定義"""
    STARTUP = "startup"         # 新設期（設立～5年）
    GROWTH = "growth"          # 成長期（6～15年）
    MATURITY = "maturity"      # 成熟期（16～30年）
    DECLINE = "decline"        # 衰退期（31年～）
    EXTINCT = "extinct"        # 消滅済み


class MarketCategory:
    """市場カテゴリ定義"""
    HIGH_SHARE = "high_share"      # 世界シェア高市場
    DECLINING = "declining"        # シェア低下市場
    LOST = "lost"                 # シェア完全失失市場


class A2AIBaseModel(ABC, BaseEstimator):
    """
    A2AI基底モデルクラス
    
    すべてのA2AIモデル（従来型、生存分析、新設企業、因果推論、統合）の
    共通基盤となる抽象基底クラス
    """
    
    def __init__(
        self,
        model_type: str,
        evaluation_metrics: List[str] = None,
        factor_variables: List[str] = None,
        time_window: Tuple[int, int] = (1984, 2024),
        market_categories: List[str] = None,
        lifecycle_aware: bool = True,
        survival_bias_correction: bool = True,
        causal_inference: bool = False,
        random_state: int = 42,
        verbose: int = 1
    ):
        """
        A2AIベースモデル初期化
        
        Parameters:
        -----------
        model_type : str
            モデルタイプ（traditional/survival/emergence/causal/integrated）
        evaluation_metrics : List[str]
            評価項目リスト（デフォルトは9項目すべて）
        factor_variables : List[str] 
            要因項目リスト（デフォルトは23項目すべて）
        time_window : Tuple[int, int]
            分析対象期間（開始年、終了年）
        market_categories : List[str]
            対象市場カテゴリ
        lifecycle_aware : bool
            ライフサイクル考慮するかどうか
        survival_bias_correction : bool
            生存バイアス補正するかどうか
        causal_inference : bool
            因果推論を行うかどうか
        random_state : int
            乱数シード
        verbose : int
            ログレベル
        """
        
        # 基本設定
        self.model_type = model_type
        self.random_state = random_state
        self.verbose = verbose
        self.time_window = time_window
        self.lifecycle_aware = lifecycle_aware
        self.survival_bias_correction = survival_bias_correction
        self.causal_inference = causal_inference
        
        # 評価項目設定（9項目）
        self.evaluation_metrics = evaluation_metrics or [
            'sales_revenue',           # 売上高
            'sales_growth_rate',       # 売上高成長率
            'operating_margin',        # 売上高営業利益率
            'net_margin',             # 売上高当期純利益率
            'roe',                    # ROE
            'value_added_ratio',      # 売上高付加価値率
            'survival_probability',    # 企業存続確率（新規）
            'emergence_success_rate',  # 新規事業成功率（新規）
            'succession_success_rate'  # 事業継承成功度（新規）
        ]
        
        # 要因項目設定（各評価項目23項目ずつ）
        self.factor_variables = factor_variables or self._get_default_factors()
        
        # 市場カテゴリ設定
        self.market_categories = market_categories or [
            MarketCategory.HIGH_SHARE,
            MarketCategory.DECLINING, 
            MarketCategory.LOST
        ]
        
        # モデル状態
        self.is_fitted = False
        self.feature_names = None
        self.target_names = None
        self.model_metadata = {}
        
        # データ処理コンポーネント
        self.scaler = None
        self.data_utils = DataUtils()
        self.math_utils = MathUtils()
        self.survival_utils = SurvivalUtils()
        self.causal_utils = CausalUtils()
        self.lifecycle_utils = LifecycleUtils()
        self.statistical_utils = StatisticalUtils()
        
        # ログ設定
        self.logger = self._setup_logging()
        
        # バリデーション
        self._validate_init_params()
    
    def _get_default_factors(self) -> List[str]:
        """デフォルト要因項目リスト（23項目）を取得"""
        base_factors = [
            # 投資・資産関連（5項目）
            'tangible_fixed_assets', 'capital_investment', 'rd_expenses',
            'intangible_assets', 'investment_securities',
            
            # 人的資源関連（4項目）  
            'employee_count', 'average_salary', 'retirement_benefit_cost',
            'welfare_cost',
            
            # 運転資本・効率性関連（5項目）
            'trade_receivables', 'inventory', 'total_assets',
            'receivables_turnover', 'inventory_turnover',
            
            # 事業展開関連（6項目）
            'overseas_sales_ratio', 'business_segments', 'sga_expenses',
            'advertising_cost', 'non_operating_income', 'order_backlog',
            
            # 新規拡張項目（3項目）
            'company_age',           # 企業年齢
            'market_entry_timing',   # 市場参入時期
            'parent_dependency'      # 親会社依存度
        ]
        return base_factors
    
    def _setup_logging(self) -> logging.Logger:
        """ログ設定"""
        logger = logging.getLogger(f"A2AI.{self.__class__.__name__}")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        if self.verbose >= 2:
            logger.setLevel(logging.DEBUG)
        elif self.verbose >= 1:
            logger.setLevel(logging.INFO)
        else:
            logger.setLevel(logging.WARNING)
            
        return logger
    
    def _validate_init_params(self):
        """初期化パラメータの検証"""
        if self.model_type not in [
            ModelType.TRADITIONAL, ModelType.SURVIVAL, 
            ModelType.EMERGENCE, ModelType.CAUSAL, ModelType.INTEGRATED
        ]:
            raise ValueError(f"Invalid model_type: {self.model_type}")
        
        if len(self.time_window) != 2 or self.time_window[0] >= self.time_window[1]:
            raise ValueError(f"Invalid time_window: {self.time_window}")
        
        if not all(cat in [MarketCategory.HIGH_SHARE, MarketCategory.DECLINING, 
                            MarketCategory.LOST] for cat in self.market_categories):
            raise ValueError(f"Invalid market_categories: {self.market_categories}")
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.DataFrame, **kwargs) -> 'A2AIBaseModel':
        """モデル学習（抽象メソッド）"""
        pass
    
    @abstractmethod  
    def predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        """予測実行（抽象メソッド）"""
        pass
    
    def preprocess_data(
        self, 
        X: pd.DataFrame, 
        y: pd.DataFrame = None,
        fit_preprocessor: bool = True
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        データ前処理の統一インターフェース
        
        Parameters:
        -----------
        X : pd.DataFrame
            特徴量データ
        y : pd.DataFrame, optional
            ターゲットデータ
        fit_preprocessor : bool
            前処理器をfitするかどうか
            
        Returns:
        --------
        X_processed : pd.DataFrame
            前処理済み特徴量
        y_processed : pd.DataFrame or None
            前処理済みターゲット
        """
        
        self.logger.info("Starting data preprocessing...")
        
        # 基本的な前処理
        X_processed = self._basic_preprocessing(X)
        
        # ライフサイクル考慮前処理
        if self.lifecycle_aware:
            X_processed = self._lifecycle_preprocessing(X_processed)
        
        # 生存バイアス補正
        if self.survival_bias_correction:
            X_processed = self._survival_bias_correction(X_processed)
        
        # 標準化
        if fit_preprocessor:
            self.scaler = RobustScaler()  # 外れ値に頑健
            X_processed[self.factor_variables] = self.scaler.fit_transform(
                X_processed[self.factor_variables]
            )
        elif self.scaler is not None:
            X_processed[self.factor_variables] = self.scaler.transform(
                X_processed[self.factor_variables]
            )
        
        # ターゲット前処理
        y_processed = None
        if y is not None:
            y_processed = self._target_preprocessing(y)
        
        # メタデータ保存
        self._save_preprocessing_metadata(X_processed, y_processed)
        
        self.logger.info("Data preprocessing completed")
        return X_processed, y_processed
    
    def _basic_preprocessing(self, X: pd.DataFrame) -> pd.DataFrame:
        """基本前処理"""
        X_processed = X.copy()
        
        # 欠損値処理
        X_processed = self.data_utils.handle_missing_values(
            X_processed, 
            method='interpolate',  # 時系列補間
            lifecycle_aware=self.lifecycle_aware
        )
        
        # 外れ値検出・処理
        X_processed = self.data_utils.handle_outliers(
            X_processed,
            method='iqr',
            factor=3.0
        )
        
        return X_processed
    
    def _lifecycle_preprocessing(self, X: pd.DataFrame) -> pd.DataFrame:
        """ライフサイクル考慮前処理"""
        # 企業年齢に基づくライフサイクルステージ分類
        X_processed = self.lifecycle_utils.classify_lifecycle_stage(X)
        
        # ライフサイクル段階別正規化
        X_processed = self.lifecycle_utils.stage_wise_normalization(X_processed)
        
        return X_processed
    
    def _survival_bias_correction(self, X: pd.DataFrame) -> pd.DataFrame:
        """生存バイアス補正"""
        # 消滅企業の重み調整
        X_processed = self.survival_utils.correct_survivorship_bias(X)
        
        # 時系列長の違いを考慮した重み付け
        X_processed = self.survival_utils.adjust_temporal_weights(X_processed)
        
        return X_processed
    
    def _target_preprocessing(self, y: pd.DataFrame) -> pd.DataFrame:
        """ターゲット前処理"""
        y_processed = y.copy()
        
        # 評価項目別の前処理
        for metric in self.evaluation_metrics:
            if metric in y_processed.columns:
                # 比率データの処理（0-1範囲に正規化）
                if 'ratio' in metric or 'rate' in metric:
                    y_processed[metric] = np.clip(y_processed[metric], 0, 1)
                
                # 成長率データの処理（極値の処理）
                elif 'growth' in metric:
                    y_processed[metric] = np.clip(
                        y_processed[metric], -0.5, 2.0  # -50%～200%
                    )
        
        return y_processed
    
    def _save_preprocessing_metadata(self, X: pd.DataFrame, y: pd.DataFrame = None):
        """前処理メタデータ保存"""
        self.model_metadata.update({
            'feature_names': list(X.columns),
            'target_names': list(y.columns) if y is not None else None,
            'n_features': X.shape[1],
            'n_samples': X.shape[0],
            'time_range': (X.index.min(), X.index.max()) if hasattr(X.index, 'min') else None,
            'preprocessing_timestamp': datetime.now().isoformat()
        })
        
        self.feature_names = list(X.columns)
        self.target_names = list(y.columns) if y is not None else None
    
    def create_time_series_splits(
        self, 
        X: pd.DataFrame, 
        n_splits: int = 5,
        test_size: int = None
    ) -> TimeSeriesSplit:
        """
        時系列交差検証用分割作成
        
        Parameters:
        -----------
        X : pd.DataFrame
            データ
        n_splits : int
            分割数
        test_size : int, optional
            テストサイズ
            
        Returns:
        --------
        TimeSeriesSplit
            時系列分割オブジェクト
        """
        if test_size is None:
            # デフォルト: 全期間の20%をテストサイズに
            test_size = len(X) // 5
        
        return TimeSeriesSplit(
            n_splits=n_splits,
            test_size=test_size
        )
    
    def evaluate_model(
        self, 
        X_test: pd.DataFrame, 
        y_test: pd.DataFrame,
        metrics: List[str] = None
    ) -> Dict[str, float]:
        """
        モデル評価
        
        Parameters:
        -----------
        X_test : pd.DataFrame
            テスト特徴量
        y_test : pd.DataFrame  
            テスト正解値
        metrics : List[str], optional
            評価指標リスト
            
        Returns:
        --------
        Dict[str, float]
            評価結果
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before evaluation")
        
        # 予測実行
        y_pred = self.predict(X_test)
        
        # デフォルト評価指標
        if metrics is None:
            metrics = ['mae', 'mse', 'rmse', 'r2']
        
        # 評価実行
        results = {}
        for metric in metrics:
            if metric == 'mae':
                results[metric] = self.statistical_utils.mean_absolute_error(y_test, y_pred)
            elif metric == 'mse':
                results[metric] = self.statistical_utils.mean_squared_error(y_test, y_pred)
            elif metric == 'rmse':
                results[metric] = np.sqrt(results.get('mse', 0))
            elif metric == 'r2':
                results[metric] = self.statistical_utils.r2_score(y_test, y_pred)
        
        return results
    
    def get_feature_importance(self) -> Dict[str, np.ndarray]:
        """特徴量重要度取得（基本実装）"""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before getting feature importance")
        
        # 基底クラスではダミー実装
        # 各派生クラスでオーバーライド
        return {
            feature: np.random.random() 
            for feature in self.feature_names
        }
    
    def save_model(self, filepath: str):
        """モデル保存"""
        import joblib
        
        model_data = {
            'model': self,
            'metadata': self.model_metadata,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'target_names': self.target_names
        }
        
        joblib.dump(model_data, filepath)
        self.logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str) -> 'A2AIBaseModel':
        """モデル読み込み"""
        import joblib
        
        model_data = joblib.load(filepath)
        model = model_data['model']
        model.model_metadata = model_data['metadata']
        model.scaler = model_data['scaler']
        model.feature_names = model_data['feature_names']
        model.target_names = model_data['target_names']
        
        return model
    
    def get_model_info(self) -> Dict[str, Any]:
        """モデル情報取得"""
        return {
            'model_type': self.model_type,
            'evaluation_metrics': self.evaluation_metrics,
            'factor_variables': self.factor_variables,
            'time_window': self.time_window,
            'market_categories': self.market_categories,
            'lifecycle_aware': self.lifecycle_aware,
            'survival_bias_correction': self.survival_bias_correction,
            'causal_inference': self.causal_inference,
            'is_fitted': self.is_fitted,
            'metadata': self.model_metadata
        }
    
    def __repr__(self) -> str:
        return (
            f"A2AIBaseModel("
            f"model_type={self.model_type}, "
            f"n_evaluation_metrics={len(self.evaluation_metrics)}, "
            f"n_factor_variables={len(self.factor_variables)}, "
            f"time_window={self.time_window}, "
            f"is_fitted={self.is_fitted})"
        )


class A2AIModelMixin:
    """A2AI共通機能のMixin"""
    
    def analyze_factor_impact(
        self, 
        factor_name: str,
        impact_range: Tuple[float, float] = (-0.5, 0.5)
    ) -> Dict[str, Any]:
        """
        要因項目の影響分析
        
        Parameters:
        -----------
        factor_name : str
            分析する要因項目名
        impact_range : Tuple[float, float]
            影響範囲（変化率の最小・最大）
            
        Returns:
        --------
        Dict[str, Any]
            影響分析結果
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before impact analysis")
        
        # 実装は各派生クラスで具体化
        return {
            'factor': factor_name,
            'impact_range': impact_range,
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def compare_market_categories(self) -> Dict[str, Any]:
        """市場カテゴリ間比較分析"""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before market comparison")
        
        # 実装は各派生クラスで具体化
        return {
            'market_categories': self.market_categories,
            'comparison_timestamp': datetime.now().isoformat()
        }
    
    def lifecycle_stage_analysis(self) -> Dict[str, Any]:
        """ライフサイクルステージ別分析"""
        if not self.lifecycle_aware:
            raise RuntimeError("Model must be lifecycle_aware for this analysis")
        
        # 実装は各派生クラスで具体化
        return {
            'lifecycle_stages': [
                LifecycleStage.STARTUP, LifecycleStage.GROWTH,
                LifecycleStage.MATURITY, LifecycleStage.DECLINE,
                LifecycleStage.EXTINCT
            ],
            'analysis_timestamp': datetime.now().isoformat()
        }


# モデル登録レジストリ
A2AI_MODEL_REGISTRY = {}

def register_a2ai_model(model_type: str):
    """A2AIモデル登録デコレータ"""
    def decorator(cls):
        A2AI_MODEL_REGISTRY[model_type] = cls
        return cls
    return decorator


def get_a2ai_model(model_type: str, **kwargs) -> A2AIBaseModel:
    """A2AIモデルファクトリ関数"""
    if model_type not in A2AI_MODEL_REGISTRY:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model_class = A2AI_MODEL_REGISTRY[model_type]
    return model_class(**kwargs)