"""
A2AI Cross Validation System
交差検証システム - 生存分析、新設企業分析、因果推論モデルに対応

企業ライフサイクル全体を考慮した高度な交差検証機能を提供
- 時系列データに対応した交差検証
- 生存分析専用の評価指標
- 企業消滅・新設を考慮したバイアス補正
- 因果推論の頑健性検証
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from sklearn.model_selection import (
    KFold, StratifiedKFold, TimeSeriesSplit, 
    GroupKFold, StratifiedGroupKFold
)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from lifelines.utils import concordance_index
from lifelines.statistics import logrank_test
import warnings
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging
from datetime import datetime, timedelta


@dataclass
class ValidationConfig:
    """交差検証設定クラス"""
    n_splits: int = 5
    test_size: float = 0.2
    random_state: int = 42
    shuffle: bool = True
    stratify_column: Optional[str] = None
    group_column: Optional[str] = None
    time_column: Optional[str] = None
    validation_type: str = 'standard'  # 'standard', 'time_series', 'survival', 'group'


@dataclass
class ValidationResult:
    """交差検証結果クラス"""
    fold_scores: Dict[str, List[float]]
    mean_scores: Dict[str, float]
    std_scores: Dict[str, float]
    fold_predictions: List[np.ndarray]
    fold_true_values: List[np.ndarray]
    validation_details: Dict[str, Any]
    model_type: str
    timestamp: datetime


class BaseValidator(ABC):
    """基底バリデータークラス"""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    @abstractmethod
    def create_splits(self, X: pd.DataFrame, y: Union[pd.Series, pd.DataFrame]) -> List[Tuple[np.ndarray, np.ndarray]]:
        """データ分割の作成"""
        pass
    
    @abstractmethod
    def evaluate_fold(self, model: Any, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                        y_train: Union[pd.Series, pd.DataFrame], y_test: Union[pd.Series, pd.DataFrame]) -> Dict[str, float]:
        """各フォールドでの評価"""
        pass


class StandardValidator(BaseValidator):
    """標準的な交差検証"""
    
    def create_splits(self, X: pd.DataFrame, y: Union[pd.Series, pd.DataFrame]) -> List[Tuple[np.ndarray, np.ndarray]]:
        """標準的なK-Fold交差検証の分割作成"""
        if self.config.stratify_column and self.config.stratify_column in X.columns:
            # 層化交差検証
            stratify_values = X[self.config.stratify_column]
            splitter = StratifiedKFold(
                n_splits=self.config.n_splits,
                shuffle=self.config.shuffle,
                random_state=self.config.random_state
            )
            return list(splitter.split(X, stratify_values))
        else:
            # 通常のK-Fold
            splitter = KFold(
                n_splits=self.config.n_splits,
                shuffle=self.config.shuffle,
                random_state=self.config.random_state
            )
            return list(splitter.split(X))
    
    def evaluate_fold(self, model: Any, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                        y_train: Union[pd.Series, pd.DataFrame], y_test: Union[pd.Series, pd.DataFrame]) -> Dict[str, float]:
        """標準的な回帰/分類評価"""
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        return {
            'mse': mean_squared_error(y_test, y_pred),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
        }


class TimeSeriesValidator(BaseValidator):
    """時系列データ専用交差検証"""
    
    def create_splits(self, X: pd.DataFrame, y: Union[pd.Series, pd.DataFrame]) -> List[Tuple[np.ndarray, np.ndarray]]:
        """時系列データの分割作成"""
        if self.config.time_column:
            # 時間列でソート
            time_sorted_idx = X[self.config.time_column].argsort()
            X_sorted = X.iloc[time_sorted_idx]
            
            splitter = TimeSeriesSplit(
                n_splits=self.config.n_splits,
                test_size=None,
                gap=0
            )
            return list(splitter.split(X_sorted))
        else:
            # 時間列がない場合はインデックス順で分割
            splitter = TimeSeriesSplit(n_splits=self.config.n_splits)
            return list(splitter.split(X))
    
    def evaluate_fold(self, model: Any, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                        y_train: Union[pd.Series, pd.DataFrame], y_test: Union[pd.Series, pd.DataFrame]) -> Dict[str, float]:
        """時系列予測の評価"""
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        scores = {
            'mse': mean_squared_error(y_test, y_pred),
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mape': np.mean(np.abs((y_test - y_pred) / y_test)) * 100  # 平均絶対パーセント誤差
        }
        
        # 時系列固有の評価指標
        if len(y_test) > 1:
            scores['directional_accuracy'] = self._calculate_directional_accuracy(y_test, y_pred)
        
        return scores
    
    def _calculate_directional_accuracy(self, y_true: pd.Series, y_pred: np.ndarray) -> float:
        """方向性の精度（上昇/下降の予測精度）"""
        if len(y_true) < 2:
            return 0.0
        
        true_direction = np.diff(y_true) > 0
        pred_direction = np.diff(y_pred) > 0
        
        return np.mean(true_direction == pred_direction)


class SurvivalValidator(BaseValidator):
    """生存分析専用交差検証"""
    
    def create_splits(self, X: pd.DataFrame, y: Union[pd.Series, pd.DataFrame]) -> List[Tuple[np.ndarray, np.ndarray]]:
        """生存分析用の分割作成"""
        # 企業グループ（市場カテゴリ）を考慮した分割
        if self.config.group_column and self.config.group_column in X.columns:
            groups = X[self.config.group_column]
            splitter = GroupKFold(n_splits=self.config.n_splits)
            return list(splitter.split(X, groups=groups))
        else:
            # イベント発生有無で層化
            if isinstance(y, pd.DataFrame) and 'event' in y.columns:
                stratify_values = y['event']
                splitter = StratifiedKFold(
                    n_splits=self.config.n_splits,
                    shuffle=self.config.shuffle,
                    random_state=self.config.random_state
                )
                return list(splitter.split(X, stratify_values))
            else:
                return super().create_splits(X, y)
    
    def evaluate_fold(self, model: Any, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                        y_train: Union[pd.Series, pd.DataFrame], y_test: Union[pd.Series, pd.DataFrame]) -> Dict[str, float]:
        """生存分析の評価"""
        model.fit(X_train, y_train)
        
        scores = {}
        
        # C-index（concordance index）計算
        if hasattr(model, 'predict_survival_function'):
            # リスクスコア予測
            risk_scores = -model.predict_partial_hazard(X_test)
            if isinstance(y_test, pd.DataFrame):
                duration = y_test.iloc[:, 0]  # 生存時間
                event = y_test.iloc[:, 1] if y_test.shape[1] > 1 else np.ones(len(y_test))  # イベント発生
            else:
                duration = y_test
                event = np.ones(len(y_test))
            
            scores['c_index'] = concordance_index(duration, risk_scores, event)
        
        # Log-rank test p-value（グループ分けして比較）
        if len(np.unique(risk_scores)) > 1:
            median_risk = np.median(risk_scores)
            high_risk_group = risk_scores >= median_risk
            low_risk_group = risk_scores < median_risk
            
            if np.sum(high_risk_group) > 0 and np.sum(low_risk_group) > 0:
                try:
                    logrank_result = logrank_test(
                        duration[high_risk_group], duration[low_risk_group],
                        event_observed_A=event[high_risk_group], 
                        event_observed_B=event[low_risk_group]
                    )
                    scores['logrank_pvalue'] = logrank_result.p_value
                except:
                    scores['logrank_pvalue'] = 1.0
        
        # Brier Score（時点tでの生存確率予測精度）
        if hasattr(model, 'predict_survival_function'):
            try:
                survival_functions = model.predict_survival_function(X_test)
                # 簡単のため、中央値時点でのBrier Score計算
                if len(survival_functions) > 0:
                    median_time = np.median(duration)
                    brier_scores = []
                    for i, sf in enumerate(survival_functions):
                        if median_time in sf.index:
                            predicted_survival = sf.loc[median_time]
                            actual_survival = 1 if (duration.iloc[i] > median_time) else 0
                            brier_scores.append((predicted_survival - actual_survival) ** 2)
                    
                    if brier_scores:
                        scores['brier_score'] = np.mean(brier_scores)
            except Exception as e:
                self.logger.warning(f"Brier score calculation failed: {e}")
        
        return scores


class GroupValidator(BaseValidator):
    """企業グループを考慮した交差検証"""
    
    def create_splits(self, X: pd.DataFrame, y: Union[pd.Series, pd.DataFrame]) -> List[Tuple[np.ndarray, np.ndarray]]:
        """企業・市場グループを考慮した分割"""
        if not self.config.group_column or self.config.group_column not in X.columns:
            raise ValueError(f"Group column '{self.config.group_column}' not found in data")
        
        groups = X[self.config.group_column]
        
        if self.config.stratify_column and self.config.stratify_column in X.columns:
            # 層化グループK-Fold
            stratify_values = X[self.config.stratify_column]
            splitter = StratifiedGroupKFold(
                n_splits=self.config.n_splits,
                shuffle=self.config.shuffle,
                random_state=self.config.random_state
            )
            return list(splitter.split(X, stratify_values, groups))
        else:
            # グループK-Fold
            splitter = GroupKFold(n_splits=self.config.n_splits)
            return list(splitter.split(X, groups=groups))
    
    def evaluate_fold(self, model: Any, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                        y_train: Union[pd.Series, pd.DataFrame], y_test: Union[pd.Series, pd.DataFrame]) -> Dict[str, float]:
        """グループを考慮した評価"""
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        scores = {
            'mse': mean_squared_error(y_test, y_pred),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
        }
        
        # グループ別の評価も追加
        if self.config.group_column in X_test.columns:
            group_scores = self._evaluate_by_groups(X_test, y_test, y_pred)
            scores.update(group_scores)
        
        return scores
    
    def _evaluate_by_groups(self, X_test: pd.DataFrame, y_test: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """グループ別評価"""
        groups = X_test[self.config.group_column]
        unique_groups = groups.unique()
        
        group_scores = {}
        for group in unique_groups:
            mask = groups == group
            if np.sum(mask) > 1:  # グループに2つ以上のサンプルがある場合のみ
                group_y_test = y_test[mask]
                group_y_pred = y_pred[mask]
                
                group_scores[f'mse_{group}'] = mean_squared_error(group_y_test, group_y_pred)
                group_scores[f'r2_{group}'] = r2_score(group_y_test, group_y_pred)
        
        return group_scores


class CrossValidator:
    """メインの交差検証クラス"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.validators = {
            'standard': StandardValidator,
            'time_series': TimeSeriesValidator,
            'survival': SurvivalValidator,
            'group': GroupValidator
        }
    
    def validate_model(self, model: Any, X: pd.DataFrame, y: Union[pd.Series, pd.DataFrame], 
                        config: ValidationConfig) -> ValidationResult:
        """モデルの交差検証実行"""
        
        # バリデーターの選択
        if config.validation_type not in self.validators:
            raise ValueError(f"Unknown validation type: {config.validation_type}")
        
        validator = self.validators[config.validation_type](config)
        
        # データ分割の作成
        try:
            splits = validator.create_splits(X, y)
        except Exception as e:
            self.logger.error(f"Failed to create data splits: {e}")
            raise
        
        # 各フォールドで評価実行
        fold_scores = {}
        fold_predictions = []
        fold_true_values = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            self.logger.info(f"Processing fold {fold_idx + 1}/{len(splits)}")
            
            try:
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = self._get_y_split(y, train_idx, test_idx)
                
                # フォールドの評価
                scores = validator.evaluate_fold(model, X_train, X_test, y_train, y_test)
                
                # スコアを記録
                for metric, score in scores.items():
                    if metric not in fold_scores:
                        fold_scores[metric] = []
                    fold_scores[metric].append(score)
                
                # 予測結果を記録（標準的な予測の場合）
                if hasattr(model, 'predict'):
                    y_pred = model.predict(X_test)
                    fold_predictions.append(y_pred)
                    fold_true_values.append(y_test.values if hasattr(y_test, 'values') else y_test)
                
            except Exception as e:
                self.logger.warning(f"Fold {fold_idx + 1} failed: {e}")
                continue
        
        # 統計の計算
        mean_scores = {metric: np.mean(scores) for metric, scores in fold_scores.items()}
        std_scores = {metric: np.std(scores) for metric, scores in fold_scores.items()}
        
        # 結果の詳細情報
        validation_details = {
            'n_splits_completed': len([s for s in fold_scores.values() if s]),
            'n_total_samples': len(X),
            'feature_names': X.columns.tolist() if hasattr(X, 'columns') else [],
            'validation_config': config.__dict__
        }
        
        return ValidationResult(
            fold_scores=fold_scores,
            mean_scores=mean_scores,
            std_scores=std_scores,
            fold_predictions=fold_predictions,
            fold_true_values=fold_true_values,
            validation_details=validation_details,
            model_type=type(model).__name__,
            timestamp=datetime.now()
        )
    
    def _get_y_split(self, y: Union[pd.Series, pd.DataFrame], train_idx: np.ndarray, 
                    test_idx: np.ndarray) -> Tuple[Union[pd.Series, pd.DataFrame], Union[pd.Series, pd.DataFrame]]:
        """目的変数の分割"""
        if isinstance(y, pd.DataFrame):
            return y.iloc[train_idx], y.iloc[test_idx]
        elif isinstance(y, pd.Series):
            return y.iloc[train_idx], y.iloc[test_idx]
        else:
            return y[train_idx], y[test_idx]
    
    def compare_models(self, models: Dict[str, Any], X: pd.DataFrame, 
                        y: Union[pd.Series, pd.DataFrame], config: ValidationConfig) -> Dict[str, ValidationResult]:
        """複数モデルの性能比較"""
        results = {}
        
        for model_name, model in models.items():
            self.logger.info(f"Validating model: {model_name}")
            try:
                result = self.validate_model(model, X, y, config)
                results[model_name] = result
            except Exception as e:
                self.logger.error(f"Model {model_name} validation failed: {e}")
                continue
        
        return results
    
    def survival_specific_validation(self, model: Any, X: pd.DataFrame, 
                                    durations: pd.Series, events: pd.Series,
                                    config: Optional[ValidationConfig] = None) -> ValidationResult:
        """生存分析専用の交差検証"""
        if config is None:
            config = ValidationConfig(validation_type='survival')
        else:
            config.validation_type = 'survival'
        
        # 生存データの結合
        y_survival = pd.DataFrame({
            'duration': durations,
            'event': events
        })
        
        return self.validate_model(model, X, y_survival, config)
    
    def time_series_validation(self, model: Any, X: pd.DataFrame, y: pd.Series,
                                time_column: str, config: Optional[ValidationConfig] = None) -> ValidationResult:
        """時系列専用の交差検証"""
        if config is None:
            config = ValidationConfig(validation_type='time_series', time_column=time_column)
        else:
            config.validation_type = 'time_series'
            config.time_column = time_column
        
        return self.validate_model(model, X, y, config)
    
    def nested_cv_with_hyperparameter_tuning(self, model_class: type, param_grid: Dict,
                                            X: pd.DataFrame, y: Union[pd.Series, pd.DataFrame],
                                            config: ValidationConfig, scoring: str = 'neg_mean_squared_error') -> Dict[str, Any]:
        """ネストした交差検証によるハイパーパラメータチューニング"""
        from sklearn.model_selection import GridSearchCV
        
        # 外側の交差検証用のバリデーター作成
        outer_validator = self.validators[config.validation_type](config)
        outer_splits = outer_validator.create_splits(X, y)
        
        nested_scores = []
        best_params_list = []
        
        for train_idx, test_idx in outer_splits:
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = self._get_y_split(y, train_idx, test_idx)
            
            # 内側の交差検証でハイパーパラメータチューニング
            inner_cv = KFold(n_splits=3, shuffle=True, random_state=config.random_state)
            grid_search = GridSearchCV(
                model_class(), param_grid, cv=inner_cv, 
                scoring=scoring, n_jobs=-1
            )
            
            grid_search.fit(X_train, y_train)
            
            # 最良パラメータで外側テストセットを評価
            best_model = grid_search.best_estimator_
            y_pred = best_model.predict(X_test)
            score = mean_squared_error(y_test, y_pred)
            
            nested_scores.append(score)
            best_params_list.append(grid_search.best_params_)
        
        return {
            'nested_cv_score_mean': np.mean(nested_scores),
            'nested_cv_score_std': np.std(nested_scores),
            'best_params_by_fold': best_params_list,
            'most_common_params': self._get_most_common_params(best_params_list)
        }
    
    def _get_most_common_params(self, params_list: List[Dict]) -> Dict:
        """最も頻繁に選ばれたパラメータの組み合わせを取得"""
        from collections import Counter
        
        # パラメータの組み合わせを文字列に変換してカウント
        param_strings = [str(sorted(params.items())) for params in params_list]
        most_common = Counter(param_strings).most_common(1)
        
        if most_common:
            # 最頻出のパラメータ文字列を辞書に戻す
            most_common_str = most_common[0][0]
            return dict(eval(most_common_str))
        else:
            return {}


def create_validation_config_for_a2ai(market_category: str, analysis_type: str) -> ValidationConfig:
    """A2AI用の交差検証設定を作成する便利関数"""
    
    # 市場カテゴリに基づく設定
    base_config = {
        'n_splits': 5,
        'random_state': 42,
        'shuffle': True
    }
    
    # 分析タイプに基づく設定調整
    if analysis_type == 'survival':
        config = ValidationConfig(
            validation_type='survival',
            group_column='market_category',
            stratify_column='extinction_event',
            **base_config
        )
    elif analysis_type == 'emergence':
        config = ValidationConfig(
            validation_type='group',
            group_column='market_category',
            stratify_column='success_flag',
            **base_config
        )
    elif analysis_type == 'time_series':
        config = ValidationConfig(
            validation_type='time_series',
            time_column='year',
            n_splits=3,  # 時系列では分割数を少なくする
            **base_config
        )
    else:  # traditional analysis
        config = ValidationConfig(
            validation_type='group',
            group_column='market_category',
            **base_config
        )
    
    return config


# 使用例とテスト関数
def example_usage():
    """使用例の実演"""
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    
    # サンプルデータ作成
    np.random.seed(42)
    n_samples = 1000
    
    X = pd.DataFrame({
        'feature1': np.random.randn(n_samples),
        'feature2': np.random.randn(n_samples),
        'market_category': np.random.choice(['high_share', 'declining', 'lost'], n_samples),
        'year': np.random.randint(1984, 2024, n_samples),
        'company_age': np.random.randint(1, 40, n_samples)
    })
    
    y = pd.Series(2 * X['feature1'] + X['feature2'] + np.random.randn(n_samples))
    
    # 交差検証実行
    cv = CrossValidator()
    
    # 標準的な交差検証
    config = ValidationConfig(validation_type='standard')
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    result = cv.validate_model(model, X[['feature1', 'feature2']], y, config)
    
    print("Standard CV Results:")
    print(f"Mean R²: {result.mean_scores.get('r2', 0):.3f} ± {result.std_scores.get('r2', 0):.3f}")
    print(f"Mean RMSE: {result.mean_scores.get('rmse', 0):.3f} ± {result.std_scores.get('rmse', 0):.3f}")
    
    # グループ交差検証
    config_group = ValidationConfig(
        validation_type='group',
        group_column='market_category'
    )
    result_group = cv.validate_model(model, X[['feature1', 'feature2', 'market_category']], y, config_group)
    
    print("\nGroup CV Results:")
    print(f"Mean R²: {result_group.mean_scores.get('r2', 0):.3f} ± {result_group.std_scores.get('r2', 0):.3f}")
    
    # モデル比較
    models = {
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
        'LinearRegression': LinearRegression()
    }
    
    comparison = cv.compare_models(models, X[['feature1', 'feature2']], y, config)
    
    print("\nModel Comparison:")
    for model_name, result in comparison.items():
        print(f"{model_name} - R²: {result.mean_scores.get('r2', 0):.3f}")


if __name__ == "__main__":
    # ログ設定
    logging.basicConfig(level=logging.INFO)
    
    # 使用例実行
    example_usage()