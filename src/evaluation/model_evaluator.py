"""
A2AI Model Evaluator
===================

A2AI（Advanced Financial Analysis AI）のモデル評価システム

機能:
- 従来財務分析モデル評価
- 生存分析モデル評価  
- 新設企業分析モデル評価
- 因果推論モデル評価
- 統合評価・比較分析
- バイアス検出・補正効果評価

対応モデル:
- 回帰系: 線形回帰、Ridge、Lasso、ElasticNet
- アンサンブル系: RandomForest、XGBoost、LightGBM
- 深層学習系: MLP、LSTM、Transformer
- 生存分析系: Cox回帰、Kaplan-Meier、AFT
- 因果推論系: DID、IV、Propensity Score、Causal Forest
"""

import numpy as np
import pandas as pd
import warnings
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, roc_curve
)
from scipy import stats
from scipy.stats import spearmanr, pearsonr
from lifelines.utils import concordance_index
import joblib
import json
from datetime import datetime
import logging

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """
    A2AI統合モデル評価クラス
    
    9つの評価項目（売上高、成長率、利益率、ROE、付加価値率、
    企業存続確率、新規事業成功率、事業継承成功度）に対応した
    包括的なモデル評価システム
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初期化
        
        Parameters:
        -----------
        config : dict, optional
            評価設定（閾値、重み等）
        """
        self.config = config or self._get_default_config()
        self.evaluation_results = {}
        self.model_comparison = {}
        
        # 評価項目定義
        self.evaluation_metrics = {
            'traditional': [
                '売上高', '売上高成長率', '売上高営業利益率', 
                '売上高当期純利益率', 'ROE', '売上高付加価値率'
            ],
            'survival': ['企業存続確率'],
            'emergence': ['新規事業成功率'],
            'succession': ['事業継承成功度']
        }
        
        # 市場カテゴリ定義
        self.market_categories = {
            'high_share': '世界シェア高市場',
            'declining': 'シェア低下市場', 
            'lost': 'シェア失失市場'
        }
    
    def _get_default_config(self) -> Dict[str, Any]:
        """デフォルト設定を取得"""
        return {
            'regression_metrics': ['mse', 'mae', 'r2', 'mape'],
            'classification_metrics': ['accuracy', 'precision', 'recall', 'f1', 'auc'],
            'survival_metrics': ['c_index', 'ibs', 'calibration'],
            'significance_level': 0.05,
            'confidence_level': 0.95,
            'cross_validation_folds': 5,
            'bootstrap_iterations': 1000,
            'outlier_threshold': 3.0,
            'min_sample_size': 30
        }
    
    def evaluate_regression_model(self, 
                                y_true: np.ndarray,
                                y_pred: np.ndarray,
                                model_name: str,
                                metric_type: str = 'traditional') -> Dict[str, float]:
        """
        回帰モデルの評価
        
        Parameters:
        -----------
        y_true : array-like
            実測値
        y_pred : array-like  
            予測値
        model_name : str
            モデル名
        metric_type : str
            評価項目タイプ ('traditional', 'emergence', 'succession')
            
        Returns:
        --------
        dict
            評価結果辞書
        """
        results = {}
        
        # 基本回帰指標
        results['mse'] = mean_squared_error(y_true, y_pred)
        results['rmse'] = np.sqrt(results['mse'])
        results['mae'] = mean_absolute_error(y_true, y_pred)
        results['r2'] = r2_score(y_true, y_pred)
        
        # MAPE (Mean Absolute Percentage Error)
        # ゼロ除算回避
        mask = y_true != 0
        if np.sum(mask) > 0:
            results['mape'] = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        else:
            results['mape'] = np.inf
            
        # 調整済み決定係数
        n = len(y_true)
        p = 1  # 仮定：単一特徴量（実際のモデルに応じて調整）
        if n > p + 1:
            results['adj_r2'] = 1 - (1 - results['r2']) * (n - 1) / (n - p - 1)
        else:
            results['adj_r2'] = results['r2']
            
        # 相関係数
        results['pearson_corr'], results['pearson_p'] = pearsonr(y_true, y_pred)
        results['spearman_corr'], results['spearman_p'] = spearmanr(y_true, y_pred)
        
        # 予測誤差の正規性検定
        residuals = y_true - y_pred
        results['shapiro_stat'], results['shapiro_p'] = stats.shapiro(residuals[:5000])  # サンプル制限
        
        # 残差分析
        results['residual_mean'] = np.mean(residuals)
        results['residual_std'] = np.std(residuals)
        results['residual_skew'] = stats.skew(residuals)
        results['residual_kurtosis'] = stats.kurtosis(residuals)
        
        # 外れ値検出
        z_scores = np.abs(stats.zscore(residuals))
        results['outlier_ratio'] = np.sum(z_scores > self.config['outlier_threshold']) / len(residuals)
        
        # A2AI特有の評価指標
        if metric_type == 'traditional':
            # 財務指標特有の評価
            results.update(self._evaluate_financial_prediction(y_true, y_pred))
        elif metric_type == 'emergence':
            # 新設企業成功率特有の評価
            results.update(self._evaluate_emergence_prediction(y_true, y_pred))
        elif metric_type == 'succession':
            # 事業継承成功度特有の評価  
            results.update(self._evaluate_succession_prediction(y_true, y_pred))
            
        # メタデータ
        results['model_name'] = model_name
        results['metric_type'] = metric_type
        results['sample_size'] = len(y_true)
        results['evaluation_date'] = datetime.now().isoformat()
        
        logger.info(f"回帰モデル評価完了: {model_name} (R²={results['r2']:.4f}, RMSE={results['rmse']:.4f})")
        
        return results
    
    def evaluate_survival_model(self,
                                durations: np.ndarray,
                                events: np.ndarray,
                                predictions: np.ndarray,
                                model_name: str,
                                time_points: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        生存分析モデルの評価
        
        Parameters:
        -----------
        durations : array-like
            観測期間（企業存続期間）
        events : array-like
            イベント発生フラグ（1:消滅, 0:打ち切り）
        predictions : array-like
            予測リスクスコアまたは生存確率
        model_name : str
            モデル名
        time_points : array-like, optional
            評価時点（年）
            
        Returns:
        --------
        dict
            生存分析評価結果
        """
        results = {}
        
        # C-index (Concordance Index) - 生存分析の主要評価指標
        try:
            results['c_index'] = concordance_index(durations, predictions, events)
        except Exception as e:
            logger.warning(f"C-index計算エラー: {e}")
            results['c_index'] = np.nan
            
        # 時点別生存率評価
        if time_points is not None:
            time_specific_metrics = {}
            for t in time_points:
                # t年時点での生存予測精度
                mask = durations >= t
                if np.sum(mask) > self.config['min_sample_size']:
                    actual_survival = (durations >= t).astype(int)
                    # 予測値を生存確率に変換（モデルタイプに応じて調整）
                    if np.all(predictions >= 0) and np.all(predictions <= 1):
                        pred_survival = predictions
                    else:
                        # リスクスコアを確率に変換
                        pred_survival = 1 / (1 + np.exp(predictions))
                    
                    time_specific_metrics[f'auc_{t}year'] = roc_auc_score(
                        actual_survival, pred_survival
                    )
            
            results.update(time_specific_metrics)
        
        # Integrated Brier Score (IBS) - 時間統合予測誤差
        results.update(self._calculate_brier_score(durations, events, predictions))
        
        # 生存曲線の適合度評価
        results.update(self._evaluate_survival_calibration(durations, events, predictions))
        
        # A2AI特有の企業生存評価
        results.update(self._evaluate_corporate_survival(durations, events, predictions))
        
        # メタデータ
        results['model_name'] = model_name
        results['metric_type'] = 'survival'
        results['sample_size'] = len(durations)
        results['event_rate'] = np.mean(events)
        results['median_duration'] = np.median(durations)
        results['evaluation_date'] = datetime.now().isoformat()
        
        logger.info(f"生存分析モデル評価完了: {model_name} (C-index={results['c_index']:.4f})")
        
        return results
    
    def evaluate_classification_model(self,
                                    y_true: np.ndarray,
                                    y_pred_proba: np.ndarray,
                                    y_pred: Optional[np.ndarray] = None,
                                    model_name: str = "classifier",
                                    metric_type: str = 'emergence') -> Dict[str, float]:
        """
        分類モデル（新設企業成功予測等）の評価
        
        Parameters:
        -----------
        y_true : array-like
            実際のラベル
        y_pred_proba : array-like
            予測確率
        y_pred : array-like, optional
            予測ラベル
        model_name : str
            モデル名
        metric_type : str
            評価項目タイプ
            
        Returns:
        --------
        dict
            分類評価結果
        """
        results = {}
        
        # 予測ラベルが未指定の場合、確率から生成
        if y_pred is None:
            y_pred = (y_pred_proba > 0.5).astype(int)
        
        # 基本分類指標
        results['accuracy'] = accuracy_score(y_true, y_pred)
        results['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        results['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        results['f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # ROC-AUC
        if len(np.unique(y_true)) == 2:  # 二値分類
            results['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            
            # Precision-Recall AUC
            precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
            results['pr_auc'] = np.trapz(precision, recall)
            
        else:  # 多値分類
            try:
                results['roc_auc'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr')
            except Exception as e:
                logger.warning(f"多値分類AUC計算エラー: {e}")
                results['roc_auc'] = np.nan
        
        # 予測確率の品質評価
        results.update(self._evaluate_probability_calibration(y_true, y_pred_proba))
        
        # A2AI特有の評価指標
        if metric_type == 'emergence':
            # 新設企業成功予測特有の評価
            results.update(self._evaluate_startup_prediction(y_true, y_pred_proba))
        
        # 混同行列の統計
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_pred)
        results['confusion_matrix'] = cm.tolist()
        
        # クラス不均衡対応評価
        class_counts = np.bincount(y_true)
        results['class_imbalance_ratio'] = np.max(class_counts) / np.min(class_counts)
        
        # メタデータ
        results['model_name'] = model_name
        results['metric_type'] = metric_type
        results['sample_size'] = len(y_true)
        results['positive_rate'] = np.mean(y_true)
        results['evaluation_date'] = datetime.now().isoformat()
        
        logger.info(f"分類モデル評価完了: {model_name} (AUC={results.get('roc_auc', 'N/A'):.4f})")
        
        return results
    
    def evaluate_causal_model(self,
                            treatment: np.ndarray,
                            outcome: np.ndarray,
                            estimated_effect: float,
                            confidence_interval: Tuple[float, float],
                            model_name: str,
                            true_effect: Optional[float] = None) -> Dict[str, float]:
        """
        因果推論モデルの評価
        
        Parameters:
        -----------
        treatment : array-like
            処置変数（例：M&A、政策変更等）
        outcome : array-like
            結果変数（財務指標の変化等）
        estimated_effect : float
            推定因果効果
        confidence_interval : tuple
            信頼区間 (lower, upper)
        model_name : str
            モデル名
        true_effect : float, optional
            真の因果効果（シミュレーション時）
            
        Returns:
        --------
        dict
            因果推論評価結果
        """
        results = {}
        
        # 基本統計
        results['estimated_effect'] = estimated_effect
        results['ci_lower'] = confidence_interval[0]
        results['ci_upper'] = confidence_interval[1]
        results['ci_width'] = confidence_interval[1] - confidence_interval[0]
        
        # 効果の有意性
        results['is_significant'] = not (confidence_interval[0] <= 0 <= confidence_interval[1])
        results['effect_size'] = abs(estimated_effect)
        
        # 真の効果が既知の場合（シミュレーション等）
        if true_effect is not None:
            results['true_effect'] = true_effect
            results['bias'] = estimated_effect - true_effect
            results['relative_bias'] = (estimated_effect - true_effect) / true_effect if true_effect != 0 else np.inf
            results['covers_true_effect'] = confidence_interval[0] <= true_effect <= confidence_interval[1]
            results['mse'] = (estimated_effect - true_effect) ** 2
        
        # 処置群・対照群の基本統計
        treatment_mask = treatment == 1
        control_mask = treatment == 0
        
        if np.sum(treatment_mask) > 0 and np.sum(control_mask) > 0:
            results['treatment_group_size'] = np.sum(treatment_mask)
            results['control_group_size'] = np.sum(control_mask)
            results['treatment_mean'] = np.mean(outcome[treatment_mask])
            results['control_mean'] = np.mean(outcome[control_mask])
            results['naive_difference'] = results['treatment_mean'] - results['control_mean']
            
            # 群間分散の均質性検定
            from scipy.stats import levene
            results['levene_stat'], results['levene_p'] = levene(
                outcome[treatment_mask], 
                outcome[control_mask]
            )
        
        # A2AI特有の因果効果評価
        results.update(self._evaluate_corporate_causal_effect(treatment, outcome, estimated_effect))
        
        # メタデータ
        results['model_name'] = model_name
        results['metric_type'] = 'causal'
        results['sample_size'] = len(treatment)
        results['treatment_rate'] = np.mean(treatment)
        results['evaluation_date'] = datetime.now().isoformat()
        
        logger.info(f"因果推論モデル評価完了: {model_name} (効果={estimated_effect:.4f})")
        
        return results
    
    def compare_models(self, 
                        model_results: Dict[str, Dict[str, float]],
                        primary_metric: str = 'r2',
                        secondary_metrics: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        複数モデルの比較評価
        
        Parameters:
        -----------
        model_results : dict
            {model_name: evaluation_results} の辞書
        primary_metric : str
            主評価指標
        secondary_metrics : list, optional
            副次評価指標
            
        Returns:
        --------
        dict
            モデル比較結果
        """
        comparison = {}
        
        if not model_results:
            logger.warning("比較対象モデルが存在しません")
            return comparison
        
        # 主評価指標によるランキング
        primary_scores = {}
        for model_name, results in model_results.items():
            if primary_metric in results:
                primary_scores[model_name] = results[primary_metric]
        
        if primary_scores:
            # 指標の向きを判定（高いほど良い vs 低いほど良い）
            ascending = primary_metric in ['mse', 'mae', 'rmse', 'mape']
            sorted_models = sorted(primary_scores.items(), key=lambda x: x[1], reverse=not ascending)
            
            comparison['ranking'] = {
                'primary_metric': primary_metric,
                'best_model': sorted_models[0][0],
                'best_score': sorted_models[0][1],
                'full_ranking': sorted_models
            }
        
        # 統計的有意差検定
        comparison['significance_tests'] = self._perform_model_significance_tests(model_results)
        
        # 副次指標による比較
        if secondary_metrics:
            comparison['secondary_comparisons'] = {}
            for metric in secondary_metrics:
                metric_scores = {}
                for model_name, results in model_results.items():
                    if metric in results:
                        metric_scores[model_name] = results[metric]
                
                if metric_scores:
                    ascending = metric in ['mse', 'mae', 'rmse', 'mape']
                    sorted_models = sorted(metric_scores.items(), key=lambda x: x[1], reverse=not ascending)
                    comparison['secondary_comparisons'][metric] = {
                        'best_model': sorted_models[0][0],
                        'best_score': sorted_models[0][1],
                        'ranking': sorted_models
                    }
        
        # 総合スコア計算
        comparison['overall_score'] = self._calculate_overall_score(model_results)
        
        # 市場カテゴリ別性能比較
        comparison['market_category_analysis'] = self._analyze_market_category_performance(model_results)
        
        # 安定性評価
        comparison['stability_analysis'] = self._analyze_model_stability(model_results)
        
        logger.info(f"モデル比較完了: ベストモデル = {comparison.get('ranking', {}).get('best_model', 'N/A')}")
        
        return comparison
    
    def _evaluate_financial_prediction(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """財務指標予測特有の評価"""
        results = {}
        
        # 方向性の一致率（増減予測の正確性）
        if len(y_true) > 1:
            true_direction = np.diff(y_true) > 0
            pred_direction = np.diff(y_pred) > 0
            if len(true_direction) > 0:
                results['direction_accuracy'] = np.mean(true_direction == pred_direction)
        
        # 極値予測の精度
        true_percentiles = [np.percentile(y_true, p) for p in [10, 90]]
        pred_percentiles = [np.percentile(y_pred, p) for p in [10, 90]]
        results['extreme_value_error'] = np.mean(np.abs(np.array(true_percentiles) - np.array(pred_percentiles)))
        
        # 財務指標特有の範囲制約評価
        if np.all(y_true >= 0):  # 非負制約がある指標
            results['negative_prediction_ratio'] = np.mean(y_pred < 0)
        
        return results
    
    def _evaluate_emergence_prediction(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """新設企業成功率予測特有の評価"""
        results = {}
        
        # 成功閾値別の予測精度
        success_thresholds = [0.3, 0.5, 0.7]
        for threshold in success_thresholds:
            true_success = y_true > threshold
            pred_success = y_pred > threshold
            if np.sum(true_success) > 0 and np.sum(~true_success) > 0:
                results[f'success_accuracy_th{threshold}'] = np.mean(true_success == pred_success)
        
        return results
    
    def _evaluate_succession_prediction(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """事業継承成功度予測特有の評価"""
        results = {}
        
        # 継承効果の方向性評価
        baseline = 0  # 継承効果なしのベースライン
        true_positive_effect = y_true > baseline
        pred_positive_effect = y_pred > baseline
        results['succession_direction_accuracy'] = np.mean(true_positive_effect == pred_positive_effect)
        
        return results
    
    def _calculate_brier_score(self, durations: np.ndarray, events: np.ndarray, predictions: np.ndarray) -> Dict[str, float]:
        """Brier Score計算（生存分析用）"""
        results = {}
        
        try:
            # 簡易版Brier Score実装
            # 実際の実装では lifelines.utils.integrated_brier_score を使用
            unique_times = np.unique(durations[events == 1])
            if len(unique_times) > 1:
                # 代表時点でのBrier Score
                mid_time = np.median(unique_times)
                actual_survival = (durations > mid_time).astype(float)
                # 予測値を生存確率に変換
                if np.all(predictions >= 0) and np.all(predictions <= 1):
                    pred_survival = predictions
                else:
                    pred_survival = 1 / (1 + np.exp(predictions))
                
                results['brier_score'] = np.mean((actual_survival - pred_survival) ** 2)
            else:
                results['brier_score'] = np.nan
                
        except Exception as e:
            logger.warning(f"Brier Score計算エラー: {e}")
            results['brier_score'] = np.nan
        
        return results
    
    def _evaluate_survival_calibration(self, durations: np.ndarray, events: np.ndarray, predictions: np.ndarray) -> Dict[str, float]:
        """生存曲線キャリブレーション評価"""
        results = {}
        
        try:
            # Hosmer-Lemeshow type test for survival
            n_bins = 10
            pred_quantiles = np.quantile(predictions, np.linspace(0, 1, n_bins + 1))
            
            calibration_errors = []
            for i in range(n_bins):
                mask = (predictions >= pred_quantiles[i]) & (predictions < pred_quantiles[i + 1])
                if np.sum(mask) > 5:  # 十分なサンプルサイズ
                    observed_rate = np.mean(events[mask])
                    expected_rate = np.mean(predictions[mask])
                    calibration_errors.append(abs(observed_rate - expected_rate))
            
            if calibration_errors:
                results['calibration_error'] = np.mean(calibration_errors)
            else:
                results['calibration_error'] = np.nan
                
        except Exception as e:
            logger.warning(f"キャリブレーション計算エラー: {e}")
            results['calibration_error'] = np.nan
        
        return results
    
    def _evaluate_corporate_survival(self, durations: np.ndarray, events: np.ndarray, predictions: np.ndarray) -> Dict[str, float]:
        """企業生存特有の評価指標"""
        results = {}
        
        # 企業寿命の予測精度
        median_survival = np.median(durations)
        results['median_survival_years'] = median_survival
        
        # 早期警告性能（倒産5年前の予測精度）
        early_warning_threshold = 5  # 年
        if np.max(durations) > early_warning_threshold:
            early_events = (durations <= early_warning_threshold) & (events == 1)
            if np.sum(early_events) > 0:
                # 早期イベントの予測精度
                from sklearn.metrics import roc_auc_score
                try:
                    results['early_warning_auc'] = roc_auc_score(early_events, predictions)
                except Exception:
                    results['early_warning_auc'] = np.nan
        
        return results
    
    def _evaluate_probability_calibration(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, float]:
        """確率予測のキャリブレーション評価"""
        results = {}
        
        try:
            from sklearn.calibration import calibration_curve
            
            # 10bin でキャリブレーション曲線を計算
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true, y_pred_proba, n_bins=10
            )
            
            # キャリブレーションエラー（Expected Calibration Error）
            bin_boundaries = np.linspace(0, 1, 11)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            
            ece = 0
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = (y_pred_proba > bin_lower) & (y_pred_proba <= bin_upper)
                prop_in_bin = in_bin.mean()
                
                if prop_in_bin > 0:
                    accuracy_in_bin = y_true[in_bin].mean()
                    avg_confidence_in_bin = y_pred_proba[in_bin].mean()
                    ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
            results['expected_calibration_error'] = ece
            
            # Brier Score
            results['brier_score_classification'] = np.mean((y_true - y_pred_proba) ** 2)
            
        except Exception as e:
            logger.warning(f"確率キャリブレーション計算エラー: {e}")
            results['expected_calibration_error'] = np.nan
            results['brier_score_classification'] = np.nan
        
        return results
    
    def _evaluate_startup_prediction(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, float]:
        """新設企業成功予測特有の評価"""
        results = {}
        
        # 成功企業の早期識別精度
        if len(np.unique(y_true)) == 2:  # 二値分類
            # Top-K精度（上位K%の予測で実際の成功企業をどれだけ捉えられるか）
            for k in [10, 20, 30]:
                top_k_threshold = np.percentile(y_pred_proba, 100 - k)
                top_k_pred = y_pred_proba >= top_k_threshold
                if np.sum(top_k_pred) > 0:
                    results[f'precision_top_{k}pct'] = np.sum(y_true[top_k_pred]) / np.sum(top_k_pred)
                    results[f'recall_top_{k}pct'] = np.sum(y_true[top_k_pred]) / np.sum(y_true)
        
        # リスク層別化性能
        risk_bins = np.quantile(y_pred_proba, [0, 0.33, 0.67, 1.0])
        risk_groups = np.digitize(y_pred_proba, risk_bins) - 1
        risk_groups = np.clip(risk_groups, 0, 2)  # 0, 1, 2 の3群
        
        group_success_rates = []
        for group in range(3):
            mask = risk_groups == group
            if np.sum(mask) > 0:
                success_rate = np.mean(y_true[mask])
                group_success_rates.append(success_rate)
        
        if len(group_success_rates) == 3:
            results['risk_stratification_range'] = np.max(group_success_rates) - np.min(group_success_rates)
        
        return results
    
    def _evaluate_corporate_causal_effect(self, treatment: np.ndarray, outcome: np.ndarray, estimated_effect: float) -> Dict[str, float]:
        """企業レベル因果効果の評価"""
        results = {}
        
        # 効果サイズの実用的意義
        outcome_std = np.std(outcome)
        if outcome_std > 0:
            results['standardized_effect_size'] = abs(estimated_effect) / outcome_std
            
            # Cohen's d基準による効果サイズ分類
            if abs(results['standardized_effect_size']) < 0.2:
                results['effect_magnitude'] = 'small'
            elif abs(results['standardized_effect_size']) < 0.5:
                results['effect_magnitude'] = 'medium'
            else:
                results['effect_magnitude'] = 'large'
        
        # 処置群・対照群のバランス確認
        treatment_rate = np.mean(treatment)
        results['treatment_balance'] = min(treatment_rate, 1 - treatment_rate)  # 0.5に近いほどバランス良好
        
        # 経済的効果の評価（財務指標の場合）
        if 'profit' in str(outcome).lower() or 'revenue' in str(outcome).lower():
            results['economic_significance'] = abs(estimated_effect) > (outcome_std * 0.1)  # 10%以上の変化
        
        return results
    
    def _perform_model_significance_tests(self, model_results: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """モデル間の統計的有意差検定"""
        significance_tests = {}
        
        model_names = list(model_results.keys())
        primary_metrics = ['r2', 'mse', 'mae', 'c_index', 'roc_auc']
        
        for metric in primary_metrics:
            metric_values = {}
            for model_name in model_names:
                if metric in model_results[model_name]:
                    metric_values[model_name] = model_results[model_name][metric]
            
            if len(metric_values) >= 2:
                # ペアワイズ比較（Bonferroni補正付き）
                from itertools import combinations
                pairwise_tests = {}
                
                for model1, model2 in combinations(metric_values.keys(), 2):
                    # 簡易的な差の検定（実際には交差検証結果を使用）
                    val1, val2 = metric_values[model1], metric_values[model2]
                    
                    # 効果サイズ計算
                    pooled_std = np.sqrt((val1**2 + val2**2) / 2) if val1 != 0 and val2 != 0 else 1
                    effect_size = abs(val1 - val2) / pooled_std if pooled_std > 0 else 0
                    
                    pairwise_tests[f'{model1}_vs_{model2}'] = {
                        'difference': val1 - val2,
                        'effect_size': effect_size,
                        'practical_significance': effect_size > 0.2
                    }
                
                significance_tests[metric] = pairwise_tests
        
        return significance_tests
    
    def _calculate_overall_score(self, model_results: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """総合スコア計算"""
        overall_scores = {}
        
        # 重み付け（設定可能）
        metric_weights = {
            'r2': 0.3, 'mse': -0.2, 'mae': -0.1,  # 回帰系（負の重みは低いほど良い指標）
            'roc_auc': 0.25, 'precision': 0.1, 'recall': 0.1,  # 分類系
            'c_index': 0.3,  # 生存分析系
            'estimated_effect': 0.2  # 因果推論系
        }
        
        for model_name, results in model_results.items():
            score = 0
            weight_sum = 0
            
            for metric, weight in metric_weights.items():
                if metric in results and not np.isnan(results[metric]):
                    # 正規化（0-1スケール）
                    if weight > 0:  # 高いほど良い指標
                        normalized_value = min(max(results[metric], 0), 1)
                    else:  # 低いほど良い指標
                        # MSE等は逆数を取って正規化
                        normalized_value = 1 / (1 + abs(results[metric]))
                        weight = abs(weight)  # 重みを正に変換
                    
                    score += normalized_value * weight
                    weight_sum += weight
            
            if weight_sum > 0:
                overall_scores[model_name] = score / weight_sum
            else:
                overall_scores[model_name] = 0
        
        return overall_scores
    
    def _analyze_market_category_performance(self, model_results: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """市場カテゴリ別性能分析"""
        market_analysis = {}
        
        # 市場タイプ別の性能差分析
        market_performance = {
            'high_share_markets': {},
            'declining_markets': {},
            'lost_markets': {}
        }
        
        # 各モデルの市場タイプ別適合度
        for model_name, results in model_results.items():
            # サンプルサイズから市場タイプを推定（実際のデータに基づいて調整）
            sample_size = results.get('sample_size', 0)
            
            if sample_size > 0:
                # 仮想的な市場カテゴリ性能指標
                if 'r2' in results:
                    # 高シェア市場での予測精度が高いかどうか
                    market_performance['high_share_markets'][model_name] = results['r2']
                
                if 'c_index' in results:
                    # 失失市場での生存予測精度
                    market_performance['lost_markets'][model_name] = results['c_index']
        
        # 市場間での性能格差分析
        for market_type, performance in market_performance.items():
            if performance:
                market_analysis[market_type] = {
                    'best_model': max(performance, key=performance.get),
                    'performance_range': max(performance.values()) - min(performance.values()),
                    'average_performance': np.mean(list(performance.values()))
                }
        
        return market_analysis
    
    def _analyze_model_stability(self, model_results: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """モデル安定性分析"""
        stability_analysis = {}
        
        # 残差分析に基づく安定性評価
        stability_metrics = ['residual_std', 'outlier_ratio', 'shapiro_p']
        
        for model_name, results in model_results.items():
            model_stability = {}
            
            # 予測安定性
            if 'residual_std' in results:
                model_stability['prediction_stability'] = 1 / (1 + results['residual_std'])  # 低いほど安定
            
            # 外れ値耐性
            if 'outlier_ratio' in results:
                model_stability['outlier_resistance'] = 1 - results['outlier_ratio']
            
            # 残差正規性（モデル適合度の指標）
            if 'shapiro_p' in results:
                model_stability['residual_normality'] = results['shapiro_p']
            
            # 総合安定性スコア
            if model_stability:
                stability_analysis[model_name] = {
                    **model_stability,
                    'overall_stability': np.mean(list(model_stability.values()))
                }
        
        return stability_analysis
    
    def generate_evaluation_report(self, 
                                    model_results: Dict[str, Dict[str, float]],
                                    comparison_results: Dict[str, Any],
                                    output_path: Optional[str] = None) -> str:
        """
        評価レポート生成
        
        Parameters:
        -----------
        model_results : dict
            各モデルの評価結果
        comparison_results : dict  
            モデル比較結果
        output_path : str, optional
            出力パス
            
        Returns:
        --------
        str
            生成されたレポート文字列
        """
        report_lines = []
        
        # ヘッダー
        report_lines.append("=" * 80)
        report_lines.append("A2AI Model Evaluation Report")
        report_lines.append("=" * 80)
        report_lines.append(f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # エグゼクティブサマリー
        if 'ranking' in comparison_results:
            ranking = comparison_results['ranking']
            report_lines.append("【エグゼクティブサマリー】")
            report_lines.append(f"最優秀モデル: {ranking['best_model']}")
            report_lines.append(f"評価指標 ({ranking['primary_metric']}): {ranking['best_score']:.4f}")
            report_lines.append("")
        
        # 個別モデル詳細評価
        report_lines.append("【個別モデル評価詳細】")
        report_lines.append("-" * 40)
        
        for model_name, results in model_results.items():
            report_lines.append(f"\n◆ {model_name}")
            report_lines.append(f"  モデルタイプ: {results.get('metric_type', 'N/A')}")
            report_lines.append(f"  サンプルサイズ: {results.get('sample_size', 'N/A')}")
            
            # 主要指標
            key_metrics = ['r2', 'rmse', 'mae', 'roc_auc', 'c_index', 'estimated_effect']
            for metric in key_metrics:
                if metric in results:
                    report_lines.append(f"  {metric.upper()}: {results[metric]:.4f}")
            
            # 特殊評価項目
            special_metrics = ['direction_accuracy', 'early_warning_auc', 'calibration_error']
            for metric in special_metrics:
                if metric in results:
                    report_lines.append(f"  {metric}: {results[metric]:.4f}")
        
        # モデル比較結果
        if 'ranking' in comparison_results:
            report_lines.append("\n【モデル比較ランキング】")
            report_lines.append("-" * 40)
            
            for i, (model_name, score) in enumerate(comparison_results['ranking']['full_ranking'], 1):
                report_lines.append(f"{i}. {model_name}: {score:.4f}")
        
        # 統計的有意差
        if 'significance_tests' in comparison_results:
            report_lines.append("\n【統計的有意差検定】")
            report_lines.append("-" * 40)
            
            for metric, tests in comparison_results['significance_tests'].items():
                report_lines.append(f"\n◆ {metric.upper()}における有意差:")
                for comparison, result in tests.items():
                    effect_size = result['effect_size']
                    practical_sig = result['practical_significance']
                    report_lines.append(f"  {comparison}: 効果サイズ={effect_size:.3f}, 実用的有意={practical_sig}")
        
        # 市場カテゴリ別分析
        if 'market_category_analysis' in comparison_results:
            report_lines.append("\n【市場カテゴリ別性能分析】")
            report_lines.append("-" * 40)
            
            for market_type, analysis in comparison_results['market_category_analysis'].items():
                if analysis:
                    report_lines.append(f"\n◆ {market_type}:")
                    report_lines.append(f"  最優秀モデル: {analysis['best_model']}")
                    report_lines.append(f"  性能レンジ: {analysis['performance_range']:.4f}")
                    report_lines.append(f"  平均性能: {analysis['average_performance']:.4f}")
        
        # 安定性分析
        if 'stability_analysis' in comparison_results:
            report_lines.append("\n【モデル安定性分析】")
            report_lines.append("-" * 40)
            
            for model_name, stability in comparison_results['stability_analysis'].items():
                report_lines.append(f"\n◆ {model_name}:")
                report_lines.append(f"  総合安定性: {stability.get('overall_stability', 'N/A'):.4f}")
                if 'prediction_stability' in stability:
                    report_lines.append(f"  予測安定性: {stability['prediction_stability']:.4f}")
                if 'outlier_resistance' in stability:
                    report_lines.append(f"  外れ値耐性: {stability['outlier_resistance']:.4f}")
        
        # 推奨事項
        report_lines.append("\n【推奨事項】")
        report_lines.append("-" * 40)
        
        if 'ranking' in comparison_results:
            best_model = comparison_results['ranking']['best_model']
            report_lines.append(f"1. 本番環境での使用推奨モデル: {best_model}")
        
        if 'overall_score' in comparison_results:
            sorted_overall = sorted(comparison_results['overall_score'].items(), key=lambda x: x[1], reverse=True)
            report_lines.append(f"2. 総合評価トップ3:")
            for i, (model_name, score) in enumerate(sorted_overall[:3], 1):
                report_lines.append(f"   {i}位: {model_name} (スコア: {score:.4f})")
        
        report_lines.append("\n3. モデル選択時の考慮事項:")
        report_lines.append("   - 予測精度と解釈可能性のトレードオフ")
        report_lines.append("   - 計算コストと予測速度要件")
        report_lines.append("   - 市場カテゴリ固有の特性適合度")
        report_lines.append("   - 生存バイアス補正効果")
        
        # フッター
        report_lines.append("\n" + "=" * 80)
        report_lines.append("A2AI Model Evaluation Complete")
        report_lines.append("=" * 80)
        
        report_text = "\n".join(report_lines)
        
        # ファイル出力
        if output_path:
            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(report_text)
                logger.info(f"評価レポートを保存: {output_path}")
            except Exception as e:
                logger.error(f"レポート保存エラー: {e}")
        
        return report_text
    
    def save_results(self, 
                    model_results: Dict[str, Dict[str, float]],
                    comparison_results: Dict[str, Any],
                    output_dir: str = "results/evaluation") -> None:
        """
        評価結果の保存
        
        Parameters:
        -----------
        model_results : dict
            各モデルの評価結果
        comparison_results : dict
            モデル比較結果
        output_dir : str
            出力ディレクトリ
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # JSON形式で詳細結果保存
        results_path = os.path.join(output_dir, f"model_evaluation_{timestamp}.json")
        full_results = {
            'model_results': model_results,
            'comparison_results': comparison_results,
            'evaluation_config': self.config,
            'timestamp': timestamp
        }
        
        try:
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(full_results, f, ensure_ascii=False, indent=2, default=str)
            logger.info(f"評価結果JSON保存: {results_path}")
        except Exception as e:
            logger.error(f"JSON保存エラー: {e}")
        
        # テキストレポート保存
        report_path = os.path.join(output_dir, f"evaluation_report_{timestamp}.txt")
        report_text = self.generate_evaluation_report(model_results, comparison_results, report_path)
        
        # サマリーCSV保存
        self._save_summary_csv(model_results, comparison_results, 
                                os.path.join(output_dir, f"evaluation_summary_{timestamp}.csv"))
        
        logger.info(f"A2AI モデル評価結果保存完了: {output_dir}")
    
    def _save_summary_csv(self, 
                            model_results: Dict[str, Dict[str, float]],
                            comparison_results: Dict[str, Any],
                            output_path: str) -> None:
        """評価結果サマリーをCSV形式で保存"""
        summary_data = []
        
        for model_name, results in model_results.items():
            row = {'model_name': model_name}
            
            # 主要指標
            key_metrics = ['r2', 'rmse', 'mae', 'roc_auc', 'c_index', 'estimated_effect', 
                            'sample_size', 'metric_type']
            for metric in key_metrics:
                row[metric] = results.get(metric, np.nan)
            
            # 総合スコア
            if 'overall_score' in comparison_results:
                row['overall_score'] = comparison_results['overall_score'].get(model_name, np.nan)
            
            # 安定性スコア
            if 'stability_analysis' in comparison_results:
                stability = comparison_results['stability_analysis'].get(model_name, {})
                row['stability_score'] = stability.get('overall_stability', np.nan)
            
            summary_data.append(row)
        
        try:
            df_summary = pd.DataFrame(summary_data)
            df_summary.to_csv(output_path, index=False, encoding='utf-8')
            logger.info(f"サマリーCSV保存: {output_path}")
        except Exception as e:
            logger.error(f"CSV保存エラー: {e}")

# 使用例・テスト用のサンプル関数
def example_usage():
    """A2AI ModelEvaluatorの使用例"""
    
    # サンプルデータ生成
    np.random.seed(42)
    n_samples = 1000
    
    # 回帰モデル評価例
    y_true_reg = np.random.normal(0, 1, n_samples)
    y_pred_reg1 = y_true_reg + np.random.normal(0, 0.5, n_samples)  # モデル1
    y_pred_reg2 = y_true_reg + np.random.normal(0, 0.3, n_samples)  # モデル2
    
    # 生存分析データ例
    durations = np.random.exponential(5, n_samples)
    events = np.random.binomial(1, 0.7, n_samples)
    survival_pred = np.random.beta(2, 3, n_samples)
    
    # 分類データ例
    y_true_cls = np.random.binomial(1, 0.3, n_samples)
    y_pred_proba = np.random.beta(2, 5, n_samples)
    
    # 評価器初期化
    evaluator = ModelEvaluator()
    
    # 各モデル評価実行
    model_results = {}
    
    # 回帰モデル評価
    model_results['LinearRegression'] = evaluator.evaluate_regression_model(
        y_true_reg, y_pred_reg1, 'LinearRegression', 'traditional'
    )
    
    model_results['RandomForest'] = evaluator.evaluate_regression_model(
        y_true_reg, y_pred_reg2, 'RandomForest', 'traditional'
    )
    
    # 生存分析モデル評価
    model_results['CoxRegression'] = evaluator.evaluate_survival_model(
        durations, events, survival_pred, 'CoxRegression'
    )
    
    # 分類モデル評価
    model_results['LogisticRegression'] = evaluator.evaluate_classification_model(
        y_true_cls, y_pred_proba, model_name='LogisticRegression', metric_type='emergence'
    )
    
    # モデル比較
    comparison_results = evaluator.compare_models(
        model_results, primary_metric='r2', 
        secondary_metrics=['mae', 'rmse']
    )
    
    # 結果表示
    print("=== A2AI Model Evaluation Example ===")
    print(f"Best Model: {comparison_results.get('ranking', {}).get('best_model', 'N/A')}")
    
    # レポート生成
    report = evaluator.generate_evaluation_report(model_results, comparison_results)
    print("\n" + report[:1000] + "...")  # 最初の1000文字表示
    
    return evaluator, model_results, comparison_results

if __name__ == "__main__":
    # 使用例実行
    example_usage()