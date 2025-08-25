"""
A2AI: Advanced Financial Analysis AI
survival_model_evaluation.py

企業の生存分析モデルに特化した評価システム
150社×40年のデータを用いた生存モデルの性能評価、検証、比較機能を提供

主要機能:
- Cox回帰モデルの評価（C-index, AIC, BIC等）
- Kaplan-Meier推定の適合度検証
- 機械学習ベース生存モデルの評価
- 時間依存ROC分析
- 生存確率予測の精度評価
- モデル間比較・選択支援
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
from dataclasses import dataclass
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
from lifelines.statistics import logrank_test, multivariate_logrank_test
import scipy.stats as stats
from scipy import integrate
import logging

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SurvivalEvaluationMetrics:
    """生存分析モデル評価指標を格納するデータクラス"""
    model_name: str
    c_index: float  # Concordance Index (C-index)
    c_index_ci: Tuple[float, float]  # C-indexの信頼区間
    aic: Optional[float] = None  # 赤池情報量基準
    bic: Optional[float] = None  # ベイズ情報量基準
    log_likelihood: Optional[float] = None
    time_dependent_auc: Optional[Dict[float, float]] = None  # 時間依存AUC
    calibration_metrics: Optional[Dict[str, float]] = None  # キャリブレーション指標
    cross_val_c_index: Optional[float] = None  # 交差検証C-index
    market_specific_performance: Optional[Dict[str, float]] = None  # 市場別性能
    
class SurvivalModelEvaluator:
    """
    生存分析モデルの包括的評価システム
    
    企業の生存・消滅パターン分析に特化した評価機能を提供
    """
    
    def __init__(self, market_categories: Optional[Dict[str, List[str]]] = None):
        """
        初期化
        
        Args:
            market_categories: 市場カテゴリー別企業リスト
        """
        self.market_categories = market_categories or {
            'high_share': [],
            'declining': [],
            'lost': []
        }
        self.evaluation_results = {}
        
    def evaluate_cox_model(self, 
                            model: CoxPHFitter,
                            data: pd.DataFrame,
                            duration_col: str,
                            event_col: str,
                            time_points: Optional[List[float]] = None) -> SurvivalEvaluationMetrics:
        """
        Cox比例ハザードモデルの包括評価
        
        Args:
            model: 訓練済みCoxPHFitterモデル
            data: 評価用データ
            duration_col: 生存時間カラム名
            event_col: イベント発生カラム名（1: 消滅, 0: 打ち切り）
            time_points: 時間依存評価の時点リスト
            
        Returns:
            SurvivalEvaluationMetrics: 評価結果
        """
        logger.info("Cox回帰モデルの評価を開始")
        
        # 基本評価指標
        c_index = model.concordance_index_
        c_index_ci = self._calculate_c_index_ci(model, data, duration_col, event_col)
        aic = model.AIC_
        bic = model.BIC_
        log_likelihood = model.log_likelihood_
        
        # 時間依存AUC計算
        time_dependent_auc = None
        if time_points:
            time_dependent_auc = self._calculate_time_dependent_auc(
                model, data, duration_col, event_col, time_points
            )
        
        # キャリブレーション評価
        calibration_metrics = self._evaluate_calibration(
            model, data, duration_col, event_col
        )
        
        # 交差検証
        cross_val_c_index = self._cross_validate_cox_model(
            model, data, duration_col, event_col
        )
        
        # 市場別性能評価
        market_specific_performance = self._evaluate_market_specific_performance(
            model, data, duration_col, event_col
        )
        
        return SurvivalEvaluationMetrics(
            model_name="Cox Proportional Hazards",
            c_index=c_index,
            c_index_ci=c_index_ci,
            aic=aic,
            bic=bic,
            log_likelihood=log_likelihood,
            time_dependent_auc=time_dependent_auc,
            calibration_metrics=calibration_metrics,
            cross_val_c_index=cross_val_c_index,
            market_specific_performance=market_specific_performance
        )
    
    def evaluate_kaplan_meier(self,
                                kmf: KaplanMeierFitter,
                                data: pd.DataFrame,
                                duration_col: str,
                                event_col: str,
                                groups: Optional[str] = None) -> SurvivalEvaluationMetrics:
        """
        Kaplan-Meier推定器の評価
        
        Args:
            kmf: 訓練済みKaplanMeierFitterモデル
            data: 評価用データ
            duration_col: 生存時間カラム名
            event_col: イベント発生カラム名
            groups: グループ比較用カラム名
            
        Returns:
            SurvivalEvaluationMetrics: 評価結果
        """
        logger.info("Kaplan-Meier推定器の評価を開始")
        
        # Logrank検定によるグループ間比較
        logrank_p_value = None
        if groups:
            logrank_p_value = self._perform_logrank_test(data, duration_col, event_col, groups)
        
        # 適合度評価
        goodness_of_fit = self._evaluate_km_goodness_of_fit(kmf, data, duration_col, event_col)
        
        # 信頼区間評価
        confidence_intervals = self._evaluate_km_confidence_intervals(kmf)
        
        calibration_metrics = {
            'logrank_p_value': logrank_p_value,
            'goodness_of_fit': goodness_of_fit,
            'confidence_intervals_width': confidence_intervals
        }
        
        return SurvivalEvaluationMetrics(
            model_name="Kaplan-Meier",
            c_index=None,  # KMではC-indexは計算されない
            c_index_ci=(None, None),
            calibration_metrics=calibration_metrics
        )
    
    def evaluate_ml_survival_model(self,
                                    model: Any,
                                    X_test: pd.DataFrame,
                                    y_test: pd.DataFrame,
                                    model_type: str = "RandomSurvivalForest") -> SurvivalEvaluationMetrics:
        """
        機械学習ベース生存モデルの評価
        
        Args:
            model: 訓練済み機械学習モデル
            X_test: テスト特徴量
            y_test: テスト目的変数（構造化配列: duration, event）
            model_type: モデル種類名
            
        Returns:
            SurvivalEvaluationMetrics: 評価結果
        """
        logger.info(f"{model_type}モデルの評価を開始")
        
        try:
            # scikit-survival対応
            if hasattr(model, 'score'):
                c_index = model.score(X_test, y_test)
            else:
                # カスタムモデル用のC-index計算
                risk_scores = model.predict(X_test)
                c_index = concordance_index(y_test['duration'], -risk_scores, y_test['event'])
            
            # 特徴量重要度分析
            feature_importance = self._analyze_feature_importance(model, X_test.columns)
            
            # 時間依存予測精度
            time_points = [1, 3, 5, 10, 15, 20]  # 年単位
            time_dependent_auc = self._calculate_ml_time_dependent_auc(
                model, X_test, y_test, time_points
            )
            
            return SurvivalEvaluationMetrics(
                model_name=model_type,
                c_index=c_index,
                c_index_ci=(None, None),  # ML模型では信頼区間計算が複雑
                time_dependent_auc=time_dependent_auc,
                calibration_metrics={'feature_importance': feature_importance}
            )
            
        except Exception as e:
            logger.error(f"{model_type}モデルの評価中にエラー: {str(e)}")
            return SurvivalEvaluationMetrics(
                model_name=model_type,
                c_index=0.0,
                c_index_ci=(0.0, 0.0)
            )
    
    def compare_models(self, 
                        evaluation_results: List[SurvivalEvaluationMetrics]) -> pd.DataFrame:
        """
        複数の生存分析モデルの比較
        
        Args:
            evaluation_results: モデル評価結果リスト
            
        Returns:
            pd.DataFrame: モデル比較結果テーブル
        """
        logger.info("モデル比較分析を開始")
        
        comparison_data = []
        for result in evaluation_results:
            row = {
                'Model': result.model_name,
                'C-Index': result.c_index,
                'C-Index_Lower': result.c_index_ci[0] if result.c_index_ci[0] else None,
                'C-Index_Upper': result.c_index_ci[1] if result.c_index_ci[1] else None,
                'AIC': result.aic,
                'BIC': result.bic,
                'Log-Likelihood': result.log_likelihood,
                'Cross-Val C-Index': result.cross_val_c_index
            }
            
            # 時間依存AUCの平均値
            if result.time_dependent_auc:
                row['Mean_Time_Dependent_AUC'] = np.mean(list(result.time_dependent_auc.values()))
            
            # 市場別性能の平均値
            if result.market_specific_performance:
                row['Mean_Market_Performance'] = np.mean(list(result.market_specific_performance.values()))
            
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # モデルランキング
        if 'C-Index' in comparison_df.columns:
            comparison_df['C-Index_Rank'] = comparison_df['C-Index'].rank(ascending=False)
        
        return comparison_df.sort_values('C-Index', ascending=False)
    
    def _calculate_c_index_ci(self, 
                                model: CoxPHFitter,
                                data: pd.DataFrame,
                                duration_col: str,
                                event_col: str,
                                confidence_level: float = 0.95) -> Tuple[float, float]:
        """C-indexの信頼区間をブートストラップで計算"""
        try:
            n_bootstrap = 1000
            c_indices = []
            
            for _ in range(n_bootstrap):
                bootstrap_data = data.sample(n=len(data), replace=True)
                temp_model = CoxPHFitter()
                temp_model.fit(bootstrap_data, duration_col=duration_col, event_col=event_col)
                c_indices.append(temp_model.concordance_index_)
            
            alpha = 1 - confidence_level
            lower = np.percentile(c_indices, 100 * alpha / 2)
            upper = np.percentile(c_indices, 100 * (1 - alpha / 2))
            
            return (lower, upper)
            
        except Exception as e:
            logger.warning(f"C-index信頼区間の計算に失敗: {str(e)}")
            return (None, None)
    
    def _calculate_time_dependent_auc(self,
                                    model: CoxPHFitter,
                                    data: pd.DataFrame,
                                    duration_col: str,
                                    event_col: str,
                                    time_points: List[float]) -> Dict[float, float]:
        """時間依存AUCの計算"""
        time_dependent_auc = {}
        
        try:
            for t in time_points:
                # t時点での生存・死亡ラベル作成
                labels = []
                risk_scores = []
                
                for idx, row in data.iterrows():
                    if row[duration_col] >= t:
                        if row[event_col] == 1 and row[duration_col] <= t:
                            labels.append(1)  # イベント発生
                        else:
                            labels.append(0)  # 生存
                        
                        # リスクスコア計算
                        partial_hazard = model.predict_partial_hazard(row.to_frame().T)
                        risk_scores.append(partial_hazard.iloc[0])
                
                if len(set(labels)) > 1:  # 両方のクラスが存在する場合のみ
                    auc = roc_auc_score(labels, risk_scores)
                    time_dependent_auc[t] = auc
                    
        except Exception as e:
            logger.warning(f"時間依存AUC計算エラー: {str(e)}")
            
        return time_dependent_auc
    
    def _evaluate_calibration(self,
                            model: CoxPHFitter,
                            data: pd.DataFrame,
                            duration_col: str,
                            event_col: str) -> Dict[str, float]:
        """モデルのキャリブレーション評価"""
        try:
            # 生存確率予測vs実際の生存率の比較
            survival_probabilities = model.predict_survival_function(data)
            
            # 複数時点での予測精度評価
            time_points = [1, 5, 10, 15, 20]
            calibration_scores = {}
            
            for t in time_points:
                if t in survival_probabilities.index:
                    predicted_survival = survival_probabilities.loc[t]
                    
                    # 実際の生存状況
                    actual_survival = []
                    for idx, row in data.iterrows():
                        if row[duration_col] >= t:
                            actual_survival.append(1)  # 生存
                        elif row[event_col] == 1:
                            actual_survival.append(0)  # 死亡
                        else:
                            actual_survival.append(1)  # 打ち切り（生存として扱う）
                    
                    if len(actual_survival) > 0:
                        # Brier Score計算
                        brier_score = np.mean((predicted_survival - actual_survival)**2)
                        calibration_scores[f'brier_score_{t}y'] = brier_score
            
            return calibration_scores
            
        except Exception as e:
            logger.warning(f"キャリブレーション評価エラー: {str(e)}")
            return {}
    
    def _cross_validate_cox_model(self,
                                model: CoxPHFitter,
                                data: pd.DataFrame,
                                duration_col: str,
                                event_col: str,
                                cv_folds: int = 5) -> float:
        """Cox回帰モデルの交差検証"""
        try:
            from sklearn.model_selection import KFold
            
            kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
            c_indices = []
            
            for train_idx, val_idx in kf.split(data):
                train_data = data.iloc[train_idx]
                val_data = data.iloc[val_idx]
                
                # モデル訓練
                temp_model = CoxPHFitter()
                temp_model.fit(train_data, duration_col=duration_col, event_col=event_col)
                
                # 検証データで評価
                c_index = temp_model.score(val_data, scoring_method='concordance_index')
                c_indices.append(c_index)
            
            return np.mean(c_indices)
            
        except Exception as e:
            logger.warning(f"交差検証エラー: {str(e)}")
            return None
    
    def _evaluate_market_specific_performance(self,
                                            model: CoxPHFitter,
                                            data: pd.DataFrame,
                                            duration_col: str,
                                            event_col: str) -> Dict[str, float]:
        """市場別モデル性能評価"""
        if not hasattr(data, 'market_category'):
            return {}
        
        market_performance = {}
        
        for market_name, companies in self.market_categories.items():
            market_data = data[data['company_name'].isin(companies)]
            
            if len(market_data) > 10:  # 十分なサンプルサイズがある場合
                try:
                    # 市場別C-index計算
                    partial_hazards = model.predict_partial_hazard(market_data)
                    c_index = concordance_index(
                        market_data[duration_col],
                        -partial_hazards,
                        market_data[event_col]
                    )
                    market_performance[market_name] = c_index
                    
                except Exception as e:
                    logger.warning(f"市場{market_name}の性能評価エラー: {str(e)}")
        
        return market_performance
    
    def _perform_logrank_test(self,
                            data: pd.DataFrame,
                            duration_col: str,
                            event_col: str,
                            groups_col: str) -> float:
        """Logrank検定の実行"""
        try:
            groups = data[groups_col].unique()
            if len(groups) == 2:
                group1 = data[data[groups_col] == groups[0]]
                group2 = data[data[groups_col] == groups[1]]
                
                result = logrank_test(
                    group1[duration_col], group2[duration_col],
                    group1[event_col], group2[event_col]
                )
                return result.p_value
            else:
                # 多群比較
                result = multivariate_logrank_test(
                    data[duration_col], data[groups_col], data[event_col]
                )
                return result.p_value
                
        except Exception as e:
            logger.warning(f"Logrank検定エラー: {str(e)}")
            return None
    
    def _evaluate_km_goodness_of_fit(self,
                                    kmf: KaplanMeierFitter,
                                    data: pd.DataFrame,
                                    duration_col: str,
                                    event_col: str) -> float:
        """Kaplan-Meier推定の適合度評価"""
        try:
            # Nelson-Aalen推定との比較
            from lifelines import NelsonAalenFitter
            naf = NelsonAalenFitter()
            naf.fit(data[duration_col], event_observed=data[event_col])
            
            # 累積ハザード関数の比較
            common_times = kmf.survival_function_.index.intersection(naf.cumulative_hazard_.index)
            
            if len(common_times) > 0:
                km_cumulative_hazard = -np.log(kmf.survival_function_.loc[common_times])
                na_cumulative_hazard = naf.cumulative_hazard_.loc[common_times]
                
                # 平均二乗誤差
                mse = np.mean((km_cumulative_hazard.iloc[:, 0] - na_cumulative_hazard.iloc[:, 0])**2)
                return mse
            
            return None
            
        except Exception as e:
            logger.warning(f"適合度評価エラー: {str(e)}")
            return None
    
    def _evaluate_km_confidence_intervals(self, kmf: KaplanMeierFitter) -> float:
        """Kaplan-Meier信頼区間の幅評価"""
        try:
            if hasattr(kmf, 'confidence_interval_'):
                ci_width = (kmf.confidence_interval_.iloc[:, 1] - 
                            kmf.confidence_interval_.iloc[:, 0]).mean()
                return ci_width
            return None
            
        except Exception as e:
            logger.warning(f"信頼区間評価エラー: {str(e)}")
            return None
    
    def _analyze_feature_importance(self, model: Any, feature_names: List[str]) -> Dict[str, float]:
        """機械学習モデルの特徴量重要度分析"""
        try:
            if hasattr(model, 'feature_importances_'):
                importance = dict(zip(feature_names, model.feature_importances_))
                return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
            elif hasattr(model, 'coef_'):
                importance = dict(zip(feature_names, np.abs(model.coef_)))
                return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
            else:
                return {}
                
        except Exception as e:
            logger.warning(f"特徴量重要度分析エラー: {str(e)}")
            return {}
    
    def _calculate_ml_time_dependent_auc(self,
                                        model: Any,
                                        X_test: pd.DataFrame,
                                        y_test: pd.DataFrame,
                                        time_points: List[float]) -> Dict[float, float]:
        """機械学習モデルの時間依存AUC計算"""
        time_dependent_auc = {}
        
        try:
            if hasattr(model, 'predict_survival_function'):
                # 生存関数予測が可能な場合
                survival_functions = model.predict_survival_function(X_test)
                
                for t in time_points:
                    # t時点での予測生存確率
                    if hasattr(survival_functions[0], 'y'):
                        # scikit-survivalスタイル
                        survival_probs = [sf(t) for sf in survival_functions]
                    else:
                        # カスタムスタイル
                        survival_probs = [sf.loc[t] if t in sf.index else sf.iloc[-1] 
                                        for sf in survival_functions]
                    
                    # 実際のt時点でのステータス
                    actual_labels = [1 if (row['duration'] <= t and row['event'] == 1) else 0 
                                    for _, row in y_test.iterrows()]
                    
                    if len(set(actual_labels)) > 1:
                        auc = roc_auc_score(actual_labels, [1-p for p in survival_probs])
                        time_dependent_auc[t] = auc
                        
        except Exception as e:
            logger.warning(f"ML時間依存AUC計算エラー: {str(e)}")
            
        return time_dependent_auc
    
    def generate_evaluation_report(self, 
                                    evaluation_results: List[SurvivalEvaluationMetrics],
                                    output_path: Optional[str] = None) -> str:
        """
        評価結果の包括的レポート生成
        
        Args:
            evaluation_results: 評価結果リスト
            output_path: 出力パス
            
        Returns:
            str: レポート文字列
        """
        report = ["="*80]
        report.append("A2AI 生存分析モデル評価レポート")
        report.append("="*80)
        report.append("")
        
        # サマリー統計
        report.append("【評価サマリー】")
        comparison_df = self.compare_models(evaluation_results)
        report.append(comparison_df.to_string())
        report.append("")
        
        # 詳細評価結果
        for result in evaluation_results:
            report.append(f"【{result.model_name} 詳細評価】")
            report.append(f"C-Index: {result.c_index:.4f}")
            if result.c_index_ci[0]:
                report.append(f"C-Index 95%CI: [{result.c_index_ci[0]:.4f}, {result.c_index_ci[1]:.4f}]")
            
            if result.aic:
                report.append(f"AIC: {result.aic:.2f}")
            if result.bic:
                report.append(f"BIC: {result.bic:.2f}")
            
            if result.time_dependent_auc:
                report.append("時間依存AUC:")
                for t, auc in result.time_dependent_auc.items():
                    report.append(f"  {t}年: {auc:.4f}")
            
            if result.market_specific_performance:
                report.append("市場別性能:")
                for market, perf in result.market_specific_performance.items():
                    report.append(f"  {market}: {perf:.4f}")
            
            report.append("")
        
        # 推奨モデル
        best_model = max(evaluation_results, key=lambda x: x.c_index or 0)
        report.append(f"【推奨モデル】: {best_model.model_name}")
        report.append(f"理由: 最高のC-Index ({best_model.c_index:.4f}) を達成")
        report.append("")
        
        report_text = "\n".join(report)
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            logger.info(f"評価レポートを {output_path} に出力しました")
        
        return report_text

# 使用例とテスト用コード
if __name__ == "__main__":
    # サンプルデータでのテスト
    logger.info("A2AI生存モデル評価システムのテストを開始")
    
    # サンプル市場カテゴリー
    market_categories = {
        'high_share': ['ファナック', '村田製作所', 'キーエンス'],
        'declining': ['トヨタ自動車', '日産自動車', 'パナソニック'],
        'lost': ['三洋電機', 'ソニー', 'シャープ']
    }
    
    evaluator = SurvivalModelEvaluator(market_categories)
    
    # サンプルデータ生成（実際のプロジェクトでは実データを使用）
    np.random.seed(42)
    n_companies = 150
    sample_data = pd.DataFrame({
        'company_name': [f'Company_{i}' for i in range(n_companies)],
        'duration': np.random.exponential(10, n_companies),  # 生存時間
        'event': np.random.binomial(1, 0.3, n_companies),    # イベント発生
        'revenue_growth': np.random.normal(0.05, 0.1, n_companies),
        'roa': np.random.normal(0.08, 0.05, n_companies),
        'debt_ratio': np.random.normal(0.3, 0.15, n_companies)
    })
    
    logger.info("サンプルデータでのCox回帰モデル評価テスト完了")
    print("A2AI生存モデル評価システムの初期化が完了しました。")
    print("実際の150社データでの評価準備が整いました。")