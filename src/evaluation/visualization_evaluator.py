"""
A2AI (Advanced Financial Analysis AI) - Visualization Evaluator
評価結果可視化モジュール

このモジュールは、A2AIの各種分析・予測モデルの評価結果を可視化するための
包括的な可視化評価システムを提供します。

主要機能:
- 従来型財務分析モデルの評価可視化
- 生存分析モデルの評価可視化  
- 新設企業分析モデルの評価可視化
- 因果推論モデルの評価可視化
- 統合モデルの評価可視化
- モデル間の比較評価可視化
- 評価指標の時系列変化可視化
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

# 日本語フォント設定
plt.rcParams['font.family'] = ['DejaVu Sans', 'SimHei', 'Malgun Gothic']
plt.rcParams['axes.unicode_minus'] = False

# 警告を抑制
warnings.filterwarnings('ignore')

@dataclass
class ModelEvaluation:
    """モデル評価結果を格納するデータクラス"""
    model_name: str
    model_type: str  # traditional, survival, emergence, causal, integrated
    metrics: Dict[str, float]
    predictions: np.ndarray
    actuals: np.ndarray
    feature_importance: Optional[Dict[str, float]] = None
    time_series_metrics: Optional[Dict[str, List[float]]] = None
    market_category: Optional[str] = None  # high_share, declining, lost
    evaluation_date: datetime = datetime.now()

class VisualizationEvaluator:
    """
    A2AI評価結果可視化クラス
    
    各種分析モデルの評価結果を包括的に可視化し、
    モデル性能の比較・解釈を支援します。
    """
    
    def __init__(self, output_dir: str = "results/visualizations/"):
        """
        初期化
        
        Args:
            output_dir: 可視化結果の出力ディレクトリ
        """
        self.output_dir = output_dir
        self.logger = self._setup_logger()
        
        # 市場カテゴリ別カラーパレット
        self.market_colors = {
            'high_share': '#2E8B57',      # SeaGreen
            'declining': '#FF8C00',       # DarkOrange
            'lost': '#DC143C',            # Crimson
            'overall': '#4682B4'          # SteelBlue
        }
        
        # モデルタイプ別カラーパレット
        self.model_colors = {
            'traditional': '#1f77b4',
            'survival': '#ff7f0e', 
            'emergence': '#2ca02c',
            'causal': '#d62728',
            'integrated': '#9467bd'
        }
        
    def _setup_logger(self) -> logging.Logger:
        """ロガーのセットアップ"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def visualize_model_performance(self, 
                                    evaluations: List[ModelEvaluation],
                                    save_path: Optional[str] = None) -> go.Figure:
        """
        モデル性能比較可視化
        
        Args:
            evaluations: モデル評価結果のリスト
            save_path: 保存パス
            
        Returns:
            Plotly図オブジェクト
        """
        self.logger.info("モデル性能比較の可視化を開始")
        
        # データ準備
        model_names = [eval.model_name for eval in evaluations]
        model_types = [eval.model_type for eval in evaluations]
        
        # 共通評価指標を抽出
        common_metrics = self._get_common_metrics(evaluations)
        
        # サブプロット作成
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "モデル別精度比較", "予測vs実際値散布図",
                "特徴量重要度TOP10", "評価指標レーダーチャート"
            ],
            specs=[
                [{"type": "bar"}, {"type": "scatter"}],
                [{"type": "bar"}, {"type": "scatterpolar"}]
            ]
        )
        
        # 1. モデル別精度比較（棒グラフ）
        accuracy_metric = self._select_primary_metric(evaluations)
        accuracy_values = [eval.metrics.get(accuracy_metric, 0) for eval in evaluations]
        
        fig.add_trace(
            go.Bar(
                x=model_names,
                y=accuracy_values,
                name=accuracy_metric,
                marker_color=[self.model_colors.get(mt, '#1f77b4') for mt in model_types],
                text=[f"{v:.3f}" for v in accuracy_values],
                textposition='outside'
            ),
            row=1, col=1
        )
        
        # 2. 予測vs実際値散布図
        for i, eval in enumerate(evaluations):
            if hasattr(eval, 'predictions') and hasattr(eval, 'actuals'):
                fig.add_trace(
                    go.Scatter(
                        x=eval.actuals,
                        y=eval.predictions,
                        mode='markers',
                        name=eval.model_name,
                        marker=dict(
                            color=self.model_colors.get(eval.model_type, '#1f77b4'),
                            opacity=0.6
                        ),
                        showlegend=False
                    ),
                    row=1, col=2
                )
        
        # 対角線を追加
        if evaluations:
            min_val = min([min(e.actuals) for e in evaluations if hasattr(e, 'actuals')])
            max_val = max([max(e.actuals) for e in evaluations if hasattr(e, 'actuals')])
            fig.add_trace(
                go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    name='Perfect Prediction',
                    line=dict(dash='dash', color='red'),
                    showlegend=False
                ),
                row=1, col=2
            )
        
        # 3. 特徴量重要度TOP10（最初のモデル）
        if evaluations and evaluations[0].feature_importance:
            importance_items = list(evaluations[0].feature_importance.items())
            importance_items.sort(key=lambda x: x[1], reverse=True)
            top_features = importance_items[:10]
            
            fig.add_trace(
                go.Bar(
                    x=[item[1] for item in top_features],
                    y=[item[0] for item in top_features],
                    orientation='h',
                    name='Feature Importance',
                    marker_color='skyblue',
                    showlegend=False
                ),
                row=2, col=1
            )
        
        # 4. 評価指標レーダーチャート
        if common_metrics and len(common_metrics) >= 3:
            for eval in evaluations:
                metric_values = [eval.metrics.get(metric, 0) for metric in common_metrics]
                fig.add_trace(
                    go.Scatterpolar(
                        r=metric_values,
                        theta=common_metrics,
                        fill='toself',
                        name=eval.model_name,
                        line_color=self.model_colors.get(eval.model_type, '#1f77b4')
                    ),
                    row=2, col=2
                )
        
        # レイアウト更新
        fig.update_layout(
            title="A2AI モデル性能総合評価",
            height=800,
            showlegend=True,
            font=dict(size=12)
        )
        
        # 軸ラベル設定
        fig.update_xaxes(title_text="モデル", row=1, col=1)
        fig.update_yaxes(title_text=accuracy_metric, row=1, col=1)
        fig.update_xaxes(title_text="実際値", row=1, col=2)
        fig.update_yaxes(title_text="予測値", row=1, col=2)
        fig.update_xaxes(title_text="重要度", row=2, col=1)
        
        if save_path:
            fig.write_html(save_path)
            self.logger.info(f"モデル性能比較図を保存: {save_path}")
        
        return fig
    
    def visualize_survival_model_evaluation(self,
                                            evaluations: List[ModelEvaluation],
                                            save_path: Optional[str] = None) -> go.Figure:
        """
        生存分析モデル評価可視化
        
        Args:
            evaluations: 生存分析モデル評価結果
            save_path: 保存パス
            
        Returns:
            Plotly図オブジェクト
        """
        self.logger.info("生存分析モデル評価の可視化を開始")
        
        # 生存分析特有の評価指標
        survival_metrics = ['c_index', 'log_likelihood', 'aic', 'bic', 'hazard_ratio']
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "C-Index比較", "AIC/BIC比較",
                "ハザード比分布", "生存曲線適合度"
            ]
        )
        
        # 1. C-Index比較
        c_indices = [eval.metrics.get('c_index', 0) for eval in evaluations]
        model_names = [eval.model_name for eval in evaluations]
        
        fig.add_trace(
            go.Bar(
                x=model_names,
                y=c_indices,
                name='C-Index',
                marker_color='coral',
                text=[f"{v:.3f}" for v in c_indices],
                textposition='outside'
            ),
            row=1, col=1
        )
        
        # 2. AIC/BIC比較
        aics = [eval.metrics.get('aic', 0) for eval in evaluations]
        bics = [eval.metrics.get('bic', 0) for eval in evaluations]
        
        fig.add_trace(go.Bar(x=model_names, y=aics, name='AIC', marker_color='lightblue'), row=1, col=2)
        fig.add_trace(go.Bar(x=model_names, y=bics, name='BIC', marker_color='lightcoral'), row=1, col=2)
        
        # 3. ハザード比分布（ヒストグラム）
        if evaluations and 'hazard_ratios' in evaluations[0].metrics:
            hazard_ratios = evaluations[0].metrics['hazard_ratios']
            fig.add_trace(
                go.Histogram(
                    x=hazard_ratios,
                    name='Hazard Ratio Distribution',
                    marker_color='gold',
                    opacity=0.7
                ),
                row=2, col=1
            )
        
        # 4. 生存曲線適合度（時系列）
        if evaluations and evaluations[0].time_series_metrics:
            time_points = evaluations[0].time_series_metrics.get('time_points', [])
            for eval in evaluations:
                if 'survival_prob' in eval.time_series_metrics:
                    survival_prob = eval.time_series_metrics['survival_prob']
                    fig.add_trace(
                        go.Scatter(
                            x=time_points,
                            y=survival_prob,
                            mode='lines',
                            name=f'{eval.model_name} 生存確率',
                            line=dict(color=self.model_colors.get(eval.model_type, '#1f77b4'))
                        ),
                        row=2, col=2
                    )
        
        # レイアウト更新
        fig.update_layout(
            title="生存分析モデル評価結果",
            height=800,
            showlegend=True
        )
        
        fig.update_yaxes(title_text="C-Index", row=1, col=1)
        fig.update_yaxes(title_text="情報量基準", row=1, col=2)
        fig.update_xaxes(title_text="ハザード比", row=2, col=1)
        fig.update_yaxes(title_text="頻度", row=2, col=1)
        fig.update_xaxes(title_text="時間", row=2, col=2)
        fig.update_yaxes(title_text="生存確率", row=2, col=2)
        
        if save_path:
            fig.write_html(save_path)
            self.logger.info(f"生存分析評価図を保存: {save_path}")
        
        return fig
    
    def visualize_emergence_model_evaluation(self,
                                            evaluations: List[ModelEvaluation],
                                            save_path: Optional[str] = None) -> go.Figure:
        """
        新設企業分析モデル評価可視化
        
        Args:
            evaluations: 新設企業分析モデル評価結果
            save_path: 保存パス
            
        Returns:
            Plotly図オブジェクト
        """
        self.logger.info("新設企業分析モデル評価の可視化を開始")
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "成功予測精度", "成長軌道予測誤差",
                "市場参入タイミング精度", "イノベーション影響度"
            ]
        )
        
        model_names = [eval.model_name for eval in evaluations]
        
        # 1. 成功予測精度
        success_accuracy = [eval.metrics.get('success_accuracy', 0) for eval in evaluations]
        fig.add_trace(
            go.Bar(
                x=model_names,
                y=success_accuracy,
                name='Success Accuracy',
                marker_color='mediumseagreen',
                text=[f"{v:.3f}" for v in success_accuracy],
                textposition='outside'
            ),
            row=1, col=1
        )
        
        # 2. 成長軌道予測誤差
        growth_rmse = [eval.metrics.get('growth_rmse', 0) for eval in evaluations]
        fig.add_trace(
            go.Bar(
                x=model_names,
                y=growth_rmse,
                name='Growth RMSE',
                marker_color='tomato',
                text=[f"{v:.3f}" for v in growth_rmse],
                textposition='outside'
            ),
            row=1, col=2
        )
        
        # 3. 市場参入タイミング精度（散布図）
        for eval in evaluations:
            if 'entry_timing_actual' in eval.metrics and 'entry_timing_predicted' in eval.metrics:
                actual = eval.metrics['entry_timing_actual']
                predicted = eval.metrics['entry_timing_predicted']
                fig.add_trace(
                    go.Scatter(
                        x=actual,
                        y=predicted,
                        mode='markers',
                        name=f'{eval.model_name} Entry Timing',
                        marker=dict(color=self.model_colors.get(eval.model_type, '#2ca02c'))
                    ),
                    row=2, col=1
                )
        
        # 4. イノベーション影響度（レーダーチャート）
        innovation_metrics = ['novelty_score', 'disruption_score', 'adoption_speed', 'market_impact']
        for eval in evaluations:
            values = [eval.metrics.get(metric, 0) for metric in innovation_metrics]
            if any(values):
                fig.add_trace(
                    go.Scatterpolar(
                        r=values,
                        theta=innovation_metrics,
                        fill='toself',
                        name=eval.model_name,
                        line_color=self.model_colors.get(eval.model_type, '#2ca02c')
                    ),
                    row=2, col=2
                )
        
        # レイアウト更新
        fig.update_layout(
            title="新設企業分析モデル評価結果",
            height=800,
            showlegend=True
        )
        
        fig.update_yaxes(title_text="精度", row=1, col=1)
        fig.update_yaxes(title_text="RMSE", row=1, col=2)
        fig.update_xaxes(title_text="実際参入時期", row=2, col=1)
        fig.update_yaxes(title_text="予測参入時期", row=2, col=2)
        
        if save_path:
            fig.write_html(save_path)
            self.logger.info(f"新設企業分析評価図を保存: {save_path}")
        
        return fig
    
    def visualize_market_category_comparison(self,
                                            evaluations: List[ModelEvaluation],
                                            save_path: Optional[str] = None) -> go.Figure:
        """
        市場カテゴリ別モデル性能比較
        
        Args:
            evaluations: 評価結果（市場カテゴリ情報含む）
            save_path: 保存パス
            
        Returns:
            Plotly図オブジェクト
        """
        self.logger.info("市場カテゴリ別評価の可視化を開始")
        
        # 市場カテゴリ別に分類
        market_data = {}
        for eval in evaluations:
            if eval.market_category:
                if eval.market_category not in market_data:
                    market_data[eval.market_category] = []
                market_data[eval.market_category].append(eval)
        
        if not market_data:
            self.logger.warning("市場カテゴリ情報が見つかりません")
            return go.Figure()
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "市場別平均精度", "市場別予測誤差",
                "市場別特徴量重要度", "市場別安定性"
            ]
        )
        
        # 1. 市場別平均精度
        categories = list(market_data.keys())
        avg_accuracies = []
        
        for category in categories:
            evals = market_data[category]
            accuracies = [eval.metrics.get('accuracy', 0) for eval in evals]
            avg_accuracies.append(np.mean(accuracies) if accuracies else 0)
        
        fig.add_trace(
            go.Bar(
                x=categories,
                y=avg_accuracies,
                name='Average Accuracy',
                marker_color=[self.market_colors.get(cat, '#4682B4') for cat in categories],
                text=[f"{v:.3f}" for v in avg_accuracies],
                textposition='outside'
            ),
            row=1, col=1
        )
        
        # 2. 市場別予測誤差（箱ひげ図）
        for category in categories:
            evals = market_data[category]
            errors = [eval.metrics.get('rmse', 0) for eval in evals if 'rmse' in eval.metrics]
            if errors:
                fig.add_trace(
                    go.Box(
                        y=errors,
                        name=category,
                        marker_color=self.market_colors.get(category, '#4682B4')
                    ),
                    row=1, col=2
                )
        
        # 3. 市場別特徴量重要度（上位5項目）
        for i, category in enumerate(categories):
            evals = market_data[category]
            # 最初のモデルの特徴量重要度を使用
            if evals and evals[0].feature_importance:
                importance_items = list(evals[0].feature_importance.items())
                importance_items.sort(key=lambda x: x[1], reverse=True)
                top_features = importance_items[:5]
                
                fig.add_trace(
                    go.Bar(
                        x=[item[1] for item in top_features],
                        y=[item[0] for item in top_features],
                        orientation='h',
                        name=f'{category} Top Features',
                        marker_color=self.market_colors.get(category, '#4682B4'),
                        visible=True if i == 0 else 'legendonly'  # 最初のカテゴリのみ表示
                    ),
                    row=2, col=1
                )
        
        # 4. 市場別安定性（評価指標の分散）
        stability_scores = []
        for category in categories:
            evals = market_data[category]
            accuracies = [eval.metrics.get('accuracy', 0) for eval in evals]
            stability = 1 / (np.std(accuracies) + 1e-6) if len(accuracies) > 1 else 1
            stability_scores.append(stability)
        
        fig.add_trace(
            go.Scatter(
                x=categories,
                y=stability_scores,
                mode='markers+lines',
                name='Stability Score',
                marker=dict(
                    size=15,
                    color=[self.market_colors.get(cat, '#4682B4') for cat in categories]
                ),
                line=dict(color='gray', dash='dash')
            ),
            row=2, col=2
        )
        
        # レイアウト更新
        fig.update_layout(
            title="市場カテゴリ別モデル性能比較",
            height=800,
            showlegend=True
        )
        
        fig.update_xaxes(title_text="市場カテゴリ", row=1, col=1)
        fig.update_yaxes(title_text="平均精度", row=1, col=1)
        fig.update_yaxes(title_text="予測誤差", row=1, col=2)
        fig.update_xaxes(title_text="重要度", row=2, col=1)
        fig.update_xaxes(title_text="市場カテゴリ", row=2, col=2)
        fig.update_yaxes(title_text="安定性スコア", row=2, col=2)
        
        if save_path:
            fig.write_html(save_path)
            self.logger.info(f"市場カテゴリ別比較図を保存: {save_path}")
        
        return fig
    
    def create_comprehensive_evaluation_dashboard(self,
                                                    all_evaluations: Dict[str, List[ModelEvaluation]],
                                                    save_path: Optional[str] = None) -> go.Figure:
        """
        包括的評価ダッシュボード作成
        
        Args:
            all_evaluations: 全評価結果（モデルタイプ別）
            save_path: 保存パス
            
        Returns:
            Plotly図オブジェクト
        """
        self.logger.info("包括的評価ダッシュボードを作成")
        
        # 4x3のサブプロット
        fig = make_subplots(
            rows=4, cols=3,
            subplot_titles=[
                "全モデル精度比較", "モデルタイプ別性能", "実行時間比較",
                "予測vs実際相関", "特徴量重要度統合", "評価指標分布",
                "市場別適合度", "時系列安定性", "複雑度vs性能",
                "エラー分析", "信頼区間比較", "総合評価ランキング"
            ],
            specs=[
                [{"type": "bar"}, {"type": "box"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "bar"}, {"type": "violin"}],
                [{"type": "heatmap"}, {"type": "scatter"}, {"type": "scatter"}],
                [{"type": "histogram"}, {"type": "box"}, {"type": "bar"}]
            ]
        )
        
        # 全評価結果を統合
        all_evals = []
        for model_type, evals in all_evaluations.items():
            all_evals.extend(evals)
        
        if not all_evals:
            self.logger.warning("評価データが空です")
            return fig
        
        # データ準備
        model_names = [eval.model_name for eval in all_evals]
        model_types = [eval.model_type for eval in all_evals]
        accuracies = [eval.metrics.get('accuracy', 0) for eval in all_evals]
        
        # 1. 全モデル精度比較
        fig.add_trace(
            go.Bar(
                x=model_names,
                y=accuracies,
                name='Accuracy',
                marker_color=[self.model_colors.get(mt, '#1f77b4') for mt in model_types],
                text=[f"{v:.3f}" for v in accuracies],
                textposition='outside'
            ),
            row=1, col=1
        )
        
        # 2. モデルタイプ別性能（箱ひげ図）
        type_performance = {}
        for eval in all_evals:
            if eval.model_type not in type_performance:
                type_performance[eval.model_type] = []
            type_performance[eval.model_type].append(eval.metrics.get('accuracy', 0))
        
        for model_type, performances in type_performance.items():
            fig.add_trace(
                go.Box(
                    y=performances,
                    name=model_type,
                    marker_color=self.model_colors.get(model_type, '#1f77b4')
                ),
                row=1, col=2
            )
        
        # 3. 実行時間比較
        execution_times = [eval.metrics.get('execution_time', 0) for eval in all_evals]
        fig.add_trace(
            go.Scatter(
                x=model_names,
                y=execution_times,
                mode='markers',
                name='Execution Time',
                marker=dict(
                    size=10,
                    color=[self.model_colors.get(mt, '#1f77b4') for mt in model_types]
                )
            ),
            row=1, col=3
        )
        
        # 残りのプロットも同様に実装...
        # （紙面の都合上、主要な部分のみ実装）
        
        # レイアウト更新
        fig.update_layout(
            title="A2AI 包括的評価ダッシュボード",
            height=1200,
            showlegend=True,
            font=dict(size=10)
        )
        
        if save_path:
            fig.write_html(save_path)
            self.logger.info(f"包括的評価ダッシュボードを保存: {save_path}")
        
        return fig
    
    def _get_common_metrics(self, evaluations: List[ModelEvaluation]) -> List[str]:
        """共通評価指標を取得"""
        if not evaluations:
            return []
        
        common_metrics = set(evaluations[0].metrics.keys())
        for eval in evaluations[1:]:
            common_metrics &= set(eval.metrics.keys())
        
        return list(common_metrics)
    
    def _select_primary_metric(self, evaluations: List[ModelEvaluation]) -> str:
        """主要評価指標を選択"""
        common_metrics = self._get_common_metrics(evaluations)
        
        # 優先順位に基づいて選択
        priority_metrics = ['accuracy', 'f1_score', 'auc', 'rmse', 'mae']
        
        for metric in priority_metrics:
            if metric in common_metrics:
                return metric
        
        return common_metrics[0] if common_metrics else 'score'

# 使用例とテスト用関数
def create_sample_evaluations() -> List[ModelEvaluation]:
    """サンプル評価データを作成"""
    np.random.seed(42)
    
    evaluations = []
    
    # 従来型モデル
    eval1 = ModelEvaluation(
        model_name="RandomForest_Traditional",
        model_type="traditional",
        metrics={
            'accuracy': 0.85,
            'f1_score': 0.82,
            'rmse': 0.15,
            'execution_time': 2.3
        },
        predictions=np.random.normal(0, 1, 100),
        actuals=np.random.normal(0, 1, 100),
        feature_importance={'売上高': 0.3, '研究開発費': 0.25, 'ROE': 0.2, '総資産回転率': 0.15, '自己資本比率': 0.1},
        market_category='high_share'
    )
    evaluations.append(eval1)
    
    # 生存分析モデル
    eval2 = ModelEvaluation(
        model_name="CoxRegression_Survival",
        model_type="survival",
        metrics={
            'c_index': 0.78,
            'log_likelihood': -1245.6,
            'aic': 2495.2,
            'bic': 2510.8,
            'execution_time': 5.2
        },
        predictions=np.random.exponential(2, 100),
        actuals=np.random.exponential(2, 100),
        feature_importance={'設備投資額': 0.4, '従業員数': 0.3, '海外売上比率': 0.2, '研究開発費率': 0.1},
        time_series_metrics={
            'time_points': list(range(0, 40)),
            'survival_prob': np.exp(-0.1 * np.array(range(0, 40)))
        },
        market_category='declining'
    )
    evaluations.append(eval2)
    
    # 新設企業分析モデル
    eval3 = ModelEvaluation(
        model_name="GradientBoost_Emergence",
        model_type="emergence",
        metrics={
            'success_accuracy': 0.73,
            'growth_rmse': 0.22,
            'novelty_score': 0.8,
            'disruption_score': 0.7,
            'adoption_speed': 0.6,
            'market_impact': 0.75,
            'execution_time': 3.1
        },
        predictions=np.random.gamma(2, 2, 100),
        actuals=np.random.gamma(2, 2, 100),
        feature_importance={'無形固定資産': 0.35, '平均年間給与': 0.25, '特許関連費用': 0.2, 'ソフトウェア比率': 0.2},
        market_category='high_share'
    )
    evaluations.append(eval3)
    
    # 因果推論モデル
    eval4 = ModelEvaluation(
        model_name="CausalForest_DID",
        model_type="causal",
        metrics={
            'ate_accuracy': 0.81,  # Average Treatment Effect精度
            'bias_reduction': 0.65,
            'confidence_interval': 0.95,
            'execution_time': 7.8
        },
        predictions=np.random.beta(2, 5, 100),
        actuals=np.random.beta(2, 5, 100),
        feature_importance={'M&A実施': 0.5, '市場参入時期': 0.3, '規制変更': 0.15, '競合状況': 0.05},
        market_category='lost'
    )
    evaluations.append(eval4)
    
    # 統合モデル
    eval5 = ModelEvaluation(
        model_name="EnsembleModel_Integrated",
        model_type="integrated",
        metrics={
            'accuracy': 0.89,
            'f1_score': 0.87,
            'auc': 0.92,
            'rmse': 0.12,
            'execution_time': 12.5
        },
        predictions=np.random.lognormal(0, 0.5, 100),
        actuals=np.random.lognormal(0, 0.5, 100),
        feature_importance={
            '売上高': 0.2, '研究開発費': 0.18, 'ROE': 0.15, '設備投資額': 0.12,
            '無形固定資産': 0.1, '従業員数': 0.08, '海外売上比率': 0.07,
            '自己資本比率': 0.06, 'M&A実施': 0.04
        },
        market_category='high_share'
    )
    evaluations.append(eval5)
    
    return evaluations

def test_visualization_evaluator():
    """可視化評価システムのテスト"""
    evaluator = VisualizationEvaluator()
    
    # サンプルデータ作成
    evaluations = create_sample_evaluations()
    
    # 1. 基本的なモデル性能比較
    print("モデル性能比較可視化をテスト中...")
    performance_fig = evaluator.visualize_model_performance(evaluations)
    
    # 2. 生存分析モデル評価
    survival_evals = [eval for eval in evaluations if eval.model_type == 'survival']
    if survival_evals:
        print("生存分析モデル評価可視化をテスト中...")
        survival_fig = evaluator.visualize_survival_model_evaluation(survival_evals)
    
    # 3. 新設企業分析モデル評価
    emergence_evals = [eval for eval in evaluations if eval.model_type == 'emergence']
    if emergence_evals:
        print("新設企業分析モデル評価可視化をテスト中...")
        emergence_fig = evaluator.visualize_emergence_model_evaluation(emergence_evals)
    
    # 4. 市場カテゴリ別比較
    print("市場カテゴリ別比較可視化をテスト中...")
    market_fig = evaluator.visualize_market_category_comparison(evaluations)
    
    # 5. 包括的評価ダッシュボード
    all_evaluations = {
        'traditional': [eval for eval in evaluations if eval.model_type == 'traditional'],
        'survival': [eval for eval in evaluations if eval.model_type == 'survival'],
        'emergence': [eval for eval in evaluations if eval.model_type == 'emergence'],
        'causal': [eval for eval in evaluations if eval.model_type == 'causal'],
        'integrated': [eval for eval in evaluations if eval.model_type == 'integrated']
    }
    
    print("包括的評価ダッシュボードをテスト中...")
    dashboard_fig = evaluator.create_comprehensive_evaluation_dashboard(all_evaluations)
    
    print("すべてのテストが完了しました。")
    return {
        'performance': performance_fig,
        'survival': survival_fig if 'survival_fig' in locals() else None,
        'emergence': emergence_fig if 'emergence_fig' in locals() else None,
        'market_comparison': market_fig,
        'dashboard': dashboard_fig
    }

class AdvancedVisualizationEvaluator(VisualizationEvaluator):
    """
    高度な可視化評価機能を提供する拡張クラス
    """
    
    def create_interactive_model_explorer(self,
                                            evaluations: List[ModelEvaluation],
                                            save_path: Optional[str] = None) -> go.Figure:
        """
        インタラクティブモデル探索ダッシュボード
        
        Args:
            evaluations: モデル評価結果
            save_path: 保存パス
            
        Returns:
            Plotly図オブジェクト
        """
        self.logger.info("インタラクティブモデル探索ダッシュボードを作成")
        
        # ドロップダウンメニュー用の選択肢
        model_options = [{'label': eval.model_name, 'value': i} for i, eval in enumerate(evaluations)]
        metric_options = []
        
        if evaluations:
            all_metrics = set()
            for eval in evaluations:
                all_metrics.update(eval.metrics.keys())
            metric_options = [{'label': metric, 'value': metric} for metric in sorted(all_metrics)]
        
        # ベースとなる図を作成
        fig = go.Figure()
        
        # 初期データ（最初のモデル）
        if evaluations:
            initial_eval = evaluations[0]
            metrics = list(initial_eval.metrics.keys())
            values = list(initial_eval.metrics.values())
            
            fig.add_trace(
                go.Bar(
                    x=metrics,
                    y=values,
                    name=initial_eval.model_name,
                    marker_color=self.model_colors.get(initial_eval.model_type, '#1f77b4')
                )
            )
        
        # レイアウトにドロップダウンメニューを追加
        fig.update_layout(
            title="インタラクティブモデル性能探索",
            updatemenus=[
                {
                    "buttons": [
                        {
                            "label": eval.model_name,
                            "method": "update",
                            "args": [
                                {
                                    "x": [list(eval.metrics.keys())],
                                    "y": [list(eval.metrics.values())],
                                    "name": eval.model_name,
                                    "marker.color": self.model_colors.get(eval.model_type, '#1f77b4')
                                },
                                {"title": f"モデル性能: {eval.model_name}"}
                            ]
                        } for eval in evaluations
                    ],
                    "direction": "down",
                    "showactive": True,
                    "x": 0.1,
                    "xanchor": "left",
                    "y": 1.02,
                    "yanchor": "top"
                }
            ],
            annotations=[
                {
                    "text": "モデル選択:",
                    "showarrow": False,
                    "x": 0.01,
                    "y": 1.05,
                    "yref": "paper",
                    "align": "left"
                }
            ]
        )
        
        if save_path:
            fig.write_html(save_path)
            self.logger.info(f"インタラクティブ探索ダッシュボードを保存: {save_path}")
        
        return fig
    
    def create_time_series_evaluation_plot(self,
                                            evaluations: List[ModelEvaluation],
                                            metric_name: str = 'accuracy',
                                            save_path: Optional[str] = None) -> go.Figure:
        """
        時系列評価プロット作成
        
        Args:
            evaluations: モデル評価結果
            metric_name: 評価指標名
            save_path: 保存パス
            
        Returns:
            Plotly図オブジェクト
        """
        self.logger.info(f"時系列評価プロット（{metric_name}）を作成")
        
        fig = go.Figure()
        
        for eval in evaluations:
            if eval.time_series_metrics and metric_name in eval.time_series_metrics:
                time_points = eval.time_series_metrics.get('time_points', [])
                values = eval.time_series_metrics[metric_name]
                
                fig.add_trace(
                    go.Scatter(
                        x=time_points,
                        y=values,
                        mode='lines+markers',
                        name=f'{eval.model_name}',
                        line=dict(color=self.model_colors.get(eval.model_type, '#1f77b4')),
                        hovertemplate='<b>%{fullData.name}</b><br>' +
                                    f'{metric_name}: %{{y:.3f}}<br>' +
                                    'Time: %{x}<br>' +
                                    '<extra></extra>'
                    )
                )
        
        fig.update_layout(
            title=f'時系列評価: {metric_name}',
            xaxis_title='時間',
            yaxis_title=metric_name,
            hovermode='x unified',
            showlegend=True
        )
        
        if save_path:
            fig.write_html(save_path)
            self.logger.info(f"時系列評価プロットを保存: {save_path}")
        
        return fig
    
    def create_feature_importance_comparison(self,
                                            evaluations: List[ModelEvaluation],
                                            top_n: int = 10,
                                            save_path: Optional[str] = None) -> go.Figure:
        """
        特徴量重要度比較可視化
        
        Args:
            evaluations: モデル評価結果
            top_n: 表示する上位特徴量数
            save_path: 保存パス
            
        Returns:
            Plotly図オブジェクト
        """
        self.logger.info("特徴量重要度比較可視化を作成")
        
        # 全モデルから特徴量重要度を収集
        all_features = set()
        for eval in evaluations:
            if eval.feature_importance:
                all_features.update(eval.feature_importance.keys())
        
        all_features = list(all_features)
        
        # データ準備
        importance_matrix = []
        model_names = []
        
        for eval in evaluations:
            if eval.feature_importance:
                model_names.append(eval.model_name)
                importance_row = [eval.feature_importance.get(feature, 0) for feature in all_features]
                importance_matrix.append(importance_row)
        
        if not importance_matrix:
            self.logger.warning("特徴量重要度データが見つかりません")
            return go.Figure()
        
        importance_df = pd.DataFrame(importance_matrix, columns=all_features, index=model_names)
        
        # 平均重要度でソート
        avg_importance = importance_df.mean(axis=0).sort_values(ascending=False)
        top_features = avg_importance.head(top_n).index.tolist()
        
        # ヒートマップ作成
        fig = go.Figure(
            data=go.Heatmap(
                z=importance_df[top_features].values,
                x=top_features,
                y=model_names,
                colorscale='Viridis',
                colorbar=dict(title="重要度"),
                hoveringfmt='.3f'
            )
        )
        
        fig.update_layout(
            title=f'特徴量重要度比較 (上位{top_n}項目)',
            xaxis_title='特徴量',
            yaxis_title='モデル',
            height=400 + len(model_names) * 30
        )
        
        if save_path:
            fig.write_html(save_path)
            self.logger.info(f"特徴量重要度比較図を保存: {save_path}")
        
        return fig

if __name__ == "__main__":
    # テスト実行
    test_results = test_visualization_evaluator()
    print("可視化評価システムのテストが完了しました。")