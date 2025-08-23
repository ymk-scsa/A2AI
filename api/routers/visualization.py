from fastapi import APIRouter, HTTPException, Query, Depends
from fastapi.responses import JSONResponse, StreamingResponse
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field
from datetime import datetime, date
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
import io
import base64
import json
from enum import Enum

# 依存関係のインポート（実際の実装では他のモジュールから）
from ...src.visualization.traditional_viz.factor_visualizer import FactorVisualizer
from ...src.visualization.survival_viz.survival_curves import SurvivalCurveGenerator
from ...src.visualization.lifecycle_viz.lifecycle_trajectories import LifecycleTrajectoryVisualizer
from ...src.visualization.emergence_viz.startup_journey import StartupJourneyVisualizer
from ...src.visualization.integrated_viz.ecosystem_networks import EcosystemNetworkVisualizer
from ...src.analysis.survival_analysis.extinction_risk_analyzer import ExtinctionRiskAnalyzer
from ...src.analysis.emergence_analysis.startup_success_analyzer import StartupSuccessAnalyzer
from ...src.utils.data_utils import get_company_data, get_market_data

router = APIRouter(prefix="/visualization", tags=["visualization"])

# Enumとモデル定義
class MarketCategory(str, Enum):
    HIGH_SHARE = "high_share"
    DECLINING = "declining"
    LOST = "lost"

class VisualizationType(str, Enum):
    SURVIVAL_CURVE = "survival_curve"
    FACTOR_IMPACT = "factor_impact"
    LIFECYCLE_TRAJECTORY = "lifecycle_trajectory"
    MARKET_COMPARISON = "market_comparison"
    ECOSYSTEM_NETWORK = "ecosystem_network"
    STARTUP_JOURNEY = "startup_journey"
    EXTINCTION_RISK = "extinction_risk"
    CORRELATION_HEATMAP = "correlation_heatmap"
    TIME_SERIES = "time_series"
    DASHBOARD = "dashboard"

class CompanyRequest(BaseModel):
    company_names: List[str] = Field(..., description="企業名のリスト")
    market_categories: Optional[List[MarketCategory]] = Field(None, description="市場カテゴリフィルタ")
    start_year: Optional[int] = Field(1984, description="分析開始年")
    end_year: Optional[int] = Field(2024, description="分析終了年")

class VisualizationRequest(BaseModel):
    visualization_type: VisualizationType
    companies: Optional[List[str]] = Field(None, description="対象企業リスト")
    market_categories: Optional[List[MarketCategory]] = Field(None, description="市場カテゴリ")
    evaluation_metrics: Optional[List[str]] = Field(None, description="評価項目")
    factor_items: Optional[List[str]] = Field(None, description="要因項目")
    time_range: Optional[Dict[str, int]] = Field(None, description="時間範囲")
    chart_config: Optional[Dict[str, Any]] = Field(None, description="チャート設定")

class SurvivalAnalysisRequest(BaseModel):
    market_categories: List[MarketCategory]
    analysis_period: Dict[str, int] = Field({"start": 1984, "end": 2024})
    risk_factors: Optional[List[str]] = Field(None, description="リスク要因")
    confidence_interval: float = Field(0.95, description="信頼区間")

class FactorImpactRequest(BaseModel):
    target_metric: str = Field(..., description="目標評価項目")
    factor_items: List[str] = Field(..., description="要因項目リスト")
    companies: Optional[List[str]] = Field(None, description="対象企業")
    market_filter: Optional[MarketCategory] = Field(None, description="市場フィルタ")
    analysis_method: str = Field("correlation", description="分析手法")

class DashboardRequest(BaseModel):
    dashboard_type: str = Field("comprehensive", description="ダッシュボードタイプ")
    market_focus: Optional[MarketCategory] = Field(None, description="市場フォーカス")
    time_period: Dict[str, int] = Field({"start": 2020, "end": 2024})
    key_metrics: List[str] = Field(["売上高", "ROE", "生存確率"], description="重要指標")

# ユーティリティ関数
def create_plotly_response(fig: go.Figure) -> Dict[str, Any]:
    """Plotlyフィギュアを JSON レスポンス用に変換"""
    return {
        "chart_data": fig.to_dict(),
        "chart_type": "plotly",
        "timestamp": datetime.now().isoformat()
    }

def create_matplotlib_response(fig: plt.Figure) -> Dict[str, Any]:
    """MatplotlibフィギュアをBase64エンコードで返す"""
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close(fig)
    
    return {
        "chart_data": f"data:image/png;base64,{image_base64}",
        "chart_type": "matplotlib",
        "timestamp": datetime.now().isoformat()
    }

# メインエンドポイント群

@router.post("/survival-analysis")
async def create_survival_analysis(request: SurvivalAnalysisRequest):
    """
    生存分析可視化
    - Kaplan-Meier生存曲線
    - ハザード比分析
    - 市場カテゴリ別比較
    """
    try:
        # 生存分析データの取得
        analyzer = ExtinctionRiskAnalyzer()
        survival_data = analyzer.get_survival_data(
            market_categories=request.market_categories,
            start_year=request.analysis_period["start"],
            end_year=request.analysis_period["end"]
        )
        
        # Kaplan-Meier曲線の生成
        curve_generator = SurvivalCurveGenerator()
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Kaplan-Meier生存曲線', 'ハザード関数', '市場別生存率比較', 'リスク要因分析'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                    [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 市場カテゴリ別の生存曲線
        colors = {"high_share": "green", "declining": "orange", "lost": "red"}
        
        for i, category in enumerate(request.market_categories):
            category_data = survival_data[survival_data['market_category'] == category.value]
            
            # Kaplan-Meier推定
            survival_curve = curve_generator.calculate_kaplan_meier(category_data)
            
            fig.add_trace(
                go.Scatter(
                    x=survival_curve['time'],
                    y=survival_curve['survival_probability'],
                    mode='lines',
                    name=f'{category.value.replace("_", " ").title()}市場',
                    line=dict(color=colors.get(category.value, 'blue'), width=3)
                ),
                row=1, col=1
            )
            
            # ハザード関数
            hazard_curve = curve_generator.calculate_hazard_function(category_data)
            fig.add_trace(
                go.Scatter(
                    x=hazard_curve['time'],
                    y=hazard_curve['hazard_rate'],
                    mode='lines',
                    name=f'{category.value.replace("_", " ").title()}ハザード',
                    line=dict(color=colors.get(category.value, 'blue'), dash='dash')
                ),
                row=1, col=2
            )
        
        # 市場別生存率比較（バーチャート）
        survival_rates = []
        for category in request.market_categories:
            category_data = survival_data[survival_data['market_category'] == category.value]
            current_survival_rate = analyzer.calculate_current_survival_rate(category_data)
            survival_rates.append({
                'market': category.value.replace("_", " ").title(),
                'survival_rate': current_survival_rate
            })
        
        survival_df = pd.DataFrame(survival_rates)
        fig.add_trace(
            go.Bar(
                x=survival_df['market'],
                y=survival_df['survival_rate'],
                marker_color=[colors.get(cat.lower().replace(" ", "_"), 'blue') for cat in survival_df['market']],
                name='現在生存率'
            ),
            row=2, col=1
        )
        
        # リスク要因分析（可能な場合）
        if request.risk_factors:
            risk_analysis = analyzer.analyze_risk_factors(survival_data, request.risk_factors)
            
            fig.add_trace(
                go.Scatter(
                    x=risk_analysis['factor_importance'],
                    y=list(range(len(risk_analysis['factor_importance']))),
                    mode='markers',
                    marker=dict(
                        size=risk_analysis['hazard_ratio'],
                        sizemode='area',
                        sizeref=2.*max(risk_analysis['hazard_ratio'])/(40.**2),
                        sizemin=4,
                        color=risk_analysis['p_value'],
                        colorscale='Reds',
                        showscale=True,
                        colorbar=dict(title="P値")
                    ),
                    text=[f"HR: {hr:.2f}" for hr in risk_analysis['hazard_ratio']],
                    name='リスク要因'
                ),
                row=2, col=2
            )
        
        # レイアウト設定
        fig.update_layout(
            height=800,
            title_text="A2AI 企業生存分析ダッシュボード",
            showlegend=True,
            title_x=0.5
        )
        
        # 軸ラベル設定
        fig.update_xaxes(title_text="年数", row=1, col=1)
        fig.update_yaxes(title_text="生存確率", row=1, col=1)
        fig.update_xaxes(title_text="年数", row=1, col=2)
        fig.update_yaxes(title_text="ハザード率", row=1, col=2)
        fig.update_xaxes(title_text="市場カテゴリ", row=2, col=1)
        fig.update_yaxes(title_text="生存率(%)", row=2, col=1)
        fig.update_xaxes(title_text="要因重要度", row=2, col=2)
        fig.update_yaxes(title_text="リスク要因", row=2, col=2)
        
        return create_plotly_response(fig)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"生存分析の生成に失敗しました: {str(e)}")

@router.post("/factor-impact")
async def create_factor_impact_analysis(request: FactorImpactRequest):
    """
    要因項目影響分析可視化
    - 相関ヒートマップ
    - 要因重要度分析
    - 時系列影響度変化
    """
    try:
        # データ取得
        company_data = get_company_data(
            companies=request.companies,
            market_filter=request.market_filter.value if request.market_filter else None
        )
        
        # 要因影響分析
        factor_viz = FactorVisualizer()
        
        # 相関分析
        correlation_matrix = factor_viz.calculate_factor_correlation(
            company_data, 
            target_metric=request.target_metric,
            factor_items=request.factor_items
        )
        
        # 複数のサブプロットを作成
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                '要因項目相関ヒートマップ',
                '要因重要度ランキング',
                '時系列影響度変化',
                '市場カテゴリ別影響度比較'
            ),
            specs=[
                [{"type": "heatmap"}, {"type": "bar"}],
                [{"type": "scatter"}, {"type": "bar"}]
            ]
        )
        
        # 1. 相関ヒートマップ
        fig.add_trace(
            go.Heatmap(
                z=correlation_matrix.values,
                x=correlation_matrix.columns,
                y=correlation_matrix.index,
                colorscale='RdBu',
                zmid=0,
                colorbar=dict(title="相関係数", x=0.48)
            ),
            row=1, col=1
        )
        
        # 2. 要因重要度ランキング
        importance_scores = factor_viz.calculate_feature_importance(
            company_data, request.target_metric, request.factor_items
        )
        
        fig.add_trace(
            go.Bar(
                x=importance_scores['importance'],
                y=importance_scores['factor'],
                orientation='h',
                marker_color='lightblue',
                name='重要度'
            ),
            row=1, col=2
        )
        
        # 3. 時系列影響度変化
        time_series_impact = factor_viz.calculate_rolling_correlation(
            company_data, request.target_metric, request.factor_items[:5]  # 上位5要因
        )
        
        for factor in time_series_impact.columns:
            if factor != 'year':
                fig.add_trace(
                    go.Scatter(
                        x=time_series_impact['year'],
                        y=time_series_impact[factor],
                        mode='lines+markers',
                        name=factor,
                        line=dict(width=2)
                    ),
                    row=2, col=1
                )
        
        # 4. 市場カテゴリ別影響度比較
        market_impact = factor_viz.calculate_market_specific_impact(
            company_data, request.target_metric, request.factor_items[:10]
        )
        
        categories = list(market_impact.keys())
        top_factors = request.factor_items[:5]
        
        for i, factor in enumerate(top_factors):
            market_values = [market_impact[cat].get(factor, 0) for cat in categories]
            fig.add_trace(
                go.Bar(
                    x=categories,
                    y=market_values,
                    name=factor,
                    offsetgroup=i
                ),
                row=2, col=2
            )
        
        # レイアウト設定
        fig.update_layout(
            height=1000,
            title_text=f"要因項目影響分析: {request.target_metric}",
            showlegend=True,
            title_x=0.5
        )
        
        return create_plotly_response(fig)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"要因影響分析の生成に失敗しました: {str(e)}")

@router.post("/lifecycle-trajectory")
async def create_lifecycle_trajectory(request: CompanyRequest):
    """
    企業ライフサイクル軌道可視化
    - 成長段階別分析
    - ライフサイクル遷移図
    - 成熟度指標
    """
    try:
        lifecycle_viz = LifecycleTrajectoryVisualizer()
        
        # 企業データ取得
        companies_data = []
        for company in request.company_names:
            company_trajectory = lifecycle_viz.calculate_lifecycle_trajectory(
                company_name=company,
                start_year=request.start_year,
                end_year=request.end_year
            )
            companies_data.append(company_trajectory)
        
        # 3Dライフサイクル軌道プロット
        fig = go.Figure()
        
        colors = px.colors.qualitative.Set1
        
        for i, company_data in enumerate(companies_data):
            # 3D軌道プロット
            fig.add_trace(
                go.Scatter3d(
                    x=company_data['成長率'],
                    y=company_data['収益性'],
                    z=company_data['効率性'],
                    mode='lines+markers',
                    marker=dict(
                        size=8,
                        color=company_data['年数'],
                        colorscale='Viridis',
                        showscale=True if i == 0 else False,
                        colorbar=dict(title="年数")
                    ),
                    line=dict(
                        width=4,
                        color=colors[i % len(colors)]
                    ),
                    name=company_data['company_name'].iloc[0],
                    text=company_data['year'],
                    hovertemplate='<b>%{fullData.name}</b><br>' +
                                '成長率: %{x:.2f}%<br>' +
                                '収益性: %{y:.2f}%<br>' +
                                '効率性: %{z:.2f}<br>' +
                                '年: %{text}<extra></extra>'
                )
            )
            
            # 開始点と終了点をマーク
            start_point = company_data.iloc[0]
            end_point = company_data.iloc[-1]
            
            # 開始点
            fig.add_trace(
                go.Scatter3d(
                    x=[start_point['成長率']],
                    y=[start_point['収益性']],
                    z=[start_point['効率性']],
                    mode='markers',
                    marker=dict(
                        size=12,
                        color='green',
                        symbol='diamond'
                    ),
                    name=f'{company_data["company_name"].iloc[0]} 開始',
                    showlegend=False
                )
            )
            
            # 終了点（企業が存続している場合）
            if not company_data['is_extinct'].iloc[-1]:
                fig.add_trace(
                    go.Scatter3d(
                        x=[end_point['成長率']],
                        y=[end_point['収益性']],
                        z=[end_point['効率性']],
                        mode='markers',
                        marker=dict(
                            size=12,
                            color='blue',
                            symbol='circle'
                        ),
                        name=f'{company_data["company_name"].iloc[0]} 現在',
                        showlegend=False
                    )
                )
            else:
                # 消滅点
                fig.add_trace(
                    go.Scatter3d(
                        x=[end_point['成長率']],
                        y=[end_point['収益性']],
                        z=[end_point['効率性']],
                        mode='markers',
                        marker=dict(
                            size=12,
                            color='red',
                            symbol='x'
                        ),
                        name=f'{company_data["company_name"].iloc[0]} 消滅',
                        showlegend=False
                    )
                )
        
        # レイアウト設定
        fig.update_layout(
            title='A2AI 企業ライフサイクル3D軌道分析',
            scene=dict(
                xaxis_title='成長率 (%)',
                yaxis_title='収益性 (ROE %)',
                zaxis_title='効率性 (総資産回転率)',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            height=700,
            title_x=0.5
        )
        
        return create_plotly_response(fig)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ライフサイクル軌道の生成に失敗しました: {str(e)}")

@router.post("/startup-success-journey")
async def create_startup_journey(
    market_category: MarketCategory = Query(..., description="対象市場カテゴリ"),
    analysis_years: int = Query(10, description="分析対象年数"),
    success_threshold: float = Query(1.2, description="成功判定閾値（成長率）")
):
    """
    新設企業の成功軌跡可視化
    - 成功企業vs失敗企業の軌跡比較
    - 成功要因分析
    - 市場参入タイミング分析
    """
    try:
        startup_viz = StartupJourneyVisualizer()
        success_analyzer = StartupSuccessAnalyzer()
        
        # 新設企業データの取得
        startup_data = success_analyzer.get_startup_companies(
            market_category=market_category.value,
            min_years=analysis_years
        )
        
        # 成功/失敗企業の分類
        success_classification = success_analyzer.classify_success_failure(
            startup_data, 
            success_threshold=success_threshold
        )
        
        # サブプロット作成
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                '成功企業 vs 失敗企業の成長軌跡',
                '成功要因重要度分析',
                '市場参入タイミング vs 成功率',
                '財務指標の推移比較'
            ),
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": True}]
            ]
        )
        
        # 1. 成長軌跡比較
        for success_type in ['success', 'failure']:
            companies = success_classification[success_type]
            
            for company in companies[:5]:  # 上位5社
                company_trajectory = startup_viz.get_company_trajectory(company)
                
                fig.add_trace(
                    go.Scatter(
                        x=company_trajectory['years_since_founding'],
                        y=company_trajectory['売上高成長率'],
                        mode='lines+markers',
                        name=f'{company} ({success_type})',
                        line=dict(
                            color='green' if success_type == 'success' else 'red',
                            width=3 if success_type == 'success' else 1,
                            dash='solid' if success_type == 'success' else 'dash'
                        )
                    ),
                    row=1, col=1
                )
        
        # 2. 成功要因重要度分析
        success_factors = success_analyzer.analyze_success_factors(success_classification)
        
        fig.add_trace(
            go.Bar(
                x=success_factors['importance_score'],
                y=success_factors['factor_name'],
                orientation='h',
                marker_color='lightgreen',
                name='成功要因重要度'
            ),
            row=1, col=2
        )
        
        # 3. 市場参入タイミング vs 成功率
        timing_analysis = success_analyzer.analyze_entry_timing(
            startup_data, market_category.value
        )
        
        fig.add_trace(
            go.Scatter(
                x=timing_analysis['entry_year'],
                y=timing_analysis['success_rate'],
                mode='markers+lines',
                marker=dict(
                    size=timing_analysis['company_count'],
                    sizemode='area',
                    sizeref=2.*max(timing_analysis['company_count'])/(50.**2),
                    sizemin=4,
                    color='blue'
                ),
                name='成功率推移',
                hovertemplate='参入年: %{x}<br>成功率: %{y:.1f}%<br>企業数: %{marker.size}<extra></extra>'
            ),
            row=2, col=1
        )
        
        # 4. 財務指標推移比較
        metrics = ['ROE', '売上高営業利益率', '総資産回転率']
        success_avg = success_analyzer.calculate_average_metrics(
            success_classification['success'], metrics
        )
        failure_avg = success_analyzer.calculate_average_metrics(
            success_classification['failure'], metrics
        )
        
        years = list(range(1, analysis_years + 1))
        
        for metric in metrics:
            # 成功企業の平均
            fig.add_trace(
                go.Scatter(
                    x=years,
                    y=success_avg[metric],
                    mode='lines',
                    name=f'成功企業{metric}',
                    line=dict(color='green', width=2)
                ),
                row=2, col=2
            )
            
            # 失敗企業の平均
            fig.add_trace(
                go.Scatter(
                    x=years,
                    y=failure_avg[metric],
                    mode='lines',
                    name=f'失敗企業{metric}',
                    line=dict(color='red', width=2, dash='dash')
                ),
                row=2, col=2
            )
        
        # レイアウト設定
        fig.update_layout(
            height=1000,
            title_text=f"A2AI 新設企業成功分析: {market_category.value.replace('_', ' ').title()}市場",
            showlegend=True,
            title_x=0.5
        )
        
        # 軸ラベル設定
        fig.update_xaxes(title_text="設立からの年数", row=1, col=1)
        fig.update_yaxes(title_text="売上高成長率 (%)", row=1, col=1)
        fig.update_xaxes(title_text="重要度スコア", row=1, col=2)
        fig.update_yaxes(title_text="成功要因", row=1, col=2)
        fig.update_xaxes(title_text="市場参入年", row=2, col=1)
        fig.update_yaxes(title_text="成功率 (%)", row=2, col=1)
        fig.update_xaxes(title_text="設立からの年数", row=2, col=2)
        fig.update_yaxes(title_text="財務指標値", row=2, col=2)
        
        return create_plotly_response(fig)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"スタートアップ分析の生成に失敗しました: {str(e)}")

@router.post("/ecosystem-network")
async def create_ecosystem_network(
    market_categories: List[MarketCategory] = Query(..., description="対象市場カテゴリ"),
    network_type: str = Query("competitive", description="ネットワーク種別"),
    time_snapshot: int = Query(2024, description="分析時点年")
):
    """
    市場エコシステムネットワーク可視化
    - 企業間関係性ネットワーク
    - 市場ポジション分析
    - 競争強度マップ
    """
    try:
        ecosystem_viz = EcosystemNetworkVisualizer()
        
        # エコシステムデータの構築
        ecosystem_data = ecosystem_viz.build_ecosystem_network(
            market_categories=[cat.value for cat in market_categories],
            snapshot_year=time_snapshot,
            network_type=network_type
        )
        
        # ネットワーク図の作成
        fig = go.Figure()
        
        # ノード（企業）の配置
        node_trace = go.Scatter(
            x=ecosystem_data['nodes']['x'],
            y=ecosystem_data['nodes']['y'],
            mode='markers+text',
            marker=dict(
                size=ecosystem_data['nodes']['size'],
                color=ecosystem_data['nodes']['color'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="市場影響力"),
                line=dict(width=2, color='black')
            ),
            text=ecosystem_data['nodes']['company_name'],
            textposition="middle center",
            textfont=dict(size=10, color='white'),
            hovertemplate='<b>%{text}</b><br>' +
                            '市場シェア: %{marker.size:.1f}%<br>' +
                            '影響力: %{marker.color:.2f}<br>' +
                            'カテゴリ: %{customdata}<extra></extra>',
            customdata=ecosystem_data['nodes']['market_category'],
            name='企業'
        )
        
        # エッジ（関係性）の描画
        edge_traces = []
        for edge in ecosystem_data['edges']:
            edge_trace = go.Scatter(
                x=[edge['x0'], edge['x1'], None],
                y=[edge['y0'], edge['y1'], None],
                mode='lines',
                line=dict(
                    width=edge['width'],
                    color=edge['color']
                ),
                hoverinfo='none',
                showlegend=False
            )
            edge_traces.append(edge_trace)
        
        # 全トレースを追加
        for edge_trace in edge_traces:
            fig.add_trace(edge_trace)
        
        fig.add_trace(node_trace)
        
        # レイアウト設定
        fig.update_layout(
            title=f'A2AI 市場エコシステムネットワーク ({time_snapshot}年)',
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            annotations=[ dict(
                text="ノードサイズ=市場シェア、色=影響力、線の太さ=関係強度",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.005, y=-0.002,
                xanchor='left', yanchor='bottom',
                font=dict(color='gray', size=12)
            )],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=700,
            title_x=0.5
        )
        
        return create_plotly_response(fig)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"エコシステムネットワークの生成に失敗しました: {str(e)}")

@router.post("/comprehensive-dashboard")
async def create_comprehensive_dashboard(request: DashboardRequest):
    """
    包括的ダッシュボード
    - 複数の分析結果を統合表示
    - リアルタイム指標監視
    - インタラクティブフィルタリング
    """
    try:
        # メイン指標の取得
        dashboard_data = get_market_data(
            market_focus=request.market_focus.value if request.market_focus else None,
            time_period=request.time_period,
            key_metrics=request.key_metrics
        )
        
        # 6x2のサブプロット作成
        fig = make_subplots(
            rows=3, cols=4,
            subplot_titles=(
                '市場別生存率', '要因重要度TOP10', '成長率分布', '収益性トレンド',
                '新設企業成功率', 'リスク要因ヒートマップ', '市場シェア推移', 'ROE vs 成長率',
                '予測精度評価', '異常値検出', '将来シナリオ', '統合スコア'
            ),
            specs=[
                [{"type": "bar"}, {"type": "bar"}, {"type": "histogram"}, {"type": "scatter"}],
                [{"type": "bar"}, {"type": "heatmap"}, {"type": "scatter"}, {"type": "scatter"}],
                [{"type": "bar"}, {"type": "scatter"}, {"type": "scatter"}, {"type": "indicator"}]
            ]
        )
        
        # 1. 市場別生存率
        survival_rates = dashboard_data['survival_analysis']['market_survival_rates']
        fig.add_trace(
            go.Bar(
                x=list(survival_rates.keys()),
                y=list(survival_rates.values()),
                marker_color=['green' if x > 80 else 'orange' if x > 60 else 'red' for x in survival_rates.values()],
                name='生存率'
            ),
            row=1, col=1
        )
        
        # 2. 要因重要度TOP10
        top_factors = dashboard_data['factor_analysis']['top_importance'][:10]
        fig.add_trace(
            go.Bar(
                x=[f['importance'] for f in top_factors],
                y=[f['factor_name'] for f in top_factors],
                orientation='h',
                marker_color='lightblue',
                name='重要度'
            ),
            row=1, col=2
        )
        
        # 3. 成長率分布
        growth_data = dashboard_data['performance_metrics']['growth_rates']
        fig.add_trace(
            go.Histogram(
                x=growth_data,
                nbinsx=30,
                marker_color='lightgreen',
                name='成長率分布'
            ),
            row=1, col=3
        )
        
        # 4. 収益性トレンド
        profitability_trend = dashboard_data['performance_metrics']['profitability_trend']
        fig.add_trace(
            go.Scatter(
                x=profitability_trend['year'],
                y=profitability_trend['avg_roe'],
                mode='lines+markers',
                line=dict(color='blue', width=3),
                name='平均ROE'
            ),
            row=1, col=4
        )
        
        # 5. 新設企業成功率
        startup_success = dashboard_data['emergence_analysis']['success_rates_by_year']
        fig.add_trace(
            go.Bar(
                x=list(startup_success.keys()),
                y=list(startup_success.values()),
                marker_color='purple',
                name='新設企業成功率'
            ),
            row=2, col=1
        )
        
        # 6. リスク要因ヒートマップ
        risk_matrix = dashboard_data['risk_analysis']['factor_risk_matrix']
        fig.add_trace(
            go.Heatmap(
                z=risk_matrix['values'],
                x=risk_matrix['factors'],
                y=risk_matrix['time_periods'],
                colorscale='Reds',
                name='リスクマップ'
            ),
            row=2, col=2
        )
        
        # 7. 市場シェア推移
        market_share_trend = dashboard_data['market_analysis']['share_trends']
        for market, trend in market_share_trend.items():
            fig.add_trace(
                go.Scatter(
                    x=trend['year'],
                    y=trend['japan_share'],
                    mode='lines',
                    name=f'{market}市場シェア',
                    line=dict(width=2)
                ),
                row=2, col=3
            )
        
        # 8. ROE vs 成長率バブルチャート
        performance_data = dashboard_data['performance_metrics']['roe_growth_correlation']
        fig.add_trace(
            go.Scatter(
                x=performance_data['growth_rate'],
                y=performance_data['roe'],
                mode='markers',
                marker=dict(
                    size=performance_data['market_cap'],
                    sizemode='area',
                    sizeref=2.*max(performance_data['market_cap'])/(50.**2),
                    sizemin=4,
                    color=performance_data['survival_probability'],
                    colorscale='RdYlGn',
                    showscale=True,
                    colorbar=dict(title="生存確率", x=0.75)
                ),
                text=performance_data['company_names'],
                name='企業ポジション'
            ),
            row=2, col=4
        )
        
        # 9. 予測精度評価
        model_accuracy = dashboard_data['model_evaluation']['prediction_accuracy']
        fig.add_trace(
            go.Bar(
                x=list(model_accuracy.keys()),
                y=list(model_accuracy.values()),
                marker_color=['green' if x > 0.8 else 'orange' if x > 0.6 else 'red' for x in model_accuracy.values()],
                name='予測精度'
            ),
            row=3, col=1
        )
        
        # 10. 異常値検出
        anomaly_data = dashboard_data['anomaly_detection']['detected_anomalies']
        fig.add_trace(
            go.Scatter(
                x=anomaly_data['timestamp'],
                y=anomaly_data['anomaly_score'],
                mode='markers',
                marker=dict(
                    color=anomaly_data['severity'],
                    colorscale='Reds',
                    size=8
                ),
                name='異常値'
            ),
            row=3, col=2
        )
        
        # 11. 将来シナリオ
        scenarios = dashboard_data['scenario_analysis']['future_scenarios']
        for scenario_name, scenario_data in scenarios.items():
            fig.add_trace(
                go.Scatter(
                    x=scenario_data['years'],
                    y=scenario_data['projected_performance'],
                    mode='lines',
                    name=f'{scenario_name}シナリオ',
                    line=dict(dash='dash' if 'pessimistic' in scenario_name else 'solid')
                ),
                row=3, col=3
            )
        
        # 12. 統合スコア（ゲージチャート）
        integrated_score = dashboard_data['integrated_metrics']['overall_health_score']
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=integrated_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "市場健康度"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "yellow"},
                        {'range': [80, 100], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ),
            row=3, col=4
        )
        
        # レイアウト設定
        fig.update_layout(
            height=1200,
            title_text="A2AI 包括的財務分析ダッシュボード",
            showlegend=True,
            title_x=0.5,
            title_font_size=20
        )
        
        # 個別軸の設定
        fig.update_xaxes(title_text="市場カテゴリ", row=1, col=1)
        fig.update_yaxes(title_text="生存率(%)", row=1, col=1)
        fig.update_xaxes(title_text="重要度", row=1, col=2)
        fig.update_yaxes(title_text="要因項目", row=1, col=2)
        fig.update_xaxes(title_text="成長率(%)", row=1, col=3)
        fig.update_yaxes(title_text="頻度", row=1, col=3)
        fig.update_xaxes(title_text="年", row=1, col=4)
        fig.update_yaxes(title_text="ROE(%)", row=1, col=4)
        
        return create_plotly_response(fig)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ダッシュボードの生成に失敗しました: {str(e)}")

@router.get("/chart-templates")
async def get_chart_templates():
    """
    利用可能なチャートテンプレートの一覧を返す
    """
    templates = {
        "survival_analysis": {
            "name": "生存分析",
            "description": "企業の生存確率とリスク要因を分析",
            "required_params": ["market_categories", "analysis_period"],
            "optional_params": ["risk_factors", "confidence_interval"]
        },
        "factor_impact": {
            "name": "要因影響分析",
            "description": "財務要因が評価指標に与える影響を分析",
            "required_params": ["target_metric", "factor_items"],
            "optional_params": ["companies", "market_filter"]
        },
        "lifecycle_trajectory": {
            "name": "ライフサイクル軌道",
            "description": "企業の成長・成熟・衰退の軌跡を3D可視化",
            "required_params": ["company_names"],
            "optional_params": ["start_year", "end_year", "market_categories"]
        },
        "startup_journey": {
            "name": "新設企業分析",
            "description": "スタートアップの成功要因と軌跡を分析",
            "required_params": ["market_category"],
            "optional_params": ["analysis_years", "success_threshold"]
        },
        "ecosystem_network": {
            "name": "エコシステムネットワーク",
            "description": "市場内の企業関係性と競争構造を可視化",
            "required_params": ["market_categories"],
            "optional_params": ["network_type", "time_snapshot"]
        },
        "comprehensive_dashboard": {
            "name": "包括的ダッシュボード",
            "description": "複数の分析結果を統合したダッシュボード",
            "required_params": [],
            "optional_params": ["dashboard_type", "market_focus", "time_period", "key_metrics"]
        }
    }
    
    return {"templates": templates, "timestamp": datetime.now().isoformat()}

@router.post("/custom-analysis")
async def create_custom_analysis(request: VisualizationRequest):
    """
    カスタム分析可視化
    - ユーザー定義の分析パラメータに基づく可視化
    - 複数の可視化タイプの組み合わせ
    """
    try:
        if request.visualization_type == VisualizationType.CORRELATION_HEATMAP:
            return await _create_correlation_heatmap(request)
        elif request.visualization_type == VisualizationType.TIME_SERIES:
            return await _create_time_series_analysis(request)
        else:
            raise HTTPException(status_code=400, detail=f"未対応の可視化タイプ: {request.visualization_type}")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"カスタム分析の生成に失敗しました: {str(e)}")

async def _create_correlation_heatmap(request: VisualizationRequest) -> Dict[str, Any]:
    """相関ヒートマップの生成"""
    # データ取得
    company_data = get_company_data(
        companies=request.companies,
        market_filter=request.market_categories[0].value if request.market_categories else None
    )
    
    # 相関行列計算
    factors_data = company_data[request.factor_items] if request.factor_items else company_data.select_dtypes(include=[np.number])
    correlation_matrix = factors_data.corr()
    
    # ヒートマップ作成
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.index,
        colorscale='RdBu',
        zmid=0,
        text=correlation_matrix.round(3).values,
        texttemplate="%{text}",
        textfont={"size": 10},
        hovertemplate='<b>%{y} vs %{x}</b><br>相関係数: %{z:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title='財務要因相関分析ヒートマップ',
        width=800,
        height=600,
        title_x=0.5
    )
    
    return create_plotly_response(fig)

async def _create_time_series_analysis(request: VisualizationRequest) -> Dict[str, Any]:
    """時系列分析の生成"""
    # データ取得
    time_range = request.time_range or {"start": 1984, "end": 2024}
    
    fig = go.Figure()
    
    # 各企業の時系列データを追加
    for i, company in enumerate(request.companies[:10]):  # 最大10社
        company_data = get_company_data([company])
        time_series_data = company_data[
            (company_data['year'] >= time_range['start']) & 
            (company_data['year'] <= time_range['end'])
        ]
        
        for metric in request.evaluation_metrics[:3]:  # 最大3指標
            fig.add_trace(
                go.Scatter(
                    x=time_series_data['year'],
                    y=time_series_data[metric],
                    mode='lines+markers',
                    name=f'{company}_{metric}',
                    line=dict(width=2),
                    hovertemplate=f'<b>{company}</b><br>年: %{{x}}<br>{metric}: %{{y:.2f}}<extra></extra>'
                )
            )
    
    fig.update_layout(
        title='企業別財務指標時系列推移',
        xaxis_title='年',
        yaxis_title='指標値',
        height=600,
        title_x=0.5,
        hovermode='x unified'
    )
    
    return create_plotly_response(fig)

@router.get("/export/{chart_type}")
async def export_chart(
    chart_type: str,
    format: str = Query("png", description="エクスポート形式 (png, pdf, svg, html)"),
    width: int = Query(1200, description="幅"),
    height: int = Query(800, description="高さ")
):
    """
    チャートのエクスポート機能
    - 様々な形式でのエクスポート対応
    - 高解像度出力
    """
    try:
        # 注意: 実際の実装では、セッション管理やキャッシュ機能が必要
        if format not in ["png", "pdf", "svg", "html"]:
            raise HTTPException(status_code=400, detail="サポートされていない形式です")
        
        # ダミーレスポンス（実際の実装では対応するチャートを生成）
        return {
            "message": f"{chart_type}チャートを{format}形式でエクスポートしました",
            "download_url": f"/downloads/{chart_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{format}",
            "file_size": f"{width*height//1000}KB",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"エクスポートに失敗しました: {str(e)}")

@router.websocket("/realtime-dashboard")
async def websocket_realtime_dashboard(websocket):
    """
    リアルタイムダッシュボード用WebSocket
    - リアルタイムデータ更新
    - インタラクティブフィルタリング
    """
    await websocket.accept()
    
    try:
        while True:
            # リアルタイムデータの取得（実際の実装では定期的なデータ更新）
            realtime_data = {
                "timestamp": datetime.now().isoformat(),
                "market_health_scores": {
                    "high_share": np.random.uniform(80, 95),
                    "declining": np.random.uniform(60, 80),
                    "lost": np.random.uniform(30, 60)
                },
                "new_alerts": [
                    {
                        "type": "risk_increase",
                        "company": "サンプル企業",
                        "message": "生存リスクが閾値を超えました",
                        "severity": "high"
                    }
                ],
                "updated_predictions": {
                    "survival_probability_changes": {
                        "improved": 5,
                        "declined": 3,
                        "stable": 142
                    }
                }
            }
            
            await websocket.send_json(realtime_data)
            await asyncio.sleep(30)  # 30秒間隔で更新
            
    except Exception as e:
        await websocket.close(code=1000, reason=f"エラーが発生しました: {str(e)}")

# ヘルスチェック用エンドポイント
@router.get("/health")
async def health_check():
    """APIヘルスチェック"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "available_visualizations": list(VisualizationType),
        "supported_formats": ["png", "pdf", "svg", "html", "json"]
    }