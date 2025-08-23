"""
A2AI - Advanced Financial Analysis AI
Emergence Analysis API Router

新設企業分析に特化したAPIエンドポイント群
企業の設立・分社・スピンオフ後の成功要因分析を提供
"""

from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from typing import List, Optional, Dict, Any, Union
from datetime import datetime, date
from pydantic import BaseModel, Field, validator
import pandas as pd
import numpy as np
from sqlalchemy.orm import Session

# A2AI内部モジュールのインポート
from src.analysis.emergence_analysis.startup_success_analyzer import StartupSuccessAnalyzer
from src.analysis.emergence_analysis.market_entry_analyzer import MarketEntryAnalyzer
from src.analysis.emergence_analysis.growth_phase_analyzer import GrowthPhaseAnalyzer
from src.analysis.emergence_analysis.innovation_impact_analyzer import InnovationImpactAnalyzer
from src.models.emergence_models.success_prediction import SuccessPredictor
from src.models.emergence_models.growth_trajectory import GrowthTrajectoryModel
from src.models.emergence_models.market_entry_timing import MarketEntryTimingModel
from src.feature_engineering.emergence_features import EmergenceFeatureEngine
from src.visualization.emergence_viz.startup_journey import StartupJourneyVisualizer
from src.visualization.emergence_viz.success_factors import SuccessFactorVisualizer
from src.utils.database_utils import get_db_session
from src.utils.data_utils import validate_company_code, normalize_financial_data
from config.settings import settings

# APIルーター初期化
router = APIRouter(
    prefix="/emergence",
    tags=["emergence_analysis"],
    responses={404: {"description": "Analysis not found"}}
)

# ===== APIリクエスト・レスポンスモデル =====

class EmergenceCompanyBase(BaseModel):
    """新設企業基本情報"""
    company_code: str = Field(..., description="企業コード")
    company_name: str = Field(..., description="企業名")
    establishment_date: date = Field(..., description="設立日")
    parent_company_code: Optional[str] = Field(None, description="親会社コード（分社の場合）")
    market_category: str = Field(..., description="市場カテゴリ（high_share/declining/lost）")
    industry_sector: str = Field(..., description="業界セクター")

class EmergenceAnalysisRequest(BaseModel):
    """新設企業分析リクエスト"""
    company_codes: List[str] = Field(..., description="分析対象企業コード群")
    analysis_start_date: Optional[date] = Field(None, description="分析開始日")
    analysis_end_date: Optional[date] = Field(None, description="分析終了日")
    benchmark_companies: Optional[List[str]] = Field(None, description="ベンチマーク企業コード群")
    analysis_type: str = Field(
        "comprehensive", 
        description="分析タイプ（comprehensive/success_factors/growth_trajectory/market_entry）"
    )
    include_visualization: bool = Field(True, description="可視化結果を含むか")
    
    @validator('analysis_type')
    def validate_analysis_type(cls, v):
        allowed_types = ['comprehensive', 'success_factors', 'growth_trajectory', 'market_entry']
        if v not in allowed_types:
            raise ValueError(f"analysis_type must be one of {allowed_types}")
        return v

class SuccessFactorMetrics(BaseModel):
    """成功要因メトリクス"""
    factor_name: str
    impact_score: float = Field(..., ge=0, le=1, description="影響度スコア（0-1）")
    confidence_interval: List[float] = Field(..., description="信頼区間 [下限, 上限]")
    statistical_significance: float = Field(..., description="統計的有意性（p値）")
    factor_category: str = Field(..., description="要因カテゴリ")

class GrowthPhaseData(BaseModel):
    """成長段階データ"""
    phase_name: str
    start_date: date
    end_date: Optional[date]
    revenue_growth_rate: float
    employee_growth_rate: float
    rd_investment_ratio: float
    profitability_score: float
    market_share_change: float

class EmergenceAnalysisResponse(BaseModel):
    """新設企業分析レスポンス"""
    analysis_id: str
    company_code: str
    company_name: str
    establishment_date: date
    analysis_timestamp: datetime
    
    # 基本統計
    operational_years: int
    current_market_position: str
    survival_probability: float = Field(..., ge=0, le=1)
    
    # 成功要因分析
    success_factors: List[SuccessFactorMetrics]
    success_score: float = Field(..., ge=0, le=1)
    
    # 成長軌道分析
    growth_phases: List[GrowthPhaseData]
    current_growth_phase: str
    projected_growth_trajectory: Dict[str, Any]
    
    # 市場参入分析
    market_entry_timing_score: float
    entry_strategy_effectiveness: Dict[str, float]
    competitive_advantage_sources: List[str]
    
    # 比較分析
    benchmark_comparison: Optional[Dict[str, Any]]
    industry_ranking: Optional[Dict[str, int]]
    
    # 予測・提言
    future_success_probability: Dict[str, float]  # {1年後, 3年後, 5年後}
    strategic_recommendations: List[str]
    risk_factors: List[str]
    
    # 可視化データ
    visualization_data: Optional[Dict[str, Any]]

class BulkEmergenceAnalysisRequest(BaseModel):
    """複数企業一括分析リクエスト"""
    company_codes: List[str] = Field(..., min_items=1, max_items=50)
    comparison_analysis: bool = Field(True, description="企業間比較分析を実施するか")
    market_segmentation: bool = Field(True, description="市場セグメント別分析を実施するか")
    output_format: str = Field("json", description="出力形式（json/csv/excel）")

class MarketEntrySuccessFactors(BaseModel):
    """市場参入成功要因分析結果"""
    market_category: str
    total_entries: int
    success_rate: float
    average_time_to_profitability: float
    key_success_factors: List[SuccessFactorMetrics]
    failure_patterns: List[Dict[str, Any]]

# ===== API依存関数 =====

def get_emergence_analyzer() -> StartupSuccessAnalyzer:
    """新設企業分析器の依存注入"""
    return StartupSuccessAnalyzer()

def get_market_entry_analyzer() -> MarketEntryAnalyzer:
    """市場参入分析器の依存注入"""
    return MarketEntryAnalyzer()

def get_growth_analyzer() -> GrowthPhaseAnalyzer:
    """成長段階分析器の依存注入"""
    return GrowthPhaseAnalyzer()

def get_success_predictor() -> SuccessPredictor:
    """成功予測モデルの依存注入"""
    return SuccessPredictor()

# ===== APIエンドポイント =====

@router.get("/companies/emerging", response_model=List[EmergenceCompanyBase])
async def get_emerging_companies(
    market_category: Optional[str] = Query(None, description="市場カテゴリでフィルタ"),
    establishment_year_from: Optional[int] = Query(None, description="設立年（開始）"),
    establishment_year_to: Optional[int] = Query(None, description="設立年（終了）"),
    min_operational_years: Optional[int] = Query(None, description="最小運営年数"),
    db: Session = Depends(get_db_session)
):
    """
    新設企業一覧取得
    
    指定された条件に基づいて新設企業の一覧を取得します。
    設立年、市場カテゴリ、運営年数などでフィルタリング可能。
    """
    try:
        # データベースから新設企業データを取得
        query = """
        SELECT company_code, company_name, establishment_date, 
                parent_company_code, market_category, industry_sector
        FROM emergence_companies ec
        WHERE 1=1
        """
        params = {}
        
        if market_category:
            query += " AND market_category = :market_category"
            params["market_category"] = market_category
            
        if establishment_year_from:
            query += " AND EXTRACT(YEAR FROM establishment_date) >= :year_from"
            params["year_from"] = establishment_year_from
            
        if establishment_year_to:
            query += " AND EXTRACT(YEAR FROM establishment_date) <= :year_to"
            params["year_to"] = establishment_year_to
            
        if min_operational_years:
            query += " AND (CURRENT_DATE - establishment_date) / 365 >= :min_years"
            params["min_years"] = min_operational_years
            
        result = db.execute(query, params).fetchall()
        
        return [
            EmergenceCompanyBase(
                company_code=row.company_code,
                company_name=row.company_name,
                establishment_date=row.establishment_date,
                parent_company_code=row.parent_company_code,
                market_category=row.market_category,
                industry_sector=row.industry_sector
            )
            for row in result
        ]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch emerging companies: {str(e)}")

@router.post("/analyze/single", response_model=EmergenceAnalysisResponse)
async def analyze_single_emergence(
    request: EmergenceAnalysisRequest,
    background_tasks: BackgroundTasks,
    analyzer: StartupSuccessAnalyzer = Depends(get_emergence_analyzer),
    predictor: SuccessPredictor = Depends(get_success_predictor),
    db: Session = Depends(get_db_session)
):
    """
    単一新設企業詳細分析
    
    指定された新設企業の包括的な成功要因分析を実施します。
    成長軌道、市場参入戦略、将来予測を含む多角的分析。
    """
    try:
        company_code = request.company_codes[0]  # 単一分析では最初の企業コードを使用
        
        # 企業基本情報の取得・検証
        if not validate_company_code(company_code, db):
            raise HTTPException(status_code=404, detail=f"Company {company_code} not found")
        
        # 財務データ取得
        financial_data = analyzer.get_financial_data(
            company_code, 
            request.analysis_start_date, 
            request.analysis_end_date
        )
        
        # 成功要因分析
        success_analysis = analyzer.analyze_success_factors(
            company_code, 
            financial_data,
            benchmark_companies=request.benchmark_companies
        )
        
        # 成長段階分析
        growth_analyzer = GrowthPhaseAnalyzer()
        growth_analysis = growth_analyzer.analyze_growth_phases(company_code, financial_data)
        
        # 市場参入分析
        entry_analyzer = MarketEntryAnalyzer()
        entry_analysis = entry_analyzer.analyze_market_entry(company_code, financial_data)
        
        # 将来予測
        future_predictions = predictor.predict_future_success(company_code, financial_data)
        
        # 戦略提言生成
        strategic_recommendations = analyzer.generate_strategic_recommendations(
            company_code, success_analysis, growth_analysis, entry_analysis
        )
        
        # 可視化データ生成（バックグラウンドタスクで実行）
        visualization_data = None
        if request.include_visualization:
            visualizer = StartupJourneyVisualizer()
            background_tasks.add_task(
                visualizer.generate_comprehensive_visualization, 
                company_code, financial_data
            )
            visualization_data = {"status": "generating", "estimated_completion": "2-3 minutes"}
        
        # レスポンス構築
        return EmergenceAnalysisResponse(
            analysis_id=f"emergence_{company_code}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            company_code=company_code,
            company_name=success_analysis['company_info']['name'],
            establishment_date=success_analysis['company_info']['establishment_date'],
            analysis_timestamp=datetime.now(),
            operational_years=success_analysis['basic_stats']['operational_years'],
            current_market_position=success_analysis['basic_stats']['market_position'],
            survival_probability=success_analysis['basic_stats']['survival_probability'],
            success_factors=[
                SuccessFactorMetrics(
                    factor_name=factor['name'],
                    impact_score=factor['impact_score'],
                    confidence_interval=factor['confidence_interval'],
                    statistical_significance=factor['p_value'],
                    factor_category=factor['category']
                )
                for factor in success_analysis['success_factors']
            ],
            success_score=success_analysis['overall_success_score'],
            growth_phases=[
                GrowthPhaseData(
                    phase_name=phase['name'],
                    start_date=phase['start_date'],
                    end_date=phase.get('end_date'),
                    revenue_growth_rate=phase['revenue_growth_rate'],
                    employee_growth_rate=phase['employee_growth_rate'],
                    rd_investment_ratio=phase['rd_investment_ratio'],
                    profitability_score=phase['profitability_score'],
                    market_share_change=phase['market_share_change']
                )
                for phase in growth_analysis['phases']
            ],
            current_growth_phase=growth_analysis['current_phase'],
            projected_growth_trajectory=growth_analysis['projections'],
            market_entry_timing_score=entry_analysis['timing_score'],
            entry_strategy_effectiveness=entry_analysis['strategy_effectiveness'],
            competitive_advantage_sources=entry_analysis['competitive_advantages'],
            benchmark_comparison=success_analysis.get('benchmark_comparison'),
            industry_ranking=success_analysis.get('industry_ranking'),
            future_success_probability=future_predictions['success_probabilities'],
            strategic_recommendations=strategic_recommendations['recommendations'],
            risk_factors=strategic_recommendations['risk_factors'],
            visualization_data=visualization_data
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router.post("/analyze/bulk", response_model=List[EmergenceAnalysisResponse])
async def analyze_bulk_emergence(
    request: BulkEmergenceAnalysisRequest,
    background_tasks: BackgroundTasks,
    analyzer: StartupSuccessAnalyzer = Depends(get_emergence_analyzer),
    db: Session = Depends(get_db_session)
):
    """
    複数新設企業一括分析
    
    複数の新設企業を一括で分析し、比較結果を提供します。
    市場セグメント別の成功パターン分析も含む。
    """
    try:
        results = []
        
        for company_code in request.company_codes:
            # 各企業の基本分析実行
            analysis_request = EmergenceAnalysisRequest(
                company_codes=[company_code],
                analysis_type="comprehensive",
                include_visualization=False  # 一括分析では可視化は無効
            )
            
            # 単一分析APIを内部呼び出し
            single_result = await analyze_single_emergence(
                analysis_request, background_tasks, analyzer, db=db
            )
            results.append(single_result)
        
        # 企業間比較分析
        if request.comparison_analysis:
            comparison_analyzer = analyzer
            comparison_results = comparison_analyzer.compare_emergence_companies(
                request.company_codes
            )
            
            # 比較結果を各企業の結果に追加
            for i, result in enumerate(results):
                result.benchmark_comparison = comparison_results[i]
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Bulk analysis failed: {str(e)}")

@router.get("/market-entry/success-factors/{market_category}", response_model=MarketEntrySuccessFactors)
async def get_market_entry_success_factors(
    market_category: str,
    entry_year_from: Optional[int] = Query(None, description="参入年（開始）"),
    entry_year_to: Optional[int] = Query(None, description="参入年（終了）"),
    analyzer: MarketEntryAnalyzer = Depends(get_market_entry_analyzer),
    db: Session = Depends(get_db_session)
):
    """
    市場別参入成功要因分析
    
    指定された市場カテゴリにおける新規参入企業の成功要因を分析します。
    成功率、成功要因、失敗パターンを統計的に提供。
    """
    try:
        # 市場参入データ取得
        entry_data = analyzer.get_market_entry_data(
            market_category, entry_year_from, entry_year_to
        )
        
        # 成功要因分析実行
        success_factors_analysis = analyzer.analyze_entry_success_factors(entry_data)
        
        return MarketEntrySuccessFactors(
            market_category=market_category,
            total_entries=success_factors_analysis['total_entries'],
            success_rate=success_factors_analysis['success_rate'],
            average_time_to_profitability=success_factors_analysis['avg_time_to_profit'],
            key_success_factors=[
                SuccessFactorMetrics(
                    factor_name=factor['name'],
                    impact_score=factor['impact_score'],
                    confidence_interval=factor['confidence_interval'],
                    statistical_significance=factor['p_value'],
                    factor_category=factor['category']
                )
                for factor in success_factors_analysis['key_factors']
            ],
            failure_patterns=success_factors_analysis['failure_patterns']
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Market entry analysis failed: {str(e)}")

@router.post("/predict/success")
async def predict_emergence_success(
    company_code: str,
    prediction_horizon_years: int = Query(5, ge=1, le=10),
    scenario: str = Query("base", description="予測シナリオ（optimistic/base/pessimistic）"),
    predictor: SuccessPredictor = Depends(get_success_predictor)
):
    """
    新設企業成功確率予測
    
    機械学習モデルを用いて新設企業の将来成功確率を予測します。
    複数シナリオでの予測結果を提供。
    """
    try:
        # 現在の財務データ取得
        current_data = predictor.get_current_financial_state(company_code)
        
        # 成功確率予測実行
        prediction_results = predictor.predict_success_probability(
            company_code, 
            current_data, 
            prediction_horizon_years,
            scenario
        )
        
        return {
            "company_code": company_code,
            "prediction_timestamp": datetime.now(),
            "prediction_horizon_years": prediction_horizon_years,
            "scenario": scenario,
            "success_probability": prediction_results['success_probability'],
            "confidence_interval": prediction_results['confidence_interval'],
            "key_risk_factors": prediction_results['risk_factors'],
            "key_success_drivers": prediction_results['success_drivers'],
            "scenario_analysis": prediction_results['scenario_breakdown'],
            "model_performance_metrics": prediction_results['model_metrics']
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@router.get("/analytics/market-trends")
async def get_emergence_market_trends(
    time_period: str = Query("10years", description="分析期間（5years/10years/20years）"),
    market_categories: Optional[List[str]] = Query(None, description="分析対象市場カテゴリ"),
    analyzer: StartupSuccessAnalyzer = Depends(get_emergence_analyzer)
):
    """
    新設企業市場トレンド分析
    
    過去の新設企業データから市場トレンドと成功パターンの変化を分析します。
    時系列での成功要因の変遷を可視化。
    """
    try:
        # トレンド分析実行
        trend_analysis = analyzer.analyze_emergence_trends(
            time_period, market_categories
        )
        
        return {
            "analysis_period": time_period,
            "market_categories": market_categories or ["all"],
            "trend_summary": trend_analysis['summary'],
            "success_rate_trends": trend_analysis['success_rate_evolution'],
            "factor_importance_changes": trend_analysis['factor_evolution'],
            "market_entry_patterns": trend_analysis['entry_patterns'],
            "innovation_impact_trends": trend_analysis['innovation_trends'],
            "survival_rate_changes": trend_analysis['survival_evolution'],
            "recommendations": trend_analysis['strategic_insights']
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Trend analysis failed: {str(e)}")

@router.get("/health")
async def health_check():
    """APIヘルスチェック"""
    return {
        "status": "healthy",
        "service": "emergence_analysis",
        "timestamp": datetime.now(),
        "version": "1.0.0"
    }