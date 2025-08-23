"""
A2AI (Advanced Financial Analysis AI) - Survival Analysis API Router
企業生存分析API - 企業の消滅リスク、生存確率、生存要因分析を提供

主要機能:
1. 企業生存確率予測
2. 消滅リスク分析
3. 生存要因特定
4. 市場カテゴリ別生存比較
5. ハザード率分析
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field
from datetime import datetime, date
import pandas as pd
import numpy as np
from enum import Enum
import logging

# A2AI internal imports
from ..models.survival_models.cox_regression import CoxRegressionModel
from ..models.survival_models.kaplan_meier import KaplanMeierEstimator
from ..models.survival_models.parametric_survival import ParametricSurvivalModel
from ..analysis.survival_analysis.extinction_risk_analyzer import ExtinctionRiskAnalyzer
from ..analysis.survival_analysis.survival_factor_analyzer import SurvivalFactorAnalyzer
from ..utils.survival_utils import SurvivalDataProcessor
from ..utils.statistical_utils import StatisticalValidator

# Router setup
router = APIRouter(prefix="/survival", tags=["survival_analysis"])
logger = logging.getLogger(__name__)

# ===== Enums and Constants =====

class MarketCategory(str, Enum):
    """市場カテゴリー定義"""
    HIGH_SHARE = "high_share"  # 世界シェア高市場
    DECLINING = "declining"    # シェア低下市場
    LOST = "lost"             # 完全失失市場

class SurvivalModel(str, Enum):
    """生存分析モデル種類"""
    COX = "cox_regression"
    KAPLAN_MEIER = "kaplan_meier"
    WEIBULL = "weibull"
    EXPONENTIAL = "exponential"
    LOG_NORMAL = "log_normal"

class TimeUnit(str, Enum):
    """時間単位"""
    YEARS = "years"
    QUARTERS = "quarters"
    MONTHS = "months"

# ===== Request/Response Models =====

class CompanyInfo(BaseModel):
    """企業基本情報"""
    company_id: str = Field(..., description="企業ID")
    company_name: str = Field(..., description="企業名")
    market_category: MarketCategory = Field(..., description="市場カテゴリー")
    industry: str = Field(..., description="業界")
    establishment_year: int = Field(..., description="設立年")
    extinction_year: Optional[int] = Field(None, description="消滅年（存続企業はNone）")

class SurvivalFactors(BaseModel):
    """生存分析用要因項目"""
    # 投資・資産関連
    tangible_assets: float = Field(..., description="有形固定資産")
    capex: float = Field(..., description="設備投資額")
    rd_expense: float = Field(..., description="研究開発費")
    intangible_assets: float = Field(..., description="無形固定資産")
    investment_securities: float = Field(..., description="投資有価証券")
    
    # 人的資源関連
    employee_count: int = Field(..., description="従業員数")
    average_salary: float = Field(..., description="平均年間給与")
    retirement_benefit_cost: float = Field(..., description="退職給付費用")
    welfare_cost: float = Field(..., description="福利厚生費")
    
    # 運転資本・効率性関連
    accounts_receivable: float = Field(..., description="売上債権")
    inventory: float = Field(..., description="棚卸資産")
    total_assets: float = Field(..., description="総資産")
    receivable_turnover: float = Field(..., description="売上債権回転率")
    inventory_turnover: float = Field(..., description="棚卸資産回転率")
    
    # 事業展開関連
    overseas_sales_ratio: float = Field(..., description="海外売上高比率")
    business_segment_count: int = Field(..., description="事業セグメント数")
    sga_expense: float = Field(..., description="販売費及び一般管理費")
    advertising_expense: float = Field(..., description="広告宣伝費")
    non_operating_income: float = Field(..., description="営業外収益")
    order_backlog: float = Field(..., description="受注残高")
    
    # 拡張要因項目
    company_age: int = Field(..., description="企業年齢")
    market_entry_timing: float = Field(..., description="市場参入時期")
    parent_dependency: float = Field(..., description="親会社依存度")

class SurvivalPredictionRequest(BaseModel):
    """生存確率予測リクエスト"""
    company_info: CompanyInfo
    survival_factors: SurvivalFactors
    prediction_horizon: int = Field(default=5, description="予測期間（年）")
    model_type: SurvivalModel = Field(default=SurvivalModel.COX, description="使用モデル")
    confidence_level: float = Field(default=0.95, description="信頼区間")

class ExtinctionRiskRequest(BaseModel):
    """消滅リスク分析リクエスト"""
    company_ids: List[str] = Field(..., description="分析対象企業IDリスト")
    risk_horizon: int = Field(default=3, description="リスク評価期間（年）")
    threshold_risk: float = Field(default=0.1, description="リスク閾値")

class SurvivalComparisonRequest(BaseModel):
    """生存比較分析リクエスト"""
    market_categories: List[MarketCategory] = Field(..., description="比較対象市場カテゴリー")
    time_period: tuple[int, int] = Field(..., description="分析期間（開始年, 終了年）")
    model_type: SurvivalModel = Field(default=SurvivalModel.KAPLAN_MEIER)

class SurvivalPredictionResponse(BaseModel):
    """生存確率予測レスポンス"""
    company_id: str
    survival_probability: Dict[int, float] = Field(..., description="年別生存確率")
    confidence_intervals: Dict[int, tuple[float, float]] = Field(..., description="信頼区間")
    risk_score: float = Field(..., description="総合リスクスコア")
    key_risk_factors: List[Dict[str, Any]] = Field(..., description="主要リスク要因")
    model_metrics: Dict[str, float] = Field(..., description="モデル評価指標")

class ExtinctionRiskResponse(BaseModel):
    """消滅リスク分析レスポンス"""
    high_risk_companies: List[Dict[str, Any]] = Field(..., description="高リスク企業")
    risk_distribution: Dict[str, int] = Field(..., description="リスク分布")
    risk_factors_ranking: List[Dict[str, Any]] = Field(..., description="リスク要因ランキング")
    early_warning_signals: List[Dict[str, Any]] = Field(..., description="早期警告シグナル")

class SurvivalComparisonResponse(BaseModel):
    """生存比較分析レスポンス"""
    market_survival_curves: Dict[str, Dict[int, float]] = Field(..., description="市場別生存曲線")
    median_survival_time: Dict[str, float] = Field(..., description="市場別中央生存時間")
    hazard_ratios: Dict[str, float] = Field(..., description="市場別ハザード比")
    statistical_significance: Dict[str, float] = Field(..., description="統計的有意性")
    survival_factors_comparison: Dict[str, List[Dict]] = Field(..., description="生存要因比較")

# ===== Dependency Functions =====

async def get_survival_analyzer() -> SurvivalFactorAnalyzer:
    """生存分析器の依存性注入"""
    return SurvivalFactorAnalyzer()

async def get_extinction_analyzer() -> ExtinctionRiskAnalyzer:
    """消滅リスク分析器の依存性注入"""
    return ExtinctionRiskAnalyzer()

async def get_data_processor() -> SurvivalDataProcessor:
    """データ処理器の依存性注入"""
    return SurvivalDataProcessor()

# ===== API Endpoints =====

@router.post("/predict", response_model=SurvivalPredictionResponse)
async def predict_survival_probability(
    request: SurvivalPredictionRequest,
    analyzer: SurvivalFactorAnalyzer = Depends(get_survival_analyzer)
) -> SurvivalPredictionResponse:
    """
    企業の生存確率予測
    
    指定された企業の財務要因を基に、将来の生存確率を予測します。
    Cox回帰、Weibull分布等の生存分析モデルを使用。
    """
    try:
        logger.info(f"生存確率予測開始: {request.company_info.company_name}")
        
        # データ前処理
        features = _convert_factors_to_features(request.survival_factors)
        
        # モデル選択と予測実行
        if request.model_type == SurvivalModel.COX:
            model = CoxRegressionModel()
            results = await model.predict_survival(
                features=features,
                horizon=request.prediction_horizon,
                confidence_level=request.confidence_level
            )
        elif request.model_type == SurvivalModel.WEIBULL:
            model = ParametricSurvivalModel(distribution="weibull")
            results = await model.predict_survival(
                features=features,
                horizon=request.prediction_horizon
            )
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported model: {request.model_type}")
        
        # リスク要因分析
        risk_factors = await analyzer.identify_risk_factors(
            company_data=features,
            market_category=request.company_info.market_category
        )
        
        # レスポンス構築
        response = SurvivalPredictionResponse(
            company_id=request.company_info.company_id,
            survival_probability=results["survival_probabilities"],
            confidence_intervals=results["confidence_intervals"],
            risk_score=results["risk_score"],
            key_risk_factors=risk_factors,
            model_metrics=results["model_metrics"]
        )
        
        logger.info(f"生存確率予測完了: {request.company_info.company_name}")
        return response
        
    except Exception as e:
        logger.error(f"生存確率予測エラー: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@router.post("/extinction-risk", response_model=ExtinctionRiskResponse)
async def analyze_extinction_risk(
    request: ExtinctionRiskRequest,
    analyzer: ExtinctionRiskAnalyzer = Depends(get_extinction_analyzer)
) -> ExtinctionRiskResponse:
    """
    企業消滅リスク分析
    
    複数企業の消滅リスクを評価し、高リスク企業の特定と
    早期警告シグナルを提供します。
    """
    try:
        logger.info(f"消滅リスク分析開始: {len(request.company_ids)}社")
        
        # リスク分析実行
        risk_analysis = await analyzer.analyze_extinction_risk(
            company_ids=request.company_ids,
            risk_horizon=request.risk_horizon,
            threshold=request.threshold_risk
        )
        
        # 高リスク企業特定
        high_risk_companies = await analyzer.identify_high_risk_companies(
            risk_analysis["risk_scores"],
            threshold=request.threshold_risk
        )
        
        # 早期警告シグナル検出
        warning_signals = await analyzer.detect_early_warning_signals(
            company_ids=request.company_ids,
            lookback_period=2
        )
        
        response = ExtinctionRiskResponse(
            high_risk_companies=high_risk_companies,
            risk_distribution=risk_analysis["risk_distribution"],
            risk_factors_ranking=risk_analysis["risk_factors_ranking"],
            early_warning_signals=warning_signals
        )
        
        logger.info(f"消滅リスク分析完了: 高リスク企業{len(high_risk_companies)}社特定")
        return response
        
    except Exception as e:
        logger.error(f"消滅リスク分析エラー: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Risk analysis failed: {str(e)}")

@router.post("/market-comparison", response_model=SurvivalComparisonResponse)
async def compare_market_survival(
    request: SurvivalComparisonRequest,
    analyzer: SurvivalFactorAnalyzer = Depends(get_survival_analyzer)
) -> SurvivalComparisonResponse:
    """
    市場間生存比較分析
    
    異なる市場カテゴリー（高シェア/低下/失失）間での
    企業生存パターンを比較分析します。
    """
    try:
        logger.info(f"市場間生存比較開始: {request.market_categories}")
        
        # 市場別生存分析実行
        comparison_results = await analyzer.compare_market_survival(
            market_categories=request.market_categories,
            time_period=request.time_period,
            model_type=request.model_type
        )
        
        # 統計的有意性検定
        significance_tests = await analyzer.test_survival_differences(
            comparison_results["survival_data"]
        )
        
        response = SurvivalComparisonResponse(
            market_survival_curves=comparison_results["survival_curves"],
            median_survival_time=comparison_results["median_survival_times"],
            hazard_ratios=comparison_results["hazard_ratios"],
            statistical_significance=significance_tests,
            survival_factors_comparison=comparison_results["factors_comparison"]
        )
        
        logger.info("市場間生存比較完了")
        return response
        
    except Exception as e:
        logger.error(f"市場間生存比較エラー: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Market comparison failed: {str(e)}")

@router.get("/survival-factors/{market_category}")
async def get_survival_factors_importance(
    market_category: MarketCategory,
    top_n: int = Query(default=10, description="上位N要因"),
    analyzer: SurvivalFactorAnalyzer = Depends(get_survival_analyzer)
) -> Dict[str, Any]:
    """
    市場別生存要因重要度取得
    
    指定した市場カテゴリーにおける企業生存に
    最も重要な要因項目を重要度順に返します。
    """
    try:
        logger.info(f"生存要因重要度取得: {market_category}")
        
        # 生存要因重要度分析
        importance_results = await analyzer.analyze_factor_importance(
            market_category=market_category,
            top_n=top_n
        )
        
        return {
            "market_category": market_category,
            "factor_importance": importance_results["factor_importance"],
            "statistical_significance": importance_results["statistical_significance"],
            "effect_directions": importance_results["effect_directions"],
            "confidence_intervals": importance_results["confidence_intervals"]
        }
        
    except Exception as e:
        logger.error(f"生存要因重要度取得エラー: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Factor importance analysis failed: {str(e)}")

@router.get("/hazard-analysis/{company_id}")
async def analyze_hazard_rate(
    company_id: str,
    time_window: int = Query(default=5, description="分析時間窓（年）"),
    analyzer: SurvivalFactorAnalyzer = Depends(get_survival_analyzer)
) -> Dict[str, Any]:
    """
    企業ハザード率分析
    
    指定企業の時間経過に伴うハザード率（瞬間消滅確率）の
    変化を分析します。
    """
    try:
        logger.info(f"ハザード率分析開始: {company_id}")
        
        # ハザード率分析実行
        hazard_analysis = await analyzer.analyze_hazard_rate(
            company_id=company_id,
            time_window=time_window
        )
        
        return {
            "company_id": company_id,
            "hazard_rates": hazard_analysis["hazard_rates"],
            "cumulative_hazard": hazard_analysis["cumulative_hazard"],
            "peak_risk_periods": hazard_analysis["peak_risk_periods"],
            "hazard_factors": hazard_analysis["contributing_factors"]
        }
        
    except Exception as e:
        logger.error(f"ハザード率分析エラー: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Hazard analysis failed: {str(e)}")

@router.get("/survival-timeline/{market_category}")
async def get_survival_timeline(
    market_category: MarketCategory,
    start_year: int = Query(..., description="開始年"),
    end_year: int = Query(..., description="終了年"),
    processor: SurvivalDataProcessor = Depends(get_data_processor)
) -> Dict[str, Any]:
    """
    市場別生存タイムライン取得
    
    指定期間における市場カテゴリー別の企業生存・消滅の
    時系列推移を返します。
    """
    try:
        logger.info(f"生存タイムライン取得: {market_category} ({start_year}-{end_year})")
        
        # 生存タイムライン構築
        timeline = await processor.build_survival_timeline(
            market_category=market_category,
            start_year=start_year,
            end_year=end_year
        )
        
        return {
            "market_category": market_category,
            "time_period": f"{start_year}-{end_year}",
            "survival_events": timeline["survival_events"],
            "extinction_events": timeline["extinction_events"],
            "emergence_events": timeline["emergence_events"],
            "survival_statistics": timeline["statistics"]
        }
        
    except Exception as e:
        logger.error(f"生存タイムライン取得エラー: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Timeline generation failed: {str(e)}")

# ===== Utility Functions =====

def _convert_factors_to_features(factors: SurvivalFactors) -> Dict[str, float]:
    """要因項目を特徴量ベクトルに変換"""
    return {
        "tangible_assets": factors.tangible_assets,
        "capex": factors.capex,
        "rd_expense": factors.rd_expense,
        "intangible_assets": factors.intangible_assets,
        "investment_securities": factors.investment_securities,
        "employee_count": float(factors.employee_count),
        "average_salary": factors.average_salary,
        "retirement_benefit_cost": factors.retirement_benefit_cost,
        "welfare_cost": factors.welfare_cost,
        "accounts_receivable": factors.accounts_receivable,
        "inventory": factors.inventory,
        "total_assets": factors.total_assets,
        "receivable_turnover": factors.receivable_turnover,
        "inventory_turnover": factors.inventory_turnover,
        "overseas_sales_ratio": factors.overseas_sales_ratio,
        "business_segment_count": float(factors.business_segment_count),
        "sga_expense": factors.sga_expense,
        "advertising_expense": factors.advertising_expense,
        "non_operating_income": factors.non_operating_income,
        "order_backlog": factors.order_backlog,
        "company_age": float(factors.company_age),
        "market_entry_timing": factors.market_entry_timing,
        "parent_dependency": factors.parent_dependency
    }

# ===== Health Check =====

@router.get("/health")
async def health_check():
    """生存分析API健全性チェック"""
    return {
        "status": "healthy",
        "service": "survival_analysis",
        "timestamp": datetime.now().isoformat(),
        "available_models": [model.value for model in SurvivalModel],
        "supported_markets": [market.value for market in MarketCategory]
    }