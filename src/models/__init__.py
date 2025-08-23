"""
A2AI (Advanced Financial Analysis AI) - Models Module

This module provides comprehensive financial analysis models for analyzing 
150 companies across 3 market categories (high-share, declining, lost-share)
with complete lifecycle analysis including survival, emergence, and causal inference.

Key Features:
- Traditional financial analysis models
- Survival analysis for company extinction prediction
- Emergence analysis for startup/spinoff success prediction
- Causal inference models for true factor impact analysis
- Integrated lifecycle analysis models

Author: A2AI Development Team
Version: 1.0.0
"""

from typing import Dict, List, Optional, Union, Any
import logging
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Model version and metadata
__version__ = "1.0.0"
__author__ = "A2AI Development Team"
__email__ = "development@a2ai.com"

# Model categories and their descriptions
MODEL_CATEGORIES = {
    "traditional": {
        "description": "Traditional financial analysis models for conventional metrics",
        "models": ["regression", "ensemble", "deep_learning", "time_series"],
        "evaluation_items": [
            "sales_revenue", "sales_growth_rate", "operating_profit_margin",
            "net_profit_margin", "roe", "value_added_ratio"
        ]
    },
    "survival": {
        "description": "Survival analysis models for company extinction prediction",
        "models": ["cox_regression", "kaplan_meier", "parametric_survival", "ml_survival"],
        "evaluation_items": ["company_survival_probability"]
    },
    "emergence": {
        "description": "Analysis models for startup/spinoff companies",
        "models": ["success_prediction", "growth_trajectory", "market_entry_timing"],
        "evaluation_items": ["startup_success_rate"]
    },
    "causal": {
        "description": "Causal inference models for true factor impact analysis",
        "models": ["did", "instrumental_variables", "propensity_score", "causal_forest"],
        "evaluation_items": ["business_succession_success"]
    },
    "integrated": {
        "description": "Integrated analysis models combining multiple approaches",
        "models": ["multi_stage", "lifecycle_trajectory", "market_ecosystem"],
        "evaluation_items": ["all_extended_metrics"]
    }
}

# Market categories from the provided company list
MARKET_CATEGORIES = {
    "high_share": {
        "description": "Markets where Japanese companies maintain high global share",
        "markets": ["robotics", "endoscopes", "machine_tools", "electronic_materials", "precision_instruments"],
        "companies_per_market": 10,
        "total_companies": 50,
        "analysis_focus": "sustainability_factors"
    },
    "declining": {
        "description": "Markets where Japanese companies are losing share",
        "markets": ["automotive", "steel", "smart_appliances", "batteries", "pc_peripherals"],
        "companies_per_market": 10,
        "total_companies": 50,
        "analysis_focus": "decline_factors"
    },
    "lost": {
        "description": "Markets where Japanese companies have completely lost share",
        "markets": ["consumer_electronics", "semiconductors", "smartphones", "pc", "telecommunications"],
        "companies_per_market": 10,
        "total_companies": 50,
        "analysis_focus": "extinction_factors"
    }
}

# Extended evaluation items (9 total: 6 traditional + 3 new)
EVALUATION_ITEMS = {
    # Traditional 6 items
    "sales_revenue": {
        "description": "Total sales revenue with 20 factor items",
        "factor_count": 23,  # 20 original + 3 extended
        "analysis_type": "traditional"
    },
    "sales_growth_rate": {
        "description": "Sales growth rate with expansion factors",
        "factor_count": 23,
        "analysis_type": "traditional"
    },
    "operating_profit_margin": {
        "description": "Operating profit margin efficiency factors",
        "factor_count": 23,
        "analysis_type": "traditional"
    },
    "net_profit_margin": {
        "description": "Net profit margin comprehensive factors",
        "factor_count": 23,
        "analysis_type": "traditional"
    },
    "roe": {
        "description": "Return on Equity with financial leverage factors",
        "factor_count": 23,
        "analysis_type": "traditional"
    },
    "value_added_ratio": {
        "description": "Value-added ratio with differentiation factors",
        "factor_count": 23,
        "analysis_type": "traditional"
    },
    # New 3 items for lifecycle analysis
    "company_survival_probability": {
        "description": "Probability of company survival over time periods",
        "factor_count": 23,
        "analysis_type": "survival"
    },
    "startup_success_rate": {
        "description": "Success rate for new/spinoff companies",
        "factor_count": 23,
        "analysis_type": "emergence"
    },
    "business_succession_success": {
        "description": "Success rate of business succession/integration",
        "factor_count": 23,
        "analysis_type": "causal"
    }
}

# Extended factor items (23 total: 20 original + 3 new for lifecycle analysis)
EXTENDED_FACTOR_ITEMS = {
    "company_age": {
        "description": "Years since company establishment",
        "category": "lifecycle",
        "data_source": "company_registration_data"
    },
    "market_entry_timing": {
        "description": "First-mover vs late-mover advantage indicator",
        "category": "strategic_timing",
        "data_source": "market_analysis"
    },
    "parent_company_dependency": {
        "description": "Dependency ratio on parent company (for spinoffs)",
        "category": "corporate_structure",
        "data_source": "financial_statements"
    }
}

# Model configuration parameters
MODEL_CONFIG = {
    "data_period": {
        "total_years": 40,
        "start_year": 1984,
        "end_year": 2024,
        "variable_periods": True,  # Allow different companies to have different data periods
        "survivorship_bias_correction": True
    },
    "company_lifecycle": {
        "surviving_companies": "continuous_data_1984_2024",
        "extinct_companies": "data_until_extinction_year",
        "new_companies": "data_from_establishment_year",
        "include_extinction_events": True,
        "include_spinoff_events": True
    },
    "statistical_methods": {
        "survival_analysis": ["cox_regression", "kaplan_meier", "parametric"],
        "causal_inference": ["did", "iv", "propensity_score", "causal_forest"],
        "missing_data": ["multiple_imputation", "inverse_probability_weighting"],
        "bias_correction": ["survivorship_bias", "selection_bias", "temporal_bias"]
    }
}

try:
    # Import base model class
    from .base_model import BaseA2AIModel, ModelValidationMixin, LifecycleAnalysisMixin
    
    # Import traditional models
    from .traditional_models.regression_models import (
        LinearRegressionA2AI, PolynomialRegressionA2AI, RidgeRegressionA2AI,
        LassoRegressionA2AI, ElasticNetA2AI, BayesianRegressionA2AI
    )
    from .traditional_models.ensemble_models import (
        RandomForestA2AI, XGBoostA2AI, LightGBMA2AI, CatBoostA2AI,
        VotingEnsembleA2AI, StackingEnsembleA2AI
    )
    from .traditional_models.deep_learning_models import (
        DNNA2AI, CNNA2AI, LSTMA2AI, TransformerA2AI, AutoencoderA2AI
    )
    from .traditional_models.time_series_models import (
        ARIMAA2AI, SARIMAXA2AI, VectorAutoRegressionA2AI, 
        StateSpaceModelA2AI, LSTMTimeSeriesA2AI
    )
    
    # Import survival analysis models
    from .survival_models.cox_regression import CoxRegressionA2AI, StratifiedCoxA2AI
    from .survival_models.kaplan_meier import KaplanMeierA2AI, LogRankTestA2AI
    from .survival_models.parametric_survival import (
        WeibullSurvivalA2AI, ExponentialSurvivalA2AI, 
        LogNormalSurvivalA2AI, GammaSurvivalA2AI
    )
    from .survival_models.machine_learning_survival import (
        RandomSurvivalForestA2AI, DeepSurvivalA2AI, 
        GradientBoostingSurvivalA2AI
    )
    
    # Import emergence analysis models
    from .emergence_models.success_prediction import (
        StartupSuccessPredictorA2AI, SpinoffSuccessAnalyzerA2AI
    )
    from .emergence_models.growth_trajectory import (
        GrowthTrajectoryAnalyzerA2AI, ScalingPatternA2AI
    )
    from .emergence_models.market_entry_timing import (
        EntryTimingAnalyzerA2AI, MarketMaturityA2AI
    )
    
    # Import causal inference models
    from .causal_inference.difference_in_differences import (
        DIDA2AI, PanelDIDA2AI, StaggeredDIDA2AI
    )
    from .causal_inference.instrumental_variables import (
        TwoSLSA2AI, LIMLREA2AI, WeakInstrumentTestA2AI
    )
    from .causal_inference.propensity_score import (
        PropensityScoreMatchingA2AI, IPWEstimatorA2AI, 
        DoublyRobustA2AI
    )
    from .causal_inference.causal_forest import (
        CausalForestA2AI, GeneralizedRandomForestA2AI
    )
    
    # Import integrated models
    from .integrated_models.multi_stage_analysis import MultiStageAnalyzerA2AI
    from .integrated_models.lifecycle_trajectory import LifecycleTrajectoryA2AI
    from .integrated_models.market_ecosystem import MarketEcosystemA2AI
    
    MODELS_IMPORTED = True
    logger.info("All A2AI model modules successfully imported")
    
except ImportError as e:
    MODELS_IMPORTED = False
    logger.warning(f"Some model modules could not be imported: {e}")
    logger.info("Basic model structure will still be available")

# Model registry for dynamic model selection
MODEL_REGISTRY = {}

def register_model(model_class, category: str, model_name: str):
    """Register a model in the A2AI model registry"""
    if category not in MODEL_REGISTRY:
        MODEL_REGISTRY[category] = {}
    MODEL_REGISTRY[category][model_name] = model_class
    logger.debug(f"Registered model {model_name} in category {category}")

def get_model(category: str, model_name: str, **kwargs):
    """Get a model instance from the registry"""
    try:
        if category in MODEL_REGISTRY and model_name in MODEL_REGISTRY[category]:
            model_class = MODEL_REGISTRY[category][model_name]
            return model_class(**kwargs)
        else:
            raise ValueError(f"Model {model_name} not found in category {category}")
    except Exception as e:
        logger.error(f"Failed to create model {model_name}: {e}")
        raise

def list_available_models() -> Dict[str, List[str]]:
    """List all available models by category"""
    return {category: list(models.keys()) for category, models in MODEL_REGISTRY.items()}

def get_recommended_models(analysis_type: str, market_category: str) -> List[str]:
    """Get recommended models for specific analysis type and market category"""
    recommendations = {
        ("traditional", "high_share"): ["RandomForestA2AI", "XGBoostA2AI", "LSTMTimeSeriesA2AI"],
        ("traditional", "declining"): ["CausalForestA2AI", "DIDA2AI", "XGBoostA2AI"],
        ("traditional", "lost"): ["CoxRegressionA2AI", "RandomSurvivalForestA2AI", "KaplanMeierA2AI"],
        ("survival", "any"): ["CoxRegressionA2AI", "RandomSurvivalForestA2AI", "WeibullSurvivalA2AI"],
        ("emergence", "any"): ["StartupSuccessPredictorA2AI", "GrowthTrajectoryAnalyzerA2AI"],
        ("causal", "any"): ["CausalForestA2AI", "DIDA2AI", "PropensityScoreMatchingA2AI"],
        ("integrated", "any"): ["MultiStageAnalyzerA2AI", "LifecycleTrajectoryA2AI", "MarketEcosystemA2AI"]
    }
    
    key = (analysis_type, market_category)
    if key in recommendations:
        return recommendations[key]
    elif (analysis_type, "any") in recommendations:
        return recommendations[(analysis_type, "any")]
    else:
        return ["BaseA2AIModel"]

# Model performance tracking
class ModelPerformanceTracker:
    """Track and compare model performance across different scenarios"""
    
    def __init__(self):
        self.performance_history = {}
        self.model_comparisons = {}
    
    def log_performance(self, model_name: str, metrics: Dict[str, float], 
                        dataset: str, analysis_type: str):
        """Log model performance metrics"""
        key = f"{model_name}_{dataset}_{analysis_type}"
        if key not in self.performance_history:
            self.performance_history[key] = []
        self.performance_history[key].append(metrics)
    
    def get_best_model(self, metric: str, dataset: str, analysis_type: str) -> str:
        """Get the best performing model for a given metric"""
        best_score = None
        best_model = None
        
        for key, history in self.performance_history.items():
            if f"_{dataset}_{analysis_type}" in key:
                model_name = key.split(f"_{dataset}_{analysis_type}")[0]
                latest_metrics = history[-1] if history else {}
                if metric in latest_metrics:
                    score = latest_metrics[metric]
                    if best_score is None or score > best_score:
                        best_score = score
                        best_model = model_name
        
        return best_model

# Global performance tracker instance
performance_tracker = ModelPerformanceTracker()

# Validation functions
def validate_model_config(config: Dict[str, Any]) -> bool:
    """Validate model configuration parameters"""
    required_keys = ["data_period", "company_lifecycle", "statistical_methods"]
    for key in required_keys:
        if key not in config:
            logger.error(f"Missing required configuration key: {key}")
            return False
    return True

def validate_evaluation_items(items: List[str]) -> bool:
    """Validate that evaluation items are supported"""
    supported_items = set(EVALUATION_ITEMS.keys())
    for item in items:
        if item not in supported_items:
            logger.error(f"Unsupported evaluation item: {item}")
            return False
    return True

def validate_market_category(category: str) -> bool:
    """Validate market category"""
    if category not in MARKET_CATEGORIES:
        logger.error(f"Unsupported market category: {category}")
        return False
    return True

# Initialize model registry if models were imported successfully
if MODELS_IMPORTED:
    try:
        # Register traditional models
        register_model(RandomForestA2AI, "traditional", "RandomForestA2AI")
        register_model(XGBoostA2AI, "traditional", "XGBoostA2AI")
        register_model(LSTMA2AI, "traditional", "LSTMA2AI")
        
        # Register survival models
        register_model(CoxRegressionA2AI, "survival", "CoxRegressionA2AI")
        register_model(RandomSurvivalForestA2AI, "survival", "RandomSurvivalForestA2AI")
        register_model(KaplanMeierA2AI, "survival", "KaplanMeierA2AI")
        
        # Register emergence models
        register_model(StartupSuccessPredictorA2AI, "emergence", "StartupSuccessPredictorA2AI")
        register_model(GrowthTrajectoryAnalyzerA2AI, "emergence", "GrowthTrajectoryAnalyzerA2AI")
        
        # Register causal inference models
        register_model(CausalForestA2AI, "causal", "CausalForestA2AI")
        register_model(DIDA2AI, "causal", "DIDA2AI")
        register_model(PropensityScoreMatchingA2AI, "causal", "PropensityScoreMatchingA2AI")
        
        # Register integrated models
        register_model(MultiStageAnalyzerA2AI, "integrated", "MultiStageAnalyzerA2AI")
        register_model(LifecycleTrajectoryA2AI, "integrated", "LifecycleTrajectoryA2AI")
        register_model(MarketEcosystemA2AI, "integrated", "MarketEcosystemA2AI")
        
        logger.info(f"Successfully registered {len(MODEL_REGISTRY)} model categories")
        
    except Exception as e:
        logger.warning(f"Some models could not be registered: {e}")

# Export main components
__all__ = [
    # Core constants
    "MODEL_CATEGORIES", "MARKET_CATEGORIES", "EVALUATION_ITEMS", 
    "EXTENDED_FACTOR_ITEMS", "MODEL_CONFIG",
    
    # Model management functions
    "register_model", "get_model", "list_available_models", 
    "get_recommended_models",
    
    # Validation functions
    "validate_model_config", "validate_evaluation_items", "validate_market_category",
    
    # Performance tracking
    "ModelPerformanceTracker", "performance_tracker",
    
    # Base classes (if imported successfully)
    "BaseA2AIModel", "ModelValidationMixin", "LifecycleAnalysisMixin"
] + ([] if not MODELS_IMPORTED else [
    # Traditional models
    "RandomForestA2AI", "XGBoostA2AI", "LSTMA2AI", "TransformerA2AI",
    
    # Survival models
    "CoxRegressionA2AI", "RandomSurvivalForestA2AI", "KaplanMeierA2AI", 
    "WeibullSurvivalA2AI",
    
    # Emergence models
    "StartupSuccessPredictorA2AI", "GrowthTrajectoryAnalyzerA2AI", 
    "EntryTimingAnalyzerA2AI",
    
    # Causal inference models
    "CausalForestA2AI", "DIDA2AI", "PropensityScoreMatchingA2AI", "TwoSLSA2AI",
    
    # Integrated models
    "MultiStageAnalyzerA2AI", "LifecycleTrajectoryA2AI", "MarketEcosystemA2AI"
])

# Module initialization message
logger.info("A2AI Models module initialized successfully")
logger.info(f"Supporting analysis of {sum(cat['total_companies'] for cat in MARKET_CATEGORIES.values())} companies")
logger.info(f"Across {len(EVALUATION_ITEMS)} evaluation items with {len(EXTENDED_FACTOR_ITEMS) + 20} factor items each")
logger.info(f"Model import status: {'SUCCESS' if MODELS_IMPORTED else 'PARTIAL'}")