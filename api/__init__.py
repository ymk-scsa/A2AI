"""
A2AI (Advanced Financial Analysis AI) API Package
==================================================

This package provides RESTful API endpoints for the A2AI financial analysis system,
enabling comprehensive analysis of corporate financial statements with focus on:

1. Traditional Financial Analysis (6 core metrics)
2. Survival Analysis (enterprise extinction probability)
3. Emergence Analysis (startup success prediction)
4. Lifecycle Analysis (corporate lifecycle dynamics)
5. Market Comparison Analysis (market share dynamics)

The API supports analysis of 150 companies across 3 market categories:
- High Market Share Markets (50 companies)
- Declining Market Share Markets (50 companies)  
- Lost Market Share Markets (50 companies)

Key Features:
- Real-time financial analysis across 9 evaluation metrics
- 120+ factor analysis with 23 factors per evaluation metric
- Survival bias correction for accurate market analysis
- Causal inference for true factor impact assessment
- Comprehensive visualization and reporting capabilities

Author: A2AI Development Team
Version: 1.0.0
License: MIT
"""

__version__ = "1.0.0"
__author__ = "A2AI Development Team"
__email__ = "dev@a2ai.com"
__license__ = "MIT"

# API Configuration
API_TITLE = "A2AI - Advanced Financial Analysis AI"
API_DESCRIPTION = """
# A2AI - Advanced Financial Analysis AI

The most comprehensive financial statement analysis system for corporate lifecycle research.

## Overview

A2AI provides advanced financial analysis capabilities including traditional metrics analysis,
survival analysis, emergence analysis, and market dynamics research across 150 Japanese companies
over 40 years (1984-2024).

## Key Analysis Capabilities

### 📊 Traditional Financial Analysis
- **6 Core Evaluation Metrics**: Revenue, Growth Rate, Operating Margin, Net Margin, ROE, Value-Added Rate
- **120 Factor Analysis**: 20 factors per evaluation metric for comprehensive analysis
- **Time Series Analysis**: 40-year longitudinal analysis capability
- **Market Comparison**: Cross-market competitive analysis

### 🏥 Survival Analysis
- **Enterprise Extinction Probability**: Cox regression and Kaplan-Meier estimation
- **Survival Factor Analysis**: Identify factors affecting corporate longevity
- **Hazard Ratio Analysis**: Risk factor quantification
- **Survivorship Bias Correction**: Eliminate selection bias in market analysis

### 🚀 Emergence Analysis  
- **Startup Success Prediction**: New enterprise success probability modeling
- **Market Entry Analysis**: Optimal market entry timing and strategy
- **Growth Phase Analysis**: Corporate growth trajectory prediction
- **Innovation Impact Assessment**: Technology adoption impact analysis

### 🔄 Lifecycle Analysis
- **Stage Transition Analysis**: Corporate lifecycle stage identification
- **Maturity Indicators**: Enterprise maturity assessment metrics
- **Rejuvenation Analysis**: Corporate renewal and transformation analysis
- **Performance by Lifecycle**: Stage-specific performance benchmarking

### 🎯 Market Dynamics Analysis
- **Market Share Evolution**: Track market share changes over time
- **Competitive Positioning**: Strategic position analysis within markets
- **Ecosystem Analysis**: Market ecosystem and network effects
- **Future Scenario Modeling**: Predictive market scenario analysis

## Target Markets & Companies

### 🟢 High Market Share Markets (50 companies)
1. **Robotics**: FANUC, Yaskawa, Kawasaki Heavy Industries, etc.
2. **Endoscopy**: Olympus, HOYA, FUJIFILM, etc.
3. **Machine Tools**: DMG MORI, Yamazaki Mazak, Okuma, etc.
4. **Electronic Materials**: Murata, TDK, Kyocera, etc.
5. **Precision Instruments**: Keyence, Shimadzu, Horiba, etc.

### 🟡 Declining Market Share Markets (50 companies)
1. **Automotive**: Toyota, Nissan, Honda, etc.
2. **Steel**: Nippon Steel, JFE Holdings, Kobe Steel, etc.
3. **Smart Appliances**: Panasonic, Sharp, Sony, etc.
4. **Battery (EV)**: Panasonic Energy, Murata, GS Yuasa, etc.
5. **PC & Peripherals**: NEC, Fujitsu, Toshiba, etc.

### 🔴 Lost Market Share Markets (50 companies)
1. **Consumer Electronics**: Sony, Panasonic, Sharp, Sanyo (extinct), etc.
2. **Semiconductors**: Toshiba Memory→Kioxia, Renesas, etc.
3. **Smartphones**: Sony Xperia, Sharp AQUOS, Kyocera (withdrawn), etc.
4. **PC Market**: Sony VAIO, NEC, Fujitsu, Toshiba dynabook, etc.
5. **Telecommunications**: NEC, Fujitsu, Hitachi, etc.

## Analysis Period
- **Primary Period**: 1984-2024 (40 years)
- **Variable Period Support**: Accommodate different corporate lifespans
- **Event Analysis**: M&A, spinoffs, bankruptcies, new establishments

## Data Sources
- **EDINET API**: Primary financial statement data source
- **Market Share Data**: Industry reports and corporate IR materials
- **External Data**: Economic indicators and industry benchmarks
- **Event Data**: Corporate restructuring and lifecycle events
"""

API_VERSION = "v1"
API_PREFIX = f"/api/{API_VERSION}"

# Market Categories
MARKET_CATEGORIES = {
    "high_share": {
        "name": "High Market Share Markets",
        "description": "Markets where Japanese companies maintain dominant global market share",
        "markets": ["robotics", "endoscopy", "machine_tools", "electronic_materials", "precision_instruments"],
        "company_count": 50,
        "status": "dominant"
    },
    "declining_share": {
        "name": "Declining Market Share Markets", 
        "description": "Markets where Japanese companies are losing global market share",
        "markets": ["automotive", "steel", "smart_appliances", "battery_ev", "pc_peripherals"],
        "company_count": 50,
        "status": "declining"
    },
    "lost_share": {
        "name": "Lost Market Share Markets",
        "description": "Markets where Japanese companies have largely lost global market dominance",
        "markets": ["consumer_electronics", "semiconductors", "smartphones", "pc_market", "telecommunications"],
        "company_count": 50,
        "status": "lost"
    }
}

# Evaluation Metrics Configuration
EVALUATION_METRICS = {
    "traditional": {
        "revenue": {
            "name": "売上高 (Revenue)",
            "description": "Total company revenue and its growth patterns",
            "factors_count": 20
        },
        "revenue_growth": {
            "name": "売上高成長率 (Revenue Growth Rate)",
            "description": "Year-over-year revenue growth analysis",
            "factors_count": 20
        },
        "operating_margin": {
            "name": "売上高営業利益率 (Operating Margin)",
            "description": "Operating profit margin and efficiency metrics",
            "factors_count": 20
        },
        "net_margin": {
            "name": "売上高当期純利益率 (Net Profit Margin)",
            "description": "Net profit margin and comprehensive profitability",
            "factors_count": 20
        },
        "roe": {
            "name": "ROE (Return on Equity)",
            "description": "Return on equity and capital efficiency",
            "factors_count": 20
        },
        "value_added_rate": {
            "name": "売上高付加価値率 (Value-Added Rate)",
            "description": "Value creation and competitive advantage metrics",
            "factors_count": 20
        }
    },
    "advanced": {
        "survival_probability": {
            "name": "企業存続確率 (Survival Probability)",
            "description": "Corporate survival and extinction risk analysis",
            "factors_count": 23
        },
        "emergence_success": {
            "name": "新規事業成功率 (Emergence Success Rate)",
            "description": "New business and startup success prediction",
            "factors_count": 23
        },
        "succession_success": {
            "name": "事業継承成功度 (Succession Success)",
            "description": "Business succession and M&A success analysis",
            "factors_count": 23
        }
    }
}

# Analysis Capabilities
ANALYSIS_CAPABILITIES = {
    "traditional_analysis": {
        "factor_impact": "Factor impact analysis on evaluation metrics",
        "market_comparison": "Cross-market performance comparison",
        "correlation": "Correlation analysis between factors and metrics",
        "regression": "Multi-variate regression analysis",
        "clustering": "Company clustering by performance patterns",
        "trend": "Time series trend analysis"
    },
    "survival_analysis": {
        "extinction_risk": "Corporate extinction risk assessment",
        "survival_factors": "Key survival factor identification",
        "hazard_ratio": "Hazard ratio analysis for risk factors",
        "survival_clustering": "Survival pattern clustering",
        "kaplan_meier": "Kaplan-Meier survival curve estimation",
        "cox_regression": "Cox proportional hazards modeling"
    },
    "emergence_analysis": {
        "startup_success": "Startup success probability prediction",
        "market_entry": "Market entry strategy optimization",
        "growth_phase": "Corporate growth phase analysis",
        "innovation_impact": "Innovation adoption impact assessment"
    },
    "lifecycle_analysis": {
        "stage_transition": "Lifecycle stage transition analysis",
        "maturity_assessment": "Corporate maturity evaluation",
        "rejuvenation": "Corporate renewal and transformation analysis",
        "performance_by_stage": "Stage-specific performance benchmarking"
    },
    "causal_inference": {
        "difference_in_differences": "DID analysis for causal effect estimation",
        "instrumental_variables": "IV estimation for endogeneity correction",
        "propensity_score": "Propensity score matching analysis",
        "causal_forest": "Machine learning-based causal inference"
    }
}

# API Error Codes
ERROR_CODES = {
    "INVALID_COMPANY": {
        "code": "E001",
        "message": "Invalid company identifier"
    },
    "INVALID_MARKET": {
        "code": "E002", 
        "message": "Invalid market category"
    },
    "INVALID_METRIC": {
        "code": "E003",
        "message": "Invalid evaluation metric"
    },
    "INVALID_PERIOD": {
        "code": "E004",
        "message": "Invalid analysis period"
    },
    "INSUFFICIENT_DATA": {
        "code": "E005",
        "message": "Insufficient data for analysis"
    },
    "MODEL_NOT_FOUND": {
        "code": "E006",
        "message": "Analysis model not found"
    },
    "ANALYSIS_FAILED": {
        "code": "E007",
        "message": "Analysis execution failed"
    },
    "RATE_LIMIT_EXCEEDED": {
        "code": "E008",
        "message": "API rate limit exceeded"
    }
}

# Response Status
RESPONSE_STATUS = {
    "SUCCESS": "success",
    "ERROR": "error",
    "WARNING": "warning",
    "PROCESSING": "processing",
    "COMPLETED": "completed",
    "FAILED": "failed"
}

# Data Analysis Periods
ANALYSIS_PERIODS = {
    "full": {
        "name": "Full Period Analysis",
        "start_year": 1984,
        "end_year": 2024,
        "description": "Complete 40-year analysis period"
    },
    "recent": {
        "name": "Recent Period Analysis", 
        "start_year": 2014,
        "end_year": 2024,
        "description": "Recent 10-year analysis period"
    },
    "pre_covid": {
        "name": "Pre-COVID Analysis",
        "start_year": 1984,
        "end_year": 2019,
        "description": "Analysis period before COVID-19 impact"
    },
    "post_covid": {
        "name": "Post-COVID Analysis",
        "start_year": 2020,
        "end_year": 2024,
        "description": "Analysis period during and after COVID-19"
    },
    "custom": {
        "name": "Custom Period Analysis",
        "description": "User-defined analysis period"
    }
}

# Model Types
MODEL_TYPES = {
    "regression": {
        "linear": "Linear Regression",
        "ridge": "Ridge Regression", 
        "lasso": "Lasso Regression",
        "elastic_net": "Elastic Net Regression"
    },
    "ensemble": {
        "random_forest": "Random Forest",
        "gradient_boosting": "Gradient Boosting",
        "xgboost": "XGBoost",
        "lightgbm": "LightGBM"
    },
    "survival": {
        "cox": "Cox Proportional Hazards",
        "kaplan_meier": "Kaplan-Meier Estimator",
        "weibull": "Weibull Survival Model",
        "log_normal": "Log-Normal Survival Model"
    },
    "causal": {
        "did": "Difference-in-Differences",
        "iv": "Instrumental Variables",
        "psm": "Propensity Score Matching",
        "causal_forest": "Causal Forest"
    }
}

# Export all public components
__all__ = [
    "__version__",
    "__author__", 
    "__email__",
    "__license__",
    "API_TITLE",
    "API_DESCRIPTION", 
    "API_VERSION",
    "API_PREFIX",
    "MARKET_CATEGORIES",
    "EVALUATION_METRICS",
    "ANALYSIS_CAPABILITIES",
    "ERROR_CODES",
    "RESPONSE_STATUS",
    "ANALYSIS_PERIODS",
    "MODEL_TYPES"
]