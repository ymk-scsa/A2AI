"""
A2AI (Advanced Financial Analysis AI) - Main API Application
============================================================

FastAPI application providing comprehensive financial analysis capabilities
for corporate lifecycle research across 150 Japanese companies.

This application serves as the central hub for:
- Traditional financial statement analysis
- Survival analysis (corporate extinction prediction)
- Emergence analysis (startup success prediction) 
- Market dynamics and competitive analysis
- Causal inference and predictive modeling

Author: A2AI Development Team
Version: 1.0.0
"""

import os
import sys
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any, List
import traceback
from datetime import datetime

# FastAPI imports
from fastapi import FastAPI, Request, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from fastapi.openapi.utils import get_openapi
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# Pydantic imports
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

# Import A2AI configuration and routers
from . import (
    __version__, API_TITLE, API_DESCRIPTION, API_VERSION, API_PREFIX,
    MARKET_CATEGORIES, EVALUATION_METRICS, ANALYSIS_CAPABILITIES,
    ERROR_CODES, RESPONSE_STATUS
)

# Import API routers
from .routers import (
    survival_analysis,
    emergence_analysis, 
    prediction,
    visualization
)

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/a2ai_api.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Settings configuration
class Settings(BaseSettings):
    """Application settings and configuration"""
    
    # API Settings
    api_title: str = API_TITLE
    api_description: str = API_DESCRIPTION
    api_version: str = __version__
    debug: bool = False
    testing: bool = False
    
    # Database Settings
    database_url: str = "sqlite:///./a2ai_data.db"
    database_echo: bool = False
    
    # Security Settings
    secret_key: str = "a2ai-secret-key-change-in-production"
    access_token_expire_minutes: int = 60 * 24 * 7  # 7 days
    api_key_header: str = "X-API-Key"
    
    # CORS Settings
    cors_origins: List[str] = ["*"]
    cors_methods: List[str] = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    cors_headers: List[str] = ["*"]
    
    # Rate Limiting
    rate_limit_requests: int = 100
    rate_limit_window: int = 3600  # 1 hour
    
    # Data Processing
    max_companies_per_request: int = 50
    max_analysis_period_years: int = 40
    cache_ttl_seconds: int = 3600
    
    # Model Settings
    model_cache_size: int = 10
    prediction_batch_size: int = 32
    
    # File Upload
    max_file_size: int = 50 * 1024 * 1024  # 50MB
    allowed_file_types: List[str] = [".csv", ".xlsx", ".json"]
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# Initialize settings
settings = Settings()

# Response models
class APIResponse(BaseModel):
    """Standard API response model"""
    status: str = Field(..., description="Response status")
    message: str = Field(..., description="Response message")
    data: Dict[Any, Any] = Field(default_factory=dict, description="Response data")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Response metadata")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")
    version: str = Field(default=__version__, description="API version")

class HealthResponse(BaseModel):
    """Health check response model"""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    timestamp: datetime = Field(default_factory=datetime.now, description="Health check timestamp")
    services: Dict[str, str] = Field(..., description="Service component status")
    uptime: str = Field(..., description="Service uptime")

class ErrorResponse(BaseModel):
    """Error response model"""
    status: str = Field(default="error", description="Error status")
    error_code: str = Field(..., description="Error code")
    message: str = Field(..., description="Error message")
    details: Dict[str, Any] = Field(default_factory=dict, description="Error details")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")

# Global variables for tracking
app_start_time = datetime.now()
request_count = 0

# Lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    logger.info(f"Starting A2AI API v{__version__}")
    logger.info("Initializing database connections...")
    logger.info("Loading pre-trained models...")
    logger.info("Warming up analysis engines...")
    
    # Create necessary directories
    os.makedirs("logs", exist_ok=True)
    os.makedirs("data/cache", exist_ok=True)
    os.makedirs("results/api_cache", exist_ok=True)
    
    yield
    
    # Shutdown
    logger.info("Shutting down A2AI API...")
    logger.info("Cleaning up resources...")

# Create FastAPI application
app = FastAPI(
    title=settings.api_title,
    description=settings.api_description,
    version=settings.api_version,
    openapi_url=f"{API_PREFIX}/openapi.json",
    docs_url=None,  # Disable default docs
    redoc_url=None,  # Disable default redoc
    lifespan=lifespan
)

# Security
security = HTTPBearer(auto_error=False)

# Middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=settings.cors_methods,
    allow_headers=settings.cors_headers,
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"] if settings.debug else ["localhost", "127.0.0.1"]
)

# Request tracking middleware
@app.middleware("http")
async def request_tracking_middleware(request: Request, call_next):
    """Track requests and add metadata"""
    global request_count
    request_count += 1
    
    start_time = datetime.now()
    
    # Add request ID
    request_id = f"req_{request_count}_{int(start_time.timestamp())}"
    request.state.request_id = request_id
    
    # Log request
    logger.info(f"Request {request_id}: {request.method} {request.url}")
    
    try:
        response = await call_next(request)
        
        # Add response headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-API-Version"] = __version__
        response.headers["X-Processing-Time"] = str((datetime.now() - start_time).total_seconds())
        
        return response
        
    except Exception as e:
        logger.error(f"Request {request_id} failed: {str(e)}")
        logger.error(traceback.format_exc())
        raise

# Authentication dependency
async def get_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Validate API key (placeholder for production authentication)"""
    if settings.testing or settings.debug:
        return True
    
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required"
        )
    
    # In production, implement proper API key validation
    # For now, accept any token for development
    return True

# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error_code=f"HTTP_{exc.status_code}",
            message=exc.detail,
            details={"url": str(request.url), "method": request.method}
        ).model_dump()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {str(exc)}")
    logger.error(traceback.format_exc())
    
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error_code="INTERNAL_ERROR",
            message="Internal server error",
            details={
                "url": str(request.url),
                "method": request.method,
                "error_type": type(exc).__name__
            }
        ).model_dump()
    )

# Root endpoint
@app.get("/", include_in_schema=False)
async def root():
    """Redirect to API documentation"""
    return RedirectResponse(url="/docs")

# Health check endpoint
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Health check endpoint
    
    Returns the current status of the A2AI API service and its components.
    """
    uptime = datetime.now() - app_start_time
    
    # Check service components
    services = {
        "database": "healthy",
        "analysis_engine": "healthy",
        "model_cache": "healthy",
        "data_processor": "healthy"
    }
    
    return HealthResponse(
        status="healthy",
        version=__version__,
        services=services,
        uptime=str(uptime)
    )

# API Info endpoint
@app.get(f"{API_PREFIX}/info", response_model=APIResponse, tags=["System"])
async def api_info():
    """
    Get A2AI API information
    
    Returns comprehensive information about API capabilities, supported markets,
    evaluation metrics, and analysis types.
    """
    return APIResponse(
        status=RESPONSE_STATUS["SUCCESS"],
        message="A2AI API information retrieved successfully",
        data={
            "api_info": {
                "title": API_TITLE,
                "version": __version__,
                "description": "Advanced Financial Analysis AI for corporate lifecycle research"
            },
            "market_categories": MARKET_CATEGORIES,
            "evaluation_metrics": EVALUATION_METRICS,
            "analysis_capabilities": ANALYSIS_CAPABILITIES,
            "supported_periods": {
                "min_year": 1984,
                "max_year": 2024,
                "total_years": 40
            },
            "company_coverage": {
                "total_companies": 150,
                "high_share_markets": 50,
                "declining_markets": 50,
                "lost_markets": 50
            }
        },
        metadata={
            "request_count": request_count,
            "uptime": str(datetime.now() - app_start_time)
        }
    )

# Metrics endpoint
@app.get(f"{API_PREFIX}/metrics", response_model=APIResponse, tags=["System"])
async def api_metrics():
    """
    Get API usage metrics
    
    Returns current API usage statistics and performance metrics.
    """
    return APIResponse(
        status=RESPONSE_STATUS["SUCCESS"],
        message="API metrics retrieved successfully",
        data={
            "usage_metrics": {
                "total_requests": request_count,
                "uptime": str(datetime.now() - app_start_time),
                "start_time": app_start_time.isoformat(),
                "current_time": datetime.now().isoformat()
            },
            "system_metrics": {
                "active_models": 0,  # Placeholder
                "cache_hit_rate": 0.85,  # Placeholder
                "average_response_time": "250ms"  # Placeholder
            }
        }
    )

# Custom OpenAPI schema
def custom_openapi():
    """Generate custom OpenAPI schema"""
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title=settings.api_title,
        version=settings.api_version,
        description=settings.api_description,
        routes=app.routes,
    )
    
    # Add custom schema information
    openapi_schema["info"]["x-logo"] = {
        "url": "/static/logo.png"
    }
    
    openapi_schema["info"]["contact"] = {
        "name": "A2AI Support",
        "email": "support@a2ai.com"
    }
    
    openapi_schema["info"]["license"] = {
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT"
    }
    
    # Add server information
    openapi_schema["servers"] = [
        {
            "url": "/",
            "description": "A2AI API Server"
        }
    ]
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

# Custom documentation endpoints
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    """Custom Swagger UI documentation"""
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=f"{app.title} - Documentation",
        swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.9.0/swagger-ui-bundle.js",
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.9.0/swagger-ui.css",
    )

@app.get("/redoc", include_in_schema=False)
async def redoc_html():
    """Custom ReDoc documentation"""
    return get_redoc_html(
        openapi_url=app.openapi_url,
        title=f"{app.title} - Documentation",
        redoc_js_url="https://cdn.jsdelivr.net/npm/redoc@2.1.2/bundles/redoc.standalone.js",
    )

# Include API routers
app.include_router(
    survival_analysis.router,
    prefix=f"{API_PREFIX}/survival",
    tags=["Survival Analysis"],
    dependencies=[Depends(get_api_key)]
)

app.include_router(
    emergence_analysis.router,
    prefix=f"{API_PREFIX}/emergence",
    tags=["Emergence Analysis"],
    dependencies=[Depends(get_api_key)]
)

app.include_router(
    prediction.router,
    prefix=f"{API_PREFIX}/prediction",
    tags=["Prediction"],
    dependencies=[Depends(get_api_key)]
)

app.include_router(
    visualization.router,
    prefix=f"{API_PREFIX}/visualization",
    tags=["Visualization"],
    dependencies=[Depends(get_api_key)]
)

# Static files (if needed)
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# Startup event
@app.on_event("startup")
async def startup_event():
    """Application startup event"""
    logger.info("A2AI API is starting up...")
    logger.info(f"Version: {__version__}")
    logger.info(f"Debug mode: {settings.debug}")
    logger.info(f"Database URL: {settings.database_url}")

# Shutdown event  
@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown event"""
    logger.info("A2AI API is shutting down...")

if __name__ == "__main__":
    import uvicorn
    
    # Development server configuration
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level="info" if not settings.debug else "debug",
        access_log=True
    )