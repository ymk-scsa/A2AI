"""
A2AI (Advanced Financial Analysis AI) - Configuration Settings
財務諸表分析AI 設定ファイル

企業ライフサイクル全体分析（生存分析、新設企業分析、事業継承分析）を
支援するための包括的設定管理
"""

import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
import yaml

class A2AISettings:
    """A2AI システム全体設定クラス"""
    
    # ============================================================================
    # 基本設定
    # ============================================================================
    
    # プロジェクト基本情報
    PROJECT_NAME = "A2AI"
    PROJECT_FULL_NAME = "Advanced Financial Analysis AI"
    VERSION = "1.0.0"
    DESCRIPTION = "企業ライフサイクル全体を対象とした高度財務諸表分析AI"
    
    # ベースディレクトリ
    BASE_DIR = Path(__file__).resolve().parent.parent
    CONFIG_DIR = BASE_DIR / "config"
    DATA_DIR = BASE_DIR / "data"
    SRC_DIR = BASE_DIR / "src"
    RESULTS_DIR = BASE_DIR / "results"
    NOTEBOOKS_DIR = BASE_DIR / "notebooks"
    DOCS_DIR = BASE_DIR / "docs"
    
    # ログ設定
    LOG_DIR = BASE_DIR / "logs"
    LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE_MAX_SIZE = 10 * 1024 * 1024  # 10MB
    LOG_BACKUP_COUNT = 5
    
    # ============================================================================
    # データ収集設定
    # ============================================================================
    
    # EDINET API設定
    EDINET_API_BASE_URL = "https://disclosure.edinet-fsa.go.jp/api/v2"
    EDINET_API_KEY = os.getenv("EDINET_API_KEY", "")  # 環境変数から取得
    EDINET_REQUEST_TIMEOUT = 30  # seconds
    EDINET_RATE_LIMIT = 1000  # requests per hour
    EDINET_RETRY_ATTEMPTS = 3
    EDINET_RETRY_DELAY = 5  # seconds
    
    # データ収集期間設定
    ANALYSIS_START_YEAR = 1984
    ANALYSIS_END_YEAR = 2024
    ANALYSIS_PERIOD = ANALYSIS_END_YEAR - ANALYSIS_START_YEAR + 1  # 41年間
    
    # 対象企業設定
    TARGET_COMPANIES_COUNT = 150
    HIGH_SHARE_MARKETS = 5  # 世界シェア高市場数
    DECLINING_MARKETS = 5   # シェア低下市場数
    LOST_MARKETS = 5        # シェア失失市場数
    COMPANIES_PER_MARKET = 10
    
    # データ収集間隔（秒）
    API_REQUEST_INTERVAL = 1.0
    BATCH_PROCESS_INTERVAL = 3600  # 1時間
    
    # ============================================================================
    # データベース設定
    # ============================================================================
    
    # SQLite設定（開発・テスト用）
    SQLITE_DB_PATH = DATA_DIR / "a2ai_database.db"
    SQLITE_ECHO = False  # SQLクエリログ出力
    
    # PostgreSQL設定（本番用）
    POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
    POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
    POSTGRES_DB = os.getenv("POSTGRES_DB", "a2ai_db")
    POSTGRES_USER = os.getenv("POSTGRES_USER", "a2ai_user")
    POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "")
    
    # データベース接続プール設定
    DB_POOL_SIZE = 10
    DB_POOL_MAX_OVERFLOW = 20
    DB_POOL_TIMEOUT = 30
    DB_POOL_RECYCLE = 3600
    
    # ============================================================================
    # データ処理設定
    # ============================================================================
    
    # 前処理設定
    MISSING_VALUE_THRESHOLD = 0.3  # 30%以上欠損の場合は除外
    OUTLIER_DETECTION_METHOD = "IQR"  # IQR, Z_SCORE, ISOLATION_FOREST
    OUTLIER_THRESHOLD = 1.5  # IQR倍数
    NORMALIZATION_METHOD = "STANDARD"  # STANDARD, MINMAX, ROBUST
    
    # 特徴量エンジニアリング設定
    TIME_SERIES_LAG_PERIODS = [1, 2, 3, 5, 10]  # ラグ期間
    MOVING_AVERAGE_WINDOWS = [3, 5, 10, 20]     # 移動平均窓
    VOLATILITY_WINDOWS = [5, 10, 20]            # ボラティリティ計算窓
    
    # データ品質チェック設定
    DATA_QUALITY_MIN_YEARS = 5      # 最小必要年数
    DATA_QUALITY_COMPLETENESS = 0.7 # 70%以上のデータ完全性要求
    
    # ============================================================================
    # 評価項目・要因項目設定
    # ============================================================================
    
    # 9つの評価項目
    EVALUATION_METRICS = {
        # 従来の6項目
        "sales_revenue": {
            "name": "売上高",
            "description": "企業の基本的な事業規模指標",
            "unit": "億円",
            "factor_count": 23  # 拡張後
        },
        "sales_growth_rate": {
            "name": "売上高成長率",
            "description": "企業の成長性を示す指標",
            "unit": "%",
            "factor_count": 23
        },
        "operating_margin": {
            "name": "売上高営業利益率",
            "description": "本業での収益性指標",
            "unit": "%",
            "factor_count": 23
        },
        "net_margin": {
            "name": "売上高当期純利益率",
            "description": "最終的な収益性指標",
            "unit": "%",
            "factor_count": 23
        },
        "roe": {
            "name": "ROE（自己資本利益率）",
            "description": "株主資本効率性指標",
            "unit": "%",
            "factor_count": 23
        },
        "value_added_ratio": {
            "name": "売上高付加価値率",
            "description": "付加価値創造能力指標",
            "unit": "%",
            "factor_count": 23
        },
        # 新規の3項目
        "survival_probability": {
            "name": "企業存続確率",
            "description": "企業が将来も存続する確率",
            "unit": "%",
            "factor_count": 23
        },
        "emergence_success_rate": {
            "name": "新規事業成功率",
            "description": "新規事業・市場参入の成功確率",
            "unit": "%",
            "factor_count": 23
        },
        "succession_success_degree": {
            "name": "事業継承成功度",
            "description": "M&A・分社化の成功度合い",
            "unit": "スコア",
            "factor_count": 23
        }
    }
    
    # 拡張要因項目（各評価項目共通の3項目）
    EXTENDED_FACTORS = {
        "company_age": {
            "name": "企業年齢",
            "description": "設立からの経過年数",
            "calculation": "current_year - establishment_year"
        },
        "market_entry_timing": {
            "name": "市場参入時期",
            "description": "先発/後発効果指標",
            "calculation": "entry_year - market_start_year"
        },
        "parent_dependency": {
            "name": "親会社依存度",
            "description": "分社企業の親会社への依存度",
            "calculation": "parent_transactions / total_transactions"
        }
    }
    
    # ============================================================================
    # 機械学習・分析設定
    # ============================================================================
    
    # モデル学習設定
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    VALIDATION_SIZE = 0.2
    CV_FOLDS = 5
    
    # モデルハイパーパラメータ
    MODEL_HYPERPARAMETERS = {
        # 従来モデル
        "random_forest": {
            "n_estimators": [100, 200, 300],
            "max_depth": [10, 20, 30, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4]
        },
        "xgboost": {
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 6, 10],
            "learning_rate": [0.01, 0.1, 0.2],
            "subsample": [0.8, 0.9, 1.0]
        },
        "neural_network": {
            "hidden_layers": [(100,), (100, 50), (200, 100, 50)],
            "activation": ["relu", "tanh"],
            "learning_rate": [0.001, 0.01, 0.1],
            "batch_size": [32, 64, 128]
        },
        # 生存分析モデル
        "cox_regression": {
            "alpha": [0.0001, 0.001, 0.01, 0.1],
            "l1_ratio": [0.0, 0.5, 1.0],
            "normalize": [True, False]
        },
        "survival_forest": {
            "n_estimators": [100, 200, 300],
            "max_depth": [10, 20, None],
            "min_samples_split": [6, 10, 20],
            "min_samples_leaf": [3, 6, 10]
        }
    }
    
    # 生存分析設定
    SURVIVAL_ANALYSIS = {
        "time_intervals": list(range(1, 41)),  # 1-40年
        "censoring_threshold": 40,  # 40年で打ち切り
        "confidence_level": 0.95,
        "risk_groups": 4,  # 高リスク、中高リスク、中低リスク、低リスク
        "hazard_ratio_threshold": 1.5
    }
    
    # 因果推論設定
    CAUSAL_INFERENCE = {
        "methods": ["DID", "IV", "PSM", "CausalForest"],
        "propensity_score_caliper": 0.1,
        "bootstrap_samples": 1000,
        "confidence_level": 0.95,
        "treatment_effect_threshold": 0.05
    }
    
    # ============================================================================
    # 市場分類設定
    # ============================================================================
    
    # 市場カテゴリ定義
    MARKET_CATEGORIES = {
        "high_share": {
            "name": "世界シェア高市場",
            "description": "日本企業が世界シェア上位を維持している市場",
            "share_threshold": 30,  # 30%以上
            "trend": "stable_or_growing",
            "examples": ["半導体製造装置", "電子部品", "産業用ロボット", "計測機器", "自動車部品"]
        },
        "declining": {
            "name": "シェア低下市場",
            "description": "日本企業の世界シェアが低下傾向の市場",
            "share_threshold": 10,  # 10-30%
            "trend": "declining",
            "examples": ["自動車", "家電", "PC", "スマートフォン", "液晶パネル"]
        },
        "lost": {
            "name": "シェア失失市場",
            "description": "日本企業が世界シェアを完全に失った市場",
            "share_threshold": 5,   # 5%未満
            "trend": "lost",
            "examples": ["DRAM", "携帯電話", "太陽電池", "白物家電", "PC"]
        }
    }
    
    # ============================================================================
    # 可視化設定
    # ============================================================================
    
    # プロット設定
    PLOT_SETTINGS = {
        "figure_size": (12, 8),
        "dpi": 300,
        "style": "seaborn-v0_8",
        "color_palette": "Set2",
        "font_size": 12,
        "title_size": 14,
        "label_size": 10,
        "legend_size": 10,
        "grid": True,
        "spine_visibility": False
    }
    
    # ダッシュボード設定
    DASHBOARD_SETTINGS = {
        "theme": "plotly_white",
        "height": 600,
        "width": 1200,
        "auto_refresh": True,
        "refresh_interval": 300000,  # 5分（ミリ秒）
        "animation_duration": 1000
    }
    
    # 色設定（市場カテゴリ別）
    MARKET_COLORS = {
        "high_share": "#2E8B57",    # SeaGreen
        "declining": "#FF8C00",     # DarkOrange  
        "lost": "#DC143C",          # Crimson
        "neutral": "#708090"        # SlateGray
    }
    
    # ============================================================================
    # API・Web設定
    # ============================================================================
    
    # FastAPI設定
    API_SETTINGS = {
        "title": "A2AI API",
        "description": "Advanced Financial Analysis AI REST API",
        "version": "1.0.0",
        "host": "0.0.0.0",
        "port": 8000,
        "debug": False,
        "reload": False,
        "workers": 4,
        "timeout": 300,
        "max_request_size": 100 * 1024 * 1024  # 100MB
    }
    
    # CORS設定
    CORS_SETTINGS = {
        "allow_origins": ["http://localhost:3000", "http://localhost:8080"],
        "allow_credentials": True,
        "allow_methods": ["GET", "POST", "PUT", "DELETE"],
        "allow_headers": ["*"]
    }
    
    # ============================================================================
    # セキュリティ設定
    # ============================================================================
    
    # 認証設定
    SECURITY_SETTINGS = {
        "secret_key": os.getenv("SECRET_KEY", "a2ai_development_key_change_in_production"),
        "algorithm": "HS256",
        "access_token_expire_minutes": 30,
        "refresh_token_expire_days": 7
    }
    
    # レート制限設定
    RATE_LIMITING = {
        "requests_per_minute": 60,
        "requests_per_hour": 1000,
        "requests_per_day": 10000,
        "burst_size": 100
    }
    
    # ============================================================================
    # パフォーマンス設定
    # ============================================================================
    
    # 並列処理設定
    PARALLEL_PROCESSING = {
        "max_workers": min(32, (os.cpu_count() or 1) + 4),
        "chunk_size": 1000,
        "timeout": 3600,  # 1時間
        "memory_limit": "8GB"
    }
    
    # キャッシュ設定
    CACHE_SETTINGS = {
        "enabled": True,
        "backend": "redis",  # redis, memory, disk
        "redis_url": os.getenv("REDIS_URL", "redis://localhost:6379/0"),
        "default_timeout": 3600,  # 1時間
        "max_memory": "2GB"
    }
    
    # ============================================================================
    # 環境別設定
    # ============================================================================
    
    ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
    
    if ENVIRONMENT == "development":
        DEBUG = True
        LOG_LEVEL = "DEBUG"
        SQLITE_ECHO = True
        API_SETTINGS["debug"] = True
        API_SETTINGS["reload"] = True
    
    elif ENVIRONMENT == "testing":
        DEBUG = False
        LOG_LEVEL = "WARNING"
        SQLITE_DB_PATH = DATA_DIR / "test_a2ai_database.db"
        
    elif ENVIRONMENT == "production":
        DEBUG = False
        LOG_LEVEL = "ERROR"
        SQLITE_ECHO = False
        API_SETTINGS["debug"] = False
        API_SETTINGS["reload"] = False
    
    # ============================================================================
    # 外部サービス設定
    # ============================================================================
    
    # メール通知設定（分析完了通知等）
    EMAIL_SETTINGS = {
        "smtp_server": os.getenv("SMTP_SERVER", "smtp.gmail.com"),
        "smtp_port": int(os.getenv("SMTP_PORT", "587")),
        "username": os.getenv("EMAIL_USERNAME", ""),
        "password": os.getenv("EMAIL_PASSWORD", ""),
        "use_tls": True,
        "from_email": os.getenv("FROM_EMAIL", "noreply@a2ai.com")
    }
    
    # Slack通知設定
    SLACK_SETTINGS = {
        "webhook_url": os.getenv("SLACK_WEBHOOK_URL", ""),
        "channel": "#a2ai-notifications",
        "username": "A2AI Bot",
        "enabled": bool(os.getenv("SLACK_WEBHOOK_URL"))
    }
    
    # ============================================================================
    # ユーティリティメソッド
    # ============================================================================
    
    @classmethod
    def get_data_path(cls, category: str, filename: str = "") -> Path:
        """データファイルパスを取得"""
        path = cls.DATA_DIR / category
        if filename:
            path = path / filename
        return path
    
    @classmethod
    def get_model_path(cls, model_type: str, model_name: str = "") -> Path:
        """モデルファイルパスを取得"""
        path = cls.RESULTS_DIR / "models" / model_type
        if model_name:
            path = path / f"{model_name}.pkl"
        return path
    
    @classmethod
    def get_database_url(cls, engine: str = "sqlite") -> str:
        """データベースURLを取得"""
        if engine == "sqlite":
            return f"sqlite:///{cls.SQLITE_DB_PATH}"
        elif engine == "postgresql":
            return (f"postgresql://{cls.POSTGRES_USER}:{cls.POSTGRES_PASSWORD}"
                    f"@{cls.POSTGRES_HOST}:{cls.POSTGRES_PORT}/{cls.POSTGRES_DB}")
        else:
            raise ValueError(f"Unsupported database engine: {engine}")
    
    @classmethod
    def create_directories(cls):
        """必要なディレクトリを作成"""
        directories = [
            cls.DATA_DIR / "raw",
            cls.DATA_DIR / "processed",
            cls.DATA_DIR / "external",
            cls.RESULTS_DIR / "models",
            cls.RESULTS_DIR / "analysis_results", 
            cls.RESULTS_DIR / "visualizations",
            cls.RESULTS_DIR / "reports",
            cls.LOG_DIR,
            cls.NOTEBOOKS_DIR,
            cls.DOCS_DIR
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def load_yaml_config(cls, config_name: str) -> Dict:
        """YAML設定ファイルを読み込み"""
        config_path = cls.CONFIG_DIR / f"{config_name}.yaml"
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        return {}
    
    @classmethod
    def save_yaml_config(cls, config_name: str, config_data: Dict):
        """YAML設定ファイルを保存"""
        config_path = cls.CONFIG_DIR / f"{config_name}.yaml"
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_data, f, default_flow_style=False, ensure_ascii=False)
    
    @classmethod
    def get_analysis_period_range(cls) -> List[int]:
        """分析対象年度リストを取得"""
        return list(range(cls.ANALYSIS_START_YEAR, cls.ANALYSIS_END_YEAR + 1))
    
    @classmethod
    def is_target_company(cls, company_code: str) -> bool:
        """対象企業かどうかを判定"""
        # TODO: 実際の企業リストとの照合実装
        return True
    
    @classmethod
    def get_market_category(cls, company_code: str) -> str:
        """企業の市場カテゴリを取得"""
        # TODO: 実際の企業-市場マッピング実装
        return "high_share"  # デフォルト
    
    @classmethod
    def get_evaluation_metrics_list(cls) -> List[str]:
        """評価項目リストを取得"""
        return list(cls.EVALUATION_METRICS.keys())
    
    @classmethod
    def get_factor_count(cls, metric: str) -> int:
        """指定評価項目の要因項目数を取得"""
        return cls.EVALUATION_METRICS.get(metric, {}).get("factor_count", 0)
    
    @classmethod
    def validate_settings(cls) -> List[str]:
        """設定の妥当性をチェック"""
        errors = []
        
        # 必須環境変数チェック
        required_env_vars = ["EDINET_API_KEY"]
        for var in required_env_vars:
            if not os.getenv(var):
                errors.append(f"Environment variable {var} is required")
        
        # ディレクトリ存在チェック
        if not cls.BASE_DIR.exists():
            errors.append(f"Base directory does not exist: {cls.BASE_DIR}")
        
        # 分析期間チェック
        if cls.ANALYSIS_START_YEAR >= cls.ANALYSIS_END_YEAR:
            errors.append("ANALYSIS_START_YEAR must be less than ANALYSIS_END_YEAR")
        
        return errors

# ============================================================================
# モジュールレベル関数
# ============================================================================

def get_settings() -> A2AISettings:
    """設定インスタンスを取得"""
    return A2AISettings()

def initialize_a2ai():
    """A2AI初期化（ディレクトリ作成、設定検証等）"""
    settings = A2AISettings()
    
    # ディレクトリ作成
    settings.create_directories()
    
    # 設定検証
    errors = settings.validate_settings()
    if errors:
        print("Configuration errors found:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    print(f"A2AI {settings.VERSION} initialized successfully")
    print(f"Base directory: {settings.BASE_DIR}")
    print(f"Analysis period: {settings.ANALYSIS_START_YEAR}-{settings.ANALYSIS_END_YEAR}")
    print(f"Target companies: {settings.TARGET_COMPANIES_COUNT}")
    print(f"Evaluation metrics: {len(settings.EVALUATION_METRICS)}")
    
    return True

# ============================================================================
# 設定値エクスポート（後方互換性のため）
# ============================================================================

# よく使用される設定値をモジュールレベルでエクスポート
settings = A2AISettings()
BASE_DIR = settings.BASE_DIR
DATA_DIR = settings.DATA_DIR
RESULTS_DIR = settings.RESULTS_DIR
EVALUATION_METRICS = settings.EVALUATION_METRICS
MARKET_CATEGORIES = settings.MARKET_CATEGORIES
RANDOM_STATE = settings.RANDOM_STATE

if __name__ == "__main__":
    # 設定ファイル単体実行時の動作
    initialize_a2ai()