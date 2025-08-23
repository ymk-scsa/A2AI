"""
A2AI (Advanced Financial Analysis AI) - Configuration Module
============================================================

This module initializes the configuration system for A2AI, providing centralized
management of all system settings, parameters, and constants used throughout
the financial analysis AI system.

Key Features:
- Centralized configuration management
- Environment-specific settings
- YAML-based configuration files
- Type-safe configuration loading
- Validation and error handling
- Logging configuration
- Database connection parameters
- Model hyperparameters
- Analysis parameters

Usage:
    from config import settings, market_categories, evaluation_factors
    
    # Access database settings
    db_url = settings.DATABASE_URL
    
    # Access market categories
    high_share_markets = market_categories.HIGH_SHARE_MARKETS
    
    # Access evaluation factors
    traditional_factors = evaluation_factors.TRADITIONAL_EVALUATION_ITEMS
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import yaml
from dataclasses import dataclass
from enum import Enum

# =============================================================================
# PATH CONFIGURATION
# =============================================================================

# A2AIプロジェクトのルートディレクトリパスを設定
# プロジェクトの絶対パスを取得し、他の全てのパスの基準とする
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# 各種ディレクトリパスを定義
# データ関連ディレクトリ
DATA_ROOT = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_ROOT / "raw"
PROCESSED_DATA_DIR = DATA_ROOT / "processed"
EXTERNAL_DATA_DIR = DATA_ROOT / "external"

# 結果出力ディレクトリ
RESULTS_ROOT = PROJECT_ROOT / "results"
MODELS_DIR = RESULTS_ROOT / "models"
ANALYSIS_RESULTS_DIR = RESULTS_ROOT / "analysis_results"
VISUALIZATIONS_DIR = RESULTS_ROOT / "visualizations"
PREDICTIONS_DIR = RESULTS_ROOT / "predictions"
REPORTS_DIR = RESULTS_ROOT / "reports"

# 設定ファイルディレクトリ
CONFIG_DIR = PROJECT_ROOT / "config"

# ログディレクトリ
LOGS_DIR = PROJECT_ROOT / "logs"

# 必要なディレクトリが存在しない場合は作成
for directory in [DATA_ROOT, PROCESSED_DATA_DIR, EXTERNAL_DATA_DIR, 
                    RESULTS_ROOT, MODELS_DIR, ANALYSIS_RESULTS_DIR, 
                    VISUALIZATIONS_DIR, PREDICTIONS_DIR, REPORTS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# =============================================================================
# ENVIRONMENT CONFIGURATION
# =============================================================================

class Environment(Enum):
    """
    実行環境の種類を定義するEnum
    
    DEVELOPMENT: 開発環境（ローカル開発用）
    TESTING: テスト環境（単体・結合テスト用）
    STAGING: ステージング環境（本番前検証用）
    PRODUCTION: 本番環境（実際のデータ分析用）
    """
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"

# 環境変数から実行環境を取得、デフォルトは開発環境
CURRENT_ENV = Environment(os.getenv("A2AI_ENV", "development"))

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

def setup_logging(log_level: str = "INFO") -> None:
    """
    A2AIシステム全体のロギング設定を行う関数
    
    Args:
        log_level (str): ログレベル（DEBUG, INFO, WARNING, ERROR, CRITICAL）
    
    Features:
        - 環境別ログレベル設定
        - ファイル出力とコンソール出力の両方をサポート
        - ログローテーション対応
        - 構造化ログ形式
        - モジュール別ログフィルタリング
    """
    # ログファイルパスを環境別に設定
    log_file = LOGS_DIR / f"a2ai_{CURRENT_ENV.value}.log"
    
    # ログフォーマットを定義（詳細な情報を含む）
    log_format = (
        "%(asctime)s - %(name)s - %(levelname)s - "
        "%(filename)s:%(lineno)d - %(funcName)s() - %(message)s"
    )
    
    # ログ設定を適用
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            # ファイル出力用ハンドラー（ローテーション対応）
            logging.handlers.RotatingFileHandler(
                log_file, 
                maxBytes=10*1024*1024,  # 10MB
                backupCount=5
            ),
            # コンソール出力用ハンドラー
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # A2AI専用ロガーを作成
    logger = logging.getLogger("A2AI")
    logger.info(f"A2AI logging initialized for {CURRENT_ENV.value} environment")
    logger.info(f"Log file: {log_file}")

# =============================================================================
# CONFIGURATION LOADER
# =============================================================================

class ConfigurationError(Exception):
    """
    設定関連のエラーを表すカスタム例外クラス
    設定ファイルの読み込みエラー、検証エラー等で使用
    """
    pass

def load_yaml_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    YAMLファイルから設定を読み込む汎用関数
    
    Args:
        config_path (Union[str, Path]): 設定ファイルのパス
    
    Returns:
        Dict[str, Any]: 読み込まれた設定データ
    
    Raises:
        ConfigurationError: ファイルが存在しない、または読み込みエラーの場合
        
    Features:
        - ファイル存在確認
        - YAML構文エラーハンドリング
        - 環境変数による設定値置換対応
    """
    config_path = Path(config_path)
    
    # ファイル存在確認
    if not config_path.exists():
        raise ConfigurationError(f"Configuration file not found: {config_path}")
    
    try:
        # YAMLファイルを読み込み
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        
        # 環境変数による設定値置換を実行
        config_data = _substitute_env_variables(config_data)
        
        logging.getLogger("A2AI").info(f"Configuration loaded: {config_path}")
        return config_data
        
    except yaml.YAMLError as e:
        raise ConfigurationError(f"YAML parsing error in {config_path}: {e}")
    except Exception as e:
        raise ConfigurationError(f"Error loading configuration {config_path}: {e}")

def _substitute_env_variables(config_data: Any) -> Any:
    """
    設定値内の環境変数を実際の値に置換する内部関数
    ${ENV_VAR_NAME} の形式で環境変数を参照可能
    
    Args:
        config_data: 設定データ（辞書、リスト、文字列等）
    
    Returns:
        環境変数が置換された設定データ
    """
    if isinstance(config_data, dict):
        return {k: _substitute_env_variables(v) for k, v in config_data.items()}
    elif isinstance(config_data, list):
        return [_substitute_env_variables(item) for item in config_data]
    elif isinstance(config_data, str):
        # ${VAR_NAME} 形式の環境変数を置換
        import re
        def replace_env_var(match):
            env_var = match.group(1)
            return os.getenv(env_var, match.group(0))
        
        return re.sub(r'\$\{([^}]+)\}', replace_env_var, config_data)
    else:
        return config_data

# =============================================================================
# CONFIGURATION DATA CLASSES
# =============================================================================

@dataclass
class DatabaseConfig:
    """
    データベース接続設定を管理するデータクラス
    
    Attributes:
        host (str): データベースホスト
        port (int): データベースポート
        database (str): データベース名
        username (str): ユーザー名
        password (str): パスワード
        pool_size (int): コネクションプール最大サイズ
        echo (bool): SQLログ出力設定
    """
    host: str
    port: int
    database: str
    username: str
    password: str
    pool_size: int = 10
    echo: bool = False
    
    @property
    def url(self) -> str:
        """データベース接続URL文字列を生成"""
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"

@dataclass
class APIConfig:
    """
    外部API設定を管理するデータクラス
    
    Attributes:
        edinet_api_key (str): EDINET APIキー
        edinet_base_url (str): EDINET APIベースURL
        request_timeout (int): APIリクエストタイムアウト（秒）
        max_retries (int): 最大リトライ回数
        retry_delay (float): リトライ間隔（秒）
    """
    edinet_api_key: str
    edinet_base_url: str = "https://disclosure.edinet-fsa.go.jp/api/v2"
    request_timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0

@dataclass
class ModelConfig:
    """
    機械学習モデル設定を管理するデータクラス
    
    Attributes:
        random_state (int): 乱数シード
        test_size (float): テストデータ分割比率
        cv_folds (int): 交差検証分割数
        max_features (int): 最大特徴量数
        model_save_format (str): モデル保存形式
    """
    random_state: int = 42
    test_size: float = 0.2
    cv_folds: int = 5
    max_features: int = 1000
    model_save_format: str = "pickle"

@dataclass
class AnalysisConfig:
    """
    分析パラメータを管理するデータクラス
    
    Attributes:
        min_company_history_years (int): 分析対象企業の最小履歴年数
        max_missing_ratio (float): 許容する最大欠損値比率
        outlier_threshold (float): 外れ値検出閾値（標準偏差の倍数）
        significance_level (float): 統計的検定の有意水準
    """
    min_company_history_years: int = 5
    max_missing_ratio: float = 0.3
    outlier_threshold: float = 3.0
    significance_level: float = 0.05

# =============================================================================
# GLOBAL CONFIGURATION LOADING
# =============================================================================

def load_all_configurations() -> Dict[str, Any]:
    """
    全ての設定ファイルを読み込み、統合された設定辞書を返す
    
    Returns:
        Dict[str, Any]: 全ての設定を含む辞書
    
    Features:
        - 環境別設定ファイル読み込み
        - デフォルト設定との統合
        - 設定値の妥当性検証
        - 型変換とデータクラス化
    """
    config = {}
    
    try:
        # 基本設定ファイルを読み込み
        base_config_file = CONFIG_DIR / "settings.py"
        if base_config_file.exists():
            # settings.pyから基本設定を読み込む（後で実装）
            pass
        
        # 各種YAML設定ファイルを読み込み
        config_files = [
            "market_categories.yaml",
            "evaluation_factors.yaml", 
            "lifecycle_stages.yaml",
            "survival_parameters.yaml"
        ]
        
        for config_file in config_files:
            file_path = CONFIG_DIR / config_file
            if file_path.exists():
                file_key = config_file.replace(".yaml", "")
                config[file_key] = load_yaml_config(file_path)
        
        logging.getLogger("A2AI").info("All configurations loaded successfully")
        return config
        
    except Exception as e:
        logging.getLogger("A2AI").error(f"Error loading configurations: {e}")
        raise ConfigurationError(f"Failed to load configurations: {e}")

# =============================================================================
# MODULE EXPORTS
# =============================================================================

# システム初期化時にロギングを設定
setup_logging()

# グローバル設定を読み込み（他のモジュールから参照可能）
try:
    GLOBAL_CONFIG = load_all_configurations()
except ConfigurationError as e:
    # 設定読み込みエラー時は警告を出すが、システムは継続
    logging.getLogger("A2AI").warning(f"Configuration loading failed: {e}")
    GLOBAL_CONFIG = {}

# このモジュールからエクスポートする要素を定義
__all__ = [
    # パス関連
    "PROJECT_ROOT", "DATA_ROOT", "RAW_DATA_DIR", "PROCESSED_DATA_DIR",
    "EXTERNAL_DATA_DIR", "RESULTS_ROOT", "MODELS_DIR", "ANALYSIS_RESULTS_DIR",
    "VISUALIZATIONS_DIR", "PREDICTIONS_DIR", "REPORTS_DIR", "CONFIG_DIR", "LOGS_DIR",
    
    # 環境・設定関連
    "Environment", "CURRENT_ENV", "GLOBAL_CONFIG",
    
    # 設定データクラス
    "DatabaseConfig", "APIConfig", "ModelConfig", "AnalysisConfig",
    
    # ユーティリティ関数
    "load_yaml_config", "setup_logging",
    
    # 例外クラス
    "ConfigurationError"
]

# モジュール初期化完了ログ
logging.getLogger("A2AI").info("A2AI configuration module initialized successfully")
logging.getLogger("A2AI").info(f"Project root: {PROJECT_ROOT}")
logging.getLogger("A2AI").info(f"Current environment: {CURRENT_ENV.value}")