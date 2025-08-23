"""
A2AI (Advanced Financial Analysis AI) - Data Collection Module

企業ライフサイクル全体（生存・消滅・新設）を対象とした財務諸表分析AI
150社×40年分の財務データ収集機能を提供

市場カテゴリ:
- 高シェア市場: ロボット、内視鏡、工作機械、電子材料、精密測定機器 (各10社)
- シェア低下市場: 自動車、鉄鋼、スマート家電、バッテリー、PC・周辺機器 (各10社)  
- 完全失失市場: 家電、半導体、スマートフォン、PC、通信機器 (各10社)

特徴:
- 企業消滅データの積極活用（倒産・事業撤退の分析価値）
- 新設企業データの体系的収集（分社・スピンオフ企業追跡）
- 生存バイアス完全対応
- EDINET API活用による自動収集
- 時系列データ整合性保証
"""

from typing import Dict, List, Optional, Union, Tuple, Any
from pathlib import Path
import logging
from datetime import datetime, date
from dataclasses import dataclass
from enum import Enum
import asyncio

# Core data collection modules
from .financial_scraper import FinancialScraper
from .market_share_collector import MarketShareCollector
from .industry_data_collector import IndustryDataCollector

# Lifecycle-specific data collection modules
from .lifecycle_data_collector import LifecycleDataCollector
from .extinction_event_tracker import ExtinctionEventTracker
from .spinoff_data_integrator import SpinoffDataIntegrator
from .emergence_data_tracker import EmergenceDataTracker
from .survival_data_generator import SurvivalDataGenerator

# Data validation and utility modules
from .data_validator import DataValidator

# Configure module logger
logger = logging.getLogger(__name__)

# Version information
__version__ = "1.0.0"
__author__ = "A2AI Development Team"
__email__ = "support@a2ai.dev"

# Market category definitions based on attached enterprise list
class MarketCategory(Enum):
    """市場カテゴリ定義"""
    HIGH_SHARE = "high_share"      # 現在もシェアが高い市場
    DECLINING = "declining"         # 現在進行形でシェア低下中の市場  
    LOST_SHARE = "lost_share"      # 完全にシェアを失った市場

class CorporateStatus(Enum):
    """企業ステータス定義"""
    ACTIVE = "active"              # 継続企業
    EXTINCT = "extinct"            # 消滅企業（倒産・吸収合併）
    SPINOFF = "spinoff"            # 分社・独立企業
    ACQUIRED = "acquired"          # 買収企業
    RESTRUCTURED = "restructured"  # 事業再編企業

class DataSource(Enum):
    """データソース定義"""
    EDINET = "edinet"             # EDINET API
    COMPANY_IR = "company_ir"     # 企業IR資料
    INDUSTRY_STATS = "industry_stats"  # 業界統計
    GOVERNMENT_DATA = "government_data"  # 政府統計
    RESEARCH_REPORTS = "research_reports"  # 調査機関レポート

@dataclass
class CompanyInfo:
    """企業情報データクラス"""
    company_code: str
    company_name: str
    market_category: MarketCategory
    industry_sector: str
    corporate_status: CorporateStatus
    establishment_date: Optional[date]
    extinction_date: Optional[date]
    parent_company: Optional[str]
    spinoff_source: Optional[str]
    edinet_code: Optional[str]
    stock_code: Optional[str]
    
class DataCollectionConfig:
    """データ収集設定クラス"""
    
    # Target companies (150 companies across 3 market categories)
    TARGET_COMPANIES = {
        MarketCategory.HIGH_SHARE: {
            "ロボット": ["ファナック", "安川電機", "川崎重工業", "不二越", "デンソーウェーブ", 
                        "三菱電機", "オムロン", "THK", "NSK", "IHI"],
            "内視鏡": ["オリンパス", "HOYA", "富士フイルム", "キヤノンメディカルシステムズ", 
                        "島津製作所", "コニカミノルタ", "ソニー", "トプコン", "エムスリー", "日立製作所"],
            "工作機械": ["DMG森精機", "ヤマザキマザック", "オークマ", "牧野フライス製作所", 
                        "ジェイテクト", "東芝機械", "アマダ", "ソディック", "三菱重工工作機械", "シギヤ精機製作所"],
            "電子材料": ["村田製作所", "TDK", "京セラ", "太陽誘電", "日本特殊陶業", 
                        "ローム", "プロテリアル", "住友電工", "日東電工", "日本碍子"],
            "精密測定機器": ["キーエンス", "島津製作所", "堀場製作所", "東京精密", "ミツトヨ", 
                            "オリンパス", "日本電産", "リオン", "アルバック", "ナブテスコ"]
        },
        MarketCategory.DECLINING: {
            "自動車": ["トヨタ自動車", "日産自動車", "ホンダ", "スズキ", "マツダ", 
                        "SUBARU", "いすゞ自動車", "三菱自動車", "ダイハツ工業", "日野自動車"],
            "鉄鋼": ["日本製鉄", "JFEホールディングス", "神戸製鋼所", "日新製鋼", "大同特殊鋼", 
                    "山陽特殊製鋼", "愛知製鋼", "中部鋼鈑", "淀川製鋼所", "日立金属"],
            "スマート家電": ["パナソニック", "シャープ", "ソニー", "東芝ライフスタイル", 
                            "日立グローバルライフソリューションズ", "アイリスオーヤマ", "三菱電機", 
                            "象印マホービン", "タイガー魔法瓶", "山善"],
            "バッテリー": ["パナソニックエナジー", "村田製作所", "GSユアサ", "東芝インフラシステムズ", 
                        "日立化成", "FDK", "NEC", "ENAX", "日本電産", "TDK"],
            "PC・周辺機器": ["NEC", "富士通クライアントコンピューティング", "東芝", "ソニー", "エレコム", 
                            "バッファロー", "ロジテック", "プリンストン", "サンワサプライ", "アイ・オー・データ機器"]
        },
        MarketCategory.LOST_SHARE: {
            "家電": ["ソニー", "パナソニック", "シャープ", "東芝ライフスタイル", "三菱電機", 
                    "日立グローバルライフソリューションズ", "三洋電機", "ビクター", "アイワ", "船井電機"],
            "半導体": ["東芝", "日立製作所", "三菱電機", "NEC", "富士通", 
                        "松下電器", "ソニー", "ルネサスエレクトロニクス", "シャープ", "ローム"],
            "スマートフォン": ["ソニー", "シャープ", "京セラ", "パナソニック", "富士通", 
                            "NEC", "日立製作所", "三菱電機", "東芝", "カシオ計算機"],
            "PC": ["ソニー", "NEC", "富士通", "東芝", "シャープ", 
                    "パナソニック", "日立製作所", "三菱電機", "カシオ計算機", "日本電気ホームエレクトロニクス"],
            "通信機器": ["NEC", "富士通", "日立製作所", "松下電器", "シャープ", 
                        "ソニー", "三菱電機", "京セラ", "カシオ計算機", "日本無線"]
        }
    }
    
    # Data collection periods (handling variable company lifespans)
    BASE_START_YEAR = 1984
    BASE_END_YEAR = 2024
    TOTAL_YEARS = BASE_END_YEAR - BASE_START_YEAR + 1  # 41 years
    
    # EDINET API configuration
    EDINET_API_CONFIG = {
        "base_url": "https://disclosure.edinet-fsa.go.jp/api/v1/",
        "timeout": 30,
        "retry_attempts": 3,
        "rate_limit_delay": 1.0  # seconds between requests
    }
    
    # Financial statement types to collect
    FINANCIAL_STATEMENT_TYPES = [
        "yuho",      # 有価証券報告書
        "rinji",     # 臨時報告書  
        "nenpou",    # 年次報告書
        "shihanki",  # 四半期報告書
    ]
    
    # Evaluation metrics (9 items for A2AI)
    EVALUATION_METRICS = [
        # Traditional metrics (6 items)
        "sales_amount",                    # 売上高
        "sales_growth_rate",              # 売上高成長率  
        "operating_profit_margin",        # 売上高営業利益率
        "net_profit_margin",              # 売上高当期純利益率
        "roe",                           # ROE
        "value_added_ratio",             # 売上高付加価値率
        # A2AI extended metrics (3 items)
        "survival_probability",           # 企業存続確率
        "emergence_success_rate",         # 新規事業成功率
        "succession_success_rate"         # 事業継承成功度
    ]
    
    # Factor variables (23 items each for A2AI enhanced analysis)
    FACTOR_VARIABLES_COUNT = 23  # Extended from 20 to 23 for lifecycle analysis

class DataCollectionOrchestrator:
    """データ収集統合管理クラス"""
    
    def __init__(self, config: Optional[DataCollectionConfig] = None):
        """
        初期化
        
        Args:
            config: データ収集設定（Noneの場合はデフォルト設定使用）
        """
        self.config = config or DataCollectionConfig()
        self.logger = logging.getLogger(f"{__name__}.DataCollectionOrchestrator")
        
        # Initialize data collectors
        self._initialize_collectors()
        
        # Data validation
        self.validator = DataValidator()
        
        # Collection status tracking
        self.collection_status = {}
        
    def _initialize_collectors(self):
        """データ収集器の初期化"""
        try:
            # Core financial data collectors
            self.financial_scraper = FinancialScraper(self.config.EDINET_API_CONFIG)
            self.market_share_collector = MarketShareCollector()
            self.industry_data_collector = IndustryDataCollector()
            
            # Lifecycle-specific collectors
            self.lifecycle_collector = LifecycleDataCollector()
            self.extinction_tracker = ExtinctionEventTracker()
            self.spinoff_integrator = SpinoffDataIntegrator()
            self.emergence_tracker = EmergenceDataTracker()
            self.survival_generator = SurvivalDataGenerator()
            
            self.logger.info("All data collectors initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize data collectors: {e}")
            raise
    
    async def collect_all_data(self, 
                                market_categories: Optional[List[MarketCategory]] = None,
                                companies: Optional[List[str]] = None,
                                start_year: Optional[int] = None,
                                end_year: Optional[int] = None) -> Dict[str, Any]:
        """
        全データ収集の統合実行
        
        Args:
            market_categories: 対象市場カテゴリ（Noneの場合は全市場）
            companies: 対象企業リスト（Noneの場合は全企業）
            start_year: 開始年（Noneの場合は1984年）
            end_year: 終了年（Noneの場合は2024年）
            
        Returns:
            収集結果の統合データ
        """
        self.logger.info("Starting comprehensive data collection for A2AI")
        
        # Set default parameters
        market_categories = market_categories or list(MarketCategory)
        start_year = start_year or self.config.BASE_START_YEAR
        end_year = end_year or self.config.BASE_END_YEAR
        
        collection_results = {
            "financial_data": {},
            "market_share_data": {},
            "industry_data": {},
            "lifecycle_data": {},
            "extinction_events": {},
            "spinoff_events": {},
            "emergence_events": {},
            "survival_data": {},
            "metadata": {
                "collection_timestamp": datetime.now(),
                "target_companies_count": 0,
                "successful_collections": 0,
                "failed_collections": 0,
                "data_period": f"{start_year}-{end_year}"
            }
        }
        
        try:
            # Phase 1: Financial statements collection
            self.logger.info("Phase 1: Collecting financial statements data")
            financial_results = await self._collect_financial_data(
                market_categories, companies, start_year, end_year
            )
            collection_results["financial_data"] = financial_results
            
            # Phase 2: Market share data collection
            self.logger.info("Phase 2: Collecting market share data")
            market_share_results = await self._collect_market_share_data(market_categories)
            collection_results["market_share_data"] = market_share_results
            
            # Phase 3: Industry benchmark data
            self.logger.info("Phase 3: Collecting industry benchmark data")
            industry_results = await self._collect_industry_data(market_categories)
            collection_results["industry_data"] = industry_results
            
            # Phase 4: Lifecycle events data
            self.logger.info("Phase 4: Collecting lifecycle events data")
            lifecycle_results = await self._collect_lifecycle_data(market_categories)
            collection_results.update(lifecycle_results)
            
            # Phase 5: Data validation and quality check
            self.logger.info("Phase 5: Validating collected data")
            validation_results = await self._validate_collected_data(collection_results)
            collection_results["validation_results"] = validation_results
            
            # Update metadata
            self._update_collection_metadata(collection_results)
            
            self.logger.info("Data collection completed successfully")
            return collection_results
            
        except Exception as e:
            self.logger.error(f"Data collection failed: {e}")
            collection_results["error"] = str(e)
            return collection_results
    
    async def _collect_financial_data(self, 
                                    market_categories: List[MarketCategory],
                                    companies: Optional[List[str]],
                                    start_year: int, 
                                    end_year: int) -> Dict[str, Any]:
        """財務諸表データ収集"""
        financial_data = {}
        
        for category in market_categories:
            category_data = {}
            target_companies = self._get_target_companies(category, companies)
            
            for company in target_companies:
                try:
                    # Get company lifecycle info
                    company_info = await self.lifecycle_collector.get_company_info(company)
                    
                    # Adjust collection period based on company lifecycle
                    adjusted_start = max(start_year, 
                                        company_info.establishment_date.year if company_info.establishment_date else start_year)
                    adjusted_end = min(end_year,
                                        company_info.extinction_date.year if company_info.extinction_date else end_year)
                    
                    # Collect financial statements
                    company_financial_data = await self.financial_scraper.collect_company_data(
                        company=company,
                        start_year=adjusted_start,
                        end_year=adjusted_end,
                        statement_types=self.config.FINANCIAL_STATEMENT_TYPES
                    )
                    
                    category_data[company] = {
                        "financial_data": company_financial_data,
                        "company_info": company_info,
                        "collection_period": f"{adjusted_start}-{adjusted_end}"
                    }
                    
                    self.logger.info(f"Collected financial data for {company} ({adjusted_start}-{adjusted_end})")
                    
                except Exception as e:
                    self.logger.warning(f"Failed to collect financial data for {company}: {e}")
                    category_data[company] = {"error": str(e)}
            
            financial_data[category.value] = category_data
        
        return financial_data
    
    async def _collect_market_share_data(self, market_categories: List[MarketCategory]) -> Dict[str, Any]:
        """市場シェアデータ収集"""
        return await self.market_share_collector.collect_market_data(market_categories)
    
    async def _collect_industry_data(self, market_categories: List[MarketCategory]) -> Dict[str, Any]:
        """業界データ収集"""
        return await self.industry_data_collector.collect_industry_benchmarks(market_categories)
    
    async def _collect_lifecycle_data(self, market_categories: List[MarketCategory]) -> Dict[str, Any]:
        """ライフサイクルイベントデータ収集"""
        lifecycle_results = {}
        
        # Extinction events (corporate failures, acquisitions, etc.)
        extinction_events = await self.extinction_tracker.track_extinction_events(market_categories)
        lifecycle_results["extinction_events"] = extinction_events
        
        # Spinoff and integration events
        spinoff_events = await self.spinoff_integrator.track_spinoff_events(market_categories)
        lifecycle_results["spinoff_events"] = spinoff_events
        
        # Emergence events (new company establishments)
        emergence_events = await self.emergence_tracker.track_emergence_events(market_categories)
        lifecycle_results["emergence_events"] = emergence_events
        
        # Generate survival analysis data
        survival_data = await self.survival_generator.generate_survival_data(
            extinction_events, emergence_events, market_categories
        )
        lifecycle_results["survival_data"] = survival_data
        
        return lifecycle_results
    
    async def _validate_collected_data(self, collection_results: Dict[str, Any]) -> Dict[str, Any]:
        """収集データの検証"""
        return await self.validator.validate_comprehensive_data(collection_results)
    
    def _get_target_companies(self, 
                            category: MarketCategory, 
                            companies: Optional[List[str]]) -> List[str]:
        """対象企業リストの取得"""
        if companies:
            return companies
        
        # Get all companies in the category
        category_companies = []
        for industry, company_list in self.config.TARGET_COMPANIES[category].items():
            category_companies.extend(company_list)
        
        return category_companies
    
    def _update_collection_metadata(self, collection_results: Dict[str, Any]):
        """収集メタデータの更新"""
        metadata = collection_results["metadata"]
        
        # Count successful and failed collections
        total_companies = 0
        successful = 0
        failed = 0
        
        for category_data in collection_results["financial_data"].values():
            for company, data in category_data.items():
                total_companies += 1
                if "error" in data:
                    failed += 1
                else:
                    successful += 1
        
        metadata.update({
            "target_companies_count": total_companies,
            "successful_collections": successful,
            "failed_collections": failed,
            "success_rate": successful / total_companies if total_companies > 0 else 0
        })

# Utility functions
def get_supported_market_categories() -> List[MarketCategory]:
    """サポートされている市場カテゴリの取得"""
    return list(MarketCategory)

def get_target_companies_by_category(category: MarketCategory) -> Dict[str, List[str]]:
    """市場カテゴリ別の対象企業取得"""
    config = DataCollectionConfig()
    return config.TARGET_COMPANIES.get(category, {})

def get_all_target_companies() -> Dict[MarketCategory, Dict[str, List[str]]]:
    """全対象企業の取得"""
    config = DataCollectionConfig()
    return config.TARGET_COMPANIES

def create_data_collector(config: Optional[DataCollectionConfig] = None) -> DataCollectionOrchestrator:
    """データ収集器の作成"""
    return DataCollectionOrchestrator(config)

# Module exports
__all__ = [
    # Enums
    "MarketCategory",
    "CorporateStatus", 
    "DataSource",
    
    # Data classes
    "CompanyInfo",
    "DataCollectionConfig",
    
    # Main orchestrator
    "DataCollectionOrchestrator",
    
    # Core collectors  
    "FinancialScraper",
    "MarketShareCollector",
    "IndustryDataCollector",
    
    # Lifecycle collectors
    "LifecycleDataCollector",
    "ExtinctionEventTracker",
    "SpinoffDataIntegrator", 
    "EmergenceDataTracker",
    "SurvivalDataGenerator",
    
    # Validation
    "DataValidator",
    
    # Utility functions
    "get_supported_market_categories",
    "get_target_companies_by_category",
    "get_all_target_companies",
    "create_data_collector",
    
    # Module info
    "__version__",
    "__author__",
    "__email__"
]

# Initialize module logger
logging.getLogger(__name__).addHandler(logging.NullHandler())