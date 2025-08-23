"""
A2AI (Advanced Financial Analysis AI)
市場シェアデータ収集システム

このモジュールは150社×40年分の市場シェアデータを収集し、
高シェア/低下/失失市場の分類根拠データを蓄積します。
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
import json
import time
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from pathlib import Path
import yaml
import requests
from bs4 import BeautifulSoup
import re
from urllib.parse import urljoin, urlparse
import sqlite3

# 設定読み込み
from config.settings import DATABASE_PATH, API_CONFIG, DATA_COLLECTION_CONFIG
from utils.logging_utils import setup_logger
from utils.database_utils import DatabaseManager
from utils.data_utils import DataCleaner, DataValidator


@dataclass
class MarketShareData:
    """市場シェアデータ構造"""
    company_name: str
    company_code: str
    market_category: str  # high_share/declining/lost
    market_sector: str    # ロボット/内視鏡/工作機械等
    year: int
    market_share_global: Optional[float] = None
    market_share_domestic: Optional[float] = None
    market_share_asia: Optional[float] = None
    market_size_million_usd: Optional[float] = None
    ranking_global: Optional[int] = None
    ranking_domestic: Optional[int] = None
    data_source: str = ""
    reliability_score: float = 0.0
    collection_date: datetime = None
    notes: str = ""


@dataclass
class MarketSegment:
    """市場セグメント定義"""
    sector_name: str
    category: str  # high_share/declining/lost
    companies: List[str]
    market_definition: str
    key_metrics: List[str]
    data_sources: List[str]


class MarketShareCollector:
    """市場シェアデータ収集メインクラス"""
    
    def __init__(self):
        self.logger = setup_logger(__name__)
        self.db_manager = DatabaseManager(DATABASE_PATH)
        self.data_cleaner = DataCleaner()
        self.data_validator = DataValidator()
        
        # 市場定義と企業マッピング読み込み
        self.market_segments = self._load_market_definitions()
        self.company_mapping = self._load_company_mapping()
        
        # データソース設定
        self.data_sources = {
            'government_stats': GovernmentStatsCollector(),
            'industry_reports': IndustryReportCollector(), 
            'corporate_ir': CorporateIRCollector(),
            'research_institutions': ResearchInstitutionCollector(),
            'market_research': MarketResearchCollector()
        }
        
        # レート制限設定
        self.rate_limiters = {}
        
    def _load_market_definitions(self) -> Dict[str, MarketSegment]:
        """市場定義ファイル読み込み"""
        config_path = Path("config/market_categories.yaml")
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            
        segments = {}
        
        # 高シェア市場定義
        for sector_name, companies in config['high_share_markets'].items():
            segments[sector_name] = MarketSegment(
                sector_name=sector_name,
                category='high_share',
                companies=companies,
                market_definition=config['market_definitions'][sector_name],
                key_metrics=['global_share', 'domestic_share', 'ranking'],
                data_sources=['jeita', 'meti', 'company_ir']
            )
            
        # シェア低下市場定義
        for sector_name, companies in config['declining_markets'].items():
            segments[sector_name] = MarketSegment(
                sector_name=sector_name,
                category='declining',
                companies=companies,
                market_definition=config['market_definitions'][sector_name],
                key_metrics=['global_share', 'share_change', 'competitive_position'],
                data_sources=['idc', 'gartner', 'company_ir']
            )
            
        # 失失市場定義
        for sector_name, companies in config['lost_markets'].items():
            segments[sector_name] = MarketSegment(
                sector_name=sector_name,
                category='lost',
                companies=companies,
                market_definition=config['market_definitions'][sector_name],
                key_metrics=['historical_share', 'exit_timeline', 'market_exit_reason'],
                data_sources=['historical_reports', 'industry_analysis']
            )
            
        return segments
        
    def _load_company_mapping(self) -> Dict[str, Dict[str, Any]]:
        """企業マッピング情報読み込み"""
        mapping = {}
        
        # 企業リストから各社の基本情報を構築
        high_share_companies = [
            # ロボット市場
            {'name': 'ファナック', 'code': '6954', 'sector': 'ロボット', 'category': 'high_share'},
            {'name': '安川電機', 'code': '6506', 'sector': 'ロボット', 'category': 'high_share'},
            {'name': '川崎重工業', 'code': '7012', 'sector': 'ロボット', 'category': 'high_share'},
            {'name': '不二越', 'code': '6474', 'sector': 'ロボット', 'category': 'high_share'},
            {'name': 'デンソーウェーブ', 'code': 'DENSO_SUB', 'sector': 'ロボット', 'category': 'high_share'},
            {'name': '三菱電機', 'code': '6503', 'sector': 'ロボット', 'category': 'high_share'},
            {'name': 'オムロン', 'code': '6645', 'sector': 'ロボット', 'category': 'high_share'},
            {'name': 'THK', 'code': '6481', 'sector': 'ロボット', 'category': 'high_share'},
            {'name': 'NSK', 'code': '6471', 'sector': 'ロボット', 'category': 'high_share'},
            {'name': 'IHI', 'code': '7013', 'sector': 'ロボット', 'category': 'high_share'},
            
            # 内視鏡市場
            {'name': 'オリンパス', 'code': '7733', 'sector': '内視鏡', 'category': 'high_share'},
            {'name': 'HOYA', 'code': '7741', 'sector': '内視鏡', 'category': 'high_share'},
            {'name': '富士フイルム', 'code': '4901', 'sector': '内視鏡', 'category': 'high_share'},
            {'name': 'キヤノンメディカルシステムズ', 'code': 'CANON_MED', 'sector': '内視鏡', 'category': 'high_share'},
            {'name': '島津製作所', 'code': '7701', 'sector': '内視鏡', 'category': 'high_share'},
            {'name': 'コニカミノルタ', 'code': '4902', 'sector': '内視鏡', 'category': 'high_share'},
            {'name': 'ソニー', 'code': '6758', 'sector': '内視鏡', 'category': 'high_share'},
            {'name': 'トプコン', 'code': '7732', 'sector': '内視鏡', 'category': 'high_share'},
            {'name': 'エムスリー', 'code': '2413', 'sector': '内視鏡', 'category': 'high_share'},
            {'name': '日立製作所', 'code': '6501', 'sector': '内視鏡', 'category': 'high_share'},
        ]
        
        declining_companies = [
            # 自動車市場
            {'name': 'トヨタ自動車', 'code': '7203', 'sector': '自動車', 'category': 'declining'},
            {'name': '日産自動車', 'code': '7201', 'sector': '自動車', 'category': 'declining'},
            {'name': 'ホンダ', 'code': '7267', 'sector': '自動車', 'category': 'declining'},
            {'name': 'スズキ', 'code': '7269', 'sector': '自動車', 'category': 'declining'},
            {'name': 'マツダ', 'code': '7261', 'sector': '自動車', 'category': 'declining'},
            {'name': 'SUBARU', 'code': '7270', 'sector': '自動車', 'category': 'declining'},
        ]
        
        lost_companies = [
            # 家電市場
            {'name': 'ソニー', 'code': '6758', 'sector': '家電', 'category': 'lost'},
            {'name': 'パナソニック', 'code': '6752', 'sector': '家電', 'category': 'lost'},
            {'name': 'シャープ', 'code': '6753', 'sector': '家電', 'category': 'lost'},
            {'name': '三洋電機', 'code': 'EXTINCT', 'sector': '家電', 'category': 'lost'},  # 消滅企業
        ]
        
        # 全企業リストを統合してマッピング構築
        all_companies = high_share_companies + declining_companies + lost_companies
        
        for company in all_companies:
            mapping[company['name']] = company
            
        return mapping
        
    async def collect_all_market_share_data(self, 
                                            start_year: int = 1984, 
                                            end_year: int = 2024) -> List[MarketShareData]:
        """全市場シェアデータ収集メイン関数"""
        self.logger.info(f"市場シェアデータ収集開始: {start_year}-{end_year}")
        
        all_data = []
        
        # 市場セグメント別に並行収集
        tasks = []
        for sector_name, segment in self.market_segments.items():
            task = self._collect_segment_data(segment, start_year, end_year)
            tasks.append(task)
            
        # 並行実行（レート制限考慮）
        segment_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in segment_results:
            if isinstance(result, Exception):
                self.logger.error(f"セグメントデータ収集エラー: {result}")
            else:
                all_data.extend(result)
                
        # データ品質検証
        validated_data = self._validate_collected_data(all_data)
        
        # データベース保存
        await self._save_to_database(validated_data)
        
        self.logger.info(f"市場シェアデータ収集完了: {len(validated_data)}レコード")
        return validated_data
        
    async def _collect_segment_data(self, 
                                    segment: MarketSegment, 
                                    start_year: int, 
                                    end_year: int) -> List[MarketShareData]:
        """セグメント別データ収集"""
        segment_data = []
        
        self.logger.info(f"セグメント収集開始: {segment.sector_name} ({segment.category})")
        
        # 企業別に収集
        for company_name in segment.companies:
            company_info = self.company_mapping.get(company_name)
            if not company_info:
                self.logger.warning(f"企業情報未定義: {company_name}")
                continue
                
            # データソース別に収集
            company_data = await self._collect_company_data(
                company_info, segment, start_year, end_year
            )
            segment_data.extend(company_data)
            
            # レート制限遵守
            await asyncio.sleep(0.5)
            
        return segment_data
        
    async def _collect_company_data(self, 
                                    company_info: Dict[str, Any],
                                    segment: MarketSegment,
                                    start_year: int, 
                                    end_year: int) -> List[MarketShareData]:
        """企業別データ収集"""
        company_data = []
        
        # データソース優先順位に基づいて収集
        for source_name in segment.data_sources:
            if source_name in self.data_sources:
                try:
                    source_collector = self.data_sources[source_name]
                    data = await source_collector.collect_company_market_share(
                        company_info, segment, start_year, end_year
                    )
                    company_data.extend(data)
                    
                except Exception as e:
                    self.logger.error(f"データソース収集エラー {source_name}: {e}")
                    
        # データ統合・重複排除
        unified_data = self._unify_company_data(company_data, company_info, segment)
        
        return unified_data
        
    def _unify_company_data(self, 
                            raw_data: List[MarketShareData],
                            company_info: Dict[str, Any],
                            segment: MarketSegment) -> List[MarketShareData]:
        """企業データ統合・重複排除"""
        if not raw_data:
            return []
            
        # 年次別にグループ化
        year_groups = {}
        for data in raw_data:
            if data.year not in year_groups:
                year_groups[data.year] = []
            year_groups[data.year].append(data)
            
        unified_data = []
        
        for year, year_data in year_groups.items():
            # 複数ソースがある場合は信頼性スコア最高を採用
            best_data = max(year_data, key=lambda x: x.reliability_score)
            
            # 他のソースからの補完データをマージ
            for other_data in year_data:
                if other_data != best_data:
                    if best_data.market_share_global is None:
                        best_data.market_share_global = other_data.market_share_global
                    if best_data.market_share_domestic is None:
                        best_data.market_share_domestic = other_data.market_share_domestic
                    if best_data.ranking_global is None:
                        best_data.ranking_global = other_data.ranking_global
                        
            unified_data.append(best_data)
            
        return unified_data
        
    def _validate_collected_data(self, data: List[MarketShareData]) -> List[MarketShareData]:
        """データ品質検証"""
        validated_data = []
        
        for record in data:
            # 基本検証
            if not self._basic_validation(record):
                continue
                
            # 論理検証
            if not self._logical_validation(record):
                continue
                
            # 時系列整合性検証
            if not self._temporal_validation(record, data):
                continue
                
            validated_data.append(record)
            
        return validated_data
        
    def _basic_validation(self, record: MarketShareData) -> bool:
        """基本検証"""
        if not record.company_name or not record.market_sector:
            return False
            
        if record.year < 1980 or record.year > 2025:
            return False
            
        if record.market_share_global is not None:
            if record.market_share_global < 0 or record.market_share_global > 100:
                return False
                
        return True
        
    def _logical_validation(self, record: MarketShareData) -> bool:
        """論理検証"""
        # グローバルシェア > 国内シェアの場合は異常
        if (record.market_share_global is not None and 
            record.market_share_domestic is not None):
            if record.market_share_global > record.market_share_domestic * 3:
                self.logger.warning(
                    f"論理異常: {record.company_name} {record.year} "
                    f"Global({record.market_share_global}) >> Domestic({record.market_share_domestic})"
                )
                
        return True
        
    def _temporal_validation(self, record: MarketShareData, all_data: List[MarketShareData]) -> bool:
        """時系列整合性検証"""
        # 同一企業・同一市場の前後年データと比較
        company_data = [d for d in all_data 
                        if d.company_name == record.company_name 
                        and d.market_sector == record.market_sector]
        
        if len(company_data) < 2:
            return True
            
        # 急激な変化をチェック（年間50%以上の変化は要注意）
        for other in company_data:
            if abs(other.year - record.year) == 1:
                if (record.market_share_global is not None and 
                    other.market_share_global is not None):
                    change_rate = abs(record.market_share_global - other.market_share_global)
                    if change_rate > 50:
                        self.logger.warning(
                            f"急激変化: {record.company_name} {record.year} "
                            f"変化率{change_rate:.1f}%"
                        )
                        
        return True
        
    async def _save_to_database(self, data: List[MarketShareData]):
        """データベース保存"""
        if not data:
            return
            
        # SQLiteテーブル作成
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS market_share_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            company_name TEXT NOT NULL,
            company_code TEXT,
            market_category TEXT NOT NULL,
            market_sector TEXT NOT NULL,
            year INTEGER NOT NULL,
            market_share_global REAL,
            market_share_domestic REAL,
            market_share_asia REAL,
            market_size_million_usd REAL,
            ranking_global INTEGER,
            ranking_domestic INTEGER,
            data_source TEXT,
            reliability_score REAL,
            collection_date TEXT,
            notes TEXT,
            UNIQUE(company_name, market_sector, year)
        )
        """
        
        await self.db_manager.execute_query(create_table_sql)
        
        # データ挿入
        insert_sql = """
        INSERT OR REPLACE INTO market_share_data 
        (company_name, company_code, market_category, market_sector, year,
            market_share_global, market_share_domestic, market_share_asia,
            market_size_million_usd, ranking_global, ranking_domestic,
            data_source, reliability_score, collection_date, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        data_tuples = []
        for record in data:
            data_tuples.append((
                record.company_name,
                record.company_code,
                record.market_category,
                record.market_sector,
                record.year,
                record.market_share_global,
                record.market_share_domestic,
                record.market_share_asia,
                record.market_size_million_usd,
                record.ranking_global,
                record.ranking_domestic,
                record.data_source,
                record.reliability_score,
                record.collection_date.isoformat() if record.collection_date else None,
                record.notes
            ))
            
        await self.db_manager.execute_many(insert_sql, data_tuples)
        self.logger.info(f"データベース保存完了: {len(data_tuples)}レコード")


class GovernmentStatsCollector:
    """政府統計データ収集クラス"""
    
    def __init__(self):
        self.logger = setup_logger(f"{__name__}.GovernmentStats")
        
    async def collect_company_market_share(self, 
                                            company_info: Dict[str, Any],
                                            segment: MarketSegment,
                                            start_year: int, 
                                            end_year: int) -> List[MarketShareData]:
        """政府統計からの市場シェアデータ収集"""
        data = []
        
        # 総務省情報通信白書
        if segment.sector_name in ['ロボット', '内視鏡', '電子材料']:
            meti_data = await self._collect_from_meti(company_info, segment, start_year, end_year)
            data.extend(meti_data)
            
        # JEITA統計
        if segment.sector_name in ['電子材料', '精密測定機器']:
            jeita_data = await self._collect_from_jeita(company_info, segment, start_year, end_year)
            data.extend(jeita_data)
            
        return data
        
    async def _collect_from_meti(self, company_info, segment, start_year, end_year):
        """経産省データ収集"""
        # 実装省略：実際のAPI呼び出しまたはスクレイピング
        # ここでは模擬データを返す
        mock_data = []
        for year in range(start_year, end_year + 1):
            if year > 2020:  # 最近のデータのみ
                mock_data.append(MarketShareData(
                    company_name=company_info['name'],
                    company_code=company_info['code'],
                    market_category=segment.category,
                    market_sector=segment.sector_name,
                    year=year,
                    market_share_global=np.random.uniform(10, 40) if segment.category == 'high_share' else np.random.uniform(1, 10),
                    data_source='METI',
                    reliability_score=0.8,
                    collection_date=datetime.now()
                ))
        return mock_data
        
    async def _collect_from_jeita(self, company_info, segment, start_year, end_year):
        """JEITA統計収集"""
        # 実装省略
        return []


class IndustryReportCollector:
    """業界レポートデータ収集クラス"""
    
    def __init__(self):
        self.logger = setup_logger(f"{__name__}.IndustryReports")
        
    async def collect_company_market_share(self, 
                                            company_info: Dict[str, Any],
                                            segment: MarketSegment,
                                            start_year: int, 
                                            end_year: int) -> List[MarketShareData]:
        """業界レポートからの市場シェアデータ収集"""
        data = []
        
        # Gartner レポート
        gartner_data = await self._collect_from_gartner(company_info, segment, start_year, end_year)
        data.extend(gartner_data)
        
        # IDC レポート
        idc_data = await self._collect_from_idc(company_info, segment, start_year, end_year)
        data.extend(idc_data)
        
        return data
        
    async def _collect_from_gartner(self, company_info, segment, start_year, end_year):
        """Gartnerレポート収集"""
        # 実装省略
        return []
        
    async def _collect_from_idc(self, company_info, segment, start_year, end_year):
        """IDCレポート収集"""
        # 実装省略  
        return []


class CorporateIRCollector:
    """企業IR資料データ収集クラス"""
    
    def __init__(self):
        self.logger = setup_logger(f"{__name__}.CorporateIR")
        
    async def collect_company_market_share(self, 
                                            company_info: Dict[str, Any],
                                            segment: MarketSegment,
                                            start_year: int, 
                                            end_year: int) -> List[MarketShareData]:
        """企業IR資料からの市場シェアデータ収集"""
        data = []
        
        # 企業IRサイトから決算説明資料を収集
        ir_data = await self._collect_from_ir_materials(company_info, segment, start_year, end_year)
        data.extend(ir_data)
        
        return data
        
    async def _collect_from_ir_materials(self, company_info, segment, start_year, end_year):
        """IR資料収集"""
        # 実装省略
        return []


class ResearchInstitutionCollector:
    """研究機関データ収集クラス"""
    
    def __init__(self):
        self.logger = setup_logger(f"{__name__}.ResearchInstitution")
        
    async def collect_company_market_share(self, 
                                            company_info: Dict[str, Any],
                                            segment: MarketSegment,
                                            start_year: int, 
                                            end_year: int) -> List[MarketShareData]:
        """研究機関からの市場シェアデータ収集"""
        # 実装省略
        return []


class MarketResearchCollector:
    """市場調査会社データ収集クラス"""
    
    def __init__(self):
        self.logger = setup_logger(f"{__name__}.MarketResearch")
        
    async def collect_company_market_share(self, 
                                            company_info: Dict[str, Any],
                                            segment: MarketSegment,
                                            start_year: int, 
                                            end_year: int) -> List[MarketShareData]:
        """市場調査会社からの市場シェアデータ収集"""
        # 実装省略
        return []


# 使用例
if __name__ == "__main__":
    async def main():
        collector = MarketShareCollector()
        
        # 全市場シェアデータ収集
        market_data = await collector.collect_all_market_share_data(
            start_year=1984, 
            end_year=2024
        )
        
        print(f"収集完了: {len(market_data)}レコード")
        
        # 高シェア市場の企業リスト表示
        high_share_companies = [d for d in market_data if d.market_category == 'high_share']
        print(f"高シェア市場企業数: {len(set(d.company_name for d in high_share_companies))}")
        
    # 実行
    asyncio.run(main())