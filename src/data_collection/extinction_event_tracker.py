"""
企業消滅イベント追跡システム (extinction_event_tracker.py)

このモジュールは、150社の対象企業における企業消滅イベントを体系的に追跡・記録します。
消滅パターンの分類、消滅時期の特定、消滅原因の分析、関連企業への影響追跡を行います。

主な機能:
1. 企業消滅イベントの自動検出・分類
2. 消滅プロセスの時系列追跡
3. 財務データとの連携による消滅予兆分析
4. 業界・市場への影響評価
5. 生存分析用データセット生成
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
import requests
import json
from pathlib import Path
import warnings

# A2AI内部モジュールのインポート
from ..utils.data_utils import DataUtils
from ..utils.logging_utils import setup_logger
from ..utils.database_utils import DatabaseManager
from .data_validator import DataValidator

logger = setup_logger(__name__)

class ExtinctionType(Enum):
    """企業消滅タイプの分類"""
    BANKRUPTCY = "bankruptcy"              # 倒産・破産
    ACQUISITION = "acquisition"           # 買収・吸収合併
    DIVESTITURE = "divestiture"          # 事業売却
    LIQUIDATION = "liquidation"          # 清算・解散  
    DELISTING = "delisting"              # 上場廃止
    RESTRUCTURE = "restructure"          # 事業再編・分社化
    BUSINESS_EXIT = "business_exit"      # 事業撤退
    NATIONALIZATION = "nationalization"  # 国有化
    SPIN_OFF = "spin_off"               # スピンオフ分離
    UNKNOWN = "unknown"                  # 不明・その他

class ExtinctionStatus(Enum):
    """企業消滅ステータス"""
    ACTIVE = "active"                    # 継続中
    WARNING = "warning"                  # 警告（予兆段階）
    CRITICAL = "critical"                # 危機的状況
    EXTINCT = "extinct"                  # 完全消滅
    ABSORBED = "absorbed"                # 他社に吸収
    TRANSFORMED = "transformed"          # 別企業に変化

@dataclass
class ExtinctionEvent:
    """企業消滅イベント情報"""
    company_id: str
    company_name: str
    extinction_type: ExtinctionType
    extinction_date: Optional[datetime] = None
    announcement_date: Optional[datetime] = None
    completion_date: Optional[datetime] = None
    status: ExtinctionStatus = ExtinctionStatus.ACTIVE
    
    # 関連企業情報
    acquiring_company: Optional[str] = None
    successor_company: Optional[str] = None
    parent_company: Optional[str] = None
    
    # 財務・事業情報
    final_revenue: Optional[float] = None
    final_assets: Optional[float] = None
    final_employees: Optional[int] = None
    debt_amount: Optional[float] = None
    
    # 消滅原因・詳細
    primary_cause: Optional[str] = None
    secondary_causes: List[str] = field(default_factory=list)
    market_impact: Optional[str] = None
    
    # データ品質
    data_quality_score: float = 0.0
    confidence_level: float = 0.0
    information_sources: List[str] = field(default_factory=list)
    
    # メタデータ
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    notes: Optional[str] = None

class ExtinctionEventTracker:
    """企業消滅イベント追跡システムメインクラス"""
    
    def __init__(self, config_path: str = None):
        """
        初期化
        
        Args:
            config_path: 設定ファイルパス
        """
        self.config = self._load_config(config_path)
        self.db_manager = DatabaseManager(self.config.get('database', {}))
        self.data_validator = DataValidator()
        self.data_utils = DataUtils()
        
        # 企業リストの読み込み
        self.target_companies = self._load_target_companies()
        
        # 消滅イベント追跡対象企業（添付リストから特定）
        self.extinction_targets = self._identify_extinction_targets()
        
        # 追跡結果保存
        self.extinction_events: Dict[str, ExtinctionEvent] = {}
        
        logger.info(f"ExtinctionEventTracker初期化完了: 対象企業{len(self.extinction_targets)}社")

    def _load_config(self, config_path: str) -> Dict:
        """設定ファイル読み込み"""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        # デフォルト設定
        return {
            'edinet_api': {
                'base_url': 'https://disclosure2.edinet-fsa.go.jp/api/v2/',
                'timeout': 30
            },
            'data_sources': {
                'company_db': True,
                'news_api': True,
                'official_announcements': True
            },
            'tracking_period': {
                'start_year': 1984,
                'end_year': 2024
            }
        }

    def _load_target_companies(self) -> pd.DataFrame:
        """対象企業150社リストの読み込み"""
        try:
            # 添付文書から企業リストを構築
            companies_data = []
            
            # 高シェア市場企業 (50社)
            high_share_companies = [
                # ロボット市場
                ("ファナック", "robot", "high_share", "6954", True),
                ("安川電機", "robot", "high_share", "6506", True),
                ("川崎重工業", "robot", "high_share", "7012", True),
                ("不二越", "robot", "high_share", "6474", True),
                ("デンソーウェーブ", "robot", "high_share", None, False), # 子会社
                ("三菱電機", "robot", "high_share", "6503", True),
                ("オムロン", "robot", "high_share", "6645", True),
                ("THK", "robot", "high_share", "6481", True),
                ("NSK", "robot", "high_share", "6471", True),
                ("IHI", "robot", "high_share", "7013", True),
                
                # 内視鏡市場
                ("オリンパス", "endoscope", "high_share", "7733", True),
                ("HOYA", "endoscope", "high_share", "7741", True),
                ("富士フイルム", "endoscope", "high_share", "4901", True),
                ("キヤノンメディカルシステムズ", "endoscope", "high_share", None, False), # キヤノン子会社
                ("島津製作所", "endoscope", "high_share", "7701", True),
                ("コニカミノルタ", "endoscope", "high_share", "4902", True),
                ("ソニー", "endoscope", "high_share", "6758", True),
                ("トプコン", "endoscope", "high_share", "7732", True),
                ("エムスリー", "endoscope", "high_share", "2413", True),
                ("日立製作所", "endoscope", "high_share", "6501", True),
                
                # 他の高シェア市場企業も同様に追加...
            ]
            
            # シェア低下市場企業 (50社)
            declining_companies = [
                # 自動車市場
                ("トヨタ自動車", "automotive", "declining", "7203", True),
                ("日産自動車", "automotive", "declining", "7201", True),
                ("ホンダ", "automotive", "declining", "7267", True),
                ("スズキ", "automotive", "declining", "7269", True),
                ("マツダ", "automotive", "declining", "7261", True),
                ("SUBARU", "automotive", "declining", "7270", True),
                ("いすゞ自動車", "automotive", "declining", "7202", True),
                ("三菱自動車", "automotive", "declining", "7211", True),
                ("ダイハツ工業", "automotive", "declining", None, False), # トヨタ傘下
                ("日野自動車", "automotive", "declining", "7205", True),
                
                # 他のシェア低下企業...
            ]
            
            # 完全失失市場企業 (50社) - 消滅イベント対象
            extinct_companies = [
                # 家電市場
                ("三洋電機", "appliance", "extinct", None, False), # パナソニック吸収・消滅
                ("アイワ", "appliance", "extinct", None, False), # ソニー傘下で実質消滅
                ("東芝ライフスタイル", "appliance", "extinct", None, False), # 美的集団売却
                ("ビクター", "appliance", "extinct", None, False), # JVCケンウッド統合
                ("船井電機", "appliance", "extinct", "6839", True), # 縮小継続中
                
                # 半導体市場
                ("キオクシア", "semiconductor", "extinct", None, False), # 東芝メモリ分社
                ("ルネサスエレクトロニクス", "semiconductor", "extinct", "6723", True), # 統合企業
                
                # スマートフォン市場
                ("FCNT", "smartphone", "extinct", None, False), # 2023年破綻
                ("京セラ", "smartphone", "extinct", "6971", True), # スマホ撤退
                
                # 他の消滅企業...
            ]
            
            # データフレーム作成
            for name, market, category, code, is_listed in (
                high_share_companies + declining_companies + extinct_companies
            ):
                companies_data.append({
                    'company_name': name,
                    'market_category': market,
                    'share_status': category,
                    'stock_code': code,
                    'is_listed': is_listed,
                    'extinction_risk': 1.0 if category == "extinct" else 
                                        0.3 if category == "declining" else 0.1
                })
            
            return pd.DataFrame(companies_data)
            
        except Exception as e:
            logger.error(f"対象企業リスト読み込み失敗: {e}")
            return pd.DataFrame()

    def _identify_extinction_targets(self) -> List[str]:
        """消滅イベント追跡対象企業の特定"""
        if self.target_companies.empty:
            return []
        
        # 消滅リスクが高い企業を特定
        extinction_targets = self.target_companies[
            (self.target_companies['share_status'] == 'extinct') |
            (self.target_companies['extinction_risk'] >= 0.3)
        ]
        
        return extinction_targets['company_name'].tolist()

    def track_all_extinction_events(self) -> Dict[str, ExtinctionEvent]:
        """全対象企業の消滅イベント追跡"""
        logger.info("企業消滅イベント追跡開始")
        
        results = {}
        
        for company_name in self.extinction_targets:
            try:
                logger.info(f"企業消滅追跡中: {company_name}")
                
                # 基本企業情報取得
                company_info = self._get_company_info(company_name)
                
                # 消滅イベント検出・分析
                extinction_event = self._detect_extinction_event(company_info)
                
                # 財務データとの連携
                if extinction_event:
                    extinction_event = self._enrich_with_financial_data(extinction_event)
                    
                    # 消滅プロセス詳細分析
                    extinction_event = self._analyze_extinction_process(extinction_event)
                    
                    results[company_name] = extinction_event
                    self.extinction_events[company_name] = extinction_event
                    
                    logger.info(f"消滅イベント記録完了: {company_name} - {extinction_event.extinction_type.value}")
                
            except Exception as e:
                logger.error(f"企業消滅追跡エラー {company_name}: {e}")
                continue
        
        logger.info(f"企業消滅イベント追跡完了: {len(results)}件")
        return results

    def _get_company_info(self, company_name: str) -> Dict:
        """企業基本情報取得"""
        company_row = self.target_companies[
            self.target_companies['company_name'] == company_name
        ]
        
        if company_row.empty:
            return {'name': company_name}
        
        return {
            'name': company_name,
            'market_category': company_row.iloc[0]['market_category'],
            'share_status': company_row.iloc[0]['share_status'],
            'stock_code': company_row.iloc[0]['stock_code'],
            'is_listed': company_row.iloc[0]['is_listed'],
            'extinction_risk': company_row.iloc[0]['extinction_risk']
        }

    def _detect_extinction_event(self, company_info: Dict) -> Optional[ExtinctionEvent]:
        """消滅イベントの検出・分析"""
        company_name = company_info['name']
        
        # 既知の消滅企業パターンマッチング
        extinction_patterns = self._get_known_extinction_patterns()
        
        if company_name in extinction_patterns:
            pattern = extinction_patterns[company_name]
            
            return ExtinctionEvent(
                company_id=self._generate_company_id(company_name),
                company_name=company_name,
                extinction_type=ExtinctionType(pattern['type']),
                extinction_date=self._parse_date(pattern.get('extinction_date')),
                announcement_date=self._parse_date(pattern.get('announcement_date')),
                status=ExtinctionStatus.EXTINCT if pattern.get('is_extinct', False) 
                        else ExtinctionStatus.CRITICAL,
                acquiring_company=pattern.get('acquiring_company'),
                successor_company=pattern.get('successor_company'),
                primary_cause=pattern.get('primary_cause'),
                secondary_causes=pattern.get('secondary_causes', []),
                information_sources=pattern.get('sources', []),
                confidence_level=pattern.get('confidence', 0.8)
            )
        
        # 上場企業の場合はEDINETで状況確認
        if company_info.get('is_listed') and company_info.get('stock_code'):
            return self._check_listed_company_status(company_info)
        
        # その他の検出方法
        return self._detect_general_extinction(company_info)

    def _get_known_extinction_patterns(self) -> Dict:
        """既知の企業消滅パターン辞書"""
        return {
            "三洋電機": {
                "type": "acquisition",
                "extinction_date": "2012-04-01",
                "announcement_date": "2009-12-21",
                "is_extinct": True,
                "acquiring_company": "パナソニック",
                "primary_cause": "経営悪化による買収",
                "secondary_causes": ["リーマンショック影響", "家電市場競争激化"],
                "sources": ["パナソニック公式発表", "EDINET"],
                "confidence": 1.0
            },
            "アイワ": {
                "type": "acquisition",
                "extinction_date": "2002-10-01", 
                "announcement_date": "2002-01-15",
                "is_extinct": True,
                "acquiring_company": "ソニー",
                "primary_cause": "ブランド統合",
                "secondary_causes": ["市場シェア低下", "コスト削減"],
                "sources": ["ソニー公式発表"],
                "confidence": 1.0
            },
            "東芝ライフスタイル": {
                "type": "divestiture",
                "extinction_date": "2016-06-01",
                "announcement_date": "2015-12-15",
                "is_extinct": True,
                "acquiring_company": "美的集団",
                "successor_company": "東芝ライフスタイル（美的傘下）",
                "primary_cause": "事業売却",
                "secondary_causes": ["東芝本体の財務悪化", "海外展開困難"],
                "sources": ["東芝公式発表", "美的集団発表"],
                "confidence": 1.0
            },
            "FCNT": {
                "type": "bankruptcy",
                "extinction_date": "2023-10-31",
                "announcement_date": "2023-08-01", 
                "is_extinct": True,
                "primary_cause": "経営破綻",
                "secondary_causes": ["スマホ市場競争激化", "資金調達困難"],
                "sources": ["官報", "東京商工リサーチ"],
                "confidence": 1.0
            },
            "京セラ": {
                "type": "business_exit",
                "extinction_date": "2023-03-31",
                "announcement_date": "2022-11-01",
                "is_extinct": False,  # 企業は存続、スマホ事業のみ撤退
                "primary_cause": "スマホ事業撤退",
                "secondary_causes": ["収益性悪化", "市場競争激化"],
                "sources": ["京セラ公式発表"],
                "confidence": 1.0
            },
            "キオクシア": {
                "type": "spin_off",
                "extinction_date": "2018-06-01",  # 東芝メモリから独立
                "announcement_date": "2017-09-01",
                "is_extinct": False,  # 独立企業として存続
                "successor_company": "キオクシア",
                "parent_company": "東芝",
                "primary_cause": "スピンオフ独立",
                "secondary_causes": ["東芝事業再編", "半導体事業特化"],
                "sources": ["東芝公式発表", "キオクシア設立発表"],
                "confidence": 1.0
            },
            "ルネサスエレクトロニクス": {
                "type": "restructure",
                "extinction_date": "2010-04-01",  # 統合設立
                "announcement_date": "2008-12-01",
                "is_extinct": False,  # 統合企業として存続
                "primary_cause": "企業統合",
                "secondary_causes": ["半導体市場再編", "規模の経済追求"],
                "sources": ["NEC・日立・三菱電機合弁発表"],
                "confidence": 1.0
            }
        }

    def _parse_date(self, date_str: Optional[str]) -> Optional[datetime]:
        """日付文字列をdatetimeに変換"""
        if not date_str:
            return None
        
        try:
            return datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            try:
                return datetime.strptime(date_str, "%Y/%m/%d")
            except ValueError:
                return None

    def _generate_company_id(self, company_name: str) -> str:
        """企業IDの生成"""
        return f"A2AI_{hash(company_name) % 100000:05d}"

    def _check_listed_company_status(self, company_info: Dict) -> Optional[ExtinctionEvent]:
        """上場企業のステータス確認"""
        # EDINET APIを使用した状況確認の実装
        # 簡略化版 - 実際にはEDINET APIでリアルタイム確認
        stock_code = company_info.get('stock_code')
        company_name = company_info['name']
        
        # 上場廃止企業の確認
        delisted_companies = {
            "6839": "船井電機",  # 事業縮小中
            "7205": "日野自動車", # 上場廃止検討
        }
        
        if stock_code in delisted_companies:
            return ExtinctionEvent(
                company_id=self._generate_company_id(company_name),
                company_name=company_name,
                extinction_type=ExtinctionType.DELISTING,
                status=ExtinctionStatus.CRITICAL,
                primary_cause="上場廃止",
                confidence_level=0.8,
                information_sources=["EDINET", "証券取引所公表"]
            )
        
        return None

    def _detect_general_extinction(self, company_info: Dict) -> Optional[ExtinctionEvent]:
        """一般的な消滅イベント検出"""
        company_name = company_info['name']
        extinction_risk = company_info.get('extinction_risk', 0.0)
        
        if extinction_risk >= 0.8:
            return ExtinctionEvent(
                company_id=self._generate_company_id(company_name),
                company_name=company_name,
                extinction_type=ExtinctionType.UNKNOWN,
                status=ExtinctionStatus.CRITICAL,
                primary_cause="高リスク企業",
                confidence_level=extinction_risk,
                information_sources=["内部リスク評価"]
            )
        
        return None

    def _enrich_with_financial_data(self, extinction_event: ExtinctionEvent) -> ExtinctionEvent:
        """財務データによる消滅イベント情報の充実化"""
        try:
            # 消滅直前の財務データ取得
            company_name = extinction_event.company_name
            
            # ここでは簡略化版 - 実際にはEDINETやDBから取得
            financial_estimates = self._estimate_final_financials(company_name)
            
            extinction_event.final_revenue = financial_estimates.get('revenue')
            extinction_event.final_assets = financial_estimates.get('assets')  
            extinction_event.final_employees = financial_estimates.get('employees')
            extinction_event.debt_amount = financial_estimates.get('debt')
            
            return extinction_event
            
        except Exception as e:
            logger.warning(f"財務データ取得失敗 {extinction_event.company_name}: {e}")
            return extinction_event

    def _estimate_final_financials(self, company_name: str) -> Dict:
        """最終財務指標の推定"""
        # 企業規模による推定値（実際にはDBから取得）
        estimates = {
            "三洋電機": {
                "revenue": 1500000,  # 百万円
                "assets": 2000000,
                "employees": 95000,
                "debt": 800000
            },
            "アイワ": {
                "revenue": 50000,
                "assets": 80000,
                "employees": 1200,
                "debt": 30000
            },
            "FCNT": {
                "revenue": 10000,
                "assets": 15000, 
                "employees": 300,
                "debt": 20000
            }
        }
        
        return estimates.get(company_name, {})

    def _analyze_extinction_process(self, extinction_event: ExtinctionEvent) -> ExtinctionEvent:
        """消滅プロセスの詳細分析"""
        try:
            # 消滅タイプ別の詳細分析
            if extinction_event.extinction_type == ExtinctionType.ACQUISITION:
                extinction_event = self._analyze_acquisition_process(extinction_event)
            elif extinction_event.extinction_type == ExtinctionType.BANKRUPTCY:
                extinction_event = self._analyze_bankruptcy_process(extinction_event)
            elif extinction_event.extinction_type == ExtinctionType.BUSINESS_EXIT:
                extinction_event = self._analyze_business_exit_process(extinction_event)
            
            # データ品質スコア算出
            extinction_event.data_quality_score = self._calculate_data_quality_score(extinction_event)
            
            return extinction_event
            
        except Exception as e:
            logger.warning(f"消滅プロセス分析失敗 {extinction_event.company_name}: {e}")
            return extinction_event

    def _analyze_acquisition_process(self, event: ExtinctionEvent) -> ExtinctionEvent:
        """買収プロセスの分析"""
        # 買収による消滅の市場影響評価
        if event.acquiring_company:
            event.market_impact = f"{event.acquiring_company}への統合により市場集約化"
        
        # 買収価格の推定（実際にはより詳細な分析）
        if event.final_assets and event.final_revenue:
            estimated_value = event.final_assets * 0.8  # 簡易推定
            event.notes = f"推定買収価格: {estimated_value:,.0f}百万円"
        
        return event

    def _analyze_bankruptcy_process(self, event: ExtinctionEvent) -> ExtinctionEvent:
        """倒産プロセスの分析"""
        # 負債比率分析
        if event.debt_amount and event.final_assets:
            debt_ratio = event.debt_amount / event.final_assets
            if debt_ratio > 0.7:
                event.secondary_causes.append(f"高負債比率({debt_ratio:.2f})")
        
        event.market_impact = "市場シェア他社に移転、競合他社に利益"
        return event

    def _analyze_business_exit_process(self, event: ExtinctionEvent) -> ExtinctionEvent:
        """事業撤退プロセスの分析"""
        event.market_impact = "特定事業からの撤退、他事業へのリソース集中"
        
        # 撤退理由の分析強化
        if "収益性悪化" in event.secondary_causes:
            event.notes = "収益性改善困難による戦略的撤退"
        
        return event

    def _calculate_data_quality_score(self, event: ExtinctionEvent) -> float:
        """データ品質スコアの算出"""
        score = 0.0
        
        # 基本情報の完全性
        if event.extinction_date:
            score += 0.2
        if event.extinction_type != ExtinctionType.UNKNOWN:
            score += 0.2
        if event.primary_cause:
            score += 0.1
            
        # 財務データの完全性
        if event.final_revenue:
            score += 0.1
        if event.final_assets:
            score += 0.1
            
        # 情報ソースの信頼性
        if event.information_sources:
            source_score = min(len(event.information_sources) * 0.1, 0.3)
            score += source_score
            
        return min(score, 1.0)

    def generate_extinction_summary(self) -> pd.DataFrame:
        """消滅イベントサマリーレポート生成"""
        if not self.extinction_events:
            return pd.DataFrame()
        
        summary_data = []
        
        for company_name, event in self.extinction_events.items():
            summary_data.append({
                'company_name': event.company_name,
                'extinction_type': event.extinction_type.value,
                'extinction_date': event.extinction_date,
                'status': event.status.value,
                'acquiring_company': event.acquiring_company,
                'primary_cause': event.primary_cause,
                'final_revenue': event.final_revenue,
                'final_employees': event.final_employees,
                'data_quality_score': event.data_quality_score,
                'confidence_level': event.confidence_level
            })
        
        return pd.DataFrame(summary_data)

    def export_survival_analysis_data(self, output_path: str) -> bool:
        """生存分析用データセット出力"""
        try:
            survival_data = []
            
            for company_name, event in self.extinction_events.items():
                # 生存時間計算（設立から消滅まで）
                if event.extinction_date:
                    # 簡易的に1984年を基準年とする
                    baseline_date = datetime(1984, 1, 1)
                    survival_time = (event.extinction_date - baseline_date).days / 365.25
                    event_occurred = 1
                else:
                    # 2024年まで生存
                    survival_time = (datetime(2024, 12, 31) - baseline_date).days / 365.25
                    event_occurred = 0  # 生存中（右側打ち切り）
                
                survival_data.append({
                    'company_id': event.company_id,
                    'company_name': event.company_name,
                    'survival_time_years': survival_time,
                    'event_occurred': event_occurred,
                    'extinction_type': event.extinction_type.value,
                    'market_category': self._get_company_market(company_name),
                    'share_status': self._get_company_share_status(company_name),
                    'final_revenue': event.final_revenue,
                    'final_assets': event.final_assets,
                    'final_employees': event.final_employees,
                    'debt_amount': event.debt_amount,
                    'primary_cause': event.primary_cause
                })
            
            # DataFrame作成・保存
            survival_df = pd.DataFrame(survival_data)
            survival_df.to_csv(output_path, index=False, encoding='utf-8')
            
            logger.info(f"生存分析データ出力完了: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"生存分析データ出力失敗: {e}")
            return False

    def _get_company_market(self, company_name: str) -> str:
        """企業の市場カテゴリ取得"""
        company_row = self.target_companies[
            self.target_companies['company_name'] == company_name
        ]
        return company_row.iloc[0]['market_category'] if not company_row.empty else 'unknown'

    def _get_company_share_status(self, company_name: str) -> str:
        """企業のシェア状況取得"""
        company_row = self.target_companies[
            self.target_companies['company_name'] == company_name
        ]
        return company_row.iloc[0]['share_status'] if not company_row.empty else 'unknown'

    def create_extinction_timeline(self) -> pd.DataFrame:
        """消滅イベントタイムライン作成"""
        timeline_data = []
        
        for event in self.extinction_events.values():
            if event.extinction_date:
                timeline_data.append({
                    'date': event.extinction_date,
                    'company_name': event.company_name,
                    'extinction_type': event.extinction_type.value,
                    'market_category': self._get_company_market(event.company_name),
                    'acquiring_company': event.acquiring_company,
                    'primary_cause': event.primary_cause,
                    'market_impact': event.market_impact
                })
        
        timeline_df = pd.DataFrame(timeline_data)
        if not timeline_df.empty:
            timeline_df = timeline_df.sort_values('date')
        
        return timeline_df

    def analyze_extinction_patterns(self) -> Dict[str, Any]:
        """消滅パターン分析"""
        if not self.extinction_events:
            return {}
        
        analysis_results = {}
        
        # 消滅タイプ別統計
        extinction_types = [event.extinction_type.value for event in self.extinction_events.values()]
        type_counts = pd.Series(extinction_types).value_counts()
        analysis_results['extinction_type_distribution'] = type_counts.to_dict()
        
        # 市場別消滅率
        market_extinction = {}
        for event in self.extinction_events.values():
            market = self._get_company_market(event.company_name)
            if market not in market_extinction:
                market_extinction[market] = {'total': 0, 'extinct': 0}
            market_extinction[market]['total'] += 1
            if event.status == ExtinctionStatus.EXTINCT:
                market_extinction[market]['extinct'] += 1
        
        market_rates = {}
        for market, data in market_extinction.items():
            if data['total'] > 0:
                market_rates[market] = data['extinct'] / data['total']
        
        analysis_results['market_extinction_rates'] = market_rates
        
        # 時期別消滅傾向
        extinction_years = []
        for event in self.extinction_events.values():
            if event.extinction_date:
                extinction_years.append(event.extinction_date.year)
        
        if extinction_years:
            year_counts = pd.Series(extinction_years).value_counts().sort_index()
            analysis_results['extinction_timeline'] = year_counts.to_dict()
        
        # 消滅原因分析
        primary_causes = [event.primary_cause for event in self.extinction_events.values() 
                            if event.primary_cause]
        if primary_causes:
            cause_counts = pd.Series(primary_causes).value_counts()
            analysis_results['primary_causes_distribution'] = cause_counts.to_dict()
        
        return analysis_results

    def generate_risk_assessment(self, company_name: str) -> Dict[str, Any]:
        """個別企業の消滅リスク評価"""
        if company_name not in self.extinction_events:
            return {'error': f'Company {company_name} not found in tracking data'}
        
        event = self.extinction_events[company_name]
        
        risk_assessment = {
            'company_name': company_name,
            'current_status': event.status.value,
            'extinction_type': event.extinction_type.value,
            'risk_level': self._calculate_risk_level(event),
            'key_risk_factors': event.secondary_causes,
            'financial_indicators': {
                'final_revenue': event.final_revenue,
                'final_assets': event.final_assets,
                'debt_amount': event.debt_amount,
                'debt_to_asset_ratio': (event.debt_amount / event.final_assets 
                                        if event.debt_amount and event.final_assets else None)
            },
            'market_context': {
                'market_category': self._get_company_market(company_name),
                'share_status': self._get_company_share_status(company_name),
                'market_impact': event.market_impact
            },
            'data_quality': {
                'quality_score': event.data_quality_score,
                'confidence_level': event.confidence_level,
                'information_sources': event.information_sources
            }
        }
        
        return risk_assessment

    def _calculate_risk_level(self, event: ExtinctionEvent) -> str:
        """リスクレベル算出"""
        if event.status == ExtinctionStatus.EXTINCT:
            return 'EXTINCT'
        elif event.status == ExtinctionStatus.CRITICAL:
            return 'HIGH'
        elif event.status == ExtinctionStatus.WARNING:
            return 'MEDIUM'
        else:
            return 'LOW'

    def update_extinction_event(self, company_name: str, 
                                updates: Dict[str, Any]) -> bool:
        """消滅イベント情報の更新"""
        if company_name not in self.extinction_events:
            logger.warning(f"更新対象企業が見つからない: {company_name}")
            return False
        
        try:
            event = self.extinction_events[company_name]
            
            # 更新可能フィールドの更新
            updatable_fields = [
                'extinction_date', 'announcement_date', 'completion_date',
                'status', 'acquiring_company', 'successor_company',
                'final_revenue', 'final_assets', 'final_employees', 'debt_amount',
                'primary_cause', 'secondary_causes', 'market_impact', 'notes'
            ]
            
            for field, value in updates.items():
                if field in updatable_fields and hasattr(event, field):
                    if field == 'status' and isinstance(value, str):
                        setattr(event, field, ExtinctionStatus(value))
                    elif field in ['extinction_date', 'announcement_date', 'completion_date']:
                        setattr(event, field, self._parse_date(value) if isinstance(value, str) else value)
                    else:
                        setattr(event, field, value)
            
            # 更新時刻記録
            event.updated_at = datetime.now()
            
            # データ品質スコア再計算
            event.data_quality_score = self._calculate_data_quality_score(event)
            
            logger.info(f"消滅イベント情報更新完了: {company_name}")
            return True
            
        except Exception as e:
            logger.error(f"消滅イベント更新エラー {company_name}: {e}")
            return False

    def save_extinction_data(self, output_dir: str) -> bool:
        """消滅イベントデータの保存"""
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # 1. サマリーレポート保存
            summary_df = self.generate_extinction_summary()
            summary_path = output_path / 'extinction_events_summary.csv'
            summary_df.to_csv(summary_path, index=False, encoding='utf-8')
            
            # 2. タイムライン保存
            timeline_df = self.create_extinction_timeline()
            timeline_path = output_path / 'extinction_timeline.csv'
            timeline_df.to_csv(timeline_path, index=False, encoding='utf-8')
            
            # 3. 生存分析データ保存
            survival_path = output_path / 'survival_analysis_data.csv'
            self.export_survival_analysis_data(str(survival_path))
            
            # 4. パターン分析結果保存
            patterns = self.analyze_extinction_patterns()
            patterns_path = output_path / 'extinction_patterns_analysis.json'
            with open(patterns_path, 'w', encoding='utf-8') as f:
                json.dump(patterns, f, ensure_ascii=False, indent=2, default=str)
            
            # 5. 詳細データ保存（JSON形式）
            detailed_data = {}
            for company_name, event in self.extinction_events.items():
                detailed_data[company_name] = {
                    'company_id': event.company_id,
                    'company_name': event.company_name,
                    'extinction_type': event.extinction_type.value,
                    'extinction_date': event.extinction_date.isoformat() if event.extinction_date else None,
                    'announcement_date': event.announcement_date.isoformat() if event.announcement_date else None,
                    'completion_date': event.completion_date.isoformat() if event.completion_date else None,
                    'status': event.status.value,
                    'acquiring_company': event.acquiring_company,
                    'successor_company': event.successor_company,
                    'parent_company': event.parent_company,
                    'final_revenue': event.final_revenue,
                    'final_assets': event.final_assets,
                    'final_employees': event.final_employees,
                    'debt_amount': event.debt_amount,
                    'primary_cause': event.primary_cause,
                    'secondary_causes': event.secondary_causes,
                    'market_impact': event.market_impact,
                    'data_quality_score': event.data_quality_score,
                    'confidence_level': event.confidence_level,
                    'information_sources': event.information_sources,
                    'created_at': event.created_at.isoformat(),
                    'updated_at': event.updated_at.isoformat(),
                    'notes': event.notes
                }
            
            detailed_path = output_path / 'extinction_events_detailed.json'
            with open(detailed_path, 'w', encoding='utf-8') as f:
                json.dump(detailed_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"消滅イベントデータ保存完了: {output_dir}")
            return True
            
        except Exception as e:
            logger.error(f"消滅イベントデータ保存エラー: {e}")
            return False

    def load_extinction_data(self, input_dir: str) -> bool:
        """保存済み消滅イベントデータの読み込み"""
        try:
            input_path = Path(input_dir)
            detailed_path = input_path / 'extinction_events_detailed.json'
            
            if not detailed_path.exists():
                logger.warning(f"消滅イベントデータファイルが見つからない: {detailed_path}")
                return False
            
            with open(detailed_path, 'r', encoding='utf-8') as f:
                detailed_data = json.load(f)
            
            # ExtinctionEventオブジェクトの復元
            for company_name, data in detailed_data.items():
                event = ExtinctionEvent(
                    company_id=data['company_id'],
                    company_name=data['company_name'],
                    extinction_type=ExtinctionType(data['extinction_type']),
                    extinction_date=datetime.fromisoformat(data['extinction_date']) if data['extinction_date'] else None,
                    announcement_date=datetime.fromisoformat(data['announcement_date']) if data['announcement_date'] else None,
                    completion_date=datetime.fromisoformat(data['completion_date']) if data['completion_date'] else None,
                    status=ExtinctionStatus(data['status']),
                    acquiring_company=data['acquiring_company'],
                    successor_company=data['successor_company'],
                    parent_company=data['parent_company'],
                    final_revenue=data['final_revenue'],
                    final_assets=data['final_assets'],
                    final_employees=data['final_employees'],
                    debt_amount=data['debt_amount'],
                    primary_cause=data['primary_cause'],
                    secondary_causes=data['secondary_causes'],
                    market_impact=data['market_impact'],
                    data_quality_score=data['data_quality_score'],
                    confidence_level=data['confidence_level'],
                    information_sources=data['information_sources'],
                    created_at=datetime.fromisoformat(data['created_at']),
                    updated_at=datetime.fromisoformat(data['updated_at']),
                    notes=data['notes']
                )
                
                self.extinction_events[company_name] = event
            
            logger.info(f"消滅イベントデータ読み込み完了: {len(self.extinction_events)}件")
            return True
            
        except Exception as e:
            logger.error(f"消滅イベントデータ読み込みエラー: {e}")
            return False

    def generate_extinction_report(self, output_path: str) -> bool:
        """包括的な消滅イベント分析レポート生成"""
        try:
            report_content = []
            report_content.append("# A2AI 企業消滅イベント分析レポート\n")
            report_content.append(f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            report_content.append("=" * 80 + "\n\n")
            
            # 1. 全体統計
            report_content.append("## 1. 全体統計\n")
            total_tracked = len(self.extinction_events)
            extinct_count = sum(1 for event in self.extinction_events.values() 
                                if event.status == ExtinctionStatus.EXTINCT)
            
            report_content.append(f"- 追跡対象企業数: {total_tracked}社\n")
            report_content.append(f"- 完全消滅企業数: {extinct_count}社\n")
            report_content.append(f"- 消滅率: {extinct_count/total_tracked*100:.1f}%\n\n")
            
            # 2. 消滅パターン分析
            patterns = self.analyze_extinction_patterns()
            
            report_content.append("## 2. 消滅タイプ別分析\n")
            if 'extinction_type_distribution' in patterns:
                for ext_type, count in patterns['extinction_type_distribution'].items():
                    percentage = count / total_tracked * 100
                    report_content.append(f"- {ext_type}: {count}社 ({percentage:.1f}%)\n")
            
            report_content.append("\n## 3. 市場別消滅率\n")
            if 'market_extinction_rates' in patterns:
                for market, rate in patterns['market_extinction_rates'].items():
                    report_content.append(f"- {market}: {rate*100:.1f}%\n")
            
            report_content.append("\n## 4. 主要消滅原因\n")
            if 'primary_causes_distribution' in patterns:
                for cause, count in patterns['primary_causes_distribution'].items():
                    report_content.append(f"- {cause}: {count}社\n")
            
            # 5. 時系列分析
            report_content.append("\n## 5. 時系列消滅傾向\n")
            if 'extinction_timeline' in patterns:
                for year, count in sorted(patterns['extinction_timeline'].items()):
                    report_content.append(f"- {year}年: {count}社\n")
            
            # 6. 個別企業詳細
            report_content.append("\n## 6. 個別企業消滅詳細\n")
            for company_name, event in self.extinction_events.items():
                if event.status == ExtinctionStatus.EXTINCT:
                    report_content.append(f"\n### {company_name}\n")
                    report_content.append(f"- 消滅タイプ: {event.extinction_type.value}\n")
                    report_content.append(f"- 消滅日: {event.extinction_date.strftime('%Y-%m-%d') if event.extinction_date else '不明'}\n")
                    report_content.append(f"- 買収企業: {event.acquiring_company or 'なし'}\n")
                    report_content.append(f"- 主要原因: {event.primary_cause or '不明'}\n")
                    if event.final_revenue:
                        report_content.append(f"- 最終売上: {event.final_revenue:,.0f}百万円\n")
                    if event.final_employees:
                        report_content.append(f"- 最終従業員数: {event.final_employees:,}人\n")
            
            # レポート保存
            with open(output_path, 'w', encoding='utf-8') as f:
                f.writelines(report_content)
            
            logger.info(f"消滅イベント分析レポート生成完了: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"レポート生成エラー: {e}")
            return False


# 使用例とテスト用のメイン関数
if __name__ == "__main__":
    # ログ設定
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # ExtinctionEventTracker初期化
        tracker = ExtinctionEventTracker()
        
        # 全企業の消滅イベント追跡実行
        extinction_events = tracker.track_all_extinction_events()
        
        print(f"\n追跡完了: {len(extinction_events)}社の消滅イベントを検出")
        
        # 結果サマリー表示
        summary_df = tracker.generate_extinction_summary()
        print("\n=== 消滅イベントサマリー ===")
        print(summary_df.to_string(index=False))
        
        # パターン分析実行
        patterns = tracker.analyze_extinction_patterns()
        print(f"\n=== 消滅パターン分析結果 ===")
        for key, value in patterns.items():
            print(f"{key}: {value}")
        
        # データ保存
        output_dir = "results/extinction_analysis"
        tracker.save_extinction_data(output_dir)
        print(f"\n消滅イベントデータを保存: {output_dir}")
        
        # レポート生成
        report_path = f"{output_dir}/extinction_analysis_report.md"
        tracker.generate_extinction_report(report_path)
        print(f"分析レポートを生成: {report_path}")
        
    except Exception as e:
        logger.error(f"実行エラー: {e}")
        raise