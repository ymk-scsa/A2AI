"""
A2AI財務諸表スクレイパー
企業のライフサイクル全体（存続・消滅・新設）に対応した財務諸表データ収集システム

主な機能:
1. EDINET APIからの財務諸表データ取得
2. 企業消滅・倒産データの収集
3. 新設企業・分社企業データの追跡
4. XBRLデータの構造化データ変換
5. 欠損値・データ継続性の管理
"""

import requests
import pandas as pd
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import logging
from pathlib import Path
import json
import time
import re
from urllib.parse import urljoin

from ..utils.data_utils import DataUtils
from ..utils.logging_utils import setup_logger


@dataclass
class CompanyInfo:
    """企業情報を管理するデータクラス"""
    edinetcode: str
    company_name: str
    market_category: str  # "high_share", "declining", "lost"
    market_sector: str    # "robot", "endoscope", etc.
    founding_date: Optional[datetime] = None
    extinction_date: Optional[datetime] = None
    parent_company: Optional[str] = None
    status: str = "active"  # "active", "extinct", "merged", "spun_off"


@dataclass
class FinancialData:
    """財務データを格納するデータクラス"""
    company_code: str
    fiscal_year: int
    report_type: str
    data: Dict
    metadata: Dict


class FinancialScraper:
    """財務諸表データ収集クラス"""
    
    def __init__(self, config_path: str = None):
        """
        初期化
        
        Args:
            config_path: 設定ファイルのパス
        """
        self.logger = setup_logger("financial_scraper")
        self.base_url = "https://disclosure.edinet-fsa.go.jp/api/v2/"
        self.session = requests.Session()
        
        # 設定読み込み
        self.config = self._load_config(config_path)
        
        # レート制限管理
        self.request_count = 0
        self.last_request_time = time.time()
        self.max_requests_per_second = 5
        
        # データキャッシュ
        self.company_cache = {}
        self.document_cache = {}
        
        # XBRLタクソノミマッピング
        self.taxonomy_mapping = self._load_taxonomy_mapping()
        
        self.logger.info("Financial Scraper初期化完了")
    
    def _load_config(self, config_path: str) -> Dict:
        """設定ファイル読み込み"""
        default_config = {
            "api_key": None,
            "max_retries": 3,
            "timeout": 30,
            "data_output_path": "./data/raw/",
            "start_year": 1984,
            "end_year": 2024,
            "target_reports": ["有価証券報告書", "四半期報告書"]
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                default_config.update(config)
        
        return default_config
    
    def _load_taxonomy_mapping(self) -> Dict:
        """XBRLタクソノミマッピング読み込み"""
        # 要因項目に対応するXBRLタグのマッピング
        return {
            # 貸借対照表項目
            "有形固定資産": ["PropertyPlantAndEquipmentNet", "TangibleFixedAssets"],
            "投資有価証券": ["InvestmentSecurities", "InvestmentInSecurities"],
            "売上債権": ["NotesAndAccountsReceivable", "TradeReceivables"],
            "棚卸資産": ["Inventories"],
            "総資産": ["TotalAssets", "Assets"],
            "自己資本": ["TotalEquity", "ShareholdersEquity"],
            
            # 損益計算書項目
            "売上高": ["Revenue", "NetSales", "OperatingRevenue"],
            "売上原価": ["CostOfSales", "CostOfRevenue"],
            "販売費及び一般管理費": ["SellingGeneralAndAdministrativeExpenses", "SGA"],
            "研究開発費": ["ResearchAndDevelopmentExpenses", "RAndD"],
            "営業利益": ["OperatingIncome", "OperatingProfit"],
            "当期純利益": ["NetIncome", "ProfitLoss"],
            
            # キャッシュフロー項目
            "設備投資額": ["PurchaseOfPropertyPlantAndEquipment", "CapitalExpenditure"],
            "営業キャッシュフロー": ["OperatingCashFlow", "CashFlowFromOperatingActivities"],
            
            # 注記情報
            "従業員数": ["NumberOfEmployees", "EmployeeCount"],
            "平均年間給与": ["AverageAnnualSalary", "AverageWage"],
            "海外売上高比率": ["OverseasSalesRatio", "ForeignSalesRatio"],
            "セグメント情報": ["SegmentInformation", "BusinessSegments"]
        }
    
    def _rate_limit_wait(self):
        """レート制限対応の待機処理"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < (1.0 / self.max_requests_per_second):
            wait_time = (1.0 / self.max_requests_per_second) - time_since_last
            time.sleep(wait_time)
        
        self.last_request_time = time.time()
        self.request_count += 1
    
    def _make_request(self, endpoint: str, params: Dict = None) -> requests.Response:
        """API リクエスト実行"""
        self._rate_limit_wait()
        
        url = urljoin(self.base_url, endpoint)
        headers = {}
        
        if self.config.get("api_key"):
            headers["X-API-Key"] = self.config["api_key"]
        
        for attempt in range(self.config["max_retries"]):
            try:
                response = self.session.get(
                    url, 
                    params=params, 
                    headers=headers,
                    timeout=self.config["timeout"]
                )
                
                if response.status_code == 200:
                    return response
                elif response.status_code == 429:  # Rate limit
                    wait_time = 2 ** attempt
                    self.logger.warning(f"Rate limit hit, waiting {wait_time}s")
                    time.sleep(wait_time)
                else:
                    self.logger.error(f"API Error: {response.status_code}")
                    
            except requests.exceptions.RequestException as e:
                self.logger.error(f"Request failed (attempt {attempt + 1}): {e}")
                if attempt < self.config["max_retries"] - 1:
                    time.sleep(2 ** attempt)
        
        raise Exception(f"Failed to make request after {self.config['max_retries']} attempts")
    
    def get_company_list(self, market_category: str = None) -> List[CompanyInfo]:
        """
        企業リスト取得
        
        Args:
            market_category: 市場カテゴリフィルタ
            
        Returns:
            企業情報リスト
        """
        try:
            response = self._make_request("documents.json", {"type": 1})  # 企業一覧
            company_data = response.json()
            
            companies = []
            for item in company_data.get("results", []):
                company = CompanyInfo(
                    edinetcode=item.get("edinetCode", ""),
                    company_name=item.get("filerName", ""),
                    market_category=self._determine_market_category(item.get("filerName", "")),
                    market_sector=self._determine_market_sector(item.get("filerName", ""))
                )
                
                if market_category is None or company.market_category == market_category:
                    companies.append(company)
            
            self.logger.info(f"取得企業数: {len(companies)}")
            return companies
            
        except Exception as e:
            self.logger.error(f"企業リスト取得エラー: {e}")
            raise
    
    def _determine_market_category(self, company_name: str) -> str:
        """企業名から市場カテゴリを判定"""
        # 添付された企業リストに基づく分類ロジック
        high_share_companies = [
            "ファナック", "安川電機", "川崎重工業", "村田製作所", "TDK", 
            "キーエンス", "島津製作所", "オリンパス", "DMG森精機"
        ]
        
        declining_companies = [
            "トヨタ自動車", "日産自動車", "パナソニック", "日本製鉄", 
            "JFE", "シャープ", "ソニー"
        ]
        
        for company in high_share_companies:
            if company in company_name:
                return "high_share"
        
        for company in declining_companies:
            if company in company_name:
                return "declining"
        
        return "lost"  # デフォルトは失失市場
    
    def _determine_market_sector(self, company_name: str) -> str:
        """企業名から市場セクタを判定"""
        sector_mapping = {
            "robot": ["ファナック", "安川電機", "川崎重工業", "不二越", "デンソーウェーブ"],
            "endoscope": ["オリンパス", "HOYA", "富士フイルム", "キヤノン"],
            "machine_tool": ["DMG森精機", "ヤマザキマザック", "オークマ", "牧野"],
            "electronic_materials": ["村田製作所", "TDK", "京セラ", "太陽誘電"],
            "precision_measurement": ["キーエンス", "島津製作所", "堀場製作所", "東京精密"]
        }
        
        for sector, companies in sector_mapping.items():
            for company in companies:
                if company in company_name:
                    return sector
        
        return "other"
    
    def get_document_list(self, company_code: str, start_date: str, end_date: str) -> List[Dict]:
        """
        指定企業の文書リスト取得
        
        Args:
            company_code: 企業コード
            start_date: 開始日 (YYYY-MM-DD)
            end_date: 終了日 (YYYY-MM-DD)
            
        Returns:
            文書情報リスト
        """
        try:
            params = {
                "date": start_date,
                "Subscription-Key": self.config.get("api_key")
            }
            
            documents = []
            current_date = datetime.strptime(start_date, "%Y-%m-%d")
            end_date_dt = datetime.strptime(end_date, "%Y-%m-%d")
            
            while current_date <= end_date_dt:
                date_str = current_date.strftime("%Y-%m-%d")
                params["date"] = date_str
                
                response = self._make_request("documents.json", params)
                data = response.json()
                
                for doc in data.get("results", []):
                    if doc.get("edinetCode") == company_code:
                        if any(report_type in doc.get("docDescription", "") 
                                for report_type in self.config["target_reports"]):
                            documents.append({
                                "docID": doc.get("docID"),
                                "edinetCode": doc.get("edinetCode"),
                                "docDescription": doc.get("docDescription"),
                                "submitDateTime": doc.get("submitDateTime"),
                                "periodStart": doc.get("periodStart"),
                                "periodEnd": doc.get("periodEnd")
                            })
                
                current_date += timedelta(days=1)
                
                # 進捗表示
                if current_date.day == 1:
                    self.logger.info(f"Progress: {date_str}")
            
            self.logger.info(f"企業 {company_code}: {len(documents)}件の文書を発見")
            return documents
            
        except Exception as e:
            self.logger.error(f"文書リスト取得エラー: {e}")
            return []
    
    def get_xbrl_data(self, doc_id: str) -> Dict:
        """
        XBRL データ取得・解析
        
        Args:
            doc_id: 文書ID
            
        Returns:
            解析済み財務データ
        """
        try:
            # キャッシュチェック
            if doc_id in self.document_cache:
                return self.document_cache[doc_id]
            
            # XBRL文書取得
            response = self._make_request(f"documents/{doc_id}", {"type": 5})  # XBRL
            
            # XML解析
            root = ET.fromstring(response.content)
            
            # 財務データ抽出
            financial_data = {}
            
            for metric_name, xbrl_tags in self.taxonomy_mapping.items():
                for tag in xbrl_tags:
                    elements = root.findall(f".//{tag}")
                    if elements:
                        # 最新の値を取得
                        for element in elements:
                            if element.text and element.text.strip():
                                try:
                                    value = float(element.text.replace(',', ''))
                                    financial_data[metric_name] = value
                                    break
                                except ValueError:
                                    financial_data[metric_name] = element.text.strip()
                                    break
                        break
            
            # メタデータ抽出
            metadata = self._extract_metadata(root)
            
            result = {
                "financial_data": financial_data,
                "metadata": metadata,
                "doc_id": doc_id
            }
            
            # キャッシュに保存
            self.document_cache[doc_id] = result
            
            return result
            
        except Exception as e:
            self.logger.error(f"XBRL データ取得エラー (doc_id: {doc_id}): {e}")
            return {}
    
    def _extract_metadata(self, root: ET.Element) -> Dict:
        """XBRLからメタデータ抽出"""
        metadata = {
            "company_name": "",
            "fiscal_year": None,
            "report_type": "",
            "accounting_standard": "",
            "currency": "JPY"
        }
        
        try:
            # 企業名
            company_elements = root.findall(".//CompanyName")
            if company_elements:
                metadata["company_name"] = company_elements[0].text
            
            # 会計年度
            year_elements = root.findall(".//FiscalYear")
            if year_elements:
                metadata["fiscal_year"] = int(year_elements[0].text)
            
            # レポートタイプ
            report_elements = root.findall(".//DocumentName")
            if report_elements:
                metadata["report_type"] = report_elements[0].text
            
            # 会計基準
            standard_elements = root.findall(".//AccountingStandard")
            if standard_elements:
                metadata["accounting_standard"] = standard_elements[0].text
                
        except Exception as e:
            self.logger.warning(f"メタデータ抽出エラー: {e}")
        
        return metadata
    
    def collect_company_data(self, company: CompanyInfo, 
                            start_year: int = None, 
                            end_year: int = None) -> List[FinancialData]:
        """
        単一企業の財務データ収集
        
        Args:
            company: 企業情報
            start_year: 開始年
            end_year: 終了年
            
        Returns:
            財務データリスト
        """
        start_year = start_year or self.config["start_year"]
        end_year = end_year or self.config["end_year"]
        
        # 企業のライフサイクルを考慮した期間調整
        if company.founding_date:
            start_year = max(start_year, company.founding_date.year)
        
        if company.extinction_date:
            end_year = min(end_year, company.extinction_date.year)
        
        self.logger.info(f"データ収集開始: {company.company_name} ({start_year}-{end_year})")
        
        financial_data_list = []
        
        for year in range(start_year, end_year + 1):
            try:
                # 年度の期間設定
                start_date = f"{year}-01-01"
                end_date = f"{year}-12-31"
                
                # 文書リスト取得
                documents = self.get_document_list(
                    company.edinetcode, start_date, end_date
                )
                
                # 各文書からデータ取得
                for doc in documents:
                    xbrl_data = self.get_xbrl_data(doc["docID"])
                    
                    if xbrl_data:
                        financial_data = FinancialData(
                            company_code=company.edinetcode,
                            fiscal_year=year,
                            report_type=doc["docDescription"],
                            data=xbrl_data["financial_data"],
                            metadata=xbrl_data["metadata"]
                        )
                        financial_data_list.append(financial_data)
                
            except Exception as e:
                self.logger.error(f"年度 {year} データ収集エラー: {e}")
                continue
        
        self.logger.info(f"データ収集完了: {company.company_name} - {len(financial_data_list)}件")
        return financial_data_list
    
    def collect_all_companies_data(self, 
                                    market_categories: List[str] = None) -> Dict[str, List[FinancialData]]:
        """
        全企業データ収集
        
        Args:
            market_categories: 対象市場カテゴリリスト
            
        Returns:
            企業別財務データ辞書
        """
        if market_categories is None:
            market_categories = ["high_share", "declining", "lost"]
        
        all_data = {}
        
        for category in market_categories:
            self.logger.info(f"市場カテゴリ '{category}' のデータ収集開始")
            
            companies = self.get_company_list(category)
            
            for company in companies:
                try:
                    company_data = self.collect_company_data(company)
                    all_data[company.edinetcode] = company_data
                    
                    # 進捗保存
                    self._save_progress(company.edinetcode, company_data)
                    
                except Exception as e:
                    self.logger.error(f"企業 {company.company_name} データ収集エラー: {e}")
                    continue
        
        return all_data
    
    def _save_progress(self, company_code: str, data: List[FinancialData]):
        """進捗データ保存"""
        try:
            output_dir = Path(self.config["data_output_path"]) / "financial_data"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # データをDataFrame化
            df_data = []
            for financial_data in data:
                row = {
                    "company_code": financial_data.company_code,
                    "fiscal_year": financial_data.fiscal_year,
                    "report_type": financial_data.report_type
                }
                row.update(financial_data.data)
                row.update({f"meta_{k}": v for k, v in financial_data.metadata.items()})
                df_data.append(row)
            
            if df_data:
                df = pd.DataFrame(df_data)
                output_file = output_dir / f"{company_code}_financial_data.csv"
                df.to_csv(output_file, index=False, encoding='utf-8')
                
                self.logger.info(f"データ保存完了: {output_file}")
                
        except Exception as e:
            self.logger.error(f"データ保存エラー: {e}")
    
    def get_extinct_companies_data(self) -> Dict[str, Dict]:
        """
        消滅企業の最終財務データ取得
        
        Returns:
            消滅企業の最終財務状態データ
        """
        extinct_companies = {
            "三洋電機": {"extinction_year": 2012, "cause": "パナソニック吸収"},
            "アイワ": {"extinction_year": 2002, "cause": "ソニー統合"},
            "FCNT": {"extinction_year": 2023, "cause": "経営破綻"},
            "日本電気ホームエレクトロニクス": {"extinction_year": 2015, "cause": "NEC再編"}
        }
        
        extinct_data = {}
        
        for company_name, info in extinct_companies.items():
            try:
                # 消滅前数年間のデータ取得
                end_year = info["extinction_year"]
                start_year = max(1984, end_year - 5)  # 消滅前5年間
                
                # 企業コード検索（企業名から推定）
                company_code = self._find_company_code(company_name)
                
                if company_code:
                    # 消滅までのデータ収集
                    company = CompanyInfo(
                        edinetcode=company_code,
                        company_name=company_name,
                        market_category="lost",
                        market_sector="extinct",
                        extinction_date=datetime(end_year, 12, 31),
                        status="extinct"
                    )
                    
                    financial_data = self.collect_company_data(
                        company, start_year, end_year
                    )
                    
                    extinct_data[company_name] = {
                        "financial_data": financial_data,
                        "extinction_info": info
                    }
                    
            except Exception as e:
                self.logger.error(f"消滅企業 {company_name} データ収集エラー: {e}")
        
        return extinct_data
    
    def _find_company_code(self, company_name: str) -> Optional[str]:
        """企業名からEDINETコード検索"""
        try:
            companies = self.get_company_list()
            
            for company in companies:
                if company_name in company.company_name or company.company_name in company_name:
                    return company.edinetcode
            
            return None
            
        except Exception as e:
            self.logger.error(f"企業コード検索エラー: {e}")
            return None
    
    def validate_data_continuity(self, company_data: List[FinancialData]) -> Dict:
        """
        データ継続性検証
        
        Args:
            company_data: 企業の財務データリスト
            
        Returns:
            検証結果
        """
        validation_result = {
            "is_continuous": True,
            "missing_years": [],
            "data_gaps": [],
            "quality_score": 1.0
        }
        
        if not company_data:
            validation_result["is_continuous"] = False
            validation_result["quality_score"] = 0.0
            return validation_result
        
        # 年度の連続性チェック
        years = sorted([data.fiscal_year for data in company_data])
        expected_years = list(range(min(years), max(years) + 1))
        missing_years = set(expected_years) - set(years)
        
        validation_result["missing_years"] = list(missing_years)
        validation_result["is_continuous"] = len(missing_years) == 0
        
        # データ品質スコア計算
        total_expected = len(expected_years)
        actual_count = len(years)
        validation_result["quality_score"] = actual_count / total_expected if total_expected > 0 else 0.0
        
        return validation_result


# 使用例とテスト用関数
def main():
    """メイン関数：使用例"""
    scraper = FinancialScraper()
    
    # 高シェア市場企業のデータ収集例
    try:
        # 特定企業のデータ収集
        company = CompanyInfo(
            edinetcode="E01784",  # ファナック
            company_name="ファナック株式会社",
            market_category="high_share",
            market_sector="robot"
        )
        
        financial_data = scraper.collect_company_data(company, 2020, 2024)
        
        print(f"収集データ数: {len(financial_data)}")
        for data in financial_data[:3]:  # 最初の3件を表示
            print(f"年度: {data.fiscal_year}")
            print(f"データ項目数: {len(data.data)}")
            print("---")
            
    except Exception as e:
        print(f"エラー: {e}")


if __name__ == "__main__":
    main()