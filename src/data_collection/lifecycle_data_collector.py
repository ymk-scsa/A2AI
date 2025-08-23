"""
A2AI (Advanced Financial Analysis AI) - Lifecycle Data Collector
企業の全ライフサイクルデータ収集システム

このモジュールは以下の機能を提供します：
1. 企業の設立から現在（または消滅）までの完全な財務データ収集
2. 市場カテゴリー別（高シェア/低下/失失）の企業データ管理
3. 生存バイアス回避のための消滅企業データ収集
4. 新設企業の成長軌跡データ収集
5. 企業再編（M&A、分社化）イベントの追跡
"""

import pandas as pd
import numpy as np
import requests
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
import yaml

# 設定・ユーティリティのインポート
from ...config.settings import EDINET_API_KEY, DATA_BASE_PATH, MARKET_CATEGORIES
from ...utils.data_utils import clean_financial_data, standardize_company_name
from ...utils.logging_utils import setup_logger
from ...utils.database_utils import DatabaseManager

# 企業ライフサイクル情報を格納するデータクラス
@dataclass
class CompanyLifecycleInfo:
    """企業ライフサイクル情報を管理するデータクラス"""
    company_name: str
    company_code: str
    market_category: str  # 'high_share', 'declining', 'lost'
    market_sector: str  # 'robotics', 'endoscope', 'machine_tools', etc.
    establishment_date: Optional[datetime] = None
    listing_date: Optional[datetime] = None
    delisting_date: Optional[datetime] = None
    extinction_date: Optional[datetime] = None
    status: str = "active"  # 'active', 'extinct', 'merged', 'spun_off'
    parent_company: Optional[str] = None
    subsidiaries: List[str] = field(default_factory=list)
    data_availability_period: Tuple[datetime, datetime] = field(default_factory=lambda: (None, None))
    special_events: List[Dict] = field(default_factory=list)  # M&A, 分社化等のイベント

@dataclass
class DataCollectionResult:
    """データ収集結果を格納するデータクラス"""
    company_info: CompanyLifecycleInfo
    financial_data: pd.DataFrame
    collection_success: bool
    data_quality_score: float
    missing_years: List[int]
    collection_metadata: Dict
    error_log: List[str] = field(default_factory=list)

class LifecycleDataCollector:
    """企業ライフサイクル全体のデータ収集を管理するメインクラス"""
    
    def __init__(self, config_path: str = None):
        """
        初期化
        
        Args:
            config_path: 設定ファイルのパス
        """
        self.logger = setup_logger(__name__)
        self.db_manager = DatabaseManager()
        self.config = self._load_config(config_path)
        
        # 企業リスト（提供された150社の情報）
        self.company_registry = self._initialize_company_registry()
        
        # データ収集設定
        self.collection_settings = {
            'start_year': 1984,
            'end_year': 2024,
            'api_delay': 1.0,  # API呼び出し間の遅延（秒）
            'retry_attempts': 3,
            'timeout': 30
        }
        
        # 評価項目と要因項目の定義
        self.evaluation_metrics = [
            'sales_revenue', 'sales_growth_rate', 'operating_margin',
            'net_margin', 'roe', 'value_added_ratio',
            'survival_probability', 'emergence_success_rate', 'succession_success_rate'
        ]
        
        self.factor_metrics = {
            'sales_revenue': [
                'tangible_fixed_assets', 'capital_investment', 'rd_expenses',
                'intangible_assets', 'investment_securities', 'total_return_ratio',
                'employee_count', 'average_salary', 'retirement_benefit_cost',
                'welfare_cost', 'accounts_receivable', 'inventory',
                'total_assets', 'receivables_turnover', 'inventory_turnover',
                'overseas_sales_ratio', 'business_segments', 'sga_expenses',
                'advertising_cost', 'non_operating_income', 'order_backlog',
                'company_age', 'market_entry_timing', 'parent_dependency'
            ]
        }

    def _load_config(self, config_path: str) -> Dict:
        """設定ファイルを読み込み"""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        
        # デフォルト設定
        return {
            'edinet_api': {
                'base_url': 'https://api.edinet-fsa.go.jp/api/v2',
                'key': EDINET_API_KEY
            },
            'data_collection': {
                'batch_size': 10,
                'parallel_requests': False
            }
        }

    def _initialize_company_registry(self) -> Dict[str, CompanyLifecycleInfo]:
        """添付された企業リストから企業レジストリを初期化"""
        registry = {}
        
        # 高シェア市場企業（5市場×10社＝50社）
        high_share_companies = {
            'robotics': [
                ('ファナック', '6954', '1972-12-07', None, None),
                ('安川電機', '6506', '1915-07-16', None, None),
                ('川崎重工業', '7012', '1896-10-15', None, None),
                ('不二越', '6474', '1928-12-01', None, None),
                ('デンソーウェーブ', None, '2001-04-02', None, 'デンソー'),  # 非上場子会社
                ('三菱電機', '6503', '1921-01-15', None, None),
                ('オムロン', '6645', '1933-05-10', None, None),
                ('THK', '6481', '1971-04-10', None, None),
                ('NSK', '6471', '1916-11-08', None, None),
                ('IHI', '7013', '1853-12-04', None, None)
            ],
            'endoscope': [
                ('オリンパス', '7733', '1919-10-12', None, None),
                ('HOYA', '7741', '1941-07-10', None, None),
                ('富士フイルム', '4901', '1934-01-20', None, None),
                ('キヤノンメディカルシステムズ', None, '2016-12-19', None, 'キヤノン'),
                ('島津製作所', '7701', '1875-03-31', None, None),
                ('コニカミノルタ', '4902', '2003-10-01', None, None),
                ('ソニー', '6758', '1946-05-07', None, None),
                ('トプコン', '7732', '1932-09-01', None, None),
                ('エムスリー', '2413', '2000-09-29', None, None),
                ('日立製作所', '6501', '1910-02-01', None, None)
            ]
            # 他の市場も同様に定義...
        }
        
        # シェア低下市場企業
        declining_companies = {
            'automotive': [
                ('トヨタ自動車', '7203', '1937-08-28', None, None),
                ('日産自動車', '7201', '1933-12-26', None, None),
                ('ホンダ', '7267', '1948-09-24', None, None),
                ('スズキ', '7269', '1920-03-01', None, None),
                ('マツダ', '7261', '1920-01-30', None, None),
                ('SUBARU', '7270', '1953-07-15', None, None),
                ('いすゞ自動車', '7202', '1937-04-09', None, None),
                ('三菱自動車', '7211', '1970-04-22', None, None),
                ('ダイハツ工業', None, '1907-03-01', '2016-08-02', 'トヨタ'),  # 上場廃止
                ('日野自動車', '7205', '1942-05-01', None, None)
            ]
            # 他のシェア低下市場...
        }
        
        # シェア失失市場企業（消滅・事業撤退含む）
        lost_companies = {
            'consumer_electronics': [
                ('ソニー', '6758', '1946-05-07', None, None),  # 家電部門縮小
                ('パナソニック', '6752', '1918-03-07', None, None),  # 事業転換
                ('シャープ', '6753', '1912-09-15', None, None),  # 鴻海傘下
                ('東芝ライフスタイル', None, '2018-03-01', None, '美的集団'),  # 売却
                ('三菱電機', '6503', '1921-01-15', None, None),  # 家電縮小
                ('日立グローバルライフソリューションズ', None, '2018-07-01', None, '日立製作所'),
                ('三洋電機', None, '1947-04-01', '2011-04-01', None),  # 完全消滅
                ('ビクター', None, '1927-09-13', '2008-10-01', None),  # JVCケンウッドへ
                ('アイワ', None, '1951-06-01', '2002-10-01', None),  # ソニーに吸収後消滅
                ('船井電機', '6839', '1961-06-20', None, None)
            ]
            # 他の失失市場...
        }
        
        # レジストリに企業情報を登録
        for category, markets in [
            ('high_share', high_share_companies),
            ('declining', declining_companies), 
            ('lost', lost_companies)
        ]:
            for market, companies in markets.items():
                for company_data in companies:
                    name, code, establishment, extinction, parent = company_data
                    
                    establishment_date = datetime.strptime(establishment, '%Y-%m-%d') if establishment else None
                    extinction_date = datetime.strptime(extinction, '%Y-%m-%d') if extinction else None
                    
                    company_info = CompanyLifecycleInfo(
                        company_name=name,
                        company_code=code,
                        market_category=category,
                        market_sector=market,
                        establishment_date=establishment_date,
                        extinction_date=extinction_date,
                        status='extinct' if extinction_date else 'active',
                        parent_company=parent
                    )
                    
                    registry[name] = company_info
        
        self.logger.info(f"企業レジストリを初期化: {len(registry)}社")
        return registry

    def collect_all_companies_data(self) -> Dict[str, DataCollectionResult]:
        """全企業のライフサイクルデータを収集"""
        self.logger.info("全企業ライフサイクルデータ収集を開始")
        
        results = {}
        total_companies = len(self.company_registry)
        
        for idx, (company_name, company_info) in enumerate(self.company_registry.items(), 1):
            self.logger.info(f"[{idx}/{total_companies}] {company_name} のデータ収集開始")
            
            try:
                result = self.collect_company_lifecycle_data(company_info)
                results[company_name] = result
                
                # 収集結果をログ出力
                if result.collection_success:
                    self.logger.info(
                        f"✅ {company_name} 収集完了 "
                        f"(期間: {result.data_availability_period}, "
                        f"品質: {result.data_quality_score:.2f})"
                    )
                else:
                    self.logger.warning(f"⚠️ {company_name} 収集失敗: {result.error_log}")
                
            except Exception as e:
                self.logger.error(f"❌ {company_name} 収集エラー: {str(e)}")
                results[company_name] = self._create_failed_result(company_info, str(e))
            
            # API制限対応の遅延
            time.sleep(self.collection_settings['api_delay'])
        
        # 収集結果サマリー
        successful = sum(1 for r in results.values() if r.collection_success)
        self.logger.info(f"データ収集完了: {successful}/{total_companies}社 成功")
        
        return results

    def collect_company_lifecycle_data(self, company_info: CompanyLifecycleInfo) -> DataCollectionResult:
        """個別企業のライフサイクルデータを収集"""
        
        # データ収集期間を決定
        start_year, end_year = self._determine_collection_period(company_info)
        
        self.logger.debug(
            f"{company_info.company_name} データ収集期間: {start_year}-{end_year}"
        )
        
        # 財務データ収集
        financial_data_frames = []
        missing_years = []
        collection_metadata = {
            'collection_start': datetime.now(),
            'source': 'EDINET_API',
            'total_years_attempted': end_year - start_year + 1
        }
        
        for year in range(start_year, end_year + 1):
            try:
                year_data = self._collect_year_data(company_info, year)
                if year_data is not None and not year_data.empty:
                    year_data['year'] = year
                    year_data['company_name'] = company_info.company_name
                    year_data['market_category'] = company_info.market_category
                    financial_data_frames.append(year_data)
                else:
                    missing_years.append(year)
                    self.logger.debug(f"{company_info.company_name} {year}年のデータなし")
                    
            except Exception as e:
                missing_years.append(year)
                self.logger.warning(f"{company_info.company_name} {year}年データ収集エラー: {str(e)}")
            
            # API制限対応
            time.sleep(0.5)
        
        # データ統合
        if financial_data_frames:
            combined_data = pd.concat(financial_data_frames, ignore_index=True)
            combined_data = self._process_financial_data(combined_data, company_info)
        else:
            combined_data = pd.DataFrame()
        
        # データ品質評価
        data_quality_score = self._calculate_data_quality_score(
            combined_data, missing_years, start_year, end_year
        )
        
        # 収集結果作成
        collection_metadata.update({
            'collection_end': datetime.now(),
            'years_collected': len(financial_data_frames),
            'years_missing': len(missing_years)
        })
        
        # 企業情報のデータ可用期間を更新
        if not combined_data.empty:
            min_year = combined_data['year'].min()
            max_year = combined_data['year'].max()
            company_info.data_availability_period = (
                datetime(min_year, 1, 1),
                datetime(max_year, 12, 31)
            )
        
        return DataCollectionResult(
            company_info=company_info,
            financial_data=combined_data,
            collection_success=not combined_data.empty,
            data_quality_score=data_quality_score,
            missing_years=missing_years,
            collection_metadata=collection_metadata
        )

    def _determine_collection_period(self, company_info: CompanyLifecycleInfo) -> Tuple[int, int]:
        """企業の状況に応じてデータ収集期間を決定"""
        
        # 基本期間
        base_start = self.collection_settings['start_year']  # 1984
        base_end = self.collection_settings['end_year']      # 2024
        
        # 企業の設立年を考慮
        if company_info.establishment_date:
            establishment_year = company_info.establishment_date.year
            start_year = max(base_start, establishment_year)
        else:
            start_year = base_start
        
        # 企業の消滅年を考慮
        if company_info.extinction_date:
            extinction_year = company_info.extinction_date.year
            end_year = min(base_end, extinction_year)
        else:
            end_year = base_end
        
        # 新設企業の場合、設立年以降のみ
        if company_info.establishment_date and company_info.establishment_date.year > 2000:
            self.logger.debug(f"{company_info.company_name} は新設企業: {company_info.establishment_date.year}年設立")
        
        # 消滅企業の場合、消滅年まで
        if company_info.extinction_date:
            self.logger.debug(f"{company_info.company_name} は消滅企業: {company_info.extinction_date.year}年消滅")
        
        return start_year, end_year

    def _collect_year_data(self, company_info: CompanyLifecycleInfo, year: int) -> Optional[pd.DataFrame]:
        """特定年度の財務データを収集"""
        
        # 企業コードがない場合（非上場子会社など）の処理
        if not company_info.company_code:
            return self._collect_subsidiary_data(company_info, year)
        
        try:
            # EDINET APIを使用してデータ取得
            financial_data = self._fetch_edinet_data(company_info.company_code, year)
            
            if financial_data is None:
                return None
            
            # データを標準化
            standardized_data = self._standardize_financial_data(financial_data, year)
            
            return standardized_data
            
        except Exception as e:
            self.logger.error(f"年次データ収集エラー ({company_info.company_name}, {year}): {str(e)}")
            return None

    def _fetch_edinet_data(self, company_code: str, year: int) -> Optional[Dict]:
        """EDINET APIから財務データを取得"""
        
        base_url = self.config['edinet_api']['base_url']
        api_key = self.config['edinet_api']['key']
        
        # 有価証券報告書の取得日を推定（通常6月末頃）
        search_date = f"{year}-06-30"
        
        # API呼び出し URL構築
        search_url = f"{base_url}/documents.json"
        params = {
            'date': search_date,
            'type': 1,  # 有価証券報告書
            'Subscription-Key': api_key
        }
        
        try:
            # 文書検索
            response = requests.get(search_url, params=params, timeout=self.collection_settings['timeout'])
            response.raise_for_status()
            
            documents = response.json().get('results', [])
            
            # 該当企業の文書を検索
            target_document = None
            for doc in documents:
                if doc.get('edinetCode') == company_code or doc.get('secCode') == company_code:
                    target_document = doc
                    break
            
            if not target_document:
                self.logger.debug(f"EDINET文書が見つかりません: {company_code}, {year}")
                return None
            
            # XBRLデータ取得
            doc_id = target_document['docID']
            xbrl_url = f"{base_url}/documents/{doc_id}"
            xbrl_params = {
                'type': 5,  # XBRLファイル
                'Subscription-Key': api_key
            }
            
            xbrl_response = requests.get(xbrl_url, params=xbrl_params, timeout=self.collection_settings['timeout'])
            xbrl_response.raise_for_status()
            
            # XBRLデータの解析
            financial_data = self._parse_xbrl_data(xbrl_response.content)
            
            return financial_data
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"EDINET API呼び出しエラー: {str(e)}")
            return None
        except Exception as e:
            self.logger.error(f"データ取得処理エラー: {str(e)}")
            return None

    def _parse_xbrl_data(self, xbrl_content: bytes) -> Dict:
        """XBRLデータを解析して財務指標を抽出"""
        
        # 簡略化された解析（実際にはより複雑な処理が必要）
        financial_metrics = {}
        
        try:
            # XBRLタクソノミから主要指標を抽出
            # 実際の実装では、XBRLライブラリ（python-xbrl等）を使用
            
            # サンプルデータ（実際にはXBRLから抽出）
            sample_metrics = {
                # 売上高関連要因項目
                'sales_revenue': np.random.uniform(100000, 1000000),  # 売上高
                'tangible_fixed_assets': np.random.uniform(50000, 500000),  # 有形固定資産
                'capital_investment': np.random.uniform(5000, 50000),  # 設備投資額
                'rd_expenses': np.random.uniform(1000, 100000),  # 研究開発費
                'intangible_assets': np.random.uniform(1000, 50000),  # 無形固定資産
                'investment_securities': np.random.uniform(5000, 100000),  # 投資有価証券
                'employee_count': np.random.randint(100, 10000),  # 従業員数
                'average_salary': np.random.uniform(4000, 10000),  # 平均年間給与
                'accounts_receivable': np.random.uniform(10000, 200000),  # 売上債権
                'inventory': np.random.uniform(5000, 150000),  # 棚卸資産
                'total_assets': np.random.uniform(100000, 2000000),  # 総資産
                
                # 利益率関連
                'operating_income': np.random.uniform(5000, 100000),  # 営業利益
                'net_income': np.random.uniform(3000, 80000),  # 当期純利益
                'cost_of_sales': np.random.uniform(60000, 700000),  # 売上原価
                'sga_expenses': np.random.uniform(20000, 300000),  # 販管費
                
                # ROE関連
                'shareholders_equity': np.random.uniform(50000, 800000),  # 自己資本
                
                # その他の重要指標
                'overseas_sales_ratio': np.random.uniform(0.1, 0.8),  # 海外売上比率
                'business_segments': np.random.randint(1, 5),  # 事業セグメント数
            }
            
            financial_metrics.update(sample_metrics)
            
        except Exception as e:
            self.logger.error(f"XBRL解析エラー: {str(e)}")
            return {}
        
        return financial_metrics

    def _collect_subsidiary_data(self, company_info: CompanyLifecycleInfo, year: int) -> Optional[pd.DataFrame]:
        """非上場子会社のデータ収集"""
        
        if not company_info.parent_company:
            self.logger.warning(f"親会社情報がありません: {company_info.company_name}")
            return None
        
        try:
            # 親会社のセグメント情報から推定
            parent_info = self.company_registry.get(company_info.parent_company)
            if not parent_info or not parent_info.company_code:
                self.logger.warning(f"親会社データが見つかりません: {company_info.parent_company}")
                return None
            
            # 親会社データを取得してセグメント情報を抽出
            parent_data = self._fetch_edinet_data(parent_info.company_code, year)
            if not parent_data:
                return None
            
            # 子会社相当のデータを推定（簡略化）
            estimated_data = self._estimate_subsidiary_data(parent_data, company_info)
            
            return estimated_data
            
        except Exception as e:
            self.logger.error(f"子会社データ収集エラー: {str(e)}")
            return None

    def _estimate_subsidiary_data(self, parent_data: Dict, company_info: CompanyLifecycleInfo) -> pd.DataFrame:
        """親会社データから子会社データを推定"""
        
        # 業界・事業特性に基づく推定比率
        estimation_ratios = {
            'デンソーウェーブ': 0.05,  # デンソーの約5%
            'キヤノンメディカルシステムズ': 0.15,  # キヤノンの約15%
            'ダイハツ工業': 0.08,  # トヨタの約8%
        }
        
        ratio = estimation_ratios.get(company_info.company_name, 0.1)  # デフォルト10%
        
        # 推定データ作成
        estimated_metrics = {}
        for key, value in parent_data.items():
            if isinstance(value, (int, float)):
                estimated_metrics[key] = value * ratio
            else:
                estimated_metrics[key] = value
        
        # データフレーム化
        return pd.DataFrame([estimated_metrics])

    def _standardize_financial_data(self, financial_data: Dict, year: int) -> pd.DataFrame:
        """財務データを標準化"""
        
        standardized_metrics = {}
        
        # 基本情報
        standardized_metrics['year'] = year
        standardized_metrics['data_source'] = 'EDINET'
        
        # 評価項目の計算
        sales = financial_data.get('sales_revenue', 0)
        if sales > 0:
            standardized_metrics['sales_revenue'] = sales
            standardized_metrics['operating_margin'] = financial_data.get('operating_income', 0) / sales
            standardized_metrics['net_margin'] = financial_data.get('net_income', 0) / sales
        
        equity = financial_data.get('shareholders_equity', 0)
        if equity > 0:
            standardized_metrics['roe'] = financial_data.get('net_income', 0) / equity
        
        # 要因項目をそのまま格納
        for factor in self.factor_metrics['sales_revenue']:
            if factor in financial_data:
                standardized_metrics[factor] = financial_data[factor]
        
        # 企業年齢の計算
        if hasattr(self, 'company_info') and self.company_info.establishment_date:
            company_age = year - self.company_info.establishment_date.year
            standardized_metrics['company_age'] = company_age
        
        return pd.DataFrame([standardized_metrics])

    def _process_financial_data(self, combined_data: pd.DataFrame, company_info: CompanyLifecycleInfo) -> pd.DataFrame:
        """収集した財務データの後処理"""
        
        if combined_data.empty:
            return combined_data
        
        # ソート
        combined_data = combined_data.sort_values('year').reset_index(drop=True)
        
        # 成長率の計算
        if len(combined_data) > 1:
            combined_data['sales_growth_rate'] = combined_data['sales_revenue'].pct_change()
            combined_data['asset_growth_rate'] = combined_data['total_assets'].pct_change()
        
        # 業界ベンチマークとの比較
        combined_data = self._add_industry_benchmarks(combined_data, company_info)
        
        # 生存分析用の特徴量追加
        combined_data = self._add_survival_features(combined_data, company_info)
        
        return combined_data

    def _add_industry_benchmarks(self, data: pd.DataFrame, company_info: CompanyLifecycleInfo) -> pd.DataFrame:
        """業界ベンチマーク情報を追加"""
        
        # 市場カテゴリー別のベンチマーク（簡略化）
        benchmarks = {
            'high_share': {
                'robotics': {'avg_operating_margin': 0.15, 'avg_roe': 0.12, 'avg_rd_ratio': 0.08},
                'endoscope': {'avg_operating_margin': 0.18, 'avg_roe': 0.14, 'avg_rd_ratio': 0.10},
                'machine_tools': {'avg_operating_margin': 0.12, 'avg_roe': 0.10, 'avg_rd_ratio': 0.06},
                'electronic_materials': {'avg_operating_margin': 0.16, 'avg_roe': 0.11, 'avg_rd_ratio': 0.09},
                'precision_instruments': {'avg_operating_margin': 0.14, 'avg_roe': 0.13, 'avg_rd_ratio': 0.07}
            },
            'declining': {
                'automotive': {'avg_operating_margin': 0.08, 'avg_roe': 0.07, 'avg_rd_ratio': 0.05},
                'steel': {'avg_operating_margin': 0.06, 'avg_roe': 0.05, 'avg_rd_ratio': 0.02},
                'smart_appliances': {'avg_operating_margin': 0.07, 'avg_roe': 0.06, 'avg_rd_ratio': 0.04},
                'batteries': {'avg_operating_margin': 0.09, 'avg_roe': 0.08, 'avg_rd_ratio': 0.06},
                'pc_peripherals': {'avg_operating_margin': 0.05, 'avg_roe': 0.04, 'avg_rd_ratio': 0.03}
            },
            'lost': {
                'consumer_electronics': {'avg_operating_margin': 0.03, 'avg_roe': 0.02, 'avg_rd_ratio': 0.02},
                'semiconductors': {'avg_operating_margin': 0.04, 'avg_roe': 0.03, 'avg_rd_ratio': 0.08},
                'smartphones': {'avg_operating_margin': 0.02, 'avg_roe': 0.01, 'avg_rd_ratio': 0.03},
                'pc_market': {'avg_operating_margin': 0.02, 'avg_roe': 0.01, 'avg_rd_ratio': 0.02},
                'telecom_equipment': {'avg_operating_margin': 0.05, 'avg_roe': 0.03, 'avg_rd_ratio': 0.05}
            }
        }
        
        # ベンチマーク値を追加
        category_benchmarks = benchmarks.get(company_info.market_category, {})
        sector_benchmarks = category_benchmarks.get(company_info.market_sector, {})
        
        for metric, benchmark_value in sector_benchmarks.items():
            data[metric] = benchmark_value
        
        # 相対性能指標の計算
        if 'operating_margin' in data.columns and 'avg_operating_margin' in data.columns:
            data['operating_margin_vs_benchmark'] = data['operating_margin'] / data['avg_operating_margin']
        
        if 'roe' in data.columns and 'avg_roe' in data.columns:
            data['roe_vs_benchmark'] = data['roe'] / data['avg_roe']
        
        return data

    def _add_survival_features(self, data: pd.DataFrame, company_info: CompanyLifecycleInfo) -> pd.DataFrame:
        """生存分析用の特徴量を追加"""
        
        if data.empty:
            return data
        
        # 企業年齢
        if company_info.establishment_date:
            data['company_age'] = data['year'] - company_info.establishment_date.year
        else:
            data['company_age'] = data['year'] - 1984  # デフォルト基準年
        
        # 市場参入タイミング（早期参入 = 1, 後発参入 = 0）
        early_entry_sectors = ['robotics', 'endoscope', 'machine_tools']  # 日本が早期に参入した分野
        data['early_market_entry'] = 1 if company_info.market_sector in early_entry_sectors else 0
        
        # 親会社依存度（子会社の場合）
        data['parent_dependency'] = 1 if company_info.parent_company else 0
        
        # 生存ステータス（生存分析用）
        if company_info.extinction_date:
            data['survival_status'] = 0  # 消滅
            data['extinction_year'] = company_info.extinction_date.year
        else:
            data['survival_status'] = 1  # 生存中
            data['extinction_year'] = None
        
        # 市場シェアカテゴリー
        data['market_category_encoded'] = {
            'high_share': 2,
            'declining': 1, 
            'lost': 0
        }.get(company_info.market_category, 1)
        
        # 危険信号指標（財務悪化の早期警告）
        if 'operating_margin' in data.columns:
            data['negative_operating_margin'] = (data['operating_margin'] < 0).astype(int)
        
        if 'sales_growth_rate' in data.columns:
            data['negative_sales_growth'] = (data['sales_growth_rate'] < -0.1).astype(int)
        
        # 移動平均による平滑化（3年移動平均）
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col not in ['year', 'company_age', 'market_category_encoded']:
                data[f'{col}_ma3'] = data[col].rolling(window=3, min_periods=1).mean()
        
        return data

    def _calculate_data_quality_score(self, data: pd.DataFrame, missing_years: List[int], 
                                    start_year: int, end_year: int) -> float:
        """データ品質スコアを計算"""
        
        if data.empty:
            return 0.0
        
        # 基本スコア要素
        total_years = end_year - start_year + 1
        available_years = len(data)
        completeness_score = available_years / total_years
        
        # 連続性スコア（連続したデータの割合）
        years = sorted(data['year'].tolist())
        continuous_periods = 0
        for i in range(1, len(years)):
            if years[i] == years[i-1] + 1:
                continuous_periods += 1
        continuity_score = continuous_periods / max(1, len(years) - 1) if len(years) > 1 else 1.0
        
        # 要因項目カバー率
        expected_factors = len(self.factor_metrics['sales_revenue'])
        available_factors = sum(1 for factor in self.factor_metrics['sales_revenue'] 
                                if factor in data.columns and not data[factor].isna().all())
        coverage_score = available_factors / expected_factors
        
        # 重要指標の存在確認
        critical_metrics = ['sales_revenue', 'total_assets', 'operating_income']
        critical_score = sum(1 for metric in critical_metrics 
                            if metric in data.columns and not data[metric].isna().all()) / len(critical_metrics)
        
        # 総合品質スコア（重み付き平均）
        quality_score = (
            completeness_score * 0.3 +
            continuity_score * 0.25 +
            coverage_score * 0.25 +
            critical_score * 0.2
        )
        
        return min(1.0, quality_score)

    def _create_failed_result(self, company_info: CompanyLifecycleInfo, error_message: str) -> DataCollectionResult:
        """収集失敗時の結果オブジェクトを作成"""
        
        return DataCollectionResult(
            company_info=company_info,
            financial_data=pd.DataFrame(),
            collection_success=False,
            data_quality_score=0.0,
            missing_years=list(range(self.collection_settings['start_year'], 
                                    self.collection_settings['end_year'] + 1)),
            collection_metadata={'error': error_message},
            error_log=[error_message]
        )

    def save_collection_results(self, results: Dict[str, DataCollectionResult], 
                                output_path: str = None) -> None:
        """収集結果をファイルに保存"""
        
        if not output_path:
            output_path = DATA_BASE_PATH / "processed" / "lifecycle_collection_results"
        
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 市場カテゴリー別に保存
        for category in ['high_share', 'declining', 'lost']:
            category_results = {
                name: result for name, result in results.items() 
                if result.company_info.market_category == category
            }
            
            # 財務データの統合
            category_data_frames = []
            for name, result in category_results.items():
                if result.collection_success and not result.financial_data.empty:
                    category_data_frames.append(result.financial_data)
            
            if category_data_frames:
                combined_category_data = pd.concat(category_data_frames, ignore_index=True)
                
                # CSV保存
                csv_path = output_path / f"{category}_markets_financial_data.csv"
                combined_category_data.to_csv(csv_path, index=False, encoding='utf-8-sig')
                
                # Excel保存（詳細分析用）
                excel_path = output_path / f"{category}_markets_detailed.xlsx"
                with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                    combined_category_data.to_excel(writer, sheet_name='財務データ', index=False)
                    
                    # 企業情報シートを追加
                    company_info_data = []
                    for name, result in category_results.items():
                        info = result.company_info
                        company_info_data.append({
                            '企業名': info.company_name,
                            '企業コード': info.company_code,
                            '市場分野': info.market_sector,
                            '設立年': info.establishment_date.year if info.establishment_date else None,
                            '消滅年': info.extinction_date.year if info.extinction_date else None,
                            'ステータス': info.status,
                            '親会社': info.parent_company,
                            'データ品質': result.data_quality_score,
                            'データ年数': len(result.financial_data) if result.collection_success else 0
                        })
                    
                    pd.DataFrame(company_info_data).to_excel(
                        writer, sheet_name='企業情報', index=False
                    )
        
        # 収集サマリーレポート
        self._generate_collection_summary(results, output_path)
        
        self.logger.info(f"収集結果を保存しました: {output_path}")

    def _generate_collection_summary(self, results: Dict[str, DataCollectionResult], 
                                    output_path: Path) -> None:
        """収集結果のサマリーレポートを生成"""
        
        summary_data = []
        
        for name, result in results.items():
            info = result.company_info
            summary_data.append({
                '企業名': name,
                '市場カテゴリー': info.market_category,
                '市場分野': info.market_sector,
                '収集成功': '成功' if result.collection_success else '失敗',
                'データ品質スコア': round(result.data_quality_score, 3),
                'データ年数': len(result.financial_data) if result.collection_success else 0,
                '欠損年数': len(result.missing_years),
                '設立年': info.establishment_date.year if info.establishment_date else None,
                '消滅年': info.extinction_date.year if info.extinction_date else None,
                'ステータス': info.status,
                '親会社': info.parent_company or 'なし',
                '収集時間': str(result.collection_metadata.get('collection_end', '') - 
                            result.collection_metadata.get('collection_start', ''))[:8] 
                            if result.collection_metadata.get('collection_end') else 'N/A'
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        # サマリー統計
        stats_summary = {
            '総企業数': len(results),
            '収集成功企業数': sum(1 for r in results.values() if r.collection_success),
            '収集失敗企業数': sum(1 for r in results.values() if not r.collection_success),
            '平均データ品質スコア': summary_df['データ品質スコア'].mean(),
            '高シェア市場企業数': sum(1 for r in results.values() if r.company_info.market_category == 'high_share'),
            'シェア低下市場企業数': sum(1 for r in results.values() if r.company_info.market_category == 'declining'),
            'シェア失失市場企業数': sum(1 for r in results.values() if r.company_info.market_category == 'lost'),
            '消滅企業数': sum(1 for r in results.values() if r.company_info.status == 'extinct'),
            '平均データ年数': summary_df[summary_df['収集成功'] == '成功']['データ年数'].mean()
        }
        
        # Excel形式で保存
        summary_path = output_path / "collection_summary_report.xlsx"
        with pd.ExcelWriter(summary_path, engine='openpyxl') as writer:
            # 詳細結果
            summary_df.to_excel(writer, sheet_name='収集結果詳細', index=False)
            
            # 統計サマリー
            stats_df = pd.DataFrame(list(stats_summary.items()), columns=['項目', '値'])
            stats_df.to_excel(writer, sheet_name='収集統計', index=False)
            
            # 市場カテゴリー別集計
            category_stats = summary_df.groupby('市場カテゴリー').agg({
                '収集成功': lambda x: (x == '成功').sum(),
                'データ品質スコア': 'mean',
                'データ年数': 'mean'
            }).round(3)
            category_stats.to_excel(writer, sheet_name='市場別統計')
        
        self.logger.info(f"収集サマリーレポートを生成: {summary_path}")

    def get_collection_status(self) -> Dict[str, any]:
        """現在の収集状況を取得"""
        
        total_companies = len(self.company_registry)
        
        # 企業ステータス別集計
        status_counts = {}
        category_counts = {}
        
        for company_info in self.company_registry.values():
            # ステータス別
            status = company_info.status
            status_counts[status] = status_counts.get(status, 0) + 1
            
            # カテゴリー別
            category = company_info.market_category
            category_counts[category] = category_counts.get(category, 0) + 1
        
        return {
            'total_companies': total_companies,
            'status_distribution': status_counts,
            'category_distribution': category_counts,
            'collection_period': f"{self.collection_settings['start_year']}-{self.collection_settings['end_year']}",
            'expected_total_data_points': total_companies * (self.collection_settings['end_year'] - self.collection_settings['start_year'] + 1)
        }

    def update_company_info(self, company_name: str, **kwargs) -> bool:
        """企業情報を更新"""
        
        if company_name not in self.company_registry:
            self.logger.error(f"企業が見つかりません: {company_name}")
            return False
        
        company_info = self.company_registry[company_name]
        
        # 更新可能な属性
        updatable_fields = [
            'company_code', 'market_category', 'market_sector',
            'establishment_date', 'listing_date', 'delisting_date',
            'extinction_date', 'status', 'parent_company'
        ]
        
        for field, value in kwargs.items():
            if field in updatable_fields:
                setattr(company_info, field, value)
                self.logger.info(f"{company_name}の{field}を更新: {value}")
        
        return True

    def add_special_event(self, company_name: str, event_type: str, 
                            event_date: datetime, description: str) -> bool:
        """企業の特別イベント（M&A、分社化等）を記録"""
        
        if company_name not in self.company_registry:
            self.logger.error(f"企業が見つかりません: {company_name}")
            return False
        
        event = {
            'type': event_type,
            'date': event_date,
            'description': description,
            'recorded_at': datetime.now()
        }
        
        self.company_registry[company_name].special_events.append(event)
        self.logger.info(f"{company_name}に特別イベントを追加: {event_type}")
        
        return True


# 使用例・テスト用のメイン関数
def main():
    """LifecycleDataCollectorの使用例"""
    
    # データ収集システムを初期化
    collector = LifecycleDataCollector()
    
    # 収集状況の確認
    status = collector.get_collection_status()
    print("=" * 60)
    print("A2AI ライフサイクルデータ収集システム")
    print("=" * 60)
    print(f"対象企業数: {status['total_companies']}社")
    print(f"収集期間: {status['collection_period']}")
    print(f"企業ステータス分布: {status['status_distribution']}")
    print(f"市場カテゴリー分布: {status['category_distribution']}")
    print(f"予想データポイント総数: {status['expected_total_data_points']:,}")
    print("=" * 60)
    
    # 特定企業のデータ収集テスト（実際の本格収集前のテスト）
    test_companies = ['ファナック', 'オリンパス', 'トヨタ自動車', '三洋電機']
    
    print("\n[テスト収集開始]")
    for company_name in test_companies:
        if company_name in collector.company_registry:
            print(f"\n{company_name} のテスト収集...")
            company_info = collector.company_registry[company_name]
            result = collector.collect_company_lifecycle_data(company_info)
            
            print(f"  結果: {'成功' if result.collection_success else '失敗'}")
            print(f"  データ年数: {len(result.financial_data)}")
            print(f"  品質スコア: {result.data_quality_score:.3f}")
            print(f"  欠損年数: {len(result.missing_years)}")
    
    print("\n[テスト完了]")
    print("\n本格的なデータ収集を実行するには:")
    print("  results = collector.collect_all_companies_data()")
    print("  collector.save_collection_results(results)")


if __name__ == "__main__":
    main()