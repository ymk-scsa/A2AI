"""
A2AI (Advanced Financial Analysis AI) Database Utilities

企業ライフサイクル分析に特化したデータベース操作ユーティリティ
- 150社の継続企業・消滅企業・新設企業データ管理
- 9つの評価項目と拡張要因項目への対応
- 生存分析・新設企業分析・事業継承分析用データ構造
- 40年間の時系列データ管理
- EDINET APIデータとの統合
"""

import sqlite3
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import json
import logging
from datetime import datetime, date
from pathlib import Path
import psycopg2
import pymongo
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Float, Date, Boolean, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.dialects.postgresql import JSONB
import yaml

logger = logging.getLogger(__name__)

Base = declarative_base()

class CompanyLifecycle(Base):
    """企業ライフサイクル基本情報テーブル"""
    __tablename__ = 'company_lifecycle'
    
    company_id = Column(String(20), primary_key=True)
    company_name = Column(String(200), nullable=False)
    market_category = Column(String(50), nullable=False)  # high_share, declining, lost
    lifecycle_status = Column(String(50), nullable=False)  # active, extinct, spinoff, merged
    establishment_date = Column(Date)
    extinction_date = Column(Date, nullable=True)
    listing_date = Column(Date, nullable=True)
    delisting_date = Column(Date, nullable=True)
    parent_company_id = Column(String(20), nullable=True)
    spinoff_source_id = Column(String(20), nullable=True)
    industry_code = Column(String(10))
    edinet_code = Column(String(10))
    securities_code = Column(String(10))
    data_start_year = Column(Integer)
    data_end_year = Column(Integer)
    total_data_years = Column(Integer)
    created_at = Column(Date, default=datetime.utcnow)
    updated_at = Column(Date, default=datetime.utcnow)

class MarketShareData(Base):
    """市場シェアデータテーブル"""
    __tablename__ = 'market_share_data'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    company_id = Column(String(20), ForeignKey('company_lifecycle.company_id'))
    year = Column(Integer, nullable=False)
    market_name = Column(String(100), nullable=False)
    global_share_pct = Column(Float)
    domestic_share_pct = Column(Float)
    regional_share_pct = Column(Float)
    market_size_billion = Column(Float)
    company_revenue_billion = Column(Float)
    rank_global = Column(Integer)
    rank_domestic = Column(Integer)
    data_source = Column(String(100))
    created_at = Column(Date, default=datetime.utcnow)

class FinancialData(Base):
    """財務データメインテーブル"""
    __tablename__ = 'financial_data'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    company_id = Column(String(20), ForeignKey('company_lifecycle.company_id'))
    fiscal_year = Column(Integer, nullable=False)
    quarter = Column(Integer, default=4)  # 4 = 年次決算
    
    # 貸借対照表
    total_assets = Column(Float)
    tangible_fixed_assets = Column(Float)
    intangible_assets = Column(Float)
    investment_securities = Column(Float)
    accounts_receivable = Column(Float)
    inventory = Column(Float)
    cash_and_deposits = Column(Float)
    total_equity = Column(Float)
    total_liabilities = Column(Float)
    interest_bearing_debt = Column(Float)
    
    # 損益計算書
    revenue = Column(Float)
    cost_of_sales = Column(Float)
    gross_profit = Column(Float)
    operating_profit = Column(Float)
    ordinary_profit = Column(Float)
    net_income = Column(Float)
    selling_admin_expenses = Column(Float)
    rd_expenses = Column(Float)
    advertising_expenses = Column(Float)
    personnel_expenses = Column(Float)
    depreciation = Column(Float)
    
    # キャッシュフロー計算書
    operating_cash_flow = Column(Float)
    investing_cash_flow = Column(Float)
    financing_cash_flow = Column(Float)
    capital_expenditure = Column(Float)
    
    # 従業員・その他情報
    employee_count = Column(Integer)
    average_salary = Column(Float)
    retirement_benefit_expenses = Column(Float)
    welfare_expenses = Column(Float)
    overseas_revenue_ratio = Column(Float)
    segment_count = Column(Integer)
    order_backlog = Column(Float)
    
    created_at = Column(Date, default=datetime.utcnow)

class EvaluationMetrics(Base):
    """9つの評価項目計算結果テーブル"""
    __tablename__ = 'evaluation_metrics'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    company_id = Column(String(20), ForeignKey('company_lifecycle.company_id'))
    fiscal_year = Column(Integer, nullable=False)
    
    # 従来6項目
    revenue_amount = Column(Float)  # 売上高
    revenue_growth_rate = Column(Float)  # 売上高成長率
    operating_profit_margin = Column(Float)  # 売上高営業利益率
    net_profit_margin = Column(Float)  # 売上高当期純利益率
    roe = Column(Float)  # ROE
    value_added_ratio = Column(Float)  # 売上高付加価値率
    
    # 新規3項目
    survival_probability = Column(Float)  # 企業存続確率
    business_success_rate = Column(Float)  # 新規事業成功率
    succession_success_score = Column(Float)  # 事業継承成功度
    
    created_at = Column(Date, default=datetime.utcnow)

class FactorMetrics(Base):
    """23要因項目計算結果テーブル"""
    __tablename__ = 'factor_metrics'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    company_id = Column(String(20), ForeignKey('company_lifecycle.company_id'))
    fiscal_year = Column(Integer, nullable=False)
    evaluation_type = Column(String(50))  # revenue, growth, margin, etc.
    
    # 投資・資産関連 (factors 1-5)
    tangible_assets_eoy = Column(Float)  # 有形固定資産期末残高
    capital_expenditure = Column(Float)  # 設備投資額
    rd_expenses = Column(Float)  # 研究開発費
    intangible_assets_eoy = Column(Float)  # 無形固定資産期末残高
    investment_securities_eoy = Column(Float)  # 投資有価証券期末残高
    
    # 人的資源関連 (factors 6-10)
    employee_count = Column(Float)  # 従業員数
    average_annual_salary = Column(Float)  # 平均年間給与
    retirement_benefit_cost = Column(Float)  # 退職給付費用
    welfare_cost = Column(Float)  # 福利厚生費
    personnel_cost_ratio = Column(Float)  # 人件費率
    
    # 運転資本・効率性関連 (factors 11-15)
    accounts_receivable_eoy = Column(Float)  # 売上債権期末残高
    inventory_eoy = Column(Float)  # 棚卸資産期末残高
    total_assets_eoy = Column(Float)  # 総資産期末残高
    receivables_turnover = Column(Float)  # 売上債権回転率
    inventory_turnover = Column(Float)  # 棚卸資産回転率
    
    # 事業展開関連 (factors 16-20)
    overseas_revenue_ratio = Column(Float)  # 海外売上高比率
    business_segment_count = Column(Float)  # 事業セグメント数
    selling_admin_expenses = Column(Float)  # 販売費及び一般管理費
    advertising_expenses = Column(Float)  # 広告宣伝費
    non_operating_income = Column(Float)  # 営業外収益
    
    # 新規拡張項目 (factors 21-23)
    company_age = Column(Float)  # 企業年齢
    market_entry_timing = Column(Float)  # 市場参入時期
    parent_dependency_ratio = Column(Float)  # 親会社依存度
    
    created_at = Column(Date, default=datetime.utcnow)

class ExtinctionEvents(Base):
    """企業消滅イベントテーブル"""
    __tablename__ = 'extinction_events'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    company_id = Column(String(20), ForeignKey('company_lifecycle.company_id'))
    extinction_date = Column(Date, nullable=False)
    extinction_type = Column(String(50))  # bankruptcy, merger, acquisition, liquidation
    acquiring_company_id = Column(String(20), nullable=True)
    final_revenue = Column(Float)
    final_assets = Column(Float)
    final_debt = Column(Float)
    final_employees = Column(Integer)
    extinction_cause = Column(Text)
    advance_warning_years = Column(Float)  # 事前警告期間（年）
    market_impact_score = Column(Float)
    created_at = Column(Date, default=datetime.utcnow)

class EmergenceEvents(Base):
    """新設企業イベントテーブル"""
    __tablename__ = 'emergence_events'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    company_id = Column(String(20), ForeignKey('company_lifecycle.company_id'))
    establishment_date = Column(Date, nullable=False)
    emergence_type = Column(String(50))  # spinoff, startup, joint_venture, acquisition
    source_company_id = Column(String(20), nullable=True)
    initial_capital = Column(Float)
    initial_employees = Column(Integer)
    initial_market_focus = Column(String(200))
    founder_background = Column(Text)
    success_milestone_years = Column(Float)  # 成功到達年数
    market_disruption_score = Column(Float)
    created_at = Column(Date, default=datetime.utcnow)

class SurvivalAnalysisData(Base):
    """生存分析データテーブル"""
    __tablename__ = 'survival_analysis_data'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    company_id = Column(String(20), ForeignKey('company_lifecycle.company_id'))
    observation_start = Column(Date)
    observation_end = Column(Date)
    event_occurred = Column(Boolean, default=False)  # 消滅イベント発生
    survival_time_years = Column(Float)
    censoring_type = Column(String(20))  # right_censored, left_censored, interval
    risk_score = Column(Float)
    hazard_ratio = Column(Float)
    created_at = Column(Date, default=datetime.utcnow)

class DatabaseManager:
    """A2AIデータベース統合管理クラス"""
    
    def __init__(self, config_path: str = "config/settings.py"):
        """
        データベース接続管理の初期化
        
        Args:
            config_path: 設定ファイルパス
        """
        self.config = self._load_config(config_path)
        self.engines = {}
        self.sessions = {}
        self._initialize_connections()
    
    def _load_config(self, config_path: str) -> Dict:
        """設定ファイル読み込み"""
        try:
            # YAMLまたはPython設定ファイル対応
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                with open(config_path, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f)
            else:
                # Python設定ファイルの場合
                import importlib.util
                spec = importlib.util.spec_from_file_location("config", config_path)
                config_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(config_module)
                return {
                    'databases': config_module.DATABASES,
                    'logging': getattr(config_module, 'LOGGING', {}),
                }
        except Exception as e:
            logger.warning(f"Config load failed: {e}, using defaults")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """デフォルト設定"""
        return {
            'databases': {
                'primary': {
                    'engine': 'sqlite',
                    'path': 'data/a2ai_primary.db'
                },
                'financial': {
                    'engine': 'sqlite', 
                    'path': 'data/a2ai_financial.db'
                },
                'market_share': {
                    'engine': 'sqlite',
                    'path': 'data/a2ai_market_share.db'
                }
            }
        }
    
    def _initialize_connections(self):
        """データベース接続初期化"""
        for db_name, db_config in self.config['databases'].items():
            try:
                if db_config['engine'] == 'sqlite':
                    # SQLite接続
                    db_path = Path(db_config['path'])
                    db_path.parent.mkdir(parents=True, exist_ok=True)
                    connection_string = f"sqlite:///{db_path}"
                    
                elif db_config['engine'] == 'postgresql':
                    # PostgreSQL接続
                    connection_string = (
                        f"postgresql://{db_config['user']}:{db_config['password']}"
                        f"@{db_config['host']}:{db_config['port']}/{db_config['database']}"
                    )
                
                # SQLAlchemyエンジン作成
                engine = create_engine(connection_string, echo=db_config.get('echo', False))
                self.engines[db_name] = engine
                
                # セッションメーカー作成
                SessionClass = sessionmaker(bind=engine)
                self.sessions[db_name] = SessionClass
                
                logger.info(f"Database {db_name} connection established")
                
            except Exception as e:
                logger.error(f"Failed to initialize database {db_name}: {e}")
    
    def create_tables(self, db_name: str = 'primary'):
        """テーブル作成"""
        try:
            engine = self.engines[db_name]
            Base.metadata.create_all(engine)
            logger.info(f"Tables created in database {db_name}")
        except Exception as e:
            logger.error(f"Failed to create tables in {db_name}: {e}")
            raise
    
    def get_session(self, db_name: str = 'primary') -> Session:
        """セッション取得"""
        return self.sessions[db_name]()
    
    def insert_company_lifecycle(self, company_data: Dict, db_name: str = 'primary'):
        """企業ライフサイクル情報挿入"""
        session = self.get_session(db_name)
        try:
            company = CompanyLifecycle(**company_data)
            session.add(company)
            session.commit()
            logger.info(f"Company {company_data['company_id']} inserted")
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to insert company {company_data.get('company_id', 'Unknown')}: {e}")
            raise
        finally:
            session.close()
    
    def bulk_insert_financial_data(self, financial_data_list: List[Dict], 
                                    db_name: str = 'primary', batch_size: int = 1000):
        """財務データ一括挿入"""
        session = self.get_session(db_name)
        try:
            for i in range(0, len(financial_data_list), batch_size):
                batch = financial_data_list[i:i + batch_size]
                financial_objects = [FinancialData(**data) for data in batch]
                session.bulk_save_objects(financial_objects)
                session.commit()
                logger.info(f"Inserted batch {i//batch_size + 1}: {len(batch)} records")
            
            logger.info(f"Total {len(financial_data_list)} financial records inserted")
            
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to bulk insert financial data: {e}")
            raise
        finally:
            session.close()
    
    def get_company_financial_history(self, company_id: str, 
                                        start_year: Optional[int] = None,
                                        end_year: Optional[int] = None,
                                        db_name: str = 'primary') -> pd.DataFrame:
        """企業の財務履歴取得"""
        session = self.get_session(db_name)
        try:
            query = session.query(FinancialData).filter(
                FinancialData.company_id == company_id
            )
            
            if start_year:
                query = query.filter(FinancialData.fiscal_year >= start_year)
            if end_year:
                query = query.filter(FinancialData.fiscal_year <= end_year)
                
            query = query.order_by(FinancialData.fiscal_year)
            results = query.all()
            
            # DataFrameに変換
            data = []
            for result in results:
                data.append({
                    'company_id': result.company_id,
                    'fiscal_year': result.fiscal_year,
                    'revenue': result.revenue,
                    'operating_profit': result.operating_profit,
                    'net_income': result.net_income,
                    'total_assets': result.total_assets,
                    'total_equity': result.total_equity,
                    'employee_count': result.employee_count,
                    # 他の必要な項目...
                })
            
            return pd.DataFrame(data)
            
        except Exception as e:
            logger.error(f"Failed to get financial history for {company_id}: {e}")
            return pd.DataFrame()
        finally:
            session.close()
    
    def get_market_category_companies(self, market_category: str,
                                        lifecycle_status: Optional[str] = None,
                                        db_name: str = 'primary') -> List[Dict]:
        """市場カテゴリー別企業リスト取得"""
        session = self.get_session(db_name)
        try:
            query = session.query(CompanyLifecycle).filter(
                CompanyLifecycle.market_category == market_category
            )
            
            if lifecycle_status:
                query = query.filter(CompanyLifecycle.lifecycle_status == lifecycle_status)
            
            results = query.all()
            companies = []
            for company in results:
                companies.append({
                    'company_id': company.company_id,
                    'company_name': company.company_name,
                    'lifecycle_status': company.lifecycle_status,
                    'establishment_date': company.establishment_date,
                    'extinction_date': company.extinction_date,
                    'total_data_years': company.total_data_years
                })
            
            return companies
            
        except Exception as e:
            logger.error(f"Failed to get companies for category {market_category}: {e}")
            return []
        finally:
            session.close()
    
    def calculate_survival_statistics(self, market_category: Optional[str] = None,
                                        db_name: str = 'primary') -> Dict:
        """生存統計計算"""
        session = self.get_session(db_name)
        try:
            # 基本クエリ
            query = session.query(CompanyLifecycle)
            
            if market_category:
                query = query.filter(CompanyLifecycle.market_category == market_category)
            
            all_companies = query.all()
            
            # 統計計算
            total_companies = len(all_companies)
            active_companies = len([c for c in all_companies if c.lifecycle_status == 'active'])
            extinct_companies = len([c for c in all_companies if c.lifecycle_status == 'extinct'])
            
            # 平均存続年数（消滅企業）
            extinct_durations = []
            for company in all_companies:
                if company.extinction_date and company.establishment_date:
                    duration = (company.extinction_date - company.establishment_date).days / 365.25
                    extinct_durations.append(duration)
            
            avg_extinction_duration = np.mean(extinct_durations) if extinct_durations else 0
            
            # 現在の平均企業年齢（存続企業）
            current_date = datetime.now().date()
            active_ages = []
            for company in all_companies:
                if company.lifecycle_status == 'active' and company.establishment_date:
                    age = (current_date - company.establishment_date).days / 365.25
                    active_ages.append(age)
            
            avg_current_age = np.mean(active_ages) if active_ages else 0
            
            return {
                'total_companies': total_companies,
                'active_companies': active_companies,
                'extinct_companies': extinct_companies,
                'survival_rate': active_companies / total_companies if total_companies > 0 else 0,
                'avg_extinction_duration_years': avg_extinction_duration,
                'avg_current_age_years': avg_current_age,
                'market_category': market_category or 'all'
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate survival statistics: {e}")
            return {}
        finally:
            session.close()
    
    def get_evaluation_metrics_timeseries(self, company_ids: List[str],
                                            start_year: Optional[int] = None,
                                            end_year: Optional[int] = None,
                                            db_name: str = 'primary') -> pd.DataFrame:
        """評価項目時系列データ取得"""
        session = self.get_session(db_name)
        try:
            query = session.query(EvaluationMetrics).filter(
                EvaluationMetrics.company_id.in_(company_ids)
            )
            
            if start_year:
                query = query.filter(EvaluationMetrics.fiscal_year >= start_year)
            if end_year:
                query = query.filter(EvaluationMetrics.fiscal_year <= end_year)
            
            query = query.order_by(EvaluationMetrics.company_id, EvaluationMetrics.fiscal_year)
            results = query.all()
            
            # DataFrame形式で返却
            data = []
            for result in results:
                data.append({
                    'company_id': result.company_id,
                    'fiscal_year': result.fiscal_year,
                    'revenue_amount': result.revenue_amount,
                    'revenue_growth_rate': result.revenue_growth_rate,
                    'operating_profit_margin': result.operating_profit_margin,
                    'net_profit_margin': result.net_profit_margin,
                    'roe': result.roe,
                    'value_added_ratio': result.value_added_ratio,
                    'survival_probability': result.survival_probability,
                    'business_success_rate': result.business_success_rate,
                    'succession_success_score': result.succession_success_score
                })
            
            return pd.DataFrame(data)
            
        except Exception as e:
            logger.error(f"Failed to get evaluation metrics timeseries: {e}")
            return pd.DataFrame()
        finally:
            session.close()
    
    def export_data_for_analysis(self, output_path: str, 
                                    market_categories: Optional[List[str]] = None,
                                    db_name: str = 'primary'):
        """分析用データエクスポート"""
        try:
            # 企業リスト取得
            companies_query = "SELECT * FROM company_lifecycle"
            if market_categories:
                placeholders = ','.join(['?' for _ in market_categories])
                companies_query += f" WHERE market_category IN ({placeholders})"
            
            engine = self.engines[db_name]
            
            # 各テーブルのデータをCSVエクスポート
            tables_to_export = {
                'companies': (companies_query, market_categories),
                'financial_data': ("SELECT * FROM financial_data", None),
                'evaluation_metrics': ("SELECT * FROM evaluation_metrics", None),
                'factor_metrics': ("SELECT * FROM factor_metrics", None),
                'market_share_data': ("SELECT * FROM market_share_data", None),
                'extinction_events': ("SELECT * FROM extinction_events", None),
                'emergence_events': ("SELECT * FROM emergence_events", None),
                'survival_analysis_data': ("SELECT * FROM survival_analysis_data", None)
            }
            
            output_dir = Path(output_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            for table_name, (query, params) in tables_to_export.items():
                try:
                    df = pd.read_sql_query(query, engine, params=params)
                    output_file = output_dir / f"{table_name}.csv"
                    df.to_csv(output_file, index=False, encoding='utf-8')
                    logger.info(f"Exported {len(df)} records to {output_file}")
                except Exception as e:
                    logger.error(f"Failed to export {table_name}: {e}")
            
            logger.info(f"Data export completed to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to export data: {e}")
            raise
    
    def backup_database(self, backup_path: str, db_name: str = 'primary'):
        """データベースバックアップ"""
        try:
            engine = self.engines[db_name]
            
            if 'sqlite' in str(engine.url):
                # SQLiteの場合
                import shutil
                source_path = str(engine.url).replace('sqlite:///', '')
                backup_file = Path(backup_path) / f"{db_name}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
                backup_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path, backup_file)
                logger.info(f"SQLite backup created: {backup_file}")
                
            else:
                # PostgreSQL等の場合
                logger.warning("Database backup for non-SQLite engines not implemented")
            
        except Exception as e:
            logger.error(f"Failed to backup database {db_name}: {e}")
            raise
    
    def close_connections(self):
        """全接続クローズ"""
        for db_name, engine in self.engines.items():
            try:
                engine.dispose()
                logger.info(f"Database {db_name} connection closed")
            except Exception as e:
                logger.error(f"Failed to close database {db_name}: {e}")

# シングルトンインスタンス
_db_manager_instance = None

def get_database_manager(config_path: str = "config/settings.py") -> DatabaseManager:
    """データベースマネージャーのシングルトンインスタンス取得"""
    global _db_manager_instance
    if _db_manager_instance is None:
        _db_manager_instance = DatabaseManager(config_path)
    return _db_manager_instance

# 便利関数
def quick_query(sql: str, db_name: str = 'primary') -> pd.DataFrame:
    """クイック SQL クエリ実行"""
    db_manager = get_database_manager()
    engine = db_manager.engines[db_name]
    return pd.read_sql_query(sql, engine)

def get_company_data(company_id: str, 
                        start_year: Optional[int] = None,
                        end_year: Optional[int] = None) -> Dict[str, pd.DataFrame]:
    """企業の全データ取得（財務・評価・要因項目）"""
    db_manager = get_database_manager()
    
    result = {
        'financial': db_manager.get_company_financial_history(
            company_id, start_year, end_year
        ),
        'evaluation': db_manager.get_evaluation_metrics_timeseries(
            [company_id], start_year, end_year
        )
    }
    
    return result

if __name__ == "__main__":
    # テスト用コード
    logging.basicConfig(level=logging.INFO)
    
    # データベースマネージャー初期化
    db_manager = DatabaseManager()
    db_manager.create_tables()
    
    # サンプルデータ挿入テスト
    sample_company = {
        'company_id': 'TEST001',
        'company_name': 'テスト株式会社',
        'market_category': 'high_share',
        'lifecycle_status': 'active',
        'establishment_date': date(1984, 4, 1),
        'industry_code': '3000',
        'edinet_code': 'E12345',
        'securities_code': '1234',
        'data_start_year': 1984,
        'data_end_year': 2024,
        'total_data_years': 40
    }
    
    try:
        db_manager.insert_company_lifecycle(sample_company)
        print("Sample company data inserted successfully")
        
        # 生存統計テスト
        stats = db_manager.calculate_survival_statistics('high_share')
        print(f"Survival statistics: {stats}")
        
        # データエクスポートテスト
        db_manager.export_data_for_analysis('results/exported_data', ['high_share'])
        print("Data export completed")
        
    except Exception as e:
        print(f"Test failed: {e}")
    finally:
        db_manager.close_connections()