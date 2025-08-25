#!/usr/bin/env python3
"""
A2AI (Advanced Financial Analysis AI) Database Setup Script
企業ライフサイクル分析対応データベースの初期化

Features:
- 150社の企業マスター管理
- 40年分の財務諸表データストレージ
- 企業消滅・新設・分社イベント管理
- 9つの評価項目 × 23の要因項目対応
- 市場シェアデータ管理
- 生存バイアス補正用メタデータ
"""

import os
import sys
import logging
from datetime import datetime, date
from pathlib import Path
import sqlite3
import pandas as pd
from typing import Optional, List, Dict, Any

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from config.settings import DATABASE_CONFIG, LOGGING_CONFIG
from src.utils.logging_utils import setup_logger

class A2AIDatabaseSetup:
    """A2AI データベースセットアップクラス"""
    
    def __init__(self, db_path: Optional[str] = None):
        """
        初期化
        
        Args:
            db_path: データベースファイルパス（Noneの場合は設定値を使用）
        """
        self.logger = setup_logger(__name__, LOGGING_CONFIG)
        
        # データベースパス設定
        if db_path is None:
            db_path = DATABASE_CONFIG.get('path', 'data/a2ai_database.db')
        
        self.db_path = Path(project_root) / db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 企業分類定義
        self.market_categories = {
            'high_share': ['ロボット', '内視鏡', '工作機械', '電子材料', '精密測定機器'],
            'declining': ['自動車', '鉄鋼', 'スマート家電', 'バッテリー', 'PC・周辺機器'],
            'lost_share': ['家電', '半導体', 'スマートフォン', 'PC', '通信機器']
        }
        
        self.logger.info(f"データベースセットアップ開始: {self.db_path}")
    
    def create_database(self) -> bool:
        """
        データベース作成・初期化
        
        Returns:
            bool: 成功時True
        """
        try:
            # 既存データベースのバックアップ
            if self.db_path.exists():
                backup_path = self.db_path.with_suffix(
                    f'.backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.db'
                )
                self.db_path.rename(backup_path)
                self.logger.info(f"既存データベースをバックアップ: {backup_path}")
            
            # データベース接続・作成
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("PRAGMA foreign_keys = ON")
                cursor = conn.cursor()
                
                # テーブル作成
                self._create_master_tables(cursor)
                self._create_financial_tables(cursor)
                self._create_lifecycle_tables(cursor)
                self._create_market_tables(cursor)
                self._create_analysis_tables(cursor)
                
                # インデックス作成
                self._create_indexes(cursor)
                
                # 初期データ投入
                self._insert_initial_data(cursor)
                
                conn.commit()
                self.logger.info("データベース作成完了")
                return True
                
        except Exception as e:
            self.logger.error(f"データベース作成エラー: {e}")
            return False
    
    def _create_master_tables(self, cursor: sqlite3.Cursor) -> None:
        """マスターテーブル作成"""
        
        # 企業マスター
        cursor.execute("""
        CREATE TABLE companies (
            company_id INTEGER PRIMARY KEY AUTOINCREMENT,
            company_name TEXT NOT NULL UNIQUE,
            company_code TEXT UNIQUE,  -- 証券コード
            market_category TEXT NOT NULL CHECK (market_category IN ('high_share', 'declining', 'lost_share')),
            market_sector TEXT NOT NULL,  -- 具体的な市場（ロボット、内視鏡等）
            founded_date DATE,
            listed_date DATE,
            delisted_date DATE,
            extinction_date DATE,
            current_status TEXT NOT NULL DEFAULT 'active' CHECK (current_status IN ('active', 'delisted', 'merged', 'bankrupt', 'spinoff')),
            parent_company_id INTEGER,
            data_start_year INTEGER NOT NULL,
            data_end_year INTEGER,
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (parent_company_id) REFERENCES companies (company_id)
        )
        """)
        
        # 市場セクターマスター
        cursor.execute("""
        CREATE TABLE market_sectors (
            sector_id INTEGER PRIMARY KEY AUTOINCREMENT,
            sector_name TEXT NOT NULL UNIQUE,
            category TEXT NOT NULL CHECK (category IN ('high_share', 'declining', 'lost_share')),
            description TEXT,
            global_market_size_usd REAL,
            japan_share_percentage REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        # 評価項目マスター
        cursor.execute("""
        CREATE TABLE evaluation_metrics (
            metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
            metric_name TEXT NOT NULL UNIQUE,
            metric_code TEXT NOT NULL UNIQUE,
            description TEXT,
            unit TEXT,
            calculation_method TEXT,
            is_traditional BOOLEAN DEFAULT TRUE,  -- 従来6項目 or 新規3項目
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        # 要因項目マスター
        cursor.execute("""
        CREATE TABLE factor_metrics (
            factor_id INTEGER PRIMARY KEY AUTOINCREMENT,
            evaluation_metric_id INTEGER NOT NULL,
            factor_name TEXT NOT NULL,
            factor_code TEXT NOT NULL,
            factor_category TEXT NOT NULL,  -- 投資・資産関連、人的資源関連等
            description TEXT,
            unit TEXT,
            calculation_method TEXT,
            data_source TEXT,  -- 貸借対照表、損益計算書、CF等
            is_extended BOOLEAN DEFAULT FALSE,  -- 拡張項目（企業年齢等）
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (evaluation_metric_id) REFERENCES evaluation_metrics (metric_id),
            UNIQUE(evaluation_metric_id, factor_code)
        )
        """)
        
        self.logger.info("マスターテーブル作成完了")
    
    def _create_financial_tables(self, cursor: sqlite3.Cursor) -> None:
        """財務データテーブル作成"""
        
        # 財務諸表メインテーブル
        cursor.execute("""
        CREATE TABLE financial_statements (
            statement_id INTEGER PRIMARY KEY AUTOINCREMENT,
            company_id INTEGER NOT NULL,
            fiscal_year INTEGER NOT NULL,
            fiscal_end_date DATE,
            statement_type TEXT NOT NULL CHECK (statement_type IN ('annual', 'quarterly')),
            accounting_standard TEXT DEFAULT 'JGAAP' CHECK (accounting_standard IN ('JGAAP', 'IFRS', 'USGAAP')),
            currency TEXT DEFAULT 'JPY',
            
            -- 基本財務指標
            revenue REAL,                    -- 売上高
            gross_profit REAL,               -- 売上総利益
            operating_profit REAL,           -- 営業利益
            ordinary_profit REAL,            -- 経常利益
            net_income REAL,                 -- 当期純利益
            total_assets REAL,               -- 総資産
            shareholders_equity REAL,        -- 株主資本
            total_liabilities REAL,          -- 総負債
            
            -- 従業員・人的データ
            employee_count INTEGER,          -- 従業員数
            average_annual_salary REAL,      -- 平均年間給与
            
            -- データ品質情報
            data_quality_score REAL DEFAULT 1.0,  -- 1.0=完全, 0.0=推定値
            is_estimated BOOLEAN DEFAULT FALSE,
            estimation_method TEXT,
            
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (company_id) REFERENCES companies (company_id),
            UNIQUE(company_id, fiscal_year, statement_type)
        )
        """)
        
        # 評価項目データテーブル
        cursor.execute("""
        CREATE TABLE evaluation_data (
            eval_data_id INTEGER PRIMARY KEY AUTOINCREMENT,
            company_id INTEGER NOT NULL,
            fiscal_year INTEGER NOT NULL,
            metric_id INTEGER NOT NULL,
            value REAL,
            calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            calculation_version TEXT DEFAULT '1.0',
            is_estimated BOOLEAN DEFAULT FALSE,
            FOREIGN KEY (company_id) REFERENCES companies (company_id),
            FOREIGN KEY (metric_id) REFERENCES evaluation_metrics (metric_id),
            UNIQUE(company_id, fiscal_year, metric_id)
        )
        """)
        
        # 要因項目データテーブル
        cursor.execute("""
        CREATE TABLE factor_data (
            factor_data_id INTEGER PRIMARY KEY AUTOINCREMENT,
            company_id INTEGER NOT NULL,
            fiscal_year INTEGER NOT NULL,
            factor_id INTEGER NOT NULL,
            value REAL,
            calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            calculation_version TEXT DEFAULT '1.0',
            is_estimated BOOLEAN DEFAULT FALSE,
            estimation_confidence REAL DEFAULT 1.0,  -- 推定信頼度
            FOREIGN KEY (company_id) REFERENCES companies (company_id),
            FOREIGN KEY (factor_id) REFERENCES factor_metrics (factor_id),
            UNIQUE(company_id, fiscal_year, factor_id)
        )
        """)
        
        self.logger.info("財務データテーブル作成完了")
    
    def _create_lifecycle_tables(self, cursor: sqlite3.Cursor) -> None:
        """企業ライフサイクル関連テーブル作成"""
        
        # 企業イベントテーブル
        cursor.execute("""
        CREATE TABLE corporate_events (
            event_id INTEGER PRIMARY KEY AUTOINCREMENT,
            company_id INTEGER NOT NULL,
            event_type TEXT NOT NULL CHECK (event_type IN (
                'founding', 'listing', 'delisting', 'merger', 'acquisition', 
                'spinoff', 'bankruptcy', 'restructure', 'name_change'
            )),
            event_date DATE NOT NULL,
            description TEXT,
            related_company_id INTEGER,  -- 相手企業（M&A等）
            financial_impact REAL,       -- 財務的影響額
            market_impact_description TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (company_id) REFERENCES companies (company_id),
            FOREIGN KEY (related_company_id) REFERENCES companies (company_id)
        )
        """)
        
        # 生存分析データテーブル
        cursor.execute("""
        CREATE TABLE survival_data (
            survival_id INTEGER PRIMARY KEY AUTOINCREMENT,
            company_id INTEGER NOT NULL,
            observation_start_year INTEGER NOT NULL,
            observation_end_year INTEGER,
            survival_time_years INTEGER,  -- 観測期間（年）
            event_occurred BOOLEAN DEFAULT FALSE,  -- イベント発生（消滅=TRUE）
            event_type TEXT,  -- 消滅理由
            censored BOOLEAN DEFAULT FALSE,  -- 打ち切りデータ
            market_category TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (company_id) REFERENCES companies (company_id),
            UNIQUE(company_id)
        )
        """)
        
        # 企業ライフサイクルステージテーブル
        cursor.execute("""
        CREATE TABLE lifecycle_stages (
            stage_id INTEGER PRIMARY KEY AUTOINCREMENT,
            company_id INTEGER NOT NULL,
            fiscal_year INTEGER NOT NULL,
            stage TEXT NOT NULL CHECK (stage IN (
                'startup', 'growth', 'maturity', 'decline', 'turnaround', 'exit'
            )),
            stage_score REAL,  -- ステージスコア（0-1）
            transition_probability REAL,  -- 次ステージ移行確率
            calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (company_id) REFERENCES companies (company_id),
            UNIQUE(company_id, fiscal_year)
        )
        """)
        
        self.logger.info("ライフサイクルテーブル作成完了")
    
    def _create_market_tables(self, cursor: sqlite3.Cursor) -> None:
        """市場データテーブル作成"""
        
        # 世界市場シェアデータテーブル
        cursor.execute("""
        CREATE TABLE market_share_data (
            share_id INTEGER PRIMARY KEY AUTOINCREMENT,
            company_id INTEGER NOT NULL,
            fiscal_year INTEGER NOT NULL,
            market_sector TEXT NOT NULL,
            global_share_percentage REAL,
            regional_share_percentage REAL,
            market_size_usd REAL,
            market_rank INTEGER,
            data_source TEXT,
            reliability_score REAL DEFAULT 1.0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (company_id) REFERENCES companies (company_id)
        )
        """)
        
        # 業界ベンチマークデータ
        cursor.execute("""
        CREATE TABLE industry_benchmarks (
            benchmark_id INTEGER PRIMARY KEY AUTOINCREMENT,
            market_sector TEXT NOT NULL,
            fiscal_year INTEGER NOT NULL,
            metric_name TEXT NOT NULL,
            benchmark_value REAL,
            benchmark_type TEXT CHECK (benchmark_type IN ('median', 'average', 'top_quartile', 'bottom_quartile')),
            sample_size INTEGER,
            data_source TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(market_sector, fiscal_year, metric_name, benchmark_type)
        )
        """)
        
        self.logger.info("市場データテーブル作成完了")
    
    def _create_analysis_tables(self, cursor: sqlite3.Cursor) -> None:
        """分析結果テーブル作成"""
        
        # 分析実行履歴
        cursor.execute("""
        CREATE TABLE analysis_runs (
            run_id INTEGER PRIMARY KEY AUTOINCREMENT,
            analysis_type TEXT NOT NULL CHECK (analysis_type IN (
                'factor_impact', 'market_comparison', 'survival_analysis', 
                'emergence_analysis', 'causal_inference', 'integrated_analysis'
            )),
            run_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            parameters TEXT,  -- JSON形式のパラメータ
            sample_size INTEGER,
            model_version TEXT,
            status TEXT DEFAULT 'running' CHECK (status IN ('running', 'completed', 'failed')),
            results_path TEXT,
            created_by TEXT DEFAULT 'system'
        )
        """)
        
        # 因果推論結果
        cursor.execute("""
        CREATE TABLE causal_effects (
            effect_id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            treatment_factor_id INTEGER NOT NULL,
            outcome_metric_id INTEGER NOT NULL,
            causal_effect REAL,
            confidence_interval_lower REAL,
            confidence_interval_upper REAL,
            p_value REAL,
            method TEXT NOT NULL,  -- 'did', 'iv', 'psm', 'causal_forest'
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (run_id) REFERENCES analysis_runs (run_id),
            FOREIGN KEY (treatment_factor_id) REFERENCES factor_metrics (factor_id),
            FOREIGN KEY (outcome_metric_id) REFERENCES evaluation_metrics (metric_id)
        )
        """)
        
        # 予測結果テーブル
        cursor.execute("""
        CREATE TABLE predictions (
            prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
            company_id INTEGER NOT NULL,
            prediction_type TEXT NOT NULL CHECK (prediction_type IN (
                'survival_probability', 'performance_forecast', 'market_share_forecast'
            )),
            target_year INTEGER NOT NULL,
            predicted_value REAL,
            confidence_interval_lower REAL,
            confidence_interval_upper REAL,
            model_name TEXT,
            prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (company_id) REFERENCES companies (company_id)
        )
        """)
        
        self.logger.info("分析テーブル作成完了")
    
    def _create_indexes(self, cursor: sqlite3.Cursor) -> None:
        """インデックス作成"""
        
        indexes = [
            # 企業関連インデックス
            "CREATE INDEX idx_companies_market_category ON companies(market_category)",
            "CREATE INDEX idx_companies_status ON companies(current_status)",
            "CREATE INDEX idx_companies_sector ON companies(market_sector)",
            
            # 財務データインデックス
            "CREATE INDEX idx_financial_year ON financial_statements(company_id, fiscal_year)",
            "CREATE INDEX idx_evaluation_data_year ON evaluation_data(company_id, fiscal_year)",
            "CREATE INDEX idx_factor_data_year ON factor_data(company_id, fiscal_year)",
            
            # ライフサイクル関連インデックス
            "CREATE INDEX idx_events_company_date ON corporate_events(company_id, event_date)",
            "CREATE INDEX idx_survival_category ON survival_data(market_category)",
            "CREATE INDEX idx_lifecycle_stages_year ON lifecycle_stages(company_id, fiscal_year)",
            
            # 市場データインデックス
            "CREATE INDEX idx_market_share_year ON market_share_data(company_id, fiscal_year)",
            "CREATE INDEX idx_benchmarks_sector_year ON industry_benchmarks(market_sector, fiscal_year)",
            
            # 分析結果インデックス
            "CREATE INDEX idx_analysis_runs_type ON analysis_runs(analysis_type, run_timestamp)",
            "CREATE INDEX idx_causal_effects_factors ON causal_effects(treatment_factor_id, outcome_metric_id)",
            "CREATE INDEX idx_predictions_company_type ON predictions(company_id, prediction_type)"
        ]
        
        for index_sql in indexes:
            cursor.execute(index_sql)
        
        self.logger.info("インデックス作成完了")
    
    def _insert_initial_data(self, cursor: sqlite3.Cursor) -> None:
        """初期データ投入"""
        
        # 市場セクターマスター投入
        sectors_data = []
        for category, sectors in self.market_categories.items():
            for sector in sectors:
                sectors_data.append((sector, category, f"{category}市場の{sector}分野"))
        
        cursor.executemany("""
        INSERT INTO market_sectors (sector_name, category, description)
        VALUES (?, ?, ?)
        """, sectors_data)
        
        # 評価項目マスター投入
        evaluation_metrics = [
            ('売上高', 'revenue', '企業の売上高', '百万円', 'financial_statements.revenue', True),
            ('売上高成長率', 'revenue_growth', '売上高の前年同期比成長率', '%', '(revenue_t - revenue_t-1) / revenue_t-1 * 100', True),
            ('売上高営業利益率', 'operating_margin', '売上高に対する営業利益の比率', '%', 'operating_profit / revenue * 100', True),
            ('売上高当期純利益率', 'net_margin', '売上高に対する当期純利益の比率', '%', 'net_income / revenue * 100', True),
            ('ROE', 'roe', '自己資本利益率', '%', 'net_income / shareholders_equity * 100', True),
            ('売上高付加価値率', 'value_added_ratio', '売上高付加価値率', '%', '付加価値 / 売上高 * 100', True),
            ('企業存続確率', 'survival_probability', '生存分析による企業存続確率', '確率', 'survival_model_prediction', False),
            ('新規事業成功率', 'emergence_success_rate', '新設企業・新規事業の成功率', '%', 'emergence_model_prediction', False),
            ('事業継承成功度', 'succession_success', 'M&A・分社化の成功度', 'スコア', 'succession_model_prediction', False)
        ]
        
        cursor.executemany("""
        INSERT INTO evaluation_metrics (metric_name, metric_code, description, unit, calculation_method, is_traditional)
        VALUES (?, ?, ?, ?, ?, ?)
        """, evaluation_metrics)
        
        # 要因項目マスターの一部投入（売上高関連のみ例示）
        cursor.execute("SELECT metric_id FROM evaluation_metrics WHERE metric_code = 'revenue'")
        revenue_metric_id = cursor.fetchone()[0]
        
        revenue_factors = [
            (revenue_metric_id, '有形固定資産', 'tangible_assets', '投資・資産関連', '有形固定資産残高', '百万円', 'balance_sheet.tangible_assets', '貸借対照表'),
            (revenue_metric_id, '設備投資額', 'capex', '投資・資産関連', '設備投資額', '百万円', 'cash_flow.capex', 'キャッシュフロー計算書'),
            (revenue_metric_id, '研究開発費', 'rd_expenses', '投資・資産関連', '研究開発費', '百万円', 'income_statement.rd_expenses', '損益計算書'),
            (revenue_metric_id, '従業員数', 'employee_count', '人的資源関連', '従業員数', '人', 'notes.employee_count', '注記情報'),
            (revenue_metric_id, '企業年齢', 'company_age', '企業特性', '設立からの経過年数', '年', 'fiscal_year - founded_year', '計算値', True)
        ]
        
        cursor.executemany("""
        INSERT INTO factor_metrics (evaluation_metric_id, factor_name, factor_code, factor_category, 
                                    description, unit, calculation_method, data_source, is_extended)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, revenue_factors)
        
        self.logger.info("初期データ投入完了")
    
    def insert_company_data(self, companies_csv_path: Optional[str] = None) -> bool:
        """
        企業データ投入
        
        Args:
            companies_csv_path: 企業リストCSVファイルパス
            
        Returns:
            bool: 成功時True
        """
        try:
            # デフォルトの企業リスト（添付ファイルベース）
            if companies_csv_path is None:
                companies_data = self._get_default_companies_data()
            else:
                companies_data = pd.read_csv(companies_csv_path)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                for _, row in companies_data.iterrows():
                    cursor.execute("""
                    INSERT OR REPLACE INTO companies (
                        company_name, market_category, market_sector, 
                        data_start_year, data_end_year, current_status
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        row['company_name'],
                        row['market_category'],
                        row['market_sector'],
                        row.get('data_start_year', 1984),
                        row.get('data_end_year', 2024),
                        row.get('current_status', 'active')
                    ))
                
                conn.commit()
                self.logger.info(f"企業データ投入完了: {len(companies_data)}社")
                return True
                
        except Exception as e:
            self.logger.error(f"企業データ投入エラー: {e}")
            return False
    
    def _get_default_companies_data(self) -> pd.DataFrame:
        """デフォルト企業データ取得"""
        
        # 添付ファイルベースの企業リスト（一部抜粋）
        companies_list = [
            # 高シェア市場 - ロボット
            {'company_name': 'ファナック', 'market_category': 'high_share', 'market_sector': 'ロボット'},
            {'company_name': '安川電機', 'market_category': 'high_share', 'market_sector': 'ロボット'},
            {'company_name': '川崎重工業', 'market_category': 'high_share', 'market_sector': 'ロボット'},
            
            # 高シェア市場 - 内視鏡
            {'company_name': 'オリンパス', 'market_category': 'high_share', 'market_sector': '内視鏡'},
            {'company_name': 'HOYA', 'market_category': 'high_share', 'market_sector': '内視鏡'},
            {'company_name': '富士フイルム', 'market_category': 'high_share', 'market_sector': '内視鏡'},
            
            # シェア低下市場 - 自動車
            {'company_name': 'トヨタ自動車', 'market_category': 'declining', 'market_sector': '自動車'},
            {'company_name': '日産自動車', 'market_category': 'declining', 'market_sector': '自動車'},
            {'company_name': 'ホンダ', 'market_category': 'declining', 'market_sector': '自動車'},
            
            # 完全失失市場 - 半導体
            {'company_name': 'ソニー', 'market_category': 'lost_share', 'market_sector': '半導体', 'current_status': 'active'},
            {'company_name': '三洋電機', 'market_category': 'lost_share', 'market_sector': '家電', 'current_status': 'merged', 'data_end_year': 2012},
        ]
        
        return pd.DataFrame(companies_list)
    
    def validate_database(self) -> bool:
        """データベース構造検証"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # テーブル存在確認
                cursor.execute("""
                SELECT name FROM sqlite_master WHERE type='table' ORDER BY name
                """)
                tables = [row[0] for row in cursor.fetchall()]
                
                expected_tables = [
                    'companies', 'market_sectors', 'evaluation_metrics', 'factor_metrics',
                    'financial_statements', 'evaluation_data', 'factor_data',
                    'corporate_events', 'survival_data', 'lifecycle_stages',
                    'market_share_data', 'industry_benchmarks',
                    'analysis_runs', 'causal_effects', 'predictions'
                ]
                
                missing_tables = set(expected_tables) - set(tables)
                if missing_tables:
                    self.logger.error(f"不足テーブル: {missing_tables}")
                    return False
                
                # データ存在確認
                cursor.execute("SELECT COUNT(*) FROM companies")
                company_count = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM evaluation_metrics")
                metrics_count = cursor.fetchone()[0]
                
                self.logger.info(f"検証結果 - 企業数: {company_count}, 評価項目数: {metrics_count}")
                return True
                
        except Exception as e:
            self.logger.error(f"データベース検証エラー: {e}")
            return False
    
    def get_database_info(self) -> Dict[str, Any]:
        """データベース情報取得"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                info = {
                    'database_path': str(self.db_path),
                    'database_size_mb': self.db_path.stat().st_size / (1024 * 1024),
                    'created_at': datetime.fromtimestamp(self.db_path.stat().st_ctime).isoformat(),
                }
                
                # テーブル別レコード数
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [row[0] for row in cursor.fetchall()]
                
                for table in tables:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    count = cursor.fetchone()[0]
                    info[f'{table}_count'] = count
                
                return info
                
        except Exception as e:
            self.logger.error(f"データベース情報取得エラー: {e}")
            return {}


def main():
    """メイン処理"""
    import argparse
    
    parser = argparse.ArgumentParser(description='A2AI Database Setup')
    parser.add_argument('--db-path', help='Database file path')
    parser.add_argument('--companies-csv', help='Companies CSV file path')
    parser.add_argument('--reset', action='store_true', help='Reset existing database')
    parser.add_argument('--validate-only', action='store_true', help='Validate database only')
    parser.add_argument('--info', action='store_true', help='Show database info')
    args = parser.parse_args()
    
    # ロギング設定
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/database_setup.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # データベースセットアップ
    db_setup = A2AIDatabaseSetup(args.db_path)
    
    try:
        if args.info:
            # データベース情報表示
            info = db_setup.get_database_info()
            print("\n=== A2AI Database Information ===")
            for key, value in info.items():
                print(f"{key}: {value}")
            return
        
        if args.validate_only:
            # 検証のみ実行
            if db_setup.validate_database():
                print("✅ データベース検証成功")
                return
            else:
                print("❌ データベース検証失敗")
                sys.exit(1)
        
        # データベース作成
        print("🚀 A2AI データベースセットアップ開始...")
        
        if not db_setup.create_database():
            print("❌ データベース作成失敗")
            sys.exit(1)
        
        # 企業データ投入
        print("📊 企業データ投入中...")
        if not db_setup.insert_company_data(args.companies_csv):
            print("⚠️ 企業データ投入失敗")
        
        # 検証実行
        print("🔍 データベース検証中...")
        if not db_setup.validate_database():
            print("❌ データベース検証失敗")
            sys.exit(1)
        
        # 情報表示
        info = db_setup.get_database_info()
        print("\n✅ A2AI データベースセットアップ完了!")
        print(f"📍 データベースパス: {info['database_path']}")
        print(f"💾 データベースサイズ: {info['database_size_mb']:.2f} MB")
        print(f"🏢 登録企業数: {info.get('companies_count', 0)}")
        print(f"📈 評価項目数: {info.get('evaluation_metrics_count', 0)}")
        print(f"⚙️ 要因項目数: {info.get('factor_metrics_count', 0)}")
        
        print("\n🎯 次のステップ:")
        print("1. collect_all_data.py でデータ収集を開始")
        print("2. preprocess_pipeline.py で前処理パイプライン実行")
        print("3. train_survival_models.py で生存分析モデル構築")
        
    except KeyboardInterrupt:
        print("\n⚠️ ユーザーによる中断")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 予期しないエラー: {e}")
        logging.exception("Setup failed")
        sys.exit(1)


class A2AICompanyDataGenerator:
    """企業データ自動生成クラス（150社完全版）"""
    
    def __init__(self):
        self.logger = setup_logger(__name__, LOGGING_CONFIG)
    
    def generate_complete_company_list(self) -> pd.DataFrame:
        """
        添付ファイルベースの完全な150社企業リスト生成
        
        Returns:
            pd.DataFrame: 150社の完全企業データ
        """
        companies = []
        
        # 高シェア市場 (50社)
        high_share_companies = {
            'ロボット': [
                'ファナック', '安川電機', '川崎重工業', '不二越', 'デンソーウェーブ',
                '三菱電機', 'オムロン', 'THK', 'NSK', 'IHI'
            ],
            '内視鏡': [
                'オリンパス', 'HOYA', '富士フイルム', 'キヤノンメディカルシステムズ',
                '島津製作所', 'コニカミノルタ', 'ソニー', 'トプコン', 'エムスリー', '日立製作所'
            ],
            '工作機械': [
                'DMG森精機', 'ヤマザキマザック', 'オークマ', '牧野フライス製作所',
                'ジェイテクト', '東芝機械', 'アマダ', 'ソディック', '三菱重工工作機械', 'シギヤ精機製作所'
            ],
            '電子材料': [
                '村田製作所', 'TDK', '京セラ', '太陽誘電', '日本特殊陶業',
                'ローム', 'プロテリアル', '住友電工', '日東電工', '日本碍子'
            ],
            '精密測定機器': [
                'キーエンス', '島津製作所', '堀場製作所', '東京精密', 'ミツトヨ',
                'オリンパス', '日本電産', 'リオン', 'アルバック', 'ナブテスコ'
            ]
        }
        
        # シェア低下市場 (50社)
        declining_companies = {
            '自動車': [
                'トヨタ自動車', '日産自動車', 'ホンダ', 'スズキ', 'マツダ',
                'SUBARU', 'いすゞ自動車', '三菱自動車', 'ダイハツ工業', '日野自動車'
            ],
            '鉄鋼': [
                '日本製鉄', 'JFEホールディングス', '神戸製鋼所', '日新製鋼', '大同特殊鋼',
                '山陽特殊製鋼', '愛知製鋼', '中部鋼鈑', '淀川製鋼所', '日立金属'
            ],
            'スマート家電': [
                'パナソニック', 'シャープ', 'ソニー', '東芝ライフスタイル',
                '日立グローバルライフソリューションズ', 'アイリスオーヤマ', '三菱電機',
                '象印マホービン', 'タイガー魔法瓶', '山善'
            ],
            'バッテリー': [
                'パナソニックエナジー', '村田製作所', 'GSユアサ', '東芝インフラシステムズ',
                '日立化成', 'FDK', 'NEC', 'ENAX', '日本電産', 'TDK'
            ],
            'PC・周辺機器': [
                'NECパーソナル', '富士通クライアントコンピューティング', '東芝ダイナブック',
                'ソニーVAIO', 'エレコム', 'バッファロー', 'ロジテック', 'プリンストン',
                'サンワサプライ', 'アイ・オー・データ機器'
            ]
        }
        
        # 完全失失市場 (50社)
        lost_share_companies = {
            '家電': [
                'ソニー', 'パナソニック', 'シャープ', '東芝ライフスタイル', '三菱電機',
                '日立グローバルライフソリューションズ', '三洋電機', 'ビクター', 'アイワ', '船井電機'
            ],
            '半導体': [
                '東芝メモリ', '日立製作所', '三菱電機', 'NEC', '富士通',
                'パナソニック', 'ソニー', 'ルネサスエレクトロニクス', 'シャープ', 'ローム'
            ],
            'スマートフォン': [
                'ソニーXperia', 'シャープAQUOS', '京セラ', 'パナソニック', '富士通arrows',
                'NEC', '日立製作所', '三菱電機', '東芝', 'カシオ計算機'
            ],
            'PC': [
                'ソニーVAIO', 'NEC', '富士通', '東芝dynabook', 'シャープ',
                'パナソニック', '日立製作所', '三菱電機', 'カシオ計算機',
                '日本電気ホームエレクトロニクス'
            ],
            '通信機器': [
                'NEC', '富士通', '日立製作所', 'パナソニック', 'シャープ',
                'ソニーエリクソン', '三菱電機', '京セラ', 'カシオ計算機', '日本無線'
            ]
        }
        
        # データ生成
        company_id = 1
        
        for category, sectors in [
            ('high_share', high_share_companies),
            ('declining', declining_companies),
            ('lost_share', lost_share_companies)
        ]:
            for sector, company_list in sectors.items():
                for company_name in company_list:
                    # 企業の設立年・上場年を推定
                    if category == 'high_share':
                        founded_year = 1950 + (company_id % 40)
                        data_start_year = max(1984, founded_year + 10)
                        data_end_year = 2024
                        status = 'active'
                    elif category == 'declining':
                        founded_year = 1945 + (company_id % 35)
                        data_start_year = max(1984, founded_year + 15)
                        data_end_year = 2024
                        status = 'active'
                    else:  # lost_share
                        founded_year = 1940 + (company_id % 50)
                        data_start_year = max(1984, founded_year + 20)
                        # 一部企業は消滅・統合
                        if company_name in ['三洋電機', 'アイワ', '日本電気ホームエレクトロニクス']:
                            data_end_year = 2000 + (company_id % 20)
                            status = 'merged'
                        elif company_name in ['富士通arrows', 'ソニーエリクソン']:
                            data_end_year = 2015 + (company_id % 8)
                            status = 'bankrupt'
                        else:
                            data_end_year = 2024
                            status = 'active'
                    
                    companies.append({
                        'company_id': company_id,
                        'company_name': company_name,
                        'market_category': category,
                        'market_sector': sector,
                        'founded_year': founded_year,
                        'data_start_year': data_start_year,
                        'data_end_year': data_end_year,
                        'current_status': status,
                        'observation_years': data_end_year - data_start_year + 1
                    })
                    company_id += 1
        
        df = pd.DataFrame(companies)
        self.logger.info(f"完全企業リスト生成完了: {len(df)}社")
        return df
    
    def save_company_list(self, output_path: str = "data/companies_master.csv") -> bool:
        """企業リストをCSVファイルに保存"""
        try:
            df = self.generate_complete_company_list()
            
            # ディレクトリ作成
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # CSV保存
            df.to_csv(output_file, index=False, encoding='utf-8-sig')
            
            # 統計情報出力
            print(f"\n📊 企業リスト統計:")
            print(f"総企業数: {len(df)}")
            print(f"市場カテゴリ別:")
            print(df['market_category'].value_counts())
            print(f"\n企業ステータス別:")
            print(df['current_status'].value_counts())
            print(f"\n平均観測年数: {df['observation_years'].mean():.1f}年")
            
            self.logger.info(f"企業リストCSV保存完了: {output_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"企業リストCSV保存エラー: {e}")
            return False


if __name__ == "__main__":
    main()