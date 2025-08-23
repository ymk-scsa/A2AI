"""
A2AI Data Validator
企業ライフサイクル対応の包括的データ検証システム

このモジュールは以下の検証を実行：
1. 企業存続性チェック（消滅・新設企業対応）
2. 財務データ整合性チェック（120要因項目）
3. 時系列データ連続性チェック（1984-2024年）
4. 市場カテゴリ分類妥当性チェック
5. 生存バイアス検出・対応
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
import logging
from dataclasses import dataclass
from enum import Enum
import yaml
import os
import warnings

class CompanyStatus(Enum):
    """企業ステータス定義"""
    ACTIVE = "active"  # 存続中
    EXTINCT = "extinct"  # 消滅（倒産・吸収）
    SPINOFF = "spinoff"  # 分社化企業
    MERGED = "merged"  # 統合・合併
    ACQUIRED = "acquired"  # 買収

class MarketCategory(Enum):
    """市場カテゴリ定義"""
    HIGH_SHARE = "high_share"  # 現在も高シェア
    DECLINING_SHARE = "declining_share"  # シェア低下中
    LOST_SHARE = "lost_share"  # 完全にシェア失失

class ValidationSeverity(Enum):
    """検証エラーの重要度"""
    CRITICAL = "critical"  # クリティカル（分析不可）
    WARNING = "warning"   # 警告（要注意）
    INFO = "info"        # 情報（軽微）

@dataclass
class ValidationResult:
    """検証結果格納クラス"""
    company_id: str
    validation_type: str
    severity: ValidationSeverity
    message: str
    details: Dict
    timestamp: datetime

@dataclass
class CompanyMetadata:
    """企業メタデータ"""
    company_id: str
    company_name: str
    market_category: MarketCategory
    market_sector: str
    status: CompanyStatus
    establishment_date: Optional[datetime]
    extinction_date: Optional[datetime]
    parent_company: Optional[str]
    spinoff_source: Optional[str]

class DataValidator:
    """
    A2AI包括的データ検証システム
    
    企業ライフサイクル全体（生存・消滅・新設）に対応した
    財務データ検証を実行
    """
    
    def __init__(self, config_path: str = "config/"):
        """
        初期化
        
        Args:
            config_path: 設定ファイルディレクトリパス
        """
        self.config_path = config_path
        self.logger = self._setup_logger()
        
        # 設定ファイル読み込み
        self.market_categories = self._load_config("market_categories.yaml")
        self.evaluation_factors = self._load_config("evaluation_factors.yaml")
        self.lifecycle_stages = self._load_config("lifecycle_stages.yaml")
        
        # 検証ルール定義
        self.validation_rules = self._initialize_validation_rules()
        
        # 検証結果格納
        self.validation_results: List[ValidationResult] = []
        
        # 企業メタデータ
        self.company_metadata: Dict[str, CompanyMetadata] = {}
        
        # データ期間設定
        self.start_year = 1984
        self.end_year = 2024
        self.total_years = self.end_year - self.start_year + 1
        
    def _setup_logger(self) -> logging.Logger:
        """ロガーセットアップ"""
        logger = logging.getLogger("A2AI_DataValidator")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def _load_config(self, filename: str) -> Dict:
        """設定ファイル読み込み"""
        try:
            with open(os.path.join(self.config_path, filename), 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            self.logger.warning(f"設定ファイル {filename} が見つかりません。デフォルト設定を使用します。")
            return self._get_default_config(filename)
    
    def _get_default_config(self, filename: str) -> Dict:
        """デフォルト設定返却"""
        if filename == "market_categories.yaml":
            return {
                "high_share_markets": ["ロボット", "内視鏡", "工作機械", "電子材料", "精密測定機器"],
                "declining_markets": ["自動車", "鉄鋼", "スマート家電", "バッテリー", "PC周辺機器"],
                "lost_markets": ["家電", "半導体", "スマートフォン", "PC", "通信機器"]
            }
        elif filename == "evaluation_factors.yaml":
            return {
                "evaluation_metrics": [
                    "売上高", "売上高成長率", "売上高営業利益率", 
                    "売上高当期純利益率", "ROE", "売上高付加価値率",
                    "企業存続確率", "新規事業成功率", "事業継承成功度"
                ]
            }
        return {}
    
    def _initialize_validation_rules(self) -> Dict:
        """検証ルール初期化"""
        return {
            "financial_data": {
                "required_fields": [
                    "売上高", "総資産", "自己資本", "当期純利益",
                    "営業利益", "売上原価", "販管費"
                ],
                "logical_constraints": {
                    "総資産": {"min": 0},
                    "売上高": {"min": 0},
                    "自己資本比率": {"min": 0, "max": 1},
                    "ROE": {"min": -1, "max": 3}  # -100%～300%
                }
            },
            "time_series": {
                "max_consecutive_missing": 3,  # 3年連続欠損まで許容
                "min_data_points": 5,          # 最低5年分のデータ必要
                "extinction_grace_period": 2   # 消滅後2年までデータ許容
            },
            "lifecycle": {
                "min_establishment_year": 1900,
                "max_future_date_tolerance": 365  # 未来日付365日まで許容
            }
        }
    
    def validate_company_metadata(self, metadata_df: pd.DataFrame) -> List[ValidationResult]:
        """
        企業メタデータ検証
        
        Args:
            metadata_df: 企業メタデータDataFrame
                columns: ['company_id', 'company_name', 'market_category', 
                            'market_sector', 'status', 'establishment_date', 
                            'extinction_date', 'parent_company', 'spinoff_source']
        
        Returns:
            検証結果リスト
        """
        results = []
        
        for _, row in metadata_df.iterrows():
            company_id = row['company_id']
            
            # メタデータ格納
            self.company_metadata[company_id] = CompanyMetadata(
                company_id=company_id,
                company_name=row['company_name'],
                market_category=MarketCategory(row['market_category']),
                market_sector=row['market_sector'],
                status=CompanyStatus(row['status']),
                establishment_date=pd.to_datetime(row.get('establishment_date')),
                extinction_date=pd.to_datetime(row.get('extinction_date')),
                parent_company=row.get('parent_company'),
                spinoff_source=row.get('spinoff_source')
            )
            
            # 1. 基本情報チェック
            results.extend(self._validate_basic_info(company_id, row))
            
            # 2. ライフサイクル整合性チェック
            results.extend(self._validate_lifecycle_consistency(company_id, row))
            
            # 3. 市場カテゴリ妥当性チェック
            results.extend(self._validate_market_category(company_id, row))
            
        return results
    
    def _validate_basic_info(self, company_id: str, row: pd.Series) -> List[ValidationResult]:
        """基本情報検証"""
        results = []
        
        # 必須フィールドチェック
        required_fields = ['company_name', 'market_category', 'market_sector', 'status']
        for field in required_fields:
            if pd.isna(row.get(field)) or row.get(field) == '':
                results.append(ValidationResult(
                    company_id=company_id,
                    validation_type="basic_info",
                    severity=ValidationSeverity.CRITICAL,
                    message=f"必須フィールド '{field}' が空です",
                    details={"field": field, "value": row.get(field)},
                    timestamp=datetime.now()
                ))
        
        # 企業ID重複チェック（後で実装される外部チェックと連携）
        
        return results
    
    def _validate_lifecycle_consistency(self, company_id: str, row: pd.Series) -> List[ValidationResult]:
        """ライフサイクル整合性検証"""
        results = []
        
        establishment_date = pd.to_datetime(row.get('establishment_date'))
        extinction_date = pd.to_datetime(row.get('extinction_date'))
        status = row.get('status')
        
        # 設立日チェック
        if pd.notna(establishment_date):
            if establishment_date.year < self.validation_rules["lifecycle"]["min_establishment_year"]:
                results.append(ValidationResult(
                    company_id=company_id,
                    validation_type="lifecycle",
                    severity=ValidationSeverity.WARNING,
                    message=f"設立年が古すぎます: {establishment_date.year}",
                    details={"establishment_date": establishment_date},
                    timestamp=datetime.now()
                ))
            
            if establishment_date > datetime.now():
                results.append(ValidationResult(
                    company_id=company_id,
                    validation_type="lifecycle",
                    severity=ValidationSeverity.CRITICAL,
                    message="設立日が未来日付です",
                    details={"establishment_date": establishment_date},
                    timestamp=datetime.now()
                ))
        
        # 消滅企業の整合性チェック
        if status in ['extinct', 'merged', 'acquired']:
            if pd.isna(extinction_date):
                results.append(ValidationResult(
                    company_id=company_id,
                    validation_type="lifecycle",
                    severity=ValidationSeverity.CRITICAL,
                    message=f"消滅企業なのに消滅日が設定されていません (status: {status})",
                    details={"status": status, "extinction_date": extinction_date},
                    timestamp=datetime.now()
                ))
            elif extinction_date > datetime.now():
                results.append(ValidationResult(
                    company_id=company_id,
                    validation_type="lifecycle",
                    severity=ValidationSeverity.CRITICAL,
                    message="消滅日が未来日付です",
                    details={"extinction_date": extinction_date},
                    timestamp=datetime.now()
                ))
        
        # 設立日・消滅日の論理関係チェック
        if pd.notna(establishment_date) and pd.notna(extinction_date):
            if establishment_date >= extinction_date:
                results.append(ValidationResult(
                    company_id=company_id,
                    validation_type="lifecycle",
                    severity=ValidationSeverity.CRITICAL,
                    message="設立日が消滅日以降になっています",
                    details={
                        "establishment_date": establishment_date,
                        "extinction_date": extinction_date
                    },
                    timestamp=datetime.now()
                ))
        
        return results
    
    def _validate_market_category(self, company_id: str, row: pd.Series) -> List[ValidationResult]:
        """市場カテゴリ妥当性検証"""
        results = []
        
        market_category = row.get('market_category')
        market_sector = row.get('market_sector')
        status = row.get('status')
        
        # 市場カテゴリと企業ステータスの整合性
        if market_category == 'lost_share' and status == 'active':
            # 完全失失市場で生存企業は例外的ケース
            results.append(ValidationResult(
                company_id=company_id,
                validation_type="market_category",
                severity=ValidationSeverity.WARNING,
                message="完全失失市場で生存企業です（要確認）",
                details={
                    "market_category": market_category,
                    "status": status,
                    "market_sector": market_sector
                },
                timestamp=datetime.now()
            ))
        
        return results
    
    def validate_financial_data(self, financial_df: pd.DataFrame, 
                                company_id: str) -> List[ValidationResult]:
        """
        財務データ検証
        
        Args:
            financial_df: 財務データDataFrame（特定企業の時系列データ）
            company_id: 企業ID
        
        Returns:
            検証結果リスト
        """
        results = []
        
        if company_id not in self.company_metadata:
            results.append(ValidationResult(
                company_id=company_id,
                validation_type="financial_data",
                severity=ValidationSeverity.CRITICAL,
                message="企業メタデータが見つかりません",
                details={},
                timestamp=datetime.now()
            ))
            return results
        
        company_meta = self.company_metadata[company_id]
        
        # 1. 必須フィールド存在チェック
        results.extend(self._validate_required_fields(financial_df, company_id))
        
        # 2. データ型・範囲チェック
        results.extend(self._validate_data_types_and_ranges(financial_df, company_id))
        
        # 3. 時系列連続性チェック
        results.extend(self._validate_time_series_continuity(financial_df, company_id, company_meta))
        
        # 4. 論理整合性チェック
        results.extend(self._validate_logical_consistency(financial_df, company_id))
        
        # 5. 企業ライフサイクル整合性チェック
        results.extend(self._validate_lifecycle_data_consistency(financial_df, company_id, company_meta))
        
        # 6. 外れ値検出
        results.extend(self._detect_outliers(financial_df, company_id))
        
        return results
    
    def _validate_required_fields(self, df: pd.DataFrame, company_id: str) -> List[ValidationResult]:
        """必須フィールド検証"""
        results = []
        required_fields = self.validation_rules["financial_data"]["required_fields"]
        
        missing_fields = [field for field in required_fields if field not in df.columns]
        
        if missing_fields:
            results.append(ValidationResult(
                company_id=company_id,
                validation_type="required_fields",
                severity=ValidationSeverity.CRITICAL,
                message=f"必須フィールドが不足: {missing_fields}",
                details={"missing_fields": missing_fields},
                timestamp=datetime.now()
            ))
        
        return results
    
    def _validate_data_types_and_ranges(self, df: pd.DataFrame, company_id: str) -> List[ValidationResult]:
        """データ型・範囲検証"""
        results = []
        constraints = self.validation_rules["financial_data"]["logical_constraints"]
        
        for field, constraint in constraints.items():
            if field not in df.columns:
                continue
                
            series = df[field]
            
            # 数値型チェック
            non_numeric = series[~pd.to_numeric(series, errors='coerce').notna()]
            if len(non_numeric) > 0:
                results.append(ValidationResult(
                    company_id=company_id,
                    validation_type="data_type",
                    severity=ValidationSeverity.CRITICAL,
                    message=f"非数値データが存在: {field}",
                    details={
                        "field": field,
                        "non_numeric_count": len(non_numeric),
                        "sample_values": non_numeric.head().tolist()
                    },
                    timestamp=datetime.now()
                ))
                continue
            
            # 数値範囲チェック
            numeric_series = pd.to_numeric(series, errors='coerce')
            
            if 'min' in constraint:
                below_min = numeric_series < constraint['min']
                if below_min.any():
                    results.append(ValidationResult(
                        company_id=company_id,
                        validation_type="data_range",
                        severity=ValidationSeverity.WARNING,
                        message=f"最小値を下回る値が存在: {field} < {constraint['min']}",
                        details={
                            "field": field,
                            "min_constraint": constraint['min'],
                            "violations_count": below_min.sum(),
                            "min_value": numeric_series.min()
                        },
                        timestamp=datetime.now()
                    ))
            
            if 'max' in constraint:
                above_max = numeric_series > constraint['max']
                if above_max.any():
                    results.append(ValidationResult(
                        company_id=company_id,
                        validation_type="data_range",
                        severity=ValidationSeverity.WARNING,
                        message=f"最大値を上回る値が存在: {field} > {constraint['max']}",
                        details={
                            "field": field,
                            "max_constraint": constraint['max'],
                            "violations_count": above_max.sum(),
                            "max_value": numeric_series.max()
                        },
                        timestamp=datetime.now()
                    ))
        
        return results
    
    def _validate_time_series_continuity(self, df: pd.DataFrame, company_id: str, 
                                        company_meta: CompanyMetadata) -> List[ValidationResult]:
        """時系列連続性検証"""
        results = []
        
        if 'year' not in df.columns:
            results.append(ValidationResult(
                company_id=company_id,
                validation_type="time_series",
                severity=ValidationSeverity.CRITICAL,
                message="年度列(year)が見つかりません",
                details={},
                timestamp=datetime.now()
            ))
            return results
        
        # 期待される年度範囲計算
        expected_start_year = self.start_year
        expected_end_year = self.end_year
        
        # 企業のライフサイクルに応じた調整
        if company_meta.establishment_date:
            expected_start_year = max(expected_start_year, company_meta.establishment_date.year)
        
        if company_meta.extinction_date:
            expected_end_year = min(expected_end_year, company_meta.extinction_date.year)
        
        # 実際のデータ年度
        actual_years = sorted(df['year'].unique())
        expected_years = list(range(expected_start_year, expected_end_year + 1))
        
        # 欠損年度チェック
        missing_years = set(expected_years) - set(actual_years)
        if missing_years:
            missing_count = len(missing_years)
            max_missing = self.validation_rules["time_series"]["max_consecutive_missing"]
            
            # 連続欠損年数計算
            consecutive_missing = self._find_consecutive_missing_years(expected_years, actual_years)
            
            severity = ValidationSeverity.CRITICAL if consecutive_missing > max_missing else ValidationSeverity.WARNING
            
            results.append(ValidationResult(
                company_id=company_id,
                validation_type="time_series",
                severity=severity,
                message=f"欠損年度が存在: {missing_count}年間, 最大連続{consecutive_missing}年",
                details={
                    "missing_years": sorted(missing_years),
                    "missing_count": missing_count,
                    "consecutive_missing": consecutive_missing,
                    "expected_range": f"{expected_start_year}-{expected_end_year}"
                },
                timestamp=datetime.now()
            ))
        
        # 最小データポイントチェック
        min_required = self.validation_rules["time_series"]["min_data_points"]
        if len(actual_years) < min_required:
            results.append(ValidationResult(
                company_id=company_id,
                validation_type="time_series",
                severity=ValidationSeverity.CRITICAL,
                message=f"データポイント数が不足: {len(actual_years)} < {min_required}",
                details={
                    "actual_data_points": len(actual_years),
                    "required_minimum": min_required,
                    "actual_years": actual_years
                },
                timestamp=datetime.now()
            ))
        
        return results
    
    def _find_consecutive_missing_years(self, expected_years: List[int], 
                                        actual_years: List[int]) -> int:
        """連続欠損年数計算"""
        actual_set = set(actual_years)
        max_consecutive = 0
        current_consecutive = 0
        
        for year in expected_years:
            if year not in actual_set:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive
    
    def _validate_logical_consistency(self, df: pd.DataFrame, company_id: str) -> List[ValidationResult]:
        """論理整合性検証"""
        results = []
        
        # 基本的な会計等式チェック（簡略版）
        if all(col in df.columns for col in ['総資産', '負債', '自己資本']):
            # 総資産 = 負債 + 自己資本（許容誤差1%）
            calculated_assets = df['負債'] + df['自己資本']
            asset_diff = np.abs(df['総資産'] - calculated_assets) / df['総資産']
            
            inconsistent_rows = asset_diff > 0.01  # 1%以上の誤差
            if inconsistent_rows.any():
                results.append(ValidationResult(
                    company_id=company_id,
                    validation_type="logical_consistency",
                    severity=ValidationSeverity.WARNING,
                    message="会計等式の不整合が検出されました（総資産 ≠ 負債 + 自己資本）",
                    details={
                        "inconsistent_years": df.loc[inconsistent_rows, 'year'].tolist(),
                        "max_deviation": asset_diff.max(),
                        "affected_rows": inconsistent_rows.sum()
                    },
                    timestamp=datetime.now()
                ))
        
        # 利益系の論理チェック
        if all(col in df.columns for col in ['売上高', '売上原価', '売上総利益']):
            # 売上総利益 = 売上高 - 売上原価
            calculated_gross_profit = df['売上高'] - df['売上原価']
            profit_diff = np.abs(df['売上総利益'] - calculated_gross_profit)
            
            # 売上高が0の行は除外
            valid_rows = df['売上高'] > 0
            if valid_rows.any():
                relative_diff = profit_diff[valid_rows] / df.loc[valid_rows, '売上高']
                inconsistent = relative_diff > 0.01
                
                if inconsistent.any():
                    results.append(ValidationResult(
                        company_id=company_id,
                        validation_type="logical_consistency",
                        severity=ValidationSeverity.WARNING,
                        message="売上総利益の計算不整合が検出されました",
                        details={
                            "inconsistent_years": df.loc[valid_rows].loc[inconsistent, 'year'].tolist(),
                            "max_deviation": relative_diff.max()
                        },
                        timestamp=datetime.now()
                    ))
        
        return results
    
    def _validate_lifecycle_data_consistency(self, df: pd.DataFrame, company_id: str,
                                            company_meta: CompanyMetadata) -> List[ValidationResult]:
        """企業ライフサイクルとデータ整合性検証"""
        results = []
        
        if 'year' not in df.columns:
            return results
        
        data_years = df['year'].tolist()
        
        # 設立前データチェック
        if company_meta.establishment_date:
            establishment_year = company_meta.establishment_date.year
            pre_establishment_data = [y for y in data_years if y < establishment_year]
            
            if pre_establishment_data:
                results.append(ValidationResult(
                    company_id=company_id,
                    validation_type="lifecycle_consistency",
                    severity=ValidationSeverity.CRITICAL,
                    message=f"設立前のデータが存在: 設立年{establishment_year}",
                    details={
                        "establishment_year": establishment_year,
                        "pre_establishment_years": pre_establishment_data
                    },
                    timestamp=datetime.now()
                ))
        
        # 消滅後データチェック
        if company_meta.extinction_date:
            extinction_year = company_meta.extinction_date.year
            grace_period = self.validation_rules["time_series"]["extinction_grace_period"]
            cutoff_year = extinction_year + grace_period
            
            post_extinction_data = [y for y in data_years if y > cutoff_year]
            
            if post_extinction_data:
                results.append(ValidationResult(
                    company_id=company_id,
                    validation_type="lifecycle_consistency",
                    severity=ValidationSeverity.WARNING,
                    message=f"消滅後のデータが存在: 消滅年{extinction_year}",
                    details={
                        "extinction_year": extinction_year,
                        "grace_period": grace_period,
                        "post_extinction_years": post_extinction_data
                    },
                    timestamp=datetime.now()
                ))
        
        return results
    
    def _detect_outliers(self, df: pd.DataFrame, company_id: str) -> List[ValidationResult]:
        """外れ値検出"""
        results = []
        
        # 数値列のみ対象
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            if column == 'year':  # 年度列は除外
                continue
                
            series = df[column].dropna()
            if len(series) < 4:  # データ不足の場合はスキップ
                continue
            
            # IQR法による外れ値検出
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR  # 3×IQR（極端な外れ値）
            upper_bound = Q3 + 3 * IQR
            
            outliers = series[(series < lower_bound) | (series > upper_bound)]
            
            if len(outliers) > 0:
                outlier_years = df.loc[outliers.index, 'year'].tolist() if 'year' in df.columns else outliers.index.tolist()
                
                results.append(ValidationResult(
                    company_id=company_id,
                    validation_type="outliers",
                    severity=ValidationSeverity.INFO,
                    message=f"外れ値が検出されました: {column}",
                    details={
                        "column": column,
                        "outlier_count": len(outliers),
                        "outlier_years": outlier_years,
                        "outlier_values": outliers.tolist(),
                        "bounds": {"lower": lower_bound, "upper": upper_bound}
                    },
                    timestamp=datetime.now()
                ))
        
        return results
    
    def validate_market_share_data(self, market_share_df: pd.DataFrame) -> List[ValidationResult]:
        """
        市場シェアデータ検証
        
        Args:
            market_share_df: 市場シェアDataFrame
                columns: ['market_sector', 'year', 'company_id', 'market_share', 'global_market_size']
        
        Returns:
            検証結果リスト
        """
        results = []
        
        # 必須列チェック
        required_columns = ['market_sector', 'year', 'company_id', 'market_share']
        missing_columns = [col for col in required_columns if col not in market_share_df.columns]
        
        if missing_columns:
            results.append(ValidationResult(
                company_id="MARKET_SHARE",
                validation_type="market_share",
                severity=ValidationSeverity.CRITICAL,
                message=f"市場シェアデータの必須列が不足: {missing_columns}",
                details={"missing_columns": missing_columns},
                timestamp=datetime.now()
            ))
            return results
        
        # 市場シェア値の妥当性チェック
        invalid_share = (market_share_df['market_share'] < 0) | (market_share_df['market_share'] > 100)
        if invalid_share.any():
            results.append(ValidationResult(
                company_id="MARKET_SHARE",
                validation_type="market_share",
                severity=ValidationSeverity.CRITICAL,
                message="無効な市場シェア値が存在します（0-100%範囲外）",
                details={
                    "invalid_count": invalid_share.sum(),
                    "sample_invalid": market_share_df.loc[invalid_share].head().to_dict('records')
                },
                timestamp=datetime.now()
            ))
        
        # 同一市場・年度での合計シェアチェック
        market_year_totals = market_share_df.groupby(['market_sector', 'year'])['market_share'].sum()
        excessive_totals = market_year_totals[market_year_totals > 105]  # 105%まで許容（計測誤差考慮）
        
        if len(excessive_totals) > 0:
            results.append(ValidationResult(
                company_id="MARKET_SHARE",
                validation_type="market_share",
                severity=ValidationSeverity.WARNING,
                message="市場シェア合計が100%を大幅に超過している市場・年度があります",
                details={
                    "excessive_cases": excessive_totals.to_dict(),
                    "max_total": excessive_totals.max()
                },
                timestamp=datetime.now()
            ))
        
        return results
    
    def validate_survival_data_consistency(self, financial_data: Dict[str, pd.DataFrame]) -> List[ValidationResult]:
        """
        生存分析データ整合性検証
        
        Args:
            financial_data: {company_id: DataFrame} 形式の財務データ辞書
        
        Returns:
            検証結果リスト
        """
        results = []
        
        for company_id, df in financial_data.items():
            if company_id not in self.company_metadata:
                continue
                
            company_meta = self.company_metadata[company_id]
            
            # 生存バイアス検出
            if company_meta.status == CompanyStatus.ACTIVE:
                # 現在存続企業のデータ終了年チェック
                if 'year' in df.columns:
                    last_data_year = df['year'].max()
                    current_year = datetime.now().year
                    
                    if last_data_year < current_year - 2:  # 2年以上古い
                        results.append(ValidationResult(
                            company_id=company_id,
                            validation_type="survival_bias",
                            severity=ValidationSeverity.WARNING,
                            message=f"存続企業なのに最新データが古い: 最終年度{last_data_year}",
                            details={
                                "last_data_year": last_data_year,
                                "current_year": current_year,
                                "data_lag": current_year - last_data_year
                            },
                            timestamp=datetime.now()
                        ))
            
            elif company_meta.status in [CompanyStatus.EXTINCT, CompanyStatus.MERGED, CompanyStatus.ACQUIRED]:
                # 消滅企業のデータ整合性
                if company_meta.extinction_date and 'year' in df.columns:
                    extinction_year = company_meta.extinction_date.year
                    last_data_year = df['year'].max()
                    
                    # 消滅年以降のデータ存在チェック
                    if last_data_year > extinction_year + 1:  # 1年の猶予
                        results.append(ValidationResult(
                            company_id=company_id,
                            validation_type="survival_bias",
                            severity=ValidationSeverity.CRITICAL,
                            message=f"消滅企業なのに消滅後のデータが存在: 消滅年{extinction_year}, 最終データ年{last_data_year}",
                            details={
                                "extinction_year": extinction_year,
                                "last_data_year": last_data_year,
                                "status": company_meta.status.value
                            },
                            timestamp=datetime.now()
                        ))
                    
                    # 消滅直前の財務悪化パターンチェック（分析価値確認）
                    extinction_period_data = df[df['year'] >= extinction_year - 3]  # 消滅前3年
                    if len(extinction_period_data) >= 2:
                        if '当期純利益' in df.columns:
                            recent_losses = extinction_period_data['当期純利益'] < 0
                            if recent_losses.sum() == 0:  # 消滅前に赤字なし
                                results.append(ValidationResult(
                                    company_id=company_id,
                                    validation_type="survival_pattern",
                                    severity=ValidationSeverity.INFO,
                                    message="消滅企業だが消滅直前に赤字なし（分析要注意）",
                                    details={
                                        "extinction_year": extinction_year,
                                        "pre_extinction_profits": extinction_period_data['当期純利益'].tolist()
                                    },
                                    timestamp=datetime.now()
                                ))
        
        return results
    
    def validate_emergence_data_patterns(self, financial_data: Dict[str, pd.DataFrame]) -> List[ValidationResult]:
        """
        新設企業データパターン検証
        
        Args:
            financial_data: {company_id: DataFrame} 形式の財務データ辞書
        
        Returns:
            検証結果リスト
        """
        results = []
        
        for company_id, df in financial_data.items():
            if company_id not in self.company_metadata:
                continue
                
            company_meta = self.company_metadata[company_id]
            
            # 新設・分社企業のデータパターンチェック
            if company_meta.status == CompanyStatus.SPINOFF:
                if 'year' in df.columns and company_meta.establishment_date:
                    establishment_year = company_meta.establishment_date.year
                    early_years_data = df[df['year'] <= establishment_year + 3]  # 設立後3年
                    
                    if len(early_years_data) >= 2:
                        # 分社企業の初期成長パターンチェック
                        if '売上高' in df.columns:
                            early_revenues = early_years_data['売上高'].values
                            if len(early_revenues) >= 2:
                                # 初年度が異常に高い場合（親会社からの事業移管）
                                first_year_revenue = early_revenues[0]
                                if first_year_revenue > 1000000000:  # 10億円以上
                                    results.append(ValidationResult(
                                        company_id=company_id,
                                        validation_type="emergence_pattern",
                                        severity=ValidationSeverity.INFO,
                                        message="分社企業の初年度売上が高額（事業移管パターン）",
                                        details={
                                            "establishment_year": establishment_year,
                                            "first_year_revenue": first_year_revenue,
                                            "parent_company": company_meta.parent_company
                                        },
                                        timestamp=datetime.now()
                                    ))
                                
                                # 初期成長率チェック
                                if len(early_revenues) >= 3:
                                    growth_rates = [(early_revenues[i+1] - early_revenues[i]) / early_revenues[i] 
                                                    for i in range(len(early_revenues)-1) if early_revenues[i] > 0]
                                    if growth_rates and max(growth_rates) > 10:  # 1000%成長
                                        results.append(ValidationResult(
                                            company_id=company_id,
                                            validation_type="emergence_pattern",
                                            severity=ValidationSeverity.WARNING,
                                            message="新設企業の異常な高成長率が検出されました",
                                            details={
                                                "max_growth_rate": max(growth_rates),
                                                "early_revenues": early_revenues.tolist()
                                            },
                                            timestamp=datetime.now()
                                        ))
            
            # 設立年のデータ品質チェック
            if company_meta.establishment_date and 'year' in df.columns:
                establishment_year = company_meta.establishment_date.year
                establishment_data = df[df['year'] == establishment_year]
                
                if len(establishment_data) == 1:
                    # 設立年の財務データ妥当性
                    establishment_row = establishment_data.iloc[0]
                    
                    # 設立年なのに過去データのような規模
                    if '総資産' in establishment_row:
                        total_assets = establishment_row['総資産']
                        if pd.notna(total_assets) and total_assets > 50000000000:  # 500億円以上
                            results.append(ValidationResult(
                                company_id=company_id,
                                validation_type="emergence_pattern",
                                severity=ValidationSeverity.WARNING,
                                message="設立年の総資産が大規模（要確認）",
                                details={
                                    "establishment_year": establishment_year,
                                    "total_assets": total_assets,
                                    "status": company_meta.status.value
                                },
                                timestamp=datetime.now()
                            ))
        
        return results
    
    def comprehensive_validation(self, metadata_df: pd.DataFrame, 
                                financial_data: Dict[str, pd.DataFrame],
                                market_share_df: Optional[pd.DataFrame] = None) -> Dict:
        """
        包括的データ検証実行
        
        Args:
            metadata_df: 企業メタデータ
            financial_data: 財務データ辞書 {company_id: DataFrame}
            market_share_df: 市場シェアデータ（オプション）
        
        Returns:
            検証結果サマリー辞書
        """
        self.logger.info("A2AI包括的データ検証を開始します...")
        
        all_results = []
        
        # 1. 企業メタデータ検証
        self.logger.info("企業メタデータを検証中...")
        metadata_results = self.validate_company_metadata(metadata_df)
        all_results.extend(metadata_results)
        
        # 2. 財務データ検証
        self.logger.info("財務データを検証中...")
        for company_id, df in financial_data.items():
            financial_results = self.validate_financial_data(df, company_id)
            all_results.extend(financial_results)
        
        # 3. 市場シェアデータ検証
        if market_share_df is not None:
            self.logger.info("市場シェアデータを検証中...")
            market_results = self.validate_market_share_data(market_share_df)
            all_results.extend(market_results)
        
        # 4. 生存分析データ整合性検証
        self.logger.info("生存分析データ整合性を検証中...")
        survival_results = self.validate_survival_data_consistency(financial_data)
        all_results.extend(survival_results)
        
        # 5. 新設企業データパターン検証
        self.logger.info("新設企業データパターンを検証中...")
        emergence_results = self.validate_emergence_data_patterns(financial_data)
        all_results.extend(emergence_results)
        
        # 検証結果格納
        self.validation_results = all_results
        
        # サマリー生成
        summary = self._generate_validation_summary(all_results)
        
        self.logger.info(f"検証完了: {len(all_results)}件の検証結果")
        self.logger.info(f"Critical: {summary['critical_count']}, Warning: {summary['warning_count']}, Info: {summary['info_count']}")
        
        return summary
    
    def _generate_validation_summary(self, results: List[ValidationResult]) -> Dict:
        """検証結果サマリー生成"""
        summary = {
            "total_validations": len(results),
            "critical_count": len([r for r in results if r.severity == ValidationSeverity.CRITICAL]),
            "warning_count": len([r for r in results if r.severity == ValidationSeverity.WARNING]),
            "info_count": len([r for r in results if r.severity == ValidationSeverity.INFO]),
            "companies_with_issues": len(set([r.company_id for r in results])),
            "validation_types": {},
            "companies_by_severity": {"critical": set(), "warning": set(), "info": set()},
            "market_categories_analysis": {}
        }
        
        # 検証タイプ別集計
        for result in results:
            validation_type = result.validation_type
            if validation_type not in summary["validation_types"]:
                summary["validation_types"][validation_type] = {"critical": 0, "warning": 0, "info": 0}
            
            summary["validation_types"][validation_type][result.severity.value] += 1
            summary["companies_by_severity"][result.severity.value].add(result.company_id)
        
        # 市場カテゴリ別分析
        for company_id, meta in self.company_metadata.items():
            category = meta.market_category.value
            if category not in summary["market_categories_analysis"]:
                summary["market_categories_analysis"][category] = {
                    "total_companies": 0,
                    "companies_with_critical": 0,
                    "companies_with_warnings": 0,
                    "avg_data_quality_score": 0
                }
            
            summary["market_categories_analysis"][category]["total_companies"] += 1
            
            company_results = [r for r in results if r.company_id == company_id]
            has_critical = any(r.severity == ValidationSeverity.CRITICAL for r in company_results)
            has_warning = any(r.severity == ValidationSeverity.WARNING for r in company_results)
            
            if has_critical:
                summary["market_categories_analysis"][category]["companies_with_critical"] += 1
            if has_warning:
                summary["market_categories_analysis"][category]["companies_with_warnings"] += 1
        
        # データ品質スコア計算（100点満点）
        for category in summary["market_categories_analysis"]:
            cat_data = summary["market_categories_analysis"][category]
            if cat_data["total_companies"] > 0:
                critical_penalty = (cat_data["companies_with_critical"] / cat_data["total_companies"]) * 50
                warning_penalty = (cat_data["companies_with_warnings"] / cat_data["total_companies"]) * 20
                cat_data["avg_data_quality_score"] = max(0, 100 - critical_penalty - warning_penalty)
        
        return summary
    
    def export_validation_report(self, output_path: str, format: str = "excel") -> str:
        """
        検証レポート出力
        
        Args:
            output_path: 出力パス
            format: 出力形式 ("excel", "csv", "json")
        
        Returns:
            出力ファイルパス
        """
        if not self.validation_results:
            raise ValueError("検証結果がありません。comprehensive_validationを先に実行してください。")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format == "excel":
            output_file = f"{output_path}/A2AI_validation_report_{timestamp}.xlsx"
            
            # DataFrameに変換
            results_data = []
            for result in self.validation_results:
                results_data.append({
                    "企業ID": result.company_id,
                    "検証タイプ": result.validation_type,
                    "重要度": result.severity.value,
                    "メッセージ": result.message,
                    "詳細": str(result.details),
                    "検証時刻": result.timestamp
                })
            
            results_df = pd.DataFrame(results_data)
            
            # Excelファイルに複数シート出力
            with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
                # 検証結果シート
                results_df.to_excel(writer, sheet_name='検証結果', index=False)
                
                # サマリーシート
                summary = self._generate_validation_summary(self.validation_results)
                summary_data = []
                for key, value in summary.items():
                    if isinstance(value, (int, float, str)):
                        summary_data.append({"項目": key, "値": value})
                
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='サマリー', index=False)
                
                # 企業別集計シート
                company_summary = self._generate_company_summary()
                company_df = pd.DataFrame(company_summary)
                company_df.to_excel(writer, sheet_name='企業別集計', index=False)
            
        elif format == "csv":
            output_file = f"{output_path}/A2AI_validation_results_{timestamp}.csv"
            results_data = []
            for result in self.validation_results:
                results_data.append({
                    "company_id": result.company_id,
                    "validation_type": result.validation_type,
                    "severity": result.severity.value,
                    "message": result.message,
                    "details": str(result.details),
                    "timestamp": result.timestamp
                })
            
            results_df = pd.DataFrame(results_data)
            results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
            
        elif format == "json":
            output_file = f"{output_path}/A2AI_validation_results_{timestamp}.json"
            results_data = []
            for result in self.validation_results:
                results_data.append({
                    "company_id": result.company_id,
                    "validation_type": result.validation_type,
                    "severity": result.severity.value,
                    "message": result.message,
                    "details": result.details,
                    "timestamp": result.timestamp.isoformat()
                })
            
            import json
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results_data, f, ensure_ascii=False, indent=2)
        
        else:
            raise ValueError(f"サポートされていない形式: {format}")
        
        self.logger.info(f"検証レポートを出力しました: {output_file}")
        return output_file
    
    def _generate_company_summary(self) -> List[Dict]:
        """企業別検証結果サマリー生成"""
        company_summary = []
        
        for company_id, meta in self.company_metadata.items():
            company_results = [r for r in self.validation_results if r.company_id == company_id]
            
            critical_count = len([r for r in company_results if r.severity == ValidationSeverity.CRITICAL])
            warning_count = len([r for r in company_results if r.severity == ValidationSeverity.WARNING])
            info_count = len([r for r in company_results if r.severity == ValidationSeverity.INFO])
            
            # データ品質スコア（0-100）
            quality_score = max(0, 100 - critical_count * 20 - warning_count * 5)
            
            company_summary.append({
                "企業ID": company_id,
                "企業名": meta.company_name,
                "市場カテゴリ": meta.market_category.value,
                "市場セクター": meta.market_sector,
                "企業ステータス": meta.status.value,
                "Critical数": critical_count,
                "Warning数": warning_count,
                "Info数": info_count,
                "品質スコア": quality_score,
                "設立年": meta.establishment_date.year if meta.establishment_date else None,
                "消滅年": meta.extinction_date.year if meta.extinction_date else None
            })
        
        return company_summary
    
    def get_companies_ready_for_analysis(self, min_quality_score: int = 70) -> List[str]:
        """
        分析準備完了企業リスト取得
        
        Args:
            min_quality_score: 最低品質スコア
        
        Returns:
            分析可能企業IDリスト
        """
        if not self.validation_results:
            raise ValueError("検証を先に実行してください。")
        
        company_summary = self._generate_company_summary()
        ready_companies = []
        
        for company in company_summary:
            # クリティカルエラーがなく、品質スコアが基準以上
            if company["Critical数"] == 0 and company["品質スコア"] >= min_quality_score:
                ready_companies.append(company["企業ID"])
        
        self.logger.info(f"分析準備完了企業数: {len(ready_companies)}/{len(company_summary)}")
        return ready_companies
    
    def get_survival_bias_affected_companies(self) -> Dict[str, List[str]]:
        """
        生存バイアス影響企業の特定
        
        Returns:
            バイアス種類別企業リスト
        """
        bias_companies = {
            "missing_extinction_data": [],  # 消滅データ不足
            "late_data_cutoff": [],         # データ終了が早い
            "survivor_only": [],            # 生存企業のみ
            "extinction_anomaly": []        # 消滅パターン異常
        }
        
        for result in self.validation_results:
            if result.validation_type == "survival_bias":
                if "存続企業なのに最新データが古い" in result.message:
                    bias_companies["late_data_cutoff"].append(result.company_id)
                elif "消滅企業なのに消滅後のデータが存在" in result.message:
                    bias_companies["extinction_anomaly"].append(result.company_id)
            elif result.validation_type == "survival_pattern":
                if "消滅企業だが消滅直前に赤字なし" in result.message:
                    bias_companies["extinction_anomaly"].append(result.company_id)
        
        return bias_companies