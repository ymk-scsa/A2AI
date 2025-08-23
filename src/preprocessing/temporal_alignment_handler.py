"""
A2AI - Advanced Financial Analysis AI
Temporal Alignment Handler

異なる存続期間を持つ企業の財務データを時系列で適切に整合させるモジュール。
- 継続存続企業（1984-2024年）
- 企業消滅・吸収企業（1984年-消滅年まで）
- 新設・分社企業（設立年-2024年）
の3つの企業タイプのデータを統合分析可能な形に変換する。
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
import warnings
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class CompanyLifecycleType(Enum):
    """企業ライフサイクルタイプ"""
    CONTINUOUS = "continuous"  # 継続存続企業
    EXTINCT = "extinct"        # 消滅企業
    EMERGED = "emerged"        # 新設企業


@dataclass
class CompanyTimeProfile:
    """企業の時系列プロファイル"""
    company_id: str
    company_name: str
    lifecycle_type: CompanyLifecycleType
    start_year: int
    end_year: int
    data_start_year: int  # 実際のデータ開始年
    data_end_year: int    # 実際のデータ終了年
    missing_periods: List[int]  # 欠損期間
    market_category: str  # high_share, declining, lost


class TemporalAlignmentHandler:
    """
    異なる存続期間企業の時系列データ整合ハンドラー
    """
    
    def __init__(self, 
                    base_start_year: int = 1984,
                    base_end_year: int = 2024,
                    min_data_years: int = 5):
        """
        Args:
            base_start_year: 基準開始年（1984年）
            base_end_year: 基準終了年（2024年）
            min_data_years: 分析に必要な最小データ年数
        """
        self.base_start_year = base_start_year
        self.base_end_year = base_end_year
        self.min_data_years = min_data_years
        self.base_years = list(range(base_start_year, base_end_year + 1))
        self.company_profiles: Dict[str, CompanyTimeProfile] = {}
        
    def create_company_profile(self,
                                company_id: str,
                                company_name: str,
                                market_category: str,
                                start_year: Optional[int] = None,
                                end_year: Optional[int] = None,
                                data_years: Optional[List[int]] = None) -> CompanyTimeProfile:
        """
        企業の時系列プロファイルを作成
        
        Args:
            company_id: 企業ID
            company_name: 企業名
            market_category: 市場カテゴリ
            start_year: 設立年（None=1984年以前設立）
            end_year: 消滅年（None=継続中）
            data_years: 実際にデータがある年のリスト
            
        Returns:
            CompanyTimeProfile: 企業時系列プロファイル
        """
        # ライフサイクルタイプを判定
        if start_year is None and end_year is None:
            lifecycle_type = CompanyLifecycleType.CONTINUOUS
            start_year = self.base_start_year
            end_year = self.base_end_year
        elif end_year is not None and end_year < self.base_end_year:
            lifecycle_type = CompanyLifecycleType.EXTINCT
            start_year = start_year or self.base_start_year
        elif start_year is not None and start_year > self.base_start_year:
            lifecycle_type = CompanyLifecycleType.EMERGED
            end_year = end_year or self.base_end_year
        else:
            lifecycle_type = CompanyLifecycleType.CONTINUOUS
            start_year = start_year or self.base_start_year
            end_year = end_year or self.base_end_year
            
        # 実際のデータ期間を決定
        if data_years is not None:
            data_start_year = min(data_years)
            data_end_year = max(data_years)
            expected_years = set(range(start_year, end_year + 1))
            actual_years = set(data_years)
            missing_periods = sorted(list(expected_years - actual_years))
        else:
            data_start_year = start_year
            data_end_year = end_year
            missing_periods = []
            
        profile = CompanyTimeProfile(
            company_id=company_id,
            company_name=company_name,
            lifecycle_type=lifecycle_type,
            start_year=start_year,
            end_year=end_year,
            data_start_year=data_start_year,
            data_end_year=data_end_year,
            missing_periods=missing_periods,
            market_category=market_category
        )
        
        self.company_profiles[company_id] = profile
        return profile
    
    def align_company_data(self,
                            company_data: pd.DataFrame,
                            company_id: str,
                            time_column: str = 'year') -> pd.DataFrame:
        """
        単一企業データを基準時系列に整合
        
        Args:
            company_data: 企業の財務データ
            company_id: 企業ID
            time_column: 時間列の名前
            
        Returns:
            pd.DataFrame: 整合済みデータ
        """
        if company_id not in self.company_profiles:
            raise ValueError(f"Company profile not found for {company_id}")
            
        profile = self.company_profiles[company_id]
        
        # 基準時系列データフレームを作成
        aligned_df = pd.DataFrame({
            time_column: self.base_years,
            'company_id': company_id,
            'company_name': profile.company_name,
            'lifecycle_type': profile.lifecycle_type.value,
            'market_category': profile.market_category
        })
        
        # 企業の存在期間フラグを追加
        aligned_df['exists'] = (
            (aligned_df[time_column] >= profile.start_year) & 
            (aligned_df[time_column] <= profile.end_year)
        )
        
        # 企業ライフサイクル段階を追加
        aligned_df['lifecycle_stage'] = self._determine_lifecycle_stage(
            aligned_df[time_column], profile
        )
        
        # 実際のデータとマージ
        if not company_data.empty:
            # company_dataの時間列を統一
            if time_column in company_data.columns:
                company_data = company_data.copy()
                company_data[time_column] = pd.to_numeric(company_data[time_column], errors='coerce')
                
                # 基準期間内のデータのみ保持
                company_data = company_data[
                    (company_data[time_column] >= self.base_start_year) &
                    (company_data[time_column] <= self.base_end_year)
                ]
                
                aligned_df = aligned_df.merge(
                    company_data, 
                    on=time_column, 
                    how='left',
                    suffixes=('', '_original')
                )
        
        # データ品質指標を追加
        aligned_df['data_availability'] = self._calculate_data_availability(
            aligned_df, profile
        )
        
        return aligned_df
    
    def align_multiple_companies(self,
                                companies_data: Dict[str, pd.DataFrame],
                                time_column: str = 'year') -> pd.DataFrame:
        """
        複数企業データを統合・整合
        
        Args:
            companies_data: {company_id: DataFrame} の辞書
            time_column: 時間列の名前
            
        Returns:
            pd.DataFrame: 全企業統合済みデータ
        """
        aligned_companies = []
        
        for company_id, company_df in companies_data.items():
            try:
                aligned_df = self.align_company_data(company_df, company_id, time_column)
                aligned_companies.append(aligned_df)
            except Exception as e:
                logger.warning(f"Failed to align data for {company_id}: {str(e)}")
                continue
                
        if not aligned_companies:
            return pd.DataFrame()
            
        # 全企業データを統合
        combined_df = pd.concat(aligned_companies, ignore_index=True)
        
        # 企業間の時系列整合性チェック
        combined_df = self._validate_temporal_consistency(combined_df, time_column)
        
        return combined_df
    
    def create_panel_data(self,
                            companies_data: Dict[str, pd.DataFrame],
                            time_column: str = 'year') -> pd.DataFrame:
        """
        パネルデータ形式で整合済みデータを作成
        
        Args:
            companies_data: 企業データ辞書
            time_column: 時間列名
            
        Returns:
            pd.DataFrame: パネル形式の整合済みデータ
        """
        aligned_df = self.align_multiple_companies(companies_data, time_column)
        
        if aligned_df.empty:
            return pd.DataFrame()
            
        # パネルデータ用のインデックス追加
        aligned_df['panel_id'] = aligned_df.groupby('company_id').ngroup()
        aligned_df['time_id'] = aligned_df.groupby(time_column).ngroup()
        
        # 生存バイアス補正用の重みを計算
        aligned_df['survival_weight'] = self._calculate_survival_weights(aligned_df)
        
        # 時系列順でソート
        aligned_df = aligned_df.sort_values([time_column, 'company_id']).reset_index(drop=True)
        
        return aligned_df
    
    def handle_missing_periods(self,
                                df: pd.DataFrame,
                                method: str = 'interpolate',
                                company_column: str = 'company_id',
                                time_column: str = 'year') -> pd.DataFrame:
        """
        欠損期間の処理
        
        Args:
            df: データフレーム
            method: 補完方法 ('interpolate', 'forward_fill', 'backward_fill', 'drop')
            company_column: 企業列名
            time_column: 時間列名
            
        Returns:
            pd.DataFrame: 欠損処理済みデータ
        """
        df_processed = df.copy()
        
        # 数値列を特定
        numeric_columns = df_processed.select_dtypes(include=[np.number]).columns.tolist()
        if time_column in numeric_columns:
            numeric_columns.remove(time_column)
            
        for company_id in df_processed[company_column].unique():
            company_mask = df_processed[company_column] == company_id
            company_data = df_processed[company_mask].sort_values(time_column)
            
            if method == 'interpolate':
                # 線形補間
                df_processed.loc[company_mask, numeric_columns] = (
                    company_data[numeric_columns].interpolate(method='linear')
                )
            elif method == 'forward_fill':
                # 前方補完
                df_processed.loc[company_mask, numeric_columns] = (
                    company_data[numeric_columns].fillna(method='ffill')
                )
            elif method == 'backward_fill':
                # 後方補完
                df_processed.loc[company_mask, numeric_columns] = (
                    company_data[numeric_columns].fillna(method='bfill')
                )
            elif method == 'drop':
                # 欠損行削除
                continue
            else:
                raise ValueError(f"Unknown method: {method}")
                
        if method == 'drop':
            df_processed = df_processed.dropna(subset=numeric_columns)
            
        return df_processed
    
    def create_lifecycle_features(self,
                                df: pd.DataFrame,
                                company_column: str = 'company_id',
                                time_column: str = 'year') -> pd.DataFrame:
        """
        ライフサイクル関連特徴量を作成
        
        Args:
            df: データフレーム
            company_column: 企業列名
            time_column: 時間列名
            
        Returns:
            pd.DataFrame: ライフサイクル特徴量追加済みデータ
        """
        df_features = df.copy()
        
        for company_id in df_features[company_column].unique():
            if company_id not in self.company_profiles:
                continue
                
            profile = self.company_profiles[company_id]
            company_mask = df_features[company_column] == company_id
            
            # 企業年齢
            df_features.loc[company_mask, 'company_age'] = (
                df_features.loc[company_mask, time_column] - profile.start_year + 1
            )
            
            # 残存期間（消滅企業の場合）
            if profile.lifecycle_type == CompanyLifecycleType.EXTINCT:
                df_features.loc[company_mask, 'remaining_years'] = (
                    profile.end_year - df_features.loc[company_mask, time_column] + 1
                )
            else:
                df_features.loc[company_mask, 'remaining_years'] = np.nan
                
            # データ取得からの経過年数
            df_features.loc[company_mask, 'data_vintage'] = (
                df_features.loc[company_mask, time_column] - profile.data_start_year + 1
            )
            
            # 正規化された時間（0-1スケール）
            total_years = profile.end_year - profile.start_year + 1
            df_features.loc[company_mask, 'normalized_time'] = (
                (df_features.loc[company_mask, time_column] - profile.start_year) / 
                max(total_years - 1, 1)
            )
            
        return df_features
    
    def _determine_lifecycle_stage(self,
                                    years: pd.Series,
                                    profile: CompanyTimeProfile) -> pd.Series:
        """ライフサイクル段階を決定"""
        total_years = profile.end_year - profile.start_year + 1
        
        # 段階分け：初期(0-20%), 成長(20-60%), 成熟(60-90%), 衰退/終了(90-100%)
        stages = []
        for year in years:
            if year < profile.start_year or year > profile.end_year:
                stages.append('non_existent')
            else:
                normalized_time = (year - profile.start_year) / max(total_years - 1, 1)
                if normalized_time <= 0.2:
                    stages.append('startup')
                elif normalized_time <= 0.6:
                    stages.append('growth')
                elif normalized_time <= 0.9:
                    stages.append('mature')
                else:
                    stages.append('decline_or_stable')
                    
        return pd.Series(stages, index=years.index)
    
    def _calculate_data_availability(self,
                                    df: pd.DataFrame,
                                    profile: CompanyTimeProfile) -> pd.Series:
        """データ可用性スコアを計算"""
        data_cols = df.select_dtypes(include=[np.number]).columns
        data_cols = [col for col in data_cols if col not in ['year', 'exists', 'panel_id', 'time_id']]
        
        if not data_cols:
            return pd.Series([0.0] * len(df), index=df.index)
            
        # 非NaN値の割合を計算
        availability = (~df[data_cols].isna()).mean(axis=1)
        
        # 企業が存在しない期間は0
        availability = availability * df['exists']
        
        return availability
    
    def _calculate_survival_weights(self, df: pd.DataFrame) -> pd.Series:
        """生存バイアス補正用の重みを計算"""
        weights = pd.Series(1.0, index=df.index)
        
        # 各年における存続企業数でウェイト調整
        for year in df['year'].unique():
            year_mask = df['year'] == year
            existing_companies = df[year_mask & df['exists']]['company_id'].nunique()
            total_companies = len(self.company_profiles)
            
            if existing_companies > 0:
                # 存続企業数の逆数でウェイト調整（生存バイアス補正）
                survival_rate = existing_companies / total_companies
                adjustment_weight = 1.0 / max(survival_rate, 0.1)  # 最小値で制限
                weights[year_mask & df['exists']] = adjustment_weight
                
        return weights
    
    def _validate_temporal_consistency(self,
                                        df: pd.DataFrame,
                                        time_column: str) -> pd.DataFrame:
        """時系列整合性を検証・修正"""
        # 時間列が適切な範囲内にあることを確認
        valid_years = (df[time_column] >= self.base_start_year) & (df[time_column] <= self.base_end_year)
        if not valid_years.all():
            logger.warning(f"Found {(~valid_years).sum()} rows with invalid years")
            df = df[valid_years].copy()
            
        # 重複する company_id + year の組み合わせを処理
        duplicates = df.duplicated(subset=['company_id', time_column], keep=False)
        if duplicates.any():
            logger.warning(f"Found {duplicates.sum()} duplicate company-year combinations")
            # 最後の値を保持
            df = df.drop_duplicates(subset=['company_id', time_column], keep='last')
            
        return df
    
    def get_alignment_summary(self) -> pd.DataFrame:
        """整合処理のサマリーを取得"""
        summary_data = []
        
        for company_id, profile in self.company_profiles.items():
            summary_data.append({
                'company_id': company_id,
                'company_name': profile.company_name,
                'lifecycle_type': profile.lifecycle_type.value,
                'market_category': profile.market_category,
                'start_year': profile.start_year,
                'end_year': profile.end_year,
                'total_years': profile.end_year - profile.start_year + 1,
                'data_start_year': profile.data_start_year,
                'data_end_year': profile.data_end_year,
                'data_years': profile.data_end_year - profile.data_start_year + 1,
                'missing_periods': len(profile.missing_periods),
                'data_coverage': (profile.data_end_year - profile.data_start_year + 1 - len(profile.missing_periods)) / 
                                max(profile.end_year - profile.start_year + 1, 1)
            })
            
        return pd.DataFrame(summary_data)
    
    def export_alignment_config(self, filepath: str):
        """整合設定をエクスポート"""
        config = {
            'base_start_year': self.base_start_year,
            'base_end_year': self.base_end_year,
            'min_data_years': self.min_data_years,
            'company_profiles': {}
        }
        
        for company_id, profile in self.company_profiles.items():
            config['company_profiles'][company_id] = {
                'company_name': profile.company_name,
                'lifecycle_type': profile.lifecycle_type.value,
                'start_year': profile.start_year,
                'end_year': profile.end_year,
                'data_start_year': profile.data_start_year,
                'data_end_year': profile.data_end_year,
                'missing_periods': profile.missing_periods,
                'market_category': profile.market_category
            }
            
        import json
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
            
    def load_alignment_config(self, filepath: str):
        """整合設定をロード"""
        import json
        with open(filepath, 'r', encoding='utf-8') as f:
            config = json.load(f)
            
        self.base_start_year = config['base_start_year']
        self.base_end_year = config['base_end_year']
        self.min_data_years = config['min_data_years']
        self.base_years = list(range(self.base_start_year, self.base_end_year + 1))
        
        self.company_profiles = {}
        for company_id, profile_data in config['company_profiles'].items():
            profile = CompanyTimeProfile(
                company_id=company_id,
                company_name=profile_data['company_name'],
                lifecycle_type=CompanyLifecycleType(profile_data['lifecycle_type']),
                start_year=profile_data['start_year'],
                end_year=profile_data['end_year'],
                data_start_year=profile_data['data_start_year'],
                data_end_year=profile_data['data_end_year'],
                missing_periods=profile_data['missing_periods'],
                market_category=profile_data['market_category']
            )
            self.company_profiles[company_id] = profile


# 使用例とテスト用のユーティリティ関数
def create_sample_alignment_handler() -> TemporalAlignmentHandler:
    """サンプルデータでテスト用ハンドラーを作成"""
    handler = TemporalAlignmentHandler()
    
    # サンプル企業プロファイル作成
    # 継続存続企業（ファナック）
    handler.create_company_profile(
        company_id="6954",
        company_name="ファナック",
        market_category="high_share",
        start_year=None,  # 1984年以前から存在
        end_year=None     # 現在も継続
    )
    
    # 消滅企業（三洋電機）
    handler.create_company_profile(
        company_id="6764",
        company_name="三洋電機",
        market_category="lost",
        start_year=1984,
        end_year=2012     # 2012年に消滅
    )
    
    # 新設企業（デンソーウェーブ）
    handler.create_company_profile(
        company_id="DENSO_WAVE",
        company_name="デンソーウェーブ",
        market_category="high_share",
        start_year=2001,  # 2001年設立
        end_year=None
    )
    
    return handler


if __name__ == "__main__":
    # 簡単なテスト実行
    handler = create_sample_alignment_handler()
    summary = handler.get_alignment_summary()
    print("Alignment Summary:")
    print(summary)