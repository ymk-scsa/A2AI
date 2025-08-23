"""
A2AI - Advanced Financial Analysis AI
Survival Features Engineering Module

This module generates survival analysis features for corporate lifecycle analysis.
It creates time-to-event features, hazard indicators, and survival-specific metrics
for the 150 companies across different market categories.

企業生存分析用の特徴量を生成するモジュール。
時間-イベント特徴量、ハザード指標、生存分析固有の指標を作成。
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
import logging
from scipy import stats
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class SurvivalFeaturesEngine:
    """
    企業生存分析用特徴量エンジニアリングクラス
    
    機能:
    - 企業存続期間の計算
    - イベント発生フラグの生成
    - ハザード率関連特徴量の生成
    - 生存曲線用特徴量の作成
    - 競合他社との生存比較特徴量
    """
    
    def __init__(self, 
                    start_year: int = 1984,
                    end_year: int = 2024,
                    log_level: str = 'INFO'):
        """
        初期化
        
        Args:
            start_year: 分析開始年
            end_year: 分析終了年
            log_level: ログレベル
        """
        self.start_year = start_year
        self.end_year = end_year
        self.analysis_period = end_year - start_year + 1
        
        # ログ設定
        logging.basicConfig(level=getattr(logging, log_level))
        self.logger = logging.getLogger(__name__)
        
        # 企業ライフサイクル段階定義
        self.lifecycle_stages = {
            'startup': (0, 5),      # 設立後5年以内
            'growth': (6, 15),      # 成長期
            'maturity': (16, 30),   # 成熟期
            'decline': (31, float('inf'))  # 衰退期
        }
        
        # 市場カテゴリ定義
        self.market_categories = {
            'high_share': 'シェア高市場',
            'declining': 'シェア低下市場', 
            'lost': 'シェア失失市場'
        }
        
    def calculate_survival_duration(self, 
                                    company_data: pd.DataFrame,
                                    company_info: Dict) -> Dict:
        """
        企業存続期間の計算
        
        Args:
            company_data: 企業の財務時系列データ
            company_info: 企業基本情報（設立年、消滅年等）
            
        Returns:
            Dict: 存続期間関連の特徴量
        """
        try:
            # 基本情報の取得
            founding_year = company_info.get('founding_year', self.start_year)
            extinction_year = company_info.get('extinction_year', None)
            company_name = company_info.get('company_name', 'Unknown')
            
            # 観測開始年の決定
            observation_start = max(founding_year, self.start_year)
            
            # 存続期間の計算
            if extinction_year is not None:
                # 企業が消滅した場合
                survival_duration = extinction_year - observation_start + 1
                is_censored = 0  # 完全観測（非打ち切り）
                event_occurred = 1  # イベント（消滅）発生
                survival_status = 'extinct'
            else:
                # 企業が存続している場合
                survival_duration = self.end_year - observation_start + 1
                is_censored = 1  # 右打ち切り
                event_occurred = 0  # イベント未発生
                survival_status = 'surviving'
            
            # 企業年齢の計算
            company_age_at_start = observation_start - founding_year
            company_age_at_end = (extinction_year or self.end_year) - founding_year
            
            # ライフサイクル段階の判定
            lifecycle_stage = self._determine_lifecycle_stage(company_age_at_end)
            
            survival_features = {
                # 基本的な存続情報
                'survival_duration': survival_duration,
                'is_censored': is_censored,
                'event_occurred': event_occurred,
                'survival_status': survival_status,
                
                # 企業年齢関連
                'company_age_at_start': company_age_at_start,
                'company_age_at_end': company_age_at_end,
                'age_at_observation': company_age_at_end,
                
                # ライフサイクル関連
                'lifecycle_stage': lifecycle_stage,
                'years_in_current_stage': self._years_in_stage(company_age_at_end, lifecycle_stage),
                
                # 観測期間関連
                'observation_start_year': observation_start,
                'observation_end_year': extinction_year or self.end_year,
                'data_availability_ratio': len(company_data) / survival_duration if survival_duration > 0 else 0,
                
                # 企業基本情報
                'founding_year': founding_year,
                'extinction_year': extinction_year,
                'company_name': company_name
            }
            
            self.logger.debug(f"{company_name}: 存続期間={survival_duration}年, ステータス={survival_status}")
            
            return survival_features
            
        except Exception as e:
            self.logger.error(f"存続期間計算エラー: {e}")
            return {}
    
    def generate_hazard_indicators(self, 
                                    financial_data: pd.DataFrame,
                                    window_size: int = 3) -> Dict:
        """
        ハザード率（リスク）指標の生成
        
        Args:
            financial_data: 財務時系列データ
            window_size: 移動平均ウィンドウサイズ
            
        Returns:
            Dict: ハザード指標特徴量
        """
        try:
            hazard_features = {}
            
            if financial_data.empty:
                return hazard_features
            
            # 財務悪化トレンド指標
            hazard_features.update(self._calculate_deterioration_trends(financial_data, window_size))
            
            # リスク蓄積指標
            hazard_features.update(self._calculate_risk_accumulation(financial_data, window_size))
            
            # 市場競争力低下指標
            hazard_features.update(self._calculate_competitiveness_decline(financial_data, window_size))
            
            # 企業体力指標
            hazard_features.update(self._calculate_corporate_resilience(financial_data, window_size))
            
            # 早期警告指標
            hazard_features.update(self._calculate_early_warning_indicators(financial_data, window_size))
            
            return hazard_features
            
        except Exception as e:
            self.logger.error(f"ハザード指標生成エラー: {e}")
            return {}
    
    def create_time_varying_features(self, 
                                    financial_data: pd.DataFrame,
                                    survival_info: Dict) -> pd.DataFrame:
        """
        時変特徴量の生成
        
        Args:
            financial_data: 財務時系列データ
            survival_info: 存続情報
            
        Returns:
            pd.DataFrame: 時変特徴量データ
        """
        try:
            if financial_data.empty:
                return pd.DataFrame()
            
            # 基本時変特徴量の作成
            time_varying_df = financial_data.copy()
            
            # 存続期間からの経過時間
            if 'year' in time_varying_df.columns:
                start_year = survival_info.get('observation_start_year', self.start_year)
                time_varying_df['time_since_start'] = time_varying_df['year'] - start_year + 1
                time_varying_df['time_to_end'] = survival_info.get('survival_duration', 0) - time_varying_df['time_since_start'] + 1
            
            # 年齢関連特徴量
            founding_year = survival_info.get('founding_year', self.start_year)
            if 'year' in time_varying_df.columns:
                time_varying_df['company_age'] = time_varying_df['year'] - founding_year + 1
                time_varying_df['age_squared'] = time_varying_df['company_age'] ** 2
                time_varying_df['age_log'] = np.log1p(time_varying_df['company_age'])
            
            # ライフサイクル段階ダミー変数
            for stage in self.lifecycle_stages.keys():
                time_varying_df[f'is_{stage}'] = 0
            
            if 'company_age' in time_varying_df.columns:
                for idx, row in time_varying_df.iterrows():
                    stage = self._determine_lifecycle_stage(row['company_age'])
                    time_varying_df.loc[idx, f'is_{stage}'] = 1
            
            # 経営指標の変化率
            time_varying_df = self._add_change_rates(time_varying_df)
            
            # 移動平均・トレンド指標
            time_varying_df = self._add_trend_indicators(time_varying_df)
            
            # リスク指標
            time_varying_df = self._add_risk_indicators(time_varying_df)
            
            return time_varying_df
            
        except Exception as e:
            self.logger.error(f"時変特徴量生成エラー: {e}")
            return pd.DataFrame()
    
    def generate_survival_comparison_features(self, 
                                            company_data: Dict,
                                            market_data: Dict,
                                            company_info: Dict) -> Dict:
        """
        同業他社との生存比較特徴量
        
        Args:
            company_data: 対象企業データ
            market_data: 市場内他社データ
            company_info: 企業情報
            
        Returns:
            Dict: 比較特徴量
        """
        try:
            comparison_features = {}
            
            market_category = company_info.get('market_category', 'unknown')
            market_name = company_info.get('market_name', 'unknown')
            
            # 同一市場内での生存率比較
            if market_category in market_data:
                market_companies = market_data[market_category].get(market_name, {})
                
                # 市場内企業の生存統計
                survival_stats = self._calculate_market_survival_stats(market_companies)
                comparison_features.update(survival_stats)
                
                # 相対ポジション
                company_survival = company_data.get('survival_duration', 0)
                if survival_stats.get('median_survival', 0) > 0:
                    comparison_features['survival_percentile'] = self._calculate_survival_percentile(
                        company_survival, market_companies
                    )
                    comparison_features['survival_vs_median'] = company_survival / survival_stats['median_survival']
                
                # ピア比較特徴量
                peer_features = self._calculate_peer_comparison(company_data, market_companies)
                comparison_features.update(peer_features)
            
            # クロス市場比較
            cross_market_features = self._calculate_cross_market_comparison(
                company_info, market_data
            )
            comparison_features.update(cross_market_features)
            
            return comparison_features
            
        except Exception as e:
            self.logger.error(f"生存比較特徴量生成エラー: {e}")
            return {}
    
    def create_survival_dataset(self, 
                                companies_data: Dict,
                                output_format: str = 'long') -> Union[pd.DataFrame, Dict]:
        """
        生存分析用データセットの作成
        
        Args:
            companies_data: 全企業データ
            output_format: 'long' or 'wide' or 'both'
            
        Returns:
            Union[pd.DataFrame, Dict]: 生存分析用データセット
        """
        try:
            survival_datasets = {}
            
            # ワイド形式データセット（企業×特徴量）
            wide_dataset = self._create_wide_survival_dataset(companies_data)
            
            # ロング形式データセット（時系列展開）
            long_dataset = self._create_long_survival_dataset(companies_data)
            
            if output_format == 'wide':
                return wide_dataset
            elif output_format == 'long':
                return long_dataset
            elif output_format == 'both':
                return {
                    'wide_format': wide_dataset,
                    'long_format': long_dataset
                }
            else:
                raise ValueError("output_format must be 'wide', 'long', or 'both'")
                
        except Exception as e:
            self.logger.error(f"生存データセット作成エラー: {e}")
            return pd.DataFrame()
    
    # プライベートメソッド群
    
    def _determine_lifecycle_stage(self, company_age: int) -> str:
        """企業年齢からライフサイクル段階を判定"""
        for stage, (min_age, max_age) in self.lifecycle_stages.items():
            if min_age <= company_age <= max_age:
                return stage
        return 'decline'  # デフォルト
    
    def _years_in_stage(self, company_age: int, stage: str) -> int:
        """現在のライフサイクル段階での滞在年数"""
        if stage not in self.lifecycle_stages:
            return 0
        
        min_age, max_age = self.lifecycle_stages[stage]
        if company_age < min_age:
            return 0
        elif company_age > max_age and max_age != float('inf'):
            return max_age - min_age + 1
        else:
            return company_age - min_age + 1
    
    def _calculate_deterioration_trends(self, data: pd.DataFrame, window: int) -> Dict:
        """財務悪化トレンド指標の計算"""
        features = {}
        
        # 主要財務指標の悪化トレンド
        key_metrics = ['売上高営業利益率', 'ROE', '自己資本比率', '売上高', '売上高成長率']
        
        for metric in key_metrics:
            if metric in data.columns:
                values = data[metric].dropna()
                if len(values) >= window:
                    # トレンド勾配
                    trend_slope = stats.linregress(range(len(values)), values).slope
                    features[f'{metric}_trend_slope'] = trend_slope
                    
                    # 悪化継続期間
                    deterioration_periods = self._count_continuous_decline(values, window)
                    features[f'{metric}_deterioration_periods'] = deterioration_periods
                    
                    # 変動係数（不安定性）
                    if values.mean() != 0:
                        features[f'{metric}_cv'] = values.std() / abs(values.mean())
        
        return features
    
    def _calculate_risk_accumulation(self, data: pd.DataFrame, window: int) -> Dict:
        """リスク蓄積指標の計算"""
        features = {}
        
        # 債務比率の上昇トレンド
        if '有利子負債比率' in data.columns:
            debt_ratio = data['有利子負債比率'].dropna()
            if len(debt_ratio) >= window:
                features['debt_accumulation_rate'] = debt_ratio.rolling(window).apply(
                    lambda x: stats.linregress(range(len(x)), x).slope if len(x) > 1 else 0
                ).mean()
        
        # 流動性リスク
        if '流動比率' in data.columns:
            liquidity = data['流動比率'].dropna()
            if len(liquidity) > 0:
                features['liquidity_risk_score'] = (liquidity < 100).sum() / len(liquidity)
        
        # 収益性の安定性
        profit_metrics = ['売上高営業利益率', '売上高当期純利益率']
        for metric in profit_metrics:
            if metric in data.columns:
                values = data[metric].dropna()
                if len(values) > window:
                    # 負の値の頻度
                    features[f'{metric}_negative_frequency'] = (values < 0).sum() / len(values)
                    
                    # ボラティリティ
                    features[f'{metric}_volatility'] = values.rolling(window).std().mean()
        
        return features
    
    def _calculate_competitiveness_decline(self, data: pd.DataFrame, window: int) -> Dict:
        """市場競争力低下指標の計算"""
        features = {}
        
        # 市場シェア関連指標
        if '売上高' in data.columns:
            revenue = data['売上高'].dropna()
            if len(revenue) >= window:
                # 売上成長率の低下
                growth_rates = revenue.pct_change().dropna()
                if len(growth_rates) > 0:
                    features['revenue_growth_decline'] = (growth_rates < 0).sum() / len(growth_rates)
                    
                    # 成長率の標準偏差（不安定性）
                    features['revenue_growth_volatility'] = growth_rates.std()
        
        # 研究開発投資の減少
        if '研究開発費率' in data.columns:
            rd_ratio = data['研究開発費率'].dropna()
            if len(rd_ratio) >= window:
                rd_trend = stats.linregress(range(len(rd_ratio)), rd_ratio).slope
                features['rd_investment_trend'] = rd_trend
        
        # 設備投資の減少
        if '設備投資額' in data.columns:
            capex = data['設備投資額'].dropna()
            if len(capex) >= window:
                capex_trend = stats.linregress(range(len(capex)), capex).slope
                features['capex_trend'] = capex_trend
        
        return features
    
    def _calculate_corporate_resilience(self, data: pd.DataFrame, window: int) -> Dict:
        """企業体力指標の計算"""
        features = {}
        
        # 財務安全性
        safety_metrics = ['自己資本比率', '流動比率', '固定比率']
        for metric in safety_metrics:
            if metric in data.columns:
                values = data[metric].dropna()
                if len(values) > 0:
                    features[f'{metric}_min'] = values.min()
                    features[f'{metric}_trend'] = stats.linregress(range(len(values)), values).slope if len(values) > 1 else 0
        
        # 収益性の回復力
        if '売上高営業利益率' in data.columns:
            operating_margin = data['売上高営業利益率'].dropna()
            if len(operating_margin) >= window:
                # 負から正への回復回数
                recovery_count = 0
                for i in range(1, len(operating_margin)):
                    if operating_margin.iloc[i-1] < 0 and operating_margin.iloc[i] >= 0:
                        recovery_count += 1
                features['profit_recovery_ability'] = recovery_count / max(1, len(operating_margin) - 1)
        
        return features
    
    def _calculate_early_warning_indicators(self, data: pd.DataFrame, window: int) -> Dict:
        """早期警告指標の計算"""
        features = {}
        
        # Z-Score（Altman）の簡易版
        if all(col in data.columns for col in ['総資産', '自己資本', '売上高', '営業利益']):
            latest_data = data.iloc[-1]
            
            # 運転資本/総資産
            working_capital_ratio = latest_data.get('運転資本', 0) / latest_data['総資産'] if latest_data['総資産'] != 0 else 0
            
            # 留保利益/総資産（簡易計算）
            retained_earnings_ratio = latest_data['自己資本'] / latest_data['総資産'] if latest_data['総資産'] != 0 else 0
            
            # EBIT/総資産
            ebit_ratio = latest_data['営業利益'] / latest_data['総資産'] if latest_data['総資産'] != 0 else 0
            
            # 売上高/総資産
            asset_turnover = latest_data['売上高'] / latest_data['総資産'] if latest_data['総資産'] != 0 else 0
            
            # 簡易Z-Score
            z_score = (1.2 * working_capital_ratio + 
                      1.4 * retained_earnings_ratio + 
                      3.3 * ebit_ratio + 
                      0.6 * asset_turnover)
            
            features['altman_z_score'] = z_score
            features['bankruptcy_risk_high'] = 1 if z_score < 1.8 else 0
            features['bankruptcy_risk_moderate'] = 1 if 1.8 <= z_score < 3.0 else 0
        
        return features
    
    def _add_change_rates(self, df: pd.DataFrame) -> pd.DataFrame:
        """変化率の追加"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col not in ['year', 'time_since_start', 'time_to_end', 'company_age']:
                # 1年変化率
                df[f'{col}_yoy_change'] = df[col].pct_change()
                
                # 3年移動平均変化率
                df[f'{col}_3y_avg_change'] = df[col].rolling(3).mean().pct_change()
        
        return df
    
    def _add_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """トレンド指標の追加"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        window_sizes = [3, 5]
        
        for col in numeric_cols:
            if col not in ['year', 'time_since_start', 'time_to_end', 'company_age']:
                for window in window_sizes:
                    # 移動平均
                    df[f'{col}_ma_{window}'] = df[col].rolling(window).mean()
                    
                    # トレンド勾配
                    df[f'{col}_trend_{window}'] = df[col].rolling(window).apply(
                        lambda x: stats.linregress(range(len(x)), x).slope if len(x) == window else np.nan
                    )
        
        return df
    
    def _add_risk_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """リスク指標の追加"""
        # 収益性リスク指標
        if '売上高営業利益率' in df.columns:
            df['profit_risk_3y'] = df['売上高営業利益率'].rolling(3).std()
            df['profit_below_zero'] = (df['売上高営業利益率'] < 0).astype(int)
        
        # 成長性リスク指標
        if '売上高成長率' in df.columns:
            df['growth_volatility_3y'] = df['売上高成長率'].rolling(3).std()
            df['negative_growth'] = (df['売上高成長率'] < 0).astype(int)
        
        # 財務安全性リスク指標
        if '自己資本比率' in df.columns:
            df['equity_ratio_risk'] = (df['自己資本比率'] < 30).astype(int)
        
        return df
    
    def _count_continuous_decline(self, series: pd.Series, min_periods: int) -> int:
        """連続悪化期間のカウント"""
        if len(series) < min_periods:
            return 0
        
        max_continuous = 0
        current_continuous = 0
        
        for i in range(1, len(series)):
            if series.iloc[i] < series.iloc[i-1]:
                current_continuous += 1
                max_continuous = max(max_continuous, current_continuous)
            else:
                current_continuous = 0
        
        return max_continuous
    
    def _calculate_market_survival_stats(self, market_companies: Dict) -> Dict:
        """市場内生存統計の計算"""
        survival_durations = []
        extinction_count = 0
        
        for company_name, company_data in market_companies.items():
            survival_info = company_data.get('survival_info', {})
            duration = survival_info.get('survival_duration', 0)
            status = survival_info.get('survival_status', 'unknown')
            
            survival_durations.append(duration)
            if status == 'extinct':
                extinction_count += 1
        
        if len(survival_durations) == 0:
            return {}
        
        return {
            'market_median_survival': np.median(survival_durations),
            'market_mean_survival': np.mean(survival_durations),
            'market_survival_std': np.std(survival_durations),
            'market_extinction_rate': extinction_count / len(survival_durations),
            'market_company_count': len(survival_durations)
        }
    
    def _calculate_survival_percentile(self, company_survival: float, market_companies: Dict) -> float:
        """市場内での生存期間パーセンタイル"""
        survival_durations = []
        
        for company_data in market_companies.values():
            survival_info = company_data.get('survival_info', {})
            duration = survival_info.get('survival_duration', 0)
            survival_durations.append(duration)
        
        if len(survival_durations) == 0:
            return 0.5
        
        return stats.percentileofscore(survival_durations, company_survival) / 100.0
    
    def _calculate_peer_comparison(self, company_data: Dict, market_companies: Dict) -> Dict:
        """同業他社比較特徴量"""
        features = {}
        
        # 同業他社との相対的な財務指標比較
        # （実装は財務データの詳細構造に依存するため、基本的な枠組みのみ示す）
        
        features['peer_count'] = len(market_companies)
        features['peer_comparison_available'] = 1 if len(market_companies) > 1 else 0
        
        return features
    
    def _calculate_cross_market_comparison(self, company_info: Dict, market_data: Dict) -> Dict:
        """クロス市場比較特徴量"""
        features = {}
        
        company_category = company_info.get('market_category', 'unknown')
        
        # 他市場カテゴリとの比較
        for category, category_name in self.market_categories.items():
            if category != company_category and category in market_data:
                # 他市場での平均生存期間などを計算
                # （詳細実装は省略）
                features[f'vs_{category}_market'] = 1
        
        return features
    
    def _create_wide_survival_dataset(self, companies_data: Dict) -> pd.DataFrame:
        """ワイド形式生存データセットの作成"""
        wide_data = []
        
        for company_name, company_data in companies_data.items():
            company_info = company_data.get('company_info', {})
            financial_data = company_data.get('financial_data', pd.DataFrame())
            survival_info = company_data.get('survival_info', {})
            
            # 基本情報
            row_data = {
                'company_name': company_name,
                'market_category': company_info.get('market_category'),
                'market_name': company_info.get('market_name'),
            }
            
            # 存続情報
            row_data.update(survival_info)
            
            # ハザード指標
            hazard_features = self.generate_hazard_indicators(financial_data)
            row_data.update(hazard_features)
            
            # 比較特徴量は省略（実装が複雑なため）
            
            wide_data.append(row_data)
        
        return pd.DataFrame(wide_data)
    
    def _create_long_survival_dataset(self, companies_data: Dict) -> pd.DataFrame:
        """ロング形式生存データセットの作成"""
        long_data = []
        
        for company_name, company_data in companies_data.items():
            company_info = company_data.get('company_info', {})
            financial_data = company_data.get('financial_data', pd.DataFrame())
            survival_info = company_data.get('survival_info', {})
            
            # 時変特徴量の生成
            time_varying = self.create_time_varying_features(financial_data, survival_info)
            
            # 企業情報を各行に追加
            for idx, row in time_varying.iterrows():
                row_data = row.to_dict()
                row_data.update({
                    'company_name': company_name,
                    'market_category': company_info.get('market_category'),
                    'market_name': company_info.get('market_name'),
                    'survival_duration': survival_info.get('survival_duration', 0),
                    'event_occurred': survival_info.get('event_occurred', 0),
                    'is_censored': survival_info.get('is_censored', 1),
                })
                long_data.append(row_data)
        
        return pd.DataFrame(long_data)

    def calculate_baseline_hazard_features(self, 
                                            survival_data: pd.DataFrame,
                                            time_col: str = 'survival_duration',
                                            event_col: str = 'event_occurred') -> pd.DataFrame:
        """
        ベースラインハザード特徴量の計算
        
        Args:
            survival_data: 生存分析データ
            time_col: 時間列名
            event_col: イベント列名
            
        Returns:
            pd.DataFrame: ベースラインハザード特徴量付きデータ
        """
        try:
            if survival_data.empty:
                return survival_data
            
            data = survival_data.copy()
            
            # 各時点でのリスクセット（まだ生存している企業数）
            unique_times = sorted(data[time_col].unique())
            
            for t in unique_times:
                # 時点tでのリスクセット
                risk_set = data[data[time_col] >= t]
                data.loc[data[time_col] == t, f'risk_set_size_at_{t}'] = len(risk_set)
                
                # 時点tでのイベント数
                events_at_t = data[(data[time_col] == t) & (data[event_col] == 1)]
                data.loc[data[time_col] == t, f'events_at_{t}'] = len(events_at_t)
            
            # カプラン・マイヤー推定量の計算要素
            data['survival_probability'] = 1.0
            data['cumulative_hazard'] = 0.0
            
            survival_prob = 1.0
            cumulative_hazard = 0.0
            
            for t in unique_times:
                mask = data[time_col] == t
                risk_set_size = data.loc[mask, f'risk_set_size_at_{t}'].iloc[0] if mask.any() else 1
                events = data.loc[mask, f'events_at_{t}'].iloc[0] if mask.any() else 0
                
                if risk_set_size > 0:
                    hazard_rate = events / risk_set_size
                    survival_prob *= (1 - hazard_rate)
                    cumulative_hazard += hazard_rate
                
                data.loc[mask, 'survival_probability'] = survival_prob
                data.loc[mask, 'cumulative_hazard'] = cumulative_hazard
            
            return data
            
        except Exception as e:
            self.logger.error(f"ベースラインハザード計算エラー: {e}")
            return survival_data

    def generate_market_survival_features(self, 
                                        companies_data: Dict,
                                        market_categories: Dict) -> Dict:
        """
        市場別生存特徴量の生成
        
        Args:
            companies_data: 全企業データ
            market_categories: 市場カテゴリ情報
            
        Returns:
            Dict: 市場別生存特徴量
        """
        try:
            market_features = {}
            
            # 市場カテゴリ別の分析
            for category in ['high_share', 'declining', 'lost']:
                category_companies = {
                    name: data for name, data in companies_data.items()
                    if data.get('company_info', {}).get('market_category') == category
                }
                
                if not category_companies:
                    continue
                
                # 市場カテゴリ別統計
                category_stats = self._analyze_market_category_survival(category_companies)
                market_features[f'{category}_market'] = category_stats
                
                # 具体的な市場別分析
                markets_in_category = {}
                for company_data in category_companies.values():
                    market_name = company_data.get('company_info', {}).get('market_name', 'unknown')
                    if market_name not in markets_in_category:
                        markets_in_category[market_name] = []
                    markets_in_category[market_name].append(company_data)
                
                # 各市場での生存パターン分析
                for market_name, market_companies in markets_in_category.items():
                    market_survival_pattern = self._analyze_market_survival_pattern(market_companies)
                    market_features[f'{category}_{market_name}'] = market_survival_pattern
            
            return market_features
            
        except Exception as e:
            self.logger.error(f"市場別生存特徴量生成エラー: {e}")
            return {}

    def create_survival_feature_summary(self, companies_data: Dict) -> Dict:
        """
        生存特徴量の統計サマリー作成
        
        Args:
            companies_data: 全企業データ
            
        Returns:
            Dict: 特徴量サマリー
        """
        try:
            summary = {
                'dataset_info': {
                    'total_companies': len(companies_data),
                    'analysis_period': f"{self.start_year}-{self.end_year}",
                    'total_years': self.analysis_period
                },
                'survival_statistics': {},
                'market_breakdown': {},
                'feature_statistics': {}
            }
            
            # 基本統計
            survival_durations = []
            event_counts = {'extinct': 0, 'surviving': 0}
            market_counts = {'high_share': 0, 'declining': 0, 'lost': 0}
            
            for company_name, company_data in companies_data.items():
                survival_info = company_data.get('survival_info', {})
                company_info = company_data.get('company_info', {})
                
                duration = survival_info.get('survival_duration', 0)
                status = survival_info.get('survival_status', 'unknown')
                category = company_info.get('market_category', 'unknown')
                
                survival_durations.append(duration)
                
                if status in event_counts:
                    event_counts[status] += 1
                
                if category in market_counts:
                    market_counts[category] += 1
            
            # 生存統計
            if survival_durations:
                summary['survival_statistics'] = {
                    'mean_survival': np.mean(survival_durations),
                    'median_survival': np.median(survival_durations),
                    'std_survival': np.std(survival_durations),
                    'min_survival': np.min(survival_durations),
                    'max_survival': np.max(survival_durations),
                    'extinction_rate': event_counts['extinct'] / len(survival_durations) if survival_durations else 0,
                    'censoring_rate': event_counts['surviving'] / len(survival_durations) if survival_durations else 0
                }
            
            # 市場別統計
            summary['market_breakdown'] = market_counts
            
            # 特徴量統計（サンプルから計算）
            sample_features = {}
            if companies_data:
                sample_company = next(iter(companies_data.values()))
                sample_financial = sample_company.get('financial_data', pd.DataFrame())
                
                if not sample_financial.empty:
                    sample_hazard = self.generate_hazard_indicators(sample_financial)
                    sample_features = {
                        'hazard_features_count': len(sample_hazard),
                        'sample_features': list(sample_hazard.keys())[:10]  # 最初の10個
                    }
            
            summary['feature_statistics'] = sample_features
            
            return summary
            
        except Exception as e:
            self.logger.error(f"生存特徴量サマリー作成エラー: {e}")
            return {}

    def validate_survival_features(self, 
                                    survival_dataset: pd.DataFrame,
                                    required_columns: List[str] = None) -> Dict:
        """
        生存特徴量の検証
        
        Args:
            survival_dataset: 生存分析データセット
            required_columns: 必須カラムリスト
            
        Returns:
            Dict: 検証結果
        """
        try:
            validation_result = {
                'is_valid': True,
                'errors': [],
                'warnings': [],
                'statistics': {}
            }
            
            if survival_dataset.empty:
                validation_result['is_valid'] = False
                validation_result['errors'].append("データセットが空です")
                return validation_result
            
            # 基本構造チェック
            if required_columns:
                missing_cols = [col for col in required_columns if col not in survival_dataset.columns]
                if missing_cols:
                    validation_result['errors'].extend([f"必須列が不足: {col}" for col in missing_cols])
                    validation_result['is_valid'] = False
            
            # データ品質チェック
            quality_checks = self._perform_data_quality_checks(survival_dataset)
            validation_result['warnings'].extend(quality_checks['warnings'])
            validation_result['errors'].extend(quality_checks['errors'])
            
            # 統計的妥当性チェック
            stat_checks = self._perform_statistical_validity_checks(survival_dataset)
            validation_result['statistics'] = stat_checks
            
            if quality_checks['errors']:
                validation_result['is_valid'] = False
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"生存特徴量検証エラー: {e}")
            return {'is_valid': False, 'errors': [str(e)], 'warnings': [], 'statistics': {}}

    # プライベートメソッドの追加実装

    def _analyze_market_category_survival(self, category_companies: Dict) -> Dict:
        """市場カテゴリ別生存分析"""
        survival_stats = {}
        
        durations = []
        extinctions = 0
        
        for company_data in category_companies.values():
            survival_info = company_data.get('survival_info', {})
            duration = survival_info.get('survival_duration', 0)
            status = survival_info.get('survival_status', 'unknown')
            
            durations.append(duration)
            if status == 'extinct':
                extinctions += 1
        
        if durations:
            survival_stats = {
                'company_count': len(durations),
                'mean_survival': np.mean(durations),
                'median_survival': np.median(durations),
                'extinction_count': extinctions,
                'extinction_rate': extinctions / len(durations),
                'survival_range': [np.min(durations), np.max(durations)]
            }
        
        return survival_stats

    def _analyze_market_survival_pattern(self, market_companies: List[Dict]) -> Dict:
        """特定市場での生存パターン分析"""
        pattern_analysis = {}
        
        # 企業設立年の分布
        founding_years = []
        extinction_years = []
        
        for company_data in market_companies:
            company_info = company_data.get('company_info', {})
            survival_info = company_data.get('survival_info', {})
            
            founding_year = survival_info.get('founding_year')
            extinction_year = survival_info.get('extinction_year')
            
            if founding_year:
                founding_years.append(founding_year)
            if extinction_year:
                extinction_years.append(extinction_year)
        
        pattern_analysis['founding_year_range'] = [min(founding_years), max(founding_years)] if founding_years else [0, 0]
        pattern_analysis['extinction_year_range'] = [min(extinction_years), max(extinction_years)] if extinction_years else [0, 0]
        pattern_analysis['market_maturity'] = max(founding_years) - min(founding_years) if len(founding_years) > 1 else 0
        
        return pattern_analysis

    def _perform_data_quality_checks(self, dataset: pd.DataFrame) -> Dict:
        """データ品質チェック"""
        checks = {'errors': [], 'warnings': []}
        
        # 欠損値チェック
        missing_ratios = dataset.isnull().mean()
        high_missing = missing_ratios[missing_ratios > 0.5]
        
        for col in high_missing.index:
            checks['warnings'].append(f"列 {col} の欠損率が高い: {high_missing[col]:.2%}")
        
        # 異常値チェック
        numeric_cols = dataset.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in dataset.columns:
                Q1 = dataset[col].quantile(0.25)
                Q3 = dataset[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = dataset[(dataset[col] < lower_bound) | (dataset[col] > upper_bound)]
                outlier_ratio = len(outliers) / len(dataset)
                
                if outlier_ratio > 0.1:
                    checks['warnings'].append(f"列 {col} の外れ値が多い: {outlier_ratio:.2%}")
        
        # 生存分析特有のチェック
        if 'survival_duration' in dataset.columns:
            negative_durations = (dataset['survival_duration'] < 0).sum()
            if negative_durations > 0:
                checks['errors'].append(f"負の生存期間: {negative_durations}件")
        
        if 'event_occurred' in dataset.columns:
            event_values = dataset['event_occurred'].unique()
            if not all(val in [0, 1] for val in event_values if pd.notna(val)):
                checks['errors'].append("イベント発生フラグは0または1である必要があります")
        
        return checks

    def _perform_statistical_validity_checks(self, dataset: pd.DataFrame) -> Dict:
        """統計的妥当性チェック"""
        stats = {}
        
        # 基本統計
        stats['row_count'] = len(dataset)
        stats['column_count'] = len(dataset.columns)
        stats['missing_data_ratio'] = dataset.isnull().sum().sum() / (len(dataset) * len(dataset.columns))
        
        # 生存分析固有の統計
        if 'survival_duration' in dataset.columns:
            stats['survival_duration_stats'] = {
                'mean': dataset['survival_duration'].mean(),
                'std': dataset['survival_duration'].std(),
                'min': dataset['survival_duration'].min(),
                'max': dataset['survival_duration'].max()
            }
        
        if 'event_occurred' in dataset.columns:
            event_rate = dataset['event_occurred'].mean()
            stats['event_rate'] = event_rate
            stats['censoring_rate'] = 1 - event_rate
        
        # 市場カテゴリ分布
        if 'market_category' in dataset.columns:
            category_counts = dataset['market_category'].value_counts().to_dict()
            stats['market_category_distribution'] = category_counts
        
        return stats

    def export_survival_features(self, 
                                survival_dataset: pd.DataFrame,
                                output_path: str,
                                format: str = 'csv') -> bool:
        """
        生存特徴量データセットのエクスポート
        
        Args:
            survival_dataset: 生存分析データセット
            output_path: 出力パス
            format: 出力フォーマット ('csv', 'parquet', 'excel')
            
        Returns:
            bool: エクスポート成功フラグ
        """
        try:
            if format.lower() == 'csv':
                survival_dataset.to_csv(output_path, index=False, encoding='utf-8-sig')
            elif format.lower() == 'parquet':
                survival_dataset.to_parquet(output_path, index=False)
            elif format.lower() == 'excel':
                survival_dataset.to_excel(output_path, index=False)
            else:
                raise ValueError(f"サポートされていないフォーマット: {format}")
            
            self.logger.info(f"生存特徴量データセットを {output_path} にエクスポートしました")
            return True
            
        except Exception as e:
            self.logger.error(f"エクスポートエラー: {e}")
            return False


# 使用例とテスト用の関数
def example_usage():
    """
    SurvivalFeaturesEngineの使用例
    """
    # インスタンス作成
    survival_engine = SurvivalFeaturesEngine(
        start_year=1984,
        end_year=2024,
        log_level='INFO'
    )
    
    # サンプルデータ作成
    sample_company_info = {
        'company_name': 'サンプル企業',
        'founding_year': 1990,
        'extinction_year': 2020,  # None if still surviving
        'market_category': 'declining',
        'market_name': '自動車市場'
    }
    
    # サンプル財務データ
    years = list(range(1990, 2021))
    sample_financial_data = pd.DataFrame({
        'year': years,
        '売上高': np.random.rand(len(years)) * 1000 + 500,
        '売上高営業利益率': np.random.rand(len(years)) * 10 - 2,
        'ROE': np.random.rand(len(years)) * 15 + 5,
        '自己資本比率': np.random.rand(len(years)) * 30 + 30,
        '研究開発費率': np.random.rand(len(years)) * 5 + 2
    })
    
    # 存続期間計算
    survival_info = survival_engine.calculate_survival_duration(
        sample_financial_data, sample_company_info
    )
    print("存続情報:", survival_info)
    
    # ハザード指標生成
    hazard_features = survival_engine.generate_hazard_indicators(sample_financial_data)
    print("ハザード指標数:", len(hazard_features))
    
    # 時変特徴量生成
    time_varying_features = survival_engine.create_time_varying_features(
        sample_financial_data, survival_info
    )
    print("時変特徴量形状:", time_varying_features.shape)
    
    return survival_engine


if __name__ == "__main__":
    example_usage()