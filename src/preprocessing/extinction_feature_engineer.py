"""
A2AI - Advanced Financial Analysis AI
企業消滅予兆特徴量生成モジュール

このモジュールは企業の消滅・倒産・事業撤退の予兆を捉えるための
特徴量を生成します。財務諸表データから企業の健全性悪化の兆候を
定量化し、生存分析で使用可能な特徴量セットを作成します。

対象企業：150社（高シェア50社、シェア低下50社、失失50社）
分析期間：1984-2024年（40年間）
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import warnings
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import KNNImputer
import logging

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ExtinctionFeatureConfig:
    """消滅予兆特徴量生成の設定クラス"""
    lookback_periods: int = 5  # 遡及分析期間（年）
    deterioration_threshold: float = 0.3  # 悪化判定閾値
    volatility_window: int = 3  # ボラティリティ計算ウィンドウ
    trend_window: int = 5  # トレンド分析ウィンドウ
    outlier_threshold: float = 3.0  # 外れ値判定（標準偏差）
    missing_value_threshold: float = 0.8  # 欠損値許容割合

class ExtinctionFeatureEngineer:
    """
    企業消滅予兆特徴量生成クラス
    
    企業の財務データから消滅リスクを示唆する特徴量を生成します。
    特に失失市場企業の消滅パターンを学習し、同様の兆候を捉える
    特徴量を構築します。
    """
    
    def __init__(self, config: ExtinctionFeatureConfig = None):
        self.config = config or ExtinctionFeatureConfig()
        self.scaler = RobustScaler()  # 外れ値に頑健なスケーラー
        self.imputer = KNNImputer(n_neighbors=5)
        self.extinction_features = {}
        
    def generate_extinction_features(self, 
                                    financial_data: pd.DataFrame,
                                    extinction_events: pd.DataFrame) -> pd.DataFrame:
        """
        企業消滅予兆特徴量の総合生成
        
        Args:
            financial_data: 財務諸表データ
            extinction_events: 企業消滅イベントデータ
            
        Returns:
            消滅予兆特徴量データフレーム
        """
        logger.info("企業消滅予兆特徴量生成を開始")
        
        # データ前処理
        processed_data = self._preprocess_data(financial_data)
        
        # 各カテゴリの特徴量生成
        features = {}
        
        # 1. 財務健全性悪化特徴量
        features.update(self._generate_financial_health_features(processed_data))
        
        # 2. 収益性悪化特徴量
        features.update(self._generate_profitability_deterioration_features(processed_data))
        
        # 3. 流動性・資金繰り危険特徴量
        features.update(self._generate_liquidity_crisis_features(processed_data))
        
        # 4. 成長性・市場競争力喪失特徴量
        features.update(self._generate_competitiveness_loss_features(processed_data))
        
        # 5. レバレッジ・債務負担特徴量
        features.update(self._generate_leverage_burden_features(processed_data))
        
        # 6. 事業効率性悪化特徴量
        features.update(self._generate_operational_efficiency_features(processed_data))
        
        # 7. 投資・イノベーション停滞特徴量
        features.update(self._generate_innovation_stagnation_features(processed_data))
        
        # 8. 市場環境変化対応力特徴量
        features.update(self._generate_market_adaptation_features(processed_data))
        
        # 9. 時系列パターン異常特徴量
        features.update(self._generate_time_series_anomaly_features(processed_data))
        
        # 10. 複合リスク指標特徴量
        features.update(self._generate_composite_risk_features(features))
        
        # 結果統合とクリーニング
        feature_df = self._combine_and_clean_features(features, processed_data)
        
        # 消滅イベントとの関連付け
        final_features = self._link_extinction_events(feature_df, extinction_events)
        
        logger.info(f"消滅予兆特徴量生成完了: {final_features.shape[1]}個の特徴量")
        return final_features
    
    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """財務データの前処理"""
        processed = data.copy()
        
        # 基本的なクリーニング
        processed = processed.replace([np.inf, -np.inf], np.nan)
        
        # 異常値の処理
        for col in processed.select_dtypes(include=[np.number]).columns:
            Q1 = processed[col].quantile(0.25)
            Q3 = processed[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            processed[col] = processed[col].clip(lower_bound, upper_bound)
        
        return processed
    
    def _generate_financial_health_features(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """財務健全性悪化特徴量生成"""
        features = {}
        
        try:
            # 自己資本比率の急激な低下
            if 'equity_ratio' in data.columns:
                features['equity_ratio_decline'] = self._calculate_decline_rate(
                    data['equity_ratio'], periods=3
                )
                features['equity_ratio_volatility'] = data.groupby('company_id')['equity_ratio'].rolling(
                    self.config.volatility_window
                ).std().reset_index(level=0, drop=True)
            
            # 有利子負債比率の急増
            if 'debt_ratio' in data.columns:
                features['debt_ratio_surge'] = self._calculate_surge_rate(
                    data['debt_ratio'], periods=2
                )
            
            # 流動比率の悪化
            if 'current_ratio' in data.columns:
                features['current_ratio_deterioration'] = self._calculate_deterioration_score(
                    data['current_ratio'], benchmark=1.5
                )
            
            # 債務超過リスク
            if 'total_assets' in data.columns and 'total_liabilities' in data.columns:
                net_assets = data['total_assets'] - data['total_liabilities']
                features['insolvency_risk'] = (net_assets < 0).astype(int)
                features['net_assets_trend'] = self._calculate_trend_score(net_assets)
            
            # 金融費用負担率の上昇
            if 'interest_expense' in data.columns and 'revenue' in data.columns:
                interest_burden = data['interest_expense'] / data['revenue']
                features['interest_burden_increase'] = self._calculate_increase_rate(
                    interest_burden, periods=2
                )
            
        except Exception as e:
            logger.warning(f"財務健全性特徴量生成でエラー: {e}")
        
        return features
    
    def _generate_profitability_deterioration_features(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """収益性悪化特徴量生成"""
        features = {}
        
        try:
            # 営業利益率の継続的悪化
            if 'operating_profit_margin' in data.columns:
                features['operating_margin_decline'] = self._calculate_consecutive_decline(
                    data['operating_profit_margin'], consecutive_periods=3
                )
                features['operating_margin_volatility'] = self._calculate_rolling_volatility(
                    data['operating_profit_margin']
                )
            
            # 売上高の減少トレンド
            if 'revenue' in data.columns:
                features['revenue_decline_trend'] = self._calculate_decline_trend(
                    data['revenue'], periods=self.config.trend_window
                )
                features['revenue_growth_deceleration'] = self._calculate_growth_deceleration(
                    data['revenue']
                )
            
            # ROE・ROAの悪化
            if 'roe' in data.columns:
                features['roe_deterioration'] = self._calculate_deterioration_score(
                    data['roe'], benchmark=0.05
                )
            
            if 'roa' in data.columns:
                features['roa_negative_trend'] = (data['roa'] < 0).astype(int)
                features['roa_decline_rate'] = self._calculate_decline_rate(data['roa'])
            
            # 売上総利益率の圧迫
            if 'gross_profit_margin' in data.columns:
                features['gross_margin_compression'] = self._calculate_compression_score(
                    data['gross_profit_margin']
                )
            
            # 赤字継続期間
            if 'net_income' in data.columns:
                features['loss_consecutive_periods'] = self._calculate_consecutive_periods(
                    data['net_income'] < 0
                )
        
        except Exception as e:
            logger.warning(f"収益性特徴量生成でエラー: {e}")
        
        return features
    
    def _generate_liquidity_crisis_features(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """流動性・資金繰り危険特徴量生成"""
        features = {}
        
        try:
            # 当座比率の危険水準
            if 'quick_ratio' in data.columns:
                features['quick_ratio_crisis'] = (data['quick_ratio'] < 0.8).astype(int)
                features['quick_ratio_decline'] = self._calculate_decline_rate(data['quick_ratio'])
            
            # 営業キャッシュフローの悪化
            if 'operating_cash_flow' in data.columns:
                features['ocf_negative_periods'] = self._calculate_consecutive_periods(
                    data['operating_cash_flow'] < 0
                )
                features['ocf_volatility'] = self._calculate_rolling_volatility(
                    data['operating_cash_flow']
                )
            
            # フリーキャッシュフローの継続的マイナス
            if 'free_cash_flow' in data.columns:
                features['fcf_negative_trend'] = self._calculate_negative_trend(
                    data['free_cash_flow']
                )
            
            # 現金・預金の急減
            if 'cash_and_deposits' in data.columns:
                features['cash_depletion_rate'] = self._calculate_depletion_rate(
                    data['cash_and_deposits']
                )
            
            # 運転資本回転率の悪化
            if all(col in data.columns for col in ['revenue', 'current_assets', 'current_liabilities']):
                working_capital = data['current_assets'] - data['current_liabilities']
                working_capital_turnover = data['revenue'] / working_capital
                features['working_capital_deterioration'] = self._calculate_deterioration_score(
                    working_capital_turnover, benchmark=2.0
                )
        
        except Exception as e:
            logger.warning(f"流動性特徴量生成でエラー: {e}")
        
        return features
    
    def _generate_competitiveness_loss_features(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """成長性・市場競争力喪失特徴量生成"""
        features = {}
        
        try:
            # 市場シェア低下（推定）
            if 'revenue' in data.columns:
                # 業界内での相対的な売上成長率の低下を代理指標として使用
                features['relative_growth_decline'] = self._calculate_relative_performance_decline(
                    data['revenue']
                )
            
            # 研究開発費の削減
            if 'rd_expenses' in data.columns and 'revenue' in data.columns:
                rd_ratio = data['rd_expenses'] / data['revenue']
                features['rd_ratio_decline'] = self._calculate_decline_rate(rd_ratio)
                features['rd_investment_stagnation'] = self._calculate_stagnation_score(
                    data['rd_expenses']
                )
            
            # 設備投資の停滞
            if 'capex' in data.columns:
                features['capex_stagnation'] = self._calculate_stagnation_score(data['capex'])
                features['capex_decline_trend'] = self._calculate_decline_trend(data['capex'])
            
            # 従業員数の減少
            if 'employee_count' in data.columns:
                features['employee_reduction'] = self._calculate_reduction_rate(
                    data['employee_count']
                )
                features['employee_reduction_consecutive'] = self._calculate_consecutive_decline(
                    data['employee_count']
                )
            
            # 海外売上比率の低下（グローバル競争力）
            if 'overseas_revenue_ratio' in data.columns:
                features['global_competitiveness_decline'] = self._calculate_decline_rate(
                    data['overseas_revenue_ratio']
                )
        
        except Exception as e:
            logger.warning(f"競争力特徴量生成でエラー: {e}")
        
        return features
    
    def _generate_leverage_burden_features(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """レバレッジ・債務負担特徴量生成"""
        features = {}
        
        try:
            # デット・エクイティ・レシオの危険水準
            if 'debt_to_equity' in data.columns:
                features['de_ratio_crisis'] = (data['debt_to_equity'] > 2.0).astype(int)
                features['de_ratio_increase'] = self._calculate_increase_rate(data['debt_to_equity'])
            
            # インタレスト・カバレッジ・レシオの悪化
            if 'ebit' in data.columns and 'interest_expense' in data.columns:
                interest_coverage = data['ebit'] / (data['interest_expense'] + 1e-6)
                features['interest_coverage_crisis'] = (interest_coverage < 1.5).astype(int)
                features['interest_coverage_decline'] = self._calculate_decline_rate(interest_coverage)
            
            # 債務償還年数の長期化
            if all(col in data.columns for col in ['total_debt', 'operating_cash_flow']):
                debt_service_years = data['total_debt'] / (data['operating_cash_flow'] + 1e-6)
                features['debt_service_burden'] = self._calculate_burden_score(debt_service_years)
            
            # 短期借入金比率の上昇
            if all(col in data.columns for col in ['short_term_debt', 'total_debt']):
                short_term_debt_ratio = data['short_term_debt'] / data['total_debt']
                features['short_term_debt_dependency'] = self._calculate_dependency_score(
                    short_term_debt_ratio
                )
        
        except Exception as e:
            logger.warning(f"レバレッジ特徴量生成でエラー: {e}")
        
        return features
    
    def _generate_operational_efficiency_features(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """事業効率性悪化特徴量生成"""
        features = {}
        
        try:
            # 総資産回転率の低下
            if 'total_asset_turnover' in data.columns:
                features['asset_turnover_decline'] = self._calculate_decline_rate(
                    data['total_asset_turnover']
                )
            
            # 売上債権回転率の悪化
            if 'receivables_turnover' in data.columns:
                features['receivables_collection_deterioration'] = self._calculate_deterioration_score(
                    data['receivables_turnover'], benchmark=6.0
                )
            
            # 棚卸資産回転率の低下
            if 'inventory_turnover' in data.columns:
                features['inventory_efficiency_decline'] = self._calculate_decline_rate(
                    data['inventory_turnover']
                )
            
            # 固定資産効率性の悪化
            if all(col in data.columns for col in ['revenue', 'fixed_assets']):
                fixed_asset_turnover = data['revenue'] / data['fixed_assets']
                features['fixed_asset_efficiency_decline'] = self._calculate_decline_rate(
                    fixed_asset_turnover
                )
            
            # 労働生産性の低下
            if all(col in data.columns for col in ['revenue', 'employee_count']):
                labor_productivity = data['revenue'] / data['employee_count']
                features['labor_productivity_decline'] = self._calculate_decline_rate(
                    labor_productivity
                )
        
        except Exception as e:
            logger.warning(f"事業効率性特徴量生成でエラー: {e}")
        
        return features
    
    def _generate_innovation_stagnation_features(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """投資・イノベーション停滞特徴量生成"""
        features = {}
        
        try:
            # 無形固定資産の停滞
            if 'intangible_assets' in data.columns:
                features['intangible_assets_stagnation'] = self._calculate_stagnation_score(
                    data['intangible_assets']
                )
            
            # IT投資の削減
            if 'it_investment' in data.columns:
                features['it_investment_decline'] = self._calculate_decline_rate(
                    data['it_investment']
                )
            
            # 特許・技術開発投資の減少
            if 'patent_expenses' in data.columns:
                features['innovation_investment_decline'] = self._calculate_decline_rate(
                    data['patent_expenses']
                )
            
            # 人材投資の削減
            if 'training_expenses' in data.columns:
                features['human_capital_investment_decline'] = self._calculate_decline_rate(
                    data['training_expenses']
                )
        
        except Exception as e:
            logger.warning(f"イノベーション特徴量生成でエラー: {e}")
        
        return features
    
    def _generate_market_adaptation_features(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """市場環境変化対応力特徴量生成"""
        features = {}
        
        try:
            # 新規事業・セグメントへの投資停滞
            if 'new_business_investment' in data.columns:
                features['market_adaptation_stagnation'] = self._calculate_stagnation_score(
                    data['new_business_investment']
                )
            
            # デジタル化投資の遅れ
            if 'digital_transformation_investment' in data.columns:
                features['digital_adaptation_lag'] = self._calculate_lag_score(
                    data['digital_transformation_investment']
                )
            
            # 販売・マーケティング費用の削減
            if 'sales_marketing_expenses' in data.columns:
                features['market_investment_decline'] = self._calculate_decline_rate(
                    data['sales_marketing_expenses']
                )
        
        except Exception as e:
            logger.warning(f"市場適応特徴量生成でエラー: {e}")
        
        return features
    
    def _generate_time_series_anomaly_features(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """時系列パターン異常特徴量生成"""
        features = {}
        
        try:
            # 財務指標の異常なボラティリティ
            key_metrics = ['revenue', 'operating_profit', 'net_income', 'total_assets']
            for metric in key_metrics:
                if metric in data.columns:
                    features[f'{metric}_volatility_anomaly'] = self._detect_volatility_anomaly(
                        data[metric]
                    )
            
            # 季節性の消失
            features['seasonality_loss'] = self._detect_seasonality_loss(data)
            
            # 突発的な指標悪化
            features['sudden_deterioration'] = self._detect_sudden_changes(data)
        
        except Exception as e:
            logger.warning(f"時系列異常特徴量生成でエラー: {e}")
        
        return features
    
    def _generate_composite_risk_features(self, features: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """複合リスク指標特徴量生成"""
        composite_features = {}
        
        try:
            # Altman Z-Score的な複合指標
            risk_components = []
            for feature_name, feature_data in features.items():
                if 'crisis' in feature_name or 'deterioration' in feature_name:
                    risk_components.append(feature_data.fillna(0))
            
            if risk_components:
                composite_features['composite_extinction_risk'] = sum(risk_components) / len(risk_components)
            
            # 財務健全性総合スコア
            health_components = []
            for feature_name, feature_data in features.items():
                if any(keyword in feature_name for keyword in ['equity', 'liquidity', 'cash']):
                    health_components.append(feature_data.fillna(0))
            
            if health_components:
                composite_features['financial_health_composite'] = sum(health_components) / len(health_components)
            
            # 事業継続性リスク総合スコア
            continuity_components = []
            for feature_name, feature_data in features.items():
                if any(keyword in feature_name for keyword in ['decline', 'stagnation', 'reduction']):
                    continuity_components.append(feature_data.fillna(0))
            
            if continuity_components:
                composite_features['business_continuity_risk'] = sum(continuity_components) / len(continuity_components)
        
        except Exception as e:
            logger.warning(f"複合リスク特徴量生成でエラー: {e}")
        
        return composite_features
    
    # ヘルパーメソッド群
    def _calculate_decline_rate(self, series: pd.Series, periods: int = 2) -> pd.Series:
        """指標の低下率を計算"""
        return (series - series.shift(periods)) / (series.shift(periods) + 1e-6)
    
    def _calculate_surge_rate(self, series: pd.Series, periods: int = 2) -> pd.Series:
        """指標の急上昇率を計算"""
        return (series - series.shift(periods)) / (series.shift(periods) + 1e-6)
    
    def _calculate_deterioration_score(self, series: pd.Series, benchmark: float) -> pd.Series:
        """ベンチマーク比での悪化スコア"""
        return (benchmark - series) / benchmark
    
    def _calculate_trend_score(self, series: pd.Series, periods: int = 5) -> pd.Series:
        """トレンドスコア（正：上昇、負：下降）"""
        return series.rolling(periods).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == periods else np.nan)
    
    def _calculate_increase_rate(self, series: pd.Series, periods: int = 2) -> pd.Series:
        """増加率計算"""
        return (series - series.shift(periods)) / (series.shift(periods) + 1e-6)
    
    def _calculate_consecutive_decline(self, series: pd.Series, consecutive_periods: int = 3) -> pd.Series:
        """連続的な減少期間の計算"""
        decline_mask = series < series.shift(1)
        consecutive_decline = pd.Series(0, index=series.index)
        
        for i in range(consecutive_periods, len(series)):
            if all(decline_mask.iloc[i-consecutive_periods+1:i+1]):
                consecutive_decline.iloc[i] = 1
        
        return consecutive_decline
    
    def _calculate_rolling_volatility(self, series: pd.Series, window: int = None) -> pd.Series:
        """ローリングボラティリティ計算"""
        window = window or self.config.volatility_window
        return series.rolling(window).std()
    
    def _calculate_decline_trend(self, series: pd.Series, periods: int) -> pd.Series:
        """減少トレンド計算"""
        return series.rolling(periods).apply(
            lambda x: 1 if len(x) == periods and x.iloc[-1] < x.iloc[0] else 0
        )
    
    def _calculate_growth_deceleration(self, series: pd.Series) -> pd.Series:
        """成長減速度計算"""
        growth_rate = series.pct_change()
        return growth_rate.diff()
    
    def _calculate_compression_score(self, series: pd.Series) -> pd.Series:
        """圧迫スコア計算"""
        rolling_mean = series.rolling(5).mean()
        return (rolling_mean - series) / (rolling_mean + 1e-6)
    
    def _calculate_consecutive_periods(self, condition: pd.Series) -> pd.Series:
        """条件を満たす連続期間の計算"""
        consecutive_count = pd.Series(0, index=condition.index)
        current_count = 0
        
        for i, value in condition.items():
            if value:
                current_count += 1
            else:
                current_count = 0
            consecutive_count[i] = current_count
        
        return consecutive_count
    
    def _calculate_negative_trend(self, series: pd.Series) -> pd.Series:
        """負の値のトレンド計算"""
        negative_mask = series < 0
        return negative_mask.rolling(3).sum() >= 2
    
    def _calculate_depletion_rate(self, series: pd.Series) -> pd.Series:
        """枯渇率計算"""
        return -(series - series.shift(4)) / (series.shift(4) + 1e-6)
    
    def _calculate_relative_performance_decline(self, series: pd.Series) -> pd.Series:
        """相対的パフォーマンス低下"""
        industry_median = series.groupby(series.index.get_level_values(0)).median()
        relative_performance = series / industry_median
        return self._calculate_decline_rate(relative_performance)
    
    def _calculate_stagnation_score(self, series: pd.Series) -> pd.Series:
        """停滞スコア計算"""
        return (series.rolling(5).std() < series.std() * 0.1).astype(int)
    
    def _calculate_reduction_rate(self, series: pd.Series) -> pd.Series:
        """削減率計算"""
        return -(series - series.shift(1)) / (series.shift(1) + 1e-6)
    
    def _calculate_burden_score(self, series: pd.Series) -> pd.Series:
        """負担スコア計算"""
        return np.log1p(series.clip(0, np.inf))
    
    def _calculate_dependency_score(self, series: pd.Series) -> pd.Series:
        """依存度スコア計算"""
        return series ** 2  # 二乗で非線形な依存度を表現
    
    def _calculate_lag_score(self, series: pd.Series) -> pd.Series:
        """遅れスコア計算"""
        industry_average = series.groupby(series.index.get_level_values(0)).mean()
        return (industry_average - series) / (industry_average + 1e-6)
    
    def _detect_volatility_anomaly(self, series: pd.Series) -> pd.Series:
        """ボラティリティ異常検出"""
        volatility = series.rolling(5).std()
        volatility_threshold = volatility.quantile(0.9)
        return (volatility > volatility_threshold).astype(int)
    
    def _detect_seasonality_loss(self, data: pd.DataFrame) -> pd.Series:
        """季節性消失検出"""
        # 簡単な季節性検出（実装を簡略化）
        if 'revenue' in data.columns:
            revenue_seasonal = data['revenue'].groupby(data.index.month).std()
            return (revenue_seasonal < revenue_seasonal.mean() * 0.5).astype(int).iloc[0]
        return pd.Series(0, index=data.index)
    
    def _detect_sudden_changes(self, data: pd.DataFrame) -> pd.Series:
        """突発的な変化検出"""
        sudden_changes = pd.Series(0, index=data.index)
        
        key_metrics = ['revenue', 'operating_profit', 'net_income']
        for metric in key_metrics:
            if metric in data.columns:
                pct_change = data[metric].pct_change()
                sudden_changes += (abs(pct_change) > 0.3).astype(int)
        
        return sudden_changes
    
    def _combine_and_clean_features(self, features: Dict[str, pd.Series], 
                                    original_data: pd.DataFrame) -> pd.DataFrame:
        """特徴量の統合とクリーニング"""
        # 特徴量データフレーム作成
        feature_df = pd.DataFrame(features, index=original_data.index)
        
        # 欠損値処理
        feature_df = feature_df.fillna(method='ffill').fillna(0)
        
        # 無限大値の処理
        feature_df = feature_df.replace([np.inf, -np.inf], np.nan)
        feature_df = feature_df.fillna(0)
        
        # 異常値の制限
        for col in feature_df.select_dtypes(include=[np.number]).columns:
            Q99 = feature_df[col].quantile(0.99)
            Q1 = feature_df[col].quantile(0.01)
            feature_df[col] = feature_df[col].clip(Q1, Q99)
        
        return feature_df
    
    def _link_extinction_events(self, feature_df: pd.DataFrame, 
                                extinction_events: pd.DataFrame) -> pd.DataFrame:
        """消滅イベントとの関連付け"""
        # 企業IDと年度でマージ
        if 'company_id' in extinction_events.columns and 'extinction_year' in extinction_events.columns:
            # 消滅フラグの追加
            feature_df = feature_df.reset_index()
            
            # 消滅予定企業のマーキング
            for _, event in extinction_events.iterrows():
                company_mask = feature_df['company_id'] == event['company_id']
                year_mask = feature_df['year'] <= event['extinction_year']
                feature_df.loc[company_mask & year_mask, 'extinction_target'] = 1
            
            # 消滅までの期間
            for _, event in extinction_events.iterrows():
                company_mask = feature_df['company_id'] == event['company_id']
                feature_df.loc[company_mask, 'years_to_extinction'] = (
                    event['extinction_year'] - feature_df.loc[company_mask, 'year']
                )
            
            feature_df = feature_df.set_index(['company_id', 'year'])
        
        # デフォルト値設定
        feature_df['extinction_target'] = feature_df.get('extinction_target', 0)
        feature_df['years_to_extinction'] = feature_df.get('years_to_extinction', np.inf)
        
        return feature_df
    
    def generate_market_specific_features(self, data: pd.DataFrame, 
                                        market_category: str) -> pd.DataFrame:
        """市場カテゴリ別特化特徴量生成"""
        logger.info(f"{market_category}市場向け特化特徴量生成")
        
        if market_category == "high_share":
            return self._generate_high_share_specific_features(data)
        elif market_category == "declining":
            return self._generate_declining_market_features(data)
        elif market_category == "lost":
            return self._generate_lost_market_features(data)
        else:
            logger.warning(f"未知の市場カテゴリ: {market_category}")
            return data
    
    def _generate_high_share_specific_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """高シェア市場特化特徴量"""
        features = data.copy()
        
        try:
            # 技術優位性の維持度
            if 'rd_expenses' in data.columns and 'revenue' in data.columns:
                rd_intensity = data['rd_expenses'] / data['revenue']
                features['rd_intensity_maintenance'] = (rd_intensity >= rd_intensity.rolling(5).mean()).astype(int)
            
            # 品質投資の継続性
            if 'quality_investment' in data.columns:
                features['quality_investment_consistency'] = self._calculate_consistency_score(
                    data['quality_investment']
                )
            
            # グローバル展開の維持
            if 'overseas_revenue_ratio' in data.columns:
                features['global_presence_stability'] = self._calculate_stability_score(
                    data['overseas_revenue_ratio']
                )
        
        except Exception as e:
            logger.warning(f"高シェア市場特徴量生成でエラー: {e}")
        
        return features
    
    def _generate_declining_market_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """シェア低下市場特化特徴量"""
        features = data.copy()
        
        try:
            # 市場適応速度の遅れ
            if 'new_technology_investment' in data.columns:
                features['adaptation_speed_lag'] = self._calculate_adaptation_lag(
                    data['new_technology_investment']
                )
            
            # コスト競争力の悪化
            if 'cost_competitiveness' in data.columns:
                features['cost_competitiveness_decline'] = self._calculate_decline_rate(
                    data['cost_competitiveness']
                )
            
            # 事業転換の遅れ
            if 'business_transformation_investment' in data.columns:
                features['transformation_delay'] = self._calculate_delay_score(
                    data['business_transformation_investment']
                )
        
        except Exception as e:
            logger.warning(f"シェア低下市場特徴量生成でエラー: {e}")
        
        return features
    
    def _generate_lost_market_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """失失市場特化特徴量"""
        features = data.copy()
        
        try:
            # 技術革新への対応遅れ
            if 'innovation_response_investment' in data.columns:
                features['innovation_response_failure'] = self._calculate_response_failure_score(
                    data['innovation_response_investment']
                )
            
            # 市場撤退の前兆
            if 'market_exit_preparation' in data.columns:
                features['exit_preparation_signals'] = self._calculate_exit_signals(
                    data['market_exit_preparation']
                )
            
            # 事業再編の失敗
            if 'restructuring_effectiveness' in data.columns:
                features['restructuring_failure'] = self._calculate_restructuring_failure(
                    data['restructuring_effectiveness']
                )
        
        except Exception as e:
            logger.warning(f"失失市場特徴量生成でエラー: {e}")
        
        return features
    
    # 市場特化ヘルパーメソッド
    def _calculate_consistency_score(self, series: pd.Series) -> pd.Series:
        """一貫性スコア計算"""
        cv = series.rolling(5).std() / (series.rolling(5).mean() + 1e-6)
        return 1 / (1 + cv)  # 変動係数の逆数
    
    def _calculate_stability_score(self, series: pd.Series) -> pd.Series:
        """安定性スコア計算"""
        return 1 - series.rolling(5).std() / (series.std() + 1e-6)
    
    def _calculate_adaptation_lag(self, series: pd.Series) -> pd.Series:
        """適応遅れスコア計算"""
        industry_median = series.groupby(series.index.get_level_values(0)).median()
        return (industry_median - series) / (industry_median + 1e-6)
    
    def _calculate_delay_score(self, series: pd.Series) -> pd.Series:
        """遅延スコア計算"""
        expected_growth = series.rolling(3).mean().shift(1)
        actual_growth = series
        return (expected_growth - actual_growth) / (expected_growth + 1e-6)
    
    def _calculate_response_failure_score(self, series: pd.Series) -> pd.Series:
        """対応失敗スコア計算"""
        response_threshold = series.quantile(0.5)
        return (series < response_threshold).astype(int)
    
    def _calculate_exit_signals(self, series: pd.Series) -> pd.Series:
        """撤退シグナル計算"""
        return (series > series.rolling(3).mean()).astype(int)
    
    def _calculate_restructuring_failure(self, series: pd.Series) -> pd.Series:
        """再編失敗スコア計算"""
        return (series < 0).astype(int)
    
    def calculate_extinction_probability_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """消滅確率計算用特徴量生成"""
        logger.info("消滅確率計算用特徴量生成開始")
        
        prob_features = data.copy()
        
        try:
            # Altman Z-Score風の複合指標
            if all(col in data.columns for col in ['current_assets', 'current_liabilities', 
                                                    'retained_earnings', 'total_assets',
                                                    'ebit', 'market_value_equity', 'total_liabilities',
                                                    'revenue']):
                
                # 運転資本/総資産
                working_capital = data['current_assets'] - data['current_liabilities']
                wc_ta = working_capital / data['total_assets']
                
                # 留保利益/総資産
                re_ta = data['retained_earnings'] / data['total_assets']
                
                # EBIT/総資産
                ebit_ta = data['ebit'] / data['total_assets']
                
                # 時価総額/負債総額
                me_tl = data['market_value_equity'] / data['total_liabilities']
                
                # 売上高/総資産
                s_ta = data['revenue'] / data['total_assets']
                
                # 修正Altman Z-Score
                prob_features['modified_z_score'] = (
                    1.2 * wc_ta + 1.4 * re_ta + 3.3 * ebit_ta + 0.6 * me_tl + 1.0 * s_ta
                )
                
                # 破綻リスクカテゴリ
                prob_features['bankruptcy_risk_level'] = pd.cut(
                    prob_features['modified_z_score'],
                    bins=[-np.inf, 1.8, 3.0, np.inf],
                    labels=[2, 1, 0]  # 2:高リスク, 1:中リスク, 0:低リスク
                ).astype(int)
            
            # 財務比率トレンド分析
            key_ratios = ['equity_ratio', 'current_ratio', 'operating_profit_margin', 'roe']
            for ratio in key_ratios:
                if ratio in data.columns:
                    # 3年間のトレンド
                    trend_3y = self._calculate_trend_score(data[ratio], periods=3)
                    prob_features[f'{ratio}_trend_3y'] = trend_3y
                    
                    # 悪化の加速度
                    acceleration = trend_3y.diff()
                    prob_features[f'{ratio}_deterioration_acceleration'] = acceleration.clip(-1, 0)
            
            # キャッシュフロー枯渇リスク
            if 'operating_cash_flow' in data.columns and 'capex' in data.columns:
                fcf = data['operating_cash_flow'] - data['capex']
                cumulative_fcf = fcf.rolling(3).sum()
                prob_features['cash_flow_depletion_risk'] = (cumulative_fcf < 0).astype(int)
            
            # 売上高減少の継続性
            if 'revenue' in data.columns:
                revenue_decline = data['revenue'].pct_change() < -0.05
                prob_features['revenue_decline_persistence'] = revenue_decline.rolling(3).sum()
        
        except Exception as e:
            logger.warning(f"消滅確率特徴量計算でエラー: {e}")
        
        return prob_features
    
    def generate_early_warning_indicators(self, data: pd.DataFrame, 
                                        lead_time: int = 3) -> pd.DataFrame:
        """早期警告指標生成"""
        logger.info(f"{lead_time}年前の早期警告指標生成")
        
        warning_features = data.copy()
        
        try:
            # 利益質の悪化（営業CFと当期純利益の乖離）
            if 'operating_cash_flow' in data.columns and 'net_income' in data.columns:
                profit_quality = data['operating_cash_flow'] - data['net_income']
                warning_features['profit_quality_deterioration'] = self._calculate_deterioration_score(
                    profit_quality, benchmark=0
                )
            
            # 資産収益性の継続的低下
            if 'roa' in data.columns:
                roa_trend = self._calculate_trend_score(data['roa'], periods=lead_time)
                warning_features['roa_declining_trend'] = (roa_trend < -0.01).astype(int)
            
            # 競争ポジションの悪化
            if 'market_share' in data.columns:
                market_share_trend = self._calculate_trend_score(data['market_share'], periods=lead_time)
                warning_features['market_position_weakening'] = (market_share_trend < 0).astype(int)
            
            # 人材流出の兆候
            if 'employee_turnover_rate' in data.columns:
                warning_features['talent_drain_risk'] = (
                    data['employee_turnover_rate'] > data['employee_turnover_rate'].rolling(5).mean()
                ).astype(int)
            
            # 投資効率の悪化
            if all(col in data.columns for col in ['capex', 'revenue']):
                capex_efficiency = data['revenue'].pct_change() / (data['capex'].pct_change() + 1e-6)
                warning_features['investment_efficiency_decline'] = (capex_efficiency < 0).astype(int)
        
        except Exception as e:
            logger.warning(f"早期警告指標生成でエラー: {e}")
        
        return warning_features
    
    def generate_distress_prediction_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """企業窮地予測特徴量生成"""
        logger.info("企業窮地予測特徴量生成開始")
        
        distress_features = data.copy()
        
        try:
            # 複数年赤字の継続
            if 'net_income' in data.columns:
                loss_periods = self._calculate_consecutive_periods(data['net_income'] < 0)
                distress_features['chronic_losses'] = (loss_periods >= 3).astype(int)
            
            # 債務不履行リスク
            if all(col in data.columns for col in ['ebit', 'interest_expense']):
                interest_coverage = data['ebit'] / (data['interest_expense'] + 1e-6)
                distress_features['default_risk'] = (interest_coverage < 1.0).astype(int)
            
            # 資産売却の急増
            if 'asset_disposal_income' in data.columns:
                asset_disposal_surge = data['asset_disposal_income'].pct_change() > 0.5
                distress_features['asset_liquidation_pressure'] = asset_disposal_surge.astype(int)
            
            # 事業セグメントの急激な縮小
            if 'segment_count' in data.columns:
                segment_reduction = self._calculate_reduction_rate(data['segment_count'])
                distress_features['business_contraction'] = (segment_reduction > 0.2).astype(int)
            
            # 監査意見の悪化
            if 'audit_opinion_score' in data.columns:
                distress_features['audit_concern'] = (data['audit_opinion_score'] < 3).astype(int)
        
        except Exception as e:
            logger.warning(f"窮地予測特徴量生成でエラー: {e}")
        
        return distress_features
    
    def create_survival_features_summary(self, features: pd.DataFrame) -> Dict[str, any]:
        """生存分析用特徴量サマリー作成"""
        summary = {
            'total_features': len(features.columns),
            'companies_count': len(features.index.get_level_values(0).unique()),
            'time_period': f"{features.index.get_level_values(1).min()}-{features.index.get_level_values(1).max()}",
            'extinction_target_count': features['extinction_target'].sum() if 'extinction_target' in features.columns else 0,
            'feature_categories': {
                'financial_health': len([col for col in features.columns if 'equity' in col or 'debt' in col]),
                'profitability': len([col for col in features.columns if 'profit' in col or 'margin' in col]),
                'liquidity': len([col for col in features.columns if 'cash' in col or 'liquidity' in col]),
                'efficiency': len([col for col in features.columns if 'turnover' in col or 'efficiency' in col]),
                'growth': len([col for col in features.columns if 'growth' in col or 'decline' in col]),
                'risk': len([col for col in features.columns if 'risk' in col or 'crisis' in col])
            },
            'data_quality': {
                'missing_ratio': features.isnull().sum().sum() / (features.shape[0] * features.shape[1]),
                'infinite_values': np.isinf(features.select_dtypes(include=[np.number])).sum().sum(),
                'zero_variance_features': (features.var() == 0).sum()
            }
        }
        
        return summary
    
    def validate_extinction_features(self, features: pd.DataFrame) -> Tuple[bool, List[str]]:
        """消滅予兆特徴量の妥当性検証"""
        issues = []
        
        # 必須特徴量の存在確認
        required_features = [
            'composite_extinction_risk',
            'financial_health_composite',
            'business_continuity_risk'
        ]
        
        for feature in required_features:
            if feature not in features.columns:
                issues.append(f"必須特徴量が不足: {feature}")
        
        # データ品質チェック
        if features.isnull().sum().sum() > len(features) * 0.1:
            issues.append("欠損値が多すぎます（10%超）")
        
        if (features.select_dtypes(include=[np.number]) == 0).all().any():
            issues.append("すべて0の特徴量が存在します")
        
        # 特徴量の分散チェック
        low_variance_features = features.columns[features.var() < 1e-6].tolist()
        if low_variance_features:
            issues.append(f"分散が極端に小さい特徴量: {low_variance_features}")
        
        # 異常値チェック
        infinite_cols = features.columns[np.isinf(features.select_dtypes(include=[np.number])).any()].tolist()
        if infinite_cols:
            issues.append(f"無限大値を含む特徴量: {infinite_cols}")
        
        is_valid = len(issues) == 0
        return is_valid, issues
    
    def save_extinction_features(self, features: pd.DataFrame, 
                                output_path: str, 
                                include_metadata: bool = True) -> None:
        """消滅予兆特徴量の保存"""
        logger.info(f"消滅予兆特徴量を保存: {output_path}")
        
        # メタデータ付きで保存
        if include_metadata:
            metadata = self.create_survival_features_summary(features)
            
            # メタデータファイル保存
            metadata_path = output_path.replace('.csv', '_metadata.json')
            import json
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2, default=str)
        
        # 特徴量データ保存
        features.to_csv(output_path, encoding='utf-8')
        logger.info("保存完了")


# 使用例とテスト関数
def test_extinction_feature_engineer():
    """ExtinctionFeatureEngineerのテスト関数"""
    # サンプルデータ作成
    np.random.seed(42)
    companies = [f"company_{i}" for i in range(10)]
    years = list(range(2020, 2025))
    
    # マルチインデックス作成
    index = pd.MultiIndex.from_product([companies, years], names=['company_id', 'year'])
    
    # サンプル財務データ
    sample_data = pd.DataFrame({
        'revenue': np.random.normal(100, 20, len(index)),
        'operating_profit': np.random.normal(10, 5, len(index)),
        'net_income': np.random.normal(5, 8, len(index)),
        'total_assets': np.random.normal(200, 50, len(index)),
        'total_liabilities': np.random.normal(120, 30, len(index)),
        'equity_ratio': np.random.uniform(0.1, 0.8, len(index)),
        'current_ratio': np.random.uniform(0.5, 3.0, len(index)),
        'operating_cash_flow': np.random.normal(12, 6, len(index)),
        'rd_expenses': np.random.normal(3, 1, len(index)),
        'capex': np.random.normal(8, 3, len(index)),
    }, index=index)
    
    # サンプル消滅イベント
    extinction_events = pd.DataFrame({
        'company_id': ['company_8', 'company_9'],
        'extinction_year': [2023, 2024],
        'extinction_type': ['bankruptcy', 'merger']
    })
    
    # 特徴量エンジニア実行
    engineer = ExtinctionFeatureEngineer()
    extinction_features = engineer.generate_extinction_features(sample_data, extinction_events)
    
    print("消滅予兆特徴量生成テスト完了")
    print(f"生成特徴量数: {extinction_features.shape[1]}")
    print(f"対象企業数: {len(extinction_features.index.get_level_values(0).unique())}")
    
    # 妥当性検証
    is_valid, issues = engineer.validate_extinction_features(extinction_features)
    print(f"妥当性検証: {'合格' if is_valid else '問題あり'}")
    if issues:
        for issue in issues:
            print(f"  - {issue}")
    
    return extinction_features

if __name__ == "__main__":
    # テスト実行
    test_features = test_extinction_feature_engineer()