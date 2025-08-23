"""
A2AI (Advanced Financial Analysis AI) - 企業消滅リスク分析モジュール
企業の消滅・倒産リスクを財務指標から予測・分析するシステム

Author: A2AI Development Team
Version: 1.0.0
"""

import numpy as np
import pandas as pd
import warnings
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime, timedelta

# 機械学習・統計ライブラリ
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
import scipy.stats as stats

# A2AI内部モジュール
from ...utils.statistical_utils import StatisticalTest, DistributionAnalyzer
from ...utils.survival_utils import SurvivalDataProcessor, HazardCalculator
from ...utils.lifecycle_utils import LifecycleStageDetector
from ..traditional_analysis.factor_impact_analyzer import FactorImpactAnalyzer

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    """リスクレベル定義"""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"
    CRITICAL = "critical"

@dataclass
class ExtinctionRiskMetrics:
    """企業消滅リスクメトリクス"""
    company_id: str
    company_name: str
    analysis_date: datetime
    risk_score: float  # 0-1の消滅リスクスコア
    risk_level: RiskLevel
    survival_probability_1y: float  # 1年生存確率
    survival_probability_3y: float  # 3年生存確率
    survival_probability_5y: float  # 5年生存確率
    hazard_ratio: float  # ハザード比
    critical_factors: List[str]  # 重要リスク要因
    warning_indicators: List[str]  # 警告指標
    time_to_extinction_estimate: Optional[float]  # 推定消滅時期（年）
    confidence_interval: Tuple[float, float]  # 信頼区間
    market_category: str  # 市場カテゴリ

@dataclass
class RiskFactorContribution:
    """リスク要因寄与度"""
    factor_name: str
    contribution_score: float  # 0-1の寄与度
    current_value: float
    industry_median: float
    risk_threshold: float
    trend_direction: str  # "improving", "stable", "deteriorating"
    importance_rank: int

class ExtinctionRiskAnalyzer:
    """企業消滅リスク分析システム"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初期化
        
        Args:
            config: 設定パラメータ
        """
        self.config = config or self._default_config()
        self.scaler = RobustScaler()
        self.models = {}
        self.feature_importance = {}
        self.risk_thresholds = self._initialize_risk_thresholds()
        self.survival_processor = SurvivalDataProcessor()
        self.hazard_calculator = HazardCalculator()
        self.lifecycle_detector = LifecycleStageDetector()
        self.factor_analyzer = FactorImpactAnalyzer()
        
    def _default_config(self) -> Dict:
        """デフォルト設定"""
        return {
            'lookback_periods': [1, 2, 3, 5],  # 分析対象期間（年）
            'risk_factors': {
                # 財務健全性指標
                'liquidity': ['current_ratio', 'quick_ratio', 'cash_ratio'],
                'leverage': ['debt_equity_ratio', 'interest_coverage_ratio', 'debt_service_coverage'],
                'profitability': ['operating_margin', 'net_margin', 'roa', 'roe'],
                'efficiency': ['asset_turnover', 'inventory_turnover', 'receivables_turnover'],
                'growth': ['revenue_growth', 'profit_growth', 'asset_growth'],
                'market_position': ['market_share', 'market_share_trend', 'competitive_position'],
                'lifecycle': ['company_age', 'market_entry_timing', 'business_maturity']
            },
            'critical_thresholds': {
                'debt_equity_ratio': 3.0,
                'current_ratio': 1.0,
                'interest_coverage_ratio': 1.5,
                'operating_margin': -0.05,
                'cash_ratio': 0.05
            },
            'model_weights': {
                'logistic_regression': 0.25,
                'random_forest': 0.25,
                'gradient_boosting': 0.25,
                'cox_regression': 0.25
            }
        }
    
    def _initialize_risk_thresholds(self) -> Dict:
        """リスクレベル閾値初期化"""
        return {
            RiskLevel.VERY_LOW: (0.0, 0.1),
            RiskLevel.LOW: (0.1, 0.25),
            RiskLevel.MEDIUM: (0.25, 0.5),
            RiskLevel.HIGH: (0.5, 0.75),
            RiskLevel.VERY_HIGH: (0.75, 0.9),
            RiskLevel.CRITICAL: (0.9, 1.0)
        }
    
    def analyze_extinction_risk(
        self, 
        financial_data: pd.DataFrame,
        company_info: pd.DataFrame,
        market_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, ExtinctionRiskMetrics]:
        """
        企業消滅リスク分析実行
        
        Args:
            financial_data: 財務データ（企業×時系列×指標）
            company_info: 企業情報（企業ID、名前、市場カテゴリ等）
            market_data: 市場データ（オプション）
            
        Returns:
            企業ごとのリスクメトリクス
        """
        logger.info("企業消滅リスク分析を開始します")
        
        # データ前処理
        processed_data = self._preprocess_data(financial_data, company_info, market_data)
        
        # 特徴量エンジニアリング
        features = self._engineer_risk_features(processed_data)
        
        # モデル学習・予測
        predictions = self._train_and_predict(features, processed_data)
        
        # リスクメトリクス計算
        risk_metrics = self._calculate_risk_metrics(
            predictions, processed_data, company_info
        )
        
        logger.info(f"分析完了: {len(risk_metrics)}社のリスク評価を実行")
        return risk_metrics
    
    def _preprocess_data(
        self, 
        financial_data: pd.DataFrame,
        company_info: pd.DataFrame,
        market_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, pd.DataFrame]:
        """データ前処理"""
        
        # 企業存続・消滅ラベル作成
        survival_data = self._create_survival_labels(financial_data, company_info)
        
        # 欠損値処理
        clean_financial = self._handle_missing_values(financial_data)
        
        # 異常値処理
        clean_financial = self._handle_outliers(clean_financial)
        
        # 業界標準化
        normalized_data = self._industry_normalize(clean_financial, company_info)
        
        return {
            'financial': normalized_data,
            'survival': survival_data,
            'company_info': company_info,
            'market_data': market_data
        }
    
    def _create_survival_labels(
        self, 
        financial_data: pd.DataFrame,
        company_info: pd.DataFrame
    ) -> pd.DataFrame:
        """生存・消滅ラベル作成"""
        
        survival_labels = []
        
        for company_id in company_info['company_id']:
            company_financial = financial_data[
                financial_data['company_id'] == company_id
            ].sort_values('year')
            
            if company_financial.empty:
                continue
                
            # 消滅判定ロジック
            last_data_year = company_financial['year'].max()
            current_year = datetime.now().year
            
            # 3年以上データが途切れている場合は消滅と判定
            is_extinct = (current_year - last_data_year) >= 3
            
            # 倒産・解散情報がある場合
            company_row = company_info[company_info['company_id'] == company_id].iloc[0]
            if 'extinction_date' in company_row and pd.notna(company_row['extinction_date']):
                is_extinct = True
                extinction_year = pd.to_datetime(company_row['extinction_date']).year
            else:
                extinction_year = None
            
            # 各年のラベル作成
            for _, row in company_financial.iterrows():
                survival_time = (
                    extinction_year - row['year'] if extinction_year 
                    else current_year - row['year']
                )
                
                survival_labels.append({
                    'company_id': company_id,
                    'year': row['year'],
                    'is_extinct': is_extinct,
                    'extinction_year': extinction_year,
                    'survival_time': max(survival_time, 0),
                    'event_observed': is_extinct and (
                        extinction_year >= row['year'] if extinction_year else False
                    )
                })
        
        return pd.DataFrame(survival_labels)
    
    def _engineer_risk_features(self, processed_data: Dict) -> pd.DataFrame:
        """リスク予測用特徴量エンジニアリング"""
        
        financial_data = processed_data['financial']
        features_list = []
        
        # 基本財務比率特徴量
        basic_features = self._calculate_basic_risk_features(financial_data)
        features_list.append(basic_features)
        
        # トレンド特徴量
        trend_features = self._calculate_trend_features(financial_data)
        features_list.append(trend_features)
        
        # 変動性特徴量
        volatility_features = self._calculate_volatility_features(financial_data)
        features_list.append(volatility_features)
        
        # 相対位置特徴量（業界内順位等）
        relative_features = self._calculate_relative_features(
            financial_data, processed_data['company_info']
        )
        features_list.append(relative_features)
        
        # ライフサイクル特徴量
        lifecycle_features = self._calculate_lifecycle_features(
            financial_data, processed_data['company_info']
        )
        features_list.append(lifecycle_features)
        
        # 特徴量統合
        combined_features = pd.concat(features_list, axis=1)
        
        return combined_features
    
    def _calculate_basic_risk_features(self, financial_data: pd.DataFrame) -> pd.DataFrame:
        """基本リスク特徴量計算"""
        
        features = financial_data.copy()
        
        # 流動性リスク指標
        features['liquidity_risk'] = np.where(
            features['current_ratio'] < 1.0, 1, 0
        )
        
        # レバレッジリスク指標
        features['leverage_risk'] = np.where(
            features['debt_equity_ratio'] > 3.0, 1, 0
        )
        
        # 収益性リスク指標
        features['profitability_risk'] = np.where(
            features['operating_margin'] < -0.05, 1, 0
        )
        
        # 利払い能力リスク
        features['interest_risk'] = np.where(
            features['interest_coverage_ratio'] < 1.5, 1, 0
        )
        
        # 複合リスクスコア
        risk_indicators = [
            'liquidity_risk', 'leverage_risk', 
            'profitability_risk', 'interest_risk'
        ]
        features['composite_risk_score'] = features[risk_indicators].sum(axis=1) / len(risk_indicators)
        
        return features[risk_indicators + ['composite_risk_score']]
    
    def _calculate_trend_features(self, financial_data: pd.DataFrame) -> pd.DataFrame:
        """トレンド特徴量計算"""
        
        trend_features = []
        
        for company_id in financial_data['company_id'].unique():
            company_data = financial_data[
                financial_data['company_id'] == company_id
            ].sort_values('year')
            
            if len(company_data) < 3:
                continue
            
            # 各指標のトレンド計算
            trends = {}
            key_metrics = [
                'revenue', 'operating_profit', 'net_profit',
                'total_assets', 'current_ratio', 'debt_equity_ratio'
            ]
            
            for metric in key_metrics:
                if metric in company_data.columns:
                    values = company_data[metric].values
                    if len(values) >= 3:
                        # 線形回帰による傾き
                        x = np.arange(len(values))
                        slope, _, r_value, _, _ = stats.linregress(x, values)
                        
                        trends[f'{metric}_trend_slope'] = slope
                        trends[f'{metric}_trend_r2'] = r_value ** 2
                        
                        # 改善・悪化判定
                        if metric in ['revenue', 'operating_profit', 'net_profit', 'current_ratio']:
                            trends[f'{metric}_deteriorating'] = 1 if slope < 0 else 0
                        else:  # debt_equity_ratio等、少ない方が良い指標
                            trends[f'{metric}_deteriorating'] = 1 if slope > 0 else 0
            
            # 会社情報を各行に追加
            for _, row in company_data.iterrows():
                trend_row = {'company_id': company_id, 'year': row['year']}
                trend_row.update(trends)
                trend_features.append(trend_row)
        
        return pd.DataFrame(trend_features) if trend_features else pd.DataFrame()
    
    def _calculate_volatility_features(self, financial_data: pd.DataFrame) -> pd.DataFrame:
        """変動性特徴量計算"""
        
        volatility_features = []
        
        for company_id in financial_data['company_id'].unique():
            company_data = financial_data[
                financial_data['company_id'] == company_id
            ].sort_values('year')
            
            if len(company_data) < 3:
                continue
            
            # 収益変動性
            if 'net_profit' in company_data.columns:
                profit_cv = company_data['net_profit'].std() / abs(company_data['net_profit'].mean())
                high_volatility = 1 if profit_cv > 1.0 else 0
            else:
                high_volatility = 0
            
            # 売上変動性
            if 'revenue' in company_data.columns:
                revenue_cv = company_data['revenue'].std() / abs(company_data['revenue'].mean())
                revenue_volatility = 1 if revenue_cv > 0.3 else 0
            else:
                revenue_volatility = 0
            
            # 各行に特徴量追加
            for _, row in company_data.iterrows():
                volatility_features.append({
                    'company_id': company_id,
                    'year': row['year'],
                    'profit_high_volatility': high_volatility,
                    'revenue_high_volatility': revenue_volatility
                })
        
        return pd.DataFrame(volatility_features) if volatility_features else pd.DataFrame()
    
    def _calculate_relative_features(
        self, 
        financial_data: pd.DataFrame,
        company_info: pd.DataFrame
    ) -> pd.DataFrame:
        """相対位置特徴量計算"""
        
        relative_features = []
        
        # 市場カテゴリ別の業界平均計算
        for market_category in company_info['market_category'].unique():
            market_companies = company_info[
                company_info['market_category'] == market_category
            ]['company_id'].tolist()
            
            market_data = financial_data[
                financial_data['company_id'].isin(market_companies)
            ]
            
            # 年別業界統計
            for year in market_data['year'].unique():
                year_market_data = market_data[market_data['year'] == year]
                
                if len(year_market_data) < 2:
                    continue
                
                # 業界四分位点計算
                key_metrics = ['revenue', 'operating_margin', 'roe', 'current_ratio']
                
                for company_id in market_companies:
                    company_year_data = year_market_data[
                        year_market_data['company_id'] == company_id
                    ]
                    
                    if company_year_data.empty:
                        continue
                    
                    relative_row = {
                        'company_id': company_id,
                        'year': year,
                        'market_category': market_category
                    }
                    
                    for metric in key_metrics:
                        if metric in year_market_data.columns:
                            company_value = company_year_data[metric].iloc[0]
                            market_percentile = stats.percentileofscore(
                                year_market_data[metric].dropna(), company_value
                            ) / 100.0
                            
                            relative_row[f'{metric}_market_percentile'] = market_percentile
                            relative_row[f'{metric}_bottom_quartile'] = 1 if market_percentile < 0.25 else 0
                    
                    relative_features.append(relative_row)
        
        return pd.DataFrame(relative_features) if relative_features else pd.DataFrame()
    
    def _calculate_lifecycle_features(
        self,
        financial_data: pd.DataFrame,
        company_info: pd.DataFrame
    ) -> pd.DataFrame:
        """ライフサイクル特徴量計算"""
        
        lifecycle_features = []
        
        for _, company in company_info.iterrows():
            company_id = company['company_id']
            
            # 企業年齢計算
            if 'founding_date' in company and pd.notna(company['founding_date']):
                founding_year = pd.to_datetime(company['founding_date']).year
            else:
                # 財務データの最初の年を代替
                company_financial = financial_data[
                    financial_data['company_id'] == company_id
                ]
                if not company_financial.empty:
                    founding_year = company_financial['year'].min()
                else:
                    continue
            
            # 各年の特徴量計算
            company_financial = financial_data[
                financial_data['company_id'] == company_id
            ]
            
            for _, row in company_financial.iterrows():
                company_age = row['year'] - founding_year
                
                # ライフサイクルステージ判定
                if company_age <= 5:
                    stage = 'startup'
                    stage_risk = 0.3  # スタートアップリスク
                elif company_age <= 15:
                    stage = 'growth'
                    stage_risk = 0.1
                elif company_age <= 30:
                    stage = 'mature'
                    stage_risk = 0.15
                else:
                    stage = 'legacy'
                    stage_risk = 0.25  # レガシー企業リスク
                
                lifecycle_features.append({
                    'company_id': company_id,
                    'year': row['year'],
                    'company_age': company_age,
                    'lifecycle_stage': stage,
                    'stage_risk_factor': stage_risk,
                    'is_startup': 1 if stage == 'startup' else 0,
                    'is_legacy': 1 if stage == 'legacy' else 0
                })
        
        return pd.DataFrame(lifecycle_features) if lifecycle_features else pd.DataFrame()
    
    def _train_and_predict(
        self, 
        features: pd.DataFrame,
        processed_data: Dict
    ) -> Dict[str, np.ndarray]:
        """機械学習モデル学習・予測"""
        
        survival_data = processed_data['survival']
        
        # 特徴量とターゲットの統合
        model_data = features.merge(
            survival_data[['company_id', 'year', 'is_extinct', 'survival_time', 'event_observed']],
            on=['company_id', 'year'],
            how='inner'
        )
        
        # 特徴量準備
        feature_cols = [col for col in model_data.columns 
                        if col not in ['company_id', 'year', 'is_extinct', 'survival_time', 'event_observed']]
        
        X = model_data[feature_cols].fillna(0)
        y_classification = model_data['is_extinct']
        
        # 特徴量正規化
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=feature_cols, index=X.index)
        
        predictions = {}
        
        # 1. ロジスティック回帰
        lr_model = LogisticRegression(random_state=42, max_iter=1000)
        lr_model.fit(X_scaled, y_classification)
        self.models['logistic'] = lr_model
        predictions['logistic'] = lr_model.predict_proba(X_scaled)[:, 1]
        
        # 2. ランダムフォレスト
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_scaled, y_classification)
        self.models['random_forest'] = rf_model
        predictions['random_forest'] = rf_model.predict_proba(X_scaled)[:, 1]
        
        # 特徴量重要度保存
        self.feature_importance['random_forest'] = dict(
            zip(feature_cols, rf_model.feature_importances_)
        )
        
        # 3. 勾配ブースティング
        gb_model = GradientBoostingClassifier(random_state=42)
        gb_model.fit(X_scaled, y_classification)
        self.models['gradient_boosting'] = gb_model
        predictions['gradient_boosting'] = gb_model.predict_proba(X_scaled)[:, 1]
        
        # 4. Cox回帰（生存分析）
        try:
            cox_data = model_data[['survival_time', 'event_observed'] + feature_cols].dropna()
            if len(cox_data) > 10:
                cox_model = CoxPHFitter()
                cox_model.fit(cox_data, duration_col='survival_time', event_col='event_observed')
                self.models['cox'] = cox_model
                
                # ハザード予測
                hazard_scores = cox_model.predict_partial_hazard(cox_data[feature_cols])
                # ハザードを確率に変換（正規化）
                cox_probs = hazard_scores / hazard_scores.max()
                predictions['cox'] = cox_probs.values
            else:
                predictions['cox'] = np.zeros(len(X_scaled))
        except Exception as e:
            logger.warning(f"Cox回帰でエラー: {e}")
            predictions['cox'] = np.zeros(len(X_scaled))
        
        # アンサンブル予測
        weights = self.config['model_weights']
        ensemble_pred = (
            weights['logistic_regression'] * predictions['logistic'] +
            weights['random_forest'] * predictions['random_forest'] +
            weights['gradient_boosting'] * predictions['gradient_boosting'] +
            weights['cox_regression'] * predictions['cox']
        )
        predictions['ensemble'] = ensemble_pred
        
        # モデルデータのインデックスを追加
        for key in predictions:
            predictions[key] = pd.Series(predictions[key], index=model_data.index)
        
        # 企業・年情報を予測結果に追加
        predictions['company_year_info'] = model_data[['company_id', 'year']]
        
        return predictions
    
    def _calculate_risk_metrics(
        self,
        predictions: Dict[str, np.ndarray],
        processed_data: Dict,
        company_info: pd.DataFrame
    ) -> Dict[str, ExtinctionRiskMetrics]:
        """リスクメトリクス計算"""
        
        risk_metrics = {}
        company_year_info = predictions['company_year_info']
        
        # 最新年度のデータのみ使用してリスク評価
        latest_year_data = company_year_info.loc[
            company_year_info.groupby('company_id')['year'].idxmax()
        ]
        
        for idx, row in latest_year_data.iterrows():
            company_id = row['company_id']
            year = row['year']
            
            # 企業情報取得
            company_row = company_info[
                company_info['company_id'] == company_id
            ]
            if company_row.empty:
                continue
            
            company_name = company_row.iloc[0].get('company_name', f'Company_{company_id}')
            market_category = company_row.iloc[0].get('market_category', 'unknown')
            
            # リスクスコア（アンサンブル予測）
            risk_score = float(predictions['ensemble'].loc[idx])
            
            # リスクレベル判定
            risk_level = self._determine_risk_level(risk_score)
            
            # 生存確率計算（Cox回帰がある場合）
            survival_probs = self._calculate_survival_probabilities(
                company_id, risk_score, processed_data
            )
            
            # 重要要因特定
            critical_factors = self._identify_critical_factors(
                company_id, year, processed_data
            )
            
            # 警告指標特定
            warning_indicators = self._identify_warning_indicators(
                company_id, year, processed_data
            )
            
            # 消滅時期推定
            time_to_extinction = self._estimate_time_to_extinction(
                risk_score, survival_probs
            )
            
            # 信頼区間計算
            confidence_interval = self._calculate_confidence_interval(
                predictions, idx
            )
            
            # メトリクス作成
            risk_metrics[company_id] = ExtinctionRiskMetrics(
                company_id=company_id,
                company_name=company_name,
                analysis_date=datetime.now(),
                risk_score=risk_score,
                risk_level=risk_level,
                survival_probability_1y=survival_probs['1y'],
                survival_probability_3y=survival_probs['3y'],
                survival_probability_5y=survival_probs['5y'],
                hazard_ratio=self._calculate_hazard_ratio(company_id, processed_data),
                critical_factors=critical_factors,
                warning_indicators=warning_indicators,
                time_to_extinction_estimate=time_to_extinction,
                confidence_interval=confidence_interval,
                market_category=market_category
            )
        
        return risk_metrics
    
    def _determine_risk_level(self, risk_score: float) -> RiskLevel:
        """リスクレベル判定"""
        for level, (min_score, max_score) in self.risk_thresholds.items():
            if min_score <= risk_score < max_score:
                return level
        return RiskLevel.CRITICAL
    
    def _calculate_survival_probabilities(
        self,
        company_id: str,
        risk_score: float,
        processed_data: Dict
    ) -> Dict[str, float]:
        """生存確率計算"""
        
        # 簡易生存確率モデル（実際の実装ではより精密なモデルを使用）
        base_survival_1y = 1 - risk_score * 0.3
        base_survival_3y = base_survival_1y ** 3
        base_survival_5y = base_survival_1y ** 5
        
        return {
            '1y': max(0.0, min(1.0, base_survival_1y)),
            '3y': max(0.0, min(1.0, base_survival_3y)),
            '5y': max(0.0, min(1.0, base_survival_5y))
        }
    
    def _identify_critical_factors(
        self,
        company_id: str,
        year: int,
        processed_data: Dict
    ) -> List[str]:
        """重要リスク要因特定"""
        
        critical_factors = []
        
        # ランダムフォレストの特徴量重要度から上位要因を特定
        if 'random_forest' in self.feature_importance:
            importance_dict = self.feature_importance['random_forest']
            # 重要度上位5つ
            top_factors = sorted(
                importance_dict.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
            
            critical_factors = [factor[0] for factor in top_factors]
        
        return critical_factors
    
    def _identify_warning_indicators(
        self,
        company_id: str,
        year: int,
        processed_data: Dict
    ) -> List[str]:
        """警告指標特定"""
        
        warning_indicators = []
        financial_data = processed_data['financial']
        
        # 該当企業の最新データ取得
        company_data = financial_data[
            (financial_data['company_id'] == company_id) & 
            (financial_data['year'] == year)
        ]
        
        if company_data.empty:
            return warning_indicators
        
        row = company_data.iloc[0]
        
        # 閾値ベースの警告指標チェック
        critical_thresholds = self.config['critical_thresholds']
        
        if 'debt_equity_ratio' in row and row['debt_equity_ratio'] > critical_thresholds['debt_equity_ratio']:
            warning_indicators.append("高レバレッジリスク（負債比率異常値）")
        
        if 'current_ratio' in row and row['current_ratio'] < critical_thresholds['current_ratio']:
            warning_indicators.append("流動性リスク（流動比率低下）")
        
        if 'interest_coverage_ratio' in row and row['interest_coverage_ratio'] < critical_thresholds['interest_coverage_ratio']:
            warning_indicators.append("利払い能力不足")
        
        if 'operating_margin' in row and row['operating_margin'] < critical_thresholds['operating_margin']:
            warning_indicators.append("営業赤字継続")
        
        if 'cash_ratio' in row and row['cash_ratio'] < critical_thresholds['cash_ratio']:
            warning_indicators.append("現金不足リスク")
        
        return warning_indicators
    
    def _estimate_time_to_extinction(
        self,
        risk_score: float,
        survival_probs: Dict[str, float]
    ) -> Optional[float]:
        """消滅時期推定"""
        
        if risk_score < 0.3:
            return None  # 低リスクの場合は推定しない
        
        # 生存確率から推定消滅時期を計算
        # 生存確率が50%を下回る時期を推定
        if survival_probs['1y'] < 0.5:
            return 1.0
        elif survival_probs['3y'] < 0.5:
            return 3.0
        elif survival_probs['5y'] < 0.5:
            return 5.0
        else:
            # 線形近似で推定
            decay_rate = -np.log(survival_probs['5y']) / 5
            time_to_50pct = -np.log(0.5) / decay_rate if decay_rate > 0 else None
            return time_to_50pct
    
    def _calculate_confidence_interval(
        self,
        predictions: Dict[str, np.ndarray],
        idx: int
    ) -> Tuple[float, float]:
        """信頼区間計算"""
        
        # 複数モデルの予測値から信頼区間を計算
        model_predictions = []
        for model_name in ['logistic', 'random_forest', 'gradient_boosting']:
            if model_name in predictions:
                model_predictions.append(predictions[model_name].loc[idx])
        
        if len(model_predictions) < 2:
            return (0.0, 1.0)
        
        predictions_array = np.array(model_predictions)
        mean_pred = np.mean(predictions_array)
        std_pred = np.std(predictions_array)
        
        # 95%信頼区間
        lower_bound = max(0.0, mean_pred - 1.96 * std_pred)
        upper_bound = min(1.0, mean_pred + 1.96 * std_pred)
        
        return (lower_bound, upper_bound)
    
    def _calculate_hazard_ratio(
        self,
        company_id: str,
        processed_data: Dict
    ) -> float:
        """ハザード比計算"""
        
        if 'cox' not in self.models:
            return 1.0
        
        try:
            # Cox回帰モデルからハザード比を計算
            cox_model = self.models['cox']
            
            # 企業の特徴量取得
            financial_data = processed_data['financial']
            company_data = financial_data[
                financial_data['company_id'] == company_id
            ].iloc[-1:]  # 最新データ
            
            if company_data.empty:
                return 1.0
            
            # 特徴量カラムのみ抽出
            feature_cols = [col for col in company_data.columns 
                            if col not in ['company_id', 'year']]
            company_features = company_data[feature_cols].fillna(0)
            
            # ハザード比計算
            hazard_ratio = cox_model.predict_partial_hazard(company_features).iloc[0]
            return float(hazard_ratio)
            
        except Exception as e:
            logger.warning(f"ハザード比計算でエラー: {e}")
            return 1.0
    
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """欠損値処理"""
        
        # 財務指標ごとの欠損値処理戦略
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        # 業界中央値で補完
        for col in numeric_cols:
            if col not in ['company_id', 'year']:
                data[col] = data[col].fillna(data[col].median())
        
        return data
    
    def _handle_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """異常値処理"""
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col not in ['company_id', 'year']:
                # IQR法による異常値検出・処理
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # 異常値をキャップ
                data[col] = np.clip(data[col], lower_bound, upper_bound)
        
        return data
    
    def _industry_normalize(
        self,
        financial_data: pd.DataFrame,
        company_info: pd.DataFrame
    ) -> pd.DataFrame:
        """業界標準化"""
        
        normalized_data = financial_data.copy()
        
        # 市場カテゴリ別に標準化
        for market_category in company_info['market_category'].unique():
            market_companies = company_info[
                company_info['market_category'] == market_category
            ]['company_id'].tolist()
            
            market_mask = normalized_data['company_id'].isin(market_companies)
            market_data = normalized_data[market_mask]
            
            # 数値カラムのみ標準化
            numeric_cols = market_data.select_dtypes(include=[np.number]).columns
            standardizable_cols = [col for col in numeric_cols 
                                    if col not in ['company_id', 'year']]
            
            if len(market_data) > 1:  # 標準化には複数サンプルが必要
                for col in standardizable_cols:
                    market_mean = market_data[col].mean()
                    market_std = market_data[col].std()
                    
                    if market_std > 0:
                        normalized_data.loc[market_mask, f'{col}_zscore'] = (
                            market_data[col] - market_mean
                        ) / market_std
        
        return normalized_data
    
    def generate_risk_report(
        self,
        risk_metrics: Dict[str, ExtinctionRiskMetrics],
        output_path: Optional[str] = None
    ) -> pd.DataFrame:
        """リスク分析レポート生成"""
        
        report_data = []
        
        for company_id, metrics in risk_metrics.items():
            report_row = {
                'company_id': metrics.company_id,
                'company_name': metrics.company_name,
                'market_category': metrics.market_category,
                'risk_score': round(metrics.risk_score, 4),
                'risk_level': metrics.risk_level.value,
                'survival_1y': round(metrics.survival_probability_1y, 4),
                'survival_3y': round(metrics.survival_probability_3y, 4),
                'survival_5y': round(metrics.survival_probability_5y, 4),
                'hazard_ratio': round(metrics.hazard_ratio, 4),
                'time_to_extinction': metrics.time_to_extinction_estimate,
                'confidence_lower': round(metrics.confidence_interval[0], 4),
                'confidence_upper': round(metrics.confidence_interval[1], 4),
                'critical_factors': '; '.join(metrics.critical_factors),
                'warning_indicators': '; '.join(metrics.warning_indicators),
                'analysis_date': metrics.analysis_date.strftime('%Y-%m-%d')
            }
            report_data.append(report_row)
        
        report_df = pd.DataFrame(report_data)
        
        # リスクスコア順でソート
        report_df = report_df.sort_values('risk_score', ascending=False)
        
        if output_path:
            report_df.to_csv(output_path, index=False, encoding='utf-8')
            logger.info(f"リスクレポートを保存: {output_path}")
        
        return report_df
    
    def get_market_category_risk_summary(
        self,
        risk_metrics: Dict[str, ExtinctionRiskMetrics]
    ) -> pd.DataFrame:
        """市場カテゴリ別リスクサマリー"""
        
        summary_data = []
        
        # 市場カテゴリ別にグループ化
        category_groups = {}
        for metrics in risk_metrics.values():
            category = metrics.market_category
            if category not in category_groups:
                category_groups[category] = []
            category_groups[category].append(metrics)
        
        # カテゴリ別統計計算
        for category, metrics_list in category_groups.items():
            risk_scores = [m.risk_score for m in metrics_list]
            survival_1y = [m.survival_probability_1y for m in metrics_list]
            
            high_risk_count = sum(1 for m in metrics_list 
                                if m.risk_level in [RiskLevel.HIGH, RiskLevel.VERY_HIGH, RiskLevel.CRITICAL])
            
            summary_data.append({
                'market_category': category,
                'company_count': len(metrics_list),
                'avg_risk_score': round(np.mean(risk_scores), 4),
                'median_risk_score': round(np.median(risk_scores), 4),
                'high_risk_count': high_risk_count,
                'high_risk_ratio': round(high_risk_count / len(metrics_list), 4),
                'avg_survival_1y': round(np.mean(survival_1y), 4),
                'min_survival_1y': round(np.min(survival_1y), 4),
                'companies_at_risk': '; '.join([
                    m.company_name for m in metrics_list 
                    if m.risk_level in [RiskLevel.VERY_HIGH, RiskLevel.CRITICAL]
                ])
            })
        
        return pd.DataFrame(summary_data).sort_values('avg_risk_score', ascending=False)
    
    def predict_future_extinctions(
        self,
        risk_metrics: Dict[str, ExtinctionRiskMetrics],
        time_horizon: int = 5
    ) -> Dict[str, List[str]]:
        """将来の企業消滅予測"""
        
        predictions = {}
        
        for year in range(1, time_horizon + 1):
            year_predictions = []
            
            for metrics in risk_metrics.values():
                # 各年の消滅確率計算
                if year == 1:
                    extinction_prob = 1 - metrics.survival_probability_1y
                elif year == 3:
                    extinction_prob = 1 - metrics.survival_probability_3y
                elif year == 5:
                    extinction_prob = 1 - metrics.survival_probability_5y
                else:
                    # 指数減衰モデルで近似
                    decay_rate = -np.log(metrics.survival_probability_1y)
                    extinction_prob = 1 - np.exp(-decay_rate * year)
                
                if extinction_prob > 0.5:  # 50%以上の確率で消滅
                    year_predictions.append({
                        'company_name': metrics.company_name,
                        'extinction_probability': extinction_prob,
                        'risk_level': metrics.risk_level.value
                    })
            
            # 確率順でソート
            year_predictions.sort(key=lambda x: x['extinction_probability'], reverse=True)
            predictions[f'year_{year}'] = year_predictions
        
        return predictions
    
    def export_analysis_results(
        self,
        risk_metrics: Dict[str, ExtinctionRiskMetrics],
        output_dir: str
    ) -> Dict[str, str]:
        """分析結果エクスポート"""
        
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        exported_files = {}
        
        # 1. 詳細リスクレポート
        detailed_report = self.generate_risk_report(risk_metrics)
        detailed_path = os.path.join(output_dir, 'detailed_extinction_risk_report.csv')
        detailed_report.to_csv(detailed_path, index=False, encoding='utf-8')
        exported_files['detailed_report'] = detailed_path
        
        # 2. 市場カテゴリサマリー
        category_summary = self.get_market_category_risk_summary(risk_metrics)
        summary_path = os.path.join(output_dir, 'market_category_risk_summary.csv')
        category_summary.to_csv(summary_path, index=False, encoding='utf-8')
        exported_files['category_summary'] = summary_path
        
        # 3. 将来予測
        future_predictions = self.predict_future_extinctions(risk_metrics)
        for year_key, predictions in future_predictions.items():
            pred_df = pd.DataFrame(predictions)
            pred_path = os.path.join(output_dir, f'extinction_predictions_{year_key}.csv')
            pred_df.to_csv(pred_path, index=False, encoding='utf-8')
            exported_files[f'predictions_{year_key}'] = pred_path
        
        # 4. モデル性能レポート
        if hasattr(self, 'models') and self.models:
            performance_data = self._generate_model_performance_report()
            perf_path = os.path.join(output_dir, 'model_performance_report.csv')
            performance_data.to_csv(perf_path, index=False, encoding='utf-8')
            exported_files['model_performance'] = perf_path
        
        logger.info(f"分析結果を{len(exported_files)}個のファイルにエクスポートしました: {output_dir}")
        return exported_files
    
    def _generate_model_performance_report(self) -> pd.DataFrame:
        """モデル性能レポート生成"""
        
        performance_data = []
        
        for model_name, model in self.models.items():
            if hasattr(model, 'score'):
                # 基本的な性能指標
                performance_row = {
                    'model_name': model_name,
                    'model_type': type(model).__name__,
                    'is_fitted': True
                }
                
                # 特徴量重要度情報（ある場合）
                if model_name in self.feature_importance:
                    importance_dict = self.feature_importance[model_name]
                    top_feature = max(importance_dict.items(), key=lambda x: x[1])
                    performance_row['top_feature'] = top_feature[0]
                    performance_row['top_feature_importance'] = top_feature[1]
                
                performance_data.append(performance_row)
        
        return pd.DataFrame(performance_data)


def main():
    """メイン実行関数（テスト用）"""
    
    # サンプルデータ生成
    np.random.seed(42)
    
    # 財務データサンプル
    companies = ['COMP_001', 'COMP_002', 'COMP_003', 'COMP_004', 'COMP_005']
    years = list(range(2019, 2024))
    
    financial_data = []
    for company in companies:
        for year in years:
            financial_data.append({
                'company_id': company,
                'year': year,
                'revenue': np.random.uniform(1000, 10000),
                'operating_profit': np.random.uniform(-500, 2000),
                'net_profit': np.random.uniform(-300, 1500),
                'total_assets': np.random.uniform(5000, 50000),
                'current_ratio': np.random.uniform(0.5, 3.0),
                'debt_equity_ratio': np.random.uniform(0.1, 5.0),
                'operating_margin': np.random.uniform(-0.1, 0.3),
                'roe': np.random.uniform(-0.2, 0.4),
                'interest_coverage_ratio': np.random.uniform(0.1, 10.0),
                'cash_ratio': np.random.uniform(0.01, 0.5)
            })
    
    financial_df = pd.DataFrame(financial_data)
    
    # 企業情報サンプル
    company_info = pd.DataFrame({
        'company_id': companies,
        'company_name': [f'Company_{i+1}' for i in range(len(companies))],
        'market_category': ['high_share', 'declining', 'lost', 'high_share', 'declining'],
        'founding_date': ['1990-01-01', '1985-05-15', '1995-12-01', '2000-03-01', '1980-08-01']
    })
    
    # 分析実行
    analyzer = ExtinctionRiskAnalyzer()
    
    try:
        risk_metrics = analyzer.analyze_extinction_risk(
            financial_data=financial_df,
            company_info=company_info
        )
        
        # 結果表示
        print("\n=== 企業消滅リスク分析結果 ===")
        for company_id, metrics in risk_metrics.items():
            print(f"\n企業: {metrics.company_name}")
            print(f"リスクスコア: {metrics.risk_score:.4f}")
            print(f"リスクレベル: {metrics.risk_level.value}")
            print(f"1年生存確率: {metrics.survival_probability_1y:.4f}")
            print(f"重要要因: {', '.join(metrics.critical_factors[:3])}")
            print(f"警告指標: {', '.join(metrics.warning_indicators)}")
        
        # レポート生成
        report = analyzer.generate_risk_report(risk_metrics)
        print(f"\n詳細レポート生成完了: {len(report)}社")
        
        # 市場カテゴリサマリー
        category_summary = analyzer.get_market_category_risk_summary(risk_metrics)
        print(f"\n市場カテゴリ分析完了: {len(category_summary)}カテゴリ")
        
    except Exception as e:
        logger.error(f"分析実行中にエラー: {e}")
        raise


if __name__ == "__main__":
    main()