"""
A2AI - Advanced Financial Analysis AI
企業ライフサイクル特徴量生成モジュール

このモジュールは企業のライフサイクル段階を特定し、各段階に応じた特徴量を生成します。
- 企業年齢と成長段階の特定
- ライフサイクル段階別の財務指標正規化
- 段階遷移確率の計算
- 市場成熟度との相関分析
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
import warnings
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from scipy import stats
from scipy.signal import find_peaks
import logging

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LifecycleFeatureGenerator:
    """
    企業ライフサイクル特徴量生成クラス
    
    企業の設立から現在（または消滅）までのライフサイクル段階を特定し、
    各段階に応じた特徴量を生成する。
    """
    
    def __init__(self, 
                    lifecycle_stages: Dict[str, Dict] = None,
                    min_years_per_stage: int = 3):
        """
        初期化
        
        Args:
            lifecycle_stages: ライフサイクル段階定義
            min_years_per_stage: 各段階の最小年数
        """
        self.min_years_per_stage = min_years_per_stage
        self.scaler = StandardScaler()
        self.minmax_scaler = MinMaxScaler()
        
        # デフォルトライフサイクル段階定義
        self.lifecycle_stages = lifecycle_stages or {
            'startup': {
                'age_range': (0, 7),
                'characteristics': ['high_growth', 'high_volatility', 'low_profitability'],
                'key_metrics': ['revenue_growth', 'r_and_d_ratio', 'cash_burn_rate']
            },
            'growth': {
                'age_range': (5, 20),
                'characteristics': ['rapid_expansion', 'market_penetration', 'scaling'],
                'key_metrics': ['market_share_growth', 'employee_growth', 'capex_ratio']
            },
            'maturity': {
                'age_range': (15, 40),
                'characteristics': ['stable_revenue', 'high_profitability', 'dividend_payment'],
                'key_metrics': ['roe', 'dividend_yield', 'operating_margin']
            },
            'decline': {
                'age_range': (25, float('inf')),
                'characteristics': ['shrinking_market', 'cost_cutting', 'restructuring'],
                'key_metrics': ['revenue_decline', 'asset_reduction', 'debt_ratio']
            },
            'transformation': {
                'age_range': (10, float('inf')),
                'characteristics': ['business_model_change', 'new_market_entry', 'innovation'],
                'key_metrics': ['r_and_d_spike', 'segment_diversification', 'acquisition_activity']
            }
        }
        
    def calculate_company_age(self, 
                            establishment_date: Union[str, datetime],
                            reference_date: Union[str, datetime] = None) -> float:
        """
        企業年齢を計算
        
        Args:
            establishment_date: 設立日
            reference_date: 基準日（デフォルトは現在日時）
            
        Returns:
            企業年齢（年）
        """
        if isinstance(establishment_date, str):
            establishment_date = pd.to_datetime(establishment_date)
        
        if reference_date is None:
            reference_date = datetime.now()
        elif isinstance(reference_date, str):
            reference_date = pd.to_datetime(reference_date)
            
        age_years = (reference_date - establishment_date).days / 365.25
        return max(0, age_years)
    
    def identify_lifecycle_stage(self, 
                                financial_data: pd.DataFrame,
                                company_age: float,
                                market_category: str = None) -> Dict[str, float]:
        """
        ライフサイクル段階を特定
        
        Args:
            financial_data: 財務データ（時系列）
            company_age: 企業年齢
            market_category: 市場カテゴリ（'high_share', 'declining', 'lost'）
            
        Returns:
            各段階への所属確率
        """
        try:
            stage_probabilities = {}
            
            # 1. 年齢ベースの基本確率
            for stage, config in self.lifecycle_stages.items():
                age_min, age_max = config['age_range']
                
                if age_min <= company_age <= age_max:
                    # 年齢範囲内での確率計算（正規分布を仮定）
                    mid_age = (age_min + age_max) / 2
                    age_std = (age_max - age_min) / 4  # 4σで範囲をカバー
                    
                    age_prob = stats.norm.pdf(company_age, mid_age, age_std)
                    stage_probabilities[stage] = age_prob
                else:
                    stage_probabilities[stage] = 0.0
            
            # 2. 財務指標ベースの調整
            if not financial_data.empty:
                stage_probabilities = self._adjust_probabilities_by_financials(
                    stage_probabilities, financial_data, market_category
                )
            
            # 3. 確率の正規化
            total_prob = sum(stage_probabilities.values())
            if total_prob > 0:
                stage_probabilities = {
                    stage: prob / total_prob 
                    for stage, prob in stage_probabilities.items()
                }
            
            return stage_probabilities
            
        except Exception as e:
            logger.error(f"ライフサイクル段階特定エラー: {e}")
            # エラー時はデフォルト値を返す
            return {stage: 1/len(self.lifecycle_stages) for stage in self.lifecycle_stages.keys()}
    
    def _adjust_probabilities_by_financials(self, 
                                            base_probabilities: Dict[str, float],
                                            financial_data: pd.DataFrame,
                                            market_category: str) -> Dict[str, float]:
        """
        財務指標による確率調整
        
        Args:
            base_probabilities: 基本確率
            financial_data: 財務データ
            market_category: 市場カテゴリ
            
        Returns:
            調整後確率
        """
        adjusted_probs = base_probabilities.copy()
        
        try:
            # 最新3年間のデータを使用
            recent_data = financial_data.tail(3)
            if recent_data.empty:
                return base_probabilities
            
            # 成長率指標
            if 'revenue_growth_rate' in recent_data.columns:
                avg_growth = recent_data['revenue_growth_rate'].mean()
                
                # 高成長 → startup/growth段階の確率向上
                if avg_growth > 0.15:  # 15%以上の成長
                    adjusted_probs['startup'] *= 1.5
                    adjusted_probs['growth'] *= 1.3
                    adjusted_probs['decline'] *= 0.5
                
                # 負成長 → decline段階の確率向上
                elif avg_growth < -0.05:  # -5%以下の成長
                    adjusted_probs['decline'] *= 2.0
                    adjusted_probs['startup'] *= 0.3
                    adjusted_probs['growth'] *= 0.5
            
            # 収益性指標
            if 'roa' in recent_data.columns:
                avg_roa = recent_data['roa'].mean()
                
                # 高収益性 → maturity段階の確率向上
                if avg_roa > 0.10:  # 10%以上のROA
                    adjusted_probs['maturity'] *= 1.4
                
                # 低収益性 → startup/decline段階の確率向上
                elif avg_roa < 0.02:  # 2%以下のROA
                    adjusted_probs['startup'] *= 1.2
                    adjusted_probs['decline'] *= 1.3
            
            # R&D投資比率
            if 'r_and_d_ratio' in recent_data.columns:
                avg_rd_ratio = recent_data['r_and_d_ratio'].mean()
                
                # 高R&D → startup/transformation段階の確率向上
                if avg_rd_ratio > 0.08:  # 8%以上のR&D比率
                    adjusted_probs['startup'] *= 1.3
                    adjusted_probs['transformation'] *= 1.4
            
            # 市場カテゴリによる調整
            if market_category == 'lost':
                # 失失市場 → decline段階の確率大幅向上
                adjusted_probs['decline'] *= 2.5
                adjusted_probs['transformation'] *= 1.5  # 事業転換の可能性
                adjusted_probs['growth'] *= 0.3
                adjusted_probs['maturity'] *= 0.5
                
            elif market_category == 'declining':
                # 低下市場 → transformation段階の確率向上
                adjusted_probs['transformation'] *= 1.8
                adjusted_probs['decline'] *= 1.3
                adjusted_probs['growth'] *= 0.7
                
            elif market_category == 'high_share':
                # 高シェア市場 → maturity段階の確率向上
                adjusted_probs['maturity'] *= 1.5
                adjusted_probs['growth'] *= 1.2
                
        except Exception as e:
            logger.warning(f"財務指標による確率調整でエラー: {e}")
            return base_probabilities
        
        return adjusted_probs
    
    def generate_stage_transition_features(self, 
                                            historical_stages: pd.DataFrame) -> Dict[str, float]:
        """
        段階遷移特徴量を生成
        
        Args:
            historical_stages: 過去のライフサイクル段階データ
            
        Returns:
            遷移特徴量
        """
        features = {}
        
        try:
            if historical_stages.empty:
                return features
            
            # 最も確率の高い段階を時系列で取得
            dominant_stages = []
            for _, row in historical_stages.iterrows():
                max_stage = max(row.to_dict().items(), key=lambda x: x[1])[0]
                dominant_stages.append(max_stage)
            
            stages_series = pd.Series(dominant_stages)
            
            # 段階変化回数
            stage_changes = (stages_series != stages_series.shift()).sum() - 1
            features['stage_transition_count'] = stage_changes
            
            # 段階安定性（同一段階継続期間の平均）
            stage_durations = []
            current_stage = stages_series.iloc[0] if len(stages_series) > 0 else None
            current_duration = 1
            
            for stage in stages_series.iloc[1:]:
                if stage == current_stage:
                    current_duration += 1
                else:
                    stage_durations.append(current_duration)
                    current_stage = stage
                    current_duration = 1
            
            if current_duration > 0:
                stage_durations.append(current_duration)
            
            features['avg_stage_duration'] = np.mean(stage_durations) if stage_durations else 0
            features['stage_stability'] = max(stage_durations) if stage_durations else 0
            
            # 現在の段階継続期間
            if len(stages_series) > 0:
                current_stage = stages_series.iloc[-1]
                current_duration = 1
                for i in range(len(stages_series) - 2, -1, -1):
                    if stages_series.iloc[i] == current_stage:
                        current_duration += 1
                    else:
                        break
                features['current_stage_duration'] = current_duration
            
            # 各段階での滞在時間比率
            stage_counts = stages_series.value_counts()
            total_periods = len(stages_series)
            
            for stage in self.lifecycle_stages.keys():
                stage_ratio = stage_counts.get(stage, 0) / total_periods
                features[f'{stage}_time_ratio'] = stage_ratio
            
            # 方向性指標（進歩的 vs 後退的遷移）
            stage_order = {'startup': 1, 'growth': 2, 'maturity': 3, 'transformation': 2.5, 'decline': 0}
            
            progression_score = 0
            for i in range(1, len(stages_series)):
                prev_score = stage_order.get(stages_series.iloc[i-1], 1)
                curr_score = stage_order.get(stages_series.iloc[i], 1)
                progression_score += curr_score - prev_score
            
            features['lifecycle_progression'] = progression_score / max(len(stages_series) - 1, 1)
            
        except Exception as e:
            logger.error(f"段階遷移特徴量生成エラー: {e}")
        
        return features
    
    def calculate_lifecycle_momentum(self, 
                                    financial_data: pd.DataFrame,
                                    window: int = 5) -> Dict[str, float]:
        """
        ライフサイクル・モメンタムを計算
        
        Args:
            financial_data: 財務データ
            window: 計算ウィンドウ（年数）
            
        Returns:
            モメンタム指標
        """
        momentum_features = {}
        
        try:
            if len(financial_data) < window:
                window = len(financial_data)
            
            recent_data = financial_data.tail(window)
            
            # 成長モメンタム
            if 'revenue_growth_rate' in recent_data.columns:
                growth_trend = self._calculate_trend(recent_data['revenue_growth_rate'])
                momentum_features['growth_momentum'] = growth_trend
            
            # 収益性モメンタム
            if 'operating_margin' in recent_data.columns:
                profitability_trend = self._calculate_trend(recent_data['operating_margin'])
                momentum_features['profitability_momentum'] = profitability_trend
            
            # 効率性モメンタム
            if 'total_asset_turnover' in recent_data.columns:
                efficiency_trend = self._calculate_trend(recent_data['total_asset_turnover'])
                momentum_features['efficiency_momentum'] = efficiency_trend
            
            # 投資モメンタム
            if 'capex_ratio' in recent_data.columns:
                investment_trend = self._calculate_trend(recent_data['capex_ratio'])
                momentum_features['investment_momentum'] = investment_trend
            
            # R&Dモメンタム
            if 'r_and_d_ratio' in recent_data.columns:
                rd_trend = self._calculate_trend(recent_data['r_and_d_ratio'])
                momentum_features['innovation_momentum'] = rd_trend
            
            # 総合モメンタム（主要指標の平均）
            key_momentums = ['growth_momentum', 'profitability_momentum', 'efficiency_momentum']
            available_momentums = [momentum_features.get(key, 0) for key in key_momentums 
                                    if key in momentum_features]
            
            if available_momentums:
                momentum_features['overall_momentum'] = np.mean(available_momentums)
            
        except Exception as e:
            logger.error(f"ライフサイクル・モメンタム計算エラー: {e}")
        
        return momentum_features
    
    def _calculate_trend(self, data_series: pd.Series) -> float:
        """
        データ系列のトレンドを計算
        
        Args:
            data_series: データ系列
            
        Returns:
            トレンド係数（正：上昇、負：下降）
        """
        try:
            if len(data_series) < 2:
                return 0.0
            
            # 欠損値を除去
            clean_data = data_series.dropna()
            if len(clean_data) < 2:
                return 0.0
            
            # 線形回帰でトレンドを計算
            x = np.arange(len(clean_data))
            y = clean_data.values
            
            slope, _, r_value, _, _ = stats.linregress(x, y)
            
            # R²で重み付けしたトレンド
            weighted_trend = slope * (r_value ** 2)
            
            return weighted_trend
            
        except Exception as e:
            logger.warning(f"トレンド計算エラー: {e}")
            return 0.0
    
    def generate_maturity_indicators(self, 
                                    financial_data: pd.DataFrame,
                                    industry_benchmarks: pd.DataFrame = None) -> Dict[str, float]:
        """
        企業成熟度指標を生成
        
        Args:
            financial_data: 財務データ
            industry_benchmarks: 業界ベンチマーク
            
        Returns:
            成熟度指標
        """
        maturity_indicators = {}
        
        try:
            if financial_data.empty:
                return maturity_indicators
            
            recent_data = financial_data.tail(3).mean()  # 最新3年平均
            
            # 1. 収益安定性
            if 'revenue' in financial_data.columns and len(financial_data) >= 5:
                revenue_cv = financial_data['revenue'].tail(5).std() / financial_data['revenue'].tail(5).mean()
                maturity_indicators['revenue_stability'] = max(0, 1 - revenue_cv)  # CV値の逆数
            
            # 2. 配当支払能力・意欲
            if 'dividend_payout_ratio' in recent_data.index:
                maturity_indicators['dividend_maturity'] = min(recent_data['dividend_payout_ratio'], 1.0)
            
            # 3. 市場地位の安定性
            if 'market_share' in recent_data.index:
                maturity_indicators['market_position_strength'] = recent_data['market_share']
            
            # 4. 資本構造の安定性
            if 'equity_ratio' in recent_data.index:
                maturity_indicators['capital_structure_stability'] = recent_data['equity_ratio']
            
            # 5. 事業多角化度
            if 'segment_count' in recent_data.index:
                # 適度な多角化が成熟を示す
                segment_count = recent_data['segment_count']
                optimal_segments = 3  # 最適セグメント数
                diversification_score = 1 - abs(segment_count - optimal_segments) / optimal_segments
                maturity_indicators['business_diversification'] = max(0, diversification_score)
            
            # 6. 研究開発の効率性（成熟企業は効率的なR&D）
            if 'r_and_d_ratio' in recent_data.index and 'revenue_growth_rate' in recent_data.index:
                rd_efficiency = recent_data['revenue_growth_rate'] / max(recent_data['r_and_d_ratio'], 0.001)
                maturity_indicators['rd_efficiency'] = min(rd_efficiency, 10)  # 上限設定
            
            # 7. 競争優位の持続性
            if 'operating_margin' in financial_data.columns and len(financial_data) >= 5:
                margin_trend = self._calculate_trend(financial_data['operating_margin'].tail(5))
                maturity_indicators['competitive_advantage_sustainability'] = max(0, margin_trend + 0.5)
            
            # 8. 総合成熟度スコア
            available_indicators = [v for v in maturity_indicators.values() if not np.isnan(v)]
            if available_indicators:
                maturity_indicators['overall_maturity'] = np.mean(available_indicators)
            
        except Exception as e:
            logger.error(f"成熟度指標生成エラー: {e}")
        
        return maturity_indicators
    
    def detect_lifecycle_anomalies(self, 
                                    lifecycle_data: pd.DataFrame,
                                    financial_data: pd.DataFrame) -> Dict[str, Union[bool, float]]:
        """
        ライフサイクル異常を検出
        
        Args:
            lifecycle_data: ライフサイクルデータ
            financial_data: 財務データ
            
        Returns:
            異常検出結果
        """
        anomalies = {}
        
        try:
            # 1. 急激な段階変化の検出
            if not lifecycle_data.empty:
                # 主要段階の変化を検出
                dominant_stages = lifecycle_data.idxmax(axis=1)
                stage_changes = (dominant_stages != dominant_stages.shift()).sum()
                
                # 異常に多い段階変化
                expected_changes = len(lifecycle_data) * 0.2  # 期待値は20%
                anomalies['excessive_stage_transitions'] = stage_changes > expected_changes * 1.5
                anomalies['stage_transition_rate'] = stage_changes / len(lifecycle_data)
            
            # 2. 年齢と段階の不整合検出
            if 'company_age' in financial_data.columns and not lifecycle_data.empty:
                latest_age = financial_data['company_age'].iloc[-1]
                latest_stages = lifecycle_data.iloc[-1]
                dominant_stage = latest_stages.idxmax()
                
                # 年齢と段階の期待適合度
                stage_config = self.lifecycle_stages.get(dominant_stage, {})
                age_range = stage_config.get('age_range', (0, float('inf')))
                
                age_stage_mismatch = not (age_range[0] <= latest_age <= age_range[1])
                anomalies['age_stage_mismatch'] = age_stage_mismatch
                
                # 不整合度のスコア化
                if age_stage_mismatch:
                    mid_age = (age_range[0] + age_range[1]) / 2
                    mismatch_score = abs(latest_age - mid_age) / mid_age
                    anomalies['mismatch_severity'] = mismatch_score
            
            # 3. 財務指標の異常パターン
            if not financial_data.empty:
                # 急激な業績変化
                if 'revenue_growth_rate' in financial_data.columns:
                    growth_data = financial_data['revenue_growth_rate'].dropna()
                    if len(growth_data) >= 3:
                        # 3年間の成長率変動が異常に大きい
                        growth_volatility = growth_data.tail(3).std()
                        anomalies['excessive_growth_volatility'] = growth_volatility > 0.5
                        anomalies['growth_volatility_score'] = growth_volatility
                
                # 収益性の急変
                if 'operating_margin' in financial_data.columns:
                    margin_data = financial_data['operating_margin'].dropna()
                    if len(margin_data) >= 2:
                        margin_change = abs(margin_data.iloc[-1] - margin_data.iloc[-2])
                        anomalies['dramatic_margin_change'] = margin_change > 0.1  # 10%以上の変化
                        anomalies['margin_change_magnitude'] = margin_change
            
            # 4. 市場環境との不整合
            # （市場カテゴリと企業パフォーマンスの整合性チェック）
            
        except Exception as e:
            logger.error(f"ライフサイクル異常検出エラー: {e}")
        
        return anomalies
    
    def generate_all_lifecycle_features(self, 
                                        company_data: Dict,
                                        financial_data: pd.DataFrame,
                                        market_category: str) -> Dict[str, Union[float, Dict]]:
        """
        全ライフサイクル特徴量を生成
        
        Args:
            company_data: 企業基本データ（設立日等）
            financial_data: 財務データ
            market_category: 市場カテゴリ
            
        Returns:
            全ライフサイクル特徴量
        """
        all_features = {}
        
        try:
            # 1. 基本ライフサイクル情報
            establishment_date = company_data.get('establishment_date')
            if establishment_date:
                company_age = self.calculate_company_age(establishment_date)
                all_features['company_age'] = company_age
                all_features['company_age_normalized'] = min(company_age / 50, 2)  # 50年で正規化、上限2
            
            # 2. ライフサイクル段階確率
            if not financial_data.empty:
                stage_probabilities = self.identify_lifecycle_stage(
                    financial_data, 
                    company_age, 
                    market_category
                )
                all_features['lifecycle_stages'] = stage_probabilities
                
                # 最も確率の高い段階
                dominant_stage = max(stage_probabilities.items(), key=lambda x: x[1])[0]
                all_features['dominant_lifecycle_stage'] = dominant_stage
                all_features['stage_confidence'] = stage_probabilities[dominant_stage]
            
            # 3. 段階遷移特徴量（履歴データがある場合）
            if len(financial_data) > 5:  # 5年以上のデータがある場合
                # 各年のライフサイクル段階を計算
                historical_stages = []
                for i in range(5, len(financial_data) + 1):
                    subset_data = financial_data.iloc[:i]
                    subset_age = self.calculate_company_age(
                        establishment_date, 
                        subset_data.index[-1] if hasattr(subset_data.index[-1], 'date') else None
                    )
                    stages = self.identify_lifecycle_stage(subset_data, subset_age, market_category)
                    historical_stages.append(stages)
                
                if historical_stages:
                    historical_df = pd.DataFrame(historical_stages)
                    transition_features = self.generate_stage_transition_features(historical_df)
                    all_features['transition_features'] = transition_features
            
            # 4. ライフサイクル・モメンタム
            momentum_features = self.calculate_lifecycle_momentum(financial_data)
            all_features['momentum_features'] = momentum_features
            
            # 5. 成熟度指標
            maturity_indicators = self.generate_maturity_indicators(financial_data)
            all_features['maturity_indicators'] = maturity_indicators
            
            # 6. 異常検出
            if len(financial_data) > 3:
                lifecycle_df = pd.DataFrame([all_features.get('lifecycle_stages', {})])
                anomalies = self.detect_lifecycle_anomalies(lifecycle_df, financial_data)
                all_features['lifecycle_anomalies'] = anomalies
            
            # 7. 市場適応度（市場カテゴリとの整合性）
            market_adaptation_score = self._calculate_market_adaptation(
                all_features, market_category
            )
            all_features['market_adaptation_score'] = market_adaptation_score
            
        except Exception as e:
            logger.error(f"全ライフサイクル特徴量生成エラー: {e}")
        
        return all_features
    
    def _calculate_market_adaptation(self, 
                                    lifecycle_features: Dict,
                                    market_category: str) -> float:
        """
        市場適応度を計算
        
        Args:
            lifecycle_features: ライフサイクル特徴量
            market_category: 市場カテゴリ
            
        Returns:
            市場適応度スコア（0-1）
        """
        adaptation_score = 0.5  # デフォルト
        
        try:
            dominant_stage = lifecycle_features.get('dominant_lifecycle_stage')
            momentum = lifecycle_features.get('momentum_features', {})
            maturity = lifecycle_features.get('maturity_indicators', {})
            
            if market_category == 'high_share':
                # 高シェア市場：成熟段階または成長段階が適している
                if dominant_stage in ['maturity', 'growth']:
                    adaptation_score = 0.8
                elif dominant_stage == 'transformation':
                    adaptation_score = 0.6
                
                # 安定性が重要
                if maturity.get('overall_maturity', 0) > 0.7:
                    adaptation_score += 0.1
                    
            elif market_category == 'declining':
                # 低下市場：変革段階が最適
                if dominant_stage == 'transformation':
                    adaptation_score = 0.9
                elif dominant_stage in ['maturity', 'decline']:
                    adaptation_score = 0.6
                
                # イノベーション・モメンタムが重要
                innovation_momentum = momentum.get('innovation_momentum', 0)
                if innovation_momentum > 0:
                    adaptation_score += 0.1
                    
            elif market_category == 'lost':
                # 失失市場：変革または衰退段階
                if dominant_stage == 'transformation':
                    adaptation_score = 0.7  # 変革による復活可能性
                elif dominant_stage == 'decline':
                    adaptation_score = 0.3  # 市場と整合するが望ましくない
                elif dominant_stage == 'startup':
                    adaptation_score = 0.8  # 新規事業での再生
                
                # 全体的モメンタムが重要
                overall_momentum = momentum.get('overall_momentum', 0)
                if overall_momentum > 0:
                    adaptation_score += 0.2
            
            # 上限・下限の設定
            adaptation_score = max(0.0, min(1.0, adaptation_score))
            
        except Exception as e:
            logger.warning(f"市場適応度計算エラー: {e}")
        
        return adaptation_score
    
    def create_lifecycle_summary_report(self, 
                                        lifecycle_features: Dict,
                                        company_name: str = "対象企業") -> str:
        """
        ライフサイクル分析サマリーレポートを作成
        
        Args:
            lifecycle_features: ライフサイクル特徴量
            company_name: 企業名
            
        Returns:
            サマリーレポート（文字列）
        """
        report_lines = []
        
        try:
            report_lines.append(f"=== {company_name} ライフサイクル分析レポート ===\n")
            
            # 1. 基本情報
            company_age = lifecycle_features.get('company_age', 0)
            report_lines.append(f"企業年齢: {company_age:.1f}年")
            
            # 2. 現在のライフサイクル段階
            dominant_stage = lifecycle_features.get('dominant_lifecycle_stage', 'unknown')
            stage_confidence = lifecycle_features.get('stage_confidence', 0)
            
            stage_names = {
                'startup': 'スタートアップ期',
                'growth': '成長期',
                'maturity': '成熟期',
                'decline': '衰退期',
                'transformation': '変革期'
            }
            
            stage_name_ja = stage_names.get(dominant_stage, '不明')
            report_lines.append(f"現在のライフサイクル段階: {stage_name_ja} (信頼度: {stage_confidence:.1%})")
            
            # 3. ライフサイクル段階の詳細確率
            stages = lifecycle_features.get('lifecycle_stages', {})
            if stages:
                report_lines.append("\n各段階確率:")
                for stage, prob in sorted(stages.items(), key=lambda x: x[1], reverse=True):
                    stage_name_ja = stage_names.get(stage, stage)
                    report_lines.append(f"  {stage_name_ja}: {prob:.1%}")
            
            # 4. モメンタム分析
            momentum = lifecycle_features.get('momentum_features', {})
            if momentum:
                report_lines.append("\nモメンタム分析:")
                
                overall_momentum = momentum.get('overall_momentum', 0)
                momentum_direction = "上昇" if overall_momentum > 0.05 else "下降" if overall_momentum < -0.05 else "横ばい"
                report_lines.append(f"  総合モメンタム: {momentum_direction} ({overall_momentum:.3f})")
                
                momentum_items = {
                    'growth_momentum': '成長モメンタム',
                    'profitability_momentum': '収益性モメンタム',
                    'efficiency_momentum': '効率性モメンタム',
                    'innovation_momentum': 'イノベーション・モメンタム'
                }
                
                for key, name in momentum_items.items():
                    if key in momentum:
                        value = momentum[key]
                        direction = "↑" if value > 0.02 else "↓" if value < -0.02 else "→"
                        report_lines.append(f"  {name}: {direction} ({value:.3f})")
            
            # 5. 成熟度指標
            maturity = lifecycle_features.get('maturity_indicators', {})
            if maturity:
                overall_maturity = maturity.get('overall_maturity', 0)
                maturity_level = "高" if overall_maturity > 0.7 else "中" if overall_maturity > 0.4 else "低"
                report_lines.append(f"\n成熟度: {maturity_level} ({overall_maturity:.1%})")
                
                maturity_items = {
                    'revenue_stability': '収益安定性',
                    'market_position_strength': '市場地位',
                    'capital_structure_stability': '資本構造安定性',
                    'business_diversification': '事業多角化'
                }
                
                for key, name in maturity_items.items():
                    if key in maturity:
                        value = maturity[key]
                        level = "高" if value > 0.7 else "中" if value > 0.4 else "低"
                        report_lines.append(f"  {name}: {level} ({value:.1%})")
            
            # 6. 段階遷移特徴量
            transition = lifecycle_features.get('transition_features', {})
            if transition:
                report_lines.append("\n段階遷移分析:")
                
                stage_changes = transition.get('stage_transition_count', 0)
                report_lines.append(f"  段階変化回数: {stage_changes}回")
                
                avg_duration = transition.get('avg_stage_duration', 0)
                report_lines.append(f"  平均段階継続期間: {avg_duration:.1f}年")
                
                progression = transition.get('lifecycle_progression', 0)
                progression_desc = "進歩的" if progression > 0.1 else "後退的" if progression < -0.1 else "安定的"
                report_lines.append(f"  ライフサイクル進行: {progression_desc} ({progression:.2f})")
            
            # 7. 市場適応度
            adaptation_score = lifecycle_features.get('market_adaptation_score', 0)
            adaptation_level = "高" if adaptation_score > 0.7 else "中" if adaptation_score > 0.4 else "低"
            report_lines.append(f"\n市場適応度: {adaptation_level} ({adaptation_score:.1%})")
            
            # 8. 異常検出
            anomalies = lifecycle_features.get('lifecycle_anomalies', {})
            if anomalies:
                anomaly_detected = any([
                    anomalies.get('excessive_stage_transitions', False),
                    anomalies.get('age_stage_mismatch', False),
                    anomalies.get('excessive_growth_volatility', False),
                    anomalies.get('dramatic_margin_change', False)
                ])
                
                if anomaly_detected:
                    report_lines.append("\n⚠️  異常検出:")
                    
                    if anomalies.get('excessive_stage_transitions', False):
                        transition_rate = anomalies.get('stage_transition_rate', 0)
                        report_lines.append(f"  - 異常に頻繁な段階変化 (変化率: {transition_rate:.1%})")
                    
                    if anomalies.get('age_stage_mismatch', False):
                        mismatch_severity = anomalies.get('mismatch_severity', 0)
                        report_lines.append(f"  - 年齢と段階の不整合 (深刻度: {mismatch_severity:.2f})")
                    
                    if anomalies.get('excessive_growth_volatility', False):
                        volatility = anomalies.get('growth_volatility_score', 0)
                        report_lines.append(f"  - 成長率の異常な変動 (変動度: {volatility:.2f})")
                    
                    if anomalies.get('dramatic_margin_change', False):
                        margin_change = anomalies.get('margin_change_magnitude', 0)
                        report_lines.append(f"  - 利益率の急激な変化 (変化幅: {margin_change:.1%})")
                else:
                    report_lines.append("\n✓ 異常検出: なし")
            
            # 9. 戦略的示唆
            report_lines.append("\n=== 戦略的示唆 ===")
            
            # 段階別の示唆
            if dominant_stage == 'startup':
                report_lines.append("• 成長基盤の確立と市場浸透に注力")
                report_lines.append("• 資金調達とキャッシュフロー管理が重要")
                report_lines.append("• 迅速な意思決定と市場適応力が競争優位の源泉")
                
            elif dominant_stage == 'growth':
                report_lines.append("• 市場シェア拡大と事業規模の拡張に注力")
                report_lines.append("• 組織体制の整備と人材確保が重要")
                report_lines.append("• 効率的な成長投資と収益性のバランス")
                
            elif dominant_stage == 'maturity':
                report_lines.append("• 市場地位の維持と収益性の最大化")
                report_lines.append("• 事業ポートフォリオの最適化")
                report_lines.append("• 配当還元と資本効率の向上")
                
            elif dominant_stage == 'decline':
                report_lines.append("• 事業再構築または撤退戦略の検討")
                report_lines.append("• コスト削減と効率化の徹底")
                report_lines.append("• 新規事業への転換可能性の探索")
                
            elif dominant_stage == 'transformation':
                report_lines.append("• 新規事業・新市場への戦略的転換")
                report_lines.append("• イノベーションと研究開発投資の拡充")
                report_lines.append("• 組織変革とビジネスモデル革新")
            
            # モメンタムに基づく示唆
            if momentum.get('overall_momentum', 0) < -0.1:
                report_lines.append("• 業績下降トレンドへの早急な対策が必要")
            elif momentum.get('innovation_momentum', 0) > 0.1:
                report_lines.append("• R&D投資の成果創出に期待、継続的な投資を推奨")
            
            # 市場適応度に基づく示唆
            if adaptation_score < 0.4:
                report_lines.append("• 市場環境との適合度が低い、戦略の見直しが必要")
                report_lines.append("• 市場トレンドへの対応力強化を推奨")
            
        except Exception as e:
            logger.error(f"レポート作成エラー: {e}")
            report_lines.append(f"レポート作成中にエラーが発生しました: {e}")
        
        return "\n".join(report_lines)
    
    def export_lifecycle_features_to_dataframe(self, 
                                                lifecycle_features: Dict) -> pd.DataFrame:
        """
        ライフサイクル特徴量をDataFrameに変換
        
        Args:
            lifecycle_features: ライフサイクル特徴量
            
        Returns:
            特徴量DataFrame
        """
        try:
            flattened_features = {}
            
            # 基本特徴量
            basic_features = ['company_age', 'company_age_normalized', 'dominant_lifecycle_stage', 
                            'stage_confidence', 'market_adaptation_score']
            
            for feature in basic_features:
                if feature in lifecycle_features:
                    flattened_features[feature] = lifecycle_features[feature]
            
            # ライフサイクル段階確率
            stages = lifecycle_features.get('lifecycle_stages', {})
            for stage, prob in stages.items():
                flattened_features[f'stage_prob_{stage}'] = prob
            
            # モメンタム特徴量
            momentum = lifecycle_features.get('momentum_features', {})
            for key, value in momentum.items():
                flattened_features[f'momentum_{key}'] = value
            
            # 成熟度指標
            maturity = lifecycle_features.get('maturity_indicators', {})
            for key, value in maturity.items():
                flattened_features[f'maturity_{key}'] = value
            
            # 段階遷移特徴量
            transition = lifecycle_features.get('transition_features', {})
            for key, value in transition.items():
                flattened_features[f'transition_{key}'] = value
            
            # 異常検出結果
            anomalies = lifecycle_features.get('lifecycle_anomalies', {})
            for key, value in anomalies.items():
                if isinstance(value, bool):
                    flattened_features[f'anomaly_{key}'] = int(value)  # boolを0/1に変換
                elif isinstance(value, (int, float)):
                    flattened_features[f'anomaly_{key}'] = value
            
            # DataFrameに変換（1行）
            df = pd.DataFrame([flattened_features])
            
            return df
            
        except Exception as e:
            logger.error(f"DataFrame変換エラー: {e}")
            return pd.DataFrame()


def main():
    """
    メイン実行関数（テスト用）
    """
    # サンプルデータの作成
    np.random.seed(42)
    dates = pd.date_range('2000-01-01', '2023-01-01', freq='A')
    
    # サンプル財務データ
    sample_financial_data = pd.DataFrame({
        'revenue': np.cumsum(np.random.normal(100, 20, len(dates))) + 1000,
        'revenue_growth_rate': np.random.normal(0.05, 0.1, len(dates)),
        'operating_margin': np.random.normal(0.1, 0.03, len(dates)),
        'roa': np.random.normal(0.08, 0.02, len(dates)),
        'total_asset_turnover': np.random.normal(1.2, 0.2, len(dates)),
        'r_and_d_ratio': np.random.normal(0.05, 0.01, len(dates)),
        'capex_ratio': np.random.normal(0.04, 0.01, len(dates)),
        'equity_ratio': np.random.normal(0.6, 0.1, len(dates)),
        'dividend_payout_ratio': np.random.normal(0.3, 0.1, len(dates)),
        'segment_count': np.random.randint(2, 6, len(dates)),
    }, index=dates)
    
    # サンプル企業データ
    sample_company_data = {
        'establishment_date': '1990-01-01',
        'company_name': 'サンプル株式会社'
    }
    
    # ライフサイクル特徴量生成器の初期化
    lifecycle_generator = LifecycleFeatureGenerator()
    
    print("=== A2AI ライフサイクル特徴量生成テスト ===\n")
    
    # 全特徴量の生成
    all_features = lifecycle_generator.generate_all_lifecycle_features(
        sample_company_data,
        sample_financial_data,
        'high_share'  # 高シェア市場
    )
    
    # レポート生成
    report = lifecycle_generator.create_lifecycle_summary_report(
        all_features,
        sample_company_data['company_name']
    )
    
    print(report)
    
    # DataFrame形式での出力
    print("\n=== 特徴量DataFrame ===")
    features_df = lifecycle_generator.export_lifecycle_features_to_dataframe(all_features)
    print(features_df.T)  # 転置して見やすく表示
    
    print("\n=== テスト完了 ===")


if __name__ == "__main__":
    main()