"""
A2AI - Advanced Financial Analysis AI
トレンド分析モジュール

企業の財務データから長期トレンドを分析し、市場カテゴリ別の特徴を抽出する。
150社×40年分のデータを対象に、生存バイアスを考慮した分析を実行。

主な機能:
- 時系列トレンド分析
- 市場カテゴリ別比較
- 企業ライフサイクル段階別分析
- 生存・消滅・新設企業の統合分析
- トレンド予測と異常検出
"""

import numpy as np
import pandas as pd
import warnings
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from pathlib import Path

# 統計・機械学習ライブラリ
from scipy import stats
from scipy.signal import savgol_filter, find_peaks
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score

# 時系列分析
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm

# 可視化（結果確認用）
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrendResult:
    """トレンド分析結果を格納するデータクラス"""
    trend_direction: str  # 'increasing', 'decreasing', 'stable', 'volatile'
    trend_strength: float  # トレンドの強さ (0-1)
    slope: float  # 線形トレンドの傾き
    r_squared: float  # 決定係数
    turning_points: List[int]  # 転換点のインデックス
    seasonal_pattern: Optional[np.ndarray]  # 季節性パターン
    anomalies: List[int]  # 異常値のインデックス
    forecast: Optional[np.ndarray]  # 予測値
    confidence_interval: Optional[Tuple[np.ndarray, np.ndarray]]  # 信頼区間


@dataclass
class MarketTrendComparison:
    """市場カテゴリ別トレンド比較結果"""
    high_share_trends: Dict[str, TrendResult]
    declining_trends: Dict[str, TrendResult]
    lost_trends: Dict[str, TrendResult]
    statistical_significance: Dict[str, float]  # p値
    effect_size: Dict[str, float]  # 効果量


class TrendAnalyzer:
    """
    財務データの長期トレンド分析を実行するクラス
    
    150社×40年分のデータから、市場カテゴリ別のトレンド特徴を抽出し、
    企業の生存・成長パターンを分析する。
    """
    
    def __init__(self, 
                    data_path: str = None,
                    config: Dict = None):
        """
        初期化
        
        Args:
            data_path: データファイルパス
            config: 設定辞書
        """
        self.data_path = data_path
        self.config = config or self._default_config()
        
        # 分析対象の評価項目（9つ）
        self.evaluation_metrics = [
            'revenue',  # 売上高
            'revenue_growth_rate',  # 売上高成長率
            'operating_profit_margin',  # 売上高営業利益率
            'net_profit_margin',  # 売上高当期純利益率
            'roe',  # ROE
            'value_added_ratio',  # 売上高付加価値率
            'survival_probability',  # 企業存続確率
            'emergence_success_rate',  # 新規事業成功率
            'succession_success_rate'  # 事業継承成功度
        ]
        
        # 市場カテゴリ
        self.market_categories = {
            'high_share': ['robot', 'endoscope', 'machine_tool', 'electronic_materials', 'precision_instruments'],
            'declining': ['automotive', 'steel', 'home_appliances', 'battery', 'pc_peripherals'],
            'lost': ['consumer_electronics', 'semiconductor', 'smartphone', 'pc', 'telecom_equipment']
        }
        
        # 分析結果格納
        self.trend_results = {}
        self.market_comparison = None
        
        logger.info("TrendAnalyzer initialized successfully")
    
    def _default_config(self) -> Dict:
        """デフォルト設定を返す"""
        return {
            'min_data_points': 10,  # 最小データポイント数
            'smoothing_window': 5,  # 平滑化ウィンドウサイズ
            'anomaly_threshold': 2.0,  # 異常値検出の閾値（標準偏差倍数）
            'trend_significance_level': 0.05,  # トレンド有意水準
            'seasonality_periods': [4, 12],  # 季節性検出期間
            'forecast_periods': 5,  # 予測期間
            'confidence_level': 0.95,  # 信頼区間
            'polynomial_degree': 2,  # 多項式回帰の次数
            'change_point_sensitivity': 0.1  # 変化点検出の感度
        }
    
    def load_data(self, 
                    data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        データを読み込み前処理を実行
        
        Args:
            data: データフレーム（指定しない場合はファイルから読み込み）
        
        Returns:
            前処理済みデータフレーム
        """
        try:
            if data is not None:
                self.data = data.copy()
            elif self.data_path:
                self.data = pd.read_csv(self.data_path)
            else:
                raise ValueError("データが指定されていません")
            
            # 基本的な前処理
            self.data = self._preprocess_data(self.data)
            
            logger.info(f"データ読み込み完了: {self.data.shape[0]}行, {self.data.shape[1]}列")
            return self.data
            
        except Exception as e:
            logger.error(f"データ読み込みエラー: {e}")
            raise
    
    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        データの前処理
        
        Args:
            data: 生データ
        
        Returns:
            前処理済みデータ
        """
        # 日付列の処理
        if 'date' in data.columns:
            data['date'] = pd.to_datetime(data['date'])
            data = data.sort_values(['company_id', 'date'])
        
        # 欠損値処理
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        data[numeric_columns] = data[numeric_columns].fillna(method='ffill').fillna(method='bfill')
        
        # 異常値の初期処理（極値のクリッピング）
        for col in numeric_columns:
            if col in self.evaluation_metrics:
                q99 = data[col].quantile(0.99)
                q01 = data[col].quantile(0.01)
                data[col] = data[col].clip(q01, q99)
        
        return data
    
    def analyze_company_trend(self, 
                            company_data: pd.Series,
                            metric: str) -> TrendResult:
        """
        個別企業の単一指標トレンド分析
        
        Args:
            company_data: 時系列データ
            metric: 分析する指標名
        
        Returns:
            トレンド分析結果
        """
        try:
            # データの準備
            values = company_data.dropna().values
            if len(values) < self.config['min_data_points']:
                logger.warning(f"データポイント不足: {len(values)}")
                return self._empty_trend_result()
            
            time_index = np.arange(len(values))
            
            # 1. 基本統計とトレンド方向の判定
            slope, r_squared = self._calculate_linear_trend(time_index, values)
            trend_direction = self._determine_trend_direction(slope, r_squared)
            trend_strength = self._calculate_trend_strength(values)
            
            # 2. 転換点の検出
            turning_points = self._detect_turning_points(values)
            
            # 3. 季節性の分析
            seasonal_pattern = self._detect_seasonality(values)
            
            # 4. 異常値の検出
            anomalies = self._detect_anomalies(values)
            
            # 5. 予測と信頼区間
            forecast, confidence_interval = self._generate_forecast(values)
            
            return TrendResult(
                trend_direction=trend_direction,
                trend_strength=trend_strength,
                slope=slope,
                r_squared=r_squared,
                turning_points=turning_points,
                seasonal_pattern=seasonal_pattern,
                anomalies=anomalies,
                forecast=forecast,
                confidence_interval=confidence_interval
            )
            
        except Exception as e:
            logger.error(f"企業トレンド分析エラー: {e}")
            return self._empty_trend_result()
    
    def _calculate_linear_trend(self, 
                                time_index: np.ndarray, 
                                values: np.ndarray) -> Tuple[float, float]:
        """線形トレンドの計算"""
        try:
            # 線形回帰
            X = time_index.reshape(-1, 1)
            reg = LinearRegression().fit(X, values)
            
            slope = reg.coef_[0]
            r_squared = reg.score(X, values)
            
            return slope, r_squared
            
        except Exception:
            return 0.0, 0.0
    
    def _determine_trend_direction(self, 
                                    slope: float, 
                                    r_squared: float) -> str:
        """トレンド方向の判定"""
        # 決定係数が低い場合は不安定とみなす
        if r_squared < 0.3:
            return 'volatile'
        
        # 傾きの絶対値が小さい場合は安定とみなす
        if abs(slope) < 0.01:
            return 'stable'
        
        return 'increasing' if slope > 0 else 'decreasing'
    
    def _calculate_trend_strength(self, values: np.ndarray) -> float:
        """トレンド強度の計算"""
        try:
            # 変動係数を基にした強度計算
            cv = np.std(values) / (np.mean(values) + 1e-8)
            
            # 自己相関を考慮した強度
            if len(values) > 2:
                autocorr = np.corrcoef(values[:-1], values[1:])[0, 1]
                if np.isnan(autocorr):
                    autocorr = 0
            else:
                autocorr = 0
            
            # 強度スコア（0-1）
            strength = min(1.0, max(0.0, autocorr + (1 - cv / 2)))
            
            return strength
            
        except Exception:
            return 0.5
    
    def _detect_turning_points(self, values: np.ndarray) -> List[int]:
        """転換点の検出"""
        try:
            if len(values) < 5:
                return []
            
            # 移動平均による平滑化
            window = min(5, len(values) // 3)
            smoothed = pd.Series(values).rolling(window, center=True).mean().dropna()
            
            if len(smoothed) < 3:
                return []
            
            # ピークと谷の検出
            peaks, _ = find_peaks(smoothed, distance=max(1, len(smoothed) // 10))
            valleys, _ = find_peaks(-smoothed, distance=max(1, len(smoothed) // 10))
            
            # 全ての転換点を統合してソート
            turning_points = sorted(list(peaks) + list(valleys))
            
            return turning_points
            
        except Exception:
            return []
    
    def _detect_seasonality(self, values: np.ndarray) -> Optional[np.ndarray]:
        """季節性パターンの検出"""
        try:
            if len(values) < 24:  # 最低2年分のデータが必要
                return None
            
            # 各周期での季節性検出を試行
            for period in self.config['seasonality_periods']:
                if len(values) >= period * 2:
                    try:
                        # 季節分解
                        ts = pd.Series(values)
                        decomposition = seasonal_decompose(ts, 
                                                            model='additive', 
                                                            period=period,
                                                            extrapolate_trend='freq')
                        
                        # 季節成分の有意性検定
                        seasonal_component = decomposition.seasonal.dropna()
                        if len(seasonal_component) > 0 and np.std(seasonal_component) > 0.1:
                            return seasonal_component.values
                            
                    except Exception:
                        continue
            
            return None
            
        except Exception:
            return None
    
    def _detect_anomalies(self, values: np.ndarray) -> List[int]:
        """異常値の検出"""
        try:
            if len(values) < 5:
                return []
            
            # Isolation Forestによる異常検出
            isolation_forest = IsolationForest(
                contamination=0.1,
                random_state=42
            )
            
            anomaly_labels = isolation_forest.fit_predict(values.reshape(-1, 1))
            anomalies = np.where(anomaly_labels == -1)[0].tolist()
            
            # 統計的手法による異常値検出（Z-score）
            z_scores = np.abs(stats.zscore(values))
            statistical_anomalies = np.where(
                z_scores > self.config['anomaly_threshold']
            )[0].tolist()
            
            # 両手法の結果を統合
            combined_anomalies = list(set(anomalies + statistical_anomalies))
            
            return sorted(combined_anomalies)
            
        except Exception:
            return []
    
    def _generate_forecast(self, 
                            values: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[Tuple[np.ndarray, np.ndarray]]]:
        """予測と信頼区間の生成"""
        try:
            if len(values) < 10:
                return None, None
            
            forecast_periods = self.config['forecast_periods']
            
            # ARIMAモデルによる予測
            try:
                # 定常性検定
                adf_result = adfuller(values)
                is_stationary = adf_result[1] < 0.05
                
                # ARIMAパラメータの自動選択
                d = 0 if is_stationary else 1
                model = ARIMA(values, order=(1, d, 1))
                fitted_model = model.fit()
                
                # 予測実行
                forecast_result = fitted_model.forecast(
                    steps=forecast_periods,
                    alpha=1-self.config['confidence_level']
                )
                
                forecast = forecast_result
                # 信頼区間は簡単のため線形トレンドベースで近似
                confidence_interval = self._calculate_confidence_interval(values, forecast_periods)
                
                return forecast, confidence_interval
                
            except Exception:
                # ARIMAが失敗した場合は線形予測を使用
                return self._linear_forecast(values, forecast_periods)
            
        except Exception:
            return None, None
    
    def _linear_forecast(self, 
                        values: np.ndarray, 
                        periods: int) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """線形予測の実行"""
        time_index = np.arange(len(values))
        X = time_index.reshape(-1, 1)
        
        reg = LinearRegression().fit(X, values)
        
        # 予測時点の時間インデックス
        future_time = np.arange(len(values), len(values) + periods).reshape(-1, 1)
        forecast = reg.predict(future_time)
        
        # 信頼区間の計算
        confidence_interval = self._calculate_confidence_interval(values, periods)
        
        return forecast, confidence_interval
    
    def _calculate_confidence_interval(self, 
                                        values: np.ndarray, 
                                        periods: int) -> Tuple[np.ndarray, np.ndarray]:
        """信頼区間の計算"""
        # 残差の標準偏差を用いた信頼区間
        residual_std = np.std(np.diff(values))
        
        # 予測期間の不確実性を考慮
        uncertainty = residual_std * np.sqrt(np.arange(1, periods + 1))
        
        # Z値（95%信頼区間の場合）
        z_value = stats.norm.ppf((1 + self.config['confidence_level']) / 2)
        
        # 最後の値を基準とした信頼区間
        last_value = values[-1]
        lower_bound = last_value - z_value * uncertainty
        upper_bound = last_value + z_value * uncertainty
        
        return lower_bound, upper_bound
    
    def _empty_trend_result(self) -> TrendResult:
        """空のトレンド結果を返す"""
        return TrendResult(
            trend_direction='unknown',
            trend_strength=0.0,
            slope=0.0,
            r_squared=0.0,
            turning_points=[],
            seasonal_pattern=None,
            anomalies=[],
            forecast=None,
            confidence_interval=None
        )
    
    def analyze_market_trends(self) -> MarketTrendComparison:
        """
        市場カテゴリ別トレンド分析の実行
        
        Returns:
            市場比較結果
        """
        try:
            logger.info("市場カテゴリ別トレンド分析開始")
            
            market_results = {}
            
            for category, markets in self.market_categories.items():
                category_results = {}
                
                for market in markets:
                    logger.info(f"分析中: {category} - {market}")
                    
                    # 市場内企業のデータを抽出
                    market_data = self.data[
                        (self.data['market_category'] == category) & 
                        (self.data['market'] == market)
                    ]
                    
                    if market_data.empty:
                        logger.warning(f"データなし: {category} - {market}")
                        continue
                    
                    # 各評価指標のトレンド分析
                    market_trends = {}
                    for metric in self.evaluation_metrics:
                        if metric in market_data.columns:
                            # 市場内全企業の平均トレンド
                            avg_trend = self._analyze_market_average_trend(market_data, metric)
                            market_trends[metric] = avg_trend
                    
                    category_results[market] = market_trends
                
                market_results[category] = category_results
            
            # 統計的比較の実行
            statistical_tests = self._perform_statistical_comparison(market_results)
            
            self.market_comparison = MarketTrendComparison(
                high_share_trends=market_results.get('high_share', {}),
                declining_trends=market_results.get('declining', {}),
                lost_trends=market_results.get('lost', {}),
                statistical_significance=statistical_tests['p_values'],
                effect_size=statistical_tests['effect_sizes']
            )
            
            logger.info("市場カテゴリ別トレンド分析完了")
            return self.market_comparison
            
        except Exception as e:
            logger.error(f"市場トレンド分析エラー: {e}")
            raise
    
    def _analyze_market_average_trend(self, 
                                    market_data: pd.DataFrame, 
                                    metric: str) -> TrendResult:
        """市場内企業の平均トレンド分析"""
        try:
            # 年次平均値の計算
            if 'date' in market_data.columns:
                market_data['year'] = market_data['date'].dt.year
                yearly_avg = market_data.groupby('year')[metric].mean()
            else:
                # 年列が存在する場合
                yearly_avg = market_data.groupby('year')[metric].mean()
            
            return self.analyze_company_trend(yearly_avg, metric)
            
        except Exception as e:
            logger.error(f"市場平均トレンド分析エラー: {e}")
            return self._empty_trend_result()
    
    def _perform_statistical_comparison(self, 
                                        market_results: Dict) -> Dict:
        """市場間の統計的比較"""
        try:
            p_values = {}
            effect_sizes = {}
            
            categories = list(market_results.keys())
            if len(categories) < 2:
                return {'p_values': p_values, 'effect_sizes': effect_sizes}
            
            for metric in self.evaluation_metrics:
                # 各カテゴリからメトリックデータを抽出
                category_data = {}
                for category in categories:
                    slopes = []
                    for market, trends in market_results[category].items():
                        if metric in trends:
                            slopes.append(trends[metric].slope)
                    
                    if slopes:
                        category_data[category] = slopes
                
                # カテゴリ間の比較（ANOVA）
                if len(category_data) >= 2:
                    groups = [data for data in category_data.values()]
                    
                    try:
                        f_stat, p_val = stats.f_oneway(*groups)
                        p_values[metric] = p_val
                        
                        # 効果量（eta squared）の計算
                        eta_squared = self._calculate_eta_squared(groups)
                        effect_sizes[metric] = eta_squared
                        
                    except Exception:
                        p_values[metric] = 1.0
                        effect_sizes[metric] = 0.0
            
            return {'p_values': p_values, 'effect_sizes': effect_sizes}
            
        except Exception as e:
            logger.error(f"統計的比較エラー: {e}")
            return {'p_values': {}, 'effect_sizes': {}}
    
    def _calculate_eta_squared(self, groups: List[List[float]]) -> float:
        """効果量（eta squared）の計算"""
        try:
            # 群内分散と群間分散の計算
            all_values = [val for group in groups for val in group]
            grand_mean = np.mean(all_values)
            
            ss_between = sum(len(group) * (np.mean(group) - grand_mean)**2 for group in groups)
            ss_within = sum(sum((val - np.mean(group))**2 for val in group) for group in groups)
            ss_total = ss_between + ss_within
            
            eta_squared = ss_between / ss_total if ss_total > 0 else 0
            return eta_squared
            
        except Exception:
            return 0.0
    
    def generate_trend_summary(self) -> Dict:
        """
        トレンド分析の要約レポート生成
        
        Returns:
            要約辞書
        """
        try:
            if not self.market_comparison:
                raise ValueError("市場分析が実行されていません")
            
            summary = {
                'analysis_timestamp': datetime.now().isoformat(),
                'total_companies': len(self.data['company_id'].unique()) if hasattr(self, 'data') else 0,
                'analysis_period': self._get_analysis_period(),
                'market_category_summary': {},
                'key_findings': [],
                'statistical_significance': self.market_comparison.statistical_significance
            }
            
            # 各市場カテゴリの要約
            categories = {
                'high_share': self.market_comparison.high_share_trends,
                'declining': self.market_comparison.declining_trends,
                'lost': self.market_comparison.lost_trends
            }
            
            for category, trends in categories.items():
                category_summary = self._summarize_category_trends(trends)
                summary['market_category_summary'][category] = category_summary
            
            # 主要発見事項の抽出
            summary['key_findings'] = self._extract_key_findings()
            
            return summary
            
        except Exception as e:
            logger.error(f"要約生成エラー: {e}")
            return {'error': str(e)}
    
    def _get_analysis_period(self) -> Dict:
        """分析期間の取得"""
        try:
            if hasattr(self, 'data') and 'date' in self.data.columns:
                return {
                    'start_date': self.data['date'].min().isoformat(),
                    'end_date': self.data['date'].max().isoformat(),
                    'total_years': (self.data['date'].max() - self.data['date'].min()).days / 365.25
                }
            else:
                return {'start_date': '1984', 'end_date': '2024', 'total_years': 40}
        except Exception:
            return {'start_date': 'unknown', 'end_date': 'unknown', 'total_years': 0}
    
    def _summarize_category_trends(self, category_trends: Dict) -> Dict:
        """カテゴリ別トレンド要約"""
        summary = {
            'total_markets': len(category_trends),
            'trend_directions': {},
            'average_trend_strength': 0.0,
            'volatile_markets': []
        }
        
        if not category_trends:
            return summary
        
        # 各指標のトレンド方向集計
        for metric in self.evaluation_metrics:
            directions = []
            strengths = []
            
            for market, trends in category_trends.items():
                if metric in trends:
                    directions.append(trends[metric].trend_direction)
                    strengths.append(trends[metric].trend_strength)
            
            if directions:
                # 最頻値の算出
                direction_counts = pd.Series(directions).value_counts()
                summary['trend_directions'][metric] = direction_counts.to_dict()
                
                # 平均強度
                if strengths:
                    summary['average_trend_strength'] = np.mean(strengths)
        
        # 不安定な市場の特定
        for market, trends in category_trends.items():
            volatile_count = sum(1 for metric_trends in trends.values() 
                                if metric_trends.trend_direction == 'volatile')
            if volatile_count > len(self.evaluation_metrics) * 0.5:
                summary['volatile_markets'].append(market)
        
        return summary
    
    def _extract_key_findings(self) -> List[str]:
        """主要発見事項の抽出"""
        findings = []
        
        if not self.market_comparison:
            return findings
        
        # 統計的有意差のある指標を特定
        significant_metrics = [
            metric for metric, p_val in self.market_comparison.statistical_significance.items()
            if p_val < 0.05
        ]
        
        if significant_metrics:
            findings.append(f"統計的有意差が確認された指標: {', '.join(significant_metrics)}")
        
        # 各カテゴリの特徴的パターン
        categories = {
            'high_share': self.market_comparison.high_share_trends,
            'declining': self.market_comparison.declining_trends,
            'lost': self.market_comparison.lost_trends
        }
        
        for category, trends in categories.items():
            if trends:
                # 成長トレンドの多い指標を特定
                growth_metrics = []
                for market, market_trends in trends.items():
                    for metric, trend_result in market_trends.items():
                        if trend_result.trend_direction == 'increasing':
                            growth_metrics.append(metric)
                
                if growth_metrics:
                    most_common = pd.Series(growth_metrics).value_counts().index[0]
                    findings.append(f"{category}市場では{most_common}の成長傾向が顕著")
        
        return findings
    
    def export_results(self, output_path: str = None) -> bool:
        """
        分析結果のエクスポート
        
        Args:
            output_path: 出力パス
        
        Returns:
            エクスポート成功フラグ
        """
        try:
            if not self.market_comparison:
                logger.error("エクスポートする結果がありません")
                return False
            
            output_dir = Path(output_path) if output_path else Path("results/analysis_results/traditional_analysis")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 1. 要約レポート
            summary = self.generate_trend_summary()
            summary_file = output_dir / f"trend_analysis_summary_{timestamp}.json"
            
            import json
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
            
            # 2. 詳細結果（CSV形式）
            detailed_results = self._prepare_detailed_results()
            detailed_file = output_dir / f"trend_analysis_detailed_{timestamp}.csv"
            detailed_results.to_csv(detailed_file, index=False, encoding='utf-8')
            
            # 3. 統計的比較結果
            stats_file = output_dir / f"statistical_comparison_{timestamp}.csv"
            stats_df = pd.DataFrame({
                'metric': list(self.market_comparison.statistical_significance.keys()),
                'p_value': list(self.market_comparison.statistical_significance.values()),
                'effect_size': [self.market_comparison.effect_size.get(k, 0) for k in self.market_comparison.statistical_significance.keys()]
            })
            stats_df.to_csv(stats_file, index=False)
            
            logger.info(f"結果をエクスポートしました: {output_dir}")
            return True
            
        except Exception as e:
            logger.error(f"エクスポートエラー: {e}")
            return False
    
    def _prepare_detailed_results(self) -> pd.DataFrame:
        """詳細結果をDataFrame形式で準備"""
        rows = []
        
        categories = {
            'high_share': self.market_comparison.high_share_trends,
            'declining': self.market_comparison.declining_trends,
            'lost': self.market_comparison.lost_trends
        }
        
        for category, markets in categories.items():
            for market, trends in markets.items():
                for metric, trend_result in trends.items():
                    rows.append({
                        'category': category,
                        'market': market,
                        'metric': metric,
                        'trend_direction': trend_result.trend_direction,
                        'trend_strength': trend_result.trend_strength,
                        'slope': trend_result.slope,
                        'r_squared': trend_result.r_squared,
                        'turning_points_count': len(trend_result.turning_points),
                        'anomalies_count': len(trend_result.anomalies),
                        'has_seasonality': trend_result.seasonal_pattern is not None,
                        'has_forecast': trend_result.forecast is not None
                    })
        
        return pd.DataFrame(rows)
    
    def visualize_trends(self, 
                        metric: str = 'revenue_growth_rate',
                        save_path: str = None) -> Dict:
        """
        トレンド可視化
        
        Args:
            metric: 可視化する指標
            save_path: 保存パス
        
        Returns:
            可視化情報辞書
        """
        try:
            if not hasattr(self, 'data'):
                raise ValueError("データが読み込まれていません")
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'Trend Analysis: {metric}', fontsize=16)
            
            # 1. 市場カテゴリ別平均トレンド
            self._plot_category_trends(axes[0, 0], metric)
            
            # 2. 企業ライフサイクル段階別分析
            self._plot_lifecycle_trends(axes[0, 1], metric)
            
            # 3. 年次変動パターン
            self._plot_yearly_variation(axes[1, 0], metric)
            
            # 4. 異常値と転換点
            self._plot_anomalies_and_turning_points(axes[1, 1], metric)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"可視化を保存: {save_path}")
            
            visualization_info = {
                'metric': metric,
                'visualization_type': 'comprehensive_trend_analysis',
                'subplots': ['category_trends', 'lifecycle_trends', 'yearly_variation', 'anomalies_turning_points'],
                'save_path': save_path
            }
            
            return visualization_info
            
        except Exception as e:
            logger.error(f"可視化エラー: {e}")
            return {'error': str(e)}
    
    def _plot_category_trends(self, ax, metric: str):
        """市場カテゴリ別トレンドプロット"""
        try:
            if 'market_category' in self.data.columns and 'year' in self.data.columns:
                category_trends = self.data.groupby(['market_category', 'year'])[metric].mean().unstack(level=0)
                
                for category in category_trends.columns:
                    ax.plot(category_trends.index, category_trends[category], 
                            label=category, marker='o', linewidth=2)
                
                ax.set_title('Market Category Trends')
                ax.set_xlabel('Year')
                ax.set_ylabel(metric)
                ax.legend()
                ax.grid(True, alpha=0.3)
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center', transform=ax.transAxes)
    
    def _plot_lifecycle_trends(self, ax, metric: str):
        """企業ライフサイクル段階別トレンドプロット"""
        try:
            # 企業年齢を基にしたライフサイクル段階の定義
            if 'company_age' in self.data.columns:
                self.data['lifecycle_stage'] = pd.cut(
                    self.data['company_age'], 
                    bins=[0, 10, 25, 40, 100], 
                    labels=['Startup', 'Growth', 'Mature', 'Legacy']
                )
                
                lifecycle_data = self.data.groupby(['lifecycle_stage', 'year'])[metric].mean().unstack(level=0)
                
                for stage in lifecycle_data.columns:
                    ax.plot(lifecycle_data.index, lifecycle_data[stage], 
                            label=stage, marker='s', linewidth=2)
                
                ax.set_title('Lifecycle Stage Trends')
                ax.set_xlabel('Year')
                ax.set_ylabel(metric)
                ax.legend()
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'Company age data not available', ha='center', va='center', transform=ax.transAxes)
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center', transform=ax.transAxes)
    
    def _plot_yearly_variation(self, ax, metric: str):
        """年次変動パターンプロット"""
        try:
            if 'year' in self.data.columns:
                yearly_stats = self.data.groupby('year')[metric].agg(['mean', 'std', 'median'])
                
                ax.plot(yearly_stats.index, yearly_stats['mean'], label='Mean', color='blue', linewidth=2)
                ax.fill_between(yearly_stats.index, 
                                yearly_stats['mean'] - yearly_stats['std'],
                                yearly_stats['mean'] + yearly_stats['std'],
                                alpha=0.3, color='blue')
                ax.plot(yearly_stats.index, yearly_stats['median'], label='Median', color='red', linestyle='--')
                
                ax.set_title('Yearly Variation Pattern')
                ax.set_xlabel('Year')
                ax.set_ylabel(metric)
                ax.legend()
                ax.grid(True, alpha=0.3)
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center', transform=ax.transAxes)
    
    def _plot_anomalies_and_turning_points(self, ax, metric: str):
        """異常値と転換点のプロット"""
        try:
            # サンプル企業のトレンド分析
            if hasattr(self, 'data') and len(self.data) > 0:
                sample_company = self.data['company_id'].iloc[0]
                company_data = self.data[self.data['company_id'] == sample_company][metric].dropna()
                
                if len(company_data) > 0:
                    trend_result = self.analyze_company_trend(company_data, metric)
                    
                    # 基本トレンド
                    x = np.arange(len(company_data))
                    ax.plot(x, company_data.values, 'b-', linewidth=2, label='Original Data')
                    
                    # 異常値のハイライト
                    if trend_result.anomalies:
                        ax.scatter([x[i] for i in trend_result.anomalies], 
                                    [company_data.iloc[i] for i in trend_result.anomalies],
                                    color='red', s=100, marker='x', label='Anomalies', zorder=5)
                    
                    # 転換点のハイライト
                    if trend_result.turning_points:
                        valid_turning_points = [tp for tp in trend_result.turning_points if tp < len(company_data)]
                        if valid_turning_points:
                            ax.scatter([x[i] for i in valid_turning_points],
                                        [company_data.iloc[i] for i in valid_turning_points],
                                        color='green', s=100, marker='^', label='Turning Points', zorder=5)
                    
                    # 線形トレンド
                    linear_trend = trend_result.slope * x + company_data.iloc[0]
                    ax.plot(x, linear_trend, 'r--', alpha=0.7, label=f'Linear Trend (slope={trend_result.slope:.4f})')
                    
                    ax.set_title(f'Anomalies & Turning Points\n(Sample: {sample_company})')
                    ax.set_xlabel('Time Period')
                    ax.set_ylabel(metric)
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                else:
                    ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center', transform=ax.transAxes)
    
    def detect_market_regime_changes(self) -> Dict:
        """
        市場レジームチェンジの検出
        
        Returns:
            レジームチェンジ情報辞書
        """
        try:
            logger.info("市場レジームチェンジ検出開始")
            
            regime_changes = {}
            
            for category in self.market_categories.keys():
                category_data = self.data[self.data['market_category'] == category]
                
                if category_data.empty:
                    continue
                
                category_changes = {}
                
                for metric in self.evaluation_metrics:
                    if metric in category_data.columns:
                        # 時系列データの準備
                        yearly_avg = category_data.groupby('year')[metric].mean()
                        
                        # 構造変化の検出（Chow test）
                        change_points = self._detect_structural_breaks(yearly_avg.values)
                        
                        if change_points:
                            # 変化点前後のトレンド比較
                            regime_info = self._analyze_regime_characteristics(yearly_avg, change_points)
                            category_changes[metric] = {
                                'change_points': change_points,
                                'regime_characteristics': regime_info
                            }
                
                regime_changes[category] = category_changes
            
            logger.info("市場レジームチェンジ検出完了")
            return regime_changes
            
        except Exception as e:
            logger.error(f"レジームチェンジ検出エラー: {e}")
            return {}
    
    def _detect_structural_breaks(self, data: np.ndarray) -> List[int]:
        """構造変化点の検出"""
        try:
            if len(data) < 10:
                return []
            
            change_points = []
            
            # 移動窓による構造変化検出
            window_size = max(5, len(data) // 5)
            
            for i in range(window_size, len(data) - window_size):
                # 前後の窓での平均の差を検定
                before_window = data[i-window_size:i]
                after_window = data[i:i+window_size]
                
                # t検定による有意差の確認
                t_stat, p_value = stats.ttest_ind(before_window, after_window)
                
                if p_value < 0.05:  # 有意水準5%
                    change_points.append(i)
            
            # 近接した変化点の統合
            if change_points:
                consolidated_points = [change_points[0]]
                for point in change_points[1:]:
                    if point - consolidated_points[-1] > window_size:
                        consolidated_points.append(point)
                return consolidated_points
            
            return []
            
        except Exception:
            return []
    
    def _analyze_regime_characteristics(self, 
                                        data: pd.Series, 
                                        change_points: List[int]) -> Dict:
        """レジーム特性の分析"""
        try:
            regimes = []
            
            # 各レジーム期間の分析
            periods = [0] + change_points + [len(data)]
            
            for i in range(len(periods) - 1):
                start_idx = periods[i]
                end_idx = periods[i + 1]
                
                regime_data = data.iloc[start_idx:end_idx]
                
                if len(regime_data) > 2:
                    # 基本統計
                    mean_level = regime_data.mean()
                    volatility = regime_data.std()
                    
                    # トレンド分析
                    x = np.arange(len(regime_data))
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x, regime_data.values)
                    
                    regimes.append({
                        'period': f"{start_idx}-{end_idx}",
                        'years': f"{data.index[start_idx]}-{data.index[end_idx-1]}" if hasattr(data.index, '__getitem__') else f"{start_idx}-{end_idx-1}",
                        'mean_level': mean_level,
                        'volatility': volatility,
                        'trend_slope': slope,
                        'trend_r_squared': r_value**2,
                        'trend_p_value': p_value
                    })
            
            return {
                'total_regimes': len(regimes),
                'regime_details': regimes,
                'most_volatile_regime': max(regimes, key=lambda x: x['volatility']) if regimes else None,
                'strongest_growth_regime': max(regimes, key=lambda x: x['trend_slope']) if regimes else None
            }
            
        except Exception as e:
            logger.error(f"レジーム特性分析エラー: {e}")
            return {}
    
    def generate_predictive_insights(self) -> Dict:
        """
        予測的インサイトの生成
        
        Returns:
            予測インサイト辞書
        """
        try:
            logger.info("予測的インサイト生成開始")
            
            insights = {
                'risk_alerts': [],
                'opportunity_signals': [],
                'market_outlook': {},
                'company_recommendations': {}
            }
            
            # 1. リスクアラート（下降トレンドの検出）
            insights['risk_alerts'] = self._detect_risk_signals()
            
            # 2. 機会シグナル（上昇トレンドの検出）
            insights['opportunity_signals'] = self._detect_opportunity_signals()
            
            # 3. 市場見通し
            insights['market_outlook'] = self._generate_market_outlook()
            
            # 4. 企業別推奨事項
            insights['company_recommendations'] = self._generate_company_recommendations()
            
            logger.info("予測的インサイト生成完了")
            return insights
            
        except Exception as e:
            logger.error(f"予測的インサイト生成エラー: {e}")
            return {'error': str(e)}
    
    def _detect_risk_signals(self) -> List[Dict]:
        """リスクシグナルの検出"""
        risk_alerts = []
        
        try:
            if not hasattr(self, 'data'):
                return risk_alerts
            
            # 各企業の最近のトレンドを分析
            recent_years = 5  # 直近5年間
            current_year = self.data['year'].max() if 'year' in self.data.columns else 2024
            recent_data = self.data[self.data['year'] >= current_year - recent_years]
            
            for company_id in recent_data['company_id'].unique():
                company_data = recent_data[recent_data['company_id'] == company_id]
                
                # 主要指標の悪化傾向をチェック
                risk_score = 0
                risk_factors = []
                
                for metric in ['revenue_growth_rate', 'operating_profit_margin', 'roe']:
                    if metric in company_data.columns:
                        trend_result = self.analyze_company_trend(company_data[metric], metric)
                        
                        if trend_result.trend_direction == 'decreasing' and trend_result.r_squared > 0.5:
                            risk_score += 1
                            risk_factors.append(f"{metric}_declining")
                        
                        # 異常値の多発
                        if len(trend_result.anomalies) > len(company_data) * 0.3:
                            risk_score += 0.5
                            risk_factors.append(f"{metric}_volatile")
                
                if risk_score >= 1.5:
                    market_category = company_data['market_category'].iloc[0] if 'market_category' in company_data.columns else 'unknown'
                    
                    risk_alerts.append({
                        'company_id': company_id,
                        'market_category': market_category,
                        'risk_score': risk_score,
                        'risk_factors': risk_factors,
                        'alert_level': 'high' if risk_score >= 2.5 else 'medium'
                    })
            
            # リスクスコア順でソート
            risk_alerts.sort(key=lambda x: x['risk_score'], reverse=True)
            
        except Exception as e:
            logger.error(f"リスクシグナル検出エラー: {e}")
        
        return risk_alerts[:20]  # 上位20社
    
    def _detect_opportunity_signals(self) -> List[Dict]:
        """機会シグナルの検出"""
        opportunity_signals = []
        
        try:
            if not hasattr(self, 'data'):
                return opportunity_signals
            
            recent_years = 5
            current_year = self.data['year'].max() if 'year' in self.data.columns else 2024
            recent_data = self.data[self.data['year'] >= current_year - recent_years]
            
            for company_id in recent_data['company_id'].unique():
                company_data = recent_data[recent_data['company_id'] == company_id]
                
                opportunity_score = 0
                opportunity_factors = []
                
                for metric in ['revenue_growth_rate', 'operating_profit_margin', 'roe', 'value_added_ratio']:
                    if metric in company_data.columns:
                        trend_result = self.analyze_company_trend(company_data[metric], metric)
                        
                        if trend_result.trend_direction == 'increasing' and trend_result.r_squared > 0.5:
                            opportunity_score += 1
                            opportunity_factors.append(f"{metric}_growing")
                        
                        # 安定した高い成長
                        if trend_result.trend_strength > 0.7:
                            opportunity_score += 0.5
                            opportunity_factors.append(f"{metric}_stable_growth")
                
                if opportunity_score >= 1.5:
                    market_category = company_data['market_category'].iloc[0] if 'market_category' in company_data.columns else 'unknown'
                    
                    opportunity_signals.append({
                        'company_id': company_id,
                        'market_category': market_category,
                        'opportunity_score': opportunity_score,
                        'opportunity_factors': opportunity_factors,
                        'signal_strength': 'strong' if opportunity_score >= 2.5 else 'moderate'
                    })
            
            opportunity_signals.sort(key=lambda x: x['opportunity_score'], reverse=True)
            
        except Exception as e:
            logger.error(f"機会シグナル検出エラー: {e}")
        
        return opportunity_signals[:20]  # 上位20社
    
    def _generate_market_outlook(self) -> Dict:
        """市場見通しの生成"""
        outlook = {}
        
        try:
            if not self.market_comparison:
                return outlook
            
            for category in ['high_share', 'declining', 'lost']:
                category_trends = getattr(self.market_comparison, f"{category}_trends", {})
                
                if not category_trends:
                    continue
                
                # カテゴリ全体の傾向分析
                growth_metrics = []
                declining_metrics = []
                stable_metrics = []
                
                for market, trends in category_trends.items():
                    for metric, trend_result in trends.items():
                        if trend_result.trend_direction == 'increasing':
                            growth_metrics.append(metric)
                        elif trend_result.trend_direction == 'decreasing':
                            declining_metrics.append(metric)
                        elif trend_result.trend_direction == 'stable':
                            stable_metrics.append(metric)
                
                # 最頻の傾向を特定
                all_trends = growth_metrics + declining_metrics + stable_metrics
                if all_trends:
                    growth_ratio = len(growth_metrics) / len(all_trends)
                    decline_ratio = len(declining_metrics) / len(all_trends)
                    stable_ratio = len(stable_metrics) / len(all_trends)
                    
                    primary_trend = 'growing' if growth_ratio > 0.4 else ('declining' if decline_ratio > 0.4 else 'mixed')
                    
                    outlook[category] = {
                        'primary_trend': primary_trend,
                        'growth_ratio': growth_ratio,
                        'decline_ratio': decline_ratio,
                        'stability_ratio': stable_ratio,
                        'key_growth_metrics': list(set(growth_metrics)),
                        'key_decline_metrics': list(set(declining_metrics)),
                        'outlook_confidence': max(growth_ratio, decline_ratio, stable_ratio)
                    }
            
        except Exception as e:
            logger.error(f"市場見通し生成エラー: {e}")
        
        return outlook
    
    def _generate_company_recommendations(self) -> Dict:
        """企業別推奨事項の生成"""
        recommendations = {}
        
        try:
            if not hasattr(self, 'data'):
                return recommendations
            
            # 業績上位企業の分析
            top_performers = self._identify_top_performers()
            
            for company_id, performance_data in top_performers.items():
                market_category = performance_data.get('market_category', 'unknown')
                
                # 成功要因の分析
                success_factors = self._analyze_success_factors(company_id)
                
                # 推奨アクション
                actions = self._generate_action_recommendations(company_id, success_factors)
                
                recommendations[company_id] = {
                    'market_category': market_category,
                    'performance_rank': performance_data.get('rank', 0),
                    'success_factors': success_factors,
                    'recommended_actions': actions,
                    'priority_level': self._determine_priority_level(performance_data)
                }
            
        except Exception as e:
            logger.error(f"企業推奨事項生成エラー: {e}")
        
        return recommendations
    
    def _identify_top_performers(self) -> Dict:
        """業績上位企業の特定"""
        top_performers = {}
        
        try:
            # 直近5年間の平均業績でランキング
            recent_years = 5
            current_year = self.data['year'].max() if 'year' in self.data.columns else 2024
            recent_data = self.data[self.data['year'] >= current_year - recent_years]
            
            # 複合スコアの計算
            key_metrics = ['revenue_growth_rate', 'operating_profit_margin', 'roe']
            available_metrics = [m for m in key_metrics if m in recent_data.columns]
            
            if not available_metrics:
                return top_performers
            
            company_scores = {}
            
            for company_id in recent_data['company_id'].unique():
                company_data = recent_data[recent_data['company_id'] == company_id]
                
                scores = []
                for metric in available_metrics:
                    metric_mean = company_data[metric].mean()
                    if not pd.isna(metric_mean):
                        scores.append(metric_mean)
                
                if scores:
                    composite_score = np.mean(scores)
                    market_category = company_data['market_category'].iloc[0] if 'market_category' in company_data.columns else 'unknown'
                    
                    company_scores[company_id] = {
                        'composite_score': composite_score,
                        'market_category': market_category
                    }
            
            # 上位20社を選択
            sorted_companies = sorted(company_scores.items(), key=lambda x: x[1]['composite_score'], reverse=True)[:20]
            
            for rank, (company_id, data) in enumerate(sorted_companies, 1):
                top_performers[company_id] = {
                    'rank': rank,
                    'composite_score': data['composite_score'],
                    'market_category': data['market_category']
                }
            
        except Exception as e:
            logger.error(f"上位企業特定エラー: {e}")
        
        return top_performers
    
    def _analyze_success_factors(self, company_id: str) -> List[str]:
        """成功要因の分析"""
        try:
            company_data = self.data[self.data['company_id'] == company_id]
            if company_data.empty:
                return []
            
            success_factors = []
            
            # 各指標のトレンド分析
            for metric in self.evaluation_metrics:
                if metric in company_data.columns:
                    trend_result = self.analyze_company_trend(company_data[metric], metric)
                    
                    if trend_result.trend_direction == 'increasing' and trend_result.trend_strength > 0.7:
                        success_factors.append(f"consistent_growth_in_{metric}")
                    
                    if len(trend_result.anomalies) < len(company_data) * 0.1:
                        success_factors.append(f"stable_{metric}")
            
            # 市場比較での優位性
            market_category = company_data['market_category'].iloc[0] if 'market_category' in company_data.columns else None
            if market_category:
                market_peers = self.data[self.data['market_category'] == market_category]
                
                for metric in ['revenue_growth_rate', 'operating_profit_margin']:
                    if metric in company_data.columns and metric in market_peers.columns:
                        company_avg = company_data[metric].mean()
                        market_avg = market_peers[metric].mean()
                        
                        if company_avg > market_avg * 1.2:  # 20%以上高い
                            success_factors.append(f"outperforming_market_in_{metric}")
            
            return success_factors
            
        except Exception as e:
            logger.error(f"成功要因分析エラー: {e}")
            return []
    
    def _generate_action_recommendations(self, 
                                        company_id: str, 
                                        success_factors: List[str]) -> List[str]:
        """アクション推奨事項の生成"""
        actions = []
        
        try:
            company_data = self.data[self.data['company_id'] == company_id]
            if company_data.empty:
                return actions
            
            # 成功要因に基づく推奨
            if any('growth' in factor for factor in success_factors):
                actions.append("Continue investing in growth drivers")
                actions.append("Consider market expansion opportunities")
            
            if any('stable' in factor for factor in success_factors):
                actions.append("Maintain operational excellence")
                actions.append("Focus on incremental improvements")
            
            if any('outperforming' in factor for factor in success_factors):
                actions.append("Leverage competitive advantages")
                actions.append("Consider strategic partnerships or acquisitions")
            
            # 業界特有の推奨事項
            market_category = company_data['market_category'].iloc[0] if 'market_category' in company_data.columns else None
            
            if market_category == 'high_share':
                actions.append("Invest in R&D to maintain technological leadership")
                actions.append("Explore emerging market opportunities")
            elif market_category == 'declining':
                actions.append("Consider business model transformation")
                actions.append("Focus on operational efficiency improvements")
            elif market_category == 'lost':
                actions.append("Evaluate strategic repositioning options")
                actions.append("Consider diversification into adjacent markets")
            
            # 重複を除去
            actions = list(set(actions))
            
        except Exception as e:
            logger.error(f"アクション推奨生成エラー: {e}")
        
        return actions[:5]  # 上位5つの推奨事項
    
    def _determine_priority_level(self, performance_data: Dict) -> str:
        """優先度レベルの決定"""
        try:
            rank = performance_data.get('rank', 100)
            composite_score = performance_data.get('composite_score', 0)
            
            if rank <= 5 and composite_score > 0.1:
                return 'high'
            elif rank <= 15 and composite_score > 0.05:
                return 'medium'
            else:
                return 'low'
                
        except Exception:
            return 'low'


# ユーティリティ関数
def calculate_trend_correlation(trend_results1: Dict, trend_results2: Dict) -> float:
    """
    2つのトレンド結果間の相関を計算
    
    Args:
        trend_results1: 第1のトレンド結果辞書
        trend_results2: 第2のトレンド結果辞書
    
    Returns:
        相関係数
    """
    try:
        slopes1 = []
        slopes2 = []
        
        common_metrics = set(trend_results1.keys()) & set(trend_results2.keys())
        
        for metric in common_metrics:
            if hasattr(trend_results1[metric], 'slope') and hasattr(trend_results2[metric], 'slope'):
                slopes1.append(trend_results1[metric].slope)
                slopes2.append(trend_results2[metric].slope)
        
        if len(slopes1) >= 2:
            correlation, _ = stats.pearsonr(slopes1, slopes2)
            return correlation if not np.isnan(correlation) else 0.0
        else:
            return 0.0
            
    except Exception:
        return 0.0


def identify_trend_clusters(market_data: pd.DataFrame, 
                            metrics: List[str], 
                            n_clusters: int = 3) -> Dict:
    """
    企業をトレンドパターンに基づいてクラスタリング
    
    Args:
        market_data: 市場データ
        metrics: 分析対象指標リスト
        n_clusters: クラスタ数
    
    Returns:
        クラスタリング結果辞書
    """
    try:
        # 各企業の各指標のトレンド特徴量を計算
        trend_features = []
        company_ids = []
        
        for company_id in market_data['company_id'].unique():
            company_data = market_data[market_data['company_id'] == company_id]
            
            features = []
            for metric in metrics:
                if metric in company_data.columns:
                    # 簡単なトレンド特徴量（傾き、変動係数）
                    values = company_data[metric].dropna()
                    if len(values) >= 3:
                        slope, _, _, _, _ = stats.linregress(range(len(values)), values)
                        cv = np.std(values) / (np.mean(values) + 1e-8)
                        features.extend([slope, cv])
                    else:
                        features.extend([0, 0])
                else:
                    features.extend([0, 0])
            
            if features:
                trend_features.append(features)
                company_ids.append(company_id)
        
        if len(trend_features) < n_clusters:
            return {'error': 'Insufficient data for clustering'}
        
        # 特徴量の標準化
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(trend_features)
        
        # K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(scaled_features)
        
        # 結果の整理
        clusters = {}
        for i in range(n_clusters):
            cluster_companies = [company_ids[j] for j in range(len(company_ids)) if cluster_labels[j] == i]
            clusters[f'cluster_{i}'] = {
                'companies': cluster_companies,
                'size': len(cluster_companies),
                'centroid': kmeans.cluster_centers_[i].tolist()
            }
        
        return {
            'clusters': clusters,
            'total_companies': len(company_ids),
            'silhouette_score': None  # 必要に応じて計算
        }
        
    except Exception as e:
        return {'error': str(e)}


def detect_cyclical_patterns(time_series: pd.Series, 
                            min_cycle_length: int = 3) -> Dict:
    """
    時系列データから循環パターンを検出
    
    Args:
        time_series: 時系列データ
        min_cycle_length: 最小サイクル長
    
    Returns:
        循環パターン情報辞書
    """
    try:
        if len(time_series) < min_cycle_length * 2:
            return {'cycles_detected': False}
        
        # FFTによる周波数分析
        values = time_series.dropna().values
        fft = np.fft.fft(values)
        freqs = np.fft.fftfreq(len(values))
        
        # パワースペクトル
        power = np.abs(fft) ** 2
        
        # 主要な周波数成分の特定
        # DC成分を除く
        non_dc_indices = np.where(freqs != 0)[0]
        if len(non_dc_indices) == 0:
            return {'cycles_detected': False}
        
        # 最も強い周波数成分
        max_power_idx = non_dc_indices[np.argmax(power[non_dc_indices])]
        dominant_freq = freqs[max_power_idx]
        dominant_period = 1 / abs(dominant_freq) if dominant_freq != 0 else len(values)
        
        # 循環の検出判定
        cycles_detected = (min_cycle_length <= dominant_period <= len(values) / 2 and
                          power[max_power_idx] > np.mean(power) * 2)
        
        result = {
            'cycles_detected': cycles_detected,
            'dominant_period': dominant_period if cycles_detected else None,
            'cycle_strength': power[max_power_idx] / np.sum(power) if cycles_detected else 0,
            'total_cycles': len(values) / dominant_period if cycles_detected and dominant_period > 0 else 0
        }
        
        return result
        
    except Exception as e:
        return {'cycles_detected': False, 'error': str(e)}


# 使用例とテスト関数
def demo_trend_analysis():
    """
    A2AI TrendAnalyzer のデモンストレーション
    """
    print("A2AI Trend Analyzer Demo")
    print("=" * 50)
    
    # サンプルデータの生成
    np.random.seed(42)
    companies = [f'company_{i:03d}' for i in range(150)]
    years = list(range(1984, 2025))
    
    sample_data = []
    
    for i, company in enumerate(companies):
        # 市場カテゴリの割り当て
        if i < 50:
            market_category = 'high_share'
            base_growth = 0.05
        elif i < 100:
            market_category = 'declining'
            base_growth = -0.02
        else:
            market_category = 'lost'
            base_growth = -0.05
        
        # 企業の基本パラメータ
        initial_revenue = np.random.uniform(100, 1000)
        volatility = np.random.uniform(0.1, 0.3)
        
        for year in years:
            # トレンド + ランダムウォーク
            age = year - 1984
            trend_factor = base_growth + np.random.normal(0, 0.01)
            
            revenue = initial_revenue * (1 + trend_factor) ** age * np.random.uniform(0.8, 1.2)
            revenue_growth = trend_factor + np.random.normal(0, volatility)
            
            sample_data.append({
                'company_id': company,
                'year': year,
                'market_category': market_category,
                'market': f'market_{i%5}',
                'revenue': revenue,
                'revenue_growth_rate': revenue_growth,
                'operating_profit_margin': np.random.uniform(-0.1, 0.2),
                'roe': np.random.uniform(-0.2, 0.3),
                'company_age': age
            })
    
    df = pd.DataFrame(sample_data)
    
    # TrendAnalyzer の初期化と実行
    analyzer = TrendAnalyzer()
    analyzer.load_data(df)
    
    print(f"データ読み込み完了: {len(df)} レコード")
    
    # 市場トレンド分析の実行
    print("\n市場トレンド分析実行中...")
    market_comparison = analyzer.analyze_market_trends()
    
    # 結果の表示
    print("\n=== 分析結果サマリー ===")
    summary = analyzer.generate_trend_summary()
    
    for category, summary_data in summary.get('market_category_summary', {}).items():
        print(f"\n{category.upper()} 市場:")
        print(f"  総市場数: {summary_data['total_markets']}")
        print(f"  平均トレンド強度: {summary_data['average_trend_strength']:.3f}")
        print(f"  不安定な市場: {summary_data['volatile_markets']}")
    
    # 予測的インサイト
    print("\n=== 予測的インサイト ===")
    insights = analyzer.generate_predictive_insights()
    
    print(f"リスクアラート: {len(insights['risk_alerts'])} 企業")
    print(f"機会シグナル: {len(insights['opportunity_signals'])} 企業")
    
    # レジームチェンジの検出
    print("\n=== レジームチェンジ検出 ===")
    regime_changes = analyzer.detect_market_regime_changes()
    
    for category, changes in regime_changes.items():
        change_count = sum(len(metric_changes.get('change_points', [])) 
                            for metric_changes in changes.values())
        print(f"{category}: {change_count} 個の構造変化を検出")
    
    print("\n=== デモ完了 ===")
    print("詳細な分析結果は analyzer オブジェクトから取得可能です。")
    
    return analyzer


if __name__ == "__main__":
    # デモの実行
    analyzer = demo_trend_analysis()
    
    # 可視化の例
    try:
        print("\n可視化例の生成中...")
        viz_info = analyzer.visualize_trends('revenue_growth_rate')
        print(f"可視化完了: {viz_info}")
        
        # 結果のエクスポート例
        print("\n結果エクスポート中...")
        export_success = analyzer.export_results()
        print(f"エクスポート{'成功' if export_success else '失敗'}")
        
    except Exception as e:
        print(f"可視化/エクスポートエラー: {e}")
    
    print("\nA2AI Trend Analyzer デモンストレーション完了")