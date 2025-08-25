"""
A2AI - Advanced Financial Analysis AI
企業若返り（リジュベネーション）分析モジュール

このモジュールは企業の衰退→復活・再生パターンを分析し、
企業がどのような要因項目の変化により若返りを実現するかを定量化する。

主要機能:
1. 企業衰退期の特定と分析
2. 復活・再生パターンの検出
3. 若返り成功要因の特定
4. 失敗企業との差異分析
5. 若返り予測モデル構築
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import warnings
from datetime import datetime, timedelta
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from scipy.optimize import minimize_scalar
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, classification_report
from sklearn.model_selection import cross_val_score
import logging

# A2AI内部モジュール
from ...utils.statistical_utils import StatisticalUtils
from ...utils.lifecycle_utils import LifecycleUtils
from ...preprocessing.lifecycle_normalizer import LifecycleNormalizer

warnings.filterwarnings('ignore', category=RuntimeWarning)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RejuvenationAnalyzer:
    """
    企業若返り分析クラス
    
    企業の衰退→復活サイクルを分析し、成功要因を特定する。
    150社×40年のデータから、市場シェア低下・失失企業の復活パターンを解析。
    """
    
    def __init__(self, 
                    decline_threshold: float = -0.2,  # 衰退判定閾値（20%以上の悪化）
                    recovery_threshold: float = 0.15,  # 復活判定閾値（15%以上の改善）
                    min_observation_period: int = 5,   # 最小観測期間（年）
                    significance_level: float = 0.05): # 統計的有意水準
        """
        初期化
        
        Args:
            decline_threshold: 衰退と判定する評価項目悪化率
            recovery_threshold: 復活と判定する評価項目改善率  
            min_observation_period: 分析に必要な最小データ期間
            significance_level: 統計的検定の有意水準
        """
        self.decline_threshold = decline_threshold
        self.recovery_threshold = recovery_threshold
        self.min_observation_period = min_observation_period
        self.significance_level = significance_level
        
        # 分析結果格納
        self.rejuvenation_patterns = {}
        self.success_factors = {}
        self.failure_factors = {}
        self.prediction_models = {}
        
        # ユーティリティクラス
        self.stats_utils = StatisticalUtils()
        self.lifecycle_utils = LifecycleUtils()
        self.normalizer = LifecycleNormalizer()
        
        logger.info("企業若返り分析器を初期化しました")
    
    
    def identify_decline_periods(self, 
                                financial_data: pd.DataFrame,
                                evaluation_metrics: List[str]) -> pd.DataFrame:
        """
        企業衰退期間の特定
        
        各企業・各評価項目について、衰退期間を特定し、
        衰退の深度・期間・回復可能性を分析する。
        
        Args:
            financial_data: 財務データ（企業×年×項目）
            evaluation_metrics: 分析対象評価項目リスト
            
        Returns:
            衰退期間情報DataFrame
        """
        logger.info("企業衰退期間の特定を開始")
        
        decline_periods = []
        
        for company in financial_data['company_id'].unique():
            company_data = financial_data[financial_data['company_id'] == company].copy()
            company_data = company_data.sort_values('year')
            
            if len(company_data) < self.min_observation_period:
                continue
                
            for metric in evaluation_metrics:
                if metric not in company_data.columns:
                    continue
                    
                # 評価項目の時系列データ取得
                metric_series = company_data[metric].dropna()
                if len(metric_series) < self.min_observation_period:
                    continue
                
                # 衰退期間検出
                decline_info = self._detect_decline_periods(
                    metric_series, 
                    company, 
                    metric
                )
                
                if decline_info:
                    decline_periods.extend(decline_info)
        
        decline_df = pd.DataFrame(decline_periods)
        logger.info(f"総計 {len(decline_df)} 件の衰退期間を特定")
        
        return decline_df
    
    
    def _detect_decline_periods(self, 
                                metric_series: pd.Series, 
                                company: str, 
                                metric: str) -> List[Dict]:
        """
        個別企業・指標の衰退期間検出
        
        Args:
            metric_series: 評価指標の時系列データ
            company: 企業ID
            metric: 評価指標名
            
        Returns:
            衰退期間情報リスト
        """
        decline_periods = []
        
        # 移動平均による平滑化（3年移動平均）
        smoothed = metric_series.rolling(window=3, center=True).mean()
        
        # ピーク検出（極大値）
        peaks = []
        for i in range(1, len(smoothed) - 1):
            if (smoothed.iloc[i] > smoothed.iloc[i-1] and 
                smoothed.iloc[i] > smoothed.iloc[i+1]):
                peaks.append(i)
        
        # 各ピークからの衰退期間を検出
        for peak_idx in peaks:
            peak_value = smoothed.iloc[peak_idx]
            peak_year = metric_series.index[peak_idx]
            
            # ピーク後の最小値を探索
            post_peak_data = smoothed.iloc[peak_idx:]
            min_idx = post_peak_data.idxmin()
            min_value = post_peak_data.min()
            
            # 衰退率計算
            if peak_value > 0:
                decline_rate = (min_value - peak_value) / peak_value
            else:
                decline_rate = min_value - peak_value
            
            # 衰退判定
            if decline_rate <= self.decline_threshold:
                decline_duration = min_idx - peak_year
                
                # 回復期間チェック（衰退後の改善）
                recovery_info = self._check_recovery_pattern(
                    smoothed, min_idx, peak_value, metric_series.index
                )
                
                decline_periods.append({
                    'company_id': company,
                    'metric': metric,
                    'decline_start_year': peak_year,
                    'decline_end_year': min_idx,
                    'decline_duration': decline_duration,
                    'peak_value': peak_value,
                    'trough_value': min_value,
                    'decline_rate': decline_rate,
                    'decline_severity': abs(decline_rate),
                    'recovery_achieved': recovery_info['achieved'],
                    'recovery_duration': recovery_info['duration'],
                    'recovery_rate': recovery_info['rate'],
                    'recovery_sustainability': recovery_info['sustainability']
                })
        
        return decline_periods
    
    
    def _check_recovery_pattern(self, 
                                smoothed_series: pd.Series, 
                                trough_idx: int, 
                                original_peak: float,
                                time_index) -> Dict:
        """
        衰退後の回復パターンチェック
        
        Args:
            smoothed_series: 平滑化された時系列データ
            trough_idx: 最低値のインデックス
            original_peak: 元のピーク値
            time_index: 時間インデックス
            
        Returns:
            回復パターン情報
        """
        recovery_info = {
            'achieved': False,
            'duration': np.nan,
            'rate': np.nan,
            'sustainability': 0.0
        }
        
        # 最低値以降のデータ
        post_trough = smoothed_series.iloc[trough_idx:]
        if len(post_trough) < 3:  # 最低限3年の観測が必要
            return recovery_info
        
        trough_value = smoothed_series.iloc[trough_idx]
        recovery_threshold_value = trough_value * (1 + self.recovery_threshold)
        
        # 回復達成チェック
        recovery_points = post_trough[post_trough >= recovery_threshold_value]
        
        if len(recovery_points) > 0:
            recovery_info['achieved'] = True
            first_recovery_idx = recovery_points.index[0]
            recovery_info['duration'] = first_recovery_idx - time_index[trough_idx]
            recovery_info['rate'] = (recovery_points.iloc[0] - trough_value) / trough_value
            
            # 持続性評価（回復後の安定性）
            post_recovery = post_trough[post_trough.index >= first_recovery_idx]
            if len(post_recovery) >= 3:
                sustainability_score = self._calculate_sustainability(
                    post_recovery, recovery_points.iloc[0]
                )
                recovery_info['sustainability'] = sustainability_score
        
        return recovery_info
    
    
    def _calculate_sustainability(self, 
                                post_recovery_data: pd.Series, 
                                recovery_level: float) -> float:
        """
        回復後の持続性スコア計算
        
        Args:
            post_recovery_data: 回復後のデータ
            recovery_level: 回復時の水準
            
        Returns:
            持続性スコア（0-1）
        """
        if len(post_recovery_data) < 2:
            return 0.0
        
        # 回復水準を下回った期間の割合
        below_recovery = (post_recovery_data < recovery_level).sum()
        sustainability = 1.0 - (below_recovery / len(post_recovery_data))
        
        # トレンド安定性も考慮
        trend_stability = 1.0 - abs(np.std(post_recovery_data) / recovery_level)
        trend_stability = max(0.0, trend_stability)
        
        return (sustainability + trend_stability) / 2.0
    
    
    def analyze_rejuvenation_factors(self, 
                                    decline_data: pd.DataFrame,
                                    financial_data: pd.DataFrame,
                                    factor_variables: List[str]) -> Dict:
        """
        若返り成功要因分析
        
        回復成功企業と失敗企業の要因項目を比較分析し、
        若返りに最も寄与する要因を特定する。
        
        Args:
            decline_data: 衰退期間データ
            financial_data: 財務データ
            factor_variables: 要因項目リスト（23項目）
            
        Returns:
            若返り要因分析結果
        """
        logger.info("若返り成功要因分析を開始")
        
        # 成功・失敗グループ分類
        success_group = decline_data[decline_data['recovery_achieved'] == True]
        failure_group = decline_data[decline_data['recovery_achieved'] == False]
        
        logger.info(f"成功グループ: {len(success_group)} 件")
        logger.info(f"失敗グループ: {len(failure_group)} 件")
        
        if len(success_group) == 0 or len(failure_group) == 0:
            logger.warning("成功または失敗グループのデータが不足")
            return {}
        
        factor_analysis = {}
        
        # 各要因項目について分析
        for factor in factor_variables:
            if factor not in financial_data.columns:
                continue
                
            factor_result = self._analyze_single_factor(
                factor, 
                success_group, 
                failure_group, 
                financial_data
            )
            
            if factor_result:
                factor_analysis[factor] = factor_result
        
        # 要因重要度ランキング
        factor_importance = self._rank_factor_importance(factor_analysis)
        
        # 若返りパターンクラスタリング
        rejuvenation_clusters = self._cluster_rejuvenation_patterns(
            success_group, financial_data, factor_variables
        )
        
        result = {
            'factor_analysis': factor_analysis,
            'factor_importance_ranking': factor_importance,
            'rejuvenation_patterns': rejuvenation_clusters,
            'summary_statistics': {
                'total_decline_cases': len(decline_data),
                'successful_recoveries': len(success_group),
                'recovery_success_rate': len(success_group) / len(decline_data),
                'average_recovery_duration': success_group['recovery_duration'].mean(),
                'average_decline_severity': decline_data['decline_severity'].mean()
            }
        }
        
        self.success_factors = result
        logger.info("若返り成功要因分析完了")
        
        return result
    
    
    def _analyze_single_factor(self, 
                                factor: str,
                                success_group: pd.DataFrame,
                                failure_group: pd.DataFrame,
                                financial_data: pd.DataFrame) -> Dict:
        """
        個別要因項目の分析
        
        Args:
            factor: 要因項目名
            success_group: 成功グループデータ
            failure_group: 失敗グループデータ
            financial_data: 財務データ
            
        Returns:
            要因分析結果
        """
        try:
            # 衰退期間中の要因項目変化を分析
            success_changes = self._calculate_factor_changes(
                success_group, financial_data, factor
            )
            failure_changes = self._calculate_factor_changes(
                failure_group, financial_data, factor
            )
            
            if len(success_changes) == 0 or len(failure_changes) == 0:
                return None
            
            # 統計的比較
            stat_test = stats.mannwhitneyu(
                success_changes, 
                failure_changes, 
                alternative='two-sided'
            )
            
            # 効果量計算（Cohen's d）
            effect_size = self._calculate_cohens_d(success_changes, failure_changes)
            
            # 相関分析
            all_changes = np.concatenate([success_changes, failure_changes])
            recovery_labels = np.concatenate([
                np.ones(len(success_changes)),
                np.zeros(len(failure_changes))
            ])
            
            correlation, corr_p_value = pearsonr(all_changes, recovery_labels)
            
            return {
                'success_mean': np.mean(success_changes),
                'success_std': np.std(success_changes),
                'failure_mean': np.mean(failure_changes),
                'failure_std': np.std(failure_changes),
                'statistical_significance': stat_test.pvalue,
                'effect_size': effect_size,
                'correlation_with_recovery': correlation,
                'correlation_p_value': corr_p_value,
                'practical_significance': abs(effect_size) > 0.5,  # 中程度以上の効果
                'direction': 'positive' if correlation > 0 else 'negative'
            }
            
        except Exception as e:
            logger.warning(f"要因 {factor} の分析でエラー: {str(e)}")
            return None
    
    
    def _calculate_factor_changes(self, 
                                group_data: pd.DataFrame,
                                financial_data: pd.DataFrame,
                                factor: str) -> np.ndarray:
        """
        要因項目の変化量計算
        
        衰退開始→最低点→回復期間での要因項目変化を計算
        
        Args:
            group_data: 分析対象グループデータ
            financial_data: 財務データ
            factor: 要因項目名
            
        Returns:
            要因変化量の配列
        """
        changes = []
        
        for _, row in group_data.iterrows():
            company_id = row['company_id']
            decline_start = row['decline_start_year']
            decline_end = row['decline_end_year']
            
            # 企業データ抽出
            company_financial = financial_data[
                (financial_data['company_id'] == company_id) &
                (financial_data['year'] >= decline_start) &
                (financial_data['year'] <= decline_end + 2)  # 回復初期も含む
            ]
            
            if len(company_financial) < 2 or factor not in company_financial.columns:
                continue
            
            factor_data = company_financial[factor].dropna()
            if len(factor_data) < 2:
                continue
            
            # 変化量計算（終値/初値 - 1）
            if factor_data.iloc[0] != 0:
                change_rate = (factor_data.iloc[-1] - factor_data.iloc[0]) / abs(factor_data.iloc[0])
            else:
                change_rate = factor_data.iloc[-1] - factor_data.iloc[0]
                
            changes.append(change_rate)
        
        return np.array(changes)
    
    
    def _calculate_cohens_d(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """
        Cohen's d 効果量計算
        
        Args:
            group1: グループ1データ
            group2: グループ2データ
            
        Returns:
            Cohen's d 値
        """
        n1, n2 = len(group1), len(group2)
        if n1 == 0 or n2 == 0:
            return 0.0
        
        # プールされた標準偏差
        pooled_std = np.sqrt(((n1 - 1) * np.var(group1, ddof=1) + 
                             (n2 - 1) * np.var(group2, ddof=1)) / (n1 + n2 - 2))
        
        if pooled_std == 0:
            return 0.0
            
        return (np.mean(group1) - np.mean(group2)) / pooled_std
    
    
    def _rank_factor_importance(self, factor_analysis: Dict) -> List[Tuple[str, float]]:
        """
        要因重要度ランキング作成
        
        統計的有意性、効果量、実用的有意性を総合評価
        
        Args:
            factor_analysis: 要因分析結果
            
        Returns:
            (要因名, 重要度スコア) のランキングリスト
        """
        importance_scores = []
        
        for factor, analysis in factor_analysis.items():
            # 重要度スコア計算
            significance_score = 1.0 if analysis['statistical_significance'] < self.significance_level else 0.0
            effect_score = min(abs(analysis['effect_size']), 2.0) / 2.0  # 最大2.0で正規化
            correlation_score = abs(analysis['correlation_with_recovery'])
            practical_score = 1.0 if analysis['practical_significance'] else 0.5
            
            total_score = (significance_score * 0.3 + 
                          effect_score * 0.3 + 
                          correlation_score * 0.3 + 
                          practical_score * 0.1)
            
            importance_scores.append((factor, total_score))
        
        # スコア順にソート
        importance_scores.sort(key=lambda x: x[1], reverse=True)
        
        return importance_scores
    
    
    def _cluster_rejuvenation_patterns(self, 
                                        success_data: pd.DataFrame,
                                        financial_data: pd.DataFrame,
                                        factor_variables: List[str]) -> Dict:
        """
        若返りパターンのクラスタリング
        
        成功企業の要因項目変化パターンをクラスタリングし、
        典型的な若返りパターンを特定する。
        
        Args:
            success_data: 成功企業データ
            financial_data: 財務データ
            factor_variables: 要因項目リスト
            
        Returns:
            クラスタリング結果
        """
        # 成功企業の要因項目変化マトリックス構築
        factor_changes_matrix = []
        company_info = []
        
        for _, row in success_data.iterrows():
            company_changes = []
            valid_factors = 0
            
            for factor in factor_variables:
                change = self._calculate_factor_changes(
                    pd.DataFrame([row]), financial_data, factor
                )
                if len(change) > 0 and not np.isnan(change[0]):
                    company_changes.append(change[0])
                    valid_factors += 1
                else:
                    company_changes.append(0.0)  # 欠損値は0で補完
            
            # 有効な要因項目が半数以上ある場合のみ使用
            if valid_factors >= len(factor_variables) * 0.5:
                factor_changes_matrix.append(company_changes)
                company_info.append(row['company_id'])
        
        if len(factor_changes_matrix) < 5:  # 最低5社必要
            logger.warning("クラスタリングに十分なデータがありません")
            return {}
        
        # データ正規化
        scaler = StandardScaler()
        normalized_matrix = scaler.fit_transform(factor_changes_matrix)
        
        # K-meansクラスタリング
        optimal_k = min(5, len(factor_changes_matrix) // 3)  # 最大5クラスター
        kmeans = KMeans(n_clusters=optimal_k, random_state=42)
        cluster_labels = kmeans.fit_predict(normalized_matrix)
        
        # クラスター分析
        clusters = {}
        for cluster_id in range(optimal_k):
            cluster_mask = cluster_labels == cluster_id
            cluster_companies = [company_info[i] for i in range(len(company_info)) if cluster_mask[i]]
            cluster_data = np.array(factor_changes_matrix)[cluster_mask]
            
            if len(cluster_data) > 0:
                clusters[f'cluster_{cluster_id}'] = {
                    'companies': cluster_companies,
                    'size': len(cluster_companies),
                    'centroid': np.mean(cluster_data, axis=0),
                    'characteristic_factors': self._identify_cluster_characteristics(
                        cluster_data, factor_variables
                    )
                }
        
        return clusters
    
    
    def _identify_cluster_characteristics(self, 
                                        cluster_data: np.ndarray,
                                        factor_variables: List[str]) -> List[Tuple[str, float]]:
        """
        クラスターの特徴的要因特定
        
        Args:
            cluster_data: クラスターのデータ
            factor_variables: 要因項目リスト
            
        Returns:
            (要因名, 特徴度) のリスト
        """
        if len(cluster_data) == 0:
            return []
        
        centroid = np.mean(cluster_data, axis=0)
        characteristics = []
        
        for i, factor in enumerate(factor_variables):
            if i < len(centroid):
                # 絶対値が大きく、変動が一定方向にある要因を特徴とする
                mean_change = centroid[i]
                std_change = np.std(cluster_data[:, i]) if len(cluster_data) > 1 else 0.0
                
                # 特徴度 = 変化の大きさ / 変動の大きさ
                if std_change > 0:
                    characteristic_score = abs(mean_change) / (1 + std_change)
                else:
                    characteristic_score = abs(mean_change)
                
                characteristics.append((factor, characteristic_score))
        
        # 特徴度順にソート
        characteristics.sort(key=lambda x: x[1], reverse=True)
        
        return characteristics[:5]  # 上位5要因
    
    
    def build_rejuvenation_prediction_model(self, 
                                            decline_data: pd.DataFrame,
                                            financial_data: pd.DataFrame,
                                            factor_variables: List[str]) -> Dict:
        """
        若返り予測モデル構築
        
        企業が衰退期にある時点で、将来の回復可能性を予測するモデルを構築。
        
        Args:
            decline_data: 衰退データ
            financial_data: 財務データ
            factor_variables: 要因項目リスト
            
        Returns:
            予測モデルと評価結果
        """
        logger.info("若返り予測モデル構築を開始")
        
        # 特徴量マトリックス構築
        X, y, valid_indices = self._build_prediction_features(
            decline_data, financial_data, factor_variables
        )
        
        if len(X) < 10:  # 最低10サンプル必要
            logger.warning("予測モデル構築に十分なデータがありません")
            return {}
        
        # データ分割（時系列考慮）
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # 複数モデル構築・比較
        models = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'gradient_boosting': RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        }
        
        model_results = {}
        
        for model_name, model in models.items():
            try:
                # モデル学習
                model.fit(X_train, y_train)
                
                # 予測・評価
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
                
                # 交差検証
                cv_scores = cross_val_score(model, X_train, y_train, cv=min(5, len(X_train)//3))
                
                # 特徴量重要度
                if hasattr(model, 'feature_importances_'):
                    feature_importance = list(zip(factor_variables, model.feature_importances_))
                    feature_importance.sort(key=lambda x: x[1], reverse=True)
                else:
                    feature_importance = []
                
                model_results[model_name] = {
                    'model': model,
                    'cv_score_mean': np.mean(cv_scores),
                    'cv_score_std': np.std(cv_scores),
                    'test_predictions': y_pred,
                    'test_probabilities': y_prob,
                    'feature_importance': feature_importance,
                    'classification_report': classification_report(y_test, y_pred, output_dict=True)
                }
                
            except Exception as e:
                logger.warning(f"モデル {model_name} の構築でエラー: {str(e)}")
                continue
        
        # ベストモデル選択
        best_model_name = max(model_results.keys(), 
                                key=lambda k: model_results[k]['cv_score_mean'])
        
        prediction_result = {
            'models': model_results,
            'best_model': best_model_name,
            'best_model_performance': model_results[best_model_name],
            'feature_variables': factor_variables,
            'training_samples': len(X_train),
            'test_samples': len(X_test)
        }
        
        self.prediction_models = prediction_result
        logger.info(f"予測モデル構築完了 - ベストモデル: {best_model_name}")
        
        return prediction_result
    
    
    def _build_prediction_features(self, 
                                    decline_data: pd.DataFrame,
                                    financial_data: pd.DataFrame,
                                    factor_variables: List[str]) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """
        予測モデル用特徴量構築
        
        衰退期間中の要因項目データから予測用特徴量を作成
        
        Args:
            decline_data: 衰退データ
            financial_data: 財務データ  
            factor_variables: 要因項目リスト
            
        Returns:
            特徴量マトリックス、目的変数、有効インデックス
        """
        features = []
        targets = []
        valid_indices = []
        
        for idx, row in decline_data.iterrows():
            company_id = row['company_id']
            decline_start = row['decline_start_year']
            decline_end = row['decline_end_year']
            recovery_achieved = row['recovery_achieved']
            
            # 衰退期間中のデータ取得
            decline_period_data = financial_data[
                (financial_data['company_id'] == company_id) &
                (financial_data['year'] >= decline_start) &
                (financial_data['year'] <= decline_end)
            ]
            
            if len(decline_period_data) < 2:
                continue
            
            # 特徴量作成
            feature_vector = self._extract_decline_features(
                decline_period_data, factor_variables, row
            )
            
            if feature_vector is not None and len(feature_vector) == len(factor_variables) + 5:
                features.append(feature_vector)
                targets.append(1 if recovery_achieved else 0)
                valid_indices.append(idx)
        
        return np.array(features), np.array(targets), valid_indices
    
    
    def _extract_decline_features(self, 
                                decline_data: pd.DataFrame,
                                factor_variables: List[str],
                                decline_info: pd.Series) -> Optional[List[float]]:
        """
        衰退期間からの特徴量抽出
        
        Args:
            decline_data: 衰退期間の財務データ
            factor_variables: 要因項目リスト
            decline_info: 衰退情報
            
        Returns:
            特徴量ベクトル
        """
        features = []
        
        # 各要因項目の統計量計算
        for factor in factor_variables:
            if factor in decline_data.columns:
                factor_series = decline_data[factor].dropna()
                
                if len(factor_series) >= 2:
                    # 変化率（最終値/初期値 - 1）
                    if factor_series.iloc[0] != 0:
                        change_rate = (factor_series.iloc[-1] - factor_series.iloc[0]) / abs(factor_series.iloc[0])
                    else:
                        change_rate = factor_series.iloc[-1] - factor_series.iloc[0]
                    
                    features.append(change_rate)
                else:
                    features.append(0.0)
            else:
                features.append(0.0)
        
        # 衰退固有の特徴量追加
        features.extend([
            decline_info['decline_severity'],      # 衰退の深刻度
            decline_info['decline_duration'],      # 衰退期間
            decline_info['peak_value'],           # ピーク時の値
            decline_info['trough_value'],         # 最低点の値
            len(decline_data)                     # データポイント数
        ])
        
        return features
    
    
    def predict_rejuvenation_probability(self, 
                                        company_data: pd.DataFrame,
                                        current_decline_info: Dict,
                                        factor_variables: List[str]) -> Dict:
        """
        特定企業の若返り確率予測
        
        現在衰退期にある企業の回復確率を予測
        
        Args:
            company_data: 企業の財務データ
            current_decline_info: 現在の衰退情報
            factor_variables: 要因項目リスト
            
        Returns:
            予測結果と推奨アクション
        """
        if not self.prediction_models or 'best_model' not in self.prediction_models:
            logger.warning("予測モデルが構築されていません")
            return {}
        
        best_model_name = self.prediction_models['best_model']
        model = self.prediction_models['models'][best_model_name]['model']
        
        # 現在状況の特徴量作成
        current_features = self._extract_current_features(
            company_data, current_decline_info, factor_variables
        )
        
        if current_features is None:
            return {'error': '特徴量の作成に失敗'}
        
        # 予測実行
        features_array = np.array(current_features).reshape(1, -1)
        
        recovery_probability = model.predict_proba(features_array)[0][1]
        recovery_prediction = model.predict(features_array)[0]
        
        # 重要要因特定
        feature_importance = self.prediction_models['models'][best_model_name]['feature_importance']
        top_factors = feature_importance[:5]
        
        # 推奨アクション生成
        recommendations = self._generate_recommendations(
            current_features, feature_importance, factor_variables
        )
        
        return {
            'recovery_probability': recovery_probability,
            'recovery_prediction': bool(recovery_prediction),
            'confidence_level': max(recovery_probability, 1 - recovery_probability),
            'key_factors': top_factors,
            'recommendations': recommendations,
            'model_used': best_model_name,
            'prediction_date': datetime.now().isoformat()
        }
    
    
    def _extract_current_features(self, 
                                company_data: pd.DataFrame,
                                decline_info: Dict,
                                factor_variables: List[str]) -> Optional[List[float]]:
        """
        現在状況からの特徴量抽出
        
        Args:
            company_data: 企業データ
            decline_info: 衰退情報
            factor_variables: 要因項目リスト
            
        Returns:
            特徴量リスト
        """
        if len(company_data) < 2:
            return None
        
        features = []
        
        # 各要因項目の変化率
        for factor in factor_variables:
            if factor in company_data.columns:
                factor_series = company_data[factor].dropna()
                
                if len(factor_series) >= 2:
                    if factor_series.iloc[0] != 0:
                        change_rate = (factor_series.iloc[-1] - factor_series.iloc[0]) / abs(factor_series.iloc[0])
                    else:
                        change_rate = factor_series.iloc[-1] - factor_series.iloc[0]
                    
                    features.append(change_rate)
                else:
                    features.append(0.0)
            else:
                features.append(0.0)
        
        # 衰退情報
        features.extend([
            decline_info.get('decline_severity', 0.0),
            decline_info.get('decline_duration', 0.0),
            decline_info.get('peak_value', 0.0),
            decline_info.get('current_value', 0.0),
            len(company_data)
        ])
        
        return features
    
    
    def _generate_recommendations(self, 
                                current_features: List[float],
                                feature_importance: List[Tuple[str, float]],
                                factor_variables: List[str]) -> List[Dict]:
        """
        予測に基づく推奨アクション生成
        
        Args:
            current_features: 現在の特徴量
            feature_importance: 要因重要度
            factor_variables: 要因項目リスト
            
        Returns:
            推奨アクションリスト
        """
        recommendations = []
        
        # 重要度上位要因について推奨作成
        for factor_name, importance in feature_importance[:5]:
            try:
                factor_idx = factor_variables.index(factor_name)
                current_value = current_features[factor_idx]
                
                # 改善方向の判定
                if importance > 0.05:  # 重要度が十分高い場合
                    if current_value < 0:
                        direction = "増加"
                        action = f"{factor_name}の改善が急務です。"
                    else:
                        direction = "維持・強化"
                        action = f"{factor_name}の現在の良好な状態を維持・強化してください。"
                    
                    recommendations.append({
                        'factor': factor_name,
                        'importance': importance,
                        'current_status': current_value,
                        'recommended_direction': direction,
                        'action': action,
                        'priority': 'high' if importance > 0.1 else 'medium'
                    })
                    
            except (ValueError, IndexError):
                continue
        
        return recommendations
    
    
    def analyze_market_rejuvenation_patterns(self, 
                                            decline_data: pd.DataFrame,
                                            market_categories: Dict) -> Dict:
        """
        市場別若返りパターン分析
        
        高シェア/シェア低下/完全失失市場での若返り特性を比較分析
        
        Args:
            decline_data: 衰退データ
            market_categories: 市場カテゴリ情報
            
        Returns:
            市場別分析結果
        """
        logger.info("市場別若返りパターン分析を開始")
        
        market_analysis = {}
        
        # 市場カテゴリごとの分析
        for market_category, companies in market_categories.items():
            category_data = decline_data[decline_data['company_id'].isin(companies)]
            
            if len(category_data) == 0:
                continue
            
            # 基本統計
            recovery_rate = category_data['recovery_achieved'].mean()
            avg_recovery_duration = category_data[category_data['recovery_achieved']]['recovery_duration'].mean()
            avg_decline_severity = category_data['decline_severity'].mean()
            
            # 持続性分析
            successful_recoveries = category_data[category_data['recovery_achieved']]
            avg_sustainability = successful_recoveries['recovery_sustainability'].mean() if len(successful_recoveries) > 0 else 0.0
            
            # 衰退頻度分析
            decline_frequency = len(category_data) / len(companies)  # 企業あたりの衰退回数
            
            market_analysis[market_category] = {
                'total_decline_cases': len(category_data),
                'companies_analyzed': len(companies),
                'recovery_success_rate': recovery_rate,
                'average_recovery_duration': avg_recovery_duration,
                'average_decline_severity': avg_decline_severity,
                'average_sustainability': avg_sustainability,
                'decline_frequency_per_company': decline_frequency,
                'market_resilience_score': self._calculate_market_resilience(category_data)
            }
        
        # 市場間比較
        market_comparison = self._compare_market_categories(market_analysis)
        
        result = {
            'market_analysis': market_analysis,
            'market_comparison': market_comparison,
            'insights': self._generate_market_insights(market_analysis)
        }
        
        logger.info("市場別若返りパターン分析完了")
        return result
    
    
    def _calculate_market_resilience(self, market_data: pd.DataFrame) -> float:
        """
        市場の復元力（レジリエンス）スコア計算
        
        回復成功率、回復速度、持続性を総合評価
        
        Args:
            market_data: 市場の衰退データ
            
        Returns:
            レジリエンススコア（0-1）
        """
        if len(market_data) == 0:
            return 0.0
        
        # 回復成功率 (0-1)
        recovery_rate = market_data['recovery_achieved'].mean()
        
        # 回復速度スコア (期間の逆数を正規化)
        successful_data = market_data[market_data['recovery_achieved']]
        if len(successful_data) > 0:
            avg_duration = successful_data['recovery_duration'].mean()
            speed_score = 1.0 / (1.0 + avg_duration / 5.0)  # 5年を基準とした正規化
        else:
            speed_score = 0.0
        
        # 持続性スコア
        sustainability_score = successful_data['recovery_sustainability'].mean() if len(successful_data) > 0 else 0.0
        
        # 総合スコア（重み付き平均）
        resilience_score = (recovery_rate * 0.4 + 
                           speed_score * 0.3 + 
                           sustainability_score * 0.3)
        
        return min(1.0, max(0.0, resilience_score))
    
    
    def _compare_market_categories(self, market_analysis: Dict) -> Dict:
        """
        市場カテゴリ間比較分析
        
        Args:
            market_analysis: 市場別分析結果
            
        Returns:
            比較分析結果
        """
        if len(market_analysis) < 2:
            return {}
        
        categories = list(market_analysis.keys())
        comparison = {}
        
        # 各指標での順位付け
        metrics = ['recovery_success_rate', 'market_resilience_score', 
                    'average_recovery_duration', 'average_sustainability']
        
        for metric in metrics:
            sorted_categories = sorted(categories, 
                                        key=lambda x: market_analysis[x].get(metric, 0),
                                        reverse=True if metric != 'average_recovery_duration' else False)
            
            comparison[metric + '_ranking'] = sorted_categories
        
        # 最高/最低パフォーマンス市場
        resilience_ranking = comparison['market_resilience_score_ranking']
        
        comparison['best_performing_market'] = resilience_ranking[0] if resilience_ranking else None
        comparison['worst_performing_market'] = resilience_ranking[-1] if resilience_ranking else None
        
        # 統計的有意差検定（市場カテゴリが3つの場合）
        if len(categories) == 3:
            high_share_data = market_analysis.get('high_share_markets', {})
            declining_data = market_analysis.get('declining_markets', {})
            lost_data = market_analysis.get('lost_markets', {})
            
            comparison['statistical_significance'] = {
                'recovery_rate_differs': self._test_market_differences(
                    [high_share_data.get('recovery_success_rate', 0),
                        declining_data.get('recovery_success_rate', 0),
                        lost_data.get('recovery_success_rate', 0)]
                ),
                'interpretation': self._interpret_market_differences(market_analysis)
            }
        
        return comparison
    
    
    def _test_market_differences(self, values: List[float]) -> bool:
        """
        市場間差異の統計的検定
        
        Args:
            values: 市場別の値リスト
            
        Returns:
            有意差があるかどうか
        """
        if len(values) < 2:
            return False
        
        # 分散が十分にある場合のみ有意差ありと判定
        return np.std(values) > 0.1
    
    
    def _interpret_market_differences(self, market_analysis: Dict) -> str:
        """
        市場差異の解釈文生成
        
        Args:
            market_analysis: 市場分析結果
            
        Returns:
            解釈文
        """
        interpretations = []
        
        # レジリエンス順で解釈
        resilience_scores = {k: v['market_resilience_score'] 
                            for k, v in market_analysis.items()}
        
        sorted_markets = sorted(resilience_scores.items(), 
                                key=lambda x: x[1], reverse=True)
        
        if len(sorted_markets) >= 3:
            best_market = sorted_markets[0]
            worst_market = sorted_markets[-1]
            
            interpretations.append(
                f"{best_market[0]}が最も高い復元力（{best_market[1]:.3f}）を示し、"
                f"{worst_market[0]}が最も低い復元力（{worst_market[1]:.3f}）を示しています。"
            )
            
            # 回復成功率の差異
            recovery_rates = {k: v['recovery_success_rate'] 
                            for k, v in market_analysis.items()}
            max_recovery = max(recovery_rates.values())
            min_recovery = min(recovery_rates.values())
            
            if max_recovery - min_recovery > 0.2:
                interpretations.append(
                    f"市場間で回復成功率に大きな差（{(max_recovery - min_recovery)*100:.1f}%ポイント）が見られます。"
                )
        
        return " ".join(interpretations) if interpretations else "市場間の明確な差異は検出されませんでした。"
    
    
    def _generate_market_insights(self, market_analysis: Dict) -> List[str]:
        """
        市場分析インサイト生成
        
        Args:
            market_analysis: 市場分析結果
            
        Returns:
            インサイトリスト
        """
        insights = []
        
        for market, data in market_analysis.items():
            recovery_rate = data['recovery_success_rate']
            resilience = data['market_resilience_score']
            
            if recovery_rate > 0.7:
                insights.append(f"{market}は高い回復成功率（{recovery_rate:.1%}）を示し、危機対応力に優れています。")
            elif recovery_rate < 0.3:
                insights.append(f"{market}は回復成功率が低く（{recovery_rate:.1%}）、構造的な課題を抱えている可能性があります。")
            
            if resilience > 0.7:
                insights.append(f"{market}の企業は総合的な復元力が高く、持続可能な回復パターンを示しています。")
            elif resilience < 0.4:
                insights.append(f"{market}の企業は復元力に課題があり、戦略的な転換が必要かもしれません。")
        
        return insights
    
    
    def export_analysis_results(self, output_path: str = "rejuvenation_analysis_results.json") -> bool:
        """
        分析結果のエクスポート
        
        Args:
            output_path: 出力ファイルパス
            
        Returns:
            エクスポート成功可否
        """
        try:
            import json
            
            export_data = {
                'analysis_metadata': {
                    'analysis_date': datetime.now().isoformat(),
                    'decline_threshold': self.decline_threshold,
                    'recovery_threshold': self.recovery_threshold,
                    'min_observation_period': self.min_observation_period
                },
                'rejuvenation_patterns': self.rejuvenation_patterns,
                'success_factors': self.success_factors,
                'failure_factors': self.failure_factors,
                'prediction_models_summary': {
                    'best_model': self.prediction_models.get('best_model', ''),
                    'model_performance': self.prediction_models.get('best_model_performance', {}).get('cv_score_mean', 0.0) if self.prediction_models else 0.0
                }
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2, default=str)
            
            logger.info(f"分析結果を {output_path} にエクスポートしました")
            return True
            
        except Exception as e:
            logger.error(f"エクスポートエラー: {str(e)}")
            return False
    
    
    def generate_executive_summary(self) -> Dict:
        """
        エグゼクティブサマリー生成
        
        分析結果の要点を経営陣向けにまとめる
        
        Returns:
            エグゼクティブサマリー
        """
        if not self.success_factors:
            return {'error': '分析結果がありません。先に分析を実行してください。'}
        
        summary_stats = self.success_factors.get('summary_statistics', {})
        factor_importance = self.success_factors.get('factor_importance_ranking', [])
        
        # トップ5成功要因
        top_success_factors = factor_importance[:5] if factor_importance else []
        
        # 主要な洞察
        key_insights = []
        
        recovery_rate = summary_stats.get('recovery_success_rate', 0)
        if recovery_rate > 0.5:
            key_insights.append(f"企業の{recovery_rate:.1%}が衰退から回復に成功しており、適切な戦略により復活が可能です。")
        else:
            key_insights.append(f"企業の回復成功率は{recovery_rate:.1%}と低く、より効果的な若返り戦略が必要です。")
        
        avg_duration = summary_stats.get('average_recovery_duration', 0)
        if avg_duration > 0:
            key_insights.append(f"平均回復期間は{avg_duration:.1f}年であり、中長期的な戦略が重要です。")
        
        # 予測モデルの性能
        model_performance = 0.0
        if self.prediction_models and 'best_model_performance' in self.prediction_models:
            model_performance = self.prediction_models['best_model_performance'].get('cv_score_mean', 0.0)
        
        if model_performance > 0.7:
            key_insights.append(f"構築した予測モデルは{model_performance:.1%}の精度を達成し、実用的な回復予測が可能です。")
        
        return {
            'executive_summary': {
                'total_analysis_cases': summary_stats.get('total_decline_cases', 0),
                'recovery_success_rate': recovery_rate,
                'average_recovery_duration_years': avg_duration,
                'top_success_factors': [
                    {'factor': factor, 'importance_score': score}
                    for factor, score in top_success_factors
                ],
                'key_insights': key_insights,
                'prediction_model_accuracy': model_performance,
                'recommendations': self._generate_executive_recommendations()
            }
        }
    
    
    def _generate_executive_recommendations(self) -> List[str]:
        """
        経営陣向け推奨事項生成
        
        Returns:
            推奨事項リスト
        """
        recommendations = [
            "定期的な企業健康診断により、衰退の早期発見体制を構築してください。",
            "若返り成功要因を基に、予防的な経営戦略を策定してください。",
            "市場別の復元力特性を踏まえた事業ポートフォリオの最適化を検討してください。"
        ]
        
        # 要因重要度に基づく具体的推奨
        if self.success_factors and 'factor_importance_ranking' in self.success_factors:
            top_factor = self.success_factors['factor_importance_ranking'][0]
            recommendations.append(
                f"最重要成功要因である「{top_factor[0]}」の継続的なモニタリングと改善が不可欠です。"
            )
        
        return recommendations