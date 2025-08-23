"""
A2AI - Advanced Financial Analysis AI
生存バイアス補正モジュール (survivorship_bias_corrector.py)

このモジュールは、財務諸表分析における生存バイアスを統計的に補正する機能を提供します。
企業の消滅・新設・継続存続パターンに基づいて、分析結果の偏りを除去します。

主な機能:
1. 生存バイアスの検出と定量化
2. 消滅企業データによる補正
3. 新設企業データの適切な重み付け
4. 時系列データの生存バイアス補正
5. 市場カテゴリー別バイアス調整

対象企業パターン:
- 継続存続企業: 約80社 (1984-2024年の40年完全データ)
- 企業消滅企業: 約35社 (消滅までの期間データ)
- 新設企業: 約35社 (設立以降のデータ)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import warnings
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
import logging
from dataclasses import dataclass
from enum import Enum

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CompanyStatus(Enum):
    """企業ステータス定義"""
    SURVIVING = "surviving"          # 継続存続
    EXTINCT = "extinct"             # 消滅・倒産
    EMERGING = "emerging"           # 新設
    ACQUIRED = "acquired"           # 買収・統合
    SPUNOFF = "spunoff"            # 分社化


class MarketCategory(Enum):
    """市場カテゴリー定義"""
    HIGH_SHARE = "high_share"       # 世界シェア高市場
    DECLINING_SHARE = "declining"   # シェア低下市場
    LOST_SHARE = "lost_share"      # シェア失失市場


@dataclass
class BiasCorrection:
    """バイアス補正結果を格納するデータクラス"""
    correction_factor: float
    confidence_interval: Tuple[float, float]
    statistical_significance: float
    correction_method: str
    sample_size_before: int
    sample_size_after: int
    bias_magnitude: float


class SurvivorshipBiasCorrector:
    """
    生存バイアス補正クラス
    
    財務諸表分析における生存バイアスを検出・補正し、
    より正確な要因項目と評価項目の関係分析を可能にします。
    """
    
    def __init__(self, 
                    correction_methods: List[str] = None,
                    confidence_level: float = 0.95,
                    min_sample_size: int = 10):
        """
        初期化
        
        Args:
            correction_methods: 補正手法のリスト
            confidence_level: 信頼水準
            min_sample_size: 最小サンプルサイズ
        """
        self.correction_methods = correction_methods or [
            'heckman', 'ipw', 'bootstrap', 'propensity_matching'
        ]
        self.confidence_level = confidence_level
        self.min_sample_size = min_sample_size
        self.scaler = StandardScaler()
        self.correction_results = {}
        
    def detect_survivorship_bias(self, 
                                data: pd.DataFrame,
                                company_status_col: str = 'company_status',
                                market_category_col: str = 'market_category') -> Dict[str, Any]:
        """
        生存バイアスの検出と定量化
        
        Args:
            data: 企業財務データ
            company_status_col: 企業ステータス列名
            market_category_col: 市場カテゴリー列名
            
        Returns:
            バイアス検出結果の辞書
        """
        logger.info("生存バイアス検出を開始")
        
        # ステータス別企業数の集計
        status_counts = data[company_status_col].value_counts()
        market_counts = data.groupby([market_category_col, company_status_col]).size().unstack(fill_value=0)
        
        # バイアス指標の計算
        surviving_ratio = status_counts.get(CompanyStatus.SURVIVING.value, 0) / len(data)
        extinct_ratio = status_counts.get(CompanyStatus.EXTINCT.value, 0) / len(data)
        emerging_ratio = status_counts.get(CompanyStatus.EMERGING.value, 0) / len(data)
        
        # 市場別バイアス分析
        market_bias_analysis = {}
        for market in MarketCategory:
            market_data = data[data[market_category_col] == market.value]
            if len(market_data) > 0:
                market_surviving_ratio = len(market_data[
                    market_data[company_status_col] == CompanyStatus.SURVIVING.value
                ]) / len(market_data)
                market_bias_analysis[market.value] = {
                    'surviving_ratio': market_surviving_ratio,
                    'sample_size': len(market_data),
                    'bias_severity': abs(market_surviving_ratio - surviving_ratio)
                }
        
        # 統計的有意性テスト
        chi2_stat, chi2_pvalue = self._chi_square_test(data, company_status_col, market_category_col)
        
        bias_detection_result = {
            'overall_bias': {
                'surviving_ratio': surviving_ratio,
                'extinct_ratio': extinct_ratio,
                'emerging_ratio': emerging_ratio,
                'bias_magnitude': 1 - surviving_ratio  # 生存企業以外の割合をバイアス指標とする
            },
            'market_specific_bias': market_bias_analysis,
            'statistical_tests': {
                'chi_square_stat': chi2_stat,
                'chi_square_pvalue': chi2_pvalue,
                'bias_significant': chi2_pvalue < (1 - self.confidence_level)
            },
            'sample_composition': {
                'total_companies': len(data),
                'status_distribution': status_counts.to_dict(),
                'market_distribution': market_counts.to_dict()
            }
        }
        
        logger.info(f"バイアス検出完了: 生存率={surviving_ratio:.3f}, バイアス強度={bias_detection_result['overall_bias']['bias_magnitude']:.3f}")
        
        return bias_detection_result
    
    def correct_heckman_selection(self, 
                                    data: pd.DataFrame, 
                                    outcome_vars: List[str],
                                    selection_vars: List[str],
                                    company_status_col: str = 'company_status') -> pd.DataFrame:
        """
        Heckman選択モデルによる生存バイアス補正
        
        Args:
            data: 企業財務データ
            outcome_vars: 結果変数（評価項目）のリスト
            selection_vars: 選択変数（要因項目）のリスト
            company_status_col: 企業ステータス列名
            
        Returns:
            バイアス補正済みデータ
        """
        logger.info("Heckman選択モデルによる補正を開始")
        
        corrected_data = data.copy()
        
        # 生存確率の推定（第1段階）
        survival_indicator = (data[company_status_col] == CompanyStatus.SURVIVING.value).astype(int)
        
        # 選択方程式用の特徴量準備
        X_selection = data[selection_vars].fillna(data[selection_vars].median())
        X_selection_scaled = self.scaler.fit_transform(X_selection)
        
        # プロビット回帰（生存確率推定）
        probit_model = LogisticRegression(random_state=42)
        probit_model.fit(X_selection_scaled, survival_indicator)
        
        # 逆ミルズ比の計算
        survival_prob = probit_model.predict_proba(X_selection_scaled)[:, 1]
        # 数値安定性のために極値をクリップ
        survival_prob = np.clip(survival_prob, 0.001, 0.999)
        
        # 逆ミルズ比 = φ(Φ^(-1)(p)) / p, ここでφは標準正規分布の密度関数、Φは累積分布関数
        inverse_mills_ratio = stats.norm.pdf(stats.norm.ppf(survival_prob)) / survival_prob
        
        # 結果変数の補正（第2段階）
        for outcome_var in outcome_vars:
            if outcome_var in data.columns:
                # 生存企業のみのデータで回帰
                surviving_mask = survival_indicator == 1
                surviving_data = data[surviving_mask]
                
                if len(surviving_data) > self.min_sample_size:
                    y = surviving_data[outcome_var].fillna(surviving_data[outcome_var].median())
                    X_outcome = surviving_data[selection_vars].fillna(surviving_data[selection_vars].median())
                    
                    # 逆ミルズ比を説明変数に追加
                    X_outcome_with_mills = np.column_stack([
                        self.scaler.fit_transform(X_outcome), 
                        inverse_mills_ratio[surviving_mask]
                    ])
                    
                    # OLS回帰
                    ols_model = LinearRegression()
                    ols_model.fit(X_outcome_with_mills, y)
                    
                    # 全サンプルに対する予測（バイアス補正済み）
                    X_all_with_mills = np.column_stack([
                        self.scaler.transform(X_selection), 
                        inverse_mills_ratio
                    ])
                    
                    corrected_values = ols_model.predict(X_all_with_mills)
                    corrected_data[f'{outcome_var}_heckman_corrected'] = corrected_values
                    
                    # 補正結果の記録
                    self.correction_results[f'{outcome_var}_heckman'] = BiasCorrection(
                        correction_factor=np.mean(corrected_values / data[outcome_var].fillna(data[outcome_var].median())),
                        confidence_interval=(np.percentile(corrected_values, 2.5), np.percentile(corrected_values, 97.5)),
                        statistical_significance=0.05,  # 仮の値、実際にはt検定等で計算
                        correction_method='heckman',
                        sample_size_before=len(surviving_data),
                        sample_size_after=len(data),
                        bias_magnitude=np.std(corrected_values - data[outcome_var].fillna(data[outcome_var].median()))
                    )
        
        logger.info("Heckman補正完了")
        return corrected_data
    
    def correct_inverse_probability_weighting(self, 
                                                data: pd.DataFrame, 
                                                outcome_vars: List[str],
                                                weight_vars: List[str],
                                                company_status_col: str = 'company_status') -> pd.DataFrame:
        """
        逆確率重み付け（IPW）による生存バイアス補正
        
        Args:
            data: 企業財務データ
            outcome_vars: 結果変数のリスト
            weight_vars: 重み計算用変数のリスト
            company_status_col: 企業ステータス列名
            
        Returns:
            重み付き補正済みデータ
        """
        logger.info("IPW補正を開始")
        
        corrected_data = data.copy()
        
        # 生存確率の推定
        survival_indicator = (data[company_status_col] == CompanyStatus.SURVIVING.value).astype(int)
        
        X_weight = data[weight_vars].fillna(data[weight_vars].median())
        X_weight_scaled = self.scaler.fit_transform(X_weight)
        
        # ランダムフォレストで生存確率を推定（より柔軟なモデル）
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_weight_scaled, survival_indicator)
        
        survival_prob = rf_model.predict(X_weight_scaled)
        survival_prob = np.clip(survival_prob, 0.01, 0.99)  # 極値をクリップ
        
        # IPW重みの計算
        ipw_weights = np.where(survival_indicator == 1, 1/survival_prob, 1/(1-survival_prob))
        
        # 重みの正規化
        ipw_weights = ipw_weights / np.mean(ipw_weights)
        
        # 結果変数の重み付き平均計算
        for outcome_var in outcome_vars:
            if outcome_var in data.columns:
                outcome_values = data[outcome_var].fillna(data[outcome_var].median())
                
                # 重み付き統計量の計算
                weighted_mean = np.average(outcome_values, weights=ipw_weights)
                weighted_std = np.sqrt(np.average((outcome_values - weighted_mean)**2, weights=ipw_weights))
                
                # 補正済み値の生成（重み付きz-score標準化後の値）
                corrected_values = (outcome_values - np.mean(outcome_values)) / np.std(outcome_values) * weighted_std + weighted_mean
                corrected_data[f'{outcome_var}_ipw_corrected'] = corrected_values
                corrected_data[f'{outcome_var}_ipw_weights'] = ipw_weights
                
                # 補正結果の記録
                self.correction_results[f'{outcome_var}_ipw'] = BiasCorrection(
                    correction_factor=weighted_mean / np.mean(outcome_values),
                    confidence_interval=(weighted_mean - 1.96*weighted_std, weighted_mean + 1.96*weighted_std),
                    statistical_significance=0.05,
                    correction_method='ipw',
                    sample_size_before=len(data),
                    sample_size_after=len(data),
                    bias_magnitude=abs(weighted_mean - np.mean(outcome_values))
                )
        
        logger.info("IPW補正完了")
        return corrected_data
    
    def correct_propensity_score_matching(self, 
                                            data: pd.DataFrame, 
                                            outcome_vars: List[str],
                                            matching_vars: List[str],
                                            company_status_col: str = 'company_status') -> pd.DataFrame:
        """
        傾向スコアマッチングによる生存バイアス補正
        
        Args:
            data: 企業財務データ
            outcome_vars: 結果変数のリスト
            matching_vars: マッチング用変数のリスト
            company_status_col: 企業ステータス列名
            
        Returns:
            マッチング補正済みデータ
        """
        logger.info("傾向スコアマッチング補正を開始")
        
        corrected_data = data.copy()
        
        # 傾向スコアの推定
        treatment = (data[company_status_col] == CompanyStatus.SURVIVING.value).astype(int)
        
        X_match = data[matching_vars].fillna(data[matching_vars].median())
        X_match_scaled = self.scaler.fit_transform(X_match)
        
        propensity_model = LogisticRegression(random_state=42)
        propensity_model.fit(X_match_scaled, treatment)
        
        propensity_scores = propensity_model.predict_proba(X_match_scaled)[:, 1]
        
        # 1:1最近傍マッチング
        matched_pairs = self._nearest_neighbor_matching(propensity_scores, treatment)
        
        # マッチング後のデータで結果変数を補正
        for outcome_var in outcome_vars:
            if outcome_var in data.columns:
                outcome_values = data[outcome_var].fillna(data[outcome_var].median())
                
                # マッチングペアでの平均治療効果（ATE）推定
                matched_effects = []
                for treated_idx, control_idx in matched_pairs:
                    if treated_idx < len(outcome_values) and control_idx < len(outcome_values):
                        effect = outcome_values.iloc[treated_idx] - outcome_values.iloc[control_idx]
                        matched_effects.append(effect)
                
                if matched_effects:
                    ate = np.mean(matched_effects)
                    ate_std = np.std(matched_effects)
                    
                    # 補正済み値の計算（非治療群に平均治療効果を加算）
                    corrected_values = outcome_values.copy()
                    control_mask = treatment == 0
                    corrected_values[control_mask] += ate
                    
                    corrected_data[f'{outcome_var}_psm_corrected'] = corrected_values
                    
                    # 補正結果の記録
                    self.correction_results[f'{outcome_var}_psm'] = BiasCorrection(
                        correction_factor=1 + ate/np.mean(outcome_values),
                        confidence_interval=(ate - 1.96*ate_std, ate + 1.96*ate_std),
                        statistical_significance=0.05,
                        correction_method='propensity_matching',
                        sample_size_before=len(data),
                        sample_size_after=len(matched_pairs),
                        bias_magnitude=abs(ate)
                    )
        
        logger.info("傾向スコアマッチング補正完了")
        return corrected_data
    
    def correct_bootstrap_resampling(self, 
                                    data: pd.DataFrame, 
                                    outcome_vars: List[str],
                                    company_status_col: str = 'company_status',
                                    n_bootstrap: int = 1000) -> pd.DataFrame:
        """
        ブートストラップリサンプリングによる生存バイアス補正
        
        Args:
            data: 企業財務データ
            outcome_vars: 結果変数のリスト
            company_status_col: 企業ステータス列名
            n_bootstrap: ブートストラップ回数
            
        Returns:
            ブートストラップ補正済みデータ
        """
        logger.info("ブートストラップ補正を開始")
        
        corrected_data = data.copy()
        
        # 企業ステータス別データ分離
        surviving_data = data[data[company_status_col] == CompanyStatus.SURVIVING.value]
        extinct_data = data[data[company_status_col] == CompanyStatus.EXTINCT.value]
        emerging_data = data[data[company_status_col] == CompanyStatus.EMERGING.value]
        
        # 各ステータスの真の分布推定
        for outcome_var in outcome_vars:
            if outcome_var in data.columns:
                bootstrap_estimates = []
                
                for _ in range(n_bootstrap):
                    # 層別ブートストラップサンプリング
                    if len(surviving_data) > 0:
                        surviving_sample = surviving_data.sample(
                            n=min(len(surviving_data), len(data)//3), 
                            replace=True, 
                            random_state=np.random.randint(0, 10000)
                        )[outcome_var].fillna(surviving_data[outcome_var].median())
                    else:
                        surviving_sample = pd.Series([])
                    
                    if len(extinct_data) > 0:
                        extinct_sample = extinct_data.sample(
                            n=min(len(extinct_data), len(data)//3), 
                            replace=True, 
                            random_state=np.random.randint(0, 10000)
                        )[outcome_var].fillna(extinct_data[outcome_var].median())
                    else:
                        extinct_sample = pd.Series([])
                    
                    if len(emerging_data) > 0:
                        emerging_sample = emerging_data.sample(
                            n=min(len(emerging_data), len(data)//3), 
                            replace=True, 
                            random_state=np.random.randint(0, 10000)
                        )[outcome_var].fillna(emerging_data[outcome_var].median())
                    else:
                        emerging_sample = pd.Series([])
                    
                    # 総合サンプルの統計量計算
                    combined_sample = pd.concat([surviving_sample, extinct_sample, emerging_sample], ignore_index=True)
                    if len(combined_sample) > 0:
                        bootstrap_estimates.append(combined_sample.mean())
                
                if bootstrap_estimates:
                    # ブートストラップ信頼区間
                    bootstrap_mean = np.mean(bootstrap_estimates)
                    bootstrap_std = np.std(bootstrap_estimates)
                    bootstrap_ci_lower = np.percentile(bootstrap_estimates, 2.5)
                    bootstrap_ci_upper = np.percentile(bootstrap_estimates, 97.5)
                    
                    # 補正済み値の生成
                    original_mean = data[outcome_var].fillna(data[outcome_var].median()).mean()
                    correction_factor = bootstrap_mean / original_mean if original_mean != 0 else 1
                    
                    corrected_values = data[outcome_var].fillna(data[outcome_var].median()) * correction_factor
                    corrected_data[f'{outcome_var}_bootstrap_corrected'] = corrected_values
                    
                    # 補正結果の記録
                    self.correction_results[f'{outcome_var}_bootstrap'] = BiasCorrection(
                        correction_factor=correction_factor,
                        confidence_interval=(bootstrap_ci_lower, bootstrap_ci_upper),
                        statistical_significance=0.05,
                        correction_method='bootstrap',
                        sample_size_before=len(data),
                        sample_size_after=len(data),
                        bias_magnitude=abs(bootstrap_mean - original_mean)
                    )
        
        logger.info("ブートストラップ補正完了")
        return corrected_data
    
    def apply_comprehensive_correction(self, 
                                        data: pd.DataFrame, 
                                        outcome_vars: List[str],
                                        feature_vars: List[str],
                                        company_status_col: str = 'company_status',
                                        market_category_col: str = 'market_category') -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        包括的な生存バイアス補正の実行
        
        Args:
            data: 企業財務データ
            outcome_vars: 結果変数のリスト
            feature_vars: 特徴量変数のリスト
            company_status_col: 企業ステータス列名
            market_category_col: 市場カテゴリー列名
            
        Returns:
            補正済みデータと補正結果のタプル
        """
        logger.info("包括的な生存バイアス補正を開始")
        
        # 1. バイアス検出
        bias_detection = self.detect_survivorship_bias(data, company_status_col, market_category_col)
        
        # 2. 各手法による補正
        corrected_data = data.copy()
        
        if 'heckman' in self.correction_methods:
            corrected_data = self.correct_heckman_selection(corrected_data, outcome_vars, feature_vars, company_status_col)
        
        if 'ipw' in self.correction_methods:
            corrected_data = self.correct_inverse_probability_weighting(corrected_data, outcome_vars, feature_vars, company_status_col)
        
        if 'propensity_matching' in self.correction_methods:
            corrected_data = self.correct_propensity_score_matching(corrected_data, outcome_vars, feature_vars, company_status_col)
        
        if 'bootstrap' in self.correction_methods:
            corrected_data = self.correct_bootstrap_resampling(corrected_data, outcome_vars, company_status_col)
        
        # 3. 補正結果の統合
        comprehensive_results = {
            'bias_detection': bias_detection,
            'correction_methods_used': self.correction_methods,
            'individual_corrections': self.correction_results,
            'summary_statistics': self._calculate_correction_summary(corrected_data, outcome_vars)
        }
        
        logger.info("包括的な生存バイアス補正完了")
        return corrected_data, comprehensive_results
    
    def _chi_square_test(self, data: pd.DataFrame, status_col: str, category_col: str) -> Tuple[float, float]:
        """カイ二乗検定による独立性の検定"""
        contingency_table = pd.crosstab(data[status_col], data[category_col])
        chi2, pvalue, _, _ = stats.chi2_contingency(contingency_table)
        return chi2, pvalue
    
    def _nearest_neighbor_matching(self, propensity_scores: np.ndarray, treatment: np.ndarray) -> List[Tuple[int, int]]:
        """最近傍傾向スコアマッチング"""
        treated_indices = np.where(treatment == 1)[0]
        control_indices = np.where(treatment == 0)[0]
        
        matches = []
        used_controls = set()
        
        for treated_idx in treated_indices:
            treated_score = propensity_scores[treated_idx]
            
            # 未使用の制御群から最も近い傾向スコアを持つ個体を探索
            best_match = None
            min_distance = float('inf')
            
            for control_idx in control_indices:
                if control_idx not in used_controls:
                    distance = abs(treated_score - propensity_scores[control_idx])
                    if distance < min_distance:
                        min_distance = distance
                        best_match = control_idx
            
            if best_match is not None:
                matches.append((treated_idx, best_match))
                used_controls.add(best_match)
        
        return matches
    
    def _calculate_correction_summary(self, corrected_data: pd.DataFrame, outcome_vars: List[str]) -> Dict[str, Any]:
        """補正結果のサマリー統計計算"""
        summary = {}
        
        for outcome_var in outcome_vars:
            var_summary = {}
            original_col = outcome_var
            
            # 各補正手法の結果を比較
            correction_cols = [col for col in corrected_data.columns if col.startswith(f'{outcome_var}_') and col.endswith('_corrected')]
            
            if original_col in corrected_data.columns:
                original_mean = corrected_data[original_col].mean()
                original_std = corrected_data[original_col].std()
                
                var_summary['original'] = {'mean': original_mean, 'std': original_std}
                
                for corr_col in correction_cols:
                    corr_mean = corrected_data[corr_col].mean()
                    corr_std = corrected_data[corr_col].std()
                    method_name = corr_col.split('_')[-2]  # 補正手法名を抽出
                    
                    var_summary[method_name] = {
                        'mean': corr_mean,
                        'std': corr_std,
                        'mean_change_ratio': (corr_mean - original_mean) / original_mean if original_mean != 0 else 0,
                        'std_change_ratio': (corr_std - original_std) / original_std if original_std != 0 else 0
                    }
            
            summary[outcome_var] = var_summary
        
        return summary
    
    def export_correction_report(self, 
                                correction_results: Dict[str, Any], 
                                output_path: str = None) -> str:
        """
        補正結果のレポート生成
        
        Args:
            correction_results: 補正結果辞書
            output_path: 出力パス
            
        Returns:
            生成されたレポート文字列
        """
        report_lines = [
            "# A2AI 生存バイアス補正レポート",
            "=" * 50,
            "",
            "## バイアス検出結果",
        ]
        
        # バイアス検出結果
        bias_detection = correction_results.get('bias_detection', {})
        overall_bias = bias_detection.get('overall_bias', {})
        
        report_lines.extend([
            f"- 生存企業比率: {overall_bias.get('surviving_ratio', 0):.3f}",
            f"- 消滅企業比率: {overall_bias.get('extinct_ratio', 0):.3f}",
            f"- 新設企業比率: {overall_bias.get('emerging_ratio', 0):.3f}",
            f"- バイアス強度: {overall_bias.get('bias_magnitude', 0):.3f}",
            ""
        ])
        
        # 市場別バイアス分析
        market_bias = bias_detection.get('market_specific_bias', {})
        if market_bias:
            report_lines.append("### 市場別バイアス分析")
            for market, analysis in market_bias.items():
                report_lines.extend([
                    f"**{market}市場:**",
                    f"  - 生存率: {analysis.get('surviving_ratio', 0):.3f}",
                    f"  - サンプル数: {analysis.get('sample_size', 0)}",
                    f"  - バイアス深刻度: {analysis.get('bias_severity', 0):.3f}",
                    ""
                ])
        
        # 統計的検定結果
        stat_tests = bias_detection.get('statistical_tests', {})
        report_lines.extend([
            "### 統計的検定結果",
            f"- カイ二乗統計量: {stat_tests.get('chi_square_stat', 0):.3f}",
            f"- p値: {stat_tests.get('chi_square_pvalue', 0):.6f}",
            f"- バイアス有意性: {'有意' if stat_tests.get('bias_significant', False) else '非有意'}",
            "",
            "## 補正手法別結果",
            ""
        ])
        
        # 個別補正結果
        individual_corrections = correction_results.get('individual_corrections', {})
        for var_method, correction in individual_corrections.items():
            if isinstance(correction, BiasCorrection):
                report_lines.extend([
                    f"### {var_method}",
                    f"- 補正係数: {correction.correction_factor:.4f}",
                    f"- 信頼区間: ({correction.confidence_interval[0]:.4f}, {correction.confidence_interval[1]:.4f})",
                    f"- 統計的有意性: {correction.statistical_significance:.4f}",
                    f"- 補正手法: {correction.correction_method}",
                    f"- 補正前サンプル数: {correction.sample_size_before}",
                    f"- 補正後サンプル数: {correction.sample_size_after}",
                    f"- バイアス強度: {correction.bias_magnitude:.4f}",
                    ""
                ])
        
        # サマリー統計
        summary_stats = correction_results.get('summary_statistics', {})
        if summary_stats:
            report_lines.extend([
                "## 補正結果サマリー統計",
                ""
            ])
            
            for outcome_var, var_summary in summary_stats.items():
                report_lines.append(f"### {outcome_var}")
                
                original_stats = var_summary.get('original', {})
                if original_stats:
                    report_lines.extend([
                        f"**元データ:**",
                        f"  - 平均: {original_stats.get('mean', 0):.4f}",
                        f"  - 標準偏差: {original_stats.get('std', 0):.4f}",
                        ""
                    ])
                
                for method, method_stats in var_summary.items():
                    if method != 'original':
                        report_lines.extend([
                            f"**{method}補正後:**",
                            f"  - 平均: {method_stats.get('mean', 0):.4f}",
                            f"  - 標準偏差: {method_stats.get('std', 0):.4f}",
                            f"  - 平均変化率: {method_stats.get('mean_change_ratio', 0):.4f}",
                            f"  - 標準偏差変化率: {method_stats.get('std_change_ratio', 0):.4f}",
                            ""
                        ])
        
        # 推奨事項
        report_lines.extend([
            "## 推奨事項と注意点",
            "",
            "### 補正手法選択ガイド",
            "- **Heckman補正**: 選択バイアスが主要因の場合に推奨",
            "- **IPW補正**: サンプル数が十分で、重み付けが適切な場合",
            "- **傾向スコアマッチング**: 因果推論において対照群設定が重要な場合",
            "- **ブートストラップ**: サンプル数が少ない場合や分布の不確実性が高い場合",
            "",
            "### 注意点",
            "- 補正後も残存する可能性のあるバイアスを考慮してください",
            "- 複数の補正手法の結果を比較検討することを推奨します",
            "- 企業の消滅・新設パターンが分析結果に与える影響を継続的に監視してください",
            "",
            f"レポート生成日時: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        ])
        
        report_text = "\n".join(report_lines)
        
        # ファイル出力
        if output_path:
            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(report_text)
                logger.info(f"補正レポートを {output_path} に出力しました")
            except Exception as e:
                logger.error(f"レポート出力エラー: {e}")
        
        return report_text
    
    def validate_correction_effectiveness(self, 
                                            original_data: pd.DataFrame,
                                            corrected_data: pd.DataFrame,
                                            outcome_vars: List[str],
                                            company_status_col: str = 'company_status') -> Dict[str, Any]:
        """
        補正効果の妥当性検証
        
        Args:
            original_data: 元データ
            corrected_data: 補正後データ
            outcome_vars: 結果変数のリスト
            company_status_col: 企業ステータス列名
            
        Returns:
            検証結果辞書
        """
        logger.info("補正効果の妥当性検証を開始")
        
        validation_results = {}
        
        # 各結果変数について検証
        for outcome_var in outcome_vars:
            var_validation = {}
            
            # 補正手法別の検証
            correction_cols = [col for col in corrected_data.columns 
                                if col.startswith(f'{outcome_var}_') and col.endswith('_corrected')]
            
            for corr_col in correction_cols:
                method_name = corr_col.split('_')[-2]
                
                # 1. 分布の正規性検定
                _, normality_pvalue = stats.normaltest(corrected_data[corr_col].dropna())
                
                # 2. 企業ステータス別の平均差検定
                surviving_data = corrected_data[
                    corrected_data[company_status_col] == CompanyStatus.SURVIVING.value
                ][corr_col].dropna()
                
                extinct_data = corrected_data[
                    corrected_data[company_status_col] == CompanyStatus.EXTINCT.value
                ][corr_col].dropna()
                
                if len(surviving_data) > 5 and len(extinct_data) > 5:
                    _, ttest_pvalue = stats.ttest_ind(surviving_data, extinct_data)
                else:
                    ttest_pvalue = np.nan
                
                # 3. 補正前後の相関
                original_values = original_data[outcome_var].dropna()
                corrected_values = corrected_data[corr_col].dropna()
                
                # 共通インデックスでの相関計算
                common_idx = original_values.index.intersection(corrected_values.index)
                if len(common_idx) > 10:
                    correlation, corr_pvalue = stats.pearsonr(
                        original_values[common_idx], 
                        corrected_values[common_idx]
                    )
                else:
                    correlation, corr_pvalue = np.nan, np.nan
                
                # 4. 分散の均一性検定
                if len(surviving_data) > 5 and len(extinct_data) > 5:
                    _, levene_pvalue = stats.levene(surviving_data, extinct_data)
                else:
                    levene_pvalue = np.nan
                
                var_validation[method_name] = {
                    'normality_test_pvalue': normality_pvalue,
                    'group_difference_pvalue': ttest_pvalue,
                    'original_correlation': correlation,
                    'original_correlation_pvalue': corr_pvalue,
                    'variance_homogeneity_pvalue': levene_pvalue,
                    'data_quality_score': self._calculate_data_quality_score(
                        normality_pvalue, ttest_pvalue, correlation, levene_pvalue
                    )
                }
            
            validation_results[outcome_var] = var_validation
        
        # 総合評価スコア計算
        overall_quality_score = self._calculate_overall_quality_score(validation_results)
        validation_results['overall_assessment'] = {
            'quality_score': overall_quality_score,
            'recommendation': self._generate_quality_recommendation(overall_quality_score),
            'validation_timestamp': pd.Timestamp.now().isoformat()
        }
        
        logger.info("補正効果の妥当性検証完了")
        return validation_results
    
    def _calculate_data_quality_score(self, 
                                        normality_p: float, 
                                        ttest_p: float, 
                                        correlation: float, 
                                        levene_p: float) -> float:
        """データ品質スコア計算"""
        score = 0.0
        
        # 正規性（0-25点）
        if not np.isnan(normality_p):
            score += 25 * min(1.0, normality_p / 0.05)
        
        # グループ間差（0-25点）
        if not np.isnan(ttest_p):
            # 有意差があることが望ましい場合
            score += 25 * (1 - min(1.0, ttest_p / 0.05))
        
        # 元データとの相関（0-25点）
        if not np.isnan(correlation):
            score += 25 * abs(correlation)
        
        # 分散均一性（0-25点）
        if not np.isnan(levene_p):
            score += 25 * min(1.0, levene_p / 0.05)
        
        return score
    
    def _calculate_overall_quality_score(self, validation_results: Dict[str, Any]) -> float:
        """総合品質スコア計算"""
        all_scores = []
        
        for outcome_var, var_validation in validation_results.items():
            if isinstance(var_validation, dict):
                for method_name, method_validation in var_validation.items():
                    if isinstance(method_validation, dict) and 'data_quality_score' in method_validation:
                        score = method_validation['data_quality_score']
                        if not np.isnan(score):
                            all_scores.append(score)
        
        return np.mean(all_scores) if all_scores else 0.0
    
    def _generate_quality_recommendation(self, quality_score: float) -> str:
        """品質スコアに基づく推奨事項生成"""
        if quality_score >= 80:
            return "優良: 補正結果は高品質です。分析に使用することを推奨します。"
        elif quality_score >= 60:
            return "良好: 補正結果は受け入れ可能な品質です。結果の解釈に注意してください。"
        elif quality_score >= 40:
            return "注意: 補正結果に問題がある可能性があります。追加の検証が必要です。"
        else:
            return "警告: 補正結果の品質が低いです。補正手法の見直しをお勧めします。"


# 使用例とテスト用コード
if __name__ == "__main__":
    # サンプルデータの生成（実際の使用では企業財務データを使用）
    np.random.seed(42)
    
    # 150社のサンプルデータ作成
    n_companies = 150
    
    # 企業ステータスの設定（添付企業リストに基づく）
    company_statuses = (
        [CompanyStatus.SURVIVING.value] * 80 +  # 継続存続企業
        [CompanyStatus.EXTINCT.value] * 35 +     # 消滅企業  
        [CompanyStatus.EMERGING.value] * 35      # 新設企業
    )
    
    # 市場カテゴリーの設定
    market_categories = (
        [MarketCategory.HIGH_SHARE.value] * 50 +      # 高シェア市場
        [MarketCategory.DECLINING_SHARE.value] * 50 + # 低下市場
        [MarketCategory.LOST_SHARE.value] * 50        # 失失市場
    )
    
    # 財務指標のサンプルデータ（生存バイアスを含む）
    sample_data = pd.DataFrame({
        'company_id': range(n_companies),
        'company_status': company_statuses,
        'market_category': market_categories,
        'sales_growth_rate': np.random.normal(0.05, 0.15, n_companies),  # 売上成長率
        'operating_margin': np.random.normal(0.08, 0.05, n_companies),   # 営業利益率
        'roe': np.random.normal(0.12, 0.08, n_companies),                # ROE
        'rd_intensity': np.random.normal(0.05, 0.03, n_companies),       # 研究開発集約度
        'debt_ratio': np.random.normal(0.3, 0.15, n_companies),          # 負債比率
        'employee_growth': np.random.normal(0.02, 0.1, n_companies),     # 従業員数成長率
        'capex_intensity': np.random.normal(0.04, 0.02, n_companies),    # 設備投資集約度
    })
    
    # 生存バイアスを人工的に導入（生存企業のパフォーマンスを向上させる）
    surviving_mask = sample_data['company_status'] == CompanyStatus.SURVIVING.value
    sample_data.loc[surviving_mask, 'sales_growth_rate'] += 0.03
    sample_data.loc[surviving_mask, 'operating_margin'] += 0.02
    sample_data.loc[surviving_mask, 'roe'] += 0.05
    
    # SurvivorshipBiasCorrectorの実行例
    print("A2AI 生存バイアス補正システム - 実行例")
    print("=" * 60)
    
    # 補正器の初期化
    bias_corrector = SurvivorshipBiasCorrector(
        correction_methods=['heckman', 'ipw', 'propensity_matching', 'bootstrap'],
        confidence_level=0.95,
        min_sample_size=10
    )
    
    # 結果変数と特徴量の定義
    outcome_variables = ['sales_growth_rate', 'operating_margin', 'roe']
    feature_variables = ['rd_intensity', 'debt_ratio', 'employee_growth', 'capex_intensity']
    
    # 包括的バイアス補正の実行
    corrected_data, correction_results = bias_corrector.apply_comprehensive_correction(
        data=sample_data,
        outcome_vars=outcome_variables,
        feature_vars=feature_variables,
        company_status_col='company_status',
        market_category_col='market_category'
    )
    
    # 補正効果の検証
    validation_results = bias_corrector.validate_correction_effectiveness(
        original_data=sample_data,
        corrected_data=corrected_data,
        outcome_vars=outcome_variables,
        company_status_col='company_status'
    )
    
    # レポート生成
    report = bias_corrector.export_correction_report(
        correction_results=correction_results,
        output_path='a2ai_survivorship_bias_report.md'
    )
    
    print("\n" + "="*60)
    print("補正結果サマリー:")
    print("="*60)
    
    # 補正前後の比較表示
    for outcome_var in outcome_variables:
        print(f"\n【{outcome_var}】")
        original_mean = sample_data[outcome_var].mean()
        print(f"  補正前平均: {original_mean:.4f}")
        
        correction_cols = [col for col in corrected_data.columns 
                            if col.startswith(f'{outcome_var}_') and col.endswith('_corrected')]
        
        for corr_col in correction_cols:
            method_name = corr_col.split('_')[-2]
            corrected_mean = corrected_data[corr_col].mean()
            change_ratio = (corrected_mean - original_mean) / original_mean * 100
            print(f"  {method_name}補正後: {corrected_mean:.4f} (変化率: {change_ratio:+.2f}%)")
    
    # 検証結果表示
    overall_quality = validation_results.get('overall_assessment', {})
    print(f"\n総合品質スコア: {overall_quality.get('quality_score', 0):.1f}/100")
    print(f"推奨事項: {overall_quality.get('recommendation', 'なし')}")
    
    print(f"\n詳細レポートが生成されました: a2ai_survivorship_bias_report.md")
    print("="*60)