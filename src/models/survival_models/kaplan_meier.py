"""
A2AI - Advanced Financial Analysis AI
Kaplan-Meier Survival Analysis Module

企業の生存確率を推定するKaplan-Meier推定量の実装
150社×40年の財務データから企業存続パターンを分析

Author: A2AI Development Team
Date: 2024
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
from dataclasses import dataclass
from datetime import datetime
import logging

# 日本語フォント設定
plt.rcParams['font.family'] = ['DejaVu Sans', 'Hiragino Sans', 'Yu Gothic', 'Meiryo', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']

@dataclass
class SurvivalData:
    """生存分析データクラス"""
    duration: np.ndarray  # 観測期間（企業存続年数）
    event: np.ndarray     # イベント発生フラグ（1: 消滅, 0: 打ち切り）
    company_id: np.ndarray  # 企業ID
    market_category: np.ndarray  # 市場カテゴリ（高シェア/低下/失失）
    
@dataclass
class KMEstimate:
    """Kaplan-Meier推定結果クラス"""
    time: np.ndarray          # 時点
    survival_prob: np.ndarray # 生存確率
    n_risk: np.ndarray        # リスク集合の企業数
    n_events: np.ndarray      # イベント発生企業数
    confidence_interval: Dict[str, np.ndarray]  # 信頼区間


class KaplanMeierEstimator:
    """
    Kaplan-Meier生存分析推定器
    
    企業の存続確率を推定し、市場カテゴリ別の生存パターンを分析
    """
    
    def __init__(self, confidence_level: float = 0.95):
        """
        初期化
        
        Args:
            confidence_level: 信頼区間の信頼水準
        """
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        self.estimates_ = {}
        self.survival_data_ = None
        self.fitted_ = False
        
        # ログ設定
        self.logger = logging.getLogger(__name__)
        
    def fit(self, survival_data: SurvivalData) -> 'KaplanMeierEstimator':
        """
        Kaplan-Meier推定を実行
        
        Args:
            survival_data: 生存分析データ
            
        Returns:
            self: 学習済みエスティメータ
        """
        self.survival_data_ = survival_data
        
        # データ検証
        self._validate_data(survival_data)
        
        # 全体のKM推定
        self.estimates_['overall'] = self._compute_km_estimate(
            survival_data.duration,
            survival_data.event
        )
        
        # 市場カテゴリ別のKM推定
        unique_categories = np.unique(survival_data.market_category)
        for category in unique_categories:
            mask = survival_data.market_category == category
            
            self.estimates_[category] = self._compute_km_estimate(
                survival_data.duration[mask],
                survival_data.event[mask]
            )
            
        self.fitted_ = True
        self.logger.info(f"Kaplan-Meier推定完了 - 対象企業数: {len(survival_data.company_id)}")
        
        return self
    
    def _validate_data(self, survival_data: SurvivalData) -> None:
        """データの検証"""
        if len(survival_data.duration) != len(survival_data.event):
            raise ValueError("duration と event の配列長が一致しません")
            
        if np.any(survival_data.duration < 0):
            raise ValueError("観測期間に負の値が含まれています")
            
        if not np.all(np.isin(survival_data.event, [0, 1])):
            raise ValueError("イベントフラグは0または1である必要があります")
    
    def _compute_km_estimate(self, durations: np.ndarray, events: np.ndarray) -> KMEstimate:
        """
        Kaplan-Meier推定の計算
        
        Args:
            durations: 観測期間
            events: イベント発生フラグ
            
        Returns:
            KMEstimate: 推定結果
        """
        # データをDataFrameに変換
        df = pd.DataFrame({
            'duration': durations,
            'event': events
        })
        
        # 時点ごとにグループ化
        grouped = df.groupby('duration').agg({
            'event': ['count', 'sum']
        }).reset_index()
        
        grouped.columns = ['time', 'n_at_risk', 'n_events']
        grouped = grouped.sort_values('time')
        
        # リスク集合の計算（後ろから累積）
        total_subjects = len(durations)
        grouped['n_risk'] = total_subjects - grouped['n_at_risk'].shift(1).fillna(0).cumsum() + grouped['n_at_risk']
        
        # 生存確率の計算
        grouped['survival_rate'] = 1 - (grouped['n_events'] / grouped['n_risk'])
        grouped['survival_prob'] = grouped['survival_rate'].cumprod()
        
        # 時点0を追加
        time_zero = pd.DataFrame({
            'time': [0],
            'n_at_risk': [0],
            'n_events': [0],
            'n_risk': [total_subjects],
            'survival_rate': [1.0],
            'survival_prob': [1.0]
        })
        
        result_df = pd.concat([time_zero, grouped], ignore_index=True)
        
        # 信頼区間の計算
        confidence_interval = self._compute_confidence_interval(result_df)
        
        return KMEstimate(
            time=result_df['time'].values,
            survival_prob=result_df['survival_prob'].values,
            n_risk=result_df['n_risk'].values,
            n_events=result_df['n_events'].values,
            confidence_interval=confidence_interval
        )
    
    def _compute_confidence_interval(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Greenwood's formulaを用いた信頼区間の計算
        
        Args:
            df: 生存分析結果のDataFrame
            
        Returns:
            信頼区間の辞書
        """
        # Greenwoodの分散推定
        df['variance_component'] = df['n_events'] / (df['n_risk'] * (df['n_risk'] - df['n_events']))
        df['variance_component'] = df['variance_component'].fillna(0)  # 0/0 = NaNを0に
        df['cumulative_variance'] = df['variance_component'].cumsum()
        
        # 標準誤差の計算
        df['se'] = df['survival_prob'] * np.sqrt(df['cumulative_variance'])
        
        # Z値
        z_value = stats.norm.ppf(1 - self.alpha / 2)
        
        # 信頼区間（logit変換を用いた方法）
        df['logit_s'] = np.log(df['survival_prob'] / (1 - df['survival_prob'] + 1e-10))
        df['se_logit'] = df['se'] / (df['survival_prob'] * (1 - df['survival_prob']) + 1e-10)
        
        df['ci_lower_logit'] = df['logit_s'] - z_value * df['se_logit']
        df['ci_upper_logit'] = df['logit_s'] + z_value * df['se_logit']
        
        ci_lower = 1 / (1 + np.exp(-df['ci_lower_logit']))
        ci_upper = 1 / (1 + np.exp(-df['ci_upper_logit']))
        
        return {
            'lower': np.clip(ci_lower.values, 0, 1),
            'upper': np.clip(ci_upper.values, 0, 1)
        }
    
    def predict_survival(self, time_points: Union[float, List[float], np.ndarray], 
                        category: str = 'overall') -> np.ndarray:
        """
        指定時点での生存確率を予測
        
        Args:
            time_points: 予測したい時点
            category: 市場カテゴリ（'overall', '高シェア市場', '低下市場', '失失市場'）
            
        Returns:
            生存確率の配列
        """
        if not self.fitted_:
            raise ValueError("fit()を先に実行してください")
            
        if category not in self.estimates_:
            raise ValueError(f"カテゴリ '{category}' は存在しません")
        
        estimate = self.estimates_[category]
        
        if np.isscalar(time_points):
            time_points = np.array([time_points])
        else:
            time_points = np.array(time_points)
        
        # 各時点での生存確率を補間
        survival_probs = np.interp(
            time_points, 
            estimate.time, 
            estimate.survival_prob
        )
        
        return survival_probs
    
    def get_median_survival_time(self, category: str = 'overall') -> Optional[float]:
        """
        中央生存期間を取得
        
        Args:
            category: 市場カテゴリ
            
        Returns:
            中央生存期間（None if not reached）
        """
        if not self.fitted_:
            raise ValueError("fit()を先に実行してください")
            
        estimate = self.estimates_[category]
        
        # 生存確率が0.5を下回る最初の時点を探す
        below_half = estimate.survival_prob < 0.5
        if np.any(below_half):
            idx = np.where(below_half)[0][0]
            return estimate.time[idx]
        else:
            return None  # 中央生存期間に達していない
    
    def log_rank_test(self, category1: str, category2: str) -> Dict[str, float]:
        """
        Log-rank検定による生存曲線の比較
        
        Args:
            category1, category2: 比較する市場カテゴリ
            
        Returns:
            検定結果の辞書
        """
        if not self.fitted_:
            raise ValueError("fit()を先に実行してください")
        
        # 各カテゴリのデータを取得
        mask1 = self.survival_data_.market_category == category1
        mask2 = self.survival_data_.market_category == category2
        
        durations1 = self.survival_data_.duration[mask1]
        events1 = self.survival_data_.event[mask1]
        durations2 = self.survival_data_.duration[mask2]
        events2 = self.survival_data_.event[mask2]
        
        # 統合データの作成
        all_times = np.unique(np.concatenate([durations1, durations2]))
        
        observed_events = 0
        expected_events = 0
        variance = 0
        
        for t in all_times:
            # 時点tでのリスク集合
            n1_risk = np.sum(durations1 >= t)
            n2_risk = np.sum(durations2 >= t)
            total_risk = n1_risk + n2_risk
            
            if total_risk == 0:
                continue
                
            # 時点tでのイベント数
            n1_events = np.sum((durations1 == t) & (events1 == 1))
            n2_events = np.sum((durations2 == t) & (events2 == 1))
            total_events = n1_events + n2_events
            
            if total_events == 0:
                continue
            
            # 期待イベント数
            expected1 = (n1_risk / total_risk) * total_events
            
            # 分散の計算
            if total_risk > 1:
                var_component = (n1_risk * n2_risk * total_events * (total_risk - total_events)) / \
                               (total_risk ** 2 * (total_risk - 1))
                variance += var_component
            
            observed_events += n1_events
            expected_events += expected1
        
        # Log-rank統計量
        if variance > 0:
            log_rank_statistic = (observed_events - expected_events) ** 2 / variance
            p_value = 1 - stats.chi2.cdf(log_rank_statistic, df=1)
        else:
            log_rank_statistic = 0
            p_value = 1.0
        
        return {
            'statistic': log_rank_statistic,
            'p_value': p_value,
            'observed_events_1': observed_events,
            'expected_events_1': expected_events,
            'category_1': category1,
            'category_2': category2
        }
    
    def plot_survival_curves(self, categories: Optional[List[str]] = None, 
                            figsize: Tuple[int, int] = (12, 8),
                            show_confidence_interval: bool = True,
                            title: str = "企業生存曲線分析") -> plt.Figure:
        """
        生存曲線のプロット
        
        Args:
            categories: プロットするカテゴリのリスト
            figsize: 図のサイズ
            show_confidence_interval: 信頼区間の表示
            title: グラフタイトル
            
        Returns:
            matplotlib Figure
        """
        if not self.fitted_:
            raise ValueError("fit()を先に実行してください")
        
        if categories is None:
            categories = list(self.estimates_.keys())
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # カラーパレット
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        for i, category in enumerate(categories):
            if category not in self.estimates_:
                self.logger.warning(f"カテゴリ '{category}' が見つかりません")
                continue
                
            estimate = self.estimates_[category]
            color = colors[i % len(colors)]
            
            # 生存曲線をプロット
            ax.plot(estimate.time, estimate.survival_prob, 
                    label=category, color=color, linewidth=2)
            
            # 信頼区間をプロット
            if show_confidence_interval:
                ax.fill_between(estimate.time, 
                                estimate.confidence_interval['lower'],
                                estimate.confidence_interval['upper'],
                                alpha=0.2, color=color)
        
        ax.set_xlabel('年数', fontsize=12)
        ax.set_ylabel('生存確率', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)
        
        # 中央生存期間をプロット
        for i, category in enumerate(categories):
            if category in self.estimates_:
                median_time = self.get_median_survival_time(category)
                if median_time is not None:
                    ax.axvline(x=median_time, color=colors[i % len(colors)], 
                                linestyle='--', alpha=0.7)
                    ax.text(median_time, 0.5, f'{category}\n中央生存期間: {median_time:.1f}年',
                            rotation=90, verticalalignment='bottom')
        
        plt.tight_layout()
        return fig
    
    def plot_risk_table(self, categories: Optional[List[str]] = None,
                        time_intervals: Optional[List[float]] = None) -> plt.Figure:
        """
        リスク集合表の作成
        
        Args:
            categories: 表示するカテゴリ
            time_intervals: 表示する時間間隔
            
        Returns:
            matplotlib Figure
        """
        if not self.fitted_:
            raise ValueError("fit()を先に実行してください")
        
        if categories is None:
            categories = list(self.estimates_.keys())
            
        if time_intervals is None:
            max_time = max([est.time.max() for est in self.estimates_.values()])
            time_intervals = np.arange(0, max_time + 1, 5)
        
        # リスク集合データの作成
        risk_data = []
        for category in categories:
            if category not in self.estimates_:
                continue
                
            estimate = self.estimates_[category]
            risk_at_intervals = []
            
            for t in time_intervals:
                # 指定時点でのリスク集合数を補間
                risk_count = np.interp(t, estimate.time, estimate.n_risk)
                risk_at_intervals.append(int(risk_count))
            
            risk_data.append(risk_at_intervals)
        
        # 表の作成
        fig, ax = plt.subplots(figsize=(12, len(categories) * 0.8 + 2))
        ax.axis('tight')
        ax.axis('off')
        
        table_data = []
        for i, category in enumerate(categories):
            if i < len(risk_data):
                row = [category] + [str(x) for x in risk_data[i]]
                table_data.append(row)
        
        headers = ['市場カテゴリ'] + [f'{t:.0f}年' for t in time_intervals]
        
        table = ax.table(cellText=table_data, colLabels=headers,
                        cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.8)
        
        # ヘッダーの装飾
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax.set_title('時点別リスク集合数', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        return fig
    
    def generate_summary_report(self) -> Dict[str, any]:
        """
        生存分析の要約レポートを生成
        
        Returns:
            要約統計の辞書
        """
        if not self.fitted_:
            raise ValueError("fit()を先に実行してください")
        
        report = {
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_companies': len(self.survival_data_.company_id),
            'categories': {}
        }
        
        for category, estimate in self.estimates_.items():
            category_summary = {
                'sample_size': int(estimate.n_risk[0]),  # 初期のリスク集合数
                'total_events': int(estimate.n_events.sum()),  # 総イベント数
                'censored': int(estimate.n_risk[0] - estimate.n_events.sum()),  # 打ち切り数
                'median_survival_time': self.get_median_survival_time(category),
                'survival_rates': {
                    '5年': float(self.predict_survival(5, category)[0]),
                    '10年': float(self.predict_survival(10, category)[0]),
                    '20年': float(self.predict_survival(20, category)[0]),
                    '30年': float(self.predict_survival(30, category)[0])
                }
            }
            report['categories'][category] = category_summary
        
        return report
    
    def save_estimates(self, filepath: str) -> None:
        """
        推定結果をファイルに保存
        
        Args:
            filepath: 保存先ファイルパス
        """
        if not self.fitted_:
            raise ValueError("fit()を先に実行してください")
        
        results = {}
        for category, estimate in self.estimates_.items():
            results[category] = {
                'time': estimate.time.tolist(),
                'survival_prob': estimate.survival_prob.tolist(),
                'n_risk': estimate.n_risk.tolist(),
                'n_events': estimate.n_events.tolist(),
                'confidence_interval': {
                    'lower': estimate.confidence_interval['lower'].tolist(),
                    'upper': estimate.confidence_interval['upper'].tolist()
                }
            }
        
        import json
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"推定結果を {filepath} に保存しました")


def create_sample_survival_data() -> SurvivalData:
    """
    サンプルデータの作成（テスト用）
    """
    np.random.seed(42)
    n_companies = 150
    
    # 市場カテゴリの生成
    categories = ['高シェア市場'] * 50 + ['低下市場'] * 50 + ['失失市場'] * 50
    
    durations = []
    events = []
    
    for category in categories:
        if category == '高シェア市場':
            # 高シェア市場：長期生存傾向
            duration = np.random.exponential(35) + 5
            event = 1 if duration < 40 else 0
        elif category == '低下市場':
            # 低下市場：中程度の生存
            duration = np.random.exponential(25) + 3
            event = 1 if duration < 35 else 0
        else:  # 失失市場
            # 失失市場：短期で消滅しやすい
            duration = np.random.exponential(15) + 1
            event = 1 if duration < 25 else 0
        
        durations.append(min(duration, 40))  # 最大40年で打ち切り
        events.append(event)
    
    return SurvivalData(
        duration=np.array(durations),
        event=np.array(events),
        company_id=np.arange(n_companies),
        market_category=np.array(categories)
    )


if __name__ == "__main__":
    # サンプル実行
    print("A2AI Kaplan-Meier生存分析モジュール - サンプル実行")
    
    # サンプルデータの生成
    sample_data = create_sample_survival_data()
    
    # Kaplan-Meier推定器の初期化と学習
    km_estimator = KaplanMeierEstimator(confidence_level=0.95)
    km_estimator.fit(sample_data)
    
    # 結果の表示
    print("\n=== 生存分析要約レポート ===")
    report = km_estimator.generate_summary_report()
    
    for category, stats in report['categories'].items():
        print(f"\n{category}:")
        print(f"  サンプル数: {stats['sample_size']}")
        print(f"  イベント発生数: {stats['total_events']}")
        print(f"  打ち切り数: {stats['censored']}")
        print(f"  中央生存期間: {stats['median_survival_time']:.1f}年" if stats['median_survival_time'] else "  中央生存期間: 未到達")
        print(f"  10年生存率: {stats['survival_rates']['10年']:.3f}")
        print(f"  20年生存率: {stats['survival_rates']['20年']:.3f}")
    
    # Log-rank検定
    print("\n=== Log-rank検定結果 ===")
    test_result = km_estimator.log_rank_test('高シェア市場', '失失市場')
    print(f"高シェア市場 vs 失失市場:")
    print(f"  統計量: {test_result['statistic']:.4f}")
    print(f"  p値: {test_result['p_value']:.4f}")
    
    print("\nKaplan-Meier分析が完了しました。")