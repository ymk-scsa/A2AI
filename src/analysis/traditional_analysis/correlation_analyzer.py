"""
A2AI (Advanced Financial Analysis AI) - Correlation Analyzer
相関分析モジュール

150社×40年分の財務データに対する包括的な相関分析機能を提供
- 評価項目と要因項目間の相関分析
- 市場カテゴリー別相関パターン分析
- 時系列相関分析
- 企業ライフサイクル段階別相関分析
- 生存バイアス補正機能付き相関分析
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import scipy.stats as stats
from scipy.stats import pearsonr, spearmanr, kendalltau
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
import logging
from pathlib import Path

# 設定とログ
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CorrelationType(Enum):
    """相関分析の種類"""
    PEARSON = "pearson"
    SPEARMAN = "spearman"
    KENDALL = "kendall"
    PARTIAL = "partial"


class MarketCategory(Enum):
    """市場カテゴリー"""
    HIGH_SHARE = "high_share"      # 現在もシェアが高い市場
    DECLINING = "declining"         # シェア低下中市場
    LOST = "lost"                  # 完全にシェア失失市場


class LifecycleStage(Enum):
    """企業ライフサイクル段階"""
    STARTUP = "startup"        # 設立初期（0-5年）
    GROWTH = "growth"          # 成長期（6-15年）
    MATURITY = "maturity"      # 成熟期（16-30年）
    MATURE = "mature"          # 老舗期（31年以上）
    DECLINING = "declining"    # 衰退期
    EXTINCT = "extinct"        # 消滅


@dataclass
class CorrelationResult:
    """相関分析結果"""
    correlation_matrix: pd.DataFrame
    p_values: pd.DataFrame
    significant_pairs: List[Tuple[str, str, float, float]]
    correlation_type: CorrelationType
    sample_size: int
    market_category: Optional[MarketCategory] = None
    lifecycle_stage: Optional[LifecycleStage] = None
    time_period: Optional[Tuple[int, int]] = None


@dataclass
class PartialCorrelationResult:
    """偏相関分析結果"""
    partial_correlation: float
    p_value: float
    controlled_variables: List[str]
    sample_size: int


class CorrelationAnalyzer:
    """
    相関分析クラス
    
    財務諸表データに対する包括的な相関分析を実行
    - 基本的な相関分析（Pearson, Spearman, Kendall）
    - 市場カテゴリー別分析
    - 時系列相関分析
    - 偏相関分析
    - 生存バイアス補正
    """
    
    def __init__(self, data: pd.DataFrame, config: Dict = None):
        """
        初期化
        
        Args:
            data: 財務データ（企業×年×指標の形式）
            config: 設定パラメータ
        """
        self.data = data.copy()
        self.config = config or {}
        self.scaler = StandardScaler()
        
        # デフォルト設定
        self.default_config = {
            'significance_level': 0.05,
            'min_sample_size': 30,
            'correlation_threshold': 0.3,
            'survivorship_bias_correction': True,
            'handle_missing_values': 'pairwise',
            'standardize_data': True
        }
        
        # 設定を統合
        self.config = {**self.default_config, **self.config}
        
        # データの前処理
        self._preprocess_data()
        
        # 評価項目と要因項目の定義
        self._define_evaluation_factors()
        
        logger.info(f"CorrelationAnalyzer initialized with {len(self.data)} records")
    
    def _preprocess_data(self):
        """データの前処理"""
        # 欠損値処理
        if self.config['handle_missing_values'] == 'drop':
            self.data = self.data.dropna()
        elif self.config['handle_missing_values'] == 'forward_fill':
            self.data = self.data.fillna(method='ffill')
        
        # 数値データのみを抽出
        numeric_columns = self.data.select_dtypes(include=[np.number]).columns
        self.data = self.data[numeric_columns]
        
        # 標準化
        if self.config['standardize_data']:
            self.standardized_data = pd.DataFrame(
                self.scaler.fit_transform(self.data),
                columns=self.data.columns,
                index=self.data.index
            )
        else:
            self.standardized_data = self.data.copy()
    
    def _define_evaluation_factors(self):
        """評価項目と要因項目の定義"""
        # 9つの評価項目
        self.evaluation_metrics = [
            '売上高',
            '売上高成長率',
            '売上高営業利益率',
            '売上高当期純利益率',
            'ROE',
            '売上高付加価値率',
            '企業存続確率',
            '新規事業成功率',
            '事業継承成功度'
        ]
        
        # 各評価項目に対する23の要因項目（代表例）
        self.factor_metrics = [
            # 投資・資産関連
            '有形固定資産', '設備投資額', '研究開発費', '無形固定資産', '投資有価証券',
            # 人的資源関連
            '従業員数', '平均年間給与', '退職給付費用', '福利厚生費',
            # 運転資本・効率性関連
            '売上債権', '棚卸資産', '総資産', '売上債権回転率', '棚卸資産回転率',
            # 事業展開関連
            '海外売上高比率', '事業セグメント数', '販売費及び一般管理費', '広告宣伝費',
            '営業外収益', '受注残高',
            # ライフサイクル関連
            '企業年齢', '市場参入時期', '親会社依存度'
        ]
    
    def basic_correlation_analysis(
        self,
        variables: List[str] = None,
        correlation_type: CorrelationType = CorrelationType.PEARSON,
        market_category: MarketCategory = None,
        lifecycle_stage: LifecycleStage = None
    ) -> CorrelationResult:
        """
        基本的な相関分析
        
        Args:
            variables: 分析対象変数のリスト
            correlation_type: 相関係数の種類
            market_category: 市場カテゴリーでフィルタ
            lifecycle_stage: ライフサイクル段階でフィルタ
            
        Returns:
            CorrelationResult: 相関分析結果
        """
        # データのフィルタリング
        filtered_data = self._filter_data(market_category, lifecycle_stage)
        
        # 分析対象変数の選択
        if variables is None:
            variables = list(filtered_data.columns)
        else:
            variables = [var for var in variables if var in filtered_data.columns]
        
        analysis_data = filtered_data[variables]
        
        # 相関係数の計算
        if correlation_type == CorrelationType.PEARSON:
            corr_matrix = analysis_data.corr(method='pearson')
        elif correlation_type == CorrelationType.SPEARMAN:
            corr_matrix = analysis_data.corr(method='spearman')
        elif correlation_type == CorrelationType.KENDALL:
            corr_matrix = analysis_data.corr(method='kendall')
        else:
            raise ValueError(f"Unsupported correlation type: {correlation_type}")
        
        # p値の計算
        p_values = self._calculate_p_values(analysis_data, correlation_type)
        
        # 有意な相関ペアの抽出
        significant_pairs = self._extract_significant_pairs(
            corr_matrix, p_values, self.config['significance_level']
        )
        
        return CorrelationResult(
            correlation_matrix=corr_matrix,
            p_values=p_values,
            significant_pairs=significant_pairs,
            correlation_type=correlation_type,
            sample_size=len(analysis_data),
            market_category=market_category,
            lifecycle_stage=lifecycle_stage
        )
    
    def evaluation_factor_correlation(
        self,
        evaluation_metric: str,
        market_category: MarketCategory = None
    ) -> Dict[str, CorrelationResult]:
        """
        特定の評価項目と要因項目間の相関分析
        
        Args:
            evaluation_metric: 評価項目
            market_category: 市場カテゴリー
            
        Returns:
            Dict[str, CorrelationResult]: 相関分析結果辞書
        """
        results = {}
        
        # 市場カテゴリー別分析
        if market_category is None:
            categories = list(MarketCategory)
        else:
            categories = [market_category]
        
        for category in categories:
            # 評価項目と要因項目のリスト作成
            variables = [evaluation_metric] + self.factor_metrics
            variables = [var for var in variables if var in self.data.columns]
            
            if len(variables) < 2:
                logger.warning(f"Insufficient variables for {category.value}")
                continue
            
            # 相関分析実行
            result = self.basic_correlation_analysis(
                variables=variables,
                market_category=category
            )
            
            results[category.value] = result
        
        return results
    
    def time_series_correlation(
        self,
        variable1: str,
        variable2: str,
        window_size: int = 5,
        market_category: MarketCategory = None
    ) -> pd.DataFrame:
        """
        時系列相関分析（移動窓相関）
        
        Args:
            variable1: 変数1
            variable2: 変数2
            window_size: 移動窓サイズ（年数）
            market_category: 市場カテゴリー
            
        Returns:
            pd.DataFrame: 時系列相関結果
        """
        # データのフィルタリング
        filtered_data = self._filter_data(market_category)
        
        if variable1 not in filtered_data.columns or variable2 not in filtered_data.columns:
            raise ValueError("Specified variables not found in data")
        
        # 企業別・年度別データの整理
        if 'company' in filtered_data.columns and 'year' in filtered_data.columns:
            # 企業×年度のピボット形式に変換
            pivot_data1 = filtered_data.pivot(index='year', columns='company', values=variable1)
            pivot_data2 = filtered_data.pivot(index='year', columns='company', values=variable2)
            
            # 移動窓相関の計算
            rolling_correlations = []
            
            for i in range(window_size, len(pivot_data1) + 1):
                window_data1 = pivot_data1.iloc[i-window_size:i].stack().dropna()
                window_data2 = pivot_data2.iloc[i-window_size:i].stack().dropna()
                
                # 共通のインデックスを取得
                common_index = window_data1.index.intersection(window_data2.index)
                
                if len(common_index) > self.config['min_sample_size']:
                    corr, p_value = pearsonr(
                        window_data1[common_index],
                        window_data2[common_index]
                    )
                    
                    rolling_correlations.append({
                        'year': pivot_data1.index[i-1],
                        'correlation': corr,
                        'p_value': p_value,
                        'sample_size': len(common_index)
                    })
            
            return pd.DataFrame(rolling_correlations)
        
        else:
            # 単純な時系列データの場合
            data1 = filtered_data[variable1].dropna()
            data2 = filtered_data[variable2].dropna()
            
            # データの整合
            common_index = data1.index.intersection(data2.index)
            
            if len(common_index) < self.config['min_sample_size']:
                raise ValueError("Insufficient data for time series correlation")
            
            aligned_data1 = data1[common_index]
            aligned_data2 = data2[common_index]
            
            # 移動窓相関
            rolling_corr = aligned_data1.rolling(window=window_size).corr(aligned_data2)
            
            return pd.DataFrame({
                'correlation': rolling_corr,
                'index': rolling_corr.index
            }).dropna()
    
    def partial_correlation_analysis(
        self,
        x: str,
        y: str,
        control_variables: List[str],
        market_category: MarketCategory = None
    ) -> PartialCorrelationResult:
        """
        偏相関分析
        
        Args:
            x: 独立変数
            y: 従属変数
            control_variables: 統制変数のリスト
            market_category: 市場カテゴリー
            
        Returns:
            PartialCorrelationResult: 偏相関分析結果
        """
        # データのフィルタリング
        filtered_data = self._filter_data(market_category)
        
        # 必要な変数が存在するかチェック
        all_variables = [x, y] + control_variables
        missing_vars = [var for var in all_variables if var not in filtered_data.columns]
        
        if missing_vars:
            raise ValueError(f"Variables not found: {missing_vars}")
        
        # データの抽出と欠損値処理
        analysis_data = filtered_data[all_variables].dropna()
        
        if len(analysis_data) < self.config['min_sample_size']:
            raise ValueError("Insufficient data for partial correlation analysis")
        
        # 偏相関係数の計算
        try:
            partial_corr, p_value = self._calculate_partial_correlation(
                analysis_data, x, y, control_variables
            )
        except Exception as e:
            logger.error(f"Error in partial correlation calculation: {e}")
            raise
        
        return PartialCorrelationResult(
            partial_correlation=partial_corr,
            p_value=p_value,
            controlled_variables=control_variables,
            sample_size=len(analysis_data)
        )
    
    def market_comparison_correlation(
        self,
        evaluation_metric: str
    ) -> Dict[str, Dict[str, float]]:
        """
        市場カテゴリー間での相関パターン比較
        
        Args:
            evaluation_metric: 評価項目
            
        Returns:
            Dict: 市場別相関パターン比較結果
        """
        comparison_results = {}
        
        for category in MarketCategory:
            try:
                # 市場別相関分析
                result = self.evaluation_factor_correlation(
                    evaluation_metric, category
                )
                
                if category.value in result:
                    corr_matrix = result[category.value].correlation_matrix
                    
                    # 評価項目と各要因項目の相関を抽出
                    if evaluation_metric in corr_matrix.index:
                        correlations = corr_matrix.loc[evaluation_metric].drop(evaluation_metric)
                        comparison_results[category.value] = correlations.to_dict()
                
            except Exception as e:
                logger.warning(f"Error in correlation analysis for {category.value}: {e}")
                comparison_results[category.value] = {}
        
        return comparison_results
    
    def lifecycle_correlation_evolution(
        self,
        variable1: str,
        variable2: str
    ) -> Dict[str, float]:
        """
        ライフサイクル段階別の相関進化分析
        
        Args:
            variable1: 変数1
            variable2: 変数2
            
        Returns:
            Dict: ライフサイクル段階別相関係数
        """
        lifecycle_correlations = {}
        
        for stage in LifecycleStage:
            try:
                result = self.basic_correlation_analysis(
                    variables=[variable1, variable2],
                    lifecycle_stage=stage
                )
                
                if len(result.correlation_matrix) >= 2:
                    corr_value = result.correlation_matrix.loc[variable1, variable2]
                    lifecycle_correlations[stage.value] = corr_value
                
            except Exception as e:
                logger.warning(f"Error in lifecycle correlation for {stage.value}: {e}")
                lifecycle_correlations[stage.value] = np.nan
        
        return lifecycle_correlations
    
    def survivorship_bias_corrected_correlation(
        self,
        variable1: str,
        variable2: str,
        include_extinct: bool = True
    ) -> Tuple[float, float]:
        """
        生存バイアス補正済み相関分析
        
        Args:
            variable1: 変数1
            variable2: 変数2
            include_extinct: 消滅企業を含めるかどうか
            
        Returns:
            Tuple[float, float]: (補正済み相関係数, p値)
        """
        if not self.config['survivorship_bias_correction']:
            # 通常の相関分析
            data = self.data[[variable1, variable2]].dropna()
            return pearsonr(data[variable1], data[variable2])
        
        # 生存企業のみのデータ
        surviving_data = self._filter_data_by_survival_status(surviving_only=True)
        surviving_corr_data = surviving_data[[variable1, variable2]].dropna()
        
        if include_extinct:
            # 消滅企業を含むデータ
            all_data = self.data[[variable1, variable2]].dropna()
            all_corr, all_p = pearsonr(all_data[variable1], all_data[variable2])
            return all_corr, all_p
        else:
            # 生存企業のみ
            if len(surviving_corr_data) >= self.config['min_sample_size']:
                surviving_corr, surviving_p = pearsonr(
                    surviving_corr_data[variable1],
                    surviving_corr_data[variable2]
                )
                return surviving_corr, surviving_p
            else:
                logger.warning("Insufficient surviving company data")
                return np.nan, np.nan
    
    def correlation_network_analysis(
        self,
        variables: List[str] = None,
        correlation_threshold: float = None
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        相関ネットワーク分析
        
        Args:
            variables: 分析対象変数
            correlation_threshold: 相関閾値
            
        Returns:
            Dict: 変数間のネットワーク構造
        """
        threshold = correlation_threshold or self.config['correlation_threshold']
        
        if variables is None:
            variables = self.evaluation_metrics + self.factor_metrics
            variables = [var for var in variables if var in self.data.columns]
        
        # 相関行列計算
        corr_result = self.basic_correlation_analysis(variables=variables)
        corr_matrix = corr_result.correlation_matrix
        
        # ネットワーク構築
        network = {}
        
        for var1 in variables:
            connections = []
            for var2 in variables:
                if var1 != var2:
                    corr_value = corr_matrix.loc[var1, var2]
                    if abs(corr_value) >= threshold:
                        connections.append((var2, corr_value))
            
            # 相関の絶対値でソート
            connections.sort(key=lambda x: abs(x[1]), reverse=True)
            network[var1] = connections
        
        return network
    
    def _filter_data(
        self,
        market_category: MarketCategory = None,
        lifecycle_stage: LifecycleStage = None
    ) -> pd.DataFrame:
        """データのフィルタリング"""
        filtered_data = self.standardized_data.copy()
        
        # 市場カテゴリーでフィルタ
        if market_category and 'market_category' in filtered_data.columns:
            filtered_data = filtered_data[
                filtered_data['market_category'] == market_category.value
            ]
        
        # ライフサイクル段階でフィルタ
        if lifecycle_stage and 'lifecycle_stage' in filtered_data.columns:
            filtered_data = filtered_data[
                filtered_data['lifecycle_stage'] == lifecycle_stage.value
            ]
        
        return filtered_data
    
    def _filter_data_by_survival_status(
        self,
        surviving_only: bool = True
    ) -> pd.DataFrame:
        """生存状況によるデータフィルタリング"""
        filtered_data = self.data.copy()
        
        if 'is_extinct' in filtered_data.columns:
            if surviving_only:
                filtered_data = filtered_data[filtered_data['is_extinct'] == False]
            else:
                # 消滅企業のみ
                filtered_data = filtered_data[filtered_data['is_extinct'] == True]
        
        return filtered_data
    
    def _calculate_p_values(
        self,
        data: pd.DataFrame,
        correlation_type: CorrelationType
    ) -> pd.DataFrame:
        """p値の計算"""
        variables = data.columns
        n_vars = len(variables)
        p_values = np.zeros((n_vars, n_vars))
        
        for i, var1 in enumerate(variables):
            for j, var2 in enumerate(variables):
                if i == j:
                    p_values[i, j] = 0.0
                else:
                    x = data[var1].dropna()
                    y = data[var2].dropna()
                    
                    # 共通インデックスを取得
                    common_index = x.index.intersection(y.index)
                    
                    if len(common_index) >= 3:
                        x_common = x[common_index]
                        y_common = y[common_index]
                        
                        try:
                            if correlation_type == CorrelationType.PEARSON:
                                _, p_val = pearsonr(x_common, y_common)
                            elif correlation_type == CorrelationType.SPEARMAN:
                                _, p_val = spearmanr(x_common, y_common)
                            elif correlation_type == CorrelationType.KENDALL:
                                _, p_val = kendalltau(x_common, y_common)
                            else:
                                p_val = np.nan
                            
                            p_values[i, j] = p_val
                        except:
                            p_values[i, j] = np.nan
                    else:
                        p_values[i, j] = np.nan
        
        return pd.DataFrame(p_values, index=variables, columns=variables)
    
    def _extract_significant_pairs(
        self,
        corr_matrix: pd.DataFrame,
        p_values: pd.DataFrame,
        alpha: float
    ) -> List[Tuple[str, str, float, float]]:
        """有意な相関ペアの抽出"""
        significant_pairs = []
        
        for i, var1 in enumerate(corr_matrix.index):
            for j, var2 in enumerate(corr_matrix.columns):
                if i < j:  # 上三角行列のみ処理
                    corr_val = corr_matrix.iloc[i, j]
                    p_val = p_values.iloc[i, j]
                    
                    if pd.notna(p_val) and p_val < alpha:
                        significant_pairs.append((var1, var2, corr_val, p_val))
        
        # 相関の絶対値でソート
        significant_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        
        return significant_pairs
    
    def _calculate_partial_correlation(
        self,
        data: pd.DataFrame,
        x: str,
        y: str,
        control_vars: List[str]
    ) -> Tuple[float, float]:
        """偏相関係数の計算"""
        from sklearn.linear_model import LinearRegression
        
        # 制御変数で回帰
        control_data = data[control_vars]
        
        # xを制御変数で回帰
        reg_x = LinearRegression().fit(control_data, data[x])
        residual_x = data[x] - reg_x.predict(control_data)
        
        # yを制御変数で回帰
        reg_y = LinearRegression().fit(control_data, data[y])
        residual_y = data[y] - reg_y.predict(control_data)
        
        # 残差同士の相関
        partial_corr, p_value = pearsonr(residual_x, residual_y)
        
        return partial_corr, p_value
    
    def generate_correlation_report(
        self,
        output_dir: str = "results/analysis_results/traditional_analysis/"
    ) -> Dict[str, Union[pd.DataFrame, Dict]]:
        """
        相関分析の包括的レポート生成
        
        Args:
            output_dir: 出力ディレクトリ
            
        Returns:
            Dict: 分析結果の統合レポート
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        report = {
            'basic_correlations': {},
            'market_comparisons': {},
            'lifecycle_evolution': {},
            'network_analysis': {},
            'survivorship_analysis': {}
        }
        
        # 基本相関分析
        for evaluation_metric in self.evaluation_metrics:
            if evaluation_metric in self.data.columns:
                try:
                    result = self.evaluation_factor_correlation(evaluation_metric)
                    report['basic_correlations'][evaluation_metric] = result
                except Exception as e:
                    logger.error(f"Error in basic correlation for {evaluation_metric}: {e}")
        
        # 市場比較分析
        for evaluation_metric in self.evaluation_metrics:
            if evaluation_metric in self.data.columns:
                try:
                    comparison = self.market_comparison_correlation(evaluation_metric)
                    report['market_comparisons'][evaluation_metric] = comparison
                except Exception as e:
                    logger.error(f"Error in market comparison for {evaluation_metric}: {e}")
        
        # ネットワーク分析
        try:
            network = self.correlation_network_analysis()
            report['network_analysis'] = network
        except Exception as e:
            logger.error(f"Error in network analysis: {e}")
        
        logger.info(f"Correlation analysis report generated in {output_dir}")
        
        return report


def main():
    """メイン実行関数（テスト用）"""
    # サンプルデータの生成
    np.random.seed(42)
    n_samples = 1000
    
    sample_data = pd.DataFrame({
        '売上高': np.random.normal(100, 20, n_samples),
        '売上高成長率': np.random.normal(0.05, 0.15, n_samples),
        '売上高営業利益率': np.random.normal(0.08, 0.05, n_samples),
        'ROE': np.random.normal(0.12, 0.08, n_samples),
        '研究開発費': np.random.normal(5, 2, n_samples),
        '従業員数': np.random.normal(1000, 300, n_samples),
        '有形固定資産': np.random.normal(50, 15, n_samples),
        'market_category': np.random.choice(['high_share', 'declining', 'lost'], n_samples),
        'is_extinct': np.random.choice([True, False], n_samples, p=[0.1, 0.9])
    })
    
    # 相関分析実行
    analyzer = CorrelationAnalyzer(sample_data)
    
    # 基本相関分析
    basic_result = analyzer.basic_correlation_analysis(
        variables=['売上高', '売上高成長率', '研究開発費', '従業員数']
    )
    
    print("=== Basic Correlation Analysis ===")
    print(basic_result.correlation_matrix)
    print(f"\nSignificant pairs: {len(basic_result.significant_pairs)}")
    
    # 市場比較
    market_comparison = analyzer.market_comparison_correlation('売上高')
    print("\n=== Market Comparison ===")
    for market, correlations in market_comparison.items():
        print(f"{market}: {len(correlations)} correlations")
    
    # 包括レポート生成
    try:
        report = analyzer.generate_correlation_report()
        print("\n=== Comprehensive Report Generated ===")
        print(f"Basic correlations: {len(report['basic_correlations'])}")
        print(f"Market comparisons: {len(report['market_comparisons'])}")
        print(f"Network connections: {len(report['network_analysis'])}")
    except Exception as e:
        print(f"Error generating report: {e}")


class CorrelationVisualization:
    """相関分析結果の可視化クラス"""
    
    def __init__(self, analyzer: CorrelationAnalyzer):
        self.analyzer = analyzer
        self.setup_plot_style()
    
    def setup_plot_style(self):
        """プロットスタイルの設定"""
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        plt.rcParams['font.size'] = 10
        plt.rcParams['figure.figsize'] = (12, 8)
    
    def plot_correlation_matrix(
        self,
        correlation_result: CorrelationResult,
        title: str = "Correlation Matrix",
        save_path: str = None
    ) -> plt.Figure:
        """
        相関行列のヒートマップ作成
        
        Args:
            correlation_result: 相関分析結果
            title: グラフタイトル
            save_path: 保存パス
            
        Returns:
            plt.Figure: matplotlib図オブジェクト
        """
        fig, ax = plt.subplots(figsize=(14, 12))
        
        # マスク作成（上三角）
        mask = np.triu(np.ones_like(correlation_result.correlation_matrix, dtype=bool))
        
        # ヒートマップ作成
        sns.heatmap(
            correlation_result.correlation_matrix,
            mask=mask,
            annot=True,
            cmap='RdYlBu_r',
            center=0,
            square=True,
            fmt='.2f',
            cbar_kws={"shrink": .8},
            ax=ax
        )
        
        # タイトルと情報追加
        ax.set_title(f'{title}\n(Sample size: {correlation_result.sample_size}, '
                    f'Type: {correlation_result.correlation_type.value})', 
                    fontsize=14, pad=20)
        
        # 軸ラベルの回転
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot_market_comparison_heatmap(
        self,
        market_comparisons: Dict[str, Dict[str, float]],
        evaluation_metric: str,
        save_path: str = None
    ) -> plt.Figure:
        """
        市場カテゴリー間相関比較ヒートマップ
        
        Args:
            market_comparisons: 市場別相関データ
            evaluation_metric: 評価項目名
            save_path: 保存パス
            
        Returns:
            plt.Figure: matplotlib図オブジェクト
        """
        # データフレーム作成
        comparison_df = pd.DataFrame(market_comparisons).T
        
        fig, ax = plt.subplots(figsize=(16, 10))
        
        # ヒートマップ作成
        sns.heatmap(
            comparison_df,
            annot=True,
            cmap='RdYlBu_r',
            center=0,
            fmt='.3f',
            cbar_kws={"shrink": .8},
            ax=ax
        )
        
        ax.set_title(f'Market Category Correlation Comparison\n{evaluation_metric}', 
                    fontsize=16, pad=20)
        ax.set_xlabel('Factor Variables', fontsize=12)
        ax.set_ylabel('Market Categories', fontsize=12)
        
        # 軸ラベルの調整
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot_time_series_correlation(
        self,
        time_series_data: pd.DataFrame,
        variable1: str,
        variable2: str,
        title: str = None,
        save_path: str = None
    ) -> plt.Figure:
        """
        時系列相関の推移プロット
        
        Args:
            time_series_data: 時系列相関データ
            variable1: 変数1の名前
            variable2: 変数2の名前
            title: グラフタイトル
            save_path: 保存パス
            
        Returns:
            plt.Figure: matplotlib図オブジェクト
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        if title is None:
            title = f'Time Series Correlation: {variable1} vs {variable2}'
        
        # 相関係数の推移
        ax1.plot(time_series_data['year'], time_series_data['correlation'], 
                'b-', linewidth=2, marker='o', markersize=4)
        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
        ax1.set_ylabel('Correlation Coefficient', fontsize=12)
        ax1.set_title(f'{title}\nCorrelation Evolution', fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        # 有意性の可視化
        significant = time_series_data['p_value'] < 0.05
        ax1.scatter(time_series_data[significant]['year'], 
                    time_series_data[significant]['correlation'],
                    color='red', s=50, alpha=0.7, label='Significant (p<0.05)')
        ax1.legend()
        
        # サンプルサイズの推移
        ax2.bar(time_series_data['year'], time_series_data['sample_size'], 
                alpha=0.6, color='green')
        ax2.set_xlabel('Year', fontsize=12)
        ax2.set_ylabel('Sample Size', fontsize=12)
        ax2.set_title('Sample Size Evolution', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot_correlation_network(
        self,
        network_data: Dict[str, List[Tuple[str, float]]],
        top_n_connections: int = 5,
        save_path: str = None
    ) -> plt.Figure:
        """
        相関ネットワーク図の作成
        
        Args:
            network_data: ネットワークデータ
            top_n_connections: 表示する上位接続数
            save_path: 保存パス
            
        Returns:
            plt.Figure: matplotlib図オブジェクト
        """
        try:
            import networkx as nx
        except ImportError:
            logger.error("NetworkX is required for network visualization")
            return None
        
        # NetworkXグラフ作成
        G = nx.Graph()
        
        # ノードとエッジの追加
        for node, connections in network_data.items():
            G.add_node(node)
            for connected_node, correlation in connections[:top_n_connections]:
                G.add_edge(node, connected_node, weight=abs(correlation))
        
        fig, ax = plt.subplots(figsize=(16, 12))
        
        # レイアウト計算
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # ノードの描画
        node_sizes = [G.degree(node) * 100 for node in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, 
                                node_color='lightblue', alpha=0.7, ax=ax)
        
        # エッジの描画
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        nx.draw_networkx_edges(G, pos, width=[w*3 for w in weights], 
                                alpha=0.6, edge_color='gray', ax=ax)
        
        # ラベルの描画
        nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold', ax=ax)
        
        ax.set_title('Correlation Network Analysis', fontsize=16, pad=20)
        ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot_lifecycle_correlation_evolution(
        self,
        lifecycle_correlations: Dict[str, float],
        variable1: str,
        variable2: str,
        save_path: str = None
    ) -> plt.Figure:
        """
        ライフサイクル段階別相関進化の可視化
        
        Args:
            lifecycle_correlations: ライフサイクル別相関データ
            variable1: 変数1の名前
            variable2: 変数2の名前
            save_path: 保存パス
            
        Returns:
            plt.Figure: matplotlib図オブジェクト
        """
        # データの準備
        stages = list(lifecycle_correlations.keys())
        correlations = list(lifecycle_correlations.values())
        
        # NaNを除去
        valid_data = [(s, c) for s, c in zip(stages, correlations) if not np.isnan(c)]
        if not valid_data:
            logger.warning("No valid correlation data for lifecycle evolution")
            return None
        
        stages, correlations = zip(*valid_data)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # バープロット
        bars = ax.bar(range(len(stages)), correlations, 
                        color=['green' if c > 0 else 'red' for c in correlations],
                        alpha=0.7)
        
        # ゼロライン
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.8)
        
        # 値のラベル表示
        for bar, corr in zip(bars, correlations):
            height = bar.get_height()
            ax.annotate(f'{corr:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3 if height >= 0 else -15),
                        textcoords="offset points",
                        ha='center', va='bottom' if height >= 0 else 'top',
                        fontsize=10, fontweight='bold')
        
        ax.set_xlabel('Lifecycle Stage', fontsize=12)
        ax.set_ylabel('Correlation Coefficient', fontsize=12)
        ax.set_title(f'Lifecycle Correlation Evolution\n{variable1} vs {variable2}', 
                    fontsize=14)
        ax.set_xticks(range(len(stages)))
        ax.set_xticklabels(stages, rotation=45, ha='right')
        ax.grid(True, axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def create_comprehensive_correlation_dashboard(
        self,
        evaluation_metric: str,
        output_dir: str = "results/visualizations/"
    ) -> Dict[str, str]:
        """
        包括的相関分析ダッシュボード作成
        
        Args:
            evaluation_metric: 分析対象の評価項目
            output_dir: 出力ディレクトリ
            
        Returns:
            Dict[str, str]: 生成されたプロットファイルのパス
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        generated_plots = {}
        
        try:
            # 基本相関分析
            basic_result = self.analyzer.basic_correlation_analysis()
            fig1 = self.plot_correlation_matrix(
                basic_result, 
                f"Overall Correlation Matrix - {evaluation_metric}"
            )
            plot_path1 = output_path / f"correlation_matrix_{evaluation_metric}.png"
            fig1.savefig(plot_path1, dpi=300, bbox_inches='tight')
            generated_plots['correlation_matrix'] = str(plot_path1)
            plt.close(fig1)
            
        except Exception as e:
            logger.error(f"Error creating correlation matrix: {e}")
        
        try:
            # 市場比較分析
            market_comparisons = self.analyzer.market_comparison_correlation(evaluation_metric)
            if market_comparisons:
                fig2 = self.plot_market_comparison_heatmap(
                    market_comparisons, evaluation_metric
                )
                plot_path2 = output_path / f"market_comparison_{evaluation_metric}.png"
                fig2.savefig(plot_path2, dpi=300, bbox_inches='tight')
                generated_plots['market_comparison'] = str(plot_path2)
                plt.close(fig2)
                
        except Exception as e:
            logger.error(f"Error creating market comparison: {e}")
        
        try:
            # ネットワーク分析
            network_data = self.analyzer.correlation_network_analysis()
            if network_data:
                fig3 = self.plot_correlation_network(network_data)
                if fig3:
                    plot_path3 = output_path / f"correlation_network_{evaluation_metric}.png"
                    fig3.savefig(plot_path3, dpi=300, bbox_inches='tight')
                    generated_plots['network'] = str(plot_path3)
                    plt.close(fig3)
                    
        except Exception as e:
            logger.error(f"Error creating network plot: {e}")
        
        logger.info(f"Created {len(generated_plots)} correlation visualizations")
        
        return generated_plots


class AdvancedCorrelationAnalyzer(CorrelationAnalyzer):
    """
    高度な相関分析機能を追加したアナライザー
    """
    
    def __init__(self, data: pd.DataFrame, config: Dict = None):
        super().__init__(data, config)
        self.setup_advanced_methods()
    
    def setup_advanced_methods(self):
        """高度な分析手法の設定"""
        try:
            from sklearn.feature_selection import mutual_info_regression
            from scipy.stats import contingency
            self.mutual_info_available = True
        except ImportError:
            self.mutual_info_available = False
            logger.warning("Mutual information analysis not available")
    
    def mutual_information_analysis(
        self,
        target_variable: str,
        feature_variables: List[str] = None,
        discrete_features: bool = False
    ) -> pd.Series:
        """
        相互情報量による変数間の関係分析
        
        Args:
            target_variable: 目的変数
            feature_variables: 説明変数のリスト
            discrete_features: 離散変数かどうか
            
        Returns:
            pd.Series: 相互情報量スコア
        """
        if not self.mutual_info_available:
            raise ImportError("Mutual information requires sklearn")
        
        from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
        
        if feature_variables is None:
            feature_variables = [col for col in self.data.columns 
                                if col != target_variable and 
                                self.data[col].dtype in ['int64', 'float64']]
        
        # データの準備
        X = self.data[feature_variables].fillna(self.data[feature_variables].mean())
        y = self.data[target_variable].fillna(self.data[target_variable].mean())
        
        # 相互情報量計算
        if discrete_features or y.dtype == 'object':
            mi_scores = mutual_info_classif(X, y)
        else:
            mi_scores = mutual_info_regression(X, y)
        
        return pd.Series(mi_scores, index=feature_variables).sort_values(ascending=False)
    
    def rolling_correlation_analysis(
        self,
        variable1: str,
        variable2: str,
        window_sizes: List[int] = [3, 5, 10],
        market_category: MarketCategory = None
    ) -> pd.DataFrame:
        """
        複数ウィンドウサイズでの移動相関分析
        
        Args:
            variable1: 変数1
            variable2: 変数2
            window_sizes: ウィンドウサイズのリスト
            market_category: 市場カテゴリー
            
        Returns:
            pd.DataFrame: 複数ウィンドウでの相関結果
        """
        results = []
        
        for window_size in window_sizes:
            try:
                rolling_corr = self.time_series_correlation(
                    variable1, variable2, window_size, market_category
                )
                rolling_corr['window_size'] = window_size
                results.append(rolling_corr)
            except Exception as e:
                logger.warning(f"Error in rolling correlation for window {window_size}: {e}")
        
        if results:
            return pd.concat(results, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def conditional_correlation_analysis(
        self,
        variable1: str,
        variable2: str,
        condition_variable: str,
        condition_thresholds: List[float] = None
    ) -> Dict[str, Tuple[float, float]]:
        """
        条件付き相関分析
        
        Args:
            variable1: 変数1
            variable2: 変数2
            condition_variable: 条件変数
            condition_thresholds: 条件閾値のリスト
            
        Returns:
            Dict: 条件別相関結果
        """
        if condition_thresholds is None:
            # 四分位点を使用
            condition_thresholds = self.data[condition_variable].quantile([0.25, 0.5, 0.75]).tolist()
        
        conditional_results = {}
        
        for i, threshold in enumerate(condition_thresholds):
            if i == 0:
                condition_name = f"{condition_variable} <= {threshold:.2f}"
                condition_data = self.data[self.data[condition_variable] <= threshold]
            else:
                prev_threshold = condition_thresholds[i-1]
                condition_name = f"{prev_threshold:.2f} < {condition_variable} <= {threshold:.2f}"
                condition_data = self.data[
                    (self.data[condition_variable] > prev_threshold) & 
                    (self.data[condition_variable] <= threshold)
                ]
            
            if len(condition_data) >= self.config['min_sample_size']:
                try:
                    corr_data = condition_data[[variable1, variable2]].dropna()
                    if len(corr_data) >= 3:
                        corr, p_val = pearsonr(corr_data[variable1], corr_data[variable2])
                        conditional_results[condition_name] = (corr, p_val)
                except Exception as e:
                    logger.warning(f"Error in conditional correlation for {condition_name}: {e}")
        
        # 高値条件も追加
        high_condition_name = f"{condition_variable} > {condition_thresholds[-1]:.2f}"
        high_condition_data = self.data[self.data[condition_variable] > condition_thresholds[-1]]
        
        if len(high_condition_data) >= self.config['min_sample_size']:
            try:
                corr_data = high_condition_data[[variable1, variable2]].dropna()
                if len(corr_data) >= 3:
                    corr, p_val = pearsonr(corr_data[variable1], corr_data[variable2])
                    conditional_results[high_condition_name] = (corr, p_val)
            except Exception as e:
                logger.warning(f"Error in conditional correlation for {high_condition_name}: {e}")
        
        return conditional_results


if __name__ == "__main__":
    main()