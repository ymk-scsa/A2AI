"""
A2AI Factor Visualizer
要因項目の可視化機能を提供するモジュール

このモジュールは120の要因項目（6つの評価項目×各20項目）の分析・可視化機能を提供します。
- 売上高の要因項目（20項目）
- 売上高成長率の要因項目（20項目）
- 売上高営業利益率の要因項目（20項目）
- 売上高当期純利益率の要因項目（20項目）
- ROEの要因項目（20項目）
- 売上高付加価値率の要因項目（20項目）
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
from typing import Dict, List, Tuple, Optional, Union
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch

# 日本語フォント設定
plt.rcParams['font.family'] = ['DejaVu Sans', 'Hiragino Kaku Gothic Pro', 'Yu Gothic', 'Meiryo', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

class FactorVisualizer:
    """
    A2AI要因項目可視化クラス
    
    財務諸表の120の要因項目を効果的に可視化するための
    包括的な可視化機能を提供
    """
    
    def __init__(self):
        """初期化"""
        # カラーパレット定義
        self.colors = {
            'high_share': '#2E8B57',      # 高シェア市場（海緑色）
            'declining': '#FF6347',        # 低下市場（トマト色）
            'lost': '#CD5C5C',            # 失失市場（インディアンレッド）
            'neutral': '#4682B4',          # ニュートラル（スチールブルー）
            'positive': '#32CD32',         # ポジティブ（ライムグリーン）
            'negative': '#DC143C',         # ネガティブ（クリムゾン）
            'accent': '#FFD700',           # アクセント（ゴールド）
            'background': '#F5F5F5'        # 背景（ホワイトスモーク）
        }
        
        # 評価項目と要因項目のマッピング
        self.evaluation_factors = {
            '売上高': [
                '有形固定資産', '設備投資額', '研究開発費', '無形固定資産', '投資有価証券',
                '従業員数', '平均年間給与', '退職給付費用', '福利厚生費',
                '売上債権', '棚卸資産', '総資産', '売上債権回転率', '棚卸資産回転率',
                '海外売上高比率', '事業セグメント数', '販管費', '広告宣伝費', '営業外収益', '受注残高'
            ],
            '売上高成長率': [
                '設備投資増加率', '研究開発費増加率', '有形固定資産増加率', '無形固定資産増加率', '総資産増加率',
                '従業員数増加率', '平均年間給与増加率', '人件費増加率', '退職給付費用増加率',
                '海外売上高比率変化', 'セグメント別売上高増加率', '販管費増加率', '広告宣伝費増加率', '営業外収益増加率',
                '売上債権増加率', '棚卸資産増加率', '売上債権回転率変化', '棚卸資産回転率変化', '総資産回転率変化', '受注残高増加率'
            ],
            '売上高営業利益率': [
                '材料費率', '労務費率', '経費率', '外注加工費率', '減価償却費率',
                '販管費率', '人件費率', '広告宣伝費率', '研究開発費率', '減価償却費率',
                '売上高付加価値率', '労働生産性', '設備効率性', '総資産回転率', '棚卸資産回転率',
                '売上高', '固定費率', '変動費率', '海外売上高比率', '事業セグメント集中度'
            ],
            '売上高当期純利益率': [
                '売上高営業利益率', '販管費率', '売上原価率', '研究開発費率', '減価償却費率',
                '受取利息・配当金', '支払利息', '為替差損益', '持分法投資損益', '営業外収益率',
                '特別利益', '特別損失', '法人税等実効税率', '法人税等調整額', '税引前当期純利益率',
                '有利子負債比率', '自己資本比率', '投資有価証券評価損益', '固定資産売却損益', '減損損失率'
            ],
            'ROE': [
                '売上高当期純利益率', '総資産回転率', '売上高営業利益率', '売上原価率', '販管費率',
                '自己資本比率', '総資産/自己資本倍率', '有利子負債/自己資本比率', '流動比率', '固定比率',
                '売上債権回転率', '棚卸資産回転率', '有形固定資産回転率', '現金及び預金/総資産比率', '投資有価証券/総資産比率',
                '配当性向', '内部留保率', '営業外収益率', '特別損益/当期純利益比率', '実効税率'
            ],
            '売上高付加価値率': [
                '研究開発費率', '無形固定資産/売上高比率', '特許関連費用', 'ソフトウェア/売上高比率', '技術ライセンス収入',
                '平均年間給与/業界平均比率', '人件費率', '従業員数/売上高比率', '退職給付費用率', '福利厚生費率',
                '売上原価率', '材料費率', '外注加工費率', '労働生産性', '設備生産性',
                '海外売上高比率', '高付加価値事業セグメント比率', 'サービス・保守収入比率', '営業利益率', 'ブランド・商標等無形資産比率'
            ]
        }
        
        # 要因項目のカテゴリ分類
        self.factor_categories = {
            '投資・資産関連': ['有形固定資産', '設備投資額', '研究開発費', '無形固定資産', '投資有価証券', '総資産'],
            '人的資源関連': ['従業員数', '平均年間給与', '退職給付費用', '福利厚生費', '人件費率'],
            '運転資本関連': ['売上債権', '棚卸資産', '売上債権回転率', '棚卸資産回転率'],
            '事業展開関連': ['海外売上高比率', '事業セグメント数', '販管費', '広告宣伝費', '営業外収益'],
            'コスト構造関連': ['売上原価率', '材料費率', '労務費率', '経費率', '外注加工費率'],
            '財務構造関連': ['自己資本比率', '有利子負債比率', '流動比率', '固定比率'],
            '効率性関連': ['労働生産性', '設備効率性', '総資産回転率'],
            '収益性関連': ['売上高営業利益率', '売上高当期純利益率', 'ROE']
        }

    def plot_factor_importance(self, data: pd.DataFrame, target_column: str, 
                                evaluation_metric: str, top_n: int = 15,
                                figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        要因項目の重要度を可視化
        
        Parameters:
        -----------
        data : pd.DataFrame
            財務データ
        target_column : str
            目的変数（評価項目）
        evaluation_metric : str
            評価指標名（'売上高', '売上高成長率', など）
        top_n : int
            表示する上位要因数
        figsize : tuple
            図のサイズ
            
        Returns:
        --------
        plt.Figure
            重要度プロット
        """
        # 要因項目を取得
        factors = self.evaluation_factors.get(evaluation_metric, [])
        available_factors = [f for f in factors if f in data.columns]
        
        if not available_factors:
            raise ValueError(f"評価指標 '{evaluation_metric}' に対応する要因項目がデータに見つかりません")
        
        # 相関係数を計算
        correlations = []
        for factor in available_factors:
            if data[factor].notna().sum() > 10:  # 最低10のサンプルが必要
                corr = data[target_column].corr(data[factor])
                if not pd.isna(corr):
                    correlations.append({
                        'factor': factor,
                        'correlation': abs(corr),
                        'direction': 'positive' if corr > 0 else 'negative'
                    })
        
        # 相関の強さでソート
        correlations_df = pd.DataFrame(correlations)
        correlations_df = correlations_df.sort_values('correlation', ascending=True).tail(top_n)
        
        # プロット作成
        fig, ax = plt.subplots(figsize=figsize)
        
        # バーの色を方向性に応じて設定
        colors = [self.colors['positive'] if direction == 'positive' 
                    else self.colors['negative'] for direction in correlations_df['direction']]
        
        bars = ax.barh(range(len(correlations_df)), correlations_df['correlation'], 
                        color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
        
        # カスタマイズ
        ax.set_yticks(range(len(correlations_df)))
        ax.set_yticklabels(correlations_df['factor'])
        ax.set_xlabel('相関係数の絶対値', fontsize=12)
        ax.set_title(f'{evaluation_metric} - 要因項目重要度 (上位{top_n}項目)', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # グリッドとスタイル
        ax.grid(True, axis='x', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        # 値をバーの上に表示
        for i, (bar, corr) in enumerate(zip(bars, correlations_df['correlation'])):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{corr:.3f}', va='center', ha='left', fontsize=9)
        
        # 凡例追加
        positive_patch = patches.Patch(color=self.colors['positive'], label='正の相関')
        negative_patch = patches.Patch(color=self.colors['negative'], label='負の相関')
        ax.legend(handles=[positive_patch, negative_patch], loc='lower right')
        
        plt.tight_layout()
        return fig

    def plot_factor_correlation_matrix(self, data: pd.DataFrame, evaluation_metric: str,
                                        figsize: Tuple[int, int] = (14, 12)) -> plt.Figure:
        """
        要因項目間の相関マトリックスを可視化
        
        Parameters:
        -----------
        data : pd.DataFrame
            財務データ
        evaluation_metric : str
            評価指標名
        figsize : tuple
            図のサイズ
            
        Returns:
        --------
        plt.Figure
            相関マトリックスヒートマップ
        """
        factors = self.evaluation_factors.get(evaluation_metric, [])
        available_factors = [f for f in factors if f in data.columns and data[f].notna().sum() > 10]
        
        if len(available_factors) < 2:
            raise ValueError(f"相関分析に必要な要因項目が不足しています（最低2項目必要）")
        
        # 相関マトリックス計算
        correlation_matrix = data[available_factors].corr()
        
        # プロット作成
        fig, ax = plt.subplots(figsize=figsize)
        
        # ヒートマップ作成
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
                    square=True, fmt='.2f', cbar_kws={'shrink': 0.8}, ax=ax)
        
        ax.set_title(f'{evaluation_metric} - 要因項目相関マトリックス', 
                    fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        return fig

    def plot_factor_distribution_by_market(self, data: pd.DataFrame, factor: str, 
                                            market_column: str = 'market_category',
                                            figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        市場カテゴリ別の要因項目分布を可視化
        
        Parameters:
        -----------
        data : pd.DataFrame
            財務データ
        factor : str
            要因項目名
        market_column : str
            市場カテゴリ列名
        figsize : tuple
            図のサイズ
            
        Returns:
        --------
        plt.Figure
            分布プロット
        """
        if factor not in data.columns:
            raise ValueError(f"要因項目 '{factor}' がデータに見つかりません")
        
        if market_column not in data.columns:
            raise ValueError(f"市場カテゴリ列 '{market_column}' がデータに見つかりません")
        
        # データ準備
        plot_data = data[[factor, market_column]].dropna()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # 1. ヒストグラム（市場別）
        markets = plot_data[market_column].unique()
        colors_map = {'高シェア': self.colors['high_share'], 
                        '低下': self.colors['declining'], 
                        '失失': self.colors['lost']}
        
        for market in markets:
            market_data = plot_data[plot_data[market_column] == market][factor]
            ax1.hist(market_data, alpha=0.7, label=f'{market}市場', 
                    color=colors_map.get(market, self.colors['neutral']), bins=20)
        
        ax1.set_xlabel(factor)
        ax1.set_ylabel('頻度')
        ax1.set_title(f'{factor} - 市場別分布')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. ボックスプロット
        box_data = [plot_data[plot_data[market_column] == market][factor].values 
                    for market in markets]
        
        bp = ax2.boxplot(box_data, labels=markets, patch_artist=True)
        
        # ボックスプロットの色設定
        for patch, market in zip(bp['boxes'], markets):
            patch.set_facecolor(colors_map.get(market, self.colors['neutral']))
            patch.set_alpha(0.7)
        
        ax2.set_ylabel(factor)
        ax2.set_title(f'{factor} - 市場別ボックスプロット')
        ax2.grid(True, alpha=0.3)
        
        # 統計的検定結果を追加
        if len(markets) == 2:
            market_data1 = plot_data[plot_data[market_column] == markets[0]][factor]
            market_data2 = plot_data[plot_data[market_column] == markets[1]][factor]
            
            # t検定実行
            try:
                t_stat, p_value = stats.ttest_ind(market_data1.dropna(), market_data2.dropna())
                ax2.text(0.02, 0.98, f'p-value: {p_value:.4f}', 
                        transform=ax2.transAxes, va='top', fontsize=10,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            except:
                pass
        
        plt.tight_layout()
        return fig

    def plot_factor_time_series(self, data: pd.DataFrame, factors: List[str], 
                                time_column: str = 'year', company_column: str = 'company',
                                companies: Optional[List[str]] = None,
                                figsize: Tuple[int, int] = (14, 10)) -> plt.Figure:
        """
        要因項目の時系列変化を可視化
        
        Parameters:
        -----------
        data : pd.DataFrame
            財務データ
        factors : list
            可視化する要因項目のリスト
        time_column : str
            時間軸列名
        company_column : str
            企業名列名
        companies : list, optional
            表示する企業のリスト（Noneの場合は全企業）
        figsize : tuple
            図のサイズ
            
        Returns:
        --------
        plt.Figure
            時系列プロット
        """
        if not all(factor in data.columns for factor in factors):
            missing = [f for f in factors if f not in data.columns]
            raise ValueError(f"要因項目がデータに見つかりません: {missing}")
        
        # データ準備
        plot_data = data.copy()
        if companies:
            plot_data = plot_data[plot_data[company_column].isin(companies)]
        
        n_factors = len(factors)
        fig, axes = plt.subplots(n_factors, 1, figsize=figsize, sharex=True)
        if n_factors == 1:
            axes = [axes]
        
        for i, factor in enumerate(factors):
            # 各企業の時系列データをプロット
            for company in plot_data[company_column].unique():
                company_data = plot_data[plot_data[company_column] == company]
                if len(company_data) > 1:  # 複数年のデータがある企業のみ
                    axes[i].plot(company_data[time_column], company_data[factor], 
                                alpha=0.6, linewidth=1, label=company if i == 0 else "")
            
            axes[i].set_ylabel(factor)
            axes[i].set_title(f'{factor} - 時系列推移')
            axes[i].grid(True, alpha=0.3)
            
            # 業界平均線を追加
            industry_avg = plot_data.groupby(time_column)[factor].mean()
            axes[i].plot(industry_avg.index, industry_avg.values, 
                        color='red', linewidth=3, label='業界平均', alpha=0.8)
        
        axes[-1].set_xlabel('年度')
        
        # 凡例は最初のサブプロットにのみ表示
        if companies and len(companies) <= 10:
            axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        return fig

    def plot_factor_pca_analysis(self, data: pd.DataFrame, evaluation_metric: str,
                                market_column: str = 'market_category',
                                figsize: Tuple[int, int] = (14, 10)) -> plt.Figure:
        """
        要因項目の主成分分析による次元削減可視化
        
        Parameters:
        -----------
        data : pd.DataFrame
            財務データ
        evaluation_metric : str
            評価指標名
        market_column : str
            市場カテゴリ列名
        figsize : tuple
            図のサイズ
            
        Returns:
        --------
        plt.Figure
            PCA散布図
        """
        factors = self.evaluation_factors.get(evaluation_metric, [])
        available_factors = [f for f in factors if f in data.columns and data[f].notna().sum() > 10]
        
        if len(available_factors) < 3:
            raise ValueError("PCA分析には最低3つの要因項目が必要です")
        
        # データ準備
        pca_data = data[available_factors + [market_column]].dropna()
        X = pca_data[available_factors]
        
        # 標準化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # PCA実行
        pca = PCA(n_components=min(len(available_factors), 10))
        X_pca = pca.fit_transform(X_scaled)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        
        # 1. 散布図（PC1 vs PC2）
        colors_map = {'高シェア': self.colors['high_share'], 
                        '低下': self.colors['declining'], 
                        '失失': self.colors['lost']}
        
        for market in pca_data[market_column].unique():
            mask = pca_data[market_column] == market
            ax1.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                        c=colors_map.get(market, self.colors['neutral']),
                        label=f'{market}市場', alpha=0.7, s=50)
        
        ax1.set_xlabel(f'第1主成分 (寄与率: {pca.explained_variance_ratio_[0]:.1%})')
        ax1.set_ylabel(f'第2主成分 (寄与率: {pca.explained_variance_ratio_[1]:.1%})')
        ax1.set_title('PCA散布図 (PC1 vs PC2)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 寄与率
        ax2.bar(range(1, len(pca.explained_variance_ratio_) + 1), 
                pca.explained_variance_ratio_, alpha=0.7, color=self.colors['neutral'])
        ax2.set_xlabel('主成分')
        ax2.set_ylabel('寄与率')
        ax2.set_title('各主成分の寄与率')
        ax2.grid(True, alpha=0.3)
        
        # 3. 累積寄与率
        cumsum_ratio = np.cumsum(pca.explained_variance_ratio_)
        ax3.plot(range(1, len(cumsum_ratio) + 1), cumsum_ratio, 
                'o-', color=self.colors['accent'], linewidth=2, markersize=6)
        ax3.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='80%ライン')
        ax3.set_xlabel('主成分')
        ax3.set_ylabel('累積寄与率')
        ax3.set_title('累積寄与率')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 第1主成分への要因項目の寄与
        pc1_contributions = pca.components_[0]
        contribution_df = pd.DataFrame({
            'factor': available_factors,
            'contribution': pc1_contributions
        }).sort_values('contribution', key=abs, ascending=True)
        
        colors = [self.colors['positive'] if x > 0 else self.colors['negative'] 
                    for x in contribution_df['contribution']]
        
        ax4.barh(range(len(contribution_df)), contribution_df['contribution'], 
                color=colors, alpha=0.7)
        ax4.set_yticks(range(len(contribution_df)))
        ax4.set_yticklabels(contribution_df['factor'])
        ax4.set_xlabel('第1主成分への寄与')
        ax4.set_title('第1主成分 - 要因項目寄与度')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig

    def create_factor_summary_dashboard(self, data: pd.DataFrame, evaluation_metric: str,
                                        market_column: str = 'market_category') -> go.Figure:
        """
        要因項目の包括的ダッシュボードを作成（Plotly）
        
        Parameters:
        -----------
        data : pd.DataFrame
            財務データ
        evaluation_metric : str
            評価指標名
        market_column : str
            市場カテゴリ列名
            
        Returns:
        --------
        plotly.graph_objects.Figure
            インタラクティブダッシュボード
        """
        factors = self.evaluation_factors.get(evaluation_metric, [])
        available_factors = [f for f in factors if f in data.columns and data[f].notna().sum() > 10]
        
        if not available_factors:
            raise ValueError(f"評価指標 '{evaluation_metric}' に対応する要因項目がデータに見つかりません")
        
        # サブプロット作成（2x2レイアウト）
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('要因項目重要度', '市場別分布', '時系列トレンド', '相関ネットワーク'),
            specs=[[{"type": "bar"}, {"type": "box"}],
                    [{"type": "scatter"}, {"type": "scatter"}]]
        )
        
        # 1. 要因項目重要度（左上）
        target_col = f'{evaluation_metric}_value'  # 仮の目的変数列名
        if target_col in data.columns:
            correlations = []
            for factor in available_factors[:10]:  # 上位10項目のみ表示
                corr = data[target_col].corr(data[factor])
                if not pd.isna(corr):
                    correlations.append({'factor': factor, 'correlation': abs(corr)})
            
            if correlations:
                corr_df = pd.DataFrame(correlations).sort_values('correlation', ascending=True)
                
                fig.add_trace(
                    go.Bar(
                        y=corr_df['factor'],
                        x=corr_df['correlation'],
                        orientation='h',
                        name='重要度',
                        marker_color=self.colors['neutral']
                    ),
                    row=1, col=1
                )
        
        # 2. 市場別分布（右上）
        if len(available_factors) > 0 and market_column in data.columns:
            sample_factor = available_factors[0]
            markets = data[market_column].unique()
            
            for market in markets:
                market_data = data[data[market_column] == market][sample_factor]
                fig.add_trace(
                    go.Box(
                        y=market_data,
                        name=f'{market}市場',
                        boxpoints='outliers'
                    ),
                    row=1, col=2
                )
        
        # 3. 時系列トレンド（左下）
        if 'year' in data.columns and len(available_factors) > 0:
            sample_factor = available_factors[0]
            yearly_avg = data.groupby('year')[sample_factor].mean()
            
            fig.add_trace(
                go.Scatter(
                    x=yearly_avg.index,
                    y=yearly_avg.values,
                    mode='lines+markers',
                    name='平均値推移',
                    line=dict(color=self.colors['accent'], width=3)
                ),
                row=2, col=1
            )
        
        # 4. 相関ネットワーク（右下）
        if len(available_factors) >= 3:
            # 上位5項目で相関計算
            top_factors = available_factors[:5]
            corr_matrix = data[top_factors].corr()
            
            # 強い相関を持つペアを抽出
            strong_corrs = []
            for i in range(len(top_factors)):
                for j in range(i+1, len(top_factors)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.3:  # 閾値0.3以上
                        strong_corrs.append({
                            'factor1': top_factors[i],
                            'factor2': top_factors[j],
                            'correlation': corr_val
                        })
            
            # ネットワーク描画用の座標計算
            n_factors = len(top_factors)
            angles = np.linspace(0, 2*np.pi, n_factors, endpoint=False)
            x_pos = np.cos(angles)
            y_pos = np.sin(angles)
            
            # ノード（要因項目）をプロット
            fig.add_trace(
                go.Scatter(
                    x=x_pos, y=y_pos,
                    mode='markers+text',
                    marker=dict(size=15, color=self.colors['neutral']),
                    text=[f[:10] + '...' if len(f) > 10 else f for f in top_factors],
                    textposition='middle center',
                    name='要因項目'
                ),
                row=2, col=2
            )
            
            # エッジ（相関）をプロット
            for corr_info in strong_corrs:
                i = top_factors.index(corr_info['factor1'])
                j = top_factors.index(corr_info['factor2'])
                
                color = self.colors['positive'] if corr_info['correlation'] > 0 else self.colors['negative']
                width = abs(corr_info['correlation']) * 5
                
                fig.add_trace(
                    go.Scatter(
                        x=[x_pos[i], x_pos[j]],
                        y=[y_pos[i], y_pos[j]],
                        mode='lines',
                        line=dict(color=color, width=width),
                        showlegend=False
                    ),
                    row=2, col=2
                )
        
        # レイアウト設定
        fig.update_layout(
            height=800,
            title_text=f"{evaluation_metric} - 要因項目分析ダッシュボード",
            title_x=0.5,
            showlegend=True
        )
        
        # 各サブプロットの軸設定
        fig.update_xaxes(title_text="相関係数", row=1, col=1)
        fig.update_yaxes(title_text="要因項目", row=1, col=1)
        
        fig.update_yaxes(title_text="値", row=1, col=2)
        
        fig.update_xaxes(title_text="年度", row=2, col=1)
        fig.update_yaxes(title_text="平均値", row=2, col=1)
        
        fig.update_xaxes(title_text="", showticklabels=False, row=2, col=2)
        fig.update_yaxes(title_text="", showticklabels=False, row=2, col=2)
        
        return fig

    def plot_factor_category_comparison(self, data: pd.DataFrame, 
                                        market_column: str = 'market_category',
                                        figsize: Tuple[int, int] = (16, 12)) -> plt.Figure:
        """
        要因項目カテゴリ別の市場比較分析
        
        Parameters:
        -----------
        data : pd.DataFrame
            財務データ
        market_column : str
            市場カテゴリ列名
        figsize : tuple
            図のサイズ
            
        Returns:
        --------
        plt.Figure
            カテゴリ比較プロット
        """
        if market_column not in data.columns:
            raise ValueError(f"市場カテゴリ列 '{market_column}' がデータに見つかりません")
        
        n_categories = len(self.factor_categories)
        fig, axes = plt.subplots(2, 4, figsize=figsize)
        axes = axes.flatten()
        
        colors_map = {'高シェア': self.colors['high_share'], 
                        '低下': self.colors['declining'], 
                        '失失': self.colors['lost']}
        
        for idx, (category, factors) in enumerate(self.factor_categories.items()):
            if idx >= len(axes):
                break
                
            ax = axes[idx]
            
            # カテゴリ内の利用可能な要因項目を取得
            available_factors = [f for f in factors if f in data.columns]
            if not available_factors:
                ax.text(0.5, 0.5, f'データなし\n({category})', 
                        ha='center', va='center', transform=ax.transAxes)
                continue
            
            # カテゴリ内要因項目の平均値を計算
            category_data = data[available_factors + [market_column]].dropna()
            category_scores = category_data.groupby(market_column)[available_factors].mean().mean(axis=1)
            
            # バープロット
            markets = category_scores.index
            bars = ax.bar(markets, category_scores.values,
                            color=[colors_map.get(m, self.colors['neutral']) for m in markets],
                            alpha=0.7, edgecolor='black', linewidth=0.5)
            
            # 値をバーの上に表示
            for bar, value in zip(bars, category_scores.values):
                if not pd.isna(value):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                            f'{value:.2f}', ha='center', va='bottom', fontsize=9)
            
            ax.set_title(f'{category}', fontsize=11, fontweight='bold')
            ax.set_ylabel('平均スコア')
            ax.grid(True, alpha=0.3, axis='y')
            ax.tick_params(axis='x', rotation=45)
        
        # 未使用のサブプロットを非表示
        for idx in range(n_categories, len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle('要因項目カテゴリ別 - 市場比較分析', fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        return fig

    def create_factor_radar_chart(self, data: pd.DataFrame, companies: List[str],
                                evaluation_metric: str, top_n: int = 8) -> go.Figure:
        """
        企業別要因項目レーダーチャート作成
        
        Parameters:
        -----------
        data : pd.DataFrame
            財務データ
        companies : list
            比較対象企業リスト
        evaluation_metric : str
            評価指標名
        top_n : int
            表示する要因項目数
            
        Returns:
        --------
        plotly.graph_objects.Figure
            レーダーチャート
        """
        if 'company' not in data.columns:
            raise ValueError("企業名列 'company' がデータに見つかりません")
        
        factors = self.evaluation_factors.get(evaluation_metric, [])
        available_factors = [f for f in factors if f in data.columns and data[f].notna().sum() > 10]
        
        if len(available_factors) < 3:
            raise ValueError("レーダーチャート作成には最低3つの要因項目が必要です")
        
        # 上位n項目を選択（相関の強さで）
        target_col = f'{evaluation_metric}_value'
        if target_col not in data.columns:
            selected_factors = available_factors[:top_n]
        else:
            correlations = [(f, abs(data[target_col].corr(data[f]))) 
                            for f in available_factors if not pd.isna(data[target_col].corr(data[f]))]
            correlations.sort(key=lambda x: x[1], reverse=True)
            selected_factors = [f[0] for f in correlations[:top_n]]
        
        # データ準備
        company_data = {}
        for company in companies:
            if company in data['company'].values:
                company_values = data[data['company'] == company][selected_factors].mean()
                # 0-1正規化
                normalized_values = ((company_values - company_values.min()) / 
                                    (company_values.max() - company_values.min())).fillna(0)
                company_data[company] = normalized_values
        
        # レーダーチャート作成
        fig = go.Figure()
        
        colors = [self.colors['high_share'], self.colors['declining'], self.colors['lost'], 
                    self.colors['neutral'], self.colors['accent']]
        
        for i, (company, values) in enumerate(company_data.items()):
            fig.add_trace(go.Scatterpolar(
                r=values.tolist() + [values.iloc[0]],  # 最初の値を最後に追加して閉じる
                theta=selected_factors + [selected_factors[0]],
                fill='toself',
                fillcolor=colors[i % len(colors)],
                line=dict(color=colors[i % len(colors)]),
                name=company,
                opacity=0.6
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=True,
            title=f"{evaluation_metric} - 企業別要因項目比較（レーダーチャート）"
        )
        
        return fig

    def generate_factor_insights_report(self, data: pd.DataFrame, evaluation_metric: str,
                                        market_column: str = 'market_category') -> Dict[str, any]:
        """
        要因項目分析のインサイトレポートを生成
        
        Parameters:
        -----------
        data : pd.DataFrame
            財務データ
        evaluation_metric : str
            評価指標名
        market_column : str
            市場カテゴリ列名
            
        Returns:
        --------
        dict
            分析インサイト辞書
        """
        factors = self.evaluation_factors.get(evaluation_metric, [])
        available_factors = [f for f in factors if f in data.columns and data[f].notna().sum() > 10]
        
        if not available_factors:
            return {'error': f'評価指標 {evaluation_metric} に対応する要因項目が見つかりません'}
        
        insights = {
            'evaluation_metric': evaluation_metric,
            'total_factors': len(available_factors),
            'analysis_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # 1. 要因項目基本統計
        factor_stats = data[available_factors].describe()
        insights['factor_statistics'] = {
            'most_variable': factor_stats.loc['std'].idxmax(),
            'least_variable': factor_stats.loc['std'].idxmin(),
            'highest_mean': factor_stats.loc['mean'].idxmax(),
            'lowest_mean': factor_stats.loc['mean'].idxmin()
        }
        
        # 2. 市場別差異分析
        if market_column in data.columns:
            market_differences = {}
            for factor in available_factors[:10]:  # 上位10項目のみ
                market_means = data.groupby(market_column)[factor].mean()
                if len(market_means) >= 2:
                    max_market = market_means.idxmax()
                    min_market = market_means.idxmin()
                    difference = market_means.max() - market_means.min()
                    market_differences[factor] = {
                        'max_market': max_market,
                        'min_market': min_market,
                        'difference': difference,
                        'coefficient_of_variation': market_means.std() / market_means.mean()
                    }
            
            # 市場間差異が最も大きい要因項目
            if market_differences:
                max_diff_factor = max(market_differences.keys(), 
                                    key=lambda x: market_differences[x]['difference'])
                insights['market_analysis'] = {
                    'most_differentiating_factor': max_diff_factor,
                    'market_differences': market_differences
                }
        
        # 3. 要因項目間相関分析
        correlation_matrix = data[available_factors].corr()
        
        # 強い正の相関ペア
        strong_positive_corrs = []
        # 強い負の相関ペア
        strong_negative_corrs = []
        
        for i in range(len(available_factors)):
            for j in range(i+1, len(available_factors)):
                corr_val = correlation_matrix.iloc[i, j]
                if not pd.isna(corr_val):
                    if corr_val > 0.7:
                        strong_positive_corrs.append({
                            'factor1': available_factors[i],
                            'factor2': available_factors[j],
                            'correlation': corr_val
                        })
                    elif corr_val < -0.7:
                        strong_negative_corrs.append({
                            'factor1': available_factors[i],
                            'factor2': available_factors[j],
                            'correlation': corr_val
                        })
        
        insights['correlation_analysis'] = {
            'strong_positive_correlations': strong_positive_corrs[:5],  # 上位5ペア
            'strong_negative_correlations': strong_negative_corrs[:5],  # 上位5ペア
            'total_strong_correlations': len(strong_positive_corrs) + len(strong_negative_corrs)
        }
        
        # 4. 時系列トレンド分析（年度データがある場合）
        if 'year' in data.columns:
            trend_analysis = {}
            for factor in available_factors[:10]:  # 上位10項目
                yearly_data = data.groupby('year')[factor].mean()
                if len(yearly_data) >= 3:
                    # 線形トレンドを計算
                    x = np.arange(len(yearly_data))
                    y = yearly_data.values
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                    
                    trend_analysis[factor] = {
                        'slope': slope,
                        'r_squared': r_value**2,
                        'p_value': p_value,
                        'trend': 'increasing' if slope > 0 else 'decreasing',
                        'strength': 'strong' if abs(r_value) > 0.7 else 'moderate' if abs(r_value) > 0.4 else 'weak'
                    }
            
            # 最も強いトレンドを持つ要因項目
            if trend_analysis:
                strongest_trend = max(trend_analysis.keys(), 
                                    key=lambda x: abs(trend_analysis[x]['slope']))
                insights['trend_analysis'] = {
                    'strongest_trend_factor': strongest_trend,
                    'trend_details': trend_analysis
                }
        
        # 5. 推奨事項
        recommendations = []
        
        if 'market_analysis' in insights:
            most_diff_factor = insights['market_analysis']['most_differentiating_factor']
            recommendations.append(f"市場間差異が最も大きい'{most_diff_factor}'を重点分析項目として検討")
        
        if strong_positive_corrs:
            recommendations.append(f"強い正の相関を持つ要因項目ペアが{len(strong_positive_corrs)}組発見されました。多重共線性に注意")
        
        if 'trend_analysis' in insights:
            strongest_factor = insights['trend_analysis']['strongest_trend_factor']
            trend_direction = insights['trend_analysis']['trend_details'][strongest_factor]['trend']
            recommendations.append(f"'{strongest_factor}'は{trend_direction}傾向が強く、将来予測の重要指標として活用可能")
        
        insights['recommendations'] = recommendations
        
        return insights

# 使用例とテスト用のヘルパー関数
def create_sample_financial_data(n_companies: int = 50, n_years: int = 10) -> pd.DataFrame:
    """
    テスト用のサンプル財務データを生成
    
    Parameters:
    -----------
    n_companies : int
        企業数
    n_years : int
        年数
        
    Returns:
    --------
    pd.DataFrame
        サンプルデータ
    """
    np.random.seed(42)
    
    # 企業と年度の組み合わせ
    companies = [f'企業{i:02d}' for i in range(1, n_companies + 1)]
    years = list(range(2015, 2015 + n_years))
    
    data = []
    
    for company in companies:
        # 市場カテゴリをランダムに割り当て
        market_category = np.random.choice(['高シェア', '低下', '失失'], p=[0.3, 0.4, 0.3])
        
        for year in years:
            row = {
                'company': company,
                'year': year,
                'market_category': market_category
            }
            
            # 売上高関連要因項目のサンプルデータ
            base_value = np.random.normal(1000, 200) if market_category == '高シェア' else np.random.normal(500, 150)
            
            row.update({
                '有形固定資産': base_value * np.random.uniform(0.3, 0.7),
                '設備投資額': base_value * np.random.uniform(0.05, 0.15),
                '研究開発費': base_value * np.random.uniform(0.03, 0.12),
                '無形固定資産': base_value * np.random.uniform(0.1, 0.3),
                '投資有価証券': base_value * np.random.uniform(0.05, 0.25),
                '従業員数': np.random.normal(1000, 300),
                '平均年間給与': np.random.normal(600, 100),
                '退職給付費用': base_value * np.random.uniform(0.01, 0.03),
                '福利厚生費': base_value * np.random.uniform(0.02, 0.05),
                '売上債権': base_value * np.random.uniform(0.1, 0.3),
                '棚卸資産': base_value * np.random.uniform(0.05, 0.2),
                '総資産': base_value * np.random.uniform(1.5, 3.0),
                '売上債権回転率': np.random.uniform(4, 12),
                '棚卸資産回転率': np.random.uniform(6, 20),
                '海外売上高比率': np.random.uniform(0.1, 0.8),
                '事業セグメント数': np.random.randint(1, 8),
                '販管費': base_value * np.random.uniform(0.15, 0.35),
                '広告宣伝費': base_value * np.random.uniform(0.01, 0.05),
                '営業外収益': base_value * np.random.uniform(0.01, 0.03),
                '受注残高': base_value * np.random.uniform(0.1, 0.4),
                # 評価項目のサンプル値
                '売上高_value': base_value,
                '売上高成長率_value': np.random.normal(0.05, 0.1),
                '売上高営業利益率_value': np.random.normal(0.08, 0.05),
                '売上高当期純利益率_value': np.random.normal(0.05, 0.04),
                'ROE_value': np.random.normal(0.1, 0.06),
                '売上高付加価値率_value': np.random.normal(0.3, 0.1)
            })
            
            data.append(row)
    
    return pd.DataFrame(data)

# テスト実行例
if __name__ == "__main__":
    # サンプルデータ生成
    sample_data = create_sample_financial_data(30, 8)
    
    # ビジュアライザー初期化
    visualizer = FactorVisualizer()
    
    # 各種可視化の実行例
    try:
        # 1. 要因項目重要度
        fig1 = visualizer.plot_factor_importance(
            sample_data, '売上高_value', '売上高', top_n=10
        )
        print("要因項目重要度プロット作成完了")
        
        # 2. 相関マトリックス
        fig2 = visualizer.plot_factor_correlation_matrix(
            sample_data, '売上高'
        )
        print("相関マトリックス作成完了")
        
        # 3. 市場別分布
        fig3 = visualizer.plot_factor_distribution_by_market(
            sample_data, '研究開発費'
        )
        print("市場別分布プロット作成完了")
        
        # 4. インサイトレポート生成
        insights = visualizer.generate_factor_insights_report(
            sample_data, '売上高'
        )
        print("インサイトレポート生成完了")
        print(f"分析対象要因項目数: {insights.get('total_factors', 0)}")
        
        # 5. インタラクティブダッシュボード
        dashboard = visualizer.create_factor_summary_dashboard(
            sample_data, '売上高'
        )
        print("インタラクティブダッシュボード作成完了")
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")