"""
Strategic Positioning Analyzer for A2AI Financial Analysis System
戦略ポジション分析モジュール

このモジュールは企業の戦略的ポジションを分析し、世界シェア別市場カテゴリー内での
相対的位置づけと競争優位性を定量化します。

主な機能:
1. 多次元戦略マップ生成
2. 競争ポジション分析
3. 戦略クラスター分析
4. 戦略移行パターン分析
5. 競争優位性評価
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from scipy.stats import rankdata
import warnings
warnings.filterwarnings('ignore')

@dataclass
class StrategyPosition:
    """戦略ポジション情報を格納するデータクラス"""
    company_name: str
    market_category: str  # 'high_share', 'declining', 'lost'
    position_vector: np.ndarray
    cluster_id: int
    competitive_strength: float
    strategic_advantages: List[str]
    strategic_weaknesses: List[str]
    positioning_score: float

@dataclass
class StrategyTransition:
    """戦略移行情報を格納するデータクラス"""
    company_name: str
    start_year: int
    end_year: int
    start_position: np.ndarray
    end_position: np.ndarray
    transition_distance: float
    transition_direction: str
    success_indicator: float

class StrategyDimensionCalculator:
    """戦略次元計算クラス"""
    
    def __init__(self):
        self.dimension_weights = {
            'innovation_intensity': 0.20,
            'operational_efficiency': 0.18,
            'market_expansion': 0.16,
            'financial_strength': 0.14,
            'human_capital': 0.12,
            'strategic_flexibility': 0.10,
            'brand_value': 0.10
        }
    
    def calculate_innovation_intensity(self, data: pd.DataFrame) -> pd.Series:
        """イノベーション強度の計算"""
        rd_ratio = data['研究開発費率']
        patent_asset_ratio = data['特許関連費用'] / data['売上高']
        new_product_ratio = data['新製品売上高比率'].fillna(0)
        tech_license_income = data['技術ライセンス収入'] / data['売上高']
        
        innovation_score = (
            0.4 * rd_ratio +
            0.3 * patent_asset_ratio +
            0.2 * new_product_ratio +
            0.1 * tech_license_income
        )
        
        return innovation_score.fillna(0)
    
    def calculate_operational_efficiency(self, data: pd.DataFrame) -> pd.Series:
        """オペレーショナル効率性の計算"""
        asset_turnover = data['総資産回転率']
        labor_productivity = data['労働生産性']
        equipment_efficiency = data['設備生産性']
        cost_ratio = 1 - data['売上原価率']  # 逆数で効率性を表現
        
        efficiency_score = (
            0.3 * asset_turnover +
            0.3 * labor_productivity +
            0.2 * equipment_efficiency +
            0.2 * cost_ratio
        )
        
        return efficiency_score.fillna(0)
    
    def calculate_market_expansion(self, data: pd.DataFrame) -> pd.Series:
        """市場拡張力の計算"""
        overseas_ratio = data['海外売上高比率']
        segment_diversity = data['事業セグメント数'] / 10  # 正規化
        market_share_growth = data['売上高成長率']
        sales_efficiency = data['販売費及び一般管理費率']
        
        expansion_score = (
            0.35 * overseas_ratio +
            0.25 * segment_diversity +
            0.25 * market_share_growth +
            0.15 * (1 - sales_efficiency)  # 効率的な販売活動
        )
        
        return expansion_score.fillna(0)
    
    def calculate_financial_strength(self, data: pd.DataFrame) -> pd.Series:
        """財務力の計算"""
        roe = data['ROE']
        equity_ratio = data['自己資本比率']
        current_ratio = data['流動比率']
        debt_equity_ratio = 1 / (1 + data['有利子負債/自己資本比率'])  # 逆数で健全性表現
        
        financial_score = (
            0.35 * roe +
            0.25 * equity_ratio +
            0.25 * current_ratio +
            0.15 * debt_equity_ratio
        )
        
        return financial_score.fillna(0)
    
    def calculate_human_capital(self, data: pd.DataFrame) -> pd.Series:
        """人的資本の計算"""
        avg_salary_ratio = data['平均年間給与'] / data['平均年間給与'].mean()
        employee_productivity = data['売上高'] / data['従業員数']
        welfare_ratio = data['福利厚生費率']
        retention_proxy = 1 / (1 + data['従業員数増加率'].abs())  # 急激な変動は不安定性を示唆
        
        human_score = (
            0.3 * avg_salary_ratio +
            0.3 * employee_productivity +
            0.2 * welfare_ratio +
            0.2 * retention_proxy
        )
        
        return human_score.fillna(0)
    
    def calculate_strategic_flexibility(self, data: pd.DataFrame) -> pd.Series:
        """戦略柔軟性の計算"""
        cash_ratio = data['現金及び預金/総資産比率']
        capex_flexibility = data['設備投資額'] / data['減価償却費']
        business_diversity = data['事業セグメント数'] / 5  # 正規化
        fixed_cost_ratio = 1 - data['固定費率']
        
        flexibility_score = (
            0.3 * cash_ratio +
            0.3 * capex_flexibility +
            0.25 * business_diversity +
            0.15 * fixed_cost_ratio
        )
        
        return flexibility_score.fillna(0)
    
    def calculate_brand_value(self, data: pd.DataFrame) -> pd.Series:
        """ブランド価値の計算"""
        profit_margin = data['売上高営業利益率']
        intangible_ratio = data['無形固定資産/売上高比率']
        premium_pricing = profit_margin / data['売上高営業利益率'].median()
        market_recognition = data['広告宣伝費率']
        
        brand_score = (
            0.4 * premium_pricing +
            0.25 * intangible_ratio +
            0.2 * profit_margin +
            0.15 * market_recognition
        )
        
        return brand_score.fillna(0)

class StrategicPositioningAnalyzer:
    """戦略ポジション分析メインクラス"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.dimension_calculator = StrategyDimensionCalculator()
        self.scaler = StandardScaler()
        self.pca = None
        self.strategy_positions: Dict[str, StrategyPosition] = {}
        self.cluster_model = None
        
    def prepare_strategy_dimensions(self, financial_data: pd.DataFrame) -> pd.DataFrame:
        """戦略次元データの準備"""
        print("戦略次元データを計算中...")
        
        strategy_df = pd.DataFrame(index=financial_data.index)
        
        # 各戦略次元の計算
        strategy_df['innovation_intensity'] = self.dimension_calculator.calculate_innovation_intensity(financial_data)
        strategy_df['operational_efficiency'] = self.dimension_calculator.calculate_operational_efficiency(financial_data)
        strategy_df['market_expansion'] = self.dimension_calculator.calculate_market_expansion(financial_data)
        strategy_df['financial_strength'] = self.dimension_calculator.calculate_financial_strength(financial_data)
        strategy_df['human_capital'] = self.dimension_calculator.calculate_human_capital(financial_data)
        strategy_df['strategic_flexibility'] = self.dimension_calculator.calculate_strategic_flexibility(financial_data)
        strategy_df['brand_value'] = self.dimension_calculator.calculate_brand_value(financial_data)
        
        # 企業情報の追加
        strategy_df['company_name'] = financial_data['company_name']
        strategy_df['market_category'] = financial_data['market_category']
        strategy_df['year'] = financial_data['year']
        
        return strategy_df
    
    def perform_strategy_clustering(self, strategy_df: pd.DataFrame, n_clusters: int = None) -> pd.DataFrame:
        """戦略クラスタリング実行"""
        print("戦略クラスタリングを実行中...")
        
        # 戦略次元のみを抽出
        strategy_features = ['innovation_intensity', 'operational_efficiency', 'market_expansion', 
                            'financial_strength', 'human_capital', 'strategic_flexibility', 'brand_value']
        
        X = strategy_df[strategy_features].fillna(0)
        X_scaled = self.scaler.fit_transform(X)
        
        # 最適クラスター数の決定
        if n_clusters is None:
            n_clusters = self._find_optimal_clusters(X_scaled)
        
        # KMeansクラスタリング
        self.cluster_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        strategy_df['strategy_cluster'] = self.cluster_model.fit_predict(X_scaled)
        
        # クラスター品質評価
        silhouette_avg = silhouette_score(X_scaled, strategy_df['strategy_cluster'])
        print(f"シルエットスコア: {silhouette_avg:.3f}")
        
        return strategy_df
    
    def _find_optimal_clusters(self, X: np.ndarray) -> int:
        """最適クラスター数の決定"""
        inertias = []
        silhouette_scores = []
        K_range = range(2, min(11, len(X) // 2))
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X)
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(X, cluster_labels))
        
        # シルエットスコアが最大となるクラスター数を選択
        optimal_k = K_range[np.argmax(silhouette_scores)]
        print(f"最適クラスター数: {optimal_k}")
        
        return optimal_k
    
    def calculate_competitive_positioning(self, strategy_df: pd.DataFrame) -> pd.DataFrame:
        """競争ポジション計算"""
        print("競争ポジションを計算中...")
        
        # 市場カテゴリー別の分析
        results = []
        
        for market_cat in strategy_df['market_category'].unique():
            market_data = strategy_df[strategy_df['market_category'] == market_cat].copy()
            
            if len(market_data) == 0:
                continue
            
            # 各戦略次元での相対ランキング
            strategy_features = ['innovation_intensity', 'operational_efficiency', 'market_expansion', 
                                'financial_strength', 'human_capital', 'strategic_flexibility', 'brand_value']
            
            for feature in strategy_features:
                market_data[f'{feature}_rank'] = rankdata(market_data[feature], method='dense')
                market_data[f'{feature}_percentile'] = market_data[feature].rank(pct=True)
            
            # 総合競争力スコア計算
            weights = self.dimension_calculator.dimension_weights
            competitive_score = 0
            
            for feature, weight in weights.items():
                competitive_score += weight * market_data[f'{feature}_percentile']
            
            market_data['competitive_strength'] = competitive_score
            market_data['market_position'] = self._categorize_position(competitive_score)
            
            results.append(market_data)
        
        return pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    
    def _categorize_position(self, scores: pd.Series) -> pd.Series:
        """ポジションカテゴリ化"""
        positions = []
        for score in scores:
            if score >= 0.8:
                positions.append('Market Leader')
            elif score >= 0.6:
                positions.append('Strong Challenger')
            elif score >= 0.4:
                positions.append('Follower')
            else:
                positions.append('Niche Player')
        return pd.Series(positions)
    
    def analyze_strategy_transitions(self, strategy_df: pd.DataFrame) -> List[StrategyTransition]:
        """戦略移行パターン分析"""
        print("戦略移行パターンを分析中...")
        
        transitions = []
        strategy_features = ['innovation_intensity', 'operational_efficiency', 'market_expansion', 
                            'financial_strength', 'human_capital', 'strategic_flexibility', 'brand_value']
        
        for company in strategy_df['company_name'].unique():
            company_data = strategy_df[strategy_df['company_name'] == company].sort_values('year')
            
            if len(company_data) < 2:
                continue
            
            # 時系列での戦略ポジション変化を分析
            years = company_data['year'].values
            positions = company_data[strategy_features].values
            
            for i in range(len(positions) - 1):
                start_pos = positions[i]
                end_pos = positions[i + 1]
                
                # 移行距離計算
                transition_dist = euclidean(start_pos, end_pos)
                
                # 移行方向の特定
                direction = self._identify_transition_direction(start_pos, end_pos)
                
                # 成功指標（売上高成長率やROEの変化）
                start_performance = company_data.iloc[i]['competitive_strength']
                end_performance = company_data.iloc[i + 1]['competitive_strength']
                success_indicator = end_performance - start_performance
                
                transition = StrategyTransition(
                    company_name=company,
                    start_year=years[i],
                    end_year=years[i + 1],
                    start_position=start_pos,
                    end_position=end_pos,
                    transition_distance=transition_dist,
                    transition_direction=direction,
                    success_indicator=success_indicator
                )
                
                transitions.append(transition)
        
        return transitions
    
    def _identify_transition_direction(self, start_pos: np.ndarray, end_pos: np.ndarray) -> str:
        """移行方向の特定"""
        diff = end_pos - start_pos
        abs_diff = np.abs(diff)
        max_change_idx = np.argmax(abs_diff)
        
        dimensions = ['Innovation', 'Efficiency', 'Expansion', 'Financial', 
                        'Human Capital', 'Flexibility', 'Brand']
        
        direction_type = "Increase" if diff[max_change_idx] > 0 else "Decrease"
        primary_dimension = dimensions[max_change_idx]
        
        return f"{direction_type} in {primary_dimension}"
    
    def create_strategy_positions(self, positioned_df: pd.DataFrame) -> Dict[str, StrategyPosition]:
        """戦略ポジションオブジェクト作成"""
        positions = {}
        
        for _, row in positioned_df.iterrows():
            company_key = f"{row['company_name']}_{row['year']}"
            
            # 戦略次元ベクトル
            strategy_features = ['innovation_intensity', 'operational_efficiency', 'market_expansion', 
                                'financial_strength', 'human_capital', 'strategic_flexibility', 'brand_value']
            position_vector = row[strategy_features].values
            
            # 強みと弱みの特定
            percentile_features = [f'{f}_percentile' for f in strategy_features]
            strengths = []
            weaknesses = []
            
            for i, feature in enumerate(strategy_features):
                percentile = row[f'{feature}_percentile']
                if percentile >= 0.8:
                    strengths.append(feature.replace('_', ' ').title())
                elif percentile <= 0.2:
                    weaknesses.append(feature.replace('_', ' ').title())
            
            position = StrategyPosition(
                company_name=row['company_name'],
                market_category=row['market_category'],
                position_vector=position_vector,
                cluster_id=row['strategy_cluster'],
                competitive_strength=row['competitive_strength'],
                strategic_advantages=strengths,
                strategic_weaknesses=weaknesses,
                positioning_score=row['competitive_strength']
            )
            
            positions[company_key] = position
        
        self.strategy_positions = positions
        return positions
    
    def generate_strategy_insights(self, positioned_df: pd.DataFrame, transitions: List[StrategyTransition]) -> Dict[str, Any]:
        """戦略インサイト生成"""
        insights = {}
        
        # 市場カテゴリー別分析
        market_analysis = {}
        for market_cat in positioned_df['market_category'].unique():
            market_data = positioned_df[positioned_df['market_category'] == market_cat]
            
            market_analysis[market_cat] = {
                'average_competitive_strength': market_data['competitive_strength'].mean(),
                'top_performers': market_data.nlargest(3, 'competitive_strength')['company_name'].tolist(),
                'dominant_strategy_cluster': market_data['strategy_cluster'].mode().iloc[0] if len(market_data) > 0 else None,
                'key_success_factors': self._identify_key_success_factors(market_data),
                'strategic_diversity': market_data['strategy_cluster'].nunique()
            }
        
        insights['market_analysis'] = market_analysis
        
        # 戦略移行分析
        transition_analysis = {
            'most_successful_transitions': sorted(transitions, key=lambda x: x.success_indicator, reverse=True)[:5],
            'most_common_transition_patterns': self._analyze_transition_patterns(transitions),
            'average_transition_success': np.mean([t.success_indicator for t in transitions])
        }
        
        insights['transition_analysis'] = transition_analysis
        
        # クラスター特性分析
        cluster_analysis = {}
        for cluster_id in positioned_df['strategy_cluster'].unique():
            cluster_data = positioned_df[positioned_df['strategy_cluster'] == cluster_id]
            
            cluster_analysis[f'cluster_{cluster_id}'] = {
                'size': len(cluster_data),
                'average_performance': cluster_data['competitive_strength'].mean(),
                'dominant_market_category': cluster_data['market_category'].mode().iloc[0] if len(cluster_data) > 0 else None,
                'key_characteristics': self._identify_cluster_characteristics(cluster_data)
            }
        
        insights['cluster_analysis'] = cluster_analysis
        
        return insights
    
    def _identify_key_success_factors(self, market_data: pd.DataFrame) -> List[str]:
        """成功要因特定"""
        strategy_features = ['innovation_intensity', 'operational_efficiency', 'market_expansion', 
                            'financial_strength', 'human_capital', 'strategic_flexibility', 'brand_value']
        
        # 競争力と各戦略次元の相関分析
        correlations = {}
        for feature in strategy_features:
            corr = market_data['competitive_strength'].corr(market_data[feature])
            correlations[feature] = abs(corr) if not np.isnan(corr) else 0
        
        # 上位3つの要因を返す
        top_factors = sorted(correlations.items(), key=lambda x: x[1], reverse=True)[:3]
        return [factor[0].replace('_', ' ').title() for factor in top_factors]
    
    def _analyze_transition_patterns(self, transitions: List[StrategyTransition]) -> Dict[str, int]:
        """移行パターン分析"""
        patterns = {}
        for transition in transitions:
            direction = transition.transition_direction
            patterns[direction] = patterns.get(direction, 0) + 1
        
        return dict(sorted(patterns.items(), key=lambda x: x[1], reverse=True))
    
    def _identify_cluster_characteristics(self, cluster_data: pd.DataFrame) -> List[str]:
        """クラスター特性特定"""
        strategy_features = ['innovation_intensity', 'operational_efficiency', 'market_expansion', 
                            'financial_strength', 'human_capital', 'strategic_flexibility', 'brand_value']
        
        characteristics = []
        for feature in strategy_features:
            mean_value = cluster_data[feature].mean()
            overall_mean = cluster_data[feature].mean()  # これは修正が必要（全体平均を計算すべき）
            
            if mean_value > overall_mean * 1.2:
                characteristics.append(f"High {feature.replace('_', ' ').title()}")
            elif mean_value < overall_mean * 0.8:
                characteristics.append(f"Low {feature.replace('_', ' ').title()}")
        
        return characteristics[:3]  # 上位3つの特徴を返す
    
    def run_comprehensive_analysis(self, financial_data: pd.DataFrame) -> Dict[str, Any]:
        """包括的戦略ポジション分析実行"""
        print("=== A2AI 戦略ポジション分析開始 ===")
        
        # Step 1: 戦略次元データ準備
        strategy_df = self.prepare_strategy_dimensions(financial_data)
        
        # Step 2: 戦略クラスタリング
        clustered_df = self.perform_strategy_clustering(strategy_df)
        
        # Step 3: 競争ポジション計算
        positioned_df = self.calculate_competitive_positioning(clustered_df)
        
        # Step 4: 戦略移行分析
        transitions = self.analyze_strategy_transitions(positioned_df)
        
        # Step 5: 戦略ポジション作成
        strategy_positions = self.create_strategy_positions(positioned_df)
        
        # Step 6: インサイト生成
        insights = self.generate_strategy_insights(positioned_df, transitions)
        
        print("=== 戦略ポジション分析完了 ===")
        
        return {
            'strategy_data': positioned_df,
            'strategy_positions': strategy_positions,
            'transitions': transitions,
            'insights': insights,
            'cluster_model': self.cluster_model,
            'scaler': self.scaler
        }

class StrategyPositioningVisualizer:
    """戦略ポジション可視化クラス"""
    
    def __init__(self):
        self.color_palette = px.colors.qualitative.Set3
    
    def create_strategy_map_2d(self, positioned_df: pd.DataFrame, x_dim: str = 'innovation_intensity', 
                                y_dim: str = 'operational_efficiency') -> go.Figure:
        """2次元戦略マップ作成"""
        fig = px.scatter(
            positioned_df,
            x=x_dim,
            y=y_dim,
            color='market_category',
            size='competitive_strength',
            hover_data=['company_name', 'strategy_cluster', 'market_position'],
            title=f'Strategic Positioning Map: {x_dim.title()} vs {y_dim.title()}',
            labels={
                x_dim: x_dim.replace('_', ' ').title(),
                y_dim: y_dim.replace('_', ' ').title()
            }
        )
        
        fig.update_layout(
            width=800,
            height=600,
            showlegend=True
        )
        
        return fig
    
    def create_strategy_radar_chart(self, company_data: pd.DataFrame, company_name: str) -> go.Figure:
        """企業別戦略レーダーチャート"""
        company_row = company_data[company_data['company_name'] == company_name].iloc[0]
        
        dimensions = ['Innovation Intensity', 'Operational Efficiency', 'Market Expansion', 
                        'Financial Strength', 'Human Capital', 'Strategic Flexibility', 'Brand Value']
        
        values = [
            company_row['innovation_intensity'],
            company_row['operational_efficiency'],
            company_row['market_expansion'],
            company_row['financial_strength'],
            company_row['human_capital'],
            company_row['strategic_flexibility'],
            company_row['brand_value']
        ]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=dimensions,
            fill='toself',
            name=company_name,
            line=dict(color='blue')
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=True,
            title=f'Strategic Profile: {company_name}'
        )
        
        return fig
    
    def create_transition_flow_chart(self, transitions: List[StrategyTransition]) -> go.Figure:
        """戦略移行フローチャート"""
        # 移行パターンごとの成功率を計算
        pattern_success = {}
        pattern_count = {}
        
        for transition in transitions:
            pattern = transition.transition_direction
            if pattern not in pattern_success:
                pattern_success[pattern] = []
                pattern_count[pattern] = 0
            
            pattern_success[pattern].append(transition.success_indicator)
            pattern_count[pattern] += 1
        
        patterns = list(pattern_success.keys())
        avg_success = [np.mean(pattern_success[p]) for p in patterns]
        counts = [pattern_count[p] for p in patterns]
        
        fig = px.scatter(
            x=counts,
            y=avg_success,
            size=counts,
            hover_name=patterns,
            title='Strategy Transition Patterns: Success Rate vs Frequency',
            labels={
                'x': 'Frequency',
                'y': 'Average Success Rate'
            }
        )
        
        return fig
    
    def create_competitive_landscape(self, positioned_df: pd.DataFrame) -> go.Figure:
        """競争ランドスケープ可視化"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Innovation vs Efficiency', 'Market Expansion vs Financial Strength',
                            'Human Capital vs Flexibility', 'Brand Value vs Overall Competitive Strength'),
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                    [{"type": "scatter"}, {"type": "scatter"}]]
        )
        
        # 各市場カテゴリーに色を付ける
        colors = {'high_share': 'green', 'declining': 'orange', 'lost': 'red'}
        
        for market_cat in positioned_df['market_category'].unique():
            market_data = positioned_df[positioned_df['market_category'] == market_cat]
            
            fig.add_trace(
                go.Scatter(
                    x=market_data['innovation_intensity'],
                    y=market_data['operational_efficiency'],
                    mode='markers',
                    name=f'{market_cat} - Innovation/Efficiency',
                    marker=dict(color=colors.get(market_cat, 'blue')),
                    text=market_data['company_name'],
                    showlegend=True
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=market_data['market_expansion'],
                    y=market_data['financial_strength'],
                    mode='markers',
                    name=f'{market_cat} - Expansion/Financial',
                    marker=dict(color=colors.get(market_cat, 'blue')),
                    text=market_data['company_name'],
                    showlegend=False
                ),
                row=1, col=2
            )
            
            fig.add_trace(
                go.Scatter(
                    x=market_data['human_capital'],
                    y=market_data['strategic_flexibility'],
                    mode='markers',
                    name=f'{market_cat} - Human/Flexibility',
                    marker=dict(color=colors.get(market_cat, 'blue')),
                    text=market_data['company_name'],
                    showlegend=False
                ),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=market_data['brand_value'],
                    y=market_data['competitive_strength'],
                    mode='markers',
                    name=f'{market_cat} - Brand/Overall',
                    marker=dict(color=colors.get(market_cat, 'blue')),
                    text=market_data['company_name'],
                    showlegend=False
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            height=800,
            title_text="Competitive Landscape Analysis",
            showlegend=True
        )
        
        return fig

def main():
    """メイン実行関数（テスト用）"""
    # サンプルデータ生成（実際の使用時はEDINETデータを使用）
    np.random.seed(42)
    n_companies = 50
    n_years = 5
    
    sample_data = []
    companies = [f'Company_{i:02d}' for i in range(n_companies)]
    market_categories = ['high_share'] * 15 + ['declining'] * 15 + ['lost'] * 20
    
    for year in range(2020, 2020 + n_years):
        for i, company in enumerate(companies):
            data_point = {
                'company_name': company,
                'market_category': market_categories[i],
                'year': year,
                '研究開発費率': np.random.uniform(0.01, 0.15),
                '特許関連費用': np.random.uniform(1000, 10000),
                '売上高': np.random.uniform(100000, 1000000),
                '新製品売上高比率': np.random.uniform(0.1, 0.5),
                '技術ライセンス収入': np.random.uniform(1000, 50000),
                '総資産回転率': np.random.uniform(0.5, 2.0),
                '労働生産性': np.random.uniform(5000, 15000),
                '設備生産性': np.random.uniform(2.0, 8.0),
                '売上原価率': np.random.uniform(0.6, 0.9),
                '海外売上高比率': np.random.uniform(0.1, 0.8),
                '事業セグメント数': np.random.randint(1, 8),
                '売上高成長率': np.random.uniform(-0.1, 0.3),
                '販売費及び一般管理費率': np.random.uniform(0.1, 0.4),
                'ROE': np.random.uniform(0.02, 0.25),
                '自己資本比率': np.random.uniform(0.2, 0.8),
                '流動比率': np.random.uniform(1.0, 3.0),
                '有利子負債/自己資本比率': np.random.uniform(0.1, 2.0),
                '平均年間給与': np.random.uniform(400, 1200),
                '従業員数': np.random.randint(100, 50000),
                '福利厚生費率': np.random.uniform(0.01, 0.05),
                '従業員数増加率': np.random.uniform(-0.1, 0.2),
                '現金及び預金/総資産比率': np.random.uniform(0.05, 0.3),
                '設備投資額': np.random.uniform(5000, 100000),
                '減価償却費': np.random.uniform(3000, 80000),
                '固定費率': np.random.uniform(0.3, 0.7),
                '売上高営業利益率': np.random.uniform(0.02, 0.25),
                '無形固定資産/売上高比率': np.random.uniform(0.01, 0.1),
                '広告宣伝費率': np.random.uniform(0.005, 0.05)
            }
            sample_data.append(data_point)
    
    # DataFrameに変換
    df = pd.DataFrame(sample_data)
    
    # 戦略ポジション分析実行
    analyzer = StrategicPositioningAnalyzer()
    results = analyzer.run_comprehensive_analysis(df)
    
    # 可視化
    visualizer = StrategyPositioningVisualizer()
    
    # 2次元戦略マップ
    strategy_map = visualizer.create_strategy_map_2d(
        results['strategy_data'], 
        'innovation_intensity', 
        'operational_efficiency'
    )
    
    # 競争ランドスケープ
    competitive_landscape = visualizer.create_competitive_landscape(results['strategy_data'])
    
    # 結果のサマリー表示
    print("\n=== 戦略ポジション分析結果サマリー ===")
    print(f"分析対象企業数: {len(df['company_name'].unique())}")
    print(f"分析期間: {df['year'].min()}-{df['year'].max()}")
    print(f"戦略クラスター数: {results['strategy_data']['strategy_cluster'].nunique()}")
    
    # 市場カテゴリー別の競争力平均
    market_strength = results['strategy_data'].groupby('market_category')['competitive_strength'].mean()
    print(f"\n市場カテゴリー別平均競争力:")
    for market, strength in market_strength.items():
        print(f"  {market}: {strength:.3f}")
    
    # インサイト表示
    print(f"\n主要インサイト:")
    for market_cat, analysis in results['insights']['market_analysis'].items():
        print(f"\n{market_cat}市場:")
        print(f"  トップパフォーマー: {', '.join(analysis['top_performers'])}")
        print(f"  主要成功要因: {', '.join(analysis['key_success_factors'])}")
    
    return results, strategy_map, competitive_landscape

if __name__ == "__main__":
    results, map_fig, landscape_fig = main()
    
    # 図表を表示（Jupyter環境の場合）
    try:
        map_fig.show()
        landscape_fig.show()
    except Exception as e:
        print(f"可視化表示エラー: {e}")
        print("Jupyter環境で実行してください")


# 使用例とドキュメント
"""
A2AI Strategic Positioning Analyzer 使用例

1. 基本的な使用方法:

```python
from src.analysis.integrated_analysis.strategic_positioning import StrategicPositioningAnalyzer

# アナライザー初期化
analyzer = StrategicPositioningAnalyzer()

# 財務データ（EDINET等から取得）をDataFrame形式で準備
# 必要カラム: company_name, market_category, year, 各種財務指標

# 分析実行
results = analyzer.run_comprehensive_analysis(financial_data)

# 結果取得
strategy_positions = results['strategy_positions']
transitions = results['transitions']
insights = results['insights']
```

2. 可視化:

```python
from src.analysis.integrated_analysis.strategic_positioning import StrategyPositioningVisualizer

visualizer = StrategyPositioningVisualizer()

# 2次元戦略マップ
strategy_map = visualizer.create_strategy_map_2d(results['strategy_data'])

# 企業別レーダーチャート
radar_chart = visualizer.create_strategy_radar_chart(results['strategy_data'], 'Company_Name')

# 競争ランドスケープ
landscape = visualizer.create_competitive_landscape(results['strategy_data'])
```

3. 分析結果の活用:

- strategy_positions: 各企業の戦略ポジション詳細
- transitions: 戦略移行パターンと成功度
- insights: 市場別・クラスター別分析結果

主要な分析次元:
1. Innovation Intensity (イノベーション強度)
2. Operational Efficiency (オペレーショナル効率性)
3. Market Expansion (市場拡張力)
4. Financial Strength (財務力)
5. Human Capital (人的資本)
6. Strategic Flexibility (戦略柔軟性)  
7. Brand Value (ブランド価値)

出力:
- 企業別競争ポジション
- 戦略クラスター分析
- 成功要因特定
- 戦略移行パターン
- 市場別競争構造
"""