"""
A2AI - Advanced Financial Analysis AI
競争ダイナミクス分析モジュール (competitive_dynamics.py)

企業間の競争関係、市場ポジション変遷、競争優位性の動的分析を実行

主要機能:
1. 競争ポジション分析 - 市場内での企業の相対的位置
2. 競争激化度測定 - 市場競争の激化・緩和パターン分析
3. 競争優位性分析 - 持続的競争優位の要因特定
4. 市場集中度分析 - HHI、CR4等による市場構造変化
5. 企業間影響分析 - 他社行動が自社に与える影響
6. 競争戦略類型化 - Porter戦略類型に基づく分類
7. 競争サイクル分析 - 競争の周期性・パターン分析
8. 新規参入影響分析 - 新設企業が既存競合に与える影響
"""

import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import pdist, squareform
import warnings
warnings.filterwarnings('ignore')

class CompetitiveDynamicsAnalyzer:
    """
    競争ダイナミクス分析クラス
    
    市場内企業間の競争関係を多角的に分析し、
    競争優位性の源泉と競争構造の変化を定量化
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初期化
        
        Args:
            config: 設定パラメータ辞書
                - markets: 市場分類情報
                - companies: 企業リスト 
                - time_periods: 分析期間設定
                - competitive_metrics: 競争指標設定
        """
        self.config = config
        self.markets = config.get('markets', {})
        self.companies = config.get('companies', [])
        self.time_periods = config.get('time_periods', {})
        self.competitive_metrics = config.get('competitive_metrics', {})
        
        # 分析結果格納
        self.competitive_positions = {}
        self.market_concentration = {}
        self.competitive_intensity = {}
        self.strategic_groups = {}
        self.competitive_responses = {}
        
    def analyze_competitive_positioning(self, 
                                        financial_data: pd.DataFrame,
                                        market_share_data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        競争ポジション分析
        
        企業の市場内での相対的位置と競争ポジションの変遷を分析
        
        Args:
            financial_data: 財務データ (企業×年×指標)
            market_share_data: 市場シェアデータ
            
        Returns:
            競争ポジション分析結果辞書
        """
        positioning_results = {}
        
        for market_category in ['high_share', 'declining', 'lost']:
            market_companies = self.markets.get(market_category, [])
            
            # 市場別データ抽出
            market_data = financial_data[
                financial_data['company'].isin(market_companies)
            ].copy()
            
            if market_data.empty:
                continue
                
            # 競争ポジションマトリクス作成
            positioning_matrix = self._create_positioning_matrix(
                market_data, market_share_data, market_companies
            )
            
            # ポジション変遷分析
            position_evolution = self._analyze_position_evolution(
                positioning_matrix
            )
            
            # 競争優位性スコア算出
            competitive_advantage = self._calculate_competitive_advantage(
                market_data
            )
            
            positioning_results[market_category] = {
                'positioning_matrix': positioning_matrix,
                'position_evolution': position_evolution,
                'competitive_advantage': competitive_advantage,
                'market_leaders': self._identify_market_leaders(positioning_matrix),
                'position_stability': self._assess_position_stability(position_evolution)
            }
            
        self.competitive_positions = positioning_results
        return positioning_results
    
    def _create_positioning_matrix(self, 
                                    market_data: pd.DataFrame,
                                    market_share_data: pd.DataFrame,
                                    companies: List[str]) -> pd.DataFrame:
        """
        競争ポジションマトリクス作成
        
        収益性(Y軸) × 市場シェア(X軸) による2次元ポジショニング
        """
        positioning_data = []
        
        for year in range(1984, 2025):
            year_data = market_data[market_data['year'] == year]
            
            if year_data.empty:
                continue
                
            for company in companies:
                company_data = year_data[year_data['company'] == company]
                
                if company_data.empty:
                    continue
                    
                # 収益性指標 (ROE, 営業利益率の加重平均)
                roe = company_data['roe'].iloc[0] if 'roe' in company_data else 0
                operating_margin = company_data['operating_margin'].iloc[0] if 'operating_margin' in company_data else 0
                profitability = (roe * 0.6 + operating_margin * 0.4)
                
                # 市場シェア
                share_data = market_share_data[
                    (market_share_data['company'] == company) & 
                    (market_share_data['year'] == year)
                ]
                market_share = share_data['market_share'].iloc[0] if not share_data.empty else 0
                
                # 成長性指標
                revenue_growth = company_data['revenue_growth'].iloc[0] if 'revenue_growth' in company_data else 0
                
                # 規模指標
                revenue = company_data['revenue'].iloc[0] if 'revenue' in company_data else 0
                
                positioning_data.append({
                    'company': company,
                    'year': year,
                    'profitability': profitability,
                    'market_share': market_share,
                    'growth_rate': revenue_growth,
                    'revenue_scale': revenue,
                    'competitive_position': self._classify_competitive_position(
                        profitability, market_share, revenue_growth
                    )
                })
        
        return pd.DataFrame(positioning_data)
    
    def _classify_competitive_position(self, 
                                        profitability: float,
                                        market_share: float, 
                                        growth_rate: float) -> str:
        """
        競争ポジション分類
        
        BCGマトリクス + 収益性を組み合わせた8分類
        """
        high_share = market_share > 15  # 高シェア閾値
        high_growth = growth_rate > 5   # 高成長閾値
        high_profit = profitability > 10  # 高収益閾値
        
        if high_share and high_growth and high_profit:
            return 'Star_Leader'  # スター・リーダー
        elif high_share and high_growth:
            return 'Star'  # スター
        elif high_share and high_profit:
            return 'Cash_Cow_Premium'  # プレミアム金のなる木
        elif high_share:
            return 'Cash_Cow'  # 金のなる木
        elif high_growth and high_profit:
            return 'Question_Mark_Premium'  # プレミアム問題児
        elif high_growth:
            return 'Question_Mark'  # 問題児
        elif high_profit:
            return 'Niche_Premium'  # ニッチ・プレミアム
        else:
            return 'Dog'  # 負け犬
    
    def _analyze_position_evolution(self, positioning_matrix: pd.DataFrame) -> pd.DataFrame:
        """
        ポジション変遷分析
        
        企業の競争ポジションの時系列変化を追跡
        """
        evolution_data = []
        
        for company in positioning_matrix['company'].unique():
            company_data = positioning_matrix[
                positioning_matrix['company'] == company
            ].sort_values('year')
            
            if len(company_data) < 2:
                continue
                
            # 各指標の変化率算出
            for i in range(1, len(company_data)):
                current = company_data.iloc[i]
                previous = company_data.iloc[i-1]
                
                profitability_change = (
                    (current['profitability'] - previous['profitability']) / 
                    max(abs(previous['profitability']), 1) * 100
                )
                
                share_change = (
                    (current['market_share'] - previous['market_share']) / 
                    max(abs(previous['market_share']), 1) * 100
                )
                
                # ポジション移動方向
                position_movement = self._analyze_position_movement(
                    previous['competitive_position'],
                    current['competitive_position']
                )
                
                evolution_data.append({
                    'company': company,
                    'from_year': previous['year'],
                    'to_year': current['year'],
                    'profitability_change': profitability_change,
                    'share_change': share_change,
                    'from_position': previous['competitive_position'],
                    'to_position': current['competitive_position'],
                    'position_movement': position_movement,
                    'evolution_score': self._calculate_evolution_score(
                        profitability_change, share_change, position_movement
                    )
                })
        
        return pd.DataFrame(evolution_data)
    
    def analyze_competitive_intensity(self, 
                                    financial_data: pd.DataFrame,
                                    market_events: pd.DataFrame = None) -> Dict[str, pd.DataFrame]:
        """
        競争激化度分析
        
        市場における競争の激化・緩和パターンを定量化
        
        Args:
            financial_data: 財務データ
            market_events: 市場イベントデータ (M&A, 新規参入等)
            
        Returns:
            競争激化度分析結果
        """
        intensity_results = {}
        
        for market_category in ['high_share', 'declining', 'lost']:
            market_companies = self.markets.get(market_category, [])
            market_data = financial_data[
                financial_data['company'].isin(market_companies)
            ].copy()
            
            if market_data.empty:
                continue
            
            # 競争激化度指標算出
            intensity_metrics = self._calculate_intensity_metrics(market_data)
            
            # 価格競争激化度
            price_competition = self._analyze_price_competition(market_data)
            
            # イノベーション競争度
            innovation_competition = self._analyze_innovation_competition(market_data)
            
            # 市場参入・退出パターン
            entry_exit_patterns = self._analyze_entry_exit_patterns(
                market_data, market_events
            )
            
            # 競争サイクル分析
            competitive_cycles = self._analyze_competitive_cycles(intensity_metrics)
            
            intensity_results[market_category] = {
                'intensity_metrics': intensity_metrics,
                'price_competition': price_competition,
                'innovation_competition': innovation_competition,
                'entry_exit_patterns': entry_exit_patterns,
                'competitive_cycles': competitive_cycles,
                'overall_intensity_score': self._calculate_overall_intensity(
                    intensity_metrics, price_competition, innovation_competition
                )
            }
            
        self.competitive_intensity = intensity_results
        return intensity_results
    
    def _calculate_intensity_metrics(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """
        競争激化度指標算出
        
        利益率分散、市場シェア変動等から競争激化度を測定
        """
        intensity_data = []
        
        for year in range(1984, 2025):
            year_data = market_data[market_data['year'] == year]
            
            if len(year_data) < 2:
                continue
            
            # 利益率の分散 (競争激化時は利益率格差拡大)
            profit_variance = year_data['operating_margin'].var()
            
            # ROEの分散
            roe_variance = year_data['roe'].var() if 'roe' in year_data else 0
            
            # 市場シェア集中度 (HHI)
            shares = year_data['market_share'].fillna(0)
            hhi = sum(shares ** 2)
            
            # 価格競争指標 (売上原価率の上昇)
            avg_cost_ratio = year_data['cost_ratio'].mean() if 'cost_ratio' in year_data else 0
            
            # R&D投資競争度
            avg_rd_ratio = year_data['rd_ratio'].mean() if 'rd_ratio' in year_data else 0
            
            # 広告宣伝費競争度  
            avg_ad_ratio = year_data['advertising_ratio'].mean() if 'advertising_ratio' in year_data else 0
            
            # 総合競争激化度スコア
            intensity_score = (
                profit_variance * 0.3 +
                roe_variance * 0.2 +
                (100 - hhi) * 0.2 +  # HHIの逆数 (分散が大きいほど競争激化)
                avg_cost_ratio * 0.1 +
                avg_rd_ratio * 0.1 +
                avg_ad_ratio * 0.1
            )
            
            intensity_data.append({
                'year': year,
                'profit_variance': profit_variance,
                'roe_variance': roe_variance,
                'hhi': hhi,
                'market_concentration': self._classify_concentration(hhi),
                'avg_cost_ratio': avg_cost_ratio,
                'avg_rd_ratio': avg_rd_ratio,
                'avg_ad_ratio': avg_ad_ratio,
                'intensity_score': intensity_score,
                'intensity_level': self._classify_intensity_level(intensity_score)
            })
        
        return pd.DataFrame(intensity_data)
    
    def analyze_strategic_groups(self, financial_data: pd.DataFrame) -> Dict[str, Any]:
        """
        戦略グループ分析
        
        類似の戦略を採る企業群を特定し、グループ間競争を分析
        
        Args:
            financial_data: 財務データ
            
        Returns:
            戦略グループ分析結果
        """
        strategic_results = {}
        
        for market_category in ['high_share', 'declining', 'lost']:
            market_companies = self.markets.get(market_category, [])
            market_data = financial_data[
                financial_data['company'].isin(market_companies)
            ].copy()
            
            if market_data.empty:
                continue
            
            # 戦略特徴量抽出
            strategic_features = self._extract_strategic_features(market_data)
            
            # クラスタリングによる戦略グループ特定
            strategic_groups = self._identify_strategic_groups(strategic_features)
            
            # グループ間競争分析
            inter_group_competition = self._analyze_inter_group_competition(
                strategic_groups, strategic_features
            )
            
            # グループ内競争分析
            intra_group_competition = self._analyze_intra_group_competition(
                strategic_groups, strategic_features
            )
            
            # 戦略移動分析
            strategic_mobility = self._analyze_strategic_mobility(
                strategic_groups, market_data
            )
            
            strategic_results[market_category] = {
                'strategic_features': strategic_features,
                'strategic_groups': strategic_groups,
                'inter_group_competition': inter_group_competition,
                'intra_group_competition': intra_group_competition,
                'strategic_mobility': strategic_mobility,
                'group_performance': self._analyze_group_performance(
                    strategic_groups, market_data
                )
            }
            
        self.strategic_groups = strategic_results
        return strategic_results
    
    def _extract_strategic_features(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """
        戦略特徴量抽出
        
        Porter の戦略類型に基づく特徴量を抽出
        """
        strategic_data = []
        
        # 企業別・年別の戦略特徴量算出
        for company in market_data['company'].unique():
            company_data = market_data[market_data['company'] == company]
            
            for year in company_data['year'].unique():
                year_data = company_data[company_data['year'] == year]
                
                if year_data.empty:
                    continue
                
                # コストリーダーシップ戦略指標
                cost_leadership_score = self._calculate_cost_leadership_score(year_data)
                
                # 差別化戦略指標
                differentiation_score = self._calculate_differentiation_score(year_data)
                
                # 集中戦略指標
                focus_score = self._calculate_focus_score(year_data)
                
                # 規模戦略指標
                scale_score = self._calculate_scale_score(year_data)
                
                strategic_data.append({
                    'company': company,
                    'year': year,
                    'cost_leadership_score': cost_leadership_score,
                    'differentiation_score': differentiation_score,
                    'focus_score': focus_score,
                    'scale_score': scale_score,
                    'strategic_type': self._classify_strategic_type(
                        cost_leadership_score, differentiation_score, 
                        focus_score, scale_score
                    )
                })
        
        return pd.DataFrame(strategic_data)
    
    def analyze_competitive_responses(self, 
                                    financial_data: pd.DataFrame,
                                    market_events: pd.DataFrame = None) -> Dict[str, pd.DataFrame]:
        """
        競争対応分析
        
        企業間の戦略的相互作用と競争対応パターンを分析
        
        Args:
            financial_data: 財務データ
            market_events: 市場イベントデータ
            
        Returns:
            競争対応分析結果
        """
        response_results = {}
        
        for market_category in ['high_share', 'declining', 'lost']:
            market_companies = self.markets.get(market_category, [])
            market_data = financial_data[
                financial_data['company'].isin(market_companies)
            ].copy()
            
            if market_data.empty:
                continue
            
            # 競争行動-対応マトリクス構築
            response_matrix = self._build_response_matrix(market_data)
            
            # 対応時差分析
            response_timing = self._analyze_response_timing(response_matrix)
            
            # 対応強度分析
            response_intensity = self._analyze_response_intensity(response_matrix)
            
            # 模倣パターン分析
            imitation_patterns = self._analyze_imitation_patterns(market_data)
            
            # 革新-追従関係分析
            innovation_followership = self._analyze_innovation_followership(
                market_data, market_events
            )
            
            response_results[market_category] = {
                'response_matrix': response_matrix,
                'response_timing': response_timing,
                'response_intensity': response_intensity,
                'imitation_patterns': imitation_patterns,
                'innovation_followership': innovation_followership,
                'competitive_rivalry_index': self._calculate_rivalry_index(
                    response_matrix, response_timing, response_intensity
                )
            }
            
        self.competitive_responses = response_results
        return response_results
    
    def analyze_new_entrant_impact(self, 
                                    financial_data: pd.DataFrame,
                                    entry_events: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        新規参入影響分析
        
        新設企業が既存競合企業に与える影響を定量化
        
        Args:
            financial_data: 財務データ
            entry_events: 新規参入イベントデータ
            
        Returns:
            新規参入影響分析結果
        """
        impact_results = {}
        
        for market_category in ['high_share', 'declining', 'lost']:
            market_companies = self.markets.get(market_category, [])
            
            # 新規参入企業特定
            new_entrants = self._identify_new_entrants(
                market_companies, entry_events
            )
            
            if not new_entrants:
                continue
            
            # 参入前後の既存企業影響分析
            pre_post_impact = self._analyze_pre_post_entry_impact(
                financial_data, market_companies, new_entrants
            )
            
            # 市場構造変化分析
            structural_changes = self._analyze_structural_changes(
                financial_data, market_companies, new_entrants
            )
            
            # 競争激化度変化
            intensity_changes = self._analyze_intensity_changes_by_entry(
                financial_data, market_companies, new_entrants
            )
            
            # 新規参入成功要因分析
            success_factors = self._analyze_entrant_success_factors(
                financial_data, new_entrants
            )
            
            impact_results[market_category] = {
                'new_entrants': new_entrants,
                'pre_post_impact': pre_post_impact,
                'structural_changes': structural_changes,
                'intensity_changes': intensity_changes,
                'success_factors': success_factors,
                'disruption_index': self._calculate_disruption_index(
                    pre_post_impact, structural_changes, intensity_changes
                )
            }
        
        return impact_results
    
    def generate_competitive_insights(self) -> Dict[str, Any]:
        """
        競争インサイト生成
        
        分析結果から戦略的インサイトを抽出
        
        Returns:
            競争インサイト辞書
        """
        insights = {
            'market_dynamics_summary': {},
            'competitive_patterns': {},
            'strategic_recommendations': {},
            'risk_warnings': {}
        }
        
        for market_category in ['high_share', 'declining', 'lost']:
            if market_category not in self.competitive_positions:
                continue
            
            # 市場ダイナミクス要約
            insights['market_dynamics_summary'][market_category] = \
                self._summarize_market_dynamics(market_category)
            
            # 競争パターン特定
            insights['competitive_patterns'][market_category] = \
                self._identify_competitive_patterns(market_category)
            
            # 戦略提言
            insights['strategic_recommendations'][market_category] = \
                self._generate_strategic_recommendations(market_category)
            
            # リスク警告
            insights['risk_warnings'][market_category] = \
                self._generate_risk_warnings(market_category)
        
        return insights
    
    # ヘルパーメソッド群
    def _analyze_position_movement(self, from_pos: str, to_pos: str) -> str:
        """ポジション移動方向分析"""
        position_hierarchy = {
            'Star_Leader': 8, 'Star': 7, 'Cash_Cow_Premium': 6,
            'Question_Mark_Premium': 5, 'Cash_Cow': 4, 'Question_Mark': 3,
            'Niche_Premium': 2, 'Dog': 1
        }
        
        from_level = position_hierarchy.get(from_pos, 0)
        to_level = position_hierarchy.get(to_pos, 0)
        
        if to_level > from_level:
            return 'Upward'
        elif to_level < from_level:
            return 'Downward'
        else:
            return 'Stable'
    
    def _calculate_evolution_score(self, profit_change: float, 
                                    share_change: float, movement: str) -> float:
        """進化スコア算出"""
        base_score = profit_change * 0.4 + share_change * 0.6
        
        movement_bonus = {
            'Upward': 10, 'Stable': 0, 'Downward': -10
        }
        
        return base_score + movement_bonus.get(movement, 0)
    
    def _classify_concentration(self, hhi: float) -> str:
        """市場集中度分類"""
        if hhi > 2500:
            return 'Highly Concentrated'
        elif hhi > 1500:
            return 'Moderately Concentrated'
        else:
            return 'Unconcentrated'
    
    def _classify_intensity_level(self, score: float) -> str:
        """競争激化度レベル分類"""
        if score > 50:
            return 'Very High'
        elif score > 30:
            return 'High'
        elif score > 15:
            return 'Moderate'
        else:
            return 'Low'
    
    def _calculate_cost_leadership_score(self, data: pd.DataFrame) -> float:
        """コストリーダーシップ戦略スコア"""
        cost_ratio = data['cost_ratio'].iloc[0] if 'cost_ratio' in data else 50
        efficiency_score = (100 - cost_ratio)  # コストが低いほど高スコア
        
        scale_efficiency = data['revenue'].iloc[0] if 'revenue' in data else 0
        scale_score = min(scale_efficiency / 1000000, 100)  # 規模効率性
        
        return (efficiency_score * 0.7 + scale_score * 0.3)
    
    def _calculate_differentiation_score(self, data: pd.DataFrame) -> float:
        """差別化戦略スコア"""
        rd_ratio = data['rd_ratio'].iloc[0] if 'rd_ratio' in data else 0
        ad_ratio = data['advertising_ratio'].iloc[0] if 'advertising_ratio' in data else 0
        margin = data['operating_margin'].iloc[0] if 'operating_margin' in data else 0
        
        return (rd_ratio * 0.4 + ad_ratio * 0.3 + margin * 0.3)
    
    def _calculate_focus_score(self, data: pd.DataFrame) -> float:
        """集中戦略スコア"""
        # セグメント集中度とニッチ市場特化度
        segment_concentration = data.get('segment_concentration', 50)
        niche_ratio = data.get('niche_market_ratio', 0)
        
        return (segment_concentration * 0.6 + niche_ratio * 0.4)
    
    def _calculate_scale_score(self, data: pd.DataFrame) -> float:
        """規模戦略スコア"""
        revenue = data['revenue'].iloc[0] if 'revenue' in data else 0
        market_share = data.get('market_share', 0)
        
        return (min(revenue / 10000000, 50) + market_share)
    
    def _classify_strategic_type(self, cost: float, diff: float, 
                                focus: float, scale: float) -> str:
        """戦略タイプ分類"""
        if cost > 60 and scale > 60:
            return 'Cost_Leadership_Scale'
        elif diff > 60:
            return 'Differentiation'
        elif focus > 60:
            return 'Focus'
        elif cost > 40:
            return 'Cost_Leadership'
        else:
            return 'Stuck_in_Middle'
    
    # 追加のヘルパーメソッドは省略（実装継続時に展開）
    def _identify_market_leaders(self, positioning_matrix: pd.DataFrame) -> pd.DataFrame:
        """市場リーダー特定"""
        # 最新年度のデータでリーダー特定
        latest_year = positioning_matrix['year'].max()
        latest_data = positioning_matrix[positioning_matrix['year'] == latest_year]
        
        return latest_data.nlargest(5, 'market_share')[['company', 'market_share', 'competitive_position']]
    
    def _assess_position_stability(self, evolution: pd.DataFrame) -> pd.DataFrame:
        """ポジション安定性評価"""
        stability_scores = []
        
        for company in evolution['company'].unique():
            company_evolution = evolution[evolution['company'] == company]
            
            # 変動の標準偏差が小さいほど安定
            share_volatility = company_evolution['share_change'].std()
            profit_volatility = company_evolution['profitability_change'].std()
            
            stability_score = 100 - (share_volatility + profit_volatility) / 2
            
            stability_scores.append({
                'company': company,
                'stability_score': max(0, stability_score),
                'share_volatility': share_volatility,
                'profit_volatility': profit_volatility,
                'position_changes': len(company_evolution),
                'stability_rank': None  # 後で順位付け
            })
        
        stability_df = pd.DataFrame(stability_scores)
        stability_df['stability_rank'] = stability_df['stability_score'].rank(ascending=False)
        
        return stability_df
    
    def _analyze_price_competition(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """価格競争激化度分析"""
        price_competition_data = []
        
        for year in range(1985, 2025):  # 前年比較のため1985から
            current_data = market_data[market_data['year'] == year]
            previous_data = market_data[market_data['year'] == year - 1]
            
            if current_data.empty or previous_data.empty:
                continue
            
            # 平均売上原価率の変化
            current_cost_ratio = current_data['cost_ratio'].mean()
            previous_cost_ratio = previous_data['cost_ratio'].mean()
            cost_ratio_increase = current_cost_ratio - previous_cost_ratio
            
            # 平均営業利益率の変化
            current_margin = current_data['operating_margin'].mean()
            previous_margin = previous_data['operating_margin'].mean()
            margin_decline = previous_margin - current_margin
            
            # 価格競争激化スコア
            price_competition_score = (cost_ratio_increase * 0.6 + margin_decline * 0.4)
            
            # 企業間利益率格差の縮小（価格競争激化の兆候）
            margin_variance_current = current_data['operating_margin'].var()
            margin_variance_previous = previous_data['operating_margin'].var()
            margin_convergence = margin_variance_previous - margin_variance_current
            
            price_competition_data.append({
                'year': year,
                'cost_ratio_increase': cost_ratio_increase,
                'margin_decline': margin_decline,
                'margin_convergence': margin_convergence,
                'price_competition_score': price_competition_score,
                'competition_level': self._classify_price_competition_level(price_competition_score)
            })
        
        return pd.DataFrame(price_competition_data)
    
    def _classify_price_competition_level(self, score: float) -> str:
        """価格競争レベル分類"""
        if score > 5:
            return 'Very Intense'
        elif score > 2:
            return 'Intense'
        elif score > 0:
            return 'Moderate'
        else:
            return 'Low'
    
    def _analyze_innovation_competition(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """イノベーション競争分析"""
        innovation_data = []
        
        for year in range(1985, 2025):
            current_data = market_data[market_data['year'] == year]
            previous_data = market_data[market_data['year'] == year - 1]
            
            if current_data.empty:
                continue
            
            # R&D投資競争度
            avg_rd_ratio = current_data['rd_ratio'].mean()
            rd_variance = current_data['rd_ratio'].var()
            
            # 前年比R&D投資増加率
            if not previous_data.empty:
                rd_growth = (avg_rd_ratio - previous_data['rd_ratio'].mean()) / max(previous_data['rd_ratio'].mean(), 0.1) * 100
            else:
                rd_growth = 0
            
            # 特許・無形資産投資競争度
            avg_intangible_ratio = current_data.get('intangible_ratio', pd.Series([0])).mean()
            
            # 広告宣伝費競争度
            avg_ad_ratio = current_data.get('advertising_ratio', pd.Series([0])).mean()
            
            # イノベーション競争総合スコア
            innovation_score = (
                avg_rd_ratio * 0.4 +
                rd_variance * 0.2 +
                abs(rd_growth) * 0.1 +
                avg_intangible_ratio * 0.2 +
                avg_ad_ratio * 0.1
            )
            
            innovation_data.append({
                'year': year,
                'avg_rd_ratio': avg_rd_ratio,
                'rd_variance': rd_variance,
                'rd_growth': rd_growth,
                'avg_intangible_ratio': avg_intangible_ratio,
                'avg_ad_ratio': avg_ad_ratio,
                'innovation_score': innovation_score,
                'innovation_level': self._classify_innovation_level(innovation_score)
            })
        
        return pd.DataFrame(innovation_data)
    
    def _classify_innovation_level(self, score: float) -> str:
        """イノベーション競争レベル分類"""
        if score > 15:
            return 'Very High'
        elif score > 10:
            return 'High'
        elif score > 5:
            return 'Moderate'
        else:
            return 'Low'
    
    def _analyze_entry_exit_patterns(self, market_data: pd.DataFrame, 
                                    market_events: pd.DataFrame = None) -> pd.DataFrame:
        """市場参入・退出パターン分析"""
        pattern_data = []
        
        # 企業の市場での存在期間を分析
        company_lifecycles = {}
        for company in market_data['company'].unique():
            company_years = sorted(market_data[market_data['company'] == company]['year'].unique())
            if company_years:
                company_lifecycles[company] = {
                    'entry_year': company_years[0],
                    'exit_year': company_years[-1] if company_years[-1] < 2024 else None,
                    'active_years': len(company_years),
                    'lifecycle_stage': self._determine_lifecycle_stage(company_years)
                }
        
        # 年別参入・退出分析
        for year in range(1984, 2025):
            # その年に新規参入した企業数
            new_entrants = [
                company for company, lifecycle in company_lifecycles.items()
                if lifecycle['entry_year'] == year
            ]
            
            # その年に退出した企業数
            exits = [
                company for company, lifecycle in company_lifecycles.items()
                if lifecycle['exit_year'] == year
            ]
            
            # アクティブ企業数
            active_companies = [
                company for company, lifecycle in company_lifecycles.items()
                if lifecycle['entry_year'] <= year and 
                (lifecycle['exit_year'] is None or lifecycle['exit_year'] > year)
            ]
            
            # 市場集中度変化
            year_data = market_data[market_data['year'] == year]
            if not year_data.empty:
                market_shares = year_data['market_share'].fillna(0)
                hhi = sum(market_shares ** 2)
            else:
                hhi = 0
            
            pattern_data.append({
                'year': year,
                'new_entrants_count': len(new_entrants),
                'exits_count': len(exits),
                'net_change': len(new_entrants) - len(exits),
                'active_companies_count': len(active_companies),
                'hhi': hhi,
                'market_turbulence': len(new_entrants) + len(exits),
                'entry_barrier_strength': self._estimate_entry_barriers(year_data, new_entrants)
            })
        
        return pd.DataFrame(pattern_data)
    
    def _determine_lifecycle_stage(self, active_years: List[int]) -> str:
        """企業ライフサイクル段階判定"""
        duration = len(active_years)
        latest_year = max(active_years)
        
        if duration <= 5:
            return 'Startup'
        elif duration <= 15:
            return 'Growth'
        elif latest_year >= 2020:
            return 'Mature_Active'
        else:
            return 'Decline_Exit'
    
    def _estimate_entry_barriers(self, year_data: pd.DataFrame, new_entrants: List[str]) -> str:
        """参入障壁強度推定"""
        if year_data.empty:
            return 'Unknown'
        
        # 既存企業の平均規模
        avg_revenue = year_data['revenue'].mean()
        
        # R&D集約度
        avg_rd_ratio = year_data['rd_ratio'].mean()
        
        # 新規参入の難易度指標
        barrier_score = (
            min(avg_revenue / 10000000, 50) * 0.5 +  # 規模障壁
            avg_rd_ratio * 0.3 +  # 技術障壁
            (5 - len(new_entrants)) * 2 * 0.2  # 参入頻度（少ないほど障壁高）
        )
        
        if barrier_score > 60:
            return 'Very High'
        elif barrier_score > 40:
            return 'High'
        elif barrier_score > 20:
            return 'Moderate'
        else:
            return 'Low'
    
    def _analyze_competitive_cycles(self, intensity_metrics: pd.DataFrame) -> Dict[str, Any]:
        """競争サイクル分析"""
        if intensity_metrics.empty:
            return {}
        
        # 競争激化度の周期性検出
        intensity_scores = intensity_metrics['intensity_score'].values
        years = intensity_metrics['year'].values
        
        # 移動平均による平滑化
        window_size = 5
        if len(intensity_scores) >= window_size:
            smoothed_scores = pd.Series(intensity_scores).rolling(window=window_size).mean().fillna(method='bfill')
            
            # ピーク・トラフ検出
            peaks = []
            troughs = []
            
            for i in range(1, len(smoothed_scores) - 1):
                if smoothed_scores[i] > smoothed_scores[i-1] and smoothed_scores[i] > smoothed_scores[i+1]:
                    peaks.append((years[i], smoothed_scores[i]))
                elif smoothed_scores[i] < smoothed_scores[i-1] and smoothed_scores[i] < smoothed_scores[i+1]:
                    troughs.append((years[i], smoothed_scores[i]))
            
            # 周期長度計算
            cycle_lengths = []
            if len(peaks) > 1:
                for i in range(1, len(peaks)):
                    cycle_lengths.append(peaks[i][0] - peaks[i-1][0])
            
            avg_cycle_length = np.mean(cycle_lengths) if cycle_lengths else 0
            
            return {
                'peaks': peaks,
                'troughs': troughs,
                'avg_cycle_length': avg_cycle_length,
                'cycle_amplitude': np.std(intensity_scores),
                'current_phase': self._determine_cycle_phase(intensity_scores[-5:]),
                'cycle_regularity': self._assess_cycle_regularity(cycle_lengths)
            }
        
        return {}
    
    def _determine_cycle_phase(self, recent_scores: np.ndarray) -> str:
        """現在の競争サイクル段階判定"""
        if len(recent_scores) < 3:
            return 'Unknown'
        
        trend = np.polyfit(range(len(recent_scores)), recent_scores, 1)[0]
        
        if trend > 2:
            return 'Intensifying'
        elif trend < -2:
            return 'Cooling'
        else:
            return 'Stable'
    
    def _assess_cycle_regularity(self, cycle_lengths: List[float]) -> str:
        """サイクル規則性評価"""
        if not cycle_lengths or len(cycle_lengths) < 2:
            return 'Insufficient_Data'
        
        coefficient_of_variation = np.std(cycle_lengths) / np.mean(cycle_lengths)
        
        if coefficient_of_variation < 0.2:
            return 'Very_Regular'
        elif coefficient_of_variation < 0.4:
            return 'Regular'
        elif coefficient_of_variation < 0.6:
            return 'Somewhat_Irregular'
        else:
            return 'Highly_Irregular'
    
    def _calculate_overall_intensity(self, intensity_metrics: pd.DataFrame,
                                    price_competition: pd.DataFrame,
                                    innovation_competition: pd.DataFrame) -> pd.DataFrame:
        """総合競争激化度算出"""
        overall_data = []
        
        for year in range(1984, 2025):
            intensity_data = intensity_metrics[intensity_metrics['year'] == year]
            price_data = price_competition[price_competition['year'] == year]
            innovation_data = innovation_competition[innovation_competition['year'] == year]
            
            # 各競争次元のスコア取得
            intensity_score = intensity_data['intensity_score'].iloc[0] if not intensity_data.empty else 0
            price_score = price_data['price_competition_score'].iloc[0] if not price_data.empty else 0
            innovation_score = innovation_data['innovation_score'].iloc[0] if not innovation_data.empty else 0
            
            # 重み付け総合スコア
            overall_score = (
                intensity_score * 0.4 +
                price_score * 0.3 +
                innovation_score * 0.3
            )
            
            overall_data.append({
                'year': year,
                'intensity_component': intensity_score,
                'price_component': price_score,
                'innovation_component': innovation_score,
                'overall_intensity': overall_score,
                'intensity_classification': self._classify_overall_intensity(overall_score)
            })
        
        return pd.DataFrame(overall_data)
    
    def _classify_overall_intensity(self, score: float) -> str:
        """総合競争激化度分類"""
        if score > 50:
            return 'Hyper_Competitive'
        elif score > 35:
            return 'Highly_Competitive'
        elif score > 20:
            return 'Moderately_Competitive'
        elif score > 10:
            return 'Competitive'
        else:
            return 'Low_Competition'
    
    def _identify_strategic_groups(self, strategic_features: pd.DataFrame) -> Dict[str, Any]:
        """戦略グループ特定"""
        if strategic_features.empty:
            return {}
        
        # 最新年度のデータでクラスタリング
        latest_year = strategic_features['year'].max()
        latest_data = strategic_features[strategic_features['year'] == latest_year]
        
        if len(latest_data) < 4:  # クラスタリングに十分なデータがない場合
            return {'error': 'Insufficient data for clustering'}
        
        # 特徴量標準化
        features = ['cost_leadership_score', 'differentiation_score', 'focus_score', 'scale_score']
        feature_data = latest_data[features].fillna(0)
        
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(feature_data)
        
        # K-means クラスタリング（3-5グループ）
        optimal_clusters = min(5, max(3, len(latest_data) // 3))
        kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(scaled_features)
        
        # グループ特徴分析
        groups = {}
        for i in range(optimal_clusters):
            group_companies = latest_data[cluster_labels == i]['company'].tolist()
            group_features = feature_data[cluster_labels == i].mean()
            
            groups[f'Group_{i+1}'] = {
                'companies': group_companies,
                'strategic_profile': {
                    'cost_leadership': group_features['cost_leadership_score'],
                    'differentiation': group_features['differentiation_score'],
                    'focus': group_features['focus_score'],
                    'scale': group_features['scale_score']
                },
                'strategic_archetype': self._determine_strategic_archetype(group_features),
                'group_size': len(group_companies)
            }
        
        return {
            'groups': groups,
            'cluster_centers': kmeans.cluster_centers_,
            'inertia': kmeans.inertia_,
            'optimal_clusters': optimal_clusters
        }
    
    def _determine_strategic_archetype(self, features: pd.Series) -> str:
        """戦略原型判定"""
        max_feature = features.idxmax()
        max_value = features.max()
        
        if max_value < 30:
            return 'Undefined_Strategy'
        elif max_feature == 'cost_leadership_score':
            return 'Cost_Leaders'
        elif max_feature == 'differentiation_score':
            return 'Differentiators'
        elif max_feature == 'focus_score':
            return 'Focusers'
        elif max_feature == 'scale_score':
            return 'Scale_Players'
        else:
            return 'Hybrid_Strategy'
    
    def _analyze_inter_group_competition(self, strategic_groups: Dict[str, Any],
                                        strategic_features: pd.DataFrame) -> pd.DataFrame:
        """グループ間競争分析"""
        if 'groups' not in strategic_groups:
            return pd.DataFrame()
        
        inter_group_data = []
        groups = strategic_groups['groups']
        
        # 各グループペアの競争関係分析
        group_names = list(groups.keys())
        for i, group1 in enumerate(group_names):
            for j, group2 in enumerate(group_names):
                if i >= j:  # 重複回避
                    continue
                
                # 戦略的距離計算
                profile1 = list(groups[group1]['strategic_profile'].values())
                profile2 = list(groups[group2]['strategic_profile'].values())
                
                strategic_distance = np.linalg.norm(np.array(profile1) - np.array(profile2))
                
                # 競争激度推定（距離が近いほど競争激化）
                competition_intensity = max(0, 100 - strategic_distance * 2)
                
                inter_group_data.append({
                    'group1': group1,
                    'group2': group2,
                    'strategic_distance': strategic_distance,
                    'competition_intensity': competition_intensity,
                    'rivalry_level': self._classify_rivalry_level(competition_intensity)
                })
        
        return pd.DataFrame(inter_group_data)
    
    def _analyze_intra_group_competition(self, strategic_groups: Dict[str, Any],
                                        strategic_features: pd.DataFrame) -> pd.DataFrame:
        """グループ内競争分析"""
        if 'groups' not in strategic_groups:
            return pd.DataFrame()
        
        intra_group_data = []
        groups = strategic_groups['groups']
        
        for group_name, group_info in groups.items():
            companies = group_info['companies']
            
            if len(companies) < 2:
                continue
            
            # グループ内企業の戦略類似性分析
            group_features = strategic_features[
                strategic_features['company'].isin(companies)
            ]
            
            if group_features.empty:
                continue
            
            # 最新年度のデータを使用
            latest_year = group_features['year'].max()
            latest_group_data = group_features[group_features['year'] == latest_year]
            
            # 戦略特徴の分散（小さいほど類似）
            feature_vars = latest_group_data[
                ['cost_leadership_score', 'differentiation_score', 'focus_score', 'scale_score']
            ].var().mean()
            
            # グループ内競争激度（分散が小さく企業数が多いほど激化）
            intra_competition = (len(companies) - 1) * 10 / max(feature_vars, 1)
            
            intra_group_data.append({
                'group': group_name,
                'companies_count': len(companies),
                'strategic_variance': feature_vars,
                'intra_competition_score': min(intra_competition, 100),
                'competition_level': self._classify_rivalry_level(intra_competition)
            })
        
        return pd.DataFrame(intra_group_data)
    
    def _classify_rivalry_level(self, intensity: float) -> str:
        """競争激度レベル分類"""
        if intensity > 70:
            return 'Very_High'
        elif intensity > 50:
            return 'High'
        elif intensity > 30:
            return 'Moderate'
        elif intensity > 10:
            return 'Low'
        else:
            return 'Very_Low'
    
    def _analyze_strategic_mobility(self, strategic_groups: Dict[str, Any],
                                    market_data: pd.DataFrame) -> pd.DataFrame:
        """戦略移動分析"""
        if 'groups' not in strategic_groups:
            return pd.DataFrame()
        
        mobility_data = []
        
        # 過去10年間の戦略グループ変遷を分析
        for company in market_data['company'].unique():
            company_data = market_data[market_data['company'] == company]
            
            # 企業の戦略進化パターン分析
            strategic_evolution = []
            for year in sorted(company_data['year'].unique())[-10:]:  # 最近10年
                year_data = company_data[company_data['year'] == year]
                if not year_data.empty:
                    strategic_evolution.append({
                        'year': year,
                        'strategic_type': year_data['strategic_type'].iloc[0] if 'strategic_type' in year_data.columns else 'Unknown'
                    })
            
            if len(strategic_evolution) < 2:
                continue
            
            # 戦略変更頻度
            strategy_changes = sum(
                1 for i in range(1, len(strategic_evolution))
                if strategic_evolution[i]['strategic_type'] != strategic_evolution[i-1]['strategic_type']
            )
            
            mobility_score = (strategy_changes / max(len(strategic_evolution) - 1, 1)) * 100
            
            mobility_data.append({
                'company': company,
                'strategy_changes': strategy_changes,
                'observation_years': len(strategic_evolution),
                'mobility_score': mobility_score,
                'mobility_level': self._classify_mobility_level(mobility_score),
                'current_strategy': strategic_evolution[-1]['strategic_type'],
                'strategic_consistency': 100 - mobility_score
            })
        
        return pd.DataFrame(mobility_data)
    
    def _classify_mobility_level(self, score: float) -> str:
        """戦略移動性レベル分類"""
        if score > 50:
            return 'Highly_Mobile'
        elif score > 30:
            return 'Moderately_Mobile'
        elif score > 10:
            return 'Low_Mobility'
        else:
            return 'Stable'
    
    def _analyze_group_performance(self, strategic_groups: Dict[str, Any],
                                    market_data: pd.DataFrame) -> pd.DataFrame:
        """戦略グループ別業績分析"""
        if 'groups' not in strategic_groups:
            return pd.DataFrame()
        
        group_performance_data = []
        
        for group_name, group_info in strategic_groups['groups'].items():
            companies = group_info['companies']
            
            # グループ企業の財務データ抽出
            group_data = market_data[market_data['company'].isin(companies)]
            
            if group_data.empty:
                continue
            
            # 最新5年間の平均業績
            recent_years = sorted(group_data['year'].unique())[-5:]
            recent_data = group_data[group_data['year'].isin(recent_years)]
            
            # 業績指標算出
            avg_roe = recent_data['roe'].mean() if 'roe' in recent_data.columns else 0
            avg_operating_margin = recent_data['operating_margin'].mean() if 'operating_margin' in recent_data.columns else 0
            avg_revenue_growth = recent_data['revenue_growth'].mean() if 'revenue_growth' in recent_data.columns else 0
            avg_market_share = recent_data['market_share'].mean() if 'market_share' in recent_data.columns else 0
            
            # リスク指標（業績変動性）
            roe_volatility = recent_data['roe'].std() if 'roe' in recent_data.columns else 0
            revenue_volatility = recent_data['revenue_growth'].std() if 'revenue_growth' in recent_data.columns else 0
            
            # 総合業績スコア
            performance_score = (
                avg_roe * 0.3 +
                avg_operating_margin * 0.3 +
                avg_revenue_growth * 0.2 +
                avg_market_share * 0.2
            )
            
            group_performance_data.append({
                'group': group_name,
                'strategic_archetype': group_info['strategic_archetype'],
                'avg_roe': avg_roe,
                'avg_operating_margin': avg_operating_margin,
                'avg_revenue_growth': avg_revenue_growth,
                'avg_market_share': avg_market_share,
                'roe_volatility': roe_volatility,
                'revenue_volatility': revenue_volatility,
                'performance_score': performance_score,
                'performance_rank': None,  # 後で順位付け
                'risk_adjusted_performance': performance_score / max(roe_volatility + revenue_volatility, 1)
            })
        
        performance_df = pd.DataFrame(group_performance_data)
        if not performance_df.empty:
            performance_df['performance_rank'] = performance_df['performance_score'].rank(ascending=False)
        
        return performance_df
    
    def _build_response_matrix(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """競争行動-対応マトリクス構築"""
        response_data = []
        
        companies = market_data['company'].unique()
        
        # 各企業ペアの戦略的相互作用分析
        for focal_company in companies:
            focal_data = market_data[market_data['company'] == focal_company].sort_values('year')
            
            for competitor in companies:
                if focal_company == competitor:
                    continue
                
                competitor_data = market_data[market_data['company'] == competitor].sort_values('year')
                
                # 共通年度でのデータマッチング
                common_years = set(focal_data['year']).intersection(set(competitor_data['year']))
                
                if len(common_years) < 3:
                    continue
                
                # 戦略的行動の相関分析
                correlations = self._calculate_strategic_correlations(
                    focal_data, competitor_data, list(common_years)
                )
                
                # 対応時差分析
                response_lags = self._analyze_response_lags(
                    focal_data, competitor_data, list(common_years)
                )
                
                response_data.append({
                    'focal_company': focal_company,
                    'competitor': competitor,
                    'rd_correlation': correlations.get('rd_correlation', 0),
                    'investment_correlation': correlations.get('investment_correlation', 0),
                    'pricing_correlation': correlations.get('pricing_correlation', 0),
                    'avg_response_lag': response_lags.get('avg_lag', 0),
                    'response_strength': correlations.get('overall_correlation', 0),
                    'competitive_relationship': self._classify_competitive_relationship(
                        correlations.get('overall_correlation', 0), response_lags.get('avg_lag', 0)
                    )
                })
        
        return pd.DataFrame(response_data)
    
    def _calculate_strategic_correlations(self, focal_data: pd.DataFrame,
                                        competitor_data: pd.DataFrame,
                                        common_years: List[int]) -> Dict[str, float]:
        """戦略行動相関分析"""
        correlations = {}
        
        try:
            # R&D投資相関
            focal_rd = focal_data[focal_data['year'].isin(common_years)]['rd_ratio'].fillna(0)
            comp_rd = competitor_data[competitor_data['year'].isin(common_years)]['rd_ratio'].fillna(0)
            
            if len(focal_rd) > 1 and len(comp_rd) > 1:
                rd_corr, _ = pearsonr(focal_rd, comp_rd)
                correlations['rd_correlation'] = rd_corr if not np.isnan(rd_corr) else 0
            
            # 設備投資相関
            focal_capex = focal_data[focal_data['year'].isin(common_years)].get('capex_ratio', focal_data[focal_data['year'].isin(common_years)].get('investment_ratio', pd.Series([0]))).fillna(0)
            comp_capex = competitor_data[competitor_data['year'].isin(common_years)].get('capex_ratio', competitor_data[competitor_data['year'].isin(common_years)].get('investment_ratio', pd.Series([0]))).fillna(0)
            
            if len(focal_capex) > 1 and len(comp_capex) > 1:
                capex_corr, _ = pearsonr(focal_capex, comp_capex)
                correlations['investment_correlation'] = capex_corr if not np.isnan(capex_corr) else 0
            
            # 価格戦略相関（営業利益率の逆相関として近似）
            focal_margin = focal_data[focal_data['year'].isin(common_years)]['operating_margin'].fillna(0)
            comp_margin = competitor_data[competitor_data['year'].isin(common_years)]['operating_margin'].fillna(0)
            
            if len(focal_margin) > 1 and len(comp_margin) > 1:
                pricing_corr, _ = pearsonr(focal_margin, comp_margin)
                correlations['pricing_correlation'] = pricing_corr if not np.isnan(pricing_corr) else 0
            
            # 総合相関スコア
            correlations['overall_correlation'] = np.mean([
                correlations.get('rd_correlation', 0),
                correlations.get('investment_correlation', 0),
                abs(correlations.get('pricing_correlation', 0))  # 価格競争では負の相関も重要
            ])
            
        except Exception as e:
            # エラーハンドリング
            correlations = {
                'rd_correlation': 0,
                'investment_correlation': 0,
                'pricing_correlation': 0,
                'overall_correlation': 0
            }
        
        return correlations
    
    def _analyze_response_lags(self, focal_data: pd.DataFrame,
                                competitor_data: pd.DataFrame,
                                common_years: List[int]) -> Dict[str, float]:
        """対応時差分析"""
        lags = []
        
        try:
            # R&D投資の急激な変化点を特定
            focal_rd_changes = self._detect_strategic_changes(
                focal_data[focal_data['year'].isin(common_years)], 'rd_ratio'
            )
            
            competitor_rd_changes = self._detect_strategic_changes(
                competitor_data[competitor_data['year'].isin(common_years)], 'rd_ratio'
            )
            
            # 変化点の時差を分析
            for focal_change in focal_rd_changes:
                for comp_change in competitor_rd_changes:
                    if comp_change['year'] > focal_change['year']:
                        lag = comp_change['year'] - focal_change['year']
                        if lag <= 3:  # 3年以内の対応のみ考慮
                            lags.append(lag)
            
            return {
                'avg_lag': np.mean(lags) if lags else 0,
                'min_lag': min(lags) if lags else 0,
                'max_lag': max(lags) if lags else 0,
                'response_count': len(lags)
            }
            
        except Exception:
            return {'avg_lag': 0, 'min_lag': 0, 'max_lag': 0, 'response_count': 0}
    
    def _detect_strategic_changes(self, data: pd.DataFrame, metric: str) -> List[Dict]:
        """戦略的変化点検出"""
        changes = []
        
        if metric not in data.columns or len(data) < 3:
            return changes
        
        values = data[metric].fillna(0).values
        years = data['year'].values
        
        # 移動平均による平滑化
        if len(values) >= 3:
            smoothed = pd.Series(values).rolling(window=3, center=True).mean().fillna(method='bfill').fillna(method='ffill')
            
            # 変化率が閾値を超える点を特定
            for i in range(1, len(smoothed)):
                change_rate = abs((smoothed[i] - smoothed[i-1]) / max(abs(smoothed[i-1]), 0.1))
                
                if change_rate > 0.2:  # 20%以上の変化
                    changes.append({
                        'year': years[i],
                        'metric': metric,
                        'change_rate': change_rate,
                        'direction': 'increase' if smoothed[i] > smoothed[i-1] else 'decrease'
                    })
        
        return changes
    
    def _classify_competitive_relationship(self, correlation: float, avg_lag: float) -> str:
        """競争関係分類"""
        if abs(correlation) > 0.7:
            if avg_lag <= 1:
                return 'Simultaneous_Response'
            else:
                return 'Follower_Leader'
        elif abs(correlation) > 0.4:
            return 'Moderate_Interaction'
        elif abs(correlation) > 0.2:
            return 'Weak_Interaction'
        else:
            return 'Independent'
    
    def _analyze_response_timing(self, response_matrix: pd.DataFrame) -> pd.DataFrame:
        """対応タイミング分析"""
        if response_matrix.empty:
            return pd.DataFrame()
        
        timing_data = []
        
        # 企業別対応速度分析
        for company in response_matrix['focal_company'].unique():
            company_responses = response_matrix[response_matrix['focal_company'] == company]
            
            avg_response_lag = company_responses['avg_response_lag'].mean()
            response_strength = company_responses['response_strength'].mean()
            
            # 対応パターン分類
            response_pattern = self._classify_response_pattern(avg_response_lag, response_strength)
            
            timing_data.append({
                'company': company,
                'avg_response_lag': avg_response_lag,
                'response_strength': response_strength,
                'response_pattern': response_pattern,
                'competitive_aggressiveness': self._calculate_aggressiveness(company_responses)
            })
        
        return pd.DataFrame(timing_data)
    
    def _classify_response_pattern(self, avg_lag: float, strength: float) -> str:
        """対応パターン分類"""
        if strength > 0.6:
            if avg_lag < 1:
                return 'Fast_Responder'
            else:
                return 'Strong_Follower'
        elif strength > 0.3:
            if avg_lag < 1:
                return 'Quick_Adapter'
            else:
                return 'Moderate_Follower'
        else:
            return 'Independent_Player'
    
    def _calculate_aggressiveness(self, company_responses: pd.DataFrame) -> float:
        """競争攻撃性算出"""
        # 高相関かつ短時差の対応が多いほど攻撃的
        aggressive_responses = company_responses[
            (company_responses['response_strength'] > 0.5) &
            (company_responses['avg_response_lag'] <= 1)
        ]
        
        return len(aggressive_responses) / max(len(company_responses), 1) * 100
    
    def _analyze_response_intensity(self, response_matrix: pd.DataFrame) -> pd.DataFrame:
        """対応強度分析"""
        if response_matrix.empty:
            return pd.DataFrame()
        
        intensity_data = []
        
        # 戦略次元別対応強度
        for company in response_matrix['focal_company'].unique():
            company_data = response_matrix[response_matrix['focal_company'] == company]
            
            rd_intensity = company_data['rd_correlation'].abs().mean()
            investment_intensity = company_data['investment_correlation'].abs().mean()
            pricing_intensity = company_data['pricing_correlation'].abs().mean()
            
            overall_intensity = (rd_intensity + investment_intensity + pricing_intensity) / 3
            
            intensity_data.append({
                'company': company,
                'rd_response_intensity': rd_intensity,
                'investment_response_intensity': investment_intensity,
                'pricing_response_intensity': pricing_intensity,
                'overall_response_intensity': overall_intensity,
                'dominant_response_dimension': self._identify_dominant_dimension(
                    rd_intensity, investment_intensity, pricing_intensity
                )
            })
        
        return pd.DataFrame(intensity_data)
    
    def _identify_dominant_dimension(self, rd: float, investment: float, pricing: float) -> str:
        """支配的対応次元特定"""
        intensities = {'R&D': rd, 'Investment': investment, 'Pricing': pricing}
        return max(intensities, key=intensities.get)
    
    def _analyze_imitation_patterns(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """模倣パターン分析"""
        imitation_data = []
        
        companies = market_data['company'].unique()
        
        for company in companies:
            company_data = market_data[market_data['company'] == company].sort_values('year')
            
            if len(company_data) < 5:
                continue
            
            # 業界平均との収束度分析
            convergence_scores = []
            
            for year in company_data['year'].unique():
                year_market_data = market_data[market_data['year'] == year]
                company_year_data = company_data[company_data['year'] == year]
                
                if year_market_data.empty or company_year_data.empty:
                    continue
                
                # 主要戦略指標での業界平均からの距離
                metrics = ['rd_ratio', 'operating_margin', 'cost_ratio']
                distances = []
                
                for metric in metrics:
                    if metric in year_market_data.columns:
                        industry_avg = year_market_data[metric].mean()
                        company_value = company_year_data[metric].iloc[0]
                        
                        if not pd.isna(industry_avg) and not pd.isna(company_value):
                            distance = abs(company_value - industry_avg) / max(abs(industry_avg), 1)
                            distances.append(distance)
                
                if distances:
                    convergence_scores.append({
                        'year': year,
                        'avg_distance_from_industry': np.mean(distances)
                    })
            
            if len(convergence_scores) < 3:
                continue
            
            # 収束トレンド分析
            distances = [score['avg_distance_from_industry'] for score in convergence_scores]
            years_range = [score['year'] for score in convergence_scores]
            
            # 線形トレンド
            if len(distances) > 1:
                trend_slope, _ = np.polyfit(years_range, distances, 1)
                
                imitation_score = max(0, -trend_slope * 10)  # 距離縮小をプラス評価
                
                imitation_data.append({
                    'company': company,
                    'convergence_trend': trend_slope,
                    'imitation_score': imitation_score,
                    'imitation_level': self._classify_imitation_level(imitation_score),
                    'current_differentiation': distances[-1] if distances else 0
                })
        
        return pd.DataFrame(imitation_data)
    
    def _classify_imitation_level(self, score: float) -> str:
        """模倣レベル分類"""
        if score > 50:
            return 'High_Imitator'
        elif score > 25:
            return 'Moderate_Imitator'
        elif score > 10:
            return 'Low_Imitator'
        else:
            return 'Independent'
    
    def _analyze_innovation_followership(self, market_data: pd.DataFrame,
                                        market_events: pd.DataFrame = None) -> Dict[str, Any]:
        """革新-追従関係分析"""
        if market_data.empty:
            return {}
        
        # R&D投資による革新リーダーシップ分析
        innovation_leaders = {}
        
        for year in range(1990, 2025, 5):  # 5年ごとの分析
            year_data = market_data[
                (market_data['year'] >= year) & 
                (market_data['year'] < year + 5)
            ]
            
            if year_data.empty:
                continue
            
            # R&D投資率上位企業を革新リーダーとして特定
            rd_leaders = year_data.groupby('company')['rd_ratio'].mean().nlargest(3)
            
            innovation_leaders[f'{year}s'] = {
                'leaders': rd_leaders.to_dict(),
                'avg_rd_ratio': rd_leaders.mean(),
                'leadership_concentration': rd_leaders.std()
            }
        
        # 追従パターン分析
        followership_patterns = self._analyze_followership_patterns(market_data)
        
        return {
            'innovation_leaders_by_period': innovation_leaders,
            'followership_patterns': followership_patterns,
            'innovation_diffusion_speed': self._calculate_diffusion_speed(market_data)
        }
    
    def _analyze_followership_patterns(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """追従パターン分析"""
        followership_data = []
        
        # 各企業の革新採用ラグを分析
        companies = market_data['company'].unique()
        
        for company in companies:
            company_data = market_data[market_data['company'] == company].sort_values('year')
            
            # R&D投資の急増タイミングを特定
            rd_increases = self._detect_strategic_changes(company_data, 'rd_ratio')
            
            if not rd_increases:
                continue
            
            # 業界全体のR&D投資トレンドとの比較
            industry_rd_trend = market_data.groupby('year')['rd_ratio'].mean()
            
            adoption_lags = []
            for increase in rd_increases:
                # 業界でR&D投資が増加し始めた時期を特定
                increase_year = increase['year']
                industry_increase_years = []
                
                for year in range(max(1984, increase_year - 5), increase_year):
                    if year in industry_rd_trend.index and year + 1 in industry_rd_trend.index:
                        if industry_rd_trend[year + 1] > industry_rd_trend[year] * 1.1:  # 10%以上増加
                            industry_increase_years.append(year)
                
                if industry_increase_years:
                    earliest_industry_increase = min(industry_increase_years)
                    lag = increase_year - earliest_industry_increase
                    adoption_lags.append(max(0, lag))
            
            if adoption_lags:
                avg_lag = np.mean(adoption_lags)
                followership_data.append({
                    'company': company,
                    'avg_adoption_lag': avg_lag,
                    'innovation_adoption_count': len(adoption_lags),
                    'followership_type': self._classify_followership_type(avg_lag)
                })
        
        return pd.DataFrame(followership_data)
    
    def _classify_followership_type(self, avg_lag: float) -> str:
        """追従タイプ分類"""
        if avg_lag <= 1:
            return 'Fast_Follower'
        elif avg_lag <= 3:
            return 'Moderate_Follower'
        elif avg_lag <= 5:
            return 'Slow_Follower'
        else:
            return 'Late_Adopter'
    
    def _calculate_diffusion_speed(self, market_data: pd.DataFrame) -> Dict[str, float]:
        """イノベーション普及速度算出"""
        # R&D投資の市場全体への普及速度を分析
        rd_adoption_data = []
        
        # 年別R&D投資企業数の推移
        for year in range(1985, 2025):
            year_data = market_data[market_data['year'] == year]
            
            if year_data.empty:
                continue
            
            # R&D投資を行っている企業数（閾値以上）
            rd_threshold = year_data['rd_ratio'].median()
            rd_adopters = len(year_data[year_data['rd_ratio'] > rd_threshold])
            total_companies = len(year_data)
            
            adoption_rate = rd_adopters / max(total_companies, 1) * 100
            
            rd_adoption_data.append({
                'year': year,
                'adoption_rate': adoption_rate
            })
        
        if len(rd_adoption_data) < 10:
            return {'diffusion_speed': 0, 'saturation_level': 0}
        
        adoption_rates = [data['adoption_rate'] for data in rd_adoption_data]
        years_range = [data['year'] for data in rd_adoption_data]
        
        # S字カーブフィッティング（簡易版）
        diffusion_speed = np.gradient(adoption_rates).mean()  # 平均普及速度
        saturation_level = max(adoption_rates)  # 飽和水準
        
        return {
            'diffusion_speed': diffusion_speed,
            'saturation_level': saturation_level
        }
    
    def _calculate_rivalry_index(self, response_matrix: pd.DataFrame,
                                response_timing: pd.DataFrame,
                                response_intensity: pd.DataFrame) -> float:
        """競争激度指数算出"""
        if response_matrix.empty:
            return 0
        
        # 複数次元の競争激度を統合
        avg_correlation = response_matrix['response_strength'].mean()
        avg_response_speed = 1 / max(response_timing['avg_response_lag'].mean(), 0.1)
        avg_intensity = response_intensity['overall_response_intensity'].mean()
        
        rivalry_index = (
            avg_correlation * 0.4 +
            min(avg_response_speed, 10) * 0.3 +  # 速度は上限設定
            avg_intensity * 0.3
        ) * 10  # スケール調整
        
        return min(rivalry_index, 100)  # 0-100に正規化
    
    def _identify_new_entrants(self, market_companies: List[str],
                                entry_events: pd.DataFrame) -> List[Dict[str, Any]]:
        """新規参入企業特定"""
        new_entrants = []
        
        for company in market_companies:
            # 企業設立年や市場参入年を特定
            if entry_events is not None and not entry_events.empty:
                entry_info = entry_events[entry_events['company'] == company]
                if not entry_info.empty:
                    entry_year = entry_info['entry_year'].iloc[0]
                    entry_type = entry_info.get('entry_type', 'Unknown').iloc[0]
                    
                    if entry_year >= 2000:  # 2000年以降の参入を新規参入とする
                        new_entrants.append({
                            'company': company,
                            'entry_year': entry_year,
                            'entry_type': entry_type
                        })
        
        return new_entrants
    
    # 最終的なインサイト生成メソッド群
    def _summarize_market_dynamics(self, market_category: str) -> Dict[str, Any]:
        """市場ダイナミクス要約"""
        summary = {}
        
        if market_category in self.competitive_positions:
            positioning = self.competitive_positions[market_category]
            
            # 主要な競争パターン
            summary['dominant_positions'] = positioning.get('market_leaders', pd.DataFrame()).to_dict('records')
            summary['position_volatility'] = positioning.get('position_stability', pd.DataFrame())['stability_score'].std()
        
        if market_category in self.competitive_intensity:
            intensity = self.competitive_intensity[market_category]
            
            summary['avg_intensity_score'] = intensity['intensity_metrics']['intensity_score'].mean()
            summary['competition_trend'] = self._determine_competition_trend(intensity['intensity_metrics'])
        
        return summary
    
    def _determine_competition_trend(self, intensity_metrics: pd.DataFrame) -> str:
        """競争トレンド判定"""
        if len(intensity_metrics) < 5:
            return 'Insufficient_Data'
        
        recent_scores = intensity_metrics.tail(5)['intensity_score'].values
        trend = np.polyfit(range(len(recent_scores)), recent_scores, 1)[0]
        
        if trend > 2:
            return 'Intensifying'
        elif trend < -2:
            return 'Cooling'
        else:
            return 'Stable'
    
    def _identify_competitive_patterns(self, market_category: str) -> Dict[str, Any]:
        """競争パターン特定"""
        patterns = {}
        
        if market_category in self.strategic_groups:
            groups = self.strategic_groups[market_category]
            
            patterns['strategic_groups_count'] = len(groups.get('groups', {}))
            patterns['dominant_strategies'] = [
                group_info.get('strategic_archetype', 'Unknown')
                for group_info in groups.get('groups', {}).values()
            ]
        
        if market_category in self.competitive_responses:
            responses = self.competitive_responses[market_category]
            
            patterns['rivalry_index'] = responses.get('competitive_rivalry_index', 0)
            patterns['imitation_prevalence'] = len(
                responses.get('imitation_patterns', pd.DataFrame())[
                    responses.get('imitation_patterns', pd.DataFrame()).get('imitation_level', '') == 'High_Imitator'
                ]
            )
        
        return patterns
    
    def _generate_strategic_recommendations(self, market_category: str) -> List[str]:
        """戦略提言生成"""
        recommendations = []
        
        # 市場カテゴリ別の基本提言
        if market_category == 'high_share':
            recommendations.extend([
                "持続的競争優位の維持: 技術革新と品質向上に継続投資",
                "新規参入阻止: 参入障壁の強化と市場支配力の維持",
                "グローバル展開: 高シェア市場での経験を活かした国際展開"
            ])
        elif market_category == 'declining':
            recommendations.extend([
                "競争力強化: コスト削減と差別化の両立による競争力回復",
                "イノベーション加速: 破壊的技術への積極投資",
                "戦略的提携: 業界再編を見据えた戦略的パートナーシップ"
            ])
        elif market_category == 'lost':
            recommendations.extend([
                "事業転換: 成長市場への資源再配分",
                "ニッチ戦略: 残存市場でのニッチポジション確立",
                "撤退判断: 適切なタイミングでの事業撤退・売却検討"
            ])
        
        # 競争分析結果に基づく追加提言
        if market_category in self.competitive_intensity:
            intensity = self.competitive_intensity[market_category]
            avg_intensity = intensity['overall_intensity_score']['overall_intensity'].mean()
            
            if avg_intensity > 50:
                recommendations.append("競争激化対応: 価格競争回避のための差別化戦略強化")
            elif avg_intensity < 20:
                recommendations.append("市場活性化: 積極的な市場拡大戦略の検討")
        
        return recommendations
    
    def _generate_risk_warnings(self, market_category: str) -> List[str]:
        """リスク警告生成"""
        warnings = []
        
        if market_category in self.competitive_intensity:
            intensity = self.competitive_intensity[market_category]
            
            # 競争激化リスク
            recent_intensity = intensity['intensity_metrics'].tail(3)['intensity_score'].mean()
            if recent_intensity > 60:
                warnings.append("競争激化リスク: 過度な価格競争による収益性悪化の懸念")
            
            # 市場集中度リスク
            recent_hhi = intensity['intensity_metrics'].tail(1)['hhi'].iloc[0]
            if recent_hhi < 1000:
                warnings.append("市場分散リスク: 低集中度による競争の不安定化")
            elif recent_hhi > 3000:
                warnings.append("寡占化リスク: 高集中度による競争当局の介入可能性")
        
        if market_category in self.strategic_groups:
            groups = self.strategic_groups[market_category]
            
            # 戦略群の空洞化リスク
            if len(groups.get('groups', {})) < 2:
                warnings.append("戦略単調化リスク: 戦略の画一化による差別化困難")
        
        # 市場カテゴリ特有のリスク
        if market_category == 'declining':
            warnings.extend([
                "市場縮小リスク: 継続的な需要減少による事業持続性の懸念",
                "技術陳腐化リスク: 既存技術の競争力喪失"
            ])
        elif market_category == 'lost':
            warnings.extend([
                "事業存続リスク: 市場からの完全撤退を余儀なくされる可能性",
                "資産座礁リスク: 専用設備・技術の価値毀損"
            ])
        
        return warnings