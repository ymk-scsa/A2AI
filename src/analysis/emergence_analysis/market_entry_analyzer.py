"""
A2AI (Advanced Financial Analysis AI) - Market Entry Analyzer

新設企業の市場参入分析モジュール
企業の市場参入タイミング、戦略、成功要因を分析し、参入パターンを特定する

Author: A2AI Development Team
Date: 2024
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import scipy.stats as stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

@dataclass
class MarketEntryMetrics:
    """市場参入分析結果を格納するデータクラス"""
    company_name: str
    market_category: str
    entry_year: int
    entry_timing_score: float  # 先発/後発スコア（0-1、1が先発）
    initial_market_share: float
    entry_strategy_type: str  # 'aggressive', 'conservative', 'niche'
    capital_intensity: float
    rd_intensity: float
    market_growth_rate: float
    competitive_density: float
    success_probability: float
    time_to_profitability: Optional[int]
    peak_market_share: float
    sustainability_score: float

class MarketEntryAnalyzer:
    """
    新設企業の市場参入分析クラス
    
    主要機能:
    - 参入タイミング分析（先発者優位 vs 後発者優位）
    - 参入戦略分類（アグレッシブ/保守的/ニッチ）
    - 成功要因特定
    - 市場環境と参入戦略の最適マッチング分析
    """
    
    def __init__(self, config: Dict = None):
        """
        初期化
        
        Args:
            config: 分析設定辞書
        """
        self.config = config or {}
        self.entry_metrics: List[MarketEntryMetrics] = []
        self.market_categories = {
            'high_share': ['ロボット', '内視鏡', '工作機械', '電子材料', '精密測定機器'],
            'declining': ['自動車', '鉄鋼', 'スマート家電', 'バッテリー', 'PC周辺機器'],
            'lost': ['家電', '半導体', 'スマートフォン', 'PC', '通信機器']
        }
        
    def analyze_market_entry_patterns(self, 
                                    financial_data: pd.DataFrame,
                                    company_data: pd.DataFrame,
                                    market_data: pd.DataFrame) -> Dict:
        """
        市場参入パターンの包括的分析
        
        Args:
            financial_data: 財務データ
            company_data: 企業基本情報（設立年、業界等）
            market_data: 市場データ（市場規模、成長率等）
            
        Returns:
            分析結果辞書
        """
        results = {
            'entry_timing_analysis': {},
            'strategy_classification': {},
            'success_factors': {},
            'market_timing_optimization': {},
            'competitive_analysis': {}
        }
        
        # 1. 市場参入タイミング分析
        results['entry_timing_analysis'] = self._analyze_entry_timing(
            financial_data, company_data, market_data
        )
        
        # 2. 参入戦略分類
        results['strategy_classification'] = self._classify_entry_strategies(
            financial_data, company_data
        )
        
        # 3. 成功要因分析
        results['success_factors'] = self._analyze_success_factors(
            financial_data, company_data, market_data
        )
        
        # 4. 市場タイミング最適化分析
        results['market_timing_optimization'] = self._optimize_market_timing(
            financial_data, market_data
        )
        
        # 5. 競争環境分析
        results['competitive_analysis'] = self._analyze_competitive_environment(
            financial_data, company_data, market_data
        )
        
        return results
    
    def _analyze_entry_timing(self, 
                                financial_data: pd.DataFrame,
                                company_data: pd.DataFrame, 
                                market_data: pd.DataFrame) -> Dict:
        """
        市場参入タイミング分析
        
        先発者優位 vs 後発者優位を定量化
        """
        timing_analysis = {}
        
        for market_cat in ['high_share', 'declining', 'lost']:
            market_companies = self._filter_companies_by_category(company_data, market_cat)
            
            # 各企業の参入タイミングスコア計算
            timing_scores = []
            market_performance = []
            
            for _, company in market_companies.iterrows():
                # 市場参入年の特定
                entry_year = self._determine_entry_year(company, financial_data)
                if entry_year is None:
                    continue
                
                # 市場成熟度計算（参入時点での市場年数）
                market_maturity = self._calculate_market_maturity(entry_year, market_cat)
                
                # 先発/後発スコア（0-1、1が完全先発）
                timing_score = max(0, 1 - market_maturity / 50)  # 50年で正規化
                
                # 参入後の市場シェア推移
                performance = self._calculate_post_entry_performance(
                    company['企業名'], financial_data, entry_year
                )
                
                timing_scores.append(timing_score)
                market_performance.append(performance)
            
            # 先発者優位 vs 後発者優位の分析
            if len(timing_scores) > 5:  # 統計的有意性のため
                correlation = stats.pearsonr(timing_scores, market_performance)[0]
                timing_analysis[market_cat] = {
                    'first_mover_advantage': correlation > 0.3,
                    'late_mover_advantage': correlation < -0.3,
                    'correlation_coefficient': correlation,
                    'optimal_timing_range': self._find_optimal_timing_range(
                        timing_scores, market_performance
                    ),
                    'timing_distribution': {
                        'early_entrants': np.sum(np.array(timing_scores) > 0.7),
                        'mid_entrants': np.sum((np.array(timing_scores) >= 0.3) & 
                                                (np.array(timing_scores) <= 0.7)),
                        'late_entrants': np.sum(np.array(timing_scores) < 0.3)
                    }
                }
        
        return timing_analysis
    
    def _classify_entry_strategies(self, 
                                    financial_data: pd.DataFrame,
                                    company_data: pd.DataFrame) -> Dict:
        """
        参入戦略の分類分析
        
        アグレッシブ/保守的/ニッチ戦略の特定
        """
        strategy_classification = {}
        
        # 戦略分類のための特徴量定義
        strategy_features = [
            'initial_capex_ratio',  # 初期設備投資比率
            'rd_intensity',         # R&D集約度
            'marketing_intensity',  # マーケティング集約度
            'employee_growth_rate', # 従業員増加率
            'geographic_expansion', # 地域展開速度
            'product_diversification' # 製品多様化度
        ]
        
        for market_cat in self.market_categories:
            market_companies = self._filter_companies_by_category(company_data, market_cat)
            
            # 各企業の戦略特徴量計算
            strategy_data = []
            company_names = []
            
            for _, company in market_companies.iterrows():
                features = self._extract_strategy_features(
                    company['企業名'], financial_data
                )
                if features is not None:
                    strategy_data.append(features)
                    company_names.append(company['企業名'])
            
            if len(strategy_data) > 3:  # 最小クラスタリング要件
                # 戦略クラスタリング
                scaler = StandardScaler()
                scaled_features = scaler.fit_transform(strategy_data)
                
                # 最適クラスタ数決定（エルボー法）
                optimal_k = self._find_optimal_clusters(scaled_features)
                
                # K-means クラスタリング
                kmeans = KMeans(n_clusters=optimal_k, random_state=42)
                strategy_labels = kmeans.fit_predict(scaled_features)
                
                # クラスタ特徴分析
                clusters_analysis = self._analyze_strategy_clusters(
                    strategy_data, strategy_labels, company_names, optimal_k
                )
                
                strategy_classification[market_cat] = {
                    'n_clusters': optimal_k,
                    'cluster_analysis': clusters_analysis,
                    'strategy_types': self._map_clusters_to_strategies(clusters_analysis),
                    'success_by_strategy': self._analyze_strategy_success(
                        strategy_labels, company_names, financial_data
                    )
                }
        
        return strategy_classification
    
    def _analyze_success_factors(self,
                                financial_data: pd.DataFrame,
                                company_data: pd.DataFrame,
                                market_data: pd.DataFrame) -> Dict:
        """
        新設企業の成功要因分析
        
        成功/失敗を分ける重要要因の特定
        """
        success_factors = {}
        
        # 成功指標の定義
        success_metrics = [
            'market_share_growth',     # 市場シェア成長
            'revenue_growth',          # 売上成長
            'profitability_achievement', # 収益性達成
            'survival_duration'        # 生存期間
        ]
        
        # 要因項目（23項目）
        factor_variables = [
            # 投資・資産関連
            'tangible_assets_ratio', 'capex_ratio', 'rd_ratio', 
            'intangible_assets_ratio', 'investment_securities_ratio',
            
            # 人的資源関連  
            'employee_count', 'average_salary', 'retirement_benefit_cost',
            'welfare_cost',
            
            # 運転資本・効率性関連
            'accounts_receivable_ratio', 'inventory_ratio', 'total_assets_ratio',
            'receivable_turnover', 'inventory_turnover',
            
            # 事業展開関連
            'overseas_sales_ratio', 'segment_count', 'sga_ratio',
            'advertising_ratio', 'non_operating_income',
            
            # 新規追加項目
            'company_age', 'market_entry_timing', 'parent_dependency'
        ]
        
        for market_cat in self.market_categories:
            market_companies = self._filter_companies_by_category(company_data, market_cat)
            
            # 成功/失敗ラベル作成
            success_labels = []
            factor_matrix = []
            company_names = []
            
            for _, company in market_companies.iterrows():
                # 成功度スコア計算
                success_score = self._calculate_success_score(
                    company['企業名'], financial_data, market_data
                )
                
                # 要因項目データ抽出
                factors = self._extract_factor_variables(
                    company['企業名'], financial_data, factor_variables
                )
                
                if success_score is not None and factors is not None:
                    success_labels.append(success_score > 0.6)  # 成功閾値
                    factor_matrix.append(factors)
                    company_names.append(company['企業名'])
            
            if len(success_labels) > 5:
                # 要因重要度分析（ランダムフォレスト）
                importance_analysis = self._analyze_factor_importance(
                    factor_matrix, success_labels, factor_variables
                )
                
                # 成功/失敗パターン分析
                pattern_analysis = self._analyze_success_patterns(
                    factor_matrix, success_labels, factor_variables
                )
                
                success_factors[market_cat] = {
                    'critical_factors': importance_analysis,
                    'success_patterns': pattern_analysis,
                    'success_rate': np.mean(success_labels),
                    'factor_correlations': self._calculate_factor_correlations(
                        factor_matrix, factor_variables
                    )
                }
        
        return success_factors
    
    def _optimize_market_timing(self,
                                financial_data: pd.DataFrame,
                                market_data: pd.DataFrame) -> Dict:
        """
        市場参入タイミング最適化分析
        
        各市場での最適参入タイミングを特定
        """
        timing_optimization = {}
        
        for market_cat in self.market_categories:
            # 市場成長フェーズの特定
            growth_phases = self._identify_market_growth_phases(market_cat, market_data)
            
            # 各フェーズでの参入成功率
            phase_success_rates = {}
            phase_characteristics = {}
            
            for phase, years in growth_phases.items():
                # 該当期間の参入企業抽出
                phase_entrants = self._extract_phase_entrants(
                    market_cat, years, financial_data
                )
                
                if len(phase_entrants) > 2:
                    # 成功率計算
                    success_rate = self._calculate_phase_success_rate(
                        phase_entrants, financial_data
                    )
                    
                    # フェーズ特徴分析
                    characteristics = self._analyze_phase_characteristics(
                        years, market_data, market_cat
                    )
                    
                    phase_success_rates[phase] = success_rate
                    phase_characteristics[phase] = characteristics
            
            # 最適タイミング決定
            optimal_phase = max(phase_success_rates.items(), 
                                key=lambda x: x[1])[0]
            
            timing_optimization[market_cat] = {
                'optimal_entry_phase': optimal_phase,
                'phase_success_rates': phase_success_rates,
                'phase_characteristics': phase_characteristics,
                'timing_recommendations': self._generate_timing_recommendations(
                    optimal_phase, phase_characteristics[optimal_phase]
                )
            }
        
        return timing_optimization
    
    def _analyze_competitive_environment(self,
                                        financial_data: pd.DataFrame,
                                        company_data: pd.DataFrame,
                                        market_data: pd.DataFrame) -> Dict:
        """
        競争環境分析
        
        参入時の競争密度と成功確率の関係分析
        """
        competitive_analysis = {}
        
        for market_cat in self.market_categories:
            market_companies = self._filter_companies_by_category(company_data, market_cat)
            
            # 年次競争密度計算
            yearly_competition = {}
            entry_success_by_competition = []
            
            for year in range(1984, 2025):
                # 該当年の市場参加企業数
                active_companies = self._count_active_companies(
                    market_cat, year, financial_data
                )
                
                # 市場規模（可能な場合）
                market_size = self._get_market_size(market_cat, year, market_data)
                
                # 競争密度 = 企業数 / 市場規模
                if market_size and market_size > 0:
                    competition_density = active_companies / market_size
                else:
                    competition_density = active_companies  # 企業数のみ
                
                yearly_competition[year] = {
                    'active_companies': active_companies,
                    'market_size': market_size,
                    'competition_density': competition_density
                }
                
                # その年の新規参入企業の後続成功率
                new_entrants = self._identify_new_entrants(market_cat, year, company_data)
                if new_entrants:
                    success_rate = self._calculate_entrant_success_rate(
                        new_entrants, financial_data, year
                    )
                    entry_success_by_competition.append({
                        'year': year,
                        'competition_density': competition_density,
                        'success_rate': success_rate,
                        'n_entrants': len(new_entrants)
                    })
            
            # 競争密度と成功率の相関分析
            if len(entry_success_by_competition) > 5:
                competition_df = pd.DataFrame(entry_success_by_competition)
                correlation = stats.pearsonr(
                    competition_df['competition_density'],
                    competition_df['success_rate']
                )[0]
                
                competitive_analysis[market_cat] = {
                    'competition_success_correlation': correlation,
                    'optimal_competition_level': self._find_optimal_competition_level(
                        competition_df
                    ),
                    'yearly_competition_data': yearly_competition,
                    'market_saturation_analysis': self._analyze_market_saturation(
                        yearly_competition
                    ),
                    'entry_barriers_evolution': self._analyze_entry_barriers_evolution(
                        yearly_competition, entry_success_by_competition
                    )
                }
        
        return competitive_analysis
    
    # ヘルパーメソッド群
    
    def _filter_companies_by_category(self, company_data: pd.DataFrame, 
                                    category: str) -> pd.DataFrame:
        """市場カテゴリーで企業を絞り込み"""
        if category in self.market_categories:
            markets = self.market_categories[category]
            return company_data[company_data['市場分野'].isin(markets)]
        return pd.DataFrame()
    
    def _determine_entry_year(self, company: pd.Series, 
                            financial_data: pd.DataFrame) -> Optional[int]:
        """企業の市場参入年を特定"""
        company_name = company['企業名']
        company_financial = financial_data[financial_data['企業名'] == company_name]
        
        if company_financial.empty:
            return None
        
        # 最初の有効な売上データがある年を参入年とする
        valid_revenues = company_financial[company_financial['売上高'] > 0]
        if not valid_revenues.empty:
            return valid_revenues['年度'].min()
        
        return None
    
    def _calculate_market_maturity(self, entry_year: int, market_cat: str) -> float:
        """市場成熟度計算（参入年時点）"""
        # 市場開始年の推定（業界により異なる）
        market_start_years = {
            'high_share': 1970,  # 高シェア市場は比較的新しい技術
            'declining': 1950,   # 低下市場は伝統的産業
            'lost': 1960        # 失失市場は技術変化の激しい分野
        }
        
        start_year = market_start_years.get(market_cat, 1960)
        return max(0, entry_year - start_year)
    
    def _calculate_post_entry_performance(self, company_name: str,
                                        financial_data: pd.DataFrame,
                                        entry_year: int) -> float:
        """参入後の市場パフォーマンス計算"""
        company_data = financial_data[financial_data['企業名'] == company_name]
        post_entry = company_data[company_data['年度'] >= entry_year]
        
        if len(post_entry) < 3:  # 最低3年のデータが必要
            return 0.0
        
        # 売上成長率の計算
        initial_revenue = post_entry['売上高'].iloc[0]
        recent_revenue = post_entry['売上高'].iloc[-1]
        years_span = len(post_entry)
        
        if initial_revenue > 0 and years_span > 0:
            growth_rate = (recent_revenue / initial_revenue) ** (1/years_span) - 1
            return max(0, growth_rate)  # 負の成長は0とする
        
        return 0.0
    
    def _find_optimal_timing_range(self, timing_scores: List[float], 
                                    performance: List[float]) -> Dict:
        """最適参入タイミング範囲の特定"""
        # パフォーマンス上位25%のタイミングスコア分析
        performance_array = np.array(performance)
        timing_array = np.array(timing_scores)
        
        top_quartile_threshold = np.percentile(performance_array, 75)
        top_performers_timing = timing_array[performance_array >= top_quartile_threshold]
        
        if len(top_performers_timing) > 0:
            return {
                'optimal_min': np.min(top_performers_timing),
                'optimal_max': np.max(top_performers_timing),
                'optimal_mean': np.mean(top_performers_timing),
                'optimal_std': np.std(top_performers_timing)
            }
        
        return {'optimal_min': 0, 'optimal_max': 1, 'optimal_mean': 0.5, 'optimal_std': 0.3}
    
    def _extract_strategy_features(self, company_name: str,
                                    financial_data: pd.DataFrame) -> Optional[List[float]]:
        """企業の戦略特徴量抽出"""
        company_data = financial_data[financial_data['企業名'] == company_name]
        
        if len(company_data) < 3:  # 最低3年のデータが必要
            return None
        
        # 設立初期3年間のデータ使用
        early_data = company_data.head(3)
        
        features = []
        
        # 初期設備投資比率
        capex_ratio = early_data['設備投資額'].sum() / early_data['売上高'].sum()
        features.append(capex_ratio if not np.isnan(capex_ratio) else 0)
        
        # R&D集約度
        rd_intensity = early_data['研究開発費'].sum() / early_data['売上高'].sum()
        features.append(rd_intensity if not np.isnan(rd_intensity) else 0)
        
        # マーケティング集約度（広告宣伝費）
        marketing_intensity = early_data['広告宣伝費'].sum() / early_data['売上高'].sum()
        features.append(marketing_intensity if not np.isnan(marketing_intensity) else 0)
        
        # 従業員増加率
        if len(early_data) >= 2:
            employee_growth = (early_data['従業員数'].iloc[-1] / early_data['従業員数'].iloc[0]) - 1
            features.append(employee_growth if not np.isnan(employee_growth) else 0)
        else:
            features.append(0)
        
        # 地域展開速度（海外売上高比率の変化）
        if len(early_data) >= 2:
            geographic_expansion = early_data['海外売上高比率'].iloc[-1] - early_data['海外売上高比率'].iloc[0]
            features.append(geographic_expansion if not np.isnan(geographic_expansion) else 0)
        else:
            features.append(0)
        
        # 製品多様化度（事業セグメント数の変化）
        if len(early_data) >= 2:
            diversification = early_data['事業セグメント数'].iloc[-1] - early_data['事業セグメント数'].iloc[0]
            features.append(diversification if not np.isnan(diversification) else 0)
        else:
            features.append(0)
        
        return features
    
    def _find_optimal_clusters(self, data: np.ndarray, max_k: int = 5) -> int:
        """エルボー法による最適クラスタ数決定"""
        inertias = []
        k_range = range(2, min(max_k + 1, len(data)))
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(data)
            inertias.append(kmeans.inertia_)
        
        # エルボー法による最適k決定
        if len(inertias) >= 2:
            # 二階微分が最大となる点を探す
            differences = np.diff(inertias)
            second_differences = np.diff(differences)
            if len(second_differences) > 0:
                elbow_idx = np.argmax(second_differences) + 2
                return min(elbow_idx, max_k)
        
        return 3  # デフォルト
    
    def _analyze_strategy_clusters(self, strategy_data: List[List[float]],
                                    labels: np.ndarray, company_names: List[str],
                                    n_clusters: int) -> Dict:
        """戦略クラスタ分析"""
        clusters_analysis = {}
        strategy_array = np.array(strategy_data)
        
        feature_names = [
            'capex_intensity', 'rd_intensity', 'marketing_intensity',
            'employee_growth', 'geographic_expansion', 'diversification'
        ]
        
        for cluster_id in range(n_clusters):
            cluster_mask = labels == cluster_id
            cluster_data = strategy_array[cluster_mask]
            cluster_companies = [company_names[i] for i in range(len(company_names)) if cluster_mask[i]]
            
            # クラスタ特徴量の統計
            cluster_stats = {}
            for i, feature in enumerate(feature_names):
                cluster_stats[feature] = {
                    'mean': np.mean(cluster_data[:, i]),
                    'std': np.std(cluster_data[:, i]),
                    'median': np.median(cluster_data[:, i])
                }
            
            clusters_analysis[f'cluster_{cluster_id}'] = {
                'size': len(cluster_companies),
                'companies': cluster_companies,
                'feature_statistics': cluster_stats,
                'cluster_center': np.mean(cluster_data, axis=0).tolist()
            }
        
        return clusters_analysis
    
    def _map_clusters_to_strategies(self, clusters_analysis: Dict) -> Dict:
        """クラスタを戦略タイプにマッピング"""
        strategy_mapping = {}
        
        for cluster_name, analysis in clusters_analysis.items():
            center = analysis['cluster_center']
            
            # 特徴量の重みづけによる戦略分類
            capex_score = center[0]  # 設備投資集約度
            rd_score = center[1]     # R&D集約度
            marketing_score = center[2]  # マーケティング集約度
            
            # 戦略タイプの決定
            if capex_score > 0.15 and rd_score > 0.05:
                strategy_type = 'aggressive_tech'  # 技術積極型
            elif marketing_score > 0.03 and center[4] > 0.1:  # geographic_expansion
                strategy_type = 'aggressive_market'  # 市場積極型
            elif rd_score > 0.08 or center[5] > 0.5:  # diversification
                strategy_type = 'niche_specialist'  # ニッチ特化型
            else:
                strategy_type = 'conservative'  # 保守的
            
            strategy_mapping[cluster_name] = strategy_type
        
        return strategy_mapping
    
    def _analyze_strategy_success(self, strategy_labels: np.ndarray,
                                company_names: List[str],
                                financial_data: pd.DataFrame) -> Dict:
        """戦略タイプ別成功分析"""
        success_by_strategy = {}
        unique_strategies = np.unique(strategy_labels)
        
        for strategy_id in unique_strategies:
            strategy_companies = [company_names[i] for i in range(len(company_names)) 
                                if strategy_labels[i] == strategy_id]
            
            success_scores = []
            for company in strategy_companies:
                score = self._calculate_simple_success_score(company, financial_data)
                if score is not None:
                    success_scores.append(score)
            
            if success_scores:
                success_by_strategy[f'strategy_{strategy_id}'] = {
                    'mean_success_score': np.mean(success_scores),
                    'success_rate': np.mean(np.array(success_scores) > 0.6),
                    'company_count': len(success_scores),
                    'top_performers': [company for i, company in enumerate(strategy_companies)
                                        if i < len(success_scores) and success_scores[i] > 0.8]
                }
        
        return success_by_strategy
    
    def _calculate_simple_success_score(self, company_name: str,
                                        financial_data: pd.DataFrame) -> Optional[float]:
        """簡単な成功スコア計算"""
        company_data = financial_data[financial_data['企業名'] == company_name]
        
        if len(company_data) < 5:
            return None
        
        # 複数指標での成功度評価
        success_components = []
        
        # 1. 売上成長（設立後10年間の年平均成長率）
        early_period = company_data.head(min(10, len(company_data)))
        if len(early_period) >= 3:
            revenue_growth = self._calculate_cagr(
                early_period['売上高'].iloc[0],
                early_period['売上高'].iloc[-1],
                len(early_period)
            )
            success_components.append(min(1.0, max(0.0, revenue_growth)))
        
        # 2. 収益性達成（営業利益率が正になるまでの期間）
        profitability_score = self._calculate_profitability_score(company_data)
        success_components.append(profitability_score)
        
        # 3. 市場地位（相対的な売上規模）
        market_position_score = self._calculate_market_position_score(
            company_name, company_data, financial_data
        )
        success_components.append(market_position_score)
        
        # 4. 持続性（事業継続年数）
        sustainability_score = min(1.0, len(company_data) / 20)  # 20年で正規化
        success_components.append(sustainability_score)
        
        # 重み付き平均で総合スコア計算
        weights = [0.3, 0.3, 0.25, 0.15]  # 成長・収益・地位・持続性
        total_score = np.average(success_components, weights=weights)
        
        return total_score
    
    def _calculate_cagr(self, start_value: float, end_value: float, periods: int) -> float:
        """年複合成長率（CAGR）計算"""
        if start_value <= 0 or end_value <= 0 or periods <= 0:
            return 0.0
        return (end_value / start_value) ** (1/periods) - 1
    
    def _calculate_profitability_score(self, company_data: pd.DataFrame) -> float:
        """収益性達成スコア計算"""
        # 営業利益率が正になった年を探す
        profitable_years = company_data[company_data['売上高営業利益率'] > 0]
        
        if profitable_years.empty:
            return 0.0  # 一度も黒字化できていない
        
        # 黒字化までの期間でスコア化
        years_to_profit = profitable_years['年度'].min() - company_data['年度'].min()
        
        # 早期黒字化ほど高スコア（5年以内で最高評価）
        return max(0.0, 1.0 - years_to_profit / 10)
    
    def _calculate_market_position_score(self, company_name: str,
                                        company_data: pd.DataFrame,
                                        all_financial_data: pd.DataFrame) -> float:
        """市場地位スコア計算"""
        if company_data.empty:
            return 0.0
        
        # 最新年度での同業他社比較
        latest_year = company_data['年度'].max()
        company_revenue = company_data[company_data['年度'] == latest_year]['売上高'].iloc[0]
        
        # 同業他社の売上データ
        same_year_data = all_financial_data[all_financial_data['年度'] == latest_year]
        industry_revenues = same_year_data['売上高'].values
        
        if len(industry_revenues) > 1:
            # パーセンタイル順位でスコア化
            percentile = stats.percentileofscore(industry_revenues, company_revenue) / 100
            return percentile
        
        return 0.5  # デフォルト中位
    
    def _identify_market_growth_phases(self, market_cat: str, 
                                        market_data: pd.DataFrame) -> Dict:
        """市場成長フェーズの特定"""
        # 市場成長率データから成長フェーズを分類
        phases = {
            'emergence': [],      # 新興期（高成長）
            'growth': [],         # 成長期（中成長）
            'maturity': [],       # 成熟期（低成長）
            'decline': []         # 衰退期（負成長）
        }
        
        # 年度別市場成長率の分析
        for year in range(1984, 2025):
            growth_rate = self._estimate_market_growth_rate(market_cat, year, market_data)
            
            if growth_rate is not None:
                if growth_rate > 0.15:
                    phases['emergence'].append(year)
                elif growth_rate > 0.05:
                    phases['growth'].append(year)
                elif growth_rate > -0.02:
                    phases['maturity'].append(year)
                else:
                    phases['decline'].append(year)
        
        return phases
    
    def _estimate_market_growth_rate(self, market_cat: str, year: int,
                                    market_data: pd.DataFrame) -> Optional[float]:
        """市場成長率の推定"""
        # 実際の実装では、市場データから成長率を取得
        # ここでは市場カテゴリーに基づく推定値を使用
        
        growth_patterns = {
            'high_share': {
                'base_growth': 0.08,
                'peak_years': range(2000, 2015),  # デジタル化の波
                'decline_start': 2020
            },
            'declining': {
                'base_growth': 0.03,
                'peak_years': range(1990, 2010),
                'decline_start': 2010
            },
            'lost': {
                'base_growth': 0.02,
                'peak_years': range(1985, 2005),
                'decline_start': 2005
            }
        }
        
        pattern = growth_patterns.get(market_cat)
        if not pattern:
            return None
        
        base_growth = pattern['base_growth']
        
        if year in pattern['peak_years']:
            return base_growth * 2  # ピーク期は2倍成長
        elif year >= pattern['decline_start']:
            return base_growth * 0.3  # 衰退期は成長鈍化
        else:
            return base_growth
    
    def _extract_phase_entrants(self, market_cat: str, years: List[int],
                                financial_data: pd.DataFrame) -> List[str]:
        """特定フェーズでの新規参入企業抽出"""
        entrants = []
        
        # 該当期間に初回データが現れる企業を新規参入企業とする
        for company in financial_data['企業名'].unique():
            company_data = financial_data[financial_data['企業名'] == company]
            first_year = company_data['年度'].min()
            
            if first_year in years:
                entrants.append(company)
        
        return entrants
    
    def _calculate_phase_success_rate(self, entrants: List[str],
                                    financial_data: pd.DataFrame) -> float:
        """フェーズ別参入成功率計算"""
        if not entrants:
            return 0.0
        
        success_count = 0
        for company in entrants:
            success_score = self._calculate_simple_success_score(company, financial_data)
            if success_score and success_score > 0.6:
                success_count += 1
        
        return success_count / len(entrants)
    
    def _analyze_phase_characteristics(self, years: List[int],
                                        market_data: pd.DataFrame,
                                        market_cat: str) -> Dict:
        """市場フェーズ特徴分析"""
        if not years:
            return {}
        
        # フェーズ期間の市場特徴量計算
        avg_growth = np.mean([
            self._estimate_market_growth_rate(market_cat, year, market_data) or 0
            for year in years
        ])
        
        return {
            'duration_years': len(years),
            'average_growth_rate': avg_growth,
            'period_start': min(years),
            'period_end': max(years),
            'market_volatility': self._calculate_market_volatility(years, market_cat),
            'technology_disruption_level': self._estimate_tech_disruption(years, market_cat)
        }
    
    def _calculate_market_volatility(self, years: List[int], market_cat: str) -> float:
        """市場ボラティリティ計算"""
        # 年次成長率の標準偏差で近似
        growth_rates = []
        for i in range(1, len(years)):
            # 前年比成長率の推定（簡略化）
            if market_cat == 'high_share':
                volatility = 0.15 if years[i] > 2008 else 0.08  # リーマンショック以降高ボラティリティ
            elif market_cat == 'declining':
                volatility = 0.12
            else:  # lost
                volatility = 0.25  # 失失市場は高ボラティリティ
            
            growth_rates.append(volatility)
        
        return np.mean(growth_rates) if growth_rates else 0.1
    
    def _estimate_tech_disruption(self, years: List[int], market_cat: str) -> float:
        """技術破壊レベル推定"""
        disruption_periods = {
            'high_share': {2000: 0.3, 2010: 0.5, 2020: 0.8},  # デジタル化の波
            'declining': {1995: 0.2, 2010: 0.4, 2020: 0.3},   # 緩やかな技術変化
            'lost': {1990: 0.4, 2000: 0.7, 2010: 0.9}         # 急激な技術変化
        }
        
        pattern = disruption_periods.get(market_cat, {})
        avg_disruption = 0.3  # デフォルト
        
        for year in years:
            for disruption_year, level in pattern.items():
                if abs(year - disruption_year) <= 5:  # 5年範囲で影響
                    avg_disruption = max(avg_disruption, level)
        
        return avg_disruption
    
    def _generate_timing_recommendations(self, optimal_phase: str,
                                        phase_characteristics: Dict) -> Dict:
        """タイミング推奨事項生成"""
        recommendations = {
            'optimal_phase': optimal_phase,
            'key_indicators': [],
            'entry_conditions': [],
            'risk_factors': []
        }
        
        if optimal_phase == 'emergence':
            recommendations['key_indicators'] = [
                '市場成長率 > 15%',
                '技術標準化前の時期',
                '競合企業数 < 5社'
            ]
            recommendations['entry_conditions'] = [
                '高いR&D投資能力',
                '技術的優位性の確保',
                'リスク許容度の高い資本'
            ]
            recommendations['risk_factors'] = [
                '市場の不確実性',
                '技術リスク',
                '早期投資回収困難'
            ]
        
        elif optimal_phase == 'growth':
            recommendations['key_indicators'] = [
                '市場成長率 5-15%',
                '顧客ニーズの明確化',
                '技術標準の確立'
            ]
            recommendations['entry_conditions'] = [
                '差別化戦略の明確化',
                '効率的な営業・マーケティング',
                '適切な設備投資計画'
            ]
            recommendations['risk_factors'] = [
                '競争激化',
                '価格競争リスク',
                '市場シェア獲得困難'
            ]
        
        elif optimal_phase == 'maturity':
            recommendations['key_indicators'] = [
                '市場成長率 0-5%',
                '技術・製品の標準化',
                '顧客ロイヤルティの重要性'
            ]
            recommendations['entry_conditions'] = [
                'コスト優位性の確保',
                'ニッチ市場での差別化',
                '効率的オペレーション'
            ]
            recommendations['risk_factors'] = [
                '新規参入障壁の高さ',
                '既存プレイヤーの反応',
                '低収益性リスク'
            ]
        
        return recommendations
    
    def _count_active_companies(self, market_cat: str, year: int,
                                financial_data: pd.DataFrame) -> int:
        """指定年の市場参加企業数カウント"""
        year_data = financial_data[financial_data['年度'] == year]
        
        # 市場カテゴリーに属する企業で、有効な売上データがある企業数
        active_count = 0
        for company in year_data['企業名'].unique():
            company_revenue = year_data[year_data['企業名'] == company]['売上高'].iloc[0] if not year_data[year_data['企業名'] == company].empty else 0
            if company_revenue > 0:
                active_count += 1
        
        return active_count
    
    def _get_market_size(self, market_cat: str, year: int,
                        market_data: pd.DataFrame) -> Optional[float]:
        """市場規模データ取得"""
        # 実際の実装では market_data から取得
        # ここでは推定値を使用
        
        market_sizes = {
            'high_share': {
                1990: 100, 2000: 300, 2010: 800, 2020: 1500
            },
            'declining': {
                1990: 1000, 2000: 1200, 2010: 1100, 2020: 900
            },
            'lost': {
                1990: 500, 2000: 2000, 2010: 1500, 2020: 800
            }
        }
        
        sizes = market_sizes.get(market_cat, {})
        
        # 線形補間で年度推定
        if not sizes:
            return None
        
        years_list = sorted(sizes.keys())
        if year <= years_list[0]:
            return sizes[years_list[0]]
        elif year >= years_list[-1]:
            return sizes[years_list[-1]]
        else:
            # 線形補間
            for i in range(len(years_list) - 1):
                if years_list[i] <= year <= years_list[i + 1]:
                    weight = (year - years_list[i]) / (years_list[i + 1] - years_list[i])
                    return sizes[years_list[i]] * (1 - weight) + sizes[years_list[i + 1]] * weight
        
        return None
    
    def _identify_new_entrants(self, market_cat: str, year: int,
                                company_data: pd.DataFrame) -> List[str]:
        """新規参入企業の特定"""
        # 指定年に初回財務データが現れる企業
        market_companies = self._filter_companies_by_category(company_data, market_cat)
        
        new_entrants = []
        for _, company in market_companies.iterrows():
            # 設立年や初回データ年が該当年の企業
            if company.get('設立年') == year or company.get('初回データ年') == year:
                new_entrants.append(company['企業名'])
        
        return new_entrants
    
    def _calculate_entrant_success_rate(self, entrants: List[str],
                                        financial_data: pd.DataFrame,
                                        entry_year: int) -> float:
        """新規参入企業の成功率計算"""
        if not entrants:
            return 0.0
        
        success_count = 0
        for company in entrants:
            # 参入から5年後の状況評価
            future_data = financial_data[
                (financial_data['企業名'] == company) & 
                (financial_data['年度'] >= entry_year) &
                (financial_data['年度'] <= entry_year + 5)
            ]
            
            if len(future_data) >= 3:  # 最低3年の追跡データが必要
                # 成功判定（売上成長 + 黒字化）
                revenue_growth = self._calculate_cagr(
                    future_data['売上高'].iloc[0],
                    future_data['売上高'].iloc[-1],
                    len(future_data)
                )
                achieved_profit = (future_data['売上高営業利益率'] > 0).any()
                
                if revenue_growth > 0.05 and achieved_profit:
                    success_count += 1
        
        return success_count / len(entrants)
    
    def _find_optimal_competition_level(self, competition_df: pd.DataFrame) -> Dict:
        """最適競争レベルの特定"""
        if len(competition_df) < 5:
            return {'optimal_density': 0.5, 'confidence': 0.0}
        
        # 成功率が最高となる競争密度を探す
        optimal_idx = competition_df['success_rate'].idxmax()
        optimal_density = competition_df.loc[optimal_idx, 'competition_density']
        max_success_rate = competition_df.loc[optimal_idx, 'success_rate']
        
        # 信頼度計算（データポイント数と成功率の分散）
        confidence = min(1.0, len(competition_df) / 20) * (1 - competition_df['success_rate'].std())
        
        return {
            'optimal_density': optimal_density,
            'max_success_rate': max_success_rate,
            'confidence': confidence,
            'supporting_observations': len(competition_df)
        }
    
    def _analyze_market_saturation(self, yearly_competition: Dict) -> Dict:
        """市場飽和分析"""
        years = sorted(yearly_competition.keys())
        company_counts = [yearly_competition[year]['active_companies'] for year in years]
        
        # 市場飽和度の推移分析
        saturation_trend = np.polyfit(range(len(company_counts)), company_counts, 2)
        
        # 飽和点の推定（二次関数の頂点）
        if saturation_trend[0] < 0:  # 下に凸の場合
            saturation_peak_idx = -saturation_trend[1] / (2 * saturation_trend[0])
            saturation_year = years[int(min(max(0, saturation_peak_idx), len(years) - 1))]
        else:
            saturation_year = years[-1]  # 単調増加の場合は最新年
        
        current_saturation = company_counts[-1] / max(company_counts) if company_counts else 0
        
        return {
            'saturation_year': saturation_year,
            'current_saturation_level': current_saturation,
            'trend_coefficient': saturation_trend.tolist(),
            'market_maturity_stage': self._determine_maturity_stage(current_saturation)
        }
    
    def _determine_maturity_stage(self, saturation_level: float) -> str:
        """市場成熟段階の判定"""
        if saturation_level < 0.3:
            return 'emerging'
        elif saturation_level < 0.7:
            return 'growing'
        elif saturation_level < 0.9:
            return 'mature'
        else:
            return 'saturated'
    
    def _analyze_entry_barriers_evolution(self, yearly_competition: Dict,
                                        entry_success_data: List[Dict]) -> Dict:
        """参入障壁の進化分析"""
        barriers_evolution = {}
        
        # 時系列での参入障壁指標計算
        years = sorted(yearly_competition.keys())
        
        for i, year in enumerate(years[:-1]):
            current_competition = yearly_competition[year]['competition_density']
            next_year = years[i + 1]
            next_competition = yearly_competition[next_year]['competition_density']
            
            # 参入障壁指標
            competition_change = next_competition - current_competition
            
            # 該当年の新規参入成功率
            year_success_data = [d for d in entry_success_data if d['year'] == year]
            success_rate = year_success_data[0]['success_rate'] if year_success_data else 0.5
            
            # 参入障壁強度（競争密度増加 × 成功率逆数）
            barrier_strength = competition_change * (1 - success_rate)
            
            barriers_evolution[year] = {
                'competition_density_change': competition_change,
                'new_entrant_success_rate': success_rate,
                'barrier_strength': barrier_strength,
                'market_accessibility': 1 - barrier_strength  # 正規化された参入容易度
            }
        
        # 全体的な参入障壁トレンド
        barrier_strengths = [data['barrier_strength'] for data in barriers_evolution.values()]
        overall_trend = 'increasing' if np.mean(barrier_strengths[-5:]) > np.mean(barrier_strengths[:5]) else 'decreasing'
        
        return {
            'yearly_barriers': barriers_evolution,
            'overall_trend': overall_trend,
            'current_barrier_level': barrier_strengths[-1] if barrier_strengths else 0.5,
            'peak_barrier_year': years[np.argmax(barrier_strengths)] if barrier_strengths else years[0]
        }
    
    def _analyze_factor_importance(self, factor_matrix: List[List[float]],
                                    success_labels: List[bool],
                                    factor_names: List[str]) -> Dict:
        """要因重要度分析（Random Forest風の簡易実装）"""
        from sklearn.ensemble import RandomForestClassifier
        
        if len(factor_matrix) < 5:
            return {}
        
        # Random Forest による特徴量重要度計算
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(factor_matrix, success_labels)
        
        importance_scores = rf.feature_importances_
        
        # 重要度順にソート
        importance_ranking = sorted(
            zip(factor_names, importance_scores),
            key=lambda x: x[1],
            reverse=True
        )
        
        return {
            'top_factors': importance_ranking[:10],
            'critical_threshold': 0.05,  # 重要度閾値
            'model_accuracy': rf.score(factor_matrix, success_labels),
            'feature_importance_distribution': {
                name: float(score) for name, score in importance_ranking
            }
        }
    
    def _analyze_success_patterns(self, factor_matrix: List[List[float]],
                                success_labels: List[bool],
                                factor_names: List[str]) -> Dict:
        """成功パターン分析"""
        success_data = np.array(factor_matrix)[np.array(success_labels)]
        failure_data = np.array(factor_matrix)[~np.array(success_labels)]
        
        patterns = {}
        
        for i, factor_name in enumerate(factor_names):
            if len(success_data) > 0 and len(failure_data) > 0:
                # 成功企業 vs 失敗企業の統計比較
                success_mean = np.mean(success_data[:, i])
                failure_mean = np.mean(failure_data[:, i])
                
                # t検定による有意差検定
                t_stat, p_value = stats.ttest_ind(
                    success_data[:, i], failure_data[:, i]
                )
                
                patterns[factor_name] = {
                    'success_mean': float(success_mean),
                    'failure_mean': float(failure_mean),
                    'difference_ratio': float(success_mean / failure_mean) if failure_mean != 0 else np.inf,
                    't_statistic': float(t_stat),
                    'p_value': float(p_value),
                    'significant_difference': p_value < 0.05,
                    'success_advantage': success_mean > failure_mean
                }
        
        return patterns
    
    def _calculate_factor_correlations(self, factor_matrix: List[List[float]],
                                        factor_names: List[str]) -> Dict:
        """要因項目間の相関分析"""
        if len(factor_matrix) < 3:
            return {}
        
        correlation_matrix = np.corrcoef(np.array(factor_matrix).T)
        
        # 高相関ペアの特定
        high_correlations = []
        n_factors = len(factor_names)
        
        for i in range(n_factors):
            for j in range(i + 1, n_factors):
                corr_value = correlation_matrix[i, j]
                if abs(corr_value) > 0.7:  # 高相関閾値
                    high_correlations.append({
                        'factor1': factor_names[i],
                        'factor2': factor_names[j],
                        'correlation': float(corr_value),
                        'relationship_type': 'positive' if corr_value > 0 else 'negative'
                    })
        
        return {
            'correlation_matrix': correlation_matrix.tolist(),
            'factor_names': factor_names,
            'high_correlations': high_correlations,
            'max_correlation': float(np.max(np.abs(correlation_matrix - np.eye(n_factors)))),
            'multicollinearity_risk': len(high_correlations) > n_factors * 0.1
        }
    
    def _extract_factor_variables(self, company_name: str,
                                financial_data: pd.DataFrame,
                                factor_variables: List[str]) -> Optional[List[float]]:
        """企業の要因項目データ抽出"""
        company_data = financial_data[financial_data['企業名'] == company_name]
        
        if company_data.empty:
            return None
        
        # 設立初期3年間の平均値を使用
        early_data = company_data.head(3)
        
        factors = []
        
        # 各要因項目の計算（簡略化実装）
        for factor in factor_variables:
            if factor in early_data.columns:
                value = early_data[factor].mean()
                factors.append(value if not np.isnan(value) else 0.0)
            else:
                # 要因項目の代替計算
                value = self._calculate_alternative_factor(factor, early_data)
                factors.append(value)
        
        return factors
    
    def _calculate_alternative_factor(self, factor_name: str, data: pd.DataFrame) -> float:
        """要因項目の代替計算"""
        # 主要な要因項目の計算ロジック
        if factor_name == 'tangible_assets_ratio':
            return data['有形固定資産'].sum() / data['総資産'].sum() if data['総資産'].sum() > 0 else 0
        elif factor_name == 'rd_ratio':
            return data['研究開発費'].sum() / data['売上高'].sum() if data['売上高'].sum() > 0 else 0
        elif factor_name == 'overseas_sales_ratio':
            return data['海外売上高比率'].mean() if '海外売上高比率' in data.columns else 0
        elif factor_name == 'company_age':
            return len(data)  # データ期間を企業年齢の代理とする
        else:
            return 0.0  # デフォルト値
    
    def _calculate_success_score(self, company_name: str,
                                financial_data: pd.DataFrame,
                                market_data: pd.DataFrame) -> Optional[float]:
        """包括的成功スコア計算"""
        company_data = financial_data[financial_data['企業名'] == company_name]
        
        if company_data.empty:
            return None
        
        # 複数次元での成功評価
        dimensions = {}
        
        # 1. 成長性スコア
        dimensions['growth'] = self._calculate_growth_score(company_data)
        
        # 2. 収益性スコア
        dimensions['profitability'] = self._calculate_profitability_score(company_data)
        
        # 3. 市場地位スコア
        dimensions['market_position'] = self._calculate_market_position_score(
            company_name, company_data, financial_data
        )
        
        # 4. 持続性スコア
        dimensions['sustainability'] = self._calculate_sustainability_score(company_data)
        
        # 5. イノベーションスコア
        dimensions['innovation'] = self._calculate_innovation_score(company_data)
        
        # 重み付き統合スコア
        weights = {
            'growth': 0.25,
            'profitability': 0.25,
            'market_position': 0.20,
            'sustainability': 0.15,
            'innovation': 0.15
        }
        
        total_score = sum(dimensions[dim] * weights[dim] for dim in dimensions)
        return total_score
    
    def _calculate_growth_score(self, company_data: pd.DataFrame) -> float:
        """成長性スコア計算"""
        if len(company_data) < 3:
            return 0.0
        
        # 売上高成長率
        revenue_growth = self._calculate_cagr(
            company_data['売上高'].iloc[0],
            company_data['売上高'].iloc[-1],
            len(company_data)
        )
        
        # 従業員数成長率
        if '従業員数' in company_data.columns:
            employee_growth = self._calculate_cagr(
                company_data['従業員数'].iloc[0],
                company_data['従業員数'].iloc[-1],
                len(company_data)
            )
        else:
            employee_growth = 0.0
        
        # 総資産成長率
        asset_growth = self._calculate_cagr(
            company_data['総資産'].iloc[0],
            company_data['総資産'].iloc[-1],
            len(company_data)
        )
        
        # 重み付き平均
        growth_score = (revenue_growth * 0.5 + employee_growth * 0.25 + asset_growth * 0.25)
        return max(0.0, min(1.0, growth_score * 2))  # 0-1に正規化
    
    def _calculate_sustainability_score(self, company_data: pd.DataFrame) -> float:
        """持続性スコア計算"""
        # 事業継続年数
        duration_score = min(1.0, len(company_data) / 25)  # 25年で最高評価
        
        # 財務安定性（自己資本比率の推移）
        if '自己資本比率' in company_data.columns:
            equity_ratio_trend = company_data['自己資本比率'].corr(
                pd.Series(range(len(company_data)))
            )
            stability_score = max(0.0, min(1.0, (equity_ratio_trend + 1) / 2))
        else:
            stability_score = 0.5
        
        # 収益安定性（営業利益率の分散）
        if '売上高営業利益率' in company_data.columns:
            profit_volatility = company_data['売上高営業利益率'].std()
            consistency_score = max(0.0, 1.0 - profit_volatility / 0.5)  # 50%変動で0点
        else:
            consistency_score = 0.5
        
        return (duration_score * 0.4 + stability_score * 0.3 + consistency_score * 0.3)
    
    def _calculate_innovation_score(self, company_data: pd.DataFrame) -> float:
        """イノベーションスコア計算"""
        innovation_indicators = []
        
        # R&D投資集約度
        if '研究開発費' in company_data.columns and '売上高' in company_data.columns:
            rd_intensity = (company_data['研究開発費'] / company_data['売上高']).mean()
            innovation_indicators.append(min(1.0, rd_intensity / 0.15))  # 15%で最高評価
        
        # 無形固定資産比率
        if '無形固定資産' in company_data.columns and '総資産' in company_data.columns:
            intangible_ratio = (company_data['無形固定資産'] / company_data['総資産']).mean()
            innovation_indicators.append(min(1.0, intangible_ratio / 0.2))  # 20%で最高評価
        
        # 新規事業展開（セグメント数増加）
        if '事業セグメント数' in company_data.columns:
            segment_growth = (
                company_data['事業セグメント数'].iloc[-1] - 
                company_data['事業セグメント数'].iloc[0]
            ) / len(company_data)
            innovation_indicators.append(min(1.0, max(0.0, segment_growth / 0.5)))
        
        return np.mean(innovation_indicators) if innovation_indicators else 0.3
    
    def generate_entry_recommendations(self, market_category: str,
                                        analysis_results: Dict) -> Dict:
        """市場参入推奨事項生成"""
        if market_category not in analysis_results.get('entry_timing_analysis', {}):
            return {'error': f'No analysis data available for {market_category}'}
        
        timing_data = analysis_results['entry_timing_analysis'][market_category]
        strategy_data = analysis_results.get('strategy_classification', {}).get(market_category, {})
        success_data = analysis_results.get('success_factors', {}).get(market_category, {})
        
        recommendations = {
            'market_category': market_category,
            'entry_timing': {},
            'strategy_recommendations': {},
            'success_factors': {},
            'risk_assessment': {},
            'implementation_roadmap': {}
        }
        
        # タイミング推奨
        if timing_data.get('first_mover_advantage'):
            recommendations['entry_timing'] = {
                'recommendation': '早期参入推奨',
                'rationale': '先発者優位が確認されている市場',
                'optimal_timing': '市場成長率 > 10%の時期',
                'key_success_factor': '技術革新とスピード'
            }
        elif timing_data.get('late_mover_advantage'):
            recommendations['entry_timing'] = {
                'recommendation': '後発参入戦略',
                'rationale': '市場が成熟してからの参入が有利',
                'optimal_timing': '市場標準確立後',
                'key_success_factor': 'コスト効率性と差別化'
            }
        else:
            recommendations['entry_timing'] = {
                'recommendation': '機会主義的参入',
                'rationale': '明確なタイミング優位性なし',
                'optimal_timing': '市場の構造変化時期',
                'key_success_factor': '適応性と機動性'
            }
        
        # 戦略推奨
        if strategy_data:
            most_successful_strategy = max(
                strategy_data.get('success_by_strategy', {}).items(),
                key=lambda x: x[1].get('mean_success_score', 0),
                default=(None, {})
            )
            
            if most_successful_strategy[0]:
                recommendations['strategy_recommendations'] = {
                    'recommended_strategy': most_successful_strategy[0],
                    'success_rate': most_successful_strategy[1].get('mean_success_score', 0),
                    'benchmark_companies': most_successful_strategy[1].get('top_performers', [])
                }
        
        # 重要成功要因
        if success_data and 'critical_factors' in success_data:
            top_factors = success_data['critical_factors'].get('top_factors', [])[:5]
            recommendations['success_factors'] = {
                'critical_factors': [factor[0] for factor in top_factors],
                'factor_importance': {factor[0]: factor[1] for factor in top_factors},
                'implementation_priority': self._prioritize_factors(top_factors)
            }
        
        return recommendations
    
    def _prioritize_factors(self, top_factors: List[Tuple[str, float]]) -> List[Dict]:
        """要因項目の実装優先度決定"""
        priorities = []
        
        factor_implementation_difficulty = {
            'rd_ratio': {'difficulty': 'high', 'time_to_impact': 'long'},
            'capex_ratio': {'difficulty': 'medium', 'time_to_impact': 'medium'},
            'marketing_intensity': {'difficulty': 'low', 'time_to_impact': 'short'},
            'employee_growth': {'difficulty': 'medium', 'time_to_impact': 'medium'},
            'overseas_expansion': {'difficulty': 'high', 'time_to_impact': 'long'}
        }
        
        for factor_name, importance in top_factors:
            difficulty_info = factor_implementation_difficulty.get(
                factor_name, 
                {'difficulty': 'medium', 'time_to_impact': 'medium'}
            )
            
            # 重要度と実装容易性を考慮した優先度スコア
            ease_score = {'low': 1.0, 'medium': 0.7, 'high': 0.4}[difficulty_info['difficulty']]
            priority_score = importance * 0.7 + ease_score * 0.3
            
            priorities.append({
                'factor': factor_name,
                'importance': float(importance),
                'implementation_difficulty': difficulty_info['difficulty'],
                'time_to_impact': difficulty_info['time_to_impact'],
                'priority_score': float(priority_score)
            })
        
        # 優先度順にソート
        priorities.sort(key=lambda x: x['priority_score'], reverse=True)
        return priorities
    
    def visualize_entry_analysis(self, analysis_results: Dict,
                                save_path: Optional[str] = None) -> None:
        """分析結果の可視化"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('A2AI Market Entry Analysis Results', fontsize=16, fontweight='bold')
        
        # 1. 市場カテゴリー別参入タイミング効果
        self._plot_timing_effects(analysis_results, axes[0, 0])
        
        # 2. 戦略タイプ別成功率
        self._plot_strategy_success_rates(analysis_results, axes[0, 1])
        
        # 3. 重要成功要因ランキング
        self._plot_success_factors(analysis_results, axes[0, 2])
        
        # 4. 競争密度と成功率の関係
        self._plot_competition_analysis(analysis_results, axes[1, 0])
        
        # 5. 市場フェーズ別参入成功率
        self._plot_phase_success_rates(analysis_results, axes[1, 1])
        
        # 6. 要因項目相関ヒートマップ
        self._plot_factor_correlations(analysis_results, axes[1, 2])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def _plot_timing_effects(self, results: Dict, ax) -> None:
        """参入タイミング効果の可視化"""
        timing_data = results.get('entry_timing_analysis', {})
        
        categories = list(timing_data.keys())
        correlations = [timing_data[cat]['correlation_coefficient'] for cat in categories]
        
        bars = ax.bar(categories, correlations, 
                        color=['green' if c > 0 else 'red' for c in correlations])
        ax.set_title('Market Entry Timing Effects')
        ax.set_ylabel('Correlation (Timing vs Performance)')
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.set_ylim(-1, 1)
        
        # 色分け凡例
        ax.text(0.02, 0.95, 'Green: First-mover advantage', transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
        ax.text(0.02, 0.85, 'Red: Late-mover advantage', transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
    
    def _plot_strategy_success_rates(self, results: Dict, ax) -> None:
        """戦略タイプ別成功率の可視化"""
        strategy_data = results.get('strategy_classification', {})
        
        all_success_rates = []
        all_strategy_names = []
        
        for market_cat, data in strategy_data.items():
            success_by_strategy = data.get('success_by_strategy', {})
            for strategy, metrics in success_by_strategy.items():
                all_success_rates.append(metrics['mean_success_score'])
                all_strategy_names.append(f"{market_cat}_{strategy}")
        
        if all_success_rates:
            bars = ax.bar(range(len(all_success_rates)), all_success_rates)
            ax.set_title('Success Rates by Entry Strategy')
            ax.set_ylabel('Mean Success Score')
            ax.set_xticks(range(len(all_strategy_names)))
            ax.set_xticklabels(all_strategy_names, rotation=45, ha='right')
            
            # 色グラデーション
            colors = plt.cm.viridis(np.linspace(0, 1, len(bars)))
            for bar, color in zip(bars, colors):
                bar.set_color(color)
    
    def _plot_success_factors(self, results: Dict, ax) -> None:
        """重要成功要因の可視化"""
        success_factors = results.get('success_factors', {})
        
        # 全市場での要因重要度を統合
        factor_importance = {}
        
        for market_cat, data in success_factors.items():
            critical_factors = data.get('critical_factors', {})
            top_factors = critical_factors.get('top_factors', [])
            
            for factor_name, importance in top_factors:
                if factor_name in factor_importance:
                    factor_importance[factor_name] += importance
                else:
                    factor_importance[factor_name] = importance
        
        # 上位10要因を表示
        sorted_factors = sorted(factor_importance.items(), 
                                key=lambda x: x[1], reverse=True)[:10]
        
        if sorted_factors:
            factors, importances = zip(*sorted_factors)
            bars = ax.barh(range(len(factors)), importances)
            ax.set_title('Top Critical Success Factors')
            ax.set_xlabel('Importance Score')
            ax.set_yticks(range(len(factors)))
            ax.set_yticklabels(factors)
            
            # 重要度に応じた色分け
            colors = plt.cm.Reds(np.linspace(0.3, 1, len(bars)))
            for bar, color in zip(bars, colors):
                bar.set_color(color)
    
    def _plot_competition_analysis(self, results: Dict, ax) -> None:
        """競争分析の可視化"""
        competitive_data = results.get('competitive_analysis', {})
        
        # 競争密度と成功率の散布図
        all_densities = []
        all_success_rates = []
        
        for market_cat, data in competitive_data.items():
            yearly_barriers = data.get('yearly_barriers', {})
            for year, barrier_data in yearly_barriers.items():
                if 'market_accessibility' in barrier_data:
                    # 競争密度の代理として市場アクセシビリティを使用
                    all_densities.append(1 - barrier_data['market_accessibility'])
                    all_success_rates.append(barrier_data.get('new_entrant_success_rate', 0.5))
        
        if all_densities and all_success_rates:
            scatter = ax.scatter(all_densities, all_success_rates, alpha=0.6, c=range(len(all_densities)), cmap='viridis')
            ax.set_title('Competition Density vs Entry Success Rate')
            ax.set_xlabel('Competition Density')
            ax.set_ylabel('Entry Success Rate')
            
            # トレンドライン
            if len(all_densities) > 5:
                z = np.polyfit(all_densities, all_success_rates, 1)
                p = np.poly1d(z)
                ax.plot(sorted(all_densities), p(sorted(all_densities)), "r--", alpha=0.8)
    
    def _plot_phase_success_rates(self, results: Dict, ax) -> None:
        """市場フェーズ別成功率の可視化"""
        timing_optimization = results.get('market_timing_optimization', {})
        
        phase_success_data = []
        phase_names = []
        
        for market_cat, data in timing_optimization.items():
            phase_success_rates = data.get('phase_success_rates', {})
            for phase, success_rate in phase_success_rates.items():
                phase_success_data.append(success_rate)
                phase_names.append(f"{market_cat}_{phase}")
        
        if phase_success_data:
            bars = ax.bar(range(len(phase_success_data)), phase_success_data)
            ax.set_title('Success Rates by Market Phase')
            ax.set_ylabel('Success Rate')
            ax.set_xticks(range(len(phase_names)))
            ax.set_xticklabels(phase_names, rotation=45, ha='right')
            
            # 成功率に応じた色分け
            colors = plt.cm.RdYlGn(np.array(phase_success_data))
            for bar, color in zip(bars, colors):
                bar.set_color(color)
    
    def _plot_factor_correlations(self, results: Dict, ax) -> None:
        """要因項目相関の可視化"""
        success_factors = results.get('success_factors', {})
        
        # 最初の市場カテゴリーの相関データを使用
        first_market = list(success_factors.keys())[0] if success_factors else None
        
        if first_market and 'factor_correlations' in success_factors[first_market]:
            corr_data = success_factors[first_market]['factor_correlations']
            correlation_matrix = np.array(corr_data['correlation_matrix'])
            factor_names = corr_data['factor_names']
            
            # 相関ヒートマップ
            im = ax.imshow(correlation_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
            ax.set_title('Factor Correlations Heatmap')
            
            # 軸ラベル（簡略化）
            tick_positions = range(0, len(factor_names), max(1, len(factor_names)//8))
            ax.set_xticks(tick_positions)
            ax.set_yticks(tick_positions)
            ax.set_xticklabels([factor_names[i][:8] for i in tick_positions], rotation=45)
            ax.set_yticklabels([factor_names[i][:8] for i in tick_positions])
            
            # カラーバー
            plt.colorbar(im, ax=ax, shrink=0.8)
    
    def export_analysis_results(self, analysis_results: Dict,
                                export_path: str) -> None:
        """分析結果のエクスポート"""
        import json
        
        # JSON形式で保存
        with open(f"{export_path}/market_entry_analysis.json", 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, ensure_ascii=False, indent=2, default=str)
        
        # CSV形式での要約レポート
        summary_data = []
        
        for market_cat in self.market_categories:
            if market_cat in analysis_results.get('entry_timing_analysis', {}):
                timing_data = analysis_results['entry_timing_analysis'][market_cat]
                
                summary_data.append({
                    'market_category': market_cat,
                    'first_mover_advantage': timing_data.get('first_mover_advantage', False),
                    'correlation_coefficient': timing_data.get('correlation_coefficient', 0),
                    'optimal_timing_mean': timing_data.get('optimal_timing_range', {}).get('optimal_mean', 0.5),
                    'early_entrants_count': timing_data.get('timing_distribution', {}).get('early_entrants', 0),
                    'late_entrants_count': timing_data.get('timing_distribution', {}).get('late_entrants', 0)
                })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(f"{export_path}/market_entry_summary.csv", index=False, encoding='utf-8')
        
        print(f"Analysis results exported to {export_path}")
    
    def run_comprehensive_analysis(self, 
                                    financial_data: pd.DataFrame,
                                    company_data: pd.DataFrame,
                                    market_data: pd.DataFrame,
                                    target_market: Optional[str] = None) -> Dict:
        """包括的市場参入分析実行"""
        print("Starting A2AI Market Entry Analysis...")
        
        # 全体分析実行
        analysis_results = self.analyze_market_entry_patterns(
            financial_data, company_data, market_data
        )
        
        # 特定市場の詳細分析（指定された場合）
        if target_market:
            detailed_analysis = self._detailed_market_analysis(
                target_market, financial_data, company_data, market_data
            )
            analysis_results['detailed_analysis'] = {target_market: detailed_analysis}
        
        # 推奨事項生成
        recommendations = {}
        for market_cat in self.market_categories:
            recommendations[market_cat] = self.generate_entry_recommendations(
                market_cat, analysis_results
            )
        
        analysis_results['recommendations'] = recommendations
        
        print("A2AI Market Entry Analysis completed.")
        return analysis_results
    
    def _detailed_market_analysis(self, market_category: str,
                                financial_data: pd.DataFrame,
                                company_data: pd.DataFrame,
                                market_data: pd.DataFrame) -> Dict:
        """特定市場の詳細分析"""
        detailed_results = {}
        
        # 市場参入企業の詳細プロファイリング
        market_companies = self._filter_companies_by_category(company_data, market_category)
        
        company_profiles = []
        for _, company in market_companies.iterrows():
            profile = self._create_company_entry_profile(
                company['企業名'], financial_data, market_data
            )
            if profile:
                company_profiles.append(profile)
        
        detailed_results['company_profiles'] = company_profiles
        
        # 時系列での市場構造変化分析
        detailed_results['market_evolution'] = self._analyze_market_evolution(
            market_category, financial_data, market_data
        )
        
        # 成功企業の共通パターン特定
        detailed_results['success_patterns'] = self._identify_success_patterns(
            company_profiles
        )
        
        # 失敗企業の共通要因分析
        detailed_results['failure_patterns'] = self._identify_failure_patterns(
            company_profiles
        )
        
        return detailed_results
    
    def _create_company_entry_profile(self, company_name: str,
                                    financial_data: pd.DataFrame,
                                    market_data: pd.DataFrame) -> Optional[Dict]:
        """企業の参入プロファイル作成"""
        company_data = financial_data[financial_data['企業名'] == company_name]
        
        if company_data.empty:
            return None
        
        entry_year = company_data['年度'].min()
        
        profile = {
            'company_name': company_name,
            'entry_year': entry_year,
            'entry_metrics': {},
            'performance_trajectory': {},
            'strategic_characteristics': {},
            'outcome_assessment': {}
        }
        
        # 参入時の指標
        entry_data = company_data[company_data['年度'] == entry_year]
        if not entry_data.empty:
            profile['entry_metrics'] = {
                'initial_revenue': float(entry_data['売上高'].iloc[0]),
                'initial_employees': float(entry_data.get('従業員数', 0).iloc[0]),
                'initial_rd_ratio': float(entry_data.get('研究開発費', 0).iloc[0] / 
                                        entry_data['売上高'].iloc[0]) if entry_data['売上高'].iloc[0] > 0 else 0,
                'initial_overseas_ratio': float(entry_data.get('海外売上高比率', 0).iloc[0])
            }
        
        # パフォーマンス軌跡（10年間）
        trajectory_data = company_data.head(10)
        if len(trajectory_data) >= 3:
            profile['performance_trajectory'] = {
                'revenue_cagr': self._calculate_cagr(
                    trajectory_data['売上高'].iloc[0],
                    trajectory_data['売上高'].iloc[-1],
                    len(trajectory_data)
                ),
                'profitability_trend': trajectory_data['売上高営業利益率'].corr(
                    pd.Series(range(len(trajectory_data)))
                ),
                'years_to_profitability': self._calculate_years_to_profitability(trajectory_data)
            }
        
        # 戦略特徴
        profile['strategic_characteristics'] = {
            'strategy_type': self._classify_individual_strategy(company_data),
            'innovation_intensity': self._calculate_innovation_score(company_data),
            'market_focus': self._analyze_market_focus(company_data),
            'growth_approach': self._classify_growth_approach(company_data)
        }
        
        # 結果評価
        profile['outcome_assessment'] = {
            'overall_success_score': self._calculate_simple_success_score(company_name, financial_data),
            'market_position': self._calculate_market_position_score(company_name, company_data, financial_data),
            'sustainability_score': self._calculate_sustainability_score(company_data),
            'status': self._determine_company_status(company_data)
        }
        
        return profile
    
    def _calculate_years_to_profitability(self, company_data: pd.DataFrame) -> Optional[int]:
        """黒字化までの年数計算"""
        profitable_years = company_data[company_data['売上高営業利益率'] > 0]
        
        if profitable_years.empty:
            return None  # 黒字化未達成
        
        first_profitable_year = profitable_years['年度'].min()
        entry_year = company_data['年度'].min()
        
        return first_profitable_year - entry_year
    
    def _classify_individual_strategy(self, company_data: pd.DataFrame) -> str:
        """個別企業の戦略分類"""
        early_data = company_data.head(3)
        
        # 戦略指標計算
        rd_intensity = (early_data.get('研究開発費', 0) / early_data['売上高']).mean()
        capex_intensity = (early_data.get('設備投資額', 0) / early_data['売上高']).mean()
        marketing_intensity = (early_data.get('広告宣伝費', 0) / early_data['売上高']).mean()
        
        # 戦略分類ロジック
        if rd_intensity > 0.08:
            return 'technology_leader'
        elif capex_intensity > 0.15:
            return 'capacity_builder'
        elif marketing_intensity > 0.05:
            return 'market_penetrator'
        else:
            return 'conservative_entrant'
    
    def _analyze_market_focus(self, company_data: pd.DataFrame) -> str:
        """市場フォーカス分析"""
        if '海外売上高比率' in company_data.columns:
            avg_overseas_ratio = company_data['海外売上高比率'].mean()
            if avg_overseas_ratio > 0.6:
                return 'global_focused'
            elif avg_overseas_ratio > 0.3:
                return 'regional_expansion'
            else:
                return 'domestic_focused'
        return 'unknown'
    
    def _classify_growth_approach(self, company_data: pd.DataFrame) -> str:
        """成長アプローチ分類"""
        if len(company_data) < 5:
            return 'insufficient_data'
        
        # 従業員数成長 vs 売上高成長の比較
        employee_growth = self._calculate_cagr(
            company_data.get('従業員数', pd.Series([1])).iloc[0],
            company_data.get('従業員数', pd.Series([1])).iloc[-1],
            len(company_data)
        )
        
        revenue_growth = self._calculate_cagr(
            company_data['売上高'].iloc[0],
            company_data['売上高'].iloc[-1],
            len(company_data)
        )
        
        if revenue_growth > employee_growth * 1.5:
            return 'productivity_driven'
        elif employee_growth > revenue_growth * 1.5:
            return 'expansion_driven'
        else:
            return 'balanced_growth'
    
    def _determine_company_status(self, company_data: pd.DataFrame) -> str:
        """企業現状ステータス判定"""
        latest_data = company_data.tail(3)
        
        # 最近3年間の動向
        revenue_trend = latest_data['売上高'].corr(pd.Series(range(len(latest_data))))
        profit_trend = latest_data.get('売上高営業利益率', pd.Series([0])).mean()
        
        if revenue_trend > 0.5 and profit_trend > 0.05:
            return 'thriving'
        elif revenue_trend > 0 and profit_trend > 0:
            return 'stable_growth'
        elif revenue_trend > 0 and profit_trend <= 0:
            return 'growth_but_unprofitable'
        elif revenue_trend <= 0 and profit_trend > 0:
            return 'declining_but_profitable'
        elif len(company_data) < 5:
            return 'early_stage'
        else:
            return 'struggling'
    
    def _analyze_market_evolution(self, market_category: str,
                                financial_data: pd.DataFrame,
                                market_data: pd.DataFrame) -> Dict:
        """市場進化分析"""
        evolution_analysis = {}
        
        # 年次での市場構造変化
        yearly_structure = {}
        
        for year in range(1984, 2025, 5):  # 5年刻みで分析
            year_data = financial_data[financial_data['年度'] == year]
            
            if not year_data.empty:
                # 市場集中度（HHI指数）
                market_revenues = year_data['売上高'].values
                total_revenue = np.sum(market_revenues)
                
                if total_revenue > 0:
                    market_shares = market_revenues / total_revenue
                    hhi_index = np.sum(market_shares ** 2)
                else:
                    hhi_index = 0
                
                # 平均企業規模
                avg_company_size = np.mean(market_revenues)
                
                # 技術集約度（R&D/売上高の市場平均）
                rd_intensity = (year_data['研究開発費'].sum() / 
                                year_data['売上高'].sum()) if year_data['売上高'].sum() > 0 else 0
                
                yearly_structure[year] = {
                    'hhi_index': float(hhi_index),
                    'average_company_size': float(avg_company_size),
                    'rd_intensity': float(rd_intensity),
                    'company_count': len(year_data),
                    'market_concentration': self._classify_market_concentration(hhi_index)
                }
        
        evolution_analysis['structural_evolution'] = yearly_structure
        
        # 進化トレンド分析
        if len(yearly_structure) > 2:
            years = sorted(yearly_structure.keys())
            hhi_trend = [yearly_structure[y]['hhi_index'] for y in years]
            size_trend = [yearly_structure[y]['average_company_size'] for y in years]
            rd_trend = [yearly_structure[y]['rd_intensity'] for y in years]
            
            evolution_analysis['trends'] = {
                'concentration_trend': 'increasing' if hhi_trend[-1] > hhi_trend[0] else 'decreasing',
                'scale_trend': 'increasing' if size_trend[-1] > size_trend[0] else 'decreasing',
                'innovation_trend': 'increasing' if rd_trend[-1] > rd_trend[0] else 'decreasing',
                'trend_coefficients': {
                    'concentration': float(np.polyfit(range(len(hhi_trend)), hhi_trend, 1)[0]),
                    'scale': float(np.polyfit(range(len(size_trend)), size_trend, 1)[0]),
                    'innovation': float(np.polyfit(range(len(rd_trend)), rd_trend, 1)[0])
                }
            }
        
        return evolution_analysis
    
    def _classify_market_concentration(self, hhi_index: float) -> str:
        """市場集中度分類"""
        if hhi_index < 0.15:
            return 'highly_competitive'
        elif hhi_index < 0.25:
            return 'moderately_competitive'
        elif hhi_index < 0.4:
            return 'moderately_concentrated'
        else:
            return 'highly_concentrated'
    
    def _identify_success_patterns(self, company_profiles: List[Dict]) -> Dict:
        """成功企業の共通パターン特定"""
        successful_companies = [
            profile for profile in company_profiles
            if profile['outcome_assessment']['overall_success_score'] and
            profile['outcome_assessment']['overall_success_score'] > 0.7
        ]
        
        if len(successful_companies) < 3:
            return {'insufficient_data': True}
        
        # 成功企業の共通特徴分析
        common_patterns = {
            'entry_timing': self._analyze_successful_timing_patterns(successful_companies),
            'strategic_approach': self._analyze_successful_strategies(successful_companies),
            'financial_characteristics': self._analyze_successful_financials(successful_companies),
            'growth_trajectory': self._analyze_successful_growth(successful_companies)
        }
        
        return common_patterns
    
    def _analyze_successful_timing_patterns(self, successful_companies: List[Dict]) -> Dict:
        """成功企業の参入タイミングパターン"""
        entry_years = [comp['entry_year'] for comp in successful_companies]
        
        # 参入年の分布分析
        year_distribution = {}
        for year in entry_years:
            decade = (year // 10) * 10
            decade_key = f"{decade}s"
            year_distribution[decade_key] = year_distribution.get(decade_key, 0) + 1
        
        # 最も成功率の高い参入時期
        peak_decade = max(year_distribution.items(), key=lambda x: x[1])[0]
        
        return {
            'optimal_entry_period': peak_decade,
            'entry_year_distribution': year_distribution,
            'average_entry_year': np.mean(entry_years),
            'entry_timing_variance': np.var(entry_years)
        }
    
    def _analyze_successful_strategies(self, successful_companies: List[Dict]) -> Dict:
        """成功企業の戦略パターン"""
        strategy_types = [comp['strategic_characteristics']['strategy_type'] 
                            for comp in successful_companies]
        
        # 戦略タイプ分布
        strategy_distribution = {}
        for strategy in strategy_types:
            strategy_distribution[strategy] = strategy_distribution.get(strategy, 0) + 1
        
        # 最も効果的な戦略
        dominant_strategy = max(strategy_distribution.items(), key=lambda x: x[1])[0]
        
        # 成功企業の平均特徴値
        avg_characteristics = {}
        for char_key in ['innovation_intensity', 'market_focus', 'growth_approach']:
            values = [comp['strategic_characteristics'].get(char_key, 0) 
                        for comp in successful_companies if isinstance(comp['strategic_characteristics'].get(char_key), (int, float))]
            if values:
                avg_characteristics[char_key] = np.mean(values)
        
        return {
            'dominant_strategy': dominant_strategy,
            'strategy_distribution': strategy_distribution,
            'average_characteristics': avg_characteristics,
            'strategy_diversity': len(strategy_distribution) / len(successful_companies)
        }
    
    def _analyze_successful_financials(self, successful_companies: List[Dict]) -> Dict:
        """成功企業の財務特徴"""
        financial_metrics = {}
        
        # 参入時の財務特徴
        entry_metrics = [comp['entry_metrics'] for comp in successful_companies]
        
        for metric in ['initial_revenue', 'initial_employees', 'initial_rd_ratio']:
            values = [metrics.get(metric, 0) for metrics in entry_metrics if metrics.get(metric) is not None]
            if values:
                financial_metrics[f'{metric}_avg'] = np.mean(values)
                financial_metrics[f'{metric}_median'] = np.median(values)
                financial_metrics[f'{metric}_std'] = np.std(values)
        
        return financial_metrics
    
    def _analyze_successful_growth(self, successful_companies: List[Dict]) -> Dict:
        """成功企業の成長軌跡分析"""
        growth_patterns = {}
        
        # 成長軌跡の特徴
        revenue_cagrs = [comp['performance_trajectory'].get('revenue_cagr', 0) 
                        for comp in successful_companies 
                        if comp['performance_trajectory'].get('revenue_cagr') is not None]
        
        profitability_years = [comp['performance_trajectory'].get('years_to_profitability') 
                                for comp in successful_companies 
                                if comp['performance_trajectory'].get('years_to_profitability') is not None]
        
        if revenue_cagrs:
            growth_patterns['average_revenue_cagr'] = np.mean(revenue_cagrs)
            growth_patterns['revenue_cagr_distribution'] = {
                'q25': np.percentile(revenue_cagrs, 25),
                'q50': np.percentile(revenue_cagrs, 50),
                'q75': np.percentile(revenue_cagrs, 75)
            }
        
        if profitability_years:
            growth_patterns['average_years_to_profit'] = np.mean(profitability_years)
            growth_patterns['fast_profitability_rate'] = np.mean([y <= 3 for y in profitability_years])
        
        return growth_patterns
    
    def _identify_failure_patterns(self, company_profiles: List[Dict]) -> Dict:
        """失敗企業の共通要因分析"""
        failed_companies = [
            profile for profile in company_profiles
            if profile['outcome_assessment']['overall_success_score'] and
            profile['outcome_assessment']['overall_success_score'] < 0.4
        ]
        
        if len(failed_companies) < 2:
            return {'insufficient_failure_data': True}
        
        # 失敗の共通パターン
        failure_patterns = {
            'common_timing_mistakes': self._analyze_failed_timing(failed_companies),
            'strategic_failures': self._analyze_failed_strategies(failed_companies),
            'financial_weaknesses': self._analyze_failed_financials(failed_companies),
            'warning_signals': self._identify_failure_warning_signals(failed_companies)
        }
        
        return failure_patterns
    
    def _analyze_failed_timing(self, failed_companies: List[Dict]) -> Dict:
        """失敗企業の参入タイミング分析"""
        entry_years = [comp['entry_year'] for comp in failed_companies]
        
        # 失敗の多い参入時期
        year_distribution = {}
        for year in entry_years:
            decade = (year // 10) * 10
            decade_key = f"{decade}s"
            year_distribution[decade_key] = year_distribution.get(decade_key, 0) + 1
        
        failure_prone_period = max(year_distribution.items(), key=lambda x: x[1])[0] if year_distribution else None
        
        return {
            'failure_prone_period': failure_prone_period,
            'entry_year_distribution': year_distribution,
            'late_entry_rate': np.mean([year > 2010 for year in entry_years])
        }
    
    def _analyze_failed_strategies(self, failed_companies: List[Dict]) -> Dict:
        """失敗企業の戦略分析"""
        failed_strategies = [comp['strategic_characteristics']['strategy_type'] 
                            for comp in failed_companies]
        
        strategy_failure_distribution = {}
        for strategy in failed_strategies:
            strategy_failure_distribution[strategy] = strategy_failure_distribution.get(strategy, 0) + 1
        
        return {
            'high_risk_strategies': strategy_failure_distribution,
            'most_risky_strategy': max(strategy_failure_distribution.items(), 
                                        key=lambda x: x[1])[0] if strategy_failure_distribution else None
        }
    
    def _analyze_failed_financials(self, failed_companies: List[Dict]) -> Dict:
        """失敗企業の財務分析"""
        financial_weaknesses = {}
        
        # 参入時の財務的弱点
        for metric in ['initial_revenue', 'initial_rd_ratio', 'initial_employees']:
            values = [comp['entry_metrics'].get(metric, 0) for comp in failed_companies
                        if comp['entry_metrics'].get(metric) is not None]
            if values:
                financial_weaknesses[f'{metric}_avg'] = np.mean(values)
        
        return financial_weaknesses
    
    def _identify_failure_warning_signals(self, failed_companies: List[Dict]) -> List[str]:
        """失敗の早期警告シグナル特定"""
        warning_signals = []
        
        # 共通する失敗パターンの特定
        profitability_delays = [comp['performance_trajectory'].get('years_to_profitability')
                                for comp in failed_companies
                                if comp['performance_trajectory'].get('years_to_profitability') is not None]
        
        if profitability_delays and np.mean(profitability_delays) > 5:
            warning_signals.append("黒字化まで5年以上要する")
        
        negative_trends = [comp['performance_trajectory'].get('profitability_trend', 0)
                            for comp in failed_companies]
        
        if np.mean(negative_trends) < -0.1:
            warning_signals.append("収益性が継続的に悪化")
        
        low_rd_companies = [comp for comp in failed_companies
                            if comp['entry_metrics'].get('initial_rd_ratio', 0) < 0.02]
        
        if len(low_rd_companies) / len(failed_companies) > 0.6:
            warning_signals.append("R&D投資比率が2%未満")
        
        return warning_signals

    def generate_market_entry_report(self, analysis_results: Dict) -> str:
        """市場参入分析レポート生成"""
        report = []
        report.append("# A2AI Market Entry Analysis Report")
        report.append("=" * 50)
        report.append()
        
        # エグゼクティブサマリー
        report.append("## Executive Summary")
        report.append()
        
        timing_analysis = analysis_results.get('entry_timing_analysis', {})
        
        # 市場別サマリー
        for market_cat, markets in self.market_categories.items():
            if market_cat in timing_analysis:
                data = timing_analysis[market_cat]
                
                report.append(f"### {market_cat.upper()} Markets")
                
                if data.get('first_mover_advantage'):
                    report.append("- **先発者優位** が確認されています")
                    report.append("- 推奨戦略: 技術革新による早期市場参入")
                elif data.get('late_mover_advantage'):
                    report.append("- **後発者優位** が確認されています") 
                    report.append("- 推奨戦略: 市場成熟後の効率的参入")
                else:
                    report.append("- 明確なタイミング優位性なし")
                    report.append("- 推奨戦略: 機会主義的・適応的参入")
                
                correlation = data.get('correlation_coefficient', 0)
                report.append(f"- タイミング-成功相関: {correlation:.3f}")
                report.append()
        
        # 重要成功要因
        report.append("## Critical Success Factors")
        report.append()
        
        success_factors = analysis_results.get('success_factors', {})
        all_factors = {}
        
        for market_cat, data in success_factors.items():
            critical_factors = data.get('critical_factors', {})
            top_factors = critical_factors.get('top_factors', [])
            
            for factor_name, importance in top_factors[:5]:
                if factor_name in all_factors:
                    all_factors[factor_name] += importance
                else:
                    all_factors[factor_name] = importance
        
        # 統合重要度ランキング
        ranked_factors = sorted(all_factors.items(), key=lambda x: x[1], reverse=True)[:10]
        
        for i, (factor, importance) in enumerate(ranked_factors, 1):
            report.append(f"{i}. **{factor}** (重要度: {importance:.3f})")
        
        report.append()
        
        # 推奨事項
        report.append("## Recommendations by Market Category")
        report.append()
        
        recommendations = analysis_results.get('recommendations', {})
        for market_cat, rec_data in recommendations.items():
            report.append(f"### {market_cat.upper()}")
            
            timing_rec = rec_data.get('entry_timing', {})
            strategy_rec = rec_data.get('strategy_recommendations', {})
            
            if timing_rec:
                report.append(f"- **推奨参入タイミング**: {timing_rec.get('recommendation', 'N/A')}")
                report.append(f"- **根拠**: {timing_rec.get('rationale', 'N/A')}")
            
            if strategy_rec:
                report.append(f"- **推奨戦略**: {strategy_rec.get('recommended_strategy', 'N/A')}")
                report.append(f"- **期待成功率**: {strategy_rec.get('success_rate', 0):.1%}")
            
            report.append()
        
        # 警告事項
        report.append("## Risk Factors and Warning Signals")
        report.append()
        
        # 各市場の詳細分析から警告シグナルを抽出
        detailed_analysis = analysis_results.get('detailed_analysis', {})
        for market_cat, details in detailed_analysis.items():
            failure_patterns = details.get('failure_patterns', {})
            warning_signals = failure_patterns.get('warning_signals', [])
            
            if warning_signals:
                report.append(f"### {market_cat} Market Warnings:")
                for signal in warning_signals:
                    report.append(f"- {signal}")
                report.append()
        
        # 結論
        report.append("## Conclusion")
        report.append()
        report.append("この分析により、市場参入における成功要因と失敗パターンが定量的に特定されました。")
        report.append("企業は自社の戦略特性と目標市場の特徴を照合し、最適な参入戦略を策定することが可能です。")
        
        return "\n".join(report)

# 使用例とテスト関数
def example_usage():
    """A2AI Market Entry Analyzer の使用例"""
    
    # サンプルデータの作成（実際の使用では実データを使用）
    np.random.seed(42)
    
    # 財務データサンプル
    companies = ['ファナック', 'オリンパス', 'トヨタ', '三洋電機', 'キオクシア']
    years = list(range(1984, 2025))
    
    financial_data = []
    for company in companies:
        for year in years:
            if company == '三洋電機' and year > 2012:  # 消滅企業
                continue
            if company == 'キオクシア' and year < 2018:  # 新設企業
                continue
                
            financial_data.append({
                '企業名': company,
                '年度': year,
                '売上高': np.random.lognormal(15, 0.5),
                '研究開発費': np.random.lognormal(12, 0.8),
                '設備投資額': np.random.lognormal(13, 0.6),
                '売上高営業利益率': np.random.normal(0.08, 0.05),
                '従業員数': np.random.lognormal(8, 0.3),
                '海外売上高比率': np.random.beta(2, 5),
                '総資産': np.random.lognormal(16, 0.4),
                '有形固定資産': np.random.lognormal(15, 0.5),
                '無形固定資産': np.random.lognormal(12, 0.7)
            })
    
    financial_df = pd.DataFrame(financial_data)
    
    # 企業データサンプル
    company_df = pd.DataFrame({
        '企業名': companies,
        '市場分野': ['ロボット', '内視鏡', '自動車', '家電', '半導体'],
        '設立年': [1972, 1919, 1937, 1947, 2018]
    })
    
    # 市場データサンプル（空のDataFrame）
    market_df = pd.DataFrame()
    
    # 分析実行
    analyzer = MarketEntryAnalyzer()
    results = analyzer.run_comprehensive_analysis(
        financial_df, company_df, market_df, target_market='high_share'
    )
    
    # レポート生成
    report = analyzer.generate_market_entry_report(results)
    print(report)
    
    # 可視化
    analyzer.visualize_entry_analysis(results)
    
    return results

if __name__ == "__main__":
    # 使用例実行
    example_results = example_usage()
    print("\nA2AI Market Entry Analysis completed successfully!")