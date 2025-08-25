"""
A2AI - Advanced Financial Analysis AI
企業成熟度指標分析モジュール (maturity_indicator.py)

企業のライフサイクル段階における成熟度を多次元で評価し、
世界シェア別市場での成熟度パターンを分析する。

企業成熟度の評価軸：
1. 財務成熟度 - 収益構造の安定性と効率性
2. 投資成熟度 - 設備投資と研究開発の戦略的配分
3. 組織成熟度 - 人的資源管理とガバナンス
4. 市場成熟度 - 市場地位と顧客基盤の安定性
5. 技術成熟度 - イノベーション創出力と技術蓄積
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import warnings
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# A2AI内部モジュールのインポート
from ...utils.data_utils import validate_dataframe, handle_missing_values
from ...utils.statistical_utils import calculate_stability_index, detect_outliers
from ...utils.lifecycle_utils import determine_lifecycle_stage


@dataclass
class MaturityDimensions:
    """成熟度の5次元評価"""
    financial_maturity: float      # 財務成熟度 (0-100)
    investment_maturity: float     # 投資成熟度 (0-100)
    organizational_maturity: float # 組織成熟度 (0-100)
    market_maturity: float        # 市場成熟度 (0-100)
    technological_maturity: float  # 技術成熟度 (0-100)
    
    @property
    def overall_maturity(self) -> float:
        """総合成熟度スコア"""
        dimensions = [
            self.financial_maturity,
            self.investment_maturity,
            self.organizational_maturity,
            self.market_maturity,
            self.technological_maturity
        ]
        return np.mean(dimensions)
    
    @property 
    def maturity_balance(self) -> float:
        """成熟度バランス指数（標準偏差の逆数）"""
        dimensions = [
            self.financial_maturity,
            self.investment_maturity,
            self.organizational_maturity,
            self.market_maturity,
            self.technological_maturity
        ]
        std = np.std(dimensions)
        return 100 - std if std < 100 else 0


@dataclass
class MaturityEvolution:
    """成熟度の時系列変化"""
    company_id: str
    years: List[int]
    maturity_trajectory: List[MaturityDimensions]
    growth_phase: str  # 'emerging', 'growing', 'mature', 'declining', 'transforming'
    maturity_trend: str  # 'improving', 'stable', 'declining', 'volatile'
    
    @property
    def maturity_velocity(self) -> float:
        """成熟度変化速度（年平均）"""
        if len(self.maturity_trajectory) < 2:
            return 0.0
        
        scores = [m.overall_maturity for m in self.maturity_trajectory]
        years_span = self.years[-1] - self.years[0]
        if years_span == 0:
            return 0.0
            
        return (scores[-1] - scores[0]) / years_span


class MaturityIndicatorAnalyzer:
    """企業成熟度指標分析クラス"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初期化
        
        Args:
            config: 分析設定パラメータ
        """
        self.config = config or self._get_default_config()
        self.scaler = StandardScaler()
        self.maturity_weights = self.config.get('maturity_weights', {
            'financial': 0.25,
            'investment': 0.20,
            'organizational': 0.20,
            'market': 0.20,
            'technological': 0.15
        })
        
    def _get_default_config(self) -> Dict:
        """デフォルト設定"""
        return {
            'min_years_for_analysis': 5,
            'outlier_threshold': 3.0,
            'stability_window': 5,
            'maturity_weights': {
                'financial': 0.25,
                'investment': 0.20,
                'organizational': 0.20,
                'market': 0.20,
                'technological': 0.15
            },
            'benchmark_companies': [],  # ベンチマーク企業リスト
            'industry_adjustments': True
        }
    
    def calculate_financial_maturity(self, df: pd.DataFrame, company_id: str) -> pd.Series:
        """
        財務成熟度の計算
        
        収益構造の安定性と効率性を評価
        - ROE安定性
        - 利益率の持続性
        - キャッシュフロー安定性
        - 財務レバレッジの適正性
        """
        company_data = df[df['company_id'] == company_id].copy()
        
        if len(company_data) < self.config['min_years_for_analysis']:
            return pd.Series(index=company_data.index, dtype=float)
        
        # ROE安定性指数 (0-25)
        roe_stability = calculate_stability_index(
            company_data['roe'], 
            window=self.config['stability_window']
        ) * 25
        
        # 利益率持続性指数 (0-25)
        profit_margin_trend = self._calculate_trend_stability(
            company_data['operating_profit_margin']
        ) * 25
        
        # キャッシュフロー安定性指数 (0-25)
        cf_stability = calculate_stability_index(
            company_data['operating_cf_ratio'],
            window=self.config['stability_window']
        ) * 25
        
        # 財務レバレッジ適正性指数 (0-25)
        leverage_score = self._calculate_leverage_maturity(
            company_data['debt_to_equity_ratio'],
            company_data['interest_coverage_ratio']
        ) * 25
        
        financial_maturity = roe_stability + profit_margin_trend + cf_stability + leverage_score
        
        return financial_maturity.clip(0, 100)
    
    def calculate_investment_maturity(self, df: pd.DataFrame, company_id: str) -> pd.Series:
        """
        投資成熟度の計算
        
        設備投資と研究開発の戦略的配分を評価
        - 設備投資効率性
        - R&D投資の持続性
        - 投資収益率
        - 投資戦略の一貫性
        """
        company_data = df[df['company_id'] == company_id].copy()
        
        if len(company_data) < self.config['min_years_for_analysis']:
            return pd.Series(index=company_data.index, dtype=float)
        
        # 設備投資効率性指数 (0-25)
        capex_efficiency = self._calculate_capex_efficiency(
            company_data['capex'],
            company_data['sales_growth_rate'],
            company_data['tangible_fixed_assets']
        ) * 25
        
        # R&D投資持続性指数 (0-25)
        rd_consistency = calculate_stability_index(
            company_data['rd_expense_ratio'],
            window=self.config['stability_window']
        ) * 25
        
        # 投資収益率指数 (0-25)
        investment_return = self._calculate_investment_return_score(
            company_data['roic'],
            company_data['investment_growth_rate']
        ) * 25
        
        # 投資戦略一貫性指数 (0-25)
        investment_strategy = self._calculate_investment_strategy_score(
            company_data['capex_ratio'],
            company_data['rd_expense_ratio'],
            company_data['acquisition_investment']
        ) * 25
        
        investment_maturity = capex_efficiency + rd_consistency + investment_return + investment_strategy
        
        return investment_maturity.clip(0, 100)
    
    def calculate_organizational_maturity(self, df: pd.DataFrame, company_id: str) -> pd.Series:
        """
        組織成熟度の計算
        
        人的資源管理とガバナンスを評価
        - 従業員生産性
        - 人件費効率性
        - 組織安定性
        - ガバナンス品質
        """
        company_data = df[df['company_id'] == company_id].copy()
        
        if len(company_data) < self.config['min_years_for_analysis']:
            return pd.Series(index=company_data.index, dtype=float)
        
        # 従業員生産性指数 (0-25)
        employee_productivity = self._calculate_productivity_score(
            company_data['labor_productivity'],
            company_data['employee_count_growth']
        ) * 25
        
        # 人件費効率性指数 (0-25)
        labor_efficiency = self._calculate_labor_efficiency(
            company_data['personnel_cost_ratio'],
            company_data['average_salary'],
            company_data['sales_per_employee']
        ) * 25
        
        # 組織安定性指数 (0-25)
        org_stability = calculate_stability_index(
            company_data['employee_count'],
            window=self.config['stability_window']
        ) * 25
        
        # ガバナンス品質指数 (0-25)
        governance_score = self._calculate_governance_score(
            company_data['dividend_payout_ratio'],
            company_data['total_payout_ratio'],
            company_data['board_independence_ratio']
        ) * 25
        
        organizational_maturity = employee_productivity + labor_efficiency + org_stability + governance_score
        
        return organizational_maturity.clip(0, 100)
    
    def calculate_market_maturity(self, df: pd.DataFrame, company_id: str) -> pd.Series:
        """
        市場成熟度の計算
        
        市場地位と顧客基盤の安定性を評価
        - 市場シェア安定性
        - 顧客基盤の多様性
        - 価格設定力
        - ブランド価値
        """
        company_data = df[df['company_id'] == company_id].copy()
        
        if len(company_data) < self.config['min_years_for_analysis']:
            return pd.Series(index=company_data.index, dtype=float)
        
        # 市場シェア安定性指数 (0-25)
        market_share_stability = calculate_stability_index(
            company_data['market_share'],
            window=self.config['stability_window']
        ) * 25
        
        # 顧客多様性指数 (0-25)
        customer_diversity = self._calculate_customer_diversity(
            company_data['overseas_sales_ratio'],
            company_data['segment_count'],
            company_data['customer_concentration']
        ) * 25
        
        # 価格設定力指数 (0-25)
        pricing_power = self._calculate_pricing_power(
            company_data['gross_profit_margin'],
            company_data['sales_growth_rate'],
            company_data['market_growth_rate']
        ) * 25
        
        # ブランド価値指数 (0-25)
        brand_value = self._calculate_brand_value_score(
            company_data['intangible_assets_ratio'],
            company_data['advertising_expense_ratio'],
            company_data['premium_pricing_ability']
        ) * 25
        
        market_maturity = market_share_stability + customer_diversity + pricing_power + brand_value
        
        return market_maturity.clip(0, 100)
    
    def calculate_technological_maturity(self, df: pd.DataFrame, company_id: str) -> pd.Series:
        """
        技術成熟度の計算
        
        イノベーション創出力と技術蓄積を評価
        - R&D効率性
        - 技術蓄積度
        - イノベーション創出力
        - 技術競争力
        """
        company_data = df[df['company_id'] == company_id].copy()
        
        if len(company_data) < self.config['min_years_for_analysis']:
            return pd.Series(index=company_data.index, dtype=float)
        
        # R&D効率性指数 (0-25)
        rd_efficiency = self._calculate_rd_efficiency(
            company_data['rd_expense_ratio'],
            company_data['patent_applications'],
            company_data['new_product_revenue_ratio']
        ) * 25
        
        # 技術蓄積度指数 (0-25)
        tech_accumulation = self._calculate_tech_accumulation(
            company_data['intangible_assets'],
            company_data['rd_investment_cumulative'],
            company_data['patent_portfolio_value']
        ) * 25
        
        # イノベーション創出力指数 (0-25)
        innovation_power = self._calculate_innovation_power(
            company_data['new_product_ratio'],
            company_data['technology_licensing_income'],
            company_data['disruptive_innovation_score']
        ) * 25
        
        # 技術競争力指数 (0-25)
        tech_competitiveness = self._calculate_tech_competitiveness(
            company_data['technology_leadership_score'],
            company_data['industry_tech_ranking'],
            company_data['tech_partnership_count']
        ) * 25
        
        technological_maturity = rd_efficiency + tech_accumulation + innovation_power + tech_competitiveness
        
        return technological_maturity.clip(0, 100)
    
    def analyze_company_maturity(self, df: pd.DataFrame, company_id: str) -> MaturityEvolution:
        """
        個別企業の成熟度分析
        
        Args:
            df: 財務データ
            company_id: 企業ID
            
        Returns:
            MaturityEvolution: 成熟度進化データ
        """
        company_data = df[df['company_id'] == company_id].copy()
        
        if len(company_data) < self.config['min_years_for_analysis']:
            warnings.warn(f"Company {company_id} has insufficient data for maturity analysis")
            return None
        
        # 各次元の成熟度を計算
        financial_maturity = self.calculate_financial_maturity(df, company_id)
        investment_maturity = self.calculate_investment_maturity(df, company_id)
        organizational_maturity = self.calculate_organizational_maturity(df, company_id)
        market_maturity = self.calculate_market_maturity(df, company_id)
        technological_maturity = self.calculate_technological_maturity(df, company_id)
        
        # 時系列での成熟度データを構築
        maturity_trajectory = []
        years = company_data['year'].tolist()
        
        for i, year in enumerate(years):
            dimensions = MaturityDimensions(
                financial_maturity=financial_maturity.iloc[i] if i < len(financial_maturity) else 0,
                investment_maturity=investment_maturity.iloc[i] if i < len(investment_maturity) else 0,
                organizational_maturity=organizational_maturity.iloc[i] if i < len(organizational_maturity) else 0,
                market_maturity=market_maturity.iloc[i] if i < len(market_maturity) else 0,
                technological_maturity=technological_maturity.iloc[i] if i < len(technological_maturity) else 0
            )
            maturity_trajectory.append(dimensions)
        
        # 成長フェーズの判定
        growth_phase = self._determine_growth_phase(company_data, maturity_trajectory)
        
        # 成熟度トレンドの判定
        maturity_trend = self._determine_maturity_trend(maturity_trajectory)
        
        return MaturityEvolution(
            company_id=company_id,
            years=years,
            maturity_trajectory=maturity_trajectory,
            growth_phase=growth_phase,
            maturity_trend=maturity_trend
        )
    
    def analyze_market_maturity_patterns(self, df: pd.DataFrame, 
                                        market_categories: Dict[str, List[str]]) -> Dict:
        """
        市場別成熟度パターン分析
        
        Args:
            df: 財務データ
            market_categories: 市場カテゴリ別企業リスト
                {'high_share': [...], 'declining': [...], 'lost': [...]}
                
        Returns:
            Dict: 市場別成熟度分析結果
        """
        market_analysis = {}
        
        for market_type, company_list in market_categories.items():
            market_maturity_data = []
            
            for company_id in company_list:
                if company_id not in df['company_id'].values:
                    continue
                    
                maturity_evolution = self.analyze_company_maturity(df, company_id)
                if maturity_evolution:
                    market_maturity_data.append(maturity_evolution)
            
            if market_maturity_data:
                market_analysis[market_type] = self._analyze_market_patterns(
                    market_maturity_data, market_type
                )
        
        return market_analysis
    
    def identify_maturity_clusters(self, df: pd.DataFrame, 
                                    n_clusters: int = 5) -> Dict:
        """
        成熟度クラスタリング分析
        
        企業を成熟度パターンに基づいてグループ化
        """
        # 全企業の最新成熟度データを収集
        company_maturity_data = []
        company_ids = df['company_id'].unique()
        
        for company_id in company_ids:
            maturity_evolution = self.analyze_company_maturity(df, company_id)
            if maturity_evolution and len(maturity_evolution.maturity_trajectory) > 0:
                latest_maturity = maturity_evolution.maturity_trajectory[-1]
                company_maturity_data.append({
                    'company_id': company_id,
                    'financial_maturity': latest_maturity.financial_maturity,
                    'investment_maturity': latest_maturity.investment_maturity,
                    'organizational_maturity': latest_maturity.organizational_maturity,
                    'market_maturity': latest_maturity.market_maturity,
                    'technological_maturity': latest_maturity.technological_maturity,
                    'overall_maturity': latest_maturity.overall_maturity,
                    'maturity_balance': latest_maturity.maturity_balance
                })
        
        if not company_maturity_data:
            return {}
        
        # データフレーム化
        maturity_df = pd.DataFrame(company_maturity_data)
        
        # クラスタリング用特徴量
        features = ['financial_maturity', 'investment_maturity', 'organizational_maturity',
                    'market_maturity', 'technological_maturity']
        
        X = maturity_df[features].values
        X_scaled = self.scaler.fit_transform(X)
        
        # K-means クラスタリング
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        
        maturity_df['cluster'] = clusters
        
        # クラスター分析
        cluster_analysis = {}
        for cluster_id in range(n_clusters):
            cluster_data = maturity_df[maturity_df['cluster'] == cluster_id]
            
            cluster_analysis[f'cluster_{cluster_id}'] = {
                'company_count': len(cluster_data),
                'companies': cluster_data['company_id'].tolist(),
                'avg_maturity': {
                    'financial': cluster_data['financial_maturity'].mean(),
                    'investment': cluster_data['investment_maturity'].mean(),
                    'organizational': cluster_data['organizational_maturity'].mean(),
                    'market': cluster_data['market_maturity'].mean(),
                    'technological': cluster_data['technological_maturity'].mean(),
                    'overall': cluster_data['overall_maturity'].mean()
                },
                'maturity_profile': self._create_cluster_profile(cluster_data),
                'characteristics': self._identify_cluster_characteristics(cluster_data)
            }
        
        return {
            'cluster_analysis': cluster_analysis,
            'clustering_model': kmeans,
            'feature_importance': self._calculate_feature_importance(X_scaled, clusters)
        }
    
    # ユーティリティメソッド群
    
    def _calculate_trend_stability(self, series: pd.Series) -> pd.Series:
        """トレンドの安定性を計算"""
        if len(series) < 3:
            return pd.Series([0.5] * len(series), index=series.index)
        
        # 線形回帰の決定係数を安定性指標とする
        stability_scores = []
        
        for i in range(len(series)):
            if i < 2:
                stability_scores.append(0.5)
                continue
                
            window_data = series.iloc[max(0, i-4):i+1]
            x = np.arange(len(window_data))
            
            if len(window_data) > 1:
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, window_data)
                stability = min(abs(r_value), 1.0)
            else:
                stability = 0.5
                
            stability_scores.append(stability)
        
        return pd.Series(stability_scores, index=series.index)
    
    def _calculate_leverage_maturity(self, debt_ratio: pd.Series, 
                                    interest_coverage: pd.Series) -> pd.Series:
        """財務レバレッジ成熟度の計算"""
        # 適正なレバレッジ水準の維持度を評価
        optimal_debt_ratio = 0.3  # 業界平均的な適正水準
        
        # 負債比率の適正性
        debt_score = 1 - abs(debt_ratio - optimal_debt_ratio) / optimal_debt_ratio
        debt_score = debt_score.clip(0, 1)
        
        # 利子支払い能力
        coverage_score = np.minimum(interest_coverage / 10, 1.0)  # 10倍以上で満点
        
        return (debt_score + coverage_score) / 2
    
    def _calculate_capex_efficiency(self, capex: pd.Series, 
                                    sales_growth: pd.Series,
                                    fixed_assets: pd.Series) -> pd.Series:
        """設備投資効率性の計算"""
        # 設備投資が売上成長に寄与している度合い
        capex_ratio = capex / fixed_assets.shift(1)
        
        # 効率性スコア = 売上成長率 / 設備投資比率
        efficiency = sales_growth / (capex_ratio + 0.01)  # ゼロ除算回避
        
        # 正規化 (0-1)
        return np.minimum(efficiency / efficiency.quantile(0.8), 1.0)
    
    def _calculate_investment_return_score(self, roic: pd.Series, 
                                            investment_growth: pd.Series) -> pd.Series:
        """投資収益率スコアの計算"""
        # ROIC水準スコア
        roic_score = np.minimum(roic / 0.15, 1.0)  # 15%以上で満点
        
        # 投資成長率の適正性
        growth_score = np.maximum(0, 1 - abs(investment_growth) / 0.5)  # 適正成長率50%
        
        return (roic_score + growth_score) / 2
    
    def _calculate_investment_strategy_score(self, capex_ratio: pd.Series,
                                            rd_ratio: pd.Series,
                                            acquisition: pd.Series) -> pd.Series:
        """投資戦略一貫性スコアの計算"""
        # 投資ポートフォリオの安定性
        total_investment = capex_ratio + rd_ratio + acquisition
        
        # 各投資項目の比率安定性
        capex_stability = calculate_stability_index(capex_ratio)
        rd_stability = calculate_stability_index(rd_ratio)
        
        return (capex_stability + rd_stability) / 2
    
    def _calculate_productivity_score(self, labor_productivity: pd.Series,
                                    employee_growth: pd.Series) -> pd.Series:
        """生産性スコアの計算"""
        # 労働生産性の改善トレンド
        productivity_trend = labor_productivity.pct_change().rolling(3).mean()
        trend_score = np.maximum(0, productivity_trend * 10)  # 10%改善で満点
        
        # 従業員数増加の効率性
        growth_efficiency = labor_productivity / (1 + employee_growth)
        efficiency_score = np.minimum(growth_efficiency / growth_efficiency.quantile(0.8), 1.0)
        
        return (trend_score + efficiency_score) / 2

    def _calculate_labor_efficiency(self, personnel_cost_ratio: pd.Series,
                                    average_salary: pd.Series,
                                    sales_per_employee: pd.Series) -> pd.Series:
        """労働効率性の計算"""
        # 人件費効率性 = 従業員当たり売上 / 平均給与
        efficiency = sales_per_employee / average_salary
        
        # 正規化
        return np.minimum(efficiency / efficiency.quantile(0.8), 1.0)
    
    def _calculate_governance_score(self, dividend_payout: pd.Series,
                                    total_payout: pd.Series,
                                    board_independence: pd.Series) -> pd.Series:
        """ガバナンススコアの計算"""
        # 配当政策の安定性
        dividend_stability = calculate_stability_index(dividend_payout)
        
        # 株主還元の適正性
        payout_score = np.minimum(total_payout / 0.3, 1.0)  # 30%程度が適正
        
        # 取締役会独立性
        independence_score = board_independence
        
        return (dividend_stability + payout_score + independence_score) / 3
    
    def _calculate_customer_diversity(self, overseas_ratio: pd.Series,
                                    segment_count: pd.Series,
                                    customer_concentration: pd.Series) -> pd.Series:
        """顧客多様性の計算"""
        # 地理的多様性
        geographic_diversity = overseas_ratio
        
        # 事業多様性
        business_diversity = np.minimum(segment_count / 5, 1.0)  # 5セグメント以上で満点
        
        # 顧客集中度の逆数
        customer_diversity = 1 - customer_concentration
        
        return (geographic_diversity + business_diversity + customer_diversity) / 3
    
    def _calculate_pricing_power(self, gross_margin: pd.Series,
                                sales_growth: pd.Series,
                                market_growth: pd.Series) -> pd.Series:
        """価格設定力の計算"""
        # 市場成長率を上回る売上成長
        growth_premium = np.maximum(0, (sales_growth - market_growth) / market_growth)
        
        # 粗利率水準
        margin_score = np.minimum(gross_margin / 0.4, 1.0)  # 40%以上で満点
        
        return (growth_premium + margin_score) / 2
    
    def _calculate_brand_value_score(self, intangible_ratio: pd.Series,
                                    advertising_ratio: pd.Series,
                                    premium_pricing: pd.Series) -> pd.Series:
        """ブランド価値スコアの計算"""
        # 無形資産比率
        intangible_score = np.minimum(intangible_ratio / 0.2, 1.0)
        
        # 広告宣伝費投資
        advertising_score = np.minimum(advertising_ratio / 0.05, 1.0)
        
        # プレミアム価格設定能力
        premium_score = premium_pricing
        
        return (intangible_score + advertising_score + premium_score) / 3
    
    def _calculate_rd_efficiency(self, rd_ratio: pd.Series,
                                patents: pd.Series,
                                new_product_ratio: pd.Series) -> pd.Series:
        """R&D効率性の計算"""
        # 特許創出効率
        patent_efficiency = patents / (rd_ratio + 0.01)
        patent_score = np.minimum(patent_efficiency / patent_efficiency.quantile(0.8), 1.0)
        
        # 新製品売上比率
        product_score = new_product_ratio
        
        return (patent_score + product_score) / 2
    
    def _calculate_tech_accumulation(self, intangible_assets: pd.Series,
                                    rd_cumulative: pd.Series,
                                    patent_value: pd.Series) -> pd.Series:
        """技術蓄積度の計算"""
        # 無形資産蓄積度
        intangible_growth = intangible_assets.pct_change().rolling(5).mean()
        intangible_score = np.maximum(0, intangible_growth * 10)
        
        # R&D累積投資効果
        rd_score = np.minimum(rd_cumulative / rd_cumulative.quantile(0.8), 1.0)
        
        # 特許ポートフォリオ価値
        patent_score = np.minimum(patent_value / patent_value.quantile(0.8), 1.0)
        
        return (intangible_score + rd_score + patent_score) / 3
    
    def _calculate_innovation_power(self, new_product_ratio: pd.Series,
                                    license_income: pd.Series,
                                    innovation_score: pd.Series) -> pd.Series:
        """イノベーション創出力の計算"""
        # 新製品比率
        product_innovation = new_product_ratio
        
        # 技術ライセンス収入
        license_score = np.minimum(license_income / license_income.quantile(0.8), 1.0)
        
        # 破壊的イノベーションスコア
        disruption_score = innovation_score
        
        return (product_innovation + license_score + disruption_score) / 3
    
    def _calculate_tech_competitiveness(self, leadership_score: pd.Series,
                                        industry_ranking: pd.Series,
                                        partnership_count: pd.Series) -> pd.Series:
        """技術競争力の計算"""
        # 技術リーダーシップスコア
        leadership = leadership_score
        
        # 業界内技術ランキング（逆数）
        ranking_score = 1 / (industry_ranking + 1)
        
        # 技術パートナーシップ数
        partnership_score = np.minimum(partnership_count / partnership_count.quantile(0.8), 1.0)
        
        return (leadership + ranking_score + partnership_score) / 3
    
    def _determine_growth_phase(self, company_data: pd.DataFrame, 
                                maturity_trajectory: List[MaturityDimensions]) -> str:
        """企業の成長フェーズを判定"""
        if len(maturity_trajectory) < 3:
            return 'unknown'
        
        # 最近3年の成熟度変化
        recent_maturity = [m.overall_maturity for m in maturity_trajectory[-3:]]
        maturity_trend = np.polyfit(range(3), recent_maturity, 1)[0]
        
        # 売上成長率の傾向
        recent_growth = company_data['sales_growth_rate'].tail(3).mean()
        
        # ROEの水準
        recent_roe = company_data['roe'].tail(3).mean()
        
        # フェーズ判定ロジック
        avg_maturity = np.mean(recent_maturity)
        
        if avg_maturity < 40:
            if recent_growth > 0.1:  # 10%以上成長
                return 'emerging'
            else:
                return 'struggling'
        elif avg_maturity < 70:
            if maturity_trend > 2 and recent_growth > 0.05:
                return 'growing' 
            elif maturity_trend < -2:
                return 'declining'
            else:
                return 'transitioning'
        else:  # avg_maturity >= 70
            if recent_roe > 0.15:
                return 'mature'
            elif maturity_trend < -3:
                return 'declining'
            else:
                return 'transforming'
    
    def _determine_maturity_trend(self, maturity_trajectory: List[MaturityDimensions]) -> str:
        """成熟度トレンドを判定"""
        if len(maturity_trajectory) < 5:
            return 'insufficient_data'
        
        # 成熟度スコアの時系列
        scores = [m.overall_maturity for m in maturity_trajectory]
        
        # トレンド分析
        x = np.arange(len(scores))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, scores)
        
        # 変動性の計算
        volatility = np.std(scores) / np.mean(scores)
        
        # トレンド判定
        if volatility > 0.2:  # 変動係数20%以上
            return 'volatile'
        elif slope > 1.5:
            return 'improving'
        elif slope < -1.5:
            return 'declining'
        else:
            return 'stable'
    
    def _analyze_market_patterns(self, market_maturity_data: List[MaturityEvolution], 
                                market_type: str) -> Dict:
        """市場別成熟度パターンの分析"""
        if not market_maturity_data:
            return {}
        
        # 各次元の平均成熟度
        avg_dimensions = {
            'financial': [],
            'investment': [], 
            'organizational': [],
            'market': [],
            'technological': []
        }
        
        growth_phases = []
        maturity_trends = []
        overall_scores = []
        
        for evolution in market_maturity_data:
            if evolution.maturity_trajectory:
                latest = evolution.maturity_trajectory[-1]
                avg_dimensions['financial'].append(latest.financial_maturity)
                avg_dimensions['investment'].append(latest.investment_maturity)
                avg_dimensions['organizational'].append(latest.organizational_maturity)
                avg_dimensions['market'].append(latest.market_maturity)
                avg_dimensions['technological'].append(latest.technological_maturity)
                overall_scores.append(latest.overall_maturity)
                
            growth_phases.append(evolution.growth_phase)
            maturity_trends.append(evolution.maturity_trend)
        
        # 統計サマリー
        market_summary = {
            'market_type': market_type,
            'company_count': len(market_maturity_data),
            'avg_maturity_dimensions': {
                dim: np.mean(scores) for dim, scores in avg_dimensions.items()
            },
            'avg_overall_maturity': np.mean(overall_scores),
            'maturity_std': np.std(overall_scores),
            'dominant_growth_phase': max(set(growth_phases), key=growth_phases.count),
            'dominant_maturity_trend': max(set(maturity_trends), key=maturity_trends.count),
            'maturity_distribution': {
                'high_maturity': sum(1 for s in overall_scores if s >= 70) / len(overall_scores),
                'medium_maturity': sum(1 for s in overall_scores if 40 <= s < 70) / len(overall_scores),
                'low_maturity': sum(1 for s in overall_scores if s < 40) / len(overall_scores)
            }
        }
        
        # 市場特性の分析
        market_characteristics = self._identify_market_characteristics(
            market_maturity_data, market_type
        )
        market_summary['characteristics'] = market_characteristics
        
        return market_summary
    
    def _identify_market_characteristics(self, market_data: List[MaturityEvolution], 
                                        market_type: str) -> Dict:
        """市場特性の特定"""
        characteristics = {}
        
        # 成熟度プロファイル分析
        dimension_scores = {
            'financial': [],
            'investment': [],
            'organizational': [],
            'market': [],
            'technological': []
        }
        
        for evolution in market_data:
            if evolution.maturity_trajectory:
                latest = evolution.maturity_trajectory[-1]
                dimension_scores['financial'].append(latest.financial_maturity)
                dimension_scores['investment'].append(latest.investment_maturity)
                dimension_scores['organizational'].append(latest.organizational_maturity)
                dimension_scores['market'].append(latest.market_maturity)
                dimension_scores['technological'].append(latest.technological_maturity)
        
        # 各次元の強み・弱み分析
        avg_scores = {dim: np.mean(scores) for dim, scores in dimension_scores.items()}
        max_dimension = max(avg_scores, key=avg_scores.get)
        min_dimension = min(avg_scores, key=avg_scores.get)
        
        characteristics['strongest_dimension'] = max_dimension
        characteristics['weakest_dimension'] = min_dimension
        characteristics['dimension_gap'] = avg_scores[max_dimension] - avg_scores[min_dimension]
        
        # 市場タイプ別特徴
        if market_type == 'high_share':
            characteristics['profile'] = 'market_leaders'
            characteristics['key_success_factors'] = self._identify_leader_success_factors(dimension_scores)
        elif market_type == 'declining':
            characteristics['profile'] = 'market_challengers' 
            characteristics['challenge_areas'] = self._identify_challenge_areas(dimension_scores)
        else:  # lost
            characteristics['profile'] = 'market_exiters'
            characteristics['failure_patterns'] = self._identify_failure_patterns(dimension_scores)
        
        return characteristics
    
    def _identify_leader_success_factors(self, dimension_scores: Dict) -> List[str]:
        """市場リーダーの成功要因を特定"""
        success_factors = []
        
        # 高スコア次元を成功要因として特定
        avg_scores = {dim: np.mean(scores) for dim, scores in dimension_scores.items()}
        
        for dim, score in avg_scores.items():
            if score >= 70:
                success_factors.append(f"high_{dim}_maturity")
        
        # バランスの良さも評価
        score_std = np.std(list(avg_scores.values()))
        if score_std < 15:
            success_factors.append("balanced_maturity_profile")
        
        return success_factors
    
    def _identify_challenge_areas(self, dimension_scores: Dict) -> List[str]:
        """課題領域を特定"""
        challenge_areas = []
        
        avg_scores = {dim: np.mean(scores) for dim, scores in dimension_scores.items()}
        
        for dim, score in avg_scores.items():
            if score < 50:
                challenge_areas.append(f"low_{dim}_maturity")
        
        return challenge_areas
    
    def _identify_failure_patterns(self, dimension_scores: Dict) -> List[str]:
        """失敗パターンを特定"""
        failure_patterns = []
        
        avg_scores = {dim: np.mean(scores) for dim, scores in dimension_scores.items()}
        
        # 全般的な低スコア
        if all(score < 60 for score in avg_scores.values()):
            failure_patterns.append("overall_low_maturity")
        
        # 特定次元の極端な低下
        for dim, score in avg_scores.items():
            if score < 30:
                failure_patterns.append(f"critical_{dim}_weakness")
        
        # 不均衡な成熟度
        score_std = np.std(list(avg_scores.values()))
        if score_std > 25:
            failure_patterns.append("unbalanced_maturity_development")
        
        return failure_patterns
    
    def _create_cluster_profile(self, cluster_data: pd.DataFrame) -> Dict:
        """クラスタープロファイルを作成"""
        features = ['financial_maturity', 'investment_maturity', 'organizational_maturity',
                    'market_maturity', 'technological_maturity']
        
        profile = {}
        for feature in features:
            profile[feature] = {
                'mean': cluster_data[feature].mean(),
                'std': cluster_data[feature].std(),
                'min': cluster_data[feature].min(),
                'max': cluster_data[feature].max()
            }
        
        profile['overall_maturity'] = {
            'mean': cluster_data['overall_maturity'].mean(),
            'std': cluster_data['overall_maturity'].std()
        }
        
        return profile
    
    def _identify_cluster_characteristics(self, cluster_data: pd.DataFrame) -> List[str]:
        """クラスター特徴を特定"""
        characteristics = []
        
        features = ['financial_maturity', 'investment_maturity', 'organizational_maturity',
                    'market_maturity', 'technological_maturity']
        
        means = {feature: cluster_data[feature].mean() for feature in features}
        
        # 高成熟度次元
        high_maturity_dims = [dim for dim, score in means.items() if score >= 70]
        if high_maturity_dims:
            characteristics.append(f"high_maturity_in_{','.join(high_maturity_dims)}")
        
        # 低成熟度次元  
        low_maturity_dims = [dim for dim, score in means.items() if score < 40]
        if low_maturity_dims:
            characteristics.append(f"low_maturity_in_{','.join(low_maturity_dims)}")
        
        # 全体的な成熟度レベル
        overall_mean = cluster_data['overall_maturity'].mean()
        if overall_mean >= 70:
            characteristics.append("highly_mature_companies")
        elif overall_mean < 40:
            characteristics.append("low_maturity_companies")
        else:
            characteristics.append("medium_maturity_companies")
        
        # バランス評価
        maturity_balance_avg = cluster_data['maturity_balance'].mean()
        if maturity_balance_avg >= 80:
            characteristics.append("well_balanced_maturity")
        elif maturity_balance_avg < 60:
            characteristics.append("unbalanced_maturity")
        
        return characteristics
    
    def _calculate_feature_importance(self, X: np.ndarray, clusters: np.ndarray) -> Dict:
        """特徴量重要度を計算"""
        from sklearn.ensemble import RandomForestClassifier
        
        # ランダムフォレストで特徴量重要度を計算
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, clusters)
        
        feature_names = ['financial_maturity', 'investment_maturity', 'organizational_maturity',
                        'market_maturity', 'technological_maturity']
        
        importance_dict = {}
        for i, importance in enumerate(rf.feature_importances_):
            importance_dict[feature_names[i]] = importance
        
        return importance_dict
    
    def generate_maturity_report(self, df: pd.DataFrame, 
                                market_categories: Dict[str, List[str]],
                                output_path: str = None) -> Dict:
        """
        成熟度分析レポートの生成
        
        Args:
            df: 財務データ
            market_categories: 市場カテゴリ別企業リスト
            output_path: レポート保存パス
            
        Returns:
            Dict: 包括的な成熟度分析レポート
        """
        report = {
            'analysis_summary': {
                'total_companies': len(df['company_id'].unique()),
                'analysis_period': f"{df['year'].min()}-{df['year'].max()}",
                'market_categories': len(market_categories)
            }
        }
        
        # 市場別分析
        print("市場別成熟度パターンを分析中...")
        market_analysis = self.analyze_market_maturity_patterns(df, market_categories)
        report['market_analysis'] = market_analysis
        
        # クラスタリング分析
        print("成熟度クラスタリングを実行中...")
        cluster_analysis = self.identify_maturity_clusters(df)
        report['cluster_analysis'] = cluster_analysis
        
        # 市場間比較
        report['market_comparison'] = self._compare_markets(market_analysis)
        
        # 主要インサイト
        report['key_insights'] = self._generate_key_insights(market_analysis, cluster_analysis)
        
        # レポート保存
        if output_path:
            self._save_report(report, output_path)
        
        return report
    
    def _compare_markets(self, market_analysis: Dict) -> Dict:
        """市場間比較分析"""
        if len(market_analysis) < 2:
            return {}
        
        comparison = {}
        
        # 平均成熟度比較
        market_maturity_scores = {}
        for market_type, analysis in market_analysis.items():
            market_maturity_scores[market_type] = analysis['avg_overall_maturity']
        
        comparison['maturity_ranking'] = sorted(
            market_maturity_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # 次元別比較
        dimensions = ['financial', 'investment', 'organizational', 'market', 'technological']
        comparison['dimension_comparison'] = {}
        
        for dim in dimensions:
            dim_scores = {}
            for market_type, analysis in market_analysis.items():
                dim_scores[market_type] = analysis['avg_maturity_dimensions'][dim]
            
            comparison['dimension_comparison'][dim] = sorted(
                dim_scores.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
        
        return comparison
    
    def _generate_key_insights(self, market_analysis: Dict, cluster_analysis: Dict) -> List[str]:
        """主要インサイトの生成"""
        insights = []
        
        if market_analysis:
            # 市場別インサイト
            market_scores = {
                market: analysis['avg_overall_maturity'] 
                for market, analysis in market_analysis.items()
            }
            
            best_market = max(market_scores, key=market_scores.get)
            worst_market = min(market_scores, key=market_scores.get)
            
            insights.append(f"最も成熟度が高い市場: {best_market} (平均スコア: {market_scores[best_market]:.1f})")
            insights.append(f"最も成熟度が低い市場: {worst_market} (平均スコア: {market_scores[worst_market]:.1f})")
            
            # 次元別強み分析
            for market_type, analysis in market_analysis.items():
                strongest_dim = analysis.get('characteristics', {}).get('strongest_dimension')
                if strongest_dim:
                    insights.append(f"{market_type}市場の強み: {strongest_dim}")
        
        if cluster_analysis and 'cluster_analysis' in cluster_analysis:
            # クラスター分析インサイト
            clusters = cluster_analysis['cluster_analysis']
            largest_cluster = max(clusters.values(), key=lambda x: x['company_count'])
            insights.append(f"最大クラスター: {largest_cluster['company_count']}社が類似の成熟度パターン")
        
        return insights
    
    def _save_report(self, report: Dict, output_path: str):
        """レポートの保存"""
        import json
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)