"""
A2AI (Advanced Financial Analysis AI) - Innovation Impact Analyzer

新設企業のイノベーション活動が財務指標や市場パフォーマンスに与える影響を
定量的に分析するモジュール。

主要機能:
1. イノベーション指標の算出・評価
2. イノベーション活動と財務パフォーマンスの因果関係分析
3. 市場破壊的イノベーションの検出・予測
4. イノベーション投資効果の時系列分析
5. 競合他社への波及効果分析
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from scipy import stats
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

@dataclass
class InnovationMetrics:
    """イノベーション指標データクラス"""
    rd_intensity: float  # 研究開発費率
    patent_count: int    # 特許出願数
    rd_productivity: float  # R&D生産性 (売上高/R&D投資)
    innovation_pipeline: float  # イノベーションパイプライン指標
    disruptive_potential: float  # 破壊的イノベーション潜在力
    technology_diversity: float  # 技術多様性指数
    collaboration_index: float  # 産学連携・協業指数

@dataclass
class ImpactResult:
    """イノベーション影響分析結果"""
    short_term_impact: Dict[str, float]    # 短期影響 (1-2年)
    medium_term_impact: Dict[str, float]   # 中期影響 (3-5年)
    long_term_impact: Dict[str, float]     # 長期影響 (5-10年)
    spillover_effect: Dict[str, float]     # 波及効果
    disruption_probability: float          # 市場破壊確率
    innovation_efficiency: float           # イノベーション効率
    risk_adjusted_return: float            # リスク調整後収益

class InnovationImpactAnalyzer:
    """イノベーション影響分析クラス"""
    
    def __init__(self, 
                    config: Optional[Dict] = None,
                    innovation_lag_periods: int = 5,
                    spillover_threshold: float = 0.15):
        """
        初期化
        
        Parameters:
        -----------
        config : Dict, optional
            設定パラメータ
        innovation_lag_periods : int
            イノベーション効果の遅延期間（年）
        spillover_threshold : float
            波及効果の閾値
        """
        self.config = config or {}
        self.innovation_lag_periods = innovation_lag_periods
        self.spillover_threshold = spillover_threshold
        self.scaler = StandardScaler()
        
        # イノベーション指標の重要度重み
        self.innovation_weights = {
            'rd_intensity': 0.25,
            'patent_count': 0.20,
            'rd_productivity': 0.20,
            'innovation_pipeline': 0.15,
            'disruptive_potential': 0.10,
            'technology_diversity': 0.05,
            'collaboration_index': 0.05
        }
        
        # 評価指標マッピング
        self.evaluation_metrics = [
            'revenue_growth_rate',
            'operating_margin',
            'market_share',
            'roe',
            'value_added_rate',
            'survival_probability'
        ]
    
    def calculate_innovation_metrics(self, 
                                    company_data: pd.DataFrame) -> InnovationMetrics:
        """
        イノベーション指標を算出
        
        Parameters:
        -----------
        company_data : pd.DataFrame
            企業の財務・特許データ
            
        Returns:
        --------
        InnovationMetrics
            算出されたイノベーション指標
        """
        try:
            # 基本指標の算出
            revenue = company_data['revenue'].iloc[-1]
            rd_expense = company_data.get('rd_expense', [0]).iloc[-1]
            
            # 1. 研究開発費率
            rd_intensity = rd_expense / revenue if revenue > 0 else 0
            
            # 2. 特許出願数（直近3年平均）
            patent_cols = [col for col in company_data.columns if 'patent' in col.lower()]
            patent_count = 0
            if patent_cols:
                recent_patents = company_data[patent_cols].iloc[-3:].mean().sum()
                patent_count = int(recent_patents) if not np.isnan(recent_patents) else 0
            
            # 3. R&D生産性
            rd_productivity = revenue / rd_expense if rd_expense > 0 else 0
            
            # 4. イノベーションパイプライン指標
            innovation_pipeline = self._calculate_innovation_pipeline(company_data)
            
            # 5. 破壊的イノベーション潜在力
            disruptive_potential = self._calculate_disruptive_potential(company_data)
            
            # 6. 技術多様性指数
            technology_diversity = self._calculate_technology_diversity(company_data)
            
            # 7. 産学連携・協業指数
            collaboration_index = self._calculate_collaboration_index(company_data)
            
            return InnovationMetrics(
                rd_intensity=rd_intensity,
                patent_count=patent_count,
                rd_productivity=rd_productivity,
                innovation_pipeline=innovation_pipeline,
                disruptive_potential=disruptive_potential,
                technology_diversity=technology_diversity,
                collaboration_index=collaboration_index
            )
            
        except Exception as e:
            print(f"イノベーション指標算出エラー: {e}")
            return InnovationMetrics(0, 0, 0, 0, 0, 0, 0)
    
    def _calculate_innovation_pipeline(self, data: pd.DataFrame) -> float:
        """イノベーションパイプライン指標算出"""
        try:
            # 特許出願推移から算出
            patent_cols = [col for col in data.columns if 'patent' in col.lower()]
            if len(patent_cols) < 3:
                return 0.0
            
            recent_data = data[patent_cols].iloc[-5:].fillna(0)
            if len(recent_data) < 2:
                return 0.0
                
            # 特許出願の増加トレンド
            pipeline_score = 0.0
            for col in patent_cols:
                if len(recent_data[col]) > 1:
                    growth = np.polyfit(range(len(recent_data[col])), recent_data[col], 1)[0]
                    pipeline_score += max(0, growth) / len(patent_cols)
            
            # 正規化 (0-1の範囲)
            return min(1.0, max(0.0, pipeline_score / 10))
            
        except:
            return 0.0
    
    def _calculate_disruptive_potential(self, data: pd.DataFrame) -> float:
        """破壊的イノベーション潜在力算出"""
        try:
            disruptive_score = 0.0
            
            # 1. 急激な研究開発費増加
            rd_data = data.get('rd_expense', pd.Series([0]))
            if len(rd_data) >= 3:
                rd_growth = rd_data.pct_change().iloc[-3:].mean()
                disruptive_score += min(0.3, max(0, rd_growth))
            
            # 2. 売上高成長率の加速度
            revenue_data = data.get('revenue', pd.Series([0]))
            if len(revenue_data) >= 4:
                growth_rates = revenue_data.pct_change().iloc[-3:]
                acceleration = growth_rates.diff().mean()
                disruptive_score += min(0.3, max(0, acceleration * 10))
            
            # 3. 新規事業セグメント比率
            segment_data = data.get('new_segment_ratio', pd.Series([0]))
            if not segment_data.empty:
                new_segment_ratio = segment_data.iloc[-1]
                disruptive_score += min(0.4, new_segment_ratio)
            
            return min(1.0, disruptive_score)
            
        except:
            return 0.0
    
    def _calculate_technology_diversity(self, data: pd.DataFrame) -> float:
        """技術多様性指数算出"""
        try:
            # 技術分野の多様性を特許分類から推定
            diversity_factors = []
            
            # 1. 事業セグメント数
            segment_count = data.get('business_segments', pd.Series([1])).iloc[-1]
            diversity_factors.append(min(1.0, segment_count / 10))
            
            # 2. 無形固定資産の多様性（推定）
            intangible_ratio = data.get('intangible_assets_ratio', pd.Series([0])).iloc[-1]
            diversity_factors.append(min(1.0, intangible_ratio * 2))
            
            # 3. 海外売上比率（グローバル技術展開の指標）
            overseas_ratio = data.get('overseas_sales_ratio', pd.Series([0])).iloc[-1]
            diversity_factors.append(min(1.0, overseas_ratio))
            
            return np.mean(diversity_factors) if diversity_factors else 0.0
            
        except:
            return 0.0
    
    def _calculate_collaboration_index(self, data: pd.DataFrame) -> float:
        """産学連携・協業指数算出"""
        try:
            collaboration_score = 0.0
            
            # 1. 研究開発費の外部委託比率（推定）
            rd_total = data.get('rd_expense', pd.Series([0])).iloc[-1]
            external_rd = data.get('external_rd_expense', pd.Series([0])).iloc[-1]
            if rd_total > 0:
                collaboration_score += min(0.5, external_rd / rd_total)
            
            # 2. 投資有価証券比率（戦略的投資の指標）
            investment_ratio = data.get('investment_securities_ratio', pd.Series([0])).iloc[-1]
            collaboration_score += min(0.3, investment_ratio)
            
            # 3. 営業外収益率（技術ライセンス等の指標）
            non_operating_income_ratio = data.get('non_operating_income_ratio', pd.Series([0])).iloc[-1]
            collaboration_score += min(0.2, non_operating_income_ratio * 5)
            
            return min(1.0, collaboration_score)
            
        except:
            return 0.0
    
    def analyze_innovation_impact(self, 
                                company_data: pd.DataFrame,
                                market_data: Optional[pd.DataFrame] = None) -> ImpactResult:
        """
        イノベーション影響を包括的に分析
        
        Parameters:
        -----------
        company_data : pd.DataFrame
            企業データ
        market_data : pd.DataFrame, optional
            市場・競合データ
            
        Returns:
        --------
        ImpactResult
            分析結果
        """
        # イノベーション指標算出
        innovation_metrics = self.calculate_innovation_metrics(company_data)
        
        # 時系列影響分析
        short_term = self._analyze_short_term_impact(company_data, innovation_metrics)
        medium_term = self._analyze_medium_term_impact(company_data, innovation_metrics)
        long_term = self._analyze_long_term_impact(company_data, innovation_metrics)
        
        # 波及効果分析
        spillover_effect = self._analyze_spillover_effect(company_data, market_data, innovation_metrics)
        
        # 市場破壊確率算出
        disruption_prob = self._calculate_disruption_probability(innovation_metrics)
        
        # イノベーション効率算出
        innovation_efficiency = self._calculate_innovation_efficiency(company_data, innovation_metrics)
        
        # リスク調整後収益算出
        risk_adjusted_return = self._calculate_risk_adjusted_return(company_data, innovation_metrics)
        
        return ImpactResult(
            short_term_impact=short_term,
            medium_term_impact=medium_term,
            long_term_impact=long_term,
            spillover_effect=spillover_effect,
            disruption_probability=disruption_prob,
            innovation_efficiency=innovation_efficiency,
            risk_adjusted_return=risk_adjusted_return
        )
    
    def _analyze_short_term_impact(self, 
                                    company_data: pd.DataFrame,
                                    innovation_metrics: InnovationMetrics) -> Dict[str, float]:
        """短期影響分析 (1-2年)"""
        impacts = {}
        
        try:
            # 研究開発投資の即効性効果
            rd_immediate_effect = innovation_metrics.rd_intensity * 0.1
            
            for metric in self.evaluation_metrics:
                if metric in company_data.columns:
                    recent_data = company_data[metric].iloc[-2:]
                    if len(recent_data) >= 2:
                        base_impact = recent_data.iloc[-1] - recent_data.iloc[-2]
                        
                        # イノベーション調整
                        innovation_adjustment = (
                            innovation_metrics.rd_productivity * 0.05 +
                            innovation_metrics.disruptive_potential * 0.03 +
                            rd_immediate_effect
                        )
                        
                        impacts[metric] = base_impact * (1 + innovation_adjustment)
                    else:
                        impacts[metric] = 0.0
                else:
                    impacts[metric] = 0.0
            
        except Exception as e:
            print(f"短期影響分析エラー: {e}")
            impacts = {metric: 0.0 for metric in self.evaluation_metrics}
        
        return impacts
    
    def _analyze_medium_term_impact(self, 
                                    company_data: pd.DataFrame,
                                    innovation_metrics: InnovationMetrics) -> Dict[str, float]:
        """中期影響分析 (3-5年)"""
        impacts = {}
        
        try:
            # イノベーションの成熟効果
            innovation_maturity = (
                innovation_metrics.patent_count / 100 * 0.1 +
                innovation_metrics.innovation_pipeline * 0.2 +
                innovation_metrics.technology_diversity * 0.15
            )
            
            for metric in self.evaluation_metrics:
                if metric in company_data.columns:
                    # 過去5年のトレンド分析
                    historical_data = company_data[metric].iloc[-5:]
                    if len(historical_data) >= 3:
                        # 線形トレンド抽出
                        x = np.arange(len(historical_data))
                        y = historical_data.values
                        trend_slope, _ = np.polyfit(x, y, 1)
                        
                        # イノベーション増幅効果
                        amplification = 1 + innovation_maturity
                        impacts[metric] = trend_slope * amplification * 3  # 3年間の予測
                    else:
                        impacts[metric] = 0.0
                else:
                    impacts[metric] = 0.0
                    
        except Exception as e:
            print(f"中期影響分析エラー: {e}")
            impacts = {metric: 0.0 for metric in self.evaluation_metrics}
        
        return impacts
    
    def _analyze_long_term_impact(self, 
                                company_data: pd.DataFrame,
                                innovation_metrics: InnovationMetrics) -> Dict[str, float]:
        """長期影響分析 (5-10年)"""
        impacts = {}
        
        try:
            # 長期的イノベーション効果
            long_term_multiplier = (
                innovation_metrics.disruptive_potential * 0.5 +
                innovation_metrics.collaboration_index * 0.3 +
                innovation_metrics.technology_diversity * 0.2
            )
            
            for metric in self.evaluation_metrics:
                if metric in company_data.columns:
                    # 複合年間成長率（CAGR）ベース予測
                    data_series = company_data[metric].iloc[-10:]  # 過去10年
                    if len(data_series) >= 5:
                        # CAGRによる長期トレンド
                        start_val = data_series.iloc[0]
                        end_val = data_series.iloc[-1]
                        years = len(data_series) - 1
                        
                        if start_val > 0:
                            cagr = (end_val / start_val) ** (1/years) - 1
                            
                            # イノベーション効果による長期増幅
                            enhanced_cagr = cagr * (1 + long_term_multiplier)
                            
                            # 7年後の予測値
                            future_value = end_val * ((1 + enhanced_cagr) ** 7)
                            impacts[metric] = future_value - end_val
                        else:
                            impacts[metric] = 0.0
                    else:
                        impacts[metric] = 0.0
                else:
                    impacts[metric] = 0.0
                    
        except Exception as e:
            print(f"長期影響分析エラー: {e}")
            impacts = {metric: 0.0 for metric in self.evaluation_metrics}
        
        return impacts
    
    def _analyze_spillover_effect(self, 
                                company_data: pd.DataFrame,
                                market_data: Optional[pd.DataFrame],
                                innovation_metrics: InnovationMetrics) -> Dict[str, float]:
        """波及効果分析"""
        spillover = {}
        
        try:
            # 基本的な波及効果スコア
            spillover_intensity = (
                innovation_metrics.disruptive_potential * 0.4 +
                innovation_metrics.patent_count / 1000 * 0.3 +
                innovation_metrics.collaboration_index * 0.3
            )
            
            spillover['industry_average_impact'] = spillover_intensity * 0.15
            spillover['supplier_impact'] = spillover_intensity * 0.10
            spillover['customer_impact'] = spillover_intensity * 0.12
            spillover['competitor_pressure'] = spillover_intensity * 0.20
            spillover['technology_diffusion'] = spillover_intensity * 0.18
            
            # 市場データがある場合の詳細分析
            if market_data is not None:
                market_volatility = market_data.std().mean() if not market_data.empty else 0
                spillover['market_volatility_impact'] = spillover_intensity * market_volatility * 0.25
            else:
                spillover['market_volatility_impact'] = 0.0
                
        except Exception as e:
            print(f"波及効果分析エラー: {e}")
            spillover = {
                'industry_average_impact': 0.0,
                'supplier_impact': 0.0,
                'customer_impact': 0.0,
                'competitor_pressure': 0.0,
                'technology_diffusion': 0.0,
                'market_volatility_impact': 0.0
            }
        
        return spillover
    
    def _calculate_disruption_probability(self, 
                                        innovation_metrics: InnovationMetrics) -> float:
        """市場破壊確率算出"""
        try:
            # 破壊的イノベーション因子の重み付き合計
            disruption_factors = [
                innovation_metrics.disruptive_potential * 0.35,
                min(1.0, innovation_metrics.rd_intensity * 10) * 0.25,
                min(1.0, innovation_metrics.patent_count / 500) * 0.15,
                innovation_metrics.technology_diversity * 0.15,
                innovation_metrics.innovation_pipeline * 0.10
            ]
            
            # ロジスティック関数で確率に変換
            disruption_score = sum(disruption_factors)
            probability = 1 / (1 + np.exp(-5 * (disruption_score - 0.5)))
            
            return min(0.95, max(0.05, probability))  # 5%-95%の範囲に制限
            
        except:
            return 0.1  # デフォルト値
    
    def _calculate_innovation_efficiency(self, 
                                        company_data: pd.DataFrame,
                                        innovation_metrics: InnovationMetrics) -> float:
        """イノベーション効率算出"""
        try:
            # 投入資源に対する成果の比率
            if innovation_metrics.rd_intensity > 0:
                efficiency_score = (
                    innovation_metrics.rd_productivity * 0.4 +
                    innovation_metrics.patent_count / innovation_metrics.rd_intensity / 1000 * 0.3 +
                    innovation_metrics.innovation_pipeline * 0.3
                )
                return min(10.0, max(0.1, efficiency_score))
            else:
                return 1.0
                
        except:
            return 1.0
    
    def _calculate_risk_adjusted_return(self, 
                                        company_data: pd.DataFrame,
                                        innovation_metrics: InnovationMetrics) -> float:
        """リスク調整後収益算出"""
        try:
            # ROEデータがある場合
            if 'roe' in company_data.columns:
                roe_data = company_data['roe'].iloc[-5:]
                if len(roe_data) >= 3:
                    mean_roe = roe_data.mean()
                    roe_volatility = roe_data.std()
                    
                    # イノベーション効果による収益向上
                    innovation_premium = (
                        innovation_metrics.rd_productivity * 0.001 +
                        innovation_metrics.disruptive_potential * 0.02 +
                        innovation_metrics.patent_count / 10000
                    )
                    
                    # リスク調整後収益
                    if roe_volatility > 0:
                        adjusted_return = (mean_roe + innovation_premium) / roe_volatility
                        return max(-1.0, min(5.0, adjusted_return))
                    else:
                        return mean_roe + innovation_premium
                else:
                    return 0.0
            else:
                return 0.0
                
        except:
            return 0.0
    
    def generate_innovation_report(self, 
                                    company_name: str,
                                    impact_result: ImpactResult) -> Dict[str, any]:
        """イノベーション影響分析レポート生成"""
        report = {
            'company_name': company_name,
            'analysis_timestamp': pd.Timestamp.now(),
            'executive_summary': {
                'disruption_probability': impact_result.disruption_probability,
                'innovation_efficiency': impact_result.innovation_efficiency,
                'risk_adjusted_return': impact_result.risk_adjusted_return
            },
            'time_horizon_impacts': {
                'short_term': impact_result.short_term_impact,
                'medium_term': impact_result.medium_term_impact,
                'long_term': impact_result.long_term_impact
            },
            'spillover_effects': impact_result.spillover_effect,
            'recommendations': self._generate_recommendations(impact_result)
        }
        
        return report
    
    def _generate_recommendations(self, impact_result: ImpactResult) -> List[str]:
        """戦略提言生成"""
        recommendations = []
        
        # 破壊的イノベーション潜在力に基づく提言
        if impact_result.disruption_probability > 0.7:
            recommendations.append("高い市場破壊潜在力を保持。積極的な研究開発投資継続を推奨")
        elif impact_result.disruption_probability > 0.4:
            recommendations.append("中程度の破壊力。特定分野への集中投資で差別化を図る")
        else:
            recommendations.append("破壊力が限定的。既存事業強化と新技術獲得戦略の検討が必要")
        
        # イノベーション効率に基づく提言
        if impact_result.innovation_efficiency > 3.0:
            recommendations.append("優秀なイノベーション効率。現在の R&D 戦略を維持・拡大")
        elif impact_result.innovation_efficiency < 1.0:
            recommendations.append("イノベーション効率改善が急務。研究開発プロセスの見直しが必要")
        
        # リスク調整後収益に基づく提言
        if impact_result.risk_adjusted_return > 1.0:
            recommendations.append("良好なリスク調整後収益。投資家への訴求力強化を推奨")
        elif impact_result.risk_adjusted_return < 0:
            recommendations.append("収益性改善が必要。イノベーション投資の選択と集中を検討")
        
        return recommendations

    def compare_innovation_impact(self, 
                                companies_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """複数企業のイノベーション影響比較分析"""
        comparison_results = []
        
        for company_name, company_data in companies_data.items():
            try:
                impact_result = self.analyze_innovation_impact(company_data)
                
                result_row = {
                    'company': company_name,
                    'disruption_probability': impact_result.disruption_probability,
                    'innovation_efficiency': impact_result.innovation_efficiency,
                    'risk_adjusted_return': impact_result.risk_adjusted_return,
                    'short_term_revenue_impact': impact_result.short_term_impact.get('revenue_growth_rate', 0),
                    'medium_term_market_share_impact': impact_result.medium_term_impact.get('market_share', 0),
                    'long_term_roe_impact': impact_result.long_term_impact.get('roe', 0)
                }
                
                comparison_results.append(result_row)
                
            except Exception as e:
                print(f"企業 {company_name} の分析エラー: {e}")
        
        return pd.DataFrame(comparison_results).sort_values('disruption_probability', ascending=False)