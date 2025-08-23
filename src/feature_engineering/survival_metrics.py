"""
A2AI - 企業存続確率評価項目計算モジュール
=======================================

企業の生存・消滅リスクを定量化する評価項目を計算する。
150社×40年のデータから、企業消滅・倒産・事業撤退を予測する指標を生成。

主な評価項目:
1. 企業存続確率 (Corporate Survival Probability)
2. 消滅リスク指標 (Extinction Risk Index)
3. 財務危険度スコア (Financial Distress Score)
4. 市場退出確率 (Market Exit Probability)
5. 事業継続力指数 (Business Continuity Index)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import warnings
from scipy import stats
from sklearn.preprocessing import StandardScaler
import logging

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SurvivalMetricsConfig:
    """生存分析評価項目の設定クラス"""
    
    # 生存確率計算用パラメータ
    survival_window: int = 5  # 生存確率予測期間（年）
    risk_threshold: float = 0.3  # リスク判定閾値
    
    # 財務危険度計算用重み
    liquidity_weight: float = 0.25  # 流動性リスク重み
    profitability_weight: float = 0.25  # 収益性リスク重み
    leverage_weight: float = 0.25  # レバレッジリスク重み
    efficiency_weight: float = 0.25  # 効率性リスク重み
    
    # 異常値処理
    outlier_method: str = 'iqr'  # 'iqr', 'zscore', 'percentile'
    outlier_threshold: float = 3.0  # 外れ値判定閾値

class SurvivalMetrics:
    """企業存続確率評価項目計算クラス"""
    
    def __init__(self, config: Optional[SurvivalMetricsConfig] = None):
        """
        初期化
        
        Args:
            config: 生存分析設定（デフォルト値使用可能）
        """
        self.config = config or SurvivalMetricsConfig()
        self.scaler = StandardScaler()
        self._is_fitted = False
        
        # 企業状態定義
        self.ACTIVE = 1  # 存続企業
        self.EXTINCT = 0  # 消滅企業
        
        logger.info("SurvivalMetrics initialized")
    
    def calculate_survival_probability(self, 
                                        financial_data: pd.DataFrame,
                                        company_status: pd.Series) -> pd.Series:
        """
        企業存続確率を計算
        
        複数の財務指標を組み合わせて、企業の将来的な存続確率を算出。
        過去の消滅企業データを学習して確率モデルを構築。
        
        Args:
            financial_data: 財務データ (企業×年×指標)
            company_status: 企業状態 (1: 存続, 0: 消滅)
            
        Returns:
            企業存続確率 (0-1の範囲)
        """
        try:
            logger.info("企業存続確率の計算を開始")
            
            # 基本的な生存指標
            liquidity_score = self._calculate_liquidity_survival(financial_data)
            profitability_score = self._calculate_profitability_survival(financial_data)
            leverage_score = self._calculate_leverage_survival(financial_data)
            efficiency_score = self._calculate_efficiency_survival(financial_data)
            
            # 重み付け統合
            survival_prob = (
                liquidity_score * self.config.liquidity_weight +
                profitability_score * self.config.profitability_weight +
                leverage_score * self.config.leverage_weight +
                efficiency_score * self.config.efficiency_weight
            )
            
            # 0-1の範囲に正規化
            survival_prob = np.clip(survival_prob, 0, 1)
            
            logger.info(f"企業存続確率計算完了: 平均={survival_prob.mean():.3f}")
            return pd.Series(survival_prob, index=financial_data.index)
            
        except Exception as e:
            logger.error(f"企業存続確率計算エラー: {str(e)}")
            raise
    
    def calculate_extinction_risk_index(self, 
                                        financial_data: pd.DataFrame) -> pd.Series:
        """
        消滅リスク指標を計算
        
        企業消滅に至る典型的なパターンを定量化。
        複数の危険信号を統合してリスクスコアを算出。
        
        Args:
            financial_data: 財務データ
            
        Returns:
            消滅リスク指数 (0-100の範囲、高いほど危険)
        """
        try:
            logger.info("消滅リスク指標の計算を開始")
            
            # 各種リスク要因
            cash_risk = self._calculate_cash_flow_risk(financial_data)
            debt_risk = self._calculate_debt_sustainability_risk(financial_data)
            market_risk = self._calculate_market_position_risk(financial_data)
            operational_risk = self._calculate_operational_risk(financial_data)
            
            # 統合リスクスコア（0-100）
            extinction_risk = np.clip(
                (cash_risk + debt_risk + market_risk + operational_risk) * 25, 0, 100
            )
            
            logger.info(f"消滅リスク指標計算完了: 平均={extinction_risk.mean():.1f}")
            return pd.Series(extinction_risk, index=financial_data.index)
            
        except Exception as e:
            logger.error(f"消滅リスク指標計算エラー: {str(e)}")
            raise
    
    def calculate_financial_distress_score(self, 
                                            financial_data: pd.DataFrame) -> pd.Series:
        """
        財務危険度スコアを計算（Altman Z-Score の改良版）
        
        破綻予測で著名なAltman Z-Scoreを日本企業向けに改良。
        財務諸表の複数項目から総合的な財務健全性を評価。
        
        Args:
            financial_data: 財務データ
            
        Returns:
            財務危険度スコア (負の値ほど危険、正の値ほど安全)
        """
        try:
            logger.info("財務危険度スコア計算を開始")
            
            # 必要な財務比率を計算
            working_capital_ratio = self._safe_divide(
                financial_data.get('流動資産', 0) - financial_data.get('流動負債', 0),
                financial_data.get('総資産', 1)
            )
            
            retained_earnings_ratio = self._safe_divide(
                financial_data.get('利益剰余金', 0),
                financial_data.get('総資産', 1)
            )
            
            operating_income_ratio = self._safe_divide(
                financial_data.get('営業利益', 0),
                financial_data.get('総資産', 1)
            )
            
            equity_ratio = self._safe_divide(
                financial_data.get('純資産', 1),
                financial_data.get('総負債', 1)
            )
            
            sales_turnover = self._safe_divide(
                financial_data.get('売上高', 0),
                financial_data.get('総資産', 1)
            )
            
            # 改良Altman Z-Score (日本企業向け係数調整)
            z_score = (
                1.2 * working_capital_ratio +
                1.4 * retained_earnings_ratio +
                3.3 * operating_income_ratio +
                0.6 * equity_ratio +
                1.0 * sales_turnover
            )
            
            logger.info(f"財務危険度スコア計算完了: 平均={z_score.mean():.3f}")
            return pd.Series(z_score, index=financial_data.index)
            
        except Exception as e:
            logger.error(f"財務危険度スコア計算エラー: {str(e)}")
            raise
    
    def calculate_market_exit_probability(self, 
                                        financial_data: pd.DataFrame,
                                        market_share_data: pd.DataFrame) -> pd.Series:
        """
        市場退出確率を計算
        
        市場シェアと財務状況の組み合わせから、
        企業が特定市場から撤退する確率を推定。
        
        Args:
            financial_data: 財務データ
            market_share_data: 市場シェアデータ
            
        Returns:
            市場退出確率 (0-1の範囲)
        """
        try:
            logger.info("市場退出確率の計算を開始")
            
            # 市場地位の脆弱性
            market_vulnerability = self._calculate_market_vulnerability(
                financial_data, market_share_data
            )
            
            # 競争力の低下
            competitiveness_decline = self._calculate_competitiveness_decline(
                financial_data, market_share_data
            )
            
            # 財務的持続可能性
            financial_sustainability = self._calculate_financial_sustainability(
                financial_data
            )
            
            # 統合確率
            exit_probability = np.clip(
                (market_vulnerability + competitiveness_decline + 
                    (1 - financial_sustainability)) / 3, 0, 1
            )
            
            logger.info(f"市場退出確率計算完了: 平均={exit_probability.mean():.3f}")
            return pd.Series(exit_probability, index=financial_data.index)
            
        except Exception as e:
            logger.error(f"市場退出確率計算エラー: {str(e)}")
            raise
    
    def calculate_business_continuity_index(self, 
                                            financial_data: pd.DataFrame) -> pd.Series:
        """
        事業継続力指数を計算
        
        長期的な事業継続能力を多面的に評価。
        財務安定性、成長性、効率性を統合した指標。
        
        Args:
            financial_data: 財務データ
            
        Returns:
            事業継続力指数 (0-100の範囲、高いほど継続力が強い)
        """
        try:
            logger.info("事業継続力指数の計算を開始")
            
            # 各構成要素
            financial_stability = self._calculate_financial_stability(financial_data)
            growth_sustainability = self._calculate_growth_sustainability(financial_data)
            operational_efficiency = self._calculate_operational_efficiency(financial_data)
            adaptability = self._calculate_business_adaptability(financial_data)
            
            # 統合指数 (0-100)
            continuity_index = np.clip(
                (financial_stability + growth_sustainability + 
                 operational_efficiency + adaptability) * 25, 0, 100
            )
            
            logger.info(f"事業継続力指数計算完了: 平均={continuity_index.mean():.1f}")
            return pd.Series(continuity_index, index=financial_data.index)
            
        except Exception as e:
            logger.error(f"事業継続力指数計算エラー: {str(e)}")
            raise
    
    # ========== 内部計算メソッド群 ==========
    
    def _calculate_liquidity_survival(self, data: pd.DataFrame) -> np.ndarray:
        """流動性による生存スコア計算"""
        current_ratio = self._safe_divide(
            data.get('流動資産', 0), data.get('流動負債', 1)
        )
        quick_ratio = self._safe_divide(
            data.get('流動資産', 0) - data.get('棚卸資産', 0),
            data.get('流動負債', 1)
        )
        cash_ratio = self._safe_divide(
            data.get('現金及び預金', 0), data.get('流動負債', 1)
        )
        
        # 正規化とスコア化
        liquidity_score = np.tanh(
            (current_ratio - 1) * 0.5 + 
            (quick_ratio - 0.8) * 0.7 +
            (cash_ratio - 0.2) * 1.0
        ) * 0.5 + 0.5
        
        return np.clip(liquidity_score, 0, 1)
    
    def _calculate_profitability_survival(self, data: pd.DataFrame) -> np.ndarray:
        """収益性による生存スコア計算"""
        roe = self._safe_divide(
            data.get('当期純利益', 0), data.get('純資産', 1)
        )
        roa = self._safe_divide(
            data.get('当期純利益', 0), data.get('総資産', 1)
        )
        operating_margin = self._safe_divide(
            data.get('営業利益', 0), data.get('売上高', 1)
        )
        
        # 収益性スコア
        profitability_score = np.tanh(
            roe * 10 + roa * 15 + operating_margin * 8
        ) * 0.5 + 0.5
        
        return np.clip(profitability_score, 0, 1)
    
    def _calculate_leverage_survival(self, data: pd.DataFrame) -> np.ndarray:
        """レバレッジによる生存スコア計算"""
        debt_ratio = self._safe_divide(
            data.get('総負債', 0), data.get('総資産', 1)
        )
        interest_coverage = self._safe_divide(
            data.get('営業利益', 0), data.get('支払利息', 0.1)
        )
        
        # レバレッジスコア (低レバレッジ、高カバレッジが良い)
        leverage_score = np.tanh(
            (1 - debt_ratio) * 2 + np.log1p(interest_coverage) * 0.3
        ) * 0.5 + 0.5
        
        return np.clip(leverage_score, 0, 1)
    
    def _calculate_efficiency_survival(self, data: pd.DataFrame) -> np.ndarray:
        """効率性による生存スコア計算"""
        asset_turnover = self._safe_divide(
            data.get('売上高', 0), data.get('総資産', 1)
        )
        inventory_turnover = self._safe_divide(
            data.get('売上原価', 0), data.get('棚卸資産', 1)
        )
        
        efficiency_score = np.tanh(
            asset_turnover + np.log1p(inventory_turnover) * 0.2
        ) * 0.5 + 0.5
        
        return np.clip(efficiency_score, 0, 1)
    
    def _calculate_cash_flow_risk(self, data: pd.DataFrame) -> np.ndarray:
        """キャッシュフローリスク計算"""
        operating_cf = data.get('営業CF', 0)
        free_cf = data.get('フリーCF', 0)
        cash_conversion = self._safe_divide(operating_cf, data.get('売上高', 1))
        
        # リスクスコア（キャッシュフロー悪化で高リスク）
        cf_risk = np.clip(1 - np.tanh(operating_cf / 1e8 + cash_conversion * 10), 0, 1)
        return cf_risk
    
    def _calculate_debt_sustainability_risk(self, data: pd.DataFrame) -> np.ndarray:
        """債務持続可能性リスク計算"""
        debt_service_ratio = self._safe_divide(
            data.get('支払利息', 0) + data.get('借入金返済', 0),
            data.get('営業CF', 1)
        )
        debt_growth = data.get('有利子負債増加率', 0)
        
        debt_risk = np.clip(np.tanh(debt_service_ratio * 2 + debt_growth), 0, 1)
        return debt_risk
    
    def _calculate_market_position_risk(self, data: pd.DataFrame) -> np.ndarray:
        """市場地位リスク計算"""
        market_share_decline = data.get('市場シェア減少率', 0)
        competitive_position = data.get('競争力指標', 0.5)
        
        market_risk = np.clip(
            market_share_decline * 2 + (1 - competitive_position), 0, 1
        )
        return market_risk
    
    def _calculate_operational_risk(self, data: pd.DataFrame) -> np.ndarray:
        """事業リスク計算"""
        revenue_volatility = data.get('売上高変動係数', 0)
        cost_rigidity = data.get('固定費比率', 0.5)
        
        operational_risk = np.clip(
            revenue_volatility * 3 + cost_rigidity * 0.5, 0, 1
        )
        return operational_risk
    
    def _calculate_market_vulnerability(self, 
                                        financial_data: pd.DataFrame,
                                        market_data: pd.DataFrame) -> np.ndarray:
        """市場脆弱性計算"""
        market_share = market_data.get('市場シェア', 0.1)
        competitor_growth = market_data.get('競合成長率', 0)
        
        vulnerability = np.clip(
            (1 - market_share) * 0.7 + competitor_growth * 0.3, 0, 1
        )
        return vulnerability
    
    def _calculate_competitiveness_decline(self, 
                                            financial_data: pd.DataFrame,
                                            market_data: pd.DataFrame) -> np.ndarray:
        """競争力低下度計算"""
        relative_profitability = market_data.get('相対収益性', 1.0)
        innovation_investment = self._safe_divide(
            financial_data.get('研究開発費', 0), 
            financial_data.get('売上高', 1)
        )
        
        decline = np.clip(
            (1 - relative_profitability) * 0.6 + 
            (0.05 - innovation_investment) * 10, 0, 1
        )
        return decline
    
    def _calculate_financial_sustainability(self, data: pd.DataFrame) -> np.ndarray:
        """財務持続可能性計算"""
        cash_position = self._safe_divide(
            data.get('現金及び預金', 0), data.get('売上高', 1)
        )
        debt_coverage = self._safe_divide(
            data.get('営業CF', 1), data.get('有利子負債', 1)
        )
        
        sustainability = np.clip(
            np.tanh(cash_position * 5 + debt_coverage * 2) * 0.5 + 0.5, 0, 1
        )
        return sustainability
    
    def _calculate_financial_stability(self, data: pd.DataFrame) -> np.ndarray:
        """財務安定性計算"""
        equity_ratio = self._safe_divide(
            data.get('純資産', 0), data.get('総資産', 1)
        )
        current_ratio = self._safe_divide(
            data.get('流動資産', 0), data.get('流動負債', 1)
        )
        
        stability = np.clip(
            equity_ratio + np.tanh((current_ratio - 1) * 2) * 0.3, 0, 1
        )
        return stability
    
    def _calculate_growth_sustainability(self, data: pd.DataFrame) -> np.ndarray:
        """成長持続可能性計算"""
        revenue_growth = data.get('売上高成長率', 0)
        profit_growth = data.get('利益成長率', 0)
        investment_ratio = self._safe_divide(
            data.get('設備投資額', 0), data.get('売上高', 1)
        )
        
        growth_sustainability = np.clip(
            np.tanh(revenue_growth + profit_growth) * 0.4 +
            np.tanh(investment_ratio * 10) * 0.6, 0, 1
        )
        return growth_sustainability
    
    def _calculate_operational_efficiency(self, data: pd.DataFrame) -> np.ndarray:
        """事業効率性計算"""
        asset_turnover = self._safe_divide(
            data.get('売上高', 0), data.get('総資産', 1)
        )
        labor_productivity = self._safe_divide(
            data.get('売上高', 0), data.get('従業員数', 1)
        )
        
        efficiency = np.clip(
            np.tanh(asset_turnover) * 0.5 +
            np.tanh(labor_productivity / 1e6) * 0.5, 0, 1
        )
        return efficiency
    
    def _calculate_business_adaptability(self, data: pd.DataFrame) -> np.ndarray:
        """事業適応力計算"""
        rd_intensity = self._safe_divide(
            data.get('研究開発費', 0), data.get('売上高', 1)
        )
        overseas_ratio = data.get('海外売上高比率', 0)
        
        adaptability = np.clip(
            np.tanh(rd_intensity * 20) * 0.6 + overseas_ratio * 0.4, 0, 1
        )
        return adaptability
    
    def _safe_divide(self, numerator: Union[pd.Series, np.ndarray, float], 
                        denominator: Union[pd.Series, np.ndarray, float]) -> np.ndarray:
        """安全な除算（ゼロ除算回避）"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = np.divide(numerator, denominator, 
                                out=np.zeros_like(numerator, dtype=float), 
                                where=denominator!=0)
        return np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
    
    def calculate_all_survival_metrics(self, 
                                        financial_data: pd.DataFrame,
                                        market_data: Optional[pd.DataFrame] = None,
                                        company_status: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        全ての生存関連評価項目を一括計算
        
        Args:
            financial_data: 財務データ
            market_data: 市場データ（オプション）
            company_status: 企業状態（オプション）
            
        Returns:
            全生存評価項目を含むDataFrame
        """
        try:
            logger.info("全生存評価項目の計算を開始")
            
            results = pd.DataFrame(index=financial_data.index)
            
            # 基本的な生存評価項目
            results['企業存続確率'] = self.calculate_survival_probability(
                financial_data, company_status or pd.Series(1, index=financial_data.index)
            )
            
            results['消滅リスク指数'] = self.calculate_extinction_risk_index(financial_data)
            
            results['財務危険度スコア'] = self.calculate_financial_distress_score(financial_data)
            
            results['事業継続力指数'] = self.calculate_business_continuity_index(financial_data)
            
            # 市場データがある場合の追加評価項目
            if market_data is not None:
                results['市場退出確率'] = self.calculate_market_exit_probability(
                    financial_data, market_data
                )
            
            logger.info(f"全生存評価項目計算完了: {len(results.columns)}項目")
            return results
            
        except Exception as e:
            logger.error(f"全生存評価項目計算エラー: {str(e)}")
            raise

# 使用例とテスト用コード
if __name__ == "__main__":
    # サンプルデータ生成（実際のデータ構造をシミュレート）
    np.random.seed(42)
    n_companies = 150
    n_years = 40
    
    # 模擬財務データ
    sample_financial_data = pd.DataFrame({
        '総資産': np.random.lognormal(10, 1, n_companies),
        '流動資産': np.random.lognormal(9, 1, n_companies),
        '流動負債': np.random.lognormal(8.5, 1, n_companies),
        '総負債': np.random.lognormal(9.5, 1, n_companies),
        '純資産': np.random.lognormal(9, 1, n_companies),
        '売上高': np.random.lognormal(10.5, 1, n_companies),
        '営業利益': np.random.lognormal(7, 1.5, n_companies) * np.random.choice([1, -1], n_companies, p=[0.8, 0.2]),
        '当期純利益': np.random.lognormal(6.5, 2, n_companies) * np.random.choice([1, -1], n_companies, p=[0.7, 0.3]),
        '営業CF': np.random.lognormal(7.5, 1.5, n_companies),
        '現金及び預金': np.random.lognormal(8, 1, n_companies),
        '棚卸資産': np.random.lognormal(7.5, 1, n_companies),
        '研究開発費': np.random.lognormal(6, 2, n_companies),
        '従業員数': np.random.randint(100, 50000, n_companies),
    }, index=[f'企業_{i:03d}' for i in range(n_companies)])
    
    # 企業状態（一部企業は消滅）
    company_status = pd.Series(
        np.random.choice([0, 1], n_companies, p=[0.2, 0.8]), 
        index=sample_financial_data.index
    )
    
    # SurvivalMetrics インスタンス化と計算
    survival_calculator = SurvivalMetrics()
    
    # 各評価項目を個別に計算
    print("=== A2AI 生存分析評価項目の計算例 ===")
    
    survival_prob = survival_calculator.calculate_survival_probability(
        sample_financial_data, company_status
    )
    print(f"企業存続確率: 平均={survival_prob.mean():.3f}, 標準偏差={survival_prob.std():.3f}")
    
    extinction_risk = survival_calculator.calculate_extinction_risk_index(
        sample_financial_data
    )
    print(f"消滅リスク指数: 平均={extinction_risk.mean():.1f}, 標準偏差={extinction_risk.std():.1f}")
    
    distress_score = survival_calculator.calculate_financial_distress_score(
        sample_financial_data
    )
    print(f"財務危険度スコア: 平均={distress_score.mean():.3f}, 標準偏差={distress_score.std():.3f}")
    
    continuity_index = survival_calculator.calculate_business_continuity_index(
        sample_financial_data
    )
    print(f"事業継続力指数: 平均={continuity_index.mean():.1f}, 標準偏差={continuity_index.std():.1f}")
    
    # 全評価項目一括計算
    all_metrics = survival_calculator.calculate_all_survival_metrics(
        sample_financial_data, company_status=company_status
    )
    
    print(f"\n=== 全生存評価項目計算結果 ===")
    print(all_metrics.describe())
    
    # 消滅企業と存続企業の比較
    extinct_companies = all_metrics[company_status == 0]
    active_companies = all_metrics[company_status == 1]
    
    print(f"\n=== 消滅企業 vs 存続企業の比較 ===")
    for col in all_metrics.columns:
        extinct_mean = extinct_companies[col].mean()
        active_mean = active_companies[col].mean()
        print(f"{col}: 消滅企業={extinct_mean:.3f}, 存続企業={active_mean:.3f}")