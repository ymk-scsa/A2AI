"""
A2AI Growth Phase Analyzer
新設企業の成長段階分析モジュール

企業ライフサイクルの各段階（スタートアップ期、成長期、成熟期、衰退期）を
財務指標の変化パターンから自動検出し、段階別の特徴量と成功要因を分析する。
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from scipy import stats
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

@dataclass
class GrowthPhase:
    """成長段階定義クラス"""
    phase_name: str
    duration_years: int
    start_year: int
    end_year: int
    key_characteristics: Dict[str, float]
    transition_triggers: List[str]
    risk_factors: List[str]

@dataclass
class PhaseAnalysisResult:
    """段階分析結果クラス"""
    company_id: str
    company_name: str
    phases_detected: List[GrowthPhase]
    current_phase: str
    phase_transition_history: List[Tuple[str, int]]
    success_probability: float
    risk_score: float
    recommendations: List[str]

class GrowthPhaseAnalyzer:
    """
    新設企業成長段階分析器
    
    財務諸表データから企業の成長段階を自動検出し、
    段階別の特徴分析と将来予測を実行する。
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初期化
        
        Args:
            config: 分析設定パラメータ
        """
        self.config = config or self._get_default_config()
        self.scaler = StandardScaler()
        self.phase_models = {}
        self.transition_models = {}
        
        # 成長段階定義
        self.growth_phases = {
            'startup': {
                'name': 'スタートアップ期',
                'typical_duration': (1, 5),
                'key_indicators': ['revenue_growth', 'employee_growth', 'rd_intensity'],
                'characteristics': {
                    'revenue_volatility': 'high',
                    'profitability': 'negative_to_low',
                    'cash_burn': 'high',
                    'growth_rate': 'very_high'
                }
            },
            'growth': {
                'name': '成長期',
                'typical_duration': (3, 10),
                'key_indicators': ['revenue_growth', 'market_share', 'operational_efficiency'],
                'characteristics': {
                    'revenue_volatility': 'medium',
                    'profitability': 'improving',
                    'cash_generation': 'positive',
                    'growth_rate': 'high'
                }
            },
            'maturity': {
                'name': '成熟期',
                'typical_duration': (5, 20),
                'key_indicators': ['profitability', 'dividend_yield', 'operational_stability'],
                'characteristics': {
                    'revenue_volatility': 'low',
                    'profitability': 'stable_high',
                    'cash_generation': 'strong',
                    'growth_rate': 'moderate'
                }
            },
            'decline': {
                'name': '衰退期',
                'typical_duration': (2, 10),
                'key_indicators': ['revenue_decline', 'margin_compression', 'market_share_loss'],
                'characteristics': {
                    'revenue_volatility': 'high',
                    'profitability': 'declining',
                    'cash_generation': 'weakening',
                    'growth_rate': 'negative'
                }
            }
        }
    
    def _get_default_config(self) -> Dict:
        """デフォルト設定を取得"""
        return {
            'min_years_for_analysis': 3,
            'phase_detection_method': 'hybrid',  # 'kmeans', 'gmm', 'rule_based', 'hybrid'
            'smoothing_window': 3,
            'transition_threshold': 0.7,
            'risk_assessment_window': 5,
            'success_metrics_weight': {
                'revenue_growth': 0.25,
                'profitability': 0.20,
                'market_position': 0.20,
                'operational_efficiency': 0.15,
                'financial_stability': 0.20
            }
        }
    
    def analyze_company_phases(self, 
                                company_data: pd.DataFrame,
                                company_id: str,
                                company_name: str) -> PhaseAnalysisResult:
        """
        単一企業の成長段階分析
        
        Args:
            company_data: 企業の時系列財務データ
            company_id: 企業ID
            company_name: 企業名
        
        Returns:
            PhaseAnalysisResult: 分析結果
        """
        if len(company_data) < self.config['min_years_for_analysis']:
            raise ValueError(f"分析には最低{self.config['min_years_for_analysis']}年のデータが必要です")
        
        # 成長指標の計算
        growth_indicators = self._calculate_growth_indicators(company_data)
        
        # 段階検出
        phases = self._detect_phases(growth_indicators)
        
        # 現在の段階判定
        current_phase = self._determine_current_phase(growth_indicators, phases)
        
        # 段階遷移履歴
        transition_history = self._build_transition_history(phases)
        
        # 成功確率計算
        success_prob = self._calculate_success_probability(growth_indicators, phases)
        
        # リスクスコア計算
        risk_score = self._calculate_risk_score(growth_indicators, current_phase)
        
        # 推奨事項生成
        recommendations = self._generate_recommendations(current_phase, growth_indicators, risk_score)
        
        return PhaseAnalysisResult(
            company_id=company_id,
            company_name=company_name,
            phases_detected=phases,
            current_phase=current_phase,
            phase_transition_history=transition_history,
            success_probability=success_prob,
            risk_score=risk_score,
            recommendations=recommendations
        )
    
    def _calculate_growth_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """成長指標の計算"""
        indicators = pd.DataFrame(index=data.index)
        
        # 売上高関連指標
        indicators['revenue_growth'] = data['売上高'].pct_change()
        indicators['revenue_volatility'] = indicators['revenue_growth'].rolling(3).std()
        indicators['revenue_acceleration'] = indicators['revenue_growth'].diff()
        
        # 利益性指標
        indicators['operating_margin'] = data['売上高営業利益率'] / 100
        indicators['net_margin'] = data['売上高当期純利益率'] / 100
        indicators['margin_trend'] = indicators['operating_margin'].diff()
        
        # 効率性指標
        indicators['asset_turnover'] = data['総資産回転率']
        indicators['roa'] = data['ROE'] / 100
        indicators['efficiency_improvement'] = indicators['asset_turnover'].pct_change()
        
        # 成長投資指標
        indicators['rd_intensity'] = data.get('研究開発費率', 0)
        indicators['capex_intensity'] = data.get('設備投資額', 0) / data['売上高']
        indicators['investment_growth'] = indicators['capex_intensity'].pct_change()
        
        # 人的資源指標
        if '従業員数' in data.columns:
            indicators['employee_growth'] = data['従業員数'].pct_change()
            indicators['revenue_per_employee'] = data['売上高'] / data['従業員数']
            indicators['productivity_growth'] = indicators['revenue_per_employee'].pct_change()
        
        # 財務健全性指標
        indicators['debt_ratio'] = 1 - data.get('自己資本比率', 50) / 100
        indicators['liquidity_ratio'] = data.get('流動比率', 100) / 100
        indicators['financial_stability'] = (indicators['liquidity_ratio'] * (1 - indicators['debt_ratio']))
        
        # 市場地位指標（利用可能な場合）
        if '海外売上高比率' in data.columns:
            indicators['global_expansion'] = data['海外売上高比率'] / 100
            indicators['market_expansion'] = indicators['global_expansion'].diff()
        
        # スムージング処理
        if self.config['smoothing_window'] > 1:
            numeric_cols = indicators.select_dtypes(include=[np.number]).columns
            indicators[numeric_cols] = indicators[numeric_cols].rolling(
                self.config['smoothing_window'], center=True).mean()
        
        return indicators.fillna(method='forward').fillna(method='backward')
    
    def _detect_phases(self, indicators: pd.DataFrame) -> List[GrowthPhase]:
        """成長段階の検出"""
        method = self.config['phase_detection_method']
        
        if method == 'rule_based':
            return self._detect_phases_rule_based(indicators)
        elif method == 'kmeans':
            return self._detect_phases_kmeans(indicators)
        elif method == 'gmm':
            return self._detect_phases_gmm(indicators)
        else:  # hybrid
            return self._detect_phases_hybrid(indicators)
    
    def _detect_phases_rule_based(self, indicators: pd.DataFrame) -> List[GrowthPhase]:
        """ルールベース段階検出"""
        phases = []
        years = indicators.index.tolist()
        
        current_phase = None
        phase_start = None
        
        for i, year in enumerate(years):
            row = indicators.loc[year]
            
            # 段階判定ロジック
            detected_phase = self._classify_phase_by_rules(row)
            
            if detected_phase != current_phase:
                # 段階変化があった場合
                if current_phase is not None and phase_start is not None:
                    # 前の段階を記録
                    phases.append(self._create_growth_phase(
                        current_phase, phase_start, years[i-1], indicators
                    ))
                
                current_phase = detected_phase
                phase_start = year
        
        # 最後の段階を追加
        if current_phase is not None and phase_start is not None:
            phases.append(self._create_growth_phase(
                current_phase, phase_start, years[-1], indicators
            ))
        
        return phases
    
    def _classify_phase_by_rules(self, row: pd.Series) -> str:
        """ルールベース段階分類"""
        # 成長率による一次判定
        revenue_growth = row.get('revenue_growth', 0)
        operating_margin = row.get('operating_margin', 0)
        financial_stability = row.get('financial_stability', 0.5)
        
        # スタートアップ期の判定
        if (revenue_growth > 0.5 or revenue_growth < -0.2) and operating_margin < 0.05 and financial_stability < 0.6:
            return 'startup'
        
        # 衰退期の判定
        if revenue_growth < -0.1 and operating_margin < 0.02:
            return 'decline'
        
        # 成熟期の判定
        if abs(revenue_growth) < 0.15 and operating_margin > 0.08 and financial_stability > 0.7:
            return 'maturity'
        
        # 成長期（デフォルト）
        return 'growth'
    
    def _detect_phases_kmeans(self, indicators: pd.DataFrame) -> List[GrowthPhase]:
        """K-means段階検出"""
        # 主要指標選択
        key_features = ['revenue_growth', 'operating_margin', 'financial_stability', 'rd_intensity']
        available_features = [f for f in key_features if f in indicators.columns]
        
        if len(available_features) < 2:
            return self._detect_phases_rule_based(indicators)
        
        data = indicators[available_features].fillna(0)
        
        # 標準化
        scaled_data = self.scaler.fit_transform(data)
        
        # K-means実行
        kmeans = KMeans(n_clusters=4, random_state=42)
        cluster_labels = kmeans.fit_predict(scaled_data)
        
        # クラスターを段階にマッピング
        return self._map_clusters_to_phases(indicators, cluster_labels, kmeans.cluster_centers_)
    
    def _detect_phases_gmm(self, indicators: pd.DataFrame) -> List[GrowthPhase]:
        """ガウス混合モデル段階検出"""
        key_features = ['revenue_growth', 'operating_margin', 'financial_stability']
        available_features = [f for f in key_features if f in indicators.columns]
        
        if len(available_features) < 2:
            return self._detect_phases_rule_based(indicators)
        
        data = indicators[available_features].fillna(0)
        scaled_data = self.scaler.fit_transform(data)
        
        # GMM実行
        gmm = GaussianMixture(n_components=4, random_state=42)
        cluster_labels = gmm.fit_predict(scaled_data)
        
        return self._map_clusters_to_phases(indicators, cluster_labels, gmm.means_)
    
    def _detect_phases_hybrid(self, indicators: pd.DataFrame) -> List[GrowthPhase]:
        """ハイブリッド段階検出（ルール + ML）"""
        # まずルールベースで大まかな分類
        rule_phases = self._detect_phases_rule_based(indicators)
        
        # MLで細かい調整
        if len(indicators) > 10:  # 十分なデータがある場合
            ml_phases = self._detect_phases_gmm(indicators)
            # ルールとMLの結果を統合
            return self._merge_phase_detections(rule_phases, ml_phases)
        
        return rule_phases
    
    def _create_growth_phase(self, phase_type: str, start_year: int, end_year: int, 
                            indicators: pd.DataFrame) -> GrowthPhase:
        """成長段階オブジェクトの作成"""
        duration = end_year - start_year + 1
        period_data = indicators.loc[start_year:end_year]
        
        # 期間の特徴量計算
        characteristics = {}
        for col in period_data.select_dtypes(include=[np.number]).columns:
            characteristics[col] = float(period_data[col].mean())
        
        # 遷移トリガー分析
        transition_triggers = self._identify_transition_triggers(period_data, phase_type)
        
        # リスク要因特定
        risk_factors = self._identify_risk_factors(period_data, phase_type)
        
        return GrowthPhase(
            phase_name=self.growth_phases[phase_type]['name'],
            duration_years=duration,
            start_year=start_year,
            end_year=end_year,
            key_characteristics=characteristics,
            transition_triggers=transition_triggers,
            risk_factors=risk_factors
        )
    
    def _determine_current_phase(self, indicators: pd.DataFrame, phases: List[GrowthPhase]) -> str:
        """現在の成長段階判定"""
        if not phases:
            return 'unknown'
        
        # 最新の段階を返す
        latest_phase = max(phases, key=lambda x: x.end_year)
        
        # 段階名から内部キーにマッピング
        phase_mapping = {v['name']: k for k, v in self.growth_phases.items()}
        
        for key, value in phase_mapping.items():
            if latest_phase.phase_name == value:
                return key
        
        return 'unknown'
    
    def _build_transition_history(self, phases: List[GrowthPhase]) -> List[Tuple[str, int]]:
        """段階遷移履歴の構築"""
        history = []
        
        for phase in sorted(phases, key=lambda x: x.start_year):
            # 段階名から内部キーにマッピング
            phase_key = None
            for k, v in self.growth_phases.items():
                if v['name'] == phase.phase_name:
                    phase_key = k
                    break
            
            if phase_key:
                history.append((phase_key, phase.start_year))
        
        return history
    
    def _calculate_success_probability(self, indicators: pd.DataFrame, 
                                        phases: List[GrowthPhase]) -> float:
        """成功確率の計算"""
        if indicators.empty or not phases:
            return 0.5
        
        # 最新データでの評価
        latest_data = indicators.iloc[-1]
        weights = self.config['success_metrics_weight']
        
        scores = {}
        
        # 収益成長スコア
        revenue_growth = latest_data.get('revenue_growth', 0)
        scores['revenue_growth'] = self._normalize_score(revenue_growth, 0, 0.5)
        
        # 収益性スコア
        operating_margin = latest_data.get('operating_margin', 0)
        scores['profitability'] = self._normalize_score(operating_margin, 0, 0.2)
        
        # 効率性スコア
        asset_turnover = latest_data.get('asset_turnover', 1)
        scores['operational_efficiency'] = self._normalize_score(asset_turnover, 0.5, 2.0)
        
        # 市場地位スコア（利用可能な場合）
        global_expansion = latest_data.get('global_expansion', 0.1)
        scores['market_position'] = self._normalize_score(global_expansion, 0, 0.5)
        
        # 財務安定性スコア
        financial_stability = latest_data.get('financial_stability', 0.5)
        scores['financial_stability'] = self._normalize_score(financial_stability, 0, 1)
        
        # 重み付き平均
        success_prob = sum(scores[key] * weights[key] for key in weights.keys() if key in scores)
        
        # 段階別調整
        current_phase = self._determine_current_phase(indicators, phases)
        phase_adjustments = {
            'startup': 0.9,  # スタートアップは高い不確実性
            'growth': 1.1,   # 成長期はプラス評価
            'maturity': 1.0, # 成熟期は現状維持
            'decline': 0.7   # 衰退期はマイナス評価
        }
        
        success_prob *= phase_adjustments.get(current_phase, 1.0)
        
        return min(max(success_prob, 0.0), 1.0)
    
    def _normalize_score(self, value: float, min_val: float, max_val: float) -> float:
        """スコア正規化（0-1範囲）"""
        if max_val == min_val:
            return 0.5
        
        normalized = (value - min_val) / (max_val - min_val)
        return min(max(normalized, 0.0), 1.0)
    
    def _calculate_risk_score(self, indicators: pd.DataFrame, current_phase: str) -> float:
        """リスクスコアの計算"""
        if indicators.empty:
            return 0.5
        
        latest_data = indicators.tail(min(self.config['risk_assessment_window'], len(indicators)))
        
        risk_factors = {}
        
        # 収益ボラティリティリスク
        revenue_volatility = latest_data.get('revenue_volatility', pd.Series([0.1])).mean()
        risk_factors['volatility_risk'] = min(revenue_volatility * 2, 1.0)
        
        # 収益性悪化リスク
        margin_trend = latest_data.get('margin_trend', pd.Series([0])).mean()
        risk_factors['profitability_risk'] = max(-margin_trend * 5, 0.0) if margin_trend < 0 else 0.0
        
        # 財務健全性リスク
        financial_stability = latest_data.get('financial_stability', pd.Series([0.7])).mean()
        risk_factors['financial_risk'] = max(1 - financial_stability, 0.0)
        
        # 成長持続性リスク
        revenue_acceleration = latest_data.get('revenue_acceleration', pd.Series([0])).mean()
        risk_factors['growth_sustainability_risk'] = max(-revenue_acceleration, 0.0) if revenue_acceleration < 0 else 0.0
        
        # 段階別リスク調整
        phase_risk_multipliers = {
            'startup': 1.3,
            'growth': 1.0,
            'maturity': 0.8,
            'decline': 1.5
        }
        
        base_risk = np.mean(list(risk_factors.values()))
        adjusted_risk = base_risk * phase_risk_multipliers.get(current_phase, 1.0)
        
        return min(max(adjusted_risk, 0.0), 1.0)
    
    def _generate_recommendations(self, current_phase: str, indicators: pd.DataFrame, 
                                risk_score: float) -> List[str]:
        """推奨事項の生成"""
        recommendations = []
        
        if indicators.empty:
            return ["十分なデータがありません"]
        
        latest_data = indicators.iloc[-1]
        
        # 段階別基本推奨事項
        phase_recommendations = {
            'startup': [
                "キャッシュフロー管理の強化",
                "コア事業への集中",
                "市場検証とピボット準備",
                "人材確保と組織体制整備"
            ],
            'growth': [
                "スケーラブルなビジネスモデルの確立",
                "市場シェア拡大戦略の実行",
                "オペレーション効率化",
                "グローバル展開の検討"
            ],
            'maturity': [
                "収益性の持続的向上",
                "新規事業・イノベーション投資",
                "コスト最適化",
                "株主還元政策の充実"
            ],
            'decline': [
                "事業ポートフォリオの見直し",
                "コスト削減と効率化",
                "新市場・新技術への転換",
                "戦略的提携・M&Aの検討"
            ]
        }
        
        recommendations.extend(phase_recommendations.get(current_phase, []))
        
        # リスク別追加推奨事項
        if risk_score > 0.7:
            recommendations.extend([
                "リスク管理体制の強化",
                "財務健全性の向上",
                "事業多角化の検討"
            ])
        
        # 指標別具体的推奨事項
        if latest_data.get('operating_margin', 0) < 0.05:
            recommendations.append("収益性改善：コスト構造の見直しとプライシング戦略の最適化")
        
        if latest_data.get('financial_stability', 0.7) < 0.6:
            recommendations.append("財務安定性向上：負債比率の改善と流動性の確保")
        
        if latest_data.get('rd_intensity', 0) < 0.03:
            recommendations.append("競争力強化：研究開発投資の拡充とイノベーション推進")
        
        return recommendations[:8]  # 最大8項目に制限
    
    def _identify_transition_triggers(self, period_data: pd.DataFrame, phase_type: str) -> List[str]:
        """段階遷移トリガーの特定"""
        triggers = []
        
        if period_data.empty:
            return triggers
        
        # 収益成長の変化
        revenue_growth_change = period_data.get('revenue_acceleration', pd.Series([0])).mean()
        if abs(revenue_growth_change) > 0.1:
            triggers.append("収益成長率の大幅変化")
        
        # 利益率の変化
        margin_change = period_data.get('margin_trend', pd.Series([0])).mean()
        if abs(margin_change) > 0.02:
            triggers.append("利益率の構造的変化")
        
        # 投資パターンの変化
        investment_change = period_data.get('investment_growth', pd.Series([0])).mean()
        if abs(investment_change) > 0.2:
            triggers.append("投資戦略の転換")
        
        return triggers
    
    def _identify_risk_factors(self, period_data: pd.DataFrame, phase_type: str) -> List[str]:
        """リスク要因の特定"""
        risks = []
        
        if period_data.empty:
            return risks
        
        # 段階別リスク要因
        phase_specific_risks = {
            'startup': ["資金調達リスク", "市場適合リスク", "競合参入リスク"],
            'growth': ["スケーリングリスク", "競争激化リスク", "オペレーションリスク"],
            'maturity': ["市場飽和リスク", "イノベーションリスク", "新規参入者脅威"],
            'decline': ["事業継続リスク", "財務悪化リスク", "市場撤退リスク"]
        }
        
        risks.extend(phase_specific_risks.get(phase_type, []))
        
        # データ基づく追加リスク
        revenue_volatility = period_data.get('revenue_volatility', pd.Series([0.1])).mean()
        if revenue_volatility > 0.3:
            risks.append("収益ボラティリティリスク")
        
        financial_stability = period_data.get('financial_stability', pd.Series([0.7])).mean()
        if financial_stability < 0.5:
            risks.append("財務健全性リスク")
        
        return risks
    
    def compare_phase_patterns(self, companies_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """複数企業の成長段階パターン比較"""
        comparison_results = []
        
        for company_id, data in companies_data.items():
            try:
                result = self.analyze_company_phases(data, company_id, company_id)
                
                comparison_results.append({
                    'company_id': company_id,
                    'total_phases': len(result.phases_detected),
                    'current_phase': result.current_phase,
                    'success_probability': result.success_probability,
                    'risk_score': result.risk_score,
                    'phase_duration_avg': np.mean([p.duration_years for p in result.phases_detected]) if result.phases_detected else 0
                })
            except Exception as e:
                print(f"企業 {company_id} の分析でエラー: {e}")
                continue
        
        return pd.DataFrame(comparison_results)
    
    def visualize_phase_analysis(self, result: PhaseAnalysisResult, save_path: Optional[str] = None):
        """成長段階分析結果の可視化"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 段階遷移タイムライン
        self._plot_phase_timeline(ax1, result)
        
        # 成功確率とリスクスコア
        self._plot_success_risk_metrics(ax2, result)
        
        # 段階別特徴量レーダーチャート
        self._plot_phase_characteristics(ax3, result)
        
        # 推奨事項表示
        self._plot_recommendations(ax4, result)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def _plot_phase_timeline(self, ax, result: PhaseAnalysisResult):
        """段階遷移タイムラインプロット"""
        if not result.phases_detected:
            ax.text(0.5, 0.5, 'データ不足', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('成長段階タイムライン')
            return
        
        colors = {'startup': 'red', 'growth': 'green', 'maturity': 'blue', 'decline': 'orange'}
        
        for i, phase in enumerate(result.phases_detected):
            phase_key = None
            for k, v in self.growth_phases.items():
                if v['name'] == phase.phase_name:
                    phase_key = k
                    break
            
            color = colors.get(phase_key, 'gray')
            ax.barh(i, phase.duration_years, left=phase.start_year, 
                    color=color, alpha=0.7, label=phase.phase_name)
            
            # 段階名を表示
            ax.text(phase.start_year + phase.duration_years/2, i, 
                    f'{phase.phase_name}\n({phase.duration_years}年)', 
                    ha='center', va='center', fontsize=8)
        
        ax.set_xlabel('年')
        ax.set_ylabel('成長段階')
        ax.set_title(f'{result.company_name} - 成長段階タイムライン')
        ax.grid(True, alpha=0.3)
    
    def _plot_success_risk_metrics(self, ax, result: PhaseAnalysisResult):
        """成功確率・リスクスコア表示"""
        metrics = ['成功確率', 'リスクスコア']
        values = [result.success_probability, result.risk_score]
        colors = ['green', 'red']
        
        bars = ax.bar(metrics, values, color=colors, alpha=0.7)
        
        # 値をバーの上に表示
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        ax.set_ylim(0, 1)
        ax.set_title('成功確率とリスクスコア')
        ax.grid(True, alpha=0.3)
    
    def _plot_phase_characteristics(self, ax, result: PhaseAnalysisResult):
        """段階別特徴量レーダーチャート"""
        if not result.phases_detected:
            ax.text(0.5, 0.5, 'データ不足', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('段階別特徴量')
            return
        
        # 最新の段階の特徴量を取得
        latest_phase = max(result.phases_detected, key=lambda x: x.end_year)
        
        # レーダーチャート用のデータ準備
        features = ['収益成長', '利益率', '財務安定性', '投資強度', '効率性']
        
        # 特徴量の値を正規化（0-1範囲）
        characteristics = latest_phase.key_characteristics
        values = []
        
        feature_keys = ['revenue_growth', 'operating_margin', 'financial_stability', 
                        'rd_intensity', 'asset_turnover']
        
        for key in feature_keys:
            value = characteristics.get(key, 0)
            # 各指標を0-1範囲に正規化
            if key == 'revenue_growth':
                normalized = self._normalize_score(value, -0.2, 0.5)
            elif key == 'operating_margin':
                normalized = self._normalize_score(value, -0.1, 0.3)
            elif key == 'financial_stability':
                normalized = self._normalize_score(value, 0, 1)
            elif key == 'rd_intensity':
                normalized = self._normalize_score(value, 0, 0.1)
            else:  # asset_turnover
                normalized = self._normalize_score(value, 0, 2)
            
            values.append(normalized)
        
        # レーダーチャートの描画
        angles = np.linspace(0, 2*np.pi, len(features), endpoint=False).tolist()
        values += values[:1]  # 円を閉じるため最初の値を追加
        angles += angles[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, color='blue', alpha=0.7)
        ax.fill(angles, values, alpha=0.25, color='blue')
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(features)
        ax.set_ylim(0, 1)
        ax.set_title(f'現在段階の特徴量\n({latest_phase.phase_name})')
        ax.grid(True)
    
    def _plot_recommendations(self, ax, result: PhaseAnalysisResult):
        """推奨事項表示"""
        ax.axis('off')
        
        recommendations_text = '\n'.join([f'• {rec}' for rec in result.recommendations[:6]])
        
        ax.text(0.05, 0.95, f'【推奨事項】\n\n{recommendations_text}', 
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
        
        # 現在の段階も表示
        ax.text(0.05, 0.3, f'現在の成長段階: {self.growth_phases.get(result.current_phase, {}).get("name", "不明")}', 
                transform=ax.transAxes, fontsize=12, weight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8))
    
    def generate_phase_report(self, result: PhaseAnalysisResult) -> str:
        """成長段階分析レポート生成"""
        report = f"""
=== {result.company_name} 成長段階分析レポート ===

【概要】
企業ID: {result.company_id}
現在の成長段階: {self.growth_phases.get(result.current_phase, {}).get('name', '不明')}
成功確率: {result.success_probability:.1%}
リスクスコア: {result.risk_score:.3f}

【成長段階履歴】
検出された段階数: {len(result.phases_detected)}
"""
        
        for i, phase in enumerate(result.phases_detected):
            report += f"""
段階{i+1}: {phase.phase_name}
期間: {phase.start_year}年 - {phase.end_year}年 ({phase.duration_years}年間)
主要特徴:
"""
            # 主要な特徴量を抽出
            key_chars = sorted(phase.key_characteristics.items(), 
                                key=lambda x: abs(x[1]), reverse=True)[:5]
            
            for char_name, char_value in key_chars:
                report += f"  - {char_name}: {char_value:.3f}\n"
            
            if phase.transition_triggers:
                report += f"遷移トリガー: {', '.join(phase.transition_triggers)}\n"
            
            if phase.risk_factors:
                report += f"リスク要因: {', '.join(phase.risk_factors[:3])}\n"
        
        report += f"""
【段階遷移パターン】
"""
        for phase_name, year in result.phase_transition_history:
            phase_display = self.growth_phases.get(phase_name, {}).get('name', phase_name)
            report += f"{year}年: {phase_display}への移行\n"
        
        report += f"""
【推奨事項】
"""
        for i, rec in enumerate(result.recommendations, 1):
            report += f"{i}. {rec}\n"
        
        report += f"""
【分析実行日時】
{datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}
"""
        
        return report
    
    def _map_clusters_to_phases(self, indicators: pd.DataFrame, 
                                cluster_labels: np.ndarray, 
                                cluster_centers: np.ndarray) -> List[GrowthPhase]:
        """クラスターを成長段階にマッピング"""
        phases = []
        years = indicators.index.tolist()
        
        # クラスター中心の特徴から段階を推定
        cluster_to_phase = {}
        
        for i, center in enumerate(cluster_centers):
            # 各クラスターの特徴量から段階を推定
            # 簡単な例：収益成長率と利益率の組み合わせで判定
            avg_growth = center[0] if len(center) > 0 else 0
            avg_margin = center[1] if len(center) > 1 else 0
            
            if avg_growth > 0.3 and avg_margin < 0.05:
                cluster_to_phase[i] = 'startup'
            elif avg_growth > 0.1 and avg_margin > 0.05:
                cluster_to_phase[i] = 'growth'
            elif abs(avg_growth) < 0.1 and avg_margin > 0.1:
                cluster_to_phase[i] = 'maturity'
            else:
                cluster_to_phase[i] = 'decline'
        
        # 連続する同一段階をまとめる
        current_phase = None
        phase_start = None
        
        for i, (year, cluster) in enumerate(zip(years, cluster_labels)):
            phase_type = cluster_to_phase.get(cluster, 'unknown')
            
            if phase_type != current_phase:
                if current_phase is not None and phase_start is not None:
                    phases.append(self._create_growth_phase(
                        current_phase, phase_start, years[i-1], indicators
                    ))
                
                current_phase = phase_type
                phase_start = year
        
        # 最後の段階
        if current_phase is not None and phase_start is not None:
            phases.append(self._create_growth_phase(
                current_phase, phase_start, years[-1], indicators
            ))
        
        return phases
    
    def _merge_phase_detections(self, rule_phases: List[GrowthPhase], 
                                ml_phases: List[GrowthPhase]) -> List[GrowthPhase]:
        """ルールベースとML結果の統合"""
        # 簡単な統合ロジック：ルールベースを基本とし、MLで境界を調整
        merged_phases = []
        
        for rule_phase in rule_phases:
            # ML結果で同期間の段階を探す
            overlapping_ml_phases = [
                ml_p for ml_p in ml_phases 
                if (ml_p.start_year <= rule_phase.end_year and 
                    ml_p.end_year >= rule_phase.start_year)
            ]
            
            if overlapping_ml_phases:
                # 最も重複の大きいML段階を選択
                best_ml_phase = max(overlapping_ml_phases, 
                                    key=lambda x: min(x.end_year, rule_phase.end_year) - 
                                                max(x.start_year, rule_phase.start_year))
                
                # 統合された段階を作成
                merged_start = max(rule_phase.start_year, best_ml_phase.start_year)
                merged_end = min(rule_phase.end_year, best_ml_phase.end_year)
                
                if merged_end > merged_start:
                    # 特徴量を平均化
                    merged_chars = {}
                    for key in rule_phase.key_characteristics.keys():
                        rule_val = rule_phase.key_characteristics.get(key, 0)
                        ml_val = best_ml_phase.key_characteristics.get(key, 0)
                        merged_chars[key] = (rule_val + ml_val) / 2
                    
                    merged_phase = GrowthPhase(
                        phase_name=rule_phase.phase_name,  # ルールベースを優先
                        duration_years=merged_end - merged_start + 1,
                        start_year=merged_start,
                        end_year=merged_end,
                        key_characteristics=merged_chars,
                        transition_triggers=rule_phase.transition_triggers,
                        risk_factors=rule_phase.risk_factors
                    )
                    
                    merged_phases.append(merged_phase)
            else:
                merged_phases.append(rule_phase)
        
        return merged_phases
    
    def export_analysis_results(self, results: List[PhaseAnalysisResult], 
                                output_path: str, format_type: str = 'excel'):
        """分析結果のエクスポート"""
        if format_type == 'excel':
            self._export_to_excel(results, output_path)
        elif format_type == 'csv':
            self._export_to_csv(results, output_path)
        else:
            raise ValueError("サポートされていない形式です。'excel'または'csv'を指定してください。")
    
    def _export_to_excel(self, results: List[PhaseAnalysisResult], output_path: str):
        """Excel形式でエクスポート"""
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # サマリーシート
            summary_data = []
            for result in results:
                summary_data.append({
                    '企業ID': result.company_id,
                    '企業名': result.company_name,
                    '現在段階': self.growth_phases.get(result.current_phase, {}).get('name', '不明'),
                    '段階数': len(result.phases_detected),
                    '成功確率': result.success_probability,
                    'リスクスコア': result.risk_score
                })
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='サマリー', index=False)
            
            # 詳細シート（企業別）
            for result in results:
                detail_data = []
                for phase in result.phases_detected:
                    detail_data.append({
                        '段階名': phase.phase_name,
                        '開始年': phase.start_year,
                        '終了年': phase.end_year,
                        '期間': phase.duration_years,
                        '主要特徴': str(phase.key_characteristics)[:50] + '...',
                        'リスク要因': ', '.join(phase.risk_factors[:3])
                    })
                
                if detail_data:
                    detail_df = pd.DataFrame(detail_data)
                    sheet_name = f'{result.company_name[:20]}'  # シート名の長さ制限
                    detail_df.to_excel(writer, sheet_name=sheet_name, index=False)
    
    def _export_to_csv(self, results: List[PhaseAnalysisResult], output_path: str):
        """CSV形式でエクスポート"""
        export_data = []
        
        for result in results:
            base_info = {
                '企業ID': result.company_id,
                '企業名': result.company_name,
                '現在段階': self.growth_phases.get(result.current_phase, {}).get('name', '不明'),
                '成功確率': result.success_probability,
                'リスクスコア': result.risk_score,
                '総段階数': len(result.phases_detected)
            }
            
            # 各段階の情報を追加
            for i, phase in enumerate(result.phases_detected):
                phase_info = base_info.copy()
                phase_info.update({
                    f'段階{i+1}_名前': phase.phase_name,
                    f'段階{i+1}_開始年': phase.start_year,
                    f'段階{i+1}_終了年': phase.end_year,
                    f'段階{i+1}_期間': phase.duration_years
                })
                export_data.append(phase_info)
        
        if export_data:
            df = pd.DataFrame(export_data)
            df.to_csv(output_path, index=False, encoding='utf-8-sig')


# 使用例とテスト用のコード
if __name__ == "__main__":
    # サンプルデータでのテスト
    analyzer = GrowthPhaseAnalyzer()
    
    # ダミーデータ作成（実際の使用時はEDINETデータを使用）
    years = range(2010, 2024)
    np.random.seed(42)
    
    sample_data = pd.DataFrame({
        '売上高': np.random.normal(1000, 200, len(years)) * np.exp(np.arange(len(years)) * 0.1),
        '売上高営業利益率': np.random.normal(8, 3, len(years)),
        '売上高当期純利益率': np.random.normal(5, 2, len(years)),
        'ROE': np.random.normal(12, 4, len(years)),
        '総資産回転率': np.random.normal(1.2, 0.3, len(years)),
        '自己資本比率': np.random.normal(60, 10, len(years)),
        '研究開発費率': np.random.normal(3, 1, len(years)),
        '従業員数': np.random.normal(1000, 100, len(years)) * np.exp(np.arange(len(years)) * 0.05)
    }, index=years)
    
    # 分析実行
    try:
        result = analyzer.analyze_company_phases(sample_data, "TEST001", "テスト企業")
        
        # 結果表示
        print("=== 成長段階分析結果 ===")
        print(f"企業名: {result.company_name}")
        print(f"現在段階: {analyzer.growth_phases.get(result.current_phase, {}).get('name', '不明')}")
        print(f"成功確率: {result.success_probability:.3f}")
        print(f"リスクスコア: {result.risk_score:.3f}")
        print(f"検出段階数: {len(result.phases_detected)}")
        
        # レポート生成
        report = analyzer.generate_phase_report(result)
        print("\n" + report)
        
        # 可視化（注釈：実際の使用時にはmatplotlibが必要）
        # analyzer.visualize_phase_analysis(result)
        
    except Exception as e:
        print(f"分析エラー: {e}")