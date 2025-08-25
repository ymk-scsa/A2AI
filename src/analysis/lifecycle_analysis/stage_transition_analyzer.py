"""
A2AI - Advanced Financial Analysis AI
企業ライフサイクル段階遷移分析器

このモジュールは企業の成長段階（スタートアップ、成長期、成熟期、衰退期、再生期）間の
遷移パターンを分析し、各段階での特徴的な財務指標変化を特定します。
150社×40年のデータを用いて、市場シェア別（高シェア/低下/失失）の遷移パターンを比較分析します。
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import warnings
from dataclasses import dataclass
from enum import Enum
import logging
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency
import networkx as nx


class LifecycleStage(Enum):
    """企業ライフサイクル段階定義"""
    STARTUP = "startup"          # 設立期・スタートアップ
    GROWTH = "growth"            # 成長期
    MATURITY = "maturity"        # 成熟期
    DECLINE = "decline"          # 衰退期
    REVIVAL = "revival"          # 再生・復活期
    UNKNOWN = "unknown"          # 分類不可


@dataclass
class TransitionEvent:
    """段階遷移イベント"""
    company_id: str
    year: int
    from_stage: LifecycleStage
    to_stage: LifecycleStage
    transition_probability: float
    key_factors: Dict[str, float]  # 遷移に影響した要因項目
    financial_context: Dict[str, float]  # 遷移時の財務状況


@dataclass
class StageCharacteristics:
    """各段階の特徴"""
    stage: LifecycleStage
    avg_duration: float  # 平均滞在期間（年）
    key_indicators: Dict[str, float]  # 特徴的な財務指標
    transition_probabilities: Dict[LifecycleStage, float]  # 他段階への遷移確率
    survival_rate: float  # 5年生存率


class StageTransitionAnalyzer:
    """企業ライフサイクル段階遷移分析器
    
    企業の財務データから自動的にライフサイクル段階を判定し、
    段階間の遷移パターンを分析します。
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Args:
            config: 分析設定辞書
        """
        self.config = config or self._default_config()
        self.logger = self._setup_logger()
        
        # 分析結果格納
        self.stage_classifications: Dict[str, Dict[int, LifecycleStage]] = {}
        self.transition_events: List[TransitionEvent] = []
        self.stage_characteristics: Dict[LifecycleStage, StageCharacteristics] = {}
        self.transition_matrix: pd.DataFrame = pd.DataFrame()
        
        # 段階判定モデル
        self.stage_classifier = None
        self.scaler = StandardScaler()
        
    def _default_config(self) -> Dict:
        """デフォルト設定"""
        return {
            'min_years_for_stage': 2,  # 段階判定に必要な最小年数
            'transition_threshold': 0.6,  # 遷移判定閾値
            'clustering_features': [
                # 成長性指標
                'sales_growth_rate', 'asset_growth_rate', 'employee_growth_rate',
                # 収益性指標
                'operating_margin', 'net_margin', 'roe',
                # 効率性指標
                'asset_turnover', 'inventory_turnover', 
                # 財務健全性指標
                'debt_ratio', 'current_ratio', 'interest_coverage',
                # 投資活動指標
                'rd_intensity', 'capex_intensity', 'age_years'
            ],
            'market_categories': ['high_share', 'declining', 'lost'],
            'evaluation_periods': [1, 3, 5, 10],  # 遷移分析期間（年）
        }
    
    def _setup_logger(self) -> logging.Logger:
        """ログ設定"""
        logger = logging.getLogger('StageTransitionAnalyzer')
        logger.setLevel(logging.INFO)
        return logger
    
    def analyze_stage_transitions(
        self, 
        financial_data: pd.DataFrame,
        company_info: pd.DataFrame,
        market_categories: pd.DataFrame
    ) -> Dict[str, any]:
        """
        企業ライフサイクル段階遷移の包括的分析
        
        Args:
            financial_data: 財務データ（会社ID、年、各種財務指標）
            company_info: 企業情報（会社ID、設立年、業界など）
            market_categories: 市場カテゴリ（会社ID、市場分類）
            
        Returns:
            分析結果辞書
        """
        self.logger.info("企業ライフサイクル段階遷移分析を開始")
        
        # 1. データ準備
        processed_data = self._prepare_data(financial_data, company_info, market_categories)
        
        # 2. 段階分類モデル構築
        self._build_stage_classifier(processed_data)
        
        # 3. 各企業の段階遷移履歴構築
        self._classify_company_stages(processed_data)
        
        # 4. 遷移イベント抽出
        self._extract_transition_events(processed_data)
        
        # 5. 段階特徴分析
        self._analyze_stage_characteristics(processed_data)
        
        # 6. 遷移確率マトリックス構築
        self._build_transition_matrix()
        
        # 7. 市場カテゴリ別比較分析
        market_analysis = self._analyze_by_market_category(processed_data)
        
        # 8. 生存分析統合
        survival_analysis = self._integrate_survival_analysis(processed_data)
        
        # 9. 結果統合
        results = {
            'stage_classifications': self.stage_classifications,
            'transition_events': self.transition_events,
            'stage_characteristics': self.stage_characteristics,
            'transition_matrix': self.transition_matrix,
            'market_analysis': market_analysis,
            'survival_analysis': survival_analysis,
            'summary_statistics': self._generate_summary_statistics()
        }
        
        self.logger.info("段階遷移分析完了")
        return results
    
    def _prepare_data(
        self, 
        financial_data: pd.DataFrame, 
        company_info: pd.DataFrame,
        market_categories: pd.DataFrame
    ) -> pd.DataFrame:
        """データ準備・特徴量エンジニアリング"""
        
        # データ結合
        data = financial_data.merge(company_info, on='company_id', how='left')
        data = data.merge(market_categories, on='company_id', how='left')
        
        # 企業年齢計算
        data['age_years'] = data['year'] - data['founded_year']
        
        # 成長率計算（前年比）
        growth_cols = ['sales_growth_rate', 'asset_growth_rate', 'employee_growth_rate']
        for company_id in data['company_id'].unique():
            company_mask = data['company_id'] == company_id
            company_data = data[company_mask].sort_values('year')
            
            # 売上高成長率
            if 'sales' in company_data.columns:
                data.loc[company_mask, 'sales_growth_rate'] = (
                    company_data['sales'].pct_change() * 100
                )
            
            # 総資産成長率
            if 'total_assets' in company_data.columns:
                data.loc[company_mask, 'asset_growth_rate'] = (
                    company_data['total_assets'].pct_change() * 100
                )
            
            # 従業員数成長率
            if 'employee_count' in company_data.columns:
                data.loc[company_mask, 'employee_growth_rate'] = (
                    company_data['employee_count'].pct_change() * 100
                )
        
        # 財務比率計算
        self._calculate_financial_ratios(data)
        
        # 欠損値処理
        data = self._handle_missing_values(data)
        
        return data
    
    def _calculate_financial_ratios(self, data: pd.DataFrame) -> None:
        """財務比率計算"""
        
        # 収益性指標
        if 'operating_income' in data.columns and 'sales' in data.columns:
            data['operating_margin'] = data['operating_income'] / data['sales'] * 100
        
        if 'net_income' in data.columns and 'sales' in data.columns:
            data['net_margin'] = data['net_income'] / data['sales'] * 100
        
        if 'net_income' in data.columns and 'shareholders_equity' in data.columns:
            data['roe'] = data['net_income'] / data['shareholders_equity'] * 100
        
        # 効率性指標
        if 'sales' in data.columns and 'total_assets' in data.columns:
            data['asset_turnover'] = data['sales'] / data['total_assets']
        
        if 'cogs' in data.columns and 'inventory' in data.columns:
            data['inventory_turnover'] = data['cogs'] / data['inventory']
        
        # 財務健全性指標
        if 'total_debt' in data.columns and 'total_assets' in data.columns:
            data['debt_ratio'] = data['total_debt'] / data['total_assets'] * 100
        
        if 'current_assets' in data.columns and 'current_liabilities' in data.columns:
            data['current_ratio'] = data['current_assets'] / data['current_liabilities']
        
        if 'operating_income' in data.columns and 'interest_expense' in data.columns:
            data['interest_coverage'] = np.where(
                data['interest_expense'] > 0,
                data['operating_income'] / data['interest_expense'],
                np.inf
            )
        
        # 投資活動指標
        if 'rd_expense' in data.columns and 'sales' in data.columns:
            data['rd_intensity'] = data['rd_expense'] / data['sales'] * 100
        
        if 'capex' in data.columns and 'sales' in data.columns:
            data['capex_intensity'] = data['capex'] / data['sales'] * 100
    
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """欠損値処理"""
        
        # 数値列の欠損値を業界・年度平均で補完
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col in self.config['clustering_features']:
                # 業界・年度グループ平均で補完
                data[col] = data.groupby(['industry', 'year'])[col].transform(
                    lambda x: x.fillna(x.mean())
                )
                # 全体平均で残り補完
                data[col] = data[col].fillna(data[col].mean())
        
        return data
    
    def _build_stage_classifier(self, data: pd.DataFrame) -> None:
        """段階分類モデル構築（教師なしクラスタリング）"""
        
        # 特徴量準備
        feature_cols = [col for col in self.config['clustering_features'] 
                        if col in data.columns]
        
        features = data[feature_cols].copy()
        features = features.dropna()
        
        if features.empty:
            raise ValueError("分析用特徴量データが不足しています")
        
        # 標準化
        features_scaled = self.scaler.fit_transform(features)
        
        # 最適クラスター数決定（シルエット分析）
        silhouette_scores = []
        K_range = range(3, 8)  # 3-7段階をテスト
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(features_scaled)
            score = silhouette_score(features_scaled, labels)
            silhouette_scores.append(score)
        
        optimal_k = K_range[np.argmax(silhouette_scores)]
        
        # 最終モデル学習
        self.stage_classifier = KMeans(
            n_clusters=optimal_k, 
            random_state=42, 
            n_init=10
        )
        cluster_labels = self.stage_classifier.fit_predict(features_scaled)
        
        # クラスターをライフサイクル段階にマッピング
        self._map_clusters_to_stages(features, cluster_labels)
        
        self.logger.info(f"段階分類モデル構築完了（{optimal_k}段階）")
    
    def _map_clusters_to_stages(
        self, 
        features: pd.DataFrame, 
        cluster_labels: np.ndarray
    ) -> None:
        """クラスターをライフサイクル段階にマッピング"""
        
        # 各クラスターの特徴分析
        features_with_clusters = features.copy()
        features_with_clusters['cluster'] = cluster_labels
        
        cluster_characteristics = {}
        for cluster_id in np.unique(cluster_labels):
            cluster_data = features_with_clusters[
                features_with_clusters['cluster'] == cluster_id
            ]
            
            char = {
                'avg_age': cluster_data['age_years'].mean() if 'age_years' in cluster_data else 0,
                'avg_growth': cluster_data['sales_growth_rate'].mean() if 'sales_growth_rate' in cluster_data else 0,
                'avg_margin': cluster_data['operating_margin'].mean() if 'operating_margin' in cluster_data else 0,
                'avg_debt': cluster_data['debt_ratio'].mean() if 'debt_ratio' in cluster_data else 0,
                'avg_rd': cluster_data['rd_intensity'].mean() if 'rd_intensity' in cluster_data else 0,
                'size': len(cluster_data)
            }
            cluster_characteristics[cluster_id] = char
        
        # ライフサイクル段階にマッピング
        self.cluster_to_stage_mapping = {}
        
        # 年齢と成長率を主要指標として段階判定
        for cluster_id, char in cluster_characteristics.items():
            age = char['avg_age']
            growth = char['avg_growth']
            margin = char['avg_margin']
            
            if age <= 5 and growth > 10:
                stage = LifecycleStage.STARTUP
            elif age <= 15 and growth > 5 and margin > 5:
                stage = LifecycleStage.GROWTH
            elif growth > -5 and margin > 3:
                stage = LifecycleStage.MATURITY
            elif growth < -5 or margin < 0:
                stage = LifecycleStage.DECLINE
            elif growth > 0 and margin > 0:  # 衰退から回復の兆候
                stage = LifecycleStage.REVIVAL
            else:
                stage = LifecycleStage.UNKNOWN
            
            self.cluster_to_stage_mapping[cluster_id] = stage
        
        self.logger.info("クラスター-段階マッピング完了")
    
    def _classify_company_stages(self, data: pd.DataFrame) -> None:
        """各企業の年次段階分類"""
        
        feature_cols = [col for col in self.config['clustering_features'] 
                        if col in data.columns]
        
        for company_id in data['company_id'].unique():
            company_data = data[data['company_id'] == company_id].copy()
            company_stages = {}
            
            for _, row in company_data.iterrows():
                year = row['year']
                
                # 特徴量準備
                features = row[feature_cols].values.reshape(1, -1)
                
                # 欠損値チェック
                if np.isnan(features).any():
                    stage = LifecycleStage.UNKNOWN
                else:
                    # 段階予測
                    features_scaled = self.scaler.transform(features)
                    cluster_pred = self.stage_classifier.predict(features_scaled)[0]
                    stage = self.cluster_to_stage_mapping.get(
                        cluster_pred, LifecycleStage.UNKNOWN
                    )
                
                company_stages[year] = stage
            
            self.stage_classifications[company_id] = company_stages
    
    def _extract_transition_events(self, data: pd.DataFrame) -> None:
        """段階遷移イベント抽出"""
        
        self.transition_events = []
        
        for company_id, stages_dict in self.stage_classifications.items():
            sorted_years = sorted(stages_dict.keys())
            
            for i in range(1, len(sorted_years)):
                prev_year = sorted_years[i-1]
                curr_year = sorted_years[i]
                
                prev_stage = stages_dict[prev_year]
                curr_stage = stages_dict[curr_year]
                
                # 遷移発生の場合
                if prev_stage != curr_stage and prev_stage != LifecycleStage.UNKNOWN:
                    
                    # 遷移時の財務コンテキスト取得
                    company_data = data[
                        (data['company_id'] == company_id) & 
                        (data['year'] == curr_year)
                    ]
                    
                    if not company_data.empty:
                        financial_context = {
                            col: company_data[col].iloc[0] 
                            for col in ['sales_growth_rate', 'operating_margin', 
                                        'roe', 'debt_ratio'] 
                            if col in company_data.columns
                        }
                        
                        # 主要要因分析（簡易版）
                        key_factors = self._identify_transition_factors(
                            company_data.iloc[0]
                        )
                        
                        event = TransitionEvent(
                            company_id=company_id,
                            year=curr_year,
                            from_stage=prev_stage,
                            to_stage=curr_stage,
                            transition_probability=0.8,  # 実際は統計的に計算
                            key_factors=key_factors,
                            financial_context=financial_context
                        )
                        
                        self.transition_events.append(event)
    
    def _identify_transition_factors(self, company_row: pd.Series) -> Dict[str, float]:
        """遷移要因特定（簡易版）"""
        
        factors = {}
        
        # 成長性要因
        if 'sales_growth_rate' in company_row:
            factors['growth_factor'] = abs(company_row['sales_growth_rate'])
        
        # 収益性要因
        if 'operating_margin' in company_row:
            factors['profitability_factor'] = abs(company_row['operating_margin'])
        
        # 財務健全性要因
        if 'debt_ratio' in company_row:
            factors['financial_health_factor'] = 100 - company_row['debt_ratio']
        
        # 投資活動要因
        if 'rd_intensity' in company_row:
            factors['innovation_factor'] = company_row['rd_intensity']
        
        return factors
    
    def _analyze_stage_characteristics(self, data: pd.DataFrame) -> None:
        """各段階の特徴分析"""
        
        self.stage_characteristics = {}
        
        for stage in LifecycleStage:
            if stage == LifecycleStage.UNKNOWN:
                continue
            
            # 該当段階のデータ抽出
            stage_companies = []
            for company_id, stages_dict in self.stage_classifications.items():
                for year, company_stage in stages_dict.items():
                    if company_stage == stage:
                        stage_companies.append((company_id, year))
            
            if not stage_companies:
                continue
            
            # 財務指標統計計算
            stage_data_list = []
            for company_id, year in stage_companies:
                company_year_data = data[
                    (data['company_id'] == company_id) & 
                    (data['year'] == year)
                ]
                if not company_year_data.empty:
                    stage_data_list.append(company_year_data.iloc[0])
            
            if not stage_data_list:
                continue
            
            stage_df = pd.DataFrame(stage_data_list)
            
            # 主要指標計算
            key_indicators = {}
            for col in ['sales_growth_rate', 'operating_margin', 'roe', 'debt_ratio', 'age_years']:
                if col in stage_df.columns:
                    key_indicators[col] = {
                        'mean': stage_df[col].mean(),
                        'median': stage_df[col].median(),
                        'std': stage_df[col].std()
                    }
            
            # 遷移確率計算
            transition_probs = self._calculate_stage_transition_probabilities(stage)
            
            # 生存率計算（5年）
            survival_rate = self._calculate_stage_survival_rate(stage, stage_companies)
            
            # 平均滞在期間
            avg_duration = self._calculate_avg_stage_duration(stage)
            
            characteristics = StageCharacteristics(
                stage=stage,
                avg_duration=avg_duration,
                key_indicators=key_indicators,
                transition_probabilities=transition_probs,
                survival_rate=survival_rate
            )
            
            self.stage_characteristics[stage] = characteristics
    
    def _calculate_stage_transition_probabilities(
        self, 
        from_stage: LifecycleStage
    ) -> Dict[LifecycleStage, float]:
        """段階からの遷移確率計算"""
        
        transitions_from_stage = [
            event for event in self.transition_events 
            if event.from_stage == from_stage
        ]
        
        if not transitions_from_stage:
            return {}
        
        transition_counts = {}
        for event in transitions_from_stage:
            to_stage = event.to_stage
            transition_counts[to_stage] = transition_counts.get(to_stage, 0) + 1
        
        total_transitions = len(transitions_from_stage)
        transition_probs = {
            stage: count / total_transitions 
            for stage, count in transition_counts.items()
        }
        
        return transition_probs
    
    def _calculate_stage_survival_rate(
        self, 
        stage: LifecycleStage, 
        stage_companies: List[Tuple[str, int]]
    ) -> float:
        """段階における5年生存率計算"""
        
        # 簡易実装：実際は生存分析モジュールと連携
        if len(stage_companies) == 0:
            return 0.0
        
        # 5年後も存続している企業の割合
        survived = 0
        total = len(set(company_id for company_id, _ in stage_companies))
        
        for company_id in set(company_id for company_id, _ in stage_companies):
            # 最新データが5年以上続いている場合は生存とみなす
            company_years = [year for cid, year in stage_companies if cid == company_id]
            if max(company_years) - min(company_years) >= 5:
                survived += 1
        
        return survived / total if total > 0 else 0.0
    
    def _calculate_avg_stage_duration(self, stage: LifecycleStage) -> float:
        """段階の平均滞在期間計算"""
        
        durations = []
        
        for company_id, stages_dict in self.stage_classifications.items():
            sorted_years = sorted(stages_dict.keys())
            current_duration = 0
            
            for year in sorted_years:
                if stages_dict[year] == stage:
                    current_duration += 1
                else:
                    if current_duration > 0:
                        durations.append(current_duration)
                        current_duration = 0
            
            # 最後まで同じ段階の場合
            if current_duration > 0:
                durations.append(current_duration)
        
        return np.mean(durations) if durations else 0.0
    
    def _build_transition_matrix(self) -> None:
        """遷移確率マトリックス構築"""
        
        stages = [stage for stage in LifecycleStage if stage != LifecycleStage.UNKNOWN]
        stage_names = [stage.value for stage in stages]
        
        # 遷移カウントマトリックス初期化
        transition_counts = pd.DataFrame(
            0, 
            index=stage_names, 
            columns=stage_names
        )
        
        # 遷移イベントからカウント
        for event in self.transition_events:
            from_stage = event.from_stage.value
            to_stage = event.to_stage.value
            
            if from_stage in stage_names and to_stage in stage_names:
                transition_counts.loc[from_stage, to_stage] += 1
        
        # 確率に変換
        self.transition_matrix = transition_counts.div(
            transition_counts.sum(axis=1), 
            axis=0
        ).fillna(0)
    
    def _analyze_by_market_category(self, data: pd.DataFrame) -> Dict[str, any]:
        """市場カテゴリ別分析"""
        
        market_analysis = {}
        
        for category in self.config['market_categories']:
            category_companies = data[
                data['market_category'] == category
            ]['company_id'].unique()
            
            # 該当カテゴリの遷移イベント
            category_transitions = [
                event for event in self.transition_events
                if event.company_id in category_companies
            ]
            
            # 段階分布
            stage_distribution = {}
            for company_id in category_companies:
                if company_id in self.stage_classifications:
                    for year, stage in self.stage_classifications[company_id].items():
                        stage_name = stage.value
                        stage_distribution[stage_name] = (
                            stage_distribution.get(stage_name, 0) + 1
                        )
            
            # 遷移パターン分析
            transition_patterns = {}
            for event in category_transitions:
                pattern = f"{event.from_stage.value}→{event.to_stage.value}"
                transition_patterns[pattern] = (
                    transition_patterns.get(pattern, 0) + 1
                )
            
            market_analysis[category] = {
                'company_count': len(category_companies),
                'transition_count': len(category_transitions),
                'stage_distribution': stage_distribution,
                'transition_patterns': transition_patterns,
                'avg_transitions_per_company': (
                    len(category_transitions) / len(category_companies) 
                    if category_companies.size > 0 else 0
                )
            }
        
        return market_analysis
    
    def _integrate_survival_analysis(self, data: pd.DataFrame) -> Dict[str, any]:
        """生存分析との統合"""
        
        # 各段階での消滅率
        extinction_by_stage = {}
        
        for stage in LifecycleStage:
            if stage == LifecycleStage.UNKNOWN:
                continue
            
            # 該当段階で消滅した企業数
            extinct_companies = []
            total_companies = set()
            
            for company_id, stages_dict in self.stage_classifications.items():
                if stage in stages_dict.values():
                    total_companies.add(company_id)
                    
                    # 最後の段階が該当段階で、その後データがない場合
                    sorted_years = sorted(stages_dict.keys())
                    if (stages_dict[sorted_years[-1]] == stage and 
                        sorted_years[-1] < data['year'].max()):
                        extinct_companies.append(company_id)
            
            extinction_rate = (
                len(extinct_companies) / len(total_companies) 
                if total_companies else 0
            )
            
            extinction_by_stage[stage.value] = {
                'extinction_rate': extinction_rate,
                'extinct_count': len(extinct_companies),
                'total_count': len(total_companies)
            }
        
        return {
            'extinction_by_stage': extinction_by_stage,
            'high_risk_transitions': self._identify_high_risk_transitions(),
            'recovery_patterns': self._analyze_recovery_patterns()
        }
    
    def _identify_high_risk_transitions(self) -> List[Dict]:
        """高リスク遷移パターン特定"""
        
        high_risk = []
        
        # 衰退・消滅につながりやすい遷移パターンを特定
        decline_transitions = [
            event for event in self.transition_events
            if event.to_stage == LifecycleStage.DECLINE
        ]
        
        # 遷移パターン別リスク分析
        pattern_risk = {}
        for event in decline_transitions:
            pattern = f"{event.from_stage.value}→{event.to_stage.value}"
            if pattern not in pattern_risk:
                pattern_risk[pattern] = {
                    'count': 0,
                    'avg_financial_health': 0,
                    'companies': []
                }
            
            pattern_risk[pattern]['count'] += 1
            pattern_risk[pattern]['companies'].append(event.company_id)
            
            # 財務健全性スコア（簡易計算）
            financial_context = event.financial_context
            health_score = 0
            if 'operating_margin' in financial_context:
                health_score += max(0, financial_context['operating_margin'])
            if 'debt_ratio' in financial_context:
                health_score += max(0, 100 - financial_context['debt_ratio'])
            
            pattern_risk[pattern]['avg_financial_health'] += health_score
        
        # 平均値計算
        for pattern, data in pattern_risk.items():
            if data['count'] > 0:
                data['avg_financial_health'] /= data['count']
                
                # リスクスコア = 遷移頻度 × (100 - 財務健全性)
                risk_score = data['count'] * (100 - data['avg_financial_health'])
                
                high_risk.append({
                    'transition_pattern': pattern,
                    'risk_score': risk_score,
                    'frequency': data['count'],
                    'avg_financial_health': data['avg_financial_health'],
                    'affected_companies': len(set(data['companies']))
                })
        
        # リスクスコア順にソート
        high_risk.sort(key=lambda x: x['risk_score'], reverse=True)
        
        return high_risk[:10]  # 上位10パターン
    
    def _analyze_recovery_patterns(self) -> Dict[str, any]:
        """回復パターン分析"""
        
        # 衰退から回復した企業の分析
        recovery_events = [
            event for event in self.transition_events
            if (event.from_stage == LifecycleStage.DECLINE and 
                event.to_stage in [LifecycleStage.REVIVAL, LifecycleStage.GROWTH, LifecycleStage.MATURITY])
        ]
        
        recovery_factors = {}
        for event in recovery_events:
            for factor, value in event.key_factors.items():
                if factor not in recovery_factors:
                    recovery_factors[factor] = []
                recovery_factors[factor].append(value)
        
        # 回復要因の統計
        recovery_factor_stats = {}
        for factor, values in recovery_factors.items():
            recovery_factor_stats[factor] = {
                'mean': np.mean(values),
                'median': np.median(values),
                'std': np.std(values)
            }
        
        return {
            'recovery_count': len(recovery_events),
            'recovery_rate': len(recovery_events) / len(self.transition_events) * 100,
            'key_recovery_factors': recovery_factor_stats,
            'successful_companies': [event.company_id for event in recovery_events]
        }
    
    def _generate_summary_statistics(self) -> Dict[str, any]:
        """要約統計生成"""
        
        total_companies = len(self.stage_classifications)
        total_transitions = len(self.transition_events)
        
        # 段階別企業数
        stage_counts = {}
        for company_stages in self.stage_classifications.values():
            for stage in company_stages.values():
                stage_name = stage.value
                stage_counts[stage_name] = stage_counts.get(stage_name, 0) + 1
        
        # 最頻遷移パターン
        transition_patterns = {}
        for event in self.transition_events:
            pattern = f"{event.from_stage.value}→{event.to_stage.value}"
            transition_patterns[pattern] = transition_patterns.get(pattern, 0) + 1
        
        most_common_transitions = sorted(
            transition_patterns.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:5]
        
        return {
            'total_companies_analyzed': total_companies,
            'total_transition_events': total_transitions,
            'avg_transitions_per_company': total_transitions / total_companies if total_companies > 0 else 0,
            'stage_distribution': stage_counts,
            'most_common_transitions': most_common_transitions,
            'analysis_period': {
                'start_year': min([min(stages.keys()) for stages in self.stage_classifications.values()]) if self.stage_classifications else None,
                'end_year': max([max(stages.keys()) for stages in self.stage_classifications.values()]) if self.stage_classifications else None
            }
        }
    
    def predict_future_transitions(
        self, 
        company_id: str, 
        prediction_horizon: int = 5
    ) -> Dict[str, any]:
        """将来の段階遷移予測
        
        Args:
            company_id: 対象企業ID
            prediction_horizon: 予測期間（年）
            
        Returns:
            予測結果辞書
        """
        
        if company_id not in self.stage_classifications:
            return {'error': f'企業ID {company_id} が見つかりません'}
        
        # 現在の段階
        company_stages = self.stage_classifications[company_id]
        current_year = max(company_stages.keys())
        current_stage = company_stages[current_year]
        
        # 遷移確率マトリックスを使用した予測
        predictions = {}
        current_probs = {current_stage: 1.0}
        
        for year in range(1, prediction_horizon + 1):
            next_probs = {}
            
            for stage, prob in current_probs.items():
                if stage.value in self.transition_matrix.index:
                    stage_transitions = self.transition_matrix.loc[stage.value]
                    
                    for next_stage_name, transition_prob in stage_transitions.items():
                        if transition_prob > 0:
                            next_stage = LifecycleStage(next_stage_name)
                            next_probs[next_stage] = (
                                next_probs.get(next_stage, 0) + prob * transition_prob
                            )
                    
                    # 同じ段階に留まる確率（遷移しない場合）
                    stay_prob = 1 - stage_transitions.sum()
                    if stay_prob > 0:
                        next_probs[stage] = next_probs.get(stage, 0) + prob * stay_prob
                else:
                    # 遷移データがない場合は同じ段階に留まる
                    next_probs[stage] = next_probs.get(stage, 0) + prob
            
            predictions[current_year + year] = {
                stage.value: prob for stage, prob in next_probs.items()
            }
            current_probs = next_probs
        
        # 最も可能性の高い遷移パス
        most_likely_path = [current_stage.value]
        current_stage_temp = current_stage
        
        for year in range(1, prediction_horizon + 1):
            if current_stage_temp.value in self.transition_matrix.index:
                next_stage_probs = self.transition_matrix.loc[current_stage_temp.value]
                most_likely_next = next_stage_probs.idxmax()
                
                # 遷移確率が閾値以上の場合のみ遷移
                if next_stage_probs[most_likely_next] > 0.3:
                    current_stage_temp = LifecycleStage(most_likely_next)
                
                most_likely_path.append(current_stage_temp.value)
            else:
                most_likely_path.append(current_stage_temp.value)
        
        return {
            'company_id': company_id,
            'current_stage': current_stage.value,
            'prediction_horizon': prediction_horizon,
            'stage_probabilities_by_year': predictions,
            'most_likely_path': most_likely_path,
            'confidence_score': self._calculate_prediction_confidence(company_id)
        }
    
    def _calculate_prediction_confidence(self, company_id: str) -> float:
        """予測信頼度計算"""
        
        # 企業の遷移履歴の安定性から信頼度を計算
        company_stages = self.stage_classifications[company_id]
        stage_changes = 0
        
        sorted_years = sorted(company_stages.keys())
        for i in range(1, len(sorted_years)):
            if company_stages[sorted_years[i]] != company_stages[sorted_years[i-1]]:
                stage_changes += 1
        
        # 変化が少ないほど予測しやすい（信頼度高）
        stability = 1 - (stage_changes / max(len(sorted_years) - 1, 1))
        
        # データ期間の長さも考慮
        data_length = len(sorted_years)
        length_factor = min(data_length / 10, 1.0)  # 10年以上で最大
        
        confidence = (stability * 0.7 + length_factor * 0.3)
        return confidence
    
    def visualize_transitions(
        self, 
        output_path: str, 
        market_category: Optional[str] = None
    ) -> Dict[str, str]:
        """段階遷移可視化
        
        Args:
            output_path: 出力ディレクトリパス
            market_category: 特定市場カテゴリに絞る場合
            
        Returns:
            生成されたファイルパス辞書
        """
        
        import os
        os.makedirs(output_path, exist_ok=True)
        generated_files = {}
        
        # 1. 遷移確率マトリックスヒートマップ
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            self.transition_matrix, 
            annot=True, 
            cmap='Blues', 
            fmt='.2f',
            cbar_kws={'label': 'Transition Probability'}
        )
        plt.title('企業ライフサイクル段階遷移確率マトリックス')
        plt.xlabel('遷移先段階')
        plt.ylabel('遷移元段階')
        plt.tight_layout()
        
        heatmap_path = os.path.join(output_path, 'transition_matrix_heatmap.png')
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        plt.close()
        generated_files['transition_matrix'] = heatmap_path
        
        # 2. 段階分布パイチャート
        stage_counts = {}
        for company_stages in self.stage_classifications.values():
            for stage in company_stages.values():
                stage_name = stage.value
                stage_counts[stage_name] = stage_counts.get(stage_name, 0) + 1
        
        plt.figure(figsize=(10, 8))
        plt.pie(
            stage_counts.values(), 
            labels=stage_counts.keys(), 
            autopct='%1.1f%%',
            startangle=90
        )
        plt.title('企業ライフサイクル段階分布')
        plt.axis('equal')
        
        pie_path = os.path.join(output_path, 'stage_distribution.png')
        plt.savefig(pie_path, dpi=300, bbox_inches='tight')
        plt.close()
        generated_files['stage_distribution'] = pie_path
        
        # 3. 遷移ネットワーク図
        plt.figure(figsize=(12, 10))
        G = nx.DiGraph()
        
        # ノード追加
        stages = list(self.transition_matrix.index)
        G.add_nodes_from(stages)
        
        # エッジ追加（閾値以上の遷移のみ）
        threshold = 0.1
        for from_stage in self.transition_matrix.index:
            for to_stage in self.transition_matrix.columns:
                prob = self.transition_matrix.loc[from_stage, to_stage]
                if prob >= threshold:
                    G.add_edge(from_stage, to_stage, weight=prob)
        
        # レイアウト
        pos = nx.spring_layout(G, k=3, iterations=50)
        
        # 描画
        nx.draw_networkx_nodes(G, pos, node_size=3000, node_color='lightblue')
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
        
        edges = G.edges()
        weights = [G[u][v]['weight'] * 5 for u, v in edges]  # 線の太さ
        nx.draw_networkx_edges(G, pos, width=weights, alpha=0.7, edge_color='gray')
        
        # エッジラベル（確率）
        edge_labels = {
            (u, v): f"{G[u][v]['weight']:.2f}"
            for u, v in edges
        }
        nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8)
        
        plt.title('企業ライフサイクル段階遷移ネットワーク')
        plt.axis('off')
        
        network_path = os.path.join(output_path, 'transition_network.png')
        plt.savefig(network_path, dpi=300, bbox_inches='tight')
        plt.close()
        generated_files['transition_network'] = network_path
        
        # 4. 時系列遷移パターン（サンプル企業）
        sample_companies = list(self.stage_classifications.keys())[:5]
        
        plt.figure(figsize=(14, 8))
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        
        for i, company_id in enumerate(sample_companies):
            stages_dict = self.stage_classifications[company_id]
            years = sorted(stages_dict.keys())
            
            # 段階を数値に変換
            stage_to_num = {
                LifecycleStage.STARTUP.value: 1,
                LifecycleStage.GROWTH.value: 2,
                LifecycleStage.MATURITY.value: 3,
                LifecycleStage.DECLINE.value: 4,
                LifecycleStage.REVIVAL.value: 2.5,
                LifecycleStage.UNKNOWN.value: 0
            }
            
            stage_nums = [stage_to_num.get(stages_dict[year].value, 0) for year in years]
            
            plt.plot(years, stage_nums, marker='o', label=f'企業{i+1}', 
                    color=colors[i % len(colors)], linewidth=2, markersize=6)
        
        plt.xlabel('年')
        plt.ylabel('ライフサイクル段階')
        plt.title('企業ライフサイクル段階遷移の時系列パターン（サンプル企業）')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Y軸ラベル
        plt.yticks([0, 1, 2, 2.5, 3, 4], 
                    ['Unknown', 'Startup', 'Growth', 'Revival', 'Maturity', 'Decline'])
        
        timeseries_path = os.path.join(output_path, 'transition_timeseries.png')
        plt.savefig(timeseries_path, dpi=300, bbox_inches='tight')
        plt.close()
        generated_files['transition_timeseries'] = timeseries_path
        
        return generated_files
    
    def export_results(self, output_path: str) -> str:
        """分析結果エクスポート
        
        Args:
            output_path: 出力ファイルパス
            
        Returns:
            出力ファイルパス
        """
        
        # 結果をExcelファイルに出力
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            
            # 1. 遷移確率マトリックス
            self.transition_matrix.to_excel(
                writer, 
                sheet_name='遷移確率マトリックス',
                index=True
            )
            
            # 2. 段階特徴統計
            stage_stats_data = []
            for stage, characteristics in self.stage_characteristics.items():
                row = {
                    '段階': stage.value,
                    '平均滞在期間': characteristics.avg_duration,
                    '5年生存率': characteristics.survival_rate
                }
                
                # 主要指標追加
                for indicator, stats in characteristics.key_indicators.items():
                    row[f'{indicator}_平均'] = stats.get('mean', 0)
                    row[f'{indicator}_中央値'] = stats.get('median', 0)
                
                stage_stats_data.append(row)
            
            stage_stats_df = pd.DataFrame(stage_stats_data)
            stage_stats_df.to_excel(
                writer,
                sheet_name='段階特徴統計',
                index=False
            )
            
            # 3. 遷移イベント一覧
            events_data = []
            for event in self.transition_events:
                events_data.append({
                    '企業ID': event.company_id,
                    '年': event.year,
                    '遷移元': event.from_stage.value,
                    '遷移先': event.to_stage.value,
                    '遷移確率': event.transition_probability,
                    '主要要因': str(event.key_factors)
                })
            
            events_df = pd.DataFrame(events_data)
            events_df.to_excel(
                writer,
                sheet_name='遷移イベント一覧',
                index=False
            )
            
            # 4. 要約統計
            summary_stats = self._generate_summary_statistics()
            summary_data = []
            
            for key, value in summary_stats.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        summary_data.append({
                            'カテゴリ': key,
                            '項目': sub_key,
                            '値': str(sub_value)
                        })
                else:
                    summary_data.append({
                        'カテゴリ': key,
                        '項目': '',
                        '値': str(value)
                    })
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(
                writer,
                sheet_name='要約統計',
                index=False
            )
        
        self.logger.info(f"分析結果を {output_path} に出力しました")
        return output_path


# 使用例とテスト用のダミーデータ生成関数
def generate_dummy_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """テスト用ダミーデータ生成"""
    
    np.random.seed(42)
    
    # 企業リスト（150社サンプル）
    companies = [f"company_{i:03d}" for i in range(150)]
    
    # 企業情報
    company_info = pd.DataFrame({
        'company_id': companies,
        'founded_year': np.random.randint(1970, 2010, 150),
        'industry': np.random.choice(['ロボット', '内視鏡', '工作機械', '電子材料', '精密測定'], 150)
    })
    
    # 市場カテゴリ
    market_categories = pd.DataFrame({
        'company_id': companies,
        'market_category': ['high_share'] * 50 + ['declining'] * 50 + ['lost'] * 50
    })
    
    # 財務データ生成（40年分）
    financial_data = []
    
    for company_id in companies:
        founded_year = company_info[company_info['company_id'] == company_id]['founded_year'].iloc[0]
        
        for year in range(1984, 2024):
            if year >= founded_year:  # 設立年以降のデータのみ
                age = year - founded_year
                
                # ライフサイクル段階に応じた財務指標生成
                if age <= 5:  # スタートアップ期
                    sales_growth = np.random.normal(20, 15)
                    operating_margin = np.random.normal(5, 10)
                elif age <= 15:  # 成長期
                    sales_growth = np.random.normal(10, 8)
                    operating_margin = np.random.normal(8, 5)
                elif age <= 25:  # 成熟期
                    sales_growth = np.random.normal(3, 5)
                    operating_margin = np.random.normal(10, 3)
                else:  # 衰退期の可能性
                    sales_growth = np.random.normal(-2, 8)
                    operating_margin = np.random.normal(5, 8)
                
                financial_data.append({
                    'company_id': company_id,
                    'year': year,
                    'sales': np.random.lognormal(10, 1) * (1 + age * 0.1),
                    'operating_income': np.random.lognormal(8, 1) * max(0.1, operating_margin/100),
                    'net_income': np.random.lognormal(7.5, 1) * max(0.05, operating_margin/100 * 0.8),
                    'total_assets': np.random.lognormal(10.5, 1) * (1 + age * 0.15),
                    'shareholders_equity': np.random.lognormal(9.8, 1) * (1 + age * 0.12),
                    'total_debt': np.random.lognormal(9, 1) * (1 + age * 0.1),
                    'current_assets': np.random.lognormal(9.5, 1),
                    'current_liabilities': np.random.lognormal(8.8, 1),
                    'inventory': np.random.lognormal(8, 1),
                    'cogs': np.random.lognormal(9.5, 1),
                    'rd_expense': np.random.lognormal(6, 1) * max(0.01, np.random.random() * 0.1),
                    'capex': np.random.lognormal(7, 1),
                    'employee_count': int(np.random.lognormal(5, 1) * (1 + age * 0.05)),
                    'interest_expense': np.random.lognormal(5, 1) * 0.1
                })
    
    financial_df = pd.DataFrame(financial_data)
    
    return financial_df, company_info, market_categories


if __name__ == "__main__":
    # 使用例
    print("A2AI 企業ライフサイクル段階遷移分析器のテスト実行")
    
    # ダミーデータ生成
    financial_data, company_info, market_categories = generate_dummy_data()
    
    # 分析器初期化
    analyzer = StageTransitionAnalyzer()
    
    # 分析実行
    try:
        results = analyzer.analyze_stage_transitions(
            financial_data=financial_data,
            company_info=company_info,
            market_categories=market_categories
        )
        
        print(f"\n=== 分析完了 ===")
        print(f"分析対象企業数: {results['summary_statistics']['total_companies_analyzed']}")
        print(f"遷移イベント総数: {results['summary_statistics']['total_transition_events']}")
        print(f"企業あたり平均遷移回数: {results['summary_statistics']['avg_transitions_per_company']:.2f}")
        
        print(f"\n=== 段階分布 ===")
        for stage, count in results['summary_statistics']['stage_distribution'].items():
            print(f"{stage}: {count}回")
        
        print(f"\n=== 最頻遷移パターン ===")
        for pattern, count in results['summary_statistics']['most_common_transitions'][:3]:
            print(f"{pattern}: {count}回")
        
        # 予測例
        if results['summary_statistics']['total_companies_analyzed'] > 0:
            sample_company = list(analyzer.stage_classifications.keys())[0]
            prediction = analyzer.predict_future_transitions(sample_company, 5)
            print(f"\n=== 企業 {sample_company} の5年予測 ===")
            print(f"現在段階: {prediction['current_stage']}")
            print(f"予測パス: {' → '.join(prediction['most_likely_path'])}")
            print(f"信頼度: {prediction['confidence_score']:.2f}")
        
        print("\nテスト完了")
        
    except Exception as e:
        print(f"エラー発生: {str(e)}")
        import traceback
        traceback.print_exc()