"""
A2AI Cox回帰生存分析モデル
企業の生存期間と23の要因項目の関係を分析し、各市場カテゴリ（高シェア/低下/失失）での
企業存続に影響する要因を特定する。

主な機能:
- Cox比例ハザードモデルによる生存分析
- 市場カテゴリ別の生存要因分析
- ハザード比による要因項目の影響度定量化
- 時間依存共変量の考慮
- 生存曲線の予測と可視化
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import warnings
from pathlib import Path
import joblib
from datetime import datetime, timedelta

# 統計・生存分析ライブラリ
try:
    from lifelines import CoxPHFitter
    from lifelines.statistics import logrank_test, multivariate_logrank_test
    from lifelines.plotting import plot_lifetimes
    from lifelines.utils import concordance_index, median_survival_times
except ImportError:
    warnings.warn("lifelines not installed. Please install with: pip install lifelines")

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from scipy import stats

# A2AI内部モジュール
from ..base_model import BaseModel
from ...utils.survival_utils import SurvivalUtils
from ...utils.statistical_utils import StatisticalUtils
from ...utils.logging_utils import setup_logger

class CoxRegressionModel(BaseModel):
    """
    Cox比例ハザードモデルクラス
    
    企業の生存期間を目的変数とし、23の拡張要因項目を説明変数として
    各要因が企業の生存（事業継続）に与える影響を分析する。
    """
    
    def __init__(self, 
                    market_category: Optional[str] = None,
                    penalizer: float = 0.1,
                    l1_ratio: float = 0.0,
                    alpha: float = 0.05,
                    tie_method: str = 'Efron',
                    robust: bool = True,
                    step_size: Optional[float] = None):
        """
        Cox回帰モデルの初期化
        
        Args:
            market_category: 分析対象市場カテゴリ ('high_share', 'declining', 'lost', None)
            penalizer: 正則化パラメータ（L2正則化の強度）
            l1_ratio: L1正則化の比率（0.0-1.0、ElasticNet用）
            alpha: 統計的有意水準
            tie_method: 同時事象の処理方法 ('Efron', 'Breslow')
            robust: ロバスト共分散行列の使用
            step_size: 最適化ステップサイズ
        """
        super().__init__()
        
        self.market_category = market_category
        self.penalizer = penalizer
        self.l1_ratio = l1_ratio
        self.alpha = alpha
        self.tie_method = tie_method
        self.robust = robust
        self.step_size = step_size
        
        # モデルインスタンス
        self.cox_model = None
        self.scaler = StandardScaler()
        
        # 分析結果格納
        self.survival_summary = {}
        self.hazard_ratios = {}
        self.concordance_scores = {}
        self.market_comparisons = {}
        
        # ログ設定
        self.logger = setup_logger(f"CoxRegression_{market_category or 'All'}")
        
        # 生存分析ユーティリティ
        self.survival_utils = SurvivalUtils()
        self.stats_utils = StatisticalUtils()
        
        self.logger.info(f"Cox回帰モデル初期化完了 - 市場カテゴリ: {market_category}")
    
    def prepare_survival_data(self, 
                            financial_data: pd.DataFrame, 
                            company_events: pd.DataFrame) -> pd.DataFrame:
        """
        生存分析用データの準備
        
        Args:
            financial_data: 財務諸表データ（150社×40年分）
            company_events: 企業イベントデータ（設立/消滅/分社等）
        
        Returns:
            生存分析用に整形されたDataFrame
        """
        self.logger.info("生存分析用データ準備開始")
        
        try:
            # 企業ごとの生存期間計算
            survival_data = []
            
            for company_id in financial_data['company_id'].unique():
                company_financials = financial_data[
                    financial_data['company_id'] == company_id
                ].sort_values('year')
                
                company_events_subset = company_events[
                    company_events['company_id'] == company_id
                ]
                
                # 生存期間とイベント情報の取得
                survival_info = self._calculate_survival_period(
                    company_financials, company_events_subset
                )
                
                if survival_info is not None:
                    # 23の拡張要因項目の計算
                    factor_features = self._calculate_factor_features(company_financials)
                    
                    # 市場カテゴリ情報の追加
                    market_info = self._get_market_category(company_id, company_events_subset)
                    
                    # データ結合
                    company_data = {
                        'company_id': company_id,
                        'duration': survival_info['duration'],
                        'event_observed': survival_info['event_observed'],
                        'market_category': market_info['category'],
                        'start_year': survival_info['start_year'],
                        'end_year': survival_info['end_year'],
                        **factor_features,
                        **market_info
                    }
                    
                    survival_data.append(company_data)
            
            survival_df = pd.DataFrame(survival_data)
            
            # 市場カテゴリでフィルタリング（指定がある場合）
            if self.market_category:
                survival_df = survival_df[
                    survival_df['market_category'] == self.market_category
                ]
            
            self.logger.info(f"生存分析データ準備完了 - {len(survival_df)}社のデータ")
            return survival_df
            
        except Exception as e:
            self.logger.error(f"生存分析データ準備エラー: {str(e)}")
            raise
    
    def _calculate_survival_period(self, 
                                    company_financials: pd.DataFrame, 
                                    company_events: pd.DataFrame) -> Optional[Dict]:
        """
        企業の生存期間計算
        
        Args:
            company_financials: 企業の財務データ
            company_events: 企業のイベントデータ
        
        Returns:
            生存期間情報の辞書
        """
        try:
            start_year = company_financials['year'].min()
            
            # 企業消滅イベントの確認
            extinction_events = company_events[
                company_events['event_type'].isin([
                    'bankruptcy', 'merger', 'acquisition', 'dissolution'
                ])
            ]
            
            if not extinction_events.empty:
                # 消滅イベントあり
                end_year = extinction_events['event_date'].min()
                event_observed = 1
            else:
                # 観測期間終了まで生存
                end_year = company_financials['year'].max()
                event_observed = 0
            
            duration = end_year - start_year
            
            # 最低観測期間のチェック
            if duration < 1:
                return None
            
            return {
                'duration': duration,
                'event_observed': event_observed,
                'start_year': start_year,
                'end_year': end_year
            }
            
        except Exception as e:
            self.logger.warning(f"生存期間計算エラー: {str(e)}")
            return None
    
    def _calculate_factor_features(self, company_financials: pd.DataFrame) -> Dict:
        """
        23の拡張要因項目の計算
        
        各評価項目（売上高、成長率等）に対応する23の要因項目：
        - 従来20項目 + 企業年齢 + 市場参入時期 + 親会社依存度
        
        Args:
            company_financials: 企業の財務データ
        
        Returns:
            要因項目の辞書
        """
        try:
            # 最新期間データを使用（代表値として）
            recent_data = company_financials.tail(5).mean()  # 直近5年平均
            
            # 従来20項目の計算
            factors = {
                # 投資・資産関連
                'tangible_fixed_assets': recent_data.get('tangible_fixed_assets', 0),
                'capex_amount': recent_data.get('capex_amount', 0),
                'rd_expenses': recent_data.get('rd_expenses', 0),
                'intangible_assets': recent_data.get('intangible_assets', 0),
                'investment_securities': recent_data.get('investment_securities', 0),
                
                # 人的資源関連
                'employee_count': recent_data.get('employee_count', 0),
                'average_salary': recent_data.get('average_salary', 0),
                'retirement_benefit_cost': recent_data.get('retirement_benefit_cost', 0),
                'welfare_expenses': recent_data.get('welfare_expenses', 0),
                
                # 運転資本・効率性関連
                'accounts_receivable': recent_data.get('accounts_receivable', 0),
                'inventory': recent_data.get('inventory', 0),
                'total_assets': recent_data.get('total_assets', 0),
                'receivables_turnover': recent_data.get('receivables_turnover', 0),
                'inventory_turnover': recent_data.get('inventory_turnover', 0),
                
                # 事業展開関連
                'overseas_sales_ratio': recent_data.get('overseas_sales_ratio', 0),
                'segment_count': recent_data.get('segment_count', 1),
                'sga_expenses': recent_data.get('sga_expenses', 0),
                'advertising_expenses': recent_data.get('advertising_expenses', 0),
                'non_operating_income': recent_data.get('non_operating_income', 0),
                'order_backlog': recent_data.get('order_backlog', 0),
            }
            
            # 新規3項目の追加
            # 21. 企業年齢（設立からの経過年数）
            factors['company_age'] = len(company_financials)
            
            # 22. 市場参入時期（先発/後発効果）
            # 業界平均設立年との比較で算出
            factors['market_entry_timing'] = self._calculate_entry_timing(company_financials)
            
            # 23. 親会社依存度（分社企業の場合）
            factors['parent_dependency'] = self._calculate_parent_dependency(company_financials)
            
            return factors
            
        except Exception as e:
            self.logger.warning(f"要因項目計算エラー: {str(e)}")
            return {}
    
    def _calculate_entry_timing(self, company_financials: pd.DataFrame) -> float:
        """
        市場参入時期の計算（先発/後発効果）
        
        Returns:
            参入時期指標（正値：先発、負値：後発）
        """
        # 簡易実装：業界平均との比較
        # 実際の実装では業界データとの比較が必要
        start_year = company_financials['year'].min()
        industry_avg_start = 1980  # 仮の業界平均設立年
        
        return industry_avg_start - start_year
    
    def _calculate_parent_dependency(self, company_financials: pd.DataFrame) -> float:
        """
        親会社依存度の計算
        
        Returns:
            親会社依存度（0-1、高いほど依存度大）
        """
        # 簡易実装：関連会社取引比率等で算出
        # 実際の実装では詳細な関連会社データが必要
        
        # 投資有価証券比率を代理指標として使用
        recent_data = company_financials.tail(1).iloc[0]
        investment_ratio = (recent_data.get('investment_securities', 0) / 
                            max(recent_data.get('total_assets', 1), 1))
        
        return min(investment_ratio, 1.0)
    
    def _get_market_category(self, 
                            company_id: str, 
                            company_events: pd.DataFrame) -> Dict:
        """
        市場カテゴリ情報の取得
        
        Returns:
            市場カテゴリ情報の辞書
        """
        # 企業リストから市場カテゴリを判定
        # 実際の実装では外部データとの連携が必要
        
        if company_id in ['ファナック', '村田製作所', 'キーエンス', 'オリンパス']:
            category = 'high_share'
        elif company_id in ['トヨタ自動車', '日産自動車', 'パナソニック']:
            category = 'declining'
        else:
            category = 'lost'
        
        return {
            'category': category,
            'market_sector': self._identify_market_sector(company_id)
        }
    
    def _identify_market_sector(self, company_id: str) -> str:
        """市場セクターの特定"""
        # 簡易実装
        sector_mapping = {
            'ファナック': 'robotics',
            '村田製作所': 'electronic_materials',
            'キーエンス': 'precision_instruments',
            'オリンパス': 'medical_endoscopy',
            'トヨタ自動車': 'automotive',
            '日産自動車': 'automotive',
            'パナソニック': 'consumer_electronics'
        }
        return sector_mapping.get(company_id, 'other')
    
    def fit(self, survival_data: pd.DataFrame) -> 'CoxRegressionModel':
        """
        Cox回帰モデルの学習
        
        Args:
            survival_data: 生存分析用データ
        
        Returns:
            学習済みモデル
        """
        self.logger.info("Cox回帰モデル学習開始")
        
        try:
            # 特徴量の準備
            feature_columns = [col for col in survival_data.columns 
                                if col not in ['company_id', 'duration', 'event_observed',
                                            'market_category', 'start_year', 'end_year']]
            
            X = survival_data[feature_columns]
            
            # 標準化
            X_scaled = pd.DataFrame(
                self.scaler.fit_transform(X),
                columns=X.columns,
                index=X.index
            )
            
            # 生存期間とイベント指標の準備
            X_scaled['duration'] = survival_data['duration']
            X_scaled['event_observed'] = survival_data['event_observed']
            
            # Cox回帰モデルの初期化と学習
            self.cox_model = CoxPHFitter(
                penalizer=self.penalizer,
                l1_ratio=self.l1_ratio,
                alpha=self.alpha,
                tie_method=self.tie_method
            )
            
            self.cox_model.fit(
                X_scaled, 
                duration_col='duration',
                event_col='event_observed',
                robust=self.robust,
                step_size=self.step_size
            )
            
            # 学習結果の保存
            self._save_training_results(X_scaled, survival_data)
            
            self.is_fitted = True
            self.logger.info("Cox回帰モデル学習完了")
            
            return self
            
        except Exception as e:
            self.logger.error(f"Cox回帰モデル学習エラー: {str(e)}")
            raise
    
    def _save_training_results(self, 
                                X_scaled: pd.DataFrame, 
                                original_data: pd.DataFrame):
        """学習結果の保存"""
        try:
            # モデル要約統計
            self.survival_summary = {
                'n_subjects': len(X_scaled),
                'n_events': X_scaled['event_observed'].sum(),
                'median_survival_time': X_scaled['duration'].median(),
                'concordance_index': self.cox_model.concordance_index_,
                'log_likelihood': self.cox_model.log_likelihood_,
                'AIC': self.cox_model.AIC_,
                'partial_AIC': self.cox_model.AIC_partial_
            }
            
            # ハザード比の計算
            self.hazard_ratios = {
                'hazard_ratios': self.cox_model.hazard_ratios_,
                'confidence_intervals': self.cox_model.confidence_intervals_,
                'p_values': self.cox_model.summary['p']
            }
            
            # 市場カテゴリ別分析
            if 'market_category' in original_data.columns:
                self._analyze_market_categories(original_data)
            
        except Exception as e:
            self.logger.warning(f"学習結果保存エラー: {str(e)}")
    
    def _analyze_market_categories(self, survival_data: pd.DataFrame):
        """市場カテゴリ別の生存分析"""
        try:
            categories = survival_data['market_category'].unique()
            
            for category in categories:
                category_data = survival_data[
                    survival_data['market_category'] == category
                ]
                
                self.market_comparisons[category] = {
                    'n_companies': len(category_data),
                    'n_events': category_data['event_observed'].sum(),
                    'median_survival': category_data['duration'].median(),
                    'event_rate': category_data['event_observed'].mean()
                }
            
            # カテゴリ間の統計的比較
            if len(categories) > 1:
                self._perform_logrank_tests(survival_data)
                
        except Exception as e:
            self.logger.warning(f"市場カテゴリ分析エラー: {str(e)}")
    
    def _perform_logrank_tests(self, survival_data: pd.DataFrame):
        """ログランク検定による市場カテゴリ間比較"""
        try:
            categories = survival_data['market_category'].unique()
            
            # 全カテゴリの多変量ログランク検定
            if len(categories) > 2:
                test_result = multivariate_logrank_test(
                    survival_data['duration'],
                    survival_data['market_category'],
                    survival_data['event_observed']
                )
                
                self.market_comparisons['multivariate_logrank'] = {
                    'test_statistic': test_result.test_statistic,
                    'p_value': test_result.p_value,
                    'is_significant': test_result.p_value < self.alpha
                }
            
            # ペアワイズ比較
            pairwise_results = {}
            for i, cat1 in enumerate(categories):
                for cat2 in categories[i+1:]:
                    data1 = survival_data[survival_data['market_category'] == cat1]
                    data2 = survival_data[survival_data['market_category'] == cat2]
                    
                    test_result = logrank_test(
                        data1['duration'], data2['duration'],
                        data1['event_observed'], data2['event_observed']
                    )
                    
                    pairwise_results[f"{cat1}_vs_{cat2}"] = {
                        'test_statistic': test_result.test_statistic,
                        'p_value': test_result.p_value,
                        'is_significant': test_result.p_value < self.alpha
                    }
            
            self.market_comparisons['pairwise_tests'] = pairwise_results
            
        except Exception as e:
            self.logger.warning(f"ログランク検定エラー: {str(e)}")
    
    def predict_survival_function(self, 
                                X: pd.DataFrame, 
                                times: Optional[np.ndarray] = None) -> pd.DataFrame:
        """
        生存関数の予測
        
        Args:
            X: 予測対象の特徴量
            times: 予測時点（Noneの場合は自動設定）
        
        Returns:
            生存確率の予測結果
        """
        if not self.is_fitted:
            raise ValueError("モデルが学習されていません")
        
        try:
            # 特徴量の標準化
            feature_columns = [col for col in X.columns 
                                if col in self.scaler.feature_names_in_]
            X_scaled = pd.DataFrame(
                self.scaler.transform(X[feature_columns]),
                columns=feature_columns,
                index=X.index
            )
            
            # 生存関数の予測
            survival_functions = self.cox_model.predict_survival_function(
                X_scaled, times=times
            )
            
            return survival_functions
            
        except Exception as e:
            self.logger.error(f"生存関数予測エラー: {str(e)}")
            raise
    
    def predict_partial_hazard(self, X: pd.DataFrame) -> pd.Series:
        """
        部分ハザードの予測
        
        Args:
            X: 予測対象の特徴量
        
        Returns:
            部分ハザードの予測値
        """
        if not self.is_fitted:
            raise ValueError("モデルが学習されていません")
        
        try:
            # 特徴量の標準化
            feature_columns = [col for col in X.columns 
                                if col in self.scaler.feature_names_in_]
            X_scaled = pd.DataFrame(
                self.scaler.transform(X[feature_columns]),
                columns=feature_columns,
                index=X.index
            )
            
            # 部分ハザードの予測
            partial_hazards = self.cox_model.predict_partial_hazard(X_scaled)
            
            return partial_hazards
            
        except Exception as e:
            self.logger.error(f"部分ハザード予測エラー: {str(e)}")
            raise
    
    def get_feature_importance(self, top_n: Optional[int] = None) -> pd.DataFrame:
        """
        特徴量重要度の取得（ハザード比基準）
        
        Args:
            top_n: 上位N個の特徴量のみ取得
        
        Returns:
            特徴量重要度のDataFrame
        """
        if not self.is_fitted:
            raise ValueError("モデルが学習されていません")
        
        try:
            importance_df = pd.DataFrame({
                'feature': self.cox_model.params_.index,
                'hazard_ratio': self.cox_model.hazard_ratios_,
                'coefficient': self.cox_model.params_,
                'p_value': self.cox_model.summary['p'],
                'ci_lower': self.cox_model.confidence_intervals_.iloc[:, 0],
                'ci_upper': self.cox_model.confidence_intervals_.iloc[:, 1]
            })
            
            # ハザード比の絶対値でソート
            importance_df['abs_log_hazard_ratio'] = np.abs(
                np.log(importance_df['hazard_ratio'])
            )
            importance_df = importance_df.sort_values(
                'abs_log_hazard_ratio', ascending=False
            )
            
            # 統計的有意性の判定
            importance_df['is_significant'] = importance_df['p_value'] < self.alpha
            
            if top_n:
                importance_df = importance_df.head(top_n)
            
            return importance_df.reset_index(drop=True)
            
        except Exception as e:
            self.logger.error(f"特徴量重要度取得エラー: {str(e)}")
            raise
    
    def cross_validate(self, 
                        survival_data: pd.DataFrame, 
                        cv_folds: int = 5) -> Dict:
        """
        交差検証による模型評価
        
        Args:
            survival_data: 生存分析用データ
            cv_folds: 交差検証の分割数
        
        Returns:
            交差検証結果の辞書
        """
        self.logger.info(f"{cv_folds}-fold交差検証開始")
        
        try:
            kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
            
            cv_scores = {
                'concordance_indices': [],
                'log_likelihoods': [],
                'partial_aic_scores': []
            }
            
            feature_columns = [col for col in survival_data.columns 
                                if col not in ['company_id', 'duration', 'event_observed',
                                            'market_category', 'start_year', 'end_year']]
            
            X = survival_data[feature_columns]
            
            for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
                self.logger.info(f"交差検証 Fold {fold+1}/{cv_folds}")
                
                # 訓練・検証データ分割
                train_data = survival_data.iloc[train_idx]
                val_data = survival_data.iloc[val_idx]
                
                # 一時的なモデル作成・学習
                temp_model = CoxRegressionModel(
                    market_category=self.market_category,
                    penalizer=self.penalizer,
                    l1_ratio=self.l1_ratio,
                    alpha=self.alpha
                )
                
                temp_model.fit(train_data)
                
                # 検証データでの評価
                val_X = val_data[feature_columns]
                val_X_scaled = pd.DataFrame(
                    temp_model.scaler.transform(val_X),
                    columns=val_X.columns
                )
                val_X_scaled['duration'] = val_data['duration'].values
                val_X_scaled['event_observed'] = val_data['event_observed'].values
                
                # Concordance index計算
                c_index = concordance_index(
                    val_data['duration'],
                    -temp_model.cox_model.predict_partial_hazard(val_X_scaled.drop(['duration', 'event_observed'], axis=1)),
                    val_data['event_observed']
                )
                
                cv_scores['concordance_indices'].append(c_index)
                cv_scores['log_likelihoods'].append(temp_model.cox_model.log_likelihood_)
                cv_scores['partial_aic_scores'].append(temp_model.cox_model.AIC_partial_)
            
            # 交差検証結果の集計
            cv_results = {
                'mean_concordance_index': np.mean(cv_scores['concordance_indices']),
                'std_concordance_index': np.std(cv_scores['concordance_indices']),
                'mean_log_likelihood': np.mean(cv_scores['log_likelihoods']),
                'std_log_likelihood': np.std(cv_scores['log_likelihoods']),
                'mean_partial_aic': np.mean(cv_scores['partial_aic_scores']),
                'std_partial_aic': np.std(cv_scores['partial_aic_scores']),
                'cv_scores': cv_scores
            }
            
            self.logger.info(f"交差検証完了 - 平均C-index: {cv_results['mean_concordance_index']:.3f}")
            
            return cv_results
            
        except Exception as e:
            self.logger.error(f"交差検証エラー: {str(e)}")
            raise
    
    def save_model(self, filepath: str):
        """モデルの保存"""
        if not self.is_fitted:
            raise ValueError("モデルが学習されていません")
        
        try:
            model_data = {
                'cox_model': self.cox_model,
                'scaler': self.scaler,
                'survival_summary': self.survival_summary,
                'hazard_ratios': self.hazard_ratios,
                'market_comparisons': self.market_comparisons,
                'market_category': self.market_category,
                'model_params': {
                    'penalizer': self.penalizer,
                    'l1_ratio': self.l1_ratio,
                    'alpha': self.alpha,
                    'tie_method': self.tie_method,
                    'robust': self.robust
                },
                'training_timestamp': datetime.now().isoformat()
            }
            
            joblib.dump(model_data, filepath)
            self.logger.info(f"モデル保存完了: {filepath}")
            
        except Exception as e:
            self.logger.error(f"モデル保存エラー: {str(e)}")
            raise
    
    @classmethod
    def load_model(cls, filepath: str) -> 'CoxRegressionModel':
        """モデルの読み込み"""
        try:
            model_data = joblib.load(filepath)
            
            # インスタンス作成
            instance = cls(
                market_category=model_data['market_category'],
                **model_data['model_params']
            )
            
            # 学習済みデータの復元
            instance.cox_model = model_data['cox_model']
            instance.scaler = model_data['scaler']
            instance.survival_summary = model_data['survival_summary']
            instance.hazard_ratios = model_data['hazard_ratios']
            instance.market_comparisons = model_data['market_comparisons']
            instance.is_fitted = True
            
            instance.logger.info(f"モデル読み込み完了: {filepath}")
            
            return instance
            
        except Exception as e:
            raise ValueError(f"モデル読み込みエラー: {str(e)}")
    
    def get_model_summary(self) -> Dict:
        """モデル要約統計の取得"""
        if not self.is_fitted:
            raise ValueError("モデルが学習されていません")
        
        return {
            'model_type': 'Cox Proportional Hazards',
            'market_category': self.market_category,
            'survival_summary': self.survival_summary,
            'top_risk_factors': self.get_feature_importance(top_n=10),
            'market_comparisons': self.market_comparisons,
            'model_performance': {
                'concordance_index': self.survival_summary.get('concordance_index', 0),
                'log_likelihood': self.survival_summary.get('log_likelihood', 0),
                'AIC': self.survival_summary.get('AIC', 0)
            }
        }
    
    def plot_survival_curves(self, 
                            X: Optional[pd.DataFrame] = None, 
                            stratify_by: Optional[str] = None,
                            save_path: Optional[str] = None) -> plt.Figure:
        """
        生存曲線のプロット
        
        Args:
            X: プロット対象の特徴量（Noneの場合は全体）
            stratify_by: 層別化変数
            save_path: 保存パス
        
        Returns:
            matplotlib Figure
        """
        if not self.is_fitted:
            raise ValueError("モデルが学習されていません")
        
        try:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            if X is not None:
                # 特定データの生存曲線
                survival_functions = self.predict_survival_function(X)
                
                for i, (idx, survival_func) in enumerate(survival_functions.iterrows()):
                    if i < 10:  # 最大10本まで表示
                        ax.plot(survival_func.index, survival_func.values, 
                                label=f'Company {idx}', alpha=0.7)
                
            else:
                # 全体の生存曲線（ベースライン）
                if hasattr(self.cox_model, 'baseline_survival_'):
                    baseline = self.cox_model.baseline_survival_
                    ax.plot(baseline.index, baseline.values, 
                            label='Baseline Survival', linewidth=2, color='red')
            
            if stratify_by and hasattr(self, 'market_comparisons'):
                # 市場カテゴリ別の生存曲線
                for category in self.market_comparisons.keys():
                    if category not in ['multivariate_logrank', 'pairwise_tests']:
                        # 簡易実装：カテゴリ別プロット
                        pass
            
            ax.set_xlabel('Time (Years)', fontsize=12)
            ax.set_ylabel('Survival Probability', fontsize=12)
            ax.set_title('Corporate Survival Curves - A2AI Analysis', fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            plt.tight_layout()
            
            if save_path:
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"生存曲線プロット保存: {save_path}")
            
            return fig
            
        except Exception as e:
            self.logger.error(f"生存曲線プロットエラー: {str(e)}")
            raise
    
    def plot_hazard_ratios(self, 
                            top_n: int = 15,
                            save_path: Optional[str] = None) -> plt.Figure:
        """
        ハザード比のフォレストプロット
        
        Args:
            top_n: 表示する上位要因数
            save_path: 保存パス
        
        Returns:
            matplotlib Figure
        """
        if not self.is_fitted:
            raise ValueError("モデルが学習されていません")
        
        try:
            importance_df = self.get_feature_importance(top_n=top_n)
            
            fig, ax = plt.subplots(figsize=(10, top_n * 0.6))
            
            # ハザード比とその信頼区間をプロット
            y_pos = np.arange(len(importance_df))
            
            # 点推定値
            colors = ['red' if hr > 1 else 'blue' for hr in importance_df['hazard_ratio']]
            ax.scatter(importance_df['hazard_ratio'], y_pos, 
                        color=colors, s=50, alpha=0.7)
            
            # 信頼区間
            for i, (_, row) in enumerate(importance_df.iterrows()):
                ax.plot([row['ci_lower'], row['ci_upper']], [i, i], 
                        color=colors[i], alpha=0.5, linewidth=2)
            
            # 基準線（HR=1）
            ax.axvline(x=1, color='gray', linestyle='--', alpha=0.7)
            
            # 軸とラベル設定
            ax.set_yticks(y_pos)
            ax.set_yticklabels([self._format_feature_name(name) 
                                for name in importance_df['feature']])
            ax.set_xlabel('Hazard Ratio (95% CI)', fontsize=12)
            ax.set_title('Corporate Survival Risk Factors - A2AI Analysis', fontsize=14)
            
            # 統計的有意性の表示
            for i, (_, row) in enumerate(importance_df.iterrows()):
                significance = '***' if row['p_value'] < 0.001 else \
                                '**' if row['p_value'] < 0.01 else \
                                '*' if row['p_value'] < 0.05 else ''
                if significance:
                    ax.text(ax.get_xlim()[1] * 0.95, i, significance, 
                            va='center', ha='right', fontsize=10)
            
            ax.grid(True, alpha=0.3, axis='x')
            plt.tight_layout()
            
            if save_path:
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"ハザード比プロット保存: {save_path}")
            
            return fig
            
        except Exception as e:
            self.logger.error(f"ハザード比プロットエラー: {str(e)}")
            raise
    
    def _format_feature_name(self, feature_name: str) -> str:
        """特徴量名の整形"""
        name_mapping = {
            'tangible_fixed_assets': 'Tangible Fixed Assets',
            'capex_amount': 'Capital Expenditure',
            'rd_expenses': 'R&D Expenses',
            'intangible_assets': 'Intangible Assets',
            'investment_securities': 'Investment Securities',
            'employee_count': 'Employee Count',
            'average_salary': 'Average Salary',
            'retirement_benefit_cost': 'Retirement Benefits',
            'welfare_expenses': 'Welfare Expenses',
            'accounts_receivable': 'Accounts Receivable',
            'inventory': 'Inventory',
            'total_assets': 'Total Assets',
            'receivables_turnover': 'Receivables Turnover',
            'inventory_turnover': 'Inventory Turnover',
            'overseas_sales_ratio': 'Overseas Sales Ratio',
            'segment_count': 'Business Segments',
            'sga_expenses': 'SG&A Expenses',
            'advertising_expenses': 'Advertising Expenses',
            'non_operating_income': 'Non-Operating Income',
            'order_backlog': 'Order Backlog',
            'company_age': 'Company Age',
            'market_entry_timing': 'Market Entry Timing',
            'parent_dependency': 'Parent Company Dependency'
        }
        return name_mapping.get(feature_name, feature_name.replace('_', ' ').title())
    
    def generate_survival_report(self, 
                                output_path: str,
                                include_plots: bool = True) -> str:
        """
        生存分析レポートの生成
        
        Args:
            output_path: レポート出力パス
            include_plots: グラフを含むかどうか
        
        Returns:
            生成されたレポートファイルのパス
        """
        if not self.is_fitted:
            raise ValueError("モデルが学習されていません")
        
        try:
            from datetime import datetime
            
            report_content = []
            report_content.append("# A2AI Corporate Survival Analysis Report")
            report_content.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report_content.append(f"Market Category: {self.market_category or 'All Markets'}")
            report_content.append("")
            
            # モデル要約
            report_content.append("## Model Summary")
            report_content.append(f"- Model Type: Cox Proportional Hazards")
            report_content.append(f"- Number of Companies: {self.survival_summary.get('n_subjects', 'N/A')}")
            report_content.append(f"- Number of Events (Extinctions): {self.survival_summary.get('n_events', 'N/A')}")
            report_content.append(f"- Median Survival Time: {self.survival_summary.get('median_survival_time', 'N/A'):.1f} years")
            report_content.append(f"- Concordance Index: {self.survival_summary.get('concordance_index', 'N/A'):.3f}")
            report_content.append("")
            
            # 主要リスク要因
            report_content.append("## Top Risk Factors")
            importance_df = self.get_feature_importance(top_n=10)
            report_content.append("| Factor | Hazard Ratio | 95% CI | P-Value | Significance |")
            report_content.append("|--------|-------------|--------|---------|-------------|")
            
            for _, row in importance_df.iterrows():
                significance = '***' if row['p_value'] < 0.001 else \
                                '**' if row['p_value'] < 0.01 else \
                                '*' if row['p_value'] < 0.05 else ''
                
                report_content.append(
                    f"| {self._format_feature_name(row['feature'])} | "
                    f"{row['hazard_ratio']:.3f} | "
                    f"({row['ci_lower']:.3f}, {row['ci_upper']:.3f}) | "
                    f"{row['p_value']:.4f} | {significance} |"
                )
            
            report_content.append("")
            
            # 市場比較
            if self.market_comparisons:
                report_content.append("## Market Category Analysis")
                for category, stats in self.market_comparisons.items():
                    if category not in ['multivariate_logrank', 'pairwise_tests']:
                        report_content.append(f"### {category.replace('_', ' ').title()} Market")
                        report_content.append(f"- Companies: {stats.get('n_companies', 'N/A')}")
                        report_content.append(f"- Extinctions: {stats.get('n_events', 'N/A')}")
                        report_content.append(f"- Median Survival: {stats.get('median_survival', 'N/A'):.1f} years")
                        report_content.append(f"- Event Rate: {stats.get('event_rate', 'N/A'):.3f}")
                        report_content.append("")
            
            # 統計的検定結果
            if 'multivariate_logrank' in self.market_comparisons:
                logrank_result = self.market_comparisons['multivariate_logrank']
                report_content.append("## Statistical Tests")
                report_content.append("### Multivariate Log-rank Test")
                report_content.append(f"- Test Statistic: {logrank_result.get('test_statistic', 'N/A'):.3f}")
                report_content.append(f"- P-Value: {logrank_result.get('p_value', 'N/A'):.4f}")
                report_content.append(f"- Significant: {logrank_result.get('is_significant', False)}")
                report_content.append("")
            
            # 解釈と提言
            report_content.append("## Key Insights")
            report_content.append(self._generate_insights())
            report_content.append("")
            
            report_content.append("## Business Implications")
            report_content.append(self._generate_business_implications())
            
            # レポートファイルの書き出し
            report_text = "\n".join(report_content)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            
            self.logger.info(f"生存分析レポート生成完了: {output_path}")
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"レポート生成エラー: {str(e)}")
            raise
    
    def _generate_insights(self) -> str:
        """主要インサイトの生成"""
        insights = []
        
        try:
            importance_df = self.get_feature_importance(top_n=5)
            
            # 最も影響の大きい要因
            top_factor = importance_df.iloc[0]
            if top_factor['hazard_ratio'] > 1:
                insights.append(
                    f"- **{self._format_feature_name(top_factor['feature'])}**が企業存続に最も大きなリスクをもたらす要因です "
                    f"(ハザード比: {top_factor['hazard_ratio']:.2f})"
                )
            else:
                insights.append(
                    f"- **{self._format_feature_name(top_factor['feature'])}**が企業存続に最も大きな保護効果をもたらす要因です "
                    f"(ハザード比: {top_factor['hazard_ratio']:.2f})"
                )
            
            # 市場カテゴリ別の特徴
            if self.market_comparisons and len(self.market_comparisons) > 1:
                categories = [(k, v) for k, v in self.market_comparisons.items() 
                                if k not in ['multivariate_logrank', 'pairwise_tests']]
                
                if categories:
                    # 最もリスクの高い市場
                    highest_risk = max(categories, key=lambda x: x[1].get('event_rate', 0))
                    insights.append(
                        f"- **{highest_risk[0].replace('_', ' ').title()}市場**が最も企業消滅リスクが高い "
                        f"(消滅率: {highest_risk[1].get('event_rate', 0):.1%})"
                    )
            
            # モデル性能に関するコメント
            c_index = self.survival_summary.get('concordance_index', 0)
            if c_index > 0.7:
                insights.append("- 高い予測精度を達成しており、要因分析の信頼性が確保されています")
            elif c_index > 0.6:
                insights.append("- 中程度の予測精度であり、さらなる要因探索が有効と考えられます")
            else:
                insights.append("- 予測精度が限定的であり、追加的な説明変数の検討が必要です")
                
        except Exception as e:
            insights.append(f"インサイト生成エラー: {str(e)}")
        
        return "\n".join(insights) if insights else "分析結果の解釈データが不足しています"
    
    def _generate_business_implications(self) -> str:
        """ビジネス含意の生成"""
        implications = []
        
        try:
            importance_df = self.get_feature_importance(top_n=3)
            
            implications.append("### 企業経営への示唆:")
            
            for _, factor in importance_df.iterrows():
                factor_name = self._format_feature_name(factor['feature'])
                
                if factor['hazard_ratio'] > 1:
                    if 'debt' in factor['feature'].lower() or 'leverage' in factor['feature'].lower():
                        implications.append(f"- **{factor_name}**の管理強化による財務安定性の向上が重要")
                    elif 'rd' in factor['feature'].lower() or 'innovation' in factor['feature'].lower():
                        implications.append(f"- **{factor_name}**の過度な投資は短期的にリスクを増加させる可能性")
                    else:
                        implications.append(f"- **{factor_name}**の最適化が企業存続確率向上の鍵")
                else:
                    implications.append(f"- **{factor_name}**の強化が企業の長期存続に有効")
            
            implications.append("")
            implications.append("### 政策・産業支援への示唆:")
            
            if self.market_category == 'declining':
                implications.append("- 衰退市場では企業の事業転換支援策が効果的")
            elif self.market_category == 'lost':
                implications.append("- 失失市場では早期の事業再編支援が必要")
            else:
                implications.append("- 競争優位性の維持・強化に向けた産業政策の重要性")
                
        except Exception as e:
            implications.append(f"ビジネス含意生成エラー: {str(e)}")
        
        return "\n".join(implications) if implications else "ビジネス含意の分析データが不足しています"


# 使用例とテスト関数
def example_usage():
    """A2AI Cox回帰モデルの使用例"""
    
    # サンプルデータの作成（実際の使用では実データを使用）
    np.random.seed(42)
    n_companies = 100
    
    # 財務諸表データの模擬作成
    financial_data = pd.DataFrame({
        'company_id': [f'Company_{i}' for i in range(n_companies)] * 10,  # 10年分
        'year': list(range(2015, 2025)) * n_companies,
        'tangible_fixed_assets': np.random.lognormal(10, 1, n_companies * 10),
        'rd_expenses': np.random.lognormal(8, 1.5, n_companies * 10),
        'employee_count': np.random.poisson(500, n_companies * 10),
        'total_assets': np.random.lognormal(12, 1, n_companies * 10)
    })
    
    # 企業イベントデータの模擬作成
    n_extinctions = 20
    extinction_companies = np.random.choice(n_companies, n_extinctions, replace=False)
    
    events_data = []
    for company_idx in extinction_companies:
        events_data.append({
            'company_id': f'Company_{company_idx}',
            'event_type': np.random.choice(['bankruptcy', 'merger', 'acquisition']),
            'event_date': np.random.randint(2016, 2024)
        })
    
    company_events = pd.DataFrame(events_data)
    
    try:
        # Cox回帰モデルの初期化
        cox_model = CoxRegressionModel(
            market_category='declining',  # 衰退市場の分析
            penalizer=0.1,
            alpha=0.05
        )
        
        # 生存分析用データの準備
        survival_data = cox_model.prepare_survival_data(financial_data, company_events)
        print(f"生存分析データ準備完了: {len(survival_data)}社")
        
        # モデル学習
        cox_model.fit(survival_data)
        print("Cox回帰モデル学習完了")
        
        # 交差検証
        cv_results = cox_model.cross_validate(survival_data, cv_folds=5)
        print(f"交差検証結果 - 平均C-index: {cv_results['mean_concordance_index']:.3f}")
        
        # 特徴量重要度の確認
        importance = cox_model.get_feature_importance(top_n=10)
        print("\n主要リスク要因:")
        print(importance[['feature', 'hazard_ratio', 'p_value']].head())
        
        # モデル要約の取得
        summary = cox_model.get_model_summary()
        print(f"\nモデル要約:")
        print(f"C-index: {summary['model_performance']['concordance_index']:.3f}")
        print(f"対象企業数: {summary['survival_summary']['n_subjects']}")
        print(f"消滅イベント数: {summary['survival_summary']['n_events']}")
        
        return cox_model, survival_data
        
    except Exception as e:
        print(f"使用例実行エラー: {str(e)}")
        return None, None


if __name__ == "__main__":
    # 使用例の実行
    model, data = example_usage()