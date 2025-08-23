"""
A2AI - Advanced Financial Analysis AI
新設企業成功分析モジュール

このモジュールは新設・分社企業の成功要因を分析し、
設立後の財務パフォーマンスと各種要因項目の関係を特定します。

対象企業例:
- キオクシア (2018年設立, 東芝メモリから独立)
- デンソーウェーブ (2001年設立, デンソーから分社)
- プロテリアル (2023年設立, 日立金属から独立)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy import stats
from scipy.stats import pearsonr, spearmanr, kendalltau
import warnings
warnings.filterwarnings('ignore')

@dataclass
class EmergenceMetrics:
    """新設企業の成功指標を定義するデータクラス"""
    # 成長性指標
    revenue_growth_rate: float  # 売上高成長率
    market_share_growth: float  # 市場シェア成長率
    employee_growth_rate: float  # 従業員数成長率
    asset_growth_rate: float    # 総資産成長率
    
    # 収益性指標
    profitability_achievement_speed: float  # 黒字化達成速度 (設立からの年数)
    roe_trajectory: float       # ROE改善軌道
    operating_margin_trend: float  # 営業利益率トレンド
    
    # 競争力指標
    market_penetration_rate: float  # 市場浸透率
    innovation_output: float    # イノベーション創出度
    competitive_positioning: float  # 競争ポジション強度
    
    # 持続可能性指標
    financial_stability: float  # 財務安定性スコア
    business_model_scalability: float  # ビジネスモデル拡張性
    survival_probability: float  # 5年生存確率

@dataclass
class EmergenceFactors:
    """新設企業成功に影響する要因項目"""
    # 初期条件要因
    initial_capital: float      # 初期資本金
    parent_company_support: float  # 親会社支援度
    founding_team_experience: float  # 創業チーム経験値
    market_timing: float        # 市場参入タイミング
    
    # 戦略要因
    rd_intensity: float         # R&D投資強度
    marketing_investment: float  # マーケティング投資
    talent_acquisition_speed: float  # 人材獲得速度
    partnership_strategy: float  # パートナーシップ戦略
    
    # 運営要因
    operational_efficiency: float  # 運営効率性
    technology_adoption: float   # 技術導入速度
    quality_management: float    # 品質管理水準
    customer_acquisition_cost: float  # 顧客獲得コスト
    
    # 市場環境要因
    market_growth_rate: float   # 市場成長率
    competitive_intensity: float  # 競争激化度
    regulatory_environment: float  # 規制環境
    economic_conditions: float   # 経済環境

class EmergenceSuccessAnalyzer:
    """新設企業成功要因分析クラス"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初期化
        
        Args:
            config: 分析パラメータ設定辞書
        """
        self.config = config or self._default_config()
        self.scaler = RobustScaler()  # 外れ値に頑健なスケーラー
        self.models = {}
        self.feature_importance = {}
        self.analysis_results = {}
        
    def _default_config(self) -> Dict:
        """デフォルト設定を返す"""
        return {
            'min_years_data': 3,  # 最小データ年数
            'success_threshold': {
                'revenue_growth': 0.10,  # 年10%成長
                'profitability_years': 5,  # 5年以内黒字化
                'market_share_min': 0.01,  # 最低1%市場シェア
                'survival_years': 5,  # 5年生存
            },
            'analysis_methods': ['random_forest', 'gradient_boosting', 'linear'],
            'cross_validation_folds': 5,
            'feature_selection_threshold': 0.01,
            'outlier_detection_method': 'iqr',
            'missing_value_strategy': 'interpolate'
        }
    
    def load_emergence_data(self, data_path: str, 
                            companies_list: List[str]) -> pd.DataFrame:
        """
        新設企業データを読み込み
        
        Args:
            data_path: データファイルパス
            companies_list: 対象企業リスト
            
        Returns:
            新設企業データフレーム
        """
        try:
            # データ読み込み (実際のEDINETデータ構造に合わせて調整)
            data = pd.read_csv(data_path)
            
            # 新設企業のみ抽出
            emergence_companies = self._identify_emergence_companies(
                data, companies_list
            )
            
            # データクリーニング
            cleaned_data = self._clean_emergence_data(emergence_companies)
            
            return cleaned_data
            
        except Exception as e:
            raise ValueError(f"新設企業データ読み込みエラー: {e}")
    
    def _identify_emergence_companies(self, data: pd.DataFrame, 
                                    companies_list: List[str]) -> pd.DataFrame:
        """
        新設・分社企業を特定
        
        Args:
            data: 全企業データ
            companies_list: 対象企業リスト
            
        Returns:
            新設企業データ
        """
        emergence_criteria = {
            # 設立年が比較的新しい企業
            'establishment_year': lambda x: x >= 1990,
            # 親会社からの分社・独立企業
            'spinoff_indicator': lambda x: pd.notna(x),
            # 新規上場企業
            'ipo_recent': lambda x: x >= 1990,
        }
        
        emergence_data = data[
            (data['company_name'].isin(companies_list)) &
            (data['establishment_year'] >= 1990 if 'establishment_year' in data.columns else True)
        ].copy()
        
        return emergence_data
    
    def _clean_emergence_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        新設企業データのクリーニング
        
        Args:
            data: 生データ
            
        Returns:
            クリーニング済みデータ
        """
        cleaned_data = data.copy()
        
        # 欠損値処理
        if self.config['missing_value_strategy'] == 'interpolate':
            numeric_columns = cleaned_data.select_dtypes(include=[np.number]).columns
            cleaned_data[numeric_columns] = cleaned_data.groupby('company_name')[numeric_columns].transform(
                lambda x: x.interpolate(method='linear')
            )
        
        # 外れ値処理
        if self.config['outlier_detection_method'] == 'iqr':
            cleaned_data = self._remove_outliers_iqr(cleaned_data)
        
        # データ型変換
        cleaned_data = self._convert_data_types(cleaned_data)
        
        return cleaned_data
    
    def _remove_outliers_iqr(self, data: pd.DataFrame) -> pd.DataFrame:
        """IQR法による外れ値除去"""
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            Q1 = data[column].quantile(0.25)
            Q3 = data[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            data[column] = np.clip(data[column], lower_bound, upper_bound)
        
        return data
    
    def _convert_data_types(self, data: pd.DataFrame) -> pd.DataFrame:
        """データ型の最適化"""
        # 日付列の変換
        date_columns = ['date', 'fiscal_year_end', 'establishment_date']
        for col in date_columns:
            if col in data.columns:
                data[col] = pd.to_datetime(data[col], errors='coerce')
        
        # カテゴリ列の変換
        category_columns = ['industry', 'market_category', 'company_type']
        for col in category_columns:
            if col in data.columns:
                data[col] = data[col].astype('category')
        
        return data
    
    def calculate_success_metrics(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        新設企業の成功指標を計算
        
        Args:
            data: 財務データ
            
        Returns:
            成功指標付きデータ
        """
        metrics_data = data.copy()
        
        # 企業別・年別のデータをグループ化
        company_groups = metrics_data.groupby('company_name')
        
        success_metrics = []
        
        for company, group in company_groups:
            group_sorted = group.sort_values('fiscal_year')
            
            # 基本的な成功指標計算
            company_metrics = self._calculate_company_success_metrics(
                group_sorted, company
            )
            success_metrics.append(company_metrics)
        
        success_df = pd.DataFrame(success_metrics)
        
        # 元データとマージ
        result_data = metrics_data.merge(
            success_df, on='company_name', how='left'
        )
        
        return result_data
    
    def _calculate_company_success_metrics(self, company_data: pd.DataFrame, 
                                            company_name: str) -> Dict:
        """
        個別企業の成功指標計算
        
        Args:
            company_data: 企業別データ
            company_name: 企業名
            
        Returns:
            成功指標辞書
        """
        metrics = {'company_name': company_name}
        
        try:
            # 設立年または最初のデータ年を取得
            establishment_year = company_data['fiscal_year'].min()
            latest_year = company_data['fiscal_year'].max()
            years_in_business = latest_year - establishment_year + 1
            
            # 売上高成長率 (年平均)
            if 'revenue' in company_data.columns and len(company_data) >= 2:
                first_revenue = company_data['revenue'].iloc[0]
                last_revenue = company_data['revenue'].iloc[-1]
                if first_revenue > 0 and last_revenue > 0:
                    metrics['revenue_growth_rate'] = (
                        (last_revenue / first_revenue) ** (1 / (len(company_data) - 1)) - 1
                    )
                else:
                    metrics['revenue_growth_rate'] = 0.0
            else:
                metrics['revenue_growth_rate'] = 0.0
            
            # 黒字化達成速度
            if 'operating_income' in company_data.columns:
                profitable_years = company_data[company_data['operating_income'] > 0]
                if not profitable_years.empty:
                    first_profitable_year = profitable_years['fiscal_year'].min()
                    metrics['profitability_achievement_speed'] = (
                        first_profitable_year - establishment_year
                    )
                else:
                    metrics['profitability_achievement_speed'] = years_in_business
            else:
                metrics['profitability_achievement_speed'] = years_in_business
            
            # ROE改善軌道
            if 'roe' in company_data.columns and len(company_data) >= 2:
                roe_trend = self._calculate_trend_slope(
                    company_data['fiscal_year'], company_data['roe']
                )
                metrics['roe_trajectory'] = roe_trend
            else:
                metrics['roe_trajectory'] = 0.0
            
            # 営業利益率トレンド
            if 'operating_margin' in company_data.columns and len(company_data) >= 2:
                margin_trend = self._calculate_trend_slope(
                    company_data['fiscal_year'], company_data['operating_margin']
                )
                metrics['operating_margin_trend'] = margin_trend
            else:
                metrics['operating_margin_trend'] = 0.0
            
            # 従業員数成長率
            if 'employee_count' in company_data.columns and len(company_data) >= 2:
                first_employees = company_data['employee_count'].iloc[0]
                last_employees = company_data['employee_count'].iloc[-1]
                if first_employees > 0 and last_employees > 0:
                    metrics['employee_growth_rate'] = (
                        (last_employees / first_employees) ** (1 / (len(company_data) - 1)) - 1
                    )
                else:
                    metrics['employee_growth_rate'] = 0.0
            else:
                metrics['employee_growth_rate'] = 0.0
            
            # 総資産成長率
            if 'total_assets' in company_data.columns and len(company_data) >= 2:
                first_assets = company_data['total_assets'].iloc[0]
                last_assets = company_data['total_assets'].iloc[-1]
                if first_assets > 0 and last_assets > 0:
                    metrics['asset_growth_rate'] = (
                        (last_assets / first_assets) ** (1 / (len(company_data) - 1)) - 1
                    )
                else:
                    metrics['asset_growth_rate'] = 0.0
            else:
                metrics['asset_growth_rate'] = 0.0
            
            # 財務安定性スコア (複数指標の合成)
            stability_indicators = []
            
            if 'current_ratio' in company_data.columns:
                avg_current_ratio = company_data['current_ratio'].mean()
                stability_indicators.append(min(avg_current_ratio / 2.0, 1.0))
            
            if 'debt_equity_ratio' in company_data.columns:
                avg_de_ratio = company_data['debt_equity_ratio'].mean()
                stability_indicators.append(max(0, 1.0 - avg_de_ratio / 1.0))
            
            if 'interest_coverage_ratio' in company_data.columns:
                avg_interest_coverage = company_data['interest_coverage_ratio'].mean()
                stability_indicators.append(min(avg_interest_coverage / 10.0, 1.0))
            
            if stability_indicators:
                metrics['financial_stability'] = np.mean(stability_indicators)
            else:
                metrics['financial_stability'] = 0.5  # デフォルト値
            
            # 生存確率 (単純化: 年数ベース)
            survival_probability = min(years_in_business / 10.0, 1.0)  # 10年で1.0
            metrics['survival_probability'] = survival_probability
            
            # イノベーション創出度 (R&D投資とその効果)
            if 'rd_expenses' in company_data.columns and 'revenue' in company_data.columns:
                rd_intensity = (
                    company_data['rd_expenses'].sum() / 
                    company_data['revenue'].sum()
                )
                metrics['innovation_output'] = rd_intensity * metrics['revenue_growth_rate']
            else:
                metrics['innovation_output'] = 0.0
            
        except Exception as e:
            print(f"企業 {company_name} の成功指標計算エラー: {e}")
            # デフォルト値設定
            default_metrics = {
                'revenue_growth_rate': 0.0,
                'profitability_achievement_speed': 10.0,
                'roe_trajectory': 0.0,
                'operating_margin_trend': 0.0,
                'employee_growth_rate': 0.0,
                'asset_growth_rate': 0.0,
                'financial_stability': 0.5,
                'survival_probability': 0.5,
                'innovation_output': 0.0
            }
            metrics.update(default_metrics)
        
        return metrics
    
    def _calculate_trend_slope(self, x: pd.Series, y: pd.Series) -> float:
        """
        トレンドの傾きを計算
        
        Args:
            x: 独立変数 (年など)
            y: 従属変数 (指標値)
            
        Returns:
            傾きの値
        """
        try:
            # 欠損値を除去
            valid_data = pd.DataFrame({'x': x, 'y': y}).dropna()
            if len(valid_data) < 2:
                return 0.0
            
            # 線形回帰による傾き計算
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                valid_data['x'], valid_data['y']
            )
            
            return slope if p_value < 0.05 else 0.0  # 有意でない場合は0
        
        except Exception:
            return 0.0
    
    def extract_success_factors(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        成功要因項目を抽出・計算
        
        Args:
            data: 財務データ
            
        Returns:
            要因項目付きデータ
        """
        factor_data = data.copy()
        
        # 基本要因項目の計算
        factor_data = self._calculate_basic_factors(factor_data)
        
        # 戦略要因項目の計算
        factor_data = self._calculate_strategic_factors(factor_data)
        
        # 運営要因項目の計算
        factor_data = self._calculate_operational_factors(factor_data)
        
        # 市場環境要因項目の計算
        factor_data = self._calculate_market_factors(factor_data)
        
        return factor_data
    
    def _calculate_basic_factors(self, data: pd.DataFrame) -> pd.DataFrame:
        """基本要因項目の計算"""
        # 初期資本金 (設立時の資本金データまたは最初の年の資本金)
        if 'capital' in data.columns:
            data['initial_capital'] = data.groupby('company_name')['capital'].transform('first')
        else:
            data['initial_capital'] = 0
        
        # 親会社支援度 (関連会社売上高比率等で近似)
        if 'related_company_sales' in data.columns and 'revenue' in data.columns:
            data['parent_company_support'] = (
                data['related_company_sales'] / data['revenue']
            ).fillna(0)
        else:
            data['parent_company_support'] = 0
        
        # 市場参入タイミング (設立年の逆数等で近似)
        if 'establishment_year' in data.columns:
            current_year = 2024
            data['market_timing'] = 1 / (current_year - data['establishment_year'] + 1)
        else:
            data['market_timing'] = 0.5
        
        return data
    
    def _calculate_strategic_factors(self, data: pd.DataFrame) -> pd.DataFrame:
        """戦略要因項目の計算"""
        # R&D投資強度
        if 'rd_expenses' in data.columns and 'revenue' in data.columns:
            data['rd_intensity'] = (data['rd_expenses'] / data['revenue']).fillna(0)
        else:
            data['rd_intensity'] = 0
        
        # マーケティング投資 (販管費のうち広告宣伝費等)
        if 'advertising_expenses' in data.columns and 'revenue' in data.columns:
            data['marketing_investment'] = (
                data['advertising_expenses'] / data['revenue']
            ).fillna(0)
        else:
            data['marketing_investment'] = 0
        
        # 人材獲得速度 (従業員数の前年比成長率)
        if 'employee_count' in data.columns:
            data['talent_acquisition_speed'] = (
                data.groupby('company_name')['employee_count']
                .pct_change().fillna(0)
            )
        else:
            data['talent_acquisition_speed'] = 0
        
        return data
    
    def _calculate_operational_factors(self, data: pd.DataFrame) -> pd.DataFrame:
        """運営要因項目の計算"""
        # 運営効率性 (総資産回転率)
        if 'revenue' in data.columns and 'total_assets' in data.columns:
            data['operational_efficiency'] = (
                data['revenue'] / data['total_assets']
            ).fillna(0)
        else:
            data['operational_efficiency'] = 0
        
        # 品質管理水準 (返品率の逆数、またはクレーム処理費用等で近似)
        # 実際のデータがない場合は売上高営業利益率で代用
        if 'operating_margin' in data.columns:
            data['quality_management'] = data['operating_margin'].fillna(0)
        else:
            data['quality_management'] = 0
        
        return data
    
    def _calculate_market_factors(self, data: pd.DataFrame) -> pd.DataFrame:
        """市場環境要因項目の計算"""
        # 市場成長率 (業界全体の成長率、ここでは簡易的に売上高成長率の業界平均)
        if 'industry' in data.columns and 'revenue' in data.columns:
            industry_growth = (
                data.groupby(['industry', 'fiscal_year'])['revenue']
                .sum().groupby('industry').pct_change().reset_index()
            )
            data = data.merge(
                industry_growth.rename(columns={'revenue': 'market_growth_rate'}),
                on=['industry', 'fiscal_year'], how='left'
            )
            data['market_growth_rate'] = data['market_growth_rate'].fillna(0)
        else:
            data['market_growth_rate'] = 0
        
        # 競争激化度 (市場内企業数の増加率等で近似)
        # 簡易的に業界内の企業数で代用
        if 'industry' in data.columns:
            company_count = (
                data.groupby(['industry', 'fiscal_year'])['company_name']
                .nunique().reset_index()
            )
            data = data.merge(
                company_count.rename(columns={'company_name': 'competitive_intensity'}),
                on=['industry', 'fiscal_year'], how='left'
            )
            data['competitive_intensity'] = data['competitive_intensity'].fillna(1)
        else:
            data['competitive_intensity'] = 1
        
        return data
    
    def analyze_success_factors(self, data: pd.DataFrame, 
                                target_variable: str = 'revenue_growth_rate') -> Dict:
        """
        成功要因分析の実行
        
        Args:
            data: 分析データ
            target_variable: 目的変数
            
        Returns:
            分析結果辞書
        """
        # 特徴量とターゲットの準備
        feature_columns = self._get_factor_columns(data)
        target_data = data[target_variable].dropna()
        feature_data = data[feature_columns].loc[target_data.index]
        
        # 欠損値処理
        feature_data = feature_data.fillna(feature_data.median())
        
        # 特徴量スケーリング
        feature_scaled = self.scaler.fit_transform(feature_data)
        feature_scaled_df = pd.DataFrame(
            feature_scaled, 
            columns=feature_columns, 
            index=feature_data.index
        )
        
        # 各種モデルでの分析
        analysis_results = {}
        
        for method in self.config['analysis_methods']:
            model_results = self._train_model(
                feature_scaled_df, target_data, method
            )
            analysis_results[method] = model_results
        
        # 特徴量重要度の統合
        importance_results = self._integrate_feature_importance(analysis_results)
        
        # 相関分析
        correlation_results = self._analyze_correlations(
            feature_data, target_data
        )
        
        # 結果の統合
        final_results = {
            'model_results': analysis_results,
            'feature_importance': importance_results,
            'correlations': correlation_results,
            'data_summary': {
                'n_samples': len(target_data),
                'n_features': len(feature_columns),
                'target_mean': target_data.mean(),
                'target_std': target_data.std()
            }
        }
        
        self.analysis_results[target_variable] = final_results
        return final_results
    
    def _get_factor_columns(self, data: pd.DataFrame) -> List[str]:
        """要因項目の列名を取得"""
        factor_columns = [
            'initial_capital', 'parent_company_support', 'market_timing',
            'rd_intensity', 'marketing_investment', 'talent_acquisition_speed',
            'operational_efficiency', 'quality_management',
            'market_growth_rate', 'competitive_intensity'
        ]
        
        # 実際に存在する列のみ返す
        available_columns = [col for col in factor_columns if col in data.columns]
        return available_columns
    
    def _train_model(self, X: pd.DataFrame, y: pd.Series, method: str) -> Dict:
        """
        指定された手法でモデル学習
        
        Args:
            X: 特徴量データ
            y: 目的変数
            method: 学習手法
            
        Returns:
            モデル結果辞書
        """
        try:
            # モデル選択
            if method == 'random_forest':
                model = RandomForestRegressor(
                    n_estimators=100, random_state=42, n_jobs=-1
                )
            elif method == 'gradient_boosting':
                model = GradientBoostingRegressor(
                    n_estimators=100, random_state=42
                )
            elif method == 'linear':
                model = Ridge(alpha=1.0)
            else:
                raise ValueError(f"未サポートの手法: {method}")
            
            # 学習
            model.fit(X, y)
            
            # 予測
            y_pred = model.predict(X)
            
            # 評価指標計算
            mse = mean_squared_error(y, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y, y_pred)
            r2 = r2_score(y, y_pred)
            
            # 交差検証
            cv_scores = cross_val_score(
                model, X, y, 
                cv=self.config['cross_validation_folds'],
                scoring='r2'
            )
            
            # 特徴量重要度
            if hasattr(model, 'feature_importances_'):
                feature_importance = dict(zip(X.columns, model.feature_importances_))
            elif hasattr(model, 'coef_'):
                feature_importance = dict(zip(X.columns, np.abs(model.coef_)))
            else:
                feature_importance = {}
            
            results = {
                'model': model,
                'predictions': y_pred,
                'metrics': {
                    'mse': mse,
                    'rmse': rmse,
                    'mae': mae,
                    'r2': r2,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std()
                },
                'feature_importance': feature_importance
            }
            
            return results
        
        except Exception as e:
            print(f"モデル学習エラー ({method}): {e}")
            return {'error': str(e)}
    
    def _integrate_feature_importance(self, model_results: Dict) -> Dict:
        """複数モデルの特徴量重要度を統合"""
        all_importance = {}
        
        for method, results in model_results.items():
            if 'feature_importance' in results:
                for feature, importance in results['feature_importance'].items():
                    if feature not in all_importance:
                        all_importance[feature] = []
                    all_importance[feature].append(importance)
        
        # 平均重要度計算
        integrated_importance = {}
        for feature, importance_list in all_importance.items():
            integrated_importance[feature] = {
                'mean': np.mean(importance_list),
                'std': np.std(importance_list),
                'methods_count': len(importance_list)
            }
        
        # 重要度順にソート
        sorted_importance = dict(sorted(
            integrated_importance.items(),
            key=lambda x: x[1]['mean'],
            reverse=True
        ))
        
        return sorted_importance
    
    def _analyze_correlations(self, features: pd.DataFrame, 
                            target: pd.Series) -> Dict:
        """
        特徴量と目的変数の相関分析
        
        Args:
            features: 特徴量データ
            target: 目的変数
            
        Returns:
            相関分析結果
        """
        correlation_results = {}
        
        for column in features.columns:
            feature_series = features[column]
            
            # 欠損値を除去してペアを作成
            valid_pairs = pd.DataFrame({
                'feature': feature_series,
                'target': target
            }).dropna()
            
            if len(valid_pairs) < 3:  # 最低3点は必要
                continue
            
            try:
                # ピアソン相関
                pearson_corr, pearson_p = pearsonr(
                    valid_pairs['feature'], valid_pairs['target']
                )
                
                # スピアマン相関
                spearman_corr, spearman_p = spearmanr(
                    valid_pairs['feature'], valid_pairs['target']
                )
                
                # ケンドールのタウ
                kendall_corr, kendall_p = kendalltau(
                    valid_pairs['feature'], valid_pairs['target']
                )
                
                correlation_results[column] = {
                    'pearson': {
                        'correlation': pearson_corr,
                        'p_value': pearson_p,
                        'significant': pearson_p < 0.05
                    },
                    'spearman': {
                        'correlation': spearman_corr,
                        'p_value': spearman_p,
                        'significant': spearman_p < 0.05
                    },
                    'kendall': {
                        'correlation': kendall_corr,
                        'p_value': kendall_p,
                        'significant': kendall_p < 0.05
                    },
                    'n_samples': len(valid_pairs)
                }
                
            except Exception as e:
                print(f"相関分析エラー ({column}): {e}")
                correlation_results[column] = {'error': str(e)}
        
        return correlation_results
    
    def classify_success_levels(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        企業を成功レベル別に分類
        
        Args:
            data: 成功指標付きデータ
            
        Returns:
            成功レベル分類付きデータ
        """
        classified_data = data.copy()
        
        # 成功指標のスコア計算
        success_score = self._calculate_comprehensive_success_score(classified_data)
        classified_data['success_score'] = success_score
        
        # 成功レベルの分類
        success_levels = self._define_success_levels(success_score)
        classified_data['success_level'] = success_levels
        
        return classified_data
    
    def _calculate_comprehensive_success_score(self, data: pd.DataFrame) -> pd.Series:
        """
        総合成功スコアの計算
        
        Args:
            data: 企業データ
            
        Returns:
            成功スコア
        """
        score_components = []
        weights = {}
        
        # 成長性スコア (重み: 0.3)
        if 'revenue_growth_rate' in data.columns:
            growth_score = np.clip(data['revenue_growth_rate'] * 10, 0, 1)
            score_components.append(growth_score)
            weights['growth'] = 0.3
        
        # 収益性スコア (重み: 0.25)
        if 'operating_margin_trend' in data.columns:
            profitability_score = np.clip(data['operating_margin_trend'] * 10, 0, 1)
            score_components.append(profitability_score)
            weights['profitability'] = 0.25
        
        # 持続可能性スコア (重み: 0.25)
        if 'financial_stability' in data.columns:
            sustainability_score = data['financial_stability']
            score_components.append(sustainability_score)
            weights['sustainability'] = 0.25
        
        # イノベーションスコア (重み: 0.2)
        if 'innovation_output' in data.columns:
            innovation_score = np.clip(data['innovation_output'] * 5, 0, 1)
            score_components.append(innovation_score)
            weights['innovation'] = 0.2
        
        # 重み付き平均スコア計算
        if score_components:
            weighted_score = np.zeros(len(data))
            total_weight = 0
            
            weight_values = list(weights.values())
            for i, component in enumerate(score_components):
                if i < len(weight_values):
                    weighted_score += component * weight_values[i]
                    total_weight += weight_values[i]
            
            if total_weight > 0:
                weighted_score /= total_weight
            
            return pd.Series(weighted_score, index=data.index)
        else:
            return pd.Series(0.5, index=data.index)  # デフォルトスコア
    
    def _define_success_levels(self, scores: pd.Series) -> pd.Series:
        """
        成功スコアに基づくレベル分類
        
        Args:
            scores: 成功スコア
            
        Returns:
            成功レベル
        """
        # パーセンタイルによる分類
        high_threshold = scores.quantile(0.8)
        medium_threshold = scores.quantile(0.6)
        low_threshold = scores.quantile(0.4)
        
        def classify_score(score):
            if score >= high_threshold:
                return 'High Success'
            elif score >= medium_threshold:
                return 'Medium-High Success'
            elif score >= low_threshold:
                return 'Medium Success'
            else:
                return 'Low Success'
        
        return scores.apply(classify_score)
    
    def generate_success_factor_report(self, analysis_results: Dict) -> Dict:
        """
        成功要因分析レポートの生成
        
        Args:
            analysis_results: 分析結果
            
        Returns:
            レポート辞書
        """
        report = {
            'summary': self._create_summary_report(analysis_results),
            'key_findings': self._extract_key_findings(analysis_results),
            'recommendations': self._generate_recommendations(analysis_results),
            'statistical_significance': self._assess_statistical_significance(analysis_results),
            'methodology': self._document_methodology()
        }
        
        return report
    
    def _create_summary_report(self, results: Dict) -> Dict:
        """サマリーレポートの作成"""
        summary = {
            'analysis_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'data_summary': results.get('data_summary', {}),
            'model_performance': {},
            'top_success_factors': {}
        }
        
        # モデル性能サマリー
        for method, model_result in results.get('model_results', {}).items():
            if 'metrics' in model_result:
                summary['model_performance'][method] = {
                    'r2_score': model_result['metrics'].get('r2', 0),
                    'rmse': model_result['metrics'].get('rmse', 0),
                    'cv_score': model_result['metrics'].get('cv_mean', 0)
                }
        
        # トップ成功要因
        feature_importance = results.get('feature_importance', {})
        top_factors = dict(list(feature_importance.items())[:5])
        summary['top_success_factors'] = top_factors
        
        return summary
    
    def _extract_key_findings(self, results: Dict) -> List[str]:
        """主要な発見事項の抽出"""
        findings = []
        
        # 特徴量重要度からの発見
        feature_importance = results.get('feature_importance', {})
        if feature_importance:
            top_factor = next(iter(feature_importance))
            findings.append(
                f"最も重要な成功要因は「{top_factor}」で、"
                f"平均重要度は{feature_importance[top_factor]['mean']:.3f}です。"
            )
        
        # 相関分析からの発見
        correlations = results.get('correlations', {})
        strong_correlations = []
        for factor, corr_data in correlations.items():
            if 'pearson' in corr_data:
                pearson_corr = corr_data['pearson']['correlation']
                if abs(pearson_corr) > 0.5 and corr_data['pearson']['significant']:
                    strong_correlations.append((factor, pearson_corr))
        
        if strong_correlations:
            findings.append(
                f"強い相関を示す要因: {', '.join([f'{factor} (r={corr:.3f})' for factor, corr in strong_correlations[:3]])}"
            )
        
        # モデル性能からの発見
        model_results = results.get('model_results', {})
        best_model = None
        best_r2 = -1
        
        for method, model_result in model_results.items():
            if 'metrics' in model_result:
                r2 = model_result['metrics'].get('r2', -1)
                if r2 > best_r2:
                    best_r2 = r2
                    best_model = method
        
        if best_model and best_r2 > 0:
            findings.append(
                f"最適なモデルは{best_model}で、R²スコアは{best_r2:.3f}です。"
            )
        
        return findings
    
    def _generate_recommendations(self, results: Dict) -> List[str]:
        """推奨事項の生成"""
        recommendations = []
        
        feature_importance = results.get('feature_importance', {})
        
        # 重要度上位3要因に基づく推奨
        top_factors = list(feature_importance.keys())[:3]
        
        factor_recommendations = {
            'rd_intensity': 'R&D投資を売上高の5-10%程度に維持し、継続的なイノベーション創出を図る。',
            'operational_efficiency': '総資産回転率の向上を通じて運営効率性を高め、投資効果を最大化する。',
            'marketing_investment': 'ブランド認知向上とカスタマーベース拡大のため、マーケティング投資を戦略的に配分する。',
            'talent_acquisition_speed': '優秀な人材の迅速な確保により組織能力を強化し、成長スピードを加速する。',
            'financial_stability': '健全な財務構造を維持し、成長投資と財務安定性のバランスを取る。',
            'parent_company_support': '親会社との適切な関係性を保ちつつ、自立的な成長基盤を構築する。'
        }
        
        for factor in top_factors:
            if factor in factor_recommendations:
                recommendations.append(factor_recommendations[factor])
        
        # データ品質に基づる推奨
        data_summary = results.get('data_summary', {})
        n_samples = data_summary.get('n_samples', 0)
        
        if n_samples < 30:
            recommendations.append(
                'より多くの企業データを収集することで、分析の統計的有意性を向上させる。'
            )
        
        return recommendations
    
    def _assess_statistical_significance(self, results: Dict) -> Dict:
        """統計的有意性の評価"""
        significance_assessment = {
            'overall_model_reliability': 'medium',
            'feature_reliability': {},
            'correlation_reliability': {}
        }
        
        # モデルの信頼性評価
        model_results = results.get('model_results', {})
        r2_scores = []
        cv_scores = []
        
        for model_result in model_results.values():
            if 'metrics' in model_result:
                r2_scores.append(model_result['metrics'].get('r2', 0))
                cv_scores.append(model_result['metrics'].get('cv_mean', 0))
        
        if r2_scores:
            avg_r2 = np.mean(r2_scores)
            avg_cv = np.mean(cv_scores)
            
            if avg_r2 > 0.7 and avg_cv > 0.6:
                significance_assessment['overall_model_reliability'] = 'high'
            elif avg_r2 > 0.5 and avg_cv > 0.4:
                significance_assessment['overall_model_reliability'] = 'medium'
            else:
                significance_assessment['overall_model_reliability'] = 'low'
        
        # 特徴量の信頼性評価
        feature_importance = results.get('feature_importance', {})
        for factor, importance_data in feature_importance.items():
            methods_count = importance_data.get('methods_count', 0)
            std_ratio = (
                importance_data.get('std', 1) / 
                max(importance_data.get('mean', 0.01), 0.01)
            )
            
            if methods_count >= 3 and std_ratio < 0.5:
                significance_assessment['feature_reliability'][factor] = 'high'
            elif methods_count >= 2 and std_ratio < 1.0:
                significance_assessment['feature_reliability'][factor] = 'medium'
            else:
                significance_assessment['feature_reliability'][factor] = 'low'
        
        # 相関の信頼性評価
        correlations = results.get('correlations', {})
        for factor, corr_data in correlations.items():
            if 'pearson' in corr_data:
                p_value = corr_data['pearson'].get('p_value', 1)
                correlation = abs(corr_data['pearson'].get('correlation', 0))
                n_samples = corr_data.get('n_samples', 0)
                
                if p_value < 0.01 and correlation > 0.5 and n_samples >= 30:
                    significance_assessment['correlation_reliability'][factor] = 'high'
                elif p_value < 0.05 and correlation > 0.3 and n_samples >= 20:
                    significance_assessment['correlation_reliability'][factor] = 'medium'
                else:
                    significance_assessment['correlation_reliability'][factor] = 'low'
        
        return significance_assessment
    
    def _document_methodology(self) -> Dict:
        """分析手法の文書化"""
        methodology = {
            'analysis_approach': 'Multiple Machine Learning Models + Statistical Analysis',
            'models_used': self.config['analysis_methods'],
            'feature_engineering': [
                'Basic factors: initial_capital, parent_company_support, market_timing',
                'Strategic factors: rd_intensity, marketing_investment, talent_acquisition_speed',
                'Operational factors: operational_efficiency, quality_management',
                'Market factors: market_growth_rate, competitive_intensity'
            ],
            'preprocessing_steps': [
                'Missing value imputation using interpolation',
                'Outlier detection and treatment using IQR method',
                'Feature scaling using RobustScaler',
                'Data type optimization'
            ],
            'evaluation_metrics': [
                'R² Score (coefficient of determination)',
                'Root Mean Squared Error (RMSE)',
                'Mean Absolute Error (MAE)',
                'Cross-validation scores'
            ],
            'statistical_tests': [
                'Pearson correlation coefficient',
                'Spearman rank correlation',
                'Kendall\'s tau correlation',
                'Statistical significance testing (p < 0.05)'
            ]
        }
        
        return methodology
    
    def save_results(self, filepath: str, results: Dict):
        """
        分析結果をファイルに保存
        
        Args:
            filepath: 保存先ファイルパス
            results: 保存する結果辞書
        """
        try:
            import json
            import pickle
            
            # JSON形式で保存可能な部分を抽出
            json_serializable_results = {
                'feature_importance': results.get('feature_importance', {}),
                'correlations': {},  # モデルオブジェクトを除外
                'data_summary': results.get('data_summary', {}),
                'model_metrics': {}
            }
            
            # 相関結果の保存用整理
            for factor, corr_data in results.get('correlations', {}).items():
                if 'error' not in corr_data:
                    json_serializable_results['correlations'][factor] = {
                        'pearson_corr': corr_data['pearson']['correlation'],
                        'pearson_p': corr_data['pearson']['p_value'],
                        'spearman_corr': corr_data['spearman']['correlation'],
                        'n_samples': corr_data['n_samples']
                    }
            
            # モデル評価指標の保存
            for method, model_result in results.get('model_results', {}).items():
                if 'metrics' in model_result:
                    json_serializable_results['model_metrics'][method] = model_result['metrics']
            
            # JSON形式で保存
            with open(f"{filepath}.json", 'w', encoding='utf-8') as f:
                json.dump(json_serializable_results, f, indent=2, ensure_ascii=False)
            
            # 完全なオブジェクト（モデルを含む）をPickle形式で保存
            with open(f"{filepath}.pickle", 'wb') as f:
                pickle.dump(results, f)
            
            print(f"分析結果を保存しました: {filepath}.json, {filepath}.pickle")
            
        except Exception as e:
            print(f"結果保存エラー: {e}")
    
    def load_results(self, filepath: str) -> Dict:
        """
        保存された分析結果を読み込み
        
        Args:
            filepath: 読み込みファイルパス
            
        Returns:
            読み込まれた結果辞書
        """
        try:
            import pickle
            
            with open(f"{filepath}.pickle", 'rb') as f:
                results = pickle.load(f)
            
            print(f"分析結果を読み込みました: {filepath}.pickle")
            return results
            
        except Exception as e:
            print(f"結果読み込みエラー: {e}")
            return {}