"""
A2AI (Advanced Financial Analysis AI) - Missing Value Handler
企業ライフサイクル全体を考慮した高度欠損値処理システム

主要機能:
1. 企業消滅・分社化・統合に伴う構造的欠損値処理
2. 150社×最大40年分データの時系列欠損値補完
3. 9つの評価項目×23の要因項目の特性別欠損処理
4. 生存バイアス対応の統計的補完手法
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
from datetime import datetime, timedelta
import logging

class MissingValueHandler:
    """
    A2AI専用高度欠損値処理クラス
    
    企業ライフサイクル全体（設立〜存続〜消滅）を考慮し、
    構造的欠損値と偶発的欠損値を区別して処理
    """
    
    def __init__(self, config: Dict = None):
        """
        初期化
        
        Args:
            config: 設定辞書
        """
        self.config = config or self._get_default_config()
        self.logger = self._setup_logging()
        
        # 企業ライフサイクル情報を保存
        self.company_lifecycle = {}
        self.market_categories = {}
        self.imputation_history = {}
        
        # 各指標の特性定義
        self.evaluation_metrics = [
            'sales_revenue', 'sales_growth_rate', 'operating_margin', 
            'net_margin', 'roe', 'value_added_ratio',
            'survival_probability', 'emergence_success_rate', 'succession_success_rate'
        ]
        
        # 要因項目の特性分類
        self.factor_categories = {
            'financial': ['total_assets', 'equity_ratio', 'debt_ratio', 'current_ratio'],
            'operational': ['employee_count', 'rd_expenses', 'capex', 'inventory_turnover'],
            'market': ['overseas_sales_ratio', 'segment_count', 'market_share'],
            'lifecycle': ['company_age', 'market_entry_timing', 'parent_dependency']
        }
    
    def _get_default_config(self) -> Dict:
        """デフォルト設定を取得"""
        return {
            'missing_threshold': 0.3,  # 欠損率30%以上で特別処理
            'time_window': 5,  # 前後5年での補完
            'min_observations': 3,  # 最小観測数
            'iteration_max': 10,  # 反復補完最大回数
            'random_state': 42,
            'n_neighbors': 5,  # KNN補完用
            'convergence_threshold': 1e-3
        }
    
    def _setup_logging(self) -> logging.Logger:
        """ログ設定"""
        logger = logging.getLogger('A2AI_MissingValueHandler')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def set_company_lifecycle_info(self, lifecycle_df: pd.DataFrame):
        """
        企業ライフサイクル情報を設定
        
        Args:
            lifecycle_df: 企業ライフサイクル情報
                columns: ['company_id', 'establishment_year', 'extinction_year', 
                            'market_category', 'lifecycle_events']
        """
        for _, row in lifecycle_df.iterrows():
            self.company_lifecycle[row['company_id']] = {
                'establishment_year': row.get('establishment_year'),
                'extinction_year': row.get('extinction_year'),
                'market_category': row.get('market_category'),
                'lifecycle_events': row.get('lifecycle_events', [])
            }
            self.market_categories[row['company_id']] = row.get('market_category')
        
        self.logger.info(f"企業ライフサイクル情報を設定: {len(self.company_lifecycle)}社")
    
    def analyze_missing_patterns(self, df: pd.DataFrame) -> Dict:
        """
        欠損パターン分析
        
        Args:
            df: 分析対象データフレーム
            
        Returns:
            欠損パターン分析結果
        """
        missing_analysis = {
            'total_missing': df.isnull().sum(),
            'missing_percentage': (df.isnull().sum() / len(df)) * 100,
            'missing_by_company': {},
            'missing_by_year': {},
            'structural_missing': {},
            'random_missing': {}
        }
        
        # 企業別欠損分析
        if 'company_id' in df.columns:
            missing_analysis['missing_by_company'] = (
                df.groupby('company_id').apply(lambda x: x.isnull().sum().sum())
            )
        
        # 年次別欠損分析
        if 'year' in df.columns:
            missing_analysis['missing_by_year'] = (
                df.groupby('year').apply(lambda x: x.isnull().sum().sum())
            )
        
        # 構造的欠損 vs ランダム欠損の分類
        for col in df.columns:
            if col in ['company_id', 'year']:
                continue
            
            # 構造的欠損の検出
            structural_pattern = self._detect_structural_missing(df, col)
            if structural_pattern['is_structural']:
                missing_analysis['structural_missing'][col] = structural_pattern
            else:
                missing_analysis['random_missing'][col] = {
                    'missing_count': df[col].isnull().sum(),
                    'missing_rate': df[col].isnull().sum() / len(df)
                }
        
        self.logger.info(f"欠損パターン分析完了: 構造的欠損{len(missing_analysis['structural_missing'])}項目, "
                        f"ランダム欠損{len(missing_analysis['random_missing'])}項目")
        
        return missing_analysis
    
    def _detect_structural_missing(self, df: pd.DataFrame, column: str) -> Dict:
        """
        構造的欠損の検出
        
        Args:
            df: データフレーム
            column: 対象列名
            
        Returns:
            構造的欠損情報
        """
        if 'company_id' not in df.columns or 'year' not in df.columns:
            return {'is_structural': False}
        
        structural_info = {
            'is_structural': False,
            'pattern_type': None,
            'affected_companies': [],
            'affected_years': []
        }
        
        # 企業消滅パターンの検出
        extinction_pattern = self._detect_extinction_pattern(df, column)
        if extinction_pattern['detected']:
            structural_info.update({
                'is_structural': True,
                'pattern_type': 'extinction',
                'affected_companies': extinction_pattern['companies']
            })
            return structural_info
        
        # 新設企業パターンの検出
        emergence_pattern = self._detect_emergence_pattern(df, column)
        if emergence_pattern['detected']:
            structural_info.update({
                'is_structural': True,
                'pattern_type': 'emergence',
                'affected_companies': emergence_pattern['companies']
            })
            return structural_info
        
        # 事業再編パターンの検出
        restructure_pattern = self._detect_restructure_pattern(df, column)
        if restructure_pattern['detected']:
            structural_info.update({
                'is_structural': True,
                'pattern_type': 'restructure',
                'affected_companies': restructure_pattern['companies'],
                'affected_years': restructure_pattern['years']
            })
        
        return structural_info
    
    def _detect_extinction_pattern(self, df: pd.DataFrame, column: str) -> Dict:
        """企業消滅パターンの検出"""
        pattern = {'detected': False, 'companies': []}
        
        for company_id in df['company_id'].unique():
            company_data = df[df['company_id'] == company_id].sort_values('year')
            
            # ライフサイクル情報がある場合
            if company_id in self.company_lifecycle:
                extinction_year = self.company_lifecycle[company_id].get('extinction_year')
                if extinction_year:
                    # 消滅年以降のデータが全て欠損の場合
                    post_extinction = company_data[company_data['year'] > extinction_year]
                    if len(post_extinction) > 0 and post_extinction[column].isnull().all():
                        pattern['detected'] = True
                        pattern['companies'].append(company_id)
        
        return pattern
    
    def _detect_emergence_pattern(self, df: pd.DataFrame, column: str) -> Dict:
        """新設企業パターンの検出"""
        pattern = {'detected': False, 'companies': []}
        
        for company_id in df['company_id'].unique():
            company_data = df[df['company_id'] == company_id].sort_values('year')
            
            if company_id in self.company_lifecycle:
                establishment_year = self.company_lifecycle[company_id].get('establishment_year')
                if establishment_year:
                    # 設立年以前のデータが全て欠損の場合
                    pre_establishment = company_data[company_data['year'] < establishment_year]
                    if len(pre_establishment) > 0 and pre_establishment[column].isnull().all():
                        pattern['detected'] = True
                        pattern['companies'].append(company_id)
        
        return pattern
    
    def _detect_restructure_pattern(self, df: pd.DataFrame, column: str) -> Dict:
        """事業再編パターンの検出"""
        pattern = {'detected': False, 'companies': [], 'years': []}
        
        for company_id in df['company_id'].unique():
            company_data = df[df['company_id'] == company_id].sort_values('year')
            
            if company_id in self.company_lifecycle:
                events = self.company_lifecycle[company_id].get('lifecycle_events', [])
                for event in events:
                    if event.get('type') in ['merger', 'spinoff', 'acquisition']:
                        event_year = event.get('year')
                        # 再編年前後で大きなデータ欠損がある場合
                        event_period = company_data[
                            (company_data['year'] >= event_year - 1) & 
                            (company_data['year'] <= event_year + 1)
                        ]
                        if len(event_period) > 0 and event_period[column].isnull().sum() > len(event_period) * 0.7:
                            pattern['detected'] = True
                            pattern['companies'].append(company_id)
                            pattern['years'].append(event_year)
        
        return pattern
    
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'comprehensive') -> pd.DataFrame:
        """
        包括的欠損値処理
        
        Args:
            df: 処理対象データフレーム
            strategy: 処理戦略 ('comprehensive', 'conservative', 'aggressive')
            
        Returns:
            欠損値処理済みデータフレーム
        """
        df_processed = df.copy()
        
        # 欠損パターン分析
        missing_analysis = self.analyze_missing_patterns(df_processed)
        
        # 戦略に応じた処理
        if strategy == 'comprehensive':
            df_processed = self._comprehensive_imputation(df_processed, missing_analysis)
        elif strategy == 'conservative':
            df_processed = self._conservative_imputation(df_processed, missing_analysis)
        elif strategy == 'aggressive':
            df_processed = self._aggressive_imputation(df_processed, missing_analysis)
        
        # 処理結果の記録
        self._record_imputation_history(df, df_processed, strategy)
        
        return df_processed
    
    def _comprehensive_imputation(self, df: pd.DataFrame, missing_analysis: Dict) -> pd.DataFrame:
        """包括的補完処理"""
        df_result = df.copy()
        
        # 1. 構造的欠損の処理
        df_result = self._handle_structural_missing(df_result, missing_analysis['structural_missing'])
        
        # 2. 時系列補完
        df_result = self._time_series_imputation(df_result)
        
        # 3. 業界ベース補完
        df_result = self._industry_based_imputation(df_result)
        
        # 4. 機械学習ベース補完
        df_result = self._ml_based_imputation(df_result)
        
        # 5. 最終的な統計補完
        df_result = self._statistical_imputation(df_result)
        
        return df_result
    
    def _conservative_imputation(self, df: pd.DataFrame, missing_analysis: Dict) -> pd.DataFrame:
        """保守的補完処理（最低限の補完のみ）"""
        df_result = df.copy()
        
        # 構造的欠損は基本的に維持（企業消滅等の意味を保持）
        # 時系列補完と統計補完のみ実施
        df_result = self._time_series_imputation(df_result, conservative=True)
        df_result = self._statistical_imputation(df_result, method='median')
        
        return df_result
    
    def _aggressive_imputation(self, df: pd.DataFrame, missing_analysis: Dict) -> pd.DataFrame:
        """積極的補完処理（可能な限り全て補完）"""
        df_result = df.copy()
        
        # 全ての手法を組み合わせ
        df_result = self._handle_structural_missing(df_result, missing_analysis['structural_missing'])
        df_result = self._time_series_imputation(df_result, forward_fill=True, backward_fill=True)
        df_result = self._industry_based_imputation(df_result, use_similar_companies=True)
        df_result = self._ml_based_imputation(df_result, max_iterations=20)
        df_result = self._statistical_imputation(df_result, method='multiple')
        
        return df_result
    
    def _handle_structural_missing(self, df: pd.DataFrame, structural_missing: Dict) -> pd.DataFrame:
        """構造的欠損の処理"""
        df_result = df.copy()
        
        for column, pattern_info in structural_missing.items():
            if pattern_info['pattern_type'] == 'extinction':
                # 企業消滅の場合、消滅年以降は0または特別値で埋める
                df_result = self._handle_extinction_missing(df_result, column, pattern_info)
            
            elif pattern_info['pattern_type'] == 'emergence':
                # 新設企業の場合、設立年以前は0または欠損のまま
                df_result = self._handle_emergence_missing(df_result, column, pattern_info)
            
            elif pattern_info['pattern_type'] == 'restructure':
                # 事業再編の場合、前後のデータから補間
                df_result = self._handle_restructure_missing(df_result, column, pattern_info)
        
        return df_result
    
    def _handle_extinction_missing(self, df: pd.DataFrame, column: str, pattern_info: Dict) -> pd.DataFrame:
        """企業消滅に伴う欠損処理"""
        df_result = df.copy()
        
        for company_id in pattern_info['affected_companies']:
            if company_id in self.company_lifecycle:
                extinction_year = self.company_lifecycle[company_id].get('extinction_year')
                if extinction_year:
                    # 消滅年以降は明示的に0に設定（企業が存在しないことを示す）
                    mask = (df_result['company_id'] == company_id) & (df_result['year'] > extinction_year)
                    df_result.loc[mask, column] = 0
                    
                    # 消滅年のデータが欠損の場合、前年データの半分で補完
                    extinction_mask = (df_result['company_id'] == company_id) & (df_result['year'] == extinction_year)
                    if df_result.loc[extinction_mask, column].isnull().any():
                        prev_year_data = df_result[
                            (df_result['company_id'] == company_id) & 
                            (df_result['year'] == extinction_year - 1)
                        ][column].values
                        if len(prev_year_data) > 0 and not pd.isna(prev_year_data[0]):
                            df_result.loc[extinction_mask, column] = prev_year_data[0] * 0.5
        
        return df_result
    
    def _handle_emergence_missing(self, df: pd.DataFrame, column: str, pattern_info: Dict) -> pd.DataFrame:
        """新設企業に伴う欠損処理"""
        df_result = df.copy()
        
        for company_id in pattern_info['affected_companies']:
            if company_id in self.company_lifecycle:
                establishment_year = self.company_lifecycle[company_id].get('establishment_year')
                if establishment_year:
                    # 設立年以前は0に設定（企業が存在しないことを示す）
                    mask = (df_result['company_id'] == company_id) & (df_result['year'] < establishment_year)
                    df_result.loc[mask, column] = 0
                    
                    # 設立年のデータが欠損の場合、業界平均の50%で補完（小規模開始を仮定）
                    establishment_mask = (df_result['company_id'] == company_id) & (df_result['year'] == establishment_year)
                    if df_result.loc[establishment_mask, column].isnull().any():
                        market_category = self.market_categories.get(company_id)
                        if market_category:
                            industry_avg = self._get_industry_average(df_result, column, market_category, establishment_year)
                            if industry_avg is not None:
                                df_result.loc[establishment_mask, column] = industry_avg * 0.5
        
        return df_result
    
    def _handle_restructure_missing(self, df: pd.DataFrame, column: str, pattern_info: Dict) -> pd.DataFrame:
        """事業再編に伴う欠損処理"""
        df_result = df.copy()
        
        for i, company_id in enumerate(pattern_info['affected_companies']):
            event_year = pattern_info['years'][i] if i < len(pattern_info['years']) else None
            if event_year:
                # 再編年前後のデータを線形補間
                company_data = df_result[df_result['company_id'] == company_id].copy()
                company_data = company_data.sort_values('year')
                
                # 再編前後の有効データを取得
                pre_data = company_data[
                    (company_data['year'] < event_year) & 
                    (company_data[column].notna())
                ].tail(2)
                
                post_data = company_data[
                    (company_data['year'] > event_year) & 
                    (company_data[column].notna())
                ].head(2)
                
                # 補間実行
                if len(pre_data) > 0 and len(post_data) > 0:
                    df_result = self._interpolate_restructure_gap(
                        df_result, company_id, column, event_year, pre_data, post_data
                    )
        
        return df_result
    
    def _time_series_imputation(self, df: pd.DataFrame, conservative: bool = False, 
                                forward_fill: bool = False, backward_fill: bool = False) -> pd.DataFrame:
        """時系列補完"""
        df_result = df.copy()
        
        if 'company_id' not in df.columns or 'year' not in df.columns:
            return df_result
        
        numeric_columns = df_result.select_dtypes(include=[np.number]).columns
        numeric_columns = [col for col in numeric_columns if col not in ['company_id', 'year']]
        
        for company_id in df_result['company_id'].unique():
            company_mask = df_result['company_id'] == company_id
            company_data = df_result[company_mask].sort_values('year')
            
            for column in numeric_columns:
                if conservative:
                    # 保守的：隣接する年のみで補間
                    df_result.loc[company_mask, column] = company_data[column].interpolate(
                        method='linear', limit=1
                    )
                else:
                    # 標準：より広範囲での補間
                    interpolated = company_data[column].interpolate(
                        method='linear', limit=self.config['time_window']
                    )
                    
                    # 前方補完・後方補完の適用
                    if forward_fill:
                        interpolated = interpolated.fillna(method='ffill', limit=2)
                    if backward_fill:
                        interpolated = interpolated.fillna(method='bfill', limit=2)
                    
                    df_result.loc[company_mask, column] = interpolated
        
        return df_result
    
    def _industry_based_imputation(self, df: pd.DataFrame, use_similar_companies: bool = False) -> pd.DataFrame:
        """業界ベース補完"""
        df_result = df.copy()
        
        if 'company_id' not in df.columns or 'year' not in df.columns:
            return df_result
        
        numeric_columns = df_result.select_dtypes(include=[np.number]).columns
        numeric_columns = [col for col in numeric_columns if col not in ['company_id', 'year']]
        
        for column in numeric_columns:
            for year in df_result['year'].unique():
                year_mask = df_result['year'] == year
                year_data = df_result[year_mask]
                
                for market_category in ['high_share', 'declining', 'lost']:
                    category_companies = [
                        cid for cid, cat in self.market_categories.items() 
                        if cat == market_category
                    ]
                    
                    category_mask = year_data['company_id'].isin(category_companies)
                    category_data = year_data[category_mask]
                    
                    # 業界平均で補完
                    if len(category_data) > 0:
                        industry_avg = category_data[column].mean()
                        if not pd.isna(industry_avg):
                            missing_mask = category_data[column].isnull()
                            indices = category_data[missing_mask].index
                            df_result.loc[indices, column] = industry_avg
        
        return df_result
    
    def _ml_based_imputation(self, df: pd.DataFrame, max_iterations: int = None) -> pd.DataFrame:
        """機械学習ベース補完"""
        df_result = df.copy()
        
        numeric_columns = df_result.select_dtypes(include=[np.number]).columns
        numeric_columns = [col for col in numeric_columns if col not in ['company_id', 'year']]
        
        if len(numeric_columns) < 2:
            return df_result
        
        # 反復補完の実行
        max_iter = max_iterations or self.config['iteration_max']
        
        try:
            # IterativeImputerを使用
            imputer = IterativeImputer(
                estimator=RandomForestRegressor(random_state=self.config['random_state']),
                max_iter=max_iter,
                random_state=self.config['random_state']
            )
            
            # 企業・年情報を保持して補完
            company_year_data = df_result[['company_id', 'year']].copy()
            numeric_data = df_result[numeric_columns].copy()
            
            # 補完実行
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                imputed_data = imputer.fit_transform(numeric_data)
            
            # 結果をデータフレームに戻す
            imputed_df = pd.DataFrame(imputed_data, columns=numeric_columns, index=df_result.index)
            df_result[numeric_columns] = imputed_df
            
        except Exception as e:
            self.logger.warning(f"機械学習ベース補完でエラー: {e}")
            # フォールバック：KNN補完
            try:
                knn_imputer = KNNImputer(n_neighbors=self.config['n_neighbors'])
                imputed_data = knn_imputer.fit_transform(df_result[numeric_columns])
                df_result[numeric_columns] = imputed_data
            except Exception as e2:
                self.logger.warning(f"KNN補完でもエラー: {e2}")
        
        return df_result
    
    def _statistical_imputation(self, df: pd.DataFrame, method: str = 'mean') -> pd.DataFrame:
        """統計的補完（最終的な補完）"""
        df_result = df.copy()
        
        numeric_columns = df_result.select_dtypes(include=[np.number]).columns
        numeric_columns = [col for col in numeric_columns if col not in ['company_id', 'year']]
        
        for column in numeric_columns:
            if method == 'mean':
                fill_value = df_result[column].mean()
            elif method == 'median':
                fill_value = df_result[column].median()
            elif method == 'mode':
                fill_value = df_result[column].mode().iloc[0] if not df_result[column].mode().empty else 0
            elif method == 'multiple':
                # 複数の方法の平均
                mean_val = df_result[column].mean()
                median_val = df_result[column].median()
                fill_value = np.mean([mean_val, median_val]) if not pd.isna(mean_val) and not pd.isna(median_val) else (mean_val or median_val or 0)
            else:
                fill_value = 0
            
            if not pd.isna(fill_value):
                df_result[column] = df_result[column].fillna(fill_value)
        
        return df_result
    
    def _get_industry_average(self, df: pd.DataFrame, column: str, 
                            market_category: str, year: int) -> Optional[float]:
        """業界平均値を取得"""
        category_companies = [
            cid for cid, cat in self.market_categories.items() 
            if cat == market_category
        ]
        
        category_data = df[
            (df['company_id'].isin(category_companies)) & 
            (df['year'] == year)
        ]
        
        if len(category_data) > 0:
            return category_data[column].mean()
        return None
    
    def _interpolate_restructure_gap(self, df: pd.DataFrame, company_id: str, column: str,
                                    event_year: int, pre_data: pd.DataFrame, 
                                    post_data: pd.DataFrame) -> pd.DataFrame:
        """事業再編ギャップの補間"""
        df_result = df.copy()
        
        if len(pre_data) == 0 or len(post_data) == 0:
            return df_result
        
        # 再編前後の値を取得
        pre_value = pre_data[column].iloc[-1]
        post_value = post_data[column].iloc[0]
        
        pre_year = pre_data['year'].iloc[-1]
        post_year = post_data['year'].iloc[0]
        
        # 再編年のデータを線形補間
        if event_year > pre_year and event_year < post_year:
            interpolated_value = pre_value + (post_value - pre_value) * (event_year - pre_year) / (post_year - pre_year)
            
            mask = (df_result['company_id'] == company_id) & (df_result['year'] == event_year)
            df_result.loc[mask, column] = interpolated_value
        
        return df_result
    
    def _record_imputation_history(self, df_original: pd.DataFrame, 
                                    df_imputed: pd.DataFrame, strategy: str):
        """補完履歴の記録"""
        missing_before = df_original.isnull().sum().sum()
        missing_after = df_imputed.isnull().sum().sum()
        
        self.imputation_history[datetime.now().isoformat()] = {
            'strategy': strategy,
            'missing_before': missing_before,
            'missing_after': missing_after,
            'improvement_rate': (missing_before - missing_after) / missing_before if missing_before > 0 else 0,
            'columns_processed': list(df_original.columns),
            'records_processed': len(df_original)
        }
        
        self.logger.info(f"欠損値処理完了 - 戦略: {strategy}, "
                        f"欠損値削減: {missing_before} → {missing_after} "
                        f"({(missing_before - missing_after) / missing_before * 100:.1f}%改善)")
    
    def validate_imputation_quality(self, df_original: pd.DataFrame, 
                                    df_imputed: pd.DataFrame) -> Dict:
        """
        補完品質の検証
        
        Args:
            df_original: 元データ
            df_imputed: 補完後データ
            
        Returns:
            品質評価結果
        """
        validation_results = {
            'overall_quality': {},
            'column_quality': {},
            'temporal_consistency': {},
            'cross_sectional_consistency': {},
            'lifecycle_consistency': {}
        }
        
        # 全体品質評価
        missing_reduction = df_original.isnull().sum().sum() - df_imputed.isnull().sum().sum()
        total_missing = df_original.isnull().sum().sum()
        
        validation_results['overall_quality'] = {
            'missing_reduction_count': missing_reduction,
            'missing_reduction_rate': missing_reduction / total_missing if total_missing > 0 else 0,
            'final_completeness': 1 - (df_imputed.isnull().sum().sum() / (len(df_imputed) * len(df_imputed.columns))),
            'outliers_introduced': self._count_potential_outliers(df_original, df_imputed)
        }
        
        # 列別品質評価
        numeric_columns = df_imputed.select_dtypes(include=[np.number]).columns
        numeric_columns = [col for col in numeric_columns if col not in ['company_id', 'year']]
        
        for column in numeric_columns:
            original_stats = df_original[column].describe()
            imputed_stats = df_imputed[column].describe()
            
            validation_results['column_quality'][column] = {
                'mean_deviation': abs(original_stats['mean'] - imputed_stats['mean']) / original_stats['mean'] if original_stats['mean'] != 0 else 0,
                'std_deviation': abs(original_stats['std'] - imputed_stats['std']) / original_stats['std'] if original_stats['std'] != 0 else 0,
                'distribution_similarity': self._calculate_distribution_similarity(df_original[column], df_imputed[column])
            }
        
        # 時系列一貫性評価
        if 'company_id' in df_imputed.columns and 'year' in df_imputed.columns:
            validation_results['temporal_consistency'] = self._validate_temporal_consistency(df_imputed)
        
        # 横断面一貫性評価
        validation_results['cross_sectional_consistency'] = self._validate_cross_sectional_consistency(df_imputed)
        
        # ライフサイクル一貫性評価
        validation_results['lifecycle_consistency'] = self._validate_lifecycle_consistency(df_imputed)
        
        return validation_results
    
    def _count_potential_outliers(self, df_original: pd.DataFrame, df_imputed: pd.DataFrame) -> int:
        """補完により導入された潜在的外れ値をカウント"""
        outliers_count = 0
        
        numeric_columns = df_imputed.select_dtypes(include=[np.number]).columns
        numeric_columns = [col for col in numeric_columns if col not in ['company_id', 'year']]
        
        for column in numeric_columns:
            # 元データの範囲を取得
            original_values = df_original[column].dropna()
            if len(original_values) == 0:
                continue
            
            q1, q3 = original_values.quantile([0.25, 0.75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            # 補完された値が範囲外かチェック
            was_missing = df_original[column].isnull()
            imputed_values = df_imputed.loc[was_missing, column]
            
            outliers_count += ((imputed_values < lower_bound) | (imputed_values > upper_bound)).sum()
        
        return outliers_count
    
    def _calculate_distribution_similarity(self, original: pd.Series, imputed: pd.Series) -> float:
        """分布類似度を計算"""
        try:
            original_clean = original.dropna()
            imputed_clean = imputed.dropna()
            
            if len(original_clean) == 0 or len(imputed_clean) == 0:
                return 0.0
            
            # Kolmogorov-Smirnov検定を使用
            ks_stat, _ = stats.ks_2samp(original_clean, imputed_clean)
            return 1 - ks_stat  # 類似度として返す（0=全く違う, 1=完全に同じ）
        except:
            return 0.0
    
    def _validate_temporal_consistency(self, df: pd.DataFrame) -> Dict:
        """時系列一貫性の検証"""
        consistency_results = {
            'trend_violations': 0,
            'sudden_jumps': 0,
            'negative_values_inappropriate': 0
        }
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        numeric_columns = [col for col in numeric_columns if col not in ['company_id', 'year']]
        
        for company_id in df['company_id'].unique():
            company_data = df[df['company_id'] == company_id].sort_values('year')
            
            for column in numeric_columns:
                values = company_data[column].dropna()
                if len(values) < 3:
                    continue
                
                # 急激な変化の検出
                pct_changes = values.pct_change().dropna()
                sudden_jumps = (abs(pct_changes) > 2.0).sum()  # 200%以上の変化
                consistency_results['sudden_jumps'] += sudden_jumps
                
                # 不適切な負値の検出
                if column in ['sales_revenue', 'total_assets', 'employee_count']:  # 通常負になりえない項目
                    negative_count = (values < 0).sum()
                    consistency_results['negative_values_inappropriate'] += negative_count
        
        return consistency_results
    
    def _validate_cross_sectional_consistency(self, df: pd.DataFrame) -> Dict:
        """横断面一貫性の検証"""
        consistency_results = {
            'industry_outliers': 0,
            'size_inconsistencies': 0
        }
        
        if 'year' not in df.columns:
            return consistency_results
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        numeric_columns = [col for col in numeric_columns if col not in ['company_id', 'year']]
        
        for year in df['year'].unique():
            year_data = df[df['year'] == year]
            
            # 市場カテゴリー別の外れ値検出
            for market_category in ['high_share', 'declining', 'lost']:
                category_companies = [
                    cid for cid, cat in self.market_categories.items() 
                    if cat == market_category
                ]
                
                category_data = year_data[year_data['company_id'].isin(category_companies)]
                
                for column in numeric_columns:
                    if len(category_data) > 3:
                        values = category_data[column].dropna()
                        if len(values) > 0:
                            q1, q3 = values.quantile([0.25, 0.75])
                            iqr = q3 - q1
                            outliers = ((values < q1 - 3 * iqr) | (values > q3 + 3 * iqr)).sum()
                            consistency_results['industry_outliers'] += outliers
        
        return consistency_results
    
    def _validate_lifecycle_consistency(self, df: pd.DataFrame) -> Dict:
        """ライフサイクル一貫性の検証"""
        consistency_results = {
            'extinction_violations': 0,
            'emergence_violations': 0,
            'restructure_violations': 0
        }
        
        for company_id, lifecycle_info in self.company_lifecycle.items():
            company_data = df[df['company_id'] == company_id]
            if len(company_data) == 0:
                continue
            
            # 消滅企業の検証
            extinction_year = lifecycle_info.get('extinction_year')
            if extinction_year:
                post_extinction = company_data[company_data['year'] > extinction_year]
                # 消滅後にゼロでない値があるかチェック
                numeric_columns = post_extinction.select_dtypes(include=[np.number]).columns
                numeric_columns = [col for col in numeric_columns if col not in ['company_id', 'year']]
                
                for column in numeric_columns:
                    non_zero_count = (post_extinction[column] != 0).sum()
                    consistency_results['extinction_violations'] += non_zero_count
            
            # 新設企業の検証
            establishment_year = lifecycle_info.get('establishment_year')
            if establishment_year:
                pre_establishment = company_data[company_data['year'] < establishment_year]
                # 設立前にゼロでない値があるかチェック
                numeric_columns = pre_establishment.select_dtypes(include=[np.number]).columns
                numeric_columns = [col for col in numeric_columns if col not in ['company_id', 'year']]
                
                for column in numeric_columns:
                    non_zero_count = (pre_establishment[column] != 0).sum()
                    consistency_results['emergence_violations'] += non_zero_count
        
        return consistency_results
    
    def get_imputation_report(self) -> Dict:
        """補完処理レポートを生成"""
        report = {
            'summary': {
                'total_imputations': len(self.imputation_history),
                'companies_processed': len(self.company_lifecycle),
                'market_categories': len(set(self.market_categories.values()))
            },
            'history': self.imputation_history,
            'configuration': self.config
        }
        
        if self.imputation_history:
            latest_key = max(self.imputation_history.keys())
            latest_result = self.imputation_history[latest_key]
            
            report['latest_processing'] = {
                'timestamp': latest_key,
                'strategy': latest_result['strategy'],
                'missing_reduction': latest_result['missing_before'] - latest_result['missing_after'],
                'improvement_rate': latest_result['improvement_rate'],
                'records_processed': latest_result['records_processed']
            }
        
        return report
    
    def export_quality_report(self, df_original: pd.DataFrame, 
                            df_imputed: pd.DataFrame, 
                            output_path: str = None) -> str:
        """品質レポートをエクスポート"""
        validation_results = self.validate_imputation_quality(df_original, df_imputed)
        
        report_lines = [
            "A2AI 欠損値処理品質レポート",
            "=" * 50,
            "",
            "【全体品質】",
            f"欠損値削減数: {validation_results['overall_quality']['missing_reduction_count']:,}",
            f"欠損値削減率: {validation_results['overall_quality']['missing_reduction_rate']:.1%}",
            f"最終完全性: {validation_results['overall_quality']['final_completeness']:.1%}",
            f"導入された外れ値: {validation_results['overall_quality']['outliers_introduced']}",
            "",
            "【列別品質】"
        ]
        
        for column, quality in validation_results['column_quality'].items():
            report_lines.extend([
                f"{column}:",
                f"  平均偏差: {quality['mean_deviation']:.3f}",
                f"  標準偏差変化: {quality['std_deviation']:.3f}",
                f"  分布類似度: {quality['distribution_similarity']:.3f}",
                ""
            ])
        
        report_lines.extend([
            "【一貫性検証】",
            f"時系列異常: 急激変化{validation_results['temporal_consistency']['sudden_jumps']}回, "
            f"不適切負値{validation_results['temporal_consistency']['negative_values_inappropriate']}回",
            f"横断面異常: 業界外れ値{validation_results['cross_sectional_consistency']['industry_outliers']}個",
            f"ライフサイクル異常: 消滅違反{validation_results['lifecycle_consistency']['extinction_violations']}回, "
            f"新設違反{validation_results['lifecycle_consistency']['emergence_violations']}回"
        ])
        
        report_text = "\n".join(report_lines)
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            self.logger.info(f"品質レポートを出力: {output_path}")
        
        return report_text


# 使用例とテスト用のヘルパー関数
def create_sample_lifecycle_data() -> pd.DataFrame:
    """サンプルライフサイクルデータの作成"""
    return pd.DataFrame({
        'company_id': ['fanuc', 'sony', 'sanyo_denki', 'denawave'],
        'establishment_year': [1972, 1946, 1945, 2001],
        'extinction_year': [None, None, 2012, None],
        'market_category': ['high_share', 'lost', 'lost', 'high_share'],
        'lifecycle_events': [
            [],
            [{'type': 'restructure', 'year': 2000}],
            [{'type': 'merger', 'year': 2012}],
            [{'type': 'spinoff', 'year': 2001}]
        ]
    })

def demonstrate_missing_value_handling():
    """使用例デモンストレーション"""
    # サンプルデータの作成
    np.random.seed(42)
    dates = pd.date_range('1984', '2024', freq='Y')
    companies = ['fanuc', 'sony', 'sanyo_denki', 'denawave']
    
    # 不完全なデータを作成
    data = []
    for year in range(1984, 2025):
        for company in companies:
            # 企業ライフサイクルに応じた欠損パターンを作成
            if company == 'sanyo_denki' and year > 2012:  # 消滅企業
                continue
            if company == 'denawave' and year < 2001:  # 新設企業
                continue
            
            row = {
                'company_id': company,
                'year': year,
                'sales_revenue': np.random.normal(1000, 200) if np.random.random() > 0.1 else np.nan,
                'total_assets': np.random.normal(2000, 400) if np.random.random() > 0.15 else np.nan,
                'employee_count': np.random.normal(5000, 1000) if np.random.random() > 0.08 else np.nan,
                'rd_expenses': np.random.normal(100, 30) if np.random.random() > 0.2 else np.nan
            }
            data.append(row)
    
    df = pd.DataFrame(data)
    
    # ライフサイクル情報の設定
    lifecycle_data = create_sample_lifecycle_data()
    
    # MissingValueHandlerの初期化
    handler = MissingValueHandler()
    handler.set_company_lifecycle_info(lifecycle_data)
    
    # 欠損値処理の実行
    print("欠損値処理前:")
    print(f"総欠損値数: {df.isnull().sum().sum()}")
    print(f"欠損率: {df.isnull().sum().sum() / (len(df) * len(df.columns)):.1%}")
    
    # 包括的処理
    df_imputed = handler.handle_missing_values(df, strategy='comprehensive')
    
    print("\n欠損値処理後:")
    print(f"総欠損値数: {df_imputed.isnull().sum().sum()}")
    print(f"欠損率: {df_imputed.isnull().sum().sum() / (len(df_imputed) * len(df_imputed.columns)):.1%}")
    
    # 品質レポートの生成
    quality_report = handler.export_quality_report(df, df_imputed)
    print("\n品質レポート:")
    print(quality_report)
    
    return handler, df, df_imputed

if __name__ == "__main__":
    demonstrate_missing_value_handling()