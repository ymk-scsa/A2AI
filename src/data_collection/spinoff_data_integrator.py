"""
A2AI - Advanced Financial Analysis AI
Spinoff Data Integrator Module

企業の分社・統合・事業継承イベントのデータ統合を行うモジュール
- 分社化前後の財務データ連続性確保
- 親会社と子会社のデータ統合
- M&A・事業統合のデータ処理
- 事業継承成功度の評価項目算出
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
import logging
from dataclasses import dataclass
from enum import Enum
import yaml

from ..utils.data_utils import DataProcessor
from ..utils.logging_utils import setup_logger
from ..utils.statistical_utils import StatisticalAnalyzer


class SpinoffEventType(Enum):
    """分社・統合イベントの種類"""
    SPINOFF = "spinoff"              # 分社化
    MERGER = "merger"                # 合併
    ACQUISITION = "acquisition"      # 買収
    DIVESTITURE = "divestiture"      # 事業売却
    SPLIT = "split"                  # 会社分割
    CONSOLIDATION = "consolidation"  # 統合


@dataclass
class SpinoffEvent:
    """分社・統合イベントの情報"""
    event_id: str
    event_type: SpinoffEventType
    event_date: datetime
    parent_company: str
    child_company: str
    business_segment: str
    transaction_value: Optional[float] = None
    ownership_ratio: float = 100.0
    description: str = ""


class SpinoffDataIntegrator:
    """分社・統合データ統合クラス"""
    
    def __init__(self, config_path: str = "config/settings.py"):
        """
        初期化
        
        Args:
            config_path: 設定ファイルパス
        """
        self.logger = setup_logger(__name__)
        self.data_processor = DataProcessor()
        self.statistical_analyzer = StatisticalAnalyzer()
        
        # 設定読み込み
        self.config = self._load_config(config_path)
        
        # 分社・統合イベント履歴
        self.spinoff_events: List[SpinoffEvent] = []
        
        # データベース接続設定
        self.db_config = self.config.get('database', {})
        
        self.logger.info("SpinoffDataIntegrator initialized")
    
    def _load_config(self, config_path: str) -> Dict:
        """設定ファイル読み込み"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                    return yaml.safe_load(f)
                else:
                    # Python設定ファイルの場合の処理
                    return {}
        except Exception as e:
            self.logger.warning(f"Config file not found or invalid: {e}")
            return {}
    
    def load_spinoff_events(self, events_data_path: str) -> None:
        """
        分社・統合イベントデータの読み込み
        
        Args:
            events_data_path: イベントデータファイルパス
        """
        try:
            events_df = pd.read_csv(events_data_path, encoding='utf-8')
            
            for _, row in events_df.iterrows():
                event = SpinoffEvent(
                    event_id=str(row['event_id']),
                    event_type=SpinoffEventType(row['event_type']),
                    event_date=pd.to_datetime(row['event_date']),
                    parent_company=str(row['parent_company']),
                    child_company=str(row['child_company']),
                    business_segment=str(row['business_segment']),
                    transaction_value=row.get('transaction_value'),
                    ownership_ratio=float(row.get('ownership_ratio', 100.0)),
                    description=str(row.get('description', ''))
                )
                self.spinoff_events.append(event)
            
            self.logger.info(f"Loaded {len(self.spinoff_events)} spinoff events")
            
        except Exception as e:
            self.logger.error(f"Failed to load spinoff events: {e}")
            raise
    
    def integrate_spinoff_financial_data(
        self, 
        parent_data: pd.DataFrame, 
        child_data: pd.DataFrame,
        event: SpinoffEvent
    ) -> pd.DataFrame:
        """
        分社化に伴う財務データの統合
        
        Args:
            parent_data: 親会社の財務データ
            child_data: 子会社の財務データ
            event: 分社イベント情報
            
        Returns:
            統合された財務データ
        """
        try:
            # 分社前後の期間を特定
            pre_spinoff_data = parent_data[
                parent_data['date'] < event.event_date
            ].copy()
            
            post_spinoff_parent = parent_data[
                parent_data['date'] >= event.event_date
            ].copy()
            
            post_spinoff_child = child_data[
                child_data['date'] >= event.event_date
            ].copy()
            
            # 分社前データの調整（分社された事業部分を除外推定）
            adjusted_pre_data = self._adjust_pre_spinoff_data(
                pre_spinoff_data, event
            )
            
            # 分社後データの統合
            integrated_post_data = self._integrate_post_spinoff_data(
                post_spinoff_parent, post_spinoff_child, event
            )
            
            # 全期間データの統合
            integrated_data = pd.concat([
                adjusted_pre_data,
                integrated_post_data
            ], ignore_index=True).sort_values('date')
            
            # 連続性チェック
            self._validate_data_continuity(integrated_data, event)
            
            self.logger.info(f"Successfully integrated spinoff data for {event.event_id}")
            
            return integrated_data
            
        except Exception as e:
            self.logger.error(f"Failed to integrate spinoff data: {e}")
            raise
    
    def _adjust_pre_spinoff_data(
        self, 
        pre_data: pd.DataFrame, 
        event: SpinoffEvent
    ) -> pd.DataFrame:
        """
        分社前データの調整（分社事業部分の推定除外）
        
        Args:
            pre_data: 分社前データ
            event: 分社イベント
            
        Returns:
            調整済み分社前データ
        """
        adjusted_data = pre_data.copy()
        
        # 分社事業の規模推定（所有比率ベース）
        adjustment_ratio = event.ownership_ratio / 100.0
        
        # 主要財務項目の調整
        financial_items = [
            'revenue', 'gross_profit', 'operating_income', 'net_income',
            'total_assets', 'total_liabilities', 'shareholders_equity',
            'operating_cash_flow', 'capex', 'rd_expense'
        ]
        
        for item in financial_items:
            if item in adjusted_data.columns:
                # 分社された事業部分を推定控除
                adjusted_data[item] = adjusted_data[item] * (1 - adjustment_ratio)
        
        # 従業員数等の調整
        if 'employee_count' in adjusted_data.columns:
            adjusted_data['employee_count'] = (
                adjusted_data['employee_count'] * (1 - adjustment_ratio)
            ).astype(int)
        
        # 調整履歴を記録
        adjusted_data['data_adjustment'] = f'pre_spinoff_adjusted_{event.event_id}'
        
        return adjusted_data
    
    def _integrate_post_spinoff_data(
        self,
        parent_data: pd.DataFrame,
        child_data: pd.DataFrame,
        event: SpinoffEvent
    ) -> pd.DataFrame:
        """
        分社後の親会社・子会社データ統合
        
        Args:
            parent_data: 分社後親会社データ
            child_data: 子会社データ
            event: 分社イベント
            
        Returns:
            統合後データ
        """
        # データの時間軸を合わせる
        common_dates = set(parent_data['date']).intersection(set(child_data['date']))
        
        integrated_data = []
        
        for date in sorted(common_dates):
            parent_row = parent_data[parent_data['date'] == date].iloc[0]
            child_row = child_data[child_data['date'] == date].iloc[0]
            
            # 統合行を作成
            integrated_row = parent_row.copy()
            
            # 財務項目の統合（所有比率考慮）
            ownership_ratio = event.ownership_ratio / 100.0
            
            financial_items = [
                'revenue', 'gross_profit', 'operating_income', 'net_income',
                'total_assets', 'total_liabilities', 'shareholders_equity',
                'operating_cash_flow', 'capex', 'rd_expense'
            ]
            
            for item in financial_items:
                if item in child_row.index:
                    child_value = child_row[item] * ownership_ratio
                    integrated_row[item] = parent_row[item] + child_value
            
            # 従業員数の統合
            if 'employee_count' in child_row.index:
                integrated_row['employee_count'] = (
                    parent_row['employee_count'] + 
                    child_row['employee_count'] * ownership_ratio
                )
            
            # 統合情報を記録
            integrated_row['integration_type'] = 'post_spinoff_integrated'
            integrated_row['child_company'] = event.child_company
            integrated_row['ownership_ratio'] = event.ownership_ratio
            
            integrated_data.append(integrated_row)
        
        return pd.DataFrame(integrated_data)
    
    def calculate_succession_success_metrics(
        self,
        pre_spinoff_data: pd.DataFrame,
        post_spinoff_data: pd.DataFrame,
        event: SpinoffEvent,
        evaluation_period: int = 3
    ) -> Dict[str, float]:
        """
        事業継承成功度指標の計算
        
        Args:
            pre_spinoff_data: 分社前データ
            post_spinoff_data: 分社後データ
            event: 分社イベント
            evaluation_period: 評価期間（年）
            
        Returns:
            事業継承成功度指標
        """
        try:
            # 評価期間のデータを抽出
            evaluation_start = event.event_date
            evaluation_end = evaluation_start + timedelta(days=365 * evaluation_period)
            
            eval_data = post_spinoff_data[
                (post_spinoff_data['date'] >= evaluation_start) &
                (post_spinoff_data['date'] <= evaluation_end)
            ]
            
            # 分社前基準期間のデータ
            baseline_start = evaluation_start - timedelta(days=365 * evaluation_period)
            baseline_data = pre_spinoff_data[
                (pre_spinoff_data['date'] >= baseline_start) &
                (pre_spinoff_data['date'] < evaluation_start)
            ]
            
            if eval_data.empty or baseline_data.empty:
                self.logger.warning(f"Insufficient data for succession metrics: {event.event_id}")
                return {}
            
            # 成功度指標の計算
            metrics = {}
            
            # 1. 売上成長率比較
            baseline_revenue = baseline_data['revenue'].mean()
            eval_revenue = eval_data['revenue'].mean()
            metrics['revenue_growth_ratio'] = (eval_revenue / baseline_revenue - 1) * 100
            
            # 2. 利益率改善度
            baseline_margin = (baseline_data['operating_income'] / baseline_data['revenue']).mean()
            eval_margin = (eval_data['operating_income'] / eval_data['revenue']).mean()
            metrics['margin_improvement'] = (eval_margin - baseline_margin) * 100
            
            # 3. ROE改善度
            baseline_roe = (baseline_data['net_income'] / baseline_data['shareholders_equity']).mean()
            eval_roe = (eval_data['net_income'] / eval_data['shareholders_equity']).mean()
            metrics['roe_improvement'] = (eval_roe - baseline_roe) * 100
            
            # 4. 資産効率改善度
            baseline_turnover = (baseline_data['revenue'] / baseline_data['total_assets']).mean()
            eval_turnover = (eval_data['revenue'] / eval_data['total_assets']).mean()
            metrics['asset_turnover_improvement'] = (eval_turnover - baseline_turnover) * 100
            
            # 5. 統合成功度スコア（重み付き平均）
            weights = {
                'revenue_growth_ratio': 0.3,
                'margin_improvement': 0.25,
                'roe_improvement': 0.25,
                'asset_turnover_improvement': 0.2
            }
            
            succession_score = sum(
                metrics.get(key, 0) * weight 
                for key, weight in weights.items()
            )
            metrics['succession_success_score'] = succession_score
            
            # 6. リスク調整後スコア
            revenue_volatility = eval_data['revenue'].std() / eval_data['revenue'].mean()
            metrics['risk_adjusted_score'] = succession_score / (1 + revenue_volatility)
            
            self.logger.info(f"Calculated succession metrics for {event.event_id}")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to calculate succession metrics: {e}")
            return {}
    
    def analyze_spinoff_impact_patterns(
        self,
        all_events_data: List[Dict]
    ) -> pd.DataFrame:
        """
        分社・統合のインパクトパターン分析
        
        Args:
            all_events_data: 全イベントのデータリスト
            
        Returns:
            パターン分析結果
        """
        try:
            analysis_results = []
            
            for event_data in all_events_data:
                event = event_data['event']
                metrics = event_data['metrics']
                
                # パターン分析
                pattern_analysis = {
                    'event_id': event.event_id,
                    'event_type': event.event_type.value,
                    'parent_company': event.parent_company,
                    'child_company': event.child_company,
                    'event_date': event.event_date,
                    'business_segment': event.business_segment,
                    'ownership_ratio': event.ownership_ratio,
                    **metrics  # 成功度指標を展開
                }
                
                # 成功・失敗の判定
                success_score = metrics.get('succession_success_score', 0)
                if success_score > 10:
                    pattern_analysis['success_category'] = 'High Success'
                elif success_score > 0:
                    pattern_analysis['success_category'] = 'Moderate Success'
                elif success_score > -10:
                    pattern_analysis['success_category'] = 'Neutral'
                else:
                    pattern_analysis['success_category'] = 'Poor Performance'
                
                # 市場環境の影響
                pattern_analysis['market_timing'] = self._analyze_market_timing(event)
                
                analysis_results.append(pattern_analysis)
            
            results_df = pd.DataFrame(analysis_results)
            
            # 統計的分析
            self._perform_pattern_statistical_analysis(results_df)
            
            self.logger.info(f"Analyzed {len(analysis_results)} spinoff impact patterns")
            
            return results_df
            
        except Exception as e:
            self.logger.error(f"Failed to analyze spinoff patterns: {e}")
            raise
    
    def _analyze_market_timing(self, event: SpinoffEvent) -> str:
        """
        分社・統合のマーケットタイミング分析
        
        Args:
            event: 分社イベント
            
        Returns:
            タイミング分析結果
        """
        # 簡易的な市場環境判定（実際の実装では外部市場データを使用）
        event_year = event.event_date.year
        
        # 経済危機・好景気の大まかな時期
        crisis_periods = [
            (1997, 1999),  # アジア通貨危機
            (2000, 2003),  # ITバブル崩壊
            (2008, 2010),  # リーマンショック
            (2020, 2021),  # コロナ禍
        ]
        
        for start, end in crisis_periods:
            if start <= event_year <= end:
                return 'Crisis Period'
        
        if event_year in [2005, 2006, 2007, 2017, 2018, 2019]:
            return 'Growth Period'
        
        return 'Stable Period'
    
    def _perform_pattern_statistical_analysis(self, results_df: pd.DataFrame) -> None:
        """
        パターン分析の統計解析
        
        Args:
            results_df: 分析結果データフレーム
        """
        try:
            # 成功要因の相関分析
            numeric_cols = results_df.select_dtypes(include=[np.number]).columns
            correlation_matrix = results_df[numeric_cols].corr()
            
            # 成功度スコアとの相関が高い要因を特定
            score_correlations = correlation_matrix['succession_success_score'].sort_values(
                ascending=False, key=abs
            )
            
            self.logger.info("Top factors correlated with succession success:")
            for factor, correlation in score_correlations.head(10).items():
                if factor != 'succession_success_score':
                    self.logger.info(f"  {factor}: {correlation:.3f}")
            
            # 事業タイプ別成功率
            if 'business_segment' in results_df.columns:
                segment_success = results_df.groupby('business_segment').agg({
                    'succession_success_score': ['mean', 'std', 'count']
                })
                self.logger.info("Success by business segment:")
                self.logger.info(segment_success.to_string())
                
        except Exception as e:
            self.logger.error(f"Failed to perform statistical analysis: {e}")
    
    def _validate_data_continuity(
        self, 
        integrated_data: pd.DataFrame, 
        event: SpinoffEvent
    ) -> bool:
        """
        データ連続性の検証
        
        Args:
            integrated_data: 統合データ
            event: 分社イベント
            
        Returns:
            連続性チェック結果
        """
        try:
            # 分社前後での急激な変化をチェック
            event_date = event.event_date
            
            # 分社前後1年のデータを比較
            pre_data = integrated_data[
                integrated_data['date'] < event_date
            ].tail(4)  # 四半期データ4期分
            
            post_data = integrated_data[
                integrated_data['date'] >= event_date
            ].head(4)  # 四半期データ4期分
            
            if pre_data.empty or post_data.empty:
                self.logger.warning(f"Insufficient data for continuity check: {event.event_id}")
                return False
            
            # 主要指標の変化率をチェック
            check_items = ['revenue', 'total_assets', 'employee_count']
            anomalies = []
            
            for item in check_items:
                if item in integrated_data.columns:
                    pre_mean = pre_data[item].mean()
                    post_mean = post_data[item].mean()
                    
                    if pre_mean > 0:
                        change_rate = abs(post_mean - pre_mean) / pre_mean
                        # 50%を超える変化は異常とみなす
                        if change_rate > 0.5:
                            anomalies.append({
                                'item': item,
                                'change_rate': change_rate,
                                'pre_mean': pre_mean,
                                'post_mean': post_mean
                            })
            
            if anomalies:
                self.logger.warning(f"Data continuity anomalies detected for {event.event_id}:")
                for anomaly in anomalies:
                    self.logger.warning(f"  {anomaly}")
                return False
            
            self.logger.info(f"Data continuity validated for {event.event_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to validate data continuity: {e}")
            return False
    
    def generate_integration_report(
        self,
        integrated_data: pd.DataFrame,
        event: SpinoffEvent,
        metrics: Dict[str, float]
    ) -> Dict[str, any]:
        """
        統合処理レポートの生成
        
        Args:
            integrated_data: 統合データ
            event: 分社イベント
            metrics: 成功度指標
            
        Returns:
            統合レポート
        """
        try:
            report = {
                'event_summary': {
                    'event_id': event.event_id,
                    'event_type': event.event_type.value,
                    'event_date': event.event_date.strftime('%Y-%m-%d'),
                    'parent_company': event.parent_company,
                    'child_company': event.child_company,
                    'business_segment': event.business_segment,
                    'ownership_ratio': event.ownership_ratio,
                    'description': event.description
                },
                'data_summary': {
                    'total_records': len(integrated_data),
                    'date_range': {
                        'start': integrated_data['date'].min().strftime('%Y-%m-%d'),
                        'end': integrated_data['date'].max().strftime('%Y-%m-%d')
                    },
                    'pre_spinoff_records': len(
                        integrated_data[integrated_data['date'] < event.event_date]
                    ),
                    'post_spinoff_records': len(
                        integrated_data[integrated_data['date'] >= event.event_date]
                    )
                },
                'succession_metrics': metrics,
                'data_quality': {
                    'missing_values': integrated_data.isnull().sum().to_dict(),
                    'continuity_validated': self._validate_data_continuity(
                        integrated_data, event
                    )
                }
            }
            
            # 主要財務指標の要約統計
            numeric_cols = integrated_data.select_dtypes(include=[np.number]).columns
            summary_stats = integrated_data[numeric_cols].describe()
            report['financial_summary'] = summary_stats.to_dict()
            
            self.logger.info(f"Generated integration report for {event.event_id}")
            
            return report
            
        except Exception as e:
            self.logger.error(f"Failed to generate integration report: {e}")
            return {}
    
    def export_integrated_data(
        self,
        integrated_data: pd.DataFrame,
        output_path: str,
        event: SpinoffEvent
    ) -> None:
        """
        統合データのエクスポート
        
        Args:
            integrated_data: 統合データ
            output_path: 出力パス
            event: 分社イベント
        """
        try:
            # ファイル名にイベント情報を含める
            filename = f"integrated_data_{event.parent_company}_{event.event_id}.csv"
            full_path = f"{output_path}/{filename}"
            
            # メタデータを含めてエクスポート
            integrated_data.to_csv(full_path, index=False, encoding='utf-8')
            
            # メタデータファイルも作成
            metadata = {
                'event_id': event.event_id,
                'event_type': event.event_type.value,
                'integration_date': datetime.now().isoformat(),
                'data_records': len(integrated_data),
                'columns': list(integrated_data.columns)
            }
            
            metadata_filename = f"metadata_{event.parent_company}_{event.event_id}.yaml"
            metadata_path = f"{output_path}/{metadata_filename}"
            
            with open(metadata_path, 'w', encoding='utf-8') as f:
                yaml.dump(metadata, f, allow_unicode=True)
            
            self.logger.info(f"Exported integrated data to {full_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to export integrated data: {e}")
            raise


# 使用例とテスト用の関数
def example_usage():
    """使用例"""
    try:
        # スピンオフデータ統合器の初期化
        integrator = SpinoffDataIntegrator()
        
        # 分社イベントデータの読み込み
        integrator.load_spinoff_events("data/raw/spinoff_events/spinoff_events.csv")
        
        # サンプルデータでの統合処理
        parent_data = pd.DataFrame({
            'date': pd.date_range('2020-01-01', '2024-12-31', freq='Q'),
            'revenue': np.random.normal(100000, 10000, 20),
            'operating_income': np.random.normal(10000, 2000, 20),
            'total_assets': np.random.normal(500000, 50000, 20),
            'employee_count': np.random.randint(1000, 1500, 20)
        })
        
        child_data = pd.DataFrame({
            'date': pd.date_range('2022-01-01', '2024-12-31', freq='Q'),
            'revenue': np.random.normal(30000, 3000, 12),
            'operating_income': np.random.normal(3000, 600, 12),
            'total_assets': np.random.normal(150000, 15000, 12),
            'employee_count': np.random.randint(300, 500, 12)
        })
        
        # サンプル分社イベント
        event = SpinoffEvent(
            event_id="SP001",
            event_type=SpinoffEventType.SPINOFF,
            event_date=pd.to_datetime('2022-01-01'),
            parent_company="パナソニック",
            child_company="パナソニックエナジー",
            business_segment="電池事業",
            ownership_ratio=80.0,
            description="電池事業の分社化"
        )
        
        # 統合処理実行
        integrated_data = integrator.integrate_spinoff_financial_data(
            parent_data, child_data, event
        )
        
        # 事業継承成功度の計算
        metrics = integrator.calculate_succession_success_metrics(
            parent_data, integrated_data, event
        )
        
        # レポート生成
        report = integrator.generate_integration_report(
            integrated_data, event, metrics
        )
        
        print("Integration completed successfully!")
        print(f"Succession success score: {metrics.get('succession_success_score', 'N/A')}")
        
        return integrated_data, report
        
    except Exception as e:
        print(f"Error in example usage: {e}")
        return None, None


if __name__ == "__main__":
    example_usage()