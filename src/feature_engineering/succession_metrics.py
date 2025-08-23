"""
A2AI (Advanced Financial Analysis AI) - 事業継承成功度評価クラス
事業継承成功度（Succession Success Metrics）の計算と分析を行う

このモジュールは以下の事業継承パターンを分析します：
1. M&A（合併・買収）
2. 分社化（スピンオフ）
3. 事業統合
4. 子会社化
5. 事業売却・撤退
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings

@dataclass
class SuccessionEvent:
    """事業継承イベントのデータクラス"""
    event_id: str
    company_code: str
    company_name: str
    event_type: str  # 'merger', 'acquisition', 'spinoff', 'integration', 'divestiture'
    event_date: datetime
    pre_event_period: int  # イベント前分析期間（年数）
    post_event_period: int  # イベント後分析期間（年数）
    target_company: Optional[str] = None
    parent_company: Optional[str] = None
    transaction_value: Optional[float] = None
    description: Optional[str] = None

class SuccessionMetrics:
    """事業継承成功度評価クラス"""
    
    def __init__(self):
        self.succession_events: List[SuccessionEvent] = []
        self.financial_data: Optional[pd.DataFrame] = None
        self.market_data: Optional[pd.DataFrame] = None
        
    def load_financial_data(self, data: pd.DataFrame) -> None:
        """
        財務データを読み込む
        
        Parameters:
        -----------
        data : pd.DataFrame
            財務諸表データ（企業コード、年度、各種財務指標を含む）
        """
        required_columns = [
            'company_code', 'fiscal_year', 'revenue', 'operating_income',
            'net_income', 'total_assets', 'equity', 'roa', 'roe'
        ]
        
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"必要な列が不足しています: {missing_columns}")
            
        self.financial_data = data.copy()
        
    def load_market_data(self, data: pd.DataFrame) -> None:
        """
        市場シェアデータを読み込む
        
        Parameters:
        -----------
        data : pd.DataFrame
            市場シェアデータ（企業コード、年度、市場シェア、市場規模等を含む）
        """
        self.market_data = data.copy()
        
    def register_succession_event(self, event: SuccessionEvent) -> None:
        """
        事業継承イベントを登録
        
        Parameters:
        -----------
        event : SuccessionEvent
            事業継承イベント情報
        """
        self.succession_events.append(event)
        
    def calculate_pre_post_comparison(self, event: SuccessionEvent) -> Dict[str, float]:
        """
        事業継承前後の財務指標比較分析
        
        Parameters:
        -----------
        event : SuccessionEvent
            分析対象の事業継承イベント
            
        Returns:
        --------
        Dict[str, float]
            事業継承前後の各種指標変化率
        """
        if self.financial_data is None:
            raise ValueError("財務データが読み込まれていません")
            
        company_data = self.financial_data[
            self.financial_data['company_code'] == event.company_code
        ].copy()
        
        event_year = event.event_date.year
        
        # 事業継承前期間の平均値計算
        pre_period_start = event_year - event.pre_event_period
        pre_period_end = event_year - 1
        pre_data = company_data[
            (company_data['fiscal_year'] >= pre_period_start) & 
            (company_data['fiscal_year'] <= pre_period_end)
        ]
        
        # 事業継承後期間の平均値計算
        post_period_start = event_year + 1
        post_period_end = event_year + event.post_event_period
        post_data = company_data[
            (company_data['fiscal_year'] >= post_period_start) & 
            (company_data['fiscal_year'] <= post_period_end)
        ]
        
        if pre_data.empty or post_data.empty:
            warnings.warn(f"企業 {event.company_code} の事業継承前後データが不十分です")
            return {}
        
        metrics = {}
        financial_indicators = ['revenue', 'operating_income', 'net_income', 'total_assets', 'equity', 'roa', 'roe']
        
        for indicator in financial_indicators:
            if indicator in pre_data.columns and indicator in post_data.columns:
                pre_avg = pre_data[indicator].mean()
                post_avg = post_data[indicator].mean()
                
                if pre_avg != 0:
                    change_rate = (post_avg - pre_avg) / abs(pre_avg) * 100
                    metrics[f'{indicator}_change_rate'] = change_rate
                    metrics[f'{indicator}_pre_avg'] = pre_avg
                    metrics[f'{indicator}_post_avg'] = post_avg
                    
        return metrics
        
    def calculate_market_position_change(self, event: SuccessionEvent) -> Dict[str, float]:
        """
        事業継承前後の市場ポジション変化分析
        
        Parameters:
        -----------
        event : SuccessionEvent
            分析対象の事業継承イベント
            
        Returns:
        --------
        Dict[str, float]
            市場シェア・競争力の変化指標
        """
        if self.market_data is None:
            warnings.warn("市場データが読み込まれていないため、市場ポジション分析をスキップします")
            return {}
            
        company_market_data = self.market_data[
            self.market_data['company_code'] == event.company_code
        ].copy()
        
        if company_market_data.empty:
            return {}
            
        event_year = event.event_date.year
        
        # 市場シェア変化
        pre_share = company_market_data[
            company_market_data['fiscal_year'] == event_year - 1
        ]['market_share'].iloc[0] if not company_market_data[
            company_market_data['fiscal_year'] == event_year - 1
        ].empty else None
        
        post_share = company_market_data[
            company_market_data['fiscal_year'] == event_year + 2
        ]['market_share'].iloc[0] if not company_market_data[
            company_market_data['fiscal_year'] == event_year + 2
        ].empty else None
        
        metrics = {}
        if pre_share is not None and post_share is not None:
            metrics['market_share_change'] = post_share - pre_share
            metrics['market_share_change_rate'] = (post_share - pre_share) / pre_share * 100 if pre_share > 0 else 0
            
        return metrics
        
    def calculate_synergy_effects(self, event: SuccessionEvent) -> Dict[str, float]:
        """
        事業継承によるシナジー効果分析
        
        Parameters:
        -----------
        event : SuccessionEvent
            分析対象の事業継承イベント
            
        Returns:
        --------
        Dict[str, float]
            シナジー効果の定量評価
        """
        if self.financial_data is None:
            raise ValueError("財務データが読み込まれていません")
            
        # 基本的な財務指標変化を取得
        basic_metrics = self.calculate_pre_post_comparison(event)
        
        if not basic_metrics:
            return {}
            
        synergy_metrics = {}
        
        # 1. 収益シナジー（売上高の改善度合い）
        if 'revenue_change_rate' in basic_metrics:
            revenue_change = basic_metrics['revenue_change_rate']
            # 業界平均成長率と比較（仮に年3%とする）
            industry_growth = 3.0
            synergy_metrics['revenue_synergy'] = revenue_change - industry_growth
            
        # 2. コストシナジー（営業利益率の改善）
        if 'operating_income_change_rate' in basic_metrics and 'revenue_change_rate' in basic_metrics:
            operating_leverage = (basic_metrics['operating_income_change_rate'] - 
                                basic_metrics['revenue_change_rate'])
            synergy_metrics['cost_synergy'] = operating_leverage
            
        # 3. 資本効率シナジー（ROA、ROEの改善）
        if 'roa_change_rate' in basic_metrics:
            synergy_metrics['capital_efficiency_synergy_roa'] = basic_metrics['roa_change_rate']
            
        if 'roe_change_rate' in basic_metrics:
            synergy_metrics['capital_efficiency_synergy_roe'] = basic_metrics['roe_change_rate']
            
        # 4. 規模の経済効果（総資産あたり売上高の変化）
        if ('revenue_post_avg' in basic_metrics and 'total_assets_post_avg' in basic_metrics and
            'revenue_pre_avg' in basic_metrics and 'total_assets_pre_avg' in basic_metrics):
            
            pre_asset_turnover = basic_metrics['revenue_pre_avg'] / basic_metrics['total_assets_pre_avg']
            post_asset_turnover = basic_metrics['revenue_post_avg'] / basic_metrics['total_assets_post_avg']
            asset_turnover_change = (post_asset_turnover - pre_asset_turnover) / pre_asset_turnover * 100
            synergy_metrics['scale_economy_effect'] = asset_turnover_change
            
        return synergy_metrics
        
    def calculate_integration_success_score(self, event: SuccessionEvent) -> Dict[str, float]:
        """
        統合成功度スコアの計算
        
        Parameters:
        -----------
        event : SuccessionEvent
            分析対象の事業継承イベント
            
        Returns:
        --------
        Dict[str, float]
            総合的な事業継承成功度指標
        """
        # 各種分析結果を統合
        financial_metrics = self.calculate_pre_post_comparison(event)
        market_metrics = self.calculate_market_position_change(event)
        synergy_metrics = self.calculate_synergy_effects(event)
        
        if not financial_metrics:
            return {'integration_success_score': 0.0}
        
        success_components = []
        
        # 1. 財務パフォーマンス改善度 (重み: 40%)
        financial_score = 0
        financial_indicators = ['revenue_change_rate', 'operating_income_change_rate', 'roa_change_rate']
        valid_indicators = 0
        
        for indicator in financial_indicators:
            if indicator in financial_metrics:
                # 正の変化を評価（上限は+50%、下限は-50%）
                normalized_score = max(-50, min(50, financial_metrics[indicator])) / 50 * 100
                financial_score += normalized_score
                valid_indicators += 1
                
        if valid_indicators > 0:
            financial_score = financial_score / valid_indicators
            success_components.append(('financial_performance', financial_score, 0.4))
        
        # 2. シナジー効果実現度 (重み: 30%)
        synergy_score = 0
        if synergy_metrics:
            synergy_indicators = ['revenue_synergy', 'cost_synergy', 'capital_efficiency_synergy_roa']
            valid_synergies = 0
            
            for indicator in synergy_indicators:
                if indicator in synergy_metrics:
                    normalized_score = max(-50, min(50, synergy_metrics[indicator])) / 50 * 100
                    synergy_score += normalized_score
                    valid_synergies += 1
                    
            if valid_synergies > 0:
                synergy_score = synergy_score / valid_synergies
                success_components.append(('synergy_realization', synergy_score, 0.3))
        
        # 3. 市場競争力維持度 (重み: 20%)
        market_score = 0
        if market_metrics and 'market_share_change_rate' in market_metrics:
            market_change = market_metrics['market_share_change_rate']
            # 市場シェア維持または向上を評価
            market_score = max(-100, min(100, market_change * 2))  # 2倍重み付け
            success_components.append(('market_competitiveness', market_score, 0.2))
        
        # 4. リスク管理度 (重み: 10%)
        risk_score = 50  # 基準点
        if 'equity_change_rate' in financial_metrics:
            equity_change = financial_metrics['equity_change_rate']
            # 自己資本の維持・向上を評価
            if equity_change >= 0:
                risk_score += min(50, equity_change)
            else:
                risk_score += max(-50, equity_change)
        success_components.append(('risk_management', risk_score, 0.1))
        
        # 総合スコア計算
        total_score = sum(score * weight for _, score, weight in success_components)
        
        # 結果の構築
        result = {'integration_success_score': total_score}
        
        # 各コンポーネントのスコアも追加
        for component_name, score, weight in success_components:
            result[f'{component_name}_score'] = score
            result[f'{component_name}_weight'] = weight
            
        return result
        
    def analyze_succession_type_effectiveness(self) -> pd.DataFrame:
        """
        事業継承タイプ別の効果分析
        
        Returns:
        --------
        pd.DataFrame
            事業継承タイプ別の成功率・効果指標
        """
        if not self.succession_events:
            return pd.DataFrame()
            
        results = []
        
        for event in self.succession_events:
            success_metrics = self.calculate_integration_success_score(event)
            financial_metrics = self.calculate_pre_post_comparison(event)
            
            result_row = {
                'event_id': event.event_id,
                'company_code': event.company_code,
                'company_name': event.company_name,
                'event_type': event.event_type,
                'event_date': event.event_date,
                'success_score': success_metrics.get('integration_success_score', 0)
            }
            
            # 財務指標の変化も追加
            for key, value in financial_metrics.items():
                result_row[key] = value
                
            results.append(result_row)
            
        df = pd.DataFrame(results)
        
        if df.empty:
            return df
            
        # 事業継承タイプ別の統計
        type_analysis = df.groupby('event_type').agg({
            'success_score': ['mean', 'std', 'count'],
            'revenue_change_rate': 'mean',
            'operating_income_change_rate': 'mean',
            'roa_change_rate': 'mean'
        }).round(2)
        
        return type_analysis
        
    def identify_success_factors(self) -> Dict[str, List[Tuple[str, float]]]:
        """
        事業継承成功要因の特定
        
        Returns:
        --------
        Dict[str, List[Tuple[str, float]]]
            事業継承成功に寄与する要因のランキング
        """
        if not self.succession_events or self.financial_data is None:
            return {}
            
        # 全てのイベントについて成功スコアと各種指標の相関を分析
        analysis_data = []
        
        for event in self.succession_events:
            success_score = self.calculate_integration_success_score(event).get('integration_success_score', 0)
            
            # イベント前の企業状況を取得
            company_data = self.financial_data[
                (self.financial_data['company_code'] == event.company_code) &
                (self.financial_data['fiscal_year'] == event.event_date.year - 1)
            ]
            
            if not company_data.empty:
                row_data = {'success_score': success_score}
                for column in ['revenue', 'operating_income', 'total_assets', 'equity', 'roa', 'roe']:
                    if column in company_data.columns:
                        row_data[f'pre_{column}'] = company_data[column].iloc[0]
                        
                analysis_data.append(row_data)
        
        if len(analysis_data) < 2:
            return {}
            
        analysis_df = pd.DataFrame(analysis_data)
        
        # 相関分析
        correlations = analysis_df.corr()['success_score'].drop('success_score').abs().sort_values(ascending=False)
        
        success_factors = {
            'top_success_factors': [(factor.replace('pre_', ''), corr) for factor, corr in correlations.head(10).items()],
            'correlation_matrix': correlations.to_dict()
        }
        
        return success_factors
        
    def generate_succession_metrics_report(self) -> Dict[str, any]:
        """
        事業継承成功度の総合レポート生成
        
        Returns:
        --------
        Dict[str, any]
            包括的な事業継承分析レポート
        """
        if not self.succession_events:
            return {'error': '事業継承イベントが登録されていません'}
            
        report = {
            'summary': {
                'total_events': len(self.succession_events),
                'event_types': list(set([event.event_type for event in self.succession_events])),
                'analysis_period': {
                    'earliest_event': min([event.event_date for event in self.succession_events]),
                    'latest_event': max([event.event_date for event in self.succession_events])
                }
            },
            'type_effectiveness': self.analyze_succession_type_effectiveness(),
            'success_factors': self.identify_success_factors()
        }
        
        # 個別イベント分析結果
        individual_results = []
        for event in self.succession_events:
            event_result = {
                'event_info': {
                    'event_id': event.event_id,
                    'company_name': event.company_name,
                    'event_type': event.event_type,
                    'event_date': event.event_date
                },
                'success_metrics': self.calculate_integration_success_score(event),
                'financial_changes': self.calculate_pre_post_comparison(event),
                'synergy_effects': self.calculate_synergy_effects(event)
            }
            individual_results.append(event_result)
            
        report['individual_analyses'] = individual_results
        
        return report

# 使用例とテスト用のサンプルデータ
def create_sample_data():
    """サンプルデータの作成（テスト用）"""
    
    # サンプル財務データ
    financial_data = pd.DataFrame({
        'company_code': ['TOSHIBA'] * 10 + ['SHARP'] * 10,
        'fiscal_year': list(range(2015, 2025)) + list(range(2013, 2023)),
        'revenue': np.random.normal(5000000, 500000, 20),
        'operating_income': np.random.normal(200000, 100000, 20),
        'net_income': np.random.normal(150000, 120000, 20),
        'total_assets': np.random.normal(8000000, 800000, 20),
        'equity': np.random.normal(3000000, 300000, 20),
        'roa': np.random.normal(2.0, 1.0, 20),
        'roe': np.random.normal(5.0, 2.0, 20)
    })
    
    # サンプル事業継承イベント
    events = [
        SuccessionEvent(
            event_id='TOSHIBA_MEMORY_SPINOFF_2018',
            company_code='TOSHIBA',
            company_name='東芝',
            event_type='spinoff',
            event_date=datetime(2018, 6, 1),
            pre_event_period=3,
            post_event_period=3,
            description='東芝メモリ分社化'
        ),
        SuccessionEvent(
            event_id='SHARP_FOXCONN_ACQUISITION_2016',
            company_code='SHARP',
            company_name='シャープ',
            event_type='acquisition',
            event_date=datetime(2016, 8, 12),
            pre_event_period=3,
            post_event_period=3,
            parent_company='Foxconn',
            description='鴻海精密工業による買収'
        )
    ]
    
    return financial_data, events

if __name__ == "__main__":
    # テスト実行
    financial_data, sample_events = create_sample_data()
    
    # SuccessionMetricsインスタンス作成
    succession_analyzer = SuccessionMetrics()
    succession_analyzer.load_financial_data(financial_data)
    
    # イベント登録
    for event in sample_events:
        succession_analyzer.register_succession_event(event)
    
    # 分析実行
    report = succession_analyzer.generate_succession_metrics_report()
    
    print("=== A2AI 事業継承成功度分析レポート ===")
    print(f"分析対象イベント数: {report['summary']['total_events']}")
    print(f"イベント種類: {report['summary']['event_types']}")
    
    for result in report['individual_analyses']:
        event_info = result['event_info']
        success_score = result['success_metrics'].get('integration_success_score', 0)
        print(f"\n{event_info['company_name']} ({event_info['event_type']}) - 成功度スコア: {success_score:.1f}")