"""
バイアス検出モジュール
日本企業の競争力分析におけるバイアスを検出・評価する
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from enum import Enum
import re
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings

class BiasType(Enum):
    """バイアスの種類を定義"""
    CONFIRMATION = "confirmation_bias"  # 確証バイアス
    ANCHORING = "anchoring_bias"       # アンカリング効果
    AVAILABILITY = "availability_bias"  # 利用可能性ヒューリスティック
    RECENCY = "recency_bias"           # 近時効果
    SURVIVORSHIP = "survivorship_bias" # 生存者バイアス
    NATIONALITY = "nationality_bias"   # 国籍バイアス
    SIZE = "size_bias"                 # 規模バイアス
    INDUSTRY = "industry_bias"         # 業界バイアス

@dataclass
class BiasDetectionResult:
    """バイアス検出結果を格納するデータクラス"""
    bias_type: BiasType
    severity: float  # 0-1のスコア
    description: str
    evidence: List[str]
    recommendations: List[str]
    confidence: float  # 検出の信頼度

class BiasDetector:
    """バイアス検出のメインクラス"""
    
    def __init__(self):
        self.japanese_companies = self._load_japanese_companies()
        self.market_categories = {
            'high_share': ['ロボット', '内視鏡', '工作機械', '電子材料', '精密測定機器'],
            'declining_share': ['自動車', '鉄鋼', 'スマート家電', 'バッテリー', 'PC・周辺機器'],
            'lost_share': ['家電', '半導体', 'スマートフォン', 'PC', '通信機器']
        }
        
    def _load_japanese_companies(self) -> Dict[str, List[str]]:
        """日本企業データを読み込み"""
        return {
            'high_share': [
                'ファナック', '安川電機', '川崎重工業', '不二越', 'デンソーウェーブ',
                'オリンパス', 'HOYA', '富士フイルム', 'キヤノンメディカルシステムズ',
                'DMG森精機', 'ヤマザキマザック', 'オークマ', '牧野フライス製作所',
                '村田製作所', 'TDK', '京セラ', '太陽誘電', 'キーエンス', '島津製作所'
            ],
            'declining_share': [
                'トヨタ自動車', '日産自動車', 'ホンダ', 'スズキ', 'マツダ',
                '日本製鉄', 'JFEホールディングス', '神戸製鋼所',
                'パナソニック', 'シャープ', 'パナソニックエナジー'
            ],
            'lost_share': [
                'ソニー', '東芝', 'NEC', '富士通', '日立製作所',
                'ルネサスエレクトロニクス', '三菱電機', '京セラ'
            ]
        }
    
    def detect_all_biases(self, analysis_data: Dict[str, Any]) -> List[BiasDetectionResult]:
        """すべてのバイアスタイプを検出"""
        results = []
        
        # 各バイアス検出メソッドを実行
        results.extend(self._detect_confirmation_bias(analysis_data))
        results.extend(self._detect_anchoring_bias(analysis_data))
        results.extend(self._detect_availability_bias(analysis_data))
        results.extend(self._detect_recency_bias(analysis_data))
        results.extend(self._detect_survivorship_bias(analysis_data))
        results.extend(self._detect_nationality_bias(analysis_data))
        results.extend(self._detect_size_bias(analysis_data))
        results.extend(self._detect_industry_bias(analysis_data))
        
        return sorted(results, key=lambda x: x.severity, reverse=True)
    
    def _detect_confirmation_bias(self, data: Dict[str, Any]) -> List[BiasDetectionResult]:
        """確証バイアスを検出"""
        results = []
        
        # データソースの偏りをチェック
        if 'data_sources' in data:
            japanese_sources = sum(1 for src in data['data_sources'] 
                                    if any(jp in src.lower() for jp in ['japan', 'nippon', 'nikkei']))
            total_sources = len(data['data_sources'])
            
            if total_sources > 0 and japanese_sources / total_sources > 0.7:
                severity = min(japanese_sources / total_sources, 1.0)
                results.append(BiasDetectionResult(
                    bias_type=BiasType.CONFIRMATION,
                    severity=severity,
                    description="日本関連の情報源に偏重している可能性",
                    evidence=[f"日本関連ソース: {japanese_sources}/{total_sources}"],
                    recommendations=[
                        "国際的な情報源を追加する",
                        "競合他国の視点も含める",
                        "中立的な第三者機関の分析を参照する"
                    ],
                    confidence=0.8
                ))
        
        # ポジティブ/ネガティブ判定の偏りをチェック
        if 'analysis_text' in data:
            text = data['analysis_text']
            positive_keywords = ['強み', '優位', '成功', '技術力', 'シェア拡大']
            negative_keywords = ['課題', '劣勢', '後退', '競争激化', 'シェア低下']
            
            pos_count = sum(text.count(word) for word in positive_keywords)
            neg_count = sum(text.count(word) for word in negative_keywords)
            
            if pos_count + neg_count > 0:
                pos_ratio = pos_count / (pos_count + neg_count)
                if pos_ratio > 0.8 or pos_ratio < 0.2:
                    severity = abs(pos_ratio - 0.5) * 2
                    results.append(BiasDetectionResult(
                        bias_type=BiasType.CONFIRMATION,
                        severity=severity,
                        description="分析の評価が極端に偏っている",
                        evidence=[f"ポジティブ表現率: {pos_ratio:.1%}"],
                        recommendations=[
                            "より中立的な視点で分析する",
                            "デメリットとメリットを均等に検討する"
                        ],
                        confidence=0.7
                    ))
        
        return results
    
    def _detect_anchoring_bias(self, data: Dict[str, Any]) -> List[BiasDetectionResult]:
        """アンカリング効果を検出"""
        results = []
        
        if 'historical_data' in data:
            hist_data = data['historical_data']
            
            # 過去の成功体験への過度な依存をチェック
            past_success_mentions = 0
            recent_performance = 0
            
            for record in hist_data:
                if 'year' in record and record['year'] < 2010:
                    if any(keyword in str(record).lower() 
                            for keyword in ['世界一', '最大手', '圧倒的', '独占']):
                        past_success_mentions += 1
                elif 'year' in record and record['year'] >= 2020:
                    if 'performance_score' in record:
                        recent_performance = record['performance_score']
            
            if past_success_mentions > 3 and recent_performance < 0.6:
                severity = min(past_success_mentions / 5.0, 1.0)
                results.append(BiasDetectionResult(
                    bias_type=BiasType.ANCHORING,
                    severity=severity,
                    description="過去の成功体験に固執している可能性",
                    evidence=[f"過去の成功言及: {past_success_mentions}回"],
                    recommendations=[
                        "現在の競争環境を重視した分析を行う",
                        "過去の前提条件の変化を考慮する"
                    ],
                    confidence=0.75
                ))
        
        return results
    
    def _detect_availability_bias(self, data: Dict[str, Any]) -> List[BiasDetectionResult]:
        """利用可能性ヒューリスティックを検出"""
        results = []
        
        if 'company_mentions' in data:
            mentions = data['company_mentions']
            
            # 有名企業への過度な注目をチェック
            famous_companies = ['トヨタ', 'ソニー', 'パナソニック', 'ホンダ', '日産']
            total_mentions = sum(mentions.values())
            famous_mentions = sum(mentions.get(company, 0) for company in famous_companies)
            
            if total_mentions > 0 and famous_mentions / total_mentions > 0.6:
                severity = famous_mentions / total_mentions
                results.append(BiasDetectionResult(
                    bias_type=BiasType.AVAILABILITY,
                    severity=severity,
                    description="有名企業に分析が偏っている",
                    evidence=[f"有名企業言及率: {famous_mentions/total_mentions:.1%}"],
                    recommendations=[
                        "中小・中堅企業も含めた包括的な分析を行う",
                        "業界全体の構造を把握する"
                    ],
                    confidence=0.8
                ))
        
        return results
    
    def _detect_recency_bias(self, data: Dict[str, Any]) -> List[BiasDetectionResult]:
        """近時効果を検出"""
        results = []
        
        if 'time_weighted_data' in data:
            time_data = data['time_weighted_data']
            recent_weight = sum(w for year, w in time_data.items() if int(year) >= 2022)
            total_weight = sum(time_data.values())
            
            if total_weight > 0 and recent_weight / total_weight > 0.8:
                severity = recent_weight / total_weight
                results.append(BiasDetectionResult(
                    bias_type=BiasType.RECENCY,
                    severity=severity,
                    description="最近の出来事に過度に影響されている",
                    evidence=[f"直近データ重み: {recent_weight/total_weight:.1%}"],
                    recommendations=[
                        "長期的なトレンドも考慮する",
                        "一時的な変動と構造的変化を区別する"
                    ],
                    confidence=0.7
                ))
        
        return results
    
    def _detect_survivorship_bias(self, data: Dict[str, Any]) -> List[BiasDetectionResult]:
        """生存者バイアスを検出"""
        results = []
        
        if 'companies_analyzed' in data:
            companies = data['companies_analyzed']
            
            # 撤退・倒産企業の分析不足をチェック
            surviving_companies = [c for c in companies if not any(
                keyword in c.lower() for keyword in ['撤退', '解散', '破綻', '売却']
            )]
            
            survival_rate = len(surviving_companies) / len(companies) if companies else 0
            
            if survival_rate > 0.9 and len(companies) > 10:
                severity = survival_rate
                results.append(BiasDetectionResult(
                    bias_type=BiasType.SURVIVORSHIP,
                    severity=severity,
                    description="成功企業のみに焦点を当てている可能性",
                    evidence=[f"生存企業率: {survival_rate:.1%}"],
                    recommendations=[
                        "失敗事例からも学習を行う",
                        "撤退企業の要因分析を含める"
                    ],
                    confidence=0.75
                ))
        
        return results
    
    def _detect_nationality_bias(self, data: Dict[str, Any]) -> List[BiasDetectionResult]:
        """国籍バイアスを検出"""
        results = []
        
        if 'competitive_analysis' in data:
            analysis = data['competitive_analysis']
            
            # 日本企業への評価の甘さをチェック
            japanese_scores = []
            foreign_scores = []
            
            for company, score in analysis.items():
                if any(jp_company in company for jp_company in self.japanese_companies['high_share'] + 
                        self.japanese_companies['declining_share'] + self.japanese_companies['lost_share']):
                    japanese_scores.append(score)
                else:
                    foreign_scores.append(score)
            
            if japanese_scores and foreign_scores:
                jp_avg = np.mean(japanese_scores)
                foreign_avg = np.mean(foreign_scores)
                
                if jp_avg - foreign_avg > 0.3:  # 日本企業の評価が0.3以上高い
                    severity = min((jp_avg - foreign_avg) / 0.5, 1.0)
                    results.append(BiasDetectionResult(
                        bias_type=BiasType.NATIONALITY,
                        severity=severity,
                        description="日本企業への評価が甘い可能性",
                        evidence=[f"日本企業平均: {jp_avg:.2f}, 外国企業平均: {foreign_avg:.2f}"],
                        recommendations=[
                            "評価基準を統一する",
                            "第三者による評価も参考にする"
                        ],
                        confidence=0.8
                    ))
        
        return results
    
    def _detect_size_bias(self, data: Dict[str, Any]) -> List[BiasDetectionResult]:
        """規模バイアスを検出"""
        results = []
        
        if 'company_sizes' in data:
            sizes = data['company_sizes']
            large_company_ratio = sum(1 for s in sizes.values() if s == 'large') / len(sizes)
            
            if large_company_ratio > 0.8:
                severity = large_company_ratio
                results.append(BiasDetectionResult(
                    bias_type=BiasType.SIZE,
                    severity=severity,
                    description="大企業に分析が偏っている",
                    evidence=[f"大企業比率: {large_company_ratio:.1%}"],
                    recommendations=[
                        "中小企業の革新性も評価する",
                        "企業規模別の分析を行う"
                    ],
                    confidence=0.7
                ))
        
        return results
    
    def _detect_industry_bias(self, data: Dict[str, Any]) -> List[BiasDetectionResult]:
        """業界バイアスを検出"""
        results = []
        
        if 'industry_focus' in data:
            industries = data['industry_focus']
            
            # 特定業界への過度な集中をチェック
            max_industry_ratio = max(industries.values()) if industries else 0
            
            if max_industry_ratio > 0.6:
                dominant_industry = max(industries, key=industries.get)
                severity = max_industry_ratio
                results.append(BiasDetectionResult(
                    bias_type=BiasType.INDUSTRY,
                    severity=severity,
                    description=f"{dominant_industry}業界に分析が偏っている",
                    evidence=[f"{dominant_industry}: {max_industry_ratio:.1%}"],
                    recommendations=[
                        "業界横断的な分析を行う",
                        "業界特有の要因を考慮する"
                    ],
                    confidence=0.75
                ))
        
        return results
    
    def generate_bias_report(self, bias_results: List[BiasDetectionResult]) -> str:
        """バイアス検出結果のレポートを生成"""
        if not bias_results:
            return "バイアスは検出されませんでした。"
        
        report = "## バイアス検出レポート\n\n"
        
        # 重要度順にソート
        high_severity = [r for r in bias_results if r.severity > 0.7]
        medium_severity = [r for r in bias_results if 0.4 <= r.severity <= 0.7]
        low_severity = [r for r in bias_results if r.severity < 0.4]
        
        if high_severity:
            report += "### 🚨 高リスク（重要度: 高）\n"
            for result in high_severity:
                report += f"- **{result.bias_type.value}** (重要度: {result.severity:.1%})\n"
                report += f"  - {result.description}\n"
                report += f"  - 推奨対策: {', '.join(result.recommendations)}\n\n"
        
        if medium_severity:
            report += "### ⚠️ 中リスク（重要度: 中）\n"
            for result in medium_severity:
                report += f"- **{result.bias_type.value}** (重要度: {result.severity:.1%})\n"
                report += f"  - {result.description}\n\n"
        
        if low_severity:
            report += "### ℹ️ 低リスク（重要度: 低）\n"
            for result in low_severity:
                report += f"- {result.bias_type.value}: {result.description}\n"
        
        # 総合的な推奨事項
        report += "\n### 📋 総合的な推奨事項\n"
        report += "1. 多様な情報源からデータを収集する\n"
        report += "2. 定量的な指標と定性的な評価のバランスを取る\n"
        report += "3. 国際的な視点を含める\n"
        report += "4. 長期的トレンドと短期的変動を区別する\n"
        report += "5. 失敗事例からも学習する\n"
        
        return report
    
    def get_bias_score(self, bias_results: List[BiasDetectionResult]) -> float:
        """総合的なバイアススコアを計算（0-1、0が最良）"""
        if not bias_results:
            return 0.0
        
        # 重み付き平均を計算
        weighted_scores = []
        for result in bias_results:
            weight = result.confidence * (1.0 if result.severity > 0.7 else 
                                        0.7 if result.severity > 0.4 else 0.4)
            weighted_scores.append(result.severity * weight)
        
        return np.mean(weighted_scores)

# 使用例とテスト用のサンプルデータ
def demo_bias_detection():
    """バイアス検出のデモンストレーション"""
    
    # サンプル分析データ
    sample_data = {
        'data_sources': [
            'nikkei.com', 'reuters.com', 'japan-times.com', 'bloomberg.jp'
        ],
        'analysis_text': '日本企業の技術力は優秀で、今後も競争優位を保つと予想される。強みを活かした戦略が重要。',
        'company_mentions': {
            'トヨタ': 15, 'ソニー': 12, 'パナソニック': 8, 
            'ファナック': 3, 'オムロン': 2
        },
        'competitive_analysis': {
            'トヨタ': 0.85, 'テスラ': 0.65, 'ソニー': 0.80, 
            'サムスン': 0.60, 'ファナック': 0.90, 'ABB': 0.70
        },
        'companies_analyzed': [
            'トヨタ', 'ソニー', 'パナソニック', 'ホンダ', '日産',
            '三菱電機', 'オムロン', 'ファナック'
        ],
        'time_weighted_data': {
            '2020': 0.1, '2021': 0.15, '2022': 0.3, '2023': 0.35, '2024': 0.1
        },
        'company_sizes': {
            'トヨタ': 'large', 'ソニー': 'large', 'パナソニック': 'large',
            'ホンダ': 'large', '日産': 'large'
        },
        'industry_focus': {
            '自動車': 0.4, 'エレクトロニクス': 0.35, 'ロボット': 0.15, 'その他': 0.1
        }
    }
    
    detector = BiasDetector()
    results = detector.detect_all_biases(sample_data)
    
    print("=== バイアス検出結果 ===")
    for result in results:
        print(f"{result.bias_type.value}: {result.severity:.2f}")
        print(f"  {result.description}")
        print(f"  推奨: {result.recommendations[0] if result.recommendations else 'なし'}")
        print()
    
    print("=== レポート ===")
    print(detector.generate_bias_report(results))
    print(f"\n総合バイアススコア: {detector.get_bias_score(results):.2f}")

if __name__ == "__main__":
    demo_bias_detection()