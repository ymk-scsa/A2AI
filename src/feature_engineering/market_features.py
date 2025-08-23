"""
A2AI - Advanced Financial Analysis AI
Market Features Module

市場カテゴリ特徴量生成モジュール
- 高シェア市場 vs シェア低下市場 vs シェア失失市場の特徴量
- 市場成熟度、競争環境、技術革新度等の特徴量生成
- 企業の市場ポジション、参入タイミング、市場依存度分析

企業分類:
- 高シェア市場: ロボット、内視鏡、工作機械、電子材料、精密測定機器 (50社)
- シェア低下市場: 自動車(EV含)、鉄鋼、スマート家電、バッテリー、PC・周辺機器 (50社)
- シェア失失市場: 家電、半導体、スマートフォン、PC、通信機器 (50社)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime, timedelta
import warnings

# 設定とログ
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MarketCategory(Enum):
    """市場カテゴリ分類"""
    HIGH_SHARE = "high_share"           # 世界シェア高市場
    DECLINING_SHARE = "declining_share" # シェア低下市場  
    LOST_SHARE = "lost_share"          # シェア失失市場

class MarketType(Enum):
    """市場タイプ分類"""
    # 高シェア市場
    ROBOT = "robot"                    # ロボット市場
    ENDOSCOPE = "endoscope"           # 内視鏡市場
    MACHINE_TOOL = "machine_tool"     # 工作機械市場
    ELECTRONIC_MATERIALS = "electronic_materials"  # 電子材料市場
    PRECISION_MEASUREMENT = "precision_measurement" # 精密測定機器市場
    
    # シェア低下市場
    AUTOMOTIVE = "automotive"         # 自動車市場（EV含む）
    STEEL = "steel"                   # 鉄鋼市場
    SMART_APPLIANCES = "smart_appliances"  # スマート家電市場
    BATTERY = "battery"               # バッテリー市場（EV用）
    PC_PERIPHERALS = "pc_peripherals" # PC・周辺機器市場
    
    # シェア失失市場
    HOME_APPLIANCES = "home_appliances"    # 家電市場
    SEMICONDUCTOR = "semiconductor"        # 半導体市場
    SMARTPHONE = "smartphone"             # スマートフォン市場
    PERSONAL_COMPUTER = "personal_computer" # PC市場
    TELECOM_EQUIPMENT = "telecom_equipment" # 通信機器市場

@dataclass
class MarketCharacteristics:
    """市場特性データクラス"""
    category: MarketCategory
    market_type: MarketType
    tech_intensity: float      # 技術集約度 (0-1)
    innovation_cycle: float    # イノベーションサイクル年数
    global_competition: float  # グローバル競争度 (0-1)
    entry_barriers: float      # 参入障壁 (0-1)
    market_maturity: float     # 市場成熟度 (0-1)
    b2b_ratio: float          # B2B比率 (0-1)
    customization_level: float # カスタマイゼーション度 (0-1)
    capital_intensity: float   # 資本集約度 (0-1)

class MarketFeaturesGenerator:
    """市場特徴量生成クラス"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初期化
        
        Args:
            config_path: 設定ファイルパス
        """
        self.market_characteristics = self._initialize_market_characteristics()
        self.company_market_mapping = self._initialize_company_market_mapping()
        
    def _initialize_market_characteristics(self) -> Dict[MarketType, MarketCharacteristics]:
        """市場特性の初期化"""
        return {
            # === 高シェア市場 ===
            MarketType.ROBOT: MarketCharacteristics(
                category=MarketCategory.HIGH_SHARE,
                market_type=MarketType.ROBOT,
                tech_intensity=0.95,     # 超高技術集約
                innovation_cycle=5.0,    # 5年サイクル
                global_competition=0.7,  # 中程度の競争
                entry_barriers=0.9,      # 極めて高い参入障壁
                market_maturity=0.6,     # 成長期-成熟期
                b2b_ratio=0.95,         # ほぼB2B
                customization_level=0.8, # 高カスタマイゼーション
                capital_intensity=0.85   # 高資本集約度
            ),
            
            MarketType.ENDOSCOPE: MarketCharacteristics(
                category=MarketCategory.HIGH_SHARE,
                market_type=MarketType.ENDOSCOPE,
                tech_intensity=0.9,
                innovation_cycle=7.0,
                global_competition=0.6,
                entry_barriers=0.95,     # 医療機器規制で極高参入障壁
                market_maturity=0.7,
                b2b_ratio=0.98,         # 医療機関向け
                customization_level=0.7,
                capital_intensity=0.8
            ),
            
            MarketType.MACHINE_TOOL: MarketCharacteristics(
                category=MarketCategory.HIGH_SHARE,
                market_type=MarketType.MACHINE_TOOL,
                tech_intensity=0.85,
                innovation_cycle=8.0,    # 比較的長期サイクル
                global_competition=0.75,
                entry_barriers=0.8,
                market_maturity=0.8,     # 成熟市場
                b2b_ratio=1.0,          # 完全B2B
                customization_level=0.9, # 極めて高カスタマイゼーション
                capital_intensity=0.9
            ),
            
            MarketType.ELECTRONIC_MATERIALS: MarketCharacteristics(
                category=MarketCategory.HIGH_SHARE,
                market_type=MarketType.ELECTRONIC_MATERIALS,
                tech_intensity=0.9,
                innovation_cycle=3.0,    # 短サイクル
                global_competition=0.8,
                entry_barriers=0.85,
                market_maturity=0.5,     # 成長期
                b2b_ratio=1.0,
                customization_level=0.6,
                capital_intensity=0.85
            ),
            
            MarketType.PRECISION_MEASUREMENT: MarketCharacteristics(
                category=MarketCategory.HIGH_SHARE,
                market_type=MarketType.PRECISION_MEASUREMENT,
                tech_intensity=0.95,     # 最高技術集約
                innovation_cycle=6.0,
                global_competition=0.7,
                entry_barriers=0.9,
                market_maturity=0.7,
                b2b_ratio=0.9,
                customization_level=0.85,
                capital_intensity=0.8
            ),
            
            # === シェア低下市場 ===
            MarketType.AUTOMOTIVE: MarketCharacteristics(
                category=MarketCategory.DECLINING_SHARE,
                market_type=MarketType.AUTOMOTIVE,
                tech_intensity=0.8,      # EV化で技術重要性増
                innovation_cycle=4.0,    # EV化で短サイクル化
                global_competition=0.95, # 極めて激しい競争
                entry_barriers=0.7,      # EV新規参入で低下
                market_maturity=0.9,     # 成熟市場
                b2b_ratio=0.1,          # 主にB2C
                customization_level=0.5,
                capital_intensity=0.9
            ),
            
            MarketType.STEEL: MarketCharacteristics(
                category=MarketCategory.DECLINING_SHARE,
                market_type=MarketType.STEEL,
                tech_intensity=0.5,      # 比較的低技術集約
                innovation_cycle=15.0,   # 長期サイクル
                global_competition=0.9,  # 価格競争激化
                entry_barriers=0.6,      # 中国参入で低下
                market_maturity=0.95,    # 完全成熟市場
                b2b_ratio=0.95,
                customization_level=0.3,
                capital_intensity=0.95   # 極高資本集約度
            ),
            
            MarketType.SMART_APPLIANCES: MarketCharacteristics(
                category=MarketCategory.DECLINING_SHARE,
                market_type=MarketType.SMART_APPLIANCES,
                tech_intensity=0.6,
                innovation_cycle=2.0,    # 短サイクル化
                global_competition=0.9,
                entry_barriers=0.4,      # 低参入障壁
                market_maturity=0.8,
                b2b_ratio=0.2,
                customization_level=0.4,
                capital_intensity=0.6
            ),
            
            MarketType.BATTERY: MarketCharacteristics(
                category=MarketCategory.DECLINING_SHARE,
                market_type=MarketType.BATTERY,
                tech_intensity=0.85,
                innovation_cycle=2.5,    # 急速な技術革新
                global_competition=0.95, # 中韓勢力との激戦
                entry_barriers=0.7,
                market_maturity=0.3,     # 成長初期段階
                b2b_ratio=0.8,          # 自動車メーカー向け
                customization_level=0.6,
                capital_intensity=0.9
            ),
            
            MarketType.PC_PERIPHERALS: MarketCharacteristics(
                category=MarketCategory.DECLINING_SHARE,
                market_type=MarketType.PC_PERIPHERALS,
                tech_intensity=0.6,
                innovation_cycle=1.5,    # 極短サイクル
                global_competition=0.85,
                entry_barriers=0.3,      # 低参入障壁
                market_maturity=0.9,
                b2b_ratio=0.4,
                customization_level=0.2,
                capital_intensity=0.4
            ),
            
            # === シェア失失市場 ===
            MarketType.HOME_APPLIANCES: MarketCharacteristics(
                category=MarketCategory.LOST_SHARE,
                market_type=MarketType.HOME_APPLIANCES,
                tech_intensity=0.4,      # 低技術集約化
                innovation_cycle=3.0,
                global_competition=0.95, # 価格競争激化
                entry_barriers=0.2,      # 極低参入障壁
                market_maturity=0.95,    # 完全成熟
                b2b_ratio=0.1,
                customization_level=0.2,
                capital_intensity=0.5
            ),
            
            MarketType.SEMICONDUCTOR: MarketCharacteristics(
                category=MarketCategory.LOST_SHARE,
                market_type=MarketType.SEMICONDUCTOR,
                tech_intensity=0.98,     # 最高技術集約
                innovation_cycle=1.5,    # 極短ムーアの法則
                global_competition=0.98, # 極限競争
                entry_barriers=0.95,     # 巨額投資必要
                market_maturity=0.8,
                b2b_ratio=0.9,
                customization_level=0.4,
                capital_intensity=0.98   # 最高資本集約度
            ),
            
            MarketType.SMARTPHONE: MarketCharacteristics(
                category=MarketCategory.LOST_SHARE,
                market_type=MarketType.SMARTPHONE,
                tech_intensity=0.8,
                innovation_cycle=1.0,    # 年次イノベーション
                global_competition=0.95,
                entry_barriers=0.4,      # プラットフォーム依存
                market_maturity=0.9,
                b2b_ratio=0.2,
                customization_level=0.3,
                capital_intensity=0.7
            ),
            
            MarketType.PERSONAL_COMPUTER: MarketCharacteristics(
                category=MarketCategory.LOST_SHARE,
                market_type=MarketType.PERSONAL_COMPUTER,
                tech_intensity=0.7,
                innovation_cycle=2.0,
                global_competition=0.9,
                entry_barriers=0.3,
                market_maturity=0.95,    # 衰退期
                b2b_ratio=0.5,
                customization_level=0.3,
                capital_intensity=0.5
            ),
            
            MarketType.TELECOM_EQUIPMENT: MarketCharacteristics(
                category=MarketCategory.LOST_SHARE,
                market_type=MarketType.TELECOM_EQUIPMENT,
                tech_intensity=0.9,
                innovation_cycle=5.0,    # 5G等大型サイクル
                global_competition=0.9,
                entry_barriers=0.8,
                market_maturity=0.8,
                b2b_ratio=0.95,
                customization_level=0.6,
                capital_intensity=0.8
            )
        }
    
    def _initialize_company_market_mapping(self) -> Dict[str, MarketType]:
        """企業と市場タイプのマッピング初期化"""
        return {
            # === 高シェア市場（ロボット） ===
            "ファナック": MarketType.ROBOT,
            "安川電機": MarketType.ROBOT,
            "川崎重工業": MarketType.ROBOT,
            "不二越": MarketType.ROBOT,
            "デンソーウェーブ": MarketType.ROBOT,
            "三菱電機": MarketType.ROBOT,
            "オムロン": MarketType.ROBOT,
            "THK": MarketType.ROBOT,
            "NSK": MarketType.ROBOT,
            "IHI": MarketType.ROBOT,
            
            # === 高シェア市場（内視鏡） ===
            "オリンパス": MarketType.ENDOSCOPE,
            "HOYA": MarketType.ENDOSCOPE,
            "富士フイルム": MarketType.ENDOSCOPE,
            "キヤノンメディカルシステムズ": MarketType.ENDOSCOPE,
            "島津製作所": MarketType.ENDOSCOPE,
            "コニカミノルタ": MarketType.ENDOSCOPE,
            "ソニー（メディカル）": MarketType.ENDOSCOPE,
            "トプコン": MarketType.ENDOSCOPE,
            "エムスリー": MarketType.ENDOSCOPE,
            "日立製作所（ヘルスケア）": MarketType.ENDOSCOPE,
            
            # === 高シェア市場（工作機械） ===
            "DMG森精機": MarketType.MACHINE_TOOL,
            "ヤマザキマザック": MarketType.MACHINE_TOOL,
            "オークマ": MarketType.MACHINE_TOOL,
            "牧野フライス製作所": MarketType.MACHINE_TOOL,
            "ジェイテクト": MarketType.MACHINE_TOOL,
            "東芝機械": MarketType.MACHINE_TOOL,
            "アマダ": MarketType.MACHINE_TOOL,
            "ソディック": MarketType.MACHINE_TOOL,
            "三菱重工工作機械": MarketType.MACHINE_TOOL,
            "シギヤ精機製作所": MarketType.MACHINE_TOOL,
            
            # === 高シェア市場（電子材料） ===
            "村田製作所": MarketType.ELECTRONIC_MATERIALS,
            "TDK": MarketType.ELECTRONIC_MATERIALS,
            "京セラ": MarketType.ELECTRONIC_MATERIALS,
            "太陽誘電": MarketType.ELECTRONIC_MATERIALS,
            "日本特殊陶業": MarketType.ELECTRONIC_MATERIALS,
            "ローム": MarketType.ELECTRONIC_MATERIALS,
            "プロテリアル": MarketType.ELECTRONIC_MATERIALS,
            "住友電工": MarketType.ELECTRONIC_MATERIALS,
            "日東電工": MarketType.ELECTRONIC_MATERIALS,
            "日本碍子": MarketType.ELECTRONIC_MATERIALS,
            
            # === 高シェア市場（精密測定機器） ===
            "キーエンス": MarketType.PRECISION_MEASUREMENT,
            "島津製作所": MarketType.PRECISION_MEASUREMENT,  # 重複（複数市場参入）
            "堀場製作所": MarketType.PRECISION_MEASUREMENT,
            "東京精密": MarketType.PRECISION_MEASUREMENT,
            "ミツトヨ": MarketType.PRECISION_MEASUREMENT,
            "オリンパス": MarketType.PRECISION_MEASUREMENT,  # 重複
            "日本電産": MarketType.PRECISION_MEASUREMENT,
            "リオン": MarketType.PRECISION_MEASUREMENT,
            "アルバック": MarketType.PRECISION_MEASUREMENT,
            "ナブテスコ": MarketType.PRECISION_MEASUREMENT,
            
            # === シェア低下市場（自動車） ===
            "トヨタ自動車": MarketType.AUTOMOTIVE,
            "日産自動車": MarketType.AUTOMOTIVE,
            "ホンダ": MarketType.AUTOMOTIVE,
            "スズキ": MarketType.AUTOMOTIVE,
            "マツダ": MarketType.AUTOMOTIVE,
            "SUBARU": MarketType.AUTOMOTIVE,
            "いすゞ自動車": MarketType.AUTOMOTIVE,
            "三菱自動車": MarketType.AUTOMOTIVE,
            "ダイハツ工業": MarketType.AUTOMOTIVE,
            "日野自動車": MarketType.AUTOMOTIVE,
            
            # === シェア低下市場（鉄鋼） ===
            "日本製鉄": MarketType.STEEL,
            "JFEホールディングス": MarketType.STEEL,
            "神戸製鋼所": MarketType.STEEL,
            "日新製鋼": MarketType.STEEL,
            "大同特殊鋼": MarketType.STEEL,
            "山陽特殊製鋼": MarketType.STEEL,
            "愛知製鋼": MarketType.STEEL,
            "中部鋼鈑": MarketType.STEEL,
            "淀川製鋼所": MarketType.STEEL,
            "日立金属": MarketType.STEEL,
            
            # === シェア低下市場（スマート家電） ===
            "パナソニック": MarketType.SMART_APPLIANCES,
            "シャープ": MarketType.SMART_APPLIANCES,
            "ソニー（家電部門）": MarketType.SMART_APPLIANCES,
            "東芝ライフスタイル": MarketType.SMART_APPLIANCES,
            "日立グローバルライフソリューションズ": MarketType.SMART_APPLIANCES,
            "アイリスオーヤマ": MarketType.SMART_APPLIANCES,
            "三菱電機": MarketType.SMART_APPLIANCES,  # 重複
            "象印マホービン": MarketType.SMART_APPLIANCES,
            "タイガー魔法瓶": MarketType.SMART_APPLIANCES,
            "山善": MarketType.SMART_APPLIANCES,
            
            # === シェア低下市場（バッテリー） ===
            "パナソニックエナジー": MarketType.BATTERY,
            "村田製作所": MarketType.BATTERY,  # 重複
            "GSユアサ": MarketType.BATTERY,
            "東芝インフラシステムズ": MarketType.BATTERY,
            "日立化成": MarketType.BATTERY,
            "FDK": MarketType.BATTERY,
            "NEC": MarketType.BATTERY,
            "ENAX": MarketType.BATTERY,
            "日本電産": MarketType.BATTERY,  # 重複
            "TDK": MarketType.BATTERY,  # 重複
            
            # === シェア低下市場（PC・周辺機器） ===
            "NEC（NECパーソナル）": MarketType.PC_PERIPHERALS,
            "富士通クライアントコンピューティング": MarketType.PC_PERIPHERALS,
            "東芝（ダイナブック）": MarketType.PC_PERIPHERALS,
            "ソニー（VAIO）": MarketType.PC_PERIPHERALS,
            "エレコム": MarketType.PC_PERIPHERALS,
            "バッファロー": MarketType.PC_PERIPHERALS,
            "ロジテック": MarketType.PC_PERIPHERALS,
            "プリンストン": MarketType.PC_PERIPHERALS,
            "サンワサプライ": MarketType.PC_PERIPHERALS,
            "アイ・オー・データ機器": MarketType.PC_PERIPHERALS,
            
            # === シェア失失市場（家電） ===
            "ソニー（家電部門）": MarketType.HOME_APPLIANCES,
            "パナソニック": MarketType.HOME_APPLIANCES,  # 重複
            "シャープ": MarketType.HOME_APPLIANCES,  # 重複
            "東芝ライフスタイル": MarketType.HOME_APPLIANCES,  # 重複
            "三菱電機（家電部門）": MarketType.HOME_APPLIANCES,
            "日立グローバルライフソリューションズ": MarketType.HOME_APPLIANCES,  # 重複
            "三洋電機": MarketType.HOME_APPLIANCES,
            "ビクター": MarketType.HOME_APPLIANCES,
            "アイワ": MarketType.HOME_APPLIANCES,
            "船井電機": MarketType.HOME_APPLIANCES,
            
            # === シェア失失市場（半導体） ===
            "東芝（メモリ部門）": MarketType.SEMICONDUCTOR,
            "日立製作所": MarketType.SEMICONDUCTOR,  # 重複
            "三菱電機": MarketType.SEMICONDUCTOR,  # 重複
            "NEC": MarketType.SEMICONDUCTOR,  # 重複
            "富士通": MarketType.SEMICONDUCTOR,  # 重複
            "松下電器": MarketType.SEMICONDUCTOR,
            "ソニー": MarketType.SEMICONDUCTOR,  # 重複
            "ルネサスエレクトロニクス": MarketType.SEMICONDUCTOR,
            "シャープ": MarketType.SEMICONDUCTOR,  # 重複
            "ローム": MarketType.SEMICONDUCTOR,  # 重複
            
            # === シェア失失市場（スマートフォン） ===
            "ソニー（Xperia）": MarketType.SMARTPHONE,
            "シャープ（AQUOS）": MarketType.SMARTPHONE,
            "京セラ": MarketType.SMARTPHONE,  # 重複
            "パナソニック": MarketType.SMARTPHONE,  # 重複
            "富士通（arrows）": MarketType.SMARTPHONE,
            "NEC": MarketType.SMARTPHONE,  # 重複
            "日立製作所": MarketType.SMARTPHONE,  # 重複
            "三菱電機": MarketType.SMARTPHONE,  # 重複
            "東芝": MarketType.SMARTPHONE,
            "カシオ計算機": MarketType.SMARTPHONE,
            
            # === シェア失失市場（PC） ===
            "ソニー（VAIO）": MarketType.PERSONAL_COMPUTER,  # 重複
            "NEC": MarketType.PERSONAL_COMPUTER,  # 重複
            "富士通": MarketType.PERSONAL_COMPUTER,  # 重複
            "東芝（dynabook）": MarketType.PERSONAL_COMPUTER,
            "シャープ": MarketType.PERSONAL_COMPUTER,  # 重複
            "パナソニック": MarketType.PERSONAL_COMPUTER,  # 重複
            "日立製作所": MarketType.PERSONAL_COMPUTER,  # 重複
            "三菱電機": MarketType.PERSONAL_COMPUTER,  # 重複
            "カシオ計算機": MarketType.PERSONAL_COMPUTER,  # 重複
            "日本電気ホームエレクトロニクス": MarketType.PERSONAL_COMPUTER,
            
            # === シェア失失市場（通信機器） ===
            "NEC": MarketType.TELECOM_EQUIPMENT,  # 重複
            "富士通": MarketType.TELECOM_EQUIPMENT,  # 重複
            "日立製作所": MarketType.TELECOM_EQUIPMENT,  # 重複
            "松下電器": MarketType.TELECOM_EQUIPMENT,  # 重複
            "シャープ": MarketType.TELECOM_EQUIPMENT,  # 重複
            "ソニー": MarketType.TELECOM_EQUIPMENT,  # 重複
            "三菱電機": MarketType.TELECOM_EQUIPMENT,  # 重複
            "京セラ": MarketType.TELECOM_EQUIPMENT,  # 重複
            "カシオ計算機": MarketType.TELECOM_EQUIPMENT,  # 重複
            "日本無線": MarketType.TELECOM_EQUIPMENT
        }
    
    def generate_market_features(self, df: pd.DataFrame, company_col: str = 'company_name') -> pd.DataFrame:
        """
        市場特徴量生成メイン関数
        
        Args:
            df: 企業財務データフレーム
            company_col: 企業名カラム名
            
        Returns:
            市場特徴量が追加されたデータフレーム
        """
        logger.info("市場特徴量生成開始")
        
        # 基本市場特徴量
        df = self._add_basic_market_features(df, company_col)
        
        # 市場ポジション特徴量
        df = self._add_market_position_features(df, company_col)
        
        # 競争環境特徴量  
        df = self._add_competitive_environment_features(df, company_col)
        
        # 市場タイミング特徴量
        df = self._add_market_timing_features(df, company_col)
        
        # 市場集中度特徴量
        df = self._add_market_concentration_features(df, company_col)
        
        # 技術革新特徴量
        df = self._add_technology_innovation_features(df, company_col)
        
        logger.info("市場特徴量生成完了")
        return df
    
    def _add_basic_market_features(self, df: pd.DataFrame, company_col: str) -> pd.DataFrame:
        """基本市場特徴量追加"""
        
        # 市場カテゴリマッピング
        df['market_category'] = df[company_col].map(
            lambda x: self._get_market_category(x).value if self._get_market_category(x) else 'unknown'
        )
        
        df['market_type'] = df[company_col].map(
            lambda x: self.company_market_mapping.get(x, MarketType.ROBOT).value
        )
        
        # 市場特性指標
        for company in df[company_col].unique():
            market_type = self.company_market_mapping.get(company, MarketType.ROBOT)
            characteristics = self.market_characteristics[market_type]
            
            mask = df[company_col] == company
            df.loc[mask, 'tech_intensity'] = characteristics.tech_intensity
            df.loc[mask, 'innovation_cycle_years'] = characteristics.innovation_cycle
            df.loc[mask, 'global_competition_level'] = characteristics.global_competition
            df.loc[mask, 'entry_barriers_level'] = characteristics.entry_barriers
            df.loc[mask, 'market_maturity_level'] = characteristics.market_maturity
            df.loc[mask, 'b2b_ratio'] = characteristics.b2b_ratio
            df.loc[mask, 'customization_level'] = characteristics.customization_level
            df.loc[mask, 'capital_intensity'] = characteristics.capital_intensity
        
        return df
    
    def _add_market_position_features(self, df: pd.DataFrame, company_col: str) -> pd.DataFrame:
        """市場ポジション特徴量追加"""
        
        # 市場シェア変動パターン
        df['market_share_trend'] = df['market_category'].map({
            'high_share': 1.0,      # シェア維持・拡大
            'declining_share': 0.0,  # シェア低下
            'lost_share': -1.0      # シェア失失
        })
        
        # 市場内競合企業数（同一市場タイプ内企業数）
        market_company_counts = df.groupby('market_type')[company_col].nunique()
        df['market_competitors_count'] = df['market_type'].map(market_company_counts)
        
        # 市場内相対ポジション（売上高ランキング）
        if 'sales_revenue' in df.columns:
            df['market_sales_rank'] = df.groupby('market_type')['sales_revenue'].rank(
                method='dense', ascending=False
            )
            df['market_sales_percentile'] = df.groupby('market_type')['sales_revenue'].rank(
                pct=True, ascending=False
            )
        
        # 市場リーダーシップ指標
        df['is_market_leader'] = (df.get('market_sales_rank', float('inf')) <= 3).astype(int)
        df['market_dominance_score'] = df.get('market_sales_percentile', 0.5)
        
        return df
    
    def _add_competitive_environment_features(self, df: pd.DataFrame, company_col: str) -> pd.DataFrame:
        """競争環境特徴量追加"""
        
        # 市場集中度指標（HHI: ハーフィンダール指数）
        if 'sales_revenue' in df.columns:
            market_hhi = self._calculate_market_hhi(df, 'market_type', 'sales_revenue')
            df['market_hhi'] = df['market_type'].map(market_hhi)
            
            # 上位3社集中度（CR3）
            market_cr3 = self._calculate_market_cr3(df, 'market_type', 'sales_revenue')
            df['market_cr3'] = df['market_type'].map(market_cr3)
        
        # 競争激化指標
        df['competition_intensity'] = (
            df['global_competition_level'] * (1 - df['entry_barriers_level'])
        )
        
        # 技術競争 vs 価格競争
        df['tech_competition_ratio'] = df['tech_intensity'] / (df['global_competition_level'] + 0.01)
        df['price_competition_pressure'] = (1 - df['entry_barriers_level']) * df['global_competition_level']
        
        # 市場の競争構造タイプ
        df['competition_structure'] = df.apply(self._classify_competition_structure, axis=1)
        
        return df
    
    def _add_market_timing_features(self, df: pd.DataFrame, company_col: str) -> pd.DataFrame:
        """市場タイミング特徴量追加"""
        
        # 市場参入時期推定（企業設立年ベース）
        if 'established_year' in df.columns:
            # 市場別最早設立年
            market_earliest_entry = df.groupby('market_type')['established_year'].min()
            df['market_earliest_entry_year'] = df['market_type'].map(market_earliest_entry)
            
            # 相対参入時期
            df['market_entry_order'] = df.groupby('market_type')['established_year'].rank(method='dense')
            df['years_after_market_start'] = df['established_year'] - df['market_earliest_entry_year']
            
            # 先発/後発優位性
            df['first_mover_advantage'] = (df['market_entry_order'] <= 3).astype(float)
            df['late_entrant_disadvantage'] = (df['market_entry_order'] > df.groupby('market_type')['market_entry_order'].transform('median')).astype(float)
        
        # イノベーションサイクル適応度
        df['innovation_adaptation_speed'] = 1 / (df['innovation_cycle_years'] + 1)
        
        # 市場成熟段階での参入タイミング適性
        df['maturity_timing_fit'] = df.apply(self._calculate_maturity_timing_fit, axis=1)
        
        return df
    
    def _add_market_concentration_features(self, df: pd.DataFrame, company_col: str) -> pd.DataFrame:
        """市場集中度特徴量追加"""
        
        # 市場カテゴリ別企業分布
        category_distribution = df['market_category'].value_counts(normalize=True)
        df['category_market_share'] = df['market_category'].map(category_distribution)
        
        # 多市場参入度（同一企業が複数市場に参入している場合）
        company_market_counts = df.groupby(company_col)['market_type'].nunique()
        df['multi_market_presence'] = df[company_col].map(company_market_counts)
        df['is_diversified_company'] = (df['multi_market_presence'] > 1).astype(int)
        
        # 市場タイプ別集中度
        market_type_concentration = df['market_type'].value_counts(normalize=True)
        df['market_type_concentration'] = df['market_type'].map(market_type_concentration)
        
        # 業界内地位指標
        df['industry_position_score'] = (
            df.get('market_sales_percentile', 0.5) * 0.4 +
            df['first_mover_advantage'] * 0.3 +
            df['is_market_leader'] * 0.3
        )
        
        return df
    
    def _add_technology_innovation_features(self, df: pd.DataFrame, company_col: str) -> pd.DataFrame:
        """技術革新特徴量追加"""
        
        # 技術革新圧力指標
        df['innovation_pressure'] = (
            df['tech_intensity'] * df['innovation_adaptation_speed']
        )
        
        # R&D投資必要度指標
        df['rd_investment_necessity'] = (
            df['tech_intensity'] * (1 - df['market_maturity_level']) * df['global_competition_level']
        )
        
        # デジタル変革圧力（市場タイプ別）
        digital_transformation_pressure = {
            MarketType.ROBOT: 0.9,
            MarketType.ENDOSCOPE: 0.8,
            MarketType.MACHINE_TOOL: 0.7,
            MarketType.ELECTRONIC_MATERIALS: 0.8,
            MarketType.PRECISION_MEASUREMENT: 0.9,
            MarketType.AUTOMOTIVE: 0.95,  # EV化
            MarketType.STEEL: 0.4,
            MarketType.SMART_APPLIANCES: 0.9,  # IoT化
            MarketType.BATTERY: 0.9,
            MarketType.PC_PERIPHERALS: 0.8,
            MarketType.HOME_APPLIANCES: 0.7,
            MarketType.SEMICONDUCTOR: 0.95,
            MarketType.SMARTPHONE: 0.9,
            MarketType.PERSONAL_COMPUTER: 0.6,
            MarketType.TELECOM_EQUIPMENT: 0.95  # 5G等
        }
        
        df['digital_transformation_pressure'] = df['market_type'].map(
            lambda x: digital_transformation_pressure.get(MarketType(x), 0.5)
        )
        
        # 技術陳腐化リスク
        df['technology_obsolescence_risk'] = (
            (1 - df['tech_intensity']) * df['innovation_adaptation_speed'] * df['global_competition_level']
        )
        
        # イノベーション防衛力
        df['innovation_defense_capability'] = (
            df['tech_intensity'] * df['entry_barriers_level'] * (1 - df['global_competition_level'])
        )
        
        return df
    
    def _get_market_category(self, company_name: str) -> Optional[MarketCategory]:
        """企業名から市場カテゴリを取得"""
        market_type = self.company_market_mapping.get(company_name)
        if market_type:
            return self.market_characteristics[market_type].category
        return None
    
    def _calculate_market_hhi(self, df: pd.DataFrame, market_col: str, sales_col: str) -> Dict[str, float]:
        """市場別ハーフィンダール指数計算"""
        hhi_dict = {}
        
        for market in df[market_col].unique():
            market_df = df[df[market_col] == market]
            if sales_col in market_df.columns:
                total_sales = market_df[sales_col].sum()
                if total_sales > 0:
                    market_shares = market_df[sales_col] / total_sales
                    hhi = (market_shares ** 2).sum()
                    hhi_dict[market] = hhi
                else:
                    hhi_dict[market] = 0.0
            else:
                hhi_dict[market] = 0.0
        
        return hhi_dict
    
    def _calculate_market_cr3(self, df: pd.DataFrame, market_col: str, sales_col: str) -> Dict[str, float]:
        """市場別上位3社集中度計算"""
        cr3_dict = {}
        
        for market in df[market_col].unique():
            market_df = df[df[market_col] == market]
            if sales_col in market_df.columns:
                total_sales = market_df[sales_col].sum()
                if total_sales > 0:
                    top3_sales = market_df[sales_col].nlargest(3).sum()
                    cr3 = top3_sales / total_sales
                    cr3_dict[market] = cr3
                else:
                    cr3_dict[market] = 0.0
            else:
                cr3_dict[market] = 0.0
        
        return cr3_dict
    
    def _classify_competition_structure(self, row) -> str:
        """競争構造分類"""
        hhi = row.get('market_hhi', 0.5)
        competition_level = row.get('global_competition_level', 0.5)
        entry_barriers = row.get('entry_barriers_level', 0.5)
        
        if hhi > 0.25:  # 高集中
            if entry_barriers > 0.7:
                return "oligopoly_high_barriers"    # 寡占・高参入障壁
            else:
                return "oligopoly_low_barriers"     # 寡占・低参入障壁
        elif hhi > 0.15:  # 中集中
            if competition_level > 0.7:
                return "moderate_concentration_high_competition"  # 中集中・激競争
            else:
                return "moderate_concentration_low_competition"   # 中集中・穏競争
        else:  # 低集中
            if competition_level > 0.8:
                return "perfect_competition"       # 完全競争的
            else:
                return "fragmented_market"         # 断片化市場
    
    def _calculate_maturity_timing_fit(self, row) -> float:
        """市場成熟段階での参入タイミング適性計算"""
        maturity = row.get('market_maturity_level', 0.5)
        entry_order = row.get('market_entry_order', 5)
        total_entrants = row.get('market_competitors_count', 10)
        
        if maturity < 0.3:  # 成長初期市場
            # 早期参入が有利
            return max(0, 1 - (entry_order - 1) / max(total_entrants, 1))
        elif maturity < 0.7:  # 成長期市場
            # 中期参入でも可能
            optimal_order = total_entrants * 0.3
            deviation = abs(entry_order - optimal_order) / total_entrants
            return max(0, 1 - deviation)
        else:  # 成熟・衰退期市場
            # 早期参入企業が圧倒的有利
            return max(0, 1 - (entry_order - 1) / 3)
    
    def generate_market_dynamics_features(self, df: pd.DataFrame, year_col: str = 'year') -> pd.DataFrame:
        """
        市場ダイナミクス特徴量生成
        
        Args:
            df: 時系列財務データフレーム
            year_col: 年度カラム名
            
        Returns:
            市場ダイナミクス特徴量が追加されたデータフレーム
        """
        logger.info("市場ダイナミクス特徴量生成開始")
        
        # 市場ライフサイクル段階推定
        df = self._add_market_lifecycle_features(df, year_col)
        
        # 競合変化分析
        df = self._add_competitive_dynamics_features(df, year_col)
        
        # 技術パラダイムシフト検出
        df = self._add_paradigm_shift_features(df, year_col)
        
        # 市場統合・分化分析
        df = self._add_market_evolution_features(df, year_col)
        
        logger.info("市場ダイナミクス特徴量生成完了")
        return df
    
    def _add_market_lifecycle_features(self, df: pd.DataFrame, year_col: str) -> pd.DataFrame:
        """市場ライフサイクル特徴量追加"""
        
        # 市場年齢計算
        market_start_years = df.groupby('market_type')[year_col].min()
        df['market_age'] = df[year_col] - df['market_type'].map(market_start_years)
        
        # ライフサイクル段階推定
        df['lifecycle_stage'] = df.apply(self._estimate_lifecycle_stage, axis=1)
        
        # 成長率ベースライフサイクル
        if 'sales_growth_rate' in df.columns:
            market_avg_growth = df.groupby(['market_type', year_col])['sales_growth_rate'].mean()
            df['market_average_growth'] = df.apply(
                lambda x: market_avg_growth.get((x['market_type'], x[year_col]), 0), axis=1
            )
            
            # ライフサイクル段階（成長率ベース）
            df['growth_based_lifecycle'] = df['market_average_growth'].apply(
                lambda x: 'introduction' if x > 0.2 else 
                            'growth' if x > 0.1 else
                            'maturity' if x > 0.02 else 'decline'
            )
        
        return df
    
    def _add_competitive_dynamics_features(self, df: pd.DataFrame, year_col: str) -> pd.DataFrame:
        """競合ダイナミクス特徴量追加"""
        
        # 市場参入・退出分析
        df = df.sort_values([year_col, 'market_type'])
        
        # 年次市場参加企業数
        yearly_participants = df.groupby(['market_type', year_col]).size()
        df['yearly_market_participants'] = df.apply(
            lambda x: yearly_participants.get((x['market_type'], x[year_col]), 0), axis=1
        )
        
        # 市場参入・退出率
        df['market_entry_rate'] = df.groupby(['market_type'])['yearly_market_participants'].pct_change()
        
        # 競合強度変化
        if 'market_hhi' in df.columns:
            df['hhi_change'] = df.groupby(['market_type'])['market_hhi'].pct_change()
            df['market_concentration_trend'] = df['hhi_change'].apply(
                lambda x: 'concentrating' if x > 0.05 else 
                            'fragmenting' if x < -0.05 else 'stable'
            )
        
        return df
    
    def _add_paradigm_shift_features(self, df: pd.DataFrame, year_col: str) -> pd.DataFrame:
        """技術パラダイムシフト特徴量追加"""
        
        # 技術変革期間推定（市場タイプ別）
        paradigm_shift_periods = {
            MarketType.AUTOMOTIVE: [2010, 2020],      # EV化
            MarketType.SEMICONDUCTOR: [1995, 2005],   # デジタル化
            MarketType.SMARTPHONE: [2007, 2015],      # スマートフォン革命
            MarketType.TELECOM_EQUIPMENT: [2018, 2025], # 5G
            MarketType.BATTERY: [2015, 2025],         # EV用リチウムイオン
        }
        
        df['in_paradigm_shift_period'] = df.apply(
            lambda x: self._check_paradigm_shift_period(
                MarketType(x['market_type']), x[year_col], paradigm_shift_periods
            ), axis=1
        )
        
        # パラダイムシフト適応度
        df['paradigm_shift_adaptation'] = df.apply(
            self._calculate_paradigm_adaptation_score, axis=1
        )
        
        return df
    
    def _add_market_evolution_features(self, df: pd.DataFrame, year_col: str) -> pd.DataFrame:
        """市場進化特徴量追加"""
        
        # 市場境界変化検出
        df['market_boundary_fluidity'] = df['multi_market_presence'] / df['market_competitors_count']
        
        # 垂直統合・水平統合度
        df['integration_level'] = df.apply(self._estimate_integration_level, axis=1)
        
        # エコシステム化度
        df['ecosystem_participation'] = (
            df['b2b_ratio'] * df['customization_level'] * df['tech_intensity']
        )
        
        return df
    
    def _estimate_lifecycle_stage(self, row) -> str:
        """ライフサイクル段階推定"""
        maturity = row.get('market_maturity_level', 0.5)
        age = row.get('market_age', 10)
        competition = row.get('global_competition_level', 0.5)
        
        if maturity < 0.3 and age < 10:
            return 'introduction'
        elif maturity < 0.6 and competition < 0.7:
            return 'growth'
        elif maturity < 0.8:
            return 'maturity'
        else:
            return 'decline'
    
    def _check_paradigm_shift_period(self, market_type: MarketType, year: int, 
                                    shift_periods: Dict[MarketType, List[int]]) -> int:
        """パラダイムシフト期間判定"""
        if market_type in shift_periods:
            start_year, end_year = shift_periods[market_type]
            return int(start_year <= year <= end_year)
        return 0
    
    def _calculate_paradigm_adaptation_score(self, row) -> float:
        """パラダイムシフト適応スコア計算"""
        tech_intensity = row.get('tech_intensity', 0.5)
        rd_necessity = row.get('rd_investment_necessity', 0.5)
        innovation_defense = row.get('innovation_defense_capability', 0.5)
        in_shift = row.get('in_paradigm_shift_period', 0)
        
        base_adaptation = (tech_intensity + rd_necessity + innovation_defense) / 3
        
        if in_shift:
            # シフト期間中は適応能力がより重要
            return base_adaptation * 1.5
        else:
            return base_adaptation
    
    def _estimate_integration_level(self, row) -> float:
        """統合レベル推定"""
        capital_intensity = row.get('capital_intensity', 0.5)
        customization = row.get('customization_level', 0.5)
        barriers = row.get('entry_barriers_level', 0.5)
        
        # 高資本集約・高カスタマイゼーション・高参入障壁 = 高統合レベル
        return (capital_intensity * 0.4 + customization * 0.3 + barriers * 0.3)
    
    def get_market_summary_statistics(self, df: pd.DataFrame) -> Dict:
        """市場特徴量サマリー統計取得"""
        
        summary = {
            'market_categories': {
                'distribution': df['market_category'].value_counts().to_dict(),
                'avg_tech_intensity': df.groupby('market_category')['tech_intensity'].mean().to_dict(),
                'avg_competition_level': df.groupby('market_category')['global_competition_level'].mean().to_dict(),
                'avg_entry_barriers': df.groupby('market_category')['entry_barriers_level'].mean().to_dict()
            },
            'market_types': {
                'distribution': df['market_type'].value_counts().to_dict(),
                'competitors_count': df.groupby('market_type')['market_competitors_count'].first().to_dict()
            },
            'competition_structure': {
                'distribution': df['competition_structure'].value_counts().to_dict()
            }
        }
        
        if 'market_hhi' in df.columns:
            summary['concentration_metrics'] = {
                'avg_hhi_by_category': df.groupby('market_category')['market_hhi'].mean().to_dict(),
                'avg_cr3_by_category': df.groupby('market_category')['market_cr3'].mean().to_dict()
            }
        
        return summary
    
    def export_market_features_definition(self, output_path: str) -> None:
        """市場特徴量定義をエクスポート"""
        
        features_definition = {
            'basic_market_features': {
                'market_category': '市場カテゴリ（high_share/declining_share/lost_share）',
                'market_type': '具体的市場タイプ（robot, automotive等）',
                'tech_intensity': '技術集約度（0-1）',
                'innovation_cycle_years': 'イノベーションサイクル年数',
                'global_competition_level': 'グローバル競争度（0-1）',
                'entry_barriers_level': '参入障壁レベル（0-1）',
                'market_maturity_level': '市場成熟度（0-1）',
                'b2b_ratio': 'B2B取引比率（0-1）',
                'customization_level': 'カスタマイゼーション度（0-1）',
                'capital_intensity': '資本集約度（0-1）'
            },
            'position_features': {
                'market_share_trend': '市場シェアトレンド（1:維持、0:低下、-1:失失）',
                'market_competitors_count': '市場内競合企業数',
                'market_sales_rank': '市場内売上高ランキング',
                'market_sales_percentile': '市場内売上高パーセンタイル',
                'is_market_leader': '市場リーダーフラグ（上位3社）',
                'market_dominance_score': '市場支配度スコア'
            },
            'competitive_features': {
                'market_hhi': '市場ハーフィンダール指数',
                'market_cr3': '上位3社集中度',
                'competition_intensity': '競争激化指標',
                'tech_competition_ratio': '技術競争対価格競争比率',
                'price_competition_pressure': '価格競争圧力',
                'competition_structure': '競争構造タイプ'
            },
            'timing_features': {
                'market_entry_order': '市場参入順序',
                'years_after_market_start': '市場開始からの経過年数',
                'first_mover_advantage': '先発優位性',
                'late_entrant_disadvantage': '後発不利性',
                'innovation_adaptation_speed': 'イノベーション適応速度',
                'maturity_timing_fit': '成熟段階参入タイミング適性'
            },
            'technology_features': {
                'innovation_pressure': '技術革新圧力',
                'rd_investment_necessity': 'R&D投資必要度',
                'digital_transformation_pressure': 'デジタル変革圧力',
                'technology_obsolescence_risk': '技術陳腐化リスク',
                'innovation_defense_capability': 'イノベーション防衛力',
                'paradigm_shift_adaptation': 'パラダイムシフト適応度'
            }
        }
        
        import json
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(features_definition, f, ensure_ascii=False, indent=2)
        
        logger.info(f"市場特徴量定義を {output_path} にエクスポートしました")