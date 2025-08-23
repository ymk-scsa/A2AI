"""
A2AI - Advanced Financial Analysis AI
Survival Data Generator Module

This module generates survival analysis datasets from the complete lifecycle 
of 150 companies across three market categories (high share, declining, lost markets).
Handles corporate extinctions, mergers, spin-offs, and new establishments.

Author: A2AI Development Team
Created: 2024
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
import logging
from pathlib import Path
import yaml
from dataclasses import dataclass
from enum import Enum

from ..utils.data_utils import DataUtils
from ..utils.statistical_utils import StatisticalUtils
from ..utils.logging_utils import setup_logger

class SurvivalEventType(Enum):
    """Enumeration of survival event types"""
    ACTIVE = "active"
    BANKRUPTCY = "bankruptcy"
    MERGER = "merger"
    ACQUISITION = "acquisition"
    SPIN_OFF = "spin_off"
    DELISTING = "delisting"
    BUSINESS_TRANSFER = "business_transfer"
    VOLUNTARY_LIQUIDATION = "voluntary_liquidation"

class MarketCategory(Enum):
    """Market category classification"""
    HIGH_SHARE = "high_share"
    DECLINING_SHARE = "declining_share"
    LOST_SHARE = "lost_share"

@dataclass
class SurvivalRecord:
    """Data structure for individual survival records"""
    company_id: str
    company_name: str
    market_category: MarketCategory
    industry_sector: str
    establishment_date: datetime
    observation_start: datetime
    observation_end: datetime
    event_occurred: bool
    event_type: SurvivalEventType
    event_date: Optional[datetime]
    survival_time: int  # in days
    censored: bool
    market_share_at_entry: Optional[float]
    market_share_at_exit: Optional[float]

class SurvivalDataGenerator:
    """
    Generates comprehensive survival analysis datasets for A2AI system.
    
    Handles:
    - Corporate life cycle data extraction
    - Event time calculation and censoring
    - Risk set construction
    - Time-varying covariates preparation
    - Market category stratification
    """
    
    def __init__(self, config_path: str = "config/survival_parameters.yaml"):
        """
        Initialize Survival Data Generator
        
        Args:
            config_path: Path to survival analysis configuration file
        """
        self.logger = setup_logger(__name__)
        self.config = self._load_config(config_path)
        self.data_utils = DataUtils()
        self.stats_utils = StatisticalUtils()
        
        # Analysis parameters
        self.study_start_date = datetime(1984, 1, 1)
        self.study_end_date = datetime(2024, 12, 31)
        self.observation_window = 40 * 365  # 40 years in days
        
        # Corporate event mappings
        self.extinction_events = self._load_extinction_events()
        self.emergence_events = self._load_emergence_events()
        self.corporate_actions = self._load_corporate_actions()
        
        self.logger.info("Survival Data Generator initialized successfully")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration parameters"""
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            self.logger.warning(f"Config file {config_path} not found, using defaults")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Default configuration parameters"""
        return {
            'minimum_observation_days': 365,
            'censoring_date': '2024-12-31',
            'include_financial_events': True,
            'include_merger_events': True,
            'include_spinoff_events': True,
            'risk_window_years': 5,
            'time_varying_covariates': True
        }
    
    def _load_extinction_events(self) -> Dict[str, Dict]:
        """Load corporate extinction event data"""
        extinction_events = {
            # Completely extinct companies from lost markets
            "三洋電機": {
                "event_date": datetime(2012, 4, 1),
                "event_type": SurvivalEventType.MERGER,
                "acquirer": "パナソニック",
                "market_category": MarketCategory.LOST_SHARE
            },
            "アイワ": {
                "event_date": datetime(2002, 3, 31),
                "event_type": SurvivalEventType.BUSINESS_TRANSFER,
                "acquirer": "ソニー",
                "market_category": MarketCategory.LOST_SHARE
            },
            "日本電気ホームエレクトロニクス": {
                "event_date": datetime(2011, 7, 1),
                "event_type": SurvivalEventType.MERGER,
                "acquirer": "NEC",
                "market_category": MarketCategory.LOST_SHARE
            },
            "船井電機": {
                "event_date": datetime(2017, 3, 31),
                "event_type": SurvivalEventType.BUSINESS_TRANSFER,
                "market_category": MarketCategory.LOST_SHARE
            },
            "FCNT": {
                "event_date": datetime(2023, 5, 31),
                "event_type": SurvivalEventType.BANKRUPTCY,
                "market_category": MarketCategory.LOST_SHARE
            },
            # Additional extinction events
            "京セラ": {
                "event_date": datetime(2023, 12, 31),
                "event_type": SurvivalEventType.BUSINESS_TRANSFER,
                "notes": "スマートフォン事業撤退",
                "market_category": MarketCategory.LOST_SHARE
            }
        }
        return extinction_events
    
    def _load_emergence_events(self) -> Dict[str, Dict]:
        """Load corporate emergence/spin-off event data"""
        emergence_events = {
            # New establishments and spin-offs
            "デンソーウェーブ": {
                "establishment_date": datetime(2001, 4, 1),
                "parent_company": "デンソー",
                "spin_off_type": "subsidiary_establishment",
                "market_category": MarketCategory.HIGH_SHARE
            },
            "キオクシア": {
                "establishment_date": datetime(2018, 6, 1),
                "parent_company": "東芝",
                "spin_off_type": "business_carve_out",
                "market_category": MarketCategory.DECLINING_SHARE
            },
            "プロテリアル": {
                "establishment_date": datetime(2023, 4, 1),
                "parent_company": "日立製作所",
                "spin_off_type": "business_spin_off",
                "market_category": MarketCategory.HIGH_SHARE
            },
            "パナソニックエナジー": {
                "establishment_date": datetime(2022, 4, 1),
                "parent_company": "パナソニック",
                "spin_off_type": "business_division",
                "market_category": MarketCategory.DECLINING_SHARE
            },
            "東芝ライフスタイル": {
                "establishment_date": datetime(2016, 7, 1),
                "acquisition_date": datetime(2017, 6, 1),
                "acquirer": "美的集団",
                "market_category": MarketCategory.LOST_SHARE
            }
        }
        return emergence_events
    
    def _load_corporate_actions(self) -> Dict[str, List[Dict]]:
        """Load major corporate actions affecting survival analysis"""
        return {
            "major_mergers": [
                {
                    "company": "日立金属",
                    "action_date": datetime(2023, 4, 1),
                    "action_type": "spin_off_to_proterial",
                    "impact": "business_continuation"
                },
                {
                    "company": "東芝メモリ",
                    "action_date": datetime(2018, 6, 1),
                    "action_type": "spin_off_to_kioxia",
                    "impact": "business_independence"
                }
            ],
            "business_transfers": [
                {
                    "company": "東芝dynabook",
                    "action_date": datetime(2019, 1, 1),
                    "acquirer": "シャープ",
                    "impact": "foreign_ownership"
                },
                {
                    "company": "富士通クライアントコンピューティング",
                    "action_date": datetime(2018, 5, 1),
                    "acquirer": "Lenovo",
                    "impact": "joint_venture"
                }
            ]
        }
    
    def generate_survival_dataset(self, 
                                companies_data: pd.DataFrame,
                                financial_data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate comprehensive survival analysis dataset
        
        Args:
            companies_data: Company master data with lifecycle information
            financial_data: Financial statement data for all companies
            
        Returns:
            Survival dataset with time-to-event information
        """
        self.logger.info("Starting survival dataset generation")
        
        survival_records = []
        
        for _, company in companies_data.iterrows():
            try:
                survival_record = self._create_survival_record(company, financial_data)
                if survival_record:
                    survival_records.append(survival_record)
                    
            except Exception as e:
                self.logger.error(f"Error processing company {company.get('company_name', 'Unknown')}: {str(e)}")
                continue
        
        # Convert to DataFrame
        survival_df = self._records_to_dataframe(survival_records)
        
        # Add time-varying covariates if enabled
        if self.config.get('time_varying_covariates', True):
            survival_df = self._add_time_varying_covariates(survival_df, financial_data)
        
        # Validate dataset
        survival_df = self._validate_survival_data(survival_df)
        
        self.logger.info(f"Generated survival dataset with {len(survival_df)} records")
        return survival_df
    
    def _create_survival_record(self, 
                                company: pd.Series, 
                                financial_data: pd.DataFrame) -> Optional[SurvivalRecord]:
        """Create individual survival record for a company"""
        
        company_name = company['company_name']
        company_id = company.get('company_id', self._generate_company_id(company_name))
        
        # Determine market category
        market_category = self._determine_market_category(company)
        
        # Get establishment date
        establishment_date = self._get_establishment_date(company_name, company)
        
        # Check for extinction events
        extinction_info = self.extinction_events.get(company_name)
        emergence_info = self.emergence_events.get(company_name)
        
        # Calculate observation period
        if emergence_info:
            obs_start = emergence_info['establishment_date']
        else:
            obs_start = max(establishment_date, self.study_start_date)
        
        # Determine event occurrence and end date
        if extinction_info:
            event_occurred = True
            event_date = extinction_info['event_date']
            event_type = extinction_info['event_type']
            obs_end = event_date
            censored = False
        else:
            event_occurred = False
            event_date = None
            event_type = SurvivalEventType.ACTIVE
            obs_end = self.study_end_date
            censored = True
        
        # Calculate survival time
        survival_time = (obs_end - obs_start).days
        
        # Skip if survival time is too short
        if survival_time < self.config.get('minimum_observation_days', 365):
            return None
        
        # Get market share information
        market_share_entry = self._get_market_share_at_date(company_name, obs_start)
        market_share_exit = self._get_market_share_at_date(company_name, obs_end)
        
        return SurvivalRecord(
            company_id=company_id,
            company_name=company_name,
            market_category=market_category,
            industry_sector=company.get('industry_sector', 'Unknown'),
            establishment_date=establishment_date,
            observation_start=obs_start,
            observation_end=obs_end,
            event_occurred=event_occurred,
            event_type=event_type,
            event_date=event_date,
            survival_time=survival_time,
            censored=censored,
            market_share_at_entry=market_share_entry,
            market_share_at_exit=market_share_exit
        )
    
    def _determine_market_category(self, company: pd.Series) -> MarketCategory:
        """Determine market category based on company classification"""
        market_info = company.get('market_category', '')
        
        if 'high_share' in market_info.lower():
            return MarketCategory.HIGH_SHARE
        elif 'declining' in market_info.lower():
            return MarketCategory.DECLINING_SHARE
        elif 'lost' in market_info.lower():
            return MarketCategory.LOST_SHARE
        else:
            # Default classification based on sector analysis
            sector = company.get('industry_sector', '')
            return self._classify_by_sector(sector)
    
    def _classify_by_sector(self, sector: str) -> MarketCategory:
        """Classify market category by industry sector"""
        high_share_sectors = ['ロボット', '内視鏡', '工作機械', '電子材料', '精密測定機器']
        declining_sectors = ['自動車', '鉄鋼', 'スマート家電', 'バッテリー', 'PC・周辺機器']
        
        for hs_sector in high_share_sectors:
            if hs_sector in sector:
                return MarketCategory.HIGH_SHARE
        
        for dec_sector in declining_sectors:
            if dec_sector in sector:
                return MarketCategory.DECLINING_SHARE
        
        return MarketCategory.LOST_SHARE
    
    def _get_establishment_date(self, company_name: str, company: pd.Series) -> datetime:
        """Get company establishment date"""
        
        # Check emergence events first
        if company_name in self.emergence_events:
            return self.emergence_events[company_name]['establishment_date']
        
        # Use provided establishment date
        if 'establishment_date' in company and pd.notna(company['establishment_date']):
            if isinstance(company['establishment_date'], str):
                try:
                    return datetime.strptime(company['establishment_date'], '%Y-%m-%d')
                except ValueError:
                    pass
            elif isinstance(company['establishment_date'], datetime):
                return company['establishment_date']
        
        # Default to study start for old companies
        return self.study_start_date
    
    def _get_market_share_at_date(self, company_name: str, target_date: datetime) -> Optional[float]:
        """Get market share at specific date (placeholder for future implementation)"""
        # This would integrate with market share data collection
        # For now, return None as placeholder
        return None
    
    def _records_to_dataframe(self, records: List[SurvivalRecord]) -> pd.DataFrame:
        """Convert survival records to pandas DataFrame"""
        
        data = []
        for record in records:
            data.append({
                'company_id': record.company_id,
                'company_name': record.company_name,
                'market_category': record.market_category.value,
                'industry_sector': record.industry_sector,
                'establishment_date': record.establishment_date,
                'observation_start': record.observation_start,
                'observation_end': record.observation_end,
                'event_occurred': record.event_occurred,
                'event_type': record.event_type.value,
                'event_date': record.event_date,
                'survival_time': record.survival_time,
                'censored': record.censored,
                'market_share_at_entry': record.market_share_at_entry,
                'market_share_at_exit': record.market_share_at_exit
            })
        
        return pd.DataFrame(data)
    
    def _add_time_varying_covariates(self, 
                                    survival_df: pd.DataFrame, 
                                    financial_data: pd.DataFrame) -> pd.DataFrame:
        """Add time-varying covariates for survival analysis"""
        
        self.logger.info("Adding time-varying covariates to survival dataset")
        
        # Create time-varying dataset in counting process format
        time_varying_records = []
        
        for _, survival_record in survival_df.iterrows():
            company_id = survival_record['company_id']
            company_name = survival_record['company_name']
            
            # Get company financial data
            company_financial = financial_data[
                financial_data['company_name'] == company_name
            ].sort_values('fiscal_year')
            
            if company_financial.empty:
                continue
            
            # Create time intervals for survival analysis
            start_year = survival_record['observation_start'].year
            end_year = survival_record['observation_end'].year
            
            for year in range(start_year, end_year + 1):
                year_financial = company_financial[
                    company_financial['fiscal_year'] == year
                ]
                
                if year_financial.empty:
                    continue
                
                # Calculate time intervals
                interval_start = max(datetime(year, 1, 1), survival_record['observation_start'])
                
                if year == end_year:
                    interval_end = survival_record['observation_end']
                    event_in_interval = survival_record['event_occurred']
                else:
                    interval_end = datetime(year, 12, 31)
                    event_in_interval = False
                
                # Create time-varying record
                time_varying_record = {
                    'company_id': company_id,
                    'company_name': company_name,
                    'market_category': survival_record['market_category'],
                    'industry_sector': survival_record['industry_sector'],
                    'interval_start': interval_start,
                    'interval_end': interval_end,
                    'interval_length': (interval_end - interval_start).days,
                    'event_occurred': event_in_interval,
                    'event_type': survival_record['event_type'],
                    'time_from_start': (interval_start - survival_record['observation_start']).days,
                    'time_to_end': (survival_record['observation_end'] - interval_end).days
                }
                
                # Add financial covariates
                if not year_financial.empty:
                    financial_record = year_financial.iloc[0]
                    
                    # Add key financial ratios as time-varying covariates
                    financial_covariates = {
                        'revenue': financial_record.get('revenue', np.nan),
                        'operating_profit_margin': financial_record.get('operating_profit_margin', np.nan),
                        'roa': financial_record.get('roa', np.nan),
                        'roe': financial_record.get('roe', np.nan),
                        'debt_ratio': financial_record.get('debt_ratio', np.nan),
                        'current_ratio': financial_record.get('current_ratio', np.nan),
                        'rd_intensity': financial_record.get('rd_intensity', np.nan),
                        'employee_count': financial_record.get('employee_count', np.nan),
                        'capex_ratio': financial_record.get('capex_ratio', np.nan),
                        'cash_ratio': financial_record.get('cash_ratio', np.nan)
                    }
                    
                    time_varying_record.update(financial_covariates)
                
                time_varying_records.append(time_varying_record)
        
        return pd.DataFrame(time_varying_records)
    
    def _validate_survival_data(self, survival_df: pd.DataFrame) -> pd.DataFrame:
        """Validate survival dataset for analysis readiness"""
        
        self.logger.info("Validating survival dataset")
        
        initial_count = len(survival_df)
        
        # Remove records with invalid survival times
        survival_df = survival_df[survival_df['survival_time'] > 0]
        
        # Remove records with missing critical information
        critical_columns = ['company_id', 'company_name', 'survival_time', 'event_occurred']
        survival_df = survival_df.dropna(subset=critical_columns)
        
        # Validate event logic
        survival_df = survival_df[
            ~((survival_df['event_occurred'] == True) & (survival_df['event_date'].isna()))
        ]
        
        final_count = len(survival_df)
        removed_count = initial_count - final_count
        
        if removed_count > 0:
            self.logger.warning(f"Removed {removed_count} invalid records during validation")
        
        # Add validation flags
        survival_df['validated'] = True
        survival_df['validation_date'] = datetime.now()
        
        return survival_df
    
    def _generate_company_id(self, company_name: str) -> str:
        """Generate unique company ID"""
        import hashlib
        return hashlib.md5(company_name.encode('utf-8')).hexdigest()[:8]
    
    def generate_market_stratified_dataset(self, 
                                            survival_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Generate market category stratified datasets"""
        
        stratified_datasets = {}
        
        for market_category in MarketCategory:
            category_data = survival_df[
                survival_df['market_category'] == market_category.value
            ].copy()
            
            if not category_data.empty:
                stratified_datasets[market_category.value] = category_data
                self.logger.info(f"Generated {market_category.value} dataset with {len(category_data)} records")
        
        return stratified_datasets
    
    def generate_risk_sets(self, survival_df: pd.DataFrame) -> Dict[int, pd.DataFrame]:
        """Generate risk sets for survival analysis"""
        
        risk_sets = {}
        
        # Get unique event times
        event_times = survival_df[survival_df['event_occurred'] == True]['survival_time'].unique()
        event_times = sorted(event_times)
        
        for event_time in event_times:
            # Companies still at risk at this time
            at_risk = survival_df[survival_df['survival_time'] >= event_time]
            
            # Companies that experienced event at this time
            events = survival_df[
                (survival_df['survival_time'] == event_time) & 
                (survival_df['event_occurred'] == True)
            ]
            
            risk_sets[event_time] = {
                'at_risk': at_risk,
                'events': events,
                'risk_set_size': len(at_risk),
                'event_count': len(events)
            }
        
        return risk_sets
    
    def export_survival_dataset(self, 
                                survival_df: pd.DataFrame, 
                                output_path: str,
                                format: str = 'csv') -> None:
        """Export survival dataset to file"""
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == 'csv':
            survival_df.to_csv(output_path, index=False, encoding='utf-8')
        elif format.lower() == 'parquet':
            survival_df.to_parquet(output_path, index=False)
        elif format.lower() == 'excel':
            survival_df.to_excel(output_path, index=False)
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        self.logger.info(f"Survival dataset exported to {output_path}")
    
    def get_survival_summary_statistics(self, survival_df: pd.DataFrame) -> Dict:
        """Generate summary statistics for survival dataset"""
        
        summary = {
            'total_companies': len(survival_df),
            'total_events': len(survival_df[survival_df['event_occurred'] == True]),
            'censored_companies': len(survival_df[survival_df['censored'] == True]),
            'event_rate': len(survival_df[survival_df['event_occurred'] == True]) / len(survival_df),
            
            'survival_time_stats': {
                'mean': survival_df['survival_time'].mean(),
                'median': survival_df['survival_time'].median(),
                'std': survival_df['survival_time'].std(),
                'min': survival_df['survival_time'].min(),
                'max': survival_df['survival_time'].max()
            },
            
            'market_category_distribution': survival_df['market_category'].value_counts().to_dict(),
            'event_type_distribution': survival_df['event_type'].value_counts().to_dict(),
            
            'by_market_category': {}
        }
        
        # Market category specific statistics
        for category in survival_df['market_category'].unique():
            category_data = survival_df[survival_df['market_category'] == category]
            
            summary['by_market_category'][category] = {
                'count': len(category_data),
                'events': len(category_data[category_data['event_occurred'] == True]),
                'event_rate': len(category_data[category_data['event_occurred'] == True]) / len(category_data),
                'mean_survival_time': category_data['survival_time'].mean(),
                'median_survival_time': category_data['survival_time'].median()
            }
        
        return summary


# Example usage and testing
if __name__ == "__main__":
    
    # Initialize survival data generator
    generator = SurvivalDataGenerator()
    
    # Example company data structure
    companies_sample = pd.DataFrame({
        'company_name': ['ファナック', '三洋電機', 'デンソーウェーブ', 'キオクシア'],
        'market_category': ['high_share', 'lost_share', 'high_share', 'declining_share'],
        'industry_sector': ['ロボット', '家電', 'ロボット', '半導体'],
        'establishment_date': ['1972-05-01', '1950-04-01', '2001-04-01', '2018-06-01']
    })
    
    # Example financial data structure (placeholder)
    financial_sample = pd.DataFrame({
        'company_name': ['ファナック'] * 5,
        'fiscal_year': [2020, 2021, 2022, 2023, 2024],
        'revenue': [1000, 1100, 1200, 1150, 1300],
        'operating_profit_margin': [0.15, 0.16, 0.14, 0.13, 0.17],
        'roa': [0.08, 0.09, 0.07, 0.06, 0.10],
        'roe': [0.12, 0.13, 0.11, 0.10, 0.15]
    })
    
    try:
        # Generate survival dataset
        survival_data = generator.generate_survival_dataset(companies_sample, financial_sample)
        print("Survival dataset generated successfully!")
        print(f"Dataset shape: {survival_data.shape}")
        
        # Generate summary statistics
        summary = generator.get_survival_summary_statistics(survival_data)
        print("\nSummary Statistics:")
        print(f"Total companies: {summary['total_companies']}")
        print(f"Total events: {summary['total_events']}")
        print(f"Event rate: {summary['event_rate']:.2%}")
        
    except Exception as e:
        print(f"Error generating survival dataset: {str(e)}")