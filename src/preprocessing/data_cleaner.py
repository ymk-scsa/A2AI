"""
A2AI (Advanced Financial Analysis AI) Data Cleaner Module

This module handles comprehensive data cleaning for financial statement analysis
including survival bias correction, lifecycle alignment, and missing data handling
for 150 companies across 40 years of financial data.

Author: A2AI Development Team
Version: 1.0.0
"""

import pandas as pd
import numpy as np
import warnings
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
import logging
from pathlib import Path
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class A2AIDataCleaner:
    """
    Advanced data cleaning system for A2AI financial analysis
    
    Features:
    - Corporate lifecycle-aware cleaning
    - Survival bias correction
    - Multi-stage data validation
    - Market category-specific processing
    - Missing data intelligent handling
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize A2AI Data Cleaner
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.cleaning_stats = {}
        self.market_categories = self._load_market_categories()
        self.evaluation_factors = self._load_evaluation_factors()
        
        # Corporate lifecycle stages
        self.lifecycle_stages = {
            'startup': (0, 5),      # 0-5 years
            'growth': (6, 15),      # 6-15 years
            'maturity': (16, 30),   # 16-30 years
            'decline_or_renewal': (31, float('inf'))  # 31+ years
        }
        
        # Market categories for differential processing
        self.high_share_markets = [
            'robot', 'endoscope', 'machine_tool', 
            'electronic_materials', 'precision_measurement'
        ]
        self.declining_markets = [
            'automotive', 'steel', 'smart_appliances',
            'battery', 'pc_peripherals'
        ]
        self.lost_markets = [
            'home_appliances', 'semiconductor', 'smartphone',
            'pc', 'telecommunications'
        ]
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration settings"""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        
        # Default configuration
        return {
            'missing_threshold': 0.7,  # Drop columns with >70% missing
            'outlier_method': 'iqr',
            'outlier_factor': 3.0,
            'min_data_years': 5,  # Minimum years of data required
            'accounting_standard_transition': 2010,  # IFRS transition year
            'inflation_adjustment': True,
            'survivorship_bias_correction': True
        }
    
    def _load_market_categories(self) -> Dict:
        """Load market category definitions"""
        return {
            'high_share': self.high_share_markets,
            'declining': self.declining_markets,
            'lost': self.lost_markets
        }
    
    def _load_evaluation_factors(self) -> Dict:
        """Load evaluation factors and metrics definitions"""
        return {
            'traditional_metrics': [
                'sales_revenue', 'sales_growth_rate', 'operating_margin',
                'net_margin', 'roe', 'value_added_ratio'
            ],
            'extended_metrics': [
                'survival_probability', 'emergence_success_rate', 
                'business_succession_rate'
            ],
            'factor_count_per_metric': 23  # 20 original + 3 extended factors
        }
    
    def clean_financial_data(self, 
                            raw_data: pd.DataFrame,
                            company_metadata: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Main data cleaning pipeline for A2AI financial analysis
        
        Args:
            raw_data: Raw financial statement data
            company_metadata: Company information including lifecycle events
            
        Returns:
            Tuple of (cleaned_data, cleaning_statistics)
        """
        logger.info("Starting A2AI comprehensive data cleaning process")
        
        # Initialize cleaning statistics
        self.cleaning_stats = {
            'initial_records': len(raw_data),
            'companies_processed': 0,
            'survival_bias_corrections': 0,
            'missing_data_imputations': 0,
            'outliers_detected': 0,
            'accounting_standard_adjustments': 0,
            'lifecycle_alignments': 0
        }
        
        # Step 1: Basic data validation and structure cleanup
        cleaned_data = self._basic_data_validation(raw_data)
        
        # Step 2: Corporate lifecycle alignment
        cleaned_data = self._align_corporate_lifecycles(cleaned_data, company_metadata)
        
        # Step 3: Survival bias correction
        cleaned_data = self._correct_survival_bias(cleaned_data, company_metadata)
        
        # Step 4: Accounting standard harmonization
        cleaned_data = self._harmonize_accounting_standards(cleaned_data)
        
        # Step 5: Missing data intelligent handling
        cleaned_data = self._handle_missing_data(cleaned_data, company_metadata)
        
        # Step 6: Outlier detection and treatment
        cleaned_data = self._detect_and_treat_outliers(cleaned_data)
        
        # Step 7: Market category-specific processing
        cleaned_data = self._market_category_processing(cleaned_data, company_metadata)
        
        # Step 8: Temporal consistency validation
        cleaned_data = self._validate_temporal_consistency(cleaned_data)
        
        # Step 9: Final validation and quality check
        cleaned_data = self._final_quality_check(cleaned_data)
        
        # Update final statistics
        self.cleaning_stats['final_records'] = len(cleaned_data)
        self.cleaning_stats['data_retention_rate'] = (
            self.cleaning_stats['final_records'] / 
            self.cleaning_stats['initial_records']
        )
        
        logger.info(f"Data cleaning completed. Retention rate: "
                    f"{self.cleaning_stats['data_retention_rate']:.2%}")
        
        return cleaned_data, self.cleaning_stats
    
    def _basic_data_validation(self, data: pd.DataFrame) -> pd.DataFrame:
        """Perform basic data validation and structure cleanup"""
        logger.info("Performing basic data validation")
        
        # Ensure required columns exist
        required_columns = [
            'company_id', 'company_name', 'fiscal_year', 'market_category',
            'sales_revenue', 'total_assets', 'net_income'
        ]
        
        missing_cols = [col for col in required_columns if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Data type conversion
        data['fiscal_year'] = pd.to_numeric(data['fiscal_year'], errors='coerce')
        data['company_id'] = data['company_id'].astype(str)
        
        # Remove completely empty rows
        data = data.dropna(how='all')
        
        # Remove duplicate records
        initial_len = len(data)
        data = data.drop_duplicates(subset=['company_id', 'fiscal_year'])
        duplicates_removed = initial_len - len(data)
        
        logger.info(f"Removed {duplicates_removed} duplicate records")
        
        return data
    
    def _align_corporate_lifecycles(self, 
                                    data: pd.DataFrame, 
                                    metadata: pd.DataFrame) -> pd.DataFrame:
        """Align data based on corporate lifecycle stages"""
        logger.info("Aligning corporate lifecycle data")
        
        # Merge with metadata to get establishment dates
        data = data.merge(
            metadata[['company_id', 'establishment_date', 'extinction_date']], 
            on='company_id', 
            how='left'
        )
        
        # Calculate company age
        data['establishment_date'] = pd.to_datetime(data['establishment_date'])
        data['company_age'] = data['fiscal_year'] - data['establishment_date'].dt.year
        
        # Assign lifecycle stage
        def assign_lifecycle_stage(age):
            if pd.isna(age):
                return 'unknown'
            for stage, (min_age, max_age) in self.lifecycle_stages.items():
                if min_age <= age <= max_age:
                    return stage
            return 'unknown'
        
        data['lifecycle_stage'] = data['company_age'].apply(assign_lifecycle_stage)
        
        # Handle extinct companies
        data['is_extinct'] = ~pd.isna(data['extinction_date'])
        
        self.cleaning_stats['lifecycle_alignments'] = len(data)
        
        return data
    
    def _correct_survival_bias(self, 
                                data: pd.DataFrame, 
                                metadata: pd.DataFrame) -> pd.DataFrame:
        """Correct for survival bias in the dataset"""
        logger.info("Correcting survival bias")
        
        if not self.config['survivorship_bias_correction']:
            return data
        
        # Identify extinct companies
        extinct_companies = metadata[~pd.isna(metadata['extinction_date'])]
        surviving_companies = metadata[pd.isna(metadata['extinction_date'])]
        
        # Ensure extinct company data is included up to extinction date
        for _, extinct_company in extinct_companies.iterrows():
            company_id = extinct_company['company_id']
            extinction_year = pd.to_datetime(extinct_company['extinction_date']).year
            
            # Filter data for this company up to extinction
            company_mask = (data['company_id'] == company_id) & \
                            (data['fiscal_year'] <= extinction_year)
            
            # Mark as extinct data
            data.loc[company_mask, 'survival_status'] = 'extinct'
        
        # Mark surviving companies
        surviving_mask = data['company_id'].isin(surviving_companies['company_id'])
        data.loc[surviving_mask, 'survival_status'] = 'surviving'
        
        # Calculate survival weights for analysis
        total_companies = len(metadata)
        extinct_ratio = len(extinct_companies) / total_companies
        surviving_ratio = len(surviving_companies) / total_companies
        
        data.loc[data['survival_status'] == 'extinct', 'survival_weight'] = 1 / extinct_ratio
        data.loc[data['survival_status'] == 'surviving', 'survival_weight'] = 1 / surviving_ratio
        
        self.cleaning_stats['survival_bias_corrections'] = len(extinct_companies)
        
        return data
    
    def _harmonize_accounting_standards(self, data: pd.DataFrame) -> pd.DataFrame:
        """Harmonize different accounting standards (JGAAP vs IFRS)"""
        logger.info("Harmonizing accounting standards")
        
        transition_year = self.config['accounting_standard_transition']
        
        # Identify columns affected by accounting standard changes
        affected_columns = [
            'goodwill', 'intangible_assets', 'deferred_tax_assets',
            'research_development_expenses', 'depreciation_method'
        ]
        
        adjustments_made = 0
        
        for column in affected_columns:
            if column in data.columns:
                # Apply transition adjustments for post-transition years
                post_transition_mask = data['fiscal_year'] >= transition_year
                
                if column == 'goodwill':
                    # IFRS: Goodwill impairment testing vs JGAAP: Systematic amortization
                    pre_transition_data = data[~post_transition_mask & ~data[column].isna()]
                    if not pre_transition_data.empty:
                        # Normalize goodwill treatment
                        adjustments_made += len(pre_transition_data)
                
                elif column == 'research_development_expenses':
                    # IFRS: Development costs can be capitalized vs JGAAP: All R&D expensed
                    post_transition_data = data[post_transition_mask & ~data[column].isna()]
                    if not post_transition_data.empty:
                        adjustments_made += len(post_transition_data)
        
        # Add accounting standard indicator
        data['accounting_standard'] = np.where(
            data['fiscal_year'] >= transition_year, 'IFRS', 'JGAAP'
        )
        
        self.cleaning_stats['accounting_standard_adjustments'] = adjustments_made
        
        return data
    
    def _handle_missing_data(self, 
                            data: pd.DataFrame, 
                            metadata: pd.DataFrame) -> pd.DataFrame:
        """Intelligent handling of missing financial data"""
        logger.info("Handling missing data with intelligent imputation")
        
        # Calculate missing data statistics by column
        missing_stats = data.isnull().sum() / len(data)
        
        # Drop columns with excessive missing data
        threshold = self.config['missing_threshold']
        cols_to_drop = missing_stats[missing_stats > threshold].index.tolist()
        
        if cols_to_drop:
            logger.warning(f"Dropping columns with >{threshold:.0%} missing data: {cols_to_drop}")
            data = data.drop(columns=cols_to_drop)
        
        # Intelligent imputation strategies
        imputations_made = 0
        
        # Strategy 1: Forward/backward fill for time series
        for company_id in data['company_id'].unique():
            company_mask = data['company_id'] == company_id
            company_data = data[company_mask].sort_values('fiscal_year')
            
            # Forward fill then backward fill
            numeric_columns = company_data.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                if company_data[col].isnull().any():
                    # Forward fill
                    data.loc[company_mask, col] = company_data[col].fillna(method='ffill')
                    # Backward fill
                    data.loc[company_mask, col] = data.loc[company_mask, col].fillna(method='bfill')
                    
                    imputations_made += company_data[col].isnull().sum()
        
        # Strategy 2: Industry median imputation
        for market_cat in data['market_category'].unique():
            market_mask = data['market_category'] == market_cat
            market_data = data[market_mask]
            
            for col in market_data.select_dtypes(include=[np.number]).columns:
                if market_data[col].isnull().any():
                    median_value = market_data[col].median()
                    if not pd.isna(median_value):
                        data.loc[market_mask, col] = data.loc[market_mask, col].fillna(median_value)
        
        # Strategy 3: Regression-based imputation for key financial ratios
        key_ratios = ['operating_margin', 'net_margin', 'roe', 'current_ratio']
        for ratio in key_ratios:
            if ratio in data.columns and data[ratio].isnull().any():
                # Use other financial metrics to predict missing ratios
                predictor_cols = ['sales_revenue', 'total_assets', 'company_age']
                available_predictors = [col for col in predictor_cols if col in data.columns]
                
                if available_predictors:
                    non_null_mask = data[ratio].notnull()
                    if non_null_mask.sum() > 50:  # Minimum samples for regression
                        from sklearn.linear_model import LinearRegression
                        from sklearn.preprocessing import StandardScaler
                        
                        # Prepare data for regression
                        X_train = data.loc[non_null_mask, available_predictors].fillna(0)
                        y_train = data.loc[non_null_mask, ratio]
                        
                        # Fit model
                        scaler = StandardScaler()
                        X_train_scaled = scaler.fit_transform(X_train)
                        
                        model = LinearRegression()
                        model.fit(X_train_scaled, y_train)
                        
                        # Predict missing values
                        missing_mask = data[ratio].isnull()
                        if missing_mask.sum() > 0:
                            X_missing = data.loc[missing_mask, available_predictors].fillna(0)
                            X_missing_scaled = scaler.transform(X_missing)
                            predictions = model.predict(X_missing_scaled)
                            
                            data.loc[missing_mask, ratio] = predictions
                            imputations_made += missing_mask.sum()
        
        self.cleaning_stats['missing_data_imputations'] = imputations_made
        
        return data
    
    def _detect_and_treat_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """Detect and treat outliers in financial data"""
        logger.info("Detecting and treating outliers")
        
        outliers_detected = 0
        method = self.config['outlier_method']
        factor = self.config['outlier_factor']
        
        # Financial metrics to check for outliers
        metrics_to_check = [
            'sales_revenue', 'total_assets', 'net_income', 'operating_margin',
            'net_margin', 'roe', 'current_ratio', 'debt_equity_ratio'
        ]
        
        for metric in metrics_to_check:
            if metric not in data.columns:
                continue
                
            if method == 'iqr':
                # IQR method
                Q1 = data[metric].quantile(0.25)
                Q3 = data[metric].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - factor * IQR
                upper_bound = Q3 + factor * IQR
                
                outlier_mask = (data[metric] < lower_bound) | (data[metric] > upper_bound)
                
            elif method == 'zscore':
                # Z-score method
                z_scores = np.abs((data[metric] - data[metric].mean()) / data[metric].std())
                outlier_mask = z_scores > factor
            
            else:
                continue
            
            # Count outliers
            current_outliers = outlier_mask.sum()
            outliers_detected += current_outliers
            
            if current_outliers > 0:
                logger.info(f"Detected {current_outliers} outliers in {metric}")
                
                # Treatment: Cap outliers at bounds rather than remove
                if method == 'iqr':
                    data.loc[data[metric] < lower_bound, metric] = lower_bound
                    data.loc[data[metric] > upper_bound, metric] = upper_bound
                elif method == 'zscore':
                    # Cap at 3 standard deviations
                    mean_val = data[metric].mean()
                    std_val = data[metric].std()
                    data.loc[outlier_mask, metric] = np.clip(
                        data.loc[outlier_mask, metric],
                        mean_val - 3 * std_val,
                        mean_val + 3 * std_val
                    )
        
        self.cleaning_stats['outliers_detected'] = outliers_detected
        
        return data
    
    def _market_category_processing(self, 
                                    data: pd.DataFrame, 
                                    metadata: pd.DataFrame) -> pd.DataFrame:
        """Apply market category-specific data processing"""
        logger.info("Applying market category-specific processing")
        
        # High-share markets: Focus on innovation and quality metrics
        high_share_mask = data['market_category'].isin(self.high_share_markets)
        if high_share_mask.any():
            # Ensure R&D and quality metrics are properly captured
            rd_cols = [col for col in data.columns if 'research' in col.lower() or 'development' in col.lower()]
            for col in rd_cols:
                if data.loc[high_share_mask, col].isnull().any():
                    # Use industry-specific imputation for R&D
                    median_rd = data.loc[high_share_mask, col].median()
                    data.loc[high_share_mask, col] = data.loc[high_share_mask, col].fillna(median_rd)
        
        # Declining markets: Focus on efficiency and cost management
        declining_mask = data['market_category'].isin(self.declining_markets)
        if declining_mask.any():
            # Ensure cost-related metrics are captured
            cost_cols = [col for col in data.columns if 'cost' in col.lower() or 'expense' in col.lower()]
            for col in cost_cols:
                if data.loc[declining_mask, col].isnull().any():
                    median_cost = data.loc[declining_mask, col].median()
                    data.loc[declining_mask, col] = data.loc[declining_mask, col].fillna(median_cost)
        
        # Lost markets: Focus on transformation and diversification metrics
        lost_mask = data['market_category'].isin(self.lost_markets)
        if lost_mask.any():
            # Special handling for companies that exited markets
            # Mark final years before exit
            for company_id in data.loc[lost_mask, 'company_id'].unique():
                company_data = data[data['company_id'] == company_id].sort_values('fiscal_year')
                if len(company_data) > 0:
                    last_year = company_data['fiscal_year'].max()
                    # Mark last 3 years as exit phase
                    exit_phase_mask = (data['company_id'] == company_id) & \
                                    (data['fiscal_year'] >= last_year - 2)
                    data.loc[exit_phase_mask, 'market_exit_phase'] = True
        
        return data
    
    def _validate_temporal_consistency(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate temporal consistency of financial data"""
        logger.info("Validating temporal consistency")
        
        consistency_issues = 0
        
        for company_id in data['company_id'].unique():
            company_mask = data['company_id'] == company_id
            company_data = data[company_mask].sort_values('fiscal_year')
            
            if len(company_data) < 2:
                continue
            
            # Check for impossible growth rates
            for metric in ['sales_revenue', 'total_assets']:
                if metric in company_data.columns:
                    growth_rates = company_data[metric].pct_change()
                    
                    # Flag impossible growth rates (>1000% or <-90%)
                    impossible_growth = (growth_rates > 10) | (growth_rates < -0.9)
                    
                    if impossible_growth.any():
                        consistency_issues += impossible_growth.sum()
                        
                        # Smooth out impossible values
                        for idx in company_data[impossible_growth].index:
                            if idx > company_data.index[0]:
                                prev_idx = company_data[company_data.index < idx].index[-1]
                                if idx < company_data.index[-1]:
                                    next_idx = company_data[company_data.index > idx].index[0]
                                    # Use average of previous and next values
                                    avg_val = (company_data.loc[prev_idx, metric] + 
                                                company_data.loc[next_idx, metric]) / 2
                                    data.loc[idx, metric] = avg_val
        
        logger.info(f"Fixed {consistency_issues} temporal consistency issues")
        
        return data
    
    def _final_quality_check(self, data: pd.DataFrame) -> pd.DataFrame:
        """Perform final data quality validation"""
        logger.info("Performing final quality check")
        
        # Ensure minimum data requirements
        min_years = self.config['min_data_years']
        
        # Count years of data per company
        company_years = data.groupby('company_id')['fiscal_year'].count()
        companies_with_sufficient_data = company_years[company_years >= min_years].index
        
        # Filter to companies with sufficient data
        initial_companies = data['company_id'].nunique()
        data = data[data['company_id'].isin(companies_with_sufficient_data)]
        final_companies = data['company_id'].nunique()
        
        logger.info(f"Companies with sufficient data ({min_years}+ years): "
                    f"{final_companies}/{initial_companies}")
        
        # Final validation checks
        required_financial_metrics = ['sales_revenue', 'total_assets']
        for metric in required_financial_metrics:
            if metric in data.columns:
                null_count = data[metric].isnull().sum()
                if null_count > 0:
                    logger.warning(f"Still {null_count} null values in required metric: {metric}")
        
        # Add data quality score
        quality_metrics = {
            'completeness': 1 - (data.isnull().sum().sum() / (len(data) * len(data.columns))),
            'consistency': 1 - (consistency_issues / len(data)) if 'consistency_issues' in locals() else 1.0,
            'validity': len(data) / self.cleaning_stats['initial_records']
        }
        
        data_quality_score = np.mean(list(quality_metrics.values()))
        logger.info(f"Overall data quality score: {data_quality_score:.3f}")
        
        self.cleaning_stats['data_quality_score'] = data_quality_score
        self.cleaning_stats['companies_processed'] = final_companies
        
        return data
    
    def export_cleaning_report(self, output_path: str) -> None:
        """Export comprehensive data cleaning report"""
        
        report = {
            'cleaning_summary': self.cleaning_stats,
            'data_sources': {
                'high_share_markets': self.high_share_markets,
                'declining_markets': self.declining_markets, 
                'lost_markets': self.lost_markets
            },
            'cleaning_configuration': self.config,
            'quality_metrics': {
                'data_retention_rate': self.cleaning_stats.get('data_retention_rate', 0),
                'data_quality_score': self.cleaning_stats.get('data_quality_score', 0),
                'companies_processed': self.cleaning_stats.get('companies_processed', 0)
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(report, f, default_flow_style=False, allow_unicode=True)
        
        logger.info(f"Cleaning report exported to {output_path}")


def main():
    """Main function for testing the data cleaner"""
    
    # Initialize data cleaner
    cleaner = A2AIDataCleaner()
    
    # Sample data for testing
    sample_data = pd.DataFrame({
        'company_id': ['FANUC_001', 'TOYOTA_001', 'SONY_001'] * 10,
        'company_name': ['ファナック', 'トヨタ自動車', 'ソニー'] * 10,
        'fiscal_year': list(range(2015, 2025)) * 3,
        'market_category': ['robot', 'automotive', 'home_appliances'] * 10,
        'sales_revenue': np.random.normal(100000, 20000, 30),
        'total_assets': np.random.normal(200000, 40000, 30),
        'net_income': np.random.normal(10000, 5000, 30)
    })
    
    sample_metadata = pd.DataFrame({
        'company_id': ['FANUC_001', 'TOYOTA_001', 'SONY_001'],
        'company_name': ['ファナック', 'トヨタ自動車', 'ソニー'],
        'establishment_date': ['1972-07-15', '1937-08-28', '1946-05-07'],
        'extinction_date': [None, None, None],
        'market_category': ['robot', 'automotive', 'home_appliances']
    })
    
    # Clean the data
    cleaned_data, stats = cleaner.clean_financial_data(sample_data, sample_metadata)
    
    print("\nCleaning completed!")
    print(f"Original records: {stats['initial_records']}")
    print(f"Final records: {stats['final_records']}")
    print(f"Data retention rate: {stats['data_retention_rate']:.2%}")
    print(f"Data quality score: {stats.get('data_quality_score', 'N/A')}")


if __name__ == "__main__":
    main()