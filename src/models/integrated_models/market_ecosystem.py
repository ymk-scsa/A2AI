"""
Market Ecosystem Analysis Model for A2AI (Advanced Financial Analysis AI)

This module implements advanced market ecosystem analysis to understand the complex
relationships between companies across different market share categories (high-share,
declining, and lost markets) and their competitive dynamics over time.

Author: A2AI Development Team
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import networkx as nx
from scipy import stats
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings
from dataclasses import dataclass
from abc import ABC, abstractmethod

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


@dataclass
class MarketPosition:
    """Data class for company market position"""
    company_id: str
    market_category: str  # 'high_share', 'declining', 'lost'
    market_segment: str   # specific market (e.g., 'robotics', 'endoscope')
    share_percentage: float
    position_rank: int
    competitive_strength: float
    year: int


@dataclass
class EcosystemMetrics:
    """Data class for ecosystem-level metrics"""
    market_concentration: float
    competitive_intensity: float
    innovation_density: float
    lifecycle_diversity: float
    survival_resilience: float
    emergence_vitality: float


class BaseEcosystemAnalyzer(ABC):
    """Abstract base class for ecosystem analyzers"""
    
    @abstractmethod
    def fit(self, data: pd.DataFrame) -> None:
        """Fit the analyzer to the data"""
        pass
    
    @abstractmethod
    def analyze(self) -> Dict:
        """Perform the ecosystem analysis"""
        pass


class NetworkTopologyAnalyzer(BaseEcosystemAnalyzer):
    """Analyzes the network topology of market ecosystems"""
    
    def __init__(self):
        self.graph = None
        self.centrality_metrics = {}
        self.community_structure = {}
    
    def fit(self, data: pd.DataFrame) -> None:
        """
        Build network graph from company relationships
        
        Args:
            data: DataFrame with company financial and market data
        """
        self.data = data
        self.graph = nx.Graph()
        self._build_network()
    
    def _build_network(self) -> None:
        """Build network graph based on market relationships and competitive positions"""
        companies = self.data['company_name'].unique()
        
        # Add nodes (companies)
        for company in companies:
            company_data = self.data[self.data['company_name'] == company].iloc[-1]
            self.graph.add_node(
                company,
                market_category=company_data.get('market_category', 'unknown'),
                market_segment=company_data.get('market_segment', 'unknown'),
                financial_strength=company_data.get('total_assets', 0),
                innovation_intensity=company_data.get('rd_expense_ratio', 0)
            )
        
        # Add edges based on competitive relationships
        for i, company1 in enumerate(companies):
            for company2 in companies[i+1:]:
                weight = self._calculate_relationship_strength(company1, company2)
                if weight > 0.1:  # Threshold for significant relationship
                    self.graph.add_edge(company1, company2, weight=weight)
    
    def _calculate_relationship_strength(self, company1: str, company2: str) -> float:
        """Calculate relationship strength between two companies"""
        data1 = self.data[self.data['company_name'] == company1]
        data2 = self.data[self.data['company_name'] == company2]
        
        if data1.empty or data2.empty:
            return 0.0
        
        # Same market segment increases relationship strength
        market_similarity = 1.0 if (data1.iloc[-1].get('market_segment') == 
                                    data2.iloc[-1].get('market_segment')) else 0.3
        
        # Similar financial characteristics increase relationship strength
        financial_similarity = self._calculate_financial_similarity(data1, data2)
        
        # Geographic overlap (if available)
        geographic_similarity = 0.5  # Default value
        
        return (market_similarity * 0.5 + financial_similarity * 0.3 + 
                geographic_similarity * 0.2)
    
    def _calculate_financial_similarity(self, data1: pd.DataFrame, data2: pd.DataFrame) -> float:
        """Calculate financial similarity between companies"""
        try:
            metrics1 = self._extract_financial_metrics(data1)
            metrics2 = self._extract_financial_metrics(data2)
            
            # Calculate correlation between financial metrics
            correlation = np.corrcoef(metrics1, metrics2)[0, 1]
            return max(0, correlation) if not np.isnan(correlation) else 0.0
        except:
            return 0.0
    
    def _extract_financial_metrics(self, data: pd.DataFrame) -> np.ndarray:
        """Extract key financial metrics for comparison"""
        recent_data = data.iloc[-5:]  # Last 5 years
        
        metrics = []
        metric_columns = [
            'sales_growth_rate', 'operating_profit_margin', 'roe',
            'total_asset_turnover', 'rd_expense_ratio', 'employee_count'
        ]
        
        for col in metric_columns:
            if col in recent_data.columns:
                metric_value = recent_data[col].mean()
                metrics.append(metric_value if not pd.isna(metric_value) else 0)
            else:
                metrics.append(0)
        
        return np.array(metrics)
    
    def analyze(self) -> Dict:
        """Perform network topology analysis"""
        if self.graph is None:
            raise ValueError("Must fit the analyzer first")
        
        # Calculate centrality metrics
        self.centrality_metrics = {
            'degree_centrality': nx.degree_centrality(self.graph),
            'betweenness_centrality': nx.betweenness_centrality(self.graph),
            'closeness_centrality': nx.closeness_centrality(self.graph),
            'eigenvector_centrality': nx.eigenvector_centrality(self.graph, max_iter=1000)
        }
        
        # Detect community structure
        try:
            self.community_structure = nx.community.greedy_modularity_communities(self.graph)
        except:
            self.community_structure = []
        
        # Calculate network-level metrics
        network_metrics = {
            'network_density': nx.density(self.graph),
            'average_clustering': nx.average_clustering(self.graph),
            'number_of_components': nx.number_connected_components(self.graph),
            'average_path_length': self._safe_average_path_length(),
            'modularity': self._calculate_modularity()
        }
        
        return {
            'centrality_metrics': self.centrality_metrics,
            'community_structure': self.community_structure,
            'network_metrics': network_metrics
        }
    
    def _safe_average_path_length(self) -> float:
        """Calculate average path length safely"""
        try:
            if nx.is_connected(self.graph):
                return nx.average_shortest_path_length(self.graph)
            else:
                # For disconnected graphs, calculate for largest component
                largest_cc = max(nx.connected_components(self.graph), key=len)
                subgraph = self.graph.subgraph(largest_cc)
                return nx.average_shortest_path_length(subgraph)
        except:
            return float('inf')
    
    def _calculate_modularity(self) -> float:
        """Calculate network modularity"""
        try:
            if len(self.community_structure) > 1:
                return nx.community.modularity(self.graph, self.community_structure)
            else:
                return 0.0
        except:
            return 0.0


class CompetitivePositioningAnalyzer(BaseEcosystemAnalyzer):
    """Analyzes competitive positioning within market ecosystems"""
    
    def __init__(self, n_clusters: int = 5):
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)  # Retain 95% variance
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.positioning_data = None
        self.clusters = None
    
    def fit(self, data: pd.DataFrame) -> None:
        """
        Prepare data for competitive positioning analysis
        
        Args:
            data: DataFrame with company financial and market data
        """
        self.data = data
        self._prepare_positioning_features()
    
    def _prepare_positioning_features(self) -> None:
        """Prepare features for positioning analysis"""
        # Group by company and calculate average metrics over recent years
        recent_years = self.data['year'].max() - 5
        recent_data = self.data[self.data['year'] >= recent_years]
        
        positioning_features = []
        
        for company in recent_data['company_name'].unique():
            company_data = recent_data[recent_data['company_name'] == company]
            
            # Calculate positioning dimensions
            features = {
                'company_name': company,
                'market_category': company_data['market_category'].iloc[0],
                'financial_strength': company_data['total_assets'].mean(),
                'profitability': company_data['operating_profit_margin'].mean(),
                'growth_rate': company_data['sales_growth_rate'].mean(),
                'innovation_intensity': company_data['rd_expense_ratio'].mean(),
                'efficiency': company_data['total_asset_turnover'].mean(),
                'market_share': company_data.get('market_share', pd.Series([0])).mean(),
                'international_presence': company_data.get('overseas_sales_ratio', pd.Series([0])).mean(),
                'employee_productivity': (company_data['sales'].mean() / 
                                        company_data['employee_count'].mean() 
                                        if company_data['employee_count'].mean() > 0 else 0),
                'debt_ratio': company_data.get('debt_ratio', pd.Series([0])).mean(),
                'survival_years': company_data['year'].max() - company_data['year'].min() + 1
            }
            
            # Fill NaN values with category means or zeros
            for key, value in features.items():
                if pd.isna(value) or np.isinf(value):
                    features[key] = 0
            
            positioning_features.append(features)
        
        self.positioning_data = pd.DataFrame(positioning_features)
    
    def analyze(self) -> Dict:
        """Perform competitive positioning analysis"""
        if self.positioning_data is None:
            raise ValueError("Must fit the analyzer first")
        
        # Prepare feature matrix for clustering
        feature_columns = [col for col in self.positioning_data.columns 
                            if col not in ['company_name', 'market_category']]
        X = self.positioning_data[feature_columns].values
        
        # Handle missing values
        X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Apply PCA for dimensionality reduction
        X_pca = self.pca.fit_transform(X_scaled)
        
        # Perform clustering
        self.clusters = self.kmeans.fit_predict(X_pca)
        
        # Calculate competitive positioning metrics
        positioning_analysis = {
            'cluster_assignments': dict(zip(self.positioning_data['company_name'], 
                                            self.clusters)),
            'cluster_centers': self.kmeans.cluster_centers_,
            'silhouette_score': silhouette_score(X_pca, self.clusters),
            'positioning_dimensions': self._analyze_positioning_dimensions(X_scaled),
            'competitive_groups': self._identify_competitive_groups(),
            'market_leadership': self._analyze_market_leadership()
        }
        
        return positioning_analysis
    
    def _analyze_positioning_dimensions(self, X_scaled: np.ndarray) -> Dict:
        """Analyze key positioning dimensions"""
        feature_columns = [col for col in self.positioning_data.columns 
                            if col not in ['company_name', 'market_category']]
        
        # Calculate feature importance based on PCA components
        feature_importance = np.abs(self.pca.components_).mean(axis=0)
        
        dimensions = {}
        for i, feature in enumerate(feature_columns):
            dimensions[feature] = {
                'importance': feature_importance[i],
                'mean_by_cluster': [X_scaled[self.clusters == cluster, i].mean() 
                                    for cluster in range(self.n_clusters)]
            }
        
        return dimensions
    
    def _identify_competitive_groups(self) -> Dict:
        """Identify competitive groups based on clustering results"""
        competitive_groups = {}
        
        for cluster_id in range(self.n_clusters):
            cluster_companies = self.positioning_data[
                self.clusters == cluster_id]['company_name'].tolist()
            cluster_categories = self.positioning_data[
                self.clusters == cluster_id]['market_category'].value_counts().to_dict()
            
            competitive_groups[f'group_{cluster_id}'] = {
                'companies': cluster_companies,
                'market_category_distribution': cluster_categories,
                'group_size': len(cluster_companies),
                'dominant_category': max(cluster_categories.keys(), 
                                        key=cluster_categories.get) if cluster_categories else 'unknown'
            }
        
        return competitive_groups
    
    def _analyze_market_leadership(self) -> Dict:
        """Analyze market leadership patterns"""
        leadership_analysis = {}
        
        # Identify leaders by market category
        for category in self.positioning_data['market_category'].unique():
            category_data = self.positioning_data[
                self.positioning_data['market_category'] == category]
            
            # Define leadership criteria
            leaders = category_data.nlargest(3, 'financial_strength')
            
            leadership_analysis[category] = {
                'top_companies': leaders['company_name'].tolist(),
                'leadership_metrics': {
                    'avg_financial_strength': leaders['financial_strength'].mean(),
                    'avg_profitability': leaders['profitability'].mean(),
                    'avg_innovation_intensity': leaders['innovation_intensity'].mean()
                }
            }
        
        return leadership_analysis


class EcosystemDynamicsAnalyzer(BaseEcosystemAnalyzer):
    """Analyzes temporal dynamics of market ecosystems"""
    
    def __init__(self, time_window: int = 5):
        self.time_window = time_window
        self.dynamics_data = None
        self.transition_matrices = {}
        self.ecosystem_evolution = {}
    
    def fit(self, data: pd.DataFrame) -> None:
        """
        Prepare data for ecosystem dynamics analysis
        
        Args:
            data: DataFrame with company financial and market data over time
        """
        self.data = data
        self._prepare_dynamics_data()
    
    def _prepare_dynamics_data(self) -> None:
        """Prepare temporal data for dynamics analysis"""
        # Create time-based snapshots of ecosystem
        years = sorted(self.data['year'].unique())
        
        ecosystem_snapshots = []
        
        for year in years:
            year_data = self.data[self.data['year'] == year]
            
            # Calculate ecosystem metrics for this year
            metrics = self._calculate_yearly_ecosystem_metrics(year_data)
            metrics['year'] = year
            
            ecosystem_snapshots.append(metrics)
        
        self.dynamics_data = pd.DataFrame(ecosystem_snapshots)
    
    def _calculate_yearly_ecosystem_metrics(self, year_data: pd.DataFrame) -> Dict:
        """Calculate ecosystem metrics for a specific year"""
        metrics = {}
        
        # Market concentration (Herfindahl Index)
        if 'market_share' in year_data.columns:
            shares = year_data['market_share'].dropna()
            metrics['market_concentration'] = (shares ** 2).sum() if len(shares) > 0 else 0
        else:
            metrics['market_concentration'] = 0
        
        # Competitive intensity
        metrics['competitive_intensity'] = len(year_data)
        
        # Innovation density
        rd_ratios = year_data['rd_expense_ratio'].dropna()
        metrics['innovation_density'] = rd_ratios.mean() if len(rd_ratios) > 0 else 0
        
        # Lifecycle diversity (based on company ages)
        if 'company_age' in year_data.columns:
            ages = year_data['company_age'].dropna()
            metrics['lifecycle_diversity'] = ages.std() if len(ages) > 1 else 0
        else:
            metrics['lifecycle_diversity'] = 0
        
        # Financial health distribution
        profit_margins = year_data['operating_profit_margin'].dropna()
        metrics['avg_profitability'] = profit_margins.mean() if len(profit_margins) > 0 else 0
        metrics['profitability_dispersion'] = profit_margins.std() if len(profit_margins) > 1 else 0
        
        # Growth momentum
        growth_rates = year_data['sales_growth_rate'].dropna()
        metrics['avg_growth_rate'] = growth_rates.mean() if len(growth_rates) > 0 else 0
        metrics['growth_volatility'] = growth_rates.std() if len(growth_rates) > 1 else 0
        
        return metrics
    
    def analyze(self) -> Dict:
        """Perform ecosystem dynamics analysis"""
        if self.dynamics_data is None:
            raise ValueError("Must fit the analyzer first")
        
        # Analyze ecosystem evolution trends
        evolution_analysis = self._analyze_ecosystem_evolution()
        
        # Identify regime changes
        regime_changes = self._identify_regime_changes()
        
        # Calculate ecosystem stability metrics
        stability_metrics = self._calculate_stability_metrics()
        
        # Analyze cross-correlation between ecosystem dimensions
        correlation_analysis = self._analyze_dimension_correlations()
        
        return {
            'evolution_analysis': evolution_analysis,
            'regime_changes': regime_changes,
            'stability_metrics': stability_metrics,
            'correlation_analysis': correlation_analysis,
            'ecosystem_trajectory': self._create_ecosystem_trajectory()
        }
    
    def _analyze_ecosystem_evolution(self) -> Dict:
        """Analyze long-term ecosystem evolution patterns"""
        evolution = {}
        
        numeric_columns = self.dynamics_data.select_dtypes(include=[np.number]).columns
        numeric_columns = [col for col in numeric_columns if col != 'year']
        
        for metric in numeric_columns:
            values = self.dynamics_data[metric].values
            years = self.dynamics_data['year'].values
            
            # Calculate trend
            if len(values) > 1:
                slope, intercept, r_value, p_value, std_err = stats.linregress(years, values)
                
                evolution[metric] = {
                    'trend_slope': slope,
                    'trend_strength': abs(r_value),
                    'trend_significance': p_value,
                    'volatility': np.std(values),
                    'mean_value': np.mean(values),
                    'direction': 'increasing' if slope > 0 else 'decreasing'
                }
            else:
                evolution[metric] = {
                    'trend_slope': 0,
                    'trend_strength': 0,
                    'trend_significance': 1,
                    'volatility': 0,
                    'mean_value': values[0] if len(values) > 0 else 0,
                    'direction': 'stable'
                }
        
        return evolution
    
    def _identify_regime_changes(self) -> List[Dict]:
        """Identify structural breaks/regime changes in ecosystem dynamics"""
        regime_changes = []
        
        # Simple change point detection based on moving averages
        for metric in ['market_concentration', 'competitive_intensity', 'innovation_density']:
            if metric in self.dynamics_data.columns:
                values = self.dynamics_data[metric].values
                years = self.dynamics_data['year'].values
                
                if len(values) >= 10:  # Need sufficient data points
                    # Calculate moving averages
                    window = min(5, len(values) // 3)
                    ma1 = np.convolve(values[:len(values)//2], 
                                        np.ones(window)/window, mode='valid')
                    ma2 = np.convolve(values[len(values)//2:], 
                                        np.ones(window)/window, mode='valid')
                    
                    if len(ma1) > 0 and len(ma2) > 0:
                        # Test for significant difference
                        t_stat, p_value = stats.ttest_ind(ma1, ma2)
                        
                        if p_value < 0.05:  # Significant change detected
                            change_point = years[len(values)//2]
                            regime_changes.append({
                                'metric': metric,
                                'change_point_year': change_point,
                                'significance': p_value,
                                'magnitude': abs(np.mean(ma2) - np.mean(ma1))
                            })
        
        return regime_changes
    
    def _calculate_stability_metrics(self) -> Dict:
        """Calculate ecosystem stability metrics"""
        stability = {}
        
        numeric_columns = self.dynamics_data.select_dtypes(include=[np.number]).columns
        numeric_columns = [col for col in numeric_columns if col != 'year']
        
        for metric in numeric_columns:
            values = self.dynamics_data[metric].values
            
            if len(values) > 1:
                # Coefficient of variation
                cv = np.std(values) / (np.mean(values) + 1e-10)
                
                # Autocorrelation (persistence)
                if len(values) > 2:
                    autocorr = np.corrcoef(values[:-1], values[1:])[0, 1]
                    autocorr = autocorr if not np.isnan(autocorr) else 0
                else:
                    autocorr = 0
                
                stability[metric] = {
                    'coefficient_of_variation': cv,
                    'autocorrelation': autocorr,
                    'stability_index': 1 / (1 + cv)  # Higher values = more stable
                }
            else:
                stability[metric] = {
                    'coefficient_of_variation': 0,
                    'autocorrelation': 0,
                    'stability_index': 1
                }
        
        return stability
    
    def _analyze_dimension_correlations(self) -> Dict:
        """Analyze correlations between ecosystem dimensions"""
        numeric_columns = self.dynamics_data.select_dtypes(include=[np.number]).columns
        numeric_columns = [col for col in numeric_columns if col != 'year']
        
        if len(numeric_columns) > 1:
            correlation_matrix = self.dynamics_data[numeric_columns].corr()
            
            # Find strongest correlations
            strong_correlations = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    corr_value = correlation_matrix.iloc[i, j]
                    if abs(corr_value) > 0.5:  # Threshold for strong correlation
                        strong_correlations.append({
                            'dimension_1': correlation_matrix.columns[i],
                            'dimension_2': correlation_matrix.columns[j],
                            'correlation': corr_value,
                            'strength': 'strong' if abs(corr_value) > 0.7 else 'moderate'
                        })
            
            return {
                'correlation_matrix': correlation_matrix.to_dict(),
                'strong_correlations': strong_correlations
            }
        else:
            return {'correlation_matrix': {}, 'strong_correlations': []}
    
    def _create_ecosystem_trajectory(self) -> Dict:
        """Create ecosystem trajectory representation"""
        trajectory = {
            'time_periods': self.dynamics_data['year'].tolist(),
            'ecosystem_snapshots': []
        }
        
        for _, row in self.dynamics_data.iterrows():
            snapshot = {
                'year': row['year'],
                'ecosystem_state': {
                    metric: row[metric] for metric in self.dynamics_data.columns 
                    if metric != 'year' and pd.notna(row[metric])
                }
            }
            trajectory['ecosystem_snapshots'].append(snapshot)
        
        return trajectory


class MarketEcosystemModel:
    """
    Integrated market ecosystem analysis model that combines network topology,
    competitive positioning, and temporal dynamics analysis.
    """
    
    def __init__(self):
        self.network_analyzer = NetworkTopologyAnalyzer()
        self.positioning_analyzer = CompetitivePositioningAnalyzer()
        self.dynamics_analyzer = EcosystemDynamicsAnalyzer()
        
        self.ecosystem_data = None
        self.analysis_results = {}
        self.ecosystem_metrics = None
    
    def fit(self, data: pd.DataFrame) -> 'MarketEcosystemModel':
        """
        Fit the integrated ecosystem model to the data
        
        Args:
            data: DataFrame with company financial and market data
            
        Returns:
            Self for method chaining
        """
        print("Fitting Market Ecosystem Model...")
        
        # Validate and prepare data
        self.ecosystem_data = self._validate_and_prepare_data(data)
        
        # Fit individual analyzers
        print("  Fitting Network Topology Analyzer...")
        self.network_analyzer.fit(self.ecosystem_data)
        
        print("  Fitting Competitive Positioning Analyzer...")
        self.positioning_analyzer.fit(self.ecosystem_data)
        
        print("  Fitting Ecosystem Dynamics Analyzer...")
        self.dynamics_analyzer.fit(self.ecosystem_data)
        
        print("Market Ecosystem Model fitted successfully.")
        return self
    
    def _validate_and_prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate and prepare data for ecosystem analysis"""
        required_columns = ['company_name', 'year', 'market_category']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Ensure market_category values are standardized
        valid_categories = ['high_share', 'declining', 'lost']
        data_copy = data.copy()
        
        # Standardize market category names
        category_mapping = {
            'high_share_markets': 'high_share',
            'declining_markets': 'declining', 
            'lost_markets': 'lost'
        }
        
        data_copy['market_category'] = data_copy['market_category'].map(
            category_mapping).fillna(data_copy['market_category'])
        
        return data_copy
    
    def analyze(self) -> Dict:
        """
        Perform comprehensive ecosystem analysis
        
        Returns:
            Dictionary containing all analysis results
        """
        if self.ecosystem_data is None:
            raise ValueError("Must fit the model first")
        
        print("Performing comprehensive ecosystem analysis...")
        
        # Perform individual analyses
        print("  Analyzing network topology...")
        network_results = self.network_analyzer.analyze()
        
        print("  Analyzing competitive positioning...")
        positioning_results = self.positioning_analyzer.analyze()
        
        print("  Analyzing ecosystem dynamics...")
        dynamics_results = self.dynamics_analyzer.analyze()
        
        # Calculate integrated ecosystem metrics
        print("  Calculating integrated ecosystem metrics...")
        self.ecosystem_metrics = self._calculate_ecosystem_metrics(
            network_results, positioning_results, dynamics_results
        )
        
        # Synthesize insights
        print("  Synthesizing ecosystem insights...")
        ecosystem_insights = self._synthesize_insights()
        
        self.analysis_results = {
            'network_analysis': network_results,
            'positioning_analysis': positioning_results,
            'dynamics_analysis': dynamics_results,
            'ecosystem_metrics': self.ecosystem_metrics,
            'ecosystem_insights': ecosystem_insights,
            'data_summary': self._generate_data_summary()
        }
        
        print("Ecosystem analysis completed successfully.")
        return self.analysis_results
    
    def _calculate_ecosystem_metrics(self, network_results: Dict, 
                                    positioning_results: Dict, 
                                    dynamics_results: Dict) -> EcosystemMetrics:
        """Calculate integrated ecosystem metrics"""
        
        # Market concentration from dynamics analysis
        market_concentration = dynamics_results.get('ecosystem_trajectory', {}).get(
            'ecosystem_snapshots', [{}])[-1].get('ecosystem_state', {}).get(
            'market_concentration', 0)
        
        # Competitive intensity from network density and positioning
        network_density = network_results.get('network_metrics', {}).get('network_density', 0)
        positioning_diversity = len(positioning_results.get('competitive_groups', {}))
        competitive_intensity = (network_density + positioning_diversity / 10) / 2
        
        # Innovation density from dynamics
        innovation_density = dynamics_results.get('ecosystem_trajectory', {}).get(
            'ecosystem_snapshots', [{}])[-1].get('ecosystem_state', {}).get(
            'innovation_density', 0)
        
        # Lifecycle diversity from dynamics
        lifecycle_diversity = dynamics_results.get('ecosystem_trajectory', {}).get(
            'ecosystem_snapshots', [{}])[-1].get('ecosystem_state', {}).get(
            'lifecycle_diversity', 0)
        
        # Survival resilience from stability metrics
        stability_metrics = dynamics_results.get('stability_metrics', {})
        avg_stability = np.mean([
            metrics.get('stability_index', 0) 
            for metrics in stability_metrics.values()
        ]) if stability_metrics else 0
        survival_resilience = avg_stability
        
        # Emergence vitality from growth dynamics
        growth_rate = dynamics_results.get('ecosystem_trajectory', {}).get(
            'ecosystem_snapshots', [{}])[-1].get('ecosystem_state', {}).get(
            'avg_growth_rate', 0)
        emergence_vitality = max(0, growth_rate / 100)  # Normalize to 0-1 scale
        
        return EcosystemMetrics(
            market_concentration=market_concentration,
            competitive_intensity=competitive_intensity,
            innovation_density=innovation_density,
            lifecycle_diversity=lifecycle_diversity,
            survival_resilience=survival_resilience,
            emergence_vitality=emergence_vitality
        )
    
    def _synthesize_insights(self) -> Dict:
        """Synthesize key insights from the ecosystem analysis"""
        insights = {
            'ecosystem_health': self._assess_ecosystem_health(),
            'competitive_landscape': self._analyze_competitive_landscape(),
            'market_evolution': self._analyze_market_evolution(),
            'strategic_recommendations': self._generate_strategic_recommendations(),
            'risk_assessment': self._assess_ecosystem_risks()
        }
        
        return insights
    
    def _assess_ecosystem_health(self) -> Dict:
        """Assess overall ecosystem health"""
        if self.ecosystem_metrics is None:
            return {'status': 'unknown', 'score': 0}
        
        # Calculate composite health score
        health_components = {
            'innovation_vitality': min(1.0, self.ecosystem_metrics.innovation_density * 10),
            'competitive_balance': min(1.0, self.ecosystem_metrics.competitive_intensity),
            'market_stability': self.ecosystem_metrics.survival_resilience,
            'growth_momentum': min(1.0, self.ecosystem_metrics.emergence_vitality * 2)
        }
        
        # Weighted health score
        weights = {'innovation_vitality': 0.3, 'competitive_balance': 0.25,
                    'market_stability': 0.25, 'growth_momentum': 0.2}
        
        health_score = sum(health_components[component] * weights[component] 
                            for component in health_components)
        
        # Determine health status
        if health_score >= 0.8:
            status = 'excellent'
        elif health_score >= 0.6:
            status = 'good'
        elif health_score >= 0.4:
            status = 'moderate'
        elif health_score >= 0.2:
            status = 'poor'
        else:
            status = 'critical'
        
        return {
            'status': status,
            'score': health_score,
            'components': health_components,
            'key_strengths': self._identify_health_strengths(health_components),
            'key_weaknesses': self._identify_health_weaknesses(health_components)
        }
    
    def _identify_health_strengths(self, components: Dict) -> List[str]:
        """Identify ecosystem health strengths"""
        strengths = []
        for component, score in components.items():
            if score >= 0.7:
                strengths.append(component.replace('_', ' ').title())
        return strengths
    
    def _identify_health_weaknesses(self, components: Dict) -> List[str]:
        """Identify ecosystem health weaknesses"""
        weaknesses = []
        for component, score in components.items():
            if score <= 0.3:
                weaknesses.append(component.replace('_', ' ').title())
        return weaknesses
    
    def _analyze_competitive_landscape(self) -> Dict:
        """Analyze the competitive landscape"""
        positioning_results = self.analysis_results.get('positioning_analysis', {})
        network_results = self.analysis_results.get('network_analysis', {})
        
        competitive_groups = positioning_results.get('competitive_groups', {})
        market_leadership = positioning_results.get('market_leadership', {})
        centrality_metrics = network_results.get('centrality_metrics', {})
        
        # Identify market leaders
        leaders = []
        if centrality_metrics.get('eigenvector_centrality'):
            top_companies = sorted(centrality_metrics['eigenvector_centrality'].items(),
                                    key=lambda x: x[1], reverse=True)[:5]
            leaders = [company for company, _ in top_companies]
        
        # Analyze competitive intensity by market category
        category_analysis = {}
        for category, leadership_info in market_leadership.items():
            category_analysis[category] = {
                'market_leaders': leadership_info.get('top_companies', []),
                'avg_financial_strength': leadership_info.get('leadership_metrics', {}).get(
                    'avg_financial_strength', 0),
                'competitive_pressure': self._calculate_competitive_pressure(category)
            }
        
        return {
            'market_leaders': leaders,
            'competitive_groups': self._summarize_competitive_groups(competitive_groups),
            'category_analysis': category_analysis,
            'market_concentration_level': self._assess_concentration_level(),
            'entry_barriers': self._assess_entry_barriers()
        }
    
    def _calculate_competitive_pressure(self, category: str) -> str:
        """Calculate competitive pressure for a market category"""
        # This is a simplified heuristic based on market category
        if category == 'high_share':
            return 'moderate'  # Established players, stable competition
        elif category == 'declining':
            return 'high'  # Intense competition as market shrinks
        elif category == 'lost':
            return 'low'  # Limited remaining competition
        else:
            return 'unknown'
    
    def _summarize_competitive_groups(self, competitive_groups: Dict) -> Dict:
        """Summarize competitive group information"""
        summary = {}
        for group_id, group_info in competitive_groups.items():
            summary[group_id] = {
                'size': group_info.get('group_size', 0),
                'dominant_category': group_info.get('dominant_category', 'unknown'),
                'representative_companies': group_info.get('companies', [])[:3]  # Top 3
            }
        return summary
    
    def _assess_concentration_level(self) -> str:
        """Assess market concentration level"""
        concentration = self.ecosystem_metrics.market_concentration if self.ecosystem_metrics else 0
        
        if concentration >= 0.8:
            return 'highly_concentrated'
        elif concentration >= 0.6:
            return 'moderately_concentrated'
        elif concentration >= 0.3:
            return 'low_concentration'
        else:
            return 'fragmented'
    
    def _assess_entry_barriers(self) -> Dict:
        """Assess market entry barriers"""
        # Based on ecosystem characteristics
        innovation_intensity = (self.ecosystem_metrics.innovation_density 
                                if self.ecosystem_metrics else 0)
        financial_barriers = self.ecosystem_metrics.market_concentration if self.ecosystem_metrics else 0
        
        barriers = {
            'innovation_barriers': 'high' if innovation_intensity > 0.05 else 'moderate' if innovation_intensity > 0.02 else 'low',
            'financial_barriers': 'high' if financial_barriers > 0.6 else 'moderate' if financial_barriers > 0.3 else 'low',
            'overall_assessment': 'high' if (innovation_intensity > 0.05 or financial_barriers > 0.6) else 'moderate'
        }
        
        return barriers
    
    def _analyze_market_evolution(self) -> Dict:
        """Analyze market evolution patterns"""
        dynamics_results = self.analysis_results.get('dynamics_analysis', {})
        evolution_analysis = dynamics_results.get('evolution_analysis', {})
        regime_changes = dynamics_results.get('regime_changes', [])
        
        # Identify key trends
        key_trends = {}
        for metric, analysis in evolution_analysis.items():
            if analysis.get('trend_strength', 0) > 0.5:  # Significant trend
                key_trends[metric] = {
                    'direction': analysis.get('direction', 'stable'),
                    'strength': analysis.get('trend_strength', 0),
                    'significance': analysis.get('trend_significance', 1)
                }
        
        # Analyze evolution phases
        evolution_phases = self._identify_evolution_phases(evolution_analysis)
        
        return {
            'key_trends': key_trends,
            'regime_changes': regime_changes,
            'evolution_phases': evolution_phases,
            'current_phase': self._determine_current_phase(evolution_analysis),
            'future_trajectory': self._predict_future_trajectory(evolution_analysis)
        }
    
    def _identify_evolution_phases(self, evolution_analysis: Dict) -> List[Dict]:
        """Identify distinct evolution phases"""
        phases = []
        
        # Simple heuristic based on growth and innovation trends
        growth_trend = evolution_analysis.get('avg_growth_rate', {}).get('direction', 'stable')
        innovation_trend = evolution_analysis.get('innovation_density', {}).get('direction', 'stable')
        
        if growth_trend == 'increasing' and innovation_trend == 'increasing':
            phases.append({'phase': 'growth_and_innovation', 'period': 'recent'})
        elif growth_trend == 'decreasing' and innovation_trend == 'decreasing':
            phases.append({'phase': 'decline_and_stagnation', 'period': 'recent'})
        elif growth_trend == 'increasing' and innovation_trend == 'decreasing':
            phases.append({'phase': 'mature_growth', 'period': 'recent'})
        elif growth_trend == 'decreasing' and innovation_trend == 'increasing':
            phases.append({'phase': 'restructuring', 'period': 'recent'})
        else:
            phases.append({'phase': 'stable_maturity', 'period': 'recent'})
        
        return phases
    
    def _determine_current_phase(self, evolution_analysis: Dict) -> str:
        """Determine current ecosystem phase"""
        phases = self._identify_evolution_phases(evolution_analysis)
        return phases[0]['phase'] if phases else 'unknown'
    
    def _predict_future_trajectory(self, evolution_analysis: Dict) -> Dict:
        """Predict future ecosystem trajectory"""
        # Simple prediction based on current trends
        predictions = {}
        
        for metric, analysis in evolution_analysis.items():
            trend_strength = analysis.get('trend_strength', 0)
            direction = analysis.get('direction', 'stable')
            
            if trend_strength > 0.7:  # Strong trend
                confidence = 'high'
            elif trend_strength > 0.4:
                confidence = 'medium'
            else:
                confidence = 'low'
            
            predictions[metric] = {
                'predicted_direction': direction,
                'confidence': confidence,
                'trend_strength': trend_strength
            }
        
        return predictions
    
    def _generate_strategic_recommendations(self) -> Dict:
        """Generate strategic recommendations based on ecosystem analysis"""
        recommendations = {
            'for_market_leaders': [],
            'for_challengers': [],
            'for_new_entrants': [],
            'for_policymakers': []
        }
        
        health_status = self._assess_ecosystem_health().get('status', 'unknown')
        concentration_level = self._assess_concentration_level()
        current_phase = self._determine_current_phase(
            self.analysis_results.get('dynamics_analysis', {}).get('evolution_analysis', {})
        )
        
        # Recommendations for market leaders
        if concentration_level in ['highly_concentrated', 'moderately_concentrated']:
            recommendations['for_market_leaders'].extend([
                'Maintain market position through continued innovation',
                'Monitor emerging competitors and potential disruptors',
                'Consider strategic partnerships to strengthen ecosystem position'
            ])
        
        if health_status in ['poor', 'critical']:
            recommendations['for_market_leaders'].extend([
                'Focus on operational efficiency and cost optimization',
                'Explore new market opportunities and diversification',
                'Consider market consolidation strategies'
            ])
        
        # Recommendations for challengers
        if current_phase in ['growth_and_innovation', 'restructuring']:
            recommendations['for_challengers'].extend([
                'Leverage innovation to differentiate from leaders',
                'Focus on underserved market segments',
                'Build strategic alliances for market access'
            ])
        
        if concentration_level == 'fragmented':
            recommendations['for_challengers'].extend([
                'Pursue aggressive growth strategies',
                'Consider acquisition opportunities',
                'Build scale through operational excellence'
            ])
        
        # Recommendations for new entrants
        entry_barriers = self._assess_entry_barriers()
        if entry_barriers.get('overall_assessment') == 'low':
            recommendations['for_new_entrants'].extend([
                'Enter market with innovative business models',
                'Focus on niche segments initially',
                'Build partnerships for market access'
            ])
        else:
            recommendations['for_new_entrants'].extend([
                'Develop strong technological differentiation',
                'Secure substantial funding for market entry',
                'Consider entering through acquisitions'
            ])
        
        # Recommendations for policymakers
        if health_status in ['poor', 'critical']:
            recommendations['for_policymakers'].extend([
                'Consider industry support programs',
                'Review regulatory framework for competitiveness',
                'Promote innovation through R&D incentives'
            ])
        
        if concentration_level == 'highly_concentrated':
            recommendations['for_policymakers'].extend([
                'Monitor market competition and antitrust issues',
                'Support new entrant development',
                'Ensure fair market access conditions'
            ])
        
        return recommendations
    
    def _assess_ecosystem_risks(self) -> Dict:
        """Assess key ecosystem risks"""
        risks = {
            'high_risks': [],
            'medium_risks': [],
            'low_risks': []
        }
        
        # Assess various risk categories
        concentration_level = self._assess_concentration_level()
        health_status = self._assess_ecosystem_health().get('status', 'unknown')
        stability_metrics = self.analysis_results.get('dynamics_analysis', {}).get('stability_metrics', {})
        
        # Market concentration risk
        if concentration_level == 'highly_concentrated':
            risks['high_risks'].append({
                'risk': 'Market concentration risk',
                'description': 'High market concentration may lead to reduced competition and innovation'
            })
        elif concentration_level == 'moderately_concentrated':
            risks['medium_risks'].append({
                'risk': 'Moderate concentration risk',
                'description': 'Some concentration present but competitive dynamics remain'
            })
        
        # Ecosystem health risk
        if health_status in ['critical', 'poor']:
            risks['high_risks'].append({
                'risk': 'Ecosystem deterioration',
                'description': 'Overall ecosystem health is declining, requiring intervention'
            })
        elif health_status == 'moderate':
            risks['medium_risks'].append({
                'risk': 'Ecosystem stability concerns',
                'description': 'Ecosystem showing signs of stress but manageable'
            })
        
        # Innovation stagnation risk
        innovation_density = (self.ecosystem_metrics.innovation_density 
                            if self.ecosystem_metrics else 0)
        if innovation_density < 0.02:
            risks['high_risks'].append({
                'risk': 'Innovation stagnation',
                'description': 'Low innovation levels may lead to competitive disadvantage'
            })
        elif innovation_density < 0.05:
            risks['medium_risks'].append({
                'risk': 'Innovation concerns',
                'description': 'Innovation levels below optimal for competitive sustainability'
            })
        
        # Market volatility risk
        avg_stability = np.mean([
            metrics.get('stability_index', 1) 
            for metrics in stability_metrics.values()
        ]) if stability_metrics else 1
        
        if avg_stability < 0.3:
            risks['high_risks'].append({
                'risk': 'High market volatility',
                'description': 'Market showing high instability and unpredictable dynamics'
            })
        elif avg_stability < 0.6:
            risks['medium_risks'].append({
                'risk': 'Market uncertainty',
                'description': 'Some market volatility present, requiring careful monitoring'
            })
        else:
            risks['low_risks'].append({
                'risk': 'Market stability',
                'description': 'Market showing stable and predictable patterns'
            })
        
        return risks
    
    def _generate_data_summary(self) -> Dict:
        """Generate summary of data used in analysis"""
        if self.ecosystem_data is None:
            return {}
        
        summary = {
            'total_companies': len(self.ecosystem_data['company_name'].unique()),
            'time_period': {
                'start_year': int(self.ecosystem_data['year'].min()),
                'end_year': int(self.ecosystem_data['year'].max()),
                'total_years': int(self.ecosystem_data['year'].max() - self.ecosystem_data['year'].min() + 1)
            },
            'market_categories': self.ecosystem_data['market_category'].value_counts().to_dict(),
            'data_completeness': self._calculate_data_completeness(),
            'key_metrics_available': self._identify_available_metrics()
        }
        
        return summary
    
    def _calculate_data_completeness(self) -> Dict:
        """Calculate data completeness metrics"""
        total_cells = len(self.ecosystem_data) * len(self.ecosystem_data.columns)
        missing_cells = self.ecosystem_data.isnull().sum().sum()
        
        return {
            'overall_completeness': 1 - (missing_cells / total_cells),
            'missing_data_by_column': (self.ecosystem_data.isnull().sum() / len(self.ecosystem_data)).to_dict()
        }
    
    def _identify_available_metrics(self) -> List[str]:
        """Identify available financial metrics in the dataset"""
        financial_metrics = [
            'sales', 'operating_profit_margin', 'roe', 'total_asset_turnover',
            'rd_expense_ratio', 'sales_growth_rate', 'total_assets', 'employee_count'
        ]
        
        return [metric for metric in financial_metrics if metric in self.ecosystem_data.columns]
    
    def get_ecosystem_summary(self) -> Dict:
        """Get a concise ecosystem summary"""
        if not self.analysis_results:
            raise ValueError("Must run analyze() first")
        
        ecosystem_insights = self.analysis_results.get('ecosystem_insights', {})
        
        summary = {
            'ecosystem_health': ecosystem_insights.get('ecosystem_health', {}),
            'current_phase': ecosystem_insights.get('market_evolution', {}).get('current_phase', 'unknown'),
            'key_leaders': ecosystem_insights.get('competitive_landscape', {}).get('market_leaders', [])[:3],
            'primary_risks': [risk['risk'] for risk in ecosystem_insights.get('risk_assessment', {}).get('high_risks', [])],
            'top_recommendations': {
                'leaders': ecosystem_insights.get('strategic_recommendations', {}).get('for_market_leaders', [])[:2],
                'challengers': ecosystem_insights.get('strategic_recommendations', {}).get('for_challengers', [])[:2]
            }
        }
        
        return summary
    
    def export_results(self, filepath: str, format: str = 'json') -> None:
        """
        Export analysis results to file
        
        Args:
            filepath: Path to save the results
            format: Export format ('json', 'pickle')
        """
        if not self.analysis_results:
            raise ValueError("Must run analyze() first")
        
        if format == 'json':
            import json
            # Convert numpy types to native Python types for JSON serialization
            serializable_results = self._make_json_serializable(self.analysis_results)
            with open(filepath, 'w') as f:
                json.dump(serializable_results, f, indent=2)
        elif format == 'pickle':
            import pickle
            with open(filepath, 'wb') as f:
                pickle.dump(self.analysis_results, f)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _make_json_serializable(self, obj):
        """Convert numpy types to JSON-serializable types"""
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict()
        else:
            return obj


# Example usage and testing functions
def create_sample_ecosystem_data() -> pd.DataFrame:
    """Create sample data for testing the ecosystem model"""
    np.random.seed(42)
    
    companies = [
        'TechCorp A', 'TechCorp B', 'TechCorp C', 'ManufCorp A', 'ManufCorp B',
        'ServiceCorp A', 'ServiceCorp B', 'StartupCorp A', 'StartupCorp B', 'LegacyCorp A'
    ]
    
    market_categories = ['high_share', 'declining', 'lost']
    market_segments = ['robotics', 'semiconductors', 'consumer_electronics']
    
    data = []
    
    for year in range(2000, 2024):
        for company in companies:
            # Simulate company evolution over time
            age = year - 2000
            
            # Market category evolution
            if 'Startup' in company and age < 10:
                category = 'high_share' if np.random.random() > 0.7 else 'declining'
            elif 'Legacy' in company:
                if age < 15:
                    category = 'declining'
                else:
                    category = 'lost'
            else:
                category = np.random.choice(market_categories, p=[0.4, 0.4, 0.2])
            
            data.append({
                'company_name': company,
                'year': year,
                'market_category': category,
                'market_segment': np.random.choice(market_segments),
                'sales': max(100, 1000 + age * 50 + np.random.normal(0, 200)),
                'total_assets': max(500, 5000 + age * 200 + np.random.normal(0, 1000)),
                'operating_profit_margin': max(0.01, 0.1 + np.random.normal(0, 0.05)),
                'roe': max(0.01, 0.15 + np.random.normal(0, 0.08)),
                'rd_expense_ratio': max(0.005, 0.05 + np.random.normal(0, 0.02)),
                'sales_growth_rate': np.random.normal(0.05, 0.15),
                'total_asset_turnover': max(0.1, 1.2 + np.random.normal(0, 0.3)),
                'employee_count': max(10, 500 + age * 20 + np.random.normal(0, 100)),
                'market_share': max(0.001, np.random.exponential(0.05)),
                'overseas_sales_ratio': np.random.uniform(0, 0.8),
                'company_age': age
            })
    
    return pd.DataFrame(data)


def main():
    """Main function for testing the MarketEcosystemModel"""
    print("Testing Market Ecosystem Model...")
    
    # Create sample data
    sample_data = create_sample_ecosystem_data()
    print(f"Created sample data with {len(sample_data)} records")
    
    # Initialize and fit the model
    ecosystem_model = MarketEcosystemModel()
    ecosystem_model.fit(sample_data)
    
    # Perform analysis
    results = ecosystem_model.analyze()
    
    # Display summary results
    print("\n" + "="*50)
    print("ECOSYSTEM ANALYSIS SUMMARY")
    print("="*50)
    
    summary = ecosystem_model.get_ecosystem_summary()
    
    print(f"\nEcosystem Health: {summary['ecosystem_health']['status'].title()}")
    print(f"Health Score: {summary['ecosystem_health']['score']:.2f}")
    print(f"Current Phase: {summary['current_phase'].replace('_', ' ').title()}")
    
    print(f"\nTop Market Leaders:")
    for i, leader in enumerate(summary['key_leaders'], 1):
        print(f"  {i}. {leader}")
    
    print(f"\nPrimary Risks:")
    for risk in summary['primary_risks']:
        print(f"  • {risk}")
    
    print(f"\nKey Recommendations for Leaders:")
    for rec in summary['top_recommendations']['leaders']:
        print(f"  • {rec}")
    
    # Export results
    try:
        ecosystem_model.export_results('ecosystem_analysis_results.json', 'json')
        print(f"\nResults exported to 'ecosystem_analysis_results.json'")
    except Exception as e:
        print(f"\nError exporting results: {e}")
    
    print("\nMarket Ecosystem Model testing completed successfully!")
    
    return ecosystem_model, results


if __name__ == "__main__":
    model, results = main()