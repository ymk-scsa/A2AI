"""
A2AI (Advanced Financial Analysis AI) - Visualization Module

This module provides comprehensive visualization capabilities for financial statement analysis
including traditional financial metrics, survival analysis, lifecycle analysis, and emergence patterns.

Key Features:
- Traditional financial visualization (factor analysis, market comparison, time series)
- Survival analysis visualization (survival curves, hazard plots, risk heatmaps)
- Lifecycle visualization (trajectories, stage transitions, maturity landscapes)
- Emergence visualization (startup journeys, success factors, market entry timing)
- Integrated visualization (ecosystem networks, competitive landscapes)
- Interactive exploration tools and dashboards

Author: A2AI Development Team
License: MIT
Version: 1.0.0
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Version and metadata
__version__ = "1.0.0"
__author__ = "A2AI Development Team"
__email__ = "a2ai-dev@example.com"

# Import all visualization modules
try:
    # Traditional visualization modules
    from .traditional_viz.factor_visualizer import FactorVisualizer
    from .traditional_viz.market_visualizer import MarketVisualizer
    from .traditional_viz.time_series_plots import TimeSeriesPlotter
    from .traditional_viz.correlation_heatmaps import CorrelationVisualizer
    from .traditional_viz.performance_dashboards import PerformanceDashboard

    # Survival analysis visualization modules
    from .survival_viz.survival_curves import SurvivalCurveVisualizer
    from .survival_viz.hazard_plots import HazardPlotter
    from .survival_viz.risk_heatmaps import RiskHeatmapVisualizer
    from .survival_viz.extinction_timeline import ExtinctionTimelineVisualizer

    # Lifecycle visualization modules
    from .lifecycle_viz.lifecycle_trajectories import LifecycleTrajectoryVisualizer
    from .lifecycle_viz.stage_transitions import StageTransitionVisualizer
    from .lifecycle_viz.maturity_landscape import MaturityLandscapeVisualizer
    from .lifecycle_viz.evolution_animation import EvolutionAnimator

    # Emergence visualization modules
    from .emergence_viz.startup_journey import StartupJourneyVisualizer
    from .emergence_viz.success_factors import SuccessFactorVisualizer
    from .emergence_viz.market_entry_timing import MarketEntryTimingVisualizer
    from .emergence_viz.innovation_diffusion import InnovationDiffusionVisualizer

    # Integrated visualization modules
    from .integrated_viz.ecosystem_networks import EcosystemNetworkVisualizer
    from .integrated_viz.competitive_landscape import CompetitiveLandscapeVisualizer
    from .integrated_viz.strategic_positioning import StrategicPositioningVisualizer
    from .integrated_viz.interactive_explorer import InteractiveExplorer

except ImportError as e:
    warnings.warn(f"Some visualization modules could not be imported: {e}")
    # Define placeholder classes for missing modules
    class PlaceholderVisualizer:
        def __init__(self, *args, **kwargs):
            pass
        def __getattr__(self, name):
            raise NotImplementedError(f"Visualization module not available: {name}")

# Market categories as defined in the project
MARKET_CATEGORIES = {
    'high_share': {
        'name': 'High Share Markets',
        'description': 'Markets where Japanese companies maintain high global market share',
        'markets': ['robots', 'endoscopy', 'machine_tools', 'electronic_materials', 'precision_instruments'],
        'color': '#2E8B57'  # Sea Green
    },
    'declining': {
        'name': 'Declining Share Markets', 
        'description': 'Markets where Japanese companies are experiencing share decline',
        'markets': ['automotive', 'steel', 'smart_appliances', 'batteries', 'pc_peripherals'],
        'color': '#FF8C00'  # Dark Orange
    },
    'lost': {
        'name': 'Lost Share Markets',
        'description': 'Markets where Japanese companies have completely lost their share',
        'markets': ['home_appliances', 'semiconductors', 'smartphones', 'personal_computers', 'telecom_equipment'],
        'color': '#DC143C'  # Crimson
    }
}

# Evaluation metrics as defined in the project (9 metrics)
EVALUATION_METRICS = {
    'traditional': [
        'revenue', 'revenue_growth_rate', 'operating_margin', 
        'net_margin', 'roe', 'value_added_ratio'
    ],
    'extended': [
        'survival_probability', 'emergence_success_rate', 'succession_success_rate'
    ]
}

# Color palettes for different analysis types
COLOR_PALETTES = {
    'market_category': [MARKET_CATEGORIES[cat]['color'] for cat in MARKET_CATEGORIES.keys()],
    'lifecycle_stages': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
    'survival_risk': ['#2E8B57', '#FFD700', '#FF8C00', '#FF6347', '#DC143C'],
    'performance': ['#0066CC', '#FF6600', '#00AA44', '#AA0044', '#6600CC'],
    'default': px.colors.qualitative.Set3
}

class A2AIVisualizationManager:
    """
    Central manager for all A2AI visualization capabilities.
    
    This class provides a unified interface to access all visualization modules
    and contains utility methods for consistent styling and theming across
    all visualizations.
    """
    
    def __init__(self, theme: str = 'modern', figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize the A2AI Visualization Manager.
        
        Parameters:
        -----------
        theme : str
            Theme for visualizations ('modern', 'classic', 'dark', 'academic')
        figsize : tuple
            Default figure size for matplotlib plots
        """
        self.theme = theme
        self.figsize = figsize
        self.market_categories = MARKET_CATEGORIES
        self.evaluation_metrics = EVALUATION_METRICS
        self.color_palettes = COLOR_PALETTES
        
        # Set up visualization theme
        self._setup_theme()
        
        # Initialize visualization modules
        self._initialize_modules()
    
    def _setup_theme(self):
        """Set up visualization theme and styling."""
        if self.theme == 'modern':
            plt.style.use('seaborn-v0_8-whitegrid')
            sns.set_palette("husl")
        elif self.theme == 'classic':
            plt.style.use('classic')
        elif self.theme == 'dark':
            plt.style.use('dark_background')
        elif self.theme == 'academic':
            plt.style.use('seaborn-v0_8-paper')
            sns.set_palette("colorblind")
        
        # Set default figure parameters
        plt.rcParams['figure.figsize'] = self.figsize
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['legend.fontsize'] = 10
        plt.rcParams['figure.titlesize'] = 16
    
    def _initialize_modules(self):
        """Initialize all visualization modules."""
        try:
            # Traditional visualization
            self.factor_viz = FactorVisualizer()
            self.market_viz = MarketVisualizer()
            self.timeseries_viz = TimeSeriesPlotter()
            self.correlation_viz = CorrelationVisualizer()
            self.performance_dashboard = PerformanceDashboard()
            
            # Survival analysis visualization
            self.survival_curves = SurvivalCurveVisualizer()
            self.hazard_plots = HazardPlotter()
            self.risk_heatmaps = RiskHeatmapVisualizer()
            self.extinction_timeline = ExtinctionTimelineVisualizer()
            
            # Lifecycle visualization
            self.lifecycle_trajectories = LifecycleTrajectoryVisualizer()
            self.stage_transitions = StageTransitionVisualizer()
            self.maturity_landscape = MaturityLandscapeVisualizer()
            self.evolution_animator = EvolutionAnimator()
            
            # Emergence visualization
            self.startup_journey = StartupJourneyVisualizer()
            self.success_factors = SuccessFactorVisualizer()
            self.market_entry_timing = MarketEntryTimingVisualizer()
            self.innovation_diffusion = InnovationDiffusionVisualizer()
            
            # Integrated visualization
            self.ecosystem_networks = EcosystemNetworkVisualizer()
            self.competitive_landscape = CompetitiveLandscapeVisualizer()
            self.strategic_positioning = StrategicPositioningVisualizer()
            self.interactive_explorer = InteractiveExplorer()
            
            self.modules_initialized = True
            
        except Exception as e:
            warnings.warn(f"Could not initialize all visualization modules: {e}")
            self.modules_initialized = False
    
    def get_market_color(self, market_category: str) -> str:
        """Get color for market category."""
        return self.market_categories.get(market_category, {}).get('color', '#808080')
    
    def get_palette(self, palette_type: str) -> List[str]:
        """Get color palette for visualization type."""
        return self.color_palettes.get(palette_type, self.color_palettes['default'])
    
    def create_comprehensive_dashboard(self, data: pd.DataFrame, company_name: str = None) -> go.Figure:
        """
        Create a comprehensive dashboard combining multiple visualization types.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Financial data for visualization
        company_name : str, optional
            Specific company to highlight
            
        Returns:
        --------
        go.Figure
            Plotly figure with comprehensive dashboard
        """
        if not self.modules_initialized:
            raise RuntimeError("Visualization modules not properly initialized")
        
        # Create subplot layout
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=[
                'Performance Overview', 'Survival Probability', 'Market Position',
                'Factor Impact', 'Lifecycle Stage', 'Risk Assessment',
                'Time Series Trends', 'Competitive Landscape', 'Strategic Position'
            ],
            specs=[
                [{"type": "scatter"}, {"type": "scatter"}, {"type": "bar"}],
                [{"type": "heatmap"}, {"type": "scatter"}, {"type": "indicator"}],
                [{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}]
            ]
        )
        
        # Add plots to each subplot (placeholder logic)
        # This would be implemented based on actual data structure
        
        fig.update_layout(
            title=f"A2AI Comprehensive Analysis Dashboard{' - ' + company_name if company_name else ''}",
            height=900,
            showlegend=True,
            template=self.theme
        )
        
        return fig
    
    def export_all_visualizations(self, data: pd.DataFrame, output_dir: str):
        """
        Export all available visualizations to specified directory.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Financial data for visualization
        output_dir : str
            Output directory path
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        if not self.modules_initialized:
            raise RuntimeError("Visualization modules not properly initialized")
        
        # Export each visualization type
        export_methods = [
            ('factor_analysis', self.factor_viz),
            ('market_comparison', self.market_viz),
            ('time_series', self.timeseries_viz),
            ('correlation_analysis', self.correlation_viz),
            ('survival_curves', self.survival_curves),
            ('hazard_analysis', self.hazard_plots),
            ('risk_heatmaps', self.risk_heatmaps),
            ('lifecycle_trajectories', self.lifecycle_trajectories),
            ('maturity_landscape', self.maturity_landscape),
            ('startup_journey', self.startup_journey),
            ('ecosystem_networks', self.ecosystem_networks),
            ('competitive_landscape', self.competitive_landscape)
        ]
        
        for name, module in export_methods:
            try:
                if hasattr(module, 'export'):
                    module.export(data, os.path.join(output_dir, f"{name}.png"))
                    print(f"Exported {name} visualization")
            except Exception as e:
                warnings.warn(f"Could not export {name}: {e}")

def get_visualization_info() -> Dict[str, Any]:
    """
    Get information about available visualization capabilities.
    
    Returns:
    --------
    dict
        Dictionary containing visualization module information
    """
    return {
        'version': __version__,
        'author': __author__,
        'market_categories': MARKET_CATEGORIES,
        'evaluation_metrics': EVALUATION_METRICS,
        'available_modules': [
            'traditional_viz', 'survival_viz', 'lifecycle_viz', 
            'emergence_viz', 'integrated_viz'
        ],
        'supported_formats': ['png', 'pdf', 'svg', 'html', 'json'],
        'color_palettes': list(COLOR_PALETTES.keys())
    }

def create_quick_visualization(data: pd.DataFrame, viz_type: str = 'overview') -> go.Figure:
    """
    Create a quick visualization for data exploration.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Financial data
    viz_type : str
        Type of visualization ('overview', 'survival', 'comparison', 'trends')
        
    Returns:
    --------
    go.Figure
        Plotly figure
    """
    manager = A2AIVisualizationManager()
    
    if viz_type == 'overview':
        return manager.create_comprehensive_dashboard(data)
    elif viz_type == 'survival' and manager.modules_initialized:
        return manager.survival_curves.plot_survival_overview(data)
    elif viz_type == 'comparison' and manager.modules_initialized:
        return manager.market_viz.plot_market_comparison(data)
    elif viz_type == 'trends' and manager.modules_initialized:
        return manager.timeseries_viz.plot_trend_overview(data)
    else:
        raise ValueError(f"Unsupported visualization type: {viz_type}")

# Export main classes and functions
__all__ = [
    'A2AIVisualizationManager',
    'get_visualization_info',
    'create_quick_visualization',
    'MARKET_CATEGORIES',
    'EVALUATION_METRICS',
    'COLOR_PALETTES',
    # Traditional viz modules
    'FactorVisualizer',
    'MarketVisualizer', 
    'TimeSeriesPlotter',
    'CorrelationVisualizer',
    'PerformanceDashboard',
    # Survival viz modules
    'SurvivalCurveVisualizer',
    'HazardPlotter',
    'RiskHeatmapVisualizer',
    'ExtinctionTimelineVisualizer',
    # Lifecycle viz modules
    'LifecycleTrajectoryVisualizer',
    'StageTransitionVisualizer',
    'MaturityLandscapeVisualizer',
    'EvolutionAnimator',
    # Emergence viz modules
    'StartupJourneyVisualizer',
    'SuccessFactorVisualizer',
    'MarketEntryTimingVisualizer',
    'InnovationDiffusionVisualizer',
    # Integrated viz modules
    'EcosystemNetworkVisualizer',
    'CompetitiveLandscapeVisualizer',
    'StrategicPositioningVisualizer',
    'InteractiveExplorer'
]

# Initialize default visualization manager instance
default_viz_manager = A2AIVisualizationManager()

if __name__ == "__main__":
    # Example usage and testing
    print("A2AI Visualization Module")
    print(f"Version: {__version__}")
    print(f"Author: {__author__}")
    print("\nAvailable visualization capabilities:")
    
    info = get_visualization_info()
    for key, value in info.items():
        if key != 'market_categories' and key != 'evaluation_metrics':
            print(f"  {key}: {value}")
    
    print(f"\nModules initialized: {default_viz_manager.modules_initialized}")
    print("Ready for financial data visualization and analysis!")