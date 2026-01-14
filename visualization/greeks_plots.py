"""
Visualization Module for Greeks and Option Analysis
Generates charts for slide deck and reports
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Tuple
import os

from utils.config import config


class GreeksVisualizer:
    """
    Create visualizations for Greeks analysis
    """
    
    def __init__(self, output_dir: str = None):
        """
        Initialize visualizer
        
        Parameters:
        -----------
        output_dir : str, optional
            Directory to save charts (default from config)
        """
        self.output_dir = output_dir or config.CHART_OUTPUT_DIR
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
    
    def plot_greeks_vs_spot(self, greeks_df: pd.DataFrame, 
                           current_spot: float, save: bool = True) -> str:
        """
        Plot Greeks vs spot price
        
        Parameters:
        -----------
        greeks_df : pd.DataFrame
            DataFrame with columns: spot_price, delta, gamma, theta_per_day, vega_percent
        current_spot : float
            Current spot price (for vertical line)
        save : bool
            Save to file
        
        Returns:
        --------
        str : File path if saved
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Option Greeks vs Spot Price', fontsize=16, fontweight='bold')
        
        # Delta
        axes[0, 0].plot(greeks_df['spot_price'], greeks_df['delta'], linewidth=2, color='blue')
        axes[0, 0].axvline(current_spot, color='red', linestyle='--', alpha=0.7, label='Current Spot')
        axes[0, 0].set_xlabel('Spot Price ($)')
        axes[0, 0].set_ylabel('Delta')
        axes[0, 0].set_title('Delta vs Spot Price')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Gamma
        axes[0, 1].plot(greeks_df['spot_price'], greeks_df['gamma'], linewidth=2, color='green')
        axes[0, 1].axvline(current_spot, color='red', linestyle='--', alpha=0.7, label='Current Spot')
        axes[0, 1].set_xlabel('Spot Price ($)')
        axes[0, 1].set_ylabel('Gamma')
        axes[0, 1].set_title('Gamma vs Spot Price')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Theta (per day)
        axes[1, 0].plot(greeks_df['spot_price'], greeks_df['theta_per_day'], 
                       linewidth=2, color='orange')
        axes[1, 0].axvline(current_spot, color='red', linestyle='--', alpha=0.7, label='Current Spot')
        axes[1, 0].set_xlabel('Spot Price ($)')
        axes[1, 0].set_ylabel('Theta ($/day)')
        axes[1, 0].set_title('Theta vs Spot Price (Daily Decay)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Vega (per 1% vol)
        axes[1, 1].plot(greeks_df['spot_price'], greeks_df['vega_percent'], 
                       linewidth=2, color='purple')
        axes[1, 1].axvline(current_spot, color='red', linestyle='--', alpha=0.7, label='Current Spot')
        axes[1, 1].set_xlabel('Spot Price ($)')
        axes[1, 1].set_ylabel('Vega ($/1% vol)')
        axes[1, 1].set_title('Vega vs Spot Price')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, 'greeks_vs_spot.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {filepath}")
            plt.close()
            return filepath
        else:
            plt.show()
            return None
    
    def plot_greeks_vs_time(self, greeks_df: pd.DataFrame, save: bool = True) -> str:
        """Plot Greeks vs time to expiry"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Option Greeks vs Time to Expiry', fontsize=16, fontweight='bold')
        
        # Delta
        axes[0, 0].plot(greeks_df['days_to_expiry'], greeks_df['delta'], linewidth=2, color='blue')
        axes[0, 0].set_xlabel('Days to Expiry')
        axes[0, 0].set_ylabel('Delta')
        axes[0, 0].set_title('Delta Decay Over Time')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].invert_xaxis()  # Time flows right to left
        
        # Gamma
        axes[0, 1].plot(greeks_df['days_to_expiry'], greeks_df['gamma'], linewidth=2, color='green')
        axes[0, 1].set_xlabel('Days to Expiry')
        axes[0, 1].set_ylabel('Gamma')
        axes[0, 1].set_title('Gamma Evolution')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].invert_xaxis()
        
        # Theta
        axes[1, 0].plot(greeks_df['days_to_expiry'], greeks_df['theta_per_day'], 
                       linewidth=2, color='orange')
        axes[1, 0].set_xlabel('Days to Expiry')
        axes[1, 0].set_ylabel('Theta ($/day)')
        axes[1, 0].set_title('Time Decay Acceleration')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].invert_xaxis()
        
        # Option Price
        axes[1, 1].plot(greeks_df['days_to_expiry'], greeks_df['option_price'], 
                       linewidth=2, color='red')
        axes[1, 1].set_xlabel('Days to Expiry')
        axes[1, 1].set_ylabel('Option Value ($)')
        axes[1, 1].set_title('Option Value Decay')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].invert_xaxis()
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, 'greeks_vs_time.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {filepath}")
            plt.close()
            return filepath
        else:
            plt.show()
            return None
    
    def plot_3d_surface(self, surface_data: Dict, save: bool = True) -> str:
        """
        Create 3D option surface plot
        
        Parameters:
        -----------
        surface_data : dict
            With keys: spot_prices, time_to_maturity, option_values
        save : bool
            Save to file
        
        Returns:
        --------
        str : File path if saved
        """
        # Create interactive Plotly 3D surface
        fig = go.Figure(data=[go.Surface(
            x=surface_data['spot_prices'][0],  # Unique spot prices
            y=surface_data['days_to_expiry'][:, 0],  # Unique days
            z=surface_data['option_values'],
            colorscale='Viridis',
            colorbar=dict(title='Option Value ($)')
        )])
        
        fig.update_layout(
            title='Option Value Surface: V(S, T)',
            scene=dict(
                xaxis_title='Spot Price ($)',
                yaxis_title='Days to Expiry',
                zaxis_title='Option Value ($)',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.3))
            ),
            width=900,
            height=700
        )
        
        if save:
            filepath_html = os.path.join(self.output_dir, 'option_surface_3d.html')
            fig.write_html(filepath_html)
            print(f"✓ Saved: {filepath_html}")
            
            # Also save static PNG
            filepath_png = os.path.join(self.output_dir, 'option_surface_3d.png')
            fig.write_image(filepath_png, width=900, height=700)
            print(f"✓ Saved: {filepath_png}")
            
            return filepath_html
        else:
            fig.show()
            return None
    
    def plot_pnl_distribution(self, pnl_data: np.ndarray, var_95: float, 
                             var_99: float, save: bool = True) -> str:
        """
        Plot P&L distribution from Monte Carlo
        
        Parameters:
        -----------
        pnl_data : np.ndarray
            P&L values
        var_95, var_99 : float
            VaR thresholds
        save : bool
            Save to file
        
        Returns:
        --------
        str : File path if saved
        """
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Histogram
        n, bins, patches = ax.hist(pnl_data, bins=50, alpha=0.7, color='steelblue', 
                                   edgecolor='black', density=True)
        
        # Overlay KDE
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(pnl_data)
        x_range = np.linspace(pnl_data.min(), pnl_data.max(), 200)
        ax.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
        
        # VaR lines
        ax.axvline(var_95, color='orange', linestyle='--', linewidth=2, 
                  label=f'VaR 95%: ${var_95:.0f}')
        ax.axvline(var_99, color='red', linestyle='--', linewidth=2, 
                  label=f'VaR 99%: ${var_99:.0f}')
        ax.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        
        # Labels
        ax.set_xlabel('P&L ($)', fontsize=12)
        ax.set_ylabel('Probability Density', fontsize=12)
        ax.set_title('30-Day P&L Distribution (Monte Carlo)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Annotations
        mean_pnl = np.mean(pnl_data)
        median_pnl = np.median(pnl_data)
        ax.axvline(mean_pnl, color='green', linestyle=':', linewidth=1.5, alpha=0.7)
        ax.text(mean_pnl, ax.get_ylim()[1]*0.9, f'Mean: ${mean_pnl:.0f}', 
               rotation=90, verticalalignment='bottom', fontsize=9)
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, 'pnl_distribution.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {filepath}")
            plt.close()
            return filepath
        else:
            plt.show()
            return None
    
    def plot_scenario_comparison(self, scenarios_df: pd.DataFrame, save: bool = True) -> str:
        """Plot scenario P&L comparison"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = ['green', 'lightgreen', 'gray', 'orange', 'red', 'darkred']
        bars = ax.barh(scenarios_df['scenario_name'], scenarios_df['total_pnl'], 
                      color=colors, edgecolor='black', linewidth=1.5)
        
        # Add value labels
        for i, (idx, row) in enumerate(scenarios_df.iterrows()):
            value = row['total_pnl']
            x_pos = value + (50 if value > 0 else -50)
            align = 'left' if value > 0 else 'right'
            ax.text(x_pos, i, f'${value:.0f}', va='center', ha=align, fontweight='bold')
        
        ax.axvline(0, color='black', linestyle='-', linewidth=2)
        ax.set_xlabel('P&L ($)', fontsize=12, fontweight='bold')
        ax.set_title('Scenario Analysis: 30-Day P&L', fontsize=14, fontweight='bold')
        ax.grid(True, axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, 'scenario_comparison.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {filepath}")
            plt.close()
            return filepath
        else:
            plt.show()
            return None
    
    def plot_volatility_forecast(self, forecasts: Dict, historical_vol: pd.Series, 
                                 save: bool = True) -> str:
        """Plot volatility forecasts"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Historical volatility
        ax.plot(historical_vol.index, historical_vol.values * 100, 
               label='Historical Vol (20-day)', linewidth=2, color='blue', alpha=0.7)
        
        # Forecast levels (horizontal lines for simplicity)
        forecast_names = ['historical_20d', 'garch', 'ensemble']
        colors_forecast = ['lightblue', 'green', 'red']
        
        for name, color in zip(forecast_names, colors_forecast):
            if name in forecasts:
                ax.axhline(forecasts[name] * 100, linestyle='--', linewidth=2, 
                          color=color, label=f'{name.upper()} Forecast', alpha=0.8)
        
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Volatility (%)', fontsize=12)
        ax.set_title('Volatility Forecast', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, 'volatility_forecast.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {filepath}")
            plt.close()
            return filepath
        else:
            plt.show()
            return None


if __name__ == "__main__":
    # Test visualizer
    viz = GreeksVisualizer()
    print(f"Charts will be saved to: {viz.output_dir}")
