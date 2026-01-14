"""
Executive Dashboard - One Visual to Rule Them All
Single-page comprehensive visual for judges
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
from typing import Dict
import os


class ExecutiveDashboard:
    """
    Create executive summary dashboard - 30-second visual
    """
    
    def __init__(self, output_dir: str = 'outputs/charts'):
        """Initialize dashboard creator"""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def create_dashboard(self, data: Dict) -> str:
        """
        Create comprehensive executive dashboard
        
        Parameters:
        -----------
        data : dict
            All analysis results
        
        Returns:
        --------
        str : Path to saved dashboard
        """
        fig = plt.figure(figsize=(20, 12))
        fig.suptitle('QuantFlow Executive Dashboard: AI-Powered Options Intelligence',
                    fontsize=24, fontweight='bold', y=0.98)
        
        gs = GridSpec(4, 5, figure=fig, hspace=0.35, wspace=0.35)
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # ROW 1: Key Metrics
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        
        # Mispricing Score Gauge
        self._create_mispricing_gauge(fig.add_subplot(gs[0, 0]), data)
        
        # Pricing Comparison
        self._create_pricing_comparison(fig.add_subplot(gs[0, 1:3]), data)
        
        # Regime Status
        self._create_regime_status(fig.add_subplot(gs[0, 3]), data)
        
        # Win Probability
        self._create_win_probability(fig.add_subplot(gs[0, 4]), data)
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # ROW 2: Greeks & Sensitivity
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        
        # Greeks Profile
        self._create_greeks_profile(fig.add_subplot(gs[1, 0:2]), data)
        
        # Sensitivity Heatmap
        self._create_sensitivity_heatmap(fig.add_subplot(gs[1, 2:4]), data)
        
        # Risk Meter
        self._create_risk_meter(fig.add_subplot(gs[1, 4]), data)
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # ROW 3: P&L Analysis
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        
        # P&L Distribution
        self._create_pnl_distribution(fig.add_subplot(gs[2, 0:3]), data)
        
        # Scenario Comparison
        self._create_scenario_bars(fig.add_subplot(gs[2, 3:5]), data)
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # ROW 4: Recommendation Banner
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        
        self._create_recommendation_banner(fig.add_subplot(gs[3, :]), data)
        
        # Save
        filepath = os.path.join(self.output_dir, 'executive_dashboard.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ Executive dashboard created: {filepath}")
        plt.close()
        
        return filepath
    
    def _create_mispricing_gauge(self, ax, data):
        """Create mispricing score gauge"""
        score = data.get('mispricing_score', 50)
        
        # Gauge background
        theta = np.linspace(0, np.pi, 100)
        
        # Color zones
        ax.fill_between(theta[:33], 0, 1, color='#ef4444', alpha=0.3,
                       transform=ax.transData._b, label='Fair')
        ax.fill_between(theta[33:66], 0, 1, color='#fbbf24', alpha=0.3, label='Moderate')
        ax.fill_between(theta[66:], 0, 1, color='#22c55e', alpha=0.3, label='Strong')
        
        # Needle
        needle_angle = np.pi * (score / 100)
        ax.arrow(0, 0, 0.8*np.cos(needle_angle), 0.8*np.sin(needle_angle),
                head_width=0.1, head_length=0.1, fc='black', ec='black', linewidth=3)
        
        # Score text
        ax.text(0, -0.2, f'{score:.0f}/100', ha='center', fontsize=20,
               fontweight='bold', color='#1e293b')
        ax.text(0, -0.35, 'Mispricing Score', ha='center', fontsize=12,
               style='italic', color='#64748b')
        
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-0.5, 1.1)
        ax.axis('off')
    
    def _create_pricing_comparison(self, ax, data):
        """Create pricing model comparison"""
        models = ['Market', 'Black-Scholes', 'Binomial', 'Monte Carlo', '✨ Fair Value']
        prices = [
            data.get('market_price', 10.2),
            data.get('bs_price', 10.5),
            data.get('binomial_price', 10.48),
            data.get('mc_price', 10.46),
            data.get('fair_value', 10.48)
        ]
        colors = ['#64748b', '#3b82f6', '#3b82f6', '#3b82f6', '#10b981']
        
        bars = ax.barh(models, prices, color=colors, edgecolor='black', linewidth=1.5)
        
        # Market price line
        ax.axvline(data.get('market_price', 10.2), color='#ef4444', linestyle='--',
                  linewidth=3, label='Market Price', alpha=0.7)
        
        # Value labels
        for i, (bar, price) in enumerate(zip(bars, prices)):
            ax.text(price + 0.15, i, f'${price:.2f}', va='center',
                   fontsize=11, fontweight='bold')
        
        ax.set_xlabel('Price ($)', fontsize=12, fontweight='bold')
        ax.set_title('Ensemble Pricing Analysis', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(axis='x', alpha=0.3)
    
    def _create_regime_status(self, ax, data):
        """Create regime status indicator"""
        regime = data.get('current_regime', 'Low Vol Bull')
        confidence = data.get('regime_confidence', 0.75)
        
        regime_colors = {
            'Low Vol Bull': '#22c55e',
            'High Vol Bull': '#f97316',
            'Low Vol Bear': '#f472b6',
            'High Vol Crisis': '#dc2626'
        }
        
        color = regime_colors.get(regime, '#64748b')
        
        # Circle indicator
        circle = mpatches.Circle((0.5, 0.55), 0.35, color=color, alpha=0.9)
        ax.add_patch(circle)
        
        # Regime text
        regime_lines = regime.split(' ')
        y_start = 0.65
        for i, line in enumerate(regime_lines):
            ax.text(0.5, y_start - i*0.12, line, ha='center', va='center',
                   fontsize=13, fontweight='bold', color='white')
        
        # Confidence
        ax.text(0.5, 0.15, f'{confidence*100:.0f}% confidence',
               ha='center', fontsize=10, style='italic', color='#64748b')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title('Market Regime', fontsize=14, fontweight='bold')
    
    def _create_win_probability(self, ax, data):
        """Create win probability circle"""
        prob = data.get('prob_profit', 0.58) * 100
        
        # Create donut chart
        wedges, texts, autotexts = ax.pie(
            [prob, 100-prob],
            colors=['#22c55e', '#e5e7eb'],
            startangle=90,
            counterclock=False,
            wedgeprops=dict(width=0.4, edgecolor='white', linewidth=3)
        )
        
        # Center text
        ax.text(0, 0, f'{prob:.0f}%', ha='center', va='center',
               fontsize=24, fontweight='bold', color='#1e293b')
        ax.text(0, -0.15, 'Win Prob', ha='center', va='center',
               fontsize=11, color='#64748b', style='italic')
        
        ax.set_title('Probability of Profit', fontsize=14, fontweight='bold')
    
    def _create_greeks_profile(self, ax, data):
        """Create Greeks profile bars"""
        greeks_data = {
            'Δ Delta': data.get('delta', 0.55),
            'Γ Gamma': data.get('gamma', 0.03) * 10,  # Scale for visibility
            'Θ Theta': abs(data.get('theta', -0.05)) * 10,
            'ν Vega': data.get('vega', 0.18)
        }
        
        labels = list(greeks_data.keys())
        values = list(greeks_data.values())
        colors = ['#3b82f6', '#10b981', '#f59e0b', '#8b5cf6']
        
        bars = ax.barh(labels, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Value labels
        for i, (bar, val, label) in enumerate(zip(bars, values, labels)):
            ax.text(val + 0.02, i, f'{val:.3f}', va='center', fontsize=10, fontweight='bold')
        
        ax.set_xlabel('Magnitude', fontsize=12, fontweight='bold')
        ax.set_title('Greeks Profile', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
    
    def _create_sensitivity_heatmap(self, ax, data):
        """Create sensitivity analysis heatmap"""
        # Sensitivity matrix: Value(Stock, Vol)
        stock_range = np.array([0.90, 0.95, 1.00, 1.05, 1.10]) * data.get('spot_price', 145)
        vol_range = np.array([0.30, 0.35, 0.40, 0.45, 0.50])
        
        # Simplified calculation (in real version, calculate with BS)
        base_value = data.get('fair_value', 10.5)
        sensitivity = np.outer(
            (stock_range / data.get('spot_price', 145) - 1),
            (vol_range / 0.40 - 1)
        ) * base_value + base_value
        
        im = ax.imshow(sensitivity, cmap='RdYlGn', aspect='auto')
        
        # Labels
        ax.set_xticks(range(len(stock_range)))
        ax.set_xticklabels([f'${x:.0f}' for x in stock_range], fontsize=9)
        ax.set_yticks(range(len(vol_range)))
        ax.set_yticklabels([f'{v*100:.0f}%' for v in vol_range], fontsize=9)
        
        ax.set_xlabel('Stock Price', fontsize=11, fontweight='bold')
        ax.set_ylabel('Volatility', fontsize=11, fontweight='bold')
        ax.set_title('Sensitivity Analysis: Option Value', fontsize=14, fontweight='bold')
        
        # Colorbar
        plt.colorbar(im, ax=ax, label='Option Value ($)')
        
        # Add values
        for i in range(len(vol_range)):
            for j in range(len(stock_range)):
                ax.text(j, i, f'${sensitivity[i, j]:.1f}',
                       ha='center', va='center', fontsize=8, color='black')
    
    def _create_risk_meter(self, ax, data):
        """Create overall risk meter"""
        # Calculate risk score (0-100, lower is better)
        var_pct = abs(data.get('var_95', -300) / (data.get('fair_value', 10.5) * 100)) * 100
        risk_score = min(var_pct, 100)
        
        # Risk level
        if risk_score < 20:
            risk_level = 'LOW'
            color = '#22c55e'
        elif risk_score < 40:
            risk_level = 'MODERATE'
            color = '#fbbf24'
        else:
            risk_level = 'HIGH'
            color = '#ef4444'
        
        # Circle
        circle = mpatches.Circle((0.5, 0.55), 0.35, color=color, alpha=0.9)
        ax.add_patch(circle)
        
        ax.text(0.5, 0.65, 'RISK', ha='center', va='center',
               fontsize=13, fontweight='bold', color='white')
        ax.text(0.5, 0.5, risk_level, ha='center', va='center',
               fontsize=14, fontweight='bold', color='white')
        
        ax.text(0.5, 0.15, f'VaR: ${abs(data.get("var_95", -300)):.0f}',
               ha='center', fontsize=9, color='#64748b')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title('Risk Assessment', fontsize=14, fontweight='bold')
    
    def _create_pnl_distribution(self, ax, data):
        """Create P&L distribution histogram"""
        pnl_dist = data.get('pnl_distribution', np.random.normal(50, 200, 10000))
        
        # Histogram
        ax.hist(pnl_dist, bins=60, color='#3b82f6', alpha=0.7, edgecolor='black')
        
        # Key lines
        ax.axvline(0, color='black', linestyle='-', linewidth=2, label='Break-even', alpha=0.5)
        ax.axvline(np.mean(pnl_dist), color='#10b981', linestyle='--',
                  linewidth=2.5, label=f'Mean: ${np.mean(pnl_dist):.0f}')
        ax.axvline(data.get('var_95', np.percentile(pnl_dist, 5)), color='#ef4444',
                  linestyle='--', linewidth=2.5,
                  label=f'VaR 95%: ${data.get("var_95", np.percentile(pnl_dist, 5)):.0f}')
        
        ax.set_xlabel('P&L ($)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_title('30-Day P&L Distribution (10,000 Monte Carlo Simulations)',
                    fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(alpha=0.3)
    
    def _create_scenario_bars(self, ax, data):
        """Create scenario comparison bars"""
        scenarios = ['Bull\n(+10%)', 'Mild Bull\n(+5%)', 'Base\n(0%)',
                    'Mild Bear\n(-5%)', 'Bear\n(-10%)', 'Crisis\n(-20%)']
        pnls = data.get('scenario_pnls', [450, 220, 50, -180, -420, -850])
        colors = ['#22c55e', '#84cc16', '#9ca3af', '#fb923c', '#ef4444', '#7f1d1d']
        
        bars = ax.barh(scenarios, pnls, color=colors, edgecolor='black', linewidth=1.5)
        
        # Zero line
        ax.axvline(0, color='black', linestyle='-', linewidth=2)
        
        # Value labels
        for i, (bar, pnl) in enumerate(zip(bars, pnls)):
            x_pos = pnl + (30 if pnl > 0 else -30)
            ha = 'left' if pnl > 0 else 'right'
            ax.text(x_pos, i, f'${pnl:.0f}', va='center', ha=ha,
                   fontsize=10, fontweight='bold')
        
        ax.set_xlabel('P&L ($)', fontsize=12, fontweight='bold')
        ax.set_title('Scenario Analysis', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
    
    def _create_recommendation_banner(self, ax, data):
        """Create recommendation banner"""
        ax.axis('off')
        
        recommendation = data.get('recommendation', 'BUY option, implement delta-neutral hedge')
        rec_type = 'BUY' if 'BUY' in recommendation else 'SELL' if 'SELL' in recommendation else 'HOLD'
        rec_color = '#22c55e' if rec_type == 'BUY' else '#ef4444' if rec_type == 'SELL' else '#fbbf24'
        
        # Recommendation box
        rect = mpatches.FancyBboxPatch((0.02, 0.25), 0.96, 0.6,
                                      boxstyle="round,pad=0.02",
                                      facecolor=rec_color, alpha=0.2,
                                      edgecolor=rec_color, linewidth=4)
        ax.add_patch(rect)
        
        # Title
        ax.text(0.5, 0.80, '🎯 AI RECOMMENDATION', ha='center', fontsize=18,
               fontweight='bold', transform=ax.transAxes)
        
        # Recommendation text
        ax.text(0.5, 0.55, recommendation, ha='center', fontsize=14,
               transform=ax.transAxes, style='italic', wrap=True)
        
        # Key metrics
        metrics = (f"Expected Return: {data.get('expected_return', 14.3):.1f}% | "
                  f"Win Prob: {data.get('prob_profit', 0.58)*100:.0f}% | "
                  f"Max Risk (VaR 95%): ${abs(data.get('var_95', -300)):.0f} | "
                  f"Sharpe Ratio: {data.get('sharpe', 1.8):.2f}")
        
        ax.text(0.5, 0.15, metrics, ha='center', fontsize=11,
               transform=ax.transAxes,
               bbox=dict(boxstyle='round,pad=0.8', facecolor='#fef3c7',
                        edgecolor='#f59e0b', linewidth=2))
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)


if __name__ == "__main__":
    # Test dashboard
    dashboard = ExecutiveDashboard()
    
    # Sample data
    test_data = {
        'mispricing_score': 67,
        'market_price': 10.20,
        'bs_price': 10.52,
        'binomial_price': 10.48,
        'mc_price': 10.46,
        'fair_value': 10.49,
        'current_regime': 'High Vol Bull',
        'regime_confidence': 0.82,
        'prob_profit': 0.58,
        'delta': 0.55,
        'gamma': 0.029,
        'theta': -0.051,
        'vega': 0.18,
        'spot_price': 145,
        'pnl_distribution': np.random.normal(52, 185, 10000),
        'var_95': -420,
        'scenario_pnls': [450,  220, 50, -180, -420, -850],
        'recommendation': 'BUY 10 contracts, hedge with SHORT 550 shares. Rehedge when spot moves >3%.',
        'expected_return': 14.3,
        'sharpe': 1.8
    }
    
    dashboard.create_dashboard(test_data)
    print("✅ Test dashboard created")
