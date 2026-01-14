"""
Sensitivity Analysis Module
Shows how option value changes with underlying parameters
"""

import numpy as np
import pandas as pd
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns

from models import BlackScholesModel


class SensitivityAnalyzer:
    """
    Perform comprehensive sensitivity analysis
    """
    
    def __init__(self, S: float, K: float, T: float, r: float, sigma: float,
                 option_type: str, q: float = 0.0):
        """Initialize sensitivity analyzer"""
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.option_type = option_type
        self.q = q
    
    def create_2d_sensitivity_table(self, 
                                   param1: str = 'stock', 
                                   param2: str = 'vol',
                                   param1_range: tuple = (0.85, 1.15),
                                   param2_range: tuple = (0.7, 1.3),
                                   n_points: int = 7) -> pd.DataFrame:
        """
        Create 2D sensitivity matrix
        
        Parameters:
        -----------
        param1, param2 : str
            'stock', 'vol', 'time', 'rate'
        param1_range, param2_range : tuple
            (min_multiplier, max_multiplier) of base value
        n_points : int
            Grid resolution
        
        Returns:
        --------
        pd.DataFrame : Sensitivity table
        """
        # Generate parameter grids
        p1_values = self._get_parameter_grid(param1, param1_range, n_points)
        p2_values = self._get_parameter_grid(param2, param2_range, n_points)
        
        # Calculate option values
        sensitivity_matrix = np.zeros((len(p2_values), len(p1_values)))
        
        for i, p2_val in enumerate(p2_values):
            for j, p1_val in enumerate(p1_values):
                # Set parameters
                params = {
                    'S': p1_val if param1 == 'stock' else (p2_val if param2 == 'stock' else self.S),
                    'K': self.K,
                    'T': p1_val if param1 == 'time' else (p2_val if param2 == 'time' else self.T),
                    'r': p1_val if param1 == 'rate' else (p2_val if param2 == 'rate' else self.r),
                    'sigma': p1_val if param1 == 'vol' else (p2_val if param2 == 'vol' else self.sigma),
                    'q': self.q
                }
                
                # Calculate option value
                bs = BlackScholesModel(**params)
                sensitivity_matrix[i, j] = bs.price(self.option_type)
        
        # Create DataFrame
        df = pd.DataFrame(
            sensitivity_matrix,
            index=[f"{self._format_param(param2, v)}" for v in p2_values],
            columns=[f"{self._format_param(param1, v)}" for v in p1_values]
        )
        
        return df
    
    def _get_parameter_grid(self, param: str, range_tuple: tuple, n_points: int) -> np.ndarray:
        """Generate parameter grid"""
        base_value = {
            'stock': self.S,
            'vol': self.sigma,
            'time': self.T,
            'rate': self.r
        }[param]
        
        return np.linspace(base_value * range_tuple[0], 
                          base_value * range_tuple[1], 
                          n_points)
    
    def _format_param(self, param: str, value: float) -> str:
        """Format parameter for display"""
        if param == 'stock':
            return f"${value:.0f}"
        elif param == 'vol':
            return f"{value*100:.0f}%"
        elif param == 'time':
            return f"{value*365:.0f}d"
        elif param == 'rate':
            return f"{value*100:.1f}%"
        return f"{value:.2f}"
    
    def plot_sensitivity_heatmap(self, sensitivity_df: pd.DataFrame, 
                                title: str = "Option Value Sensitivity",
                                save_path: str = None) -> None:
        """Plot sensitivity heatmap"""
        plt.figure(figsize=(10, 8))
        
        sns.heatmap(sensitivity_df.astype(float), annot=True, fmt='.2f', 
                   cmap='RdYlGn', center=self._get_base_value(),
                   cbar_kws={'label': 'Option Value ($)'},
                   linewidths=0.5, linecolor='gray')
        
        plt.title(title, fontsize=14, fontweight='bold', pad=20)
        plt.xlabel(sensitivity_df.columns.name or 'Parameter 1', fontsize=12)
        plt.ylabel(sensitivity_df.index.name or 'Parameter 2', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Sensitivity heatmap saved: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def _get_base_value(self) -> float:
        """Get base option value"""
        bs = BlackScholesModel(self.S, self.K, self.T, self.r, self.sigma, self.q)
        return bs.price(self.option_type)
    
    def breakeven_analysis(self) -> Dict:
        """
        Calculate breakeven points
        
        Returns:
        --------
        dict : Breakeven analysis
        """
        base_value = self._get_base_value()
        
        # Stock price breakeven
        stock_be = self.K + base_value  # For calls
        if self.option_type == 'put':
            stock_be = self.K - base_value
        
        # Probability of reaching breakeven (assume log-normal)
        from scipy.stats import norm
        
        drift = (self.r - self.q - 0.5 * self.sigma**2) * self.T
        diffusion = self.sigma * np.sqrt(self.T)
        
        if self.option_type == 'call':
            prob_be = 1 - norm.cdf((np.log(stock_be / self.S) - drift) / diffusion)
        else:
            prob_be = norm.cdf((np.log(stock_be / self.S) - drift) / diffusion)
        
        # Required stock move
        stock_move_pct = (stock_be - self.S) / self.S
        
        return {
            'breakeven_stock_price': stock_be,
            'breakeven_probability': prob_be,
            'required_move_pct': stock_move_pct,
            'required_move_dollars': stock_be - self.S,
            'option_premium_paid': base_value
        }
    
    def generate_sensitivity_report(self, output_path: str = None) -> str:
        """Generate comprehensive sensitivity report"""
        report = f"""
# Sensitivity Analysis Report

**Option**: {self.option_type.upper()} ${self.K:.2f}, Expiry: {self.T*365:.0f} days
**Current Stock**: ${self.S:.2f}
**Implied Vol**: {self.sigma*100:.2f}%
**Analysis Date**: {pd.Timestamp.now().strftime('%Y-%m-%d')}

---

## 1. Stock Price vs Volatility Sensitivity

"""
        # Create sensitivity table
        stock_vol_sensitivity = self.create_2d_sensitivity_table('stock', 'vol')
        report += stock_vol_sensitivity.to_markdown() + "\n\n"
        
        report += """
**Key Insights**:
- Upper left (low stock, low vol): Lowest option values
- Lower right (high stock, high vol): Highest option values
- Diagonal shows interaction effects

---

## 2. Breakeven Analysis

"""
        breakeven = self.breakeven_analysis()
        report += f"""
- **Breakeven Stock Price**: ${breakeven['breakeven_stock_price']:.2f}
- **Current Stock**: ${self.S:.2f}
- **Required Move**: {breakeven['required_move_pct']*100:+.2f}% (${breakeven['required_move_dollars']:+.2f})
- **Probability of Reaching Breakeven**: {breakeven['breakeven_probability']*100:.1f}%
- **Option Premium Cost**: ${breakeven['option_premium_paid']:.2f}

"""
        
        report += """
---

## 3. Sensitivity Insights

### Upside Scenarios
"""
        # Calculate upside scenarios
        for stock_move in [0.05, 0.10, 0.15]:
            S_new = self.S * (1 + stock_move)
            bs_up = BlackScholesModel(S_new, self.K, self.T, self.r, self.sigma, self.q)
            value_up = bs_up.price(self.option_type)
            pnl = (value_up - self._get_base_value()) * 100  # Per contract
            
            report += f"- Stock +{stock_move*100:.0f}%: Option = ${value_up:.2f}, P&L = ${pnl:+.0f} ({(pnl/(self._get_base_value()*100))*100:+.1f}%)\n"
        
        report += "\n### Downside Scenarios\n"
        for stock_move in [-0.05, -0.10, -0.15]:
            S_new = self.S * (1 + stock_move)
            bs_down = BlackScholesModel(S_new, self.K, self.T, self.r, self.sigma, self.q)
            value_down = bs_down.price(self.option_type)
            pnl = (value_down - self._get_base_value()) * 100
            
            report += f"- Stock {stock_move*100:.0f}%: Option = ${value_down:.2f}, P&L = ${pnl:+.0f} ({(pnl/(self._get_base_value()*100))*100:+.1f}%)\n"
        
        report += """
---

## 4. Risk Summary

**Asymmetry**: """ + ('Upside' if self.option_type == 'call' else 'Downside') + """ is magnified due to leverage.
**Max Loss**: Limited to premium paid (${:.2f} per contract)
**Max Gain**: """ + ('Unlimited (call)' if self.option_type == 'call' else f'Limited to ${(self.K - 0)*100:.0f} (put)')
        
        report = report.format(self._get_base_value())
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
            print(f"✓ Sensitivity report saved: {output_path}")
        
        return report


if __name__ == "__main__":
    # Test sensitivity analyzer
    analyzer = SensitivityAnalyzer(
        S=145, K=140, T=93/365, r=0.045, sigma=0.35, option_type='call'
    )
    
    # 2D sensitivity table
    print("Stock vs Vol Sensitivity:")
    table = analyzer.create_2d_sensitivity_table('stock', 'vol')
    print(table)
    
    # Breakeven
    print("\nBreakeven Analysis:")
    be = analyzer.breakeven_analysis()
    print(f"Breakeven: ${be['breakeven_stock_price']:.2f} ({be['breakeven_probability']*100:.1f}% probability)")
    
    # Generate report
    analyzer.generate_sensitivity_report('outputs/sensitivity_report.md')
