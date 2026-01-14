"""
Multi-Option Portfolio Analysis
Aggregates Greeks and risk across multiple options
"""

import numpy as np
import pandas as pd
from typing import List, Dict
from dataclasses import dataclass

from models import BlackScholesModel, GreeksCalculator


@dataclass
class OptionPosition:
    """Represents a single option position"""
    ticker: str
    option_type: str  # 'call' or 'put'
    strike: float
    expiry: str
    quantity: int  # Number of contracts (signed: positive = long, negative = short)
    entry_price: float


class PortfolioAnalyzer:
    """
    Analyze portfolio of multiple option positions
    """
    
    def __init__(self, spot_price: float, risk_free_rate: float, dividend_yield: float = 0.0):
        """
        Initialize portfolio analyzer
        
        Parameters:
        -----------
        spot_price : float
            Current underlying price
        risk_free_rate : float
            Risk-free rate
        dividend_yield : float
            Dividend yield
        """
        self.S = spot_price
        self.r = risk_free_rate
        self.q = dividend_yield
        self.positions = []
        
    def add_position(self, position: OptionPosition, implied_vol: float, time_to_maturity: float):
        """Add option position to portfolio"""
        self.positions.append({
            'position': position,
            'implied_vol': implied_vol,
            'time_to_maturity': time_to_maturity
        })
    
    def calculate_portfolio_greeks(self) -> Dict:
        """
        Calculate aggregate portfolio Greeks
        
        Returns:
        --------
        dict : Portfolio-level Greeks
        """
        print(f"\n{'='*70}")
        print(f"📊 PORTFOLIO GREEKS ANALYSIS")
        print(f"{'='*70}\n")
        
        total_delta = 0
        total_gamma = 0
        total_theta = 0
        total_vega = 0
        total_rho = 0
        total_value = 0
        
        position_details = []
        
        for item in self.positions:
            pos = item['position']
            iv = item['implied_vol']
            T = item['time_to_maturity']
            
            # Calculate Greeks for this position
            bs = BlackScholesModel(self.S, pos.strike, T, self.r, iv, self.q)
            greeks = bs.all_greeks(pos.option_type)
            current_price = bs.price(pos.option_type)
            
            # Scale by quantity (1 contract = 100 shares)
            quantity_shares = pos.quantity * 100
            
            position_delta = greeks['delta'] * quantity_shares
            position_gamma = greeks['gamma'] * quantity_shares
            position_theta = greeks['theta_per_day'] * abs(pos.quantity)  # Per contract
            position_vega = greeks['vega_percent'] * abs(pos.quantity)    # Per contract
            position_rho = greeks['rho_percent'] * abs(pos.quantity)      # Per contract
            position_value = current_price * abs(pos.quantity) * 100
            
            # Aggregate
            total_delta += position_delta
            total_gamma += position_gamma
            total_theta += position_theta * np.sign(pos.quantity)  # Account for short positions
            total_vega += position_vega * np.sign(pos.quantity)
            total_rho += position_rho * np.sign(pos.quantity)
            total_value += position_value * np.sign(pos.quantity)
            
            position_details.append({
                'ticker': pos.ticker,
                'option': f"{pos.option_type.upper()} ${pos.strike} {pos.expiry}",
                'quantity': pos.quantity,
                'current_price': current_price,
                'value': position_value * np.sign(pos.quantity),
                'delta': position_delta,
                'gamma': position_gamma,
                'theta': position_theta * np.sign(pos.quantity),
                'vega': position_vega * np.sign(pos.quantity),
                'pnl': (current_price - pos.entry_price) * pos.quantity * 100
            })
        
        # Create position breakdown DataFrame
        positions_df = pd.DataFrame(position_details)
        
        # Print individual positions
        print("Individual Positions:")
        print(positions_df[['option', 'quantity', 'current_price', 'delta', 'theta']].to_string(index=False))
        
        # Print portfolio Greeks
        print(f"\n{'='*70}")
        print(f"📐 PORTFOLIO GREEKS (Aggregate)")
        print(f"{'='*70}\n")
        print(f"Total Delta:      {total_delta:>10.2f}  (${abs(total_delta * self.S):>10,.0f} equivalent)")
        print(f"Total Gamma:      {total_gamma:>10.4f}")
        print(f"Total Theta:      ${total_theta:>10.2f}/day")
        print(f"Total Vega:       ${total_vega:>10.2f}/1% vol")
        print(f"Total Rho:        ${total_rho:>10.2f}/1% rate")
        print(f"\nPortfolio Value:  ${total_value:>10,.2f}")
        
        return {
            'total_delta': total_delta,
            'total_gamma': total_gamma,
            'total_theta': total_theta,
            'total_vega': total_vega,
            'total_rho': total_rho,
            'portfolio_value': total_value,
            'positions': positions_df,
            'dollar_delta': total_delta * self.S,
            'is_delta_neutral': abs(total_delta) < 0.1
        }
    
    def calculate_portfolio_hedge(self, portfolio_greeks: Dict) -> Dict:
        """
        Calculate hedge required for delta-neutrality
        
        Parameters:
        -----------
        portfolio_greeks : dict
            Portfolio Greeks from calculate_portfolio_greeks()
        
        Returns:
        --------
        dict : Hedge recommendation
        """
        total_delta = portfolio_greeks['total_delta']
        
        # Shares to hedge
        hedge_shares = -total_delta  # Opposite sign
        hedge_notional = hedge_shares * self.S
        
        print(f"\n{'='*70}")
        print(f"🛡️  PORTFOLIO HEDGE RECOMMENDATION")
        print(f"{'='*70}\n")
        print(f"Current Portfolio Delta: {total_delta:.2f}")
        print(f"Dollar Delta: ${portfolio_greeks['dollar_delta']:,.0f}")
        print(f"\n📝 To achieve delta-neutrality:")
        print(f"   {'SHORT' if hedge_shares < 0 else 'LONG'} {abs(hedge_shares):.0f} shares @ ${self.S:.2f}")
        print(f"   Notional: ${abs(hedge_notional):,.0f}")
        
        if abs(total_delta) < 0.1:
            print(f"\n✅ Portfolio is already delta-neutral (Delta = {total_delta:.3f})")
        elif abs(total_delta) < 10:
            print(f"\n🟡 Portfolio has small delta exposure (Delta = {total_delta:.2f})")
        else:
            print(f"\n🔴 Portfolio has significant delta exposure (Delta = {total_delta:.0f})")
        
        return {
            'current_delta': total_delta,
            'hedge_shares': hedge_shares,
            'hedge_notional': hedge_notional,
            'is_neutral': abs(total_delta) < 0.1
        }
    
    def greeks_contribution_analysis(self, portfolio_greeks: Dict) -> pd.DataFrame:
        """
        Analyze which positions contribute most to portfolio Greeks
        
        Returns:
        --------
        pd.DataFrame : Contribution analysis
        """
        positions_df = portfolio_greeks['positions']
        
        # Calculate contributions
        positions_df['delta_contribution_pct'] = (
            positions_df['delta'] / portfolio_greeks['total_delta'] * 100
        )
        positions_df['gamma_contribution_pct'] = (
            positions_df['gamma'] / portfolio_greeks['total_gamma'] * 100
        )
        
        print(f"\n{'='*70}")
        print(f"📊 GREEKS CONTRIBUTION ANALYSIS")
        print(f"{'='*70}\n")
        
        # Top delta contributors
        print("Top Delta Contributors:")
        top_delta = positions_df.nlargest(3, 'delta_contribution_pct')[
            ['option', 'delta', 'delta_contribution_pct']
        ]
        for _, row in top_delta.iterrows():
            print(f"  {row['option']}: {row['delta']:.2f} ({row['delta_contribution_pct']:.1f}%)")
        
        # Top gamma contributors
        print("\nTop Gamma Contributors:")
        top_gamma = positions_df.nlargest(3, 'gamma_contribution_pct')[
            ['option', 'gamma', 'gamma_contribution_pct']
        ]
        for _, row in top_gamma.iterrows():
            print(f"  {row['option']}: {row['gamma']:.4f} ({row['gamma_contribution_pct']:.1f}%)")
        
        return positions_df
    
    def portfolio_scenario_analysis(self, price_shocks: List[float]) -> pd.DataFrame:
        """
        Analyze portfolio P&L under different price scenarios
        
        Parameters:
        -----------
        price_shocks : list
            List of price changes (e.g., [-0.10, -0.05, 0, 0.05, 0.10])
        
        Returns:
        --------
        pd.DataFrame : Scenario results
        """
        print(f"\n{'='*70}")
        print(f"📈 PORTFOLIO SCENARIO ANALYSIS")
        print(f"{'='*70}\n")
        
        results = []
        
        for shock in price_shocks:
            S_new = self.S * (1 + shock)
            total_pnl = 0
            
            for item in self.positions:
                pos = item['position']
                iv = item['implied_vol']
                T = item['time_to_maturity']
                
                # Original price
                bs_orig = BlackScholesModel(self.S, pos.strike, T, self.r, iv, self.q)
                price_orig = bs_orig.price(pos.option_type)
                
                # New price
                bs_new = BlackScholesModel(S_new, pos.strike, T, self.r, iv, self.q)
                price_new = bs_new.price(pos.option_type)
                
                # P&L for this position
                position_pnl = (price_new - price_orig) * pos.quantity * 100
                total_pnl += position_pnl
            
            results.append({
                'price_shock_pct': shock * 100,
                'new_price': S_new,
                'portfolio_pnl': total_pnl
            })
            
            print(f"Shock {shock*100:+.0f}%: Spot ${S_new:.2f} → P&L ${total_pnl:>10,.0f}")
        
        return pd.DataFrame(results)


if __name__ == "__main__":
    # Example portfolio analysis
    analyzer = PortfolioAnalyzer(spot_price=145, risk_free_rate=0.045)
    
    # Add positions
    analyzer.add_position(
        OptionPosition(ticker='NVDA', option_type='call', strike=140, 
                      expiry='2026-04-17', quantity=10, entry_price=10.50),
        implied_vol=0.35,
        time_to_maturity=93/365
    )
    
    analyzer.add_position(
        OptionPosition(ticker='NVDA', option_type='put', strike=150,
                      expiry='2026-04-17', quantity=-5, entry_price=12.00),
        implied_vol=0.38,
        time_to_maturity=93/365
    )
    
    # Calculate portfolio Greeks
    greeks = analyzer.calculate_portfolio_greeks()
    
    # Hedge recommendation
    hedge = analyzer.calculate_portfolio_hedge(greeks)
    
    # Contribution analysis
    contrib = analyzer.greeks_contribution_analysis(greeks)
    
    # Scenario analysis
    scenarios = analyzer.portfolio_scenario_analysis([-0.10, -0.05, 0, 0.05, 0.10])
