
import numpy as np
from typing import Dict

class PositionSizer:
    """
    Risk management engine for optimal position sizing
    """
    
    def __init__(self, portfolio_value: float = 100000.0, max_risk_pct: float = 0.02):
        self.portfolio_value = portfolio_value
        self.max_risk_pct = max_risk_pct
        
    def suggest_stop_loss(self, spot_price: float, volatility: float, T_years: float) -> float:
        """
        Suggest stop loss based on volatility (2 standard deviations)
        """
        # Expected move over time horizon
        move = spot_price * volatility * np.sqrt(T_years)
        # 2 SD buffer
        stop_price = spot_price - (2 * move)
        return max(0.01, stop_price)
        
    def calculate_position_size(self, entry_price: float, stop_loss_price: float) -> Dict:
        """
        Calculate suggested position size based on risk limits
        
        Args:
            entry_price: Cost per unit (e.g. option premium)
            stop_loss_price: Price to sell at
            
        Returns:
            Dict with sizing details
        """
        if entry_price <= 0:
            return {
                'recommended_contracts': 0,
                'total_cost': 0.0,
                'pct_portfolio': 0.0,
                'total_risk': 0.0
            }

        # Dollar risk per unit
        risk_per_unit = entry_price - stop_loss_price
        
        # If stop loss is above entry (short) or invalid logic, assume 100% risk for options
        if risk_per_unit <= 0:
             risk_per_unit = entry_price # Max risk is premium
             
        # Max dollar risk allowed
        max_risk_dollars = self.portfolio_value * self.max_risk_pct
        
        # Units according to risk
        units_by_risk = max_risk_dollars / risk_per_unit
        
        # Contracts (1 contract = 100 shares typically, but here prices are likely per share)
        # Assuming entry_price is per-share premium. Contract cost = entry_price * 100
        contract_cost = entry_price * 100
        risk_per_contract = risk_per_unit * 100
        
        contracts = int(max_risk_dollars / risk_per_contract)
        
        # Cap at 10% portfolio allocation
        max_capital = self.portfolio_value * 0.10
        contracts_cap_capital = int(max_capital / contract_cost)
        
        recommended = max(0, min(contracts, contracts_cap_capital))
        
        total_cost = recommended * contract_cost
        
        return {
            'recommended_contracts': recommended,
            'total_cost': total_cost,
            'pct_portfolio': (total_cost / self.portfolio_value) * 100,
            'total_risk': recommended * risk_per_contract
        }
