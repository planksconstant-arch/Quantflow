"""
Backtesting Framework for Options Strategies
Simulates historical trading with transaction costs
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from datetime import datetime, timedelta

from models import BlackScholesModel
from utils import format_currency, format_percentage


class OptionsBacktester:
    """
    Backtest options trading strategies with realistic costs
    """
    
    def __init__(self, initial_capital: float = 10000, commission_per_contract: float = 0.65):
        """
        Initialize backtester
        
        Parameters:
        -----------
        initial_capital : float
            Starting capital
        commission_per_contract : float
            Commission per option contract per side (e.g., $0.65 at most brokers)
        """
        self.initial_capital = initial_capital
        self.commission = commission_per_contract
        self.trades = []
        self.equity_curve = []
        
    def simulate_option_trade(self, 
                             entry_date: datetime,
                             exit_date: datetime,
                             S_entry: float,
                             S_exit: float,
                             K: float,
                             option_type: str,
                             iv_entry: float,
                             iv_exit: float,
                             r: float,
                             q: float = 0.0,
                             position_size: int = 1,
                             bid_ask_spread_pct: float = 0.02) -> Dict:
        """
        Simulate a single option trade with realistic costs
        
        Parameters:
        -----------
        entry_date, exit_date : datetime
            Trade dates
        S_entry, S_exit : float
            Stock prices at entry and exit
        K : float
            Strike price
        option_type : str
            'call' or 'put'
        iv_entry, iv_exit : float
            Implied volatilities
        r : float
            Risk-free rate
        q : float
            Dividend yield
        position_size : int
            Number of contracts (1 contract = 100 shares)
        bid_ask_spread_pct : float
            Bid-ask spread as % of mid price (default 2%)
        
        Returns:
        --------
        dict : Trade results
        """
        # Time to maturity at entry and exit
        expiry = exit_date + timedelta(days=30)  # Assume 30-day option
        T_entry = (expiry - entry_date).days / 365
        T_exit = (expiry - exit_date).days / 365
        
        # Entry price (buy at ask)
        bs_entry = BlackScholesModel(S_entry, K, T_entry, r, iv_entry, q)
        mid_entry = bs_entry.price(option_type)
        ask_entry = mid_entry * (1 + bid_ask_spread_pct / 2)
        
        # Exit price (sell at bid)
        bs_exit = BlackScholesModel(S_exit, K, max(T_exit, 0.001), r, iv_exit, q)
        mid_exit = bs_exit.price(option_type)
        bid_exit = mid_exit * (1 - bid_ask_spread_pct / 2)
        
        # Transaction costs
        entry_commission = self.commission * position_size
        exit_commission = self.commission * position_size
        slippage_entry = mid_entry * 0.005 * position_size  # 0.5% slippage
        slippage_exit = mid_exit * 0.005 * position_size
        
        # Total costs
        entry_cost = (ask_entry * 100 * position_size + entry_commission + slippage_entry)
        exit_proceeds = (bid_exit * 100 * position_size - exit_commission - slippage_exit)
        
        # P&L
        gross_pnl = (mid_exit - mid_entry) * 100 * position_size
        transaction_costs = (ask_entry - mid_entry + mid_exit - bid_exit) * 100 * position_size + entry_commission + exit_commission + slippage_entry + slippage_exit
        net_pnl = exit_proceeds - entry_cost
        
        return {
            'entry_date': entry_date,
            'exit_date': exit_date,
            'entry_price': ask_entry,
            'exit_price': bid_exit,
            'mid_entry': mid_entry,
            'mid_exit': mid_exit,
            'position_size': position_size,
            'gross_pnl': gross_pnl,
            'transaction_costs': transaction_costs,
            'net_pnl': net_pnl,
            'return_pct': (net_pnl / entry_cost) * 100,
            'holding_days': (exit_date - entry_date).days
        }
    
    def backtest_mispricing_strategy(self, 
                                     historical_data: pd.DataFrame,
                                     mispricing_threshold: float = 5.0,
                                     hold_days: int = 30) -> Dict:
        """
        Backtest strategy that buys undervalued options
        
        Parameters:
        -----------
        historical_data : pd.DataFrame
            Must have columns: date, spot_price, strike, option_type, market_price, 
            fair_value, implied_vol, risk_free_rate
        mispricing_threshold : float
            Buy if (fair_value - market_price) / market_price > threshold %
        hold_days : int
            Holding period
        
        Returns:
        --------
        dict : Backtest results
        """
        print(f"\n{'='*70}")
        print(f"📊 BACKTESTING MISPRICING STRATEGY")
        print(f"{'='*70}\n")
        print(f"Threshold: {mispricing_threshold}% undervalued")
        print(f"Holding Period: {hold_days} days")
        print(f"Initial Capital: {format_currency(self.initial_capital)}\n")
        
        capital = self.initial_capital
        trades = []
        
        # Simulate trades
        for i, row in historical_data.iterrows():
            mispricing_pct = ((row['fair_value'] - row['market_price']) / row['market_price']) * 100
            
            if mispricing_pct > mispricing_threshold and capital > row['market_price'] * 100:
                # Calculate position size (use 10% of capital per trade)
                position_value = capital * 0.10
                position_size = int(position_value / (row['market_price'] * 100))
                
                if position_size > 0:
                    # Find exit point
                    exit_idx = min(i + hold_days, len(historical_data) - 1)
                    exit_row = historical_data.iloc[exit_idx]
                    
                    # Simulate trade
                    trade = self.simulate_option_trade(
                        entry_date=row['date'],
                        exit_date=exit_row['date'],
                        S_entry=row['spot_price'],
                        S_exit=exit_row['spot_price'],
                        K=row['strike'],
                        option_type=row['option_type'],
                        iv_entry=row['implied_vol'],
                        iv_exit=exit_row['implied_vol'],
                        r=row['risk_free_rate'],
                        position_size=position_size
                    )
                    
                    # Update capital
                    capital += trade['net_pnl']
                    trades.append(trade)
                    
                    print(f"Trade {len(trades)}: {row['option_type'].upper()} ${row['strike']} "
                          f"| Entry: ${trade['mid_entry']:.2f} | Exit: ${trade['mid_exit']:.2f} "
                          f"| P&L: {format_currency(trade['net_pnl'])} ({trade['return_pct']:+.1f}%)")
        
        # Calculate performance metrics
        trades_df = pd.DataFrame(trades)
        
        total_return = capital - self.initial_capital
        total_return_pct = (total_return / self.initial_capital) * 100
        
        # Sharpe ratio (annualized)
        if len(trades_df) > 1:
            returns = trades_df['return_pct'].values / 100
            sharpe = (np.mean(returns) * 252 / np.mean(trades_df['holding_days'])) / (np.std(returns) * np.sqrt(252 / np.mean(trades_df['holding_days'])))
        else:
            sharpe = np.nan
        
        # Win rate
        wins = trades_df[trades_df['net_pnl'] > 0]
        win_rate = len(wins) / len(trades_df) if len(trades_df) > 0 else 0
        
        # Max drawdown
        cumulative_pnl = trades_df['net_pnl'].cumsum()
        running_max = cumulative_pnl.expanding().max()
        drawdown = cumulative_pnl - running_max
        max_drawdown = drawdown.min() if len(drawdown) > 0 else 0
        
        # Average trade metrics
        avg_win = wins['net_pnl'].mean() if len(wins) > 0 else 0
        losses = trades_df[trades_df['net_pnl'] < 0]
        avg_loss = losses['net_pnl'].mean() if len(losses) > 0 else 0
        profit_factor = abs(wins['net_pnl'].sum() / losses['net_pnl'].sum()) if len(losses) > 0 and losses['net_pnl'].sum() != 0 else np.inf
        
        results = {
            'total_trades': len(trades_df),
            'winning_trades': len(wins),
            'losing_trades': len(losses),
            'win_rate': win_rate,
            'total_return': total_return,
            'total_return_pct': total_return_pct,
            'final_capital': capital,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'avg_holding_days': trades_df['holding_days'].mean() if len(trades_df) > 0 else 0,
            'total_transaction_costs': trades_df['transaction_costs'].sum() if len(trades_df) > 0 else 0,
            'trades': trades_df
        }
        
        # Print summary
        print(f"\n{'='*70}")
        print(f"📈 BACKTEST RESULTS")
        print(f"{'='*70}\n")
        print(f"Total Trades: {results['total_trades']}")
        print(f"Win Rate: {format_percentage(results['win_rate'])}")
        print(f"\n💰 Returns:")
        print(f"  Total Return: {format_currency(results['total_return'])} ({results['total_return_pct']:+.2f}%)")
        print(f"  Final Capital: {format_currency(results['final_capital'])}")
        print(f"\n📊 Risk Metrics:")
        print(f"  Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown: {format_currency(results['max_drawdown'])}")
        print(f"\n🎯 Trade Quality:")
        print(f"  Avg Win: {format_currency(results['avg_win'])}")
        print(f"  Avg Loss: {format_currency(results['avg_loss'])}")
        print(f"  Profit Factor: {results['profit_factor']:.2f}")
        print(f"  Transaction Costs: {format_currency(results['total_transaction_costs'])}")
        
        return results
    
    def compare_strategies(self, results: Dict, benchmark_return: float) -> None:
        """Compare strategy to benchmark"""
        print(f"\n{'='*70}")
        print(f"📊 STRATEGY vs BENCHMARK")
        print(f"{'='*70}\n")
        print(f"Strategy Return: {results['total_return_pct']:+.2f}%")
        print(f"Benchmark Return: {benchmark_return:+.2f}%")
        print(f"Alpha: {(results['total_return_pct'] - benchmark_return):+.2f}%")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        
        if results['total_return_pct'] > benchmark_return:
            print(f"\n✅ Strategy OUTPERFORMED benchmark by {(results['total_return_pct'] - benchmark_return):.2f}%")
        else:
            print(f"\n❌ Strategy UNDERPERFORMED benchmark by {abs(results['total_return_pct'] - benchmark_return):.2f}%")


if __name__ == "__main__":
    # Example backtest with synthetic data
    backtester = OptionsBacktester(initial_capital=10000)
    
    # Create synthetic historical data
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='W')
    historical_data = pd.DataFrame({
        'date': dates,
        'spot_price': 100 + np.random.randn(len(dates)).cumsum(),
        'strike': 105,
        'option_type': 'call',
        'market_price': 5 + np.random.randn(len(dates)) * 0.5,
        'fair_value': 5.5 + np.random.randn(len(dates)) * 0.5,
        'implied_vol': 0.25 + np.random.randn(len(dates)) * 0.02,
        'risk_free_rate': 0.045
    })
    
    # Run backtest
    results = backtester.backtest_mispricing_strategy(historical_data, mispricing_threshold=5.0)
    
    # Compare to benchmark (e.g., 8% annual return)
    backtester.compare_strategies(results, benchmark_return=8.0)
