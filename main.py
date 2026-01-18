"""
QuantFlow Main Entry Point with Phase 2 ML Integration
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict

from data import MarketDataFetcher
from models import BlackScholesModel, BinomialTreeModel, MonteCarloSimulation, GreeksCalculator
from models.ml import VolatilityForecaster, MispricingDetector, RegimeDetector
from analysis import ScenarioAnalyzer
from models.risk.risk_manager import PositionSizer
from utils import config, time_to_maturity, format_currency, format_percentage


class QuantFlow:
    """
    Main QuantFlow Options Intelligence System with ML
    """
    
    def __init__(self, ticker: str = None, option_type: str = None,
                 strike: float = None, expiry: str = None):
        """Initialize QuantFlow system"""
        self.ticker = ticker or config.TICKER
        self.option_type = (option_type or config.OPTION_TYPE).lower()
        self.strike = strike or config.STRIKE
        self.expiry = expiry or config.EXPIRY
        
        # Initialize data fetcher
        self.data_fetcher = MarketDataFetcher(self.ticker)
        
        # Will be populated by fetch_data()
        self.market_data = None
        self.S = None
        self.K = None
        self.T = None
        self.r = None
        self.sigma = None
        self.q = None
        
    def fetch_data(self, force_refresh: bool = False):
        """Fetch all market data"""
        print(f"\n{'='*70}")
        print(f"QUANTFLOW OPTIONS INTELLIGENCE SYSTEM v2.0")
        print(f"{'='*70}")
        print(f"\nAnalyzing: {config.get_option_identifier()}")
        print(f"   Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        self.market_data = self.data_fetcher.get_all_market_data(force_refresh=force_refresh)
        
        # Extract key parameters
        self.S = self.market_data['spot_price']
        self.K = self.strike
        self.T = time_to_maturity(self.expiry)
        self.r = self.market_data['risk_free_rate']
        self.q = self.market_data['dividend_yield']
        self.sigma = self.market_data['option']['impliedVolatility']
        
        return self.market_data
    
    def get_ensemble_pricing(self) -> Dict:
        """Calculate prices from all three models"""
        if self.S is None:
            self.fetch_data()
        
        print(f"\n{'='*70}")
        print(f"ENSEMBLE PRICING ANALYSIS")
        print(f"{'='*70}\n")
        
        # Black-Scholes
        bs = BlackScholesModel(self.S, self.K, self.T, self.r, self.sigma, self.q)
        bs_price = bs.price(self.option_type)
        print(f"  Black-Scholes:  {format_currency(bs_price)}")
        
        # Binomial Tree
        binomial = BinomialTreeModel(
            self.S, self.K, self.T, self.r, self.sigma,
            config.BINOMIAL_STEPS, self.q
        )
        binomial_european = binomial.price(self.option_type, 'european')
        binomial_american = binomial.price(self.option_type, 'american')
        print(f"  Binomial (European):  {format_currency(binomial_european)}")
        print(f"  Binomial (American):  {format_currency(binomial_american)}")
        
        # Monte Carlo
        mc = MonteCarloSimulation(
            self.S, self.K, self.T, self.r, self.sigma,
            config.MC_SIMULATIONS, self.q
        )
        mc_result = mc.price(self.option_type)
        mc_price = mc_result['price']
        mc_ci = (mc_result['ci_95_lower'], mc_result['ci_95_upper'])
        print(f"  Monte Carlo:  {format_currency(mc_price)} "
              f"[95% CI: {format_currency(mc_ci[0])} - {format_currency(mc_ci[1])}]")
        
        # Ensemble
        ensemble_price = (bs_price + binomial_european + mc_price) / 3
        
        # Fallback if pricing fails or returns near-zero (when it shouldn't)
        if ensemble_price < 0.01:
             print("! Ensemble price near zero, using fallback calculation")
             ensemble_price = max(bs_price, 0.01)

        print(f"\nEnsemble Fair Value:  {format_currency(ensemble_price)}")
        
        # Market price logic with Stale Data Protection
        raw_market_price = self.market_data['option']['lastPrice']
        is_model_valid = self.market_data['option'].get('model_is_valid', True)
        
        market_price = raw_market_price
        price_source = "Market Data"
        
        # Fallback Conditions:
        # 1. Explicitly flagged as invalid by fetcher (Arbitrage detected)
        # 2. Price is zero/negative
        if not is_model_valid or raw_market_price <= 0.01:
             print(f"! NOTICE: Market Data Stale/Invalid (Last: {raw_market_price}). Using FAIR VALUE.")
             market_price = ensemble_price
             price_source = "Fair Value (Est)"
             
        print(f"Market Price:  {format_currency(market_price)} [{price_source}]")
        
        # Divergence
        divergence = ((ensemble_price - market_price) / market_price) * 100
        divergence_dollars = ensemble_price - market_price
        
        print(f"\nDivergence:  {format_currency(divergence_dollars)} ({divergence:+.2f}%)")
        
        if abs(divergence) < 2:
            assessment = "FAIRLY PRICED"
        elif divergence > 0:
            assessment = "UNDERVALUED (Market < Fair Value)"
        else:
            assessment = "OVERVALUED (Market > Fair Value)"
        
        print(f"   Assessment: {assessment}\n")
        
        return {
            'black_scholes': bs_price,
            'binomial_european': binomial_european,
            'binomial_american': binomial_american,
            'monte_carlo': mc_price,
            'monte_carlo_ci': mc_ci,
            'ensemble_fair_value': ensemble_price,
            'market_price': market_price,
            'divergence_pct': divergence,
            'divergence_dollars': divergence_dollars,
            'assessment': assessment
        }
    
    def get_greeks(self) -> Dict:
        """Calculate Greeks"""
        if self.S is None:
            self.fetch_data()
        
        print(f"\n{'='*70}")
        print(f"GREEKS ANALYSIS")
        print(f"{'='*70}\n")
        
        calc = GreeksCalculator(
            self.S, self.K, self.T, self.r, self.sigma,
            self.option_type, self.q
        )
        
        greeks_df = calc.get_all_greeks_summary()
        print(greeks_df.to_string(index=False))
        
        greeks = calc.get_analytical_greeks()
        
        return greeks
    
    def run_ml_analysis(self) -> Dict:
        """
        Run ML-powered analysis (Phase 2)
        """
        if self.S is None:
            self.fetch_data()
        
        results = {}
        
        # === VOLATILITY FORECASTING ===
        vol_forecaster = VolatilityForecaster(
            self.market_data['historical'],
            self.market_data['vix']
        )
        vol_forecasts = vol_forecaster.forecast_volatility(horizon=5)
        results['volatility_forecast'] = vol_forecasts
        
        # === REGIME DETECTION ===
        regime_detector = RegimeDetector(n_states=4)
        regime_features = regime_detector.prepare_features(
            self.market_data['historical'],
            self.market_data['vix']
        )
        regime_results = regime_detector.fit(regime_features)
        current_regime = regime_detector.predict_regime(regime_features.tail(20))
        results['regime'] = current_regime
        
        print(f"\n{'='*70}")
        print(f"MARKET REGIME ANALYSIS")
        print(f"{'='*70}\n")
        print(f"Current Regime: {current_regime['regime_label']}")
        print(f"Confidence: {current_regime['confidence']*100:.1f}%\n")
        
        # Regime-adjusted Greeks
        base_greeks = self.get_greeks()
        adjusted_greeks = regime_detector.regime_adjusted_greeks(base_greeks, current_regime)
        
        print(f"Regime-Adjusted Greeks:")
        print(f"   Delta: {adjusted_greeks['delta']:.4f} "
              f"[{adjusted_greeks['delta_lower']:.4f} - {adjusted_greeks['delta_upper']:.4f}]")
        print(f"   Gamma: {adjusted_greeks['gamma']:.4f} (×{adjusted_greeks['gamma']/base_greeks['gamma']:.2f})")
        print(f"\n   {adjusted_greeks['recommendation']}")
        
        results['adjusted_greeks'] = adjusted_greeks
        
        # === MISPRICING DETECTION ===
        # Note: Would need historical training data - demo version
        print(f"\n{'='*70}")
        print(f"MISPRICING DETECTION")
        print(f"{'='*70}\n")
        
        # Calculate mispricing features
        pricing = self.get_ensemble_pricing()
        option_data = self.market_data['option']
        
        print(f"Pricing Error: {pricing['divergence_pct']:+.2f}%")
        print(f"IV vs Forecast Vol: {self.sigma*100:.2f}% vs {vol_forecasts['ensemble']*100:.2f}%")
        
        vol_spread = (self.sigma - vol_forecasts['ensemble']) / vol_forecasts['ensemble'] * 100
        print(f"Volatility Spread: {vol_spread:+.2f}%")
        
        # Simple mispricing score (without full ML training)
        # Score based on pricing error + vol spread
        pricing_score = abs(pricing['divergence_pct']) * 5  # Weight pricing
        vol_score = abs(vol_spread) * 3  # Weight vol spread
        mispricing_score = min(pricing_score + vol_score, 100)
        
        if mispricing_score > 70:
            mispricing_assessment = "STRONG MISPRICING SIGNAL"
        elif mispricing_score > 40:
            mispricing_assessment = "MODERATE MISPRICING"
        else:
            mispricing_assessment = "FAIRLY PRICED"
        
        print(f"\nMispricing Score: {mispricing_score:.1f}/100")
        print(f"Assessment: {mispricing_assessment}")
        
        results['mispricing_score'] = mispricing_score
        results['mispricing_assessment'] = mispricing_assessment
        
        return results
    
    def run_scenario_analysis(self) -> Dict:
        """Run scenario analysis"""
        if self.S is None:
            self.fetch_data()
        
        # Risk Management & Position Sizing
        sizer = PositionSizer(portfolio_value=100000) # Default $100k portfolio
        
        # Calculate optimal stop loss (2 SD move)
        suggested_stop = sizer.suggest_stop_loss(self.S, self.sigma, self.T * 365)
        
        # Get sizing recommendation
        # If we are BUYING, entry is the option price (Fair Value preferred if market is stale)
        pricing = self.get_ensemble_pricing()
        entry_price = pricing['market_price']
        
        sizing = sizer.calculate_position_size(entry_price, stop_loss_price=entry_price * 0.5) # Assuming 50% max loss on option
        
        print(f"\n{'='*70}")
        print(f"RISK MANAGEMENT & SIZING")
        print(f"{'='*70}\n")
        print(f"Portfolio: $100,000  |  Max Risk: 2%  |  Stop Loss: 50% of Premium")
        print(f"Recommended Position: {sizing['recommended_contracts']} contracts")
        print(f"Cost: {format_currency(sizing['total_cost'])} ({sizing['pct_portfolio']:.1f}% of Portfolio)")
        print(f"Max Risk: {format_currency(sizing['total_risk'])}")
        
        analyzer = ScenarioAnalyzer(
            self.S, self.K, self.T, self.r, self.sigma,
            self.option_type, self.q, position_size=sizing['recommended_contracts']
        )
        
        # Standard scenarios
        scenarios = analyzer.run_standard_scenarios(time_horizon_days=30)
        
        # Monte Carlo distribution
        mc_dist = analyzer.monte_carlo_distribution(n_simulations=10000, time_horizon_days=30)
        
        return {
            'scenarios': scenarios,
            'monte_carlo_distribution': mc_dist
        }
    
    def run_phase2_demo(self):
        """Run complete Phase 2 demonstration"""
        print("\n" + "="*70)
        print("QUANTFLOW PHASE 2 DEMONSTRATION - ML ESSENTIALS")
        print("="*70)
        
        # Fetch data
        self.fetch_data()
        
        # Phase 1: Pricing & Greeks
        pricing = self.get_ensemble_pricing()
        greeks = self.get_greeks()
        
        # Phase 2: ML Analysis
        ml_results = self.run_ml_analysis()
        
        # Scenario Analysis
        scenario_results = self.run_scenario_analysis()
        
        # Summary
        print(f"\n{'='*70}")
        print(f"PHASE 2 COMPLETE - EXECUTIVE SUMMARY")
        print(f"{'='*70}\n")
        
        print(f"Option:  {config.get_option_identifier()}")
        print(f"Market: {format_currency(pricing['market_price'])}  |  Fair Value: {format_currency(pricing['ensemble_fair_value'])}")
        print(f"{pricing['assessment']}")
        
        print(f"\nRegime: {ml_results['regime']['regime_label']} ({ml_results['regime']['confidence']*100:.1f}% confidence)")
        print(f"   {ml_results['adjusted_greeks']['recommendation']}")
        
        print(f"\nMispricing: {ml_results['mispricing_score']:.0f}/100 - {ml_results['mispricing_assessment']}")
        
        mc_dist = scenario_results['monte_carlo_distribution']
        print(f"\n30-Day P&L (Monte Carlo):")
        print(f"   Mean: {format_currency(mc_dist['mean_pnl'])}  |  VaR(95%): {format_currency(mc_dist['var_95'])}")
        print(f"   Probability of Profit: {format_percentage(mc_dist['prob_profit'])}")
        
        print(f"\n{'='*70}")
        print(f"Next Steps: Phase 3 - Deliverables (Slides + Memo)")
        print(f"{'='*70}\n")
        
        return {
            'pricing': pricing,
            'greeks': greeks,
            'ml_results': ml_results,
            'scenario_results': scenario_results
        }


def main():
    """Main entry point"""
    qf = QuantFlow()
    results = qf.run_phase2_demo()
    return results


if __name__ == "__main__":
    main()
