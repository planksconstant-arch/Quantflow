import sys
import os

# Force utf-8 for stdout
sys.stdout.reconfigure(encoding='utf-8')

sys.path.append('.')
from main import QuantFlow
from utils import config

def debug_pricing():
    print("DEBUG: Starting Pricing Diagnosis...")
    
    # Initialize
    ticker = 'NVDA'
    strike = 140.0
    expiry = '2026-04-17'
    
    print(f"DEBUG: Initializing QuantFlow({ticker}, start={strike}, {expiry})")
    qf = QuantFlow(ticker=ticker, option_type='call', strike=strike, expiry=expiry)
    
    # Fetch Data
    print("DEBUG: Fetching Data...")
    qf.fetch_data()
    
    # Check Inputs
    print("\n" + "="*30)
    print("INPUT DATA VALIDATION")
    print("="*30)
    print(f"S (Spot Price): {qf.S}")
    print(f"K (Strike): {qf.K}")
    print(f"T (Time): {qf.T}")
    print(f"r (Risk Free): {qf.r}")
    print(f"sigma (Vol): {qf.sigma}")
    print(f"q (Div Yield): {qf.q}")
    
    if qf.S is None:
        print("CRITICAL ERROR: Spot price is None!")
        return

    # Run Individual Models manually to catch the culprit
    from models import BlackScholesModel, BinomialTreeModel, MonteCarloSimulation
    
    print("\n" + "="*30)
    print("MODEL VALIDATION")
    print("="*30)
    
    # 1. Black Scholes
    try:
        bs = BlackScholesModel(qf.S, qf.K, qf.T, qf.r, qf.sigma, qf.q)
        bs_price = bs.price('call')
        print(f"Black-Scholes Price: {bs_price}")
    except Exception as e:
        print(f"Black-Scholes Failed: {e}")

    # 2. Binomial
    try:
        binomial = BinomialTreeModel(qf.S, qf.K, qf.T, qf.r, qf.sigma, 50, qf.q)
        bin_price = binomial.price('call', 'european')
        print(f"Binomial Price: {bin_price}")
    except Exception as e:
        print(f"Binomial Failed: {e}")

    # 3. Monte Carlo
    try:
        mc = MonteCarloSimulation(qf.S, qf.K, qf.T, qf.r, qf.sigma, 1000, qf.q)
        mc_res = mc.price('call')
        print(f"Monte Carlo Price: {mc_res['price']}")
    except Exception as e:
        print(f"Monte Carlo Failed: {e}")

if __name__ == "__main__":
    debug_pricing()
