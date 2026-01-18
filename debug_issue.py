
import sys
import os

# Enforce UTF-8 encoding for Windows consoles
if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

sys.path.append(os.getcwd())

from main import QuantFlow
from data.fetch_market_data import MarketDataFetcher
import pandas as pd

def debug_pricing():
    print("running debug...")
    
    # 1. Test MarketDataFetcher directly
    print("\n--- MarketDataFetcher ---")
    fetcher = MarketDataFetcher("NVDA")
    try:
        data = fetcher.get_all_market_data()
        print(f"Spot Price: {data.get('spot_price')}")
        print(f"Risk Free Rate: {data.get('risk_free_rate')}")
        print(f"Dividend Yield: {data.get('dividend_yield')}")
        
        opt = data.get('option', {})
        print(f"Option Price: {opt.get('lastPrice')}")
        print(f"Implied Vol: {opt.get('impliedVolatility')}")
        
        hist = data.get('historical')
        if isinstance(hist, pd.DataFrame):
            print(f"Historical Data: {len(hist)} rows")
            print(hist.tail())
        else:
            print(f"Historical Data: {hist}")
            
    except Exception as e:
        print(f"Fetcher Error: {e}")

    # 2. Test QuantFlow Orchestrator
    print("\n--- QuantFlow Class ---")
    try:
        qf = QuantFlow(ticker="NVDA", option_type="put", strike=140.0, expiry='2025-06-20') # Use a far out date to be safe
        pricing = qf.get_ensemble_pricing()
        print("\nPricing Results:")
        for k, v in pricing.items():
            print(f"{k}: {v}")
            
        ml_res = qf.run_ml_analysis()
        print("\nML Results:")
        print(f"Regime: {ml_res.get('regime')}")
        
    except Exception as e:
        print(f"QuantFlow Error: {e}")

if __name__ == "__main__":
    debug_pricing()
