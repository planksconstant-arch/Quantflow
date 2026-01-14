import sys
import os

# Redirect stdout to a file
log_file = open('debug_output_v3.txt', 'w', encoding='utf-8')
sys.stdout = log_file

sys.path.append('.')
from main import QuantFlow
from models import BinomialTreeModel

def debug_pricing():
    print("DEBUG: Starting Pricing Diagnosis...")
    
    ticker = 'NVDA'
    strike = 140.0
    expiry = '2026-04-17'
    
    print(f"DEBUG: Initializing QuantFlow({ticker}, start={strike}, {expiry})")
    qf = QuantFlow(ticker=ticker, option_type='call', strike=strike, expiry=expiry)
    qf.fetch_data()
    
    print(f"S: {qf.S}")
    print(f"r: {qf.r}")
    print(f"sigma: {qf.sigma}")
    print(f"T: {qf.T}")
    
    # Check Binomial manually
    print("Checking Binomial Tree...")
    binomial = BinomialTreeModel(qf.S, qf.K, qf.T, qf.r, qf.sigma, 50, qf.q)
    print(f"u: {binomial.u}")
    print(f"d: {binomial.d}")
    print(f"p: {binomial.p}")
    
    price = binomial.price('call', 'european')
    print(f"Binomial Price: {price}")

if __name__ == "__main__":
    try:
        debug_pricing()
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        log_file.close()

