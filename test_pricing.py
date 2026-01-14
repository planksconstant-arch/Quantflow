"""Quick test to check what's wrong with pricing"""
import sys
sys.path.append('.')

from main import QuantFlow

# Initialize with NVDA defaults
qf = QuantFlow(ticker='NVDA', option_type='call', strike=140.0, expiry='2026-04-17')

# Fetch data
print("Fetching data...")
qf.fetch_data()

# Get pricing
print("\n" + "="*70)
print("Testing get_ensemble_pricing()")
print("="*70)

pricing = qf.get_ensemble_pricing()

print("\n" + "="*70)
print("RESULTS:")
print("="*70)
for key, value in pricing.items():
    print(f"{key}: {value}")
