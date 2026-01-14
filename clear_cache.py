# Clear Streamlit cache script
import os
import shutil

cache_dir = os.path.expanduser("~/.streamlit/cache")
if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)
    print(f"✅ Cleared Streamlit cache at {cache_dir}")
else:
    print(f"ℹ️  No cache found at {cache_dir}")

print("\n🔄 Please do the following:")
print("1. In your browser, press Ctrl+Shift+R (hard refresh)")
print("2. Or close the browser tab and reopen http://localhost:8501")
print("3. Click 'Run Analysis' again")
