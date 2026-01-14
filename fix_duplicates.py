import sys

# Read the file
with open('app.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

print(f"Original file: {len(lines)} lines")

# Remove lines 704-953 (duplicate code inside display_greeks function)
cleaned_lines = lines[:703] + lines[954:]

print(f"After cleanup: {len(cleaned_lines)} lines")
print(f"Removed: {len(lines) - len(cleaned_lines)} lines of duplicate code")

# Write back
with open('app.py', 'w', encoding='utf-8') as f:
    f.writelines(cleaned_lines)

print("\n✅ FIXED! File cleaned successfully.")
print("The dashboard should work now after Streamlit auto-reloads.")
