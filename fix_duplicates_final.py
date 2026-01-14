
# Remove duplicate lines 704-982 from app.py
with open('app.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

print(f"Original lines: {len(lines)}")

# We valid content is 0-703 (inclusive, so 704 lines)
# Then skip 704-982 (indices 704 to 982)
# Keep 983 onwards

keep_lines = lines[:703] + lines[982:]

print(f"New lines: {len(keep_lines)}")

with open('app.py', 'w', encoding='utf-8') as f:
    f.writelines(keep_lines)

print("✅ Removed duplicate code block.")
