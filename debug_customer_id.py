import pandas as pd
import os

CSV_PATH = "churn analysis.xlsx"
TARGET_ID = "34110-MAH"

print("CWD:", os.getcwd())
print("Exists?", os.path.exists(CSV_PATH))

df = pd.read_excel(CSV_PATH)
df.columns = [str(c).strip() for c in df.columns]
print("Columns (first 40):", df.columns.tolist()[:40])

cid_candidates = ["customerid", "customer_id", "customer id", "customer", "id", "custid", "cust_id", "cust id"]
found = None
for c in df.columns:
    if c.lower() in cid_candidates:
        found = c
        break
print("Found CID column:", found)

if found:
    df[found] = df[found].astype(str)
    df = df.set_index(found, drop=False)

print("Sample index values (first 10):", list(df.index[:10]))

print("Exact match for", TARGET_ID, ":", TARGET_ID in df.index)

matches_contains = [idx for idx in df.index if TARGET_ID in str(idx)]
print("Index values containing TARGET_ID as substring (first 10):", matches_contains[:10])

lower_target = TARGET_ID.lower()
lower_index_map = {str(idx).lower(): idx for idx in df.index}
print("Case-insensitive match for", lower_target, ":", lower_target in lower_index_map)
if lower_target in lower_index_map:
    print("Actual index key:", lower_index_map[lower_target])




