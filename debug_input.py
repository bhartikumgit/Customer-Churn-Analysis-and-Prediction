# debug_input.py
import joblib, pandas as pd, numpy as np, json, os

MODEL_PATH = "rf_model.pkl"
CSV_PATH = "churn analysis.xlsx"   # or training_data.csv

print("=== Loading model ===")
model = joblib.load(MODEL_PATH)
feat = getattr(model, "feature_names_in_", None)
print("Model.feature_names_in_ (len):", (len(feat) if feat is not None else None))
print("First 30 model features:", list(feat)[:30] if feat is not None else "NONE")

print("\n=== Loading master table ===")
if CSV_PATH.lower().endswith((".xls", ".xlsx")):
    df = pd.read_excel(CSV_PATH)
else:
    df = pd.read_csv(CSV_PATH, encoding="utf-8", errors="replace")
df.columns = [c.strip() for c in df.columns]
print("Master columns (first 50):", df.columns.tolist()[:50])

# find candidate CustomerIDs
candidates = [c for c in df.columns if 'customer' in c.lower() or 'id'==c.lower() or c.lower().endswith('id')]
print("CustomerID candidates:", candidates[:10])

# pick first customer id to debug
if candidates:
    cid_col = candidates[0]
    cid = str(df.iloc[0][cid_col])
    print("Using sample CustomerID from column", cid_col, "value:", cid)
else:
    cid = None
    print("No CustomerID-like column found. Will inspect first row values.")

row = df.iloc[0].to_dict()
print("\nFirst-row raw (master):")
print(json.dumps({k:str(row[k])[:120] for k in list(row.keys())[:60]}, indent=2))

print("\n=== Try to build input as app would ===")
# attempt to build input following same rules as app
from ast import literal_eval
# load encoders if present
encoders = None
try:
    encoders = joblib.load("label_encoders.pkl")
    print("Loaded encoders dict keys:", list(encoders.keys())[:30])
except Exception as e:
    print("No encoders or failed to load:", e)

# build candidate input dict using model features if available
if feat is None:
    feat = [c for c in df.columns if c.lower()!='customerid'][:50]

input_row = {}
# if cid available use that row
if cid is not None:
    rec = df.iloc[0]
    for f in feat:
        input_row[f] = rec[f] if f in df.columns else None
else:
    # grab first row but mapping by name-lower
    lc = {c.lower(): c for c in df.columns}
    for f in feat:
        input_row[f] = None
        if f.lower() in lc:
            input_row[f] = df.iloc[0][lc[f.lower()]]

print("\nConstructed input sample (first 40 features):")
print(json.dumps({k: (None if input_row[k] is None else str(input_row[k])[:80]) for k in feat[:40]}, indent=2))

# show defaults: compute medians/modes for numeric/cat
defaults = {}
for f in feat:
    if f in df.columns and pd.api.types.is_numeric_dtype(df[f]):
        defaults[f] = float(df[f].median(skipna=True))
    elif f in df.columns:
        try:
            defaults[f] = str(df[f].mode().iat[0])
        except Exception:
            defaults[f] = ""
    else:
        defaults[f] = 0 if f.lower().endswith(("charges","revenue","total","monthly","tenure")) else ""
print("\nSample defaults (first 20):")
print(json.dumps({k: defaults[k] for k in list(defaults.keys())[:20]}, indent=2))

# if model has feature_importances_
if hasattr(model, "feature_importances_"):
    fi = list(model.feature_importances_)
    print("\nModel.feature_importances_ (len):", len(fi))
    print("First 20 importances:", fi[:20])
else:
    print("\nModel has no feature_importances_ attribute.")

# show numeric coercion result for first features
print("\nNumeric coercion check (attempt to convert first 20 input values):")
num_conv = {}
for k in feat[:20]:
    v = input_row.get(k)
    try:
        num_conv[k] = float(v) if v is not None else None
    except Exception:
        num_conv[k] = None
print(json.dumps(num_conv, indent=2))
