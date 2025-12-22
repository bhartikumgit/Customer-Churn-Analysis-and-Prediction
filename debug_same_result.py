import os
import pandas as pd
import joblib

from app import model, df_master, feature_names, DEFAULTS, prepare_input

print("CWD:", os.getcwd())
print("df_master shape:", None if df_master is None else df_master.shape)
print("First 10 feature_names:", feature_names[:10])

sample_ids = ["93520-GUJ", "57256-BIH", "72357-MAD"]
print("Sample IDs present?:", {cid: (cid in df_master.index) for cid in sample_ids})

for cid in sample_ids:
    if df_master is None or cid not in df_master.index:
        continue
    print("\n=== Debug for", cid, "===")
    form = {"customer_id": cid}
    df = prepare_input(form)
    print("Prepared row values (first 15 features):")
    print({col: df.iloc[0][col] for col in df.columns[:15]})
    pred = model.predict(df)
    prob = None
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(df)
        prob = float(p[0,1]) if p.shape[1] == 2 else p[0].tolist()
    print("Prediction:", pred[0], "Prob:", prob)




