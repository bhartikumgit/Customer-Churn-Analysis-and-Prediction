from flask import Flask, request, render_template, jsonify, send_file
import joblib
import pandas as pd
import numpy as np
import os
import math
import warnings
import io
import json

app = Flask(__name__)

MODEL_PATH = "rf_model.pkl"
ENCODERS_PATH = "label_encoders.pkl"
CSV_PATH = "churn analysis.xlsx"   # change if needed (supports .csv, .xls, .xlsx)

# ---------------- Load model & encoders ----------------
model = joblib.load(MODEL_PATH)

# encoders: joblib then pickle fallback
try:
    encoders = joblib.load(ENCODERS_PATH)
except Exception:
    try:
        import pickle
        with open(ENCODERS_PATH, "rb") as f:
            encoders = pickle.load(f)
    except Exception:
        encoders = None

# ---------------- Safe file reader (CSV with encoding fallbacks / Excel) ----------------
def safe_read_table(path):
    """
    Read CSV or Excel robustly. For CSV, try utf-8, latin1, utf-16, utf-16-le, utf-16-be, cp1252.
    For Excel extensions, use read_excel.
    Returns a pandas.DataFrame or raises an exception.
    """
    path_lower = path.lower()
    if path_lower.endswith((".xls", ".xlsx")):
        # Load all sheets, then pick the one that actually contains a
        # CustomerID-like column (e.g. Customer_ID, CustomerID, etc.).
        # Prefer the join sheet 'vw_joindata' if it exists (matches your screenshot),
        # otherwise fall back to any sheet that has a CustomerID-like column.
        try:
            xls = pd.read_excel(path, sheet_name=None)
        except Exception:
            # Fallback: old behaviour
            return pd.read_excel(path)

        if isinstance(xls, dict):
            # 0) If there is a sheet explicitly named 'vw_joindata', use it.
            if "vw_joindata" in xls:
                print(f"[INFO] Loaded Excel sheet 'vw_joindata' from '{path}'")
                return xls["vw_joindata"]

            cid_candidates = [
                "customerid",
                "customer_id",
                "customer id",
                "customer",
                "id",
                "custid",
                "cust_id",
                "cust id",
            ]
            chosen_name = None
            # 1) Prefer a sheet whose columns contain a CustomerID-like field
            for name, df_sheet in xls.items():
                cols_lower = [str(c).strip().lower() for c in df_sheet.columns]
                if any(c in cols_lower for c in cid_candidates):
                    chosen_name = name
                    break
            if chosen_name is None:
                # 2) Fallback: first sheet
                chosen_name = list(xls.keys())[0]
            print(f"[INFO] Loaded Excel sheet '{chosen_name}' from '{path}'")
            return xls[chosen_name]
        else:
            return xls
    # otherwise assume CSV-like -- attempt several encodings
    encodings_to_try = ["utf-8", "latin1", "utf-16", "utf-16-le", "utf-16-be", "cp1252"]
    last_err = None
    for enc in encodings_to_try:
        try:
            return pd.read_csv(path, low_memory=False, encoding=enc)
        except Exception as e:
            last_err = e
            continue
    # final fallback: attempt pandas engine with python (sometimes helps)
    try:
        return pd.read_csv(path, low_memory=False, engine="python")
    except Exception:
        raise last_err if last_err is not None else RuntimeError("Failed to read file")

# ---------------- Load + normalize master table (robust) ----------------
if os.path.exists(CSV_PATH):
    try:
        df_master = safe_read_table(CSV_PATH)
    except Exception as e:
        raise RuntimeError(f"Failed to load master dataset '{CSV_PATH}': {e}")

    # strip whitespace from column names
    df_master.columns = [str(c).strip() for c in df_master.columns]

    # --- align CSV columns to model feature names (case-insensitive + simple normalization) ---
    lower_to_col = {c.lower(): c for c in df_master.columns}

    if hasattr(model, "feature_names_in_"):
        model_feats = list(model.feature_names_in_)
        rename_map = {}
        unmatched = []
        # simplified_map removes underscores/spaces for fuzzy matches
        simplified_map = {c.lower().replace("_", "").replace(" ", ""): c for c in df_master.columns}
        for mf in model_feats:
            mf_lower = mf.lower()
            if mf_lower in lower_to_col:
                orig_col = lower_to_col[mf_lower]
                if orig_col != mf:
                    rename_map[orig_col] = mf
            else:
                key = mf_lower.replace("_", "").replace(" ", "")
                if key in simplified_map:
                    rename_map[simplified_map[key]] = mf
                else:
                    unmatched.append(mf)
        if rename_map:
            df_master = df_master.rename(columns=rename_map)
            print(f"[INFO] Renamed {len(rename_map)} columns to match model feature names.")
        if unmatched:
            print(f"[WARN] {len(unmatched)} model features not found in CSV columns. Example missing: {unmatched[:8]}")
    else:
        print("[INFO] Model does not expose feature_names_in_; skipping alignment step.")

    # --- find CustomerID-like column and set index ---
    cid_candidates = ["customerid", "customer_id", "customer id", "customer", "id", "custid", "cust_id", "cust id"]
    found_cid = None
    for c in df_master.columns:
        if c.lower() in cid_candidates:
            found_cid = c
            break
    if found_cid:
        if found_cid != "CustomerID":
            df_master = df_master.rename(columns={found_cid: "CustomerID"})
            print(f"[INFO] Renamed column '{found_cid}' -> 'CustomerID'")
        # Normalize CustomerID values to string and strip whitespace so lookups are reliable
        df_master["CustomerID"] = df_master["CustomerID"].astype(str).str.strip()
        df_master = df_master.set_index("CustomerID", drop=False)
    else:
        print("[WARN] No CustomerID-like column found in master CSV. Customer lookup will fail.")

    # quick sanity prints (helpful during debugging; remove later)
    try:
        print("[INFO] df_master columns aligned (first 40):", df_master.columns.tolist()[:40])
        if hasattr(model, "feature_names_in_"):
            sample_feats = list(model.feature_names_in_)[:10]
            available = [c for c in sample_feats if c in df_master.columns]
            if available:
                print("[INFO] sample values from first row for model features:")
                print(df_master.iloc[0][available].to_dict())
    except Exception:
        pass
else:
    df_master = None

# ---------------- Features & defaults ----------------
feature_names = None
if hasattr(model, "feature_names_in_"):
    feature_names = list(model.feature_names_in_)
elif df_master is not None:
    # fall back to all columns except CustomerID (if present)
    feature_names = [c for c in df_master.columns if c != "CustomerID"]
else:
    feature_names = None

DEFAULTS = {}
if df_master is not None and feature_names is not None:
    for col in feature_names:
        if col not in df_master.columns:
            continue
        ser = df_master[col]
        if pd.api.types.is_numeric_dtype(ser):
            try:
                DEFAULTS[col] = float(ser.median(skipna=True))
            except Exception:
                DEFAULTS[col] = 0.0
        else:
            try:
                DEFAULTS[col] = str(ser.mode().iat[0])
            except Exception:
                DEFAULTS[col] = ""
else:
    for c in (feature_names or []):
        DEFAULTS[c] = 0 if c.lower().endswith(("charges", "revenue", "total", "monthly", "tenure")) else ""

# ---------------- CSV lookup ----------------
def get_customer_features_from_csv(customer_id: str):
    """
    Return dict of features for given customer_id if present in df_master.
    Supports:
      - Exact CustomerID match on index (string)
      - Case-insensitive CustomerID match
      - Row index (0-based) if a pure integer is provided
    """
    if df_master is None or feature_names is None:
        return None

    cid_raw = str(customer_id)
    cid = cid_raw.strip()

    # 1) Exact index match
    if cid in df_master.index:
        row = df_master.loc[cid]
    else:
        # 2) Case-insensitive index match
        index_lower = {str(idx).lower(): idx for idx in df_master.index}
        if cid.lower() in index_lower:
            actual_idx = index_lower[cid.lower()]
            row = df_master.loc[actual_idx]
        else:
            # 3) Treat as 0-based row index if numeric
            try:
                row_idx = int(cid)
                if 0 <= row_idx < len(df_master):
                    row = df_master.iloc[row_idx]
                else:
                    return None
            except (ValueError, TypeError):
                return None

    out = {}
    for f in feature_names:
        if f in row:
            out[f] = row[f]
    return out

# ---------------- Prepare input (merge + encode) ----------------
def prepare_input(form_data: dict):
    """
    Build a one-row DataFrame ready for model.predict:
      - Lookup CSV by customer_id (if provided)
      - Apply manual overrides from form_data
      - Fill missing with DEFAULTS
      - Apply encoders (if available) to categorical cols
      - Coerce numeric columns to numeric
    """
    if feature_names is None:
        raise RuntimeError("Model feature names unknown. Cannot prepare input.")

    # 1) base dict from CSV lookup if possible
    data = {}
    cid = form_data.get("customer_id") or form_data.get("CustomerID") or None
    if cid and df_master is not None:
        csv_vals = get_customer_features_from_csv(cid)
        if csv_vals:
            for k, v in dict(csv_vals).items():
                data[k] = v

    # 2) overrides from incoming form/json
    for k, v in form_data.items():
        if k == "customer_id":
            continue
        if k in feature_names:
            data[k] = v

    # 3) fill missing with defaults
    for f in feature_names:
        if f not in data or data[f] in [None, "", "None", np.nan]:
            data[f] = DEFAULTS.get(f, "")

    # 4) create DataFrame using model order
    df = pd.DataFrame([data], columns=feature_names)

    # 5) Apply encoders (if loaded). Support sklearn-like encoders, dicts, mapping by classes_
    if encoders and isinstance(encoders, dict):
        for col, enc in encoders.items():
            if col not in df.columns:
                continue
            vals = df[col].astype(str).fillna("nan")
            # sklearn-like transform
            try:
                if hasattr(enc, "transform"):
                    transformed = enc.transform(vals)
                    df[col] = transformed
                    continue
            except Exception:
                pass
            # mapping dict
            if isinstance(enc, dict):
                df[col] = vals.map(enc).fillna(-1)
                continue
            # classes_ fallback
            if hasattr(enc, "classes_"):
                classes = list(enc.classes_)
                mapped = vals.apply(lambda v: classes.index(v) if v in classes else -1)
                df[col] = mapped
                continue
            # last resort: try transform on list
            try:
                df[col] = enc.transform(vals.tolist())
            except Exception:
                pass

    # 6) numeric coercion for numeric-default columns
    for c in df.columns:
        if isinstance(DEFAULTS.get(c), (int, float, np.integer, np.floating)):
            try:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(DEFAULTS[c])
            except Exception:
                pass

    return df

# ---------------- SHAP setup ----------------
SHAP_AVAILABLE = False
explainer = None
try:
    import shap
    warnings.filterwarnings("ignore", category=UserWarning)
    explainer = shap.TreeExplainer(model)
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False
    explainer = None

def get_top_factors_shap(df_row, top_n=6):
    try:
        sv = explainer.shap_values(df_row)
        if isinstance(sv, list) and len(sv) >= 2:
            shap_vals_for_pos = sv[1][0]
        else:
            shap_vals_for_pos = sv[0]
        arr = np.array(shap_vals_for_pos)
        idx = np.argsort(np.abs(arr))[::-1][:top_n]
        return [(df_row.columns[i], float(arr[i])) for i in idx]
    except Exception:
        return []

def get_top_factors_simple(df_row, top_n=6):
    out = []
    try:
        if not hasattr(model, "feature_importances_"):
            return []
        importances = np.array(model.feature_importances_)
        deviations = []
        for c in df_row.columns:
            try:
                val = float(df_row.iloc[0][c])
                default = float(DEFAULTS.get(c, 0) if DEFAULTS.get(c, "") != "" else 0)
                deviations.append(val - default)
            except Exception:
                deviations.append(0.0)
        deviations = np.array(deviations)
        scores = importances * deviations
        if np.all(scores == 0):
            # If all scores are 0, use global importances
            idx = np.argsort(importances)[::-1][:top_n]
            return [(df_row.columns[i], float(importances[i])) for i in idx]
        idx = np.argsort(np.abs(scores))[::-1][:top_n]
        return [(df_row.columns[i], float(scores[i])) for i in idx]
    except Exception:
        return []

def get_top_factors(df_row, top_n=6):
    if SHAP_AVAILABLE:
        res = get_top_factors_shap(df_row, top_n=top_n)
        if res:
            return res
    return get_top_factors_simple(df_row, top_n=top_n)

# ---------------- Analytics (global) ----------------
ANALYTICS = {}

def compute_global_analytics():
    global ANALYTICS
    if df_master is None or feature_names is None:
        ANALYTICS = {"available": False, "message": "No CSV loaded"}
        return
    X = df_master.copy()
    for f in feature_names:
        if f not in X.columns:
            X[f] = DEFAULTS.get(f, 0 if isinstance(DEFAULTS.get(f), (int, float)) else DEFAULTS.get(f, ""))
    Xf = X[feature_names].copy()
    for c in Xf.columns:
        if isinstance(DEFAULTS.get(c), (int, float, np.integer, np.floating)):
            Xf[c] = pd.to_numeric(Xf[c], errors="coerce").fillna(DEFAULTS.get(c))
    try:
        preds = model.predict(Xf)
        probs = model.predict_proba(Xf)[:,1] if hasattr(model, "predict_proba") else np.zeros(len(preds))
    except Exception as e:
        ANALYTICS = {"available": False, "message": f"Prediction on CSV failed: {e}"}
        return
    df_master["_pred"] = preds
    df_master["_prob"] = probs
    churn_rate = float(np.mean(preds))
    stay_rate = 1.0 - churn_rate
    analytics = {
        "available": True,
        "n_customers": int(len(df_master)),
        "churn_rate": churn_rate,
        "stay_rate": stay_rate,
    }
    if "State" in df_master.columns:
        state_churn = df_master.groupby("State")["_pred"].mean().sort_values(ascending=False)
        analytics["top_states_by_churn"] = state_churn.head(8).to_dict()
    if "Internet_Type" in df_master.columns:
        it_churn = df_master.groupby("Internet_Type")["_pred"].mean().sort_values(ascending=False)
        analytics["internet_type_churn"] = it_churn.to_dict()
    elif "Internet_Service" in df_master.columns:
        is_churn = df_master.groupby("Internet_Service")["_pred"].mean().sort_values(ascending=False)
        analytics["internet_service_churn"] = is_churn.to_dict()
    if hasattr(model, "feature_importances_"):
        try:
            fi = np.array(model.feature_importances_)
            pairs = list(zip(feature_names, fi))
            pairs_sorted = sorted(pairs, key=lambda x: x[1], reverse=True)
            analytics["top_global_features"] = [(k, float(v)) for k, v in pairs_sorted[:12]]
        except Exception:
            analytics["top_global_features"] = []
    else:
        analytics["top_global_features"] = []
    recs = []
    if churn_rate > 0.25:
        recs.append("High overall churn (>25%). Investigate recent product or billing changes.")
    elif churn_rate > 0.12:
        recs.append("Moderate churn. Consider targeted retention campaigns for high-risk segments.")
    if "internet_type_churn" in analytics:
        top = analytics["internet_type_churn"]
        if isinstance(top, dict):
            sorted_types = sorted(top.items(), key=lambda x: x[1], reverse=True)
            if sorted_types and sorted_types[0][1] - sorted_types[-1][1] > 0.12:
                recs.append(f"{sorted_types[0][0]} customers churn much more than others. Investigate service issues for this network type.")
    if "Tenure_in_Months" in df_master.columns:
        short_tenure_churn = df_master[df_master["Tenure_in_Months"] <= 3]["_pred"].mean()
        if not math.isnan(short_tenure_churn) and short_tenure_churn > churn_rate + 0.05:
            recs.append("Very short-tenure customers (<=3 months) show elevated churn. Consider onboarding improvements or early offers.")
    analytics["recommendations"] = recs
    # Add additional metrics
    if "Tenure_in_Months" in df_master.columns:
        analytics["avg_tenure"] = float(df_master["Tenure_in_Months"].mean())
    if "Monthly_Charge" in df_master.columns:
        analytics["avg_monthly_charge"] = float(df_master["Monthly_Charge"].mean())
    ANALYTICS = analytics

# ---------------- Routes ----------------
@app.route("/", methods=["GET"])
def home():
    # landing page with two choices: customer-level analysis or analytics dashboard
    return render_template("home.html")

@app.route("/predict_form", methods=["GET"])
def predict_form():
    # the existing index.html form (customer-level analysis)
    return render_template("index.html", feature_names=feature_names, csv_loaded=(df_master is not None))

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if request.is_json:
            raw = request.get_json()
        else:
            raw = {k: (v if v != "" else None) for k, v in request.form.items()}

        # Normalize / fetch customer_id (can be true CustomerID like '93520-GUJ'
        # or a 0-based row index like '0', '1', ...)
        customer_id = raw.get("customer_id") or raw.get("CustomerID") or raw.get("Customer_ID")
        if customer_id is not None:
            customer_id = str(customer_id).strip()

        # If a customer_id is provided, first try to pull that exact row from the
        # master table. This restores the original behaviour that worked for row
        # indices, and adds proper CustomerID support without touching the model.
        if customer_id and df_master is not None:
            row = None
            # 1) Try as real CustomerID in the index
            if customer_id in df_master.index:
                row = df_master.loc[customer_id]
            else:
                # 2) Try as 0-based row index
                try:
                    idx = int(customer_id)
                    if 0 <= idx < len(df_master):
                        row = df_master.iloc[idx]
                except (ValueError, TypeError):
                    row = None

            if row is not None:
                # Use the dataset row as the base input, like before.
                raw = row.to_dict()
                # Keep a normalised id field for display / downstream logic.
                raw["customer_id"] = customer_id
            else:
                return jsonify({"error": f"Customer ID or row index '{customer_id}' not found in dataset"}), 400

        df = prepare_input(raw)
        is_default = all(df.iloc[0][f] == DEFAULTS.get(f, 0 if f.lower().endswith(("charges", "revenue", "total", "monthly", "tenure")) else "") for f in feature_names)
        pred = model.predict(df)
        prob = None
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(df)
            if probs.shape[1] == 2:
                prob = float(probs[0, 1])
            else:
                prob = probs[0].tolist()
        result = {
            "prediction": int(pred[0]) if hasattr(pred[0], "__int__") else str(pred[0]),
            "probability": prob
        }
        try:
            factors = get_top_factors(df, top_n=6)
        except Exception:
            factors = []
        
        # Decode inputs for display
        display_inputs = {}
        for c in df.columns:
            v = df.iloc[0][c]
            if encoders and c in encoders:
                try:
                    display_inputs[c] = encoders[c].inverse_transform([int(v)])[0]
                except:
                    display_inputs[c] = v
            else:
                # Round numerical values to 2 decimal places
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    display_inputs[c] = round(float(v), 2)
                else:
                    display_inputs[c] = v
        
        if not request.is_json:
            return render_template("result.html",
                                   result=result,
                                   inputs=display_inputs,
                                   factors=factors,
                                   is_default=is_default)
        return jsonify({"result": result, "factors": factors})
    except Exception as e:
        return jsonify({"error": "Prediction failed", "detail": str(e)}), 500

@app.route("/batch_predict", methods=["GET", "POST"])
def batch_predict():
    """
    Upload CSV/XLSX. If 'CustomerID' column present, we'll lookup each CustomerID in training CSV.
    Otherwise we expect the uploaded file to contain model feature columns matching feature_names.
    Returns a downloadable CSV with _pred, _prob, and _factors (JSON string).
    """
    if request.method == "GET":
        return """
        <html><body>
        <h3>Upload CSV / Excel for batch predictions</h3>
        <form method="post" enctype="multipart/form-data">
          <input type="file" name="datafile" accept=".csv, .xls, .xlsx" />
          <br/><br/>
          <button type="submit">Upload & Predict</button>
        </form>
        </body></html>
        """
    if "datafile" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    f = request.files["datafile"]
    filename = f.filename.lower()
    try:
        if filename.endswith(".csv"):
            uploaded = pd.read_csv(f, low_memory=False)
        elif filename.endswith((".xls", ".xlsx")):
            uploaded = pd.read_excel(f)
        else:
            return jsonify({"error": "Unsupported file type. Use CSV or Excel."}), 400
    except Exception as e:
        return jsonify({"error": "Failed to read uploaded file", "detail": str(e)}), 400

    uploaded.columns = [c.strip() for c in uploaded.columns]
    results = []
    factor_rows = []
    for idx, row in uploaded.iterrows():
        if "CustomerID" in uploaded.columns:
            form_dict = {"customer_id": str(row.get("CustomerID"))}
            for col in uploaded.columns:
                if col in feature_names:
                    val = row.get(col)
                    if pd.isna(val):
                        continue
                    form_dict[col] = val
        else:
            form_dict = {}
            for col in uploaded.columns:
                if col in feature_names:
                    val = row.get(col)
                    if pd.isna(val):
                        continue
                    form_dict[col] = val
        try:
            x = prepare_input(form_dict)
            pred = model.predict(x)[0]
            prob = None
            if hasattr(model, "predict_proba"):
                p = model.predict_proba(x)
                prob = float(p[0,1]) if p.shape[1] == 2 else p[0].tolist()
            try:
                factors = get_top_factors(x, top_n=5)
            except Exception:
                factors = []
            results.append({"_pred": int(pred), "_prob": prob})
            factor_rows.append(factors)
        except Exception as e:
            results.append({"_pred": None, "_prob": None})
            factor_rows.append([])
    res_df = uploaded.copy().reset_index(drop=True)
    preds_series = [r["_pred"] for r in results]
    probs_series = [r["_prob"] for r in results]
    res_df["_pred"] = preds_series
    res_df["_prob"] = probs_series
    res_df["_factors"] = [json.dumps(fr) for fr in factor_rows]
    buf = io.StringIO()
    res_df.to_csv(buf, index=False)
    buf.seek(0)
    return send_file(
        io.BytesIO(buf.getvalue().encode("utf-8")),
        mimetype="text/csv",
        as_attachment=True,
        download_name="batch_predictions.csv"
    )

@app.route("/customer/<customer_id>", methods=["GET"])
def customer_detail(customer_id):
    """
    Show customer details by ID. If a templates/customer.html exists, render it with context.
    Otherwise return JSON with the customer's data and last prediction/prob (if computed).
    """
    cid = str(customer_id)
    if df_master is None or cid not in df_master.index:
        return jsonify({"error": "CustomerID not found in CSV", "customer_id": cid}), 404
    row = df_master.loc[cid]
    # minimal dict of visible attributes (pick commonly useful fields if present)
    keys_to_show = ["Gender", "Age", "Married", "State", "Tenure_in_Months",
                    "Phone_Service", "Internet_Service", "Internet_Type", "Monthly_Charge"]
    out = {}
    # case-insensitive mapping for keys_to_show
    col_map = {c.lower(): c for c in df_master.columns}
    for k in keys_to_show:
        k_lower = k.lower()
        if k_lower in col_map:
            out[k] = row[col_map[k_lower]]
    # attach last computed prediction/prob if analytics run
    pred = row.get("_pred") if "_pred" in row else None
    prob = row.get("_prob") if "_prob" in row else None
    out["_pred"] = int(pred) if pred is not None and not pd.isna(pred) else None
    out["_prob"] = float(prob) if prob is not None and not pd.isna(prob) else None

    # attempt to provide SHAP explanation for this single customer (if SHAP available)
    try:
        form_dict = {"customer_id": cid}
        x = prepare_input(form_dict)
        factors = get_top_factors(x, top_n=6)
    except Exception:
        factors = []

    # render template if available
    if os.path.exists(os.path.join("templates", "customer.html")):
        return render_template("customer.html", customer_id=cid, data=out, factors=factors)
    return jsonify({"customer_id": cid, "data": out, "factors": factors})

@app.route("/analytics", methods=["GET"])
def analytics_page():
    if os.path.exists(os.path.join("templates", "analytics.html")):
        return render_template("analytics.html", analytics=ANALYTICS)
    else:
        return jsonify(ANALYTICS)

@app.route("/analyse", methods=["GET", "POST"])
def analyse_dataset():
    if request.method == "GET":
        return render_template("analyse.html")
    if "dataset" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    f = request.files["dataset"]
    if f.filename == "":
        return jsonify({"error": "No file selected"}), 400
    try:
        if f.filename.lower().endswith(".csv"):
            df = pd.read_csv(f, low_memory=False)
        elif f.filename.lower().endswith((".xls", ".xlsx")):
            df = pd.read_excel(f)
        else:
            return jsonify({"error": "Unsupported file type. Use CSV or Excel."}), 400
    except Exception as e:
        return jsonify({"error": f"Failed to read file: {e}"}), 400

    df.columns = [str(c).strip() for c in df.columns]

    # --- align CSV columns to model feature names (case-insensitive + simple normalization) ---
    lower_to_col = {c.lower(): c for c in df.columns}

    if hasattr(model, "feature_names_in_"):
        model_feats = list(model.feature_names_in_)
        rename_map = {}
        unmatched = []
        # simplified_map removes underscores/spaces for fuzzy matches
        simplified_map = {c.lower().replace("_", "").replace(" ", ""): c for c in df.columns}
        for mf in model_feats:
            mf_lower = mf.lower()
            if mf_lower in lower_to_col:
                orig_col = lower_to_col[mf_lower]
                if orig_col != mf:
                    rename_map[orig_col] = mf
            else:
                key = mf_lower.replace("_", "").replace(" ", "")
                if key in simplified_map:
                    rename_map[simplified_map[key]] = mf
                else:
                    unmatched.append(mf)
        if rename_map:
            df = df.rename(columns=rename_map)
            print(f"[INFO] Renamed {len(rename_map)} columns to match model feature names.")
        if unmatched:
            print(f"[WARN] {len(unmatched)} model features not found in uploaded CSV columns. Example missing: {unmatched[:8]}")

    # Find CustomerID
    cid_candidates = ["customerid", "customer_id", "customer", "id", "custid"]
    found_cid = None
    for c in df.columns:
        if c.lower() in cid_candidates:
            found_cid = c
            break
    if found_cid and found_cid != "CustomerID":
        df = df.rename(columns={found_cid: "CustomerID"})
    if "CustomerID" in df.columns:
        df["CustomerID"] = df["CustomerID"].astype(str)
        df = df.set_index("CustomerID", drop=False)

    # Compute analytics similar to compute_global_analytics but on df
    X = df.copy()
    for f in feature_names:
        if f not in X.columns:
            X[f] = DEFAULTS.get(f, 0 if isinstance(DEFAULTS.get(f), (int, float)) else DEFAULTS.get(f, ""))
    Xf = X[feature_names].copy()
    for c in Xf.columns:
        if isinstance(DEFAULTS.get(c), (int, float, np.integer, np.floating)):
            Xf[c] = pd.to_numeric(Xf[c], errors="coerce").fillna(DEFAULTS.get(c))

    # Apply encoders (if loaded). Support sklearn-like encoders, dicts, mapping by classes_
    if encoders and isinstance(encoders, dict):
        for col, enc in encoders.items():
            if col not in Xf.columns:
                continue
            vals = Xf[col].astype(str).fillna("nan")
            # sklearn-like transform
            try:
                if hasattr(enc, "transform"):
                    transformed = enc.transform(vals)
                    Xf[col] = transformed
                    continue
            except Exception:
                pass
            # mapping dict
            if isinstance(enc, dict):
                Xf[col] = vals.map(enc).fillna(-1)
                continue
            # classes_ fallback
            if hasattr(enc, "classes_"):
                classes = list(enc.classes_)
                mapped = vals.apply(lambda v: classes.index(v) if v in classes else -1)
                Xf[col] = mapped
                continue
            # last resort: try transform on list
            try:
                Xf[col] = enc.transform(vals.tolist())
            except Exception:
                pass

    try:
        preds = model.predict(Xf)
        probs = model.predict_proba(Xf)[:,1] if hasattr(model, "predict_proba") else np.zeros(len(preds))
        df["_pred"] = preds
        df["_prob"] = probs
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"}), 500

    churn_rate = float(np.mean(preds))
    stay_rate = 1.0 - churn_rate
    analytics = {
        "available": True,
        "n_customers": int(len(df)),
        "churn_rate": churn_rate,
        "stay_rate": stay_rate,
    }
    if "State" in df.columns:
        state_churn = df.groupby("State")["_pred"].mean().sort_values(ascending=False)
        analytics["top_states_by_churn"] = state_churn.head(8).to_dict()
    if "Internet_Type" in df.columns:
        it_churn = df.groupby("Internet_Type")["_pred"].mean().sort_values(ascending=False)
        analytics["internet_type_churn"] = it_churn.to_dict()
    elif "Internet_Service" in df.columns:
        is_churn = df.groupby("Internet_Service")["_pred"].mean().sort_values(ascending=False)
        analytics["internet_service_churn"] = is_churn.to_dict()
    if hasattr(model, "feature_importances_"):
        fi = np.array(model.feature_importances_)
        pairs = list(zip(feature_names, fi))
        pairs_sorted = sorted(pairs, key=lambda x: x[1], reverse=True)
        analytics["top_global_features"] = [(k, f"{round(float(v * 100), 2)}%") for k, v in pairs_sorted[:12]]
    recs = []
    if churn_rate > 0.25:
        recs.append("High overall churn (>25%). Investigate recent product or billing changes.")
    elif churn_rate > 0.12:
        recs.append("Moderate churn. Consider targeted retention campaigns for high-risk segments.")
    if "internet_type_churn" in analytics:
        top = analytics["internet_type_churn"]
        if isinstance(top, dict):
            sorted_types = sorted(top.items(), key=lambda x: x[1], reverse=True)
            if sorted_types and sorted_types[0][1] - sorted_types[-1][1] > 0.12:
                recs.append(f"{sorted_types[0][0]} customers churn much more than others. Investigate service issues for this network type.")
    if "Tenure_in_Months" in df.columns:
        short_tenure_churn = df[df["Tenure_in_Months"] <= 3]["_pred"].mean()
        if not math.isnan(short_tenure_churn) and short_tenure_churn > churn_rate + 0.05:
            recs.append("Very short-tenure customers (<=3 months) show elevated churn. Consider onboarding improvements or early offers.")
    analytics["recommendations"] = recs
    # Add additional metrics
    if "Tenure_in_Months" in df.columns:
        analytics["avg_tenure"] = float(df["Tenure_in_Months"].mean())
    if "Monthly_Charge" in df.columns:
        analytics["avg_monthly_charge"] = float(df["Monthly_Charge"].mean())

    return render_template("analytics.html", analytics=analytics)

if __name__ == "__main__":
    try:
        app.run(debug=True, host="127.0.0.1", port=5001)
    except Exception as e:
        print(f"Failed to start app: {e}")
        import traceback
        traceback.print_exc()
