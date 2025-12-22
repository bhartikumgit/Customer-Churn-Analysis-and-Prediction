from flask import Flask, request, render_template, jsonify
import pickle
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

MODEL_PATH = "rf_model.pkl"
ENCODERS_PATH = "label_encoders.pkl"

# ---- Load model & encoders at startup ----
def safe_load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

model = None
encoders = None
feature_names = None

try:
    model = safe_load_pickle(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load model from {MODEL_PATH}: {e}")

# Attempt to infer expected feature names:
if hasattr(model, "feature_names_in_"):
    feature_names = list(model.feature_names_in_)
elif hasattr(model, "columns"):
    # some pipeline objects may have .columns
    try:
        feature_names = list(model.columns)
    except Exception:
        feature_names = None
else:
    feature_names = None

# Load encoders if available
try:
    encoders = safe_load_pickle(ENCODERS_PATH)
except FileNotFoundError:
    encoders = None
except Exception as e:
    raise RuntimeError(f"Failed to load encoders from {ENCODERS_PATH}: {e}")

# Helper to preprocess a single-row dict -> DataFrame matching training features
def preprocess_input(data: dict):
    """
    data: dict of feature_name -> value (strings for categoricals, numbers for numerics)
    returns: pandas.DataFrame with one row prepared for model.predict
    """
    if feature_names:
        missing = [f for f in feature_names if f not in data]
        if missing:
            raise ValueError(f"Missing required features: {missing}")

        row = {f: data[f] for f in feature_names}
        df = pd.DataFrame([row], columns=feature_names)
    else:
        # not able to infer expected features - assume whatever user passes
        df = pd.DataFrame([data])

    # Apply label encoders if provided
    if encoders and isinstance(encoders, dict):
        for col, le in encoders.items():
            if col in df.columns:
                vals = df[col].astype(str).fillna("nan")
                try:
                    # if LabelEncoder-like object with .transform
                    df[col] = le.transform(vals)
                except Exception:
                    # fallback: if encoder stored mapping dict
                    if isinstance(le, dict):
                        df[col] = vals.map(le).fillna(-1)
                    else:
                        # Unknown encoder structure: attempt mapping by classes_
                        if hasattr(le, "classes_"):
                            classes = list(le.classes_)
                            mapped = vals.apply(lambda v: classes.index(v) if v in classes else -1)
                            df[col] = mapped
                        else:
                            # leave as-is and let model fail if incompatible
                            pass

    # Ensure numeric columns are numeric where possible
    for c in df.columns:
        # don't coerce if there are encoders for the column (already numeric)
        if encoders and isinstance(encoders, dict) and c in encoders:
            continue
        try:
            df[c] = pd.to_numeric(df[c])
        except Exception:
            # leave as string if cannot convert (model or pipeline should handle it)
            pass

    return df

# ---- Routes ----
@app.route("/", methods=["GET"])
def index():
    # Provide features list to the template if available so form can be generated
    return render_template("index.html", feature_names=feature_names)

@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts either:
      - form POST (from web form)
      - JSON payload: { "feature1": value1, ... }
    Returns prediction and probability.
    """
    try:
        if request.is_json:
            payload = request.get_json()
            if not isinstance(payload, dict):
                return jsonify({"error": "JSON body must be an object mapping feature names to values."}), 400
            data = payload
        else:
            # form data: request.form (ImmutableMultiDict)
            data = {k: (v if v != "" else None) for k, v in request.form.items()}

        df = preprocess_input(data)

        # If model is pipeline, call predict directly
        pred = model.predict(df)
        # If classifier with predict_proba
        prob = None
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(df)
            # If binary, probs[:,1] is positive class
            if probs.shape[1] == 2:
                prob = float(probs[0, 1])
            else:
                # for multi-class, return array
                prob = probs[0].tolist()

        response = {
            "prediction": int(pred[0]) if (hasattr(pred[0], "__int__")) else str(pred[0]),
            "probability": prob
        }

        # If HTML form, render template
        if not request.is_json:
            return render_template("result.html", result=response, inputs=data)
        return jsonify(response)

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        # For debugging, include exception message; remove in production
        return jsonify({"error": "Prediction failed", "detail": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
