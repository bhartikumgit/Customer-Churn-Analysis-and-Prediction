# debug_unpickle.py
import pickle
import joblib
import cloudpickle
import dill
import io
import os

FN = "rf_model.pkl"

print("file size:", os.path.getsize(FN), "bytes")

# quick header inspect
with open(FN, "rb") as f:
    h = f.read(64)
print("first 64 bytes:", h[:64])

def try_pickle():
    try:
        with open(FN, "rb") as f:
            obj = pickle.load(f)
        print("pickle.load succeeded; type:", type(obj))
        return True
    except Exception as e:
        print("pickle.load failed:", repr(e))
        return False

def try_joblib():
    try:
        obj = joblib.load(FN)
        print("joblib.load succeeded; type:", type(obj))
        return True
    except Exception as e:
        print("joblib.load failed:", repr(e))
        return False

def try_cloudpickle():
    try:
        with open(FN, "rb") as f:
            obj = cloudpickle.load(f)
        print("cloudpickle.load succeeded; type:", type(obj))
        return True
    except Exception as e:
        print("cloudpickle.load failed:", repr(e))
        return False

def try_dill():
    try:
        with open(FN, "rb') as f:  # if your editor munges quotes, correct them
            obj = dill.load(f)
        print("dill.load succeeded; type:", type(obj))
        return True
    except Exception as e:
        print("dill.load failed:", repr(e))
        return False

print("\n== try pickle ==")
p = try_pickle()

print("\n== try joblib ==")
j = try_joblib()

print("\n== try cloudpickle ==")
c = try_cloudpickle()

print("\n== try dill ==")
d = try_dill()

print("\nSummary: pickle, joblib, cloudpickle, dill ->", p, j, c, d)
