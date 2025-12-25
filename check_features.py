import joblib

model = joblib.load('rf_model.pkl')
print("Required features:")
features = list(model.feature_names_in_)
for i, feat in enumerate(features, 1):
    print(f"{i}. {feat}")
print(f"\nTotal: {len(features)} features")