import joblib
import pickle
import gzip
import os

print("Loading model saved with joblib...")

# Load the model data (it's a dictionary)
model_data = joblib.load('rf_model.pkl')

print(f"✓ Model data loaded successfully!")
print(f"Keys in model data: {model_data.keys()}")

# Extract components
model = model_data['model']
encoders = model_data['encoders']
feature_names = model_data['feature_names']

print(f"\nModel type: {type(model)}")
print(f"Number of features: {len(feature_names)}")
print(f"Number of encoders: {len(encoders)}")

# Save encoders separately (as before)
print("\nSaving label encoders...")
with open('label_encoders.pkl', 'wb') as f:
    pickle.dump(encoders, f)

# Save feature names
print("Saving feature names...")
with open('feature_names.txt', 'w') as f:
    for feature in feature_names:
        f.write(f"{feature}\n")

# Compress and save ONLY the model
print("Compressing model...")
with gzip.open('rf_model.pkl.gz', 'wb', compresslevel=9) as f:
    pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

compressed_size = os.path.getsize('rf_model.pkl.gz') / 1024 / 1024
original_size = os.path.getsize('rf_model.pkl') / 1024 / 1024

print(f"\n✓ Compression complete!")
print(f"Original size: {original_size:.2f} MB")
print(f"Compressed size: {compressed_size:.2f} MB")
print(f"Compression ratio: {(1 - compressed_size/original_size) * 100:.1f}%")

# Test loading the compressed model
print("\nTesting compressed model...")
with gzip.open('rf_model.pkl.gz', 'rb') as f:
    test_model = pickle.load(f)

print("✓ Compressed model loads successfully!")
print(f"Model type: {type(test_model)}")
print(f"Number of estimators: {test_model.n_estimators}")

print("\n" + "="*70)
print("✓ ALL FILES READY FOR DEPLOYMENT!")
print("="*70)
print("Files created:")
print("  • rf_model.pkl.gz (compressed model)")
print("  • label_encoders.pkl (encoders)")
print("  • feature_names.txt (feature list)")
print("\nYou can now run: python app.py")
print("="*70)