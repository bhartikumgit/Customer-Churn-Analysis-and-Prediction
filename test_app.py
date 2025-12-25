"""
Quick Test Script - Run this to test all routes
"""

import requests
import time

BASE_URL = "http://localhost:5000"

print("\n" + "="*60)
print("🧪 TESTING CHURN PREDICTION APP")
print("="*60 + "\n")

# Make sure the app is running
print("⏳ Checking if app is running...")
try:
    response = requests.get(BASE_URL, timeout=5)
    if response.status_code == 200:
        print("✅ App is running!\n")
    else:
        print("❌ App returned error:", response.status_code)
        exit()
except:
    print("❌ App is not running!")
    print("   Please run: python app.py")
    exit()

# Test each route
tests = [
    ("Home Page", "/"),
    ("Upload Page", "/upload"),
]

print("Testing routes:\n")
for name, route in tests:
    try:
        response = requests.get(BASE_URL + route, timeout=5)
        if response.status_code == 200:
            print(f"✅ {name:20} → {route}")
        else:
            print(f"❌ {name:20} → {route} (Error: {response.status_code})")
    except Exception as e:
        print(f"❌ {name:20} → {route} (Error: {str(e)})")

print("\n" + "="*60)
print("✅ Basic tests complete!")
print("="*60 + "\n")

print("Manual tests needed:")
print("1. Go to http://localhost:5000")
print("2. Click 'Upload Your Dataset'")
print("3. Upload your CSV file")
print("4. Click 'Analyze All Customers' button")
print("5. Click 'Analyze One Customer' and select customer")
print("\nLet me know what happens at each step!")