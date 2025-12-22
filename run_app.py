#!/usr/bin/env python
"""Simple script to run the Flask app with better error handling"""
import sys
import os

# Change to script directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("Starting Churn Prediction App")
print("=" * 60)
print(f"Working directory: {os.getcwd()}")
print(f"Python: {sys.executable}")
print()

try:
    from app import app
    print("✓ App module loaded successfully")
    print("✓ Starting Flask server on http://127.0.0.1:5001")
    print("=" * 60)
    print("Press Ctrl+C to stop the server")
    print("=" * 60)
    print()
    
    app.run(debug=True, host="127.0.0.1", port=5001, use_reloader=False)
except KeyboardInterrupt:
    print("\n\nServer stopped by user")
except Exception as e:
    print(f"\n\nERROR: Failed to start server: {e}")
    import traceback
    traceback.print_exc()
    input("\nPress Enter to exit...")




