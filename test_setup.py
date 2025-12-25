"""
Test Setup Script
Run this to check if everything is configured correctly
"""

import os
import sys

print("\n" + "="*60)
print("🔍 CHURN APP SETUP CHECKER")
print("="*60 + "\n")

# 1. Check Python version
print("1️⃣  Checking Python version...")
python_version = sys.version.split()[0]
print(f"   ✅ Python {python_version}")

# 2. Check required packages
print("\n2️⃣  Checking required packages...")
required_packages = ['flask', 'pandas', 'numpy']
missing_packages = []

for package in required_packages:
    try:
        __import__(package)
        print(f"   ✅ {package} installed")
    except ImportError:
        print(f"   ❌ {package} NOT installed")
        missing_packages.append(package)

if missing_packages:
    print(f"\n   ⚠️  Missing packages: {', '.join(missing_packages)}")
    print("   Install them with:")
    print(f"   pip install {' '.join(missing_packages)}")
else:
    print("   ✅ All required packages installed!")

# 3. Check file structure
print("\n3️⃣  Checking file structure...")

current_dir = os.getcwd()
print(f"   Current directory: {current_dir}")

# Check for app.py
if os.path.exists('app.py'):
    print("   ✅ app.py found")
else:
    print("   ❌ app.py NOT found")
    print("      Make sure you save app.py in this folder!")

# Check for templates folder
if os.path.exists('templates'):
    print("   ✅ templates folder found")
    
    # List files in templates
    template_files = os.listdir('templates')
    required_templates = ['home.html', 'index.html', 'result.html', 'analytics.html']
    
    print("\n   Checking template files:")
    for template in required_templates:
        if template in template_files:
            print(f"   ✅ {template}")
        else:
            print(f"   ❌ {template} NOT found")
else:
    print("   ❌ templates folder NOT found")
    print("      Create a folder named 'templates'")

# 4. Check ports
print("\n4️⃣  Checking available ports...")
import socket

def is_port_available(port):
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(('localhost', port))
        sock.close()
        return result != 0
    except:
        return False

ports_to_check = [5000, 5001, 8080]
for port in ports_to_check:
    if is_port_available(port):
        print(f"   ✅ Port {port} is available")
    else:
        print(f"   ⚠️  Port {port} is in use (try another port)")

# 5. Summary
print("\n" + "="*60)
print("📊 SUMMARY")
print("="*60)

if not missing_packages and os.path.exists('app.py') and os.path.exists('templates'):
    print("✅ Setup looks good! You can run: python app.py")
else:
    print("⚠️  Some issues found. Fix them and try again.")
    print("\n📝 Quick Fix Checklist:")
    if missing_packages:
        print(f"   [ ] Install packages: pip install {' '.join(missing_packages)}")
    if not os.path.exists('app.py'):
        print("   [ ] Save app.py in current folder")
    if not os.path.exists('templates'):
        print("   [ ] Create 'templates' folder")
    print("   [ ] Put all HTML files in 'templates' folder")

print("\n" + "="*60 + "\n")