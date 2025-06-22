# debug_credentials.py
import os
from dotenv import load_dotenv

print("🔍 Debugging API Credentials")
print("=" * 40)

# Load .env file
load_dotenv()

# Check if credentials are loaded
api_key = os.getenv('BINANCE_TESTNET_API_KEY')
secret = os.getenv('BINANCE_TESTNET_SECRET')

print(f"📁 Current directory: {os.getcwd()}")
print(f"📄 .env file exists: {os.path.exists('.env')}")

if api_key:
    print(f"✅ API Key loaded: {api_key[:8]}...{api_key[-4:]} (length: {len(api_key)})")
else:
    print("❌ API Key not found")

if secret:
    print(f"✅ Secret loaded: {secret[:8]}...{secret[-4:]} (length: {len(secret)})")
else:
    print("❌ Secret not found")

# Check .env file content (safely)
if os.path.exists('.env'):
    print("\n📝 .env file contents:")
    with open('.env', 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines, 1):
            if line.strip():
                # Hide actual values for security
                if '=' in line:
                    key, value = line.split('=', 1)
                    masked_value = value[:4] + '*' * (len(value) - 8) + value[-4:] if len(value) > 8 else '*' * len(value)
                    print(f"  Line {i}: {key.strip()}={masked_value.strip()}")
                else:
                    print(f"  Line {i}: {line.strip()}")

print("\n🔧 Troubleshooting Tips:")
if not api_key or not secret:
    print("❌ Credentials not loaded properly")
    print("   1. Check .env file is in the same directory as the script")
    print("   2. Ensure no spaces around = sign")
    print("   3. No quotes around values")
    print("   4. File should have Unix line endings")
else:
    print("✅ Credentials loaded successfully")
    print("   Issue might be with testnet URL configuration")