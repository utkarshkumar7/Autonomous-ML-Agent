#!/usr/bin/env python3
"""
Test script to verify the virtual environment setup
"""

import sys
import daytona

def main():
    print("🚀 Virtual Environment Test")
    print("=" * 40)
    
    # Python version
    print(f"🐍 Python Version: {sys.version}")
    print(f"📍 Python Executable: {sys.executable}")
    
    # Daytona info
    print(f"📦 Daytona Version: {daytona.__version__ if hasattr(daytona, '__version__') else 'Unknown'}")
    print(f"📍 Daytona Location: {daytona.__file__}")
    
    # Environment info
    print(f"🌍 Environment: autonomous-ml-agent")
    print(f"✅ All tests passed! Environment is ready for development.")
    
    return True

if __name__ == "__main__":
    main()
