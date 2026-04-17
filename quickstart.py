"""
SafeGear AI - Quick Start Script
One-command setup and launch for first-time users.

Usage:
    python quickstart.py
"""

import subprocess
import sys
import os
from pathlib import Path

def print_banner():
    """Print welcome banner."""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║                    🦺 SAFEGEAR AI 🦺                         ║
    ║                                                              ║
    ║     Real-Time Safety Compliance Detection System             ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required.")
        print(f"   Current version: {version.major}.{version.minor}")
        return False
    print(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
    return True

def install_dependencies():
    """Install required packages."""
    print("\n📦 Installing dependencies...")
    print("   This may take a few minutes...")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt",
            "--quiet"
        ])
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def setup_models():
    """Run model setup script."""
    print("\n🤖 Setting up YOLO models...")
    
    if not Path("setup_models.py").exists():
        print("❌ setup_models.py not found!")
        return False
    
    try:
        subprocess.check_call([sys.executable, "setup_models.py"])
        return True
    except subprocess.CalledProcessError as e:
        print(f"⚠️ Model setup had issues: {e}")
        print("   The app will download models on first run.")
        return True  # Non-critical

def launch_app():
    """Launch Streamlit application."""
    print("\n🚀 Launching SafeGear AI...")
    print("   Opening browser at http://localhost:8501")
    print("   Press Ctrl+C to stop the server\n")
    
    try:
        subprocess.call([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.headless", "false"
        ])
    except KeyboardInterrupt:
        print("\n\n👋 SafeGear AI stopped. See you next time!")

def main():
    """Main quickstart function."""
    print_banner()
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check if we're in the right directory
    if not Path("app.py").exists():
        print("❌ Error: app.py not found!")
        print("   Please run this script from the SafeGear AI directory.")
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("\n⚠️ Please install dependencies manually:")
        print("   pip install -r requirements.txt")
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    # Setup models
    setup_models()
    
    # Launch app
    launch_app()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Setup cancelled. Run 'python quickstart.py' to try again.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)
