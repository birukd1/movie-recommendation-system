#!/usr/bin/env python3
"""
Project Initialization Script
Sets up the Movie Recommendation System for first-time use
"""

import os
import sys
import subprocess
import time

def print_header(text):
    """Print a formatted header"""
    print("\n" + "="*60)
    print(f"🎬 {text}")
    print("="*60)

def print_step(step, text):
    """Print a formatted step"""
    print(f"\n{step}. {text}")
    print("-" * 40)

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"Running: {description}")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Success!")
            return True
        else:
            print(f"❌ Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Exception: {str(e)}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} is not compatible. Need Python 3.8+")
        return False

def install_dependencies():
    """Install required Python packages"""
    packages = [
        "pandas>=1.3.0",
        "numpy>=1.21.0", 
        "scikit-learn>=1.0.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "fastapi>=0.70.0",
        "uvicorn>=0.15.0",
        "requests>=2.25.0"
    ]
    
    for package in packages:
        if not run_command(f"pip install {package}", f"Installing {package}"):
            return False
    return True

def run_demo():
    """Run the system demo"""
    print("Running comprehensive demo...")
    return run_command("python demo.py", "System demo")

def run_tests():
    """Run the test suite"""
    print("Running test suite...")
    return run_command("python -m pytest test_system.py -v", "Unit tests")

def start_api_server():
    """Start the API server in background"""
    print("Starting API server...")
    print("The server will be available at http://localhost:8000")
    print("Press Ctrl+C to stop the server when done testing")
    
    try:
        subprocess.run("python api_server.py", shell=True)
    except KeyboardInterrupt:
        print("\n🛑 API server stopped")

def main():
    """Main initialization process"""
    print_header("MOVIE RECOMMENDATION SYSTEM SETUP")
    
    print("Welcome to the Movie Recommendation System!")
    print("This script will set up everything needed to run the system.")
    
    # Step 1: Check Python version
    print_step(1, "Checking Python Version")
    if not check_python_version():
        print("Please install Python 3.8 or higher and try again.")
        return False
    
    # Step 2: Install dependencies
    print_step(2, "Installing Dependencies")
    if not install_dependencies():
        print("Failed to install dependencies. Please check your internet connection.")
        return False
    
    # Step 3: Run demo
    print_step(3, "Running System Demo")
    if not run_demo():
        print("Demo failed. Please check the error messages above.")
        return False
    
    # Step 4: Run tests (optional, as they can be slow)
    print_step(4, "Running Tests (Optional)")
    response = input("Run full test suite? This may take a few minutes (y/N): ").lower()
    if response in ['y', 'yes']:
        run_tests()
    else:
        print("Skipping tests. You can run them later with: python test_system.py")
    
    # Step 5: API server option
    print_step(5, "API Server")
    response = input("Start the API server now? (y/N): ").lower()
    if response in ['y', 'yes']:
        start_api_server()
    else:
        print("You can start the API server later with: python api_server.py")
    
    # Final success message
    print_header("SETUP COMPLETE!")
    print("🎉 The Movie Recommendation System is ready to use!")
    print("\nNext steps:")
    print("1. Start API server: python api_server.py")
    print("2. Test API: python client_example.py")
    print("3. View documentation: README.md")
    print("4. Explore notebooks: notebooks/analysis.ipynb")
    print("\nFor deployment options, see: DEPLOYMENT.md")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n❌ Setup failed. Please check the error messages and try again.")
        sys.exit(1)
    else:
        print("\n✅ Setup completed successfully!")
        sys.exit(0)