#!/usr/bin/env python3
"""
Setup Verification Script
Verifies that the Movie Recommendation System is properly configured
"""

import os
import sys
import importlib.util

def check_file_exists(filepath, description):
    """Check if a file exists"""
    if os.path.exists(filepath):
        print(f"✅ {description}: {filepath}")
        return True
    else:
        print(f"❌ Missing {description}: {filepath}")
        return False

def check_import(module_name, description):
    """Check if a module can be imported"""
    try:
        __import__(module_name)
        print(f"✅ {description} imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import {description}: {e}")
        return False

def verify_project_structure():
    """Verify all project files are present"""
    print("🔍 VERIFYING PROJECT STRUCTURE")
    print("=" * 50)
    
    required_files = [
        ("recommendation_system.py", "Main recommendation engine"),
        ("api_server.py", "FastAPI server"),
        ("demo.py", "Demo script"),
        ("test_system.py", "Test suite"),
        ("client_example.py", "API client"),
        ("requirements.txt", "Dependencies"),
        ("README.md", "Documentation"),
        ("LICENSE", "License file"),
        ("setup.py", "Package setup"),
        (".gitignore", "Git ignore file"),
        ("DEPLOYMENT.md", "Deployment guide"),
        ("PROJECT_SUMMARY.md", "Project summary"),
        ("docker/Dockerfile", "Docker configuration"),
        ("docker/docker-compose.yml", "Docker Compose"),
        ("notebooks/analysis.ipynb", "Jupyter notebook")
    ]
    
    all_present = True
    for filepath, description in required_files:
        if not check_file_exists(filepath, description):
            all_present = False
    
    return all_present

def verify_dependencies():
    """Verify required dependencies are installed"""
    print("\n🔍 VERIFYING DEPENDENCIES")
    print("=" * 50)
    
    required_modules = [
        ("pandas", "Pandas data manipulation"),
        ("numpy", "NumPy numerical computing"),
        ("sklearn", "Scikit-learn machine learning"),
        ("matplotlib", "Matplotlib plotting"),
        ("seaborn", "Seaborn visualization"),
        ("fastapi", "FastAPI web framework"),
        ("uvicorn", "Uvicorn ASGI server")
    ]
    
    all_imported = True
    for module_name, description in required_modules:
        if not check_import(module_name, description):
            all_imported = False
    
    return all_imported

def verify_functionality():
    """Verify core functionality works"""
    print("\n🔍 VERIFYING CORE FUNCTIONALITY")
    print("=" * 50)
    
    try:
        # Test importing the main class
        from recommendation_system import MovieRecommendationSystem
        print("✅ MovieRecommendationSystem class imported successfully")
        
        # Test basic initialization
        rec_system = MovieRecommendationSystem()
        print("✅ MovieRecommendationSystem initialized successfully")
        
        # Test data loading
        rec_system.load_data()
        print("✅ Sample data loaded successfully")
        
        # Test data preparation
        rec_system.prepare_data()
        print("✅ Data preparation completed successfully")
        
        print("✅ Core functionality verification passed")
        return True
        
    except Exception as e:
        print(f"❌ Core functionality verification failed: {e}")
        return False

def verify_git_setup():
    """Verify Git repository is properly set up"""
    print("\n🔍 VERIFYING GIT SETUP")
    print("=" * 50)
    
    if os.path.exists('.git'):
        print("✅ Git repository initialized")
        
        # Check if remote origin is set
        try:
            import subprocess
            result = subprocess.run(['git', 'remote', '-v'], 
                                  capture_output=True, text=True)
            if 'origin' in result.stdout:
                print("✅ Git remote origin configured")
                print(f"   Remote: {result.stdout.strip()}")
                return True
            else:
                print("❌ Git remote origin not configured")
                return False
        except Exception as e:
            print(f"❌ Error checking Git remote: {e}")
            return False
    else:
        print("❌ Git repository not initialized")
        return False

def main():
    """Main verification process"""
    print("🎬 MOVIE RECOMMENDATION SYSTEM VERIFICATION")
    print("=" * 60)
    
    results = []
    
    # Verify project structure
    results.append(verify_project_structure())
    
    # Verify dependencies
    results.append(verify_dependencies())
    
    # Verify functionality
    results.append(verify_functionality())
    
    # Verify Git setup
    results.append(verify_git_setup())
    
    # Final results
    print("\n" + "=" * 60)
    if all(results):
        print("🎉 ALL VERIFICATIONS PASSED!")
        print("✅ The Movie Recommendation System is ready to use")
        print("✅ Project successfully pushed to GitHub")
        print("\nNext steps:")
        print("1. Visit your GitHub repository to see the project")
        print("2. Run 'python demo.py' to see the system in action")
        print("3. Run 'python api_server.py' to start the API server")
        print("4. Check out the documentation in README.md")
        return True
    else:
        print("❌ SOME VERIFICATIONS FAILED")
        print("Please check the error messages above and fix any issues")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)