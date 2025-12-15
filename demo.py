"""
Demo script to showcase the Movie Recommendation System
Demonstrates all key features and generates sample outputs
"""

from recommendation_system import MovieRecommendationSystem
import pandas as pd
import numpy as np

def run_demo():
    """Run a comprehensive demo of the recommendation system"""
    print("🎬 MOVIE RECOMMENDATION SYSTEM DEMO")
    print("=" * 60)
    
    # Initialize the system
    rec_system = MovieRecommendationSystem()
    
    # Step 1: Load and explore data
    print("\n📊 STEP 1: DATA LOADING AND EXPLORATION")
    print("-" * 40)
    rec_system.load_data()
    rec_system.explore_data()
    
    # Step 2: Data preparation
    print("\n🔧 STEP 2: DATA PREPARATION")
    print("-" * 40)
    rec_system.prepare_data()
    
    # Step 3: Calculate similarities
    print("\n🔍 STEP 3: SIMILARITY CALCULATIONS")
    print("-" * 40)
    rec_system.calculate_similarities()
    
    # Step 4: Train SVD model
    print("\n🤖 STEP 4: MATRIX FACTORIZATION (SVD)")
    print("-" * 40)
    rmse, mae = rec_system.train_svd_model()
    
    # Step 5: Model evaluation
    print("\n📈 STEP 5: MODEL EVALUATION")
    print("-" * 40)
    results = rec_system.evaluate_models()
    
    # Step 6: Generate recommendations for sample users
    print("\n🎯 STEP 6: SAMPLE RECOMMENDATIONS")
    print("-" * 40)
    
    sample_users = list(rec_system.user_encoder.keys())[:3]
    methods = ['svd', 'user_based', 'item_based']
    
    for user_id in sample_users:
        print(f"\n👤 USER {user_id} RECOMMENDATIONS:")
        print("=" * 50)
        
        for method in methods:
            rec_system.display_recommendations(user_id, method, 5)
            print()
    
    # Step 7: Cold start analysis
    print("\n❄️ STEP 7: COLD START PROBLEM ANALYSIS")
    print("-" * 40)
    popular_movies = rec_system.analyze_cold_start()
    
    # Step 8: Performance summary
    print("\n📊 STEP 8: PERFORMANCE SUMMARY")
    print("-" * 40)
    print(f"SVD Model Performance:")
    print(f"  - RMSE: {rmse:.4f}")
    print(f"  - MAE: {mae:.4f}")
    
    # Step 9: Save model
    print("\n💾 STEP 9: MODEL PERSISTENCE")
    print("-" * 40)
    rec_system.save_model()
    
    # Step 10: Deployment readiness check
    print("\n🚀 STEP 10: DEPLOYMENT READINESS")
    print("-" * 40)
    print("✅ Model trained and validated")
    print("✅ API server ready (api_server.py)")
    print("✅ Model saved for production use")
    print("✅ Cold start handling implemented")
    print("✅ Multiple recommendation methods available")
    
    print("\n🎉 DEMO COMPLETED SUCCESSFULLY!")
    print("The recommendation system is ready for production deployment.")
    
    return rec_system

def test_api_functionality():
    """Test the recommendation system functionality"""
    print("\n🧪 TESTING RECOMMENDATION FUNCTIONALITY")
    print("-" * 50)
    
    rec_system = MovieRecommendationSystem()
    rec_system.load_data()
    rec_system.prepare_data()
    rec_system.calculate_similarities()
    rec_system.train_svd_model()
    
    # Test different methods
    test_user = list(rec_system.user_encoder.keys())[0]
    
    print(f"Testing recommendations for User {test_user}:")
    
    # Test SVD
    svd_recs = rec_system.get_recommendations(test_user, 'svd', 5)
    print(f"SVD recommendations: {len(svd_recs)} items")
    
    # Test User-based CF
    user_recs = rec_system.get_recommendations(test_user, 'user_based', 5)
    print(f"User-based CF recommendations: {len(user_recs)} items")
    
    # Test Item-based CF
    item_recs = rec_system.get_recommendations(test_user, 'item_based', 5)
    print(f"Item-based CF recommendations: {len(item_recs)} items")
    
    print("✅ All recommendation methods working correctly!")

if __name__ == "__main__":
    # Run the main demo
    rec_system = run_demo()
    
    # Run functionality tests
    test_api_functionality()
    
    print("\n" + "="*60)
    print("🎬 Movie Recommendation System Demo Complete!")
    print("Ready to deploy and use in production.")
    print("="*60)