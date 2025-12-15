"""
Test suite for the Movie Recommendation System
Validates functionality and performance of all components
"""

import unittest
import numpy as np
import pandas as pd
from recommendation_system import MovieRecommendationSystem
import tempfile
import os

class TestMovieRecommendationSystem(unittest.TestCase):
    """Test cases for the recommendation system"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.rec_system = MovieRecommendationSystem()
        self.rec_system.load_data()  # Uses sample data
        self.rec_system.prepare_data()
        self.rec_system.calculate_similarities()
        
    def test_data_loading(self):
        """Test data loading functionality"""
        self.assertIsNotNone(self.rec_system.ratings_df)
        self.assertIsNotNone(self.rec_system.movies_df)
        self.assertGreater(len(self.rec_system.ratings_df), 0)
        self.assertGreater(len(self.rec_system.movies_df), 0)
        
    def test_data_preparation(self):
        """Test data preparation and encoding"""
        self.assertIsNotNone(self.rec_system.user_item_matrix)
        self.assertIsNotNone(self.rec_system.user_encoder)
        self.assertIsNotNone(self.rec_system.item_encoder)
        self.assertGreater(len(self.rec_system.user_encoder), 0)
        self.assertGreater(len(self.rec_system.item_encoder), 0)
        
    def test_similarity_calculation(self):
        """Test similarity matrix calculations"""
        self.assertIsNotNone(self.rec_system.user_similarity)
        self.assertIsNotNone(self.rec_system.item_similarity)
        
        # Check matrix dimensions
        n_users = len(self.rec_system.user_encoder)
        n_items = len(self.rec_system.item_encoder)
        
        self.assertEqual(self.rec_system.user_similarity.shape, (n_users, n_users))
        self.assertEqual(self.rec_system.item_similarity.shape, (n_items, n_items))
        
    def test_svd_training(self):
        """Test SVD model training"""
        rmse, mae = self.rec_system.train_svd_model()
        
        self.assertIsNotNone(self.rec_system.svd_model)
        self.assertIsInstance(rmse, float)
        self.assertIsInstance(mae, float)
        self.assertGreater(rmse, 0)
        self.assertGreater(mae, 0)
        
    def test_user_based_recommendations(self):
        """Test user-based collaborative filtering"""
        test_user = list(self.rec_system.user_encoder.keys())[0]
        recommendations = self.rec_system.user_based_recommendations(test_user, 5)
        
        self.assertIsInstance(recommendations, list)
        self.assertLessEqual(len(recommendations), 5)
        
        if recommendations:
            movie_id, rating = recommendations[0]
            self.assertIsInstance(movie_id, (int, np.integer))
            self.assertIsInstance(rating, (float, np.floating))
            
    def test_item_based_recommendations(self):
        """Test item-based collaborative filtering"""
        test_user = list(self.rec_system.user_encoder.keys())[0]
        recommendations = self.rec_system.item_based_recommendations(test_user, 5)
        
        self.assertIsInstance(recommendations, list)
        self.assertLessEqual(len(recommendations), 5)
        
    def test_svd_recommendations(self):
        """Test SVD-based recommendations"""
        # Train SVD model first
        self.rec_system.train_svd_model()
        
        test_user = list(self.rec_system.user_encoder.keys())[0]
        recommendations = self.rec_system.svd_recommendations(test_user, 5)
        
        self.assertIsInstance(recommendations, list)
        self.assertLessEqual(len(recommendations), 5)
        
    def test_get_recommendations_method(self):
        """Test the unified get_recommendations method"""
        self.rec_system.train_svd_model()
        test_user = list(self.rec_system.user_encoder.keys())[0]
        
        # Test all methods
        methods = ['user_based', 'item_based', 'svd']
        
        for method in methods:
            recommendations = self.rec_system.get_recommendations(test_user, method, 3)
            self.assertIsInstance(recommendations, list)
            self.assertLessEqual(len(recommendations), 3)
            
    def test_invalid_method(self):
        """Test error handling for invalid recommendation method"""
        test_user = list(self.rec_system.user_encoder.keys())[0]
        
        with self.assertRaises(ValueError):
            self.rec_system.get_recommendations(test_user, 'invalid_method')
            
    def test_nonexistent_user(self):
        """Test handling of non-existent user"""
        nonexistent_user = 999999
        recommendations = self.rec_system.user_based_recommendations(nonexistent_user)
        self.assertEqual(recommendations, [])
        
    def test_cold_start_analysis(self):
        """Test cold start problem analysis"""
        popular_movies = self.rec_system.analyze_cold_start()
        
        self.assertIsInstance(popular_movies, list)
        self.assertGreater(len(popular_movies), 0)
        
    def test_model_persistence(self):
        """Test model saving and loading"""
        # Train the model
        self.rec_system.train_svd_model()
        
        # Save model to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_file:
            temp_path = tmp_file.name
            
        try:
            # Save model
            self.rec_system.save_model(temp_path)
            self.assertTrue(os.path.exists(temp_path))
            
            # Create new instance and load model
            new_rec_system = MovieRecommendationSystem()
            new_rec_system.load_model(temp_path)
            
            # Verify loaded model
            self.assertIsNotNone(new_rec_system.svd_model)
            self.assertIsNotNone(new_rec_system.user_encoder)
            self.assertIsNotNone(new_rec_system.item_encoder)
            
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                
    def test_evaluation_metrics(self):
        """Test model evaluation functionality"""
        self.rec_system.train_svd_model()
        results = self.rec_system.evaluate_models()
        
        self.assertIsInstance(results, dict)
        if 'SVD' in results:
            self.assertIn('RMSE', results['SVD'])
            self.assertIn('MAE', results['SVD'])
            
    def test_data_quality(self):
        """Test data quality and consistency"""
        # Check for required columns
        required_rating_cols = ['userId', 'movieId', 'rating']
        required_movie_cols = ['movieId', 'title']
        
        for col in required_rating_cols:
            self.assertIn(col, self.rec_system.ratings_df.columns)
            
        for col in required_movie_cols:
            self.assertIn(col, self.rec_system.movies_df.columns)
            
        # Check rating range
        ratings = self.rec_system.ratings_df['rating']
        self.assertTrue(ratings.min() >= 1)
        self.assertTrue(ratings.max() <= 5)
        
    def test_matrix_properties(self):
        """Test user-item matrix properties"""
        matrix = self.rec_system.user_item_matrix
        
        # Check matrix is not empty
        self.assertGreater(matrix.shape[0], 0)
        self.assertGreater(matrix.shape[1], 0)
        
        # Check values are in valid range
        non_zero_values = matrix[matrix > 0]
        if len(non_zero_values) > 0:
            self.assertTrue((non_zero_values >= 1).all())
            self.assertTrue((non_zero_values <= 5).all())


class TestAPIFunctionality(unittest.TestCase):
    """Test API-related functionality"""
    
    def setUp(self):
        """Set up test fixtures for API tests"""
        self.rec_system = MovieRecommendationSystem()
        self.rec_system.load_data()
        self.rec_system.prepare_data()
        self.rec_system.calculate_similarities()
        self.rec_system.train_svd_model()
        
    def test_recommendation_formatting(self):
        """Test recommendation output formatting for API"""
        test_user = list(self.rec_system.user_encoder.keys())[0]
        recommendations = self.rec_system.get_recommendations(test_user, 'svd', 3)
        
        # Format for API response
        formatted_recs = []
        for movie_id, predicted_rating in recommendations:
            movie_info = self.rec_system.movies_df[
                self.rec_system.movies_df['movieId'] == movie_id
            ]
            if not movie_info.empty:
                formatted_recs.append({
                    "movie_id": int(movie_id),
                    "title": movie_info.iloc[0]['title'],
                    "predicted_rating": round(float(predicted_rating), 2)
                })
                
        self.assertIsInstance(formatted_recs, list)
        if formatted_recs:
            rec = formatted_recs[0]
            self.assertIn('movie_id', rec)
            self.assertIn('title', rec)
            self.assertIn('predicted_rating', rec)


def run_performance_benchmark():
    """Run performance benchmarks"""
    print("\n🚀 PERFORMANCE BENCHMARK")
    print("-" * 40)
    
    import time
    
    rec_system = MovieRecommendationSystem()
    
    # Benchmark data loading
    start_time = time.time()
    rec_system.load_data()
    load_time = time.time() - start_time
    print(f"Data loading: {load_time:.3f} seconds")
    
    # Benchmark data preparation
    start_time = time.time()
    rec_system.prepare_data()
    prep_time = time.time() - start_time
    print(f"Data preparation: {prep_time:.3f} seconds")
    
    # Benchmark similarity calculation
    start_time = time.time()
    rec_system.calculate_similarities()
    sim_time = time.time() - start_time
    print(f"Similarity calculation: {sim_time:.3f} seconds")
    
    # Benchmark SVD training
    start_time = time.time()
    rec_system.train_svd_model()
    svd_time = time.time() - start_time
    print(f"SVD training: {svd_time:.3f} seconds")
    
    # Benchmark recommendation generation
    test_user = list(rec_system.user_encoder.keys())[0]
    
    start_time = time.time()
    for _ in range(10):
        rec_system.get_recommendations(test_user, 'svd', 10)
    rec_time = (time.time() - start_time) / 10
    print(f"Average recommendation time: {rec_time:.4f} seconds")
    
    print(f"\nTotal benchmark time: {load_time + prep_time + sim_time + svd_time:.3f} seconds")


if __name__ == '__main__':
    # Run unit tests
    print("🧪 RUNNING UNIT TESTS")
    print("=" * 50)
    unittest.main(verbosity=2, exit=False)
    
    # Run performance benchmark
    run_performance_benchmark()
    
    print("\n✅ ALL TESTS COMPLETED!")