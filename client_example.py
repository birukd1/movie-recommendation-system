"""
Example client to test the Movie Recommendation API
Demonstrates how to interact with the recommendation system
"""

import requests
import json

# API base URL
BASE_URL = "http://localhost:8000"

def test_api_endpoints():
    """Test all API endpoints"""
    print("🧪 TESTING MOVIE RECOMMENDATION API")
    print("=" * 50)
    
    try:
        # Test root endpoint
        print("\n1. Testing root endpoint...")
        response = requests.get(f"{BASE_URL}/")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        # Test health check
        print("\n2. Testing health check...")
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        # Get available users
        print("\n3. Getting available users...")
        response = requests.get(f"{BASE_URL}/users")
        users_data = response.json()
        print(f"Status: {response.status_code}")
        print(f"Total users: {users_data['total_users']}")
        print(f"Sample user IDs: {users_data['user_ids'][:10]}")
        
        # Test recommendations for a specific user
        test_user_id = users_data['user_ids'][0]
        print(f"\n4. Getting recommendations for user {test_user_id}...")
        
        # Test SVD recommendations
        response = requests.get(f"{BASE_URL}/user/{test_user_id}/recommendations?method=svd&n_recommendations=5")
        print(f"SVD Recommendations - Status: {response.status_code}")
        if response.status_code == 200:
            recs = response.json()
            print(f"Method: {recs['method']}")
            print("Top 5 SVD Recommendations:")
            for i, rec in enumerate(recs['recommendations'], 1):
                print(f"  {i}. {rec['title']} (Rating: {rec['predicted_rating']})")
        
        # Test User-based CF recommendations
        response = requests.get(f"{BASE_URL}/user/{test_user_id}/recommendations?method=user_based&n_recommendations=5")
        print(f"\nUser-based CF Recommendations - Status: {response.status_code}")
        if response.status_code == 200:
            recs = response.json()
            print("Top 5 User-based CF Recommendations:")
            for i, rec in enumerate(recs['recommendations'], 1):
                print(f"  {i}. {rec['title']} (Rating: {rec['predicted_rating']})")
        
        # Test Item-based CF recommendations
        response = requests.get(f"{BASE_URL}/user/{test_user_id}/recommendations?method=item_based&n_recommendations=5")
        print(f"\nItem-based CF Recommendations - Status: {response.status_code}")
        if response.status_code == 200:
            recs = response.json()
            print("Top 5 Item-based CF Recommendations:")
            for i, rec in enumerate(recs['recommendations'], 1):
                print(f"  {i}. {rec['title']} (Rating: {rec['predicted_rating']})")
        
        # Test POST endpoint
        print(f"\n5. Testing POST recommendations endpoint...")
        payload = {
            "user_id": test_user_id,
            "method": "svd",
            "n_recommendations": 3
        }
        response = requests.post(f"{BASE_URL}/recommendations", json=payload)
        print(f"POST Recommendations - Status: {response.status_code}")
        if response.status_code == 200:
            recs = response.json()
            print("Top 3 POST Recommendations:")
            for i, rec in enumerate(recs['recommendations'], 1):
                print(f"  {i}. {rec['title']} (Rating: {rec['predicted_rating']})")
        
        # Test popular movies
        print(f"\n6. Getting popular movies...")
        response = requests.get(f"{BASE_URL}/movies/popular?limit=5")
        print(f"Popular Movies - Status: {response.status_code}")
        if response.status_code == 200:
            popular = response.json()
            print("Top 5 Popular Movies:")
            for i, movie in enumerate(popular['popular_movies'], 1):
                print(f"  {i}. {movie['title']} ({movie['genres']})")
        
        print("\n✅ API testing completed successfully!")
        
    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to the API server.")
        print("Make sure the server is running with: python api_server.py")
    except Exception as e:
        print(f"❌ Error: {str(e)}")

def benchmark_api_performance():
    """Benchmark API response times"""
    print("\n🚀 API PERFORMANCE BENCHMARK")
    print("-" * 40)
    
    import time
    
    try:
        # Get a test user
        response = requests.get(f"{BASE_URL}/users")
        test_user_id = response.json()['user_ids'][0]
        
        # Benchmark recommendation generation
        methods = ['svd', 'user_based', 'item_based']
        
        for method in methods:
            times = []
            for _ in range(10):
                start_time = time.time()
                response = requests.get(f"{BASE_URL}/user/{test_user_id}/recommendations?method={method}&n_recommendations=10")
                end_time = time.time()
                if response.status_code == 200:
                    times.append(end_time - start_time)
            
            if times:
                avg_time = sum(times) / len(times)
                print(f"{method.upper()} method: {avg_time:.4f} seconds (avg)")
        
    except Exception as e:
        print(f"❌ Benchmark error: {str(e)}")

if __name__ == "__main__":
    test_api_endpoints()
    benchmark_api_performance()