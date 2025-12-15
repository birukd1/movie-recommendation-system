"""
Movie Recommendation System
A comprehensive recommendation engine using collaborative filtering and matrix factorization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.decomposition import TruncatedSVD
import warnings
warnings.filterwarnings('ignore')

class MovieRecommendationSystem:
    """
    A comprehensive movie recommendation system implementing multiple approaches:
    - User-based collaborative filtering
    - Item-based collaborative filtering  
    - Matrix factorization using SVD
    """
    
    def __init__(self):
        self.ratings_df = None
        self.movies_df = None
        self.user_item_matrix = None
        self.user_similarity = None
        self.item_similarity = None
        self.svd_model = None
        self.user_encoder = {}
        self.item_encoder = {}
        self.user_decoder = {}
        self.item_decoder = {}
        
    def load_data(self, ratings_path=None, movies_path=None):
        """Load and prepare the MovieLens dataset"""
        if ratings_path and movies_path:
            self.ratings_df = pd.read_csv(ratings_path)
            self.movies_df = pd.read_csv(movies_path)
        else:
            # Create sample data for demonstration
            self._create_sample_data()
            
        print("Dataset loaded successfully!")
        print(f"Ratings shape: {self.ratings_df.shape}")
        print(f"Movies shape: {self.movies_df.shape}")
        
    def _create_sample_data(self):
        """Create sample MovieLens-style data for demonstration"""
        np.random.seed(42)
        
        # Generate sample ratings data
        n_users = 1000
        n_movies = 500
        n_ratings = 50000
        
        user_ids = np.random.randint(1, n_users + 1, n_ratings)
        movie_ids = np.random.randint(1, n_movies + 1, n_ratings)
        ratings = np.random.choice([1, 2, 3, 4, 5], n_ratings, p=[0.1, 0.1, 0.2, 0.3, 0.3])
        timestamps = np.random.randint(800000000, 1600000000, n_ratings)
        
        self.ratings_df = pd.DataFrame({
            'userId': user_ids,
            'movieId': movie_ids,
            'rating': ratings,
            'timestamp': timestamps
        })
        
        # Remove duplicates
        self.ratings_df = self.ratings_df.drop_duplicates(['userId', 'movieId'])
        
        # Generate sample movies data
        genres = ['Action', 'Comedy', 'Drama', 'Horror', 'Romance', 'Sci-Fi', 'Thriller']
        movie_titles = [f"Movie_{i}" for i in range(1, n_movies + 1)]
        movie_genres = [np.random.choice(genres) for _ in range(n_movies)]
        
        self.movies_df = pd.DataFrame({
            'movieId': range(1, n_movies + 1),
            'title': movie_titles,
            'genres': movie_genres
        })
        
    def explore_data(self):
        """Perform comprehensive exploratory data analysis"""
        print("=== DATA EXPLORATION ===")
        
        # Basic statistics
        print("\nRatings Dataset Info:")
        print(self.ratings_df.info())
        print("\nRatings Statistics:")
        print(self.ratings_df.describe())
        
        print("\nMovies Dataset Info:")
        print(self.movies_df.info())
        
        # Check for missing values
        print(f"\nMissing values in ratings: {self.ratings_df.isnull().sum().sum()}")
        print(f"Missing values in movies: {self.movies_df.isnull().sum().sum()}")
        
        # Rating distribution
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 3, 1)
        self.ratings_df['rating'].value_counts().sort_index().plot(kind='bar')
        plt.title('Rating Distribution')
        plt.xlabel('Rating')
        plt.ylabel('Count')
        
        plt.subplot(2, 3, 2)
        user_rating_counts = self.ratings_df['userId'].value_counts()
        plt.hist(user_rating_counts, bins=50, edgecolor='black')
        plt.title('Distribution of Ratings per User')
        plt.xlabel('Number of Ratings')
        plt.ylabel('Number of Users')
        
        plt.subplot(2, 3, 3)
        movie_rating_counts = self.ratings_df['movieId'].value_counts()
        plt.hist(movie_rating_counts, bins=50, edgecolor='black')
        plt.title('Distribution of Ratings per Movie')
        plt.xlabel('Number of Ratings')
        plt.ylabel('Number of Movies')
        
        plt.subplot(2, 3, 4)
        avg_ratings = self.ratings_df.groupby('movieId')['rating'].mean()
        plt.hist(avg_ratings, bins=30, edgecolor='black')
        plt.title('Distribution of Average Movie Ratings')
        plt.xlabel('Average Rating')
        plt.ylabel('Number of Movies')
        
        plt.subplot(2, 3, 5)
        avg_user_ratings = self.ratings_df.groupby('userId')['rating'].mean()
        plt.hist(avg_user_ratings, bins=30, edgecolor='black')
        plt.title('Distribution of Average User Ratings')
        plt.xlabel('Average Rating')
        plt.ylabel('Number of Users')
        
        plt.subplot(2, 3, 6)
        # Sparsity analysis
        n_users = self.ratings_df['userId'].nunique()
        n_movies = self.ratings_df['movieId'].nunique()
        n_ratings = len(self.ratings_df)
        sparsity = 1 - (n_ratings / (n_users * n_movies))
        
        plt.bar(['Filled', 'Empty'], [1-sparsity, sparsity])
        plt.title(f'Matrix Sparsity: {sparsity:.3f}')
        plt.ylabel('Proportion')
        
        plt.tight_layout()
        plt.savefig('eda_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\nDataset Statistics:")
        print(f"Number of users: {n_users}")
        print(f"Number of movies: {n_movies}")
        print(f"Number of ratings: {n_ratings}")
        print(f"Matrix sparsity: {sparsity:.3f}")
        
    def prepare_data(self):
        """Prepare data for recommendation algorithms"""
        print("=== DATA PREPARATION ===")
        
        # Filter users and movies with minimum interactions
        min_ratings_per_user = 5
        min_ratings_per_movie = 5
        
        # Filter users
        user_counts = self.ratings_df['userId'].value_counts()
        valid_users = user_counts[user_counts >= min_ratings_per_user].index
        self.ratings_df = self.ratings_df[self.ratings_df['userId'].isin(valid_users)]
        
        # Filter movies
        movie_counts = self.ratings_df['movieId'].value_counts()
        valid_movies = movie_counts[movie_counts >= min_ratings_per_movie].index
        self.ratings_df = self.ratings_df[self.ratings_df['movieId'].isin(valid_movies)]
        
        print(f"After filtering - Users: {self.ratings_df['userId'].nunique()}, Movies: {self.ratings_df['movieId'].nunique()}")
        
        # Create encoders for user and movie IDs
        unique_users = sorted(self.ratings_df['userId'].unique())
        unique_movies = sorted(self.ratings_df['movieId'].unique())
        
        self.user_encoder = {user: idx for idx, user in enumerate(unique_users)}
        self.item_encoder = {movie: idx for idx, movie in enumerate(unique_movies)}
        self.user_decoder = {idx: user for user, idx in self.user_encoder.items()}
        self.item_decoder = {idx: movie for movie, idx in self.item_encoder.items()}
        
        # Create user-item matrix
        self.user_item_matrix = self.ratings_df.pivot_table(
            index='userId', 
            columns='movieId', 
            values='rating'
        ).fillna(0)
        
        print(f"User-item matrix shape: {self.user_item_matrix.shape}")
        
    def calculate_similarities(self):
        """Calculate user-user and item-item similarities"""
        print("=== CALCULATING SIMILARITIES ===")
        
        # User-based similarity (cosine similarity)
        user_matrix = self.user_item_matrix.values
        self.user_similarity = cosine_similarity(user_matrix)
        
        # Item-based similarity
        item_matrix = self.user_item_matrix.T.values
        self.item_similarity = cosine_similarity(item_matrix)
        
        print("Similarity matrices calculated successfully!")
        print(f"User similarity matrix shape: {self.user_similarity.shape}")
        print(f"Item similarity matrix shape: {self.item_similarity.shape}")
    def user_based_recommendations(self, user_id, n_recommendations=10):
        """Generate recommendations using user-based collaborative filtering"""
        if user_id not in self.user_encoder:
            return []
            
        user_idx = self.user_encoder[user_id]
        user_ratings = self.user_item_matrix.iloc[user_idx].values
        
        # Find similar users
        user_similarities = self.user_similarity[user_idx]
        similar_users_idx = np.argsort(user_similarities)[::-1][1:51]  # Top 50 similar users
        
        # Calculate weighted ratings for unrated movies
        recommendations = {}
        
        for movie_idx in range(len(self.user_item_matrix.columns)):
            if user_ratings[movie_idx] == 0:  # User hasn't rated this movie
                weighted_sum = 0
                similarity_sum = 0
                
                for similar_user_idx in similar_users_idx:
                    similar_user_rating = self.user_item_matrix.iloc[similar_user_idx, movie_idx]
                    if similar_user_rating > 0:
                        similarity = user_similarities[similar_user_idx]
                        weighted_sum += similarity * similar_user_rating
                        similarity_sum += abs(similarity)
                
                if similarity_sum > 0:
                    predicted_rating = weighted_sum / similarity_sum
                    movie_id = self.user_item_matrix.columns[movie_idx]
                    recommendations[movie_id] = predicted_rating
        
        # Sort and return top N recommendations
        sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
        return sorted_recommendations[:n_recommendations]
    
    def item_based_recommendations(self, user_id, n_recommendations=10):
        """Generate recommendations using item-based collaborative filtering"""
        if user_id not in self.user_encoder:
            return []
            
        user_idx = self.user_encoder[user_id]
        user_ratings = self.user_item_matrix.iloc[user_idx].values
        
        recommendations = {}
        
        for movie_idx in range(len(self.user_item_matrix.columns)):
            if user_ratings[movie_idx] == 0:  # User hasn't rated this movie
                weighted_sum = 0
                similarity_sum = 0
                
                # Find movies the user has rated
                rated_movies_idx = np.where(user_ratings > 0)[0]
                
                for rated_movie_idx in rated_movies_idx:
                    similarity = self.item_similarity[movie_idx, rated_movie_idx]
                    if similarity > 0:
                        weighted_sum += similarity * user_ratings[rated_movie_idx]
                        similarity_sum += abs(similarity)
                
                if similarity_sum > 0:
                    predicted_rating = weighted_sum / similarity_sum
                    movie_id = self.user_item_matrix.columns[movie_idx]
                    recommendations[movie_id] = predicted_rating
        
        # Sort and return top N recommendations
        sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
        return sorted_recommendations[:n_recommendations]
    
    def train_svd_model(self, n_components=50):
        """Train SVD model using scikit-learn's TruncatedSVD"""
        print("=== TRAINING SVD MODEL ===")
        
        # Create user-item matrix for SVD
        user_item_dense = self.user_item_matrix.values
        
        # Apply SVD
        self.svd_model = TruncatedSVD(n_components=n_components, random_state=42)
        user_factors = self.svd_model.fit_transform(user_item_dense)
        item_factors = self.svd_model.components_
        
        # Reconstruct the matrix
        self.reconstructed_matrix = np.dot(user_factors, item_factors)
        
        # Calculate RMSE and MAE on known ratings
        known_mask = user_item_dense > 0
        known_ratings = user_item_dense[known_mask]
        predicted_ratings = self.reconstructed_matrix[known_mask]
        
        # Clip predictions to valid rating range
        predicted_ratings = np.clip(predicted_ratings, 1, 5)
        
        rmse = np.sqrt(mean_squared_error(known_ratings, predicted_ratings))
        mae = mean_absolute_error(known_ratings, predicted_ratings)
        
        print(f"SVD Components: {n_components}")
        print(f"Explained Variance Ratio: {self.svd_model.explained_variance_ratio_.sum():.4f}")
        print(f"Training RMSE: {rmse:.4f}")
        print(f"Training MAE: {mae:.4f}")
        
        return rmse, mae
    
    def svd_recommendations(self, user_id, n_recommendations=10):
        """Generate recommendations using SVD model"""
        if not hasattr(self, 'reconstructed_matrix') or user_id not in self.user_encoder:
            return []
        
        user_idx = self.user_encoder[user_id]
        user_ratings = self.user_item_matrix.iloc[user_idx].values
        predicted_ratings = self.reconstructed_matrix[user_idx]
        
        # Get recommendations for unrated movies
        recommendations = []
        for movie_idx in range(len(predicted_ratings)):
            if user_ratings[movie_idx] == 0:  # User hasn't rated this movie
                movie_id = self.user_item_matrix.columns[movie_idx]
                predicted_rating = np.clip(predicted_ratings[movie_idx], 1, 5)
                recommendations.append((movie_id, predicted_rating))
        
        # Sort and return top N recommendations
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:n_recommendations]
    
    def evaluate_models(self):
        """Compare performance of different recommendation approaches"""
        print("=== MODEL EVALUATION ===")
        
        # Split data for evaluation
        train_df, test_df = train_test_split(self.ratings_df, test_size=0.2, random_state=42)
        
        # Create test user-item matrix
        test_matrix = test_df.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)
        
        results = {}
        
        # Evaluate SVD model
        if hasattr(self, 'reconstructed_matrix'):
            svd_predictions = []
            svd_actuals = []
            
            for _, row in test_df.iterrows():
                if row['userId'] in self.user_encoder and row['movieId'] in self.user_item_matrix.columns:
                    user_idx = self.user_encoder[row['userId']]
                    movie_idx = list(self.user_item_matrix.columns).index(row['movieId'])
                    pred = np.clip(self.reconstructed_matrix[user_idx, movie_idx], 1, 5)
                    svd_predictions.append(pred)
                    svd_actuals.append(row['rating'])
            
            if svd_predictions:
                svd_rmse = np.sqrt(mean_squared_error(svd_actuals, svd_predictions))
                svd_mae = mean_absolute_error(svd_actuals, svd_predictions)
                results['SVD'] = {'RMSE': svd_rmse, 'MAE': svd_mae}
        
        # Create comparison visualization
        if results:
            plt.figure(figsize=(12, 5))
            
            models = list(results.keys())
            rmse_scores = [results[model]['RMSE'] for model in models]
            mae_scores = [results[model]['MAE'] for model in models]
            
            x = np.arange(len(models))
            width = 0.35
            
            plt.subplot(1, 2, 1)
            plt.bar(x, rmse_scores, width, label='RMSE')
            plt.xlabel('Models')
            plt.ylabel('RMSE')
            plt.title('Model Comparison - RMSE')
            plt.xticks(x, models)
            
            plt.subplot(1, 2, 2)
            plt.bar(x, mae_scores, width, label='MAE', color='orange')
            plt.xlabel('Models')
            plt.ylabel('MAE')
            plt.title('Model Comparison - MAE')
            plt.xticks(x, models)
            
            plt.tight_layout()
            plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        return results
    
    def get_recommendations(self, user_id, method='svd', n_recommendations=10):
        """Get recommendations using specified method"""
        if method == 'user_based':
            return self.user_based_recommendations(user_id, n_recommendations)
        elif method == 'item_based':
            return self.item_based_recommendations(user_id, n_recommendations)
        elif method == 'svd':
            return self.svd_recommendations(user_id, n_recommendations)
        else:
            raise ValueError("Method must be 'user_based', 'item_based', or 'svd'")
    
    def display_recommendations(self, user_id, method='svd', n_recommendations=10):
        """Display recommendations with movie titles"""
        recommendations = self.get_recommendations(user_id, method, n_recommendations)
        
        print(f"\n=== TOP {n_recommendations} RECOMMENDATIONS FOR USER {user_id} ({method.upper()}) ===")
        
        if not recommendations:
            print("No recommendations available for this user.")
            return
        
        for i, (movie_id, predicted_rating) in enumerate(recommendations, 1):
            movie_title = self.movies_df[self.movies_df['movieId'] == movie_id]['title'].iloc[0]
            print(f"{i}. {movie_title} (Predicted Rating: {predicted_rating:.2f})")
    
    def analyze_cold_start(self):
        """Analyze and address cold start problem"""
        print("=== COLD START ANALYSIS ===")
        
        # Identify new users (users with very few ratings)
        user_rating_counts = self.ratings_df['userId'].value_counts()
        new_users = user_rating_counts[user_rating_counts <= 5].index
        
        print(f"Number of users with ≤5 ratings: {len(new_users)}")
        print(f"Percentage of cold start users: {len(new_users)/len(user_rating_counts)*100:.2f}%")
        
        # Popular movies recommendation for cold start
        popular_movies = self.ratings_df.groupby('movieId').agg({
            'rating': ['mean', 'count']
        }).round(2)
        popular_movies.columns = ['avg_rating', 'rating_count']
        popular_movies = popular_movies[popular_movies['rating_count'] >= 50]
        popular_movies = popular_movies.sort_values(['avg_rating', 'rating_count'], ascending=False)
        
        print("\nTop 10 Popular Movies for Cold Start Users:")
        for i, (movie_id, row) in enumerate(popular_movies.head(10).iterrows(), 1):
            movie_title = self.movies_df[self.movies_df['movieId'] == movie_id]['title'].iloc[0]
            print(f"{i}. {movie_title} (Avg: {row['avg_rating']:.2f}, Count: {row['rating_count']})")
        
        return popular_movies.head(20).index.tolist()
    
    def save_model(self, filepath='recommendation_model.pkl'):
        """Save the trained model"""
        import pickle
        
        model_data = {
            'user_item_matrix': self.user_item_matrix,
            'user_similarity': self.user_similarity,
            'item_similarity': self.item_similarity,
            'svd_model': self.svd_model,
            'reconstructed_matrix': getattr(self, 'reconstructed_matrix', None),
            'user_encoder': self.user_encoder,
            'item_encoder': self.item_encoder,
            'user_decoder': self.user_decoder,
            'item_decoder': self.item_decoder,
            'movies_df': self.movies_df
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='recommendation_model.pkl'):
        """Load a trained model"""
        import pickle
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.user_item_matrix = model_data['user_item_matrix']
        self.user_similarity = model_data['user_similarity']
        self.item_similarity = model_data['item_similarity']
        self.svd_model = model_data['svd_model']
        self.reconstructed_matrix = model_data.get('reconstructed_matrix')
        self.user_encoder = model_data['user_encoder']
        self.item_encoder = model_data['item_encoder']
        self.user_decoder = model_data['user_decoder']
        self.item_decoder = model_data['item_decoder']
        self.movies_df = model_data['movies_df']
        
        print(f"Model loaded from {filepath}")


def main():
    """Main execution function"""
    print("🎬 Movie Recommendation System")
    print("=" * 50)
    
    # Initialize the recommendation system
    rec_system = MovieRecommendationSystem()
    
    # Load data (using sample data for demonstration)
    rec_system.load_data()
    
    # Explore the data
    rec_system.explore_data()
    
    # Prepare data for modeling
    rec_system.prepare_data()
    
    # Calculate similarities for collaborative filtering
    rec_system.calculate_similarities()
    
    # Train SVD model
    rec_system.train_svd_model()
    
    # Evaluate models
    results = rec_system.evaluate_models()
    
    # Generate sample recommendations
    sample_users = list(rec_system.user_encoder.keys())[:5]
    
    for user_id in sample_users:
        print(f"\n{'='*60}")
        rec_system.display_recommendations(user_id, 'svd', 5)
        rec_system.display_recommendations(user_id, 'user_based', 5)
        rec_system.display_recommendations(user_id, 'item_based', 5)
    
    # Analyze cold start problem
    popular_movies = rec_system.analyze_cold_start()
    
    # Save the model
    rec_system.save_model()
    
    print("\n🎯 Recommendation system training completed successfully!")
    print("Model saved and ready for deployment.")


if __name__ == "__main__":
    main()