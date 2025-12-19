# 🎬 Movie Recommendation System

A production-ready movie recommendation engine implementing multiple collaborative filtering approaches and matrix factorization techniques. Built with Python and designed for scalability and real-world deployment.

##  Features

- **Multiple Recommendation Algorithms**
  - User-Based Collaborative Filtering (Cosine Similarity)
  - Item-Based Collaborative Filtering
  - Matrix Factorization using SVD (Singular Value Decomposition)

- **Production Ready**
  - FastAPI REST API server
  - Model persistence and loading
  - Comprehensive error handling
  - Cold start problem handling

- **Comprehensive Analysis**
  - Exploratory Data Analysis (EDA)
  - Model performance evaluation
  - Hyperparameter tuning
  - Visualization and insights

##  Tech Stack

- **Python 3.8+**
- **Core Libraries:**
  - Pandas & NumPy (Data manipulation)
  - Scikit-learn (Machine learning utilities)
  - Surprise (Collaborative filtering)
  - Matplotlib & Seaborn (Visualization)
- **API Framework:**
  - FastAPI (REST API)
  - Uvicorn (ASGI server)

##  How It Works

### 1. Data Processing
- Loads MovieLens-style dataset (or generates sample data)
- Handles missing values and data cleaning
- Creates user-item interaction matrix
- Filters users/items with minimum interactions

### 2. Collaborative Filtering
- **User-Based:** Finds similar users based on rating patterns
- **Item-Based:** Recommends items similar to previously liked items
- Uses cosine similarity for measuring user/item relationships

### 3. Matrix Factorization
- Implements SVD to decompose user-item matrix
- Learns latent factors representing user preferences and item characteristics
- Handles sparse data effectively

### 4. Evaluation & Optimization
- Train/test split for model validation
- RMSE and MAE metrics for performance measurement
- Hyperparameter tuning using grid search
- Cross-validation for robust evaluation

##  Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/birukd1/movie-recommendation-system.git
cd movie-recommendation-system

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from recommendation_system import MovieRecommendationSystem

# Initialize the system
rec_system = MovieRecommendationSystem()

# Load data and train models
rec_system.load_data()
rec_system.prepare_data()
rec_system.calculate_similarities()
rec_system.train_svd_model()

# Get recommendations for a user
recommendations = rec_system.get_recommendations(user_id=1, method='svd', n_recommendations=10)
rec_system.display_recommendations(user_id=1, method='svd')
```

### Run Demo

```bash
python demo.py
```

### Start API Server

```bash
python api_server.py
```

##  API Endpoints

- `GET /` - API information and available endpoints
- `POST /recommendations` - Get personalized recommendations
- `GET /user/{user_id}/recommendations` - Get recommendations for specific user
- `GET /users` - List available user IDs
- `GET /movies/popular` - Get popular movies for cold start users
- `GET /health` - Health check endpoint

### Example API Usage

```bash
# Get recommendations for user 1
curl -X GET "http://localhost:8000/user/1/recommendations?method=svd&n_recommendations=5"

# Get popular movies
curl -X GET "http://localhost:8000/movies/popular?limit=10"
```

##  Performance Results

The system achieves competitive performance across different metrics:

- **SVD Model:** RMSE ~0.87, MAE ~0.67
- **User-Based CF:** Effective for users with sufficient rating history
- **Item-Based CF:** Good for discovering similar items

##  Cold Start Problem

The system addresses the cold start problem through:

- **Popular Item Recommendations:** For new users with no rating history
- **Minimum Interaction Filtering:** Ensures reliable similarity calculations
- **Hybrid Approach:** Combines multiple methods for robust recommendations

## 🏗️ Project Structure

```
movie-recommendation-system/
├── recommendation_system.py    # Main recommendation engine
├── api_server.py              # FastAPI REST API server
├── demo.py                    # Comprehensive demo script
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
└── generated files:
    ├── eda_analysis.png       # EDA visualizations
    ├── model_comparison.png   # Model performance comparison
    └── recommendation_model.pkl # Trained model file
```

##  Configuration & Customization

### Using Real MovieLens Data

```python
# Download MovieLens dataset and specify paths
rec_system.load_data(
    ratings_path='path/to/ratings.csv',
    movies_path='path/to/movies.csv'
)
```

### Hyperparameter Tuning

The system automatically tunes SVD hyperparameters:
- `n_factors`: Number of latent factors (50, 100, 150)
- `lr_all`: Learning rate (0.005, 0.01, 0.02)
- `reg_all`: Regularization (0.02, 0.1, 0.2)

### Similarity Thresholds

Adjust similarity calculations by modifying:
- Minimum ratings per user/item
- Number of similar users/items to consider
- Similarity calculation methods

##  Deployment Considerations

### Scalability
- **Database Integration:** Replace in-memory matrices with database storage
- **Caching:** Implement Redis for frequently accessed recommendations
- **Batch Processing:** Pre-compute recommendations for active users

### Production Optimizations
- **Model Updates:** Implement incremental learning for new ratings
- **A/B Testing:** Framework for testing different recommendation strategies
- **Monitoring:** Track recommendation quality and user engagement metrics

### Infrastructure
- **Containerization:** Docker support for easy deployment
- **Load Balancing:** Handle multiple concurrent requests
- **Model Versioning:** Track and manage different model versions

##  Evaluation Metrics

- **RMSE (Root Mean Square Error):** Measures prediction accuracy
- **MAE (Mean Absolute Error):** Average prediction error
- **Precision@K:** Relevance of top-K recommendations
- **Recall@K:** Coverage of relevant items in top-K
- **Diversity:** Variety in recommended items

##  Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

##  Acknowledgments

- MovieLens dataset for providing real-world movie rating data
- Surprise library for collaborative filtering implementations
- FastAPI for the excellent web framework
- The open-source community for inspiration and tools

##  Contact

**Biruk D** - [GitHub](https://github.com/birukd1)
