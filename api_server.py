"""
FastAPI server for the Movie Recommendation System
Provides REST API endpoints for getting recommendations
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from recommendation_system import MovieRecommendationSystem
import os

app = FastAPI(
    title="Movie Recommendation API",
    description="A production-ready movie recommendation system API",
    version="1.0.0"
)

# Global recommendation system instance
rec_system = None

class RecommendationRequest(BaseModel):
    user_id: int
    method: str = "svd"
    n_recommendations: int = 10

class RecommendationResponse(BaseModel):
    user_id: int
    method: str
    recommendations: List[dict]

@app.on_event("startup")
async def startup_event():
    """Initialize the recommendation system on startup"""
    global rec_system
    rec_system = MovieRecommendationSystem()
    
    # Check if model exists, otherwise train a new one
    if os.path.exists('recommendation_model.pkl'):
        rec_system.load_model()
        print("Loaded existing model")
    else:
        print("Training new model...")
        rec_system.load_data()
        rec_system.prepare_data()
        rec_system.calculate_similarities()
        rec_system.train_svd_model()
        rec_system.save_model()
        print("Model trained and saved")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Movie Recommendation System API",
        "version": "1.0.0",
        "endpoints": {
            "/recommendations": "POST - Get movie recommendations",
            "/user/{user_id}/recommendations": "GET - Get recommendations for specific user",
            "/health": "GET - Health check"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": rec_system is not None}

@app.post("/recommendations", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    """Get movie recommendations for a user"""
    if rec_system is None:
        raise HTTPException(status_code=500, detail="Recommendation system not initialized")
    
    if request.method not in ["svd", "user_based", "item_based"]:
        raise HTTPException(status_code=400, detail="Method must be 'svd', 'user_based', or 'item_based'")
    
    try:
        recommendations = rec_system.get_recommendations(
            request.user_id, 
            request.method, 
            request.n_recommendations
        )
        
        # Format recommendations with movie titles
        formatted_recommendations = []
        for movie_id, predicted_rating in recommendations:
            movie_info = rec_system.movies_df[rec_system.movies_df['movieId'] == movie_id]
            if not movie_info.empty:
                formatted_recommendations.append({
                    "movie_id": int(movie_id),
                    "title": movie_info.iloc[0]['title'],
                    "genres": movie_info.iloc[0]['genres'],
                    "predicted_rating": round(float(predicted_rating), 2)
                })
        
        return RecommendationResponse(
            user_id=request.user_id,
            method=request.method,
            recommendations=formatted_recommendations
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")

@app.get("/user/{user_id}/recommendations")
async def get_user_recommendations(
    user_id: int, 
    method: str = "svd", 
    n_recommendations: int = 10
):
    """Get recommendations for a specific user (GET endpoint)"""
    request = RecommendationRequest(
        user_id=user_id,
        method=method,
        n_recommendations=n_recommendations
    )
    return await get_recommendations(request)

@app.get("/users")
async def get_available_users():
    """Get list of available user IDs"""
    if rec_system is None:
        raise HTTPException(status_code=500, detail="Recommendation system not initialized")
    
    user_ids = [int(uid) for uid in list(rec_system.user_encoder.keys())[:50]]  # Convert to int
    return {"user_ids": user_ids, "total_users": int(len(rec_system.user_encoder))}

@app.get("/movies/popular")
async def get_popular_movies(limit: int = 20):
    """Get popular movies for cold start users"""
    if rec_system is None:
        raise HTTPException(status_code=500, detail="Recommendation system not initialized")
    
    try:
        popular_movies = rec_system.analyze_cold_start()
        popular_list = []
        
        for movie_id in popular_movies[:limit]:
            movie_info = rec_system.movies_df[rec_system.movies_df['movieId'] == movie_id]
            if not movie_info.empty:
                popular_list.append({
                    "movie_id": int(movie_id),
                    "title": movie_info.iloc[0]['title'],
                    "genres": movie_info.iloc[0]['genres']
                })
        
        return {"popular_movies": popular_list}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting popular movies: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)