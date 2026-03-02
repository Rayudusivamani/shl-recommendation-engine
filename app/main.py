# app/main.py
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from typing import List, Optional, Dict
import uvicorn
import os
from dotenv import load_dotenv

from app.retriever.embedding_retriever import SHLRetriever
from app.models.schemas import (
    HealthResponse,
    RecommendationRequest,
    RecommendationResponse,
    Assessment
)

load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="SHL Assessment Recommendation System",
    description="API for recommending SHL assessments based on job descriptions",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize retriever
retriever = SHLRetriever()

# Load pre-built index
INDEX_PATH = os.getenv("INDEX_PATH", "data/processed/index")
if os.path.exists(INDEX_PATH):
    retriever.load(INDEX_PATH)
else:
    print("Warning: Index not found. Please run build_index first.")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint
    """
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        index_loaded=retriever.index is not None,
        num_assessments=len(retriever.assessments) if retriever.assessments else 0
    )

@app.post("/recommend", response_model=RecommendationResponse)
async def recommend_assessments(request: RecommendationRequest):
    """
    Recommend SHL assessments based on job description or natural language query
    """
    try:
        # Determine k (between 5 and 10)
        k = min(max(request.top_k, 5), 10) if request.top_k else 10
        
        # Get query from request
        query = request.query or request.job_description
        
        if not query:
            raise HTTPException(status_code=400, detail="Either query or job_description must be provided")
        
        # Retrieve assessments
        if request.use_hybrid:
            results = retriever.hybrid_retrieve(query, k=k)
        else:
            results = retriever.retrieve(query, k=k)
        
        # Convert to response format
        recommendations = []
        for assessment, score in results:
            # Extract test type (ensure it's a list)
            test_type = assessment.get('test_type', [])
            if isinstance(test_type, str):
                test_type = [test_type]
            
            # Create assessment object
            rec = Assessment(
                url=assessment.get('url', ''),
                name=assessment.get('name', ''),
                adaptive_support=assessment.get('adaptive_support', 'No'),
                description=assessment.get('description', '')[:200] + '...' if len(assessment.get('description', '')) > 200 else assessment.get('description', ''),
                duration=assessment.get('duration', 0),
                remote_support=assessment.get('remote_support', 'No'),
                test_type=test_type,
                relevance_score=score
            )
            recommendations.append(rec)
        
        # Balance recommendations if needed
        if request.balance_domains:
            recommendations = balance_recommendations(recommendations, query)
        
        return RecommendationResponse(
            query=query,
            recommendations=recommendations[:10],
            count=len(recommendations[:10])
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/recommend", response_model=RecommendationResponse)
async def recommend_assessments_get(
    q: str = Query(..., description="Natural language query or job description"),
    top_k: int = Query(10, ge=1, le=10),
    balance: bool = Query(False, description="Balance recommendations across domains")
):
    """
    GET endpoint for recommendations (simpler interface)
    """
    request = RecommendationRequest(
        query=q,
        top_k=top_k,
        balance_domains=balance
    )
    return await recommend_assessments(request)

def balance_recommendations(recommendations: List[Assessment], query: str) -> List[Assessment]:
    """
    Balance recommendations across different test types (K, P, etc.)
    """
    # Separate by test type
    k_type = [r for r in recommendations if 'K' in r.test_type]
    p_type = [r for r in recommendations if 'P' in r.test_type]
    other = [r for r in recommendations if not ('K' in r.test_type or 'P' in r.test_type)]
    
    # Determine if query requires balance (check for keywords)
    query_lower = query.lower()
    needs_balance = any(word in query_lower for word in ['collaborat', 'team', 'behavior', 'personality', 'soft skill'])
    
    if needs_balance and k_type and p_type:
        # Take top from each category
        balanced = []
        for i in range(min(len(k_type), len(p_type), 5)):
            balanced.append(k_type[i])
            balanced.append(p_type[i])
        
        # Add remaining
        remaining = k_type[len(balanced)//2:] + p_type[len(balanced)//2:] + other
        balanced.extend(remaining[:10-len(balanced)])
        
        return balanced[:10]
    
    return recommendations

@app.get("/search")
async def search_assessments(
    query: str = Query(..., description="Search query"),
    limit: int = Query(10, ge=1, le=50)
):
    """
    Simple search endpoint (for debugging)
    """
    results = retriever.retrieve(query, k=limit)
    return {
        "query": query,
        "results": [
            {
                "name": a['name'],
                "url": a['url'],
                "score": score,
                "test_type": a.get('test_type', [])
            }
            for a, score in results
        ]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)