# app/models/schemas.py
from pydantic import BaseModel, HttpUrl, Field
from typing import List, Optional, Dict, Any
from enum import Enum

class HealthResponse(BaseModel):
    status: str
    version: str
    index_loaded: bool
    num_assessments: int

class RecommendationRequest(BaseModel):
    query: Optional[str] = Field(None, description="Natural language query")
    job_description: Optional[str] = Field(None, description="Job description text")
    url: Optional[HttpUrl] = Field(None, description="URL containing job description")
    top_k: Optional[int] = Field(10, ge=1, le=10, description="Number of recommendations (1-10)")
    use_hybrid: bool = Field(True, description="Use hybrid retrieval")
    balance_domains: bool = Field(True, description="Balance recommendations across domains")

class Assessment(BaseModel):
    url: str
    name: str
    adaptive_support: str = Field(..., pattern="^(Yes|No)$")
    description: str
    duration: int = Field(..., ge=0)
    remote_support: str = Field(..., pattern="^(Yes|No)$")
    test_type: List[str]
    relevance_score: Optional[float] = None

class RecommendationResponse(BaseModel):
    query: str
    recommendations: List[Assessment]
    count: int