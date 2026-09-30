from typing import List
from pydantic import BaseModel, Field


class RecommendRequest(BaseModel):
    user_id: int
    k: int = Field(default=10, ge=1, le=50)


class RecommendationItem(BaseModel):
    movie_id: int
    title: str
    score: float
    genres: List[str] = []


class RecommendResponse(BaseModel):
    user_id: int
    recommendations: List[RecommendationItem]