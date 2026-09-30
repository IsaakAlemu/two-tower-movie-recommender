import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from app.schemas import RecommendRequest, RecommendResponse
from app.engine import get_recommendations, load_artifacts


@asynccontextmanager
async def lifespan(app: FastAPI):
    if os.getenv("TESTING") != "1":
        load_artifacts()
    yield


app = FastAPI(title="Two-Tower Movie Recommender", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/recommend", response_model=RecommendResponse)
def recommend(request: RecommendRequest):
    try:
        recommendations = get_recommendations(request.user_id, k=request.k)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return RecommendResponse(user_id=request.user_id, recommendations=recommendations)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)