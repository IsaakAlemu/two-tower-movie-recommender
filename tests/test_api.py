from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_recommend_unknown_user(monkeypatch):
    def fake_get_recommendations(user_id, k=10):
        raise ValueError("unknown user")

    monkeypatch.setattr(
        "app.main.get_recommendations",
        fake_get_recommendations,
    )

    response = client.post(
        "/recommend",
        json={
            "user_id": 0,
            "k": 10,
        },
    )

    assert response.status_code == 404


def test_recommend_known_user(monkeypatch):
    def fake_get_recommendations(user_id, k=10):
        return [
            {
                "movie_id": i,
                "title": f"Movie {i}",
                "score": 1.0 - i * 0.01,
                "genres": [],
            }
            for i in range(k)
        ]

    monkeypatch.setattr(
        "app.main.get_recommendations",
        fake_get_recommendations,
    )

    response = client.post(
        "/recommend",
        json={
            "user_id": 1,
            "k": 10,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["user_id"] == 1
    assert len(data["recommendations"]) == 10