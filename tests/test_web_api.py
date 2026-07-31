from fastapi.testclient import TestClient

from kakeya.web_api import ExperimentRequest, app


def test_image_codec_defaults_use_fifty_epoch_capacity_run() -> None:
    request = ExperimentRequest()

    assert request.method == "image_codec"
    assert request.epochs == 80
    assert request.train_limit == 128


def test_health_and_environment_endpoints() -> None:
    with TestClient(app) as client:
        health = client.get("/api/health")
        environment = client.get("/api/environment")

    assert health.status_code == 200
    assert health.json() == {"status": "ok"}
    assert environment.status_code == 200
    assert "packages" in environment.json()


def test_builtin_image_probe_is_available() -> None:
    with TestClient(app) as client:
        response = client.get("/api/test-image")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content.startswith(b"\x89PNG")


def test_defaults_endpoint_returns_image_codec_weights() -> None:
    with TestClient(app) as client:
        response = client.get("/api/defaults")

    assert response.status_code == 200
    assert response.json()["method"] == "image_codec"
    assert "capacity" in response.json()["stage_weights"]


def test_request_builds_method_specific_objective() -> None:
    request = ExperimentRequest(
        method="image_codec",
        train_limit=32,
        num_projections=16,
        batch_size=4,
        k=10,
    )

    config = request.experiment_config()

    assert config["train_limit"] == 32
    assert config["test_limit"] is None
    obj = config["objective"]
    assert obj["num_projections"] == 16
    assert obj["k"] == 3
    assert obj["lambda_rate"] == 0.01
    assert obj["lambda_kakeya"] == 0.001
    # stage_weights should be present for image_codec method
    assert "stage_weights" in obj
    assert "capacity" in obj["stage_weights"]
    assert "transition" in obj["stage_weights"]
    assert "finetune" in obj["stage_weights"]


def test_invalid_epoch_count_returns_validation_error() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/api/experiments",
            json={"method": "image_codec", "epochs": 0},
        )

    assert response.status_code == 422
