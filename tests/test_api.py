import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module(relative_path, module_name):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_predict_returns_explainable_guidance():
    predict = load_module("api/predict.py", "predict")
    payload = {
        "age": 45,
        "lump_present": 1,
        "pain_in_breast": 0,
        "skin_dimpling": 0,
        "nipple_discharge": 0,
        "family_history": 1,
        "mammogram_abnormality": 0,
        "tumor_size_mm": 25,
        "bmi": 27,
        "glucose_level": 110,
        "blood_pressure": 120,
        "cholesterol": 180,
    }

    normalized = predict.normalize_payload(payload)
    guidance = predict.predict_with_model(normalized)

    assert guidance["source"] == "vercel_ml_model"
    assert guidance["level"] in {"low", "moderate", "high"}
    assert 0 <= guidance["probability"] <= 1
    assert len(guidance["top_factors"]) == 5
    assert "feature" in guidance["top_factors"][0]
    assert "direction" in guidance["top_factors"][0]


def test_predict_rejects_invalid_age():
    predict = load_module("api/predict.py", "predict_invalid")
    payload = {
        "age": 12,
        "lump_present": 0,
        "pain_in_breast": 0,
        "skin_dimpling": 0,
        "nipple_discharge": 0,
        "family_history": 0,
        "mammogram_abnormality": 0,
    }

    try:
        predict.normalize_payload(payload)
    except ValueError as exc:
        assert "Age must be between 18 and 120" in str(exc)
    else:
        raise AssertionError("Expected invalid age to raise ValueError")


def test_health_model_summary_loads():
    health = load_module("api/health.py", "health")
    model = health.load_model()

    assert model["model_type"] == "logistic_regression"
    assert model["metrics"]["test_records"] == 200


def test_model_artifacts_are_valid():
    model = json.loads((ROOT / "aws/model.json").read_text(encoding="utf-8"))
    report = json.loads((ROOT / "ml/model_report.json").read_text(encoding="utf-8"))

    assert model["features"]
    assert model["feature_importance"]
    assert report["model_comparison"]
    assert report["feature_importance"]
