import json
import math
from http.server import BaseHTTPRequestHandler
from pathlib import Path


MODEL = None


class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(204)
        self.send_headers()

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length).decode("utf-8") if length else "{}"
            payload = json.loads(body)
            normalized = normalize_payload(payload)
            guidance = predict_with_model(normalized)
            self.send_json(
                200,
                {
                    "guidance": guidance,
                    "model": model_summary(),
                    "message": "Vercel ML inference completed successfully.",
                },
            )
        except ValueError as exc:
            self.send_json(400, {"error": str(exc)})
        except Exception:
            self.send_json(500, {"error": "Unexpected server error."})

    def do_GET(self):
        self.send_json(
            200,
            {
                "status": "ok",
                "message": "Breast Wellness ML API is running.",
                "model": model_summary(),
            },
        )

    def send_json(self, status_code, payload):
        self.send_response(status_code)
        self.send_headers()
        self.wfile.write(json.dumps(payload).encode("utf-8"))

    def send_headers(self):
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.end_headers()


def normalize_payload(payload):
    age = to_int(payload.get("age"), "age")
    if age < 18 or age > 120:
        raise ValueError("Age must be between 18 and 120.")

    return {
        "age": age,
        "lump_present": to_binary(payload.get("lump_present"), "lump_present"),
        "pain_in_breast": to_binary(payload.get("pain_in_breast"), "pain_in_breast"),
        "skin_dimpling": to_binary(payload.get("skin_dimpling"), "skin_dimpling"),
        "nipple_discharge": to_binary(payload.get("nipple_discharge"), "nipple_discharge"),
        "family_history": to_binary(payload.get("family_history"), "family_history"),
        "mammogram_abnormality": to_binary(payload.get("mammogram_abnormality"), "mammogram_abnormality"),
        "tumor_size_mm": to_optional_number(payload.get("tumor_size_mm")),
        "bmi": to_optional_number(payload.get("bmi")),
        "glucose_level": to_optional_number(payload.get("glucose_level")),
        "blood_pressure": to_optional_number(payload.get("blood_pressure")),
        "cholesterol": to_optional_number(payload.get("cholesterol")),
    }


def to_int(value, field_name):
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a number.") from exc


def to_binary(value, field_name):
    number = to_int(value, field_name)
    if number not in (0, 1):
        raise ValueError(f"{field_name} must be 0 or 1.")
    return number


def to_optional_number(value):
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_model():
    global MODEL
    if MODEL is not None:
        return MODEL

    path = Path(__file__).resolve().parents[1] / "aws" / "model.json"
    MODEL = json.loads(path.read_text(encoding="utf-8"))
    return MODEL


def predict_with_model(values):
    model = load_model()
    features, display_features = feature_vector(values, model)
    standardized = [
        (value - model["means"][index]) / model["stds"][index]
        for index, value in enumerate(features)
    ]
    raw_score = sum(weight * value for weight, value in zip(model["weights"], standardized)) + model["bias"]
    probability = sigmoid(raw_score)
    thresholds = model.get("thresholds", {"low": 0.35, "moderate": 0.65})

    if probability >= thresholds["moderate"]:
        level = "high"
        message = "The ML model found a higher pattern match. Please arrange a clinical evaluation."
    elif probability >= thresholds["low"]:
        level = "moderate"
        message = "The ML model found a moderate pattern match. A routine clinical check may help."
    else:
        level = "low"
        message = "The ML model found a lower pattern match. Continue awareness and routine screening."

    return {
        "source": "vercel_ml_model",
        "level": level,
        "probability": round(probability, 4),
        "score": round(probability * 100, 2),
        "message": message,
        "top_factors": top_factors(model, standardized, display_features),
        "thresholds": thresholds,
    }


def feature_vector(values, model):
    age = values["age"]
    tumor_size = impute(values["tumor_size_mm"], model, "Tumor_size_mm")
    bmi = impute(values["bmi"], model, "BMI")
    glucose = impute(values["glucose_level"], model, "Glucose_level")
    blood_pressure = impute(values["blood_pressure"], model, "Blood_pressure")
    cholesterol = impute(values["cholesterol"], model, "Cholesterol")
    features = [
        values["lump_present"],
        values["pain_in_breast"],
        values["skin_dimpling"],
        values["nipple_discharge"],
        values["family_history"],
        age,
        tumor_size,
        bmi,
        glucose,
        blood_pressure,
        cholesterol,
        tumor_size * age,
        1 if 36 <= age <= 50 else 0,
        1 if 51 <= age <= 65 else 0,
        1 if age >= 66 else 0,
        1 if 18.5 <= bmi < 25 else 0,
        1 if 25 <= bmi < 30 else 0,
        1 if bmi >= 30 else 0,
    ]
    display_features = dict(zip(model["features"], features))
    return features, display_features


def top_factors(model, standardized, display_features):
    factors = []
    for index, feature in enumerate(model["features"]):
        contribution = model["weights"][index] * standardized[index]
        factors.append(
            {
                "feature": feature,
                "value": round(display_features[feature], 4),
                "contribution": round(contribution, 4),
                "direction": "raises probability" if contribution > 0 else "lowers probability",
            }
        )
    return sorted(factors, key=lambda item: abs(item["contribution"]), reverse=True)[:5]


def impute(value, model, feature_name):
    if value is not None:
        return value
    index = model["features"].index(feature_name)
    return model["means"][index]


def sigmoid(value):
    if value < -500:
        return 0.0
    if value > 500:
        return 1.0
    return 1 / (1 + math.exp(-value))


def model_summary():
    model = load_model()
    return {
        "available": True,
        "platform": "vercel",
        "type": model.get("model_type"),
        "version": model.get("version"),
        "metrics": model.get("metrics"),
        "comparison": model.get("comparison"),
        "disclaimer": "Educational model only; not clinically validated.",
    }
