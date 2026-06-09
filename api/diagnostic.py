import json
import math
from http.server import BaseHTTPRequestHandler
from pathlib import Path


MODEL = None


class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(204)
        self.send_headers()

    def do_GET(self):
        self.send_json(
            200,
            {
                "status": "ok",
                "message": "Diagnostic WDBC model API is running.",
                "model": model_summary(),
            },
        )

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length).decode("utf-8") if length else "{}"
            payload = json.loads(body)
            features = normalize_payload(payload)
            prediction = predict(features)
            self.send_json(
                200,
                {
                    "prediction": prediction,
                    "model": model_summary(),
                    "message": "Diagnostic model inference completed successfully.",
                },
            )
        except ValueError as exc:
            self.send_json(400, {"error": str(exc)})
        except Exception:
            self.send_json(500, {"error": "Unexpected server error."})

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
    model = load_model()
    features = payload.get("features")
    if isinstance(features, dict):
        return [to_float(features.get(name), name) for name in model["features"]]
    if isinstance(features, list):
        if len(features) != len(model["features"]):
            raise ValueError(f"features must contain {len(model['features'])} values.")
        return [to_float(value, f"features[{index}]") for index, value in enumerate(features)]
    raise ValueError("Provide features as an object keyed by feature name or a 30-value list.")


def predict(features):
    model = load_model()
    standardized = [
        (value - model["means"][index]) / model["stds"][index]
        for index, value in enumerate(features)
    ]
    probability = sigmoid(sum(weight * value for weight, value in zip(model["weights"], standardized)) + model["bias"])
    label = "malignant" if probability >= 0.5 else "benign"
    return {
        "source": "wdbc_diagnostic_model",
        "label": label,
        "malignant_probability": round(probability, 4),
        "confidence": round(max(probability, 1 - probability), 4),
        "top_factors": top_factors(model, standardized, features),
    }


def top_factors(model, standardized, raw_features):
    factors = []
    for index, feature in enumerate(model["features"]):
        contribution = model["weights"][index] * standardized[index]
        factors.append(
            {
                "feature": feature,
                "value": round(raw_features[index], 5),
                "contribution": round(contribution, 5),
                "direction": "raises malignant probability" if contribution > 0 else "lowers malignant probability",
            }
        )
    return sorted(factors, key=lambda item: abs(item["contribution"]), reverse=True)[:8]


def model_summary():
    model = load_model()
    return {
        "available": True,
        "platform": "vercel",
        "type": model.get("model_type"),
        "version": model.get("version"),
        "features": model.get("features"),
        "metrics": model.get("metrics"),
        "disclaimer": "Educational model only; not clinically validated.",
    }


def load_model():
    global MODEL
    if MODEL is not None:
        return MODEL
    path = Path(__file__).resolve().parents[1] / "aws" / "diagnostic_model.json"
    MODEL = json.loads(path.read_text(encoding="utf-8"))
    return MODEL


def to_float(value, field_name):
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be numeric.") from exc


def sigmoid(value):
    if value < -500:
        return 0.0
    if value > 500:
        return 1.0
    return 1 / (1 + math.exp(-value))
