import json
import math
import os
import time
import uuid
from decimal import Decimal
from pathlib import Path

try:
    import boto3
except ImportError:  # Allows local syntax checks without boto3 installed.
    boto3 = None


CORS_HEADERS = {
    "Access-Control-Allow-Origin": os.environ.get("ALLOWED_ORIGIN", "*"),
    "Access-Control-Allow-Headers": "Content-Type",
    "Access-Control-Allow-Methods": "OPTIONS,POST",
}
MODEL = None


def lambda_handler(event, context):
    if event.get("requestContext", {}).get("http", {}).get("method") == "OPTIONS":
        return response(204, {})

    method = event.get("requestContext", {}).get("http", {}).get("method") or event.get("httpMethod")
    if method and method != "POST":
        return response(405, {"error": "Only POST is supported."})

    try:
        payload = parse_body(event)
        normalized = normalize_payload(payload)
        prediction = predict_with_model(normalized)
        guidance = prediction or compute_guidance(normalized)
        record_id = save_record(normalized, guidance)

        return response(
            200,
            {
                "id": record_id,
                "guidance": guidance,
                "model": model_summary(),
                "message": "Awareness guidance generated successfully.",
            },
        )
    except ValueError as exc:
        return response(400, {"error": str(exc)})
    except Exception:
        return response(500, {"error": "Unexpected server error."})


def parse_body(event):
    raw_body = event.get("body") or "{}"
    if event.get("isBase64Encoded"):
        raise ValueError("Base64 payloads are not supported.")

    try:
        return json.loads(raw_body)
    except json.JSONDecodeError as exc:
        raise ValueError("Request body must be valid JSON.") from exc


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
        "local_guidance_level": str(payload.get("local_guidance_level", ""))[:30],
        "local_guidance_score": to_int(payload.get("local_guidance_score", 0), "local_guidance_score"),
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


def predict_with_model(values):
    model = load_model()
    if not model:
        return None

    features = feature_vector(values, model)
    standardized = standardize(features, model)
    probability = predict_probability(model, standardized)
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
        "source": "ml_model",
        "level": level,
        "probability": round(probability, 4),
        "score": round(probability * 100, 2),
        "message": message,
    }


def standardize(features, model):
    return [
        (value - model["means"][index]) / model["stds"][index]
        for index, value in enumerate(features)
    ]


def predict_probability(model, standardized):
    if model.get("model_type") == "k_nearest_neighbors":
        distances = sorted(
            (
                (squared_distance(standardized, train_vector), label)
                for train_vector, label in zip(model["train_vectors"], model["train_labels"])
            ),
            key=lambda item: item[0],
        )
        nearest = distances[: model["k"]]
        return sum(label for _, label in nearest) / len(nearest)

    score = sum(weight * value for weight, value in zip(model["weights"], standardized)) + model["bias"]
    return sigmoid(score)


def squared_distance(left, right):
    return sum((a - b) ** 2 for a, b in zip(left, right))


def load_model():
    global MODEL
    if MODEL is not None:
        return MODEL

    path = Path(__file__).with_name("model.json")
    if not path.exists():
        MODEL = False
        return None

    MODEL = json.loads(path.read_text(encoding="utf-8"))
    return MODEL


def feature_vector(values, model):
    age = values["age"]
    tumor_size = impute(values["tumor_size_mm"], model, "Tumor_size_mm")
    bmi = impute(values["bmi"], model, "BMI")
    glucose = impute(values["glucose_level"], model, "Glucose_level")
    blood_pressure = impute(values["blood_pressure"], model, "Blood_pressure")
    cholesterol = impute(values["cholesterol"], model, "Cholesterol")
    return [
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
    if not model:
        return {"available": False, "source": "fallback_guidance"}

    return {
        "available": True,
        "type": model.get("model_type"),
        "version": model.get("version"),
        "features": model.get("features"),
        "metrics": model.get("metrics"),
        "disclaimer": "Educational model only; not clinically validated.",
    }


def compute_guidance(values):
    score = 0
    if values["age"] >= 50:
        score += 2
    elif values["age"] >= 40:
        score += 1

    score += values["lump_present"] * 3
    score += values["mammogram_abnormality"] * 3
    score += values["skin_dimpling"] * 2
    score += values["nipple_discharge"] * 2
    score += values["family_history"]
    score += values["pain_in_breast"]

    if score >= 7:
        return {
            "level": "high",
            "score": score,
            "message": "Several warning signs were selected. A clinical evaluation is recommended.",
        }

    if score >= 3:
        return {
            "level": "moderate",
            "score": score,
            "message": "Some changes were selected. A routine clinical check may be helpful.",
        }

    return {
        "level": "low",
        "score": score,
        "message": "No major warning pattern was selected. Continue awareness and routine screening.",
    }


def save_record(values, guidance):
    record_id = str(uuid.uuid4())
    table_name = os.environ.get("TABLE_NAME")
    if not table_name or boto3 is None:
        return record_id

    item = decimalize(
        {
            "id": record_id,
            "created_at": int(time.time()),
            "inputs": values,
            "guidance": guidance,
        }
    )

    table = boto3.resource("dynamodb").Table(table_name)
    table.put_item(Item=item)
    return record_id


def decimalize(value):
    if isinstance(value, dict):
        return {key: decimalize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [decimalize(item) for item in value]
    if isinstance(value, (int, float)):
        return Decimal(str(value))
    return value


def response(status_code, body):
    return {
        "statusCode": status_code,
        "headers": CORS_HEADERS,
        "body": json.dumps(body),
    }
