import csv
import json
import math
import random
import statistics
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATA_PATH = DATA_DIR / "wdbc.csv"
MODEL_PATH = ROOT / "aws" / "diagnostic_model.json"
REPORT_PATH = ROOT / "ml" / "diagnostic_model_report.json"

UCI_WDBC_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"

FEATURES = [
    "radius_mean",
    "texture_mean",
    "perimeter_mean",
    "area_mean",
    "smoothness_mean",
    "compactness_mean",
    "concavity_mean",
    "concave_points_mean",
    "symmetry_mean",
    "fractal_dimension_mean",
    "radius_se",
    "texture_se",
    "perimeter_se",
    "area_se",
    "smoothness_se",
    "compactness_se",
    "concavity_se",
    "concave_points_se",
    "symmetry_se",
    "fractal_dimension_se",
    "radius_worst",
    "texture_worst",
    "perimeter_worst",
    "area_worst",
    "smoothness_worst",
    "compactness_worst",
    "concavity_worst",
    "concave_points_worst",
    "symmetry_worst",
    "fractal_dimension_worst",
]


def main():
    ensure_dataset()
    dataset = load_dataset()
    train, test = split_dataset(dataset, test_ratio=0.2, seed=42)
    model = train_logistic_regression(train)
    metrics = evaluate(model, test)
    feature_importance = explain_model(model)

    artifact = {
        "model_type": "diagnostic_logistic_regression",
        "version": "1.0.0",
        "dataset_sources": [
            "Kaggle: uciml/breast-cancer-wisconsin-data",
            "Kaggle: yasserh/breast-cancer-dataset",
            "Fallback mirror: UCI WDBC public data file",
        ],
        "target": "diagnosis",
        "target_mapping": {"M": 1, "B": 0},
        "features": FEATURES,
        "means": model["means"],
        "stds": model["stds"],
        "weights": model["weights"],
        "bias": model["bias"],
        "thresholds": {"benign": 0.5},
        "metrics": metrics,
        "feature_importance": feature_importance,
        "notes": [
            "Diagnostic model uses Wisconsin Diagnostic Breast Cancer cell-nuclei features.",
            "It is an educational model and is not clinically validated for diagnosis.",
        ],
    }

    report = {
        "dataset": {
            "file": str(DATA_PATH.relative_to(ROOT)),
            "records": len(dataset),
            "features": FEATURES,
            "target": "diagnosis",
            "sources": artifact["dataset_sources"],
        },
        "deployed_model": "Diagnostic Logistic Regression",
        "metrics": metrics,
        "feature_importance": feature_importance,
        "limitations": [
            "The dataset is a benchmark diagnostic dataset, not a deployed clinical workflow.",
            "Inputs are FNA image-derived cell measurements, not self-check questionnaire fields.",
            "The model is educational and not clinically validated.",
        ],
    }

    MODEL_PATH.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {MODEL_PATH}")
    print(f"Wrote {REPORT_PATH}")
    print(json.dumps(metrics, indent=2))


def ensure_dataset():
    if DATA_PATH.exists():
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading WDBC data to {DATA_PATH}")
    with urllib.request.urlopen(UCI_WDBC_URL, timeout=30) as response:
        raw = response.read().decode("utf-8")

    with DATA_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "diagnosis", *FEATURES])
        for line in raw.splitlines():
            writer.writerow(line.split(","))


def load_dataset():
    rows = []
    with DATA_PATH.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            features = [float(row[name]) for name in FEATURES]
            target = 1 if row["diagnosis"] == "M" else 0
            rows.append((features, target))
    return rows


def split_dataset(dataset, test_ratio, seed):
    shuffled = list(dataset)
    random.Random(seed).shuffle(shuffled)
    test_size = max(1, int(len(shuffled) * test_ratio))
    return shuffled[test_size:], shuffled[:test_size]


def train_logistic_regression(train):
    means, stds = standardization_stats([features for features, _ in train])
    weights = [0.0] * len(FEATURES)
    bias = 0.0
    learning_rate = 0.12
    regularization = 0.001

    for _ in range(2500):
        grad_w = [0.0] * len(FEATURES)
        grad_b = 0.0
        for raw_features, target in train:
            features = standardize(raw_features, means, stds)
            prediction = sigmoid(dot(weights, features) + bias)
            error = prediction - target
            grad_b += error
            for index, value in enumerate(features):
                grad_w[index] += error * value

        n = len(train)
        bias -= learning_rate * grad_b / n
        for index in range(len(weights)):
            penalty = regularization * weights[index]
            weights[index] -= learning_rate * ((grad_w[index] / n) + penalty)

    return {"means": means, "stds": stds, "weights": weights, "bias": bias}


def evaluate(model, test):
    confusion = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    losses = []
    for features, target in test:
        probability = predict_probability(model, features)
        label = 1 if probability >= 0.5 else 0
        losses.append(log_loss(probability, target))
        if label == 1 and target == 1:
            confusion["tp"] += 1
        elif label == 0 and target == 0:
            confusion["tn"] += 1
        elif label == 1 and target == 0:
            confusion["fp"] += 1
        else:
            confusion["fn"] += 1

    total = sum(confusion.values())
    accuracy = (confusion["tp"] + confusion["tn"]) / total
    precision = safe_div(confusion["tp"], confusion["tp"] + confusion["fp"])
    recall = safe_div(confusion["tp"], confusion["tp"] + confusion["fn"])
    f1 = safe_div(2 * precision * recall, precision + recall)
    return {
        "test_records": total,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "log_loss": round(statistics.fmean(losses), 4),
        "confusion_matrix": confusion,
    }


def explain_model(model):
    rows = []
    for name, weight in zip(FEATURES, model["weights"]):
        rows.append(
            {
                "feature": name,
                "weight": round(weight, 6),
                "absolute_weight": round(abs(weight), 6),
                "direction": "raises malignant probability" if weight > 0 else "lowers malignant probability",
            }
        )
    return sorted(rows, key=lambda row: row["absolute_weight"], reverse=True)


def predict_probability(model, features):
    standardized = standardize(features, model["means"], model["stds"])
    return sigmoid(dot(model["weights"], standardized) + model["bias"])


def standardization_stats(feature_rows):
    columns = list(zip(*feature_rows))
    means = [statistics.fmean(column) for column in columns]
    stds = []
    for column in columns:
        std = statistics.pstdev(column)
        stds.append(std if std > 0 else 1.0)
    return means, stds


def standardize(features, means, stds):
    return [(value - means[index]) / stds[index] for index, value in enumerate(features)]


def sigmoid(value):
    if value < -500:
        return 0.0
    if value > 500:
        return 1.0
    return 1 / (1 + math.exp(-value))


def dot(left, right):
    return sum(a * b for a, b in zip(left, right))


def log_loss(probability, target):
    clipped = min(max(probability, 1e-9), 1 - 1e-9)
    return -(target * math.log(clipped) + (1 - target) * math.log(1 - clipped))


def safe_div(numerator, denominator):
    return numerator / denominator if denominator else 0.0


if __name__ == "__main__":
    main()
