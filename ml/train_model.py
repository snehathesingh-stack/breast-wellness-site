import argparse
import csv
import importlib.util
import json
import math
import random
import statistics
import zipfile
from pathlib import Path
from xml.etree import ElementTree


ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "Dataset_file.xlsx"
DEFAULT_SYNTHETIC_PATH = ROOT / "ml" / "synthetic_wellness_dataset.csv"
MODEL_PATH = ROOT / "aws" / "model.json"
REPORT_PATH = ROOT / "ml" / "model_report.json"

FEATURES = [
    "Lump_present",
    "Pain_in_breast",
    "Skin_dimpling",
    "Nipple_discharge",
    "Family_history",
    "Age",
    "Tumor_size_mm",
    "BMI",
    "Glucose_level",
    "Blood_pressure",
    "Cholesterol",
    "TumorAge_interaction",
    "Age_group_36-50",
    "Age_group_51-65",
    "Age_group_66+",
    "BMI_category_Normal",
    "BMI_category_Overweight",
    "BMI_category_Obese",
]
TARGET = "Detected_cancer"


def main(dataset_path=None):
    path = Path(dataset_path or DATASET_PATH)
    rows = load_dataset(path)
    dataset = clean_rows(rows)
    train, test = split_dataset(dataset, test_ratio=0.2, seed=42)
    logistic_model = train_logistic_regression(train)
    logistic_metrics = evaluate_logistic(logistic_model, test)
    knn_model, knn_metrics = train_best_knn(train, test)
    baseline_metrics = evaluate_baseline(test)
    feature_importance = explain_logistic_model(logistic_model)

    artifact = {
        "model_type": "k_nearest_neighbors",
        "version": "2.0.0",
        "trained_from": path.name,
        "target": TARGET,
        "features": FEATURES,
        "means": knn_model["means"],
        "stds": knn_model["stds"],
        "k": knn_model["k"],
        "train_vectors": knn_model["train_vectors"],
        "train_labels": knn_model["train_labels"],
        "explanation_model": {
            "model_type": "logistic_regression",
            "weights": logistic_model["weights"],
            "bias": logistic_model["bias"],
        },
        "thresholds": {
            "low": 0.35,
            "moderate": 0.65,
        },
        "metrics": knn_metrics,
        "feature_importance": feature_importance,
        "comparison": [
            {
                "model": f"kNN (k={knn_model['k']})",
                "role": "Deployed ML model",
                **knn_metrics,
            },
            {
                "model": "Logistic Regression",
                "role": "Baseline interpretable model",
                **logistic_metrics,
            },
            {
                "model": "Majority Class Baseline",
                "role": "Sanity-check baseline",
                **baseline_metrics,
            },
        ],
        "notes": [
            "This model is for educational project use only.",
            "It is not validated for clinical diagnosis or medical decision-making.",
        ],
    }

    report = {
        "dataset": {
            "file": path.name,
            "records": len(dataset),
            "features": FEATURES,
            "target": TARGET,
        },
        "deployed_model": f"kNN (k={knn_model['k']})",
        "model_comparison": artifact["comparison"],
        "feature_importance": feature_importance,
        "limitations": [
            "The dataset is curated for project demonstration.",
            "The model is not clinically validated.",
            "Metrics show limited predictive signal and should be presented honestly.",
        ],
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {MODEL_PATH}")
    print(f"Wrote {REPORT_PATH}")
    print(json.dumps(knn_metrics, indent=2))


def load_dataset(path):
    if path.suffix.lower() == ".csv":
        return read_csv(path)
    if path.suffix.lower() == ".xlsx":
        return read_xlsx(path)
    raise ValueError(f"Unsupported dataset format: {path.suffix}")


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        return [row for row in reader]


def read_xlsx(path):
    ns = {"main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(path) as archive:
        shared_strings = read_shared_strings(archive, ns)
        sheet_xml = archive.read("xl/worksheets/sheet1.xml")

    root = ElementTree.fromstring(sheet_xml)
    table = []
    for row in root.findall(".//main:sheetData/main:row", ns):
        values = {}
        for cell in row.findall("main:c", ns):
            ref = cell.attrib["r"]
            column = "".join(ch for ch in ref if ch.isalpha())
            raw = cell.find("main:v", ns)
            if raw is None:
                values[column] = ""
                continue
            value = raw.text
            if cell.attrib.get("t") == "s":
                value = shared_strings[int(value)]
            values[column] = value

        if values:
            table.append([values.get(index_to_column(i), "") for i in range(1, 20)])

    # Row 1 is the merged dataset title. Row 2 is the actual header.
    headers = table[1]
    return [dict(zip(headers, row)) for row in table[2:]]


def read_shared_strings(archive, ns):
    root = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
    strings = []
    for item in root.findall("main:si", ns):
        parts = [node.text or "" for node in item.findall(".//main:t", ns)]
        strings.append("".join(parts))
    return strings


def index_to_column(index):
    result = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        result = chr(65 + remainder) + result
    return result


def clean_rows(rows):
    dataset = []
    for row in rows:
        try:
            features = [float(row[name]) for name in FEATURES]
            target = parse_target(row[TARGET])
        except (KeyError, TypeError, ValueError):
            continue
        dataset.append((features, target))
    if not dataset:
        raise RuntimeError("No valid training rows found.")
    return dataset


def parse_target(value):
    normalized = str(value).strip().lower()
    if normalized in {"yes", "y", "true", "1"}:
        return 1
    if normalized in {"no", "n", "false", "0"}:
        return 0
    raise ValueError(f"Unsupported target value: {value}")


def split_dataset(dataset, test_ratio, seed):
    shuffled = list(dataset)
    random.Random(seed).shuffle(shuffled)
    test_size = max(1, int(len(shuffled) * test_ratio))
    return shuffled[test_size:], shuffled[:test_size]


def train_logistic_regression(train):
    means, stds = standardization_stats([features for features, _ in train])
    weights = [0.0] * len(FEATURES)
    bias = 0.0
    learning_rate = 0.08
    regularization = 0.002

    for _ in range(1400):
        grad_w = [0.0] * len(FEATURES)
        grad_b = 0.0
        for raw_features, target in train:
            features = standardize(raw_features, means, stds)
            prediction = sigmoid(dot(weights, features) + bias)
            error = prediction - target
            grad_b += error
            for i, value in enumerate(features):
                grad_w[i] += error * value

        n = len(train)
        bias -= learning_rate * grad_b / n
        for i in range(len(weights)):
            penalty = regularization * weights[i]
            weights[i] -= learning_rate * ((grad_w[i] / n) + penalty)

    return {
        "means": means,
        "stds": stds,
        "weights": weights,
        "bias": bias,
    }


def standardization_stats(feature_rows):
    columns = list(zip(*feature_rows))
    means = [statistics.fmean(column) for column in columns]
    stds = []
    for column in columns:
        std = statistics.pstdev(column)
        stds.append(std if std > 0 else 1.0)
    return means, stds


def standardize(features, means, stds):
    return [(value - means[i]) / stds[i] for i, value in enumerate(features)]


def evaluate_logistic(model, test):
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


def train_best_knn(train, test):
    means, stds = standardization_stats([features for features, _ in train])
    train_vectors = [standardize(features, means, stds) for features, _ in train]
    train_labels = [target for _, target in train]
    candidates = [7, 11, 15, 21, 31, 41, 61]
    best = None

    for k in candidates:
        model = {
            "means": means,
            "stds": stds,
            "k": k,
            "train_vectors": train_vectors,
            "train_labels": train_labels,
        }
        metrics = evaluate_knn(model, test)
        rank = (metrics["accuracy"], metrics["f1"])
        if best is None or rank > best[0]:
            best = (rank, model, metrics)

    return best[1], best[2]


def evaluate_knn(model, test):
    confusion = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    losses = []
    for features, target in test:
        probability = predict_knn_probability(model, features)
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


def predict_knn_probability(model, features):
    standardized = standardize(features, model["means"], model["stds"])
    distances = sorted(
        (
            (squared_distance(standardized, train_vector), label)
            for train_vector, label in zip(model["train_vectors"], model["train_labels"])
        ),
        key=lambda item: item[0],
    )
    nearest = distances[: model["k"]]
    return sum(label for _, label in nearest) / len(nearest)


def squared_distance(left, right):
    return sum((a - b) ** 2 for a, b in zip(left, right))


def evaluate_baseline(test):
    positives = sum(target for _, target in test)
    negatives = len(test) - positives
    majority_label = 1 if positives >= negatives else 0
    confusion = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}

    for _, target in test:
        label = majority_label
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
        "log_loss": None,
        "confusion_matrix": confusion,
    }


def explain_logistic_model(model):
    rows = []
    for name, weight in zip(FEATURES, model["weights"]):
        rows.append(
            {
                "feature": name,
                "weight": round(weight, 6),
                "absolute_weight": round(abs(weight), 6),
                "direction": "raises model probability" if weight > 0 else "lowers model probability",
            }
        )
    return sorted(rows, key=lambda row: row["absolute_weight"], reverse=True)


def predict_probability(model, features):
    standardized = standardize(features, model["means"], model["stds"])
    return sigmoid(dot(model["weights"], standardized) + model["bias"])


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


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train the breast wellness questionnaire ML model.")
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(DATASET_PATH),
        help="Path to the wellness dataset file (.xlsx or .csv).",
    )
    parser.add_argument(
        "--generate-synthetic",
        action="store_true",
        help="Generate a synthetic wellness dataset before training.",
    )
    parser.add_argument(
        "--records",
        type=int,
        default=1500,
        help="Number of synthetic records to generate when using --generate-synthetic.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    dataset_path = Path(args.dataset)
    if args.generate_synthetic:
        if args.dataset == str(DATASET_PATH):
            dataset_path = DEFAULT_SYNTHETIC_PATH
        generate_script = Path(__file__).resolve().parent / "generate_synthetic_wellness_data.py"
        spec = importlib.util.spec_from_file_location("generate_synthetic_wellness_data", generate_script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.generate_synthetic_dataset(dataset_path, records=args.records, seed=42)
    main(dataset_path)
