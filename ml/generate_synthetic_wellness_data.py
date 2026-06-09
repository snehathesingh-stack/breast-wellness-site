import csv
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "ml" / "synthetic_wellness_dataset.csv"

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
HEADERS = FEATURES + [TARGET]


def sigmoid(score):
    return 1 / (1 + math.exp(-score))


def clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


def generate_row():
    age = int(random.triangular(18, 85, 38))
    bmi = clamp(random.gauss(26, 5), 16.0, 44.0)
    lump_present = 1 if random.random() < 0.2 else 0
    pain_in_breast = 1 if random.random() < 0.18 else 0
    skin_dimpling = 1 if random.random() < 0.1 else 0
    nipple_discharge = 1 if random.random() < 0.12 else 0
    family_history = 1 if random.random() < 0.28 else 0

    base_tumor_size = 8.0 if lump_present == 0 else 20.0
    tumor_size_mm = clamp(random.gauss(base_tumor_size, 11), 1.0, 80.0)

    glucose_level = clamp(random.gauss(100 + 8 * (bmi > 30), 15), 70, 180)
    blood_pressure = clamp(random.gauss(118 + 4 * (bmi > 30), 12), 90, 185)
    cholesterol = clamp(random.gauss(185 + 8 * (age > 50), 28), 120, 320)

    risk_score = -5.5
    risk_score += 1.45 * lump_present
    risk_score += 1.25 * pain_in_breast
    risk_score += 1.4 * skin_dimpling
    risk_score += 1.1 * nipple_discharge
    risk_score += 0.95 * family_history
    risk_score += 0.03 * (age - 18)
    risk_score += 0.05 * (tumor_size_mm - 8)
    risk_score += 0.04 * (bmi - 22)
    risk_score += 0.02 * ((glucose_level - 95) / 10)
    risk_score += 0.015 * ((blood_pressure - 115) / 10)
    risk_score += 0.012 * ((cholesterol - 180) / 10)
    risk_score += 0.0012 * tumor_size_mm * age

    probability = sigmoid(risk_score)
    detected_cancer = 1 if random.random() < probability else 0

    age_group_36_50 = 1 if 36 <= age <= 50 else 0
    age_group_51_65 = 1 if 51 <= age <= 65 else 0
    age_group_66_plus = 1 if age >= 66 else 0
    bmi_normal = 1 if 18.5 <= bmi < 25 else 0
    bmi_overweight = 1 if 25 <= bmi < 30 else 0
    bmi_obese = 1 if bmi >= 30 else 0
    tumor_age_interaction = tumor_size_mm * age

    return {
        "Lump_present": lump_present,
        "Pain_in_breast": pain_in_breast,
        "Skin_dimpling": skin_dimpling,
        "Nipple_discharge": nipple_discharge,
        "Family_history": family_history,
        "Age": round(age, 1),
        "Tumor_size_mm": round(tumor_size_mm, 1),
        "BMI": round(bmi, 1),
        "Glucose_level": round(glucose_level, 1),
        "Blood_pressure": round(blood_pressure, 1),
        "Cholesterol": round(cholesterol, 1),
        "TumorAge_interaction": round(tumor_age_interaction, 1),
        "Age_group_36-50": age_group_36_50,
        "Age_group_51-65": age_group_51_65,
        "Age_group_66+": age_group_66_plus,
        "BMI_category_Normal": bmi_normal,
        "BMI_category_Overweight": bmi_overweight,
        "BMI_category_Obese": bmi_obese,
        "Detected_cancer": detected_cancer,
    }


def generate_synthetic_dataset(output_path=None, records=1500, seed=None):
    if seed is not None:
        random.seed(seed)

    output_path = Path(output_path) if output_path else OUTPUT_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=HEADERS)
        writer.writeheader()
        for _ in range(records):
            writer.writerow(generate_row())

    print(f"Generated synthetic wellness dataset: {output_path} ({records} records)")
    return output_path


if __name__ == "__main__":
    generate_synthetic_dataset(OUTPUT_PATH, records=1500, seed=42)
