# Breast Wellness: Cloud-Based Awareness and Risk Guidance Tool

Breast Wellness is a static web application for breast health awareness. It helps users record basic self-check information, receive non-diagnostic guidance, save entries locally, export check-ins, and optionally send an anonymous record to an AWS Lambda backend through API Gateway.

This project is designed for deployment on AWS:

- Frontend: Amazon S3 static website hosting
- Backend: AWS Lambda with Python
- API: Amazon API Gateway HTTP API
- Storage: Amazon DynamoDB
- ML: Logistic Regression trained from the included dataset with a pure-Python script

Live website:

http://breast-cancer-prediction-site.s3-website.ap-south-1.amazonaws.com

## Important Medical Disclaimer

This application is for awareness and education only. It does not diagnose breast cancer, estimate a clinical probability, replace screening, or replace advice from a qualified healthcare professional.

## Features

- Breast wellness self-check form
- Age and symptom-based awareness guidance
- Clear low, moderate, and higher concern messaging
- Optional anonymous AWS logging
- Visible AWS success/failure status in the frontend
- Local browser storage for private check-in history
- CSV export
- Monthly Google Calendar reminder
- Clinic search shortcut
- AWS Lambda starter backend with optional DynamoDB persistence
- Reproducible ML training script
- Exported Lambda-ready model artifact
- Cloud ML probability returned from API Gateway/Lambda

## Repository Structure

```text
.
├── breast-cancer-prediction.html   # Static frontend for S3 hosting
├── Dataset_file.xlsx               # Curated project dataset
├── README.md                       # Project documentation
├── ml/
│   └── train_model.py              # Reproducible Logistic Regression training
└── aws/
    ├── lambda_function.py          # Lambda handler for API Gateway
    ├── model.json                  # Exported trained model artifact
    └── README.md                   # AWS setup notes
```

## Dataset

The included spreadsheet is a curated breast cancer awareness dataset:

- File: `Dataset_file.xlsx`
- Sheet: `Sheet 1 - curated_breast_cancer`
- Approximate size: 1,000 records
- Columns include:
  - `Lump_present`
  - `Pain_in_breast`
  - `Skin_dimpling`
  - `Nipple_discharge`
  - `Family_history`
  - `Age`
  - `Tumor_size_mm`
  - `BMI`
  - `Glucose_level`
  - `Blood_pressure`
  - `Cholesterol`
  - `Detected_cancer`

## Machine Learning

The project now includes a reproducible Logistic Regression pipeline:

```text
python ml/train_model.py
```

The script:

1. Reads `Dataset_file.xlsx` directly.
2. Extracts the curated breast cancer awareness dataset.
3. Trains a Logistic Regression model with standardized features.
4. Evaluates the model on a fixed 20% test split.
5. Writes `aws/model.json` for Lambda inference.

Current generated model metrics:

| Metric | Value |
| --- | ---: |
| Test records | 200 |
| Accuracy | 0.54 |
| Precision | 0.5385 |
| Recall | 0.56 |
| F1 | 0.549 |
| Log loss | 0.6937 |

These metrics show that the included dataset has limited predictive signal. The model is included to demonstrate an end-to-end ML workflow, not to provide clinical-grade prediction.

## Frontend Setup

Open the HTML file directly in a browser for local testing:

```text
breast-cancer-prediction.html
```

For AWS S3:

1. Create an S3 bucket.
2. Enable static website hosting.
3. Upload `breast-cancer-prediction.html`.
4. Set the index document to `breast-cancer-prediction.html`.
5. Configure public read access or use CloudFront with origin access control.

## AWS Backend Setup

1. Run `python ml/train_model.py` to regenerate `aws/model.json` if needed.
2. Create a Lambda function using Python 3.12 or newer.
3. Upload both `aws/lambda_function.py` and `aws/model.json`.
4. Create a DynamoDB table with partition key `id` as a string.
5. Add Lambda environment variables:

```text
TABLE_NAME=your-dynamodb-table-name
ALLOWED_ORIGIN=http://breast-cancer-prediction-site.s3-website.ap-south-1.amazonaws.com
```

6. Give the Lambda IAM role permission to write to DynamoDB:

```text
dynamodb:PutItem
```

7. Create an API Gateway HTTP API route:

```text
POST /predict
```

8. Connect the route to the Lambda function.
9. Enable CORS for your S3 website origin.
10. Update `API_ENDPOINT` in `breast-cancer-prediction.html` if your API URL is different.

## API Payload

The frontend sends this shape to Lambda:

```json
{
  "age": 45,
  "lump_present": 0,
  "pain_in_breast": 1,
  "skin_dimpling": 0,
  "nipple_discharge": 0,
  "family_history": 1,
  "mammogram_abnormality": 0,
  "tumor_size_mm": null,
  "bmi": null,
  "glucose_level": null,
  "blood_pressure": null,
  "cholesterol": null,
  "local_guidance_level": "moderate",
  "local_guidance_score": 3
}
```

The Lambda returns:

```json
{
  "id": "generated-record-id",
  "guidance": {
    "source": "ml_model",
    "level": "moderate",
    "probability": 0.5271,
    "score": 52.71,
    "message": "The ML model found a moderate pattern match. A routine clinical check may help."
  },
  "model": {
    "available": true,
    "type": "logistic_regression",
    "version": "1.0.0"
  },
  "message": "Awareness guidance generated successfully."
}
```

## Current Limitations

- The app includes a deployed lightweight ML model artifact for Lambda, but it is educational and not clinically validated.
- The dataset is curated for project demonstration and should not be treated as clinical evidence.
- Health-related data should be handled carefully; avoid collecting names, phone numbers, exact addresses, or other personal identifiers.

## Recommended Next Improvements

- Improve dataset quality and retrain with clinically meaningful, validated data.
- Add model explainability output such as top contributing features.
- Add Lambda unit tests and CI.
- Add API Gateway custom domain and HTTPS frontend through CloudFront.
- Add privacy policy text if the app is used beyond a class/demo setting.
