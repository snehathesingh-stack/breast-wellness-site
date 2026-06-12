# Breast Wellness: Cloud-Based Awareness and Risk Guidance Tool

Breast Wellness is a static web application for breast health awareness. It helps users record basic self-check information, receive non-diagnostic guidance, save entries locally, export check-ins, and optionally run an anonymous ML inference request through a Vercel serverless API.

This project is designed for deployment on Vercel:

- Frontend: Vercel static hosting
- Backend: Vercel Python serverless function
- Optional AWS backend: Lambda, API Gateway, and DynamoDB
- ML: k-nearest-neighbors model selected from a pure-Python training script

Recommended live hosting:

https://vercel.com/new/clone?repository-url=https://github.com/snehathesingh-stack/breast-wellness-site

## Important Medical Disclaimer

This application is for awareness and education only. It does not diagnose breast cancer, estimate a clinical probability, replace screening, or replace advice from a qualified healthcare professional.

## Features

- Breast wellness self-check form
- Age and symptom-based awareness guidance
- Clear low, moderate, and higher concern messaging
- Optional anonymous AI cloud check
- Visible Vercel ML success/failure status in the frontend
- Explainable AI response with top contributing model factors
- Local browser storage for private check-in history
- CSV export
- Monthly Google Calendar reminder
- Clinic search shortcut
- Model report page with metrics, confusion matrix, comparison, and limitations
- High-accuracy Wisconsin Diagnostic Breast Cancer model track
- AWS Lambda starter backend with optional DynamoDB persistence
- Reproducible ML training script
- Exported Lambda-ready model artifact
- Cloud ML probability returned from Vercel serverless API

## Repository Structure

```text
.
├── breast-cancer-prediction.html   # Static frontend for S3 hosting
├── index.html                      # S3 website root page
├── model-report.html               # ML model report page
├── diagnostic-report.html          # Diagnostic model report page
├── vercel.json                     # Vercel static hosting config
├── Dataset_file.xlsx               # Curated project dataset
├── DEPLOYMENT.md                   # S3 403 fix and deployment guide
├── README.md                       # Project documentation
├── api/
│   ├── predict.py                  # Vercel Python ML inference endpoint
│   ├── diagnostic.py               # Vercel WDBC diagnostic model endpoint
│   └── health.py                   # Vercel model health endpoint
├── .github/workflows/
│   ├── validate.yml                # CI validation checks
│   └── deploy-s3.yml               # Optional GitHub Actions S3 deployment
├── ml/
│   ├── train_model.py              # Reproducible wellness model training
│   ├── train_diagnostic_model.py   # Reproducible WDBC diagnostic model training
│   ├── model_report.json
│   └── diagnostic_model_report.json
├── data/
│   └── wdbc.csv                    # Wisconsin Diagnostic Breast Cancer data
├── tests/
│   ├── run_tests.py                # Dependency-free test runner
│   ├── test_api.py
│   └── test_static_assets.py
└── aws/
    ├── lambda_function.py          # Lambda handler for API Gateway
    ├── model.json                  # Exported trained model artifact
    ├── s3-bucket-policy-public-read.json
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

The wellness model can also be regenerated from a balanced synthetic dataset that covers low, moderate, and high questionnaire patterns. This improves the educational ML demo, but it is still not clinical evidence.

The diagnostic model uses the Wisconsin Diagnostic Breast Cancer dataset requested from Kaggle:

- https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data
- https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset

Because Kaggle downloads require authentication, the training script also supports the public UCI WDBC data file as a fallback mirror for the same benchmark-style diagnostic data.

## Machine Learning

The project now includes a reproducible pure-Python ML pipeline:

```text
python ml/train_model.py
python ml/train_model.py --generate-synthetic --records 1500
```

The script:

1. Reads `Dataset_file.xlsx` directly.
2. Extracts the curated breast cancer awareness dataset.
3. Trains Logistic Regression as an interpretable baseline.
4. Benchmarks lightweight kNN candidates.
5. Selects the best model on a fixed 20% test split.
6. Writes `aws/model.json` for Vercel/AWS inference.

Current generated model metrics:

| Metric | Value |
| --- | ---: |
| Test records | 240 |
| Accuracy | 0.9083 |
| Precision | 0.96 |
| Recall | 0.8759 |
| F1 | 0.916 |
| Log loss | 0.3336 |

These metrics come from the balanced synthetic wellness dataset. They show the deployed demo model is learning the synthetic questionnaire patterns, but the model is still included to demonstrate an end-to-end ML workflow, not to provide clinical-grade prediction.

## Diagnostic Model

The diagnostic model uses 30 Wisconsin Diagnostic Breast Cancer numeric cell-nuclei features. It is trained separately from the wellness questionnaire model:

```text
python ml/train_diagnostic_model.py
```

Current diagnostic model metrics:

| Metric | Value |
| --- | ---: |
| Test records | 113 |
| Accuracy | 0.9912 |
| Precision | 0.9744 |
| Recall | 1.0 |
| F1 | 0.987 |
| Log loss | 0.0374 |

This higher accuracy is possible because the diagnostic dataset contains biopsy image-derived numeric features, not broad self-check questionnaire fields.

## Vercel Deployment

The easiest working deployment is Vercel:

1. Open this import link:

```text
https://vercel.com/new/clone?repository-url=https://github.com/snehathesingh-stack/breast-wellness-site
```

2. Choose your GitHub account.
3. Keep the default project settings.
4. Click **Deploy**.

Vercel will serve:

```text
/
```

from `index.html`, and it will serve ML predictions from:

```text
/api/predict
```

It also exposes model service health at:

```text
/api/health
```

The model report page is available at:

```text
/model-report.html
```

The diagnostic model API and report are available at:

```text
/api/diagnostic
/diagnostic-report.html
```

The frontend uses a relative API URL:

```text
/api/predict
```

so it works automatically after Vercel deployment.

If someone else opens the site and Vercel asks for access, check that you are sharing the production URL, for example:

```text
https://breast-wellness-site.vercel.app
```

Preview links under a personal/team project URL can be protected by Vercel Deployment Protection. In Vercel, open **Project Settings > Deployment Protection** and turn off Vercel Authentication or Password Protection for Production if this is meant to be publicly visible.

## Testing and Validation

Run the dependency-free test suite:

```text
python tests/run_tests.py
```

The GitHub Actions workflow `.github/workflows/validate.yml` runs:

- Python syntax checks
- JSON artifact validation
- Frontend JavaScript syntax checks
- API and static asset tests

## Local Frontend Setup

Open the HTML file directly in a browser for local testing:

```text
index.html
```

## Optional AWS S3 Setup

AWS is no longer required for the live frontend. If you still want S3:

1. Create an S3 bucket.
2. Enable static website hosting.
3. Upload `index.html` and `breast-cancer-prediction.html`.
4. Set the index document to `index.html`.
5. Configure public read access using `aws/s3-bucket-policy-public-read.json` or use CloudFront with origin access control.

If the S3 website shows `403 Forbidden` or `AllAccessDisabled`, follow `DEPLOYMENT.md`. If the AWS account is closed, use Vercel instead.

Optional AWS automatic deployment is available with `.github/workflows/deploy-s3.yml`. Add `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` as GitHub repository secrets, then each push to `main` can upload the frontend files to S3.

## Optional AWS Backend Setup

1. Run `python ml/train_model.py` to regenerate `aws/model.json` if needed.
   - To regenerate using a fresh synthetic wellness dataset, run `python ml/train_model.py --generate-synthetic --records 1500`.
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

## Vercel API Payload

The frontend sends this shape to `/api/predict`:

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

The Vercel API returns:

```json
{
  "id": "generated-record-id",
  "guidance": {
    "source": "vercel_ml_model",
    "level": "moderate",
    "probability": 0.5271,
    "score": 52.71,
    "message": "The ML model found a moderate pattern match. A routine clinical check may help.",
    "top_factors": []
  },
  "model": {
    "available": true,
    "type": "logistic_regression",
    "version": "1.0.0"
  },
  "message": "Vercel ML inference completed successfully."
}
```

## Current Limitations

- The app includes a deployed lightweight ML model artifact for Vercel and Lambda, but it is educational and not clinically validated.
- The dataset is curated for project demonstration and should not be treated as clinical evidence.
- Health-related data should be handled carefully; avoid collecting names, phone numbers, exact addresses, or other personal identifiers.

## Recommended Next Improvements

- Improve dataset quality and retrain with clinically meaningful, validated data.
- Add API and frontend tests.
- Add a short public deployment verification checklist for Vercel production releases.
- Add API Gateway custom domain and HTTPS frontend through CloudFront.
- Add privacy policy text if the app is used beyond a class/demo setting.
