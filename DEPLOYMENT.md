# Deployment Guide

This project is ready for S3 static website hosting, Lambda, API Gateway, and DynamoDB.

## Fixing 403 Forbidden on S3

If the website shows:

```text
403 Forbidden
Code: AllAccessDisabled
Message: All access to this object has been disabled
```

the problem is in S3 access settings, not the HTML code.

### Required S3 Settings

Bucket name:

```text
breast-cancer-prediction-site
```

Static website hosting:

```text
Enabled
Index document: index.html
```

Files to upload to the bucket root:

```text
index.html
breast-cancer-prediction.html
```

### Public Access

In the bucket permissions tab:

1. Open **Block public access**.
2. Turn off bucket-level public access blocking.
3. Save changes.
4. Open **Bucket policy**.
5. Paste the contents of `aws/s3-bucket-policy-public-read.json`.

If the page still shows `AllAccessDisabled`, open the S3 service-level **Block Public Access settings for this account** and check whether account-level blocking is overriding the bucket policy.

## Manual S3 Upload

From the repo root:

```bash
aws s3 cp index.html s3://breast-cancer-prediction-site/index.html --content-type text/html
aws s3 cp breast-cancer-prediction.html s3://breast-cancer-prediction-site/breast-cancer-prediction.html --content-type text/html
```

Then open:

```text
http://breast-cancer-prediction-site.s3-website.ap-south-1.amazonaws.com
```

## GitHub Actions Deployment

The workflow in `.github/workflows/deploy-s3.yml` can upload the frontend to S3 every time `main` is pushed.

Add these GitHub repository secrets:

```text
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
```

The IAM user or role needs:

```text
s3:PutObject
s3:PutObjectAcl
s3:DeleteObject
s3:ListBucket
```

for this bucket:

```text
breast-cancer-prediction-site
```

## Lambda Deployment Files

Upload both files together:

```text
aws/lambda_function.py
aws/model.json
```

API Gateway route:

```text
POST /predict
```

If your deployed API URL changes, update `API_ENDPOINT` in both:

```text
index.html
breast-cancer-prediction.html
```
