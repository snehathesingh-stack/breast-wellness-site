# 🩺 Breast Wellness: Cloud-Based Breast Cancer Prediction and Awareness System  

## 📘 Project Overview  
**Breast Wellness** is a cloud-based breast cancer prediction and awareness application developed as part of the Data Mining.  

The system uses **data mining and machine learning** to predict breast cancer risk levels (Low, Medium, High) based on user input. It is hosted on **Amazon Web Services (AWS)** using **S3**, **Lambda**, **API Gateway**, and **DynamoDB**.  

This project combines technology and health awareness by allowing users to:
- Input key parameters (age, family history, presence of lump, etc.)
- Receive instant risk predictions powered by a Logistic Regression model
- Learn about breast self-examination and preventive screening  

**Live Website:**  
👉 [http://breast-cancer-prediction-site.s3-website.ap-south-1.amazonaws.com](http://breast-cancer-prediction-site.s3-website.ap-south-1.amazonaws.com)  

---
**Google Colab Notebook:**  
[Open Colab](https://colab.research.google.com/gist/snehathesingh-stack/bcbf902a604cc08ffcbaf16b2fb2872f/untitled2.ipynb) 

## 🧠 Features  

- **Machine Learning Model:** Logistic Regression trained on the Breast Cancer Wisconsin dataset.  
- **Cloud Integration:**  
  - Frontend hosted on **AWS S3**  
  - Backend executed using **AWS Lambda** and **API Gateway**  
  - Database using **AWS DynamoDB**  
- **User Awareness:** Includes self-check guidance and educational resources.  
- **Accuracy:** 95% model performance on validation data.  

---

## ⚙️ Tech Stack  

| Component | Technology Used |
|------------|-----------------|
| **Frontend** | HTML5, CSS3, JavaScript |
| **Model Development** | Python (pandas, numpy, scikit-learn) |
| **Backend** | AWS Lambda (Python runtime) |
| **Database** | AWS DynamoDB |
| **Cloud Hosting** | AWS S3, API Gateway |
| **Development Platform** | Google Colab |
| **Version Control** | GitHub |

---

## 📊 Dataset Description (Used For data mining)


**Dataset Name:** Breast Cancer Wisconsin (Diagnostic) Dataset  
- **Records:** 569 instances  
- **Attributes:** 32 (30 features + ID + diagnosis)  
- **Target Variable:**  
  - *M* = Malignant (Cancerous)  
  - *B* = Benign (Non-cancerous)  

**Dataset Sources:**  
- Zenodo DOI: [https://doi.org/10.5281/zenodo.5084116](https://doi.org/10.5281/zenodo.5084116)  
- IEEE DataPort: [https://dx.doi.org/10.21227/6sda-hn78](https://dx.doi.org/10.21227/6sda-hn78)  

---

## 🧩 Project Architecture  

```plaintext
User Input (Web Form)
        ↓
Frontend (AWS S3)
        ↓
API Gateway → AWS Lambda (Python)
        ↓
Model (Logistic Regression .pkl)
        ↓
Prediction Result (JSON)
        ↓
DynamoDB (Logs Stored)
