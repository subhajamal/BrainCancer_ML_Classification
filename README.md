# BrainCancer_ML_Classification
# 🧠 Brain Cancer Classification using Radiomics & Machine Learning

This repository contains our final team project for **HIDS509: Health Informatics**, where we applied machine learning models to classify brain cancer types using radiomic features from **TCGA** and **REMBRANDT** MRI datasets.

## 🧪 Project Overview

We extracted Pyradiomics features from T1-weighted MRI scans and trained models using TCGA-labeled data to predict cancer type in REMBRANDT patients.

### 🔍 Overview:
1. **Feature Extraction**: Pyradiomics on 64 REMBRANDT patients (T1 modality + segmented masks)
2. **Model Training**: Random Forest & SVM on TCGA radiomic features
3. **Prediction**: Apply trained model on REMBRANDT dataset
4. **Evaluation**: Accuracy, confusion matrix, precision, recall
5. **Reporting**: Written report + slide presentation

## 📂 Data Access

Due to size, all data files are hosted on Google Drive:

- [REMBRANDT Dataset](https://drive.google.com/drive/folders/1yGBBQaaEVAXmg_nPgFsTokFldnaM7-z8)
- [TCGA Radiomic Features](https://drive.google.com/drive/folders/1PqjNQHGBCJLmR8LQQMAJKf6lVAwX83Gd)
- [TCGA Clinical Data](https://docs.google.com/spreadsheets/d/1MN5nVm8ZxSOib-Go6BOGr9DUbqvbqnSX)

## ⚙️ Tools & Libraries

- Python, scikit-learn, Pyradiomics, NumPy, Pandas, matplotlib
- MRI Format: NIfTI (.nii)
- Image Segmentation: GlisterBoost (pre-generated masks)

## 🤖 Models Used

- Random Forest (with feature importance)
- Support Vector Machines (SVM)

## 📈 Results

- Accuracy metrics and confusion matrix included in final report
- Feature importance ranked and interpreted



## 📎 Report & Presentation

The full report and presentation slides are included in the `report/` folder.
