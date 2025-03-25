# 🧠 Brain Cancer Classification using Radiomics & Machine Learning

This project applies machine learning models to classify brain cancer types using radiomic features extracted from MRI scans. We used the **TCGA** and **REMBRANDT** datasets to build, train, and evaluate models as part of my **HIDS509 Final Project** at Georgetown University.

---

## 📁 Repository Structure

BrainCancer_ML_Classification/ ├── notebooks/ # Jupyter notebooks for each major analysis step │ ├── feature_extraction_rembrandt.ipynb │ ├── model_training_tcga.ipynb │ ├── rembrandt_prediction.ipynb │ ├── prediction_evaluation.ipynb │ └── merged_tcga_rembrandt_analysis.ipynb │ ├── results/ # Output files (radiomics features, clinical data, labels) │ ├── rembrandt_radiomic_features.csv │ ├── extracted_features.csv │ ├── tcga_radiomics_features.csv │ ├── tcga_clinical_labels.xlsx │ └── rembrandt_ground_truth_labels.txt │ ├── requirements.txt # Python dependencies ├── README.md # This file └── .gitignore # Ignore outputs, checkpoints, system files


---

## 🔬 Project Overview

### 🧩 Step 1 – Radiomics Feature Extraction
- Used `pyradiomics` to extract features from T1-weighted MRI scans of 64 REMBRANDT patients.
- Segmentations were provided using the **GlisterBoost** algorithm.

### 🤖 Step 2 – ML Model Training on TCGA
- Trained several machine learning models (SVM, Random Forest, k-NN, Gradient Boosting) using radiomics features from the **TCGA** dataset.
- Performed hyperparameter tuning using `GridSearchCV` and `RandomizedSearchCV`.

### 🔍 Step 3 – Prediction on REMBRANDT Patients
- Applied trained models to predict cancer types in REMBRANDT patients using the features extracted in Step 1.

### 📊 Step 4 – Evaluation
- Compared predictions with ground truth labels using:
  - Confusion matrix
  - Precision, recall, F1-score
  - Classification reports

---

## 📂 Data Access

Due to size, raw data files (MRI scans and segmentation masks) are hosted externally.

**Google Drive (Raw Data):**
- REMBRANDT dataset (images + masks)
- TCGA radiomics + clinical data

📌 *See notebook paths and comments for links & usage.*

---

## 💻 Tools Used

- `pyradiomics`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`
- Data formats: `.nii.gz`, `.csv`, `.xlsx`
- Environment: Google Colab + Google Drive

---

## 📈 Results

We observed promising performance across multiple classifiers. Feature importance was extracted and interpreted for biomedical relevance. Confusion matrices and reports are included in the `prediction_evaluation.ipynb`.

---

## 🎓 Team Acknowledgement

This project was developed as part of **HIDS509: Health Informatics and Data Science** coursework at Georgetown University.

