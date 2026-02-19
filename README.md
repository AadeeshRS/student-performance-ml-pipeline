# 🎓 Student Performance Prediction — ML Pipeline

![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-Web_App-000000?logo=flask&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker&logoColor=white)
![AWS](https://img.shields.io/badge/AWS-ECR%20%2F%20ECS-FF9900?logo=amazonaws&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikit-learn&logoColor=white)

An **end-to-end machine learning project** that predicts a student's math score based on demographic and academic features. Built with a modular ML pipeline, a Flask web interface, Docker containerization, and CI/CD deployment to AWS.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Project Architecture](#project-architecture)
- [Tech Stack](#tech-stack)
- [Dataset](#dataset)
- [ML Models](#ml-models)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Training the Model](#training-the-model)
- [Running the Web App](#running-the-web-app)
- [Docker](#docker)
- [CI/CD & Deployment](#cicd--deployment)
- [Screenshots](#screenshots)
- [License](#license)

---

## Overview

This project implements a complete ML pipeline from data ingestion to deployment:

1. **Data Ingestion** — Reads raw CSV data, performs train/test split
2. **Data Transformation** — Applies imputation, scaling, and one-hot encoding via sklearn pipelines
3. **Model Training** — Evaluates 7 regression models with hyperparameter tuning, selects the best
4. **Prediction Pipeline** — Loads saved model & preprocessor for real-time inference
5. **Web App** — Flask-based UI where users input features and get predicted math scores
6. **Deployment** — Dockerized and deployed to AWS ECR/ECS via GitHub Actions

---

## Project Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        GitHub Actions CI/CD                     │
│              (Build Docker Image → Push to ECR → Deploy ECS)    │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                         Flask Web App                           │
│                    (app.py / application.py)                     │
│         User submits form → Predict Pipeline → Result           │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                        ML Pipeline (src/)                       │
│                                                                 │
│  ┌──────────────┐  ┌──────────────────┐  ┌─────────────────┐   │
│  │   Data       │→ │   Data           │→ │   Model         │   │
│  │   Ingestion  │  │   Transformation │  │   Trainer       │   │
│  └──────────────┘  └──────────────────┘  └─────────────────┘   │
│                                                                 │
│  Artifacts: model.pkl, preprocessor.pkl, train/test CSVs        │
└─────────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Category | Technologies |
|----------|-------------|
| **Language** | Python 3.10 |
| **ML Libraries** | scikit-learn, CatBoost, XGBoost |
| **Web Framework** | Flask |
| **Data** | Pandas, NumPy, Seaborn, Matplotlib |
| **Serialization** | dill |
| **Containerization** | Docker |
| **CI/CD** | GitHub Actions |
| **Cloud** | AWS ECR, AWS ECS |

---

## Dataset

**Source:** `notebook/Data/StudentsPerformance.csv`

**Target variable:** `math score`

**Features:**

| Feature | Type | Values |
|---------|------|--------|
| `gender` | Categorical | male, female |
| `race/ethnicity` | Categorical | group A–E |
| `parental level of education` | Categorical | some high school → master's degree |
| `lunch` | Categorical | standard, free/reduced |
| `test preparation course` | Categorical | none, completed |
| `reading score` | Numerical | 0–100 |
| `writing score` | Numerical | 0–100 |

---

## ML Models

The pipeline evaluates **7 regression models** with hyperparameter tuning and selects the best based on R² score:

| Model | Hyperparameters Tuned |
|-------|----------------------|
| Linear Regression | — |
| Decision Tree | criterion, max_depth, min_samples_split |
| Random Forest | n_estimators, max_depth, min_samples_split |
| Gradient Boosting | n_estimators, learning_rate, max_depth |
| XGBoost | n_estimators, learning_rate, max_depth |
| AdaBoost | n_estimators, learning_rate |
| K-Neighbors | n_neighbors, weights, p |

> The best model is automatically saved to `artifact/model.pkl` if its R² score ≥ 0.6.

---

## Project Structure

```
├── .github/
│   └── workflows/
│       └── main.yml              # CI/CD pipeline (ECR + ECS deploy)
├── .ebextensions/                # AWS Elastic Beanstalk config
├── artifact/                     # Generated artifacts (model, preprocessor, data)
├── notebook/
│   └── Data/
│       └── StudentsPerformance.csv
├── src/
│   ├── __init__.py
│   ├── exception.py              # Custom exception handler
│   ├── logger.py                 # Logging configuration
│   ├── utils.py                  # Utility functions (save_object, evaluate_models)
│   ├── components/
│   │   ├── data_ingestion.py     # Read data, train/test split
│   │   ├── data_transformation.py # Preprocessing pipeline
│   │   └── model_trainer.py      # Model training & selection
│   └── pipeline/
│       ├── predict_pipeline.py   # Inference pipeline
│       └── train_pipeline.py     # Training orchestration
├── templates/
│   ├── index.html                # Landing page
│   └── home.html                 # Prediction form & results
├── app.py                        # Flask entry point
├── application.py                # Flask entry point (alternate)
├── Dockerfile                    # Docker container config
├── requirements.txt              # Python dependencies
├── setup.py                      # Package setup
└── README.md
```

---

## Setup & Installation

### Prerequisites
- Python 3.10+
- pip

### Install

```bash
# Clone the repository
git clone https://github.com/AadeeshRS/student-performance-ml-pipeline.git
cd student-performance-ml-pipeline

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

---

## Training the Model

```bash
python src/components/data_ingestion.py
```

This will:
1. Read the CSV dataset
2. Split into train/test sets → `artifact/train.csv`, `artifact/test.csv`
3. Build preprocessing pipeline → `artifact/preprocessor.pkl`
4. Train & evaluate all models → `artifact/model.pkl`

---

## Running the Web App

```bash
python app.py
```

Open **http://localhost:5000** in your browser.

- **`/`** — Landing page
- **`/predictdata`** — Prediction form (enter features → get predicted math score)

---

## Docker

### Build the image

```bash
docker build -t student-performance-ml .
```

### Run the container

```bash
docker run -p 5000:5000 student-performance-ml
```

Visit **http://localhost:5000**

### Push to Docker Hub

```bash
docker tag student-performance-ml aadeeshrs/student-performance-ml:latest
docker push aadeeshrs/student-performance-ml:latest
```

---

## CI/CD & Deployment

The project uses **GitHub Actions** to automatically build and deploy to **AWS ECR/ECS** on every push to `main`.

### Workflow: `.github/workflows/main.yml`

```
Push to main → Build Docker Image → Push to ECR → Deploy to ECS
```

### Required GitHub Secrets

| Secret | Description |
|--------|-------------|
| `AWS_ACCESS_KEY_ID` | IAM access key ID |
| `AWS_SECRET_ACCESS_KEY` | IAM secret access key |
| `AWS_REGION` | AWS region (e.g. `ap-south-1`) |
| `AWS_ECR_LOGIN_URI` | ECR registry URI (e.g. `123456789.dkr.ecr.ap-south-1.amazonaws.com`) |
| `ECR_REPOSITORY_NAME` | ECR repository name |

Add these in: **GitHub repo → Settings → Secrets and variables → Actions**

---


## License

This project is for **educational purposes**.

---

<p align="center">
  Made with ❤️ by <a href="https://github.com/AadeeshRS">Aadeesh RS</a>
</p>