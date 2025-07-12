# 🧠 Mental Health Prediction Dashboard

A comprehensive Streamlit-based application for predicting mental health conditions using machine learning. The dashboard enables seamless data upload, model training, evaluation, prediction, and MLflow experiment tracking — all in one interactive and modular interface.

---

## 🚀 Features

- Upload training, testing, and prediction datasets
- Train machine learning models using configurable settings
- Evaluate model performance with visual metrics
- Make predictions on single or bulk data inputs
- Track experiments with MLflow integration
- Dockerized for easy deployment and scalability

---

## 📁 Project Structure
mental_health_dashboard/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── config.py                       # Configuration settings
├── README.md                       # Project documentation
├── .env                           # Environment variables
├── .gitignore                     # Git ignore file
├── data/
│   ├── train/                     # Training data directory
│   ├── test/                      # Test data directory
│   └── predict/                   # Bulk prediction data directory
├── models/
│   ├── __init__.py
│   ├── data_preprocessing.py      # Data preprocessing utilities
│   ├── model_trainer.py           # Model training pipeline
│   ├── model_predictor.py         # Model prediction utilities
│   └── model_evaluator.py         # Model evaluation utilities
├── utils/
│   ├── __init__.py
│   ├── file_utils.py              # File handling utilities
│   ├── mlflow_utils.py            # MLflow utilities
│   └── visualization.py           # Visualization utilities
├── pages/
│   ├── data_upload.py           # Data upload page
│   ├── model_training.py        # Model training page
│   ├── model_evaluation.py      # Model evaluation page
│   ├── predictions.py           # Prediction page
│   └── experiment_tracking.py   # MLflow experiment tracking page
├── notebooks/
│   └── exploratory_analysis.ipynb # Jupyter notebook for EDA
└── saved_models/                  # Directory for saved model artifacts


## ⚙️ Setup & Usage

### 🔹 Option 1: Virtual Environment (venv)
```bash
python -m venv env
env\Scripts\activate.bat           # For Windows
pip install -r requirements.txt
streamlit run app.py
```

### 🔹 Option 2: Conda Environment
```bash
conda create -p conda_env
conda activate E:\MSC\SEMESTER2\mental_health_prediction\conda_env
pip install -r requirements.txt
streamlit run app.py
```

🐳 Docker Deployment
🧱 Build and Run
```bash
docker build -t mental-health-app .
docker run -p 8501:8501 mental-health-app
```

📦 Docker Compose (Recommended)
```bash
docker-compose up --build
```

Access:
- Streamlit App: http://localhost:8501
- MLflow UI: http://localhost:5000

📜 License
Licensed under the Apache License 2.0.
You may freely use, modify, and distribute this software with proper attribution.




