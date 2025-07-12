import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "saved_models"
TRAIN_DATA_DIR = DATA_DIR / "train"
TEST_DATA_DIR = DATA_DIR / "test"
PREDICT_DATA_DIR = DATA_DIR / "predict"

# Create directories if they don't exist
for dir_path in [DATA_DIR, MODELS_DIR, TRAIN_DATA_DIR, TEST_DATA_DIR, PREDICT_DATA_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# MLflow configuration
# MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
# MLFLOW_TRACKING_URI = "http://localhost:5000" 
MLFLOW_TRACKING_URI = "http://mlflow:5000"
MLFLOW_EXPERIMENT_NAME = "mental_health_prediction"

# Model configuration
VALID_PROFESSIONS = [
    'Chef', 'Teacher', 'Business Analyst', 'Financial Analyst', 'Chemist', 'Electrician', 
    'Software Engineer', 'Data Scientist', 'Plumber', 'Marketing Manager', 'Accountant', 
    'Entrepreneur', 'HR Manager', 'UX/UI Designer', 'Content Writer', 'Educational Consultant', 
    'Civil Engineer', 'Manager', 'Pharmacist', 'Architect', 'Mechanical Engineer', 
    'Customer Support', 'Consultant', 'Judge', 'Researcher', 'Pilot', 'Graphic Designer', 
    'Travel Consultant', 'Digital Marketer', 'Lawyer', 'Research Analyst', 'Sales Executive', 
    'Doctor', 'Unemployed', 'Investment Banker', 'Family Consultant', 'Medical Doctor', 
    'Working Professional', 'Student', 'Analyst'
]

VALID_SLEEP_DURATION = [
    'Less than 5 hours', '5-6 hours', '6-7 hours', '6-8 hours', 
    '7-8 hours', '8-9 hours', 'More than 8 hours'
]

VALID_DEGREES = [
    'BHM', 'LLB', 'B.Pharm', 'BBA', 'MCA', 'MD', 'BSc', 'ME', 'B.Arch', 'BCA', 'BE',
    'MA', 'B.Ed', 'B.Com', 'MBA', 'M.Com', 'MHM', 'BA', 'Class 12', 'M.Tech', 'PhD',
    'M.Ed', 'MSc', 'B.Tech', 'LLM', 'MBBS', 'M.Pharm', 'MPA', 'M.Arch', 'BEd', 
    'B.Sc', 'MTech', 'BPharm', 'BPA', 'ACA', 'LHM', 'M.S', 'HCA'
]

VALID_DIETARY_HABITS = ['Healthy', 'Unhealthy', 'Moderate']

# Feature configurations
NUMERICAL_FEATURES = [
    'Age', 'Academic Pressure', 'Work Pressure', 'CGPA', 'Study Satisfaction', 
    'Job Satisfaction', 'Work/Study Hours', 'Financial Stress'
]

CATEGORICAL_FEATURES = [
    'Gender', 'Working Professional or Student', 'Dietary Habits', 
    'Have you ever had suicidal thoughts ?', 'Family History of Mental Illness',
    'Profession', 'Sleep Duration', 'Degree'
]

ONEHOT_FEATURES = [
    'Gender', 'Working Professional or Student', 'Dietary Habits', 
    'Have you ever had suicidal thoughts ?', 'Family History of Mental Illness'
]

FREQUENCY_FEATURES = ['Profession', 'Sleep Duration', 'Degree']

# Model parameters
MODEL_CONFIGS = {
    'GradientBoosting': {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    },
    'XGBoost': {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    },
    'LightGBM': {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    },
    'CatBoost': {
        'iterations': [100, 200, 300],
        'learning_rate': [0.01, 0.08, 0.1],
        'depth': [6, 8, 10]
    }
}