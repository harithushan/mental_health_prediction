import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
import mlflow.catboost
from datetime import datetime
import joblib
import os
from pathlib import Path
from config import MODELS_DIR, CATEGORICAL_FEATURES, MODEL_CONFIGS
from models.data_preprocessing import DataPreprocessor

class ModelTrainer:
    """Main model training class with MLflow integration."""
    
    def __init__(self):
        self.models = {}
        self.best_models = {}
        self.preprocessor = DataPreprocessor()
        self.results = {}
        
    def get_model_instance(self, model_name, **params):
        """Get model instance based on name and parameters."""
        if model_name == 'GradientBoosting':
            return GradientBoostingClassifier(**params)
        elif model_name == 'XGBoost':
            return XGBClassifier(**params)
        elif model_name == 'LightGBM':
            return LGBMClassifier(**params)
        elif model_name == 'CatBoost':
            # Add categorical feature indices for CatBoost
            categorical_feature_indices = list(range(len(CATEGORICAL_FEATURES)))
            return CatBoostClassifier(cat_features=categorical_feature_indices, **params)
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    def train_single_model(self, X_train, X_test, y_train, y_test, model_name, 
                          hyperparameter_tuning=False, cv_folds=5):
        """Train a single model with optional hyperparameter tuning."""
        
        with mlflow.start_run(run_name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Log basic info
            mlflow.log_param("model_type", model_name)
            mlflow.log_param("training_samples", len(X_train))
            mlflow.log_param("test_samples", len(X_test))
            mlflow.log_param("hyperparameter_tuning", hyperparameter_tuning)
            
            try:
                # Prepare data based on model type
                if model_name == 'CatBoost':
                    X_train_processed, train_preprocessor = self.preprocessor.fit_transform(
                        X_train, y_train, model_type='catboost'
                    )
                    X_test_processed = self.preprocessor.transform(
                        X_test, train_preprocessor, model_type='catboost'
                    )
                else:
                    X_train_processed, train_preprocessor = self.preprocessor.fit_transform(
                        X_train, y_train, model_type='standard'
                    )
                    X_test_processed = self.preprocessor.transform(
                        X_test, train_preprocessor, model_type='standard'
                    )
                
                if hyperparameter_tuning:
                    # Hyperparameter tuning
                    param_grid = MODEL_CONFIGS.get(model_name, {})
                    model = self.get_model_instance(model_name)
                    
                    # Create pipeline
                    pipeline = Pipeline([
                        ('model', model)
                    ])
                    
                    # Adjust parameter names for pipeline
                    param_grid_pipeline = {f'model__{k}': v for k, v in param_grid.items()}
                    
                    # Randomized search
                    random_search = RandomizedSearchCV(
                        pipeline, 
                        param_grid_pipeline, 
                        n_iter=10, 
                        cv=cv_folds, 
                        scoring='accuracy',
                        n_jobs=-1,
                        random_state=42
                    )
                    
                    random_search.fit(X_train_processed, y_train)
                    best_model = random_search.best_estimator_
                    best_params = random_search.best_params_
                    
                    # Log best parameters
                    for param, value in best_params.items():
                        mlflow.log_param(param, value)
                    
                    mlflow.log_metric("best_cv_score", random_search.best_score_)
                    
                else:
                    # Use default parameters
                    model = self.get_model_instance(model_name)
                    best_model = Pipeline([('model', model)])
                    best_model.fit(X_train_processed, y_train)
                
                # Make predictions
                y_pred = best_model.predict(X_test_processed)
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                
                # Cross-validation score
                cv_scores = cross_val_score(best_model, X_train_processed, y_train, 
                                          cv=cv_folds, scoring='accuracy')
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()
                
                # Log metrics
                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("cv_mean", cv_mean)
                mlflow.log_metric("cv_std", cv_std)
                
                # Log model
                if model_name == 'XGBoost':
                    mlflow.xgboost.log_model(best_model.named_steps['model'], "model")
                elif model_name == 'LightGBM':
                    mlflow.lightgbm.log_model(best_model.named_steps['model'], "model")
                elif model_name == 'CatBoost':
                    mlflow.catboost.log_model(best_model.named_steps['model'], "model")
                else:
                    mlflow.sklearn.log_model(best_model, "model")
                
                # Save model locally
                model_path = MODELS_DIR / f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
                joblib.dump({
                    'model': best_model,
                    'preprocessor': train_preprocessor,
                    'model_type': model_name
                }, model_path)
                
                # Store results
                self.results[model_name] = {
                    'model': best_model,
                    'preprocessor': train_preprocessor,
                    'accuracy': accuracy,
                    'cv_mean': cv_mean,
                    'cv_std': cv_std,
                    'predictions': y_pred,
                    'model_path': str(model_path)
                }
                
                return {
                    'model_name': model_name,
                    'accuracy': accuracy,
                    'cv_mean': cv_mean,
                    'cv_std': cv_std,
                    'model_path': str(model_path)
                }
                
            except Exception as e:
                mlflow.log_param("error", str(e))
                print(f"Error training {model_name}: {str(e)}")
                return None
    
    def train_all_models(self, df, target_column, test_size=0.2, 
                        hyperparameter_tuning=False, cv_folds=5, selected_models= None):
        """Train all available models."""
        
        # Prepare data
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Model names to train
        if selected_models is None:
            model_names = ['GradientBoosting', 'XGBoost', 'LightGBM', 'CatBoost']
        else:
            model_names = selected_models
       
        
        results = []
        
        for model_name in model_names:
            print(f"Training {model_name}...")
            result = self.train_single_model(
                X_train, X_test, y_train, y_test, model_name, 
                hyperparameter_tuning, cv_folds
            )
            if result:
                results.append(result)
        
        return results, X_test, y_test
    
    def get_best_model(self):
        """Get the best performing model."""
        if not self.results:
            return None
        
        best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['accuracy'])
        return best_model_name, self.results[best_model_name]
    
    def load_model(self, model_path):
        """Load a saved model."""
        try:
            model_data = joblib.load(model_path)
            return model_data
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return None
    
    def predict_with_model(self, model_path, X_new):
        """Make predictions with a saved model."""
        try:
            model_data = self.load_model(model_path)
            if model_data is None:
                return None
            
            model = model_data['model']
            preprocessor = model_data['preprocessor']
            model_type = model_data.get('model_type', 'standard')
            
            # Preprocess the new data
            if model_type == 'CatBoost':
                X_processed = self.preprocessor.transform(X_new, preprocessor, model_type='catboost')
            else:
                X_processed = self.preprocessor.transform(X_new, preprocessor, model_type='standard')
            
            # Make predictions
            predictions = model.predict(X_processed)
            prediction_proba = model.predict_proba(X_processed)
            
            return predictions, prediction_proba
            
        except Exception as e:
            print(f"Error making predictions: {str(e)}")
            return None, None
        