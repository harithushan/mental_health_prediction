import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from config import (
    VALID_PROFESSIONS, VALID_SLEEP_DURATION, VALID_DEGREES, VALID_DIETARY_HABITS,
    NUMERICAL_FEATURES, CATEGORICAL_FEATURES, ONEHOT_FEATURES, FREQUENCY_FEATURES
)

class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Custom frequency encoder for categorical features."""
    
    def __init__(self):
        self.encoding_map = {}
    
    def fit(self, X, y=None):
        for col in X.columns:
            frequency_encoding = X[col].value_counts() / len(X)
            self.encoding_map[col] = frequency_encoding.to_dict()
        return self
    
    def transform(self, X):
        X = X.copy()
        for col in X.columns:
            X[col] = X[col].map(self.encoding_map[col]).fillna(0)
        return X

class DataPreprocessor:
    """Main data preprocessing class."""
    
    def __init__(self):
        self.preprocessor = None
        self.catboost_preprocessor = None
        self.feature_names = None
    
    def clean_data(self, df):
        """Clean and validate the input data."""
        df = df.copy()
        
        # Clean profession
        if 'Profession' in df.columns:
            df['Profession'] = df['Profession'].apply(
                lambda x: x if pd.isna(x) or x in VALID_PROFESSIONS else 'others'
            )
        
        # Clean sleep duration
        if 'Sleep Duration' in df.columns:
            df['Sleep Duration'] = df['Sleep Duration'].apply(
                lambda x: x if x in VALID_SLEEP_DURATION else 'Unknown'
            )
        
        # Clean degree
        if 'Degree' in df.columns:
            df['Degree'] = df['Degree'].apply(
                lambda x: x if pd.isna(x) or x in VALID_DEGREES else 'Unknown'
            )
        
        # Clean dietary habits
        if 'Dietary Habits' in df.columns:
            df['Dietary Habits'] = df['Dietary Habits'].apply(
                lambda x: x if x in VALID_DIETARY_HABITS else 'Unknown'
            )
        
        # Drop irrelevant columns
        columns_to_drop = ['Name', 'City', 'id']
        df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
        
        return df
    
    def handle_missing_values(self, df):
        """Handle missing values in the dataset."""
        df = df.copy()
        
        # Fill numerical features with 0
        numerical_cols = ['Job Satisfaction', 'Academic Pressure', 'Study Satisfaction', 
                         'Financial Stress', 'Work Pressure']
        
        for col in numerical_cols:
            if col in df.columns:
                df[col].fillna(0, inplace=True)
        
        # Fill categorical features with mode
        categorical_cols = ['Profession', 'Dietary Habits', 'Degree']
        
        for col in categorical_cols:
            if col in df.columns:
                mode_value = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
                df[col].fillna(mode_value, inplace=True)
        
        # Handle CGPA separately
        if 'CGPA' in df.columns:
            df['CGPA'].fillna(df['CGPA'].mean(), inplace=True)
        
        return df
    
    def create_preprocessing_pipeline(self, model_type='standard'):
        """Create preprocessing pipeline based on model type."""
        
        if model_type == 'catboost':
            # For CatBoost, use simpler preprocessing
            categorical_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing'))
            ])
            
            numerical_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ])
            
            self.catboost_preprocessor = ColumnTransformer([
                ('cat', categorical_pipeline, CATEGORICAL_FEATURES),
                ('num', numerical_pipeline, NUMERICAL_FEATURES)
            ])
            
            return self.catboost_preprocessor
        
        else:
            # For other models, use frequency and one-hot encoding
            categorical_pipeline_f = Pipeline([
                ('frequency_encoder', FrequencyEncoder())
            ])
            
            categorical_pipeline_one = Pipeline([
                ('onehot_encoder', OneHotEncoder(handle_unknown='ignore', drop='first'))
            ])
            
            numerical_pipeline = Pipeline([
                ('scaler', StandardScaler())
            ])
            
            self.preprocessor = ColumnTransformer([
                ('frequency', categorical_pipeline_f, FREQUENCY_FEATURES),
                ('onehot', categorical_pipeline_one, ONEHOT_FEATURES),
                ('num', numerical_pipeline, NUMERICAL_FEATURES)
            ])
            
            return self.preprocessor
    
    def fit_transform(self, X, y=None, model_type='standard'):
        """Fit and transform the data."""
        # Clean the data
        X_clean = self.clean_data(X)
        X_clean = self.handle_missing_values(X_clean)
        
        # Create and fit preprocessing pipeline
        preprocessor = self.create_preprocessing_pipeline(model_type)
        
        # Fit and transform
        X_transformed = preprocessor.fit_transform(X_clean)
        
        return X_transformed, preprocessor
    
    def transform(self, X, preprocessor, model_type='standard'):
        """Transform new data using fitted preprocessor."""
        # Clean the data
        X_clean = self.clean_data(X)
        X_clean = self.handle_missing_values(X_clean)
        
        # Transform
        X_transformed = preprocessor.transform(X_clean)
        
        return X_transformed
    
    def get_feature_names(self, preprocessor):
        """Get feature names after preprocessing."""
        try:
            if hasattr(preprocessor, 'get_feature_names_out'):
                return preprocessor.get_feature_names_out()
            else:
                return None
        except:
            return None

def preprocess_data(df, target_column=None, model_type='standard'):
    """Convenience function to preprocess data."""
    preprocessor = DataPreprocessor()
    
    if target_column and target_column in df.columns:
        X = df.drop(columns=[target_column])
        y = df[target_column]
        X_transformed, fitted_preprocessor = preprocessor.fit_transform(X, y, model_type)
        return X_transformed, y, fitted_preprocessor
    else:
        X_transformed, fitted_preprocessor = preprocessor.fit_transform(df, model_type=model_type)
        return X_transformed, fitted_preprocessor