import joblib
import pandas as pd
from models.data_preprocessing import DataPreprocessor

class ModelPredictor:
    def __init__(self):
        self.preprocessor = DataPreprocessor()

    def load_model(self, model_path):
        try:
            model_data = joblib.load(model_path)
            return model_data
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return None

    def predict(self, model_path, X_new):
        model_data = self.load_model(model_path)
        if model_data is None:
            return None, None

        model = model_data['model']
        preprocessor = model_data['preprocessor']
        model_type = model_data.get('model_type', 'standard')

        # Preprocess input data
        if model_type == 'CatBoost':
            X_processed = self.preprocessor.transform(X_new, preprocessor, model_type='catboost')
        else:
            X_processed = self.preprocessor.transform(X_new, preprocessor, model_type='standard')

        predictions = model.predict(X_processed)
        prediction_proba = model.predict_proba(X_processed)
        return predictions, prediction_proba