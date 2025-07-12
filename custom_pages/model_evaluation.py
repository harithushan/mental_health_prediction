import streamlit as st
import pandas as pd
import joblib
from config import MODELS_DIR, TEST_DATA_DIR
from models.model_trainer import ModelTrainer
from utils.visualization import plot_confusion_matrix
from sklearn.metrics import classification_report
import plotly.express as px

def show_model_evaluation_page():
    """Display model evaluation page."""
    st.markdown('<h2 class="sub-header">📊 Model Evaluation</h2>', unsafe_allow_html=True)

    model_files = list(MODELS_DIR.glob("*.pkl"))
    test_files = list(TEST_DATA_DIR.glob("*.csv"))

    if not model_files:
        st.warning("⚠️ No trained models found.")
        return
    if not test_files:
        st.warning("⚠️ No test datasets found. Please upload a test dataset.")
        return

    model_choice = st.selectbox("Select trained model", model_files, format_func=lambda x: x.name)
    test_choice = st.selectbox("Select test dataset", test_files, format_func=lambda x: x.name)

    if st.button("🔍 Evaluate"):
        df = pd.read_csv(test_choice)
        if 'Depression' not in df.columns:
            st.error("❌ Test data must contain 'Depression' column.")
            return

        trainer = ModelTrainer()
        model_data = joblib.load(model_choice)
        model = model_data['model']
        preproc = model_data['preprocessor']
        model_type = model_data.get('model_type', 'standard')

        X = df.drop(columns=['Depression'])
        y = df['Depression']

        # Preprocess + predict
        X_proc = trainer.preprocessor.transform(X, preproc, model_type=model_type)
        y_pred = model.predict(X_proc)
        proba = model.predict_proba(X_proc)

        st.metric("Accuracy", f"{(y_pred == y).mean():.4f}")
        st.plotly_chart(plot_confusion_matrix(y, y_pred), use_container_width=True)
        report = classification_report(y, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())

        prob_df = pd.DataFrame(proba, columns=model.classes_)
        st.markdown("#### Prediction Probability Samples")
        st.dataframe(prob_df.head())
