import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from config import MODELS_DIR, PREDICT_DATA_DIR
from models.model_trainer import ModelTrainer
from models.data_preprocessing import DataPreprocessor
import plotly.express as px
import plotly.graph_objects as go
import plotly.express as px

def show_predictions_page():
    """Display the predictions page."""

    st.markdown('<h2 class="sub-header">🔮 Predictions</h2>', unsafe_allow_html=True)

    model_files = list(MODELS_DIR.glob("*.pkl"))

    if not model_files:
        st.warning("⚠️ No trained models found. Please train some models first.")
        return

    tab1, tab2 = st.tabs(["🔍 Single Prediction", "📊 Bulk Predictions"])

    # -----------------------------------
    # 🔍 Single Prediction Tab
    # -----------------------------------
    with tab1:
        st.markdown("### Single Prediction")
        selected_model = st.selectbox(
            "Select Model:",
            model_files,
            format_func=lambda x: x.name,
            key="single_model"
        )

        if selected_model:
            st.markdown("#### Enter Patient Information")
            col1, col2 = st.columns(2)
            with col1:
                age = st.number_input("Age", min_value=10, max_value=100, value=25)
                gender = st.selectbox("Gender", ["Male", "Female"])
                working_status = st.selectbox("Working Professional or Student", ["Working Professional", "Student"])
                profession = st.selectbox("Profession", ['Chef', 'Teacher', 'Business Analyst', 'Financial Analyst', 'Software Engineer', 'Data Scientist', 'Marketing Manager', 'Student', 'Doctor', 'Others'])
                degree = st.selectbox("Degree", ['B.Tech', 'MBA', 'BSc', 'MSc', 'PhD', 'Class 12', 'Others'])
            with col2:
                cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, value=7.0, step=0.1)
                work_study_hours = st.number_input("Work/Study Hours", min_value=0, max_value=24, value=8)
                sleep_duration = st.selectbox("Sleep Duration", ['Less than 5 hours', '5-6 hours', '6-7 hours', '7-8 hours', '8-9 hours', 'More than 8 hours'])
                dietary_habits = st.selectbox("Dietary Habits", ["Healthy", "Moderate", "Unhealthy"])
                suicidal_thoughts = st.selectbox("Suicidal Thoughts", ["No", "Yes"])

            st.markdown("#### Stress & Satisfaction Levels (0-10)")
            col1, col2 = st.columns(2)
            with col1:
                academic_pressure = st.slider("Academic Pressure", 0, 10, 5)
                work_pressure = st.slider("Work Pressure", 0, 10, 5)
                financial_stress = st.slider("Financial Stress", 0, 10, 5)
            with col2:
                study_satisfaction = st.slider("Study Satisfaction", 0, 10, 7)
                job_satisfaction = st.slider("Job Satisfaction", 0, 10, 7)
                family_history = st.selectbox("Family History of Mental Illness", ["No", "Yes"])

            if st.button("🔮 Make Prediction", type="primary"):
                input_data = pd.DataFrame({
                    'Age': [age],
                    'Gender': [gender],
                    'Working Professional or Student': [working_status],
                    'Profession': [profession],
                    'Degree': [degree],
                    'CGPA': [cgpa],
                    'Work/Study Hours': [work_study_hours],
                    'Sleep Duration': [sleep_duration],
                    'Dietary Habits': [dietary_habits],
                    'Have you ever had suicidal thoughts ?': [suicidal_thoughts],
                    'Academic Pressure': [academic_pressure],
                    'Work Pressure': [work_pressure],
                    'Financial Stress': [financial_stress],
                    'Study Satisfaction': [study_satisfaction],
                    'Job Satisfaction': [job_satisfaction],
                    'Family History of Mental Illness': [family_history]
                })

                try:
                    trainer = ModelTrainer()
                    predictions, prediction_proba = trainer.predict_with_model(selected_model, input_data)

                    if predictions is not None:
                        pred_text = "🚨 Depression Risk: HIGH" if predictions[0] == 1 else "✅ Depression Risk: LOW"
                        confidence = max(prediction_proba[0]) * 100

                        col1, col2 = st.columns(2)
                        with col1:
                            st.error(pred_text) if predictions[0] == 1 else st.success(pred_text)
                        with col2:
                            st.metric("Confidence", f"{confidence:.1f}%")

                        st.markdown("#### Probability Distribution")
                        prob_df = pd.DataFrame({
                            'Class': ['No Depression', 'Depression'],
                            'Probability': prediction_proba[0]
                        })
                        fig = px.bar(prob_df, x='Class', y='Probability', color='Class',
                                     color_discrete_map={'No Depression': 'green', 'Depression': 'red'},
                                     title="Prediction Probability")
                        st.plotly_chart(fig, use_container_width=True)

                        # Recommendations
                        st.markdown("#### Recommendations")
                        if predictions[0] == 1:
                            st.markdown("""
                            **⚠️ High Risk – Take Action:**
                            - 🏥 Talk to a professional
                            - 🧘 Practice mindfulness
                            - 💬 Seek support from friends/family
                            """)
                        else:
                            st.markdown("""
                            **✅ Low Risk – Maintain Mental Health:**
                            - 🧘 Stay active and balanced
                            - 🤝 Stay connected with loved ones
                            """)
                    else:
                        st.error("❌ Failed to generate prediction.")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

    # -----------------------------------
    # 📊 Bulk Predictions Tab
    # -----------------------------------
    with tab2:
        st.markdown("### Bulk Predictions")
        predict_files = list(PREDICT_DATA_DIR.glob("*.csv"))

        if not predict_files:
            st.warning("⚠️ No prediction data found.")
            return

        selected_model = st.selectbox("Select Model:", model_files, format_func=lambda x: x.name, key="bulk_model")
        selected_data = st.selectbox("Select Prediction Data:", predict_files, format_func=lambda x: x.name)

        if selected_model and selected_data:
            try:
                df = pd.read_csv(selected_data)
                st.markdown("#### Data Preview")
                st.dataframe(df.head())

                if st.button("🔮 Run Bulk Predictions", type="primary"):
                    with st.spinner("Running predictions..."):
                        trainer = ModelTrainer()
                        predictions, prediction_proba = trainer.predict_with_model(selected_model, df)

                        if predictions is not None:
                            df["Predicted_Depression"] = predictions
                            df["Confidence"] = [max(prob) for prob in prediction_proba]

                            st.success("✅ Predictions completed!")
                            st.dataframe(df.head())

                            dist = df["Predicted_Depression"].value_counts().rename(index={0: "No Depression", 1: "Depression"})
                            fig = px.bar(
                                x=dist.index,
                                y=dist.values,
                                labels={"x": "Prediction", "y": "Count"},
                                color=dist.index,
                                color_discrete_map={"No Depression": "green", "Depression": "red"},
                                title="Depression Prediction Distribution"
                            )
                            st.plotly_chart(fig, use_container_width=True)

                            csv = df.to_csv(index=False).encode("utf-8")
                            st.download_button("📥 Download Predictions as CSV", data=csv, file_name="bulk_predictions.csv", mime="text/csv")
                        else:
                            st.error("❌ Prediction failed.")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")