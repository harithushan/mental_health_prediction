import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import mlflow
import time
from config import TRAIN_DATA_DIR, MLFLOW_EXPERIMENT_NAME, MODELS_DIR
from models.model_trainer import ModelTrainer
from utils.visualization import plot_training_results, plot_confusion_matrix
import plotly.express as px
import plotly.graph_objects as go

def show_model_training_page():
    """Display the model training page."""
    st.markdown('<h2 class="sub-header">🤖 Model Training</h2>', unsafe_allow_html=True)

    train_files = list(TRAIN_DATA_DIR.glob("*.csv"))
    if not train_files:
        st.warning("⚠️ No training data found. Please upload training data first.")
        return

    st.markdown("### Select Training Data")
    selected_file = st.selectbox("Choose a training file:", train_files, format_func=lambda x: x.name)

    if selected_file:
        try:
            df = pd.read_csv(selected_file)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows", df.shape[0])
            with col2:
                st.metric("Columns", df.shape[1])
            with col3:
                st.metric("Missing Values", df.isnull().sum().sum())

            if 'Depression' not in df.columns:
                st.error("❌ Target column 'Depression' not found in the selected file.")
                return

            st.markdown("### Target Distribution")
            target_dist = df['Depression'].value_counts()
            fig = px.bar(x=target_dist.index, y=target_dist.values,
                         labels={'x': 'Depression', 'y': 'Count'},
                         title='Target Variable Distribution')
            st.plotly_chart(fig, use_container_width=True)

            # Config
            st.markdown("### Training Configuration")
            col1, col2 = st.columns(2)
            with col1:
                test_size = st.slider("Test Size (%)", 10, 40, 20, step=5) / 100
                cv_folds = st.slider("Cross-Validation Folds", 3, 10, 5, step=1)
            with col2:
                hyperparameter_tuning = st.checkbox("Enable Hyperparameter Tuning", value=False)
                models_to_train = st.multiselect(
                    "Select Models to Train",
                    ['GradientBoosting', 'XGBoost', 'LightGBM', 'CatBoost'],
                    default=['GradientBoosting', 'XGBoost', 'LightGBM', 'CatBoost']
                )

            if st.button("🚀 Start Training", type="primary"):
                if not models_to_train:
                    st.error("❌ Please select at least one model to train.")
                    return

                mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
                trainer = ModelTrainer()

                total_models = len(models_to_train)
                start_time = time.time()

                # Progress bars
                st.markdown("### 🔄 Training Progress")
                overall_progress_container = st.container()
                with overall_progress_container:
                    col1, col2, col3 = st.columns([2, 1, 1])
                    with col1:
                        overall_progress = st.progress(0)
                        overall_status = st.empty()
                    with col2:
                        model_counter = st.empty()
                    with col3:
                        time_display = st.empty()

                current_model_container = st.container()
                with current_model_container:
                    st.markdown("#### Current Model Progress")
                    current_model_progress = st.progress(0)
                    current_model_status = st.empty()

                live_results_container = st.container()
                with live_results_container:
                    st.markdown("#### Completed Models")
                    completed_models_placeholder = st.empty()

                results_container = st.container()

                def update_overall_progress(completed, total):
                    progress = completed / total
                    overall_progress.progress(progress)
                    overall_status.markdown(f"**Overall Progress:** {completed}/{total} models completed")
                    model_counter.metric("Models", f"{completed}/{total}")
                    elapsed = time.time() - start_time
                    if completed > 0:
                        avg_time = elapsed / completed
                        remaining = avg_time * (total - completed)
                        time_display.metric("Time", f"{elapsed:.1f}s", f"~{remaining:.1f}s remaining")
                    else:
                        time_display.metric("Time", f"{elapsed:.1f}s")

                def update_current_model_progress(name, status, val=0.0):
                    current_model_progress.progress(val)
                    current_model_status.markdown(f"**{name}:** {status}")

                def display_completed_models(results):
                    if results:
                        with completed_models_placeholder.container():
                            for r in results:
                                col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
                                with col1:
                                    st.success(f"✅ {r['model_name']}")
                                with col2:
                                    st.metric("Accuracy", f"{r['accuracy']:.4f}")
                                with col3:
                                    st.metric("CV Mean", f"{r['cv_mean']:.4f}")
                                with col4:
                                    st.metric("CV Std", f"{r['cv_std']:.4f}")

                with st.spinner("Initializing training..."):
                    try:
                        update_overall_progress(0, total_models)
                        update_current_model_progress("Initializing", "Preparing data...", 0.1)

                        from sklearn.model_selection import train_test_split
                        X = df.drop('Depression', axis=1)
                        y = df['Depression']
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=test_size, random_state=42, stratify=y
                        )

                        completed_results = []
                        all_results = []

                        for i, model_name in enumerate(models_to_train):
                            update_current_model_progress(model_name, "Starting training...", 0.1)

                            try:
                                update_current_model_progress(model_name, "Training...", 0.4)

                                model_result = trainer.train_single_model(
                                    X_train, X_test, y_train, y_test,
                                    model_name,
                                    hyperparameter_tuning=hyperparameter_tuning,
                                    cv_folds=cv_folds
                                )

                                if model_result:
                                    update_current_model_progress(model_name, "Evaluating...", 0.8)
                                    completed_results.append(model_result)
                                    all_results.append(model_result)
                                    update_current_model_progress(model_name, "✅ Completed!", 1.0)
                                else:
                                    st.warning(f"⚠️ {model_name} training failed. Skipping.")
                                    update_current_model_progress(model_name, "❌ Failed", 0.0)

                            except Exception as model_error:
                                st.error(f"❌ Error training {model_name}: {str(model_error)}")
                                update_current_model_progress(model_name, f"Failed: {str(model_error)}", 0)

                            update_overall_progress(i + 1, total_models)
                            display_completed_models(completed_results)
                            time.sleep(0.4)

                        results = all_results

                        if not results:
                            st.error("❌ No models trained successfully. Please check logs.")
                            return

                        # Final results
                        overall_status.markdown("**✅ All models training completed!**")
                        current_model_status.markdown("**🎉 Training session finished successfully!**")

                        with results_container:
                            st.markdown("---")
                            st.success("✅ Training completed successfully!")

                            total_time = time.time() - start_time
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Models Trained", len(results))
                            with col2:
                                st.metric("Total Time", f"{total_time:.1f}s")
                            with col3:
                                avg_time = total_time / len(results)
                                st.metric("Avg Time/Model", f"{avg_time:.1f}s")
                            with col4:
                                best_accuracy = max(r['accuracy'] for r in results)
                                st.metric("Best Accuracy", f"{best_accuracy:.4f}")

                            st.markdown("### 📊 Detailed Training Results")
                            results_df = pd.DataFrame(results)
                            st.dataframe(results_df.style.highlight_max(axis=0))

                            best_model_name, best_model_info = trainer.get_best_model()
                            st.markdown(f"### 🏆 Best Model: **{best_model_name}**")
                            st.metric("Best Accuracy", f"{best_model_info['accuracy']:.4f}")

                            st.markdown("### 📈 Model Comparison")
                            fig_acc = px.bar(results_df, x='model_name', y='accuracy',
                                             title='Model Accuracy Comparison',
                                             color='accuracy',
                                             color_continuous_scale='viridis')
                            st.plotly_chart(fig_acc, use_container_width=True)

                            fig_cv = go.Figure()
                            for _, row in results_df.iterrows():
                                fig_cv.add_trace(go.Bar(
                                    x=[row['model_name']],
                                    y=[row['cv_mean']],
                                    error_y=dict(type='data', array=[row['cv_std']]),
                                    name=row['model_name']
                                ))
                            fig_cv.update_layout(title='Cross-Validation Scores',
                                                 xaxis_title='Model',
                                                 yaxis_title='CV Score',
                                                 showlegend=False)
                            st.plotly_chart(fig_cv, use_container_width=True)

                            st.markdown("### 🎯 Confusion Matrix - Best Model")
                            if best_model_name in trainer.results:
                                y_pred = trainer.results[best_model_name]['predictions']
                                fig_cm = plot_confusion_matrix(y_test, y_pred)
                                st.plotly_chart(fig_cm, use_container_width=True)

                            st.markdown("### 📋 Model Details")
                            for result in results:
                                with st.expander(f"{result['model_name']} Details"):
                                    st.write(f"**Accuracy:** {result['accuracy']:.4f}")
                                    st.write(f"**CV Mean:** {result['cv_mean']:.4f}")
                                    st.write(f"**CV Std:** {result['cv_std']:.4f}")
                                    st.write(f"**Model Path:** {result['model_path']}")

                    except Exception as e:
                        st.error(f"❌ Error during training: {str(e)}")
                        overall_progress.progress(0)
                        current_model_progress.progress(0)
                        overall_status.markdown("**❌ Training failed!**")
                        current_model_status.markdown(f"**Error:** {str(e)}")

        except Exception as e:
            st.error(f"❌ Error loading training data: {str(e)}")

    st.markdown("---")
    st.markdown('<h3 class="sub-header">💾 Saved Models</h3>', unsafe_allow_html=True)
    model_files = list(MODELS_DIR.glob("*.pkl"))

    if model_files:
        st.markdown("### Available Models")
        for model_file in model_files:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"📄 {model_file.name}")
            with col2:
                if st.button(f"Delete", key=f"del_model_{model_file.name}"):
                    model_file.unlink()
                    st.success(f"Deleted {model_file.name}")
                    st.rerun()
    else:
        st.write("No saved models found. Train some models first!")

    st.markdown("---")
    st.markdown('<h3 class="sub-header">📊 Experiment Tracking</h3>', unsafe_allow_html=True)
    st.info("All training runs are automatically tracked using MLflow.")

    try:
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
        if experiment:
            runs = client.search_runs(experiment_ids=[experiment.experiment_id])
            st.metric("Total Experiment Runs", len(runs))
        else:
            st.metric("Total Experiment Runs", 0)
    except:
        st.metric("Total Experiment Runs", "N/A")