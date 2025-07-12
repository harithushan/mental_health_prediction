import streamlit as st
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
from config import MLFLOW_EXPERIMENT_NAME
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from utils.mlflow_utils import get_mlflow_status

def show_experiment_tracking_page():
    """Display the experiment tracking page with MLflow runs."""
    
    st.markdown('<h2 class="sub-header">📊 Experiment Tracking</h2>', unsafe_allow_html=True)
    
    try:
        client = MlflowClient()
        
        # Get or create experiment
        experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
        
        if experiment is None:
            st.warning(f"⚠️ Experiment '{MLFLOW_EXPERIMENT_NAME}' not found. Train some models first!")
            return
        
        # Search for runs
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            max_results=100,
            order_by=["start_time DESC"]
        )
        
        if not runs:
            st.info("📝 No experiment runs found. Train some models to see results here!")
            return
        
        st.success(f"✅ Found {len(runs)} experiment runs")
        
        # Convert runs to DataFrame for easier manipulation
        runs_data = []
        for run in runs:
            run_data = {
                'run_id': run.info.run_id,
                'run_name': run.data.tags.get('mlflow.runName', 'Unnamed'),
                'start_time': datetime.fromtimestamp(run.info.start_time / 1000).strftime('%Y-%m-%d %H:%M:%S'),
                'status': run.info.status,
                'model_type': run.data.params.get('model_type', 'Unknown'),
                'accuracy': run.data.metrics.get('accuracy', None),
                'cv_mean': run.data.metrics.get('cv_mean', None),
                'cv_std': run.data.metrics.get('cv_std', None),
                'hyperparameter_tuning': run.data.params.get('hyperparameter_tuning', 'False'),
                'training_samples': run.data.params.get('training_samples', None),
                'test_samples': run.data.params.get('test_samples', None)
            }
            runs_data.append(run_data)
        
        df_runs = pd.DataFrame(runs_data)
        
        # Display summary statistics
        st.markdown("### 📈 Summary Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Runs", len(df_runs))
        with col2:
            successful_runs = len(df_runs[df_runs['status'] == 'FINISHED'])
            st.metric("Successful Runs", successful_runs)
        with col3:
            if df_runs['accuracy'].notna().any():
                best_accuracy = df_runs['accuracy'].max()
                st.metric("Best Accuracy", f"{best_accuracy:.4f}")
            else:
                st.metric("Best Accuracy", "N/A")
        with col4:
            unique_models = df_runs['model_type'].nunique()
            st.metric("Model Types", unique_models)
        
        # Filter options
        st.markdown("### 🔍 Filter Runs")
        col1, col2 = st.columns(2)
        
        with col1:
            selected_models = st.multiselect(
                "Select Model Types",
                options=df_runs['model_type'].unique(),
                default=df_runs['model_type'].unique()
            )
        
        with col2:
            selected_status = st.multiselect(
                "Select Status",
                options=df_runs['status'].unique(),
                default=df_runs['status'].unique()
            )
        
        # Filter dataframe
        filtered_df = df_runs[
            (df_runs['model_type'].isin(selected_models)) & 
            (df_runs['status'].isin(selected_status))
        ]
        
        if filtered_df.empty:
            st.warning("⚠️ No runs match the selected filters.")
            return
        
        # Display runs table
        st.markdown("### 📊 Experiment Runs")
        
        # Format the dataframe for display
        display_df = filtered_df.copy()
        display_df['run_id'] = display_df['run_id'].str[:8] + "..."  # Truncate run ID
        
        # Sort by accuracy if available
        if 'accuracy' in display_df.columns and display_df['accuracy'].notna().any():
            display_df = display_df.sort_values('accuracy', ascending=False)
        
        st.dataframe(
            display_df[['run_name', 'model_type', 'start_time', 'status', 'accuracy', 'cv_mean', 'cv_std']],
            use_container_width=True
        )
        
        # Visualizations
        st.markdown("### 📈 Performance Visualizations")
        
        # Filter for runs with accuracy data
        viz_df = filtered_df[filtered_df['accuracy'].notna()]
        
        if not viz_df.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                # Accuracy comparison by model type
                fig_acc = px.box(
                    viz_df, 
                    x='model_type', 
                    y='accuracy',
                    title='Accuracy Distribution by Model Type',
                    points='all'
                )
                fig_acc.update_layout(xaxis_title='Model Type', yaxis_title='Accuracy')
                st.plotly_chart(fig_acc, use_container_width=True)
            
            with col2:
                # Accuracy over time
                viz_df['start_time_dt'] = pd.to_datetime(viz_df['start_time'])
                fig_time = px.scatter(
                    viz_df, 
                    x='start_time_dt', 
                    y='accuracy',
                    color='model_type',
                    title='Accuracy Over Time',
                    hover_data=['run_name']
                )
                fig_time.update_layout(xaxis_title='Time', yaxis_title='Accuracy')
                st.plotly_chart(fig_time, use_container_width=True)
            
            # Cross-validation comparison
            cv_df = viz_df[viz_df['cv_mean'].notna()]
            if not cv_df.empty:
                st.markdown("### 📊 Cross-Validation Performance")
                fig_cv = go.Figure()
                
                for model_type in cv_df['model_type'].unique():
                    model_data = cv_df[cv_df['model_type'] == model_type]
                    fig_cv.add_trace(go.Bar(
                        x=model_data['run_name'],
                        y=model_data['cv_mean'],
                        error_y=dict(type='data', array=model_data['cv_std']),
                        name=model_type
                    ))
                
                fig_cv.update_layout(
                    title='Cross-Validation Scores by Run',
                    xaxis_title='Run Name',
                    yaxis_title='CV Score',
                    barmode='group'
                )
                st.plotly_chart(fig_cv, use_container_width=True)
        
        else:
            st.info("📝 No accuracy data available for visualization. Train some models to see performance charts!")
        
        # Detailed run view
        st.markdown("### 🔍 Detailed Run View")
        
        if not filtered_df.empty:
            selected_run_idx = st.selectbox(
                "Select a run to view details:",
                range(len(filtered_df)),
                format_func=lambda x: f"{filtered_df.iloc[x]['run_name']} - {filtered_df.iloc[x]['model_type']}"
            )
            
            selected_run_id = filtered_df.iloc[selected_run_idx]['run_id']
            
            # Get detailed run information
            run = client.get_run(selected_run_id)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Parameters:**")
                for key, value in run.data.params.items():
                    st.write(f"• {key}: {value}")
            
            with col2:
                st.markdown("**Metrics:**")
                for key, value in run.data.metrics.items():
                    st.write(f"• {key}: {value:.4f}")
            
            # Tags
            if run.data.tags:
                st.markdown("**Tags:**")
                for key, value in run.data.tags.items():
                    if not key.startswith('mlflow.'):  # Skip internal MLflow tags
                        st.write(f"• {key}: {value}")
        
        # Comparison tool
        st.markdown("### 🔄 Compare Runs")
        
        if len(filtered_df) >= 2:
            compare_runs = st.multiselect(
                "Select runs to compare (max 5):",
                options=filtered_df.index,
                format_func=lambda x: f"{filtered_df.iloc[x]['run_name']} - {filtered_df.iloc[x]['model_type']}",
                max_selections=5
            )
            
            if len(compare_runs) >= 2:
                comparison_df = filtered_df.iloc[compare_runs]
                
                # Show comparison table
                st.markdown("**Comparison Table:**")
                comparison_cols = ['run_name', 'model_type', 'accuracy', 'cv_mean', 'cv_std', 'hyperparameter_tuning']
                st.dataframe(comparison_df[comparison_cols], use_container_width=True)
                
                # Comparison chart
                if comparison_df['accuracy'].notna().any():
                    fig_compare = px.bar(
                        comparison_df, 
                        x='run_name', 
                        y='accuracy',
                        color='model_type',
                        title='Accuracy Comparison'
                    )
                    st.plotly_chart(fig_compare, use_container_width=True)
        
        else:
            st.info("📝 You need at least 2 runs to use the comparison tool.")
    
    except Exception as e:
        st.error(f"❌ Error accessing MLflow tracking: {str(e)}")
        st.info("💡 Make sure MLflow server is running and accessible.")
        
        # Debug information
        with st.expander("Debug Information"):
            st.write(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")
            st.write(f"Experiment Name: {MLFLOW_EXPERIMENT_NAME}")
            st.write(f"Error Details: {str(e)}")