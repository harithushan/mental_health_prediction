import mlflow
import mlflow.tracking
from mlflow.tracking import MlflowClient
from config import MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME
import streamlit as st
import pandas as pd
import os
import logging
from datetime import datetime


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@st.cache_resource
def setup_mlflow():
    """Initialize MLflow tracking with proper error handling."""
    try:
        # Set tracking URI
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        logger.info(f"MLflow tracking URI set to: {MLFLOW_TRACKING_URI}")
        
        # Create client
        client = MlflowClient()
        
        # Check if experiment exists, create if not
        experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
        if experiment is None:
            experiment_id = mlflow.create_experiment(MLFLOW_EXPERIMENT_NAME)
            logger.info(f"Created experiment: {MLFLOW_EXPERIMENT_NAME} with ID: {experiment_id}")
        else:
            logger.info(f"Using existing experiment: {MLFLOW_EXPERIMENT_NAME}")
            
        return True
        
    except Exception as e:
        logger.error(f"Error setting up MLflow: {str(e)}")
        st.error(f"Failed to setup MLflow: {str(e)}")
        return False

def set_experiment(experiment_name: str = None):
    """Set the active MLflow experiment."""
    try:
        if experiment_name is None:
            experiment_name = MLFLOW_EXPERIMENT_NAME
        
        mlflow.set_experiment(experiment_name)
        logger.info(f"Active experiment set to: {experiment_name}")
        return True
        
    except Exception as e:
        logger.error(f"Error setting experiment: {str(e)}")
        return False

def log_params(params: dict):
    """Log parameters to MLflow."""
    try:
        for k, v in params.items():
            mlflow.log_param(k, v)
        logger.info(f"Logged {len(params)} parameters")
        
    except Exception as e:
        logger.error(f"Error logging parameters: {str(e)}")

def log_metrics(metrics: dict):
    """Log metrics to MLflow."""
    try:
        for k, v in metrics.items():
            if v is not None:  # Only log non-None values
                mlflow.log_metric(k, v)
        logger.info(f"Logged {len(metrics)} metrics")
        
    except Exception as e:
        logger.error(f"Error logging metrics: {str(e)}")

def start_run(run_name: str = None):
    """Start an MLflow run."""
    try:
        return mlflow.start_run(run_name=run_name)
    except Exception as e:
        logger.error(f"Error starting run: {str(e)}")
        return None

def end_run():
    """End the current MLflow run."""
    try:
        mlflow.end_run()
    except Exception as e:
        logger.error(f"Error ending run: {str(e)}")

def get_experiment_runs(experiment_name: str = None):
    """Get all runs for an experiment."""
    try:
        if experiment_name is None:
            experiment_name = MLFLOW_EXPERIMENT_NAME
            
        client = MlflowClient()
        experiment = client.get_experiment_by_name(experiment_name)
        
        if experiment is None:
            logger.warning(f"Experiment {experiment_name} not found")
            return []
        
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            max_results=1000,
            order_by=["start_time DESC"]
        )
        
        return runs
        
    except Exception as e:
        logger.error(f"Error getting experiment runs: {str(e)}")
        return []

def get_run_details(run_id: str):
    """Get detailed information about a specific run."""
    try:
        client = MlflowClient()
        run = client.get_run(run_id)
        return run
        
    except Exception as e:
        logger.error(f"Error getting run details: {str(e)}")
        return None

def delete_experiment_run(run_id: str):
    """Delete a specific run."""
    try:
        client = MlflowClient()
        client.delete_run(run_id)
        logger.info(f"Deleted run: {run_id}")
        return True
        
    except Exception as e:
        logger.error(f"Error deleting run: {str(e)}")
        return False

def get_mlflow_status():
    """Check MLflow connection status."""
    try:
        client = MlflowClient()
        # Try to list experiments to test connection
        experiments = client.search_experiments()
        return {
            'status': 'connected',
            'tracking_uri': mlflow.get_tracking_uri(),
            'experiments_count': len(experiments)
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'tracking_uri': mlflow.get_tracking_uri()
        }


def show_mlflow_debug_page():
    """Debug page for MLflow issues."""
    
    st.markdown("# 🐛 MLflow Debug Page")
    
    # Check MLflow status
    st.markdown("## Connection Status")
    status = get_mlflow_status()
    
    if status['status'] == 'connected':
        st.success("✅ MLflow connection successful!")
        st.write(f"**Tracking URI:** {'http://localhost:5000/'}") #status['tracking_uri']
        st.write(f"**Experiments found:** {status['experiments_count']}")
        st.write(f"**Current Experiment:** {MLFLOW_EXPERIMENT_NAME}")
        
    else:
        st.error("❌ MLflow connection failed!")
        st.write(f"**Error:** {status['error']}")
        st.write(f"**Tracking URI:** {'http://localhost:5000/'}") #status['tracking_uri']
    
    # List runs for the main experiment
    st.markdown("## All Experiment Runs")
    try:
        client = MlflowClient(tracking_uri= status['tracking_uri'])
        experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
        
        if experiment is None:
            st.warning(f"⚠️ Experiment '{MLFLOW_EXPERIMENT_NAME}' not found.")
        else:
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                max_results=1000,
                order_by=["start_time DESC"]
            )

            if runs:
                st.success(f"Found {len(runs)} runs in experiment '{MLFLOW_EXPERIMENT_NAME}'")

                runs_info = []
                for run in runs[:50]:  # Display more if needed
                    run_info = {
                        'run_id': run.info.run_id,
                        'run_name': run.data.tags.get('mlflow.runName', 'Unnamed'),
                        'start_time': datetime.fromtimestamp(run.info.start_time / 1000).strftime('%Y-%m-%d %H:%M:%S'),
                        'status': run.info.status,
                        **run.data.params,
                        **{f"metric_{k}": v for k, v in run.data.metrics.items()},
                        **{f"tag_{k}": v for k, v in run.data.tags.items() if not k.startswith("mlflow.")}
                    }
                    runs_info.append(run_info)

                df_runs = pd.DataFrame(runs_info)

                st.markdown("### 📌 Select Columns to Display")
                all_columns = df_runs.columns.tolist()
                default_cols = ["run_id", "run_name", "start_time", "status"] + [col for col in all_columns if "accuracy" in col]
                select_all = st.checkbox("Select All Columns", value=True)

                if select_all:
                    selected_cols = all_columns
                else:
                    selected_cols = st.multiselect("Choose Columns to Display", options=all_columns, default=default_cols)

                if not selected_cols:
                    st.warning("⚠️ No columns selected.")
                else:
                    df_display = df_runs[selected_cols].copy()
                    if "run_id" in df_display.columns:
                        df_display["run_id"] = df_display["run_id"].str[:8] + "..."
                    st.dataframe(df_display, use_container_width=True)
            else:
                st.info(f"No runs found in experiment '{MLFLOW_EXPERIMENT_NAME}'")
    except Exception as e:
        st.error(f"❌ Error listing runs: {str(e)}")

    # # List runs for the main experiment
    # st.markdown("## Runs in Main Experiment")
    # try:
    #     runs = get_experiment_runs(MLFLOW_EXPERIMENT_NAME)
    #     if runs:
    #         st.success(f"Found {len(runs)} runs in experiment '{MLFLOW_EXPERIMENT_NAME}'")
            
    #         # Show basic run info
    #         runs_info = []
    #         for run in runs[:10]:  # Show first 10 runs
    #             runs_info.append({
    #                 'Run ID': run.info.run_id[:8] + "...",
    #                 'Status': run.info.status,
    #                 'Start Time': run.info.start_time,
    #                 'Model Type': run.data.params.get('model_type', 'Unknown'),
    #                 'Accuracy': run.data.metrics.get('accuracy', 'N/A')
    #             })
            
    #         st.dataframe(runs_info)
            
    #     else:
    #         st.info(f"No runs found in experiment '{MLFLOW_EXPERIMENT_NAME}'")
            
    # except Exception as e:
    #     st.error(f"Error listing runs: {str(e)}")
    
    # Test basic operations
    st.markdown("## Test Operations")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Test Create Experiment"):
            try:
                client = MlflowClient()
                test_exp_name = "test_experiment_debug"
                
                # Check if exists
                existing = client.get_experiment_by_name(test_exp_name)
                if existing is None:
                    exp_id = mlflow.create_experiment(test_exp_name)
                    st.success(f"✅ Created test experiment: {exp_id}")
                else:
                    st.info(f"ℹ️ Test experiment already exists: {existing.experiment_id}")
                    
            except Exception as e:
                st.error(f"❌ Error creating experiment: {str(e)}")
    
    with col2:
        if st.button("Test Start Run"):
            try:
                mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
                with mlflow.start_run(run_name="debug_test_run"):
                    mlflow.log_param("test_param", "debug_value")
                    mlflow.log_metric("test_metric", 0.95)
                    st.success("✅ Test run completed successfully!")
                    
            except Exception as e:
                st.error(f"❌ Error with test run: {str(e)}")
    
    # List experiments
    st.markdown("## Available Experiments")
    try:
        client = MlflowClient()
        experiments = client.search_experiments()
        
        if experiments:
            exp_data = []
            for exp in experiments:
                exp_data.append({
                    'Name': exp.name,
                    'ID': exp.experiment_id,
                    'Lifecycle Stage': exp.lifecycle_stage,
                    'Creation Time': exp.creation_time
                })
            
            st.dataframe(exp_data)
        else:
            st.info("No experiments found.")
            
    except Exception as e:
        st.error(f"Error listing experiments: {str(e)}")
    
    # Raw MLflow info
    st.markdown("## Raw MLflow Information")
    with st.expander("Show Raw Data"):
        try:
            client = MlflowClient()
            experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
            
            if experiment:
                st.write("**Experiment Object:**")
                st.json({
                    'experiment_id': experiment.experiment_id,
                    'name': experiment.name,
                    'artifact_location': experiment.artifact_location,
                    'lifecycle_stage': experiment.lifecycle_stage,
                    'creation_time': experiment.creation_time,
                    'last_update_time': experiment.last_update_time
                })
                
                # Get first run details
                runs = client.search_runs(experiment_ids=[experiment.experiment_id], max_results=1)
                if runs:
                    run = runs[0]
                    st.write("**Sample Run Object:**")
                    st.json({
                        'run_id': run.info.run_id,
                        'status': run.info.status,
                        'start_time': run.info.start_time,
                        'end_time': run.info.end_time,
                        'params': dict(run.data.params),
                        'metrics': dict(run.data.metrics),
                        'tags': dict(run.data.tags)
                    })
            else:
                st.write("No experiment found to show details.")
                
        except Exception as e:
            st.error(f"Error getting raw data: {str(e)}")