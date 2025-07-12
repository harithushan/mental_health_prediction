import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import os
from streamlit_option_menu import option_menu
import mlflow
from config import MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME, BASE_DIR
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import os
from streamlit_option_menu import option_menu
from utils.mlflow_utils import setup_mlflow



# -------------------- Page Configuration --------------------

# -------------------- Initialize MLflow --------------------

setup_mlflow()

# -------------------- Custom CSS --------------------
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    font-weight: bold;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.sub-header {
    font-size: 1.5rem;
    color: #ff7f0e;
    margin-bottom: 1rem;
}
.metric-container {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
}
.success-message {
    background-color: #d4edda;
    color: #155724;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 1rem 0;
}
.error-message {
    background-color: #f8d7da;
    color: #721c24;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# -------------------- Main Title --------------------
st.markdown('<h1 class="main-header">🧠 Mental Health Prediction Dashboard</h1>', unsafe_allow_html=True)

# -------------------- Sidebar Navigation --------------------
from PIL import Image
from pathlib import Path

# Load local image (relative path)
image_path = Path("assets/logo.png")
image = Image.open(image_path)

# Show in sidebar or anywhere

with st.sidebar:
    # st.image("https://via.placeholder.com/200x100/1f77b4/ffffff?text=Mental+Health+AI", width=200)
    st.image(image, width=200)
    

    selected = option_menu(
        menu_title="Navigation",
        options=["Home", "Data Upload", "Model Training", "Model Evaluation", "Predictions", "Experiment Tracking", "MLflow Debug"],
        icons=["house", "cloud-upload", "gear", "bar-chart", "cpu", "graph-up", "bug"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "#fafafa"},
            "icon": {"color": "#1f77b4", "font-size": "18px"},
            "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px", "--hover-color": "#eee"},
            "nav-link-selected": {"background-color": "#1f77b4"},
        }
    )

# -------------------- Cached File Count --------------------
@st.cache_data
def count_files(directory, extension="*.csv"):
    return len(list(Path(directory).glob(extension))) if Path(directory).exists() else 0

# -------------------- Home Page --------------------
if selected == "Home":
    st.set_page_config(
    page_title="Mental Health Prediction Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
    )
    st.markdown('<h2 class="sub-header">Welcome to the Mental Health Prediction Dashboard</h2>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        ### 📊 Data Management
        - Upload training and test datasets
        - Validate data quality
        - Manage bulk prediction data
        """)

    with col2:
        st.markdown("""
        ### 🤖 Model Training
        - Train multiple ML models
        - Hyperparameter optimization
        - Cross-validation
        """)

    with col3:
        st.markdown("""
        ### 📈 Predictions & Tracking
        - Make predictions on new data
        - Track experiments with MLflow
        - Model versioning and comparison
        """)

    # -------------------- Project Stats --------------------
    st.markdown('<h3 class="sub-header">Project Overview</h3>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Training Datasets", count_files("data/train", extension="*.csv"))
    with col2:
        st.metric("Test Datasets", count_files("data/test", extension="*.csv"))
    with col3:
        st.metric("Prediction Datasets", count_files("data/predict", extension="*.csv"))
    with col4:
        st.metric("Saved Models", count_files("saved_models", extension="*.pkl"))


    # -------------------- Instructions --------------------
    st.markdown('<h3 class="sub-header">Getting Started</h3>', unsafe_allow_html=True)
    st.markdown("""
    1. **Upload Data**: Navigate to the Data Upload page to upload your training and test datasets
    2. **Train Models**: Use the Model Training page to train and optimize your models
    3. **Evaluate**: Check model performance on the Model Evaluation page
    4. **Predict**: Make predictions on new data using the Predictions page
    5. **Track**: Monitor experiments and compare models using the Experiment Tracking page
    """)

# -------------------- Routed Pages --------------------
elif selected == "Data Upload":
    st.set_page_config(
    page_title= None,
    page_icon= None,
    layout="wide",
    initial_sidebar_state="expanded"
    )
    from custom_pages.data_upload import show_data_upload_page
    show_data_upload_page()

elif selected == "Model Training":
    st.set_page_config(
    page_title= None,
    page_icon= None,
    layout="wide",
    initial_sidebar_state="expanded"
    )
    from custom_pages.model_training import show_model_training_page
    show_model_training_page()

elif selected == "Model Evaluation":
    st.set_page_config(
    page_title= None,
    page_icon= None,
    layout="wide",
    initial_sidebar_state="expanded"
    )
    from custom_pages.model_evaluation import show_model_evaluation_page
    show_model_evaluation_page()

elif selected == "Predictions":    
    st.set_page_config(
    page_title= None,
    page_icon= None,
    layout="wide",
    initial_sidebar_state="expanded"
    )
    from custom_pages.predictions import show_predictions_page
    show_predictions_page()

elif selected == "Experiment Tracking":
    st.set_page_config(
    page_title= None,
    page_icon= None,
    layout="wide",
    initial_sidebar_state="expanded"
    )
    from custom_pages.experiment_tracking import show_experiment_tracking_page
    show_experiment_tracking_page()
    
elif selected == "MLflow Debug":
    st.set_page_config(
    page_title= None,
    page_icon= None,
    layout="wide",
    initial_sidebar_state="expanded"
    )
    from utils.mlflow_utils import show_mlflow_debug_page
    show_mlflow_debug_page()


# -------------------- Footer --------------------
st.markdown("---")
st.markdown(
    '<div style="text-align: center; color: #666; margin-top: 2rem;">Mental Health Prediction Dashboard | Built with Streamlit & MLflow</div>',
    unsafe_allow_html=True
)