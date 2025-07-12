import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import os
from config import TRAIN_DATA_DIR, TEST_DATA_DIR, PREDICT_DATA_DIR
from utils.file_utils import save_uploaded_file, validate_csv_format

def show_data_upload_page():
    """Display the data upload page."""
    
    st.markdown('<h2 class="sub-header">📊 Data Upload</h2>', unsafe_allow_html=True)
    
    # Create tabs for different data types
    tab1, tab2, tab3 = st.tabs(["📚 Training Data", "🧪 Test Data", "🔮 Prediction Data"])
    
    with tab1:
        st.markdown("### Upload Training Data")
        st.markdown("Upload your training datasets here. These will be used to train your models.")
        
        uploaded_file = st.file_uploader(
            "Choose a CSV file for training",
            type=['csv'],
            key="train_uploader",
            help="Upload a CSV file containing your training data with the target column 'Depression'"
        )
        
        if uploaded_file is not None:
            try:
                # Read and validate the file
                df = pd.read_csv(uploaded_file)
                
                # Display basic info
                st.success(f"✅ File uploaded successfully! Shape: {df.shape}")
                
                # Show data preview
                st.markdown("#### Data Preview")
                st.dataframe(df.head())
                
                # Show data info
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Dataset Info")
                    st.write(f"**Rows:** {df.shape[0]}")
                    st.write(f"**Columns:** {df.shape[1]}")
                    st.write(f"**Memory Usage:** {df.memory_usage().sum() / 1024:.2f} KB")
                
                with col2:
                    st.markdown("#### Missing Values")
                    missing_values = df.isnull().sum()
                    if missing_values.sum() > 0:
                        st.dataframe(missing_values[missing_values > 0])
                    else:
                        st.write("No missing values found!")
                
                # Check for target column
                if 'Depression' in df.columns:
                    st.success("✅ Target column 'Depression' found!")
                    target_dist = df['Depression'].value_counts()
                    st.markdown("#### Target Distribution")
                    st.bar_chart(target_dist)
                else:
                    st.warning("⚠️ Target column 'Depression' not found. Please ensure your training data contains the target column.")
                
                # Save file option
                if st.button("Save Training Data", key="save_train"):
                    if save_uploaded_file(uploaded_file, TRAIN_DATA_DIR):
                        st.success("✅ Training data saved successfully!")
                        st.rerun()
                    else:
                        st.error("❌ Failed to save training data.")
                        
            except Exception as e:
                st.error(f"❌ Error reading file: {str(e)}")
    
    with tab2:
        st.markdown("### Upload Test Data")
        st.markdown("Upload your test datasets here. These will be used to evaluate your models.")
        
        uploaded_file = st.file_uploader(
            "Choose a CSV file for testing",
            type=['csv'],
            key="test_uploader",
            help="Upload a CSV file containing your test data"
        )
        
        if uploaded_file is not None:
            try:
                # Read and validate the file
                df = pd.read_csv(uploaded_file)
                
                # Display basic info
                st.success(f"✅ File uploaded successfully! Shape: {df.shape}")
                
                # Show data preview
                st.markdown("#### Data Preview")
                st.dataframe(df.head())
                
                # Show data info
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Dataset Info")
                    st.write(f"**Rows:** {df.shape[0]}")
                    st.write(f"**Columns:** {df.shape[1]}")
                    st.write(f"**Memory Usage:** {df.memory_usage().sum() / 1024:.2f} KB")
                
                with col2:
                    st.markdown("#### Missing Values")
                    missing_values = df.isnull().sum()
                    if missing_values.sum() > 0:
                        st.dataframe(missing_values[missing_values > 0])
                    else:
                        st.write("No missing values found!")
                
                # Save file option
                if st.button("Save Test Data", key="save_test"):
                    if save_uploaded_file(uploaded_file, TEST_DATA_DIR):
                        st.success("✅ Test data saved successfully!")
                        st.rerun()
                    else:
                        st.error("❌ Failed to save test data.")
                        
            except Exception as e:
                st.error(f"❌ Error reading file: {str(e)}")
    
    with tab3:
        st.markdown("### Upload Prediction Data")
        st.markdown("Upload datasets here for bulk predictions.")
        
        uploaded_file = st.file_uploader(
            "Choose a CSV file for predictions",
            type=['csv'],
            key="predict_uploader",
            help="Upload a CSV file containing data you want to make predictions on"
        )
        
        if uploaded_file is not None:
            try:
                # Read and validate the file
                df = pd.read_csv(uploaded_file)
                
                # Display basic info
                st.success(f"✅ File uploaded successfully! Shape: {df.shape}")
                
                # Show data preview
                st.markdown("#### Data Preview")
                st.dataframe(df.head())
                
                # Show data info
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Dataset Info")
                    st.write(f"**Rows:** {df.shape[0]}")
                    st.write(f"**Columns:** {df.shape[1]}")
                    st.write(f"**Memory Usage:** {df.memory_usage().sum() / 1024:.2f} KB")
                
                with col2:
                    st.markdown("#### Missing Values")
                    missing_values = df.isnull().sum()
                    if missing_values.sum() > 0:
                        st.dataframe(missing_values[missing_values > 0])
                    else:
                        st.write("No missing values found!")
                
                # Save file option
                if st.button("Save Prediction Data", key="save_predict"):
                    if save_uploaded_file(uploaded_file, PREDICT_DATA_DIR):
                        st.success("✅ Prediction data saved successfully!")
                        st.rerun()
                    else:
                        st.error("❌ Failed to save prediction data.")
                        
            except Exception as e:
                st.error(f"❌ Error reading file: {str(e)}")
    
    # Show existing files
    st.markdown("---")
    st.markdown('<h3 class="sub-header">📁 Existing Files</h3>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### Training Data")
        train_files = list(TRAIN_DATA_DIR.glob("*.csv"))
        if train_files:
            for file in train_files:
                st.write(f"📄 {file.name}")
                if st.button(f"Delete {file.name}", key=f"del_train_{file.name}"):
                    file.unlink()
                    st.success(f"Deleted {file.name}")
                    st.rerun()
        else:
            st.write("No training files found")
    
    with col2:
        st.markdown("#### Test Data")
        test_files = list(TEST_DATA_DIR.glob("*.csv"))
        if test_files:
            for file in test_files:
                st.write(f"📄 {file.name}")
                if st.button(f"Delete {file.name}", key=f"del_test_{file.name}"):
                    file.unlink()
                    st.success(f"Deleted {file.name}")
                    st.rerun()
        else:
            st.write("No test files found")
    
    with col3:
        st.markdown("#### Prediction Data")
        predict_files = list(PREDICT_DATA_DIR.glob("*.csv"))
        if predict_files:
            for file in predict_files:
                st.write(f"📄 {file.name}")
                if st.button(f"Delete {file.name}", key=f"del_predict_{file.name}"):
                    file.unlink()
                    st.success(f"Deleted {file.name}")
                    st.rerun()
        else:
            st.write("No prediction files found")