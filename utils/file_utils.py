import os
from pathlib import Path
import shutil

def save_uploaded_file(uploaded_file, target_dir: Path) -> bool:
    """Save uploaded Streamlit file to target directory."""
    try:
        file_path = target_dir / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return True
    except Exception as e:
        print(f"Error saving file: {e}")
        return False

def validate_csv_format(df, required_columns=None) -> bool:
    """Basic CSV validation. Optionally ensure required columns exist."""
    if df is None or not hasattr(df, 'columns'):
        return False
    if required_columns:
        return set(required_columns).issubset(set(df.columns))
    return True

def cleanup_directory(target_dir: Path):
    """Delete all files in the given directory."""
    for file in target_dir.glob("*"):
        try:
            file.unlink()
        except Exception:
            pass