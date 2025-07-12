# Base image with Python and OS packages
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_PORT=8501 \
    STREAMLIT_HOME=/app

# Set work directory
WORKDIR ${STREAMLIT_HOME}

# Install OS dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    wget \
    libglib2.0-0 \
    libsm6 \
    libxrender-dev \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy app source code
COPY . .

# Expose Streamlit default port
EXPOSE ${STREAMLIT_PORT}

# Run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]