# Use a slim Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies needed for compiling packages like hdbscan and lightgbm
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first to leverage Docker caching
COPY requirements.txt .

# Install dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the default Streamlit port (Render will override this via $PORT)
EXPOSE 8501

# Start Streamlit, binding to the PORT environment variable injected by Render
CMD ["sh", "-c", "streamlit run app/main.py --server.port $PORT --server.address 0.0.0.0"]
