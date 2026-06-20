# Use a slim Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements file first to leverage Docker caching
COPY requirements.txt .

# Install dependencies (using --prefer-binary to use precompiled wheels and avoid compiling packages like hdbscan)
RUN pip3 install --no-cache-dir --prefer-binary -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the default Streamlit port (Render will override this via $PORT)
EXPOSE 8501

# Start Streamlit, binding to the PORT environment variable injected by Render
CMD ["sh", "-c", "streamlit run app/main.py --server.port $PORT --server.address 0.0.0.0"]
