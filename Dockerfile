# Use the standard Python image which comes pre-packed with build-essential tools
FROM python:3.10

# Set working directory
WORKDIR /app

# Copy requirements file first to leverage Docker caching
COPY requirements.txt .

# Install dependencies (they will compile successfully using pre-installed compilers)
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the default Streamlit port (Render will override this via $PORT)
EXPOSE 8501

# Start Streamlit, binding to the PORT environment variable injected by Render
CMD ["sh", "-c", "streamlit run app/main.py --server.port $PORT --server.address 0.0.0.0"]
