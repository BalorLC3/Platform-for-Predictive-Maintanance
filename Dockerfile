FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code and model artifact
COPY src/ /app/src/
COPY checkpoints/lstm_model_inference.pth /app/notebooks/lstm_model_inference.pth

# Expose the port the API will run on
EXPOSE 8000

# command to run application using Uvicorn
# --host 0.0.0.0 makes it accessible from outside the container
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]