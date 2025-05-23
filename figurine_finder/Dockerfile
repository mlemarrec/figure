# Base Image
FROM python:3.9-slim

# Set Environment Variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set Working Directory
WORKDIR /app

# Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
# We copy the app and model directories
COPY ./app ./app
COPY ./model ./model

# Expose port
EXPOSE 8000

# Command to run the application
# Assumes main.py is in the 'app' directory and the FastAPI instance is named 'app'
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
