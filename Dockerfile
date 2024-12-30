# Dockerfile

# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy necessary files
COPY requirements.txt /app/requirements.txt
COPY fruits_recognition_model.h5 /app/fruits_recognition_model.h5
COPY app.py /app/app.py

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port the app runs on
EXPOSE 5000

# Run the application
CMD ["python", "app.py"]
