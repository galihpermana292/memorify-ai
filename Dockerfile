# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install the missing system dependency for OpenCV
RUN apt-get update && apt-get install -y libgl1-mesa-glx

# Copy and install your Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application code into the container
COPY . .

# Expose the port the app will run on
EXPOSE 8080

# Define the command to run your app when the container starts
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]