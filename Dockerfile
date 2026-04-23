# Base Python image for the Flask service.
FROM python:3.11-slim

# Set the working directory for the container.
WORKDIR /app

# Install libraries required by OpenCV and other image utilities.
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the YOLO model weights into the container.
COPY theBest.pt .

# Copy the application code.
COPY . .

# Run the Flask app when the container starts.
CMD ["python3", "app.py"]
