# Use a lighter python version
FROM python:3.10-slim

# Prevent python from writing pyc files to disc
ENV PYTHONDONTWRITEBYTECODE=1
# Prevent python from buffering stdout and stderr
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies (needed for OpenCV/Pillow usually)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]