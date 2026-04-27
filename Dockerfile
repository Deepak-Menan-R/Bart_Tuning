# Use official Python 3.9 image
FROM python:3.9

# Install required system dependencies before switching to non-root user
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Set up a non-root user
RUN useradd -m -u 1000 user
USER user

# Set environment variables
ENV PATH="/home/user/.local/bin:$PATH"
WORKDIR /app

# Copy dependency file and install Python packages
COPY --chown=user requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy the entire project to the container
COPY --chown=user . /app

# Expose the port Hugging Face Spaces expects
EXPOSE 7860

# Start the FastAPI application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
