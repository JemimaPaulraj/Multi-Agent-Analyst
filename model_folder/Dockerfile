FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PORT=8080

WORKDIR /opt/program

# System deps (often needed for prophet/cmdstanpy)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all scripts
COPY app.py .
COPY serve .
COPY train .

# Fix Windows line endings (CRLF -> LF) and make executable
RUN sed -i 's/\r$//' /opt/program/serve && chmod +x /opt/program/serve
RUN sed -i 's/\r$//' /opt/program/train && chmod +x /opt/program/train

# Add program directory to PATH so SageMaker can find 'serve' and 'train' commands
ENV PATH="/opt/program:${PATH}"

EXPOSE 8080

# No ENTRYPOINT - SageMaker will call either:
#   - docker run <image> train   (for training jobs)
#   - docker run <image> serve   (for inference endpoints)
