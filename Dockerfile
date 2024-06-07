# install ubuntu and python 3.10
FROM ubuntu:22.04

ENV PYTHON_VERSION 3.10
# Set the working directory in the container
WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
  && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    wget \
    curl \
    git \
    vim \
    openssh-client \
    build-essential \
    zip \
    unzip \
    ffmpeg \
    libsm6 \
    libxext6 \
    python3-pip

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Install the required Python packages
RUN pip install --no-cache-dir -r requirements.txt

RUN git config --global user.email "GolovachLena@example.com"
RUN git config --global user.name "Stariyi BOG"


# Copy the rest of the application code into the container
COPY . .
RUN apt install git-all -y
RUN git init
RUN git add .
RUN git commit -m "cool"


# Set environment variables to avoid issues with stdout/stderr buffering
ENV PYTHONUNBUFFERED=1

# Specify the entrypoint for the container
ENTRYPOINT ["python", "commands.py"]
