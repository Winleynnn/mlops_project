# Use the official Python image as the base image
# ARG BASE_CONTAINER=ubuntu
# ARG UBUNTU_VERSION=22.04

# FROM $BASE_CONTAINER:$UBUNTU_VERSION
FROM python:3.10-slim
ARG PYTHON_VERSION=3.10

ENV PYTHON_VERSION 3.10
# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Install the required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Set environment variables to avoid issues with stdout/stderr buffering
ENV PYTHONUNBUFFERED=1

# Specify the entrypoint for the container
ENTRYPOINT ["python", "commands.py"]