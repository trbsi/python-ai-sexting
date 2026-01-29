#!/bin/bash

# ==========================
# Deployment Script
# ==========================
# Exit on any error
set -e

# Variables
VENV_DIR="venv"                    # Name of Python virtual environment
REQ_FILE="lora_requirements.txt"   # Requirements file

# ==========================
# Install Python, pip, and dependencies
# ==========================
echo "Installing Python3, pip, and required tools..."
sudo apt-get update
sudo apt-get install -y python3 python3-venv python3-pip screen git nano

# ==========================
# Create and activate Python virtual environment
# ==========================
echo "Creating Python virtual environment..."
python3 -m venv "$VENV_DIR"

echo "Activating virtual environment..."
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# Upgrade pip inside venv
pip install --upgrade pip

# ==========================
# Install packages from requirements file
# ==========================
echo "Installing Python packages from $REQ_FILE..."
pip install -r "$REQ_FILE"
pip install -U bitsandbytes

# ==========================
# Copy .env file
# ==========================
cp .env.example .env

echo "Deployment complete. Virtual environment is ready in $VENV_DIR."
