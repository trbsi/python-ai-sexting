#!/bin/bash

# ==========================
# Zip and Download Script
# ==========================
# Exit on any error
set -e

# Variables
VENV_DIR="venv"          # Virtual environment folder to exclude
ZIP_FILE="lora_model.zip"   # Name of zip file

# ==========================
# Step 1: Zip project folder
# ==========================
echo "Zipping current folder into $ZIP_FILE."
zip -r "$ZIP_FILE" ./trained_model

# ==========================
# Step 2: Copy zip to local laptop
# ==========================
echo "e.g.: scp -P 37149  root@86.57.175.52:/workspace/repo/lora_model.zip ."
