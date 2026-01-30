#!/usr/bin/env bash

set -e  # exit on error

if [ -z "$1" ]; then
  echo "Usage: $0 <zip_file_url>"
  exit 1
fi

ZIP_URL="$1"
ZIP_NAME=$(basename "$ZIP_URL")

echo "Downloading $ZIP_URL ..."
if command -v curl >/dev/null 2>&1; then
  curl -L -o "$ZIP_NAME" "$ZIP_URL"
elif command -v wget >/dev/null 2>&1; then
  wget -O "$ZIP_NAME" "$ZIP_URL"
else
  echo "Error: curl or wget is required"
  exit 1
fi

echo "Unzipping $ZIP_NAME ..."
unzip -o "$ZIP_NAME"

echo "Done ✅"
