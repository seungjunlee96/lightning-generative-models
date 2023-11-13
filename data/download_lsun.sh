#!/bin/bash

# Clone the LSUN repository
git clone https://github.com/fyu/lsun.git ./data/dataset/LSUN

# Change directory to LSUN
cd ./data/dataset/LSUN

# Check if a command-line argument is provided
if [ "$1" == "all" ]; then
    # Download all datasets
    python3 download.py
else
    # Default to download only bedroom dataset
    python3 download.py -c bedroom
fi

# Unzip all zip files in the directory
for file in *.zip; do
    unzip -o "$file"
done

# Remove all zip files
rm *.zip
