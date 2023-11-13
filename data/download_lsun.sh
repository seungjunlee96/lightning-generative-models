#!/bin/bash

# Clone the LSUN repository
git clone https://github.com/fyu/lsun.git ./data/dataset/LSUN

# Change directory to LSUN
cd ./data/dataset/LSUN

# Run the download script
python3 download.py

# Unzip all zip files in the directory
for file in *.zip; do
    unzip -o "$file"
done

# Remove all zip files
rm *.zip
