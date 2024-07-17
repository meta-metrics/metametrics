#!/bin/bash

# Install requirements
echo "Installing requirements ..."
pip install . > /dev/null 2>&1 && echo "Requirements installed successfully."

# Installing BLEURT
cd meta_metrics/metrics
if [ ! -d "bleurt" ]; then
    echo "Cloning BLEURT repository ..."
    git clone https://github.com/google-research/bleurt.git
else
    echo "Skipping git clone, BLEURT already exists."
fi
cd bleurt
echo "Installing BLEURT ..."
pip install . > /dev/null 2>&1 && echo "BLEURT installed successfully."

# Prompt the user for the huggingface token
read -p "Please enter your HF token: " HF_TOKEN
export HF_TOKEN
echo "HF_TOKEN is set to: $HF_TOKEN"
