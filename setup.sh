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
cd .. # change directory to metrics
# Install YiSi
if [ ! -d "yisi" ]; then
    echo "Cloning YiSi repository ..."
    git clone https://github.com/chikiulo/yisi.git
    cd yisi 
    cd src
    make all -j 4 > /dev/null 2>&1 && echo "YiSi installed successfully."
else
    echo "Skipping git clone, yisi already exists."
fi

# Prompt the user for the huggingface token
read -p "Please enter your HF token: " HF_TOKEN
export HF_TOKEN
echo "HF_TOKEN is set to: $HF_TOKEN"

    