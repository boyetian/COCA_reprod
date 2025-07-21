#!/bin/bash

# This script downloads the ImageNet-C dataset.
# Note: You will need to have an account on the ImageNet website to download the data.

# Create a data directory if it doesn't exist
# mkdir -p data

# Download the ImageNet-C dataset
# You will need to replace the following URL with the actual download link from the ImageNet website.
# The file is ~76GB.
wget -c https://zenodo.org/record/2235448/files/imagenet-c.tar 

# Download the Tiny ImageNet-C dataset for smaller experiments
# wget https://zenodo.org/records/2536630/files/Tiny-ImageNet-C.tar?download=1

# Extract the dataset
tar -xvf imagenet-c.tar