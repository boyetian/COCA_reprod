#!/bin/bash

# This script downloads the ImageNet-C dataset.

# Create a data directory if it doesn't exist
mkdir -p ImageNet-C
cd ImageNet-C

# Download the ImageNet-C dataset
# You will need to replace the following URL with the actual download link from the ImageNet website.
# The file is ~76GB.
wget -c https://zenodo.org/records/2235448/files/blur.tar
tar -xvf blur.tar
rm blur.tar

wget -c https://zenodo.org/records/2235448/files/digital.tar
tar -xvf digital.tar
rm digital.tar

wget -c https://zenodo.org/records/2235448/files/noise.tar
tar -xvf noise.tar
rm noise.tar

wget -c https://zenodo.org/records/2235448/files/weather.tar
tar -xvf weather.tar
rm weather.tar

# Download the Tiny ImageNet-C dataset for smaller experiments
# wget https://zenodo.org/records/2536630/files/Tiny-ImageNet-C.tar