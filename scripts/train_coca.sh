#!/bin/bash

# This script runs the COCA test-time adaptation.

# Usage: ./scripts/train_coca.sh <config_file>

# Example: ./scripts/train_coca.sh configs/vit_base_mobilvit.yaml

python main.py --config $1 