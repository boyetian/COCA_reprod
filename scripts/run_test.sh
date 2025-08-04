#!/bin/bash

CONFIG_FILE="configs/resnet50_resnet18.yaml"

export PYTHONPATH="$PYTHONPATH:$(dirname "$0")/.."

python test_accuracy.py --config $CONFIG_FILE
