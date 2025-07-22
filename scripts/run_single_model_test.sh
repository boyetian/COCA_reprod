#!/bin/bash

CONFIG_FILE="configs/resnet50_resnet18.yaml"

python scripts/test_single_model_accuracy.py --config $CONFIG_FILE
