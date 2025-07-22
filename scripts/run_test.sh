#!/bin/bash

CONFIG_FILE="configs/resnet50_resnet18.yaml"

python scripts/test_accuracy.py --config $CONFIG_FILE
