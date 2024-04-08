#!/bin/bash
# Script to train all models

# List of models to be trained
MODELS=("simple-cnn" "mobile-net" "shuffle-net" "resnet18" "resnet34")

# Loop to train each model
for MODEL in "${MODELS[@]}"; do
    echo "Starting training for: $MODEL"
    python train.py --model $MODEL

    # Optional: Add a command to pause between trainings, if needed
    sleep 5
done

echo "All trainings are completed."
