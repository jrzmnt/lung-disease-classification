#!/bin/bash
# Script to test all models

# List of models to be tested
MODELS=("simple-cnn" "mobile-net" "shuffle-net" "resnet18" "resnet34")

# Loop to test each model
for MODEL in "${MODELS[@]}"; do
    echo "Starting testing for: $MODEL"
    python test.py --model $MODEL

    # Optional: Add a command to pause between tests, if needed
    sleep 5
done

echo "All tests are completed."
