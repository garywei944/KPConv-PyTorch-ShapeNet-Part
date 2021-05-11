#!/usr/bin/env bash

# Activate the virtual environment
. ./.venv/bin/activate

for radius in 0.25 0.15 0.05; do
  python -m src.models.train_shapnet_part --in-radius $radius
done
