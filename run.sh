#!/usr/bin/env bash

# Activate the virtual environment
. ./.venv/bin/activate

# Run the following commands to reproduce all the experiments

# Tuning in_radius
for radius in 0.25 0.15 0.05; do
  python -m src.models.train_shapnet_part --in-radius $radius
done

# Try different categories
for ctg in Car Lamp; do
  python -m src.models.train_shapnet_part --ctg $ctg --in-radius 0.2
done

# Tuning in_features_dim
for dim in 1 4; do
  python -m src.models.train_shapnet_part --feature-dim $dim
done
