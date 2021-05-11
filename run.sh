#!/usr/bin/env bash

# Activate the virtual environment
. ./.venv/bin/activate

for radius in 0.25 0.15 0.05; do
  python -m src.models.train_shapnet_part --in-radius $radius
done

wait

for ctg in Car Lamp; do
  python -m src.models.train_shapnet_part --ctg $ctg --in-radius 0.2
done

wait
