#!/usr/bin/env bash

# Activate the virtual environment
. ./.venv/bin/activate

# Train S3DIS dataset with configuration file `./config/s3dis_2060.yml`
python -m src.models.train_s3dis -c s3dis_2060.yml

## Plot the loss, time, and IoU
#python -m src.visualization.plot_convergence
#
## Run the test
#python -m src.test.test_models
#
## Interactive 3D visualization
#python -m src.visualization.visualize_deformations
