#!/usr/bin/env bash

conda env create -f environment.yml

TORCH=$(python -c "import torch; print(torch.__version__)")
CUDA=$(python -c "import torch; print(torch.version.cuda)" | sed -ne 's/\.//gp')

pip install torch-scatter -f "https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html"
pip install torch-sparse -f "https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html"
pip install torch-cluster -f "https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html"
pip install torch-spline-conv -f "https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html"
pip install torch-geometric
