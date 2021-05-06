#!/usr/bin/env bash

# Set up the environment for this project
# Refer to https://github.com/HuguesTHOMAS/KPConv-PyTorch/blob/master/INSTALL.md for more detials

# Make new conda environment
conda env create -f environment.yml
conda activate kpconv

# Compile the C++ wrapper, tested on Ubuntu 20.04
function cpp_wrappers() {
  cd src/cpp_wrappers || exit
  bash compile_wrappers.sh
}

(cpp_wrappers)

