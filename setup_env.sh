#!/usr/bin/env bash

# Set up the environment for this project
# Refer to https://github.com/HuguesTHOMAS/KPConv-PyTorch/blob/master/INSTALL.md for more detials

# Make a new virtual environment
python3 -m venv .venv
. ./.venv/bin/activate

# Install PyTorch
pip3 install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
# Install all requirements
pip3 install numpy scikit-learn PyYAML matplotlib jupyter mayavi PyQt5 python-dotenv

# Compile the C++ wrapper, tested on Ubuntu 20.04
function cpp_wrappers() {
  cd src/cpp_wrappers || exit
  bash compile_wrappers.sh
}

(cpp_wrappers)
