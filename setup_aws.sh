#!/usr/bin/env bash

function dataset() {
  mkdir -p ~/Downloads
  cd ~/Downloads || exit
  wget --no-check-certificate https://shapenet.cs.stanford.edu/ericyi/shapenetcore_partanno_segmentation_benchmark_v0.zip
  unzip shapenetcore_partanno_segmentation_benchmark_v0.zip
}

function setup_env() {
  cp -f .env.template .env

  python3 -m venv .venv
  . ./.venv/bin/activate

  pip3 install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
  pip3 install numpy scikit-learn PyYAML python-dotenv
}

(dataset &)
(setup_env &)

wait
