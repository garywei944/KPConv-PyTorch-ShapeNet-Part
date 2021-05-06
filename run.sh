#!/usr/bin/env bash

PYTHONPATH=$(pwd):$PYTHONPATH
export PYTHONPATH

python src/models/train_s3dis.py
