#!/bin/bash
# This script is used to run the training server with default settings.

export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=4,5,6,7

accelerate launch ./api_server.py \
-t \
-p 8001