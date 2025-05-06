#!/bin/bash
# This script is used to run the inference server with default settings.

export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=4

accelerate launch ./api_server.py \
-i \
-p 8001