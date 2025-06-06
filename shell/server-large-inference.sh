#!/bin/bash
# This script is used to run the inference server with large model.

export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=4,5,6,7


python ./api_server.py \
-i \
-p 8001 \
--model Qwen/Qwen3-8B