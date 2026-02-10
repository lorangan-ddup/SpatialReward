#!/bin/bash
# Generic API Inference for MERBench

SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
cd $SHELL_FOLDER

# Paths
DATA_PATH="/path/to/your/data"
OUTPUT_DIR="results"
MODEL="gpt-5"

# API Settings (Can also be set via env vars)
API_KEY="your_api_key"
API_BASE="https://api.openai.com/v1"

mkdir -p ${OUTPUT_DIR}
mkdir -p logs

echo "=============================================="
echo "Generic API Inference: ${MODEL}"
echo "=============================================="

python3 inference.py \
    --data_path ${DATA_PATH} \
    --output_path ${OUTPUT_DIR}/${MODEL}.json \
    --api_key ${API_KEY} \
    --api_base_url ${API_BASE} \
    --model ${MODEL} \
    --max_workers 500 \
    2>&1 | tee logs/inference_${MODEL}.log

if [ $? -ne 0 ]; then
    echo "❌ Inference failed!"
    exit 1
fi

python3 ../evaluate.py \
    --result_file ${OUTPUT_DIR}/${MODEL}.json \
    --output ${OUTPUT_DIR}/${MODEL}_accuracy.json

echo "✅ Completed!"
