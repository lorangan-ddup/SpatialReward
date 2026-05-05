#!/bin/bash
# MMRB2 Image Editing Benchmark Evaluation Script
# Set the script directory
SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
cd $SHELL_FOLDER

# source /path/to/anaconda/bin/activate
# conda activate spatialreward
# export HF_HOME="/path/to/huggingface"

# ============ Configuration ============
# Data and Model
DATA_PATH="/path/to/your/data"
CHECKPOINT_PATH="SpatialReward/SpatialReward-8B"

# Output directory
OUTPUT_DIR="results/mmrb2"

# Score aggregation method
# - "min": Take minimum of SC dimensions
# - "mean": Take average of SC dimensions
# - "weighted_power": Use weighted power formula
SCORE_AGGREGATION="weighted_power"

# Weighted Power parameters (only used when SCORE_AGGREGATION="weighted_power")
# Formula: ((w1*s1+w2*s2)**a) * ((w3*s3+w4*s4)**(1-a))
WEIGHTED_POWER_PARAMS="0.6 0.4 0.5 0.5 0.8"

# Inference configuration
TEMPERATURE=0.7
TENSOR_PARALLEL_SIZE=8
MAX_NUM_SEQS=256
MAX_MODEL_LEN=12240
MAX_NUM_BATCHED_TOKENS=65536
GPU_MEMORY_UTILIZATION=0.85
BATCH_SIZE=512

# Add timestamp to output filename
ADD_TIMESTAMP=true

# Single image only mode
SINGLE_IMAGE_ONLY=false

# Create output directory and logs
mkdir -p ${OUTPUT_DIR}
mkdir -p logs

# Build inference command
CMD="CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python torch_offline.py \
    --data_path ${DATA_PATH} \
    --output_path ${OUTPUT_DIR}/results.json \
    --model ${CHECKPOINT_PATH} \
    --temperature ${TEMPERATURE} \
    --tensor_parallel_size ${TENSOR_PARALLEL_SIZE} \
    --max_num_seqs ${MAX_NUM_SEQS} \
    --max_model_len ${MAX_MODEL_LEN} \
    --max_num_batched_tokens ${MAX_NUM_BATCHED_TOKENS} \
    --gpu_memory_utilization ${GPU_MEMORY_UTILIZATION} \
    --batch_size ${BATCH_SIZE} \
    --score_aggregation ${SCORE_AGGREGATION} \
    --top_p 0.9 \
    --top_k 20 \
    --max_tokens 4096 \
    --num_workers 16"

# Add weighted_power_params if using weighted_power aggregation
if [ "${SCORE_AGGREGATION}" = "weighted_power" ]; then
    CMD="${CMD} --weighted_power_params ${WEIGHTED_POWER_PARAMS}"
fi

# Add timestamp flag if enabled
if [ "${ADD_TIMESTAMP}" = true ]; then
    CMD="${CMD} --add_timestamp"
fi

# Add single_image_only flag if enabled
if [ "${SINGLE_IMAGE_ONLY}" = true ]; then
    CMD="${CMD} --single_image_only"
    echo "Only evaluating single-image editing tasks"
else
    echo "Evaluating all tasks (single-image + multi-image fusion)"
fi

echo "Starting MMRB2 inference..."
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Temperature: ${TEMPERATURE}"
echo "Data: ${DATA_PATH}"
echo "Output: ${OUTPUT_DIR}"
echo "=========================================="

eval ${CMD} 2>&1 | tee -a logs/mmrb2.log

if [ $? -ne 0 ]; then
    echo "Inference failed!"
    exit 1
fi

echo ""
echo "Inference completed successfully!"
echo "Calculating accuracy..."
    
# Find the results file
if [ "${ADD_TIMESTAMP}" = true ]; then
    RESULT_FILE=$(ls -t ${OUTPUT_DIR}/results_*.json 2>/dev/null | head -1)
    if [ -z "$RESULT_FILE" ]; then
        echo "No timestamped results file found!"
        exit 1
    fi
else
    RESULT_FILE="${OUTPUT_DIR}/results.json"
fi

python calculate_accuracy.py \
    --result_file "${RESULT_FILE}" \
    --benchmark_file ${DATA_PATH} \
    --output_file "${RESULT_FILE%.json}_evaluation.json" 2>&1 | tee -a logs/mmrb2_accuracy.log

if [ $? -ne 0 ]; then
    echo "Accuracy calculation failed!"
    exit 1
fi

echo ""
echo "=========================================="
echo "Pipeline completed successfully!"
echo "=========================================="
echo "Results saved to: ${OUTPUT_DIR}/"
echo "Files created:"
echo "   - results*.json: Raw inference results"
echo "   - results*_evaluation.json: Accuracy metrics"
echo "=========================================="
