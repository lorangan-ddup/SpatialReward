#!/bin/bash
# MERBench (Multi-Edit Reward Benchmark) Evaluation Script

# source /path/to/anaconda/bin/activate
# conda activate spatialreward
# export HF_HOME="/path/to/huggingface"

# Set the script directory
SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
cd $SHELL_FOLDER

# ============ Configuration ============
# Data and Model
# Local path:  DATA_PATH="/path/to/MERBench"
# HuggingFace: DATA_PATH="SpatialReward/MER-Bench"
DATA_PATH="SpatialReward/MER-Bench"
CHECKPOINT_PATH="SpatialReward/SpatialReward-8B"

# Output directory 
OUTPUT_DIR="results/merbench"

# Score aggregation method
# - "min": Take minimum of SC dimensions
# - "mean": Take average of SC dimensions
# - "weighted_power": Use weighted power formula
SCORE_AGGREGATION="weighted_power"

# Weighted Power parameters (only used when SCORE_AGGREGATION="weighted_power")
# Formula: ((w1*s1+w2*s2)**a) * ((w3*s3+w4*s4)**(1-a))
WEIGHTED_POWER_PARAMS="0.6 0.4 0.5 0.5 0.8"

# Configuration
TEMPERATURE=0.0
TENSOR_PARALLEL_SIZE=8
MAX_NUM_SEQS=256
MAX_MODEL_LEN=6144
MAX_NUM_BATCHED_TOKENS=32768
GPU_MEMORY_UTILIZATION=0.85
BATCH_SIZE=512

MAX_PIXELS=589824   
MIN_PIXELS=3136     

# Create directories
mkdir -p ${OUTPUT_DIR}
mkdir -p logs

echo "=============================================="
echo "MERBench Evaluation"
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Temperature: ${TEMPERATURE}"
echo "Output: ${OUTPUT_DIR}"
echo "Score Aggregation: ${SCORE_AGGREGATION}"
echo "=============================================="

# Run inference
echo ""
echo "[Running inference...]"

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
    --num_workers 16 \
    --max_pixels ${MAX_PIXELS} \
    --min_pixels ${MIN_PIXELS}"

if [ "${SCORE_AGGREGATION}" = "weighted_power" ]; then
    CMD="${CMD} --weighted_power_params ${WEIGHTED_POWER_PARAMS}"
fi

eval ${CMD} 2>&1 | tee -a logs/merbench.log

if [ $? -ne 0 ]; then
    echo "❌ Inference failed!"
    exit 1
fi

echo ""
echo "=============================================="
echo "Calculating Accuracy..."
echo "=============================================="

python calculate_accuracy.py \
    --result_file ${OUTPUT_DIR}/results.json \
    --data_path ${DATA_PATH} 2>&1 | tee -a logs/merbench_accuracy.log

if [ $? -ne 0 ]; then
    echo "❌ Accuracy calculation failed!"
    exit 1
fi

echo ""
echo "=============================================="
echo "✅ Pipeline completed successfully!"
echo "=============================================="
echo "📁 Results saved to: ${OUTPUT_DIR}/"
echo "📄 Files created:"
echo "   - all_results.json (all pair types)"
echo "   - accuracy_report.json"
echo "=============================================="
