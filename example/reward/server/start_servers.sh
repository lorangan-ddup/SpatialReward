#!/bin/bash
# source /path/to/anaconda/bin/activate
# conda activate spatialreward
# export HF_HOME="/path/to/huggingface"
# Spatial Reward Server Startup Script
# This script starts the reward server workers on multiple machines

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CONFIG_PATH="$SCRIPT_DIR/server_configs/SpatialReward.yml"

# Read configuration
WORKER_BASE_PORT=$(python3 -c "import yaml; config=yaml.safe_load(open('$CONFIG_PATH')); print(config['server']['worker_base_port'])")
TENSOR_PARALLEL_SIZE=$(python3 -c "import yaml; config=yaml.safe_load(open('$CONFIG_PATH')); print(config['reward']['tensor_parallel_size'])")

# Calculate number of servers per machine
NUM_SERVERS=$((8 / TENSOR_PARALLEL_SIZE))

echo "🚀 Starting Spatial Reward Servers"
echo "   Script Dir: $SCRIPT_DIR"
echo "   Config: $CONFIG_PATH"
echo "   Base Port: $WORKER_BASE_PORT"
echo "   Tensor Parallel Size: $TENSOR_PARALLEL_SIZE"
echo "   Servers per machine: $NUM_SERVERS"
echo ""

# Create logs directory
mkdir -p "$SCRIPT_DIR/logs"

# Start servers
for ((i=0; i<$NUM_SERVERS; i++)); do
    PORT=$((WORKER_BASE_PORT + i))
    GPU_ID=$i  # Assign GPU: server 0 → GPU 0, server 1 → GPU 1, etc.
    
    echo "Starting server $i on port $PORT (GPU $GPU_ID)..."
    
    cd "$SCRIPT_DIR" && CUDA_VISIBLE_DEVICES=$GPU_ID nohup python reward_server.py \
        --host 0.0.0.0 \
        --port $PORT \
        --config_path "$CONFIG_PATH" \
        > logs/reward_server_${PORT}.log 2>&1 &
    
    echo "  Process ID: $! (GPU: $GPU_ID)"
    sleep 2
done

echo ""
echo "✅ All reward servers started!"
echo "Check logs in: $SCRIPT_DIR/logs/reward_server_*.log"