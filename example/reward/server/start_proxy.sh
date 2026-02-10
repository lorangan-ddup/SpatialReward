#!/bin/bash
# source /path/to/anaconda/bin/activate
# conda activate spatialreward
# export HF_HOME="/path/to/huggingface"
# Start the Reward Proxy Server
# The proxy distributes requests to multiple worker servers
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CONFIG_PATH="$SCRIPT_DIR/server_configs/SpatialReward.yml"


echo "🚀 Starting Spatial Reward Proxy Server"
echo "   Script Dir: $SCRIPT_DIR"
echo "   Config: $CONFIG_PATH"
echo ""

cd "$SCRIPT_DIR" && python reward_proxy.py \
    --host 0.0.0.0 \
    --config_path "$CONFIG_PATH"

echo ""
echo "Proxy server stopped."
