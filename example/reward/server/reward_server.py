#!/usr/bin/env python3
"""
Spatial Reward Server - Flask Server (Handles requests and queue management only)
"""
import dotenv
dotenv.load_dotenv(override=True)

from typing import List, Dict, Tuple
import argparse
import pickle
import json
import warnings
import threading
from queue import Queue
import uuid
import time

from flask import Flask, request, jsonify
from PIL import Image
import yaml

# Import scorer
from spatial_reward_scorer import SpatialRewardScorer

warnings.filterwarnings("ignore")

app = Flask(__name__)

# Global queue and result storage
request_queue = Queue()
results = {}

# =============================================================================
# WORKER THREAD (Background processing thread)
# =============================================================================

def vlm_worker(scorer: SpatialRewardScorer):
    """Background worker thread, fetches tasks from queue and processes them"""
    print("🚀 Worker thread started, waiting for tasks...")
    
    while True:
        try:
            task_id, input_images, output_images, metadata = request_queue.get()
            
            # Call scorer
            outputs = scorer.score(input_images, output_images, metadata)
            
            # Build result
            result_payload = []
            for (reward, reasoning), meta in zip(outputs, metadata):
                result_payload.append({
                    "score": 1.0 if reward >= 0.5 else 0.0,
                    "reward": reward,
                    "reasoning": reasoning,
                    "strict_reward": reward,
                    "meta_data": meta,
                    "group_reward": {meta.get("tag", "spatial"): reward},
                    "group_strict_reward": {meta.get("tag", "spatial"): reward},
                })
            
            results[task_id] = pickle.dumps(result_payload)
            
        except Exception as e:
            print(f"❌ Worker error for task {task_id[:8]}: {e}")
            import traceback
            traceback.print_exc()
            
            error_result = {"error": f"Internal server error: {e}"}
            results[task_id] = pickle.dumps(error_result)
            
        finally:
            request_queue.task_done()

# =============================================================================
# REQUEST PARSING
# =============================================================================

def parse_and_validate_request(raw_data: bytes) -> Tuple[List, List, List, str]:
    """
    Parse request data
    
    Returns:
        (input_images, output_images, metadata, error_msg)
    """
    try:
        data = pickle.loads(raw_data)
        input_images_data = data['input_images']
        output_images_data = data['output_image']
        metadata = data['meta_data']
    except Exception as e:
        return None, None, None, f"Failed to parse request: {e}"
    
    # Convert images to RGB
    output_images = []
    for img_data in output_images_data:
        output_images.append(img_data.convert('RGB'))
    
    input_images = []
    for img_list_data in input_images_data:
        img_list = []
        for img_data in img_list_data:
            img_list.append(img_data.convert('RGB'))
        input_images.append(img_list)
    
    # Process metadata
    parsed_metadata = []
    for meta in metadata:
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except json.JSONDecodeError:
                meta = {'instruction': meta}
        
        if not isinstance(meta, dict):
            return None, None, None, "Metadata must be dict or JSON string"
        
        parsed_metadata.append(meta)
    
    return input_images, output_images, parsed_metadata, None

# =============================================================================
# FLASK ROUTES
# =============================================================================

@app.route('/', methods=['POST'])
def evaluate_batch():
    """Receive scoring request"""

    # Parse request
    input_images, output_images, metadata, error_msg = parse_and_validate_request(request.data)
    if error_msg:
        print(f"❌ Request validation failed: {error_msg}")
        return jsonify({"error": error_msg}), 400
    
    # Create task
    task_id = str(uuid.uuid4())
    request_queue.put((task_id, input_images, output_images, metadata))
    
    print(f"📥 Task {task_id[:8]} enqueued: {len(output_images)} images, queue size: {request_queue.qsize()}", flush=True)
    
    # Wait for result
    timeout_seconds = 600
    start_time = time.time()
    
    while True:
        if task_id in results:
            result_data = results.pop(task_id)
            elapsed = time.time() - start_time
            print(f"📤 Task {task_id[:8]} completed in {elapsed:.2f}s")
            return result_data, 200, {'Content-Type': 'application/octet-stream'}
        
        if time.time() - start_time > timeout_seconds:
            print(f"⌛ Task {task_id[:8]} timed out")
            return jsonify({"error": "Request timed out"}), 504
        
        time.sleep(0.05)

@app.route('/ping', methods=['GET'])
def ping():
    """Health check"""
    return jsonify({"status": "ok"}), 200

# =============================================================================
# MAIN
# =============================================================================

def arg_parser():
    parser = argparse.ArgumentParser(description='Spatial Reward Server')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Server host')
    parser.add_argument('--port', type=int, default=18888, help='Server port')
    parser.add_argument('--config_path', type=str, 
                       default='server_configs/SpatialReward.yml',
                       help='Config file path')
    return parser.parse_args()

def main(args):
    
    # 1. Load config
    print("⚡ Loading config...")
    config = yaml.safe_load(open(args.config_path, "r"))
    
    # 2. Initialize scorer
    print("⚡ Initializing scorer...")
    scorer = SpatialRewardScorer(config["reward"])
    
    # 3. Start background worker thread
    worker_thread = threading.Thread(target=vlm_worker, args=(scorer,), daemon=True)
    worker_thread.start()
    
    # 4. Start Flask server
    print(f"🔥 Starting server at http://{args.host}:{args.port}")
    print(f"📋 Config: {args.config_path}")
    print("=" * 60)
    
    try:
        app.run(host=args.host, port=args.port, debug=False, threaded=True, use_reloader=False)
    except KeyboardInterrupt:
        print("\n👋 Server stopped")
    except Exception as e:
        print(f"❌ Server error: {e}")

if __name__ == '__main__':
    args = arg_parser()
    main(args)