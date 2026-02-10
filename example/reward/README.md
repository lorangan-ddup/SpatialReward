# SpatialReward Server - API Documentation

> **Note**: throughout this documentation, `<SERVER_IP>` is used as a placeholder. Replace it with your actual server IP.

## 1. Overview

The Spatial Reward Server provides image editing quality evaluation based on Qwen3VL. It uses a **Proxy-Worker** architecture where a central proxy distributes requests to multiple GPU workers based on the editing instructions.

### Architecture

- **Client**: Sends batch requests.
- **Proxy**: Load balances and groups requests by `instruction`.
- **Workers**: Parallel inference on multiple GPUs.

## 2. Server Setup & Quick Start

### Start Server

On your server machine:

```bash
cd /path/to/reward_server_spatial_reward

# 1. Start worker servers (one per GPU, typically 8)
bash start_servers.sh

# 2. Start proxy server
bash start_proxy.sh
```

### Worker Allocation & Instruction Grouping

The proxy server intelligently routes requests:
- Requests with the **same instruction** are grouped together.
- These groups are sent to the same worker to maximize **KV cache** usage and inference throughput.
- **Tip**: When sending batch requests, sorting samples by instruction can significantly improve performance.

## 3. Client Usage

We provide a sample client implementation in `client/reward_client_edit.py`. This client is adapted from the [EditScore repository](https://github.com/VectorSpaceLab/EditScore) and handles communication with the SpatialReward server.

### Reproducing OmniGen2 RL

To reproduce the Online RL training for OmniGen2 using SpatialReward:

1.  Clone the **EditScore** repository:
    ```bash
    git clone https://github.com/VectorSpaceLab/EditScore
    ```
2.  Navigate to `examples/OmniGen2-RL/`.
3.  **Replace** the existing `reward_server` directory with this `SpatialReward` server code.
4.  Follow the training instructions in the EditScore repository.

### Adapting to Other Frameworks

If you want to integrate SpatialReward into other RL frameworks, please refer to `client/reward_client_edit.py`. This script demonstrates:
- How to format requests (image serialization, metadata).
- How to handle server responses (parsing scores, rewards, and reasoning).
- Error handling and retries.


### Client Initialization

The client supports configurable timeout and retry logic:

```python
client = RewardClient(
    proxy_host="<SERVER_IP>",  # Server IP
    proxy_port=23456,          # Proxy port
    timeout=300,               # Request timeout (seconds)
    max_retries=3              # Number of retries
)
```

**Key Implementation Reference:**
See `client/reward_client_edit.py` for the `RewardClient` class and `evaluate` method.

## 4. API Specification

### Endpoint
`POST http://{proxy_host}:{proxy_port}/`

### Request Body (Pickled Dictionary)

```python
{
    "input_images": List[List[PIL.Image.Image]],  # Original images
    "output_image": List[PIL.Image.Image],        # Edited images
    "meta_datas": List[Dict[str, Any]],           # Metadata (Must include 'instruction')
    "server_type": str                            # Default: "vlm"
}
```

#### Field Details

1.  **`input_images`**: A list of lists, where each inner list contains the source images for a single sample (typically just one image).
2.  **`output_image`**: A list of edited images, one per sample.
3.  **`meta_datas`**: A list of dictionaries. **Required**: `instruction` key. Optional fields like `tag` or `image_id` are preserved in the response.
4.  **`server_type`**: Identifier for the backend model (default: "vlm").

**Important**: The `instruction` field in `meta_datas` is mandatory for request routing.

### Response Body

```python
{
    "scores": List[float],        # Binary scores (0/1)
    "rewards": List[float],       # Continuous rewards [0, 1]
    "reasoning": List[str],       # Detailed reasoning text
    "meta_data": List[Dict]       # Original metadata + server info
    "strict_rewards": List[float] # Alias for rewards
}
```

#### Field Details

1.  **`scores`**: Binary decision (1.0 if reward ≥ threshold, else 0.0).
2.  **`rewards`**: The continuous reward value [0, 1], calculated from SC and PQ scores.
3.  **`reasoning`**: Contains the breakdown of scores (SC, PQ) and the VLM's textual reasoning.
4.  **`meta_data`**: The original metadata with an added `original_index` field.

## 5. Scoring Details

The reward is a weighted combination of **Semantic Correctness (SC)** and **Perceptual Quality (PQ)**:

-   **SC (Semantic Correctness)**:
    -   Instruction Following (0-25)
    -   Consistency (0-25)
-   **PQ (Perceptual Quality)**:
    -   Naturalness (0-25)
    -   Artifact-free (0-25)

**Calculation (Default Weights)**:

```python
SC_score = (0.6 * score1 + 0.4 * score2) / 2.5
PQ_score = (0.5 * naturalness + 0.5 * artifacts) / 2.5
O_score = SC_score^0.8 * PQ_score^0.2
reward = O_score / 10  # Normalized to [0, 1]
```

## 6. Troubleshooting

- **Server Won't Start**: Check `nvidia-smi` and ensure ports are free. Kill old processes with `pkill -9 -f reward_server`.
- **Connection Error**: Verify `<SERVER_IP>` and firewall settings. Use `curl http://<SERVER_IP>:23456/ping` to test.
- **Slow Performance**: Check if instructions are being grouped effective. Randomly shuffling instructions in a batch reduces cache hits.
