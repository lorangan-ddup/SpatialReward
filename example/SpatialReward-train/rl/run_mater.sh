#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MAX_PIXELS=401408 \
NPROC_PER_NODE=8 \
NNODES=4 \
NODE_RANK=0 \
MASTER_ADDR="127.0.0.1" \
MASTER_PORT=29500 \
swift rlhf \
    --rlhf_type grpo \
    --model /path/to/your/checkpoint/spatial_reward \
    --model_type ${MODEL_TYPE:-qwen3_vl} \
    --reward_funcs gemini_consistency \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_max_model_len 5120 \
    --vllm_tensor_parallel_size 1 \
    --vllm_limit_mm_per_prompt '{"image": 6, "video": 0}' \
    --vllm_gpu_memory_utilization 0.5 \
    --vllm_enforce_eager true \
    --train_type full \
    --torch_dtype bfloat16 \
    --dataset '/path/to/your/dataset/data.json' \
    --dataset_shuffle true \
    --train_dataloader_shuffle true \
    --load_from_cache_file true \
    --max_completion_length 1024 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --learning_rate 5e-7 \
    --gradient_accumulation_steps 2 \
    --save_strategy 'steps' \
    --eval_strategy 'no' \
    --split_dataset_ratio 0 \
    --eval_steps 40000 \
    --save_steps 50 \
    --save_total_limit 10000 \
    --save_only_model false \
    --logging_steps 1 \
    --output_dir output/spatial_reward \
    --warmup_ratio 0.03 \
    --dataloader_num_workers 8 \
    --num_generations 4 \
    --temperature 0.9 \
    --repetition_penalty 1.05 \
    --deepspeed zero3 \
    --log_completions true \
    --report_to tensorboard \
    --num_iterations 1 \
    --async_generate false \
    --beta 0.02 \
    --top_p 0.9 \
    --enable_jsonl_output True \
    --top_k 50 2>&1 | tee log/node0.log