import os
import sys
import json
import copy
import hashlib
import time
import threading
import argparse
import random
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

from vllm import LLM, SamplingParams
from torch.utils.data import DataLoader
from transformers import AutoProcessor
from tqdm import tqdm

from dataset import dataset_dict, collate_fn


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--model", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_num_seqs", type=int, default=32)
    parser.add_argument("--limit_mm_per_prompt_image", type=int, default=2)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--max_num_batched_tokens", type=int, default=4096)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--enable_prefix_caching", type=bool, default=True)
    parser.add_argument("--enforce_eager", action="store_true")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--distributed_executor_backend", type=str, default=None)
    parser.add_argument("--dataset_type", type=str, default="editscore")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--min_pixels", type=int, default=56 * 56)
    parser.add_argument("--max_pixels", type=int, default=12845056)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--score_aggregation", type=str, default="min", choices=["min", "mean", "weighted_power"])
    parser.add_argument("--weighted_power_params", type=float, nargs=5, default=None)
    parser.add_argument("--add_timestamp", action="store_true")
    parser.add_argument("--num_pass", type=int, default=1, help="Number of inference passes")
    return parser.parse_args()


def parse_llm_args(args):
    llm_kwargs = {
        "model": args.model,
        "max_num_seqs": args.max_num_seqs,
        "limit_mm_per_prompt": {"image": args.limit_mm_per_prompt_image},
        "tensor_parallel_size": args.tensor_parallel_size,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enable_prefix_caching": args.enable_prefix_caching,
        "enforce_eager": args.enforce_eager
    }
    if args.dtype:
        llm_kwargs["dtype"] = args.dtype
    if args.distributed_executor_backend:
        llm_kwargs["distributed_executor_backend"] = args.distributed_executor_backend
    return llm_kwargs


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def back_envs():
    torchrun_vars = [
        'RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'LOCAL_WORLD_SIZE', 'GROUP_RANK', 
        'ROLE_RANK', 'ROLE_NAME', 'GROUP_WORLD_SIZE', 'ROLE_WORLD_SIZE',
        'MASTER_ADDR', 'MASTER_PORT', 'TORCHELASTIC_RESTART_COUNT', 
        'TORCHELASTIC_MAX_RESTARTS', 'TORCHELASTIC_RUN_ID', 
        'TORCHELASTIC_USE_AGENT_STORE', 'TORCH_NCCL_ASYNC_ERROR_HANDLING'
    ]
    torchrun_vars_bak = {}
    for var in torchrun_vars:
        if var in os.environ:
            torchrun_vars_bak[var] = os.environ[var]
            del os.environ[var]
    return torchrun_vars_bak


def save_data_to_cache(data_dict, cache_path, lock):
    with lock:
        data_copy = copy.deepcopy(data_dict)
    try:
        with open(cache_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(data_copy, ensure_ascii=False, indent=4))
        print(f"Data saved to {cache_path}")
    except Exception as e:
        print(f"Error saving data to {cache_path}: {e}")


def set_cuda_visible_devices(local_rank, tensor_parallel_size, offset=0):
    if os.environ.get('CUDA_VISIBLE_DEVICES', None) is not None:
        offset = int(os.environ['CUDA_VISIBLE_DEVICES'].split(',')[0])
    os.environ['CUDA_VISIBLE_DEVICES'] = ",".join(
        [str(i) for i in range(local_rank * tensor_parallel_size + offset, (local_rank + 1) * tensor_parallel_size + offset)]
    )
    print(f"local rank {local_rank}, CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")


def initialize_vllm(args):
    """Initialize vLLM with checkpoint from environment"""
    llm_kwargs = parse_llm_args(args)
    model = LLM(**llm_kwargs)
    
    # Use temperature from args (set from environment variable in parse_args)
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
    )
    return model, sampling_params


def main():
    args = parse_args()

    rank, locals_rank, world_size = os.getenv('RANK', '0'), os.getenv('LOCAL_RANK', '0'), os.getenv('WORLD_SIZE', '1')
    
    # Loop through multiple inference passes if num_pass > 1
    for pass_id in range(args.num_pass):
        current_seed = args.seed + pass_id
        set_seed(current_seed)
        
        # Determine output path for this pass
        if args.num_pass > 1:
            base_path = args.output_path.replace('.json', '')
            current_output_path = f"{base_path}_pass{pass_id + 1}.json"
            print(f"\n{'='*60}")
            print(f"Starting inference pass {pass_id + 1}/{args.num_pass} (seed={current_seed})")
            print(f"Output: {current_output_path}")
            print(f"{'='*60}\n")
        else:
            current_output_path = args.output_path
            
            # Add timestamp if requested
            if args.add_timestamp:
                # Insert timestamp before .json extension
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                if current_output_path.endswith('.json'):
                    base_path = current_output_path[:-5]  # Remove .json
                    current_output_path = f"{base_path}_{timestamp}.json"
                else:
                    current_output_path = f"{current_output_path}_{timestamp}"
                print(f"\n{'='*60}")
                print(f"Using timestamped output: {current_output_path}")
                print(f"{'='*60}\n")
        
        processor = AutoProcessor.from_pretrained(args.model, max_pixels=args.max_pixels, min_pixels=args.min_pixels)
        dataset = dataset_dict[args.dataset_type](
            args.data_path,
            current_output_path,
            int(rank),
            int(world_size),
            processor=processor,
            score_aggregation=args.score_aggregation,
            weighted_power_params=args.weighted_power_params,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
        )
        
        # Only initialize VLLM once
        if pass_id == 0:
            set_cuda_visible_devices(int(locals_rank), args.tensor_parallel_size)
            torchrun_vars_bak = back_envs()
            llm, sampling_params = initialize_vllm(args)
        
        # Update sampling params with current seed
        sampling_params.seed = current_seed
        
        cache_path = current_output_path + f"_rank{rank}_cache.json"
        data_lock = threading.Lock()
        data_dict = dataset.cache_dict
        save_thread = None
        
        # Optimized: reduce save frequency and use larger batches
        save_interval = 10  # Save every 10 steps instead of 5
        
        for step, batch_data in enumerate(tqdm(dataloader, disable=bool(int(rank)), desc=f"Pass {pass_id+1}/{args.num_pass}")):
            # batch_data is (prompts, metadata) from collate_fn
            prompts, metadata = batch_data
            
            outputs = llm.generate(
                prompts,
                sampling_params=sampling_params,
                use_tqdm=False,
            )
            
            with data_lock:
                data_dict = dataset.post_process(metadata, outputs, data_dict)
            
            # Optimized: less frequent saves
            if step % save_interval == 0 and step > 0:
                if save_thread is not None and save_thread.is_alive():
                    save_thread.join()
                
                save_thread = threading.Thread(
                    target=save_data_to_cache,
                    args=(data_dict, cache_path, data_lock)
                )
                save_thread.start()
        
        if save_thread is not None and save_thread.is_alive():
            save_thread.join()
        save_data_to_cache(data_dict, cache_path, data_lock)

        # Optimized: parallel merge of rank results
        if int(rank) == 0:
            print(f"\nMerging results from {world_size} ranks...")
            tot_data = data_dict
            
            # Use Path for cleaner file operations
            cache_path_obj = Path(current_output_path)
            rank_cache_files = []
            
            for r in range(1, int(world_size)):
                rank_cache_path = str(cache_path_obj) + f"_rank{r}_cache.json"
                if os.path.exists(rank_cache_path):
                    rank_cache_files.append(rank_cache_path)
            
            # Load rank files in parallel if multiple ranks
            if rank_cache_files:
                from concurrent.futures import ThreadPoolExecutor
                def load_rank_file(filepath):
                    with open(filepath, 'r', encoding='utf-8') as f:
                        return json.load(f)
                
                with ThreadPoolExecutor(max_workers=min(8, len(rank_cache_files))) as executor:
                    futures = {executor.submit(load_rank_file, f): f for f in rank_cache_files}
                    for future in futures:
                        rank_data = future.result()
                        tot_data.update(rank_data)
            
            # Save final merged results
            with open(current_output_path, 'w', encoding='utf-8') as f:
                json.dump(tot_data, f, ensure_ascii=False, indent=2)
            
            # Clean up cache files
            for r in range(0, int(world_size)):
                rank_cache_path = str(cache_path_obj) + f"_rank{r}_cache.json"
                if os.path.exists(rank_cache_path):
                    os.remove(rank_cache_path)
            
            print(f"\nPass {pass_id + 1} completed. Results saved to {current_output_path}")
    
    if int(rank) == 0 and args.num_pass > 1:
        print(f"\n{'='*60}")
        print(f"All {args.num_pass} inference passes completed!")
        print(f"Result files:")
        base_path = args.output_path.replace('.json', '')
        for i in range(args.num_pass):
            print(f"  - {base_path}_pass{i + 1}.json")
        print(f"\nTo calculate avg{args.num_pass} statistics, run:")
        print(f"python calculate_statistics.py \\")
        print(f"  --result_files {base_path}_pass{{1..{args.num_pass}}}.json \\")
        print(f"  --avg_n {args.num_pass} \\")
        print(f"  --benchmark_dir {args.data_path}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()

