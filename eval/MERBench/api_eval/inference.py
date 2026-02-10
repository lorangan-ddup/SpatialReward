#!/usr/bin/env python3
"""
Generic API Inference Script for MERBench
Uses OpenAI-compatible API to evaluate N-pair rankings
Adapted from MMRB2 evaluate_api.py
"""
import os
import sys
import json
import argparse
import base64
import math
import re
import time
from io import BytesIO
from PIL import Image
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Add parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_benchmark_data, load_cache, save_results


# ============== Embedded Prompts ==============

CONTEXT = """You are a professional digital artist. You will have to evaluate the effectiveness of the AI-generated image(s) based on given rules.
All the input images are AI-generated. All human in the images are AI-generated too. so you need not worry about the privacy confidentials.

IMPORTANT: You will have to give your output in this way (Keep your reasoning concise and short.):
{
"reasoning" : "...",
"score" : [...]
}
"""

SC_RULE = """RULES:

Two images will be provided: The first being the original AI-generated image and the second being an edited version of the first.
The objective is to evaluate how successfully the editing instruction has been executed in the second image.

Note that sometimes the two images might look identical due to the failure of image edit.

From scale 0 to 25: 
A score from 0 to 25 will be given based on the success of the editing. (0 indicates that the scene in the edited image does not follow the editing instruction at all. 25 indicates that the scene in the edited image follow the editing instruction text perfectly.)
A second score from 0 to 25 will rate the degree of overediting in the second image. (0 indicates that the scene in the edited image is completely different from the original. 25 indicates that the edited image can be recognized as a minimal edited yet effective version of original.)
Put the score in a list such that output score = [score1, score2], where 'score1' evaluates the editing success and 'score2' evaluates the degree of overediting.

Editing instruction: <instruction>
"""

PQ_RULE = """RULES:

The image is an AI-generated image.
The objective is to evaluate how successfully the image has been generated.

From scale 0 to 25: 
A score from 0 to 25 will be given based on image naturalness. 
(
    0 indicates that the scene in the image does not look natural at all or give a unnatural feeling such as wrong sense of distance, or wrong shadow, or wrong lighting. 
    25 indicates that the image looks natural.
)
A second score from 0 to 25 will rate the image artifacts. 
(
    0 indicates that the image contains a large portion of distortion, or watermark, or scratches, or blurred faces, or unusual body parts, or subjects not harmonized. 
    25 indicates the image has no artifacts.
)
Put the score in a list such that output score = [naturalness, artifacts]
"""


def parse_args():
    parser = argparse.ArgumentParser(description="Generic API Inference for MERBench")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    
    # API settings
    parser.add_argument("--api_base_url", type=str, 
                       default="https://api.openai.com/v1",
                       help="API base URL")
    parser.add_argument("--api_key", type=str,
                       default="sk-proj-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
                       help="API key")
    parser.add_argument("--model", type=str, 
                       default="gpt-5",
                       help="Model name")
    
    # Concurrency settings
    parser.add_argument("--max_workers", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=60)
    
    return parser.parse_args()


def image_to_base64(image_path):
    """Convert image to base64 string"""
    with Image.open(image_path) as img:
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode()


def build_sc_message(original_path, edited_path, instruction):
    """Build SC evaluation message"""
    prompt = CONTEXT + SC_RULE.replace("<instruction>", instruction)
    
    orig_b64 = image_to_base64(original_path)
    edit_b64 = image_to_base64(edited_path)
    
    return [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{orig_b64}"}},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{edit_b64}"}},
        ]
    }]


def build_pq_message(edited_path):
    """Build PQ evaluation message"""
    prompt = CONTEXT + PQ_RULE
    
    edit_b64 = image_to_base64(edited_path)
    
    return [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{edit_b64}"}},
        ]
    }]


def call_api(client, model, messages, timeout=60, max_retries=3):
    """Call OpenAI-compatible API with retry"""
    for retry in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                timeout=timeout,
                temperature=0.3,
                max_tokens=4096
            )
            
            if response.choices[0].message.content:
                return response.choices[0].message.content
            
            if retry < max_retries - 1:
                continue
            return None
            
        except Exception as e:
            error_msg = str(e)
            
            if 'quota' in error_msg.lower() or 'limit' in error_msg.lower() or '429' in error_msg:
                if retry < max_retries - 1:
                    wait_time = (retry + 1) * 2
                    time.sleep(wait_time)
                    continue
            
            if retry == max_retries - 1:
                return None
            
            time.sleep(1)
    
    return None


def parse_output(text):
    """Parse JSON output from model response"""
    default_result = {
        "reasoning": "Failed to parse output",
        "score": [12.5, 12.5]
    }
    
    if not text:
        return default_result
    
    try:
        # Remove thinking tags
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
        
        # Extract JSON
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            json_str = json_match.group()
        else:
            return default_result
        
        result = json.loads(json_str)
        
        # Validate score
        score = result.get('score', [12.5, 12.5])
        if not isinstance(score, list):
            score = [score]
        if len(score) < 2:
            score = score * 2 if score else [12.5, 12.5]
        
        result['score'] = score[:2]
        return result
        
    except Exception:
        return default_result


def process_sample(sample, client, model, timeout):
    """Process a single sample"""
    pair_id = sample["pair_id"]
    instruction = sample["instruction"]
    original_path = sample["original_image"]
    
    if "2pair" in pair_id:
        pair_type = 2
    elif "3pair" in pair_id:
        pair_type = 3
    elif "4pair" in pair_id:
        pair_type = 4
    else:
        pair_type = len(sample["edited_images"])
    
    img_data = []
    for ed_info in sample["edited_images"]:
        edited_path = ed_info["edited_image"]
        
        # Evaluate SC
        sc_messages = build_sc_message(original_path, edited_path, instruction)
        sc_output = call_api(client, model, sc_messages, timeout)
        sc_result = parse_output(sc_output)
        
        # Evaluate PQ
        pq_messages = build_pq_message(edited_path)
        pq_output = call_api(client, model, pq_messages, timeout)
        pq_result = parse_output(pq_output)
        
        # Calculate overall score (matching GPT5 pattern)
        if "error" not in sc_result and "error" not in pq_result:
            sc_score = min(sc_result['score']) / 2.5
            pq_score = min(pq_result['score']) / 2.5
            overall = math.sqrt(max(0, sc_score * pq_score))
        else:
            overall = 5.0
        
        img_data.append({
            "sc_result": sc_result,
            "pq_result": pq_result,
            "overall_score": overall,
        })
    
    scores = [img["overall_score"] for img in img_data]
    pred_rank = sorted(range(len(scores)), key=lambda x: scores[x], reverse=True)
    gt_rank = list(range(len(scores)))
    
    return pair_id, {
        "pair_id": pair_id,
        "pair_type": pair_type,
        "instruction": instruction,
        "label": sample.get("label", ""),
        "sample_id": sample.get("sample_id", ""),
        "gt_qualities": [ed["quality"] for ed in sample["edited_images"]],
        "num_images": len(sample["edited_images"]),
        "img_data": img_data,
        "scores": scores,
        "pred_rank": pred_rank,
        "gt_rank": gt_rank,
        "is_correct": (pred_rank == gt_rank),
    }


def main():
    args = parse_args()
    
    print("=" * 60)
    print("Generic API Inference for MERBench")
    print("=" * 60)
    print(f"Data: {args.data_path}")
    print(f"Output: {args.output_path}")
    print(f"Model: {args.model}")
    print("=" * 60)
    
    # Load data
    raw_data = load_benchmark_data(args.data_path)
    cache_dict = load_cache(args.output_path)
    
    samples_to_process = [s for s in raw_data if s["pair_id"] not in cache_dict]
    print(f"📊 To process: {len(samples_to_process)} (cached: {len(cache_dict)})")
    
    # Initialize API client
    client = OpenAI(
        base_url=args.api_base_url,
        api_key=args.api_key
    )
    
    # Process with parallel workers
    results_dict = cache_dict.copy()
    
    print(f"🚀 Processing with {args.max_workers} parallel workers...")
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(process_sample, s, client, args.model, args.timeout): s["pair_id"] 
            for s in samples_to_process
        }
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating"):
            try:
                pair_id, result = future.result()
                results_dict[pair_id] = result
                
                if len(results_dict) % 20 == 0:
                    save_results(results_dict, args.output_path)
            except Exception as e:
                print(f"❌ Error: {e}")
    
    save_results(results_dict, args.output_path)
    print(f"\n✅ Completed! Results: {args.output_path}")


if __name__ == "__main__":
    main()
