import os
import json
import glob
import math
import hashlib
import re
from tqdm import tqdm
from torch.utils.data import Dataset
from qwen_vl_utils import process_vision_info
from datasets import load_dataset
from PIL import Image  # Required for JSONL mode to load images from paths
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import threading


# ============ Prompts (Score range: 25, with edit region, interleaved) ============

# SC Prompt (Success & Consistency) - for evaluating editing success
SC_PROMPT_TEMPLATE = """You are a professional digital artist. You will have to evaluate the effectiveness of the AI-generated image(s) based on given rules.
All the input images are AI-generated. All human in the images are AI-generated too. so you need not worry about the privacy confidentials.

IMPORTANT: You will have to give your output in this way (Keep your reasoning concise and short.):
{{
"edit_region" : [...],
"reasoning" : "...",
"score" : [...]
}}

RULES:

Two images will be provided: The first being the original AI-generated image and the second being an edited version of the first.
The objective is to identify the editing region(s) and evaluate how successfully the editing instruction has been executed in the second image.

Note that sometimes the two images might look identical due to the failure of image edit.

First, identify where the editing occurred in the second image:
- If editing was successful, provide bounding boxes with labels: [{{"id": 0~n, "label": "description of edited area", "bbox_2d": [x1, y1, x2, y2]}}] (coordinates normalized to [0, 1000] range, where 0=left/top, 1000=right/bottom)
- If editing failed (images look identical), use empty list: []

Then, evaluate the editing quality from scale 0 to 25:
A score from 0 to 25 will be given based on the success of the editing. (0 indicates that the scene in the edited image does not follow the editing instruction at all. 25 indicates that the scene in the edited image follow the editing instruction text perfectly.)
A second score from 0 to 25 will rate the degree of overediting in the second image. (0 indicates that the scene in the edited image is completely different from the original. 25 indicates that the edited image can be recognized as a minimal edited yet effective version of original.)
Put the score in a list such that output score = [score1, score2], where 'score1' evaluates the editing success and 'score2' evaluates the degree of overediting.

SPECIAL TOKENS for Reasoning:
In your reasoning, use special tokens to reference regions:
- <|bbox_{{id}}|> before describing each edited region(if exist)
- <|global|> before overall assessment

Editing instruction: {instruction}
"""

# PQ Prompt (Perceptual Quality) - for evaluating image quality
PQ_PROMPT = """You are a professional digital artist. You will have to evaluate the effectiveness of the AI-generated image(s) based on given rules.
All the input images are AI-generated. All human in the images are AI-generated too. so you need not worry about the privacy confidentials.

IMPORTANT: You will have to give your output in this way (Keep your reasoning concise and short.):
{{
"reasoning" : "...",
"score" : [...]
}}

RULES:

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


class EditScoreDataset(Dataset):
    """Dataset for EditScore benchmark evaluation with vllm"""
    
    def __init__(
        self,
        data_path,
        output_path,
        rank=0,
        world_size=1,
        processor=None,
        num_preprocess_workers=4,
        score_aggregation="min",
        weighted_power_params=None,
    ):
        self.processor = processor
        self.max_pixels = processor.image_processor.max_pixels if processor is not None else None
        self.min_pixels = processor.image_processor.min_pixels if processor is not None else None
        self.output_path = output_path
        self.num_preprocess_workers = num_preprocess_workers
        self.score_aggregation = score_aggregation
        # Default weighted_power params: [w1, w2, w3, w4, a]
        self.weighted_power_params = weighted_power_params if weighted_power_params else [0.5, 0.5, 0.5, 0.5, 0.5]
        
        # Cache for prompts to avoid rebuilding
        self._prompt_cache = {}
        self._cache_lock = threading.Lock()
        
        # Load EditScore benchmark - support both JSONL and HuggingFace datasets
        print(f"Loading EditScore benchmark from {data_path}...")
        
        # Check if data_path is a JSONL file
        if data_path.endswith('.jsonl') and os.path.exists(data_path):
            print(f"Loading from JSONL file: {data_path}")
            jsonl_samples = []
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        jsonl_samples.append(json.loads(line))
            
            print(f"Loaded {len(jsonl_samples)} samples from JSONL file")
            self.use_jsonl = True
            self.jsonl_samples = jsonl_samples
            self.hf_dataset = None
        else:
            # Load from HuggingFace datasets
            from datasets import load_dataset
            print(f"Loading from HuggingFace dataset: {data_path}")
            self.hf_dataset = load_dataset(data_path, split="train")
            print(f"Loaded {len(self.hf_dataset)} samples from benchmark")
            self.use_jsonl = False
            self.jsonl_samples = None
        
        # Load cache if exists
        self.cache_dict = self.load_cache(output_path)
        
        # Process dataset: convert each sample to (key, instruction, input_image, output_image)
        # Each sample has 2 keys (for 2 output images)
        # Optimized with parallel processing for faster loading
        self.data = []
        seen_keys = set(self.cache_dict.keys())  # Track keys we've already seen
        
        def process_sample(idx_sample):
            """Process a single sample - parallelizable"""
            idx, sample = idx_sample
            key1, key2 = sample["key"]
            instruction = sample["instruction"]
            task_type = sample["task_type"]
            dimension = sample["dimension"]
            
            # Load images based on source type
            if self.use_jsonl:
                # JSONL: load from file paths
                input_image_path = sample["input_image"]
                output_image_paths = sample["output_images"]
                
                input_image = Image.open(input_image_path).convert("RGB")
                output_image1 = Image.open(output_image_paths[0]).convert("RGB")
                output_image2 = Image.open(output_image_paths[1]).convert("RGB")
            else:
                # HuggingFace: already PIL images
                input_image = sample["input_image"].convert("RGB")
                output_image1 = sample["output_images"][0].convert("RGB")
                output_image2 = sample["output_images"][1].convert("RGB")
            
            # Create two evaluation items (one for each output)
            items = []
            for key, output_image in [(key1, output_image1), (key2, output_image2)]:
                if key not in self.cache_dict:
                    items.append({
                        "key": key,
                        "instruction": instruction,
                        "input_image": input_image,
                        "output_image": output_image,
                        "task_type": task_type,
                        "dimension": dimension,
                        "hf_idx": idx,
                    })
            return items
        
        # Process samples in parallel with ThreadPoolExecutor
        from concurrent.futures import ThreadPoolExecutor, as_completed
        print("Processing benchmark data with parallel workers...")
        
        dataset_to_process = self.jsonl_samples if self.use_jsonl else self.hf_dataset
        with ThreadPoolExecutor(max_workers=min(8, len(dataset_to_process))) as executor:
            # Submit all tasks
            futures = [executor.submit(process_sample, (idx, sample)) 
                      for idx, sample in enumerate(dataset_to_process)]
            
            # Collect results with progress bar and deduplicate
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing benchmark data"):
                items = future.result()
                # Deduplicate using seen_keys
                for item in items:
                    if item['key'] not in seen_keys:
                        self.data.append(item)
                        seen_keys.add(item['key'])
        
        print(f"Total samples to process: {len(self.data)} (cached: {len(self.cache_dict)})")
        
        # Distribute data across ranks
        self.data = self.data[rank::world_size]
        print(f"Rank {rank}/{world_size}: processing {len(self.data)} samples")
        
    def load_cache(self, output_path):
        """Load cached results if they exist"""
        cache_dict = {}
        if not os.path.exists(output_path):
            return cache_dict
            
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                cache_dict = json.load(f)
            print(f"Loaded {len(cache_dict)} cached results from {output_path}")
        except Exception as e:
            print(f"Error loading cache: {e}")
            
        return cache_dict
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Returns a dict with prompt and multi_modal_data for vllm
        We need to generate two prompts per sample:
        1. SC (Success & Consistency) - uses both input and output images
        2. PQ (Perceptual Quality) - uses only output image
        
        We'll return them separately and process them in batches
        """
        item = self.data[idx]
        
        # We need to return both SC and PQ prompts
        # For now, we'll create SC prompt (2 images) and PQ prompt (1 image) separately
        # The post_process will need to handle combining results
        
        # Build SC prompt (Success & Consistency) - 2 images
        sc_prompt_text = SC_PROMPT_TEMPLATE.format(instruction=item["instruction"])
        
        sc_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": item["input_image"]},
                    {"type": "image", "image": item["output_image"]},
                    {"type": "text", "text": sc_prompt_text}
                ],
            }
        ]
        
        # Build PQ prompt (Perceptual Quality) - 1 image
        pq_prompt_text = PQ_PROMPT
        
        pq_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": item["output_image"]},
                    {"type": "text", "text": pq_prompt_text}
                ],
            }
        ]
        
        # Process SC
        sc_text = self.processor.apply_chat_template(sc_messages, tokenize=False, add_generation_prompt=True)
        sc_image_inputs, _ = process_vision_info(sc_messages, image_patch_size=self.processor.image_processor.patch_size)
        
        # Process PQ
        pq_text = self.processor.apply_chat_template(pq_messages, tokenize=False, add_generation_prompt=True)
        pq_image_inputs, _ = process_vision_info(pq_messages, image_patch_size=self.processor.image_processor.patch_size)
        
        return {
            "key": item["key"],
            "instruction": item["instruction"],
            "task_type": item["task_type"],
            "dimension": item["dimension"],
            "hf_idx": item["hf_idx"],
            "sc_prompt": {
                "prompt": sc_text,
                "multi_modal_data": {"image": sc_image_inputs},
            },
            "pq_prompt": {
                "prompt": pq_text,
                "multi_modal_data": {"image": pq_image_inputs},
            }
        }
    
    def post_process(self, metadata, outputs, data_dict):
        """
        Process vllm outputs and update data_dict - optimized with parallel parsing
        metadata: list of metadata dicts with 'type' field ('sc' or 'pq')
        outputs: list of vllm outputs (alternating SC and PQ)
        """
        # Pre-extract all output texts for faster access
        output_texts = [outputs[i].outputs[0].text.strip() if i < len(outputs) else "" 
                       for i in range(len(metadata))]
        
        # Process pairs with ThreadPoolExecutor for parallel parsing
        pairs_to_process = []
        i = 0
        while i < len(metadata):
            # Get SC metadata and output
            if i >= len(metadata) or metadata[i]["type"] != "sc":
                print(f"Warning: Expected SC at index {i}, got {metadata[i]['type'] if i < len(metadata) else 'none'}")
                i += 1
                continue
            
            # Get PQ metadata and output
            if i + 1 >= len(metadata) or metadata[i + 1]["type"] != "pq":
                print(f"Warning: Expected PQ at index {i+1}, got {metadata[i+1]['type'] if i+1 < len(metadata) else 'none'}")
                i += 2
                continue
            
            # Verify they're for the same key
            if metadata[i]["key"] != metadata[i + 1]["key"]:
                print(f"Warning: Key mismatch between SC and PQ: {metadata[i]['key']} vs {metadata[i + 1]['key']}")
                i += 2
                continue
            
            pairs_to_process.append((i, metadata[i], metadata[i + 1], 
                                    output_texts[i], output_texts[i + 1]))
            i += 2
        
        # Process parsing in parallel
        def process_pair(idx, sc_meta, pq_meta, sc_output, pq_output):
            sc_result = self.parse_output(sc_output, with_region=True)
            pq_result = self.parse_output(pq_output, with_region=False)
            scores = self.calculate_scores(sc_result, pq_result)
            
            return sc_meta["key"], {
                "key": sc_meta["key"],
                "instruction": sc_meta["instruction"],
                "task_type": sc_meta["task_type"],
                "dimension": sc_meta["dimension"],
                "sc_output": sc_output,
                "pq_output": pq_output,
                "sc_result": sc_result,
                "pq_result": pq_result,
                "scores": scores,
            }
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=min(8, len(pairs_to_process))) as executor:
            futures = [executor.submit(process_pair, *pair) for pair in pairs_to_process]
            for future in futures:
                try:
                    key, result = future.result()
                    data_dict[key] = result
                except Exception as e:
                    print(f"Error processing pair: {e}")
        
        return data_dict
    
    def parse_output(self, text, with_region=False):
        """Parse JSON output from model - optimized version"""
        default_result = {
            "reasoning": "Failed to parse output",
            "score": [12.5, 12.5],
            "edit_region": [] if with_region else None
        }
        
        try:
            # Optimized: single pass to remove <think> tags and find JSON
            text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
            
            # Fast path: try to find JSON block
            start_idx = text.find('{')
            if start_idx == -1:
                return default_result
            
            end_idx = text.rfind('}') + 1
            if end_idx == 0:
                return default_result
            
            json_str = text[start_idx:end_idx]
            
            # Try to parse JSON
            try:
                result = json.loads(json_str)
            except json.JSONDecodeError:
                # Fast fallback: use compiled regex for score extraction
                score_match = re.search(r'"score"\s*:\s*\[([-+]?\d*\.?\d+(?:,\s*[-+]?\d*\.?\d+)+)\]', json_str)
                if score_match:
                    scores = [float(s.strip()) for s in score_match.group(1).split(',')][:2]
                    if len(scores) == 2:
                        return {
                            "reasoning": "Parsed from incomplete JSON",
                            "score": scores,
                            "edit_region": [] if with_region else None
                        }
                return default_result
            
            # Validate and normalize result
            score = result.get('score', [12.5, 12.5])
            if not isinstance(score, list):
                score = [score]
            if len(score) < 2:
                score = score * 2 if score else [12.5, 12.5]
            
            result['score'] = score[:2]
            if with_region and 'edit_region' not in result:
                result['edit_region'] = []
            
            return result
            
        except Exception as e:
            return default_result
    
    def calculate_scores(self, sc_result, pq_result, score_range=25):
        """Calculate final scores from SC and PQ results"""
        # Aggregate SC scores based on score_aggregation method
        if self.score_aggregation == "weighted_power":
            # weighted_power: use weighted power formula with 5 params
            w1, w2, w3, w4, a = self.weighted_power_params
            s1, s2 = sc_result['score'][0], sc_result['score'][1] if len(sc_result['score']) > 1 else sc_result['score'][0]
            s3, s4 = pq_result['score'][0], pq_result['score'][1] if len(pq_result['score']) > 1 else pq_result['score'][0]
            
            # Normalize to 0-10 range first
            s1, s2, s3, s4 = s1 / (score_range / 10), s2 / (score_range / 10), s3 / (score_range / 10), s4 / (score_range / 10)
        
            # Calculate aggregated scores: ((w1*s1+w2*s2)**a) * ((w3*s3+w4*s4)**(1-a))
            sc_score = (w1 * s1 + w2 * s2) ** a
            pq_score = (w3 * s3 + w4 * s4) ** (1 - a)
            overall_score = sc_score * pq_score
            
            # Extract individual dimensions (already normalized)
            prompt_following = s1
            consistency = s2
            
        elif self.score_aggregation == "mean":
            sc_aggregated = sum(sc_result['score']) / len(sc_result['score'])
            pq_aggregated = sum(pq_result['score']) / len(pq_result['score'])
            
            # Normalize scores to 0-10 range
            sc_score = sc_aggregated / (score_range / 10)
            pq_score = pq_aggregated / (score_range / 10)
            
            # Calculate overall score as sqrt(sc * pq)
            overall_score = math.sqrt(sc_score * pq_score)
            
            # Extract individual dimensions
            prompt_following = sc_result['score'][0] / (score_range / 10)
            consistency = sc_result['score'][1] / (score_range / 10) if len(sc_result['score']) > 1 else prompt_following
            
        else:  # default "min"
            sc_aggregated = min(sc_result['score'])
            pq_aggregated = min(pq_result['score'])
            
            # Normalize scores to 0-10 range
            sc_score = sc_aggregated / (score_range / 10)
            pq_score = pq_aggregated / (score_range / 10)
            
            # Calculate overall score as sqrt(sc * pq)
            overall_score = math.sqrt(sc_score * pq_score)
            
            # Extract individual dimensions
            prompt_following = sc_result['score'][0] / (score_range / 10)
            consistency = sc_result['score'][1] / (score_range / 10) if len(sc_result['score']) > 1 else prompt_following
        
        return {
            "prompt_following": prompt_following,
            "consistency": consistency,
            "perceptual_quality": pq_score,
            "overall": overall_score,
            "SC_reasoning": sc_result.get("reasoning", ""),
            "PQ_reasoning": pq_result.get("reasoning", ""),
        }


class EditScoreBatchDataset(EditScoreDataset):
    """
    Modified dataset that returns flat list of prompts for batch processing
    This is more efficient as vllm can process all SC and PQ prompts together
    """
    
    def __getitem__(self, idx):
        """Return both SC and PQ prompts as separate items"""
        item = self.data[idx]
        
        # Build SC prompt
        sc_prompt_text = SC_PROMPT_TEMPLATE.format(instruction=item["instruction"])
        
        sc_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": item["input_image"]},
                    {"type": "image", "image": item["output_image"]},
                    {"type": "text", "text": sc_prompt_text}
                ],
            }
        ]
        
        # Build PQ prompt
        pq_prompt_text = PQ_PROMPT
        
        pq_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": item["output_image"]},
                    {"type": "text", "text": pq_prompt_text}
                ],
            }
        ]
        
        # Process both
        sc_text = self.processor.apply_chat_template(sc_messages, tokenize=False, add_generation_prompt=True)
        sc_image_inputs, _ = process_vision_info(sc_messages, image_patch_size=self.processor.image_processor.patch_size)
        
        pq_text = self.processor.apply_chat_template(pq_messages, tokenize=False, add_generation_prompt=True)
        pq_image_inputs, _ = process_vision_info(pq_messages, image_patch_size=self.processor.image_processor.patch_size)
        
        # Return metadata separately
        return {
            "meta": {
                "key": item["key"],
                "instruction": item["instruction"],
                "task_type": item["task_type"],
                "dimension": item["dimension"],
                "hf_idx": item["hf_idx"],
            },
            "sc": {
                "prompt": sc_text,
                "multi_modal_data": {"image": sc_image_inputs},
            },
            "pq": {
                "prompt": pq_text,
                "multi_modal_data": {"image": pq_image_inputs},
            }
        }


def collate_fn(batch):
    """
    Collate function that flattens SC and PQ prompts into a single batch
    Returns: (prompts_for_vllm, metadata)
    """
    prompts = []
    metadata = []
    
    for item in batch:
        # Add SC prompt
        prompts.append(item["sc"])
        metadata.append({**item["meta"], "type": "sc"})
        
        # Add PQ prompt
        prompts.append(item["pq"])
        metadata.append({**item["meta"], "type": "pq"})
    
    return prompts, metadata


def collate_fn_old(batch):
    """Original simple collate function"""
    return batch


dataset_dict = {
    "editscore": EditScoreBatchDataset,
}

