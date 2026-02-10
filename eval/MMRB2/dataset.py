import os
import json
import math
import re
from tqdm import tqdm
from torch.utils.data import Dataset
from qwen_vl_utils import process_vision_info
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


# ============ Prompts (Score range: 25, with edit region, interleaved for single-image) ============

# SC Prompt for single-image editing
SC_PROMPT_SINGLE_IMAGE = """You are a professional digital artist. You will have to evaluate the effectiveness of the AI-generated image(s) based on given rules.
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

# SC Prompt for multi-image editing/fusion
SC_PROMPT_MULTI_IMAGE = """You are a professional digital artist. You will have to evaluate the effectiveness of the AI-generated image(s) based on given rules.
All the input images are AI-generated. All human in the images are AI-generated too. so you need not worry about the privacy confidentials.

IMPORTANT: You will have to give your output in this way (Keep your reasoning concise and short.):
{{
"edit_region" : [...],
"reasoning" : "...",
"score" : [...]
}}

RULES:

Multiple input images and one edited output image will be provided. The first {num_input_images} images are the original input images used for editing, and the last image is the edited/fused result.
The objective is to identify which elements from the input images are successfully integrated and evaluate how successfully the editing instruction has been executed.

Note that sometimes the edited image might not successfully integrate all input elements due to the failure of image editing/fusion.

First, identify the successfully integrated elements in the edited image:
- For each element from input images that appears in the edited image, provide bounding boxes: [{{"id": 0~n, "label": "brief description", "bbox_2d": [x1, y1, x2, y2]}}] (coordinates normalized to [0, 1000] range, where 0=left/top, 1000=right/bottom)
- If fusion failed (no integration), use empty list: []

Then, evaluate the editing quality from scale 0 to 25:
A score from 0 to 25 will be given based on the success of the editing. (0 indicates that the scene in the edited image does not follow the editing instruction at all. 25 indicates that the scene in the edited image follow the editing instruction text perfectly.)
A second score from 0 to 25 will rate the degree of overediting in the edited image. (0 indicates that the scene in the edited image is completely different from the original inputs or has excessive artifacts. 25 indicates that the edited image is a well-balanced fusion of the input images.)
Put the score in a list such that output score = [score1, score2], where 'score1' evaluates the editing success and 'score2' evaluates the fusion quality.

SPECIAL TOKENS for Reasoning:
In your reasoning, use special token <|bbox_{{id}}|> before describing each edited region to reference it.

Editing instruction: {instruction}
"""

# PQ Prompt (Perceptual Quality) - same for both single and multi-image
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


class MMRB2EditDataset(Dataset):
    """Dataset for MMRB2 Image Editing benchmark evaluation with vllm
    
    MMRB2 format: Each sample has:
    - prompt_content: [(type, content), ...] - instruction + input images
    - response_a: {model_name, response_content: [(type, content), ...]}
    - response_b: {model_name, response_content: [(type, content), ...]}
    - chosen: "A" or "B"
    
    We create 2 evaluation items per sample:
    - Item A: input_image + instruction + output_image_a
    - Item B: input_image + instruction + output_image_b
    
    Then compare scores to determine which output is better.
    """
    
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
        single_image_only=False,
    ):
        self.processor = processor
        self.max_pixels = processor.image_processor.max_pixels if processor is not None else None
        self.min_pixels = processor.image_processor.min_pixels if processor is not None else None
        self.output_path = output_path
        self.num_preprocess_workers = num_preprocess_workers
        self.score_aggregation = score_aggregation
        # Default weighted_power params: [w1, w2, w3, w4, a]
        self.weighted_power_params = weighted_power_params if weighted_power_params else [0.5, 0.5, 0.5, 0.5, 0.5]
        self.single_image_only = single_image_only
        
        # Cache for prompts to avoid rebuilding
        self._prompt_cache = {}
        self._cache_lock = threading.Lock()
        
        # Load MMRB2 data - support both HuggingFace dataset and JSON file
        print(f"Loading MMRB2 edit benchmark from {data_path}...")
        
        # Check if data_path is a HuggingFace dataset (format: "dataset_name" or "org/dataset_name")
        if not os.path.exists(data_path) and '/' in data_path:
            # Assume it's a HuggingFace dataset
            from datasets import load_dataset
            print(f"Loading from HuggingFace dataset: {data_path}")
            hf_dataset = load_dataset(data_path, 'edit', split='test')
            self.benchmark_pairs = list(hf_dataset)
            print(f"Loaded {len(self.benchmark_pairs)} pairs from HuggingFace dataset")
            # For HF datasets, images are PIL.Image objects already loaded
            self.base_dir = None
            self.use_hf_dataset = True
        elif os.path.isfile(data_path):
            # Load from JSON file
            print(f"Loading from JSON file: {data_path}")
            with open(data_path, 'r', encoding='utf-8') as f:
                mmrb2_data = json.load(f)
            self.benchmark_pairs = mmrb2_data["pairs"]
            print(f"Loaded {len(self.benchmark_pairs)} pairs from JSON file")
            # Determine base directory for loading images
            self.base_dir = os.path.dirname(data_path)
            self.use_hf_dataset = False
        else:
            raise ValueError(f"Invalid data_path: {data_path}. Must be either a HuggingFace dataset name or path to edit.json")

        
        # Load cache if exists
        self.cache_dict = self.load_cache(output_path)
        
        # Process dataset: create evaluation items for both A and B outputs
        self.data = []
        
        def process_pair(pair):
            """Process a single pair - parallelizable"""
            items = []
            pair_id = pair["id"]
            
            # Extract instruction and input images from prompt_content
            instruction = None
            input_images = []
            
            for content_type, content_value in pair["prompt_content"]:
                if content_type == "text":
                    instruction = content_value
                elif content_type == "image":
                    # Handle both HF datasets (PIL images) and JSON files (paths)
                    if self.use_hf_dataset:
                        # HF dataset: content_value is already a PIL Image
                        if hasattr(content_value, 'convert'):  # Check if it's a PIL Image
                            input_images.append(content_value.convert("RGB"))
                    else:
                        # JSON file: content_value is a path
                        img_path = os.path.join(self.base_dir, content_value)
                        if os.path.exists(img_path):
                            input_images.append(Image.open(img_path).convert("RGB"))
            
            if instruction is None:
                print(f"Warning: No instruction found for pair {pair_id}")
                return []
            
            # Check number of input images  
            num_input_images = len(input_images)
            
            # Skip multi-image tasks if single_image_only is enabled
            if self.single_image_only and num_input_images > 1:
                return []  # Skip this pair
            
            # For single-image editing, use the first (and only) input image
            input_image = input_images[0] if input_images else None
            
            if input_image is None:
                print(f"Warning: No input image found for pair {pair_id}")
                return []
            
            # Process response A
            output_a_images = []
            for content_type, content_value in pair["response_a"]["response_content"]:
                if content_type == "image":
                    if self.use_hf_dataset:
                        if hasattr(content_value, 'convert'):
                            output_a_images.append(content_value.convert("RGB"))
                    else:
                        img_path = os.path.join(self.base_dir, content_value)
                        if os.path.exists(img_path):
                            output_a_images.append(Image.open(img_path).convert("RGB"))
            
            # Process response B
            output_b_images = []
            for content_type, content_value in pair["response_b"]["response_content"]:
                if content_type == "image":
                    if self.use_hf_dataset:
                        if hasattr(content_value, 'convert'):
                            output_b_images.append(content_value.convert("RGB"))
                    else:
                        img_path = os.path.join(self.base_dir, content_value)
                        if os.path.exists(img_path):
                            output_b_images.append(Image.open(img_path).convert("RGB"))
            
            # For editing task, we expect 1 output image per response
            output_a = output_a_images[0] if output_a_images else None
            output_b = output_b_images[0] if output_b_images else None
            
            if output_a is None or output_b is None:
                print(f"Warning: Missing output images for pair {pair_id}")
                return []
            
            # Create evaluation items for A and B
            key_a = f"{pair_id}_A"
            key_b = f"{pair_id}_B"
            
            if key_a not in self.cache_dict:
                items.append({
                    "key": key_a,
                    "pair_id": pair_id,
                    "response_type": "A",
                    "instruction": instruction,
                    "input_image": input_image,  # Keep for backward compatibility
                    "input_images": input_images,  # Store all input images
                    "output_image": output_a,
                    "ground_truth": pair["chosen"],
                })
            
            if key_b not in self.cache_dict:
                items.append({
                    "key": key_b,
                    "pair_id": pair_id,
                    "response_type": "B",
                    "instruction": instruction,
                    "input_image": input_image,  # Keep for backward compatibility
                    "input_images": input_images,  # Store all input images
                    "output_image": output_b,
                    "ground_truth": pair["chosen"],
                })
            
            return items
        
        # Process pairs in parallel
        print("Processing MMRB2 data with parallel workers...")
        with ThreadPoolExecutor(max_workers=min(8, len(self.benchmark_pairs))) as executor:
            futures = [executor.submit(process_pair, pair) 
                      for pair in self.benchmark_pairs]
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing pairs"):
                items = future.result()
                self.data.extend(items)
        
        print(f"Total items to process: {len(self.data)} (cached: {len(self.cache_dict)})")
        
        # Report filtering statistics
        if self.single_image_only:
            total_pairs = len(self.benchmark_pairs)
            processed_pairs = len(self.data) // 2  # Each pair creates 2 items (A and B)
            filtered_out = total_pairs - processed_pairs
            print(f"📊 Filtering: Kept {processed_pairs}/{total_pairs} single-image pairs, filtered {filtered_out} multi-image pairs")
        
        # Distribute data across ranks
        self.data = self.data[rank::world_size]
        print(f"Rank {rank}/{world_size}: processing {len(self.data)} items")
    
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
        Returns both SC and PQ prompts for vllm processing
        1. SC (Success & Consistency) - uses all input images + output image
        2. PQ (Perceptual Quality) - uses only output image
        """
        item = self.data[idx]
        
        # Get all input images (for multi-image support)
        input_images = item.get("input_images", [item["input_image"]])
        if not isinstance(input_images, list):
            input_images = [input_images]
        num_input_images = len(input_images)
        
        # Build SC prompt
        if num_input_images > 1:
            # Multi-image editing/fusion
            sc_prompt_text = SC_PROMPT_MULTI_IMAGE.format(
                instruction=item["instruction"],
                num_input_images=num_input_images
            )
        else:
            # Single-image editing (with interleaved reasoning)
            sc_prompt_text = SC_PROMPT_SINGLE_IMAGE.format(instruction=item["instruction"])
        
        # Build SC messages with all images
        sc_content = []
        # Add all input images
        for img in input_images:
            sc_content.append({"type": "image", "image": img})
        # Add output image
        sc_content.append({"type": "image", "image": item["output_image"]})
        # Add text prompt
        sc_content.append({"type": "text", "text": sc_prompt_text})
        
        sc_messages = [
            {
                "role": "user",
                "content": sc_content,
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
                "pair_id": item["pair_id"],
                "response_type": item["response_type"],
                "instruction": item["instruction"],
                "ground_truth": item["ground_truth"],
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
    
    def post_process(self, metadata, outputs, data_dict):
        """
        Process vllm outputs and update data_dict
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
                "pair_id": sc_meta["pair_id"],
                "response_type": sc_meta["response_type"],
                "instruction": sc_meta["instruction"],
                "ground_truth": sc_meta["ground_truth"],
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
        """Parse JSON output from model"""
        default_result = {
            "reasoning": "Failed to parse output",
            "score": [12.5, 12.5],
            "edit_region": [] if with_region else None
        }
        
        try:
            # Remove <think> tags and find JSON block
            text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
            
            start_idx = text.find('{')
            if start_idx == -1:
                return default_result
            
            end_idx = text.rfind('}') + 1
            if end_idx == 0:
                return default_result
            
            json_str = text[start_idx:end_idx]
            
            try:
                result = json.loads(json_str)
            except json.JSONDecodeError:
                # Fallback: extract scores using regex
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


dataset_dict = {
    "mmrb2_edit": MMRB2EditDataset,
}

