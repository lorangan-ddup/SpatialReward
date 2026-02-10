import os
import json
import math
import re
from torch.utils.data import Dataset
from qwen_vl_utils import process_vision_info
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


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


class MultiEditRewardBenchDataset(Dataset):
    """Dataset for MultiEditRewardBench evaluation with vllm
    
    Supports loading multiple pair files (2pair, 3pair, 4pair) together.
    GT is ranked by order in edited_images array (first is best).
    For 3pair/4pair, all rankings must be correct to count as correct.
    """
    
    def __init__(
        self,
        data_path,
        output_path,
        rank=0,
        world_size=1,
        processor=None,
        score_aggregation="min",
        weighted_power_params=None,
    ):
        self.processor = processor
        self.max_pixels = processor.image_processor.max_pixels if processor is not None else None
        self.min_pixels = processor.image_processor.min_pixels if processor is not None else None
        self.output_path = output_path
        self.score_aggregation = score_aggregation
        self.weighted_power_params = weighted_power_params if weighted_power_params else [0.5, 0.5, 0.5, 0.5, 0.5]
        
        print(f"Score aggregation: {score_aggregation}")
        
        # Load benchmark data - support single file or directory with multiple files
        print(f"Loading MultiEditRewardBench from {data_path}...")
        
        raw_data = []
        if os.path.isdir(data_path):
            # Load all pair files from directory
            for pair_file in ["2pair.json", "3pair.json", "4pair.json"]:
                file_path = os.path.join(data_path, pair_file)
                if os.path.exists(file_path):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        file_data = json.load(f)
                        raw_data.extend(file_data)
                        print(f"  Loaded {len(file_data)} samples from {pair_file}")
        else:
            # Single file
            with open(data_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
        
        print(f"Total loaded: {len(raw_data)} samples")
        
        # Load cache if exists
        self.cache_dict = self.load_cache(output_path)
        
        # Process dataset
        self.data = []
        seen_keys = set(self.cache_dict.keys())
        
        def process_sample(sample):
            """Process a single sample"""
            pair_id = sample["pair_id"]
            
            if pair_id in seen_keys:
                return None
            
            instruction = sample["instruction"]
            original_image_path = sample["original_image"]
            edited_images = sample["edited_images"]
            num_images = len(edited_images)
            
            # Detect pair type from pair_id or num_images
            if "2pair" in pair_id:
                pair_type = 2
            elif "3pair" in pair_id:
                pair_type = 3
            elif "4pair" in pair_id:
                pair_type = 4
            else:
                pair_type = num_images
            
            # Load images
            try:
                original_image = Image.open(original_image_path).convert("RGB")
                loaded_edited_images = []
                for ed_info in edited_images:
                    img = Image.open(ed_info["edited_image"]).convert("RGB")
                    loaded_edited_images.append(img)
            except Exception as e:
                print(f"Error loading images for {pair_id}: {e}")
                return None
            
            gt_qualities = [ed["quality"] for ed in edited_images]
            
            return {
                "pair_id": pair_id,
                "pair_type": pair_type,
                "instruction": instruction,
                "original_image": original_image,
                "edited_images": loaded_edited_images,
                "label": sample.get("label", ""),
                "sample_id": sample.get("sample_id", ""),
                "gt_qualities": gt_qualities,
            }
        
        # Process samples in parallel
        print("Processing benchmark data...")
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(process_sample, sample) for sample in raw_data]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Loading data"):
                item = future.result()
                if item is not None:
                    self.data.append(item)
        
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
        """Returns SC and PQ prompts for each edited image"""
        item = self.data[idx]
        
        sc_prompt_text = SC_PROMPT_TEMPLATE.format(instruction=item["instruction"])
        
        pq_prompt_text = PQ_PROMPT
        
        prompts = []
        for i, edited_img in enumerate(item["edited_images"]):
            sc_messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": item["original_image"]},
                        {"type": "image", "image": edited_img},
                        {"type": "text", "text": sc_prompt_text}
                    ],
                }
            ]
            
            pq_messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": edited_img},
                        {"type": "text", "text": pq_prompt_text}
                    ],
                }
            ]
            
            sc_text = self.processor.apply_chat_template(sc_messages, tokenize=False, add_generation_prompt=True)
            sc_image_inputs, _ = process_vision_info(sc_messages, image_patch_size=self.processor.image_processor.patch_size)
            
            pq_text = self.processor.apply_chat_template(pq_messages, tokenize=False, add_generation_prompt=True)
            pq_image_inputs, _ = process_vision_info(pq_messages, image_patch_size=self.processor.image_processor.patch_size)
            
            prompts.append({
                "sc": {"prompt": sc_text, "multi_modal_data": {"image": sc_image_inputs}},
                "pq": {"prompt": pq_text, "multi_modal_data": {"image": pq_image_inputs}},
            })
        
        return {
            "meta": {
                "pair_id": item["pair_id"],
                "pair_type": item["pair_type"],
                "instruction": item["instruction"],
                "label": item["label"],
                "sample_id": item["sample_id"],
                "gt_qualities": item["gt_qualities"],
                "num_images": len(item["edited_images"]),
            },
            "prompts": prompts,
        }
    
    def post_process(self, metadata, outputs, data_dict):
        """Process vllm outputs and update data_dict"""
        output_texts = [outputs[i].outputs[0].text.strip() if i < len(outputs) else "" 
                       for i in range(len(metadata))]
        
        # Group by pair_id
        pair_results = {}
        for i, meta in enumerate(metadata):
            pair_id = meta["pair_id"]
            if pair_id not in pair_results:
                pair_results[pair_id] = {
                    "pair_id": pair_id,
                    "pair_type": meta["pair_type"],
                    "instruction": meta["instruction"],
                    "label": meta["label"],
                    "sample_id": meta["sample_id"],
                    "gt_qualities": meta["gt_qualities"],
                    "num_images": meta["num_images"],
                    "img_data": [{} for _ in range(meta["num_images"])],
                }
            
            img_idx = meta["img_idx"]
            prompt_type = meta["prompt_type"]
            result = self._parse_output(output_texts[i], with_region=True if prompt_type == 'sc' else False)
            
            pair_results[pair_id]["img_data"][img_idx][f"{prompt_type}_output"] = output_texts[i]
            pair_results[pair_id]["img_data"][img_idx][f"{prompt_type}_result"] = result
        
        # Calculate scores and ranking
        for pair_id, result in pair_results.items():
            scores = []
            for img_data in result["img_data"]:
                sc_result = img_data.get("sc_result", {"score": [12.5, 12.5]})
                pq_result = img_data.get("pq_result", {"score": [12.5, 12.5]})
                overall_score = self._calculate_score(sc_result, pq_result)
                img_data["overall_score"] = overall_score
                scores.append(overall_score)
            
            result["scores"] = scores
            
            indexed_scores = list(enumerate(scores))
            sorted_scores = sorted(indexed_scores, key=lambda x: x[1], reverse=True)
            pred_rank = [x[0] for x in sorted_scores]
            
            gt_rank = list(range(len(scores)))
            
            result["pred_rank"] = pred_rank
            result["gt_rank"] = gt_rank
            result["is_correct"] = (pred_rank == gt_rank)
            
            data_dict[pair_id] = result
        
        return data_dict
    
    def _calculate_score(self, sc_result, pq_result, score_range=25):
        """Calculate overall score based on aggregation method"""
        if self.score_aggregation == "weighted_power":
            w1, w2, w3, w4, a = self.weighted_power_params
            s1 = sc_result['score'][0] if len(sc_result['score']) > 0 else 12.5
            s2 = sc_result['score'][1] if len(sc_result['score']) > 1 else s1
            s3 = pq_result['score'][0] if len(pq_result['score']) > 0 else 12.5
            s4 = pq_result['score'][1] if len(pq_result['score']) > 1 else s3
            
            s1, s2, s3, s4 = s1 / (score_range / 10), s2 / (score_range / 10), s3 / (score_range / 10), s4 / (score_range / 10)
            
            sc_score = (w1 * s1 + w2 * s2) ** a
            pq_score = (w3 * s3 + w4 * s4) ** (1 - a)
            return sc_score * pq_score
        
        elif self.score_aggregation == "mean":
            sc_aggregated = sum(sc_result['score']) / len(sc_result['score']) if sc_result['score'] else 12.5
            pq_aggregated = sum(pq_result['score']) / len(pq_result['score']) if pq_result['score'] else 12.5
            
            sc_score = sc_aggregated / (score_range / 10)
            pq_score = pq_aggregated / (score_range / 10)
            return math.sqrt(sc_score * pq_score)
        
        else:  # "min"
            sc_aggregated = min(sc_result['score']) if sc_result['score'] else 12.5
            pq_aggregated = min(pq_result['score']) if pq_result['score'] else 12.5
            
            sc_score = sc_aggregated / (score_range / 10)
            pq_score = pq_aggregated / (score_range / 10)
            return math.sqrt(sc_score * pq_score)
    
    def _parse_output(self, text, with_region=False):
        """Parse JSON output from model"""
        default_result = {
            "reasoning": "Failed to parse output",
            "score": [12.5, 12.5],
            "edit_region": [] if with_region else None
        }
        
        try:
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
            
            score = result.get('score', [12.5, 12.5])
            if not isinstance(score, list):
                score = [score]
            if len(score) < 2:
                score = score * 2 if score else [12.5, 12.5]
            
            result['score'] = score[:2]
            if with_region and 'edit_region' not in result:
                result['edit_region'] = []
            
            return result
        
        except Exception:
            return default_result


def collate_fn(batch):
    """Collate function that flattens SC and PQ prompts into a batch"""
    prompts = []
    metadata = []
    
    for item in batch:
        for i, prompt_pair in enumerate(item["prompts"]):
            prompts.append(prompt_pair["sc"])
            metadata.append({**item["meta"], "img_idx": i, "prompt_type": "sc"})
            
            prompts.append(prompt_pair["pq"])
            metadata.append({**item["meta"], "img_idx": i, "prompt_type": "pq"})
    
    return prompts, metadata


dataset_dict = {
    "multiedit": MultiEditRewardBenchDataset,
}
