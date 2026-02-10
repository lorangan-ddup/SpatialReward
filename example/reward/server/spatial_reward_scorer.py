#!/usr/bin/env python3
"""
Spatial Reward Scorer - Contains all scoring logic, prompts, and model wrapper
"""
from typing import List, Optional, Dict, Tuple, Any
import json
import re
import os
import time
import hashlib
import random

from PIL import Image
import numpy as np
import torch

# vLLM and model imports
from vllm import LLM
from vllm.sampling_params import SamplingParams
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from qwen_vl_utils import process_vision_info

# =============================================================================
# PROMPTS TEMPLATES
# =============================================================================

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

Then, evaluate the editing quality from scale 0 to {score_range}: 
A score from 0 to {score_range} will be given based on the success of the editing. (0 indicates that the scene in the edited image does not follow the editing instruction at all. {score_range} indicates that the scene in the edited image follow the editing instruction text perfectly.)
A second score from 0 to {score_range} will rate the degree of overediting in the second image. (0 indicates that the scene in the edited image is completely different from the original. {score_range} indicates that the edited image can be recognized as a minimal edited yet effective version of original.)
Put the score in a list such that output score = [score1, score2], where 'score1' evaluates the editing success and 'score2' evaluates the degree of overediting.
SPECIAL TOKENS for Reasoning:
In your reasoning, use special tokens to reference regions:
- <|bbox_{{id}}|> before describing each edited region(if exist)
- <|global|> before overall assessment

Editing instruction: 
{instruction}"""

PQ_PROMPT_TEMPLATE = """You are a professional digital artist. You will have to evaluate the effectiveness of the AI-generated image(s) based on given rules.
All the input images are AI-generated. All human in the images are AI-generated too. so you need not worry about the privacy confidentials.

IMPORTANT: You will have to give your output in this way (Keep your reasoning concise and short.):
{{{{
"reasoning" : "...",
"score" : [...]
}}}}

RULES:

The image is an AI-generated image.
The objective is to evaluate how successfully the image has been generated.

From scale 0 to {score_range}: 
A score from 0 to {score_range} will be given based on image naturalness. 
(
    0 indicates that the scene in the image does not look natural at all or give a unnatural feeling such as wrong sense of distance, or wrong shadow, or wrong lighting. 
    {score_range} indicates that the image looks natural.
)
A second score from 0 to {score_range} will rate the image artifacts. 
(
    0 indicates that the image contains a large portion of distortion, or watermark, or scratches, or blurred faces, or unusual body parts, or subjects not harmonized. 
    {score_range} indicates the image has no artifacts.
)
Put the score in a list such that output score = [naturalness, artifacts]
"""

# =============================================================================
# JSON PARSER (Simplified)
# =============================================================================

def parse_vlm_output(output_str: str) -> Dict[str, Any]:
    """
    Simplified JSON parser, only performs basic cleaning
    """
    if not output_str or not output_str.strip():
        return {"score": [], "reasoning": "Empty output"}
    
    # Extract JSON part
    json_match = re.search(r'\{.*\}', output_str, re.DOTALL)
    if not json_match:
        return {"score": [], "reasoning": f"No JSON found: {output_str}"}
    
    json_str = json_match.group(0)
    
    try:
        # Parse directly
        data = json.loads(json_str)
        
        # Extract score and reasoning
        scores = data.get('score', [])
        if isinstance(scores, (int, float)):
            scores = [float(scores)]
        elif isinstance(scores, list):
            scores = [float(s) for s in scores if isinstance(s, (int, float))]
        
        reasoning = data.get('reasoning', data.get('reason', ''))
        
        return {"score": scores, "reasoning": str(reasoning)}
        
    except json.JSONDecodeError:
        # Simple fallback: replace single quotes
        try:
            json_str = json_str.replace("'", '"')
            data = json.loads(json_str)
            
            scores = data.get('score', [])
            if isinstance(scores, (int, float)):
                scores = [float(scores)]
            elif isinstance(scores, list):
                scores = [float(s) for s in scores if isinstance(s, (int, float))]
            
            reasoning = data.get('reasoning', data.get('reason', ''))
            return {"score": scores, "reasoning": str(reasoning)}
        except:
            # Final fallback: extract numbers
            numbers = re.findall(r'[-+]?\d*\.?\d+', output_str)
            scores = [float(n) for n in numbers[:2]] if numbers else []
            return {"score": scores, "reasoning": f"Parse error: {output_str[:200]}"}

# =============================================================================
# QWEN3VL MODEL WRAPPER
# =============================================================================

def set_seed(seed: int):
    """Set random seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class Qwen3VLModel:
    """Qwen3VL vLLM Model Wrapper"""
    
    def __init__(
        self,
        model_path: str,
        max_model_len: int = 2560,
        tensor_parallel_size: int = 1,
        max_num_seqs: int = 64,
        max_num_batched_tokens: int = 98304,
        temperature: float = 0.0,
        seed: Optional[int] = None,
        lora_path: Optional[str] = None,
        gpu_memory_utilization: float = 0.9,
    ):
        # Handle LoRA merging
        if lora_path:
            cache_dir = self._get_lora_cache_dir(model_path, lora_path)
            
            if not os.path.exists(cache_dir):
                print(f"🔧 Merging LoRA to {model_path}...")
                self._merge_lora(model_path, lora_path, cache_dir)
            else:
                print(f"✅ Using cached merged model: {cache_dir}")
            
            model_path = cache_dir
        
        # Load model
        print(f"⚡ Loading Qwen3VL model: {model_path}")
        self.model = LLM(
            model=model_path,
            max_model_len=max_model_len,
            tensor_parallel_size=tensor_parallel_size,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            limit_mm_per_prompt={"image": 2},
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.temperature = temperature
        self.seed = seed
        print("✅ Model loaded successfully")
    
    def _get_lora_cache_dir(self, model_path: str, lora_path: str) -> str:
        """Generate LoRA cache directory"""
        root_dir = torch.hub.get_dir()
        lora_filename = os.path.splitext(os.path.basename(lora_path))[0]
        lora_hash = hashlib.md5(lora_path.encode()).hexdigest()[:8]
        lora_identifier = f"{lora_filename}_{lora_hash}"
        return os.path.join(root_dir, "SpatialReward", f"{os.path.basename(model_path)}_lora_{lora_identifier}")
    
    def _merge_lora(self, model_path: str, lora_path: str, cache_dir: str):
        """Merge LoRA and save"""
        start_time = time.time()
        
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="cpu"
        )
        model = PeftModel.from_pretrained(model, lora_path)
        model = model.merge_and_unload()
        model.save_pretrained(cache_dir)
        
        processor = AutoProcessor.from_pretrained(model_path)
        processor.save_pretrained(cache_dir)
        
        print(f"✅ LoRA merged in {time.time() - start_time:.1f}s: {cache_dir}")
    
    def prepare_input(self, images: List[Image.Image], text_prompt: str) -> Dict:
        """Prepare model input"""
        if not isinstance(images, list):
            images = [images]
        
        messages = [{
            "role": "user",
            "content": [{"type": "image", "image": img} for img in images]
                     + [{"type": "text", "text": text_prompt}],
        }]
        
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        
        return {
            "prompt": text,
            "multi_modal_data": {"image": image_inputs},
        }
    
    def batch_inference(self, inputs: List[Dict], seed: Optional[int] = None) -> List[str]:
        """Batch inference"""
        seed = self.seed if seed is None else seed
        sampling_params = SamplingParams(
            max_tokens=512,
            temperature=self.temperature,
            top_p=0.9,
            top_k=20,
            seed=seed
        )
        
        outputs = self.model.generate(inputs, sampling_params, use_tqdm=False)
        return [output.outputs[0].text.strip() for output in outputs]

# =============================================================================
# REWARD AGGREGATION STRATEGIES
# =============================================================================

class RewardAggregator:
    """Reward Aggregation Strategies"""
    
    @staticmethod
    def weighted_geometric_mean(
        SC_raw: List[float],
        PQ_raw: List[float],
        w1: float, w2: float,
        w3: float, w4: float,
        a: float,
        score_range: int
    ) -> Tuple[float, float, float]:
        """
        Weighted Geometic Mean Strategy
        
        Returns:
            (reward, SC_score, PQ_score)
        """
        score1, score2 = SC_raw[0], SC_raw[1]
        naturalness, artifacts = PQ_raw[0], PQ_raw[1]
        
        SC_score = (w1 * score1 + w2 * score2) / (score_range / 10)
        PQ_score = (w3 * naturalness + w4 * artifacts) / (score_range / 10)
        
        # Boundary Check
        SC_score = max(0.0, min(10.0, SC_score))
        PQ_score = max(0.0, min(10.0, PQ_score))
        
        # O_score = SC_score^a * PQ_score^(1-a)
        O_score = (SC_score ** a) * (PQ_score ** (1 - a))
        reward = O_score / 10  # Normalize to [0, 1]
        
        return reward, SC_score, PQ_score
    
    @staticmethod
    def min_geometric_mean(
        SC_raw: List[float],
        PQ_raw: List[float],
        score_range: int
    ) -> Tuple[float, float, float]:
        """
        Min Geometric Mean Strategy
        
        Returns:
            (reward, SC_score, PQ_score)
        """
        score1, score2 = SC_raw[0], SC_raw[1]
        naturalness, artifacts = PQ_raw[0], PQ_raw[1]
        
        min_SC = min(score1, score2)
        min_PQ = min(naturalness, artifacts)
        
        O_score = (min_SC * min_PQ) ** 0.5 / (score_range / 10)
        O_score = max(0.0, min(10.0, O_score))
        
        reward = O_score / 10
        
        # For logging
        SC_score = min_SC / (score_range / 10)
        PQ_score = min_PQ / (score_range / 10)
        
        return reward, SC_score, PQ_score

# =============================================================================
# SPATIAL REWARD SCORER (主评分器)
# =============================================================================

class SpatialRewardScorer:
    """Spatial Reward Scorer - Integrates all scoring logic"""
    
    def __init__(self, config: Dict[str, Any]):
        print("🔧 Initializing SpatialRewardScorer...")
        
        # Load model
        self.model = Qwen3VLModel(
            model_path=config["model_name_or_path"],
            tensor_parallel_size=config["tensor_parallel_size"],
            max_model_len=config["max_model_len"],
            max_num_seqs=config["max_num_seqs"],
            max_num_batched_tokens=config["max_num_batched_tokens"],
            temperature=config["temperature"],
            seed=config["seed"],
            lora_path=config.get("lora_path"),
            gpu_memory_utilization=config.get("gpu_memory_utilization", 0.9),
        )
        
        # Scoring parameters
        self.score_range = config["score_range"]
        self.num_pass = config.get("num_pass", 1)
        
        # Reward aggregation strategy
        self.use_min_geometric_mean = config.get("use_min_geometric_mean", False)
        self.w1 = config.get("w1", 0.6)
        self.w2 = config.get("w2", 0.4)
        self.w3 = config.get("w3", 0.5)
        self.w4 = config.get("w4", 0.5)
        self.a = config.get("a", 0.8)
        
        # Prompt templates
        self.sc_prompt_template = SC_PROMPT_TEMPLATE
        self.pq_prompt_template = PQ_PROMPT_TEMPLATE
        
        # Logging configuration
        strategy = "Min Geometric Mean" if self.use_min_geometric_mean else "Weighted Geometric Mean"
        print(f"📊 Aggregation strategy: {strategy}")
        if not self.use_min_geometric_mean:
            print(f"   w1={self.w1}, w2={self.w2}, w3={self.w3}, w4={self.w4}, a={self.a}")
        print("✅ SpatialRewardScorer initialized")
    
    def score(
        self,
        input_images: List[List[Image.Image]],
        output_images: List[Image.Image],
        metadata: List[Dict[str, Any]]
    ) -> List[Tuple[float, str]]:
        """
        Main scoring function
        
        Args:
            input_images: List[List[Image]] - input images list for each sample
            output_images: List[Image] - output images for each sample
            metadata: List[Dict] - metadata for each sample (must contain 'instruction')
        
        Returns:
            List[(reward, reasoning)] - reward and reasoning for each sample
        """
        # Prepare SC prompts (Original + Edited images)
        sc_inputs = []
        for input_imgs, output_img, meta in zip(input_images, output_images, metadata):
            instruction = meta.get('instruction', '')
            prompt_text = self.sc_prompt_template.format(
                score_range=self.score_range,
                instruction=instruction
            )
            images = input_imgs + [output_img]
            sc_inputs.append(self.model.prepare_input(images, prompt_text))
        
        # Prepare PQ prompts (Edited image only)
        pq_inputs = []
        for output_img in output_images:
            prompt_text = self.pq_prompt_template.format(score_range=self.score_range)
            pq_inputs.append(self.model.prepare_input([output_img], prompt_text))
        
        # Batch inference
        all_inputs = sc_inputs + pq_inputs
        results = self.model.batch_inference(all_inputs)
        
        # Separate SC and PQ results
        num_samples = len(input_images)
        sc_outputs = results[:num_samples]
        pq_outputs = results[num_samples:]
        
        # Parse and calculate reward
        outputs = []
        for sc_out, pq_out in zip(sc_outputs, pq_outputs):
            sc_parsed = parse_vlm_output(sc_out)
            pq_parsed = parse_vlm_output(pq_out)
            
            # Get raw scores, set default values
            SC_raw = sc_parsed.get('score', [self.score_range / 2, self.score_range / 2])
            PQ_raw = pq_parsed.get('score', [self.score_range / 2, self.score_range / 2])
            
            # Ensure at least 2 scores
            if len(SC_raw) < 2:
                SC_raw = SC_raw + [self.score_range / 2] * (2 - len(SC_raw))
            if len(PQ_raw) < 2:
                PQ_raw = PQ_raw + [self.score_range / 2] * (2 - len(PQ_raw))
            
            # Calculate reward
            if self.use_min_geometric_mean:
                reward, SC_score, PQ_score = RewardAggregator.min_geometric_mean(
                    SC_raw, PQ_raw, self.score_range
                )
            else:
                reward, SC_score, PQ_score = RewardAggregator.weighted_geometric_mean(
                    SC_raw, PQ_raw,
                    self.w1, self.w2, self.w3, self.w4, self.a,
                    self.score_range
                )
            
            # Construct detailed reasoning info
            reasoning = self._format_reasoning(
                SC_raw, PQ_raw, SC_score, PQ_score, reward * 10,
                sc_parsed['reasoning'], pq_parsed['reasoning'],
                sc_out, pq_out
            )
            
            outputs.append((reward, reasoning))
        
        return outputs
    
    def _format_reasoning(
        self,
        SC_raw: List[float],
        PQ_raw: List[float],
        SC_score: float,
        PQ_score: float,
        O_score: float,
        sc_reasoning: str,
        pq_reasoning: str,
        sc_output: str,
        pq_output: str
    ) -> str:
        """Format reasoning info"""
        lines = [
            f"SC_raw_scores: {SC_raw}",
            f"PQ_raw_scores: {PQ_raw}",
            f"SC_score (weighted): {SC_score:.3f}",
            f"PQ_score (weighted): {PQ_score:.3f}",
            f"O_score: {O_score:.3f}",
            f"SC_reasoning: {sc_reasoning}",
            f"PQ_reasoning: {pq_reasoning}",
            f"SC_raw_output: {sc_output}",
            f"PQ_raw_output: {pq_output}",
        ]
        return "\n".join(lines)
