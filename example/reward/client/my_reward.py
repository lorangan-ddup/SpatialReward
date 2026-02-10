# Copyright 2026 Spatial Reward Server
# src/flow_factory/rewards/my_reward.py

from accelerate import Accelerator
from typing import Optional, List, Union
from PIL import Image
import torch
import pickle
import requests
import logging

from .abc import PointwiseRewardModel, GroupwiseRewardModel, RewardModelOutput
from ..hparams import *

logger = logging.getLogger(__name__)


class MyPointwiseRewardModel(PointwiseRewardModel):
    """
    Spatial Reward Model - Evaluates image editing quality
    
    Connects to Spatial Reward Server via HTTP.
    Server must be running before training.
    
    Required fields: prompt, condition_images, image
    """
    
    required_fields = ("prompt", "condition_images", "image")
    use_tensor_inputs = False
    
    def __init__(self, config: RewardArguments, accelerator: Accelerator):
        super().__init__(config, accelerator)
        
        # Server configuration
        self.server_url = getattr(config, 'server_url', 'http://localhost:23456')
        self.timeout = getattr(config, 'timeout', 300)
        self.retry_attempts = getattr(config, 'retry_attempts', 3)
        
        # Validate server
        self._check_server_connection()
        
        logger.info(f"✅ Spatial Reward Model initialized")
        logger.info(f"   Server: {self.server_url}")
    
    def _check_server_connection(self):
        """Check if reward server is reachable"""
        try:
            response = requests.get(f"{self.server_url}/ping", timeout=5)
            if response.status_code == 200:
                logger.info(f"✅ Server {self.server_url} is reachable")
            else:
                raise RuntimeError(f"Server returned status {response.status_code}")
        except Exception as e:
            raise RuntimeError(
                f"❌ Cannot connect to reward server at {self.server_url}\n"
                f"   Make sure the server is running:\n"
                f"     bash start_servers.sh\n"
                f"     bash start_proxy.sh"
            )
    
    @torch.no_grad()
    def __call__(
        self,
        prompt: List[str],
        image: Optional[List[Image.Image]] = None,
        video: Optional[List[List[Image.Image]]] = None,
        condition_images: Optional[List[Union[List[Image.Image], torch.Tensor]]] = None,
        condition_videos: Optional[List[Union[List[List[Image.Image]], torch.Tensor]]] = None,
    ) -> RewardModelOutput:
        """Compute spatial editing rewards"""
        batch_size = len(prompt)
        
        if image is None or condition_images is None:
            logger.warning("Missing inputs, returning zero rewards")
            return RewardModelOutput(
                rewards=torch.zeros(batch_size, device=self.device)
            )
        
        # Prepare server request
        input_images = []
        output_images = []
        meta_datas = []
        
        for i in range(batch_size):
            # Extract condition image
            if isinstance(condition_images[i], list):
                cond_img = condition_images[i][0] if condition_images[i] else None
            else:
                cond_img = condition_images[i]
            
            if cond_img is None:
                continue
            
            input_images.append([cond_img])
            output_images.append(image[i])
            meta_datas.append({"instruction": prompt[i]})
        
        # Request server
        try:
            rewards_list = self._request_server(input_images, output_images, meta_datas)
        except Exception as e:
            logger.error(f"Server request failed: {e}")
            return RewardModelOutput(
                rewards=torch.zeros(batch_size, device=self.device)
            )
        
        rewards = torch.tensor(rewards_list, dtype=torch.float32, device=self.device)
        
        return RewardModelOutput(rewards=rewards)
    
    def _request_server(self, input_images, output_images, meta_datas):
        """Send request to reward server"""
        request_data = {
            'input_images': input_images,
            'output_image': output_images,
            'meta_datas': meta_datas,
            'server_type': 'vlm'
        }
        
        for attempt in range(self.retry_attempts):
            try:
                pickled_data = pickle.dumps(request_data)
                response = requests.post(
                    self.server_url,
                    data=pickled_data,
                    headers={'Content-Type': 'application/octet-stream'},
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    result = pickle.loads(response.content)
                    return result.get('rewards', [])
                else:
                    logger.error(f"HTTP error: {response.status_code}")
                    
            except Exception as e:
                logger.error(f"Request exception (attempt {attempt + 1}/{self.retry_attempts}): {e}")
                if attempt < self.retry_attempts - 1:
                    import time
                    time.sleep(2 ** attempt)
        
        raise RuntimeError(f"All {self.retry_attempts} attempts failed")


class MyGroupwiseRewardModel(GroupwiseRewardModel):
    """Groupwise version with ranking (optional)"""
    
    required_fields = ("prompt", "condition_images", "image")
    
    def __init__(self, config: RewardArguments, accelerator: Accelerator):
        super().__init__(config, accelerator)
        
        self.server_url = getattr(config, 'server_url', 'http://localhost:23456')
        self.timeout = getattr(config, 'timeout', 300)
        self.retry_attempts = getattr(config, 'retry_attempts', 3)
        
        # Check server
        try:
            response = requests.get(f"{self.server_url}/ping", timeout=5)
            if response.status_code != 200:
                raise RuntimeError(f"Server not ready")
        except Exception as e:
            raise RuntimeError(f"Cannot connect to {self.server_url}: {e}")
        
        logger.info(f"✅ Spatial Reward Model (Groupwise) initialized")
    
    @torch.no_grad()
    def __call__(
        self,
        prompt: List[str],
        image: Optional[List[Image.Image]] = None,
        video: Optional[List[List[Image.Image]]] = None,
        condition_images: Optional[List[Union[List[Image.Image], torch.Tensor]]] = None,
        condition_videos: Optional[List[Union[List[List[Image.Image]], torch.Tensor]]] = None,
    ) -> RewardModelOutput:
        """Compute rewards with ranking"""
        # Use MyPointwiseRewardModel to get scores in batches
        pointwise = MyPointwiseRewardModel(self.config, self.accelerator)
        
        group_size = len(prompt)
        all_scores = []
        
        for i in range(0, group_size, self.config.batch_size):
            batch_end = min(i + self.config.batch_size, group_size)
            batch_output = pointwise(
                prompt=prompt[i:batch_end],
                image=image[i:batch_end] if image else None,
                condition_images=condition_images[i:batch_end] if condition_images else None,
            )
            all_scores.append(batch_output.rewards)
        
        raw_scores = torch.cat(all_scores, dim=0)
        
        # Convert to ranks
        ranks = raw_scores.argsort().argsort()
        rewards = ranks.float() / max(group_size - 1, 1)
        
        return RewardModelOutput(rewards=rewards)