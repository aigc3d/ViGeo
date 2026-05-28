import os
import json
import torch
import numpy as np
from PIL import Image
from mmengine.dataset import Compose

class base_data(torch.utils.data.Dataset):
    def __init__(
        self,
        data_path: str,
        data_name: str,
        train_test_split: str,
        max_interval: int = 8,
        min_depth: float = 0.0001,
        max_depth: float = 200,
        shuffle_sequence_prob: float = 0.0,
        pipeline: list[dict] | None = None,
    ):
        super().__init__()
        self.data_name = data_name
        self.data_path = data_path
        self.scenes = self.load_annotations(train_test_split)
        self.max_interval = max_interval
        self.shuffle_sequence_prob = shuffle_sequence_prob

        self.min_depth = min_depth
        self.max_depth = max_depth

        if pipeline is not None:
            self.pipeline = Compose(pipeline)

    def load_annotations(self, train_test_split):
        with open(train_test_split, 'r', encoding='utf-8') as f:
            scenes = json.load(f)
        return scenes
    
    def sample_video(self, idx):
        return dict()

    def shuffle_sequence_order(self, sequence, rng: np.random.Generator):
        if self.shuffle_sequence_prob <= 0 or len(sequence) <= 1:
            return sequence
        if rng.random() < self.shuffle_sequence_prob:
            return rng.permutation(sequence).tolist()
        return sequence
    
    def sample_sequence(self, n: int, seq_len: int, rng: np.random.Generator):
        if n < seq_len:
            i = np.arange(seq_len)
            cycle_pos = i % (2 * n)
            sampled = np.where(cycle_pos < n, cycle_pos, 2 * n - 1 - cycle_pos)
            return self.shuffle_sequence_order(sampled.tolist(), rng)
            
        if seq_len == 1:
            return self.shuffle_sequence_order([int(rng.choice(n))], rng)

        eff_max_interval = max(1, min(self.max_interval, n - 1))
        
        sampled = []
        
        curr_pos = int(rng.integers(0, n))
        
        direction = 1 

        for _ in range(seq_len):
            sampled.append(curr_pos)
            
            step = int(rng.integers(1, eff_max_interval + 1))
            
            next_pos = curr_pos + direction * step
            
            if next_pos >= n:
                direction = -1 
                step = int(rng.integers(1, eff_max_interval + 1))
                next_pos = curr_pos + direction * step
                next_pos = max(0, next_pos) 
                
            elif next_pos < 0:
                direction = 1 
                step = int(rng.integers(1, eff_max_interval + 1))
                next_pos = curr_pos + direction * step
                next_pos = min(n - 1, next_pos)
                
            curr_pos = next_pos
            
        return self.shuffle_sequence_order(sampled, rng)
    
    def __len__(self):
        return len(self.scenes)
    
    def load_image(self, image_path):
        image = Image.open(os.path.join(self.data_path, image_path))
        image = np.asarray(image, dtype=np.uint8)[:, :, :3]
        return image
    
    def load_normal(self, normal_path):
        try:
            data = np.load(os.path.join(self.data_path, normal_path), allow_pickle=True).item()
            normal = data['normal']
            
            normal_mask = data['normal_mask']
            normal[~normal_mask] = 0.0
            
            norm = np.linalg.norm(normal, axis=-1, keepdims=True)
            normal = np.divide(normal, norm, out=np.zeros_like(normal), where=norm > 1e-6)
            return normal.astype(np.float32)

        except Exception as e:
            return None

    
    def normalize_intrinsic(self, intrinsic: np.ndarray, height: int | float, width: int | float):
        intrinsic[0, :] = intrinsic[0, :] / width
        intrinsic[1, :] = intrinsic[1, :] / height
        return intrinsic
    
    def filter_depth(self, depth):
        return np.where((depth > self.min_depth) & (depth < self.max_depth), depth, 0)
    
    def __getitem__(self, idx):
        sample = self.sample_video(idx)
        sample['data_name'] = self.data_name
        if hasattr(self, "pipeline"):
            sample = self.pipeline(sample)
        
        return sample