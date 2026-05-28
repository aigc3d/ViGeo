import random
import numpy as np
from torch.utils.data import BatchSampler

class MultiScaleBatchSampler(BatchSampler):
    """
    Batch Sampler for multi-scale training supporting:
    - Mixing multiple datasets (sampling according to weights)
    - Each batch has a consistent resolution
    - The number of iterations per epoch can be precisely controlled (e.g., 2000 iterations)
    - Support for DDP (each GPU samples independently)
    - No need to maintain a global index, avoiding complications from datasets of different lengths

    Note: This sampler returns indices in tuples (dataset_idx, sample_idx, height, width).
    """

    def __init__(
        self,
        dataset_lengths: list[int],    # Lengths of each dataset, e.g., [len(ds1), len(ds2)]
        weights: list[float],          # Sampling weights, such as [0.6, 0.4]
        sampler = None,
        num_iterations: int = 2000,    # Number of iterations (batches) per epoch
        warm_epoch: int = 10, # warm epoch
        batch_size: int | None = None,
        ensure_multiple_of: int = 14,
        max_image_num: int = 48,
        image_num: int | None = None,
        image_num_range: list[int] = [2, 24], # Number range for images within a sequence
        area_range: list[float] = [112896, 409600],      # Area range for resolutions, roughly from ~336x336 to 640x640
        aspect_ratio_range: list[float] = [0.5, 2.0],   # Aspect ratio range
        seed: int = 42                  # Random seed
    ):

        if len(dataset_lengths) != len(weights):
            raise ValueError("The length of datasets and weights must match.")

        self.dataset_lengths = dataset_lengths
        self.weights = np.array(weights) / np.sum(weights)  # Normalize weights

        self.ensure_multiple_of = ensure_multiple_of
        self.max_image_num = max_image_num
        self.image_num_range = image_num_range
        self.num_iterations = num_iterations
        self.area_range = area_range
        self.aspect_ratio_range = aspect_ratio_range
        self.seed = seed
        self.batch_size = batch_size
        self.image_num = image_num
        self.sampler = sampler
        self.warm_epoch = warm_epoch
        self.epoch = 0

    def _generate_resolution(self, rng: random.Random):
        if self.epoch < self.warm_epoch:
            ratio = min(1, self.epoch * 1.0 / self.warm_epoch)
            area_range = [self.area_range[0], self.area_range[0] + (self.area_range[1] - self.area_range[0]) * ratio]
        else:
            area_range = self.area_range
        
        area = rng.uniform(area_range[0], area_range[1])
        ar = rng.uniform(self.aspect_ratio_range[0], self.aspect_ratio_range[1])
        h = int(round(((area / ar) ** 0.5) / self.ensure_multiple_of) * self.ensure_multiple_of)
        w = int(round(((area * ar) ** 0.5) / self.ensure_multiple_of) * self.ensure_multiple_of)
        return h, w

    def __iter__(self):
        import torch.distributed as dist
        global_rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        # Use epoch + seed to ensure different randomness for each epoch
        
        rng_res = random.Random(self.seed + self.epoch)
        rng_data = random.Random(self.seed + self.epoch * 1000 + global_rank)
        if self.num_iterations is not None:
            iterator = range(self.num_iterations)
        else:
            iterator = iter(int, -1)

        for _ in iterator:
            # 1. Generate a uniform resolution for the current batch
            tgt_h, tgt_w = self._generate_resolution(rng_res)
            image_num = rng_res.randint(self.image_num_range[0], self.image_num_range[1]) if self.image_num is None else self.image_num
            batch_size = self.max_image_num // image_num if self.batch_size is None else self.batch_size
            # 2. Build the list of indices for the current batch
            batch = []
            for _ in range(batch_size):
                # Sample a dataset according to weights
                dataset_idx = rng_data.choices(range(len(self.dataset_lengths)), weights=self.weights)[0]
                # Randomly sample one sample from the selected dataset
                seq_idx = rng_data.randint(0, self.dataset_lengths[dataset_idx] - 1)
                seed = rng_data.randint(0, 2**32 - 1)  # Full uint32 range
                # Store composite index: data_idx, seq_idx, seq_len, seed, tgt_h, tgt_w = index 
                batch.append((dataset_idx, seq_idx, image_num, seed, tgt_h, tgt_w))
            
            yield batch  # Return a batch

    def __len__(self):
        """
        Returns the number of batches (iterations) per epoch.
        """
        return self.num_iterations if self.num_iterations is not None else 0

    def set_epoch(self, epoch: int):
        """
        Called by DDP to notify the sampler about the current epoch.
        """
        self.epoch = epoch