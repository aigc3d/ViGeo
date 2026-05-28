import os
import cv2
import torch
import mmengine
import numpy as np
from typing import Literal
from .virtual_lidar import virtual_lidar_sampler

@mmengine.TRANSFORMS.register_module()
class sample_points(object):
    def __init__(
        self,
        split: Literal["train", "eval"] = "eval",
        mode: str | list[str] = "random",
        weights: list[int | float] | None = None,
        depth_noise: list[float] | None = None,
        lidar_lines: int | list[int] = 64,
        sfm_drop_rate: float = 0.0,
        sample_num_range: int | float | list[int | float] | None = [100, 2000],
        partial_masks_dir: str | None = None,
    ):
        super().__init__()
        self.split = split
        self.mode = mode
        self.weights = self._normalize_weights(mode, weights)
        self.depth_noise = depth_noise
        self.lidar_lines = lidar_lines  # Note: This parameter is now only used by virtual_lidar_sampler
        self.sfm_drop_rate = sfm_drop_rate
        self.sample_num_range = sample_num_range
        self.virtual_lidar_sampler = virtual_lidar_sampler()
        self.valid = 0.0001
        self.partial_masks = self._load_partial_masks(partial_masks_dir)

    def _normalize_weights(self, mode: str | list[str], weights: list[int | float] | None):
        if isinstance(mode, str):
            return None

        if weights is None:
            raise ValueError("weights must be provided when mode is a list.")
        if len(mode) != len(weights):
            raise ValueError("mode and weights must have the same length.")

        weights = np.array(weights, dtype=np.float64)
        weight_sum = weights.sum()
        if weight_sum <= 0:
            raise ValueError("weights must sum to a positive value.")
        return weights / weight_sum

    def _uses_partial_mode(self):
        return self.mode == "partial" or (
            isinstance(self.mode, list) and "partial" in self.mode
        )

    def _load_partial_masks(self, partial_masks_dir: str | None):
        if not self._uses_partial_mode():
            return []
        if partial_masks_dir is None:
            raise ValueError("partial_masks_dir must be set when mode includes 'partial'.")
        return [
            os.path.join(partial_masks_dir, f) for f in os.listdir(partial_masks_dir)
        ]

    def sample_mode(self, rng: np.random.Generator):
        """Selects the sampling mode based on the configured weights."""
        if isinstance(self.mode, str):
            return self.mode
        return rng.choice(self.mode, p=self.weights)
    
    def add_noise(self, depth: torch.Tensor, generator: torch.Generator):
        """
        Adds noise to a depth map in torch.Tensor format.

        Args:
            depth (torch.Tensor): Input depth map, shape: [1, H, W].
            generator (torch.Generator): PyTorch random number generator for reproducibility.

        Returns:
            torch.Tensor: Noisy depth map.
        """

        if self.depth_noise is None:
            return depth

        valid_mask = depth > self.valid

        if not torch.any(valid_mask):
            return depth
        
        noise_prob_low, noise_prob_high = self.depth_noise
        # Generate a random noise probability within the configured range
        noise_prob = torch.rand(1, generator=generator).item() * (noise_prob_high - noise_prob_low) + noise_prob_low
        
        random_mask = torch.rand(depth.shape, device=depth.device, generator=generator) < noise_prob
        noise_mask = random_mask & valid_mask

        if not torch.any(noise_mask):
            return depth
        
        depth_valid = depth[valid_mask]
        # Add a check to prevent torch.quantile from failing on an empty tensor
        if depth_valid.numel() == 0:
            return depth
        
        depth_min = torch.clamp(torch.quantile(depth_valid, 0.1) * 0.3, min=0.0001)
        depth_max = torch.quantile(depth_valid, 0.9) * 3

        # Generate noise values within the calculated range
        noise_values = torch.rand(depth.shape, device=depth.device, generator=generator) * (depth_max - depth_min) + depth_min
        
        # Use torch.where for replacement to avoid in-place modification
        noisy_depth = torch.where(noise_mask, noise_values, depth)

        return noisy_depth

    def sample_num(self, rng: np.random.Generator):
        """Samples the number of points to keep based on the configuration."""
        if self.sample_num_range is None:
            raise ValueError("sample_num_range is None, cannot determine sample number")

        if isinstance(self.sample_num_range, (int, float)):
            if isinstance(self.sample_num_range, int):
                return self.sample_num_range
            value = float(self.sample_num_range)
        elif isinstance(self.sample_num_range, list):
            if len(self.sample_num_range) != 2:
                raise ValueError("sample_num_range list must have exactly 2 elements: [min, max]")
            min_val, max_val = map(float, self.sample_num_range)
            if not (min_val <= max_val):
                raise ValueError("min must be <= max in sample_num_range list")
            value = rng.uniform(min_val, max_val)
        else:
            raise TypeError("sample_num_range must be int, float, or list of two numbers")

        return value if value <= 1.0 else int(value)

    def random_sample(self, depth: torch.Tensor, rng: np.random.Generator):
        """
        Performs random sampling on a depth map in torch.Tensor format.
        """
        valid_mask = (depth > self.valid).squeeze(0) # Remove channel dim -> (H, W)
        flat_valid = torch.nonzero(valid_mask.flatten(), as_tuple=False).squeeze(1)
        num_valid = flat_valid.numel()

        sample_num_val = self.sample_num(rng)
        k = int(num_valid * sample_num_val) if isinstance(sample_num_val, float) else sample_num_val
        k = max(0, min(k, num_valid))

        if k == 0:
            return torch.zeros_like(depth)

        # Use NumPy rng to generate a permutation, then convert to a Torch tensor
        perm = torch.from_numpy(rng.permutation(num_valid)).to(flat_valid.device)
        selected_flat_idx = flat_valid[perm[:k]]
        
        sparse_flat = torch.zeros_like(depth.flatten())
        flat_depth = depth.flatten()
        sparse_flat[selected_flat_idx] = flat_depth[selected_flat_idx]
        
        return sparse_flat.view_as(depth)
    
    def sfm_sample(self, image: torch.Tensor, depth: torch.Tensor, generator: torch.Generator, mode: str = "sift"):
        """
        Samples a depth map using SIFT/ORB keypoints detected from the corresponding image.
        """
        _c, h, w = image.shape
        
        # 1. Convert PyTorch Tensor [C, H, W] to OpenCV-compatible NumPy array [H, W, C]
        img_np = image.permute(1, 2, 0).cpu().numpy()
        
        # Robustly handle different input image data types
        if img_np.dtype == np.float32 or img_np.dtype == np.float64:
            # Assumes float images are in [0, 1] range
            gray_uint8 = np.clip(img_np * 255, 0, 255).astype(np.uint8)
        elif img_np.dtype == np.uint8:
            gray_uint8 = img_np
        else:
            raise TypeError(f"Unsupported image dtype for sfm_sample: {img_np.dtype}")

        # Convert to grayscale, as required by feature detectors
        if gray_uint8.ndim == 3 and gray_uint8.shape[2] == 3:
            gray = cv2.cvtColor(gray_uint8, cv2.COLOR_RGB2GRAY)
        elif gray_uint8.ndim == 2: # Already grayscale
            gray = gray_uint8
        else:
            raise ValueError(f"Cannot convert image with shape {gray_uint8.shape} to grayscale.")

        # 2. Detect keypoints
        if mode == "sift":
            detector = cv2.SIFT_create()
        elif mode == "orb":
            detector = cv2.ORB_create(nfeatures=100000, scoreType=cv2.ORB_FAST_SCORE)
        else:
            raise ValueError(f"Unsupported feature mode: {mode}")
        
        keypoints = detector.detect(gray)

        # 3. Create a boolean mask from keypoint locations
        mask = torch.zeros((h, w), dtype=torch.bool, device=image.device)
        for kp in keypoints:
            x, y = map(round, kp.pt)
            if 0 <= x < w and 0 <= y < h:
                mask[y, x] = True
        
        # 4. Apply dropout to simulate feature matching failures
        if self.sfm_drop_rate > 0.0:
            drop_range = self.sfm_drop_rate # Can be a range or a fixed value
            drop_prob = torch.rand(1, generator=generator).item() * drop_range
            keep_prob = 1.0 - drop_prob
            drop_mask = torch.rand(mask.shape, device=mask.device, generator=generator) < keep_prob
            mask = mask & drop_mask
        
        # 5. Apply the final mask to the depth map
        # unsqueeze(0) matches the mask [H, W] to the depth [1, H, W]
        sparse_depth = torch.where(mask.unsqueeze(0), depth, torch.tensor(0.0, device=depth.device))

        return sparse_depth

    def partial_sample(self, depth: torch.Tensor, rng: np.random.Generator):
        mask_path = rng.choice(self.partial_masks)
        base_mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        _, h, w = depth.shape # [1, h, w]

        angle = rng.uniform(-45, 45)
        scale = rng.uniform(0.7, 1.5)
        
        center = (base_mask.shape[1] // 2, base_mask.shape[0] // 2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, scale)
        augmented_mask = cv2.warpAffine(base_mask, rot_mat, (base_mask.shape[1], base_mask.shape[0]), borderValue=0)
        if rng.random() > 0.5:
            augmented_mask = cv2.flip(augmented_mask, 1)
        mask_resized = cv2.resize(augmented_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        mask_tensor = torch.from_numpy(mask_resized > 0).to(dtype=torch.bool, device=depth.device)
    
        sparse_depth = torch.where(mask_tensor.unsqueeze(0), depth, torch.tensor(0.0, device=depth.device))
        return sparse_depth
    
    def __call__(self, sample: dict) -> dict:
        """
        Main call function to process a data sample.
        """
        # 1. Set up random generators for reproducibility
        seed = sample.get('seed', None)
        rng = np.random.default_rng(seed)  # NumPy rng for non-tensor operations
        
        generator = torch.Generator() # PyTorch generator for tensor operations
        if seed is not None:
            generator.manual_seed(seed)
        
        prior_seq = sample['prior'].clone() if 'prior' in sample else sample['depth'].clone()
        image_seq = sample['image'].clone()
        
        seq_len = prior_seq.shape[0]
        sparse_priors_list = []

        for s in range(seq_len):
            mode = self.sample_mode(rng)
            noisy_prior = self.add_noise(prior_seq[s], generator)

            if mode == "random":
                sparse_prior_frame = self.random_sample(noisy_prior, rng)
            elif mode in ["sift", "orb"]:
                current_image = image_seq[s]
                sparse_prior_frame = self.sfm_sample(current_image, noisy_prior, generator, mode=mode)
            elif mode == "virtual_lidar":
                sparse_prior_frame = self.virtual_lidar_sampler.sample(noisy_prior, self.split, self.lidar_lines, rng)
            elif mode == "partial":
                sparse_prior_frame = self.partial_sample(noisy_prior, rng)
            else:
                raise NotImplementedError(f"Sampling mode '{mode}' is not implemented.")

            if torch.count_nonzero(sparse_prior_frame > self.valid) < 20:
                sparse_prior_frame = self.random_sample(noisy_prior, rng)
            
            sparse_priors_list.append(sparse_prior_frame)

        final_sparse_sequence = torch.stack(sparse_priors_list, dim=0)
        sample["prior"] = final_sparse_sequence.to(torch.float32)
        return sample
