import torch
import utils3d
import mmengine
import numpy as np
from .perspective_crop import perspective_crop
from .tools import (
    warp_perspective, 
    warp_depth, 
    warp_pose,
    warp_normal
)

@mmengine.TRANSFORMS.register_module()
class perspective_crop_eval(perspective_crop):
    """
    A deterministic perspective cropping transformation designed specifically for evaluation (validation/test) phases.

    Unlike its parent class `perspective_crop` (used for training with random augmentations),
    this class applies a **fixed, center-aligned, maximum-coverage** cropping strategy:
      - The output view is centered on the original image center.
      - The field of view is maximized while respecting the input aspect ratio and fitting within the original view.
      - No random augmentation (e.g., jittering, blurring, flipping) is applied.
      - Output dimensions are constrained to be multiples of a specified value (e.g., 16) to satisfy network architecture requirements (like U-Net downsampling).

    This ensures consistent, reproducible, and fair evaluation by avoiding stochasticity and preserving as much valid scene content as possible.

    Args:
        ensure_multiple_of (int): The output width and height will be rounded to the nearest multiple of this value.
                                  Useful for models requiring input dimensions divisible by powers of 2. Default: 16.
        **kwargs: Additional arguments passed to the parent `perspective_crop` class (e.g., width, height are ignored).
    """
    def __init__(
        self,
        tgt_width: int | None = None,
        tgt_height: int | None = None,
        ensure_multiple_of: int = 16,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.ensure_multiple_of = ensure_multiple_of
        self.tgt_width = tgt_width
        self.tgt_height = tgt_height
    
    def compute_transform(self, src_intrinsics: np.ndarray, tgt_aspect: float):
        """
        Compute a deterministic transformation (intrinsic + rotation) for center-aligned cropping.

        The goal is to extract the largest possible view from the original camera that:
          - Is centered on the principal point.
          - Maintains the target aspect ratio.
          - Fits entirely within the original field of view after perspective warping.

        Args:
            src_intrinsics (np.ndarray): Original camera intrinsic matrix (3x3), assumed to be normalized by image size.
            tgt_aspect (float): Target aspect ratio (width / height) for the output view.

        Returns:
            tuple:
                - tgt_intrinsics (np.ndarray): Computed target intrinsic matrix (3x3).
                - R (np.ndarray): Rotation matrix (3x3) aligning the new view direction with the z-axis.
        """
        raw_horizontal, raw_vertical = abs(1.0 / src_intrinsics[0, 0]), abs(1.0 / src_intrinsics[1, 1])
        
        # set expected target view field
        tgt_horizontal = min(raw_horizontal, raw_vertical * tgt_aspect)
        tgt_vertical = tgt_horizontal / tgt_aspect

        # set target view direction
        cu, cv = 0.5, 0.5
        direction = utils3d.np.unproject_cv(np.array([[cu, cv]], dtype=np.float32), np.array([1.0], dtype=np.float32), intrinsics=src_intrinsics)[0]
        R = utils3d.np.rotation_matrix_from_vectors(direction, np.array([0, 0, 1], dtype=np.float32))

        # restrict target view field within the raw view
        corners = np.array([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=np.float32)
        corners = np.concatenate([corners, np.ones((4, 1), dtype=np.float32)], axis=1) @ (np.linalg.inv(src_intrinsics).T @ R.T)   # corners in viewport's camera plane
        corners = corners[:, :2] / corners[:, 2:3]

        warp_horizontal, warp_vertical = abs(1.0 / src_intrinsics[0, 0]), abs(1.0 / src_intrinsics[1, 1])
        for i in range(4):
            intersection, _ = utils3d.numpy.ray_intersection(
                np.array([0., 0.]), np.array([[tgt_aspect, 1.0], [tgt_aspect, -1.0]]),
                corners[i - 1], corners[i] - corners[i - 1],
            )
            warp_horizontal, warp_vertical = min(warp_horizontal, 2 * np.abs(intersection[:, 0]).min()), min(warp_vertical, 2 * np.abs(intersection[:, 1]).min())
        tgt_horizontal, tgt_vertical = min(tgt_horizontal, warp_horizontal), min(tgt_vertical, warp_vertical)

        fx, fy = 1.0 / tgt_horizontal, 1.0 / tgt_vertical
        tgt_intrinsics = utils3d.numpy.intrinsics_from_focal_center(fx, fy, 0.5, 0.5).astype(np.float32)
        return tgt_intrinsics, R

    def constrain_to_multiple_of(self, x: int):
        """
        Round the given dimension to the nearest multiple of `self.ensure_multiple_of`.

        Ensures compatibility with deep networks that require input sizes divisible by powers of 2.

        Args:
            x (int): Input dimension (width or height).

        Returns:
            int: Dimension rounded to the nearest multiple of `ensure_multiple_of`.
        """
        return (np.round(x / self.ensure_multiple_of) * self.ensure_multiple_of).astype(int)
    
    def __call__(self, sample):
        """
        Apply the deterministic perspective crop to the input sample.

        Processes image, depth, intrinsic, and optional prior. No image augmentations are applied.

        Args:
            sample (dict): Input sample with keys:
                - 'image': RGB image (H, W, 3), float in [0,1] or uint8.
                - 'depth': Depth map (H, W).
                - 'intrinsic': Camera intrinsic matrix (3, 3).
                - 'prior': Optional prior depth (e.g., from another model).

        Returns:
            dict: Transformed sample with:
                - 'image': Cropped and warped image, normalized to [0,1].
                - 'depth': Warped depth map.
                - 'intrinsic': New intrinsic matrix.
                - 'prior': Warped prior (if present).
        """
        image = sample['image']
        depth = sample['depth']
        pose = sample['pose']
        intrinsic = sample['intrinsic']
        normal = sample['normal']

        raw_height, raw_width = image[0].shape[0], image[0].shape[1]
        if self.tgt_height is not None and self.tgt_width is not None:
            tgt_height, tgt_width = self.tgt_height, self.tgt_width
        else:
            tgt_height, tgt_width = self.constrain_to_multiple_of(raw_height), self.constrain_to_multiple_of(raw_width)
        
        tgt_aspect = tgt_width / tgt_height
        
        # perspective augmentation
        tgt_intrinsic, R = self.compute_transform(intrinsic[0], tgt_aspect=tgt_aspect)
        transform = tgt_intrinsic @ R @ np.linalg.inv(intrinsic[0])
        
        tgt_image = []
        tgt_depth = []
        tgt_intrinsics = []
        tgt_poses = []
        tgt_normal = []

        for image_i, depth_i, pose_i, normal_i in zip(image, depth, pose, normal):
            tgt_image.append(warp_perspective(image_i, transform, tgt_size=(tgt_height, tgt_width), interpolation='lanczos'))
            tgt_depth.append(warp_depth(depth_i, transform, (tgt_height, tgt_width)))
            tgt_intrinsics.append(tgt_intrinsic)
            tgt_poses.append(warp_pose(pose_i, R) if pose_i is not None else None)
            tgt_normal.append(warp_normal(normal_i, transform, (tgt_height, tgt_width), R) if normal_i is not None else None)

        sample['image'] = torch.stack([torch.from_numpy(image_i).permute(2, 0, 1) for image_i in tgt_image], dim=0) / 255.0 # [S, 3, H, W]
        sample['depth'] = torch.stack([torch.from_numpy(depth_i).unsqueeze(0) for depth_i in tgt_depth], dim=0) # [S, 1, H, W]
        sample['normal'] = torch.zeros_like(sample['image']) if tgt_normal[0] is None else torch.stack([torch.from_numpy(normal_i).permute(2, 0, 1) for normal_i in tgt_normal], dim=0) # [S, 3, H, W]
        sample['intrinsic'] = torch.stack([torch.from_numpy(intrinsic_i).unsqueeze(0) for intrinsic_i in tgt_intrinsics], dim=0) # [S, 1, 3, 3]
        sample['pose'] = torch.eye(4, 4, dtype=torch.float32).unsqueeze(0).repeat(len(tgt_poses), 1, 1) if tgt_poses[0] is None else \
            torch.stack([torch.from_numpy(pose_i).unsqueeze(0) for pose_i in tgt_poses], dim=0)
        return sample
