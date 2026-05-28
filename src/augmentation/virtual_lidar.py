import cv2
import torch
import numpy as np
from .nnfill import fill_in_fast
from numpy.random import Generator

class virtual_lidar_sampler(object):
    def __init__(
        self,
    ):
        super().__init__()
        self.cx_range = [0.2, 0.8]
        self.cy_range = [0.2, 0.8]
        self.focal_x_range = [1.5, 2.0]
        self.focal_y_range = [1.5, 2.0]
        self.expand_range = [1.1, 1.5]
        self.offset_range = (-0.06, 0.06), (-0.02, 0.02)  # (dx, dy)
        self.eps = 1e-8

    def sample_translation(self, rng: Generator):
        """
        Samples a small 3D translation in the camera's X and Y directions.

        Args:
            rng (Generator): Random number generator.

        Returns:
            Tuple[float, float]: Translation offsets (dx, dy).
        """
        (min_dx, max_dx), (min_dy, max_dy) = self.offset_range
        dx = rng.uniform(min_dx, max_dx)
        dy = rng.uniform(min_dy, max_dy)
        return dx, dy
    
    def sample_virtual_camera(self, h: int, w: int, rng: Generator):
        """
        Samples intrinsic parameters for a virtual camera and computes target image shape.

        Args:
            h (int): Original image height.
            w (int): Original image width.
            rng (Generator): Random number generator.

        Returns:
            Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
                - src_intrinsic: Original virtual camera intrinsic matrix.
                - tgt_intrinsic: Target camera intrinsic (adjusted for expansion).
                - (tgt_h, tgt_w): Target image dimensions.
        """
        expand_factor = rng.uniform(*self.expand_range)
        tgt_h, tgt_w = int(expand_factor * h), int(expand_factor * w)

        cx = rng.uniform(self.cx_range[0] * w, self.cx_range[1] * w)
        cy = rng.uniform(self.cy_range[0] * h, self.cy_range[1] * h)
        focal_x = rng.uniform(self.focal_x_range[0] * w, self.focal_x_range[1] * w)
        focal_y = rng.uniform(self.focal_y_range[0] * h, self.focal_y_range[1] * h)

        intrinsic = np.eye(3)
        intrinsic[0, 0], intrinsic[1, 1] = focal_x, focal_y
        intrinsic[0, 2], intrinsic[1, 2] = cx, cy

        tgt_intrinsic = intrinsic.copy()
        tgt_intrinsic[0, 2] += (expand_factor - 1.0) / 2.0 * w
        tgt_intrinsic[1, 2] += (expand_factor - 1.0) / 2.0 * h

        return intrinsic, tgt_intrinsic, (tgt_h, tgt_w)
    
    def _backproject_valid_points(self, depth: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
        """
        Back-projects valid (non-zero) depth values to 3D camera coordinates.

        Args:
            depth (np.ndarray): Input depth map of shape (H, W).
            intrinsic (np.ndarray): Camera intrinsic matrix (3x3).

        Returns:
            np.ndarray: 3D points in camera coordinates of shape (3, N).
        """
        v, u = np.nonzero(depth)
        z = depth[v, u]
        pixel_coords = np.vstack([u, v, np.ones_like(u)]) * z  # [3, N]
        points_3d = np.linalg.inv(intrinsic) @ pixel_coords  # [3, N]
        return points_3d
    
    def _project_3d_to_2d(
        self, points_3d: np.ndarray, intrinsic: np.ndarray, shape: tuple[int, int]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Projects 3D points to 2D image coordinates.

        Args:
            points_3d (np.ndarray): 3D points in camera coordinates (3, N).
            intrinsic (np.ndarray): Camera intrinsic matrix (3x3).
            shape (Tuple[int, int]): Target image shape (H, W).

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                - uv_rounded: Rounded pixel coordinates (2, N).
                - z: Depth values (N,).
                - valid: Boolean mask indicating valid projections.
        """
        proj = intrinsic @ points_3d  # [3, N]
        z = proj[2]
        uv = proj[:2] / (z + self.eps)
        uv_rounded = np.round(uv).astype(int)

        h, w = shape
        valid = (
            (uv_rounded[0] >= 0) & (uv_rounded[0] < w) &
            (uv_rounded[1] >= 0) & (uv_rounded[1] < h) &
            (z > 0)
        )

        uv_valid = uv_rounded[:, valid]
        z_valid = z[valid]

        return uv_valid, z_valid, valid  # uv_valid: (2, M), z_valid: (M,)

    
    def _build_depth_map(
        self, proj_uv: np.ndarray, proj_z: np.ndarray, shape: tuple[int, int]
    ) -> np.ndarray:
        """
        Builds a dense depth map from projected 3D points, filling occluded areas.

        Args:
            proj_uv (np.ndarray): Projected pixel coordinates (2, N).
            proj_z (np.ndarray): Corresponding depth values (N,).
            shape (Tuple[int, int]): Output depth map shape (H, W).

        Returns:
            np.ndarray: Dense depth map with inpainted holes.
        """
        h, w = shape
        depth_map = np.full((h, w), np.inf, dtype=proj_z.dtype)
        np.minimum.at(depth_map, (proj_uv[1], proj_uv[0]), proj_z)
        depth_map[depth_map == np.inf] = 0.0
        depth_map = fill_in_fast(depth_map, max_depth=np.max(depth_map))
        return depth_map

    def _simulate_lidar_mask(
        self, intrinsic: np.ndarray, shape: tuple[int, int], rng: Generator = None, lidar_lines: int = None, fixed_pattern: bool = False
    ) -> np.ndarray:
        """
        Simulates a LiDAR scanning pattern as a binary mask.

        Args:
            intrinsic (np.ndarray): Camera intrinsic matrix.
            shape (Tuple[int, int]): Image shape (H, W).
            rng (Generator): Random generator (for training).
            lidar_lines (int): Number of vertical lines (for eval).
            fixed_pattern (bool): Whether to use fixed pitch/yaw ranges.

        Returns:
            np.ndarray: Binary mask of shape (H, W) with 1.0 at LiDAR hit points.
        """
        h, w = shape
        if fixed_pattern:
            pitch_min, pitch_max = -0.5, 0.5
            num_lines = lidar_lines
            num_points = 200
        else:
            pitch_min = rng.uniform(-0.20, -0.15)
            pitch_max = rng.uniform(0.25, 0.30)
            num_lines = rng.integers(2, 64)
            num_points = rng.integers(400, 1000)

        yaw_min, yaw_max = -np.pi / 2.1, np.pi / 2.1
        pitch = np.linspace(pitch_min, pitch_max, num_lines)
        yaw = np.linspace(yaw_min, yaw_max, num_points)
        pitch_grid, yaw_grid = np.meshgrid(pitch, yaw, indexing='ij')  # [L, P]

        x = np.cos(pitch_grid) * np.sin(yaw_grid)
        y = np.sin(pitch_grid)
        z = np.cos(pitch_grid) * np.cos(yaw_grid)
        norm = np.sqrt(x**2 + y**2 + z**2)
        x, y, z = x / norm, y / norm, z / norm

        directions = np.stack([x, y, z], axis=0).reshape(3, -1)  # [3, L*P]
        proj = intrinsic @ directions
        uv = proj[:2] / (proj[2] + self.eps)
        uv_rounded = np.round(uv).astype(int)

        mask = np.zeros((h, w), dtype=np.float32)
        valid = (
            (uv_rounded[0] >= 0) & (uv_rounded[0] < w) &
            (uv_rounded[1] >= 0) & (uv_rounded[1] < h)
        )
        valid_uv = uv_rounded[:, valid]
        mask[valid_uv[1], valid_uv[0]] = 1.0
        return mask
    
    def train_sample(
        self, depth: np.ndarray, rng: Generator):
        """
        Generates a sparse depth map for training with geometric augmentation.

        Steps:
            1. Sample virtual camera and translation.
            2. Back-project and transform 3D points.
            3. Project to expanded view and simulate LiDAR scan.
            4. Reproject to original view with occlusion filtering.

        Args:
            depth (np.ndarray): Dense input depth map (H, W).
            rng (Generator): Random number generator.

        Returns:
            np.ndarray: Sparse depth map (H, W) with LiDAR-like sparsity.
        """
        h, w = depth.shape

        src_intrinsic, tgt_intrinsic, tgt_shape = self.sample_virtual_camera(h, w, rng)
        dx, dy = self.sample_translation(rng)

        src_points_3d = self._backproject_valid_points(depth, src_intrinsic)

        tgt_points_3d = src_points_3d.copy()
        tgt_points_3d[0] -= dx
        tgt_points_3d[1] -= dy

        proj_uv, proj_z, valid_mask = self._project_3d_to_2d(tgt_points_3d, tgt_intrinsic, tgt_shape)
        tgt_depth_map = self._build_depth_map(proj_uv, proj_z, tgt_shape)

        lidar_mask = self._simulate_lidar_mask(tgt_intrinsic, tgt_shape, rng = rng)

        sampled_tgt_depth = tgt_depth_map * lidar_mask

        sampled_3d = self._backproject_valid_points(sampled_tgt_depth, tgt_intrinsic)
        src_reprojected_3d = sampled_3d.copy()
        src_reprojected_3d[0] += dx
        src_reprojected_3d[1] += dy
        
        src_uv, src_z, src_valid = self._project_3d_to_2d(src_reprojected_3d, src_intrinsic, (h, w))

        sparse_depth = np.full((h, w), np.inf, dtype=depth.dtype)
        valid_coords = (
            (src_uv[0] >= 0) & (src_uv[0] < w) &
            (src_uv[1] >= 0) & (src_uv[1] < h)
        )
        x_valid = np.clip(src_uv[0, valid_coords].astype(int), 0, w - 1)
        y_valid = np.clip(src_uv[1, valid_coords].astype(int), 0, h - 1)
        z_valid = src_z[valid_coords]
        np.minimum.at(sparse_depth, (y_valid, x_valid), z_valid)
        sparse_depth[sparse_depth == np.inf] = 0.0

        valid_original = (depth > 0.0)
        sparse_depth *= valid_original.astype(sparse_depth.dtype)
        invalid_mask = (sparse_depth > 0.0) & (sparse_depth < depth)
        sparse_depth[invalid_mask] = 0.0

        return sparse_depth
    
    def eval_sample(
        self, depth: np.ndarray, lidar_lines: int, rng: Generator
    ):
        """
        Generates a sparse depth map for evaluation with fixed parameters.

        Uses centered camera and fixed LiDAR line count for reproducibility.

        Args:
            depth (np.ndarray): Dense input depth map (H, W).
            lidar_lines (int): Number of simulated LiDAR lines.
            rng (Generator): Random generator (unused in fixed mode).

        Returns:
            np.ndarray: Sparse depth map (H, W).
        """
        
        h, w = depth.shape
        cx, cy = 0.5 * w, 0.5 * h
        focal = h

        if isinstance(lidar_lines, (list, tuple)):
            actual_lines = rng.integers(low=lidar_lines[0], high=lidar_lines[1] + 1)
        else:
            actual_lines = lidar_lines

        intrinsic = np.eye(3)
        intrinsic[0, 0] = focal
        intrinsic[1, 1] = focal
        intrinsic[0, 2] = cx
        intrinsic[1, 2] = cy

        lidar_mask = self._simulate_lidar_mask(intrinsic, (h, w), lidar_lines=actual_lines, fixed_pattern=True)
        valid_original = (depth > 0.0)
        sparse_depth = depth * lidar_mask * valid_original.astype(float)
        return sparse_depth

    def sample(
        self,
        depth: torch.Tensor,
        split: str,
        lidar_lines: int = 64,
        rng: Generator = None
    ) -> torch.Tensor:
        """
        Main interface for generating sparse depth maps, handling torch.Tensor inputs.

        Args:
            depth (torch.Tensor): Input dense depth map, shape [1, H, W].
            split (str): 'train' or 'eval'.
            lidar_lines (int): Number of LiDAR lines (used in 'eval' mode).
            rng (Generator): Optional random generator for reproducibility.

        Returns:
            torch.Tensor: Sparse depth map, shape [1, H, W].
        """
        if rng is None:
            rng = np.random.default_rng()

        original_device = depth.device
        original_dtype = depth.dtype
        
        depth_np = depth.squeeze(0).cpu().numpy()

        if split == 'train':
            sparse_depth_np = self.train_sample(depth_np, rng)
        elif split == 'eval':
            sparse_depth_np = self.eval_sample(depth_np, lidar_lines, rng)
        else:
            raise ValueError(f"Invalid split '{split}'. Must be 'train' or 'eval'.")
        
        sparse_depth_tensor = torch.from_numpy(sparse_depth_np).unsqueeze(0).to(device=original_device, dtype=original_dtype)

        return sparse_depth_tensor
