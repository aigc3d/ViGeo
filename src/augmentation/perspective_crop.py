import torch
import utils3d
import mmengine
import numpy as np
from .tools import (
    warp_perspective, 
    warp_depth, 
    image_augmentation, 
    warp_pose,
    warp_normal
)

@mmengine.TRANSFORMS.register_module()
class perspective_crop(object):
    """
    A transformation class for performing perspective cropping on images and their corresponding depth maps.
    
    Args:
        center_augmentation (float): Degree of augmentation applied to the image center. Default is 0.5.
        image_augmentation (List[str]): List of image augmentations to apply. Defaults include 'jittering', 'jpeg_loss', 'blurring'.
        fov_range_absolute (List[float]): Absolute field of view range. Default is [30, 150].
        fov_range_relative (List[float]): Relative field of view range. Default is [0.5, 1.0].
    """
    def __init__(
        self,
        center_augmentation: float = 0.5,
        image_augmentation_list: list[str] = ["jittering", "jpeg_loss", "blurring"],
        fov_range_absolute: list[float] = [30, 150],
        fov_range_relative: list[float] = [0.5, 1.0],
        **kwargs
    ):
        self.center_augmentation = center_augmentation
        self.image_augmentation_list = image_augmentation_list
        self.fov_range_absolute = fov_range_absolute
        self.fov_range_relative = fov_range_relative

    def sample_perspective(self, src_intrinsics: np.ndarray, tgt_aspect: float, rng: np.random.Generator):
        """
        Sample a new perspective by calculating target intrinsics and rotation matrix.
        
        Args:
            src_intrinsics (np.ndarray): Source camera intrinsic matrix.
            tgt_aspect (float): Target aspect ratio.
            rng (np.random.Generator): Random number generator for reproducibility.
            
        Returns:
            tuple: Target intrinsic matrix and rotation matrix.
        """
        raw_horizontal, raw_vertical = abs(1.0 / src_intrinsics[0, 0]), abs(1.0 / src_intrinsics[1, 1])
        raw_fov_x, raw_fov_y = utils3d.np.intrinsics_to_fov(src_intrinsics)

        # 1. set target fov
        fov_range_absolute_min, fov_range_absolute_max = self.fov_range_absolute
        fov_range_relative_min, fov_range_relative_max = self.fov_range_relative
        tgt_fov_x_min = min(fov_range_relative_min * raw_fov_x, utils3d.focal_to_fov(utils3d.fov_to_focal(fov_range_relative_min * raw_fov_y) / tgt_aspect))
        tgt_fov_x_max = min(fov_range_relative_max * raw_fov_x, utils3d.focal_to_fov(utils3d.fov_to_focal(fov_range_relative_max * raw_fov_y) / tgt_aspect))
        tgt_fov_x_min, tgt_fov_x_max = max(np.deg2rad(fov_range_absolute_min), tgt_fov_x_min), min(np.deg2rad(fov_range_absolute_max), tgt_fov_x_max)
        tgt_fov_x = rng.uniform(min(tgt_fov_x_min, tgt_fov_x_max), tgt_fov_x_max)
        tgt_fov_y = utils3d.focal_to_fov(utils3d.np.fov_to_focal(tgt_fov_x) * tgt_aspect)

        # 2. set target image center (principal point) and the corresponding z-direction in raw camera space
        center_augmentation = self.center_augmentation
        center_dtheta = center_augmentation * rng.uniform(-0.5, 0.5) * (raw_fov_x - tgt_fov_x)
        center_dphi = center_augmentation * rng.uniform(-0.5, 0.5) * (raw_fov_y - tgt_fov_y)
        cu, cv = 0.5 + 0.5 * np.tan(center_dtheta) / np.tan(raw_fov_x / 2), 0.5 + 0.5 *  np.tan(center_dphi) / np.tan(raw_fov_y / 2)
        direction = utils3d.np.unproject_cv(np.array([[cu, cv]], dtype=np.float32), np.array([1.0], dtype=np.float32), intrinsics=src_intrinsics)[0]
        
        # 3. obtain the rotation matrix for homography warping (new_ext = R * old_ext)
        R = utils3d.np.rotation_matrix_from_vectors(direction, np.array([0, 0, 1], dtype=np.float32))

        # 4. shrink the target view to fit into the warped image
        corners = np.array([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=np.float32)
        corners = np.concatenate([corners, np.ones((4, 1), dtype=np.float32)], axis=1) @ (np.linalg.inv(src_intrinsics).T @ R.T)   # corners in viewport's camera plane
        corners = corners[:, :2] / corners[:, 2:3]
        tgt_horizontal, tgt_vertical = np.tan(tgt_fov_x / 2) * 2, np.tan(tgt_fov_y / 2) * 2
        warp_horizontal, warp_vertical = float('inf'), float('inf')
        for i in range(4):
            intersection, _ = utils3d.np.ray_intersection(
                np.array([0., 0.]), np.array([[tgt_aspect, 1.0], [tgt_aspect, -1.0]]),
                corners[i - 1], corners[i] - corners[i - 1],
            )
            warp_horizontal, warp_vertical = min(warp_horizontal, 2 * np.abs(intersection[:, 0]).min()), min(warp_vertical, 2 * np.abs(intersection[:, 1]).min())
        tgt_horizontal, tgt_vertical = min(tgt_horizontal, warp_horizontal), min(tgt_vertical, warp_vertical)
        
        # 5. obtain the target intrinsics
        fx, fy = 1 / tgt_horizontal, 1 / tgt_vertical
        tgt_intrinsics = utils3d.np.intrinsics_from_focal_center(fx, fy, 0.5, 0.5).astype(np.float32)

        return tgt_intrinsics, R
        
    def __call__(self, sample):
        rng = np.random.default_rng(sample['seed'])

        # list
        image = sample['image']
        depth = sample['depth']
        sky_mask = sample['sky_mask']
        pose = sample['pose']
        intrinsic = sample['intrinsic']
        normal = sample['normal']

        tgt_width = sample.get('tgt_width', image[0].shape[1])
        tgt_height = sample.get('tgt_height', image[0].shape[0])
        tgt_aspect = tgt_width / tgt_height
        
        # perspective augmentation
        tgt_intrinsic, R = self.sample_perspective(intrinsic[0], tgt_aspect=tgt_aspect, rng=rng)
        transform = tgt_intrinsic @ R @ np.linalg.inv(intrinsic[0])
        
        tgt_image = []
        tgt_sky_mask = []
        tgt_depth = []
        tgt_intrinsics = []
        tgt_poses = []
        tgt_normal = []

        for image_i, sky_mask_i, depth_i, pose_i, normal_i in zip(image, sky_mask, depth, pose, normal):
            tgt_image.append(warp_perspective(image_i, transform, tgt_size=(tgt_height, tgt_width), interpolation='lanczos'))
            tgt_sky_mask.append(warp_perspective(sky_mask_i, transform, sparse_mask=sky_mask_i > 0, tgt_size=(tgt_height, tgt_width), interpolation='nearest'))
            tgt_depth.append(warp_depth(depth_i, transform, (tgt_height, tgt_width)))
            tgt_intrinsics.append(tgt_intrinsic)
            tgt_poses.append(warp_pose(pose_i, R) if pose_i is not None else None)
            tgt_normal.append(warp_normal(normal_i, transform, (tgt_height, tgt_width), R) if normal_i is not None else None)
        
        sample['raw_image'] = torch.stack([torch.from_numpy(image_i).permute(2, 0, 1) for image_i in tgt_image], dim=0) / 255.0
        # image augmentation
        tgt_image = [image_augmentation(image, depth, rng, self.image_augmentation_list) for image, depth in zip(tgt_image, tgt_depth)]
        
        sample['image'] = torch.stack([torch.from_numpy(image_i).permute(2, 0, 1) for image_i in tgt_image], dim=0) / 255.0 # [S, 3, H, W]
        sample['depth'] = torch.stack([torch.from_numpy(depth_i).unsqueeze(0) for depth_i in tgt_depth], dim=0) # [S, 1, H, W]
        sample['intrinsic'] = torch.stack([torch.from_numpy(intrinsic_i) for intrinsic_i in tgt_intrinsics], dim=0) # [S, 1, 3, 3]
        sample['pose'] = torch.eye(4, 4, dtype=torch.float32).repeat(len(tgt_poses), 1, 1) if tgt_poses[0] is None else \
            torch.stack([torch.from_numpy(pose_i) for pose_i in tgt_poses], dim=0)
        sample['normal'] = torch.zeros_like(sample['image']) if tgt_normal[0] is None else torch.stack([torch.from_numpy(normal_i).permute(2, 0, 1) for normal_i in tgt_normal], dim=0) # [S, 3, H, W]
        sample['sky_mask'] = torch.stack([torch.from_numpy(sky_mask_i).unsqueeze(0) for sky_mask_i in tgt_sky_mask], dim=0) # [S, 1, H, W]
        return sample
