import cv2
import torch
import utils3d
import numpy as np
import torchvision.transforms.functional as TF
from PIL import Image
from typing import Literal
from scipy.signal import fftconvolve

def warp_perspective(
    src_map: np.ndarray = None, 
    transform: np.ndarray = None,
    tgt_size: tuple[int, int] = None, 
    interpolation: Literal['nearest', 'bilinear', 'lanczos'] = 'nearest',
    sparse_mask: np.ndarray = None,
):
    """Perspective warping with careful resampling.
    - For `lanczos`, use PIL to resize first to reduce aliasing.
    - For `nearest` with sparse input, use mask-aware nearest resize to avoid losing points.
    - For `bilinear` or `nearest` with dense input, directly use cv2.remap.

    - `transform` is the matrix that transforms homogeneous pixel coordinates of source image to those of target image, i.e., `p_tgt = transform @ p_src`.
    """
    
    tgt_height, tgt_width = tgt_size
    src_height, src_width = src_map.shape[:2]

    # source to target transform
    transform_pixel = np.array([[tgt_width, 0, -0.5], [0, tgt_height, -0.5], [0, 0, 1]], dtype=np.float32) @ transform @ np.array([[1 / src_width, 0, 0.5 / src_width], [0, 1 / src_height, 0.5 / src_height], [0, 0, 1]], dtype=np.float32)
    # Get scale factor at the target center
    w = np.dot(np.linalg.inv(transform_pixel)[2, :], np.array([tgt_width / 2, tgt_height / 2, 1], dtype=np.float32))
    scale_x, scale_y = w * np.linalg.norm(transform_pixel[:2, :2], axis=0)

    if interpolation == 'lanczos' and (scale_x < 0.8 or scale_y < 0.8):
        # If lanczos & downsampling, use PIL to resize first to reduce aliasing
        src_height, src_width = max(round(src_height * scale_y * 1.25), 16), max(round(src_width * scale_x * 1.25), 16)
        src_map = np.array(Image.fromarray(src_map).resize((src_width, src_height), Image.Resampling.LANCZOS))
    elif interpolation == 'nearest' and sparse_mask is not None and (scale_x < 1 or scale_y < 1):
        # If nearest and sparse, use mask-aware nearest resize first to avoid losing points
        src_height, src_width = max(round(src_height * scale_y), 16), max(round(src_width * scale_x), 16)
        src_map, _ = utils3d.np.masked_nearest_resize(src_map, mask=sparse_mask, size=(src_height, src_width))

    # Recompute the pixel-space transform after resizing
    transform_pixel = np.array([[tgt_width, 0, -0.5], [0, tgt_height, -0.5], [0, 0, 1]], dtype=np.float32) @ transform @ np.array([[1 / src_width, 0, 0.5 / src_width], [0, 1 / src_height, 0.5 / src_height], [0, 0, 1]], dtype=np.float32)
    
    # Remap
    cv2_interpolation = {'nearest': cv2.INTER_NEAREST, 'bilinear': cv2.INTER_LINEAR, 'lanczos': cv2.INTER_LANCZOS4}[interpolation]
    tgt_map = cv2.warpPerspective(src_map, transform_pixel, (tgt_width, tgt_height), flags=cv2_interpolation)

    return tgt_map

def warp_depth(depth: np.ndarray, transform: np.ndarray, tgt_size: tuple[int, int]):
    """
    Warp the depth map according to the given transformation matrix.
        
    Args:
        depth (np.ndarray): Depth map.
        transform (np.ndarray): Transformation matrix.
        tgt_size (Tuple[int, int]): Target size (height, width).
            
    Returns:
        np.ndarray: Warped depth map.
    """
    tgt_height, tgt_width = tgt_size

    warped_depth_nearest = warp_perspective(depth, transform, (tgt_height, tgt_width), interpolation='nearest', sparse_mask=(depth > 0))
    tgt_uvhomo = np.concatenate([utils3d.np.uv_map((tgt_height, tgt_width)), np.ones((tgt_height, tgt_width, 1), dtype=np.float32)], axis=-1)
    warped_depth = warped_depth_nearest / np.dot(tgt_uvhomo, np.linalg.inv(transform)[2, :])
    return warped_depth

def warp_normal(normal: np.ndarray, transform: np.ndarray, tgt_size: tuple[int, int], R: np.ndarray):
    """
    Warp the normal map according to the given transformation matrix.
        
    Args:
        normal (np.ndarray): Normal map.
        transform (np.ndarray): Transformation matrix.
        tgt_size (Tuple[int, int]): Target size (height, width).
            
    Returns:
        np.ndarray: Warped normal map.
    """
    tgt_height, tgt_width = tgt_size

    warped_normal = warp_perspective(normal, transform, (tgt_height, tgt_width), interpolation='bilinear')
    warped_normal = warped_normal @ R.T

    norm = np.linalg.norm(warped_normal, axis=-1, keepdims=True)
    warped_normal = np.divide(warped_normal, norm, out=np.zeros_like(warped_normal), where=norm > 1e-6)
    return warped_normal

def warp_pose(pose: np.ndarray, R: np.ndarray):
    pose = pose.copy()
    RT = R.T  # precompute transpose
    pose[:3, :3] = pose[:3, :3] @ RT
    return pose

def image_augmentation(image: np.ndarray, depth: np.ndarray, rng: np.random.Generator, image_augmentation_list: list[str]):
    if 'jittering' in image_augmentation_list:
        image = torch.from_numpy(image).permute(2, 0, 1)
        image = TF.adjust_brightness(image, rng.uniform(0.9, 1.1))
        image = TF.adjust_contrast(image, rng.uniform(0.9, 1.1))
        image = TF.adjust_saturation(image, rng.uniform(0.9, 1.1))
        image = TF.adjust_hue(image, rng.uniform(-0.05, 0.05))
        image = TF.adjust_gamma(image, rng.uniform(0.9, 1.1))
        image = image.permute(1, 2, 0).numpy()

    if 'shot_noise' in image_augmentation_list:
        if rng.uniform() < 0.5: 
            k = np.exp(rng.uniform(np.log(100), np.log(10000))) / 255
            image = (rng.poisson(image * k) / k).clip(0, 255).astype(np.uint8)

    if 'dof' in image_augmentation_list:
        if rng.uniform() < 0.5:
            dof_strength = rng.integers(12)
            with np.errstate(divide='ignore', invalid='ignore'):
                disp = np.where(depth > 0, 1 / depth, 0)
            finite_mask = depth > 0

            if np.any(finite_mask): 
                disp[~finite_mask] = np.nan
                disp_min, disp_max = disp[finite_mask].min(), disp[finite_mask].max()
                disp = cv2.inpaint(np.nan_to_num(disp, nan=1), np.isnan(disp).astype(np.uint8), 3, cv2.INPAINT_TELEA).clip(0, disp_max)
                dof_focus = rng.uniform(disp_min, disp_max)
                image = depth_of_field(image, disp, dof_focus, dof_strength)
    
    if 'jpeg_loss' in image_augmentation_list:
        if rng.uniform() < 0.5:
            image = cv2.imdecode(cv2.imencode('.jpg', image,
                [cv2.IMWRITE_JPEG_QUALITY, rng.integers(20, 100)])[1], cv2.IMREAD_COLOR)
    
    if 'blurring' in image_augmentation_list:
        if rng.uniform() < 0.5:    
            ratio = rng.uniform(0.25, 1)
            height, width = image.shape[:2]
            image = cv2.resize(cv2.resize(image, (int(width * ratio), int(height * ratio)), interpolation=cv2.INTER_AREA), 
                (width, height), interpolation=rng.choice([cv2.INTER_LINEAR_EXACT, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4]))

    return image

def disk_kernel(radius: int) -> np.ndarray:
    """
    Generate disk kernel with given radius.
    
    Args:
        radius (int): Radius of the disk (in pixels).
    
    Returns:
        np.ndarray: (2*radius+1, 2*radius+1) normalized convolution kernel.
    """
    # Create coordinate grid centered at (0,0)
    L = np.arange(-radius, radius + 1)
    X, Y = np.meshgrid(L, L)
    # Generate disk: region inside circle with radius R is 1
    kernel = ((X**2 + Y**2) <= radius**2).astype(np.float32)
    # Normalize the kernel
    kernel /= np.sum(kernel)
    return kernel


def disk_blur(image: np.ndarray, radius: int) -> np.ndarray:
    """
    Apply disk blur to an image using FFT convolution.

    Args:
        image (np.ndarray): Input image, can be grayscale or color.
        radius (int): Blur radius (in pixels).

    Returns:
        np.ndarray: Blurred image.
    """
    if radius == 0:
        return image
    kernel = disk_kernel(radius)
    if image.ndim == 2:
        blurred = fftconvolve(image, kernel, mode='same')
    elif image.ndim == 3:
        channels = []
        for i in range(image.shape[2]):
            blurred_channel = fftconvolve(image[..., i], kernel, mode='same')
            channels.append(blurred_channel)
        blurred = np.stack(channels, axis=-1)
    else:
        raise ValueError("Image must be 2D or 3D.")
    return blurred

def depth_of_field(
    img: np.ndarray, 
    disp: np.ndarray, 
    focus_disp : float, 
    max_blur_radius : int = 10,
) -> np.ndarray:
    """
    Apply depth of field effect to an image.

    Args:
        img (numpy.ndarray): (H, W, 3) input image.
        depth (numpy.ndarray): (H, W) depth map of the scene.
        focus_depth (float): Focus depth of the lens.
        strength (float): Strength of the depth of field effect.
        max_blur_radius (int): Maximum blur radius (in pixels).
        
    Returns:
        numpy.ndarray: (H, W, 3) output image with depth of field effect applied.
    """
    # Precalculate dilated depth map for each blur radius
    max_disp = np.max(disp)
    disp = disp / max_disp
    focus_disp = focus_disp / max_disp
    dilated_disp = []
    for radius in range(max_blur_radius + 1):
        dilated_disp.append(cv2.dilate(disp, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)), iterations=1))
        
    # Determine the blur radius for each pixel based on the depth map
    blur_radii = np.clip(np.abs(disp - focus_disp) * max_blur_radius, 0, max_blur_radius).astype(np.int32)
    for radius in range(max_blur_radius + 1):
        dialted_blur_radii = np.clip(np.abs(dilated_disp[radius] - focus_disp) * max_blur_radius, 0, max_blur_radius).astype(np.int32)
        mask = (dialted_blur_radii >= radius) & (dialted_blur_radii >= blur_radii) & (dilated_disp[radius] > disp)
        blur_radii[mask] = dialted_blur_radii[mask]
    blur_radii = np.clip(blur_radii, 0, max_blur_radius)
    blur_radii = cv2.blur(blur_radii, (5, 5))

    # Precalculate the blurred image for each blur radius
    unique_radii = np.unique(blur_radii)
    precomputed = {}
    for radius in range(max_blur_radius + 1):
        if radius not in unique_radii:
            continue
        precomputed[radius] = disk_blur(img, radius)
        
    # Composite the blurred image for each pixel
    output = np.zeros_like(img)
    for r in unique_radii:
        mask = blur_radii == r
        output[mask] = precomputed[r][mask]
        
    return output
