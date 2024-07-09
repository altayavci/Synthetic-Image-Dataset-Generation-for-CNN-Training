from scipy.ndimage import binary_dilation
from PIL import ImageFilter, Image
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import torch
import warnings

warnings.filterwarnings("ignore")

from utils.image_transforms import apply_image_transform


def get_masked_depth_map(img: Image):
    [cropped, crop_mask, depth_map] = apply_image_transform(img)

    crop_mask_np = np.array(crop_mask.convert("L"))
    crop_mask_binary = crop_mask_np > 128

    dilated_mask = binary_dilation(crop_mask_binary, iterations=10)
    dilated_mask = Image.fromarray((dilated_mask * 255).astype(np.uint8))
    dilated_mask_blurred = dilated_mask.filter(ImageFilter.GaussianBlur(radius=10))
    dilated_mask_blurred_np = np.array(dilated_mask_blurred) / 255.0

    depth_map_np = np.array(depth_map.convert("L")) / 255.0
    masked_depth_map_np = depth_map_np * dilated_mask_blurred_np
    masked_depth_map_np = (masked_depth_map_np * 255).astype(np.uint8)
    masked_depth_map = Image.fromarray(masked_depth_map_np).convert("RGB")

    return cropped, masked_depth_map
