import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

depth_estimator = None
feature_extractor = None


def init():
    global depth_estimator, feature_extractor
    feature_extractor = AutoImageProcessor.from_pretrained(
        "LiheYoung/depth-anything-large-hf"
    )
    depth_estimator = AutoModelForDepthEstimation.from_pretrained(
        "LiheYoung/depth-anything-large-hf"
    ).to("cuda")


@torch.inference_mode()
def get_depth_map(image: Image) -> Image:
    image_to_depth = feature_extractor(images=image, return_tensors="pt").to("cuda")
    with torch.no_grad():
        depth_map = depth_estimator(**image_to_depth).predicted_depth

    width, height = image.size
    depth_map = torch.nn.functional.interpolate(
        depth_map.unsqueeze(1).float(),
        size=(height, width),
        mode="bicubic",
        align_corners=False,
    )
    depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_map = (depth_map - depth_min) / (depth_max - depth_min)
    image = torch.cat([depth_map] * 3, dim=1)

    image = image.permute(0, 2, 3, 1).cpu().numpy()[0]
    image = Image.fromarray((image * 255.0).clip(0, 255).astype(np.uint8))
    return image
