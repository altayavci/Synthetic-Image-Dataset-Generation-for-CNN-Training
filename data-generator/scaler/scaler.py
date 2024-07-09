import os
import requests
import math

import cv2
import numpy as np
from PIL import Image
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet


upsampler = None


def init():
    global upsampler

    if not os.path.exists("scaler/weights"):
        os.mkdir("scaler/weights")
        url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth"
        response = requests.get(url)
        with open("scaler/weights/RealESRNet_x4plus.pth", "wb") as f:
            f.write(response.content)
    model = RRDBNet(
        num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4
    )
    upsampler = RealESRGANer(
        scale=4,
        model_path="scaler/weights/RealESRNet_x4plus.pth",
        model=model,
        device="cuda",
    )


UPSCALE_PIXEL_THRESHOLD = 1
DOWNSCALE_PIXEL_THRESHOLD = 1


def upscale(image, outscale):
    original_numpy = np.array(image)
    original_opencv = cv2.cvtColor(original_numpy, cv2.COLOR_RGB2BGR)

    output, _ = upsampler.enhance(original_opencv, outscale=outscale)
    upscaled = Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))

    return upscaled


def maybe_upscale(original, outscale, megapixels=1.0):
    original_width, original_height = original.size
    original_pixels = original_width * original_height
    target_pixels = megapixels * 1024 * 1024

    if original_pixels < target_pixels:
        scale_by = math.sqrt(target_pixels / original_pixels)
        target_width = original_width * scale_by
        target_height = original_height * scale_by

        if (
            target_width - original_width >= 1
            or target_height - original_height >= UPSCALE_PIXEL_THRESHOLD
        ):
            upscaled = upscale(original, outscale)
            return upscaled
    return original


def maybe_downscale(original, megapixels=1.0):
    original_width, original_height = original.size
    original_pixels = original_width * original_height
    target_pixels = megapixels * 1024 * 1024

    if original_pixels > target_pixels:
        scale_by = math.sqrt(target_pixels / original_pixels)
        target_width = original_width * scale_by
        target_height = original_height * scale_by

        if (
            original_width - target_width >= 1
            or original_height - target_height >= DOWNSCALE_PIXEL_THRESHOLD
        ):
            target_width = round(target_width)
            target_height = round(target_height)
            downscaled = original.resize((target_width, target_height), Image.LANCZOS)
            return downscaled
    return original


def ensure_resolution(original, outscale, megapixels=1.0):
    return maybe_downscale(maybe_upscale(original, outscale, megapixels), megapixels)
