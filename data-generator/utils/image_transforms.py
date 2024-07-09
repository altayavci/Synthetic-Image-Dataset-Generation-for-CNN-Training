import imgaug.augmenters as iaa
import numpy as np
from PIL import Image
import torch

from utils.image_utils import trim_white
from segmenter.segmenter import init as init_segmenter, segment
from depth_estimator.depth_estimator import init as init_depth_estimator, get_depth_map

init_segmenter()
init_depth_estimator()


def get_shift_amount(trimmed_img_shape: np.array, img_shape: np.array):
    vertical, horizontal = img_shape - trimmed_img_shape
    horizontal = horizontal // 2
    vertical = vertical // 2
    if (vertical, horizontal) == (0, 0):
        return (0, 0)
    try:  # temporary solution (try-except block)
        vertical = np.random.randint(-1 * int(vertical * 0.8), int(vertical * 0.8))
        horizontal = np.random.randint(
            -1 * int(horizontal * 0.8), int(horizontal * 0.8)
        )
    except ValueError:
        pass
    return (vertical, horizontal)


def apply_image_transform(image: Image):
    # image : scaled
    image_np = np.array(image).astype(np.uint8)
    seq_resize_pad = iaa.Sequential(
        [
            iaa.Resize({"height": image_np.shape[0], "width": image_np.shape[1]}),
            iaa.Pad(px=10, pad_cval=255),
        ]
    )
    img_resized_padded = seq_resize_pad(image=image_np)
    img_pil = Image.fromarray(img_resized_padded.astype(np.uint8)).convert("RGB")

    [img_cropped, img_masked] = segment(img_pil)
    torch.cuda.empty_cache()

    depth_map = get_depth_map(img_pil)
    torch.cuda.empty_cache()

    depth_map_np = np.array(depth_map)
    img_cropped_np = np.array(img_cropped)
    img_masked_np = np.array(img_masked)

    (shift_y, shift_x) = get_shift_amount(
        trimmed_img_shape=np.array(trim_white(img_cropped)).shape[:-1],
        img_shape=np.array(img_cropped_np.shape[:-1]),
    )
    seq_shift = iaa.Affine(translate_px={"x": shift_x, "y": shift_y})

    cropped_shifted = seq_shift(image=img_cropped_np)
    masked_shifted = seq_shift(image=img_masked_np)
    depth_map_shifted = seq_shift(image=depth_map_np)

    cropped_shifted = Image.fromarray(cropped_shifted.astype(np.uint8)).convert("RGBA")
    masked_shifted = Image.fromarray(masked_shifted.astype(np.uint8)).convert("RGB")
    depth_map_shifted = Image.fromarray(depth_map_shifted.astype(np.uint8)).convert(
        "RGB"
    )

    return [cropped_shifted, masked_shifted, depth_map_shifted]
