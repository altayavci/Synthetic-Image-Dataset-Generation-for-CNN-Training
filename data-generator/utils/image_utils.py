from PIL import Image, ImageChops
import numpy as np
from itertools import product


def trim_white(img: Image):
    bg = Image.new(img.mode, img.size, img.getpixel((0, 0)))
    diff = ImageChops.difference(img, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        img = img.crop(bbox)
    return img


def crop_centered(image: Image, target_size):
    original_width, original_height = image.size
    target_width, target_height = target_size

    left = (original_width - target_width) / 2
    top = (original_height - target_height) / 2
    right = (original_width + target_width) / 2
    bottom = (original_height + target_height) / 2

    return image.crop((left, top, right, bottom))


def mask_init_generator(
    overlay,
    position: tuple = (0, 0),
    scale_f: int = 2,
    crop: int = 30,
    rotation: int = 0,
):
    black_background = Image.new(
        "RGB", (overlay.size[0] * scale_f, overlay.size[1] * scale_f), (0, 0, 0)
    )
    white_background = Image.new("RGB", black_background.size, (255, 255, 255))

    black_box = Image.new(
        "RGB", (overlay.size[0] - crop, overlay.size[1] - crop), (0, 0, 0)
    )
    overlay = overlay.rotate(rotation, expand=True)
    black_box = black_box.rotate(rotation, expand=True)

    black_background.paste(overlay, position)
    white_background.paste(black_box, position)
    return black_background, white_background


def background_kernel(overlay: Image, scale_f: int, crop: int):
    init_imgs, mask_imgs = [], []
    x_axis = np.arange(0, overlay.size[1] * scale_f, overlay.size[1])
    y_axis = np.arange(0, overlay.size[0] * scale_f, overlay.size[0])
    combinations = list(product(y_axis, x_axis))
    for comb in combinations:
        init_img, mask_img = mask_init_generator(overlay, comb, scale_f, crop)
        init_imgs.append(init_img)
        mask_imgs.append(mask_img)
    return (init_imgs, mask_imgs)


def size_decider(img: np.array):
    height, width = img.shape[:-1]
    ratio = height / width
    size = (880, 880)
    if ratio > 1:
        size = (1024, 768)
    elif ratio < 1:
        size = (768, 1024)
    return size


def all_to_parts(img):
    combs = [
        (0, 0, 320, 320),
        (320, 0, 640, 320),
        (0, 320, 320, 640),
        (320, 320, 640, 640),
        (0, 640, 320, 960),
        (320, 640, 640, 960),
    ]
    parts = []
    for comb in combs:
        cropped_img = img.crop(comb)
        parts.append(cropped_img)
    return parts
