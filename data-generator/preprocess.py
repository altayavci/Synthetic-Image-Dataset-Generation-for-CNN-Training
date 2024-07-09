from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import torch
import warnings

warnings.filterwarnings("ignore")
from tqdm import tqdm

from scaler.scaler import init as init_upscaler, ensure_resolution
from masked_depther.masked_depther import get_masked_depth_map
from utils.dataset import get_images
import parameters
from dotenv import load_dotenv


init_upscaler()
opt = parameters.get()
load_dotenv()

if not os.path.exists("scaler/scaled"):
    os.makedirs("scaler/scaled")
if not os.path.exists("segmenter/cropped"):
    os.makedirs("segmenter/cropped")
if not os.path.exists("masked_depther/masked_depth"):
    os.makedirs("masked_depther/masked_depth")
if not os.path.exists("generated"):
    os.makedirs("generated")
if not os.path.exists("outpainted"):
    os.makedirs("outpainted")

ROOT = str(os.getenv("ROOT_PATH"))
DATASET_PATH = os.path.join(ROOT, opt.dataset)
torch.manual_seed(0)


class Preprocess:

    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

    def background_replacing(self):
        imgs = get_images(self.dataset_path, sample_dataset=opt.sample_dataset)
        for path in tqdm(imgs, total=len(imgs)):
            img_path = os.path.join(self.dataset_path, path)
            img = Image.open(img_path).convert("RGB")
            save_path = os.path.splitext(path)[0]
            scaled = ensure_resolution(img, 2, megapixels=opt.megapixels)
            torch.cuda.empty_cache()

            plt.imsave(f"scaler/scaled/{save_path}.png", np.array(scaled))

            for i in range(opt.num_per_aug_img):
                cropped, masked_depth_map = get_masked_depth_map(scaled)
                plt.imsave(f"segmenter/cropped/{i}_{save_path}.png", np.array(cropped))
                plt.imsave(
                    f"masked_depther/masked_depth/{i}_{save_path}.png",
                    np.array(masked_depth_map),
                )

    def multiview(self):
        imgs = get_images(self.dataset_path, sample_dataset=opt.sample_dataset)
        for path in tqdm(imgs, total=len(imgs)):
            img_path = os.path.join(self.dataset_path, path)
            scaled = Image.open(img_path).convert("RGB")
            save_path = os.path.splitext(path)[0]

            for i in range(opt.num_per_aug_img):
                _, masked_depth_map = get_masked_depth_map(scaled)
                plt.imsave(
                    f"masked_depther/masked_depth/{i}_{save_path}.png",
                    np.array(masked_depth_map),
                )

    def basic(self):
        imgs = get_images(self.dataset_path, sample_dataset=opt.sample_dataset)
        for path in tqdm(imgs, total=len(imgs)):
            img_path = os.path.join(self.dataset_path, path)
            img = Image.open(img_path).convert("RGB")
            save_path = os.path.splitext(path)[0]
            scaled = ensure_resolution(img, 2, megapixels=opt.megapixels)
            torch.cuda.empty_cache()

            plt.imsave(f"scaler/scaled/{save_path}.png", np.array(scaled))

            for i in range(opt.num_per_aug_img):
                _, masked_depth_map = get_masked_depth_map(scaled)

                plt.imsave(
                    f"masked_depther/masked_depth/{i}_{save_path}.png",
                    np.array(masked_depth_map),
                )


if __name__ == "__main__":
    worker = Preprocess(DATASET_PATH)
    if opt.multiview:
        worker.multiview()
    elif opt.background_replacing:
        worker.background_replacing()
    else:
        worker.basic()
