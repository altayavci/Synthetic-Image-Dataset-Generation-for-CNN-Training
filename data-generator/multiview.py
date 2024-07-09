import torch
from diffusers import (
    DiffusionPipeline,
    EulerAncestralDiscreteScheduler,
    AsymmetricAutoencoderKL,
)
import parameters
import os
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from dotenv import load_dotenv
import gc

from utils.image_utils import all_to_parts
from scaler.scaler import init as init_scaler, ensure_resolution


load_dotenv()
opt = parameters.get()
init_scaler()

if not os.path.exists("scaler/scaled"):
    os.makedirs("scaler/scaled")

ROOT = str(os.getenv("ROOT_PATH"))
SCALED_PATH = os.path.join(ROOT, "scaled")
DATASET_PATH = os.path.join(ROOT, opt.dataset)

pipe_zero123 = DiffusionPipeline.from_pretrained(
    "sudo-ai/zero123plus-v1.2",
    custom_pipeline="sudo-ai/zero123plus-pipeline",
    torch_dtype=torch.float16,
).to("cuda")

pipe_zero123.scheduler = EulerAncestralDiscreteScheduler.from_config(
    pipe_zero123.scheduler.config, timestep_spacing="trailing"
)

pipe_zero123.vae = AsymmetricAutoencoderKL.from_pretrained(
    opt.sd_autoencoder, torch_dtype=torch.float16
).to("cuda")

for path in tqdm(os.listdir(DATASET_PATH), total=len(os.listdir(DATASET_PATH))):
    img_path = os.path.join(DATASET_PATH, path)
    save_path = os.path.splitext(path)[0]

    img_pil = Image.open(img_path).convert("RGB")
    img_pil = ensure_resolution(img_pil, 2, opt.megapixels)
    img_all = pipe_zero123(img_pil, num_inference_steps=75).images[0]
    img_parts = all_to_parts(img_all)
    torch.cuda.empty_cache()
    for i, part in enumerate(img_parts):
        img_part = ensure_resolution(part, 4, opt.megapixels).resize(
            img_pil.size, Image.LANCZOS
        )
        plt.imsave(f"scaler/scaled/{i}_{save_path}.png", np.array(img_part))

torch.cuda.empty_cache()
torch.cuda.empty_cache()
gc.collect()
del pipe_zero123
