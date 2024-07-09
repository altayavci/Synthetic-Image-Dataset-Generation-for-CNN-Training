import os
import torch
from diffusers import (
    StableDiffusionInpaintPipeline,
    EulerAncestralDiscreteScheduler,
    AsymmetricAutoencoderKL,
)
from compel import Compel
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from tqdm import tqdm
import random

from utils.image_utils import background_kernel, size_decider
import parameters


torch.backends.cuda.matmul.allow_tf32 = True
load_dotenv()
opt = parameters.get()

POSITIVE_PROMPT_SUFFIX = str(os.getenv("POSITIVE_PROMPT_SUFFIX_OUTPAINTING"))
NEGATIVE_PROMPT_SUFFIX = str(os.getenv("NEGATIVE_PROMPT_SUFFIX_OUTPAINTING"))

ROOT = str(os.getenv("ROOT_PATH"))
GENERATED_PATH = os.path.join(ROOT, "generated")


pipe_outpaint = StableDiffusionInpaintPipeline.from_pretrained(
    opt.sd_inpainting,
    torch_dtype=torch.float16,
    # revision="fp16"
).to("cuda")

pipe_outpaint.scheduler = EulerAncestralDiscreteScheduler.from_config(
    pipe_outpaint.scheduler.config
)

pipe_outpaint.vae = AsymmetricAutoencoderKL.from_pretrained(
    opt.sd_autoencoder, torch_dtype=torch.float16
).to("cuda")

compel = Compel(
    tokenizer=pipe_outpaint.tokenizer, text_encoder=pipe_outpaint.text_encoder
)
positive_prompt_embeds = compel(
    f"({opt.positive_prompt},{POSITIVE_PROMPT_SUFFIX}).blend({opt.positive_prompt_weight},{opt.positive_prompt_suffix_weight})"
)
negative_prompt_embeds = compel(
    f"({opt.negative_prompt},{NEGATIVE_PROMPT_SUFFIX}).blend({opt.negative_prompt_weight},{opt.negative_prompt_suffix_weight})"
)

imgs = [img for img in os.listdir(GENERATED_PATH) if ".png" in img]
for img in tqdm(imgs, total=len(imgs)):
    img_path = os.path.join(GENERATED_PATH, img)
    img_pil = Image.open(img_path).convert("RGB")
    inits, masks = background_kernel(
        img_pil, scale_f=opt.scale_factor, crop=opt.crop_size
    )
    height, width = size_decider(np.array(img_pil))
    seed = random.randint(0, 2**63 - 1)
    generator = torch.Generator(device="cuda").manual_seed(seed)
    torch.cuda.empty_cache()

    generated_imgs = pipe_outpaint(
        prompt_embeds=positive_prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        num_images_per_prompt=(opt.scale_factor) ** 2,
        image=inits,
        mask_image=masks,
        height=height,
        width=width,
        generator=generator,
        num_inference_steps=opt.num_inference_steps,
    ).images

    for i, gen_img in enumerate(generated_imgs):
        plt.imsave(f"outpainted/{i}_{img}", np.array(gen_img))
torch.cuda.empty_cache()
