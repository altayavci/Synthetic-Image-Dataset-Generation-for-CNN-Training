from diffusers import (
    ControlNetModel,
    AutoencoderKL,
    StableDiffusionXLControlNetPipeline,
    UniPCMultistepScheduler,
    StableDiffusionXLImg2ImgPipeline,
)
import torch
from compel import Compel, ReturnedEmbeddingsType
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import warnings

warnings.filterwarnings("ignore")
from tqdm import tqdm
import random
import gc

from captioner.captioner import init as init_captioner, derive_caption
from utils.image_utils import crop_centered
import parameters
from dotenv import load_dotenv

init_captioner()
load_dotenv()
opt = parameters.get()

POSITIVE_PROMPT_SUFFIX = str(os.getenv("POSITIVE_PROMPT_SUFFIX_BACKGROUND_REPLACER"))
NEGATIVE_PROMPT_SUFFIX = str(os.getenv("NEGATIVE_PROMPT_SUFFIX_BACKGROUND_REPLACER"))
ROOT = str(os.getenv("ROOT_PATH"))

SCALED_PATH = os.path.join(ROOT, "scaler/scaled")
MASKED_DEPTH_PATH = os.path.join(ROOT, "masked_depther/masked_depth")
GENERATED_PATH = os.path.join(ROOT, "generated")

SDXL_NUM_INFERENCE_STEPS = 30
REFINER_NUM_INFERENCE_STEPS = 45

depth_controlnet = ControlNetModel.from_pretrained(
    opt.sdxl_depth_controlnet, use_safetensors=True, torch_dtype=torch.float16
).to("cuda")

vae = AutoencoderKL.from_pretrained(opt.sdxl_autoencoder, torch_dtype=torch.float16).to(
    "cuda"
)

scheduler = UniPCMultistepScheduler.from_config(
    opt.sdxl_controlnet, subfolder="scheduler"
)
pipe_controlnet_sdxl = StableDiffusionXLControlNetPipeline.from_pretrained(
    opt.sdxl_controlnet,
    controlnet=[depth_controlnet],
    scheduler=scheduler,
    vae=vae,
    variant="fp16",
    use_safetensors=True,
    torch_dtype=torch.float16,
).to("cuda")

if opt.sdxl_refiner:
    pipe_controlnet_sdxl = StableDiffusionXLControlNetPipeline.from_pretrained(
        opt.sdxl_controlnet,
        controlnet=[depth_controlnet],
        scheduler=scheduler,
        vae=vae,
        variant="fp16",
        use_safetensors=True,
        torch_dtype=torch.float16,
        output_type="latent",
    ).to("cuda")
    pipe_refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        text_encoder_2=pipe_controlnet_sdxl.text_encoder_2,
        vae=pipe_controlnet_sdxl.vae,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    ).to("cuda")

pipe_controlnet_sdxl.enable_model_cpu_offload()
pipe_controlnet_sdxl.enable_attention_slicing()
pipe_controlnet_sdxl.enable_xformers_memory_efficient_attention()

for path in tqdm(
    os.listdir(MASKED_DEPTH_PATH), total=len(os.listdir(MASKED_DEPTH_PATH))
):
    scaled_img_path = os.path.join(SCALED_PATH, "_".join(path.split("_")[1:]))
    masked_depth_img_path = os.path.join(MASKED_DEPTH_PATH, path)

    scaled_img = Image.open(scaled_img_path).convert("RGB")
    masked_depth_img = Image.open(masked_depth_img_path).convert("RGB")

    caption = derive_caption(scaled_img)
    torch.cuda.empty_cache()

    compel = Compel(
        tokenizer=[pipe_controlnet_sdxl.tokenizer, pipe_controlnet_sdxl.tokenizer_2],
        text_encoder=[
            pipe_controlnet_sdxl.text_encoder,
            pipe_controlnet_sdxl.text_encoder_2,
        ],
        returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
        requires_pooled=[False, True],
    )
    final_positive_prompt = [
        f"({caption}){opt.captioner_prompt_weight}, ({opt.positive_prompt}){opt.positive_prompt_weight}, ({POSITIVE_PROMPT_SUFFIX}){opt.positive_prompt_suffix_weight}"
    ]
    positive_prompt_embds, pooled_positive_prompt_embds = compel(final_positive_prompt)
    final_negative_prompt = [
        f"({opt.negative_prompt}){opt.negative_prompt_weight}, ({NEGATIVE_PROMPT_SUFFIX}){opt.negative_prompt_suffix_weight}"
    ]
    negative_prompt_embds, pooled_negative_prompt_embds = compel(final_negative_prompt)

    seed = random.randint(0, 2**63 - 1)
    generator = torch.Generator(device="cuda").manual_seed(seed)

    if opt.sdxl_refiner:
        compel_refiner = Compel(
            tokenizer=pipe_refiner.tokenizer_2,
            text_encoder=pipe_refiner.text_encoder_2,
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=True,
        )
        refiner_prompt_embeds, refiner_pooled_prompt_embeds = compel_refiner(
            final_positive_prompt
        )
        refiner_negative_prompt_embeds, refiner_negative_pooled_prompt_embeds = (
            compel_refiner(final_negative_prompt)
        )

    generated_images = pipe_controlnet_sdxl(
        prompt_embeds=positive_prompt_embds,
        pooled_prompt_embeds=pooled_positive_prompt_embds,
        negative_prompt_embeds=negative_prompt_embds,
        negative_pooled_prompt_embeds=pooled_negative_prompt_embds,
        num_inference_steps=SDXL_NUM_INFERENCE_STEPS,
        num_images_per_prompt=opt.num_per_img_prompt,
        controlnet_conditioning_scale=opt.controlnet_conditioning_scale,
        guidance_scale=opt.guidance_scale,
        generator=generator,
        image=[masked_depth_img],
    ).images

    for i, generated_img in enumerate(generated_images):
        if opt.sdxl_refiner:
            generated_img = pipe_refiner(
                prompt_embeds=refiner_prompt_embeds,
                pooled_prompt_embeds=refiner_pooled_prompt_embeds,
                negative_prompt_embeds=refiner_negative_prompt_embeds,
                negative_pooled_prompt_embeds=refiner_negative_pooled_prompt_embeds,
                generator=generator,
                num_inference_steps=REFINER_NUM_INFERENCE_STEPS,
                strength=(REFINER_NUM_INFERENCE_STEPS - SDXL_NUM_INFERENCE_STEPS)
                / REFINER_NUM_INFERENCE_STEPS,
                guidance_scale=opt.guidance_scale,
                image=generated_img,
            ).images[0]

        if opt.background_replacing:
            cropped_img_path = os.path.join(
                os.path.join(ROOT, "segmenter/cropped"), path
            )
            cropped_img = Image.open(cropped_img_path)
            crop_centered_img = crop_centered(cropped_img, generated_img.size).convert(
                "RGBA"
            )
            generated_img = Image.alpha_composite(
                generated_img.convert("RGBA"), crop_centered_img
            )
        plt.imsave(f"generated/{i}_{path}", np.array(generated_img))

    if opt.background_replacing:
        os.remove(cropped_img_path)
    os.remove(masked_depth_img_path)

torch.cuda.empty_cache()
gc.collect()
del pipe_controlnet_sdxl
