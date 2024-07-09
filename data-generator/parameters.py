import argparse


def get():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=False, type=str, help="root path")
    parser.add_argument(
        "--dataset",
        required=False,
        type=str,
        help="name of the dataset, (must be inside of the root path)",
    )
    parser.add_argument("--sample_dataset", required=False, type=float, default=None)
    parser.add_argument("--megapixels", required=False, type=int, default=1)
    parser.add_argument(
        "--num_per_aug_img",
        required=False,
        type=int,
        default=3,
        help="number of the augmented images",
    )
    parser.add_argument("--multiview", action="store_true")
    parser.add_argument("--background_replacing", action="store_true")
    parser.add_argument("--positive_prompt", required=False, type=str)
    parser.add_argument("--negative_prompt", required=False, type=str, default="")
    parser.add_argument("--num_inference_steps", required=False, type=int, default=30)
    parser.add_argument("--num_per_img_prompt", required=False, type=int, default=2)
    parser.add_argument(
        "--controlnet_conditioning_scale", required=False, type=float, default=0.5
    )
    parser.add_argument("--guidance_scale", required=False, type=float, default=8.0)
    parser.add_argument("--sdxl_refiner", action="store_true")
    parser.add_argument(
        "--sdxl_depth_controlnet",
        required=False,
        type=str,
        default="diffusers/controlnet-depth-sdxl-1.0-small",
    )
    parser.add_argument(
        "--sdxl_autoencoder",
        required=False,
        type=str,
        default="madebyollin/sdxl-vae-fp16-fix",
    )
    parser.add_argument(
        "--sdxl_controlnet",
        required=False,
        type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0",
    )
    parser.add_argument("--scale_factor", required=False, type=int, default=2)
    parser.add_argument("--crop_size", required=False, type=int, default=30)
    parser.add_argument(
        "--sd_inpainting",
        required=False,
        type=str,
        default="stabilityai/stable-diffusion-2-inpainting",
    )
    parser.add_argument(
        "--sd_autoencoder",
        required=False,
        type=str,
        default="cross-attention/asymmetric-autoencoder-kl-x-2",
    )
    parser.add_argument("--seed", required=False, type=int, default=42)
    parser.add_argument(
        "--positive_prompt_suffix_weight", required=False, type=float, default=0.3
    )
    parser.add_argument(
        "--positive_prompt_weight", required=False, type=float, default=0.5
    )
    parser.add_argument(
        "--captioner_prompt_weight", required=False, type=float, default=0.5
    )
    parser.add_argument(
        "--negative_prompt_suffix_weight", required=False, type=float, default=0.6
    )
    parser.add_argument(
        "--negative_prompt_weight", required=False, type=float, default=0.8
    )
    return parser.parse_args()
