from __future__ import annotations

import math
import random
import sys
from tqdm import tqdm
from argparse import ArgumentParser
import einops
import k_diffusion as K
import numpy as np
import torch.nn as nn
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image, ImageOps
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from blip_models.blip import blip_decoder
import os
import cv2
from segment_anything import sam_model_registry, SamPredictor
import torch.distributed as dist


sys.path.append("./stable_diffusion")

from stable_diffusion.ldm.util import instantiate_from_config

import json

def generate_seg(input_image, seg_path, MODEL_PATH):
    DEVICE = f'cuda:{3}' if torch.cuda.is_available() else "cpu"

    # 3. Load model
    sam = sam_model_registry["vit_h"](checkpoint=MODEL_PATH)
    sam.to(device=DEVICE)
    predictor = SamPredictor(sam)

    # 4. Read image & encode
    image = np.array(input_image)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)

    # 5. Generate automatic masks for the whole image
    from segment_anything import SamAutomaticMaskGenerator
    mask_generator = SamAutomaticMaskGenerator(sam)
    masks = mask_generator.generate(image_rgb)

    colored = np.zeros_like(image_rgb)
    for idx, m in enumerate(masks):
        colored[m["segmentation"]] = np.random.randint(0, 255, 3)
    img_pil = Image.fromarray(cv2.cvtColor(colored, cv2.COLOR_RGB2BGR))
    cv2.imwrite(seg_path, cv2.cvtColor(colored, cv2.COLOR_RGB2BGR))
    return img_pil



class CFGDenoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, z, sigma, cond, uncond, text_cfg_scale, image_cfg_scale,seg_cfg_scale):
        cfg_z = einops.repeat(z, "1 ... -> n ...", n=4)
        cfg_sigma = einops.repeat(sigma, "1 ... -> n ...", n=4)
        cfg_cond = {
            "c_crossattn": [torch.cat([cond["c_crossattn"][0], uncond["c_crossattn"][0],uncond["c_crossattn"][0], uncond["c_crossattn"][0]])],
            "c_concat1": [torch.cat([cond["c_concat1"][0], cond["c_concat1"][0], uncond["c_concat1"][0], uncond["c_concat1"][0]])],
            "c_concat2": [torch.cat([cond["c_concat2"][0], cond["c_concat2"][0],cond["c_concat2"][0], uncond["c_concat2"][0]])],
        }
        out_cond, out_img_cond, out_seg_cond, out_uncond = self.inner_model(cfg_z, cfg_sigma, cond=cfg_cond).chunk(4)
        return out_uncond + text_cfg_scale * (out_cond - out_img_cond) + image_cfg_scale * (out_img_cond - out_seg_cond)+seg_cfg_scale * (out_seg_cond - out_uncond)



def get_text_for_image(image_filename, json_file):
    with open(json_file, 'r', encoding='utf-8') as infile:
        image_text_data = json.load(infile)
    
    if image_filename in image_text_data:
        return image_text_data[image_filename]
    else:
        return None

def load_model_from_config(config, ckpt, vae_ckpt=None, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    if vae_ckpt is not None:
        print(f"Loading VAE from {vae_ckpt}")
        vae_sd = torch.load(vae_ckpt, map_location="cpu")["state_dict"]
        sd = {
            k: vae_sd[k[len("first_stage_model.") :]] if k.startswith("first_stage_model.") else v
            for k, v in sd.items()
        }
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)
    return model

def load_demo_image(image_size,img_url):
    
    raw_image = Image.open(img_url).convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize((image_size,image_size),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ]) 
    image = transform(raw_image).unsqueeze(0)   
    return image

def cleanup():
    dist.destroy_process_group()
def infer():
    parser = ArgumentParser()
    parser.add_argument("--resolution", default=512, type=int)
    parser.add_argument("--steps", default=100, type=int)
    parser.add_argument("--config", default="configs/generate.yaml", type=str)
    parser.add_argument("--ckpt", default="weight/after_phase_2.ckpt", type=str)
    parser.add_argument("--blip_decoder_ckpt", default="blip_models/model_base_caption_capfilt_large.pth", type=str)
    parser.add_argument("--sam_ckpt", default="segment-anything-main/segment-anything-main/sam_vit_h_4b8939.pth", type=str)
    parser.add_argument("--vae_ckpt", default=None, type=str)
    parser.add_argument("--input",  default="View_Transformation/AnyMatch_results/image1/", type=str)
    parser.add_argument("--output", default="View_Transformation/AnyMatch_results/image_ir/", type=str)
    parser.add_argument("--output-seg", default="View_Transformation/AnyMatch_results/image_seg/", type=str)
    # parser.add_argument("--input",  default="vis_image/", type=str)
    # parser.add_argument("--output", default="ir_images/", type=str)
    # parser.add_argument("--output-seg", default="vis_image_seg/", type=str)
    parser.add_argument("--edit", default="turn the RGB image into the infrared one",type=str)
    parser.add_argument("--cfg-text", default=1.5, type=float)  #7.5
    parser.add_argument("--cfg-image", default=5.5, type=float)  #1.5
    parser.add_argument("--cfg-seg", default=3.5, type=float)  #1.5
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    model = load_model_from_config(config, args.ckpt, args.vae_ckpt)
    # model = model.eval()
    model = model.eval().to(f'cuda:{3}')

    model_wrap = K.external.CompVisDenoiser(model)
    model_wrap_cfg = CFGDenoiser(model_wrap)

    null_token = model.get_learned_conditioning([""])

    blip_model = blip_decoder(pretrained=args.blip_decoder_ckpt, image_size=384, vit='base')
    blip_model.eval().to(f'cuda:{3}')

    # Get all image files from the input directory
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    # Input and output paths (relative to project root)
    input_dir = os.path.join(project_root, args.input)
    output_dir_ir = os.path.join(project_root, args.output)
    output_dir_seg = os.path.join(project_root, args.output_seg)

    # script_dir = os.path.dirname(os.path.abspath(__file__))
    # input_dir = os.path.join(script_dir, args.input)
    # output_dir_ir = os.path.join(script_dir, args.output)
    # output_dir_seg = os.path.join(script_dir, args.output_seg)

    # Ensure output directories exist
    os.makedirs(output_dir_ir, exist_ok=True)
    os.makedirs(output_dir_seg, exist_ok=True)

    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(image_extensions)]
    image_files.sort()  # Sort to ensure consistent order

    if len(image_files) == 0:
        print(f"Error: No image files found in {input_dir}")
        return

    print(f"Found {len(image_files)} images, starting processing...")

    for img_name in tqdm(image_files, desc="Processing images", unit="img"):
        image_path = os.path.join(input_dir, img_name)
        output_image_path = os.path.join(output_dir_ir, img_name)
        output_seg_path = os.path.join(output_dir_seg, img_name)

        seed = random.randint(0, 100000) if args.seed is None else args.seed

        # Generate caption using BLIP
        image = load_demo_image(image_size=384, img_url=image_path)
        image = image.to(f'cuda:{3}')
        with torch.no_grad():
            caption = blip_model.generate(image, sample=True, top_p=0.9, max_length=20, min_length=5)
        edit_prompt = "turn the visible image of " + caption[0] + " into infrared"

        # Read input image and generate segmentation
        input_image = Image.open(image_path).convert("RGB")

        width0, height0 = input_image.size
        factor = args.resolution / max(width0, height0)
        factor = math.ceil(min(width0, height0) * factor / 64) * 64 / min(width0, height0)
        width = int((width0 * factor) // 64) * 64
        height = int((height0 * factor) // 64) * 64
        input_image = ImageOps.fit(input_image, (width, height), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))

        input_seg = generate_seg(input_image, seg_path=output_seg_path, MODEL_PATH = args.sam_ckpt)
        input_seg = ImageOps.fit(input_seg, (width, height), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))

        cond = {}
        cond["c_crossattn"] = [model.get_learned_conditioning([edit_prompt])]
        input_image_tensor = 2 * torch.tensor(np.array(input_image)).float() / 255 - 1
        input_seg_tensor = 2 * torch.tensor(np.array(input_seg)).float() / 255 - 1
        input_image_tensor = rearrange(input_image_tensor, "h w c -> 1 c h w").to(model.device)
        input_seg_tensor = rearrange(input_seg_tensor, "h w c -> 1 c h w").to(model.device)
        cond["c_concat1"] = [model.encode_first_stage(input_image_tensor).mode()]
        cond["c_concat2"] = [model.encode_first_stage(input_seg_tensor).mode()]

        uncond = {}
        uncond["c_crossattn"] = [null_token]
        uncond["c_concat1"] = [torch.zeros_like(cond["c_concat1"][0])]
        uncond["c_concat2"] = [torch.zeros_like(cond["c_concat2"][0])]

        sigmas = model_wrap.get_sigmas(args.steps)

        extra_args = {
            "cond": cond,
            "uncond": uncond,
            "text_cfg_scale": args.cfg_text,
            "image_cfg_scale": args.cfg_image,
            "seg_cfg_scale": args.cfg_seg,
        }
        torch.manual_seed(seed)
        z = torch.randn_like(cond["c_concat1"][0]) * sigmas[0]
        z = K.sampling.sample_euler_ancestral(model_wrap_cfg, z, sigmas, extra_args=extra_args)
        x = model.decode_first_stage(z)
        x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)
        x = 255.0 * rearrange(x, "1 c h w -> h w c")
        edited_image = Image.fromarray(x.type(torch.uint8).cpu().numpy())
        edited_image = ImageOps.fit(edited_image, (width0, height0), method=Image.Resampling.LANCZOS,
                                    centering=(0.5, 0.5))
        edited_image.save(output_image_path)

    print("All images processed successfully!")

    
def main():
    infer()
if __name__ == "__main__":
    main()