from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision
from einops import rearrange
from PIL import Image
from torch.utils.data import Dataset

import requests
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from blip_models.blip import blip_decoder
import json
import os

def load_demo_image(image_size,device,img_url):
    
    raw_image = Image.open(img_url).convert('RGB')   

    w,h = raw_image.size
    
    transform = transforms.Compose([
        transforms.Resize((image_size,image_size),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ]) 
    image = transform(raw_image).unsqueeze(0)   
    return image

class EditDataset(Dataset):
    def __init__(
        self,
        path: str,
        split: str = "train",
        splits: tuple[float, float, float] = (0.9, 0.05, 0.05),
        min_resize_res: int = 256,
        max_resize_res: int = 256,
        crop_res: int = 256,
        flip_prob: float = 0.0,
    ):
        assert split in ("train", "val", "test")
        assert sum(splits) == 1
        self.path = path
        self.min_resize_res = min_resize_res
        self.max_resize_res = max_resize_res
        self.crop_res = crop_res
        self.flip_prob = flip_prob

        with open(Path(self.path, "seeds.json")) as f:
            self.seeds = json.load(f)

        split_0, split_1 = {
            "train": (0.0, splits[0]),
            "val": (splits[0], splits[0] + splits[1]),
            "test": (splits[0] + splits[1], 1.0),
        }[split]

        idx_0 = math.floor(split_0 * len(self.seeds))
        idx_1 = math.floor(split_1 * len(self.seeds))
        self.seeds = self.seeds[idx_0:idx_1]
    
        self.blip_model = blip_decoder(pretrained="./blip_models/model__base_caption.pth", image_size=384, vit='base')
        self.blip_model.eval()

    def __len__(self) -> int:
        return len(self.seeds)
    
    def __getitem__(self, i: int) -> dict[str, Any]:
        name = self.seeds[i]
        ir_dir = Path(self.path, "ir")
        rgb_dir = Path(self.path, "rgb")
        seg_dir = Path(self.path, "seg")
        
        image_filename = name[0]
        
        
        image = load_demo_image(image_size=384, device='cuda:1',img_url=os.path.join(rgb_dir,image_filename))
        with torch.no_grad():
            caption = self.blip_model.generate(image, sample=True, top_p=0.9, max_length=20, min_length=5) 
        prompt = "turn the visible image of "+caption[0]+" into infrared"

        image_0 = Image.open(ir_dir.joinpath(name[0]))
        image_1 = Image.open(rgb_dir.joinpath(name[0]))
        image_2 = Image.open(seg_dir.joinpath(name[0].split(".")[0]+".png"))
        
        reize_res = torch.randint(self.min_resize_res, self.max_resize_res + 1, ()).item()
        image_0 = image_0.resize((reize_res, reize_res), Image.Resampling.LANCZOS)
        image_1 = image_1.resize((reize_res, reize_res), Image.Resampling.LANCZOS)
        image_2 = image_2.resize((reize_res, reize_res), Image.Resampling.LANCZOS)

        image_0 = rearrange(2 * torch.tensor(np.array(image_0)).float() / 255 - 1, "h w c -> c h w")
        image_1 = rearrange(2 * torch.tensor(np.array(image_1)).float() / 255 - 1, "h w c -> c h w")
        image_2 = rearrange(2 * torch.tensor(np.array(image_2)).float() / 255 - 1, "h w c -> c h w")

        crop = torchvision.transforms.RandomCrop(self.crop_res)
        flip = torchvision.transforms.RandomHorizontalFlip(float(self.flip_prob))
        image_0, image_1, image_2 = flip(crop(torch.cat((image_0, image_1,image_2)))).chunk(3)

        return dict(edited=image_0, edit=dict(c_concat1=image_1, c_concat2=image_2, c_crossattn=prompt))
