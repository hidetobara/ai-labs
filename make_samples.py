import os
import torch
import argparse
import numpy as np
from torch.utils.data import Dataset, DataLoader
from diffusers import StableDiffusionPipeline, DiffusionPipeline, DDPMScheduler
from diffusers.optimization import get_scheduler
from transformers import CLIPTextModel, CLIPTokenizer
from accelerate import Accelerator
from PIL import Image
from tqdm.auto import tqdm
from safetensors.torch import load_file

DEVICE = "cuda:0"
DTYPE = torch.float16

class Sampler:
    ADD_PROMPT = "materpiece, best quality, "

    def __init__(self, output_dir="./tmp", num_samples_per_prompt=10, image_size=1024):
        self.sdxl_pipeline = None
        self.output_dir = output_dir
        self.num_samples_per_prompt = num_samples_per_prompt
        self.image_size = image_size
        
        os.makedirs(output_dir, exist_ok=True)
        
    def generate_images(self, header:str, prompt_list: list[str]):
        """SDXLを使って画像を生成し、保存する"""
        print("Generating images from SDXL...")
        for i, prompt in enumerate(prompt_list):
            print(f"Generating images for prompt: {prompt}")
            for j in range(self.num_samples_per_prompt):
                # SDXLで画像を生成
                image = self.sdxl_pipeline(self.ADD_PROMPT + prompt, height=self.image_size, width=self.image_size, num_inference_steps=35).images[0]
                
                cells = []
                for cell in prompt.split(','):
                    cells.append(cell.strip().replace(" ", "_"))
                name_dir = "/".join(cells)

                # 画像とプロンプトを保存
                os.makedirs(os.path.join(self.output_dir, name_dir), exist_ok=True)
                image.save(os.path.join(self.output_dir, name_dir, f"{header}-{i}-{j}.png"))
    

    def setup(self,
        sdxl_model_id="stabilityai/stable-diffusion-xl-base-1.0",
        sdxl_finetuned_unet_path=None,  # ファインチューニング済みSDXL UNetのパス
    ):
        # SDXLパイプラインをロード
        print("Loading SDXL pipeline...")
        self.sdxl_pipeline = DiffusionPipeline.from_pretrained(
            sdxl_model_id, torch_dtype=DTYPE
        )
        self.sdxl_pipeline.to(DEVICE)
        self.sdxl_pipeline.enable_attention_slicing()
        
        # ファインチューニング済みSDXL UNetをロード（指定されている場合）
        if sdxl_finetuned_unet_path is not None and os.path.exists(sdxl_finetuned_unet_path):
            print(f"Loading fine-tuned SDXL UNet from {sdxl_finetuned_unet_path}")
            # UNetだけを読み込む場合
            new_state_dict = load_file(sdxl_finetuned_unet_path)
            self.sdxl_pipeline.unet.load_state_dict(new_state_dict, strict=False)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SD1.5 from SDXL outputs")
    parser.add_argument("--header", default="sdxl", help="header of sample files")
    parser.add_argument("--prompt_file", type=str, default=None, help="File containing prompts, one per line")
    parser.add_argument("--output_dir", type=str, default="./images/tmp", help="Output directory")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples per prompt")
    parser.add_argument("--image_size", type=int, default=1024, help="Image size")
    parser.add_argument("--sdxl_model", type=str, default="stabilityai/stable-diffusion-xl-base-1.0", help="Base SDXL model ID")
    parser.add_argument("--sdxl_unet", type=str, default=None, help="Path to fine-tuned SDXL UNet")
    
    args = parser.parse_args()
    
    if args.prompt_file:
        # プロンプトファイルを読み込む
        prompt_list = []
        with open(args.prompt_file, "r") as f:
            prompt_list = []
            for line in f.readlines():
                line = line.strip()
                if len(line) == 0 or line.startswith("#"):
                    continue
                prompt_list.append(line)
        sampler = Sampler(output_dir=args.output_dir, num_samples_per_prompt=args.num_samples, image_size=args.image_size)
        sampler.setup(args.sdxl_model, args.sdxl_unet)
        sampler.generate_images(header=args.header, prompt_list=prompt_list)
