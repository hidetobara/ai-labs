import argparse
import os
import datetime
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from torchvision import transforms
from PIL import Image
from safetensors.torch import load_file, save_file


LEARNING_RATE = 1e-4
DEVICE = "cuda:0"
DTYPE = torch.bfloat16
TOKEN = os.environ["HF_TOKEN"]
NUM_STEPS = 50
# ノイズ無い状態
TIMESTEP_START = 0
# ノイズだらけの状態 最高1000
TIMESTEP_END = 700
POSITIVE = "(masterpiece), best quality, best composition, "
NEGATIVE = "low quality, bad anatomy, nsfw, many human, credit, sign, cap, lowres, text, error, missing fingers, cropped, worst quality, low quality, jpeg artifacts, signature, watermark, username, missing fingers"

# データセットの定義
class ImageFolderDataset(Dataset):
    def __init__(self, root_dir, tokenizer, size=512):
        self.image_paths = []
        self.prompts = []
        self.size = size
        
        for dirpath, _, filenames in os.walk(root_dir):
            for file in filenames:
                if not file.lower().endswith(('png', 'jpg', 'jpeg')):
                    continue

                cells = []
                for cell in dirpath.split("/")[1:]:
                    if cell.startswith("__"):
                        cells = []
                        break
                    elif cell.startswith("_"):
                        continue
                    cells.append(cell.replace('_', ' '))
                if len(cells) == 0:
                    continue

                prompt = ", ".join(cells)
                img_path = os.path.join(dirpath, file)
                cap_path = Path(img_path).with_suffix(".cap")
                if os.path.exists(cap_path):
                    with open(cap_path, 'r') as f:
                        for line in f:
                            prompt = line.strip()
                            if len(prompt) > 0 and not prompt.startswith("#"):
                                print("prompt found=", prompt)
                                break

                self.image_paths.append(img_path)
                self.prompts.append(prompt)
        print("length=", len(self.image_paths), "size=", size, "an example of paths=", dirpath)
        print("prompts=", set(self.prompts), flush=True)
        
        self.tokenizer = tokenizer
        self.transform = transforms.Compose([
            transforms.Resize(self.size),
            transforms.CenterCrop(self.size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        image = self.transform(image).to(DTYPE)
        text = self.prompts[idx]
        text_input = self.tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=77)
        return {"image": image, "text_input": text_input}

# トレーニングループ
def train(args):
    # モデルのロード（U-Net のみ学習対象）
    model_id = "runwayml/stable-diffusion-v1-5" if args.load_model is None else args.load_model
    pipeline = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=DTYPE, token=TOKEN).to(DEVICE)

    # U-Net だけを Fine-Tuning する
    for param in pipeline.vae.parameters():
        param.requires_grad = False
    for param in pipeline.text_encoder.parameters():
        param.requires_grad = False

    unet = pipeline.unet.to(DEVICE)
    if args.unet:
        unet.load_state_dict(load_file(args.unet, device=DEVICE), strict=False)
    unet.train()

    # フォルダからデータセットをロード
    size = args.image_size if args.image_size else 512
    dataset = ImageFolderDataset(args.images, pipeline.tokenizer, size=size)
    dataloader = DataLoader(dataset, batch_size=args.batch, shuffle=True)

    # オプティマイザとスケジューラ
    optimizer = torch.optim.AdamW(unet.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    start_time = time.time()
    for epoch in range(args.epoch):
        sum_loss = 0
        for batch in dataloader:
            images = batch["image"].to(DEVICE)
            text_input = batch["text_input"]["input_ids"].to(DEVICE)
            
            timesteps = torch.randint(TIMESTEP_START, TIMESTEP_END, (images.shape[0],), device=DEVICE).long()

            latents = pipeline.vae.encode(images).latent_dist.sample()
            latents = latents * 0.18215
            noise = torch.randn_like(latents).to(DEVICE)
            hidden = pipeline.text_encoder(text_input)
            noisy_latents = pipeline.scheduler.add_noise(latents, noise, timesteps).to(DTYPE)
            noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states=hidden.last_hidden_state)["sample"]

            loss = F.mse_loss(noise_pred, noise)
            sum_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        passed = (time.time() - start_time) / 60
        print(f"Epoch [{epoch}], Loss: {sum_loss:.2f}, LR: {scheduler.get_last_lr()[0]:.6f}, Passed: {passed:.1f} min", flush=True)
        scheduler.step()

    # 学習済み U-Net を統合して SD1.5 モデルとして保存
    save_path = args.save_model
    pipeline.unet = unet.to(dtype=DTYPE)
    pipeline.save_pretrained(save_path)
    print(f"Full fine-tuned SD1.5 model saved at {save_path}")

def read_prompts(path) -> list:
    prompt_list = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if len(line) == 0 or line.startswith("#"):
                continue
            prompt_list.append(line)
    return prompt_list

def save_images(images: list, output: str):
    now = datetime.datetime.now()
    header = now.strftime("%y%m%d-%H%M%S")
    for i, image in enumerate(images):
        output_path = f"{output}_{header}_{i:02}.png"
        image.save(output_path)

def generate_image(args):
    if args.prompt:
        prompt_list = [args.prompt]
    elif args.prompt_file:
        prompt_list = read_prompts(args.prompt_file)
    else:
        raise Exception("No prompts !")
    negatives = [NEGATIVE for _ in range(args.batch)]
    size = args.image_size if args.image_size else 512

    # モデルをロード
    if args.init_image:
        init_images = [Image.open(args.init_image).convert("RGB").resize((size, size)) for _ in range(args.batch)]
        pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(args.load_model, torch_dtype=DTYPE, safety_checker=None).to(DEVICE)
        for prompt in prompt_list:
            with torch.autocast(DEVICE, dtype=DTYPE):
                prompts = [POSITIVE + prompt for _ in range(args.batch)]
                images = pipeline(prompts, negative_prompt=negatives, image=init_images, num_inference_steps=NUM_STEPS, guidance_scale=5.0, strength=0.8).images
                save_images(images, args.output)
    else:
        pipeline = StableDiffusionPipeline.from_pretrained(args.load_model, torch_dtype=DTYPE, safety_checker=None).to(DEVICE)
        for prompt in prompt_list:
            with torch.autocast(DEVICE, dtype=DTYPE):
                prompts = [POSITIVE + prompt for _ in range(args.batch)]
                images = pipeline(prompts, negative_prompt=negatives, height=size, width=size, num_inference_steps=NUM_STEPS, guidance_scale=7.0).images
                save_images(images, args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine Tuning SD1.5")
    parser.add_argument('--image_size', default=1024, type=int, help="image size")
    parser.add_argument('--unet', default=None, help="load unet")
    parser.add_argument('--images', help="training images")
    parser.add_argument('--epoch', default=5, type=int, help="epoch")
    parser.add_argument('--batch', default=4, type=int, help="Batch")
    parser.add_argument('--save_model', default="./tuned/sd15", help="save SD1.5")
    parser.add_argument('--load_model', default=None)
    parser.add_argument('--init_image', default=None)
    parser.add_argument('--prompt', default=None)
    parser.add_argument('--prompt_file', default=None)
    parser.add_argument('--output', default="./output/sd", help="dump image path")
    args = parser.parse_args()

    if args.images:
        train(args)
    elif args.load_model and (args.prompt or args.prompt_file):
        generate_image(args)

# python3 train_sd.py --image_size 512 --images images/ --epoch 20 > train.out &
# python3 train_sd.py --load_model ./tuned/sd15/ --prompt_file ./data/test_prompts.txt --image_size 512
# python3 train_sd.py --image_size 768 --images images/ --epoch 50 --load_model ./tuned/sd15d --unet ./models/chilloutmix.safetensors
# python3 train_sd.py --image_size 768 --images images/_v2/ --epoch 20 --load_model ./tuned/sd15_all
# python3 train_sd.py --load_model ./tuned/sd15/ --prompt_file ./data/test_prompts.txt --image_size 768
# python3 train_sd.py --load_model ./tuned/sd15/ --prompt "masterpiece, best quality, girl, anime style, lalafell, she has a magic wand, in the dark ruins, around many colorful crystals"
# python3 train_sd.py --load_model ./tuned/sd15/ --prompt "masterpiece, best quality, 1girl, loli, blue shorts, half zip shirt, dancing, looking at me, solo focus, smile, tight buttocks, growing breasts, clear face, perfect lighting, solo"
# python3 train_sd.py --image_size 768 --images images/_v2/ --epoch 15 --load_model ./tuned/sd15_768/
