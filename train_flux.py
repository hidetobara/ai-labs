import argparse
import os
import datetime
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from diffusers import FluxPipeline
from torchvision import transforms
from PIL import Image
from safetensors.torch import load_file, save_file
import numpy as np


LEARNING_RATE = 1e-5  # Flux用により小さな学習率
DEVICE = "cuda:0"
DTYPE = torch.bfloat16
TOKEN = os.environ.get("HF_TOKEN", None)
NUM_STEPS = 28  # Fluxのデフォルト推論ステップ数
POSITIVE = ""  # Fluxは自然言語プロンプトなのでプレフィックス不要
NEGATIVE = ""  # Fluxはネガティブプロンプトを使用しない

# データセットの定義
class ImageFolderDataset(Dataset):
    def __init__(self, root_dir, tokenizer, size=1024):  # Fluxは1024x1024がデフォルト
        self.image_paths = []
        self.prompts = []
        self.size = size
        
        for dirpath, _, filenames in os.walk(root_dir):
            for file in filenames:
                if file.lower().endswith(('png', 'jpg', 'jpeg')):
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

                    self.image_paths.append(os.path.join(dirpath, file))
                    self.prompts.append(", ".join(cells))
        print("length=", len(self.image_paths), "size=", size, "an example of paths=", dirpath)
        print("prompts=", set(self.prompts), flush=True)
        
        self.tokenizer = tokenizer
        # Fluxのための正規化値を調整
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
        # Fluxは2つのプロンプトを使用するが、同じテキストを使用
        return {"image": image, "prompt": text}

# トレーニングループ
def train(args):
    # Fluxモデルのロード
    model_id = "black-forest-labs/FLUX.1-dev" if args.load_model is None else args.load_model
    pipeline = FluxPipeline.from_pretrained(model_id, torch_dtype=DTYPE, token=TOKEN).to(DEVICE)

    # VAEとテキストエンコーダーは固定、Transformerのみ学習
    for param in pipeline.vae.parameters():
        param.requires_grad = False
    for param in pipeline.text_encoder.parameters():
        param.requires_grad = False
    for param in pipeline.text_encoder_2.parameters():  # Fluxには2つのテキストエンコーダーがある
        param.requires_grad = False

    transformer = pipeline.transformer.to(DEVICE)
    if args.transformer:
        transformer.load_state_dict(load_file(args.transformer, device=DEVICE), strict=False)
    transformer.train()

    # フォルダからデータセットをロード
    size = args.image_size if args.image_size else 1024
    dataset = ImageFolderDataset(args.images, None, size=size)  # tokenizerは不要
    dataloader = DataLoader(dataset, batch_size=args.batch, shuffle=True)

    # オプティマイザとスケジューラ
    optimizer = torch.optim.AdamW(transformer.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    start_time = time.time()
    for epoch in range(args.epoch):
        sum_loss = 0
        for batch in dataloader:
            images = batch["image"].to(DEVICE)
            prompts = batch["prompt"]
            
            # Fluxは1000ステップのスケジューラーを使用
            timesteps = torch.randint(0, 1000, (images.shape[0],), device=DEVICE).long()

            # VAEエンコーディング（Fluxのスケーリング係数を使用）
            with torch.no_grad():
                latents = pipeline.vae.encode(images).latent_dist.sample()
                latents = latents * pipeline.vae.config.scaling_factor
                
                # Fluxのlatent次元を確認してパッキング
                # latents shape: [batch_size, channels, height, width]
                batch_size, channels, height, width = latents.shape
                
                # パッチサイズに基づいてlatentsをreshape
                patch_size = 2  # Fluxのデフォルトパッチサイズ
                latents = latents.view(
                    batch_size, 
                    channels, 
                    height // patch_size, 
                    patch_size, 
                    width // patch_size, 
                    patch_size
                )
                latents = latents.permute(0, 2, 4, 1, 3, 5)
                latents = latents.reshape(
                    batch_size, 
                    (height // patch_size) * (width // patch_size),
                    channels * patch_size * patch_size
                )
            
            noise = torch.randn_like(latents).to(DEVICE)
            
            # テキストエンコーディング（FluxはT5とCLIPの両方を使用）
            with torch.no_grad():
                prompt_embeds, pooled_prompt_embeds, text_ids = pipeline.encode_prompt(
                    prompt=prompts, 
                    prompt_2=prompts,  # 同じプロンプトを使用
                    device=DEVICE, 
                    num_images_per_prompt=1
                )
            
            # Flow matching用のノイズ追加 (直接線形補間)
            # t を [0, 1] の範囲に正規化
            t = timesteps.float() / 1000.0
            t = t.view(-1, 1, 1)  # [batch_size, 1, 1] for broadcasting
            
            # Flow matching: x_t = (1-t) * x_0 + t * noise
            noisy_latents = (1 - t) * latents + t * noise
            noisy_latents = noisy_latents.to(DTYPE)
            
            # Flow matching target: noise - x_0
            target = noise - latents
            
            # Transformerによるノイズ予測
            # 画像の高さと幅を計算（パッチサイズで割った値）
            img_ids = torch.zeros(batch_size, height // patch_size, width // patch_size, 3).to(DEVICE)
            for i in range(height // patch_size):
                for j in range(width // patch_size):
                    img_ids[:, i, j, 1] = i
                    img_ids[:, i, j, 2] = j
            img_ids = img_ids.reshape(batch_size, -1, 3)
            
            # FluxTransformerの正しいパラメータ名を使用
            # pooled_projectionsは不要かもしれないので除外してテスト
            model_pred = transformer(
                hidden_states=noisy_latents,
                timestep=timesteps,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=img_ids,
                return_dict=False
            )[0]

            loss = F.mse_loss(model_pred, target)
            sum_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        passed = (time.time() - start_time) / 60
        print(f"Epoch [{epoch}], Loss: {sum_loss:.2f}, LR: {scheduler.get_last_lr()[0]:.6f}, Passed: {passed:.1f} min", flush=True)
        scheduler.step()

    # 学習済みTransformerを統合してFluxモデルとして保存
    save_path = args.save_model
    pipeline.transformer = transformer.to(dtype=DTYPE)
    pipeline.save_pretrained(save_path)
    print(f"Full fine-tuned Flux model saved at {save_path}")

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
    
    size = args.image_size if args.image_size else 1024

    # Fluxモデルをロード
    pipeline = FluxPipeline.from_pretrained(args.load_model, torch_dtype=DTYPE).to(DEVICE)
    
    for prompt in prompt_list:
        with torch.autocast(DEVICE, dtype=DTYPE):
            # Fluxは単一のプロンプトで動作し、バッチサイズ分繰り返す
            prompts = [POSITIVE + prompt for _ in range(args.batch)]
            images = pipeline(
                prompts, 
                height=size, 
                width=size, 
                num_inference_steps=NUM_STEPS, 
                guidance_scale=3.5,  # Fluxの推奨値
                max_sequence_length=512  # Flux用の長いシーケンス長
            ).images
            save_images(images, args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine Tuning Flux.1")
    parser.add_argument('--image_size', default=1024, type=int, help="image size")
    parser.add_argument('--transformer', default=None, help="load transformer weights")
    parser.add_argument('--images', help="training images")
    parser.add_argument('--epoch', default=5, type=int, help="epoch")
    parser.add_argument('--batch', default=1, type=int, help="Batch (Fluxは大きなメモリを使用するため小さめ)")
    parser.add_argument('--save_model', default="./tuned/flux", help="save Flux model")
    parser.add_argument('--load_model', default=None)
    parser.add_argument('--prompt', default=None)
    parser.add_argument('--prompt_file', default=None)
    parser.add_argument('--output', default="./output/flux", help="dump image path")
    args = parser.parse_args()

    if args.images:
        train(args)
    elif args.load_model and (args.prompt or args.prompt_file):
        generate_image(args)

# 使用例:
# python3 train_flux.py --image_size 1024 --images images/ --epoch 20 --batch 1 --load_model black-forest-labs/FLUX.1-dev
# python3 train_flux.py --load_model ./tuned/flux/ --prompt_file ./data/test_prompts.txt --image_size 1024
# python3 train_flux.py --load_model ./tuned/flux/ --prompt "A beautiful anime girl with magical powers standing in a mystical forest with glowing crystals"