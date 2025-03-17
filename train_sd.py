import argparse
import os
import datetime

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from torchvision import transforms
from PIL import Image
from safetensors.torch import load_file

DEVICE = "cuda:0"
TOKEN = os.environ["HF_TOKEN"]
NEGATIVE = "low quality, bad anatomy, nsfw, many human"

# データセットの定義
class ImageFolderDataset(Dataset):
    def __init__(self, root_dir, tokenizer, size=512):
        self.image_paths = []
        self.prompts = []
        self.size = size
        
        for dirpath, _, filenames in os.walk(root_dir):
            for file in filenames:
                if file.lower().endswith(('png', 'jpg', 'jpeg')):
                    self.image_paths.append(os.path.join(dirpath, file))
                    cells = dirpath.split("/")
                    if len(cells) < 3:
                        prompt = cells[-1]
                    else:
                        prompt = cells[-2] + ", " + cells[-1]
                    self.prompts.append(prompt.replace("_", " "))
        print("length=", len(self.image_paths), "size=", size, "path=", dirpath)
        print("prompts=", set(self.prompts))
        
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
        image = self.transform(image).to(torch.bfloat16)
        text = self.prompts[idx]
        text_input = self.tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=77)
        return {"image": image, "text_input": text_input}

# トレーニングループ
def train(args):
    # モデルのロード（U-Net のみ学習対象）
    model_id = "runwayml/stable-diffusion-v1-5" if args.load_model is None else args.load_model
    pipeline = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16, token=TOKEN).to(DEVICE)

    # U-Net だけを Fine-Tuning する
    for param in pipeline.vae.parameters():
        param.requires_grad = False
    for param in pipeline.text_encoder.parameters():
        param.requires_grad = False

    unet = pipeline.unet.to(DEVICE)
    if args.unet:
        unet.load_state_dict(load_file(args.unet, device=DEVICE))
    unet.train()

    # フォルダからデータセットをロード
    size = 256 if args.layout else 512
    dataset = ImageFolderDataset(args.images, pipeline.tokenizer, size=size)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # オプティマイザとスケジューラ
    optimizer = torch.optim.AdamW(unet.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    for epoch in range(args.epoch):
        sum_loss = 0
        for batch in dataloader:
            images = batch["image"].to(DEVICE)
            text_input = batch["text_input"]["input_ids"].to(DEVICE)
            
            timesteps = torch.randint(0, 1000, (images.shape[0],), device=DEVICE).long()

            latents = pipeline.vae.encode(images).latent_dist.sample()
            latents = latents * 0.18215
            noise = torch.randn_like(latents).to(DEVICE)
            hidden = pipeline.text_encoder(text_input)
            noisy_latents = pipeline.scheduler.add_noise(latents, noise, timesteps).to(torch.bfloat16)
            noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states=hidden.last_hidden_state)["sample"]

            loss = F.mse_loss(noise_pred, noise)
            sum_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        print(f"Epoch [{epoch}], Loss: {sum_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
        scheduler.step()

    # 学習済み U-Net を統合して SD1.5 モデルとして保存
    save_path = args.save_model
    pipeline.unet = unet.to(dtype=torch.bfloat16)
    pipeline.save_pretrained(save_path)
    print(f"Full fine-tuned SD1.5 model saved at {save_path}")

def generate_image(args):
    BATCH = 6
    prompts = [args.prompt + ", (masterpiece), best shot" for _ in range(BATCH)]
    negatives = [NEGATIVE for _ in range(BATCH)]

    now = datetime.datetime.now()
    header = now.strftime("%y%m%d-%H%M")

    size = 256 if args.layout else 768

    # モデルをロード
    if args.init_image:
        init_images = [Image.open(args.init_image).convert("RGB").resize((size, size)) for _ in range(BATCH)]
        pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(args.load_model, torch_dtype=torch.bfloat16, safety_checker=None).to(DEVICE)
        with torch.autocast(DEVICE, dtype=torch.bfloat16):
            images = pipeline(prompts, negative_prompt=negatives, image=init_images, num_inference_steps=50, guidance_scale=5.0, strength=0.75).images
    else:
        pipeline = StableDiffusionPipeline.from_pretrained(args.load_model, torch_dtype=torch.bfloat16, safety_checker=None).to(DEVICE)
        with torch.autocast(DEVICE, dtype=torch.bfloat16):
            images = pipeline(prompts, negative_prompt=negatives, height=size, width=size, num_inference_steps=50, guidance_scale=7.0).images

    for i, image in enumerate(images):
        output_path = f"{args.output}_{header}_{i:02}.png"
        image.save(output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine Tuning SD1.5")
    parser.add_argument('--layout', action="store_true")
    parser.add_argument('--style', action="store_true")
    parser.add_argument('--unet', default=None, help="load unet")
    parser.add_argument('--images', help="training images")
    parser.add_argument('--epoch', default=10, type=int, help="epoch")
    parser.add_argument('--save_model', default="./models/sd15", help="save SD1.5")
    parser.add_argument('--load_model', default=None)
    parser.add_argument('--init_image', default=None)
    parser.add_argument('--prompt', default="1 girl with blue shorts")
    parser.add_argument('--output', default="./output/sd", help="dump image path")
    args = parser.parse_args()

    if args.images:
        train(args)
    if args.prompt:
        generate_image(args)

# python3 train_sd.py --images ./images/
# python3 train_sd.py --load_model ./output/fine_tuned_sd15/ --prompt '1 girl with blue shorts'