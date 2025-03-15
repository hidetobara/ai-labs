import argparse
import os
import math

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPTokenizer
from torchvision import transforms
from PIL import Image
from diffusers import StableDiffusionXLPipeline


DEVICE = "cuda:0"
TOKEN = os.environ["HF_TOKEN"]
NEGATIVE = "low quality, bad anatomy, nsfw, many human"
LEARNING_RATE = 1e-5


# データセットの定義
class ImageFolderDataset(Dataset):
    def __init__(self, root_dir, tokenizer, tokenizer2):
        self.image_paths = []
        self.prompts = []
        
        for dirpath, _, filenames in os.walk(root_dir):
            for file in filenames:
                if file.lower().endswith(('png', 'jpg', 'jpeg')):
                    self.image_paths.append(os.path.join(dirpath, file))
                    prompt = dirpath.split("/")[-1].replace("_", " ")
                    self.prompts.append(prompt)
        print("length=", len(self.image_paths), "prompts=", set(self.prompts))
        
        self.tokenizer = tokenizer
        self.tokenizer2 = tokenizer2
        self.transform = transforms.Compose([
            transforms.Resize((1024, 1024)),  # 直接1024x1024にリサイズ
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),  # RGBチャンネルごとに正規化
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        image = self.transform(image)
        text = self.prompts[idx]
        text_input = self.tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=77)
        text_input_2 = self.tokenizer2(text, return_tensors="pt", padding="max_length", truncation=True, max_length=77)
        return {"image": image, "text_input": text_input, "text_input_2": text_input_2}


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
        device=DEVICE
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def train(args):
    # SDXLモデルをロード
    model_id = "stabilityai/stable-diffusion-xl-base-1.0" if args.load_model is None else args.load_model
    pipeline = StableDiffusionXLPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16, token=TOKEN).to(DEVICE)
    
    # U-Net のみを Fine-Tuning
    for param in pipeline.vae.parameters():
        param.requires_grad = False
    for param in pipeline.text_encoder.parameters():
        param.requires_grad = False
    for param in pipeline.text_encoder_2.parameters():  # SDXLは2つのエンコーダを持つ
        param.requires_grad = False

    unet = pipeline.unet.to(DEVICE)
    if args.unet:
        unet.load_state_dict(torch.load(args.unet, map_location=DEVICE))
    unet.train()

    # データセットの準備
    dataset = ImageFolderDataset(args.images, pipeline.tokenizer, pipeline.tokenizer_2)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # オプティマイザとスケジューラ
    optimizer = torch.optim.AdamW(unet.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    for epoch in range(args.epoch):
        sum_loss = 0
        for batch in dataloader:
            images = batch["image"].to(DEVICE).to(torch.bfloat16)
            text_input = batch["text_input"]["input_ids"].to(DEVICE)
            text_input_2 = batch["text_input_2"]["input_ids"].to(DEVICE)
            
            timesteps = torch.randint(0, 1000, (images.shape[0],), device=DEVICE).long()
            
            latents = pipeline.vae.encode(images).latent_dist.sample() * 0.18215
            noise = torch.randn_like(latents).to(DEVICE)
            hidden_1 = pipeline.text_encoder(text_input)
            hidden_2 = pipeline.text_encoder_2(text_input_2)
            
            print("P1=", hidden_1["pooler_output"].shape)
            print("H2=", hidden_2["last_hidden_state"].shape)
            em1 = hidden_1["last_hidden_state"][-1][-2]
            em2 = hidden_2["last_hidden_state"][-1][-2]
            print("em1=", em1.shape)
            print("em2=", em2.shape)
            text_embedding = torch.cat([em1, em2], dim=2).to(torch.bfloat16)

            def compute_time_ids(original_size, crops_coords_top_left):
                # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
                target_size = (1024, 1024)
                add_time_ids = list(original_size + crops_coords_top_left + target_size)
                add_time_ids = torch.tensor([add_time_ids], device=DEVICE, dtype=torch.bfloat16)
                return add_time_ids

            add_time_ids = torch.cat(
                [compute_time_ids(s, c) for s, c in zip([(1024,1024),(1024,1024)], [(0,0),(0,0)])]
            )

            # Predict the noise residual
            unet_added_conditions = {"time_ids": add_time_ids}
            #prompt_embeds = batch["prompt_embeds"].to(DEVICE, dtype=torch.bfloat16)
            pooled_prompt_embeds = hidden_1["pooler_output"].to(DEVICE)
            unet_added_conditions.update({"text_embeds": pooled_prompt_embeds})

            noisy_latents = pipeline.scheduler.add_noise(latents, noise, timesteps).to(torch.bfloat16)
            # UNet へ入力
            noise_pred = unet(
                noisy_latents,
                timesteps,
                text_embedding,
                added_cond_kwargs=unet_added_conditions,
            ).sample
            
            loss = F.mse_loss(noise_pred, noise)
            sum_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f"Epoch [{epoch}], Loss: {sum_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
        scheduler.step()
    
    # 学習済みU-Netを保存
    save_path = args.save_model
    pipeline.unet = unet.to(dtype=torch.bfloat16)
    pipeline.save_pretrained(save_path)
    print(f"Fine-tuned SDXL model saved at {save_path}")


def generate_image(args):
    BATCH = 4
    prompts = [args.prompt + ", photo award, best shot" for _ in range(BATCH)]
    negatives = [NEGATIVE for _ in range(BATCH)]
    # モデルをロード
    if args.init_image:
        init_images = [Image.open(args.init_image).convert("RGB").resize((1024, 1024)) for _ in range(BATCH)]
    else:
        init_images = None

    pipeline = StableDiffusionXLPipeline.from_pretrained(args.load_model, torch_dtype=torch.bfloat16).to(DEVICE)
    
    # 画像を生成
    with torch.autocast(DEVICE, dtype=torch.bfloat16):
        if init_images:
            # うまく動かない？
            images = pipeline(prompts, negative_prompt=negatives, image=init_images, strength=0.5, num_inference_steps=30, guidance_scale=3.0).images
        else:
            images = pipeline(prompts, negative_prompt=negatives, num_inference_steps=50, guidance_scale=3.0).images

    for i, image in enumerate(images):
        output_path = f"{args.output}_{i}.png"
        image.save(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine Tuning SDXL")
    parser.add_argument('--unet', default=None, help="load unet")
    parser.add_argument('--images', help="training images")
    parser.add_argument('--epoch', default=5, type=int, help="epoch")
    parser.add_argument('--save_model', default="./output/fine_tuned_sdxl", help="save SDXL")
    parser.add_argument('--load_model', default=None)
    parser.add_argument('--init_image', default=None)
    parser.add_argument('--prompt', default="1 girl with blue shorts")
    parser.add_argument('--output', default="./output/sdxl", help="dump image path")
    args = parser.parse_args()

    if args.images:
        train(args)
    if args.load_model and args.prompt:
        generate_image(args)

# python3 train_sdxl.py --images ./images/
# python3 train_sdxl.py --load_model ./output/fine_tuned_sdxl/ --prompt '1 girl with blue shorts'
