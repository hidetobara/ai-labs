import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel, DPMSolverSDEScheduler, DDIMScheduler, AutoencoderKL

DTYPE = torch.bfloat16

# モデルのロード
model_id = "runwayml/stable-diffusion-v1-5"
controlnet_id = "lllyasviel/sd-controlnet-canny"

# ControlNetのロード
controlnet = ControlNetModel.from_pretrained(controlnet_id, torch_dtype=DTYPE)

# Stable Diffusion パイプラインの作成
pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
    model_id, controlnet=controlnet, torch_dtype=DTYPE
)

# SDE系サンプラーを設定
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)

# VAE モデル（latent 変換用）
vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", torch_dtype=DTYPE)

# GPUを使用可能なら使用
DEVICE = "cuda:0"
pipe.to(DEVICE)
vae.to(DEVICE)

# Cannyエッジ検出（ControlNet用）
def preprocess_canny(image_path, image_size=(512,512)):
    image = cv2.imread(image_path)
    image = cv2.resize(image, image_size)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(image, 50, 100)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)  # RGB 形式に変換
    cv2.imwrite("canny.png", edges)
    return Image.fromarray(edges)

# 初期画像を PIL で開き、Tensor に変換
def preprocess_init_image(image_path):
    image = Image.open(image_path).convert("RGB").resize((512, 512))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # 正規化
    ])
    image_tensor = transform(image).unsqueeze(0).to(DEVICE, dtype=DTYPE)
    return image_tensor

# 画像を latent 空間にエンコード
def encode_latents(image_tensor):
    RESCALE = 0.18215
    #RESCALE = 1.0
    with torch.no_grad():
        latents = vae.encode(image_tensor).latent_dist.sample() * RESCALE
    return latents

# 入力画像を処理
init_image_path = "output/sdxl_3.png"

control_image = preprocess_canny(init_image_path)
init_image_tensor = preprocess_init_image(init_image_path)

# 初期画像を latent に変換
init_latents = encode_latents(init_image_tensor)

# プロンプト設定
prompt = "oilpainting style, masterpiece, best quality"

def null_safety(images, **kwargs):
    return images, [False]
pipe.safety_checker = null_safety

# 途中のステップから開始
num_inference_steps = 30  # 全体のステップ数
start_step = 5  # 途中のステップ（例: 30ステップ目から再開）

pipe.scheduler.set_timesteps(num_inference_steps)
pipe.scheduler.steps_offset = start_step

# 指定したステップから開始するための `t_index` を取得
t_index = num_inference_steps - start_step  # 例: 50ステップ中30ステップ目から開始
pipe.scheduler.timesteps = pipe.scheduler.timesteps[t_index:]  # 残りのタイムステップ
#print("S=", pipe.scheduler.timesteps)

# 画像生成 (latent + ControlNet)
output = pipe(
    prompt,
    num_inference_steps=num_inference_steps,  # 全体のステップ数
    control_image=control_image,  # ControlNet用エッジ画像
    image=Image.open(init_image_path).convert("RGB").resize((512, 512)),
    controlnet_conditioning_scale=0.9,  # ControlNet の影響度
    guidance_scale=7.0,  # クラシックな CFG (高いほどプロンプト重視)
    strength=0.9, # ステップ数に影響
#    latents=init_image_tensor, # 初期画像
).images[0]

# 画像を保存
output.save("output/refine.png")

print("画像を生成しました: output.png")
