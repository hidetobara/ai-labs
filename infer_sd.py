import argparse

import torch
import cv2
from PIL import Image
from torchvision import transforms
from diffusers import StableDiffusionPipeline, StableDiffusionControlNetImg2ImgPipeline, ControlNetModel, DPMSolverSDEScheduler, DDIMScheduler, AutoencoderKL
from safetensors.torch import load_file


class ImageGenerator:
    DTYPE = torch.bfloat16
    DEVICE = "cuda:0"
    MODEL_ID = "runwayml/stable-diffusion-v1-5"
    CONTROLNET_ID = "lllyasviel/sd-controlnet-canny"
    ADD_PROMPT = ", masterpiece, best quality"

    INFERENCE_STEPS = 50
    SIZE = (512, 512)
    RESCALE = 0.18215

    def __init__(self, unet_path=None):
        # ControlNetのロード
        self.controlnet = ControlNetModel.from_pretrained(self.CONTROLNET_ID, torch_dtype=self.DTYPE)

        # Stable Diffusion パイプラインの作成
        self.pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            self.MODEL_ID, controlnet=self.controlnet, torch_dtype=self.DTYPE
        )

        # SDE系サンプラーを設定
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config, use_karras_sigmas=True)

        # VAE モデル（latent 変換用）
        self.vae = AutoencoderKL.from_pretrained(self.MODEL_ID, subfolder="vae", torch_dtype=self.DTYPE)

        # safety解除
        def null_safety(images, **kwargs):
            return images, [False]
        self.pipe.safety_checker = null_safety

        if unet_path:
            print("UNET=", unet_path)
            new_state_dict = load_file(unet_path)
            self.pipe.unet.load_state_dict(new_state_dict, strict=False)

        # GPUを使用可能なら使用
        self.pipe.to(self.DEVICE)
        self.vae.to(self.DEVICE)

    # Cannyエッジ検出（ControlNet用）
    def preprocess_canny(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.resize(image, self.SIZE, self.SIZE)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(image, 64, 128)
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)  # RGB 形式に変換
        return Image.fromarray(edges)

    # 初期画像を PIL で開き、Tensor に変換
    def preprocess_init_image(self, image_path):
        image = Image.open(image_path).convert("RGB").resize(self.SIZE)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # 正規化
        ])
        image_tensor = transform(image).unsqueeze(0).to(self.DEVICE, dtype=self.DTYPE)
        return image_tensor

    # 画像を latent 空間にエンコード
    def encode_latents(self, image_tensor):
        with torch.no_grad():
            return self.vae.encode(image_tensor).latent_dist.sample() * self.RESCALE

    def generate(self, input_path: str, output_path: str, prompt: str):
        control_image = self.preprocess_canny(input_path)

        # 初期画像を latent に変換
        init_image_tensor = self.preprocess_init_image(input_path)
        init_latent = self.encode_latents(init_image_tensor).squeeze(dim=0)
        #init_image = Image.open(input_path).convert("RGB").resize(self.SIZE)

        # 画像生成 (latent + ControlNet)
        output = self.pipe(
            prompt + self.ADD_PROMPT,
            num_inference_steps=self.INFERENCE_STEPS,  # 全体のステップ数
            control_image=control_image,  # ControlNet用エッジ画像
            #image=init_image, # 画像を使う場合
            image=init_latent, # Latentを使う
            controlnet_conditioning_scale=0.7,  # ControlNet の影響度
            guidance_scale=7.0,  # クラシックな CFG (高いほどプロンプト重視)
            strength=0.9, # ステップ数に影響、どれだけ元から改変させるか
        )
        for n, o in enumerate(output.images):
            o.save(f"{output_path}_{n}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine Tuning SDXL")
    parser.add_argument('--unet', default=None, help="load unet")
    parser.add_argument('--input', default=None, help="input path")
    parser.add_argument('--output', default="output/gen", help="output path")
    parser.add_argument('--prompt', default="watercolor style")
    args = parser.parse_args()

    instance = ImageGenerator(unet_path=args.unet)
    if args.input:
        instance.generate(args.input, output_path=args.output, prompt=args.prompt)
