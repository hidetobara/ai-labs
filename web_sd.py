import argparse
import io

import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel, DDIMScheduler, AutoencoderKL
from safetensors.torch import load_file
import gradio as gr

class ImageGenerator:
    DTYPE = torch.bfloat16
    DEVICE = "cuda:0"
    MODEL_ID = "runwayml/stable-diffusion-v1-5"
    CONTROLNET_ID = "lllyasviel/sd-controlnet-canny"
    ADD_PROMPT = ", masterpiece, best quality"

    INFERENCE_STEPS = 50
    SIZE = (768, 768)
    RESCALE = 0.18215

    def __init__(self, unet_path=None):
        # ControlNetのロード
        self.controlnet = ControlNetModel.from_pretrained(self.CONTROLNET_ID, torch_dtype=self.DTYPE)

        # Stable Diffusion パイプラインの作成
        self.pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            self.MODEL_ID, controlnet=self.controlnet, torch_dtype=self.DTYPE
        )

        # サンプラーを設定
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
    def preprocess_canny(self, image, canny_filter=32):
        if isinstance(image, str):  # 文字列（ファイルパス）の場合
            image = cv2.imread(image)
        elif isinstance(image, np.ndarray):  # NumPy配列の場合（GradioからのRGB画像）
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        image = cv2.resize(image, self.SIZE)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(image, canny_filter, canny_filter*3)
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)  # RGB 形式に変換
        return Image.fromarray(edges)

    # 初期画像をTensorに変換
    def preprocess_init_image(self, image):
        if isinstance(image, str):  # 文字列（ファイルパス）の場合
            pil_image = Image.open(image).convert("RGB").resize(self.SIZE)
        elif isinstance(image, np.ndarray):  # NumPy配列の場合（GradioからのRGB画像）
            pil_image = Image.fromarray(image).convert("RGB").resize(self.SIZE)
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # 正規化
        ])
        image_tensor = transform(pil_image).unsqueeze(0).to(self.DEVICE, dtype=self.DTYPE)
        return image_tensor

    # 画像を latent 空間にエンコード
    def encode_latents(self, image_tensor):
        with torch.no_grad():
            return self.vae.encode(image_tensor).latent_dist.sample() * self.RESCALE

    def generate(self, input_image, prompt, canny_filter=32, strength=0.9):
        control_image = self.preprocess_canny(input_image, canny_filter)

        # 初期画像を latent に変換
        init_image_tensor = self.preprocess_init_image(input_image)
        init_latent = self.encode_latents(init_image_tensor).squeeze(dim=0)

        # 画像生成 (latent + ControlNet)
        output = self.pipe(
            prompt + self.ADD_PROMPT,
            num_inference_steps=self.INFERENCE_STEPS,  # 全体のステップ数
            control_image=control_image,  # ControlNet用エッジ画像
            image=init_latent,  # Latentを使う
            controlnet_conditioning_scale=0.8,  # ControlNet の影響度
            guidance_scale=7.0,  # クラシックな CFG (高いほどプロンプト重視)
            strength=strength,  # ステップ数に影響、どれだけ元から改変させるか
        )
        return output.images[0]


# Gradioアプリケーションの設定
def create_app(unet_path=None):
    generator = ImageGenerator(unet_path=unet_path)
    
    def process_image(input_image, prompt, canny, strength):
        generated_image = generator.generate(input_image, prompt, canny, strength)
        # PNG形式で保存
        img_byte_arr = io.BytesIO()
        generated_image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        # PILイメージとして返す（Gradioが内部でPNG形式として処理）
        return Image.open(img_byte_arr)
    
    # Gradioインターフェースの作成
    demo = gr.Interface(
        fn=process_image,
        inputs=[
            gr.Image(type="numpy", label="入力画像"),
            gr.Textbox(label="プロンプト", value="watercolor style"),
            gr.Slider(minimum=1, maximum=128, value=16, step=1, label="Canny"),
            gr.Slider(minimum=0.1, maximum=1.0, value=0.9, step=0.05, label="Strength")
        ],
        outputs=gr.Image(type="pil", label="生成画像", format="png"),
        title="Stable Diffusion ControlNet 画像生成",
        description="画像とプロンプトを入力して、Stable Diffusionで画像を生成します。Strengthは元画像からの変化の度合いを制御します。",
        examples=[
            ["/app/sample/watercolor.png", "watercolor style", 0.9],
            ["/app/sample/oil_painting.png", "oil painting style", 0.9],
        ]
    )
    return demo

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stable Diffusion Gradio App")
    parser.add_argument('--unet', default=None, help="load unet")
    args = parser.parse_args()

    app = create_app(unet_path=args.unet)
    app.launch(server_port=7860, server_name='0.0.0.0', share=False)  # shareをTrueにすると一時的な公開URLが生成されます
