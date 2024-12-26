import os, sys, argparse, glob

import torch
from PIL import Image
from torchvision import transforms
from diffusers import AutoencoderKL

# VAEモデルのロード
def load_vae_model(model_name="stabilityai/sd-vae-ft-mse"):
    vae = AutoencoderKL.from_pretrained(model_name).to("cuda")
    vae.eval()  # 評価モードに設定
    return vae

# 画像を1024x1024にリサイズしてテンソルに変換
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((512, 512)),  # リサイズ
        transforms.ToTensor(),  # テンソルに変換
        transforms.Normalize([0.5], [0.5])  # 正規化
    ])
    return transform(image).unsqueeze(0)  # バッチ次元を追加

# 画像をLatent表現に変換
def encode_to_latent(vae, image_tensor):
    with torch.no_grad():
        latent = vae.encode(image_tensor.to("cuda")).latent_dist.sample()  # サンプリング
        latent = latent * 0.18215  # スケール調整
    return latent

# Latent表現を画像に戻す関数
def decode_from_latent(vae, latent_tensor):
    with torch.no_grad():
        # スケール調整を戻す
        latent_tensor = latent_tensor / 0.18215
        # VAEを使って画像を生成
        reconstructed_image = vae.decode(latent_tensor).sample

    # 画像の値を[0, 1]の範囲にスケール
    reconstructed_image = (reconstructed_image.clamp(-1, 1) + 1) / 2
    reconstructed_image = reconstructed_image.permute(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)
    reconstructed_image = reconstructed_image.cpu().numpy()  # NumPy配列に変換
    return reconstructed_image

def to_pil_image(reconstructed_image):
    return Image.fromarray((reconstructed_image[0] * 255).astype("uint8"))

# メイン処理
def main(image_folder):
    vae = load_vae_model()
    latents = []
    paths = []
    for ext in ["png", "jpg"]:
        paths.extend(glob.glob(os.path.join(image_folder, f"*.{ext}")))
    for path in paths:
        image_tensor = preprocess_image(path)
        latent = encode_to_latent(vae, image_tensor)
        #print(f"Latent shape: {latent.shape}")
        min_val = latent.min().item()
        max_val = latent.max().item()
        mean_val = latent.mean().item()
        print(f"Latent min: {min_val}, max: {max_val}, average: {mean_val}")
        latents.append(latent)

    stacked_latents = torch.stack(latents, dim=0)  # [N, B, C, H, W]
    average_latent = stacked_latents.mean(dim=0)  # 平均を計算 (N次元を削除)

    image_tensor = decode_from_latent(vae, average_latent)
    image = to_pil_image(image_tensor)
    image.save("output/test.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="実験する")
    parser.add_argument('-i', '--image', default=None, help="画像へのパス")
    parser.add_argument('-f', '--folder', default=None, help="画像フォルダ")
    args = parser.parse_args()

    if args.folder:
        main(args.folder)
