import argparse
import random
import time

from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from src.custom_vae import CustomVAE

# ハイパーパラメータ
LOAD_SIZE = 512
SIZE = 256
batch_size = 12
epochs = 50
learning_rate = 0.001

def train(args):
    def crop_random_images(images):
        if images.shape[2] <= SIZE and images.shape[3] <= SIZE:
            return images
        h = random.randint(0, images.shape[2] - SIZE)
        w = random.randint(0, images.shape[3] - SIZE)
        new_images = images[:, :, h:h+SIZE, w:w+SIZE]
        return new_images

    # データセットのロード
    transform = transforms.Compose([
        transforms.Resize((LOAD_SIZE, LOAD_SIZE)),
        transforms.ToTensor()
    ])
    dataset = ImageFolder(root="/app/images/", transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # モデルの初期化
    vae = CustomVAE().cuda()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(vae.parameters(), lr=learning_rate)
    scheduler = ExponentialLR(optimizer, gamma=0.9)

    # 学習ループ
    start = time.time()
    for epoch in range(epochs):
        for images, _ in dataloader:
            images = crop_random_images(images)
            images = images.cuda()
            optimizer.zero_grad()
            recon = vae(images)
            loss = criterion(recon, images)
            loss.backward()
            optimizer.step()
            passed = time.time() - start
        print(f"Epoch: [{epoch+1}/{epochs}], Loss: {loss.item():.5f}, Time: {passed:.1f} sec.")
        scheduler.step()

    # 学習済みモデルの保存
    vae.save(f"models/my_vae.{args.name}.pth")

def infer(args):
    # モデルのロード
    vae = CustomVAE.load(f"models/my_vae.{args.name}.pth").cuda()
    vae.eval()

    # 画像のロードと前処理
    def preprocess_image(image_path):
        transform = transforms.Compose([
            transforms.Resize((SIZE, SIZE)),
            transforms.ToTensor()
        ])
        image = Image.open(image_path).convert("RGB")
        image = transform(image).unsqueeze(0).cuda()
        return image

    # 画像のエンコード
    def encode_image(image_path):
        image = preprocess_image(image_path)
        with torch.no_grad():
            encoded = vae.encode(image)
        return encoded
    
    # 確認
    def check(tensor):
        print(f"Tensor shape: {tensor.shape}")
        min_val = tensor.min().item()
        max_val = tensor.max().item()
        mean_val = tensor.mean().item()
        std_val = tensor.std().item()
        print(f"Latent min: {min_val}, max: {max_val}, average: {mean_val} std: {std_val}")

    # エンコードされたテンソルを元の画像に戻して保存
    def decode_and_save(encoded_tensor, output_path):
        with torch.no_grad():
            decoded_image = vae.decode(encoded_tensor)
        decoded_image = decoded_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
        decoded_image = (decoded_image * 255).astype("uint8")
        Image.fromarray(decoded_image).save(output_path)

    encoded = encode_image(args.input)
    check(encoded)
    if args.output:
        decode_and_save(encoded, args.output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VAE")
    parser.add_argument('--name', default="512")
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--input', default=None)
    parser.add_argument('--output', default=None)
    args = parser.parse_args()
    
    if args.train:
        train(args)
    if args.input:
        infer(args)
