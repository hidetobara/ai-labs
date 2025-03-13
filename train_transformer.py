import os, sys, glob, argparse, random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader, Dataset

from PIL import Image
from torchvision import transforms
from diffusers import AutoencoderKL

from src.custom_vae import CustomVAE

# 設定
VAE_PATH = "./models/my_vae.512.pth"

# Transformer Encoder-Decoder Model
class TransformerLatentModel(nn.Module):
    def __init__(self, latent_dim, num_heads, num_layers, hidden_dim):
        super(TransformerLatentModel, self).__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=latent_dim, nhead=num_heads, dim_feedforward=hidden_dim, batch_first=True),
            num_layers=num_layers,
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=latent_dim, nhead=num_heads, dim_feedforward=hidden_dim, batch_first=True),
            num_layers=num_layers,
        )
        self.latent_proj = nn.Linear(latent_dim, latent_dim)

    def forward(self, src, tgt):
        # Encoder
        memory = self.encoder(src)
        # Decoder
        output = self.decoder(memory, tgt)
        # Project to Latent Space
        output = self.latent_proj(output)
        return output

# Custom Dataset
class LatentDataset(Dataset):
    def __init__(self, latent_inputs, latent_outputs, fn=None):
        if fn is None:
            fn = lambda x: x
        # [Batch, Channels, Height, Width] を Transformer の形式に変換
        self.inputs = [fn(tensor) for tensor in latent_inputs]
        self.outputs = [fn(tensor) for tensor in latent_outputs]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]

class Trainer:
    LATENT_DIM = 512  # Embedding Dim
    NUM_HEADS = 16
    NUM_LAYERS = 16
    HIDDEN_DIM = 128
    IMAGE_SIZE = 256
    LATENT_SIZE = 16
    SEQUENCE_LENGTH = LATENT_SIZE * LATENT_SIZE

    def __init__(self):
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.vae = CustomVAE.load(VAE_PATH).to(self.DEVICE)
        self.vae.eval()

        self.model = TransformerLatentModel(self.LATENT_DIM, self.NUM_HEADS, self.NUM_LAYERS, self.HIDDEN_DIM).to(self.DEVICE)

    def reshape_latents(self, latents):
        """
        Latent tensor [Channels, Height, Width] を
        Transformer用の形式 [Sequence Length, Embedding Dim] に変換
        """
        batch, channels, height, width = latents.shape
        sequence_length = height * width
        reshaped_tensor = latents.permute(0, 2, 3, 1).reshape(batch, sequence_length, channels)
        return reshaped_tensor

    def reshape_back_latents(self, tensors, height, width):
        """
        Transformer用の形式 [Sequence Length, Embedding Dim] を
        [Channels, Height, Width] に戻す
        """
        batch, sequence_length, channels = tensors.shape
        reshaped_tensors = tensors.reshape(batch, height, width, channels).permute(0, 3, 1, 2)
        return reshaped_tensors
    
    def check_latent(self, latent):
        print(f"Latent shape: {latent.shape}")
        min_val = latent.min().item()
        max_val = latent.max().item()
        mean_val = latent.mean().item()
        std_val = latent.std().item()
        print(f"Latent min: {min_val}, max: {max_val}, average: {mean_val} std: {std_val}")

    def generate_positional_encoding(self, target):
        """
        Generate sinusoidal positional encoding for a 3D latent tensor.
        Args:
            C: Number of channels in the latent representation.
            H: Height of the latent representation.
            W: Width of the latent representation.
            device: Device (CPU or GPU) to use for the positional encoding.
        Returns:
            A positional encoding tensor of shape [C, H, W].
        """
        B, C, H, W = target.shape
        # Create a grid of positions
        y_embed = torch.arange(0, H, device=self.DEVICE).unsqueeze(1).repeat(1, W)  # Shape: [H, W]
        x_embed = torch.arange(0, W, device=self.DEVICE).unsqueeze(0).repeat(H, 1)  # Shape: [H, W]

        # Normalize positions to range [0, 1]
        y_embed = y_embed / H
        x_embed = x_embed / W

        # Initialize positional encoding tensor
        pe = torch.zeros(B, C, H, W, device=self.DEVICE)  # Shape: [C, H, W]

        # Compute sinusoidal positional encoding
        for i in range(C // 2):
            div_term = 10000 ** (2 * i / C)
            pe[:, 2 * i, :, :] = torch.sin(x_embed / div_term)  # Sin for x
            pe[:, 2 * i + 1, :, :] = torch.cos(y_embed / div_term)  # Cos for y
        return pe

    # 画像をリサイズしてテンソルに変換
    def preprocess_image(self, image_path, width=None):
        if width is None:
            width = self.IMAGE_SIZE
        image = Image.open(image_path).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize((width, width)),  # リサイズ
            transforms.ToTensor(),  # テンソルに変換
            #transforms.Normalize([0.5], [0.5])  # 正規化
        ])
        tensor = transform(image)
        #print("origin RGB=", tensor[0,0,:,:].mean(), tensor[0,1,:,:].mean(), tensor[0,2,:,:].mean())
        return tensor

    # 画像をLatent表現に変換
    def encode_to_token_latents(self, image_tensors):
        with torch.no_grad():
            e = self.vae.encode(image_tensors.to(self.DEVICE))
            t = self.vae.to_token(e)
            latents = self.vae.from_token(t)
        return latents

    def encode_to_latents(self, image_tensors):
        with torch.no_grad():
            latents = self.vae.encode(image_tensors.to(self.DEVICE))
        return latents

    # Latent表現を画像に戻す関数
    def decode_from_latents(self, latent_tensors):
        with torch.no_grad():
            # VAEを使って画像を生成
            reconstructed_images = self.vae.decode(latent_tensors)

        # 画像の値を[0, 1]の範囲にスケール
        mean_val = reconstructed_images.mean().item()
        std_val = reconstructed_images.std().item()
        print("MEAN, STD=", mean_val, std_val)
        reconstructed_images = (reconstructed_images - mean_val) / (std_val * 2)
        reconstructed_images = (reconstructed_images.clamp(-1, 1) + 1) / 2
        #reconstructed_image = torch.reshape(reconstructed_image.cpu(), (1, 3, self.ORIGIN_WIDTH, self.ORIGIN_WIDTH))
        return reconstructed_images

    def crop_random_images(self, images):
        if images.shape[2] <= self.IMAGE_SIZE and images.shape[3] <= self.IMAGE_SIZE:
            return images
        h = random.randint(0, images.shape[2] - self.IMAGE_SIZE)
        w = random.randint(0, images.shape[3] - self.IMAGE_SIZE)
        new_images = images[:, :, h:h+self.IMAGE_SIZE, w:w+self.IMAGE_SIZE]
        return new_images

    def to_pil_image(self, tensor):
        array = tensor.cpu().permute(1, 2, 0).numpy()
        return Image.fromarray((array * 255).astype("uint8"))

    def load_images(self, folder, size=256):
        images = [] # (N, C, H, W)
        paths = []
        for ext in ["png", "jpg", "JPG"]:
            paths.extend(glob.glob(os.path.join(folder, f"*.{ext}")))
        for path in paths:
            image = self.preprocess_image(path, size)
            images.append(image)
        return images

    # Training Loop
    def train_model(self, dataloader, optimizer, scheduler, loss_fn, epochs=10):
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for src, dst in dataloader:
                # ランダムに刈り取る
                cropped_src = self.crop_random_images(src)
                cropped_dst = cropped_src
                # latentsへ
                latents_src = self.encode_to_latents(cropped_src)
                latents_dst = self.encode_to_latents(cropped_dst)
                # 追加positional encodings
                latents_src += self.generate_positional_encoding(latents_src) * 0.5
                # sequenceへ
                sequence_src = self.reshape_latents(latents_src)
                sequence_dst = self.reshape_latents(latents_dst)

                optimizer.zero_grad()
                new_sequences = self.model(sequence_src, sequence_src)
                loss = loss_fn(new_sequences, sequence_dst)

                loss.backward()
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader)}")

    # Main Program
    def main(self, args):
        # Hyperparameters
        batch_size = 4
        learning_rate = 0.001 # 0.001

        # Fake data (replace with actual latent data)
        images = self.load_images(args.folder, 256)

        # Dataset and DataLoader
        dataset = LatentDataset(images, images)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Model, Loss, Optimizer
        self.model = TransformerLatentModel(self.LATENT_DIM, self.NUM_HEADS, self.NUM_LAYERS, self.HIDDEN_DIM).to(self.DEVICE)
        if args.load:
            print("LOADED=", args.load)
            self.model.load_state_dict(torch.load(args.load, weights_only=True))
        loss_fn = nn.MSELoss()  # Latent regression task
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = ExponentialLR(optimizer, gamma=0.9)

        # Train the model
        self.train_model(dataloader, optimizer, scheduler, loss_fn, epochs=args.epochs)

        # Save the model
        if args.model:
            torch.save(self.model.state_dict(), args.model)
            print("Model saved!")

    def inference(self, args):
        self.model.load_state_dict(torch.load(args.load, weights_only=True))

        tensor = self.preprocess_image(args.input, 256)
        tensors = tensor.unsqueeze(0)
        latents = self.encode_to_token_latents(tensors)
        positonal = self.generate_positional_encoding(latents) * 0.5
        sequences = self.reshape_latents(latents + positonal)
        with torch.no_grad():
            new_sequences = self.model(sequences, sequences)
        new_latents = self.reshape_back_latents(new_sequences, self.LATENT_SIZE, self.LATENT_SIZE)
        self.check_latent(new_latents[0])
        #print("L2=", new_latents)
        tensors = self.decode_from_latents(new_latents)
        #print("T=", tensors)
        image = self.to_pil_image(tensors[0])

        if args.output:
            image.save(args.output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="実験する")
    parser.add_argument('--model', default="my_transformer.pth", help="モデル名")
    parser.add_argument('--load', default=None, help="読み込みモデル名")
    parser.add_argument('--epochs', default=30, type=int, help="エポック数")
    parser.add_argument('-i', '--input', default=None, help="画像へのパス")
    parser.add_argument('-o', '--output', default=None, help="出力パス")
    parser.add_argument('-f', '--folder', default=None, help="画像フォルダ")
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--infer', action='store_true')
    args = parser.parse_args()

    trainer = Trainer()
    if args.train and args.folder:
        trainer.main(args)
    if args.infer and args.input and args.load:
        trainer.inference(args)

# python3 train_transformer.py --train --epochs 30 --folder images/blue1/
# python3 train_transformer.py --infer --load my_transformer.pth --input sample/01.jpg --output output/tf-01.jpg
