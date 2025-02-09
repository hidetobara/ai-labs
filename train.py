import os, sys, glob, argparse, random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from PIL import Image
from torchvision import transforms
from diffusers import AutoencoderKL


# Transformer Encoder-Decoder Model
class TransformerLatentModel(nn.Module):
    def __init__(self, latent_dim, num_heads, num_layers, hidden_dim, dropout=0.1):
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
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, tgt):
        # Encoder
        memory = self.encoder(src)
        # Dropout
        memory = self.dropout(memory)
        # Decoder
        output = self.decoder(tgt, memory)
        # Project to Latent Space
        output = self.latent_proj(output)
        return output

# Custom Dataset
class LatentDataset(Dataset):
    def __init__(self, latent_inputs, latent_targets, fn=None):
        if fn is None:
            fn = lambda x: x
        # [Batch, Channels, Height, Width] を Transformer の形式に変換
        self.inputs = [fn(tensor) for tensor in latent_inputs]
        self.targets = [fn(tensor) for tensor in latent_targets]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

class Trainer:
    LATENT_DIM = 4  # Embedding Dim
    NUM_HEADS = 4
    NUM_LAYERS = 8
    HIDDEN_DIM = 128
    ORIGIN_WIDTH = 512
    CROP_LEN = 32 # 64
    SEQUENCE_LENGTH = CROP_LEN * CROP_LEN

    def __init__(self):
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.vae = self.load_vae_model()
        self.model = TransformerLatentModel(self.LATENT_DIM, self.NUM_HEADS, self.NUM_LAYERS, self.HIDDEN_DIM).to(self.DEVICE)

    def load_vae_model(self, model_name="stabilityai/sd-vae-ft-mse"):
        vae = AutoencoderKL.from_pretrained(model_name).to(self.DEVICE)
        vae.eval()  # 評価モードに設定
        return vae

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
    
    def check_latent(self, latents):
        print(f"Latent shape: {latents.shape}")
        min_val = latents[0].min().item()
        max_val = latents[0].max().item()
        mean_val = latents[0].mean().item()
        std_val = latents[0].std().item()
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
            width = self.ORIGIN_WIDTH
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
    def encode_to_latents(self, image_tensors):
        with torch.no_grad():
            latents = self.vae.encode(image_tensors.to(self.DEVICE)).latent_dist.sample()  # サンプリング
            latents = latents * 0.18215  # スケール調整
        return latents

    # Latent表現を画像に戻す関数
    def decode_from_latents(self, latent_tensors):
        with torch.no_grad():
            # スケール調整を戻す
            latent_tensors = latent_tensors / 0.18215
            # VAEを使って画像を生成
            reconstructed_images = self.vae.decode(latent_tensors).sample

        # 画像の値を[0, 1]の範囲にスケール
        reconstructed_images = (reconstructed_images.clamp(-1, 1) + 1) / 2
        #reconstructed_image = torch.reshape(reconstructed_image.cpu(), (1, 3, self.ORIGIN_WIDTH, self.ORIGIN_WIDTH))
        return reconstructed_images

    def crop_random_images(self, images):
        if images.shape[2] == self.CROP_LEN and images.shape[3] == self.CROP_LEN:
            return images
        h = random.randint(0, images.shape[2] - self.CROP_LEN)
        w = random.randint(0, images.shape[3] - self.CROP_LEN)
        new_images = images[:, :, h:h+self.CROP_LEN, w:w+self.CROP_LEN]
        return new_images

    def to_pil_image(self, tensor):
        array = tensor.cpu().permute(1, 2, 0).numpy()
        return Image.fromarray((array * 255).astype("uint8"))

    def load_images(self, folder):
        images = [] # (N, C, H, W)
        paths = []
        for ext in ["png", "jpg", "JPG"]:
            paths.extend(glob.glob(os.path.join(folder, f"*.{ext}")))
        for path in paths:
            image = self.preprocess_image(path)
            images.append(image)
        return images

    # Training Loop
    def train_model(self, dataloader, optimizer, loss_fn, epochs=10):
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for src, tgt in dataloader:
                # ランダムに64x64サイズに刈り取る
                cropped_tgt = self.crop_random_images(src)
                cropped_src = cropped_tgt + torch.empty(cropped_tgt.shape).uniform_(-0.05, 0.05)
                # latentsへ
                latents_src = self.encode_to_latents(cropped_src)
                latents_tgt = self.encode_to_latents(cropped_tgt)
                # 追加positional encodings
                latents_src += self.generate_positional_encoding(latents_src)
                latents_tgt += self.generate_positional_encoding(latents_tgt)
                # sequenceへ
                sequence_src = self.reshape_latents(latents_src)
                sequence_tgt = self.reshape_latents(latents_tgt)

                optimizer.zero_grad()
                output = self.model(sequence_src, sequence_src)
                loss = loss_fn(output, sequence_tgt)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader)}")

    # Main Program
    def main(self, args):
        # Hyperparameters
        batch_size = 4
        learning_rate = 0.0001 # 0.001

        # Fake data (replace with actual latent data)
        images = self.load_images(args.folder)

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

        # Train the model
        self.train_model(dataloader, optimizer, loss_fn, epochs=args.epochs)

        # Save the model
        if args.model:
            torch.save(self.model.state_dict(), args.model)
            print("Model saved!")

    def inference(self, args):
        filename = os.path.basename(args.image)
        self.model.load_state_dict(torch.load(args.load, weights_only=True))

        tensor = self.preprocess_image(args.image, 256)
        tensors = tensor.unsqueeze(0)
        latents = self.encode_to_latents(tensors)
        positonal = self.generate_positional_encoding(latents)
        sequences = self.reshape_latents(latents + positonal)
        with torch.no_grad():
            new_sequences = self.model(sequences, sequences)
        new_latents = self.reshape_back_latents(new_sequences, self.CROP_LEN, self.CROP_LEN) - positonal
        tensors = self.decode_from_latents(new_latents)
        image = self.to_pil_image(tensors[0])
        image.save(os.path.join("output", filename))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="実験する")
    parser.add_argument('--model', default="tfl.pth", help="モデル名")
    parser.add_argument('--load', default=None, help="読み込みモデル名")
    parser.add_argument('--epochs', default=30, type=int, help="エポック数")
    parser.add_argument('-i', '--image', default=None, help="画像へのパス")
    parser.add_argument('-f', '--folder', default=None, help="画像フォルダ")
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--infer', action='store_true')
    args = parser.parse_args()

    trainer = Trainer()
    if args.train and args.folder:
        trainer.main(args)
    if args.infer and args.image and args.load:
        trainer.inference(args)

# python3 train.py --train --epochs 30 --folder images/doll/
