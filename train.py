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
    def __init__(self, latent_dim, num_heads, num_layers, hidden_dim, dropout=0.05):
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
        if True:
            # Encoder
            memory = self.encoder(src)
            # Dropout
            memory = self.dropout(memory)
            # Decoder
            output = self.decoder(tgt, memory)
        if False:
            # Decoder
            output = self.decoder(tgt, src)
        # Project to Latent Space
        output = self.latent_proj(output)
        return output

# Custom Dataset
class LatentDataset(Dataset):
    def __init__(self, latent_inputs, latent_targets, fn=None):
        if fn is None:
            fn = lambda x: x
        # [Batch, Channels, Height, Width] を Transformer の形式に変換
        self.latent_inputs = [fn(tensor) for tensor in latent_inputs]
        self.latent_targets = [fn(tensor) for tensor in latent_targets]

    def __len__(self):
        return len(self.latent_inputs)

    def __getitem__(self, idx):
        return self.latent_inputs[idx], self.latent_targets[idx]

class Trainer:
    LATENT_DIM = 4  # Embedding Dim
    SEQUENCE_LENGTH = 64 * 64  # Sequence Length
    NUM_HEADS = 4
    NUM_LAYERS = 8
    HIDDEN_DIM = 128
    ORIGIN_WIDTH = 512
    CROP_LEN = 64

    def __init__(self):
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.vae = self.load_vae_model()
        self.model = TransformerLatentModel(self.LATENT_DIM, self.NUM_HEADS, self.NUM_LAYERS, self.HIDDEN_DIM).to(self.DEVICE)

    def load_vae_model(self, model_name="stabilityai/sd-vae-ft-mse"):
        vae = AutoencoderKL.from_pretrained(model_name).to(self.DEVICE)
        vae.eval()  # 評価モードに設定
        return vae

    def reshape_latent(self, latent_tensor):
        """
        Latent tensor [Channels, Height, Width] を
        Transformer用の形式 [Sequence Length, Embedding Dim] に変換
        """
        channels, height, width = latent_tensor.shape
        sequence_length = height * width
        reshaped_tensor = latent_tensor.permute(1, 2, 0).reshape(sequence_length, channels)
        return reshaped_tensor

    def reshape_back_latent(self, reshaped_tensor, height, width):
        """
        Transformer用の形式 [Sequence Length, Embedding Dim] を
        [Channels, Height, Width] に戻す
        """
        sequence_length, channels = reshaped_tensor.shape
        reshaped_tensor = reshaped_tensor.reshape(height, width, channels).permute(2, 0, 1)
        return reshaped_tensor
    
    def check_latent(self, latent):
        print(f"Latent shape: {latent.shape}")
        min_val = latent.min().item()
        max_val = latent.max().item()
        mean_val = latent.mean().item()
        std_val = latent.std().item()
        print(f"Latent min: {min_val}, max: {max_val}, average: {mean_val} std: {std_val}")

    # 画像をリサイズしてテンソルに変換
    def preprocess_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize((self.ORIGIN_WIDTH, self.ORIGIN_WIDTH)),  # リサイズ
            transforms.ToTensor(),  # テンソルに変換
            #transforms.Normalize([0.5], [0.5])  # 正規化
        ])
        tensor = transform(image).unsqueeze(0)
        #print("origin RGB=", tensor[0,0,:,:].mean(), tensor[0,1,:,:].mean(), tensor[0,2,:,:].mean())
        return tensor

    # 画像をLatent表現に変換
    def encode_to_latent(self, image_tensor):
        with torch.no_grad():
            latent = self.vae.encode(image_tensor.to(self.DEVICE)).latent_dist.sample()  # サンプリング
            latent = latent * 0.18215  # スケール調整
        new_latent = torch.squeeze(latent, dim=0)
        return new_latent

    def generate_positional_encoding(self, C, H, W):
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
        # Create a grid of positions
        y_embed = torch.arange(0, H, device=self.DEVICE).unsqueeze(1).repeat(1, W)  # Shape: [H, W]
        x_embed = torch.arange(0, W, device=self.DEVICE).unsqueeze(0).repeat(H, 1)  # Shape: [H, W]

        # Normalize positions to range [0, 1]
        y_embed = y_embed / H
        x_embed = x_embed / W

        # Initialize positional encoding tensor
        pe = torch.zeros(C, H, W, device=self.DEVICE)  # Shape: [C, H, W]

        # Compute sinusoidal positional encoding
        for i in range(C // 2):
            div_term = 10000 ** (2 * i / C)
            pe[2 * i, :, :] = torch.sin(x_embed / div_term)  # Sin for x
            pe[2 * i + 1, :, :] = torch.cos(y_embed / div_term)  # Cos for y

        return pe

    # Latent表現を画像に戻す関数
    def decode_from_latent(self, latent_tensor):
        with torch.no_grad():
            # スケール調整を戻す
            latent_tensor = latent_tensor / 0.18215
            # VAEを使って画像を生成
            reconstructed_image = self.vae.decode(latent_tensor).sample

        # 画像の値を[0, 1]の範囲にスケール
        reconstructed_image = (reconstructed_image.clamp(-1, 1) + 1) / 2
        reconstructed_image = torch.reshape(reconstructed_image.cpu(), (3, 512, 512))
        print("returned RGB=", reconstructed_image[0,:,:].mean(), reconstructed_image[1,:,:].mean(), reconstructed_image[2,:,:].mean())
        return reconstructed_image

    def crop_random_latent(self, latent):
        if latent.shape[1] == self.CROP_LEN and latent.shape[2] == self.CROP_LEN:
            return latent
        h = random.randint(0, latent.shape[1] - self.CROP_LEN)
        w = random.randint(0, latent.shape[2] - self.CROP_LEN)
        new_latent = latent[:,h:h+self.CROP_LEN,w:w+self.CROP_LEN]
        return new_latent

    def to_pil_image(self, tensor):
        array = tensor.permute(1, 2, 0).numpy()
        return Image.fromarray((array * 255).astype("uint8"))

    def load_images(self, folder):
        latents = [] # (N, C, H, W)
        paths = []
        for ext in ["png", "jpg", "JPG"]:
            paths.extend(glob.glob(os.path.join(folder, f"*.{ext}")))
        for path in paths:
            image_tensor = self.preprocess_image(path)
            latent = self.encode_to_latent(image_tensor)
            latents.append(latent)
        return latents

    # Training Loop
    def train_model(self, dataloader, optimizer, loss_fn, epochs=10):
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for src, tgt in dataloader:
                # ランダムに64x64サイズに刈り取る
                cropped_src = [self.reshape_latent(self.crop_random_latent(l) + self.generate_positional_encoding(4, self.CROP_LEN, self.CROP_LEN)) for l in src]
                cropped_tgt = [self.reshape_latent(self.crop_random_latent(l) + self.generate_positional_encoding(4, self.CROP_LEN, self.CROP_LEN)) for l in tgt]
                cropped_src = torch.stack(cropped_src).to(self.DEVICE)
                cropped_tgt = torch.stack(cropped_tgt).to(self.DEVICE)

                optimizer.zero_grad()
                output = self.model(cropped_src, cropped_src)
                loss = loss_fn(output, cropped_tgt)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader)}")

    # Main Program
    def main(self, args):
        # Hyperparameters
        batch_size = 4
        learning_rate = 0.001

        # Fake data (replace with actual latent data)
        latent_inputs = self.load_images(args.folder)
        latent_targets = self.load_images(args.folder)

        # Dataset and DataLoader
        dataset = LatentDataset(latent_inputs, latent_targets)
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

        image_tensor = self.preprocess_image(args.image)
        latent = self.reshape_latent(self.crop_random_latent(self.encode_to_latent(image_tensor)))
        src = torch.reshape(latent, (1, self.SEQUENCE_LENGTH, self.LATENT_DIM))
        with torch.no_grad():
            new_tensor = self.model(src, src)
        new_latent = self.reshape_back_latent(new_tensor[0], self.CROP_LEN, self.CROP_LEN)
        tensor = self.decode_from_latent(torch.reshape(new_latent, (1, self.LATENT_DIM, self.CROP_LEN, self.CROP_LEN)) )
        image = self.to_pil_image(tensor)
        image.save(os.path.join("output", filename))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="実験する")
    parser.add_argument('--model', default="transformer_latent.pth", help="モデル名")
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

    # python3 train.py --train --epochs 300 --folder images/blue3/