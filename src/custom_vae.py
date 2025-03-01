from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# 定数
INPUT_CHANNELS = 3   # RGB 画像
LATENT_CHANNELS = 512  # 潜在空間のチャンネル数（変更可能）

# カスタム VAE モデル
class CustomVAE(nn.Module):
    """
    [256, 256, 3] -> [C, 16, 16]
    """
    def __init__(self):
        super(CustomVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(INPUT_CHANNELS, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, LATENT_CHANNELS, kernel_size=4, stride=2, padding=1)
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(LATENT_CHANNELS, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, INPUT_CHANNELS, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        return self.encoder(x) * 0.5

    def decode(self, x):
        return self.decoder(x * 2.0)

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon
    
    def save(self, path):
        torch.save(self.state_dict(), path)

    @staticmethod
    def load(path):
        i = CustomVAE()
        i.load_state_dict(torch.load(path, weights_only=True))
        return i
