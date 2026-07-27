import torch
from torch import nn


class DCGANGenerator28x28(nn.Module):
    """
    DCGAN-style generator for 28x28 grayscale images.
    Input: latent noise vector of shape (batch_size, latent_dim, 1, 1)
    Output: fake image tensor of shape (batch_size, 1, 28, 28)
    """
    def __init__(self, latent_dim=100, feature_maps=64, out_channels=1):
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, feature_maps * 4, kernel_size=7, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_maps * 2, feature_maps, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(True),

            nn.Conv2d(feature_maps, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)


class DCGANDiscriminator28x28(nn.Module):
    """
    DCGAN-style discriminator for 28x28 grayscale images.
    Input: image tensor of shape (batch_size, 1, 28, 28)
    Output: probability tensor of shape (batch_size, 1)
    """
    def __init__(self, in_channels=1, feature_maps=64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, feature_maps, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_maps, feature_maps * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_maps * 2, feature_maps * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten(),
            nn.Linear(feature_maps * 4 * 4 * 4, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)