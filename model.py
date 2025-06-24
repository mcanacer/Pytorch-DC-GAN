import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, z_dim=100, nfd=64, nc=3):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(z_dim, nfd * 8, 4, 1, 0, bias=False),
            nn.GroupNorm(8, nfd * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(nfd * 8, nfd * 4, 4, 2, 1, bias=False),
            nn.GroupNorm(8, nfd * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(nfd * 4, nfd * 2, 4, 2, 1, bias=False),
            nn.GroupNorm(8, nfd * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(nfd * 2, nfd, 4, 2, 1, bias=False),
            nn.GroupNorm(8, nfd),
            nn.ReLU(True),

            nn.ConvTranspose2d(nfd, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

        self.apply(self._initialize_weights)

    def _initialize_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        return self.net(x)


class Discriminator(nn.Module):
    def __init__(self, nc=3, nfd=64):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(nc, nfd, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),  # No norm in first layer

            nn.Conv2d(nfd, nfd * 2, 4, 2, 1, bias=False),
            nn.GroupNorm(8, nfd * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(nfd * 2, nfd * 4, 4, 2, 1, bias=False),
            nn.GroupNorm(8, nfd * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(nfd * 4, nfd * 8, 4, 2, 1, bias=False),
            nn.GroupNorm(8, nfd * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(nfd * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

        self.apply(self._initialize_weights)

    def _initialize_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        return self.net(x)

