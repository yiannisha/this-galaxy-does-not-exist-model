import torch
from torch import nn

class Generator (nn.Module):
    '''
    Generator model.
    Generates images from random noise.
    '''
    
    def __init__ (self, latent_size: int) -> None:
        super().__init__()
        
        self.network = nn.Sequential(
            # in: latent_size x 1 x 1

            nn.ConvTranspose2d(latent_size, 1024, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            # out: 1024 x 4 x 4

            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # out: 512 x 8 x 8

            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # out: 256 x 16 x 16

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # out: 128 x 32 x 32
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # out: 64 x 64 x 64
            
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            # out: 32 x 128 x 128
            
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            # out: 16 x 256 x 256
            
            nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1, bias=False),
            # out: 3 x 512 x 512

            nn.Tanh()
            # out: 3 x 512 x 512
        )

    def forward (self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)