import torch
import torch.nn as nn

class Discriminator (nn.Module):
    '''
    Discriminator model.
    Classifies images as real or fake.
    '''
    
    def __init__ (self) -> None:
        super().__init__()
        
        self.network = nn.Sequential(
        # in: 3 x 512 x 512
        
        nn.Conv2d(3, 16, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(16),
        nn.LeakyReLU(0.2, inplace=True),
        # out: 16 x 256 x 256
        
        nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(32),
        nn.LeakyReLU(0.2, inplace=True),
        # out: 32 x 128 x 128

        nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(64),
        nn.LeakyReLU(0.2, inplace=True),
        # out: 64 x 64 x 64
        
        nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(128),
        nn.LeakyReLU(0.2, inplace=True),
        # out: 128 x 32 x 32
        
        nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(256),
        nn.LeakyReLU(0.2, inplace=True),
        # out: 256 x 16 x 16
        
        nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(512),
        nn.LeakyReLU(0.2, inplace=True),
        # out: 512 x 8 x 8

        nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(1024),
        nn.LeakyReLU(0.2, inplace=True),
        # out: 1024 x 4 x 4

        nn.Conv2d(1024, 1, kernel_size=4, stride=1, padding=0, bias=False),
        # out: 1 x 1 x 1

        nn.Flatten(),
        nn.Sigmoid()
    )

    def forward (self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)