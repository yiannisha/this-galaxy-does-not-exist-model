'''
Helper functions for the project.
'''
import os
import torch
import torch.nn as nn
from torchvision.utils import save_image

from .DeviceDataLoader import DeviceDataLoader

from typing import Any

def get_device () -> torch.device:
    '''
    :return: The device to use for training.
    '''
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def to_device (data: Any, device: torch.device) -> Any:
    '''
    Move tensor(s) to chosen device.
    '''
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

def save_samples(index: int, generator: nn.Model, discriminator: nn.Model, latent_size: int) -> None:
    '''
    Saves fake images from the generator & model checkpoints.
    '''
    latent_tensors = torch.randn(64, latent_size, 1, 1, device=get_device())
    fake_images = generator(latent_tensors)
    fake_fname = 'generated-images-{0:0=4d}.png'.format(index)
    
    # create checkppoint directory
    dirpath = os.path.join(os.path.dirname(__file__), '..', 'models', f'checkpoint-{index}')
    os.makedirs(dirpath, exist_ok=True)
    
    # save models
    generator.to('cpu')
    discriminator.to('cpu')
    torch.save(generator.state_dict(), os.path.join(dirpath, 'generator.pth'))
    torch.save(discriminator.state_dict(), os.path.join(dirpath, 'discriminator.pth'))
    
    # save images
    save_image(denormalize(fake_images), os.path.join(dirpath, fake_fname), nrow=8)
    
    print('Saving', fake_fname)
    
def denormalize (images: torch.Tensor) -> torch.Tensor:
    '''
    Denormalize images.
    '''
    return (images * 0.5) + 0.5