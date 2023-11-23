import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

from utils import get_device, save_samples, DeviceDataLoader

from typing import Tuple, List

class Trainer:
    '''
    A class for training the model.
    '''
    
    def __init__ ( self, discriminator: nn.Module, generator: nn.Module, dataset: Dataset, latent_size: int, batch_size: int ) -> None:
        '''
        :param discriminator: The discriminator model.
        :param generator: The generator model.
        :param dataset: The dataset to train on.
        :param latent_size: The size of the latent space.
        :param batch_size: The batch size.
        '''
        
        # save the models
        self.discriminator = discriminator
        self.generator = generator
        
        # save the dataset
        self.dataset = dataset
        self.dataloader = DeviceDataLoader(
            DataLoader(self.dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True),
            get_device()
        )
        
        # save the latent size
        self.latent_size = latent_size
        
        # save the batch size
        self.batch_size = batch_size
        
    def train_discriminator (self, opt_d: torch.optim.Optimizer, images: torch.Tensor) -> Tuple[float, float, float]:
        '''
        Train the discriminator model.
        '''
        
        # Clear discriminator gradients
        opt_d.zero_grad()

        # Pass real images through discriminator
        real_preds = self.discriminator(images)
        real_targets = torch.ones(images.size(0), 1, device=get_device())
        real_loss = F.binary_cross_entropy(real_preds, real_targets)
        real_score = torch.mean(real_preds).item()
        
        # Generate fake images
        latent = torch.randn(self.batch_size, self.latent_size, 1, 1, device=get_device())
        fake_images = self.generator(latent)

        # Pass fake images through discriminator
        fake_targets = torch.zeros(fake_images.size(0), 1, device=get_device())
        fake_preds = self.discriminator(fake_images)
        fake_loss = F.binary_cross_entropy(fake_preds, fake_targets)
        fake_score = torch.mean(fake_preds).item()

        # Update discriminator weights
        loss = real_loss + fake_loss
        loss.backward()
        opt_d.step()
        
        return loss.item(), real_score, fake_score
    
    def train_generator (self, opt_g: torch.optim.Optimizer) -> float:
        '''
        Train the generator model.
        '''
        
        # Clear generator gradients
        opt_g.zero_grad()

        # Generate fake images
        latent = torch.randn(self.batch_size, self.latent_size, 1, 1, device=get_device())
        fake_images = self.generator(latent)

        # Try to fool the discriminator
        preds = self.discriminator(fake_images)
        targets = torch.ones(self.batch_size, 1, device=get_device())
        loss = F.binary_cross_entropy(preds, targets)

        # Update generator weights
        loss.backward()
        opt_g.step()
        
        return loss.item()
    
    def fit (self, epochs: int, lr: float, start_idx: int = 1) -> Tuple[List[float], List[float], List[float], List[float]]:
        torch.cuda.empty_cache()
        
        # Losses & scores
        losses_g = []
        losses_d = []
        real_scores = []
        fake_scores = []
        
        # Create optimizers
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.999))
        
        for epoch in range(epochs):
            for real_images, _ in self.dataloader:
                # Train discriminator
                loss_d, real_score, fake_score = self.train_discriminator(real_images, opt_d)
                # Train generator
                loss_g = self.train_generator(opt_g)
                
            # Record losses & scores
            losses_g.append(loss_g)
            losses_d.append(loss_d)
            real_scores.append(real_score)
            fake_scores.append(fake_score)
            
            # Log losses & scores (last batch)
            print("Epoch [{}/{}], loss_g: {:.4f}, loss_d: {:.4f}, real_score: {:.4f}, fake_score: {:.4f}".format(
                epoch+1, epochs, loss_g, loss_d, real_score, fake_score))
        
            # Save generated images
            save_samples(epoch+start_idx, self.generator, self.discriminator, self.latent_size)
    
        return losses_g, losses_d, real_scores, fake_scores