from models import Generator, Discriminator
from preprocessing import NebulaDataset, download
from utils import to_device, get_device
from training import Trainer

from torch.utils.data import Subset

if __name__ == '__main__':
    
    device = get_device()
    print('device: ', device)
    
    # set hyperparameters
    batch_size = 128
    latent_size = 2048
    
    # create new models
    gen = to_device(Generator(latent_size=latent_size), device)
    discr = to_device(Discriminator(), device)
    
    # downlaod/load dataset
    hugging_face_dataset = download()
    dataset = NebulaDataset(hugging_face_dataset)
    
    # subset = Subset(dataset, range(0, 10))
    
    # train the models
    trainer = Trainer(discr, gen, dataset, latent_size, batch_size, device)
    
    trainer.fit(epochs=25, lr=0.0002, start_idx=1)