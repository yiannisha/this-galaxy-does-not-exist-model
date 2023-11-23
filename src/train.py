from models import generator, discriminator
from preprocessing import NebulaDataset, download
from utils import to_device, get_device
from training import Trainer

if __name__ == '__main__':
    
    # create new models
    generator = to_device(generator, get_device())
    discriminator = to_device(discriminator, get_device())
    
    # set hyperparameters
    batch_size = 1
    latent_size = 2048
    
    # downlaod/load dataset
    hugging_face_dataset = download()
    dataset = NebulaDataset(hugging_face_dataset[:10])
    
    # train the models
    trainer = Trainer(discriminator, generator, dataset, latent_size, batch_size)
    
    trainer.fit(epochs=25, lr=0.0002, start_idx=1)