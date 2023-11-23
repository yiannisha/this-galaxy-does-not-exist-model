from models import Generator, Discriminator
from preprocessing import NebulaDataset, download
from utils import to_device, get_device
from training import Trainer

if __name__ == '__main__':
    
    # set hyperparameters
    batch_size = 1
    latent_size = 2048
    
    # create new models
    gen = to_device(Generator(latent_size=latent_size), get_device())
    discr = to_device(Discriminator(), get_device())
    
    # downlaod/load dataset
    hugging_face_dataset = download()
    dataset = NebulaDataset(hugging_face_dataset[:10])
    
    # train the models
    trainer = Trainer(disc, gen, dataset, latent_size, batch_size)
    
    trainer.fit(epochs=25, lr=0.0002, start_idx=1)