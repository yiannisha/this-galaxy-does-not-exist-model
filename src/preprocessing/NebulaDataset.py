import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from datasets import DatasetDict

from typing import Tuple

class NebulaDataset (Dataset):
    '''
    A custom pytorch dataset for the Nebula dataset.
    Necessary to train the discriminator model.
    '''
    
    def __init__ (self, hugging_face_dataset: DatasetDict) -> None:
        '''
        :param hugging_face_dataset: A hugging face dataset object.
        '''
        self.dataset = hugging_face_dataset
        
    def __len__ (self) -> int:
        '''
        :return: The length of the dataset.
        '''
        return len(self.dataset)
    
    def __getitem__ (self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        :param idx: The index of the item to get.
        :return: A tuple containing the input and output tensors.
        '''
        
        # images are stored as PIL images, so we need to convert them to tensors
        image = self.dataset[idx]['image']
        
        # transform image to 512x512
        image = T.Resize((512, 512))(image)
        
        # convert image to tensor
        image = T.ToTensor()(image)
        
        # normalize images to [-1, 1]
        image = T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(image)
        
        # labels are all 1s because the images are real
        label = torch.tensor(1)
        
        return image, label