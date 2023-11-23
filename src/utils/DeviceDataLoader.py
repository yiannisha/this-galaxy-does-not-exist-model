import torch
from torch.utils.data import DataLoader

from . import to_device

from typing import Tuple, Union, Any, Generator

class DeviceDataLoader():
    '''
    Wraps a DataLoader to move batches of data to the correct device
    '''
    
    def __init__ (self, dataloader: DataLoader, device: Union[torch.device, int]):
        self.dl = dataloader
        self.device = device
        
    def __iter__ (self) -> Generator[Tuple[Any, Any], Tuple[Any, Any], None]:
        '''
        Converts dataloader batch to device and returns it.
        '''

        for b in self.dl:
            yield to_device(b, self.device)
    
    def __len__ (self):
        return len(self.dl)