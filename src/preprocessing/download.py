import os
from datasets import load_dataset, dataset_dict

datadir = os.path.join(os.path.dirname(__file__), '..', '..', 'data')

def download() -> dataset_dict.DatasetDict:
    '''
    Downloads the dataset into the `data` directory the first time it is run.
    Loads dataset from cache on subsequent runs.
    
    :return: dataset as a HuggingFace DatasetDict
    '''
    return load_dataset('Supermaxman/esa-hubble', split='train', cache_dir=datadir)