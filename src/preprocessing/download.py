import os
from datasets import load_dataset

datadir = os.path.join(os.path.dirname(__file__), '..', '..', 'data')

def download() -> datasets.dataset_dict.DatasetDict:
    return load_dataset('Supermaxman/esa-hubble', split='train', cache_dir=datadir)