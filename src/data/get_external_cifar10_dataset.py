import os
import tarfile
import urllib.request
from glob import glob

def get_external_cifar10_datasets():    
    CIFAR10_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    EXTERNAL_DATA_PATH = '/home/jovyan/cifar10-practice/data/external'
    RAW_DATA_PATH = '/home/jovyan/cifar10-practice/data/raw'
    fname = f'{EXTERNAL_DATA_PATH}/cifar10.tar.gz'
    if not glob(fname): 
        print(f'Request data from {CIFAR10_URL}')
        urllib.request.urlretrieve(CIFAR10_URL, fname)
        print(f'Downloaded data from {CIFAR10_URL}')
    raw_data = tarfile.open(fname)
    raw_data.extractall(RAW_DATA_PATH)    
    print(f'Extract data to {RAW_DATA_PATH} finished')
    raw_data.close()   
    

if __name__ == "__main__":
    get_external_cifar10_datasets()