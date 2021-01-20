import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_dev():
    return device
