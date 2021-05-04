import torch
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
from dataloader import get_dataloaders
from models import Four_Layer_SG_1, Four_Layer_SG_2
import numpy as np
import warnings
from tqdm import tqdm
from time import time
import json

warnings.filterwarnings("ignore")


def train(model, epoch, train_loader, log_interval, verbose=False):
    model.train()
    model.cpu()
    optimizer = optim.Adam(model.parameters(), lr=3e-5)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cpu(), target.cpu()
        optimizer.zero_grad()
        try:
            output = model(data)
        except:
            output = model(data, target)
    return output, target


if __name__ == "__main__":
    n_epochs = 100
    log_interval = 20

    # set torch device
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    batch_size_train = 256  # specified in the paper
    train_loader, _ = get_dataloaders(1000, 200, batch_size_train)

    model = Four_Layer_SG_1()
    stats = []

    for epoch in tqdm(range(1, n_epochs + 1), leave=False):
        start_time = time()
        output, target = train(model, epoch, train_loader, log_interval, verbose=False)
        torch.save(output, "x.pt")
        torch.save(target, "y.pt")

