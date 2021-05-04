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
from models import (
    Four_Layer,
    Four_Layer_SG,
)
import numpy as np
from train import train, test
import warnings
from tqdm import tqdm
from time import time
import json

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    n_epochs = 100
    log_interval = 20

    # set torch device
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    batch_size_train = 256  # specified in the paper
    train_loader, test_loader = get_dataloaders(1000, 200, batch_size_train)

    model_dict = {
        # 4-layer convnets
        "Four_Layer_SG": Four_Layer_SG,
        # "Four_layer": Four_Layer,
    }

    # with open("data.json", "r") as fp:
    #     data = json.load(fp)

    for key in model_dict.keys():
        model = model_dict[key]()
        stats = []
        test(model, test_loader)
        print(key, "\n")
        for epoch in tqdm(range(1, n_epochs + 1), leave=False):
            start_time = time()

            train(model, epoch, train_loader, log_interval, verbose=False)

            end_time = time()
            time_taken = end_time - start_time

            metrics = test(model, test_loader)
            metrics.append(time_taken)

            stats.append(metrics)

        torch.cuda.empty_cache()

