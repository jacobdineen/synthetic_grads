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
    Eight_Layer,
    VGG16_custom,
    Four_Layer_SG,
    Eight_Layer_SG,
    VGG16_SG2,
)
import numpy as np
from train import train, train_sg, test
import warnings
from tqdm import tqdm
from time import time
import json


warnings.filterwarnings("ignore")

if __name__ == "__main__":
    n_epochs = 25
    log_interval = 20

    # set random seed
    random_seed = 42
    torch.manual_seed(random_seed)

    # set torch device
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    batch_size_train = 100  # specified in the paper
    train_loader, test_loader = get_dataloaders(500, 100, batch_size_train)

    model_dict = {
        # 4-layer convnets
        "Four_Layer_SG": Four_Layer_SG,
        "Four_layer": Four_Layer,
        #     #8-layer convnets
        "Eight_Layer_SG": Eight_Layer_SG,
        "Eight_Layer": Eight_Layer,
        # VGG Models
        "VGG16_SG": VGG16_SG2,
        "VGG16": VGG16_custom,
    }

    # with open("data.json", "r") as fp:
    #     data = json.load(fp)
    data = {}
    for key in model_dict.keys():
        model = model_dict[key]()
        stats = []
        print(key, "\n")
        for epoch in tqdm(range(1, n_epochs + 1), leave=False):
            start_time = time()

            if "SG" not in key:
                train(model, epoch, train_loader, log_interval, verbose=False)
            else:
                train_sg(model, epoch, train_loader, log_interval, verbose=False)

            end_time = time()
            time_taken = end_time - start_time

            metrics = test(model, test_loader)
            metrics.append(time_taken)

            stats.append(metrics)

        data[key] = stats
        # clear gpu memory
        torch.cuda.empty_cache()

    # with open("data.json", "w") as fp:
    #     json.dump(data, fp)
