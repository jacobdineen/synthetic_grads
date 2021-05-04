from sklearn.metrics import precision_recall_fscore_support
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
import numpy as np
import warnings
from tqdm import tqdm
from time import time
import json
from models import Four_Layer_SG_1, Four_Layer_SG_2
import pandas as pd

warnings.filterwarnings("ignore")


def train(model, data, target, epoch, log_interval, verbose=False):
    model.train()
    model.cpu()
    optimizer = optim.Adam(model.parameters(), lr=3e-5)
    for batch_idx, (dat, tar) in enumerate(train_loader):
        data, target = data.cpu(), target.cpu()
        optimizer.zero_grad()
        try:
            output = model(data)
        except:
            output = model(data, target)
        loss = F.nll_loss(output, target, size_average=False)
        loss.backward()
        optimizer.step()
    return output


def test(model, test_loader):
    test_metrics = []
    model.eval()
    model.cpu()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cpu(), target.cpu()
            data, target = Variable(data, volatile=True), Variable(target)
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()

            pred = output.data.max(1, keepdim=True)[
                1
            ]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            total += target.size(0)

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100 * correct / total

    prec, recall, f1, support = precision_recall_fscore_support(
        target.cpu(), pred.cpu(), average="weighted"
    )

    test_metrics.append((test_accuracy.item(), prec, recall, f1, support))

    return test_metrics


if __name__ == "__main__":
    n_epochs = 100
    log_interval = 20
    output_path = os.path.join("./data", "output.csv")
    batch_size_train = 256  # specified in the paper
    train_loader, test_loader = get_dataloaders(1000, 200, batch_size_train)
    model = Four_Layer_SG_2()
    stats = []

    for epoch in tqdm(range(1, n_epochs + 1), leave=False):
        start_time = time()
        data, target = torch.load("x.pt"), torch.load("y.pt")

        train(model, epoch, data, target, log_interval, verbose=False)

        end_time = time()
        time_taken = end_time - start_time

        metrics = test(model, test_loader)
        metrics.append(time_taken)

        stats.append(metrics)
        print(stats, flush=True)
    pd.DataFrame(stats).to_csv("./data/data.csv")

