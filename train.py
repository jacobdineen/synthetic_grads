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


def train(model, epoch, train_loader, log_interval, verbose=False):
    model.train()
    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=3e-5)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target, size_average=False).item()
        loss.backward()
        optimizer.step()
        if verbose:
            if batch_idx % log_interval == 0:
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch,
                        batch_idx * len(data),
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(data),
                        loss.item(),
                    )
                )


def test(model, test_loader):
    test_metrics = []

    model.eval()
    model.cuda()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
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

    print(
        "\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), test_accuracy
        )
    )
    return test_metrics


def train_sg(model, epoch, train_loader, log_interval, verbose=False):
    model.train()
    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=3e-5)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data, target)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if verbose:
            if batch_idx % log_interval == 0:
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch,
                        batch_idx * len(data),
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(data),
                        loss.data.item(),
                    )
                )

