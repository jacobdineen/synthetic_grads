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


def train(model, epoch, train_loader, log_interval):
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    model.train()
    model.cuda()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
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


#             train_counter.append((batch_idx * 64) +
#                                  ((epoch - 1) * len(train_loader.dataset)))


def test(model, test_loader):
    test_metrics = []

    model.eval()
    model.cuda()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            test_loss += F.nll_loss(
                output.cpu(), target.cpu(), size_average=False
            ).item()
            pred = output.cpu().data.max(1, keepdim=True)[1]
            correct += pred.cpu().eq(target.cpu().data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)

    test_accuracy = 100.0 * correct / len(test_loader.dataset)
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


def train_sg(model, epoch, train_loader, log_interval):
    model.train()
    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data, target)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
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

