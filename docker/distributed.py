import numpy as np
import tensorflow as tf

tf.compat.v1.disable_eager_execution()
cluster = tf.train.ClusterSpec({"local": ["localhost:2222", "localhost:2223"]})

x = tf.compat.v1.placeholder(tf.float32, 100)


with tf.device("/job:local/task:1"):
    first_batch = tf.slice(x, [0], [50])
    mean1 = tf.reduce_mean(first_batch)
    print(mean1)

with tf.device("/job:local/task:0"):
    second_batch = tf.slice(x, [50], [-1])
    mean2 = tf.reduce_mean(second_batch)
    mean = (mean1 + mean2) / 2
    print(mean)


with tf.compat.v1.Session("grpc://localhost:2222") as sess:
    result = sess.run(mean, feed_dict={x: np.random.random(100)})
    print(result)


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
from sklearn.metrics import precision_recall_fscore_support
import dni

warnings.filterwarnings("ignore")


def one_hot(indexes, n_classes):
    result = torch.FloatTensor(indexes.size() + (n_classes,))
    result = result.cpu()
    result.zero_()
    indexes_rank = len(indexes.size())
    result.scatter_(
        dim=indexes_rank, index=indexes.data.unsqueeze(dim=indexes_rank), value=1
    )
    return Variable(result)


class Four_Layer_SG(nn.Module):
    """4-layer CNN as described in the paper"""

    def __init__(self):
        super(Four_Layer_SG, self).__init__()

        self.block_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=(12, 12),
                stride=(4, 4),
                padding=(4, 4),
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(12 * 12 * 32, 512),
            nn.ReLU(True),
            nn.Dropout(p=0.65),
            nn.Linear(512, 10),
        )

        self.backward_interface_1 = dni.BackwardInterface(
            dni.BasicSynthesizer(output_dim=12, n_hidden=1)
        )

    def forward(self, x, y=None):
        with tf.device("/job:local/task:1"):
            x = self.block_1(x)
            if self.training:
                context = one_hot(y, 10)
                with dni.synthesizer_context(context):
                    x = self.backward_interface_1(x)
        with tf.device("/job:local/task:0"):
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
        with tf.compat.v1.Session("grpc://localhost:2222") as sess:

            return F.log_softmax(x, dim=1)


class Four_Layer_SG_2(nn.Module):
    """4-layer CNN as described in the paper"""

    def __init__(self):
        super(Four_Layer_SG_2, self).__init__()

        self.classifier = nn.Sequential(
            nn.Linear(12 * 12 * 32, 512),
            nn.ReLU(True),
            nn.Dropout(p=0.65),
            nn.Linear(512, 10),
        )

    def forward(self, x, y=None):
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)


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


if __name__ == "__main__":
    n_epochs = 100
    log_interval = 20
    dirpath = os.getcwd()
    output_path = os.path.join(dirpath, "output.csv")

    # set torch device
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    batch_size_train = 256  # specified in the paper
    train_loader, _ = get_dataloaders(1000, 200, batch_size_train)

    model = Four_Layer_SG_1()
    stats = []

    for epoch in tqdm(range(1, n_epochs + 1), leave=False):
        start_time = time()
        output, target = train(model, epoch, train_loader, log_interval, verbose=False)
        torch.save(output, "./data/x.pt")
        torch.save(target, "./data/y.pt")
