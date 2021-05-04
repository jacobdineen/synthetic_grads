import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import dni
from torch.autograd import Variable
import torch


def one_hot(indexes, n_classes):
    result = torch.FloatTensor(indexes.size() + (n_classes,))
    result = result.cpu()
    result.zero_()
    indexes_rank = len(indexes.size())
    result.scatter_(
        dim=indexes_rank, index=indexes.data.unsqueeze(dim=indexes_rank), value=1
    )
    return Variable(result)


class Four_Layer_SG_1(nn.Module):
    """4-layer CNN as described in the paper"""

    def __init__(self):
        super(Four_Layer_SG_1, self).__init__()

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

        self.backward_interface_1 = dni.BackwardInterface(
            dni.BasicSynthesizer(output_dim=12, n_hidden=1)
        )

    def forward(self, x, y=None):
        x = self.block_1(x)
        if self.training:
            context = one_hot(y, 10)
            with dni.synthesizer_context(context):
                x = self.backward_interface_1(x)
        return x


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
