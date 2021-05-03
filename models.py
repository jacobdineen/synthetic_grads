import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import dni
from torch.autograd import Variable
import torch


class Four_Layer(nn.Module):
    """4-layer CNN as described in the paper"""

    def __init__(self):
        super(Four_Layer, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(6, 6))
        self.fc1 = nn.Linear(6400, 128)
        self.fc2 = nn.Linear(128, 10)
        self.output = nn.Softmax(dim=1)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), (4, 4)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class Eight_Layer(nn.Module):
    """8-layer CNN as described in the paper"""

    def __init__(self):
        super(Eight_Layer, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3)
        self.fc1 = nn.Linear(4 * 4 * 128, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.output = nn.Softmax(dim=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = F.relu(F.max_pool2d(self.conv4(x), 2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


class VGG16_custom(nn.Module):
    """VGG architecture CNN as described in the paper"""

    def __init__(self):
        super(VGG16_custom, self).__init__()
        # modify the original VGG model to have a 10 unit output linear layer
        self.model = models.vgg16(False)
        self.model.classifier[6].out_features = 10
        self.conv1 = nn.Conv2d(1, 3, 1)
        self.output = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)  # reshape channels from 1 to 3 for vgg
        x = self.model(x)
        return F.log_softmax(x, dim=1)


def one_hot(indexes, n_classes):
    result = torch.FloatTensor(indexes.size() + (n_classes,))
    result = result.cuda()
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
        x = self.block_1(x)
        if self.training:
            context = one_hot(y, 10)
            with dni.synthesizer_context(context):
                x = self.backward_interface_1(x)
        x = x.view(x.size(0), -1)

        x = self.classifier(x)

        return F.log_softmax(x, dim=1)


class Eight_Layer_SG(nn.Module):
    """8-layer CNN as described in the paper"""

    def __init__(self):
        super(Eight_Layer_SG, self).__init__()
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

        self.block_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=(12, 12),
                stride=(4, 4),
                padding=(4, 4),
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(576, 128),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(64, 10),
        )

        self.backward_interface_1 = dni.BackwardInterface(
            dni.BasicSynthesizer(output_dim=12, n_hidden=256)
        )

        self.backward_interface_2 = dni.BackwardInterface(
            dni.BasicSynthesizer(output_dim=3, n_hidden=128)
        )

    def forward(self, x, y=None):
        x = self.block_1(x)
        if self.training:
            context = one_hot(y, 10)
            with dni.synthesizer_context(context):
                x = self.backward_interface_1(x)
        x = self.block_2(x)

        # if self.training:
        #     context = one_hot(y, 10)
        #     with dni.synthesizer_context(context):
        #         x = self.backward_interface_2(x)
        x = x.view(x.size(0), -1)

        x = self.classifier(x)

        return F.log_softmax(x, dim=1)


class VGG16_SG2(nn.Module):
    """VGG architecture CNN as described in the paper"""

    def __init__(self):
        super(VGG16_SG2, self).__init__()
        # modify the original VGG model to have a 10 unit output linear layer
        self.model = models.vgg16(True)
        self.model.classifier[0].in_features = 512
        self.conv1 = nn.Conv2d(1, 3, 1)
        self.block1 = self.model.features[:5]
        self.block2 = self.model.features[5:10]
        self.block3 = self.model.features[10:17]
        self.block4 = self.model.features[17:24]
        self.block5 = self.model.features[24:]
        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, 10),
        )
        self.backward_interface_1 = dni.BackwardInterface(
            dni.BasicSynthesizer(output_dim=24, n_hidden=20)
        )

        self.backward_interface_2 = dni.BackwardInterface(
            dni.BasicSynthesizer(output_dim=12, n_hidden=20)
        )
        self.backward_interface_3 = dni.BackwardInterface(
            dni.BasicSynthesizer(output_dim=6, n_hidden=20)
        )
        self.backward_interface_4 = dni.BackwardInterface(
            dni.BasicSynthesizer(output_dim=3, n_hidden=20)
        )
        self.backward_interface_5 = dni.BackwardInterface(
            dni.BasicSynthesizer(output_dim=1, n_hidden=20)
        )

    def forward(self, x, y=None):
        x = self.conv1(x)  # reshape channels from 1 to 3 for vgg
        x = self.block1(x)
        if self.training:
            context = one_hot(y, 10)
            with dni.synthesizer_context(context):
                x = self.backward_interface_1(x)
        x = self.block2(x)
        # if self.training:
        #     context = one_hot(y, 10)
        #     with dni.synthesizer_context(context):
        #         x = self.backward_interface_2(x)
        x = self.block3(x)
        # if self.training:
        #     context = one_hot(y, 10)
        #     with dni.synthesizer_context(context):
        #         x = self.backward_interface_3(x)
        x = self.block4(x)
        # if self.training:
        #     context = one_hot(y, 10)
        #     with dni.synthesizer_context(context):
        #         x = self.backward_interface_4(x)
        x = self.block5(x)
        # if self.training:
        #     context = one_hot(y, 10)
        #     with dni.synthesizer_context(context):
        #         x = self.backward_interface_5(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)

