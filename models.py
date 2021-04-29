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

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), (4, 4)))
        x = x.view(-1, 6400)
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

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = F.relu(F.max_pool2d(self.conv4(x), 2))
        x = x.view(-1, 4 * 4 * 128)
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
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(6, 6))
        self.fc1 = nn.Linear(6400, 128)
        self.fc2 = nn.Linear(128, 10)
        self.backward_interface_1 = dni.BackwardInterface(
            dni.BasicSynthesizer(output_dim=10, n_hidden=100)
        )

    def forward(self, x, y=None):
        x = F.relu(F.max_pool2d(self.conv1(x), (4, 4)))
        if self.training:
            context = one_hot(y, 10)
            with dni.synthesizer_context(context):
                x = self.backward_interface_1(x)
        x = x.view(-1, 6400)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class Eight_Layer_SG(nn.Module):
    """8-layer CNN as described in the paper"""

    def __init__(self):
        super(Eight_Layer_SG, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)

        self.conv2 = nn.Conv2d(32, 32, kernel_size=3)

        self.backward_interface_1 = dni.BackwardInterface(
            dni.BasicSynthesizer(output_dim=22, n_hidden=50)
        )

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3)

        self.backward_interface_2 = dni.BackwardInterface(
            dni.BasicSynthesizer(output_dim=10, n_hidden=50)
        )

        self.conv4 = nn.Conv2d(64, 128, kernel_size=3)

        self.backward_interface_3 = dni.BackwardInterface(
            dni.BasicSynthesizer(output_dim=4, n_hidden=50)
        )

        self.fc1 = nn.Linear(4 * 4 * 128, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x, y=None):
        x = F.relu(self.conv1(x))

        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        if self.training:
            context = one_hot(y, 10)
            with dni.synthesizer_context(context):
                x = self.backward_interface_1(x)

        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        if self.training:
            context = one_hot(y, 10)
            with dni.synthesizer_context(context):
                x = self.backward_interface_2(x)

        x = F.relu(F.max_pool2d(self.conv4(x), 2))
        if self.training:
            context = one_hot(y, 10)
            with dni.synthesizer_context(context):
                x = self.backward_interface_3(x)

        x = x.view(-1, 4 * 4 * 128)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


class VGG16_SG(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG16_SG, self).__init__()

        # calculate same padding:
        # (w - k + 2*p)/s + 1 = o
        # => p = (s(o-1) - w + k)/2
        self.init_conv = nn.Conv2d(1, 3, 1)
        self.block_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=64,
                kernel_size=(3, 3),
                stride=(1, 1),
                # (1(32-1)- 32 + 3)/2 = 1
                padding=1,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=1,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        )

        self.block_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=1,
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=1,
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        )

        self.block_3 = nn.Sequential(
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=1,
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=1,
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=1,
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        )

        self.block_4 = nn.Sequential(
            nn.Conv2d(
                in_channels=256,
                out_channels=512,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=1,
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=1,
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=1,
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        )

        self.block_5 = nn.Sequential(
            nn.Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=1,
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=1,
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=1,
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        )

        self.backward_interface_1 = dni.BackwardInterface(
            dni.BasicSynthesizer(output_dim=24, n_hidden=10)
        )

        self.backward_interface_2 = dni.BackwardInterface(
            dni.BasicSynthesizer(output_dim=12, n_hidden=10)
        )

        self.backward_interface_3 = dni.BackwardInterface(
            dni.BasicSynthesizer(output_dim=6, n_hidden=10)
        )

        self.backward_interface_4 = dni.BackwardInterface(
            dni.BasicSynthesizer(output_dim=3, n_hidden=10)
        )

        self.backward_interface_5 = dni.BackwardInterface(
            dni.BasicSynthesizer(output_dim=1, n_hidden=10)
        )

        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.65),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.65),
            nn.Linear(4096, num_classes),
        )

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                nn.init.kaiming_uniform_(
                    m.weight, mode="fan_in", nonlinearity="leaky_relu"
                )
                #                 nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    m.bias.detach().zero_()

        # self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

    def forward(self, x, y=None):
        x = self.init_conv(x)
        x = self.block_1(x)
        if self.training:
            context = one_hot(y, 10)
            with dni.synthesizer_context(context):
                x = self.backward_interface_1(x)
        x = self.block_2(x)
        if self.training:
            context = one_hot(y, 10)
            with dni.synthesizer_context(context):
                x = self.backward_interface_2(x)
        x = self.block_3(x)
        if self.training:
            context = one_hot(y, 10)
            with dni.synthesizer_context(context):
                x = self.backward_interface_3(x)
        x = self.block_4(x)
        if self.training:
            context = one_hot(y, 10)
            with dni.synthesizer_context(context):
                x = self.backward_interface_4(x)
        x = self.block_5(x)
        if self.training:
            context = one_hot(y, 10)
            with dni.synthesizer_context(context):
                x = self.backward_interface_5(x)
        # x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        # probas = nn.Softmax(logits)
        return F.log_softmax(x, dim=1)
        # return logits
