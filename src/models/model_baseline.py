from torch import Tensor
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, PointNetConv, NNConv
from src.models.MLP_torch_geo import MLP
from src.utils.utilities import *


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, conv_number, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.hidden = nn.ModuleList()
        for k in range(conv_number - 2):
            self.conv = GCNConv(hidden_channels, hidden_channels)
            self.hidden.append(self.conv)
            self.add_module("hidden_layer" + str(k), self.hidden[-1])
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        # x: Node feature matrix of shape [num_nodes, in_channels]
        # edge_index: Graph connectivity matrix of shape [2, num_edges]
        x = self.conv1(x, edge_index).relu()
        for layer in self.hidden:
            x = layer(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x


class GCN_Net(torch.nn.Module):
    def __init__(
        self, width, ker_width, depth, in_width=1, out_width=1, net_type="gcn"
    ):
        super(GCN_Net, self).__init__()
        self.depth = depth
        self.width = width

        self.fc_in = torch.nn.Linear(in_width, width)
        self.net_type = net_type

        if net_type == "gcn":
            self.conv1 = GCNConv(width, width)
            self.conv2 = GCNConv(width, width)
            self.conv3 = GCNConv(width, width)
            self.conv4 = GCNConv(width, width)
        elif net_type == "pointnet":
            self.conv1 = PointNetConv(
                MLP(channel_list=[width + 2, width], batch_norm=False),
                MLP(channel_list=[width, width], batch_norm=False),
            )
            self.conv2 = PointNetConv(
                MLP(channel_list=[width + 2, width], batch_norm=False),
                MLP(channel_list=[width, width], batch_norm=False),
            )
            self.conv3 = PointNetConv(
                MLP(channel_list=[width + 2, width], batch_norm=False),
                MLP(channel_list=[width, width], batch_norm=False),
            )
            self.conv4 = PointNetConv(
                MLP(channel_list=[width + 2, width], batch_norm=False),
                MLP(channel_list=[width, width], batch_norm=False),
            )
        elif net_type == "mlp":
            self.conv1 = torch.nn.Linear(width, width)
            self.conv2 = torch.nn.Linear(width, width)
            self.conv3 = torch.nn.Linear(width, width)
            self.conv4 = torch.nn.Linear(width, width)

        self.fc_out1 = torch.nn.Linear(width, ker_width)
        self.fc_out2 = torch.nn.Linear(ker_width, 1)

    def forward(self, data):
        x, X, edge_index = data.x, data.pos, data.edge_index
        # if self.net_type == 'gcn':
        x = torch.cat([X, x], dim=1)
        # print(X.device)

        x = self.fc_in(x)

        for t in range(self.depth):
            if self.net_type == "gcn":
                x = x + self.conv1(x, edge_index)
                x = F.relu(x)
                x = x + self.conv2(x, edge_index)
                x = F.relu(x)
                x = x + self.conv3(x, edge_index)
                x = F.relu(x)
                x = x + self.conv4(x, edge_index)
                x = F.relu(x)
            elif self.net_type == "pointnet":
                x = x + self.conv1(x, X, edge_index)
                x = F.relu(x)
                x = x + self.conv2(x, X, edge_index)
                x = F.relu(x)
                x = x + self.conv3(x, X, edge_index)
                x = F.relu(x)
                x = x + self.conv4(x, X, edge_index)
                x = F.relu(x)
            elif self.net_type == "mlp":
                x = self.conv1(x)
                x = F.relu(x)
                x = self.conv2(x)
                x = F.relu(x)
                x = self.conv3(x)
                x = F.relu(x)
                x = self.conv4(x)
                x = F.relu(x)
        x = F.leaky_relu(self.fc_out1(x))
        x = self.fc_out2(x)
        return x


class KernelGKN(torch.nn.Module):
    def __init__(self, width, ker_width, depth, ker_in, in_width=1, out_width=1):
        super(KernelGKN, self).__init__()
        self.depth = depth

        self.fc1 = torch.nn.Linear(in_width, width)

        kernel = DenseNet(
            [ker_in, ker_width // 2, ker_width, width ** 2], torch.nn.ReLU
        )
        self.conv1 = NNConv(
            width, width, kernel, aggr="mean", root_weight=False, bias=False
        )

        self.fc2 = torch.nn.Linear(width, ker_width)
        self.fc3 = torch.nn.Linear(ker_width, 1)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.fc1(x)
        for k in range(self.depth):
            x = self.conv1(x, edge_index, edge_attr)
            if k != self.depth - 1:
                x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x


def Multi_Layer_Perceptron(dim):
    return torch.nn.Sequential(
        *[
            torch.nn.Sequential(torch.nn.Linear(dim[i - 1], dim[i]), torch.nn.ReLU(),)
            for i in range(1, len(dim))
        ]
    )
