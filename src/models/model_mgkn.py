import torch.nn.functional as F
from torch_geometric.nn import NNConv
from src.utils.utilities import *

########################################################################
#
#  The neural networks architecture
#
########################################################################


class KernelInduced(torch.nn.Module):
    def __init__(self, width, ker_width, depth, ker_in, points, level, in_width=1, out_width=1, cycle_type='v',
                 shared=True, skip_connection=True, in_cycle_shared=True):
        super(KernelInduced, self).__init__()
        self.depth = depth
        self.width = width
        self.level = level
        self.points = points
        self.points_total = np.sum(points)
        self.cycle_type = cycle_type

        # in
        self.fc_in = torch.nn.Linear(in_width, width)
        self.convs = {}

        if level == 2:
            net_order = [
                '0-1-0',  # downward-0
                '1-1-0', '1-0-0', '0-0-0'  # upward-0
            ]
        elif level == 1:
            net_order = ['0-0-0']
        if self.cycle_type == 'v':
            if level == 4:
                net_order = [
                    '0-1-0', '1-2-0', '2-3-0',  # downward-0
                    '3-3-0', '3-2-0', '2-2-0', '2-1-0', '1-1-0', '1-0-0', '0-0-0'  # upward-0
                ]
            elif level == 3:
                net_order = [
                    '0-1-0', '1-2-0',  # downward-0
                    '2-2-0', '2-1-0', '1-1-0', '1-0-0', '0-0-0'  # upward-0
                ]
        elif self.cycle_type == 'f':
            if level == 4:
                if not in_cycle_shared:
                    net_order = [
                        '0-1-0', '1-2-0', '2-3-0',  # downward-0
                        '3-3-0', '3-2-0', '2-2-0',  # upward-0
                        '2-3-1',  # downward-1
                        '3-3-1', '3-2-1', '2-2-1', '2-1-1', '1-1-1',  # upward-1
                        '1-2-2', '2-3-2',  # downward-2
                        '3-3-2', '3-2-2', '2-2-2', '2-1-2', '1-1-2', '1-0-2', '0-0-2',  # upward-2
                    ]
                else:
                    net_order = [
                        '0-1-0', '1-2-0', '2-3-0',  # downward-0
                        '3-3-0', '3-2-0', '2-2-0',  # upward-0
                        '2-3-0',  # downward-1
                        '3-3-0', '3-2-0', '2-2-0', '2-1-0', '1-1-0',  # upward-1
                        '1-2-0', '2-3-0',  # downward-2
                        '3-3-0', '3-2-0', '2-2-0', '2-1-0', '1-1-0', '1-0-0', '0-0-0',  # upward-2
                    ]
            elif level == 3:
                if not in_cycle_shared:
                    net_order = [
                        '0-1-0', '1-2-0',  # downward-0
                        '2-2-0', '2-1-0', '1-1-0',  # upward-1
                        '1-2-1',  # downward-2
                        '2-2-1', '2-1-1', '1-1-1', '1-0-1', '0-0-1',  # upward-2
                    ]
                else:
                    net_order = [
                        '0-1-0', '1-2-0',  # downward-0
                        '2-2-0', '2-1-0', '1-1-0',  # upward-1
                        '1-2-0',  # downward-2
                        '2-2-0', '2-1-0', '1-1-0', '1-0-0', '0-0-0',  # upward-2
                    ]

        elif self.cycle_type == 'w':
            if level == 4:
                if not in_cycle_shared:
                    net_order = [
                        '0-1-0', '1-2-0', '2-3-0',  # downward-0
                        '3-3-0', '3-2-0', '2-2-0',  # upward-0
                        '2-3-1',  # downward-1
                        '3-3-1', '3-2-1', '2-2-1', '2-1-1', '1-1-1',  # upward-1
                        '1-2-2', '2-3-2',  # downward-2
                        '3-3-2', '3-2-2', '2-2-2',  # upward-2
                        '2-3-3',  # downward-3
                        '3-3-3', '3-2-3', '2-2-3', '2-1-3', '1-1-3', '1-0-3', '0-0-3',  # upward-3
                    ]
                else:
                    net_order = [
                        '0-1-0', '1-2-0', '2-3-0',  # downward-0
                        '3-3-0', '3-2-0', '2-2-0',  # upward-0
                        '2-3-0',  # downward-1
                        '3-3-0', '3-2-0', '2-2-0', '2-1-0', '1-1-0',  # upward-1
                        '1-2-0', '2-3-0',  # downward-2
                        '3-3-0', '3-2-0', '2-2-0',  # upward-2
                        '2-3-0',  # downward-3
                        '3-3-0', '3-2-0', '2-2-0', '2-1-0', '1-1-0', '1-0-0', '0-0-0',  # upward-3
                    ]
            elif level == 3:
                if not in_cycle_shared:
                    net_order = [
                        '0-1-0', '1-2-0',  # downward-0
                        '2-2-0', '2-1-1', '1-1-1',  # upward-1
                        '1-2-1', '2-2-1', '2-1-2', '1-1-2', '1-0-2', '0-0-2',  # upward-3
                    ]
                else:
                    net_order = [
                        '0-1-0', '1-2-0',  # downward-0
                        '2-2-0', '2-1-0', '1-1-0',  # upward-1
                        '1-2-0', '2-2-0', '2-1-0', '1-1-0', '1-0-0', '0-0-0',  # upward-3
                    ]

        self.net_order = list()
        if not shared:
            for d in range(depth):
                self.net_order += [f'{idx}-{d}' for idx in net_order]
        else:
            self.net_order = [f'{idx}-0' for idx in net_order] * depth

        for idx in self.net_order:
            pre, post = [int(x) for x in idx.split('-')][:2]
            if pre < post:
                ker_width_l = ker_width // (2 ** post)
                kernel_l = DenseNet([ker_in, ker_width_l, width ** 2], torch.nn.ReLU)
            elif pre > post:
                ker_width_l = ker_width // (2 ** pre)
                kernel_l = DenseNet([ker_in, ker_width_l, width ** 2], torch.nn.ReLU)
            else:
                ker_width_l = ker_width // (2 ** pre)
                kernel_l = DenseNet([ker_in, ker_width_l, ker_width_l, width ** 2], torch.nn.ReLU)

            self.convs[idx] = NNConv(width, width, kernel_l, aggr='mean', root_weight=False, bias=False)

        self.convs = nn.ModuleDict(self.convs)

        # out
        self.fc_out1 = torch.nn.Linear(width, ker_width)
        self.fc_out2 = torch.nn.Linear(ker_width, 1)
        self.skip_connection = skip_connection

    def forward(self, data):
        if self.level > 1:
            edge_index_down, edge_attr_down, range_down = data.edge_index_down, data.edge_attr_down, data.edge_index_down_range
            edge_index_mid, edge_attr_mid, range_mid = data.edge_index_mid, data.edge_attr_mid, data.edge_index_range
            edge_index_up, edge_attr_up, range_up = data.edge_index_up, data.edge_attr_up, data.edge_index_up_range
        else:
            edge_index_mid, edge_attr_mid = data.edge_index, data.edge_attr
            range_mid = [[0, None]]

        x = self.fc_in(data.x)
        for idx in self.net_order:
            pre, post = [int(x) for x in idx.split('-')][:2]
            if pre < post:
                edge_index = edge_index_down[:, range_down[pre, 0]:range_down[pre, 1]]
                edge_attr = edge_attr_down[range_down[pre, 0]:range_down[pre, 1], :]
            elif pre > post:
                edge_index = edge_index_up[:, range_up[post, 0]:range_up[post, 1]]
                edge_attr = edge_attr_up[range_up[post, 0]:range_up[post, 1], :]
            else:
                edge_index = edge_index_mid[:, range_mid[pre, 0]:range_mid[pre, 1]]
                edge_attr = edge_attr_mid[range_mid[pre, 0]:range_mid[pre, 1], :]

            if self.skip_connection:
                x = x + self.convs[idx](x, edge_index, edge_attr)
            else:
                x = self.convs[idx](x, edge_index, edge_attr)
            x = F.relu(x)
        x = F.relu(self.fc_out1(x[:self.points[0]]))
        x = self.fc_out2(x)
        return x
