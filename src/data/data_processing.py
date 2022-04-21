from torch_geometric.data import Data
from src.utils.utilities import RandomMultiMeshGenerator, RandomMeshGenerator
import torch


def convert_data_to_mesh(
    data_input_list, data_output, ntrain, k, s, m, radius_inner, radius_inter=0
):
    """

    Parameters
    ----------
    data_input_list: list of input data (for e.g. for 1D Burger it would be train_a and for Darcy flow [train_a, train_a_smooth, train_a_gradx, train_a_grady]
    data_output: output data, usually is train_u for training
    ntrain: number of training data
    k
    s
    m: The number of sampled nodes for each level
    radius_inner: The inner radius of the kernel integration
    radius_inter: The inter radius of the kernel integration

    Returns
    -------

    """
    assert data_output.shape[0] == ntrain
    level = len(m)

    # Depending on the data, the mesh is bigger. For now it is handcrafted, so need to understand why the real_space and mesh_size dimensions are different for Burger and Darcy
    if len(data_input_list) == 4:
        real_space = [[0, 1], [0, 1]]
        mesh_size = [s, s]
    else:
        real_space = [[0, 1]]
        mesh_size = [s]
    # Depending on the level, the mesh is different
    if level > 1:
        meshgenerator = RandomMultiMeshGenerator(
            real_space=real_space, mesh_size=mesh_size, level=level, sample_sizes=m
        )
        data_train = []
        for j in range(ntrain):
            for i in range(k):
                idx, idx_all = meshgenerator.sample()
                grid, grid_all = meshgenerator.get_grid()
                (
                    edge_index,
                    edge_index_down,
                    edge_index_up,
                ) = meshgenerator.ball_connectivity(radius_inner, radius_inter)
                (
                    edge_index_range,
                    edge_index_down_range,
                    edge_index_up_range,
                ) = meshgenerator.get_edge_index_range()
                edge_attr, edge_attr_down, edge_attr_up = meshgenerator.attributes(
                    theta=data_input_list[0][j, :]
                )
                data_point = [grid_all]
                for data_input in data_input_list:
                    data_point.append(data_input[j, idx_all].reshape(-1, 1))
                x = torch.cat(data_point, dim=1)
                data_train.append(
                    Data(
                        x=x,
                        y=data_output[j, idx[0]],
                        edge_index_mid=edge_index,
                        edge_index_down=edge_index_down,
                        edge_index_up=edge_index_up,
                        edge_index_range=edge_index_range,
                        edge_index_down_range=edge_index_down_range,
                        edge_index_up_range=edge_index_up_range,
                        edge_attr_mid=edge_attr,
                        edge_attr_down=edge_attr_down,
                        edge_attr_up=edge_attr_up,
                        sample_idx=idx[0],
                    )
                )
    else:
        meshgenerator = RandomMeshGenerator(
            real_space=real_space, mesh_size=mesh_size, sample_size=m[0]
        )
        data_train = []
        for j in range(ntrain):
            for i in range(k):
                idx = meshgenerator.sample()
                grid = meshgenerator.get_grid()
                edge_index = meshgenerator.ball_connectivity(radius_inner[0])
                edge_attr = meshgenerator.attributes(theta=data_input_list[0][j, :])
                data_point = [grid]
                for data_input in data_input_list:
                    data_point.append(data_input[j, idx].reshape(-1, 1))
                x = torch.cat(data_point, dim=1)
                data_train.append(
                    Data(
                        x=x,
                        y=data_output[j, idx],
                        edge_index=edge_index,
                        edge_attr=edge_attr,
                        sample_idx=idx,
                    )
                )

    return data_train
