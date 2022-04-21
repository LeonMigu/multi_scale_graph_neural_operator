from timeit import default_timer
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from src.utils.utilities import *
from src.models.model_baseline import GCN_Net

################
# MAIN PROGRAM #
################

if __name__ == "__main__":
    ########################################################################
    #
    #  Hyperparameters
    #
    ########################################################################

    PATH = ""
    TRAIN_PATH = "/data/migus/mgkn_revamped/burgers_data_R10.mat"
    TEST_PATH = "/data/migus/mgkn_revamped/burgers_data_R10.mat"

    r = 8
    s = 2 ** 13 // r
    K = s

    n = s
    k = 2

    m = [320]

    if len(m) == 1:
        radius_inner = [0.5 / 8]
        radius_inter = 0
    if len(m) == 2:
        radius_inner = [0.5 / 8, 0.5 / 4]
        radius_inter = [0.5 / 8 * 1.41]
    if len(m) == 3:
        radius_inner = [0.5 / 8, 0.5 / 4, 0.5 / 2]
        radius_inter = [0.5 / 8 * 1.41, 0.5 / 4 * 1.41]
    if len(m) == 4:
        radius_inner = [0.5 / 8, 0.5 / 4, 0.5 / 2, 0.5]
        radius_inter = [0.5 / 8 * 1.41, 0.5 / 4 * 1.41, 0.5 / 2 * 1.41]

    batch_size = 1
    batch_size2 = 1

    ntrain = 100
    ntest = 100

    level = len(m)
    print("resolution", s)

    width = 54
    ker_width = 128
    depth = 1
    node_features = 3

    epochs = 200
    init_gain = 0.8
    learning_rate = 0.001
    scheduler_step = 10
    scheduler_gamma = 0.85

    #######
    # GPU #
    #######

    gpu = 0
    torch.cuda.set_device(gpu)
    print(f"Using GPU: {torch.cuda.current_device()}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    # if device=='cuda':
    #     torch.cuda.set_device(0)
    torch.set_num_threads(1)

    ###############
    # RANDOM SEED for data generation #
    ###############

    RANDOM_SEED = 0

    torch.backends.cudnn.deterministic = False
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)

    ################################################################
    # load data
    ################################################################

    runtime = np.zeros(2,)

    t1 = default_timer()

    reader = MatReader(TRAIN_PATH, device_cuda=gpu)
    train_a = reader.read_field("a")[:ntrain, ::r].reshape(ntrain, -1)
    train_u = reader.read_field("u")[:ntrain, ::r].reshape(ntrain, -1)

    reader.load_file(TEST_PATH)
    test_a = reader.read_field("a")[-ntest:, ::r].reshape(ntest, -1)
    test_u = reader.read_field("u")[-ntest:, ::r].reshape(ntest, -1)

    a_normalizer = GaussianNormalizer(train_a)
    train_a = a_normalizer.encode(train_a)
    test_a = a_normalizer.encode(test_a)

    u_normalizer = UnitGaussianNormalizer(train_u)
    train_u = u_normalizer.encode(train_u)

    X, edge_index, _ = grid_edge(2 ** 5, 2 ** 5)

    data_train = []
    for j in range(ntrain):
        for i in range(k):
            x = torch.cat([train_a[j].reshape(-1, 1)], dim=1)
            data_train.append(Data(x=x, pos=X, y=train_u[j], edge_index=edge_index))

    print(x.shape)
    print(edge_index.shape)

    data_test = []
    for j in range(ntest):
        x = torch.cat([test_a[j].reshape(-1, 1)], dim=1)
        data_test.append(Data(x=x, pos=X, y=test_u[j], edge_index=edge_index))

    print(x.shape)
    print(edge_index.shape)

    train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(data_test, batch_size=batch_size2, shuffle=False)

    ########################################################################
    #
    #  Training
    #
    ########################################################################
    ###############
    # RANDOM SEED for training #
    ###############

    RANDOM_SEED = 1

    torch.backends.cudnn.deterministic = False
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)

    t2 = default_timer()

    print("preprocessing finished, time used:", t2 - t1)

    net_type = "mlp" # choice between mlp, gcn and pointnet
    model = GCN_Net(
        width=width,
        ker_width=ker_width,
        depth=depth,
        in_width=node_features,
        out_width=1,
        net_type=net_type,
    ).to(device)

    init_weights(model, init_type="orthogonal", init_gain=init_gain)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=5e-4
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=scheduler_step, gamma=scheduler_gamma
    )

    myloss = LpLoss(size_average=False)
    u_normalizer.to(device)
    model.train()
    for ep in range(epochs):
        t1 = default_timer()
        train_mse = 0.0
        train_l2 = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            mse = F.mse_loss(out.view(-1, 1), batch.y.view(-1, 1))
            # mse.backward()

            l2 = myloss(
                u_normalizer.decode(out.view(batch_size, -1),),
                u_normalizer.decode(batch.y.view(batch_size, -1),),
            )
            l2.backward()

            optimizer.step()
            train_mse += mse.item()
            train_l2 += l2.item()

        scheduler.step()
        t2 = default_timer()
        ttime_epoch = t2 - t1
        ttrain = train_l2 / (ntrain * k)

        print(ep, t2 - t1, train_mse / len(train_loader), train_l2 / (ntrain * k))

        runtime[0] = t2 - t1
        t1 = default_timer()
        model.eval()
        test_l2 = 0.0
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                out = model(batch)
                out = u_normalizer.decode(out.view(batch_size2, -1),)
                test_l2 += myloss(out, batch.y.view(batch_size2, -1)).item()

        ttest = test_l2 / ntest
        t2 = default_timer()
        print(ep, t2 - t1, test_l2 / ntest)

        runtime[1] = t2 - t1
    model_total_params = sum(p.numel() for p in model.parameters())
    model_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )