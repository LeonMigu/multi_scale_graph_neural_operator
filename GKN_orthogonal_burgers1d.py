from timeit import default_timer
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from src.utils.utilities import *
from src.models.model_baseline import KernelGKN
from src.data.data_processing import convert_data_to_mesh

################
# MAIN PROGRAM #
################

if __name__ == "__main__":
    ########################################################################
    #
    #  Hyperparameters
    #
    ########################################################################

    PATH = "."
    TRAIN_PATH = f"{PATH}/data/burgers_data_R10.mat"
    TEST_PATH = f"{PATH}/data/burgers_data_R10.mat"

    r = 16
    s = 2 ** 13 // r
    K = s

    n = s
    k = 2

    m = [512]

    radius_inner = [0.5 / 8]
    radius_inter = 0

    batch_size = 1
    batch_size2 = 1

    ntrain = 100
    ntest = 100

    # rbf_sigma = 0.2

    level = len(m)
    print("resolution", s)

    width = 64
    ker_width = 256
    depth = 4
    edge_features = 4
    node_features = 2

    epochs = 200
    init_gain = 0.8
    learning_rate = 0.0001
    scheduler_step = 10
    scheduler_gamma = 0.85

    #######
    # GPU #
    #######

    gpu = 1
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

    data_train = convert_data_to_mesh(
        [train_a], train_u, ntrain, k, s, m, radius_inner, radius_inter
    )

    data_test = convert_data_to_mesh(
        [test_a], test_u, ntest, k, s, m, radius_inner, radius_inter
    )
    print(data_test[0])
    # pdb.set_trace()
    train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(data_test, batch_size=batch_size2, shuffle=False)

    ########################################################################
    #
    #  Training
    #
    ########################################################################

    RANDOM_SEED = 1

    torch.backends.cudnn.deterministic = False
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)

    t2 = default_timer()

    print("preprocessing finished, time used:", t2 - t1)

    # pdb.set_trace()
    model = KernelGKN(width, ker_width, depth, edge_features, node_features).cuda()

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
