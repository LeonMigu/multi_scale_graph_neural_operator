import torch.nn.functional as F

from torch_geometric.data import DataLoader
from src.utils.utilities import *

from timeit import default_timer
from src.models.model_mgkn import KernelInduced
from src.data.data_processing import convert_data_to_mesh

torch.manual_seed(0)
np.random.seed(0)


PATH = '.'
TRAIN_PATH = f'{PATH}/data/piececonst_r241_N1024_smooth1.mat'
TEST_PATH = f'{PATH}/data/piececonst_r241_N1024_smooth2.mat'


r = 1
s = int(((241 - 1)/r) + 1)
n = s**2
k = 1

# this is too large
# m = [6400, 1600, 400, 100, 25]
# radius_inner = [0.5/16, 0.5/8, 0.5/4, 0.5/2, 0.5]
# radius_inter = [0.5/16 * 1.41, 0.5/8* 1.41, 0.5/4* 1.41, 0.5/2* 1.41]

for case in [1]:

    print('!!!!!!!!!!!!!! case ', case, ' !!!!!!!!!!!!!!!!!!!!!!!!')

    if case == 0:
        m = [1600, 400, 100, 25]
        radius_inner = [ 0.5/8, 0.5/4, 0.5/2, 0.5]
        radius_inter = [0.5/8* 1.41, 0.5/4* 1.41, 0.5/2* 1.41]

    if case == 1:
        m = [1600, 400, 100]
        radius_inner = [0.5/8, 0.5/4, 0.5/2]
        radius_inter = [0.5/8* 1.41, 0.5/4* 1.41]

    if case == 2:
        m = [1600, 400]
        radius_inner = [0.5/8, 0.5/4]
        radius_inter = [0.5/8* 1.41]

    level = len(m)
    print('resolution', s)

    ntrain = 100
    ntest = 100

    # don't change this
    batch_size = 1
    batch_size2 = 1

    width = 64
    ker_width = 256
    depth = 4
    edge_features = 6
    node_features = 6

    epochs = 200
    learning_rate = 0.001
    scheduler_step = 10
    scheduler_gamma = 0.8

    path = f'neurips1_multigraph_s'+str(s)+'_ntrain'+str(ntrain)+'_kerwidth'+str(ker_width) + '_m0' + str(m[0])
    path_model = f'{PATH}/model/' + path
    path_train_err = f'{PATH}/results/' + path + 'train.txt'
    path_test_err = f'{PATH}/results/' + path + 'test.txt'
    path_runtime = f'{PATH}/results/' + path + 'time.txt'
    path_image = f'{PATH}/results/' + path

    runtime = np.zeros(2,)

    t1 = default_timer()


    reader = MatReader(TRAIN_PATH)
    train_a = reader.read_field('coeff')[:ntrain,::r,::r].reshape(ntrain,-1)
    train_a_smooth = reader.read_field('Kcoeff')[:ntrain,::r,::r].reshape(ntrain,-1)
    train_a_gradx = reader.read_field('Kcoeff_x')[:ntrain,::r,::r].reshape(ntrain,-1)
    train_a_grady = reader.read_field('Kcoeff_y')[:ntrain,::r,::r].reshape(ntrain,-1)
    train_u = reader.read_field('sol')[:ntrain,::r,::r].reshape(ntrain,-1)

    reader.load_file(TEST_PATH)
    test_a = reader.read_field('coeff')[:ntest,::r,::r].reshape(ntest,-1)
    test_a_smooth = reader.read_field('Kcoeff')[:ntest,::r,::r].reshape(ntest,-1)
    test_a_gradx = reader.read_field('Kcoeff_x')[:ntest,::r,::r].reshape(ntest,-1)
    test_a_grady = reader.read_field('Kcoeff_y')[:ntest,::r,::r].reshape(ntest,-1)
    test_u = reader.read_field('sol')[:ntest,::r,::r].reshape(ntest,-1)


    a_normalizer = GaussianNormalizer(train_a)
    train_a = a_normalizer.encode(train_a)
    test_a = a_normalizer.encode(test_a)
    as_normalizer = GaussianNormalizer(train_a_smooth)
    train_a_smooth = as_normalizer.encode(train_a_smooth)
    test_a_smooth = as_normalizer.encode(test_a_smooth)
    agx_normalizer = GaussianNormalizer(train_a_gradx)
    train_a_gradx = agx_normalizer.encode(train_a_gradx)
    test_a_gradx = agx_normalizer.encode(test_a_gradx)
    agy_normalizer = GaussianNormalizer(train_a_grady)
    train_a_grady = agy_normalizer.encode(train_a_grady)
    test_a_grady = agy_normalizer.encode(test_a_grady)

    u_normalizer = UnitGaussianNormalizer(train_u)
    train_u = u_normalizer.encode(train_u)
    # test_u = y_normalizer.encode(test_u)


    data_train = convert_data_to_mesh([train_a, train_a_smooth, train_a_gradx, train_a_grady], train_u, ntrain, k,
                                        s, m, radius_inner, radius_inter)
    data_test = convert_data_to_mesh([test_a, test_a_smooth, test_a_gradx, test_a_grady], test_u, ntest, k, s,
                                        m, radius_inner, radius_inter)

    train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(data_test, batch_size=batch_size2, shuffle=False)

    t2 = default_timer()

    print('preprocessing finished, time used:', t2-t1)
    device = torch.device('cuda:0')

    cycle_type = "f"
    print('cycle type:', cycle_type)
    in_cycle_shared = True
    shared = False
    skip_connection = True
    print('in cycle shared:', in_cycle_shared, 'iteration shared:', shared)
    

    model = KernelInduced(width=width, ker_width=ker_width, depth=depth, ker_in=edge_features,
                          points=m, level=level, in_width=node_features,  out_width=1, cycle_type=cycle_type, shared=shared, skip_connection=skip_connection, in_cycle_shared=in_cycle_shared).to(device)
    init_weights(model, init_type='orthogonal', init_gain=1.414)

    print('model parameters:', get_n_params(model))


    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

    myloss = LpLoss(size_average=False)
    u_normalizer.to(device)
    ttrain = np.zeros((epochs,))
    ttest = np.zeros((epochs,))
    model.train()
    for ep in range(epochs):
        t1 = default_timer()
        train_mse = 0.0
        train_l2 = 0.0
        for batch in train_loader:
            batch = batch.to(device)

            optimizer.zero_grad()
            out = model(batch)
            mse = F.mse_loss(out.view(-1, 1), batch.y.view(-1,1))
            # mse.backward()

            l2 = myloss(
                u_normalizer.decode(out.view(batch_size, -1), sample_idx=batch.sample_idx.view(batch_size, -1)),
                u_normalizer.decode(batch.y.view(batch_size, -1), sample_idx=batch.sample_idx.view(batch_size, -1)))
            l2.backward()

            optimizer.step()
            train_mse += mse.item()
            train_l2 += l2.item()

        scheduler.step()
        t2 = default_timer()
        ttrain[ep] = train_l2 / (ntrain * k)

        print(ep, t2 - t1, train_mse / len(train_loader), train_l2 / (ntrain * k))

    runtime[0] =  t2 - t1

    t1 = default_timer()
    model.eval()
    test_l2 = 0.0
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch)
            out = u_normalizer.decode(out.view(batch_size2, -1), sample_idx=batch.sample_idx.view(batch_size2, -1))
            test_l2 += myloss(out, batch.y.view(batch_size2, -1)).item()

    ttest[ep] = test_l2 / ntest
    t2 = default_timer()
    print(ep, t2 - t1, test_l2 / ntest)

    runtime[1] =  t2 - t1

    np.savetxt(path_train_err, ttrain)
    np.savetxt(path_test_err, ttest)
    np.savetxt(path_runtime, runtime)
    torch.save(model, path_model)

