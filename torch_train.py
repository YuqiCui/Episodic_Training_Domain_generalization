from lib.data_loader import load_vlcs
from lib.torch_lib import torch_vlcs_model
from lib.utils import *
import torch
import torch.utils.data as Data
import torch.nn.functional as F
import os
from sklearn.metrics import accuracy_score
np.random.seed(1447)
data_root = 'data/'
datas,  domain_name = load_vlcs(data_root)
fea = ['fc1', 'fc2']
cls = ['fc3']
ckpt_root = 'torch_ckpt/'
if not os.path.exists(ckpt_root):
    os.mkdir(ckpt_root)
lr = 1e-3
momentum = 0.9
weight_decay = 5e-5
batch_size = 32
step_warm = 1
step_tol_train = 500
w_f = 0.1
w_c = 0.1
w_r = 0.1
patience = 80
init_ckpt = ckpt_root+'init_model.pkl'


def train_data_set(x_train, y_train, batch_size):
    x_t_train = torch.from_numpy(x_train)
    y_t_train = torch.from_numpy(y_train)
    dataset = Data.TensorDataset(x_t_train, y_t_train)
    return Data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


def get_next_batch(d_iter):
    try:
        batch_x, batch_y = next(d_iter)
    except:
        d_iter = iter(d_iter)
        batch_x, batch_y = next(d_iter)
    return batch_x, batch_y, d_iter


def get_model_weights(model, prefix):
    """

    :param model: a torch.nn.Module or a string to the weight pickle location
    :param prefix:
    :return:
    """
    if isinstance(model, str):
        weight_dict = torch.load(model)
    else:
        weight_dict = model.state_dict()
    lists = []
    weights = {}
    for n in prefix:
        lists.extend([k for k in list(weight_dict.keys()) if n in k])
    for n in lists:
        weights[n] = weight_dict[n]
    return weights


def set_require_grad(model, attr, requires_grad=True):
    lists = getattr(model, attr)
    for l in lists:
        l.weight.requires_grad = requires_grad
        l.bias.requires_grad = requires_grad


def get_params(model, group):
    for name, p in model.named_parameters():
        n = name.split('.')[0]
        if n in group:
            yield p


def eval(model, x_test, y_test):
    model.eval()
    loader = train_data_set(x_test, y_test, batch_size=batch_size)
    preds = []
    labels = []
    for step, (batch_x, batch_y) in enumerate(loader):
        batch_x = batch_x.cuda()
        out = model(batch_x)
        pred = torch.max(F.softmax(out, dim=1), 1)[1]
        preds.append(pred.cpu().numpy().squeeze())
        labels.append(batch_y.numpy())
    preds = np.concatenate(preds)
    labels = np.concatenate(labels)
    return accuracy_score(preds, labels)


for i in range(4):
    best_acc = -1
    count = 0
    leave_out_domain = i
    x_test, y_test = datas[i][1], datas[i][3]
    train_id = np.delete(np.arange(4), i)
    print('[Start Training] leave out domain: {}'.format(domain_name[leave_out_domain]))
    domain_iters_init = [train_data_set(datas[jj][0], datas[jj][2], batch_size=batch_size) for jj in train_id]

    # --------- warm up model --------
    x_train, y_train = concat_train(datas, leave_out_domain, domain_name)
    model = torch_vlcs_model(4096, 5).cuda()
    d_models = [torch_vlcs_model(4096, 5).cuda() for _ in range(len(train_id))]
    d_optims = [torch.optim.SGD(d_m.parameters(), lr=lr,
                                momentum=momentum, weight_decay=weight_decay) for d_m in d_models]
    soft_loss = torch.nn.CrossEntropyLoss()

    torch.save(model.state_dict(), init_ckpt)

    for it, model_warm in enumerate(d_models):
        optim = d_optims[it]
        for epoch in range(step_warm):
            model_warm.train()
            for step, (batch_x, batch_y) in enumerate(domain_iters_init[it]):
                batch_x = batch_x.cuda()
                batch_y = batch_y.long().cuda()

                out = model_warm(batch_x)
                loss = soft_loss(out, batch_y)
                optim.zero_grad()
                loss.backward()
                optim.step()

            acc = eval(model_warm, x_test, y_test)
            print('[STEP {}][Warm up Acc] acc: {:.4}'.format(epoch, acc))

    # ------- train model --------
    iter_per_domain = [iter(n) for n in domain_iters_init]
    optim_F = torch.optim.SGD(get_params(model, fea), lr=lr, momentum=momentum, weight_decay=weight_decay)
    optim_C = torch.optim.SGD(get_params(model, cls), lr=lr, momentum=momentum, weight_decay=weight_decay)

    for _ in range(step_tol_train):
        for model_j in range(len(train_id)):
            try:
                batch_x, batch_y = next(iter_per_domain[model_j])
            except StopIteration:
                iter_per_domain[model_j] = iter(domain_iters_init[model_j])
                batch_x, batch_y = next(iter_per_domain[model_j])
            batch_x = batch_x.cuda()
            batch_y = batch_y.long().cuda()

            out = d_models[model_j](batch_x)
            loss = soft_loss(out, batch_y)
            d_optims[model_j].zero_grad()
            loss.backward()
            d_optims[model_j].step()

        for data_i in range(len(train_id)):
            for model_j in range(len(train_id)):
                if data_i == model_j:
                    continue
                try:
                    batch_x, batch_y = next(iter_per_domain[data_i])
                except StopIteration:
                    iter_per_domain[data_i] = iter(domain_iters_init[data_i])
                    batch_x, batch_y = next(iter_per_domain[data_i])

                model.train()
                batch_x = batch_x.cuda()
                batch_y = batch_y.long().cuda()

                out_agg = model(batch_x)
                loss_agg = soft_loss(out_agg, batch_y)

                out_f = model(batch_x, weights={**get_model_weights(model, fea),
                                                **get_model_weights(d_models[model_j], cls)})
                loss_f = soft_loss(out_f, batch_y)

                out_r = model(batch_x, weights={**get_model_weights(model, fea),
                                                **get_model_weights(init_ckpt, cls)})
                loss_r = soft_loss(out_r, batch_y)
                fea_weights = get_model_weights(model, fea)
                loss = loss_agg + w_f * loss_f + w_r * loss_r
                optim_F.zero_grad()
                loss.backward()
                optim_F.step()

                out_agg = model(batch_x)
                loss_agg = soft_loss(out_agg, batch_y)
                out_c = model(batch_x, weights={**get_model_weights(model, cls),
                                                **get_model_weights(d_models[model_j], fea)})
                loss_c = soft_loss(out_c, batch_y)
                loss_ = loss_agg + w_c * loss_c
                optim_C.zero_grad()
                loss_.backward()
                optim_C.step()

        acc = eval(model, x_test, y_test)
        if acc > best_acc:
            best_acc = acc
        else:
            count += 1
            if count >= patience:
                print('[{}] best acc: {:.4}'.format(domain_name[leave_out_domain], acc))
                break

