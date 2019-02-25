import numpy as np


def concat_train(datas, leave_out_domain, domain_name):
    x_train, y_train = [], []
    for i in range(len(domain_name)):
        if i == leave_out_domain:
            continue
        x_train.append(datas[i][0])
        y_train.append(datas[i][2])
    return np.concatenate(x_train, axis=0), np.concatenate(y_train)