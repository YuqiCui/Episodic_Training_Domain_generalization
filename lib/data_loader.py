import numpy as np
import scipy.io as sp
import os
from sklearn.model_selection import train_test_split
name = [
    'VOC2007.mat',
    'LabelMe.mat',
    'Caltech101.mat',
    'SUN09.mat'
]


def load_vlcs(root='../data/'):
    datas = []
    for n in name:
        f = sp.loadmat(os.path.join(root, n))
        data = f['data']
        x = data[:, :4096]
        y = data[:, -1]
        y = y.astype(int)
        y = y - 1
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
        datas.append((x_train, x_test, y_train, y_test))
    return datas, name




if __name__ == '__main__':
    load_vlcs()