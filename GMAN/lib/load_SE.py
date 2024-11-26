import os
import numpy as np
import pandas as pd

def load_st_SE(dataset):
    #output B, N, D
    if dataset == 'PEMSD3':
        data_path = os.path.join('./data/PEMS03/SE(PEMSD3).txt')
        # data = np.load(data_path)['data'][:, :, 0]  #onley the first dimension, traffic flow data
    elif dataset == 'PEMSD4':
        data_path = os.path.join('./data/PEMS04/SE(PEMSD4).txt')
        # data = np.load(data_path)['data'][:, :, 0]  #onley the first dimension, traffic flow data
    elif dataset == 'PEMSD7':
        data_path = os.path.join('./data/PEMS07/SE(PEMSD7).txt')
        # data = np.load(data_path)['data'][:, :, 0]  #onley the first dimension, traffic flow data
    elif dataset == 'PEMSD8':
        data_path = os.path.join('./data/PEMS08/SE(PEMSD8).txt')
        # data = np.load(data_path)['data'][:, :, 0]  #onley the first dimension, traffic flow data
    else:
        raise ValueError
    f = open(data_path, mode = 'r')
    lines = f.readlines()
    temp = lines[0].split(' ')
    N, dims = int(temp[0]), int(temp[1])
    SE = np.zeros(shape = (N, dims), dtype = np.float32)
    for line in lines[1 :]:
        temp = line.split(' ')
        index = int(temp[0])
        SE[index] = temp[1 :]
    print('Load %s Dataset shaped: ' % dataset, SE.shape)
    return SE

#
# data_path = os.path.join('../data/PeMS07/PEMS07.npz')
# data = np.load(data_path)['data'][:, :, 0]  # onley the first dimension, traffic flow data
