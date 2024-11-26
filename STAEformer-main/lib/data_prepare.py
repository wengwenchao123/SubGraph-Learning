import torch
import numpy as np
import os
from .utils import print_log, StandardScaler, vrange
import pandas as pd
# ! X shape: (B, T, N, C)


# def get_dataloaders_from_index_data(
#     data_dir, tod=False, dow=False, dom=False, batch_size=64, log=None
# ):
#     data = np.load(os.path.join(data_dir, "data.npz"))["data"].astype(np.float32)
#
#     features = [0]
#     if tod:
#         features.append(1)
#     if dow:
#         features.append(2)
#     # if dom:
#     #     features.append(3)
#     data = data[..., features]
#
#     index = np.load(os.path.join(data_dir, "index.npz"))
#
#     train_index = index["train"]  # (num_samples, 3)
#     val_index = index["val"]
#     test_index = index["test"]
#
#     x_train_index = vrange(train_index[:, 0], train_index[:, 1])
#     y_train_index = vrange(train_index[:, 1], train_index[:, 2])
#     x_val_index = vrange(val_index[:, 0], val_index[:, 1])
#     y_val_index = vrange(val_index[:, 1], val_index[:, 2])
#     x_test_index = vrange(test_index[:, 0], test_index[:, 1])
#     y_test_index = vrange(test_index[:, 1], test_index[:, 2])
#
#     x_train = data[x_train_index]
#     y_train = data[y_train_index][..., :1]
#     x_val = data[x_val_index]
#     y_val = data[y_val_index][..., :1]
#     x_test = data[x_test_index]
#     y_test = data[y_test_index][..., :1]
#
#     scaler = StandardScaler(mean=x_train[..., 0].mean(), std=x_train[..., 0].std())
#
#     x_train[..., 0] = scaler.transform(x_train[..., 0])
#     x_val[..., 0] = scaler.transform(x_val[..., 0])
#     x_test[..., 0] = scaler.transform(x_test[..., 0])
#
#     print_log(f"Trainset:\tx-{x_train.shape}\ty-{y_train.shape}", log=log)
#     print_log(f"Valset:  \tx-{x_val.shape}  \ty-{y_val.shape}", log=log)
#     print_log(f"Testset:\tx-{x_test.shape}\ty-{y_test.shape}", log=log)
#
#     trainset = torch.utils.data.TensorDataset(
#         torch.FloatTensor(x_train), torch.FloatTensor(y_train)
#     )
#     valset = torch.utils.data.TensorDataset(
#         torch.FloatTensor(x_val), torch.FloatTensor(y_val)
#     )
#     testset = torch.utils.data.TensorDataset(
#         torch.FloatTensor(x_test), torch.FloatTensor(y_test)
#     )
#
#     trainset_loader = torch.utils.data.DataLoader(
#         trainset, batch_size=batch_size, shuffle=True
#     )
#     valset_loader = torch.utils.data.DataLoader(
#         valset, batch_size=batch_size, shuffle=False
#     )
#     testset_loader = torch.utils.data.DataLoader(
#         testset, batch_size=batch_size, shuffle=False
#     )
#
#     return trainset_loader, valset_loader, testset_loader, scaler

def get_dataloaders_from_index_data(
    data_dir,
    tod=False,
    dow=False,
    dom=False,
    batch_size=64,
    log=None,
    history_seq_len=12,
    future_seq_len=12,
    train_ratio=0.6,
    valid_ratio=0.2,
    shift=False,
):
    data_name = "data_shift" if shift else "data"
    if os.path.isfile(os.path.join(data_dir, f"{data_name}.npz")) == True:
        data = np.load(os.path.join(data_dir, f"{data_name}.npz"))["data"].astype(
            np.float32
        )
    else:
        df = (
            pd.read_hdf(os.path.join(data_dir, f"{data_name}.h5")).fillna(0).astype(int)
        )
        num_samples, num_nodes = df.shape
        data = np.expand_dims(df.values, axis=-1)

        feature_list = [data]
        time_ind = (
            df.index.values - df.index.values.astype("datetime64[D]")
        ) / np.timedelta64(1, "D")
        time_of_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
        feature_list.append(time_of_day)
        dow_tiled = np.tile(df.index.dayofweek, [1, num_nodes, 1]).transpose((2, 1, 0))
        day_of_week = dow_tiled
        feature_list.append(day_of_week)
        data = np.concatenate(feature_list, axis=-1)

    l, n, f = data.shape
    num_samples = l - (history_seq_len + future_seq_len) + 1
    train_num_short = round(num_samples * train_ratio)
    valid_num_short = round(num_samples * valid_ratio)
    test_num_short = num_samples - train_num_short - valid_num_short

    index_list = np.array(
        [
            (t - history_seq_len, t, t + future_seq_len)
            for t in range(history_seq_len, num_samples + history_seq_len)
        ]
    )

    train_index = index_list[:train_num_short]
    val_index = index_list[train_num_short : train_num_short + valid_num_short]
    test_index = index_list[
        train_num_short
        + valid_num_short : train_num_short
        + valid_num_short
        + test_num_short
    ]

    x_train_index = vrange(train_index[:, 0], train_index[:, 1])
    y_train_index = vrange(train_index[:, 1], train_index[:, 2])
    x_val_index = vrange(val_index[:, 0], val_index[:, 1])
    y_val_index = vrange(val_index[:, 1], val_index[:, 2])
    x_test_index = vrange(test_index[:, 0], test_index[:, 1])
    y_test_index = vrange(test_index[:, 1], test_index[:, 2])

    x_train = data[x_train_index]
    y_train = data[y_train_index][..., :1]
    x_val = data[x_val_index]
    y_val = data[y_val_index][..., :1]
    x_test = data[x_test_index]
    y_test = data[y_test_index][..., :1]

    scaler = StandardScaler(mean=x_train[..., 0].mean(), std=x_train[..., 0].std())

    x_train[..., 0] = scaler.transform(x_train[..., 0])
    x_val[..., 0] = scaler.transform(x_val[..., 0])
    x_test[..., 0] = scaler.transform(x_test[..., 0])

    print_log(f"Trainset:\tx-{x_train.shape}\ty-{y_train.shape}", log=log)
    print_log(f"Valset:  \tx-{x_val.shape}  \ty-{y_val.shape}", log=log)
    print_log(f"Testset:\tx-{x_test.shape}\ty-{y_test.shape}", log=log)

    trainset = torch.utils.data.TensorDataset(
        torch.FloatTensor(x_train), torch.FloatTensor(y_train)
    )
    valset = torch.utils.data.TensorDataset(
        torch.FloatTensor(x_val), torch.FloatTensor(y_val)
    )
    testset = torch.utils.data.TensorDataset(
        torch.FloatTensor(x_test), torch.FloatTensor(y_test)
    )

    trainset_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True
    )
    valset_loader = torch.utils.data.DataLoader(
        valset, batch_size=batch_size, shuffle=False
    )
    testset_loader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False
    )

    return trainset_loader, valset_loader, testset_loader, scaler