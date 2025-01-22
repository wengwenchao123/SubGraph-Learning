import os
import argparse
import numpy as np

import sys
sys.path.append(os.path.abspath(__file__ + '/../../..'))

import torch
torch.set_num_threads(3)

from src.models.dgcrn import DGCRN
from src.engines.dgcrn_engine import DGCRN_Engine
from src.utils.args import get_public_config
from src.utils.dataloader import load_dataset, load_adj_from_numpy, get_dataset_info
from src.utils.graph_algo import normalize_adj_mx
from src.utils.metrics import masked_mae
from src.utils.logging import get_logger

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


# --dataset SD --device cuda:0 --years 2019 --model_name dgcrn --seed 2023 --bs 64 --tpd 96
# --dataset GBA --device cuda:3 --years 2019 --model_name dgcrn --seed 2023 --bs 16 --tpd 96 --topk 250 --memory_node 10
# --dataset PEMS08 --device cuda:0 --model_name dgcrn --seed 2023 --bs 64 --tpd 288
def get_config():  #一些基础的参数的设置写在utils/args.py里面
    parser = get_public_config()
    parser.add_argument('--gcn_depth', type=int, default=2)
    parser.add_argument('--rnn_size', type=int, default=64)
    parser.add_argument('--hyperGNN_dim', type=int, default=16)
    parser.add_argument('--node_dim', type=int, default=40)
    parser.add_argument('--tanhalpha', type=int, default=3)
    parser.add_argument('--cl_decay_step', type=int, default=2500)  #教师学习的步数
    parser.add_argument('--step_size', type=int, default=1000)   #这是多少item后预测步数+1
    parser.add_argument('--tpd', type=int, default=288) #这个应该是指时间间隔，用于这里的DGCRN来换算decoder的时间的
    parser.add_argument('--use_subgraph', type=eval, default=True)
    parser.add_argument('--topk', type=int, default=60)
    parser.add_argument('--memory_node', type=int, default=4)
    parser.add_argument('--lrate', type=float, default=1e-3)
    parser.add_argument('--wdecay', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--clip_grad_value', type=float, default=5)
    args = parser.parse_args()

    log_dir = './experiments/{}/{}/'.format(args.model_name, args.dataset)
    logger = get_logger(log_dir, __name__, 'record_s{}.log'.format(args.seed))
    logger.info(args)
    
    return args, log_dir, logger


def main():
    args, log_dir, logger = get_config()
    # set_seed(args.seed)
    device = torch.device(args.device)
    
    data_path, adj_path, node_num = get_dataset_info(args.dataset)
    logger.info('Adj path: ' + adj_path)

    adj_mx = load_adj_from_numpy(adj_path)
    adj_mx = normalize_adj_mx(adj_mx, 'doubletransition')
    supports = [torch.tensor(i).to(device) for i in adj_mx]
    
    dataloader, scaler = load_dataset(data_path, args, logger)

    model = DGCRN(node_num=node_num,
                  input_dim=args.input_dim,
                  output_dim=args.output_dim,
                  device=device,
                  predefined_adj=supports,
                  gcn_depth=args.gcn_depth,
                  rnn_size=args.rnn_size,
                  hyperGNN_dim=args.hyperGNN_dim,
                  node_dim=args.node_dim,
                  middle_dim=2,
                  list_weight=[0.05, 0.95, 0.95],
                  tpd=args.tpd,
                  tanhalpha=args.tanhalpha,
                  cl_decay_step=args.cl_decay_step,
                  dropout=args.dropout,
                  use_subgraph = args.use_subgraph,
                  topk= args.topk,
                  memory_node =args.memory_node
                  )
    
    loss_fn = masked_mae
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lrate, weight_decay=args.wdecay)
    # scheduler = None
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 120], gamma=0.5)

    engine = DGCRN_Engine(device=device,
                          model=model,
                          dataloader=dataloader,
                          scaler=scaler,
                          sampler=None,
                          loss_fn=loss_fn,
                          lrate=args.lrate,
                          optimizer=optimizer,
                          scheduler=scheduler,
                          clip_grad_value=args.clip_grad_value,
                          max_epochs=args.max_epochs,
                          patience=args.patience,
                          log_dir=log_dir,
                          logger=logger,
                          seed=args.seed,
                          step_size=args.step_size,
                          horizon=args.horizon
                          )

    if args.mode == 'train':
        engine.train()
    else:
        engine.evaluate(args.mode)


if __name__ == "__main__":
    main()