import torch
import torch.nn.functional as F
import torch.nn as nn


class feature_aggregation(nn.Module):
    def __init__(self,K,N):
        super(feature_aggregation,self).__init__()
        self.K = K
        self.N = N

    #无原始节点版本
    def forward(self,x, adj, batch_indices, indices):
        B,N,D = x.shape
        # selected_nodes = x[batch_indices, indices]

        # x1 = torch.matmul(a, x)
        selected_nodes = x[batch_indices, indices]
        node_new = torch.matmul(adj, selected_nodes).reshape(B, self.K * self.N, D)

        # 重塑indices1以匹配scatter_add_的需要
        indices_expanded = indices.unsqueeze(-1).expand(-1, -1, -1, D).reshape(B, self.K * self.N, D)
        dict1 = torch.zeros(x.shape,device=x.device).clone()#.cuda()
        dict1.scatter_add_(1, indices_expanded, node_new)

        dict_refined = torch.ones((B, N),device=x.device) * 10 ** -14
        # dict_refined = torch.zeros(B, N).cuda()
        # dict_refined = torch.ones(B, N, requires_grad=True).cuda()
        # 获取 indices1 的扁平版本和对应的批次索引
        flat_indices = indices.flatten()
        batch_indices1 = torch.arange(indices.size(0),device=x.device).repeat_interleave(indices.size(1) * indices.size(2))#.cuda()

        # 构建用于 scatter_add 的源张量，因为我们需要在每个索引处累加 1
        ones_source = torch.ones_like(flat_indices,device=x.device, dtype=dict_refined.dtype)#.cuda()

        # 使用 scatter_add 在正确的位置累加 1
        # 由于 indices1 有重复的索引，我们需要先转换它们为线性索引
        linear_indices = flat_indices + batch_indices1 * dict_refined.size(1)
        # 得到的索引dict_refined
        dict_refined.put_(linear_indices, ones_source, accumulate=True)

        x1 = dict1 / dict_refined.unsqueeze(-1).expand(B, N, D)
        return x1