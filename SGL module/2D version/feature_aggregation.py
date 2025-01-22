import torch
import torch.nn.functional as F
import torch.nn as nn


class feature_aggregation(nn.Module):
    def __init__(self,K,N):
        super(feature_aggregation,self).__init__()
        self.K = K
        self.N = N

    def forward(self,x, adj, batch_indices, indices):
        B,N,D = x.shape

        selected_nodes = x[batch_indices, indices]
        node_new = torch.matmul(adj, selected_nodes).reshape(B, self.K * self.N, D)

        indices_expanded = indices.unsqueeze(-1).expand(-1, -1, -1, D).reshape(B, self.K * self.N, D)
        dict1 = torch.zeros(x.shape,device=x.device).clone()#.cuda()
        dict1.scatter_add_(1, indices_expanded, node_new)

        dict_refined = torch.ones((B, N),device=x.device) * 10 ** -14

        flat_indices = indices.flatten()
        batch_indices1 = torch.arange(indices.size(0),device=x.device).repeat_interleave(indices.size(1) * indices.size(2))#.cuda()

        ones_source = torch.ones_like(flat_indices,device=x.device, dtype=dict_refined.dtype)#.cuda()

        linear_indices = flat_indices + batch_indices1 * dict_refined.size(1)

        dict_refined.put_(linear_indices, ones_source, accumulate=True)

        x1 = dict1 / dict_refined.unsqueeze(-1).expand(B, N, D)
        return x1