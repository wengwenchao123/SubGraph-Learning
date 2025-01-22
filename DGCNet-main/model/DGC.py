import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from collections import OrderedDict
import math
import torch_scatter

class nodeselection(nn.Module):
    def __init__(self, topk, memory_node, time_dim):
        super(nodeselection, self).__init__()
        self.K= topk
        self.node_embeddings = nn.Parameter(torch.randn(memory_node, time_dim), requires_grad=True)

    def forward(self, node_feature,node_embeddings=None):
        nodevec1 = node_feature

        nodevec2 = self.node_embeddings

        supports2 = torch.softmax(torch.matmul(nodevec2, nodevec1.transpose(-2, -1)), dim=-1)

        # 子图和对应节点索引生成部分
        values, indices = supports2.topk(self.K, dim=-1, largest=True, sorted=True)

        batch_indices = torch.arange(nodevec1.size(0),device=nodevec1.device).unsqueeze(-1).unsqueeze(-1).expand(-1, indices.size(1),
                                                                                   indices.size(2))#.cuda()
        # 这是得到的记忆网络的topk索引对应节点提取到的特征
        selected_nodes_features = nodevec1[batch_indices, indices]
        return selected_nodes_features, batch_indices, indices


class feature_aggregation(nn.Module):
    def __init__(self,K,N):
        super(feature_aggregation,self).__init__()
        self.K = K
        self.N = N

    #无原始节点版本
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

class DGC(nn.Module):
    def __init__(self, args,dim_in, dim_out, cheb_k, embed_dim, time_dim):
        super(DGC, self).__init__()
        self.cheb_k = cheb_k
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, cheb_k, dim_in, dim_out))
        self.weights = nn.Parameter(torch.FloatTensor(cheb_k,dim_in, dim_out))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))
        self.bias = nn.Parameter(torch.FloatTensor(dim_out))

        self.hyperGNN_dim = 16
        self.middle_dim = 2
        self.embed_dim = embed_dim
        self.time_dim = time_dim
        self.fc1=nn.Sequential( #疑问，这里为什么要用三层linear来做，为什么激活函数是sigmoid
                OrderedDict([('fc1', nn.Linear(dim_in, self.hyperGNN_dim)),
                             #('sigmoid1', nn.ReLU()),
                             ('sigmoid1', nn.Sigmoid()),
                             ('fc2', nn.Linear(self.hyperGNN_dim, self.middle_dim)),
                             #('sigmoid1', nn.ReLU()),
                             ('sigmoid2', nn.Sigmoid()),
                             ('fc3', nn.Linear(self.middle_dim, self.time_dim))]))
        self.fc2=nn.Sequential( #疑问，这里为什么要用三层linear来做，为什么激活函数是sigmoid
                OrderedDict([('fc1', nn.Linear(self.time_dim, self.middle_dim)),
                             #('sigmoid1', nn.ReLU()),
                             ('sigmoid1', nn.Sigmoid()),
                             ('fc2', nn.Linear(self.middle_dim, self.hyperGNN_dim)),
                             #('sigmoid1', nn.ReLU()),
                             ('sigmoid2', nn.Sigmoid()),
                             ('fc3', nn.Linear(self.hyperGNN_dim, dim_in))]))

        self.N= args.memory_node
        self.K = args.topk
        self.use_subgraph = args.use_subgraph
        self.node_embeddings = nn.Parameter(torch.randn(self.N, self.time_dim), requires_grad=True)
        self.nodeselection= nodeselection(self.K,self.N,self.time_dim)
        self.fal = feature_aggregation(self.K,self.N)

    def forward(self, x, node_embeddings):
        #x shaped[B, N, C], node_embeddings shaped [N, D] -> supports shaped [N, N]
        #output shape [B, N, C]
        filter = self.fc1(x)
        nodevec = torch.mul(node_embeddings[0].unsqueeze(-2),filter)
        # nodevec = self.clustering1(nodevec)
        # x1 = self.clustering2(x)
        if self.use_subgraph:
            selected_nodes_features, batch_indices, indices = self.nodeselection(nodevec)

            A = DGC.get_laplacian(
                F.relu(torch.matmul(selected_nodes_features, selected_nodes_features.transpose(-2, -1))))

            x_g2 = self.fal(x, A, batch_indices, indices)

        else:
            supports2 = DGC.get_laplacian(F.relu(torch.matmul(nodevec, nodevec.transpose(2, 1))))
            x_g2 = torch.einsum("bnm,bmc->bnc", supports2, x)

        x_g = torch.stack([x,x_g2],dim=1)


        weights = torch.einsum('nd,dkio->nkio', node_embeddings[1], self.weights_pool)    #[B,N,embed_dim]*[embed_dim,chen_k,dim_in,dim_out] =[B,N,cheb_k,dim_in,dim_out]
                                                                                  #[N, cheb_k, dim_in, dim_out]=[nodes,cheb_k,hidden_size,output_dim]
        bias = torch.matmul(node_embeddings[1], self.bias_pool) #N, dim_out                 #[che_k,nodes,nodes]* [batch,nodes,dim_in]=[B, cheb_k, N, dim_in]


        x_g = x_g.permute(0, 2, 1, 3)  # B, N, cheb_k, dim_in
        # x_gconv = torch.einsum('bnki,bnkio->bno', x_g, weights) + bias  #b, N, dim_out
        x_gconv = torch.einsum('bnki,nkio->bno', x_g, weights) + bias  #b, N, dim_out
        # x_gconv = torch.einsum('bnki,kio->bno', x_g, self.weights) + self.bias    #[B,N,cheb_k,dim_in] *[N,cheb_k,dim_in,dim_out] =[B,N,dim_out]

        # x_gconv =self.fc3(x)

        return x_gconv

    @staticmethod
    def get_laplacian(graph,normalize=True):
        """
        return the laplacian of the graph.

        :param graph: the graph structure without self loop, [N, N].
        :param normalize: whether to used the normalized laplacian.
        :return: graph laplacian.
        """
        if normalize:
            D = torch.diag_embed(torch.sum(graph, dim=-1) ** (-1 / 2))
            #L = I - torch.matmul(torch.matmul(D, graph), D)
            L = torch.matmul(torch.matmul(D, graph), D)
        else:
            graph = graph + I
            D = torch.diag_embed(torch.sum(graph, dim=-1) ** (-1 / 2))
            L = torch.matmul(torch.matmul(D, graph), D)
        return L