import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from collections import OrderedDict

class nodeselection(nn.Module):
    def __init__(self, topk, memory_node, time_dim):
        super(nodeselection, self).__init__()
        self.K= topk
        # self.memory_node = memory_node
        self.node_embeddings = nn.Parameter(torch.randn(memory_node, time_dim), requires_grad=True)

    def forward(self, node_feature,node_embeddings=None):
        nodevec1 = node_feature
        # nodevec2 = node_feature[1]
        # nodevec3 = torch.cat((nodevec1, nodevec2), -1)
        # nodevec4 = torch.mul(torch.cat((node_embeddings[0], node_embeddings[1]), -1).unsqueeze(-2),
        #                      self.node_embeddings)
        nodevec2 = self.node_embeddings

        # nodevec4 = torch.mul(node_embeddings[1].unsqueeze(-2), filter2)  # [B,N,dim_in]

        supports2 = torch.softmax(torch.matmul(nodevec2, nodevec1.transpose(-2, -1)), dim=-1)

        # 子图和对应节点索引生成部分
        values, indices = supports2.topk(self.K, dim=-1, largest=True, sorted=True)

        # batch_indices =
        # batch_indices = torch.cuda.FloatTensor(batch_indices) if torch.cuda.is_available() else torch.FloatTensor(batch_indices)
        # torch.randn(shape, out=batch_indices)


        batch_indices = torch.arange(nodevec1.size(0),device=nodevec1.device).unsqueeze(-1).unsqueeze(-1).expand(-1, indices.size(1),
                                                                                   indices.size(2))#.cuda()
        # 这是得到的记忆网络的topk索引对应节点提取到的特征
        selected_nodes_features = nodevec1[batch_indices, indices]
        # selected_nodes_features2 = nodevec2[batch_indices, indices]
        # selected_nodes_features = [selected_nodes_features1,selected_nodes_features2]
        return selected_nodes_features, batch_indices, indices


class feature_aggregation(nn.Module):
    def __init__(self,K,N):
        super(feature_aggregation,self).__init__()
        self.K = K
        self.N = N

    #无原始节点版本
    def forward(self,x, adj, batch_indices, indices):
        B,N,D = x.shape
        # selected_nodes = x[batch_indices, indices]

        # x=x.to('cuda:1')
        # adj=adj.to('cuda:1')
        # batch_indices=batch_indices.to('cuda:1')
        # indices=indices.to('cuda:1')
        # x1 = torch.matmul(a, x)
        selected_nodes = x[batch_indices, indices]
        # selected_nodes1 = x[:, indices]
        # selected_nodes12 = selected_nodes1[0]
        # selected_nodes13 =selected_nodes1[1]
        node_new = torch.matmul(adj, selected_nodes).reshape(B, self.K * self.N, D)

        # 重塑indices1以匹配scatter_add_的需要
        indices_expanded = indices.unsqueeze(-1).expand(-1, -1, -1, D).reshape(B, self.K * self.N, D)
        dict1 = torch.zeros(x.shape,device=x.device).clone()#.cuda()
        dict1.scatter_add_(1, indices_expanded, node_new)
        # x1 = dict1

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
        # x1 = dict1
        return x1 #.to('cuda:0')

class DGCN(nn.Module):
    def __init__(self, args,dim_in, dim_out, cheb_k, embed_dim):
        super(DGCN, self).__init__()
        self.cheb_k = cheb_k
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, cheb_k, dim_in, dim_out))
        self.weights = nn.Parameter(torch.FloatTensor(cheb_k,dim_in, dim_out))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))
        self.bias = nn.Parameter(torch.FloatTensor(dim_out))
        self.hyperGNN_dim = 16
        self.middle_dim = 2
        self.embed_dim = embed_dim
        self.fc=nn.Sequential( #疑问，这里为什么要用三层linear来做，为什么激活函数是sigmoid
                OrderedDict([('fc1', nn.Linear(dim_in, self.hyperGNN_dim)),
                             #('sigmoid1', nn.ReLU()),
                             ('sigmoid1', nn.Sigmoid()),
                             ('fc2', nn.Linear(self.hyperGNN_dim, self.middle_dim)),
                             #('sigmoid1', nn.ReLU()),
                             ('sigmoid2', nn.Sigmoid()),
                             ('fc3', nn.Linear(self.middle_dim, self.embed_dim))]))

        self.N= args.memory_node
        self.K = args.topk
        self.use_subgraph =args.use_subgraph
        # self.node_embeddings = nn.Parameter(torch.randn(self.N, self.embed_dim), requires_grad=True)
        self.nodeselection= nodeselection(self.K,self.N,self.embed_dim)
        self.fal = feature_aggregation(self.K,self.N)

    def forward(self, x, node_embeddings):
        #x shaped[B, N, C], node_embeddings shaped [N, D] -> supports shaped [N, N]
        #output shape [B, N, C]
        # B, N, D = x.shape
        # M, _ = self.node_embeddings.shape

        # node_num = node_embeddings[0].shape[1]
        # supports1 = torch.eye(node_num).to(node_embeddings[0].device)



        filter = self.fc(x)
        nodevec = torch.mul(node_embeddings[0], filter) #[B,N,dim_in]
        if self.use_subgraph:
            selected_nodes_features, batch_indices, indices = self.nodeselection(nodevec)

            A = DGCN.get_laplacian(
                F.relu(torch.matmul(selected_nodes_features, selected_nodes_features.transpose(-2, -1))))

            x_g2 = self.fal(x, A, batch_indices, indices)
        else:
            supports2 = DGCN.get_laplacian(F.relu(torch.matmul(nodevec, nodevec.transpose(2, 1))))
            x_g2 = torch.einsum("bnm,bmc->bnc", supports2, x)


        # nodevec2 = self.node_embeddings  # [B,N,dim_in]
        #
        # # cata = self.fc2(nodevec2)
        # # supports2 = torch.matmul(nodevec2, nodevec1.transpose(-2, -1))
        # supports2 = torch.softmax(torch.matmul(nodevec2, nodevec1.transpose(-2, -1)), dim=-1)
        #
        # values1, indices1 = supports2.topk(self.K, dim=-1, largest=True, sorted=True)
        # # 用来定位是那个batch的数据的索引
        # batch_indices = torch.arange(x.size(0)).unsqueeze(-1).unsqueeze(-1).expand(-1, indices1.size(1),
        #                                                                            indices1.size(2)).cuda()
        # # 这是得到的记忆网络的topk索引对应的特征
        # selected_nodes_features = nodevec1[batch_indices, indices1]
        # selected_nodes = x[batch_indices, indices1]

        # graph = DGCN.get_laplacian(F.relu(torch.matmul(nodevec1, nodevec1.transpose(-2, -1))))
        # x_g3 =  torch.matmul(graph, x)
        # print(graph)
        # node_new = torch.matmul(A, selected_nodes).reshape(B, self.K * self.N, D)
        # indices1_expanded = indices1.unsqueeze(-1).expand(-1, -1, -1, D).reshape(B, self.K * self.N, D)
        #
        # # 跟加入节点本身的不一样，这里不包含节点本身，所以这里本身只需要设置为0
        # # 但是如果有节点不在任何子图中，则做除法时分母为0会报错，所以要设置一个足够小的值来代替0
        # # 同时不需要像加入节点版那样让X和新的节点特征相加
        # dict_refined = torch.ones(B, N).cuda()* 10 ** -14
        # # dict_refined = self.dict_refined.expand(B,-1).clone()
        # # dict_refined = torch.ones(B, N).cuda()
        # # 用来测试完整图的情况是否一致
        # # dict_refined = torch.zeros(B, N).cuda()
        #
        # # 获取 indices1 的扁平版本和对应的批次索引
        # # 由于 indices1 有重复的索引，我们需要先转换它们为线性索引
        # flat_indices = indices1.flatten()
        # # 这个batch_indices也是用来定位这个索引属于哪个batch的
        # batch_indices1 = torch.arange(indices1.size(0)).repeat_interleave(indices1.size(1) * indices1.size(2)).cuda()
        #
        # # 构建用于 scatter_add 的源张量，用来对应每个索引代表的次数,弄成这样子是因为下面put_的格式要求，不然直接+1了
        # ones_source = torch.ones_like(flat_indices, dtype=dict_refined.dtype).cuda()
        #
        # # 使用 scatter_add 在正确的位置累加1,弄成对应线性batch的索引版本
        # linear_indices = flat_indices + batch_indices1 * dict_refined.size(1)
        # # 得到的索引dict_refined,accumulate=True会在相同索引的地方进行累加，这样子在每个线性batch中一样的索引值的地方会累加
        # dict_refined.put_(linear_indices, ones_source, accumulate=True)
        #
        # # 做平均
        # # cata = self.fc2(nodevec2)
        # # dict1 =torch.einsum("bmn,bmc->bnc", supports2, cata)
        # # dict1 = x.clone()
        #
        # # 测试挑选后完整图是否一致
        # dict1 = torch.zeros(x.shape).clone().cuda()
        # dict1.scatter_add_(1, indices1_expanded, node_new)
        #
        # x_g2 = dict1 / dict_refined.unsqueeze(-1).expand(B, N, D)

        # x_g1 = torch.einsum("nm,bmc->bnc", supports1, x)
        # x_g2 = torch.einsum("bnm,bmc->bnc", supports2, x)
        #x_g3 = torch.einsum("bnm,bmc->bnc", supports3, x)

        x_g = torch.stack([x,x_g2],dim=1)

        # supports = torch.stack(support_set, dim=0)   #[2,nodes,nodes]  也就是这里把单位矩阵和自适应矩阵拼在一起了
        # x_g = torch.einsum("knm,bmc->bknc", supports, x)

        # weights = torch.einsum('bnd,dkio->bnkio', nodevec, self.weights_pool)

        weights = torch.einsum('nd,dkio->nkio', node_embeddings[1], self.weights_pool)    #[B,N,embed_dim]*[embed_dim,chen_k,dim_in,dim_out] =[B,N,cheb_k,dim_in,dim_out]
                                                                                  #[N, cheb_k, dim_in, dim_out]=[nodes,cheb_k,hidden_size,output_dim]
        bias = torch.matmul(node_embeddings[1], self.bias_pool) #N, dim_out                 #[che_k,nodes,nodes]* [batch,nodes,dim_in]=[B, cheb_k, N, dim_in]

        x_g = x_g.permute(0, 2, 1, 3)  # B, N, cheb_k, dim_in
        # x_gconv = torch.einsum('bnki,bnkio->bno', x_g, weights) + bias  #b, N, dim_out
        x_gconv = torch.einsum('bnki,nkio->bno', x_g, weights) + bias  #b, N, dim_out
        # x_gconv = torch.einsum('bnki,kio->bno', x_g, self.weights) + self.bias    #[B,N,cheb_k,dim_in] *[N,cheb_k,dim_in,dim_out] =[B,N,dim_out]

        return x_gconv



    # def forward(self, x, node_embeddings):
    #     #x shaped[B, N, C], node_embeddings shaped [N, D] -> supports shaped [N, N]
    #     #output shape [B, N, C]
    #     B, N, D = x.shape
    #     M, _ = self.node_embeddings.shape
    #
    #     # node_num = node_embeddings[0].shape[1]
    #     # supports1 = torch.eye(node_num).to(node_embeddings[0].device)
    #
    #     # filter = self.fc(x)
    #     # nodevec = torch.tanh(torch.mul(node_embeddings[0], filter))  #[B,N,dim_in]
    #     # supports2 = DGCN.get_laplacian(F.relu(torch.matmul(nodevec, nodevec.transpose(2, 1))), supports1)
    #
    #     filter1 = self.fc(x)
    #     nodevec1 = torch.mul(node_embeddings[0], filter1)
    #
    #     nodevec2 = self.node_embeddings  # [B,N,dim_in]
    #
    #     # cata = self.fc2(nodevec2)
    #     # supports2 = torch.matmul(nodevec2, nodevec1.transpose(-2, -1))
    #     supports2 = torch.softmax(torch.matmul(nodevec2, nodevec1.transpose(-2, -1)), dim=-1)
    #
    #     values1, indices1 = supports2.topk(self.K, dim=-1, largest=True, sorted=True)
    #     batch_indices = torch.arange(x.size(0)).unsqueeze(-1).unsqueeze(-1).expand(-1, indices1.size(1),
    #                                                                                indices1.size(2)).cuda()
    #     # 这是得到的记忆网络的topk索引对应的特征
    #     selected_nodes_features = nodevec1[batch_indices, indices1]
    #     selected_nodes = x[batch_indices, indices1]
    #     # graph = F.relu(torch.matmul(selected_nodes_features, selected_nodes_features.transpose(-2, -1)))
    #     # print(graph)
    #
    #     A = self.get_laplacian(F.relu(torch.matmul(selected_nodes_features, selected_nodes_features.transpose(-2, -1))))
    #     node_new = torch.matmul(A, selected_nodes).reshape(B, self.K * self.N, D)
    #
    #     # 重塑indices1以匹配scatter_add_的需要
    #     indices1_expanded = indices1.unsqueeze(-1).expand(-1, -1, -1, D).reshape(B, self.K * self.N, D)
    #     dict1 = x.clone()
    #     dict1.scatter_add_(1, indices1_expanded, node_new)
    #
    #     dict_refined = torch.ones(B, N, requires_grad=True).cuda()
    #     # 获取 indices1 的扁平版本和对应的批次索引
    #     flat_indices = indices1.flatten()
    #     batch_indices = torch.arange(indices1.size(0)).repeat_interleave(indices1.size(1) * indices1.size(2)).cuda()
    #
    #     # 构建用于 scatter_add 的源张量，因为我们需要在每个索引处累加 1
    #     ones_source = torch.ones_like(flat_indices, dtype=dict_refined.dtype).cuda()
    #
    #     # 使用 scatter_add 在正确的位置累加 1
    #     # 由于 indices1 有重复的索引，我们需要先转换它们为线性索引
    #     linear_indices = flat_indices + batch_indices * dict_refined.size(1)
    #     # 得到的索引dict_refined
    #     dict_refined.put_(linear_indices, ones_source, accumulate=True)
    #
    #     x_g2 = dict1 / dict_refined.unsqueeze(-1).expand(B, N, D)
    #     # x_g1 = torch.einsum("nm,bmc->bnc", supports1, x)
    #     # x_g2 = torch.einsum("bnm,bmc->bnc", supports2, x)
    #     #x_g3 = torch.einsum("bnm,bmc->bnc", supports3, x)
    #     x_g = torch.stack([x,x_g2],dim=1)
    #
    #     # supports = torch.stack(support_set, dim=0)   #[2,nodes,nodes]  也就是这里把单位矩阵和自适应矩阵拼在一起了
    #     # x_g = torch.einsum("knm,bmc->bknc", supports, x)
    #
    #     # weights = torch.einsum('bnd,dkio->bnkio', nodevec, self.weights_pool)
    #
    #     weights = torch.einsum('nd,dkio->nkio', node_embeddings[1], self.weights_pool)    #[B,N,embed_dim]*[embed_dim,chen_k,dim_in,dim_out] =[B,N,cheb_k,dim_in,dim_out]
    #                                                                               #[N, cheb_k, dim_in, dim_out]=[nodes,cheb_k,hidden_size,output_dim]
    #     bias = torch.matmul(node_embeddings[1], self.bias_pool) #N, dim_out                 #[che_k,nodes,nodes]* [batch,nodes,dim_in]=[B, cheb_k, N, dim_in]
    #
    #     x_g = x_g.permute(0, 2, 1, 3)  # B, N, cheb_k, dim_in
    #     # x_gconv = torch.einsum('bnki,bnkio->bno', x_g, weights) + bias  #b, N, dim_out
    #     x_gconv = torch.einsum('bnki,nkio->bno', x_g, weights) + bias  #b, N, dim_out
    #     # x_gconv = torch.einsum('bnki,kio->bno', x_g, self.weights) + self.bias    #[B,N,cheb_k,dim_in] *[N,cheb_k,dim_in,dim_out] =[B,N,dim_out]
    #
    #     return x_gconv

    @staticmethod
    def get_laplacian(graph, normalize=True):
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
