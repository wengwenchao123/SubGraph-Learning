import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.autograd import Variable
from collections import OrderedDict
from src.base.model import BaseModel

class nodeselection(nn.Module):
    def __init__(self, topk, memory_node, time_dim):
        super(nodeselection, self).__init__()
        self.K= topk
        # self.memory_node = memory_node
        self.node_embeddings = nn.Parameter(torch.randn(memory_node, 2 * time_dim), requires_grad=True)

    def forward(self, node_feature):
        nodevec1 = node_feature[0]
        nodevec2 = node_feature[1]
        nodevec3 = torch.cat((nodevec1, nodevec2), -1)
        # nodevec4 = torch.mul(torch.cat((node_embeddings[0], node_embeddings[1]), -1).unsqueeze(-2),
        #                      self.node_embeddings)
        nodevec4 = self.node_embeddings

        # nodevec4 = torch.mul(node_embeddings[1].unsqueeze(-2), filter2)  # [B,N,dim_in]

        supports2 = torch.softmax(torch.matmul(nodevec4, nodevec3.transpose(-2, -1)), dim=-1)

        # 子图和对应节点索引生成部分
        values, indices = supports2.topk(self.K, dim=-1, largest=True, sorted=True)
        batch_indices = torch.arange(nodevec1.size(0),device=nodevec1.device).unsqueeze(-1).unsqueeze(-1).expand(-1, indices.size(1),
                                                                                   indices.size(2))
        # 这是得到的记忆网络的topk索引对应节点提取到的特征
        selected_nodes_features1 = nodevec1[batch_indices, indices]
        selected_nodes_features2 = nodevec2[batch_indices, indices]
        selected_nodes_features = [selected_nodes_features1,selected_nodes_features2]
        return selected_nodes_features, batch_indices, indices


class DGCRN(BaseModel):
    '''
    Reference code: https://github.com/tsinghua-fib-lab/Traffic-Benchmark/tree/master/methods/DGCRN
    '''
    def __init__(self, device, predefined_adj, gcn_depth, rnn_size, hyperGNN_dim, node_dim, \
                 middle_dim, list_weight, tpd, tanhalpha, cl_decay_step, dropout,use_subgraph,topk,memory_node, **args):
        super(DGCRN, self).__init__(**args)
        self.device = device
        self.predefined_adj = predefined_adj
        self.hidden_size = rnn_size
        self.tpd = tpd
        self.alpha = tanhalpha
        self.cl_decay_step = cl_decay_step
        self.use_curriculum_learning = True

        self.emb1 = nn.Embedding(self.node_num, node_dim)
        self.emb2 = nn.Embedding(self.node_num, node_dim)
        self.lin1 = nn.Linear(node_dim, node_dim)
        self.lin2 = nn.Linear(node_dim, node_dim)
        self.idx = torch.arange(self.node_num).to(self.device)

        dims_hyper = [self.hidden_size + self.input_dim, hyperGNN_dim, middle_dim, node_dim]
        self.GCN1_tg = gcn(dims_hyper, gcn_depth, dropout, *list_weight, 'hyper')
        self.GCN2_tg = gcn(dims_hyper, gcn_depth, dropout, *list_weight, 'hyper')

        self.GCN1_tg_de = gcn(dims_hyper, gcn_depth, dropout, *list_weight, 'hyper')
        self.GCN2_tg_de = gcn(dims_hyper, gcn_depth, dropout, *list_weight, 'hyper')

        self.GCN1_tg_1 = gcn(dims_hyper, gcn_depth, dropout, *list_weight, 'hyper')
        self.GCN2_tg_1 = gcn(dims_hyper, gcn_depth, dropout, *list_weight, 'hyper')

        self.GCN1_tg_de_1 = gcn(dims_hyper, gcn_depth, dropout, *list_weight, 'hyper')
        self.GCN2_tg_de_1 = gcn(dims_hyper, gcn_depth, dropout, *list_weight, 'hyper')

        self.fc_final = nn.Linear(self.hidden_size, self.output_dim)

        dims = [self.input_dim + self.hidden_size, self.hidden_size]

        self.K = topk
        self.M = memory_node
        self.use_subgraph = use_subgraph
        self.nodeselection = nodeselection(self.K, self.M, node_dim)

        self.gz1 = gcn(dims, gcn_depth, dropout, *list_weight, 'RNN', self.use_subgraph,self.K,self.M)
        self.gz2 = gcn(dims, gcn_depth, dropout, *list_weight, 'RNN', self.use_subgraph,self.K,self.M)
        self.gr1 = gcn(dims, gcn_depth, dropout, *list_weight, 'RNN', self.use_subgraph,self.K,self.M)
        self.gr2 = gcn(dims, gcn_depth, dropout, *list_weight, 'RNN', self.use_subgraph,self.K,self.M)
        self.gc1 = gcn(dims, gcn_depth, dropout, *list_weight, 'RNN', self.use_subgraph,self.K,self.M)
        self.gc2 = gcn(dims, gcn_depth, dropout, *list_weight, 'RNN', self.use_subgraph,self.K,self.M)

        self.gz1_de = gcn(dims, gcn_depth, dropout, *list_weight, 'RNN', self.use_subgraph,self.K,self.M)
        self.gz2_de = gcn(dims, gcn_depth, dropout, *list_weight, 'RNN', self.use_subgraph,self.K,self.M)
        self.gr1_de = gcn(dims, gcn_depth, dropout, *list_weight, 'RNN', self.use_subgraph,self.K,self.M)
        self.gr2_de = gcn(dims, gcn_depth, dropout, *list_weight, 'RNN', self.use_subgraph,self.K,self.M)
        self.gc1_de = gcn(dims, gcn_depth, dropout, *list_weight, 'RNN', self.use_subgraph,self.K,self.M)
        self.gc2_de = gcn(dims, gcn_depth, dropout, *list_weight, 'RNN', self.use_subgraph,self.K,self.M)

        # self.K = 60
        # self.M = 4
        # self.use_subgraph = True
        # self.nodeselection = nodeselection(self.K, self.M, node_dim)

    def preprocessing(self, adj, predefined_adj):
        adj = adj + torch.eye(adj.shape[-1],device=self.device)
        adj = adj / torch.unsqueeze(adj.sum(-1), -1)
        return [adj, predefined_adj]


    def step(self, input, Hidden_State, Cell_State, predefined_adj, type='encoder', i=None):
        x = input
        x = x.transpose(1, 2).contiguous()

        nodevec1 = self.emb1(self.idx)
        nodevec2 = self.emb2(self.idx)

        hyper_input = torch.cat(
            (x, Hidden_State.view(-1, self.node_num, self.hidden_size)), 2)

        if type == 'encoder':
            filter1 = self.GCN1_tg(hyper_input, predefined_adj[0]) + \
                      self.GCN1_tg_1(hyper_input, predefined_adj[1])

            filter2 = self.GCN2_tg(hyper_input, predefined_adj[0]) + \
                      self.GCN2_tg_1(hyper_input, predefined_adj[1])

        if type == 'decoder':
            filter1 = self.GCN1_tg_de(hyper_input, predefined_adj[0]) + \
                      self.GCN1_tg_de_1(hyper_input, predefined_adj[1])

            filter2 = self.GCN2_tg_de(hyper_input, predefined_adj[0]) + \
                      self.GCN2_tg_de_1(hyper_input, predefined_adj[1])

        nodevec1 = torch.tanh(self.alpha * torch.mul(nodevec1, filter1))
        nodevec2 = torch.tanh(self.alpha * torch.mul(nodevec2, filter2))


        if self.use_subgraph:
            nodevec =[nodevec1,nodevec2]
            [selected_nodes_features1, selected_nodes_features2], batch_indices, indices = self.nodeselection(nodevec)
            #子图生成
            a = torch.matmul(selected_nodes_features1, selected_nodes_features2.transpose(-2, -1)) - torch.matmul(
                selected_nodes_features2, selected_nodes_features1.transpose(-2, -1))
        else:
            a = torch.matmul(nodevec1, nodevec2.transpose(2, 1)) - torch.matmul(
                nodevec2, nodevec1.transpose(2, 1))

        adj = F.relu(torch.tanh(self.alpha * a))

        adp = self.preprocessing(adj, predefined_adj[0])
        adpT = self.preprocessing(adj.transpose(-1, -2), predefined_adj[1])

        Hidden_State = Hidden_State.view(-1, self.node_num, self.hidden_size)
        Cell_State = Cell_State.view(-1, self.node_num, self.hidden_size)

        combined = torch.cat((x, Hidden_State), -1)

        if self.use_subgraph:
            if type == 'encoder':
                z = torch.sigmoid(self.gz1(combined, adp,batch_indices, indices) + self.gz2(combined, adpT,batch_indices, indices))
                r = torch.sigmoid(self.gr1(combined, adp,batch_indices, indices) + self.gr2(combined, adpT,batch_indices, indices))

                temp = torch.cat((x, torch.mul(r, Hidden_State)), -1)
                Cell_State = torch.tanh(self.gc1(temp, adp,batch_indices, indices) + self.gc2(temp, adpT,batch_indices, indices))

            elif type == 'decoder':
                z = torch.sigmoid(
                    self.gz1_de(combined, adp,batch_indices, indices) + self.gz2_de(combined, adpT,batch_indices, indices))
                r = torch.sigmoid(
                    self.gr1_de(combined, adp,batch_indices, indices) + self.gr2_de(combined, adpT,batch_indices, indices))
                temp = torch.cat((x, torch.mul(r, Hidden_State)), -1)
                Cell_State = torch.tanh(
                    self.gc1_de(temp, adp,batch_indices, indices) + self.gc2_de(temp, adpT,batch_indices, indices))
        else:
            if type == 'encoder':
                z = torch.sigmoid(self.gz1(combined, adp) + self.gz2(combined, adpT))
                r = torch.sigmoid(self.gr1(combined, adp) + self.gr2(combined, adpT))

                temp = torch.cat((x, torch.mul(r, Hidden_State)), -1)
                Cell_State = torch.tanh(self.gc1(temp, adp) + self.gc2(temp, adpT))

            elif type == 'decoder':
                z = torch.sigmoid(
                    self.gz1_de(combined, adp) + self.gz2_de(combined, adpT))
                r = torch.sigmoid(
                    self.gr1_de(combined, adp) + self.gr2_de(combined, adpT))
                temp = torch.cat((x, torch.mul(r, Hidden_State)), -1)
                Cell_State = torch.tanh(
                    self.gc1_de(temp, adp) + self.gc2_de(temp, adpT))

        Hidden_State = torch.mul(z, Hidden_State) + torch.mul(
            1 - z, Cell_State)
        return Hidden_State.view(-1, self.hidden_size), Cell_State.view(
            -1, self.hidden_size)


    def compute_future_info(self, his):
        b, _, n, _ = his.shape
        tod, dow = his[:,0,0,:], his[:,1,0,:]
        time_unit = 1 / self.tpd * self.horizon
        day_unit = 1 / 7

        out_tod = torch.full_like(tod, 0)
        out_dow = torch.full_like(dow, 0)
        for i in range(b):
            temp = tod[i] + time_unit
            temp2 = dow[i,-1].repeat(self.horizon)

            idxs = torch.where(temp >= 1)[0]
            if len(idxs) != 0:
                temp[idxs] -= 1

                idx = torch.where(temp == 0)[0]
                if len(idx) != 0:
                    temp2[idx:] += day_unit

            out_tod[i] = temp
            out_dow[i] = temp2
        
        out_tod = out_tod.unsqueeze(-1).expand(-1, -1, n).unsqueeze(-1)
        out_dow = out_dow.unsqueeze(-1).expand(-1, -1, n).unsqueeze(-1)
        
        out = torch.cat((out_tod, out_dow), dim=-1).transpose(1, 3)
        return out


    def forward(self, input, label=None, batches_seen=None, task_level=12):  # (b, t, n, f)
        x = input[...,:self.input_dim].transpose(1, 3)
        label = label.transpose(1, 3)

        batch_size = x.size(0)
        Hidden_State, Cell_State = self.initHidden(batch_size * self.node_num,
                                                   self.hidden_size)

        outputs = None
        for i in range(self.seq_len):
            Hidden_State, Cell_State = self.step(torch.squeeze(x[..., i]),
                                                 Hidden_State, Cell_State,
                                                 self.predefined_adj, 'encoder', i)
            if outputs is None:
                outputs = Hidden_State.unsqueeze(1)
            else:
                outputs = torch.cat((outputs, Hidden_State.unsqueeze(1)), 1)

        timeofday = self.compute_future_info(input.transpose(1, 3)[:,1:,:,:])[:,:self.input_dim-self.output_dim,...]
        decoder_input = torch.zeros((batch_size, self.output_dim, self.node_num), device=self.device)
        outputs_final = []
        for i in range(task_level):
            try:
                decoder_input = torch.cat([decoder_input, timeofday[..., i]], dim=1)
            except:
                print(decoder_input.shape, timeofday.shape)
                sys.exit(0)
            Hidden_State, Cell_State = self.step(decoder_input, Hidden_State,
                                                 Cell_State, self.predefined_adj,
                                                 'decoder', None)

            decoder_output = self.fc_final(Hidden_State)
            decoder_input = decoder_output.view(batch_size, self.node_num,
                                                self.output_dim).transpose(1, 2)
            outputs_final.append(decoder_output)

            if self.training and self.use_curriculum_learning:
                c = np.random.uniform(0, 1)
                if c < self.compute_sampling_threshold(batches_seen):
                    decoder_input = label[:, :1, :, i]

        outputs_final = torch.stack(outputs_final, dim=1)

        outputs_final = outputs_final.view(batch_size, self.node_num,
                                           task_level, self.output_dim).transpose(1, 2)
        return outputs_final


    def initHidden(self, batch_size, hidden_size):
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            Hidden_State = Variable(
                torch.zeros([batch_size, hidden_size],device=self.device))#.to(self.device))
            Cell_State = Variable(
                torch.zeros([batch_size, hidden_size],device=self.device))#.to(self.device))
            nn.init.orthogonal_(Hidden_State)
            nn.init.orthogonal_(Cell_State)
            return Hidden_State, Cell_State
        else:
            Hidden_State = Variable(torch.zeros(batch_size, hidden_size))
            Cell_State = Variable(torch.zeros(batch_size, hidden_size))
            return Hidden_State, Cell_State


    def compute_sampling_threshold(self, batches_seen):
        x = self.cl_decay_step / (
            self.cl_decay_step + np.exp(batches_seen / self.cl_decay_step))
        return x

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
        dict1 = torch.zeros(x.shape,device=x.device).clone()
        dict1.scatter_add_(1, indices_expanded, node_new)

        dict_refined = torch.ones((B, N),device=x.device) * 10 ** -14
        # dict_refined = torch.zeros(B, N).cuda()
        # dict_refined = torch.ones(B, N, requires_grad=True).cuda()
        # 获取 indices1 的扁平版本和对应的批次索引
        flat_indices = indices.flatten()
        batch_indices1 = torch.arange(indices.size(0),device=x.device).repeat_interleave(indices.size(1) * indices.size(2))

        # 构建用于 scatter_add 的源张量，因为我们需要在每个索引处累加 1
        ones_source = torch.ones_like(flat_indices,device=x.device, dtype=dict_refined.dtype)

        # 使用 scatter_add 在正确的位置累加 1
        # 由于 indices1 有重复的索引，我们需要先转换它们为线性索引
        linear_indices = flat_indices + batch_indices1 * dict_refined.size(1)
        # 得到的索引dict_refined
        dict_refined.put_(linear_indices, ones_source, accumulate=True)

        x1 = dict1 / dict_refined.unsqueeze(-1).expand(B, N, D)
        return x1

class gcn(nn.Module):
    def __init__(self, dims, gdep, dropout, alpha, beta, gamma, type=None, use_subgraph=None,K=None,M=None):
        super(gcn, self).__init__()
        self.use_subgraph= use_subgraph
        if type == 'RNN':
            if self.use_subgraph:
                self.gconv = feature_aggregation(K,M)
            else:
                self.gconv = gconv_RNN()
            self.gconv_preA = gconv_hyper()
            self.mlp = nn.Linear((gdep + 1) * dims[0], dims[1])

        elif type == 'hyper':
            self.gconv = gconv_hyper()
            self.mlp = nn.Sequential(
                OrderedDict([('fc1', nn.Linear((gdep + 1) * dims[0], dims[1])),
                             ('sigmoid1', nn.Sigmoid()),
                             ('fc2', nn.Linear(dims[1], dims[2])),
                             ('sigmoid2', nn.Sigmoid()),
                             ('fc3', nn.Linear(dims[2], dims[3]))]))
        self.gdep = gdep
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.type_GNN = type


    def forward(self, x, adj,batch_indices=None,indices=None):
        h = x
        out = [h]
        if self.type_GNN == 'RNN':
            if self.use_subgraph:
                for _ in range(self.gdep):
                    h = self.alpha * x + self.beta * self.gconv(x,adj[0],batch_indices,indices) + self.gamma * self.gconv_preA(h, adj[1])
                    out.append(h)
            else:
                for _ in range(self.gdep):
                    h = self.alpha * x + self.beta * self.gconv(
                        h, adj[0]) + self.gamma * self.gconv_preA(h, adj[1])
                    out.append(h)
        else:
            for _ in range(self.gdep):
                h = self.alpha * x + self.gamma * self.gconv(h, adj)
                out.append(h)
        ho = torch.cat(out, dim=-1)
        ho = self.mlp(ho)
        return ho


class gconv_RNN(nn.Module):
    def __init__(self):
        super(gconv_RNN, self).__init__()


    def forward(self, x, A):
        x = torch.einsum('nvc,nvw->nwc', (x, A))
        return x.contiguous()


class gconv_hyper(nn.Module):
    def __init__(self):
        super(gconv_hyper, self).__init__()


    def forward(self, x, A):
        x = torch.einsum('nvc,vw->nwc', (x, A))
        return x.contiguous()