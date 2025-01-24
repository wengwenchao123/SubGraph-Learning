import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pywt

class temporalEmbedding(nn.Module):
    def __init__(self, D):
        super(temporalEmbedding, self).__init__()
        self.ff_te = FeedForward([295,D,D])

    def forward(self, TE, T=288, W=7):
        '''
        TE:[B,T,2]
        '''
        dayofweek = torch.empty(TE.shape[0], TE.shape[1], 7).to(TE.device) # [B,T,7]
        timeofday = torch.empty(TE.shape[0], TE.shape[1], T).to(TE.device) # [B,T,288]
        for i in range(TE.shape[0]):
            dayofweek[i] = F.one_hot(TE[..., 0][i].to(torch.int64) % W, W)
        for j in range(TE.shape[0]):
            timeofday[j] = F.one_hot(TE[..., 1][j].to(torch.int64) % T, T)
        TE = torch.cat((dayofweek, timeofday), dim=-1) # [B,T,295]
        TE = TE.unsqueeze(dim=2) # [B,T,1,295]
        TE = self.ff_te(TE) # [B,T,1,F]

        return TE  # [B,T,1,F]

class FeedForward(nn.Module):
    def __init__(self, fea, res_ln=False):
        super(FeedForward, self).__init__()
        self.res_ln = res_ln
        self.L = len(fea) - 1
        self.linear = nn.ModuleList([nn.Linear(fea[i], fea[i+1]) for i in range(self.L)])
        self.ln = nn.LayerNorm(fea[self.L], elementwise_affine=False)

    def forward(self, inputs):
        x = inputs
        for i in range(self.L):
            x = self.linear[i](x)
            if i != self.L-1:
                x = F.relu(x)
        if self.res_ln:
            x += inputs
            x = self.ln(x)
        return x

class nodeselection(nn.Module):
    def __init__(self, topk, memory_num, time_dim):
        super(nodeselection, self).__init__()
        self.K = topk
        self.memory_num = memory_num
        self.node_embeddings = nn.Parameter(torch.randn(memory_num, 2*time_dim), requires_grad=True)

    def forward(self, node_feature, node_embeddings=None):
        nodevec1 = node_feature[0]
        B, T, N, D = nodevec1.shape

        nodevec4 = self.node_embeddings


        supports2 = torch.softmax(torch.matmul(nodevec4, nodevec3.transpose(-2, -1)), dim=-1)

        
        values, indices = supports2.topk(self.K, dim=-1, largest=True, sorted=True)

        batch_indices = torch.arange(nodevec1.size(0), device=nodevec1.device).view(B, 1, 1, 1).expand(-1, T,
                                                                                                       self.memory_num,
                                                                                                       self.K)
        # B, T, N, D = nodevec1.shape
        # _, _, K, M = indices.shape

        time_indices = torch.arange(T, device=nodevec1.device).view(1, T, 1, 1).expand(B, -1, self.memory_num,
                                                                                           self.K)

        selected_nodes_features1 = nodevec1[batch_indices, time_indices, indices]
        selected_nodes_features2 = nodevec2[batch_indices, time_indices, indices]
        selected_nodes_features = [selected_nodes_features1,selected_nodes_features2]
        return selected_nodes_features, batch_indices,time_indices, indices


class feature_aggregation(nn.Module):
    def __init__(self, K, N):
        super(feature_aggregation, self).__init__()
        self.K = K
        self.N = N

   
    def forward(self, x, adj, batch_indices, time_indices, indices):
        B, T, N, D = x.shape
        _, _, M, K =indices.shape
        # selected_nodes = x[batch_indices, indices]

        # x1 = torch.matmul(a, x)
        selected_nodes = x[batch_indices, time_indices, indices]
        node_new = torch.matmul(adj, selected_nodes).reshape(B, T, self.K * self.N, D)

        indices_expanded = indices.unsqueeze(-1).expand(-1, -1, -1, -1, D).reshape(B, T, self.K * self.N, D)
        dict1 = torch.zeros(x.shape, device=x.device).clone()  # .cuda()
        dict1.scatter_add_(-2, indices_expanded, node_new)

        dict_refined = torch.ones((B, T, N), device=x.device) * 10 ** -14

        flat_indices = indices.flatten()
        batch_indices1 = torch.arange(B*T, device=x.device).repeat_interleave(M*K)
            # indices.size(1) * indices.size(2))  # .cuda()

        ones_source = torch.ones_like(flat_indices, device=x.device, dtype=dict_refined.dtype)  # .cuda()

        linear_indices = flat_indices + batch_indices1 * N # N = dict_refined.size(-1)
        dict_refined.put_(linear_indices, ones_source, accumulate=True)

        x1 = dict1 / dict_refined.unsqueeze(-1).expand(B, T, N, D)
        return x1

class sparseSpatialAttention(nn.Module):
    def __init__(self, hidden_size, log_samples,topk,memory_node,which_to_use):
        super(sparseSpatialAttention, self).__init__()
        self.qfc = FeedForward([hidden_size, hidden_size])
        self.kfc = FeedForward([hidden_size, hidden_size])
        self.vfc = FeedForward([hidden_size, hidden_size])
        self.ofc = FeedForward([hidden_size, hidden_size])

        self._hidden_size = hidden_size
        self._log_samples = log_samples

        self.ln = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.ff = FeedForward([hidden_size,hidden_size,hidden_size], True)
        self.proj = nn.Linear(hidden_size, 1)

        self.topk = topk
        self.memory_node = memory_node
        self.which_to_use =which_to_use
        self.nodeselection = nodeselection(self.topk, self.memory_node, hidden_size)
        self.fal = feature_aggregation(self.topk,self.memory_node)
    def forward(self, x, adj, eigvec, eigvalue):
        '''
        [B,T,N,D]
        '''
        # add spatial positional encoding
        x_ = x + torch.matmul(eigvec.transpose(0,1).squeeze(-1), torch.diag_embed(eigvalue))

        Q = self.qfc(x_)
        K = self.kfc(x_)
        V = self.vfc(x_)

        B, T, N, D = Q.shape

        if self.which_to_use == "use_subgraph" :
            node_feature = [Q, K]
            [selected_nodes_Q, selected_nodes_K], batch_indices, time_indices, indices = self.nodeselection(
                node_feature)
            attn_score = (
                                 selected_nodes_Q @ selected_nodes_K.transpose(-1, -2)
                         ) / self._hidden_size ** 0.5  # (num_heads * batch_size, ..., tgt_length, src_length)

            attn_score1 = torch.softmax(attn_score, dim=-1)

            value = self.fal(V, attn_score1, batch_indices, time_indices, indices)

        elif self.which_to_use == "use_ESGAT" :
            # # use gat results to reduce Q
            K_expand = K.unsqueeze(-3).expand(B, T, N, N, D)
            K_sample = K_expand[:, :, torch.arange(N).unsqueeze(1), adj, :]
            V_expand = V.unsqueeze(-3).expand(B, T, N, N, D)
            V_sample = V_expand[:, :, torch.arange(N).unsqueeze(1), adj, :]
            Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1))
            GAT_results = torch.matmul(Q_K_sample, V_sample).squeeze(-2)
            M = self.proj(GAT_results).squeeze(-1)
            samples = int(self._log_samples * math.log(N, 2))
            M_top = M.topk(samples, sorted=False)[1]

            # use the reduced Q to calculate Q_K
            Q_reduce = Q[torch.arange(B)[:, None, None],
                         torch.arange(T)[None, :, None],
                         M_top, :]
            Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))

            Q_K /= (self._hidden_size ** 0.5)

            attn = torch.softmax(Q_K, dim=-1)

            # copy operation
            cp = attn.argmax(dim=-2, keepdim=True).transpose(-2,-1)
            value = torch.matmul(attn, V).unsqueeze(-3).expand(B, T, N, M_top.shape[-1], V.shape[-1])[torch.arange(B)[:, None, None, None],
                                                                                                     torch.arange(T)[None, :, None, None],
                                                                                                     torch.arange(N)[None, None, :, None],cp,:].squeeze(-2)
        elif self.which_to_use == "use_full_GAT" :
            K = K.transpose(
                -1, -2
            )  # (num_heads * batch_size, ..., head_dim, src_length)

            attn_score = (
                                 Q @ K
                         ) / (self._hidden_size ** 0.5)  # (num_heads * batch_size, ..., tgt_length, src_length)

            attn_score = torch.softmax(attn_score, dim=-1)
            value = attn_score @ V  # (num_heads * batch_size, ..., tgt_length, head_dim)

        value = self.ofc(value) + x_
        value = self.ln(value)
        return self.ff(value)

class temporalAttention(nn.Module):
    def __init__(self, hidden_size):
        super(temporalAttention, self).__init__()
        self.qfc = FeedForward([hidden_size,hidden_size])
        self.kfc = FeedForward([hidden_size,hidden_size])
        self.vfc = FeedForward([hidden_size,hidden_size])
        self.ofc = FeedForward([hidden_size,hidden_size])
        self._hidden_size = hidden_size

        self.ln = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.ff = FeedForward([hidden_size,hidden_size,hidden_size], True)

    def forward(self, x, te, Mask=True):
        '''
        x:[B,T,N,F]
        te:[B,T,N,F]
        '''
        x += te

        query = self.qfc(x).permute(0,2,1,3) #[B,N,T,F]
        key = self.kfc(x).permute(0,2,3,1) 
        value = self.vfc(x).permute(0,2,1,3) 

        attention = torch.matmul(query, key) # [B,N,T,T]
        attention /= (self._hidden_size ** 0.5) # scaled

        if Mask:
            batch_size = x.shape[0]
            num_steps = x.shape[1]
            num_vertexs = x.shape[2]
            mask = torch.ones(num_steps, num_steps).to(x.device) # [T,T]
            mask = torch.tril(mask) # [T,T]
            mask = torch.unsqueeze(torch.unsqueeze(mask, dim=0), dim=0) # [1,1,T,T]
            mask = mask.repeat(batch_size, num_vertexs, 1, 1) # [B,N,T,T]
            mask = mask.to(torch.bool)
            zero_vec = (-2 ** 15 + 1)*torch.ones_like(attention).to(x.device) # [B,N,T,T]
            attention = torch.where(mask, attention, zero_vec)

        attention = F.softmax(attention, -1) # [B,N,T,T]

        value = torch.matmul(attention, value).permute(0,2,1,3) # [B,N,T,d]
        value = self.ofc(value)
        value += x

        value = self.ln(value)

        return self.ff(value)

class Chomp1d(nn.Module):
    """
    extra dimension will be added by padding, remove it
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :, :-self.chomp_size].contiguous()

class temporalConvNet(nn.Module):
    def __init__(self, hidden_size, kernel_size=2, dropout=0.2, levels=1):
        super(temporalConvNet, self).__init__()
        layers = []
        for i in range(levels):
            dilation_size = 2 ** i
            padding = (kernel_size - 1) * dilation_size
            self.conv = nn.Conv2d(hidden_size, hidden_size, (1, kernel_size), dilation=(1, dilation_size), padding=(0, padding))
            self.chomp = Chomp1d(padding)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(dropout)

            layers += [nn.Sequential(self.conv, self.chomp, self.relu, self.dropout)]
        self.tcn = nn.Sequential(*layers)

    def forward(self, xh):
        xh = self.tcn(xh.transpose(1,3)).transpose(1,3)
        return xh

class adaptiveFusion(nn.Module):
    def __init__(self, hidden_size):
        super(adaptiveFusion, self).__init__()
        self.qlfc = FeedForward([hidden_size,hidden_size])
        self.khfc = FeedForward([hidden_size,hidden_size])
        self.vhfc = FeedForward([hidden_size,hidden_size])
        self.ofc = FeedForward([hidden_size,hidden_size])
        self._hidden_size = hidden_size

        self.ln = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.ff = FeedForward([hidden_size,hidden_size,hidden_size], True)

    def forward(self, xl, xh, te, Mask=True):
        '''
        xl: [B,T,N,F]
        xh: [B,T,N,F]
        tp: [B,T,1,F]
        '''
        xl += te
        xh += te

        query = self.qlfc(xl).permute(0,2,1,3) # [B,N,T,F]
        keyh = torch.relu(self.khfc(xh)).permute(0,2,3,1)
        valueh = torch.relu(self.vhfc(xh)).permute(0,2,1,3)

        attentionh = torch.matmul(query, keyh) # [B,N,T,T]
    
        if Mask:
            batch_size = xl.shape[0]
            num_steps = xl.shape[1]
            num_vertexs = xl.shape[2]
            mask = torch.ones(num_steps, num_steps).to(xl.device) # [T,T]
            mask = torch.tril(mask) # [T,T]
            mask = torch.unsqueeze(torch.unsqueeze(mask, dim=0), dim=0) # [1,1,T,T]
            mask = mask.repeat(batch_size, num_vertexs, 1, 1) # [B,N,T,T]
            mask = mask.to(torch.bool)
            zero_vec = (-2 ** 15 + 1)*torch.ones_like(attentionh).to(xl.device) # [B,N,T,T]
            attentionh = torch.where(mask, attentionh, zero_vec)
        attentionh /= (self._hidden_size ** 0.5) # scaled
        attentionh = F.softmax(attentionh, -1) # [B,N,T,T]


        value = torch.matmul(attentionh, valueh).permute(0,2,1,3)
        value = self.ofc(value)
        value = value + xl #+ xh

        value = self.ln(value)

        return self.ff(value)

class dualEncoder(nn.Module):
    def __init__(self, hidden_size, log_samples, adj_gat, graphwave,topk,memory_node,which_to_use):
        super(dualEncoder, self).__init__()
        self.tcn = temporalConvNet(hidden_size)
        self.tatt = temporalAttention(hidden_size)

        self.ssal = sparseSpatialAttention(hidden_size, log_samples,topk,memory_node,which_to_use)
        self.ssah = sparseSpatialAttention(hidden_size, log_samples,topk,memory_node,which_to_use)
    
        eigvalue = torch.from_numpy(graphwave[0].astype(np.float32))
        self.eigvalue = nn.Parameter(eigvalue, requires_grad=True)
        self.eigvec = torch.from_numpy(graphwave[1].astype(np.float32)).transpose(0,1).unsqueeze(-1)
        self.adj = torch.from_numpy(adj_gat)
    
    def forward(self, xl, xh, te):
        xl = self.tatt(xl, te)
        xh = self.tcn(xh)
    
        spa_statesl = self.ssal(xl, self.adj.to(xl.device), self.eigvec.to(xl.device), self.eigvalue.to(xl.device))
        spa_statesh = self.ssah(xh, self.adj.to(xl.device), self.eigvec.to(xl.device), self.eigvalue.to(xl.device))
        xl = spa_statesl + xl
        xh = spa_statesh + xh
    
        return xl, xh

class STWave(nn.Module):
    """
    Paper: When Spatio-Temporal Meet Wavelets: Disentangled Traffic Forecasting via Efficient Spectral Graph Attention Networks
    Link: https://ieeexplore.ieee.org/document/10184591
    Ref Official Code: https://github.com/LMissher/STWave
    Venue: ICDE 2023
    Task: Spatial-Temporal Forecasting
    Hints: PyWavelets and pytorch_wavelets packages are needed
    """
    def __init__(self, input_dim, hidden_size, layers, seq_len, horizon, log_samples, adj_gat, graphwave, time_in_day_size, day_in_week_size, wave_type, wave_levels,topk,memory_node,which_to_use):
        super(STWave, self).__init__()
        self.start_emb_l = FeedForward([input_dim, hidden_size, hidden_size])
        self.start_emb_h = FeedForward([input_dim, hidden_size, hidden_size])
        self.te_emb = temporalEmbedding(hidden_size)

        self.dual_encoder = nn.ModuleList([dualEncoder(hidden_size, log_samples, adj_gat, graphwave,topk,memory_node,which_to_use) for i in range(layers)])
        self.adaptive_fusion = adaptiveFusion(hidden_size)
    
        self.pre_l = nn.Conv2d(seq_len, horizon, (1,1))
        self.pre_h = nn.Conv2d(seq_len, horizon, (1,1))
    
        self.end_emb = FeedForward([hidden_size, hidden_size, input_dim])
        self.end_emb_l = FeedForward([hidden_size, hidden_size, input_dim])

        self.td = time_in_day_size
        self.dw = day_in_week_size
        self.id = input_dim
        self.wt = wave_type
        self.wl = wave_levels

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool, **kwargs):
        '''
        x:[B,T,N,D]
        '''
        x = history_data
        te = torch.cat([x[:,:,0,1:2]*self.td, x[:,:,0,2:]*self.dw], -1)
        ADD = torch.arange(te.shape[1]).to(x.device).unsqueeze(0).unsqueeze(2) + 1
        TEYTOD = (te[:,-1:,0:1] + ADD) % self.td
        TEYDOW = (torch.floor((te[:,-1:,0:1] + ADD) / self.td) + te[...,1:2]) % self.dw
        te = torch.cat([te, torch.cat([TEYTOD, TEYDOW], -1)], 1)
        te = te[...,[1,0]]

        inputs = x[...,:self.id]
        xl, xh = disentangle(inputs[...,0:1].cpu().numpy(), self.wt, self.wl)

        xl, xh, TE = self.start_emb_l(xl.to(x.device)), self.start_emb_h(xh.to(x.device)), self.te_emb(te, self.td, self.dw)

        for enc in self.dual_encoder:
            xl, xh = enc(xl, xh, TE[:,:xl.shape[1],:,:])
    
        hat_y_l = self.pre_l(xl)
        hat_y_h = self.pre_h(xh)

        hat_y = self.adaptive_fusion(hat_y_l, hat_y_h, TE[:,xl.shape[1]:,:,:])
        hat_y, hat_y_l = self.end_emb(hat_y), self.end_emb_l(hat_y_l)
    
        if self.training:
            label_yl, _ = disentangle(future_data[...,0:1].cpu().numpy(), self.wt, self.wl)
            return torch.cat([hat_y, hat_y_l, label_yl.to(x.device)], -1)
    
        return hat_y


def disentangle(x, w, j):
    x = x.transpose(0,3,2,1) # [B,D,N,T]
    coef = pywt.wavedec(x, w, level=j)
    coefl = [coef[0]]
    for i in range(len(coef)-1):
        coefl.append(None)
    coefh = [None]
    for i in range(len(coef)-1):
        coefh.append(coef[i+1])
    xl = pywt.waverec(coefl, w).transpose(0,3,2,1)
    xh = pywt.waverec(coefh, w).transpose(0,3,2,1)

    return torch.from_numpy(xl), torch.from_numpy(xh)
