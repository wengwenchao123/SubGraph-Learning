import torch
import torch.nn.functional as F
import torch.nn as nn

class STEmbModel(torch.nn.Module):
    def __init__(self, SEDims, TEDims, OutDims):
        super(STEmbModel, self).__init__()
        self.TEDims = TEDims
        self.fc3 = torch.nn.Linear(SEDims, OutDims)
        self.fc4 = torch.nn.Linear(OutDims, OutDims)
        self.fc5 = torch.nn.Linear(TEDims, OutDims)
        self.fc6 = torch.nn.Linear(OutDims, OutDims)
        # self.device = device


    def forward(self, SE, TE):
        SE = SE.unsqueeze(0).unsqueeze(0)
        SE = self.fc4(F.relu(self.fc3(SE)))
        dayofweek_index = TE[..., 1]
        timeofday_index = TE[..., 0]
        dayofweek = F.one_hot(dayofweek_index, num_classes = 7)
        timeofday = F.one_hot(timeofday_index, num_classes = self.TEDims-7)
        TE = torch.cat((dayofweek, timeofday), dim=-1)
        TE = TE.unsqueeze(2).to(torch.float)
        TE = self.fc6(F.relu(self.fc5(TE)))
        sum_tensor = torch.add(SE, TE)
        return sum_tensor

class nodeselection(nn.Module):
    def __init__(self, topk, memory_num, time_dim):
        super(nodeselection, self).__init__()
        self.K = topk
        self.memory_num = memory_num
        self.node_embeddings = nn.Parameter(torch.randn(memory_num, 2*time_dim), requires_grad=True)

    def forward(self, node_feature, node_embeddings=None):
        nodevec1 = node_feature[0]
        B, T, N, D = nodevec1.shape

        nodevec2 = node_feature[1]
        nodevec3 = torch.cat((nodevec1, nodevec2), -1)

        nodevec4 = self.node_embeddings

        supports2 = torch.softmax(torch.matmul(nodevec4, nodevec3.transpose(-2, -1)), dim=-1)

        values, indices = supports2.topk(self.K, dim=-1, largest=True, sorted=True)

        batch_indices = torch.arange(nodevec1.size(0), device=nodevec1.device).view(B, 1, 1, 1).expand(-1, T,
                                                                                                       self.memory_num,
                                                                                                       self.K)


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

class SpatialAttentionModel(torch.nn.Module):
    def __init__(self, K, d,args):
        super(SpatialAttentionModel, self).__init__()
        D = K*d
        self.fc7 = torch.nn.Linear(2*D, D)
        self.fc8 = torch.nn.Linear(2*D, D)
        self.fc9 = torch.nn.Linear(2*D, D)
        self.fc10 = torch.nn.Linear(D, D)
        self.fc11 = torch.nn.Linear(D, D)
        self.K = K
        self.d = d
        self.softmax = torch.nn.Softmax(dim=-1)
        self.topk = args.topk
        self.memory_node = args.memory_node
        self.use_subgraph = args.use_subgraph
        self.nodeselection = nodeselection(self.topk, self.memory_node, self.d)
        self.fal = feature_aggregation(self.topk,self.memory_node)

    def forward(self, X, STE):
        batchsize = X.shape[0]
        X = torch.cat((X, STE), dim=-1)
        query = F.relu(self.fc7(X))
        key = F.relu(self.fc8(X))
        value = F.relu(self.fc9(X))

        query = torch.cat(torch.split(query, self.d, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.d, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.d, dim=-1), dim=0)

        if self.use_subgraph :
            node_feature = [query, key]
            [selected_nodes_query, selected_nodes_key], batch_indices, time_indices, indices = self.nodeselection(
                node_feature)
            attention = torch.matmul(selected_nodes_query, torch.transpose(selected_nodes_key, -1, -2))
            attention /= (self.d ** 0.5)
            attention = self.softmax(attention)
            X = self.fal(value, attention, batch_indices, time_indices, indices)
            # X = torch.matmul(attention, value)
        else:

            attention = torch.matmul(query, torch.transpose(key, 2, 3))
            attention /= (self.d ** 0.5)
            attention = self.softmax(attention)
            X = torch.matmul(attention, value)
        X = torch.cat(torch.split(X, batchsize, dim=0), dim=-1)
        X = self.fc11(F.relu(self.fc10(X)))
        return X
        

class TemporalAttentionModel(torch.nn.Module):
    def __init__(self, K, d):
        super(TemporalAttentionModel, self).__init__()
        D = K*d
        self.fc12 = torch.nn.Linear(2*D, D)
        self.fc13 = torch.nn.Linear(2*D, D)
        self.fc14 = torch.nn.Linear(2*D, D)
        self.fc15 = torch.nn.Linear(D, D)
        self.fc16 = torch.nn.Linear(D, D)
        self.K = K
        self.d = d
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, X, STE):
        batchsize=X.shape[0]
        X = torch.cat((X, STE), dim=-1)
        query = F.relu(self.fc12(X))
        key = F.relu(self.fc13(X))
        value = F.relu(self.fc14(X))
        query = torch.cat(torch.split(query, self.d, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.d, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.d, dim=-1), dim=0)
        query = torch.transpose(query, 2, 1)
        key = torch.transpose(torch.transpose(key, 1, 2), 2, 3)
        value = torch.transpose(value, 2, 1)
        attention = torch.matmul(query, key)
        attention /= (self.d ** 0.5)
        attention = self.softmax(attention)
        X = torch.matmul(attention, value)
        X = torch.transpose(X, 2, 1)
        X = torch.cat(torch.split(X, batchsize, dim=0), dim=-1)
        X = self.fc16(F.relu(self.fc15(X)))
        return X



class GatedFusionModel(torch.nn.Module):
    def __init__(self, K, d):
        super(GatedFusionModel, self).__init__()
        D = K*d
        self.fc17 = torch.nn.Linear(D, D)
        self.fc18 = torch.nn.Linear(D, D)
        self.fc19 = torch.nn.Linear(D, D)
        self.fc20 = torch.nn.Linear(D, D)
        self.sigmoid = torch.nn.Sigmoid()
        
    def forward(self, HS, HT):
        XS = self.fc17(HS)
        XT = self.fc18(HT)
        z = self.sigmoid(torch.add(XS, XT))
        H = torch.add((z* HS), ((1-z)* HT))
        H = self.fc20(F.relu(self.fc19(H)))
        return H


class STAttModel(torch.nn.Module):
    def __init__(self, K, d,args):
        super(STAttModel, self).__init__()
        self.spatialAttention = SpatialAttentionModel(K, d,args)
        self.temporalAttention = TemporalAttentionModel(K, d)
        self.gatedFusion = GatedFusionModel(K, d)

    def forward(self, X, STE):
        HS = self.spatialAttention(X, STE)
        HT = self.temporalAttention(X, STE)
        H = self.gatedFusion(HS, HT)
        return torch.add(X, H)


class TransformAttentionModel(torch.nn.Module):
    def __init__(self, K, d):
        super(TransformAttentionModel, self).__init__()
        D = K * d
        self.fc21 = torch.nn.Linear(D, D)
        self.fc22 = torch.nn.Linear(D, D)
        self.fc23 = torch.nn.Linear(D, D)
        self.fc24 = torch.nn.Linear(D, D)
        self.fc25 = torch.nn.Linear(D, D)
        self.K = K
        self.d = d
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, X, STE_P, STE_Q):
        query = F.relu(self.fc21(STE_Q))
        key = F.relu(self.fc22(STE_P))
        value = F.relu(self.fc23(X))
        query = torch.cat(torch.split(query, self.d, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.d, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.d, dim=-1), dim=0)
        query = torch.transpose(query, 2, 1)
        key = torch.transpose(torch.transpose(key, 1, 2), 2, 3)
        value = torch.transpose(value, 2, 1)
        attention = torch.matmul(query, key)
        attention /= (self.d ** 0.5)
        attention = self.softmax(attention)
        X = torch.matmul(attention, value)
        X = torch.transpose(X, 2, 1)
        X = torch.cat(torch.split(X, X.shape[0]//self.K, dim=0), dim=-1)
        X = self.fc25(F.relu(self.fc24(X)))
        return X


class GMAN(torch.nn.Module):
    def __init__(self, K, d, SEDims, TEDims,SE,args):
        super(GMAN, self).__init__()
        D = K*d
        self.fc1 = torch.nn.Linear(1, D)
        self.fc2 = torch.nn.Linear(D, D)
        self.STEmb = STEmbModel(SEDims, TEDims, K*d)
        self.STAttBlockEnc = STAttModel(K, d,args)
        self.STAttBlockDec = STAttModel(K, d,args)
        self.transformAttention = TransformAttentionModel(K, d)
        self.fc26 = torch.nn.Linear(D, D)
        self.fc27 = torch.nn.Linear(D, 1)
        self.SE =SE

    def forward(self, X, Y):

        TE_P = X[:,:,0,1:].to(torch.long)
        TE_Q = Y[:,:,0,1:].to(torch.long)
        X = X[..., :1]
        TE=torch.cat([TE_P,TE_Q],dim=1)
        X = self.fc2(F.relu(self.fc1(X)))
        STE = self.STEmb(self.SE, TE)
        STE_P = STE[:, : X.shape[1]]
        STE_Q = STE[:, X.shape[1] :]
        X = self.STAttBlockEnc(X, STE_P)
        X = self.transformAttention(X, STE_P, STE_Q)
        X = self.STAttBlockDec(X, STE_Q)
        X = self.fc27(F.relu(self.fc26(X)))
        return X


def mae_loss(pred, label, device):
    mask = (label != 0)
    mask = mask.type(torch.FloatTensor).to(device)
    mask /= torch.mean(mask)
    mask[mask!=mask] = 0
    loss = torch.abs(pred - label)
    loss *= mask
    loss[loss!=loss] = 0
    loss = torch.mean(loss)
    return loss
