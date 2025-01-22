import torch
import torch.nn.functional as F
import torch.nn as nn

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
