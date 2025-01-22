import torch
import torch.nn.functional as F
import torch.nn as nn

class nodeselection(nn.Module):
    def __init__(self, topk, memory_node, time_dim):
        super(nodeselection, self).__init__()
        self.K= topk
        self.node_embeddings = nn.Parameter(torch.randn(memory_node, time_dim), requires_grad=True)

    def forward(self, node_feature,node_embeddings=None):
        nodevec1 = node_feature

        nodevec2 = self.node_embeddings

        supports2 = torch.softmax(torch.matmul(nodevec2, nodevec1.transpose(-2, -1)), dim=-1)

        values, indices = supports2.topk(self.K, dim=-1, largest=True, sorted=True)

        batch_indices = torch.arange(nodevec1.size(0),device=nodevec1.device).unsqueeze(-1).unsqueeze(-1).expand(-1, indices.size(1),
                                                                                   indices.size(2))#.cuda()

        selected_nodes_features = nodevec1[batch_indices, indices]

        return selected_nodes_features, batch_indices, indices