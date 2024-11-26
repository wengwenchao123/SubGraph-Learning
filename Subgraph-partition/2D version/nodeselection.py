import torch
import torch.nn.functional as F
import torch.nn as nn

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