import torch.nn as nn
import torch
from torchinfo import summary

class AttentionLayer_t(nn.Module):
    """Perform attention across the -2 dim (the -1 dim is `model_dim`).

    Make sure the tensor is permuted to correct shape before attention.

    E.g.
    - Input shape (batch_size, in_steps, num_nodes, model_dim).
    - Then the attention will be performed across the nodes.

    Also, it supports different src and tgt length.

    But must `src length == K length == V length`.

    """

    def __init__(self, model_dim, num_heads=8, mask=False):
        super().__init__()

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.mask = mask

        self.head_dim = model_dim // num_heads

        self.FC_Q = nn.Linear(model_dim, model_dim)
        self.FC_K = nn.Linear(model_dim, model_dim)
        self.FC_V = nn.Linear(model_dim, model_dim)

        self.out_proj = nn.Linear(model_dim, model_dim)

    def forward(self, query, key, value):
        # Q    (batch_size, ..., tgt_length, model_dim)
        # K, V (batch_size, ..., src_length, model_dim)
        batch_size = query.shape[0]
        tgt_length = query.shape[-2]
        src_length = key.shape[-2]

        query = self.FC_Q(query)
        key = self.FC_K(key)
        value = self.FC_V(value)

        # Qhead, Khead, Vhead (num_heads * batch_size, ..., length, head_dim)
        query = torch.cat(torch.split(query, self.head_dim, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.head_dim, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.head_dim, dim=-1), dim=0)

        key = key.transpose(
            -1, -2
        )  # (num_heads * batch_size, ..., head_dim, src_length)

        attn_score = (
            query @ key
        ) / self.head_dim**0.5  # (num_heads * batch_size, ..., tgt_length, src_length)

        if self.mask:
            mask = torch.ones(
                tgt_length, src_length, dtype=torch.bool, device=query.device
            ).tril()  # lower triangular part of the matrix
            attn_score.masked_fill_(~mask, -torch.inf)  # fill in-place

        attn_score = torch.softmax(attn_score, dim=-1)
        out = attn_score @ value  # (num_heads * batch_size, ..., tgt_length, head_dim)
        out = torch.cat(
            torch.split(out, batch_size, dim=0), dim=-1
        )  # (batch_size, ..., tgt_length, head_dim * num_heads = model_dim)

        out = self.out_proj(out)

        return out

class SelfAttentionLayer_t(nn.Module):
    def __init__(
        self, model_dim, feed_forward_dim=2048, num_heads=8, dropout=0, mask=False
    ):
        super().__init__()

        self.attn = AttentionLayer_t(model_dim, num_heads, mask)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, dim=-2):
        x = x.transpose(dim, -2)
        # x: (batch_size, ..., length, model_dim)
        residual = x
        out = self.attn(x, x, x)  # (batch_size, ..., length, model_dim)
        out = self.dropout1(out)
        out = self.ln1(residual + out)

        residual = out
        out = self.feed_forward(out)  # (batch_size, ..., length, model_dim)
        out = self.dropout2(out)
        out = self.ln2(residual + out)

        out = out.transpose(dim, -2)
        return out


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

        ones_source = torch.ones_like(flat_indices, device=x.device, dtype=dict_refined.dtype)  # .cuda()
        
        linear_indices = flat_indices + batch_indices1 * N # N = dict_refined.size(-1)

        dict_refined.put_(linear_indices, ones_source, accumulate=True)

        x1 = dict1 / dict_refined.unsqueeze(-1).expand(B, T, N, D)
        return x1


class AttentionLayer_s(nn.Module):
    """Perform attention across the -2 dim (the -1 dim is `model_dim`).

    Make sure the tensor is permuted to correct shape before attention.

    E.g.
    - Input shape (batch_size, in_steps, num_nodes, model_dim).
    - Then the attention will be performed across the nodes.

    Also, it supports different src and tgt length.

    But must `src length == K length == V length`.

    """

    def __init__(self, model_dim, num_heads=8,mask=False, use_subgraph=True,topk=100,memory_node=10):
        super().__init__()

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.mask = mask

        self.head_dim = model_dim // num_heads

        self.FC_Q = nn.Linear(model_dim, model_dim)
        self.FC_K = nn.Linear(model_dim, model_dim)
        self.FC_V = nn.Linear(model_dim, model_dim)

        self.out_proj = nn.Linear(model_dim, model_dim)

        self.topk = topk
        self.memory_node = memory_node
        self.use_subgraph = use_subgraph
        self.nodeselection = nodeselection(self.topk, self.memory_node, self.head_dim)
        self.fal = feature_aggregation(self.topk,self.memory_node)
    def forward(self, query, key, value):
        # Q    (batch_size, ..., tgt_length, model_dim)
        # K, V (batch_size, ..., src_length, model_dim)
        batch_size = query.shape[0]
        tgt_length = query.shape[-2]
        src_length = key.shape[-2]


        query = self.FC_Q(query)
        key = self.FC_K(key)
        value = self.FC_V(value)

        # x_g2 = self.fal(x, A, batch_indices, time_indices, indices)

        # Qhead, Khead, Vhead (num_heads * batch_size, ..., length, head_dim)
        query = torch.cat(torch.split(query, self.head_dim, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.head_dim, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.head_dim, dim=-1), dim=0)

        if self.use_subgraph :
            node_feature = [query, key]
            [selected_nodes_query, selected_nodes_key], batch_indices, time_indices, indices = self.nodeselection(
                node_feature)
            attn_score = (
                                 selected_nodes_query @ selected_nodes_key.transpose(-1, -2)
                         ) / self.head_dim ** 0.5  # (num_heads * batch_size, ..., tgt_length, src_length)

            attn_score1 = torch.softmax(attn_score, dim=-1)

            out = self.fal(value, attn_score1, batch_indices, time_indices, indices)
            out = torch.cat(
                torch.split(out, batch_size, dim=0), dim=-1
            )  # (batch_size, ..., tgt_length, head_dim * num_heads = model_dim)
        else:
            key = key.transpose(
                -1, -2
            )  # (num_heads * batch_size, ..., head_dim, src_length)

            attn_score = (
                                 query @ key
                         ) / self.head_dim ** 0.5  # (num_heads * batch_size, ..., tgt_length, src_length)

            attn_score = torch.softmax(attn_score, dim=-1)
            out = attn_score @ value  # (num_heads * batch_size, ..., tgt_length, head_dim)
            out = torch.cat(
                torch.split(out, batch_size, dim=0), dim=-1
            )  # (batch_size, ..., tgt_length, head_dim * num_heads = model_dim)




        out = self.out_proj(out)

        return out


class SelfAttentionLayer_s(nn.Module):
    def __init__(
        self, model_dim, feed_forward_dim=2048, num_heads=8, dropout=0, use_subgraph=True,topk=10,memory_node=100,mask=False
    ):
        super().__init__()

        self.attn = AttentionLayer_s(model_dim, num_heads, mask,use_subgraph,topk,memory_node)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, dim=-2):
        x = x.transpose(dim, -2)
        # x: (batch_size, ..., length, model_dim)
        residual = x
        out = self.attn(x, x, x)  # (batch_size, ..., length, model_dim)
        out = self.dropout1(out)
        out = self.ln1(residual + out)

        residual = out
        out = self.feed_forward(out)  # (batch_size, ..., length, model_dim)
        out = self.dropout2(out)
        out = self.ln2(residual + out)

        out = out.transpose(dim, -2)
        return out

class STAEformer(nn.Module):
    def __init__(
        self,
        num_nodes,
        in_steps=12,
        out_steps=12,
        steps_per_day=288,
        input_dim=3,
        output_dim=1,
        input_embedding_dim=24,
        tod_embedding_dim=24,
        dow_embedding_dim=24,
        topk=100,
        memory_node=10,
        use_subgraph=True,
        spatial_embedding_dim=0,
        adaptive_embedding_dim=80,
        feed_forward_dim=256,
        num_heads=4,
        num_layers=3,
        dropout=0.1,
        use_mixed_proj=True,
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.steps_per_day = steps_per_day
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_embedding_dim = input_embedding_dim
        self.tod_embedding_dim = tod_embedding_dim
        self.dow_embedding_dim = dow_embedding_dim
        self.spatial_embedding_dim = spatial_embedding_dim
        self.adaptive_embedding_dim = adaptive_embedding_dim
        self.model_dim = (
            input_embedding_dim
            + tod_embedding_dim
            + dow_embedding_dim
            + spatial_embedding_dim
            + adaptive_embedding_dim
        )
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.use_mixed_proj = use_mixed_proj

        self.input_proj = nn.Linear(input_dim, input_embedding_dim)
        if tod_embedding_dim > 0:
            self.tod_embedding = nn.Embedding(steps_per_day, tod_embedding_dim)
        if dow_embedding_dim > 0:
            self.dow_embedding = nn.Embedding(7, dow_embedding_dim)
        if spatial_embedding_dim > 0:
            self.node_emb = nn.Parameter(
                torch.empty(self.num_nodes, self.spatial_embedding_dim)
            )
            nn.init.xavier_uniform_(self.node_emb)
        if adaptive_embedding_dim > 0:
            self.adaptive_embedding = nn.init.xavier_uniform_(
                nn.Parameter(torch.empty(in_steps, num_nodes, adaptive_embedding_dim))
            )

        if use_mixed_proj:
            self.output_proj = nn.Linear(
                in_steps * self.model_dim, out_steps * output_dim
            )
        else:
            self.temporal_proj = nn.Linear(in_steps, out_steps)
            self.output_proj = nn.Linear(self.model_dim, self.output_dim)

        self.attn_layers_t = nn.ModuleList(
            [
                SelfAttentionLayer_t(self.model_dim, feed_forward_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )

        self.attn_layers_s = nn.ModuleList(
            [
                SelfAttentionLayer_s(self.model_dim, feed_forward_dim, num_heads, dropout,use_subgraph,topk,memory_node)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x):
        # x: (batch_size, in_steps, num_nodes, input_dim+tod+dow=3)
        batch_size = x.shape[0]

        if self.tod_embedding_dim > 0:
            tod = x[..., 1]
        if self.dow_embedding_dim > 0:
            dow = x[..., 2]
        x = x[..., : self.input_dim]

        x = self.input_proj(x)  # (batch_size, in_steps, num_nodes, input_embedding_dim)
        features = [x]
        if self.tod_embedding_dim > 0:
            tod_emb = self.tod_embedding(
                (tod * self.steps_per_day).long()
            )  # (batch_size, in_steps, num_nodes, tod_embedding_dim)
            features.append(tod_emb)
        if self.dow_embedding_dim > 0:
            dow_emb = self.dow_embedding(
                dow.long()
            )  # (batch_size, in_steps, num_nodes, dow_embedding_dim)
            features.append(dow_emb)
        if self.spatial_embedding_dim > 0:
            spatial_emb = self.node_emb.expand(
                batch_size, self.in_steps, *self.node_emb.shape
            )
            features.append(spatial_emb)
        if self.adaptive_embedding_dim > 0:
            adp_emb = self.adaptive_embedding.expand(
                size=(batch_size, *self.adaptive_embedding.shape)
            )
            features.append(adp_emb)
        x = torch.cat(features, dim=-1)  # (batch_size, in_steps, num_nodes, model_dim)

        for attn in self.attn_layers_t:
            x = attn(x, dim=1)
        for attn in self.attn_layers_s:
            x = attn(x, dim=2)
        # (batch_size, in_steps, num_nodes, model_dim)

        if self.use_mixed_proj:
            out = x.transpose(1, 2)  # (batch_size, num_nodes, in_steps, model_dim)
            out = out.reshape(
                batch_size, self.num_nodes, self.in_steps * self.model_dim
            )
            out = self.output_proj(out).view(
                batch_size, self.num_nodes, self.out_steps, self.output_dim
            )
            out = out.transpose(1, 2)  # (batch_size, out_steps, num_nodes, output_dim)
        else:
            out = x.transpose(1, 3)  # (batch_size, model_dim, num_nodes, in_steps)
            out = self.temporal_proj(
                out
            )  # (batch_size, model_dim, num_nodes, out_steps)
            out = self.output_proj(
                out.transpose(1, 3)
            )  # (batch_size, out_steps, num_nodes, output_dim)

        return out


if __name__ == "__main__":
    model = STAEformer(207, 12, 12)
    summary(model, [64, 12, 207, 3])
