import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.utils import expand_as_pair
import dgl.function as fn


class overAll(nn.Module):
    def __init__(self, node_size, node_hidden,
                 rel_size,
                 rel_matrix,
                 ent_matrix,
                 dropout_rate=0., depth=2,
                 device='cpu'
                 ):
        super(overAll, self).__init__()
        self.e_encoder = Dual(depth=depth, device=device)
        self.r_encoder = Dual(depth=depth, device=device)
        self.device = device
        self.ent_adj = self.get_spares_matrix_by_index(ent_matrix, (node_size, node_size), self.device)
        self.rel_adj = self.get_spares_matrix_by_index(rel_matrix, (node_size, rel_size), self.device)
        self.dropout_rate = dropout_rate
        self.depth = depth
        self.ent_emb = self.init_emb(node_size, node_hidden, init_func='uniform')
        self.rel_emb = self.init_emb(rel_size, node_hidden, init_func='uniform')

    @staticmethod
    def get_spares_matrix_by_index(index, size, device='cuda'):
        index = torch.LongTensor(index)
        adj = torch.sparse.FloatTensor(torch.transpose(index, 0, 1),
                                       torch.ones_like(index[:, 0], dtype=torch.float), size)
        # dim ??
        return torch.sparse.softmax(adj, dim=1).to(device)

    @staticmethod
    def init_emb(*size, init_func='xavier'):
        entities_emb = nn.Parameter(torch.randn(size))
        if init_func == 'xavier':
            torch.nn.init.xavier_normal_(entities_emb)
        elif init_func == 'zero':
            torch.nn.init.zeros_(entities_emb)
        elif init_func == 'uniform':
            torch.nn.init.uniform_(entities_emb, -.05, .05)
        else:
            raise NotImplementedError
        return entities_emb

    def forward(self, g, g_r):
        g = g.to(self.device)
        g_r = g_r.to(self.device)
        # inputs = [adj_matrix, r_index, r_val, rel_matrix, ent_matrix, train_pairs]
        ent_feature = torch.matmul(self.ent_adj, self.ent_emb)
        rel_feature = torch.matmul(self.rel_adj, self.rel_emb)

        out_feature_ent = self.e_encoder(g, g_r, ent_feature, self.rel_emb)
        out_feature_rel = self.r_encoder(g, g_r, rel_feature, self.rel_emb)
        out_feature = torch.cat((out_feature_ent, out_feature_rel), dim=-1)
        out_feature = F.dropout(out_feature, p=self.dropout_rate, training=self.training)
        return out_feature


class Dual(nn.Module):
    def __init__(self,
                 use_bias=True,
                 depth=1,
                 activation=torch.tanh,
                 device='cpu'
                 ):
        super(Dual, self).__init__()

        self.activation = activation
        self.depth = depth
        self.use_bias = use_bias
        self.attn_kernels = []

        # try
        node_F = 128
        rel_F = 128
        self.ent_F = node_F
        ent_F = self.ent_F
        self.gate_kernel = overAll.init_emb(ent_F * (self.depth + 1), ent_F * (self.depth + 1))
        self.proxy = overAll.init_emb(64, node_F * (self.depth + 1))
        if self.use_bias:
            self.bias = overAll.init_emb(1, ent_F * (self.depth + 1), init_func='zero')
        for d in range(self.depth):
            attn_kernel = overAll.init_emb(node_F, 1)
            self.attn_kernels.append(attn_kernel.to(device))

    def forward(self, g: dgl.graph, g_r: dgl.heterograph, features, rel_emb):
        outputs = []
        features = self.activation(features)
        outputs.append(features)
        for l in range(self.depth):
            attention_kernel = self.attn_kernels[l]
            g_r.nodes['relation'].data['features'] = rel_emb
            g_r['in'].update_all(fn.copy_u('features', 'm'), fn.mean('m', 'emb'), etype='in')
            rels_sum = F.normalize(g_r.nodes['index'].data['emb'], p=2, dim=1)
            g.ndata['features'] = features
            g.apply_edges(fn.copy_u('features', 'neighs'))
            neighs = g.edata['neighs']
            neighs = neighs - 2 * torch.sum(neighs * rels_sum, 1, keepdim=True) * rels_sum
            att1 = torch.squeeze(torch.matmul(rels_sum, attention_kernel), dim=-1)

            att = torch.sparse.FloatTensor(torch.vstack(g.edges()), att1, (g.num_nodes(), g.num_nodes()))
            att = torch.sparse.softmax(att, dim=0)
            new_feature = neighs * torch.unsqueeze(att._values(), dim=-1)
            g.edata['feat'] = new_feature
            g.update_all(fn.copy_e('feat', 'm'), fn.sum('m', 'layer' + str(l)))
            features = g.ndata['layer' + str(l)]
            features = self.activation(features)
            outputs.append(features)

        outputs = torch.cat(outputs, dim=1)
        proxy_att = torch.matmul(F.normalize(outputs, dim=-1),
                                 torch.transpose(F.normalize(self.proxy, dim=-1), 0, 1))
        proxy_att = F.softmax(proxy_att, dim=-1)  # eq.3
        proxy_feature = outputs - torch.matmul(proxy_att, self.proxy)

        if self.use_bias:
            gate_rate = F.sigmoid(torch.matmul(proxy_feature, self.gate_kernel) + self.bias)
        else:
            gate_rate = F.sigmoid(torch.matmul(proxy_feature, self.gate_kernel))
        outputs = gate_rate * outputs + (1 - gate_rate) * proxy_feature
        return outputs
