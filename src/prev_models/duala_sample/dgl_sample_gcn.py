import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.dataloading import MultiLayerFullNeighborSampler
from dgl.utils import expand_as_pair
import dgl.function as fn


class overAll(nn.Module):
    def __init__(self, node_size, node_hidden,
                 rel_size,
                 g,
                 g_r,
                 n_r,
                 dropout_rate=0., depth=2,
                 device='cpu'
                 ):
        super(overAll, self).__init__()
        self.e_encoder = Dual(depth=depth, device=device)
        self.r_encoder = Dual(depth=depth, device=device)
        self.device = device
        self.dropout_rate = dropout_rate
        self.depth = depth
        self.ent_emb = self.init_emb(node_size, node_hidden, init_func='uniform')
        self.rel_emb = self.init_emb(rel_size, node_hidden, init_func='uniform')

        # new adding
        self.g = g
        self.g_r = g_r
        self.n_r = n_r

    @staticmethod
    def init_emb(*size, init_func='xavier'):
        # TODO BIAS
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

    def forward(self, blocks):
        #
        src_nodes = blocks[0].srcdata[dgl.NID]
        dst_nodes = blocks[-1].dstdata[dgl.NID]
        # torch.equal(src_nodes[:len(dst_nodes)], dst_nodes
        g = self.g
        # g.ndata['ent_emb'] = self.ent_emb
        temp_block = MultiLayerFullNeighborSampler(1).sample_blocks(g, src_nodes)[0]
        temp_block = temp_block.to(self.device)
        temp_block.srcdata['ent_emb'] = self.ent_emb[temp_block.srcdata[dgl.NID]]
        temp_block.update_all(fn.copy_u('ent_emb', 'm'), fn.mean('m', 'feature'))
        ent_feature = temp_block.dstdata['feature']

        n_r = self.n_r
        r_temp_block = MultiLayerFullNeighborSampler(1).sample_blocks(n_r, {'entity': src_nodes.type(torch.int32)})[0]
        r_temp_block = r_temp_block.to(self.device)
        r_temp_block.srcnodes['relation'].data['rel_emb'] = self.rel_emb[
            r_temp_block.srcdata[dgl.NID]['relation'].to(dtype=torch.long)]
        # n_r.nodes['relation'].data['rel_emb'] = self.rel_emb
        r_temp_block.update_all(fn.copy_u('rel_emb', 'm'), fn.mean('m', 'r_neigh'), etype='link')
        rel_feature = r_temp_block.dstnodes['entity'].data['r_neigh']

        out_feature_ent = self.e_encoder(blocks, self.g_r, ent_feature, self.rel_emb)
        out_feature_rel = self.r_encoder(blocks, self.g_r, rel_feature, self.rel_emb)
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

    def forward(self, blocks, g_r: dgl.heterograph, features, rel_emb):
        outputs = []
        features = self.activation(features)
        # append dst feature
        dst_nodes = blocks[-1].dstdata[dgl.NID]
        outputs.append(features[:len(dst_nodes)])
        for l in range(self.depth):
            attention_kernel = self.attn_kernels[l]

            # 这里应该选择和block匹配的边的信息，进行聚合
            eid = blocks[l].edata[dgl.EID]

            # g_r.nodes['relation'].data['features'] = rel_emb
            temp_block = MultiLayerFullNeighborSampler(1).sample_blocks(g_r, {'index': eid})[0]
            temp_block = temp_block.to(rel_emb.device)
            temp_block.srcnodes['relation'].data['features'] = rel_emb[
                temp_block.srcdata[dgl.NID]['relation'].to(dtype=torch.long)]
            temp_block.update_all(fn.copy_u('features', 'm'), fn.mean('m', 'emb'), etype='in')
            rels_sum = F.normalize(temp_block.dstnodes['index'].data['emb'], p=2, dim=1)
            blocks[l].srcdata['features'] = features
            blocks[l].apply_edges(fn.copy_u('features', 'neighs'))
            neighs = blocks[l].edata['neighs']

            # neight的维度是边的数量
            neighs = neighs - 2 * torch.sum(neighs * rels_sum, 1, keepdim=True) * rels_sum
            att1 = torch.squeeze(torch.matmul(rels_sum, attention_kernel), dim=-1)
            # print('att_len', len(att1))
            # print('block num edge',len(torch.vstack(blocks[l].edges())[0]))
            # print('block edge',torch.vstack(blocks[l].edges()) )

            # 这里有问题
            #
            # blocks[l].edata['att'] = att1
            # blocks[l].apply_edges(fn.ed)
            # att = torch.sparse.FloatTensor(torch.vstack(blocks[l].edges()), att1,
            #                                (blocks[l].number_of_src_nodes(), blocks[l].number_of_dst_nodes()))
            from dgl.nn.functional import edge_softmax
            att = edge_softmax(blocks[l], att1.flatten(), norm_by='dst')
            # from util2 import sparse_softmax
            # att = sparse_softmax(att,1)
            # att = torch.sparse.softmax(att, dim=0)

            # new_feature = neighs * torch.unsqueeze(att._values(), dim=-1)
            new_feature = neighs * torch.unsqueeze(att, dim=-1)
            blocks[l].edata['feat'] = new_feature

            blocks[l].update_all(fn.copy_e('feat', 'm'), fn.sum('m', 'layer' + str(l)))
            features = blocks[l].dstdata['layer' + str(l)]

            features = self.activation(features)
            # 这里依然要注意append的东西
            dst_nodes = blocks[-1].dstdata[dgl.NID]
            outputs.append(features[:len(dst_nodes)])

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
