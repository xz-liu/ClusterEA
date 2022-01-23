import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.dataloading import MultiLayerFullNeighborSampler
from dgl.utils import expand_as_pair
import dgl.function as fn


class overAllRREA(nn.Module):
    def __init__(self, node_size, node_hidden,
                 rel_size,
                 g,
                 g_r,
                 n_r,
                 dropout_rate=0., depth=2,
                 device='cpu'
                 ):
        super(overAllRREA, self).__init__()
        self.e_encoder = RREA(depth=depth, device=device, node_hidden=node_hidden)
        self.r_encoder = RREA(depth=depth, device=device, node_hidden=node_hidden)
        self.device = device
        self.dropout_rate = dropout_rate
        self.depth = depth
        self.ent_emb = self.init_emb(node_size, node_hidden, init_func='uniform')
        self.rel_emb = self.init_emb(rel_size, node_hidden, init_func='uniform')

        # new adding
        self.g = g
        self.g_r = g_r
        self.n_r = n_r
        self.g_r = self.g_r.to(self.device)
        self.n_r = self.n_r.to(self.device)

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
        n_r.nodes['relation'].data['rel_emb'] = self.rel_emb
        r_temp_block = MultiLayerFullNeighborSampler(1).sample_blocks(n_r, {'entity': src_nodes.type(torch.int32)})[0]
        r_temp_block.update_all(fn.copy_u('rel_emb', 'm'), fn.mean('m', 'r_neigh'), etype='link')
        rel_feature = r_temp_block.dstnodes['entity'].data['r_neigh']

        out_feature_ent = self.e_encoder(blocks, self.g_r, ent_feature, self.rel_emb)
        out_feature_rel = self.r_encoder(blocks, self.g_r, rel_feature, self.rel_emb)
        out_feature = torch.cat((out_feature_ent, out_feature_rel), dim=-1)

        out_feature = F.dropout(out_feature, p=self.dropout_rate, training=self.training)
        return out_feature


class RREA(nn.Module):
    def __init__(self,
                 use_bias=True,
                 depth=1,
                 activation=torch.tanh,
                 device='cpu', node_hidden=100,
                 ):
        super(RREA, self).__init__()

        self.activation = activation
        self.depth = depth
        self.use_bias = use_bias
        self.attn_kernels = []

        dim = node_hidden

        node_F = dim
        rel_F = dim
        self.ent_F = node_F
        ent_F = self.ent_F

        for d in range(self.depth):
            attn_kernel = overAllRREA.init_emb(3 * node_F, 1)
            self.attn_kernels.append(attn_kernel.to(device))

    def forward(self, blocks, g_r: dgl.heterograph, features, rel_emb):
        outputs = []
        features = self.activation(features)
        # append dst feature
        dst_nodes = blocks[-1].dstdata[dgl.NID]
        outputs.append(features[:len(dst_nodes)])
        for l in range(self.depth):
            attention_kernel = self.attn_kernels[l]

            # compute rels_sum
            eid = blocks[l].edata[dgl.EID]
            g_r.nodes['relation'].data['features'] = rel_emb
            temp_block = MultiLayerFullNeighborSampler(1).sample_blocks(g_r, {'index': eid})[0]
            temp_block.update_all(fn.copy_u('features', 'm'), fn.mean('m', 'emb'), etype='in')
            rels_sum = F.normalize(temp_block.dstnodes['index'].data['emb'], p=2, dim=1)
            blocks[l].srcdata['features'] = features
            blocks[l].apply_edges(fn.copy_u('features', 'neighs'))
            neighs = blocks[l].edata['neighs']

            # add self
            src, trg = blocks[l].edges()
            selfs = features[trg]
            # neighs = features[src]

            # different from dual
            neighs = neighs - 2 * torch.sum(neighs * rels_sum, 1, keepdim=True) * rels_sum
            new_conbine = torch.cat([selfs, neighs, rels_sum], dim=1)
            att1 = torch.squeeze(torch.matmul(new_conbine, attention_kernel), dim=-1)

            from dgl.nn.functional import edge_softmax
            att = edge_softmax(blocks[l], att1.flatten(), norm_by='dst')
            new_feature = neighs * torch.unsqueeze(att, dim=-1)
            blocks[l].edata['feat'] = new_feature
            blocks[l].update_all(fn.copy_e('feat', 'm'), fn.sum('m', 'layer' + str(l)))
            features = blocks[l].dstdata['layer' + str(l)]

            features = self.activation(features)
            dst_nodes = blocks[-1].dstdata[dgl.NID]
            outputs.append(features[:len(dst_nodes)])  # only append dst node

        outputs = torch.cat(outputs, dim=1)
        return outputs
