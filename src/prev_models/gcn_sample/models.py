import torch.nn as nn
import torch.nn.functional as F
import dgl.nn
from dataset import *
# import time

class GCN(nn.Module):
    def __init__(self, in_feats, out_feats, middle=200, device='cuda', first_layer_weight=False):
        super(GCN, self).__init__()
        self.in_dim = in_feats
        self.conv1 = dgl.nn.GraphConv(in_feats, middle, norm='none', weight=first_layer_weight, bias=True)
        self.conv2 = dgl.nn.GraphConv(middle, out_feats, norm='none', weight=True, bias=True)
        #
        # self.edge_weight_norm = dgl.nn.EdgeWeightNorm(norm='both')
        # self.conv2 = dgl.nn.GraphConv(in_feats, out_feats)
        self.out_feats = out_feats
        self.device = device
        self._init_weight(first_layer_weight)

    def _init_weight(self, first_layer_weight):
        from torch.nn.init import xavier_normal_, zeros_
        if first_layer_weight:
            xavier_normal_(self.conv1.weight)
        zeros_(self.conv1.bias)
        xavier_normal_(self.conv2.weight)
        zeros_(self.conv2.bias)

    def forward(self, blocks):
        blocks = [blocks[0].to(self.device), blocks[1].to(self.device)]
        in_feats = blocks[0].srcdata['feature']
        edge_weight = blocks[0].edata['weight']
        # if edge_weight is not None:
        #     edge_weight = self.edge_weight_norm(g, edge_weight)
        h = self.conv1(blocks[0], in_feats, edge_weight=edge_weight)
        h = F.relu(h)

        edge_weight_2 = blocks[1].edata['weight']
        # print(h.size(), in_feats.size())
        h = self.conv2(blocks[1], h, edge_weight=edge_weight_2)
        return h


def get_output(model, graph, seed, fanouts):
    from dgl.dataloading import MultiLayerNeighborSampler
    time1 = time.time()
    blocks = MultiLayerNeighborSampler(fanouts).sample_blocks(graph, torch.from_numpy(seed))
    time2 = time.time()
    blocks = [block.to(model.device) for block in blocks]
    output_embedding = model(blocks)
    time3 = time.time()
    return output_embedding, time2 - time1, time3 - time2


class LargeGCNFramework(nn.Module):
    def __init__(self, triple1, triple2, ent_sizes, model, device='cuda', embeddings=None,
                 fixed_embedding=False):
        super(LargeGCNFramework, self).__init__()
        self.model = model
        if model is not None:
            self.in_dim = model.in_dim
        elif embeddings is not None:
            self.in_dim = embeddings[0].size(1)
        else:
            raise NotImplementedError
        if embeddings is None:
            self.embedding_1 = self._init_entity_emb(ent_sizes[0], fixed_embedding)
            self.embedding_2 = self._init_entity_emb(ent_sizes[1], fixed_embedding)
        else:
            self.embedding_1 = nn.Parameter(torch.tensor(embeddings[0]), requires_grad=not fixed_embedding)
            self.embedding_2 = nn.Parameter(torch.tensor(embeddings[1]), requires_grad=not fixed_embedding)
        g1 = ConstructGraph(triple1, ent_sizes[0])
        g2 = ConstructGraph(triple2, ent_sizes[1])
        g1.ndata['feature'] = self.embedding_1
        g2.ndata['feature'] = self.embedding_2

        self.g1 = g1
        self.g2 = g2
        self.device = device
        self.ent_sizes = ent_sizes

    def _init_entity_emb(self, num, fixed_embedding=False):
        w = torch.empty(num, self.in_dim)
        torch.nn.init.xavier_normal_(w)
        # truncated_normal_(w)
        entities_emb = nn.Parameter(w, requires_grad=not fixed_embedding)
        return entities_emb

    @torch.no_grad()
    def _get_initial_embedding(self):
        return self.embedding_1, self.embedding_2

    @torch.no_grad()
    def _get_final_embedding(self):
        if self.model is None:
            return self._get_initial_embedding()
        em1 = self.model([self.g1.to(self.device)] * 2)
        em2 = self.model([self.g2.to(self.device)] * 2)
        return em1, em2

    @staticmethod
    def _merge_links(subsample):
        ranges = [[0, len(x)] for x in subsample]
        for i in range(len(ranges)):
            if i > 0:
                ranges[i][0] += ranges[i - 1][1]
                ranges[i][1] += ranges[i][0]
        merged_nodes = np.concatenate(subsample, axis=0)
        return merged_nodes, ranges

    @staticmethod
    def _split_links(merged_embedding, ranges, k, random_resample=False):
        if random_resample:
            len_curr = ranges[k][1] - ranges[k][0]
            idx = torch.randint(merged_embedding.size(0), (len_curr,))
            return merged_embedding[idx]
        else:
            return merged_embedding[ranges[k][0]: ranges[k][1]]

    def forward(self, link, neg1_total, neg2_total, fanout=(100, 100), margin=3, random_resample=True):
        seed_1 = link.T[0]
        seed_2 = link.T[1]
        self.model.to(self.device)
        merged1, r1 = self._merge_links([seed_1] + neg1_total)
        merged2, r2 = self._merge_links([seed_2] + neg2_total)
        emb1, left_sample_time, left_forward_time = get_output(self.model, self.g1, merged1, fanout)
        emb2, right_sample_time, right_forward_time = get_output(self.model, self.g2, merged2, fanout)
        # emb1 = norm_process(emb1)
        # emb2 = norm_process(emb2)
        sample_time = left_sample_time + right_sample_time
        forward_time = left_forward_time + right_forward_time
        pos1, pos2 = self._split_links(emb1, r1, 0, False), self._split_links(emb2, r2, 0, False)
        from ..duala_sample.wrapper import batch_align_loss
        embeddings = torch.vstack([pos1, pos2, emb1[pos1.size(0):], emb2[pos2.size(0):]])

        loss = batch_align_loss(pos1.size(0), embeddings.size(0) - 2 * pos1.size(0), embeddings, device=self.device)
        return loss, sample_time, forward_time


if __name__ == '__main__':
    model = GCN(100, 100)

#  python main.py --phase 0 --dataset small --lang fr --save_prefix gcn_align_nosplit --model gcn-align --src_split 1 --trg_split 1
#  python main.py --phase 4 --dataset small --lang fr --save_prefix gcn_align_nosplit  --eval_which s
