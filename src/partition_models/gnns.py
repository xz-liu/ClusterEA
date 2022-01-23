import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import EAData
import dgl.data
from utils import ConstructGraph
import dgl.nn


class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN, self).__init__()
        self.conv1 = dgl.nn.GraphConv(in_feats, h_feats, norm='none', weight=True, bias=True)
        self.conv2 = dgl.nn.GraphConv(h_feats, num_classes, norm='none', weight=True, bias=True)

    def forward(self, g, in_feat):
        # print(in_feat.requires_grad)
        edge_weight = g.edata.get('weight')
        h = self.conv1(g, in_feat, edge_weight=edge_weight)
        # print(h.requires_grad)
        h = F.relu(h)
        # print(h.requires_grad)
        h = self.conv2(g, h, edge_weight=edge_weight)
        # print(h.requires_grad)
        return h


import torch
import torch.nn as nn
import dgl.function as fn
from dgl.nn import GATConv, GINConv


class GAT(nn.Module):
    def __init__(self,
                 num_layers=1,
                 in_dim=600,
                 num_hidden=8,
                 num_classes=10,
                 num_heads=8,
                 num_out_heads=1,
                 activation=F.elu,
                 feat_drop=0,
                 attn_drop=0,
                 negative_slope=0.2,
                 residual=False):
        super(GAT, self).__init__()
        heads = ([num_heads] * num_layers) + [num_out_heads]

        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        # input projection (no residual)
        self.gat_layers.append(GATConv(
            in_dim, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GATConv(
                num_hidden * heads[l - 1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation))
        # output projection
        self.gat_layers.append(GATConv(
            num_hidden * heads[-2], num_classes, heads[-1],
            feat_drop, attn_drop, negative_slope, residual, None))

    def forward(self, g, inputs):
        h = inputs
        for l in range(self.num_layers):
            h = self.gat_layers[l](g, h).flatten(1)
        # output projection
        logits = self.gat_layers[-1](g, h).mean(1)
        return logits


class GIN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GIN, self).__init__()
        self.linear1 = nn.Linear(in_feats, h_feats)
        self.linear2 = nn.Linear(h_feats, num_classes)
        self.conv1 = dgl.nn.GINConv(self.linear1, 'max')
        self.conv2 = dgl.nn.GINConv(self.linear2, 'max')

    def forward(self, g, in_feat):
        # print(in_feat.requires_grad)
        edge_weight = g.edata.get('weight')
        h = self.conv1(g, in_feat, edge_weight=edge_weight)
        # print(h.requires_grad)
        h = F.relu(h)
        # print(h.requires_grad)
        h = self.conv2(g, h, edge_weight=edge_weight)
        # print(h.requires_grad)
        return h


class DummyModule(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p


class NodeClassification:
    def __init__(self, model_name, data: EAData, triples, feature_dim=600, hidden_dim=16, k=10, device='cuda',
                 construct_which=(0, 1), max_iter=500):
        if model_name == 'gcn':
            self.model = GCN(feature_dim, hidden_dim, k)
        elif model_name == 'gat':
            self.model = GAT(in_dim=feature_dim, num_hidden=hidden_dim, num_classes=k)
        elif model_name == 'gin':
            self.model = GIN(feature_dim, hidden_dim, k)
        else:
            raise NotImplementedError
        self.g = self._construct_graph(data, triples, construct_which)
        self.device = device
        self.max_iter = max_iter

    def _construct_graph(self, data: EAData, triples, construct_which):
        g = {}
        for w in construct_which:
            triple = triples[w]
            num_ents = len(data.ents[w])
            g[w] = ConstructGraph(triple, num_ents)

        return g

    @torch.enable_grad()
    def fit(self, features, idx, labels, which=0):
        g = self.g[which]
        device = self.device
        self.model.to(device)
        g = g.to(device)
        features = features.clone().detach().to(device)
        # model = nn.ModuleList([DummyModule(nn.Parameter(features, requires_grad=True)), self.model])
        model = self.model
        labels = torch.from_numpy(labels).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        g.ndata['feature'] = features

        for e in range(self.max_iter):
            # Forward
            logits = self.model(g, features)
            # Compute loss
            # Note that you should only compute the losses of the nodes in the training set.
            loss = F.cross_entropy(logits[idx], labels)
            # from torchviz import make_dot
            # make_dot(loss).render("attached", format="png")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if e % 5 == 0:
                print('In epoch {}, loss: {:.3f}'.format(
                    e, loss))

    def predict(self, features, which):
        device = self.device
        g = self.g[which]
        self.model.to(device)
        g = g.to(device)
        features = features.detach().to(device)
        logits = self.model(g, features)
        pred = logits.argmax(1)
        return pred.cpu().numpy()

    def clear_memory(self):
        del self.model
        del self.g
        import gc
        gc.collect()
