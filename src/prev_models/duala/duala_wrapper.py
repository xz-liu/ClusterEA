from .duala import *
import dgl
import numpy as np
import torch
import time
from utils_largeea import *
from .loss import align_loss
from .data_util import *
from tqdm import *


def get_embedding(index_a, index_b, vec):
    vec = vec.detach().numpy()
    Lvec = np.array([vec[e] for e in index_a])
    Rvec = np.array([vec[e] for e in index_b])
    Lvec = Lvec / (np.linalg.norm(Lvec, axis=-1, keepdims=True) + 1e-5)
    Rvec = Rvec / (np.linalg.norm(Rvec, axis=-1, keepdims=True) + 1e-5)
    return Lvec, Rvec


def constructRelGraph(r_index):
    src, trg = [], []
    for index in r_index:
        src.append(index[1])
        trg.append(index[0])
    return dgl.heterograph({
        ('relation', 'in', 'index'): (torch.tensor(src), torch.tensor(trg)), })


def constructGraph(adj_matrix):
    src, trg = adj_matrix[:, 0], adj_matrix[:, 1]
    # todo src trg
    g = dgl.graph((src, trg))
    # g = dgl.graph(( trg,src))
    return g


class DualAMNWrapper:
    def __init__(self, *args, ent_sizes=None, rel_sizes=None, triples=None, **kwargs):

        entsz = ent_sizes[0] + ent_sizes[1]
        relsz = rel_sizes[0] + rel_sizes[1]
        print(len(triples), entsz, relsz)
        adj_matrix, r_index, r_val, adj_features, rel_features = get_matrix(triples, entsz, relsz)

        # train_pair, dev_pair, adj_matrix, r_index, r_val, adj_features, rel_features = get_matrix(kwargs.get())
        adj_matrix = np.stack(adj_matrix.nonzero(), axis=1)
        rel_matrix, rel_val = np.stack(rel_features.nonzero(), axis=1), rel_features.data
        ent_matrix, ent_val = np.stack(adj_features.nonzero(), axis=1), adj_features.data

        node_size = adj_features.shape[0]
        rel_size = rel_features.shape[1]
        triple_size = len(adj_matrix)  # not triple size, but number of diff(h, t)
        node_hidden = 128
        rel_hidden = 128
        batch_size = 1024
        dropout_rate = 0.3
        lr = 0.005
        gamma = 1
        depth = 2
        device = 'cuda'
        # print('adj_matrix', adj_matrix)
        # print('rel_matrix', r_index)

        g = constructGraph(adj_matrix)
        g_r = constructRelGraph(r_index)

        print('begin')
        # inputs = [adj_input, index_input, val_input, rel_adj, ent_adj]

        model = overAll(node_size=node_size, node_hidden=node_hidden,
                        rel_size=rel_size, rel_matrix=rel_matrix,
                        ent_matrix=ent_matrix, dropout_rate=dropout_rate,
                        depth=depth, device=device)
        model = model.to(device)
        # opt = torch.optim.RMSprop(model.parameters(), lr=lr)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        print('model constructed')

        self.opt = opt
        self.model = model
        self.g = g
        self.g_r = g_r
        self.train_pair = []
        self.batch_size = batch_size
        self.gamma = gamma
        self.node_size = node_size
        self.device = device
        self.ent_sizes = ent_sizes
        # np.random.shuffle(rest_set_1)
        # np.random.shuffle(rest_set_2)

        # here is dual-m

    @torch.enable_grad()
    def train1step(self, epoch):
        print(len(self.dev_pair), len(self.train_pair))
        self.model.train()
        train_pair = self.train_pair
        for i in trange(epoch):
            batch_size = self.batch_size
            np.random.shuffle(train_pair)
            for pairs in [train_pair[i * batch_size:(i + 1) * batch_size] for i in
                          range(len(train_pair) // batch_size + 1)]:
                # inputs = [adj_matrix, r_index, r_val, rel_matrix, ent_matrix]
                output = self.model(self.g, self.g_r)
                # print(output)
                loss = align_loss(pairs, output, self.gamma, self.node_size, self.device)
                print(loss)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
            if i == epoch - 1:
                self.output = self.get_embedding()

        self.CSLS_test()

        from evaluation import get_hits

    def CSLS_test(self, thread_number=16, csls=10, accurate=True):
        if len(self.dev_pair) == 0:
            print('EVAL--No dev')
            return
        # vec = self.get_embedding()
        vec = self.output
        Lvec = np.array([vec[e1] for e1, e2 in self.dev_pair])
        Rvec = np.array([vec[e2] for e1, e2 in self.dev_pair])
        Lvec = Lvec / np.linalg.norm(Lvec, axis=-1, keepdims=True)
        Rvec = Rvec / np.linalg.norm(Rvec, axis=-1, keepdims=True)
        from ..rrea.CSLS import eval_alignment_by_sim_mat
        eval_alignment_by_sim_mat(Lvec, Rvec, [1, 5, 10], thread_number, csls=csls, accurate=accurate)
        return None

    def convert_ent_id(self, ent_ids, which=0):
        return ent_ids

    def convert_rel_id(self, rel_ids, which=0):
        raise NotImplementedError()

    def append_pairs(self, old_pair, new_pair):
        if len(old_pair) == 0:
            return new_pair
        px, py = set(), set()
        for e1, e2 in old_pair:
            px.add(e1)
            py.add(e2)
        filtered = []
        added = 0
        for e1, e2 in new_pair:
            if e1 not in px and e2 not in py:
                filtered.append([e1, e2])
                added += 1
        print('Update pairs:{} newly added'.format(added))
        if len(filtered) == 0:
            return old_pair
        filtered = np.array(filtered)
        return np.concatenate([filtered, old_pair], axis=0)

    def update_trainset(self, pairs, append=True):
        trainset = [self.convert_ent_id(p, i) for i, p in enumerate(pairs)]

        curr_pair = np.array(trainset).T
        if append:
            if append == 'REVERSE':
                self.train_pair = self.append_pairs(curr_pair, self.train_pair)
            else:
                self.train_pair = self.append_pairs(self.train_pair, curr_pair)
        else:
            self.train_pair = curr_pair
        # print('srs-iteration-update-train-pair')

    def update_devset(self, pairs):
        # pairs = [pairs[0], pairs[1]]
        devset = [self.convert_ent_id(p, i) for i, p in enumerate(pairs)]

        self.dev_pair = np.array(devset).T
        self.dev_pair = np.array(
            list(set([(e1, e2) for e1, e2 in self.dev_pair]) - set([(e1, e2) for e1, e2 in self.train_pair])))
        rest_set_1 = [e1 for e1, e2 in self.dev_pair]
        rest_set_2 = [e2 for e1, e2 in self.dev_pair]
        self.rs = [rest_set_1, rest_set_2]

    def get_embedding(self):
        with torch.no_grad():
            self.model.eval()
            vec = self.model(self.g, self.g_r).to('cpu')
            self.model.train()
            return vec.detach().numpy()

    def get_curr_embeddings(self, device=None):
        device = device or 'cpu'
        with torch.no_grad():
            # self.model.eval()
            # vec = self.model(self.g, self.g_r).to(device)
            vec = torch.from_numpy(self.output)
            sep = self.ent_sizes[0]
            Lvec, Rvec = vec[:sep], vec[sep:]

            Lvec = Lvec / (torch.linalg.norm(Lvec, axis=-1, keepdims=True) + 1e-5)
            Rvec = Rvec / (torch.linalg.norm(Rvec, axis=-1, keepdims=True) + 1e-5)
            return Lvec, Rvec

    def mraea_iteration(self):
        print('mraea-iteration-update-train-pair')
        vec = self.output
        # np.random.shuffle(rest_set_1)
        # rest_set_1 = [e1 for e1, e2 in self.dev_pair]
        # rest_set_2 = [e2 for e1, e2 in self.dev_pair]
        # # np.random.shuffle(rest_set_2)
        # for e1, e2 in self.train_pair:
        #     e1, e2 = int(e1), int(e2)
        #     if e1 in rest_set_1:
        #         rest_set_1.remove(e1)
        #     if e2 in rest_set_2:
        #         rest_set_2.remove(e2)
        rest_set_1, rest_set_2 = self.rs
        new_pair = []
        Lvec = np.array([vec[e] for e in rest_set_1])
        Rvec = np.array([vec[e] for e in rest_set_2])
        Lvec = Lvec / np.linalg.norm(Lvec, axis=-1, keepdims=True)
        Rvec = Rvec / np.linalg.norm(Rvec, axis=-1, keepdims=True)
        from ..rrea.CSLS import eval_alignment_by_sim_mat
        A, _ = eval_alignment_by_sim_mat(Lvec, Rvec, [1, 5, 10], 16, 10, True, False)
        B, _ = eval_alignment_by_sim_mat(Rvec, Lvec, [1, 5, 10], 16, 10, True, False)
        A = sorted(list(A))
        B = sorted(list(B))

        for a, b in A:
            if B[b][1] == a:
                new_pair.append([rest_set_1[a], rest_set_2[b]])
        print("generate new semi-pairs: %d." % len(new_pair))

        for e1, e2 in new_pair:
            if e1 in rest_set_1:
                rest_set_1.remove(e1)

        for e1, e2 in new_pair:
            if e2 in rest_set_2:
                rest_set_2.remove(e2)

        self.train_pair = np.concatenate([self.train_pair, np.array(new_pair)], axis=0)

        self.rs = [rest_set_1, rest_set_2]
    #
    # def mraea_iteration(self):
    #     Lvec, Rvec = get_embedding(rest_set_1, rest_set_2, output.cpu())
    #     A, B = evaluater.CSLS_cal(Lvec, Rvec, False)
    #     for i, j in enumerate(A):
    #         if B[j] == i:
    #             new_pair.append([rest_set_1[j], rest_set_2[i]])
    #
    #     train_pair = np.concatenate([train_pair, np.array(new_pair)], axis=0)
    #     for e1, e2 in new_pair:
    #         if e1 in rest_set_1:
    #             rest_set_1.remove(e1)
    #
    #     for e1, e2 in new_pair:
    #         if e2 in rest_set_2:
    #             rest_set_2.remove(e2)
    #     epoch = 5
