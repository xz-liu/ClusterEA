import torch
import torch.nn as nn
from nxmetis import metis
from .kmeans import KMeans
from metis import Partition
from sklearn.neural_network import MLPClassifier
from dataset import *
from torch import Tensor
import numpy as np
import nxmetis

from .gnns import NodeClassification
from .sklearn_models import SKLearnPartition


class PartitionTrainer():
    def __init__(self, data: EAData, partition_k, pt=None):
        self.data = data
        self.metis = pt or Partition(data)
        self.train_map = self.metis.train_map
        self.train_set = self.metis.train_set
        self.partiton_k = partition_k
        src = 0
        trg = 1
        trg_triplets = self.metis.share_triplets(self.data.triples[src], self.data.triples[trg],
                                                 self.train_set[src], self.train_map[src])
        src_triplets = self.metis.share_triplets(self.data.triples[trg], self.data.triples[src],
                                                 self.train_set[trg], self.train_map[trg])
        self.triples = [src_triplets, trg_triplets]

    def _construct(self, classifier, which, **kwargs):
        self.torch_model = True
        if classifier in ['xgb', ]:
            self.torch_model = False
            return SKLearnPartition(classifier, **kwargs)
        elif classifier in ['gcn', 'gat', 'gin']:
            return NodeClassification(classifier, self.data, self.triples, construct_which=which, k=self.partiton_k,
                                      **kwargs)
        else:
            raise NotImplementedError()

    def build_model(self, model_name='xgb', siam_model=False, **kwargs):
        self.classifier_name = model_name
        if siam_model:
            model = self._construct(model_name, {0, 1}, **kwargs)
            self.model1, self.model2 = model, model

        else:
            self.model1, self.model2 = self._construct(model_name, (0,), **kwargs), \
                                       self._construct(model_name, (1,), **kwargs)

    def build_initial_graphs(self, src=0):
        trg = 1 - src

        src_triplets, trg_triplets = self.triples[src], self.triples[trg]
        src_graph = self.metis.construct_graph(src_triplets, cnt_as_weight=True)
        trg_graph = self.metis.construct_graph(trg_triplets, cnt_as_weight=True)
        return src_graph, trg_graph

    def initial_partition(self, src_graph, k, src):
        mincut, src_nodes = nxmetis.partition(src_graph, k)

        src_train, trg_train = self.metis.subgraph_trainset(src_nodes, src)
        print('mincut=', mincut)
        return src_nodes, src_train, trg_train

    def build_idx_and_target(self, nodes_list) -> Tuple[Tensor, np.ndarray]:
        idx = []
        cls = []

        for i, nodes in enumerate(nodes_list):
            idx += nodes
            cls += [i for _ in range(len(nodes))]

        return torch.tensor(idx), np.array(cls)

    def partition_to_node_ids(self, partition, k):
        if isinstance(partition, Tensor):
            partition = partition.cpu().numpy()
        if isinstance(partition, np.ndarray):
            partition = partition.flatten()
        node_ids = [list() for _ in range(k)]
        for idx, p in enumerate(partition):
            node_ids[p].append(idx)

        return node_ids

    def partition_bidirectional(self, f_left, f_right, *args, **kwargs):
        batch_1 = self.partition(f_left, f_right, 0, *args, **kwargs)
        batch_2 = self.partition(f_left, f_right, 1, *args, **kwargs)
        src_nodes, trg_nodes, src_train, trg_train = [], [], [], []

        for ids0, ids1, train0, train1 in batch_sampler_consistency(self.data, batch_1, batch_2, self.train_map):
            src_nodes.append(ids0)
            trg_nodes.append(ids1)
            src_train.append(train0)
            trg_train.append(train1)

        return src_nodes, trg_nodes, src_train, trg_train

    def partition(self, f_left: Tensor, f_right: Tensor, src=0, how_to='kmeans'):
        k = self.partiton_k
        models = [self.model1, self.model2]
        model1 = models[src]
        model2 = models[1 - src]
        if src == 1:
            f_left, f_right = f_right, f_left
        fit_model1 = True
        cluster_time = time.time()
        if how_to == 'metis':
            src_graph, trg_graph = self.build_initial_graphs(src)
            src_nodes, src_train, trg_train = self.initial_partition(src_graph, k, src)
            fit_model1 = False

        elif how_to in ['kmeans', 'spectral']:
            # from sklearn.cluster import KMeans, SpectralClustering
            f_left_norm = norm_process(f_left)
            f_right_norm = norm_process(f_right)
            idx = self.data.ill(self.data.train, 'cpu')
            idx_left, idx_right = idx[src], idx[1 - src]
            ft_left, ft_right = f_left_norm[idx_left], f_right_norm[idx_right]
            f_train = torch.cat([ft_left, ft_right], dim=1)
            # f_train = f_train.cpu().numpy()
            if how_to == 'kmeans':
                cluster = KMeans(n_clusters=k)
            # elif how_to == 'spectral':
            #     cluster = SpectralClustering(k)
            else:
                raise NotImplementedError
            tic = time.time()
            clusters = cluster.fit_predict(f_train)
            add_log('fit_kmeans_cuda', time.time() - tic)
            src_train = [[] for _ in range(k)]
            trg_train = [[] for _ in range(k)]
            for i, c in enumerate(clusters):
                src_train[c].append(idx_left[i].item())
                trg_train[c].append(idx_right[i].item())

        elif how_to == 'random':
            idx = self.data.train
            src_train = [[] for _ in range(k)]
            trg_train = [[] for _ in range(k)]
            bs = len(idx) // k
            now = 0
            cnt = 0
            for e1, e2 in idx:
                src_train[now].append(e1)
                trg_train[now].append(e2)
                cnt += 1
                if cnt > bs:
                    cnt = 0
                    now += 1
        else:
            raise NotImplementedError

        src_idx, src_target = self.build_idx_and_target(src_train)
        trg_idx, trg_target = self.build_idx_and_target(trg_train)
        # self.fit(self.model1, f_left[src_idx], src_target)
        time0 = time.time()
        add_log(f'cluster_src={src}_{how_to}', time0 - cluster_time)
        if fit_model1:
            model1.fit(f_left, src_idx, src_target, src)
        time1 = time.time()
        model2.fit(f_right, trg_idx, trg_target, 1 - src)
        time2 = time.time()
        add_log(f'fit_src={src}_{self.classifier_name}_model1', time1 - time0)
        add_log(f'fit_src={src}_{self.classifier_name}_model2', time2 - time1)
        add_log(f'fit_src={src}_{self.classifier_name}_total', time2 - time0)
        time0 = time.time()
        if fit_model1:
            partition1 = model1.predict(f_left, src)
            src_nodes = self.partition_to_node_ids(partition1, k)
            model1.clear_memory()
        time1 = time.time()
        partition2 = model2.predict(f_right, 1 - src)
        model2.clear_memory()
        trg_nodes = self.partition_to_node_ids(partition2, k)
        time2 = time.time()
        add_log(f'predict_src={src}_{self.classifier_name}_model1', time1 - time0)
        add_log(f'predict_src={src}_{self.classifier_name}_model2', time2 - time1)
        add_log(f'predict_src={src}_{self.classifier_name}_total', time2 - time0)
        if src == 1:
            src_nodes, trg_nodes = trg_nodes, src_nodes
            src_train, trg_train = trg_train, src_train
        return src_nodes, trg_nodes, src_train, trg_train


def gen_partition(corr_ind_1, src_nodes_1, trg_nodes_1, src_train_1, trg_train_1, corr_val_1, mapping_1,
                  triple1_batch_1, triple2_batch_1):
    train_pair_cnt = 0
    test_pair_cnt = 0
    from align_batch import place_triplets, overlaps, make_pairs

    IDs_s_1 = []
    IDs_t_1 = []
    Trains_s_1 = []
    Trains_t_1 = []
    Triples_s_1 = []
    Triples_t_1 = []
    for src_id, src_corr in enumerate(corr_ind_1):
        ids1_1, train1_1 = src_nodes_1[src_id], src_train_1[src_id]
        train2_1, ids2_1, triple2_1 = [], [], []
        corr_rate = 0.
        for trg_rank, trg_id in enumerate(src_corr):
            train2_1 += trg_train_1[trg_id]
            ids2_1 += trg_nodes_1[trg_id]
            triple2_1 += triple2_batch_1[trg_id]
            corr_rate += corr_val_1[src_id][trg_rank]
        ids1_1, ids2_1, train1_1, train2_1 = map(set, [ids1_1, ids2_1, train1_1, train2_1])

        IDs_s_1.append(ids1_1)
        IDs_t_1.append(ids2_1)
        Trains_s_1.append(train1_1)
        Trains_t_1.append(train2_1)
        Triples_s_1.append(set(triple1_batch_1[src_id]))
        Triples_t_1.append(set(triple2_1))

        print('Train corr=', corr_rate)

        train_pairs = make_pairs(train1_1, train2_1, mapping_1)
        train_pair_cnt += len(train_pairs)
        test_pairs = make_pairs(ids1_1, ids2_1, mapping_1)
        test_pair_cnt += len(test_pairs)

    print("*************************************************************")
    print("Total trainig pairs: " + str(train_pair_cnt))
    print("Total testing pairs: " + str(test_pair_cnt - train_pair_cnt))
    print("Total links: " + str(test_pair_cnt))
    print("*************************************************************")

    return IDs_s_1, IDs_t_1, Trains_s_1, Trains_t_1, Triples_s_1, Triples_t_1


def batch_sampler_consistency(data, batchs_1, batchs_2, train_map, top_k_corr=1, which=0,
                              share_triples=True, random=False, **kwargs):
    time_now = time.time()
    from align_batch import place_triplets, overlaps, make_pairs
    print("\n*************************************************************")
    print("Partition left 2 right: ")
    print("*************************************************************")

    src_nodes_1, trg_nodes_1, src_train_1, trg_train_1 = batchs_1

    triple1_batch_1, removed1_1 = place_triplets(data.triples[which], src_nodes_1)
    triple2_batch_1, removed2_1 = place_triplets(data.triples[1 - which], trg_nodes_1)

    corr_1 = torch.from_numpy(overlaps(
        [set(train_map[which][i] for i in s) for s in src_train_1],
        [set(s) for s in trg_train_1]
    ))

    mapping_1 = train_map[which]
    corr_val_1, corr_ind_1 = map(lambda x: x.numpy(), corr_1.topk(top_k_corr))

    # corr_ind = corr_ind.numpy()
    print('partition complete, time=', time.time() - time_now)

    IDs_s_1, IDs_t_1, Trains_s_1, Trains_t_1, Triples_s_1, Triples_t_1 = gen_partition(corr_ind_1, src_nodes_1,
                                                                                       trg_nodes_1, src_train_1,
                                                                                       trg_train_1, corr_val_1,
                                                                                       mapping_1, triple1_batch_1,
                                                                                       triple2_batch_1)

    print("\n*************************************************************")
    print("Partition right 2 left: ")
    print("*************************************************************")

    trg_nodes_2, src_nodes_2, trg_train_2, src_train_2 = batchs_2

    triple1_batch_2, removed1_2 = place_triplets(data.triples[which], src_nodes_2)
    triple2_batch_2, removed2_2 = place_triplets(data.triples[1 - which], trg_nodes_2)

    corr2 = torch.from_numpy(overlaps(
        [set(train_map[which][i] for i in s) for s in src_train_2],  # no change here
        [set(s) for s in trg_train_2]
    ))

    mapping_2 = train_map[which]  # converted for corr2, so here might still use this mapping from source 2 target
    corr_val_2, corr_ind_2 = map(lambda x: x.numpy(), corr2.topk(top_k_corr))
    print('partition complete, time=', time.time() - time_now)

    IDs_s_2, IDs_t_2, Trains_s_2, Trains_t_2, Triples_s_2, Triples_t_2 = gen_partition(corr_ind_2, src_nodes_2,
                                                                                       trg_nodes_2, src_train_2,
                                                                                       trg_train_2, corr_val_2,
                                                                                       mapping_2, triple1_batch_2,
                                                                                       triple2_batch_2)

    print("\n*************************************************************")
    print("Combination: ")
    print("*************************************************************")

    corr_3 = torch.from_numpy(overlaps(
        [set(s) for s in src_train_1],
        [set(s) for s in src_train_2]
    ))

    corr_val_3, corr_ind_3 = map(lambda x: x.numpy(), corr_3.topk(top_k_corr))

    train_pair_cnt = 0
    test_pair_cnt = 0
    real_test_pairs_cnt = 0
    real_train_pairs_cnt = 0
    train_pair_unq = []
    test_pair_unq = []

    real_test_sourceids = set([item[0] for item in data.test])

    for src1_id, src1_corr in enumerate(corr_ind_3):
        ids_s_1, trains_s_1, triples_s_1 = IDs_s_1[src1_id], Trains_s_1[src1_id], Triples_s_1[src1_id]
        ids_t_1, trains_t_1, triples_t_1 = IDs_t_1[src1_id], Trains_t_1[src1_id], Triples_t_1[src1_id]

        corr_rate = 0.
        for src2_rank, src2_id in enumerate(src1_corr):
            ids_s_2, trains_s_2, triples_s_2 = IDs_s_2[src2_id], Trains_s_2[src2_id], Triples_s_2[src2_id]
            ids_t_2, trains_t_2, triples_t_2 = IDs_t_2[src2_id], Trains_t_2[src2_id], Triples_t_2[src2_id]
            corr_rate += corr_val_3[src1_id][src2_rank]

            ids_s_1 = ids_s_1.union(ids_s_2)
            ids_t_1 = ids_t_1.union(ids_t_2)
            trains_s_1 = trains_s_1.union(trains_s_2)
            trains_t_1 = trains_t_1.union(trains_t_2)
            triples_s_1 = triples_s_1.union(triples_s_2)
            triples_t_1 = triples_t_1.union(triples_t_2)

        # print('Train corr=', corr_rate)

        train_pairs = make_pairs(trains_s_1, trains_t_1, mapping_1)
        train_pair_cnt += len(train_pairs)
        test_pairs = make_pairs(ids_s_1, ids_t_1, mapping_1)
        test_pair_cnt += len(test_pairs)

        real_train_pairs = [item for item in train_pairs if item[0] not in real_test_sourceids]
        real_train_pairs_cnt += len(real_train_pairs)

        real_test_pairs = [item for item in test_pairs if item[0] in real_test_sourceids]
        real_test_pairs_cnt += len(real_test_pairs)

        train_pair_unq.extend(real_train_pairs)
        test_pair_unq.extend(real_test_pairs)

        yield map(list, [ids_s_1, ids_t_1, trains_s_1, trains_t_1])

        # yield [list(triples_s_1), list(triples_t_1), ids_s_1, ids_t_1, train_pairs, test_pairs, real_test_pairs]

        # print(str(len(ids_s_1)) + '\t' + str(len(ids_t_1)))
        # print(len(train_pairs))
        # print(len(test_pairs) - len(train_pairs))
        # ids1_2, ids2_2, train1_2, train2_2 = map(set, [ids1_2, ids2_2, train1_2, train2_2])

    print("\n*************************************************************")
    print("Total trainig pairs: " + str(train_pair_cnt))
    print("Real trainig pairs: " + str(real_train_pairs_cnt))
    print("Real testing pairs: " + str(real_test_pairs_cnt))
    print("Total links: " + str(test_pair_cnt))
    train_pair_unq = set(train_pair_unq)
    test_pair_unq = set(test_pair_unq)
    print("Real trainig pairs uniq: " + str(len(train_pair_unq)))
    print("Real testing pairs uniq: " + str(len(test_pair_unq)))
    print("Total links uniq: " + str(len(test_pair_unq) + len(train_pair_unq)))
    print("*************************************************************\n")


if __name__ == '__main__':
    ea = OpenEAData('../OpenEA_dataset_v1.1/EN_FR_15K_V1/', train_ratio=0.3)
    k = 10
    metis = Partition(ea)
    src_nodes, trg_nodes, src_train, trg_train = metis.partition(src_k=k, trg_k=k)
