from dataset import *
import torch.nn as nn
from evaluation import get_hits
from metis import Partition

from common.sinkhorn import *
from partition_models.trainer import PartitionTrainer

from sparse_eval import sparse_top_k
from align_batch import SelectedCandidates
from utils import get_batch_sim, get_batch_csls_sim


def overlaps(src: List[set], trg: List[set]):
    return np.array([[float(len(s.intersection(t))) / (float(len(s)) + 0.01) for t in trg] for s in src])


class LargeFramework:
    def __init__(self, ea: EAData, max_sinkhorn_sz, device='cuda'):
        self.max_sinkhorn_sz = max_sinkhorn_sz
        self.device = device
        self.ea = ea
        partition = Partition(ea)
        self.tset, self.tmap = partition.test_set, partition.train_map
        self.partition = partition

    @torch.no_grad()
    def get_metis_cps_similarity_matrix(self, k=10, src=0, top_k_corr=1,
                                        embeddings=None, enhance='sinkhorn'):

        print('---begin-partition-test--metis')
        # left_embedding, right_embedding = self._get_final_embedding() \
        #     if embeddings is None else tuple(embeddings)

        partition = Partition(self.ea)

        def make_pairs(src):
            return set([(x[0], x[1]) for x in src])

        src_nodes, trg_nodes, src_train, trg_train = \
            partition.partition(src, k, k, True)
        partition.eval_partition(src_nodes, trg_nodes, src_train, trg_train)
        # self.eval_sim(
        #     self.get_partition_sim(left_embedding, right_embedding, partition,
        #                            src_nodes, trg_nodes, src_train, trg_train, src, top_k_corr, enhance))

    @torch.no_grad()
    def get_partition_nodes(self, k=10, src=0,
                            embeddings=None,
                            partition_method='kmeans', model='xgb', model_dim=600, max_iter=500,
                            bid=False, partition=None):
        left_embedding, right_embedding = tuple(embeddings)

        partition = partition or self.partition
        print('---begin-partition-test--model')
        pt = PartitionTrainer(self.ea, k, partition)
        pt.build_model(model, False, feature_dim=model_dim, max_iter=max_iter)

        src_nodes, trg_nodes, src_train, trg_train = pt.partition_bidirectional(left_embedding, right_embedding,
                                                                                how_to=partition_method) if bid else \
            pt.partition(left_embedding, right_embedding, src,
                         how_to=partition_method)
        # partition.eval_partition(src_nodes, trg_nodes, src_train, trg_train)
        #
        # print(list(map(len, src_nodes)))
        # print(list(map(len, trg_nodes)))
        return partition, src_nodes, trg_nodes, src_train, trg_train

    @torch.no_grad()
    def get_partition_similarity_matrix(self, k=10, src=0, top_k_corr=1,
                                        embeddings=None, enhance='sinkhorn',
                                        partition_method='kmeans', model='xgb', model_dim=600, max_iter=500,
                                        norm=False, bid=False):

        left_embedding, right_embedding = tuple(embeddings)
        if k == 1:
            return self.create_batch_sim(left_embedding, right_embedding, enhance,
                                         [left_embedding.size(0), right_embedding.size(0)], whole_batch=True, norm=norm)
        # begin partition test

        # partition.eval_partition(src_nodes, trg_nodes, src_train, trg_train)
        # from align_batch import AlignmentBatch
        #

        partition, src_nodes, trg_nodes, src_train, trg_train = \
            self.get_partition_nodes(k=k, src=src,
                                     embeddings=embeddings,
                                     partition_method=partition_method,
                                     model=model,
                                     model_dim=model_dim,
                                     max_iter=max_iter,
                                     bid=False)
        tic = time.time()

        ret = self.get_partition_sim(left_embedding, right_embedding, partition,
                                     src_nodes, trg_nodes, src_train, trg_train, 0, top_k_corr, enhance, norm)
        toc = time.time()
        add_log(f'sinkhorn_time_{partition_method}_{model}_{src}', toc - tic)
        return ret

    def get_eval_ids(self, ids1, ids2, src=0, keep_train=False):
        tset = self.tset
        tmap = self.tmap
        curr = 0
        ids1_set, ids2_set = set(ids1), set(ids2)
        new_ids1, new_ids2 = [], []
        while curr < len(ids1):
            id1 = ids1[curr]
            if (id1 in tset[src] or keep_train) and tmap[src][id1] in ids2_set:
                new_ids1.append(id1)
            curr += 1
        curr = 0
        while curr < len(ids2):
            id2 = ids2[curr]
            if (id2 in tset[1 - src] or keep_train) and tmap[1 - src][id2] in ids1_set:
                new_ids2.append(id2)
            curr += 1

        return new_ids1, new_ids2

    def create_batch_sim(self, left, right, enhance='none', size=None, assoc0=None, assoc1=None, norm=False,
                         whole_batch=False, src=0, return_use_sinkhorn=False):
        use_sinkhorn = 0
        if size is None:
            size = [left.size(0), right.size(0)]
        if whole_batch:
            assoc0 = list(range(size[0]))
            assoc1 = list(range(size[1]))
            assoc0, assoc1 = self.get_eval_ids(assoc0, assoc1)
        if assoc0 is not None:
            assoc0 = torch.tensor(assoc0)
            left = left[torch.tensor(assoc0)]
        if assoc1 is not None:
            assoc1 = torch.tensor(assoc1)
            right = right[torch.tensor(assoc1)]
        sz = left.size(0) * right.size(0)
        if src == 1:
            left, right = right, left
            assoc0, assoc1 = assoc1, assoc0
            size = [size[1], size[0]]
        if enhance == 'sinkhorn' and sz < (self.max_sinkhorn_sz ** 2):
            # sim = get_batch_csls_sim((left_embedding[assoc0], right_embedding[assoc1]), topk=300, split=False)
            if norm:
                dist_func = cosine_distance
            else:
                dist_func = l1_dist
            # if norm:
            #     dist_func = cosine_distance
            # sim = Lin_Sinkhorn(left, right, 1, 500, self.device)
            sim = matrix_sinkhorn(left, right, dist_func=dist_func, device=self.device)
            spmat = remain_topk_sim(sim, dim=0, k=50, split=False)
            ind = spmat._indices()
            val = spmat._values()
            use_sinkhorn = 1
        elif whole_batch:
            sim_function = get_batch_csls_sim if enhance == 'csls' else get_batch_sim
            ind, val = sim_function((left.to(self.device),
                                     right.to(self.device)))

        else:
            sim = cosine_sim(left.to(self.device), right.to(self.device))
            spmat = remain_topk_sim(sim, dim=0, k=50, split=False)
            ind = spmat._indices()
            val = spmat._values()
        ind = torch.stack(
            [ind[0] if assoc0 is None else assoc0[ind[0]],
             ind[1] if assoc1 is None else assoc1[ind[1]]]
        )
        batch_sim = ind2sparse(ind.to(self.device), size, values=val.to(self.device))
        if return_use_sinkhorn:
            return batch_sim, use_sinkhorn
        return batch_sim

    def get_partition_sim(self, left_embedding, right_embedding, partition,
                          src_nodes, trg_nodes, src_train, trg_train, src=0, top_k_corr=2, enhance='sinkhorn',
                          norm=False, keep_train=False):

        merged_sim = None
        topk = (1, 5, 10, 50, 100)
        size = [left_embedding.size(0), right_embedding.size(0)]
        corr = torch.from_numpy(overlaps(
            [set(partition.train_map[src][i] for i in s) for s in src_train],
            [set(s) for s in trg_train]
        ))
        use_sinkhorn = 0
        # mapping = partition.train_map[src]
        corr_val, corr_ind = map(lambda x: x.numpy(), corr.topk(top_k_corr))
        # final_lr = None
        total = 1e-8
        for src_id, src_corr in enumerate(corr_ind):
            ids1, train1 = src_nodes[src_id], src_train[src_id]
            # train2, ids2, triple2 = [], [], []
            corr_rate = 0.
            ids2 = []
            for trg_rank, trg_id in enumerate(src_corr):
                # train2 += trg_train[trg_id]
                ids2 += trg_nodes[trg_id]
                # triple2 += triple2_batch[trg_id]
                corr_rate += corr_val[src_id][trg_rank]
            ids1, ids2 = self.get_eval_ids(ids1, ids2, keep_train=keep_train)
            assoc0 = torch.tensor(ids1)
            assoc1 = torch.tensor(ids2)
            if len(ids1) == 0 or len(ids2) == 0:
                continue
            # _, toplr, total_now = get_hits(left_embedding, right_embedding, self.ea.test, top_k=topk,
            #                                src_nodes=ids1, trg_nodes=ids2)
            # toplr = np.array(toplr)
            # total += total_now
            # final_lr = toplr if final_lr is None else final_lr + toplr
            # print('Calculate batch [{} * {}]'.format(assoc0.numel(), assoc1.numel()),
            #       list(map(lambda x: x.device, [left_embedding, right_embedding, assoc0, assoc1])))
            batch_sim, curr_use_sinkhorn = self.create_batch_sim(left_embedding, right_embedding, enhance, size, assoc0,
                                                                 assoc1,
                                                                 norm, return_use_sinkhorn=True)
            batch_sim = batch_sim.cpu()

            use_sinkhorn += curr_use_sinkhorn
            print('Total sinkhorn=', use_sinkhorn)
            if merged_sim is None:
                merged_sim = batch_sim
            else:
                merged_sim += batch_sim
            # merged_sim = merged_sim.coalesce()
            print(total)
        # print(final_lr / total)
        # print(final_lr / len(self.ea.test))

        return merged_sim

    @torch.no_grad()
    def get_test_sim(self, sim):

        d: EAData = self.ea
        stru_sim = sim
        candidates = SelectedCandidates(d.test, *d.ents)

        stru_sim = candidates.filter_sim_mat(sim)
        return stru_sim

    @torch.no_grad()
    def eval_sim(self, sim, batch_size=3000):
        d: EAData = self.ea
        print('Size of sim:', sim.size(), sim._values().size())
        sim = self.get_test_sim(sim)
        print('Size of test sim:', sim.size(), sim._values().size())
        acc = sparse_top_k(sim.to(self.device), d.ill(d.test, self.device), self.device, batch_size=batch_size)
        return sim, acc
        # sparse_top_k(sim.to(self.device), d.ill(d.test, self.device), self.device)

    @torch.no_grad()
    def csls_matrix(self, sims, emb1: Tensor, emb2: Tensor, csls_k=10, gpu=False, device='cpu', apply_csls=True,
                    factor=1.,
                    norm=True):
        from sparse_eval import global_level_semantic_sim
        left_val = global_level_semantic_sim((emb1, emb2), k=csls_k, norm=norm, split=True, gpu=gpu)[0].mean(
            -1) * factor
        right_val = global_level_semantic_sim((emb2, emb1), k=csls_k, norm=norm, split=True, gpu=gpu)[0].mean(
            -1) * factor

        sim_mat = None
        if isinstance(sims, tuple):
            for curr_sim in sims:
                if curr_sim is None:
                    continue
                sim_mat = curr_sim.to(device) if sim_mat is None else sim_mat + curr_sim.to(device)
        else:
            sim_mat = sims.to(device)

        sim_mat = sim_mat.to(device)
        if not apply_csls:
            return sim_mat
        # sim_mat = sim_mat.coalesce()
        sim_ind = sim_mat._indices()
        sim_val = sim_mat._values()
        sim_val = sim_val * 2
        # device = sim_mat.device
        # sim_val += 2
        sim_val -= left_val[sim_ind[0]].to(device)
        sim_val -= right_val[sim_ind[1]].to(device)
        # sim_val = torch.relu(sim_val)
        sim_val -= sim_val.min()
        sim = ind2sparse(sim_ind, sim_mat.size(), values=sim_val)
        sim = self.get_test_sim(sim)
        sparse_minmax(sim)
        return sim
