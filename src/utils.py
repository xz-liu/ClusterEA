import numpy as np
import scipy.sparse as sp
import torch
from tqdm import tqdm


global_dict = {}


def add_log(key, value):
    global_dict[key] = value


def func(triples):
    head = {}
    cnt = {}
    for tri in triples:
        if tri[1] not in cnt:  # relation
            cnt[tri[1]] = 1
            head[tri[1]] = {tri[0]}
        else:
            cnt[tri[1]] += 1
            head[tri[1]].add(tri[0])
    r2f = {}
    for r in cnt:
        r2f[r] = len(head[r]) / cnt[r]
    return r2f


def ifunc(triples):
    tail = {}
    cnt = {}
    for tri in triples:
        if tri[1] not in cnt:
            cnt[tri[1]] = 1
            tail[tri[1]] = {tri[2]}
        else:
            cnt[tri[1]] += 1
            tail[tri[1]].add(tri[2])
    r2if = {}
    for r in cnt:
        r2if[r] = len(tail[r]) / cnt[r]
    return r2if


def get_weighted_adj(triples, e):
    r2f = func(triples)
    r2if = ifunc(triples)
    M = {}
    for tri in triples:
        if tri[0] == tri[2]:
            continue
        if (tri[0], tri[2]) not in M:
            M[(tri[0], tri[2])] = max(r2if[tri[1]], 0.3)
        else:
            M[(tri[0], tri[2])] += max(r2if[tri[1]], 0.3)
        if (tri[2], tri[0]) not in M:
            M[(tri[2], tri[0])] = max(r2f[tri[1]], 0.3)
        else:
            M[(tri[2], tri[0])] += max(r2f[tri[1]], 0.3)
    row, col, data = [], [], []
    for key in M:
        row.append(key[1])
        col.append(key[0])
        data.append(M[key])
    # return row, col, data
    return sp.coo_matrix((data, (row, col)), shape=(e, e))


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col))
        values = mx.data
        shape = mx.shape
        return torch.LongTensor(coords), torch.FloatTensor(values), shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def add_cnt_for(mp, val, begin=None):
    if begin is None:
        if val not in mp:
            mp[val] = len(mp)
        return mp, mp[val]
    else:
        if val not in mp:
            mp[val] = begin
            begin += 1
        return mp, mp[val], begin


def argprint(**kwargs):
    return '\n'.join([str(k) + "=" + str(v) for k, v in kwargs.items()])


def set_seed(seed):
    if seed:
        import random
        import numpy
        import torch
        import tensorflow
        tensorflow.compat.v1.random.set_random_seed(seed)
        numpy.random.seed(seed)
        random.seed(seed)
        torch.random.manual_seed(seed)


import faiss.contrib.torch_utils


def faiss_search_impl(emb_q, emb_id, emb_size, shift, k=50, search_batch_sz=50000, gpu=True, l2=False):
    index = faiss.IndexFlatL2(emb_size) if l2 else faiss.IndexFlat(emb_size)
    if gpu:
        index = faiss.index_cpu_to_all_gpus(index)
    else:
        emb_id = emb_id.cpu()
        emb_q = emb_q.cpu()
    index.add(emb_id)
    print('Total index =', index.ntotal)
    vals, inds = [], []
    for i_batch in tqdm(range(0, len(emb_q), search_batch_sz)):
        val, ind = index.search(emb_q[i_batch:min(i_batch + search_batch_sz, len(emb_q))], k)
        val = torch.tensor(val)
        val = 1 - val
        vals.append(val)
        inds.append(torch.tensor(ind) + shift)
        # print(vals[-1].size())
        # print(inds[-1].size())
    index.reset()
    del index, emb_id, emb_q
    vals, inds = torch.cat(vals), torch.cat(inds)
    return vals, inds


from utils_largeea import *
import faiss


@torch.no_grad()
def global_level_semantic_sim(embs, k=50, search_batch_sz=500000, index_batch_sz=500000
                              , split=False, norm=True, gpu=True, l2=False):
    print('FAISS number of GPUs=', faiss.get_num_gpus())
    size = [embs[0].size(0), embs[1].size(0)]
    emb_size = embs[0].size(1)
    if norm:
        embs = apply(norm_process, *embs)
    # emb_q, emb_id = apply(lambda x: x.cpu().numpy(), *embs)
    emb_q, emb_id = embs
    del embs
    # gc.collect()
    vals, inds = [], []
    total_size = emb_id.shape[0]
    for i_batch in range(0, total_size, index_batch_sz):
        i_end = min(total_size, i_batch + index_batch_sz)
        val, ind = faiss_search_impl(emb_q, emb_id[i_batch:i_end], emb_size, i_batch, k, search_batch_sz, gpu, l2)
        vals.append(val)
        inds.append(ind)

    vals, inds = torch.cat(vals, dim=1), torch.cat(inds, dim=1)
    vals = vals.clone().detach()
    inds = inds.clone().detach()
    print(vals.size(), inds.size())

    return topk2spmat(vals, inds, size, 0, torch.device('cpu'), split)


def get_batch_csls_sim(embed, topk=50, csls=10, split=True, sim_func=cosine_sim):
    device = embed[0].device
    from utils_largeea import cosine_sim
    from sparse_eval import bi_csls_matrix
    sim0 = sim_func(embed[0], embed[1])
    sim1 = sim_func(embed[1], embed[0])
    sim = bi_csls_matrix(sim0, sim1, k=csls, return2=False)
    val, ind = sim.topk(topk)
    spmat = topk2spmat(val, ind, sim.size(), device=device)
    if split:
        return spmat._indices(), spmat._values()
    else:
        return spmat


def get_batch_sim(embed, topk=50, split=True, norm=True, gpu=True, l2=False):
    # embed = self.get_gnn_embed()
    size = apply(lambda x: x.size(0), *embed)
    # x2y_val, x2y_argmax = fast_topk(2, embed[0], embed[1])
    # y2x_val, y2x_argmax = fast_topk(2, embed[1], embed[0])
    # ind, val = filter_mapping(x2y_argmax, y2x_argmax, size, (x2y_val, y2x_val), 0)

    torch.cuda.empty_cache()
    spmat = global_level_semantic_sim(embed, k=topk, gpu=gpu, norm=norm, l2=l2).to(embed[0].device)
    if split:
        return spmat._indices(), spmat._values()
    else:
        return spmat


import dgl


@torch.no_grad()
def ConstructGraph(triples, num_ent):
    support = get_weighted_adj(triples, num_ent)
    support = preprocess_adj(support)
    edge_index, edge_weight, _ = support
    src_nodes, trg_nodes = edge_index[0].tolist(), edge_index[1].tolist()
    g = dgl.graph((src_nodes, trg_nodes))
    # g = dgl.add_self_loop(g)
    # data = np.array(data + [1 for _ in range(g.num_edges() - len(data))])
    # edge_weight = dgl.nn.EdgeWeightNorm(norm='both')(g, torch.from_numpy(data).to(dtype=torch.float32))
    g.edata['weight'] = edge_weight
    return g


@torch.no_grad()
def ConstructNaiveGraph(triples, *args, **kwargs):
    g = dgl.graph(([t[0] for t in triples], [t[2] for t in triples]))
    return g
