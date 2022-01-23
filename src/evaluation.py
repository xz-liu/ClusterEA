from dataset import *

import faiss
import scipy.spatial


def get_hits_slow(em1, em2, test_pair, top_k=(1, 10)):
    em1 = em1.detach().numpy()
    em2 = em2.detach().numpy()
    Lvec = np.array([em1[e1] for e1, e2 in test_pair])
    Rvec = np.array([em2[e2] for e1, e2 in test_pair])
    sim = scipy.spatial.distance.cdist(Lvec, Rvec, metric='cityblock')
    top_lr = [0] * len(top_k)
    for i in range(Lvec.shape[0]):
        rank = sim[i, :].argsort()
        rank_index = np.where(rank == i)[0][0]
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                top_lr[j] += 1
    top_rl = [0] * len(top_k)
    for i in range(Rvec.shape[0]):
        rank = sim[:, i].argsort()
        rank_index = np.where(rank == i)[0][0]
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                top_rl[j] += 1
    print('For each left:')
    for i in range(len(top_lr)):
        print('Hits@%d: %.2f%%' % (top_k[i], top_lr[i] / len(test_pair) * 100))
    print('For each right:')
    for i in range(len(top_rl)):
        print('Hits@%d: %.2f%%' % (top_k[i], top_rl[i] / len(test_pair) * 100))


def my_dist_func(L, R, k=100):
    dim = len(L[0])
    torch.cuda.empty_cache()
    index = faiss.IndexFlat(dim)
    index = faiss.index_cpu_to_all_gpus(index)
    index.add(R)
    D, I = index.search(L, k)
    print("FAISS complete")
    return I


def get_hits(em1, em2, test_pair, top_k=(1, 5, 10, 50, 100), partition=1, norm=False, src_nodes=None, trg_nodes=None):
    if isinstance(em1, Tensor):
        em1 = em1.cpu().detach().numpy()
        em2 = em2.cpu().detach().numpy()
    if norm:
        # em1= norm_process(torch.from_numpy(em1)).detach().numpy()
        # em2= norm_process(torch.from_numpy(em2)).detach().numpy()
        em1 = em1 / np.linalg.norm(em1, axis=-1, keepdims=True)
        em2 = em2 / np.linalg.norm(em2, axis=-1, keepdims=True)

    def filter_pair(pair, src, trg):
        if src is None or trg is None:
            return pair
        src = set(src)
        trg = set(trg)
        return list(filter(lambda x: x[0] in src and x[1] in trg, pair))

    batch_size = len(test_pair) // partition
    print(batch_size)
    total_size = 0
    top_lr = [0] * len(top_k)
    for x in range(partition):
        left = x * batch_size
        right = left + batch_size if left + batch_size < len(test_pair) else len(test_pair)
        filtered = filter_pair(test_pair[left:right], src_nodes, trg_nodes)
        print(len(filtered))
        if len(filtered) == 0:
            continue
        total_size += len(filtered)
        Lvec = np.array([em1[e1] for e1, e2 in filtered])
        Rvec = np.array([em2[e2] for e1, e2 in filtered])
        ranks = my_dist_func(Lvec, Rvec)
        for i in range(Lvec.shape[0]):
            rank = ranks[i]
            rank_index = np.where(rank == i)[0][0] if i in rank else 1000
            for j in range(len(top_k)):
                if rank_index < top_k[j]:
                    top_lr[j] += 1
    print('For each left:')
    print('Total size=', total_size)
    for i in range(len(top_lr)):
        print('Hits@%d: %.2f%%' % (top_k[i], top_lr[i] / (total_size + 1e-8) * 100))

    return top_k, top_lr, total_size
