import utils
from utils import *
from dataset import *


def get_bi_mapping(src2trg, trg2src, lens) -> Tensor:
    srclen, trglen = lens
    with torch.no_grad():
        i = torch.arange(srclen, device=src2trg.device).to(torch.long)
        return trg2src[src2trg[i]] == i


def filter_mapping(src2trg: Tensor, trg2src: Tensor, lens, values: Tuple[Tensor, Tensor], th):
    vals2t, valt2s = values
    added_s = np.zeros(lens[0], dtype=np.int)
    added_t = np.zeros(lens[1], dtype=np.int)
    pair_s, pair_t = [], []
    val = []
    for i in range(lens[0]):
        j = src2trg[i, 0]
        # print(now)
        if added_t[j] == 1 or i != trg2src[j, 0]:
            continue
        gap_x2y = vals2t[i, 0] - vals2t[i, 1]
        gap_y2x = valt2s[j, 0] - valt2s[j, 1]
        if gap_y2x < th or gap_x2y < th:
            continue
        added_s[i] = 1
        added_t[j] = 1
        pair_s.append(i)
        pair_t.append(j)
        val.append(vals2t[i, 0])

    return torch.tensor([pair_s, pair_t]), torch.tensor(val)


def rearrange_ids(nodes, merge: bool, *to_map):
    ent_mappings = [{}, {}]
    rel_mappings = [{}, {}]
    ent_ids = [[], []]
    shift = 0
    for w, node_set in enumerate(nodes):
        for n in node_set:
            ent_mappings[w], nn, shift = add_cnt_for(ent_mappings[w], n, shift)
            ent_ids[w].append(nn)
        shift = len(ent_ids[w]) if merge else 0
    mapped = []
    shift = 0
    curr = 0
    for i, need in enumerate(to_map):
        now = []
        if len(need) == 0:
            mapped.append([])
            continue
        is_triple = len(need[0]) == 3
        for tu in need:
            if is_triple:
                h, t = ent_mappings[curr][tu[0]], ent_mappings[curr][tu[-1]]
                rel_mappings[curr], r, shift = add_cnt_for(rel_mappings[curr], tu[1], shift)
                now.append((h, r, t))
            else:
                now.append((ent_mappings[0][tu[0]], ent_mappings[1][tu[-1]]))
        mapped.append(now)
        curr += is_triple
        if not merge:
            shift = 0
    rel_ids = [list(rm.values()) for rm in rel_mappings]

    return ent_mappings, rel_mappings, ent_ids, rel_ids, mapped


def make_assoc(maps, src_len, trg_len, merge):
    assoc = np.empty(src_len + trg_len, dtype=np.int)
    shift = 0 if merge else 1
    shift = shift * src_len
    for idx, ent_mp in enumerate(maps):
        for k, v in ent_mp.items():
            assoc[v + idx * shift] = k
    return torch.tensor(assoc)


def filter_ent_list(id_map, ent_collection):
    id_ent_mp = {}
    if isinstance(ent_collection, dict):
        for ent, i in ent_collection.items():
            if i in id_map:
                id_ent_mp[ent] = id_map[i]
        return id_ent_mp
    else:
        for i, ent in enumerate(ent_collection):
            if i in id_map:
                id_ent_mp[ent] = id_map[i]

        return sorted(id_ent_mp.keys(), key=lambda x: id_ent_mp[x])


class SelectedCandidates:
    def __init__(self, pairs, e1, e2):
        self.total_len = len(pairs)
        pairs = np.array(pairs).T
        self.pairs = pairs
        self.ent_maps = rearrange_ids(pairs, False)[0]
        self.assoc = make_assoc(self.ent_maps, *([self.total_len] * 2), False)
        self.shift = self.total_len
        self.ents = [x for x in map(filter_ent_list, self.ent_maps, [e1, e2])]
        self.sz = [len(e1), len(e2)]
        pass

    def convert_sim_mat(self, sim):
        # selected sim(dense) to normal sim(sparse)
        ind, val = sim._indices(), sim._values()
        assoc = self.assoc.to(sim.device)
        ind = torch.stack(
            [assoc[ind[0]],
             assoc[ind[1] + self.shift]]
        )
        return ind2sparse(ind, self.sz, values=val)

    @torch.no_grad()
    def filter_sim_mat(self, sim):
        # '''
        # filter normal sim with selected candidates
        def build_filter_array(sz, nodes, device):
            a = torch.zeros(sz).to(torch.bool).to(device)
            a[torch.from_numpy(nodes).to(device)] = True
            a = torch.logical_not(a)
            ret = torch.arange(sz).to(device)
            ret[a] = -1
            return ret

        ind, val = sim._indices(), sim._values()
        ind0, ind1 = map(lambda x, xsz, xn: build_filter_array(xsz, xn, x.device)[x],
                         ind, sim.size(), self.pairs)

        remain = torch.bitwise_and(ind0 >= 0, ind1 >= 0)
        return ind2sparse(ind[:, remain], sim.size(), values=val[remain])


def place_triplets(triplets, nodes_batch):  # after divide the nodes, place the triples!!
    batch = defaultdict(list)
    node2batch = {}
    for i, nodes in enumerate(nodes_batch):
        for n in nodes:
            node2batch[n] = i
    removed = 0
    for h, r, t in triplets:
        h_batch, t_batch = node2batch.get(h, -1), node2batch.get(t, -1)
        if h_batch == t_batch and h_batch >= 0:
            batch[h_batch].append((h, r, t))
        else:
            removed += 1
    print('split triplets complete, total {} triplets removed'.format(removed))

    return batch, removed


def make_pairs(src, trg, mp):
    return list(filter(lambda p: p[1] in trg, [(e, mp[e]) for e in set(filter(lambda x: x in mp, src))]))


class WholeBatch:
    def __init__(self, ea: EAData, backbone='dual-amn', *args, **kwargs):
        self.backbone = backbone
        if self.backbone in ['rrea', 'gcn-align', 'dual-amn', 'mraea', 'dual-large', 'rrea-large']:
            t1, t2 = ea.triples
            es = len(ea.ent1)
            rs = len(ea.rel1)
            t2 = [[h + es, r + rs, t + es] for h, r, t in t2]
            test_ill = [[e1, e2 + es] for e1, e2 in ea.test]
            train_ill = [[e1, e2 + es] for e1, e2 in ea.train]
            from prev_models import ModelWrapper
            self.model = ModelWrapper(self.backbone,
                                      triples=t1 + t2,
                                      link=torch.tensor(test_ill),
                                      ent_sizes=[len(ids) for ids in ea.ents],
                                      rel_sizes=[len(ids) for ids in ea.rels],
                                      device='cuda',
                                      dim=200,
                                      )

            self.model.update_trainset(np.array(train_ill).T)
            self.model.update_devset(np.array(test_ill).T)
        elif self.backbone == 'gcn-large':
            from prev_models.gcn_sample.train import SamplingGCNTrainer
            self.model = SamplingGCNTrainer(ea)
        else:
            raise NotImplementedError


from metis import Partition, overlaps
from collections import defaultdict


def get_whole_batch(data: EAData, backbone='dual-amn', *args, **kwargs):
    return WholeBatch(data, backbone=backbone,
                      *args, **kwargs)
