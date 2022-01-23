import numpy as np
from dataset import EAData


class RandomUniquePartition(object):
    def __init__(self, ea: EAData, batch_size=2000, neg_k=5, shuffle=True):
        links = ea.train
        self.shuffle = True
        self.batch_size = batch_size
        self.links = links
        self.neg_k = neg_k
        self.ea = ea
        self.resample_negative_pairs()

    def resample_negative_pairs(self):
        self.ng1 = {}
        self.ng2 = {}

        # def resample_negative_pairs(self):
        #     total1, total2 = len(neg1), len(neg2)
        #     batch_size = self.batch_size
        #     neg_k = self.neg_k
        #     print('resample!')
        #     ng1, ng2 = {}, {}
        #     for i in range(0, len(self.links), batch_size):
        #         neg_seed_1 = []
        #         neg_seed_2 = []
        #         for l in range(0, batch_size * neg_k, batch_size):
        #             neg_seed_1.append(neg1[l + left_begin:l + left_begin + batch_size])
        #             neg_seed_2.append(neg2[l + right_begin:l + right_begin + batch_size])
        #
        #         ng1[i] = neg_seed_1
        #         ng2[i] = neg_seed_2

        # self.ng1, self.ng2 = ng1, ng2

    def sample(self):
        batch_size = self.batch_size
        links = self.links
        if self.shuffle:
            np.random.shuffle(links)

        total1 = len(self.ea.ent1)
        total2 = len(self.ea.ent2)
        for i in range(0, len(links), batch_size):
            r = len(links) if i + batch_size > len(links) else i + batch_size
            curr_links = links[i:r]
            # print(len(curr_links))
            bs_now = r - i
            if i in self.ng1:
                neg_seed_1 = self.ng1[i]
                neg_seed_2 = self.ng2[i]
            else:
                l1_set = set([l[0] for l in curr_links])
                l2_set = set([l[1] for l in curr_links])
                neg1 = list(filter(lambda x: x not in l1_set, range(total1)))
                neg2 = list(filter(lambda x: x not in l2_set, range(total2)))
                np.random.shuffle(neg1)
                np.random.shuffle(neg2)
                neg1 = neg1[:bs_now * self.neg_k]
                neg2 = neg2[:bs_now * self.neg_k]
                neg_seed_1 = []
                neg_seed_2 = []
                for l in range(0, bs_now * self.neg_k, bs_now):
                    neg_seed_1.append(neg1[l:l + bs_now])
                    neg_seed_2.append(neg2[l:l + bs_now])
                    # print('---------')
                    # print(len(neg_seed_1[-1]), bs_now)
                    # print(len(neg_seed_2[-1]), bs_now)

                self.ng1[i] = neg_seed_1
                self.ng2[i] = neg_seed_2

            yield curr_links, neg_seed_1, neg_seed_2


def RandomPartition(ea: EAData, batch_size=2000, neg_k=5, shuffle=True):
    links = ea.train
    total1 = len(ea.ent1)
    total2 = len(ea.ent2)
    if shuffle:
        np.random.shuffle(links)
    for i in range(0, len(links), batch_size):
        r = len(links) if i + batch_size > len(links) else i + batch_size
        # neg_seed_1 = np.random.randint(0, total1, size=r-i)
        # neg_seed_2 = np.random.randint(0, total2, size=r-i)
        neg_seed_1 = []
        neg_seed_2 = []
        for _ in range(neg_k):
            neg_seed_1.append(np.random.choice(total1, r - i, replace=False))
            neg_seed_2.append(np.random.choice(total2, r - i, replace=False))
        yield links[i: r], neg_seed_1, neg_seed_2
