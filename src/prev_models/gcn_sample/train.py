from utils import set_seed

set_seed(0)

from .models import *
from evaluation import get_hits
from .partition import RandomUniquePartition
from dataset import *

# ds = 'srp'
# scale = 'large'
# lang = 'fr'
# fanout = -1
dim = 200
train_epoch = 10
batch_size = 2000
learning_rate = 0.001
fanouts = [8, 8]
neg_k = 2
margin = 3.0
weight_decay = 0
gcn_first_layer_weight = False
# For each left:
# Hits@1: 4.83%
# Hits@10: 9.80%
# Hits@50: 14.19%
# Hits@100: 16.81%
# Epoch : 499
device = 'cuda'


# get_hits(em1.cpu(), em2.cpu(), ea.test, partition=)

class SamplingGCNTrainer():
    def __init__(self, ea):
        pass

        # ea = OpenEAData('../dbp_yg_large/')
        print(ea)

        model = GCN(dim, dim, dim, first_layer_weight=gcn_first_layer_weight, device=device)
        model.to(device)

        print('model to cuda complete')
        framework = LargeGCNFramework(*ea.triples, [len(ea.ent1), len(ea.ent2)], model, device)
        # put embedding to GPU
        framework.to(device)

        # opt = Lamb(framework.parameters(), lr=learning_rate, weight_decay=weight_decay)
        opt = torch.optim.Adam(framework.parameters(), lr=learning_rate, weight_decay=weight_decay)
        g1, g2 = framework.g1, framework.g2
        self.opt, self.g1, self.g2, self.framework, self.model, self.ea = opt, g1, g2, framework, model, ea

    def train1step(self, train_epoch):
        opt, g1, g2, framework, model, ea = self.opt, self.g1, self.g2, self.framework, self.model, self.ea
        partition = RandomUniquePartition(ea, batch_size=batch_size, neg_k=neg_k, shuffle=False)

        for epoch in range(train_epoch):
            total_loss = 0
            curr_sample_time = 0.
            curr_forward_time = 0.
            curr_loss_time = 0.
            print(f'Epoch : {epoch}')
            for link, neg1_total, neg2_total in partition.sample():
                loss, sample_time, forward_time = framework(link, neg1_total, neg2_total, fanouts, margin)
                # print('add_loss')
                total_loss += loss.item()
                # print('zero_grad')
                loss_update_time = time.time()
                opt.zero_grad()
                loss.backward()
                # print(loss.item())
                # print('step')
                opt.step()
                loss_update_time = time.time() - loss_update_time
                curr_loss_time += loss_update_time
                curr_sample_time += sample_time
                curr_forward_time += forward_time

            partition.resample_negative_pairs()

            print(f'Loss : {total_loss}', 'SampleTime:', curr_sample_time, 'ForwardTime:', curr_forward_time,
                  'LossTime:',
                  curr_loss_time)
            # acc = testing(ea, model, g1, g2)
            # get_hits(a, b, ea.link)
            # with torch.no_grad():
            #     em1 = model([g1.to(device)] * 2)
            #     em2 = model([g2.to(device)] * 2)
            #     get_hits(em1.cpu(), em2.cpu(), ea.test, norm=True, partition=1)
            #     # framework.eval_with_metis(partition_k)
            #     saveobj((ea, framework.cpu(), em1.cpu(), em2.cpu()), 'tmp/large_ea_results.pkl')
            #     framework = framework.to(device)

    def get_curr_embeddings(self, device='cpu'):
        opt, g1, g2, framework, model, ea = self.opt, self.g1, self.g2, self.framework, self.model, self.ea
        with torch.no_grad():
            em1 = model([g1.to(model.device)] * 2)
            em2 = model([g2.to(model.device)] * 2)
            # get_hits(em1.cpu(), em2.cpu(), ea.test, norm=True, partition=1)
            # framework.eval_with_metis(partition_k)
            # saveobj((ea, framework.cpu(), em1.cpu(), em2.cpu()), 'tmp/large_ea_results.pkl')
            # framework = framework.to(device)
            return em1.to(device), em2.to(device)
