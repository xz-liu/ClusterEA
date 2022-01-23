from utils import *
from utils_largeea import *
from tqdm import tqdm
from sparse_eval import evaluate_sim_matrix
import torch.nn as nn
import torch.optim as optim


def default(*args, **kwargs):
    pass


class ModelWrapper:
    def __init__(self, name, **kwargs):

        self.tf = True
        print('Model name is', name)
        if name in ['mraea', 'rrea']:
            from .rrea.rrea import TFModelWrapper
            self.model = TFModelWrapper(name, **kwargs)
        elif name == 'gcn-align':
            from .gcn_align import GCNAlignWrapper
            self.model = GCNAlignWrapper(**kwargs)
        elif name == 'dual-amn':
            from .duala.duala_wrapper import DualAMNWrapper
            self.model = DualAMNWrapper(**kwargs)
        elif name in ['dual-large', 'rrea-large']:
            from .duala_sample.wrapper import DualASamplingWrapper
            self.model = DualASamplingWrapper(name,**kwargs)
        else:
            raise NotImplementedError

    def __getattr__(self, item):
        SHARED_METHODS = ['update_trainset',
                          'update_devset',
                          'train1step',
                          'test_train_pair_acc',
                          'get_curr_embeddings',
                          'mraea_iteration'
                          ]
        if item in SHARED_METHODS:
            if self.tf:
                if hasattr(self.model, item):
                    return object.__getattribute__(self.model, item)
                return default
            else:
                return object.__getattribute__(self, '_' + item)
        else:
            return self.__getattribute__(item)
