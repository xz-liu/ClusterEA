from xgboost import XGBClassifier
from utils_largeea import *

from sklearn.neural_network import MLPClassifier


class SKLearnPartition:
    def __init__(self, classifier, **kwargs):
        if classifier == 'xgb':
            self.model = XGBClassifier(tree_method='gpu_hist', gpu_id=0, predictor='gpu_predictor',
                                       updater='grow_gpu_hist',
                                       **kwargs)

    @torch.no_grad()
    def _before_apply(self, features, idx=None):
        if idx is not None:
            features = features[idx]
        if isinstance(features, Tensor):
            features = features.cpu().numpy()
        return features

    def fit(self, feature, idx, target, *args, **kwargs):
        return self.model.fit(self._before_apply(feature, idx), target)

    def predict(self, feature, *args, **kwargs):
        return self.model.predict(self._before_apply(feature))

    def clear_memory(self):
        del self.model
        import gc
        gc.collect()
