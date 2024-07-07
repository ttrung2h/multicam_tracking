import numpy as np

class Cluster:
    def __init__(self, V) -> None:
        self.V = V
        self.weights = self._compute_node_weights()
        self.center = self._compute_center() # location feature, shape=(2,)
        self.appearance = self._compute_appearance() # appearance feature, shape=(512,)
        self.idw = self._compute_id_weights() # id weight, a dictionary, e.g. {id1: 0.5, id2: 0.5}
        self.max_conf = self._compute_max_conf()

    def _compute_node_weights(self):
        conf_scores = []
        for v in self.V:
            if 0 < v['conf'] <= 1:
                conf_scores.append(v['conf'])
            else:
                raise Exception(f"Confident score {v['conf']} invalid")
        return np.array(conf_scores) / np.sum(conf_scores)

    def _compute_center(self):
        locs = []
        for v in self.V:
            locs.append(v['loc'])
        return np.average(locs, axis=0, weights=self.weights)

    def _compute_appearance(self):
        apps = []
        for v in self.V:
            apps.append(v['app'])
        return np.average(apps, axis=0, weights=self.weights)

    def _compute_id_weights(self):
        idw = dict()
        for i, v in enumerate(self.V):
            idw[v['gid']] = self.weights[i]
        return idw

    def _compute_max_conf(self):
        return np.max([v['conf'] for v in self.V])

    def __repr__(self) -> str:
        return f"{self.V}"