from cluster_tracking.cluster import Cluster
from filterpy.kalman import KalmanFilter
import numpy as np

class TrackState:
    INIT = 0
    ONLINE = 1
    OFFLINE = 2
    DELETE = 3

class Track:
    # ID used for creating new track
    last_id = 0

    def __init__(self, cluster: Cluster, min_hits, max_umatch, app_inertia, id_inertia) -> None:
        # Params
        self.MIN_HITS = min_hits
        self.MAX_UNMATCH = max_umatch
        self.APPEARANCE_INERTIA = app_inertia
        self.ID_INERTIA = id_inertia

        # Init
        self.kf = self._init_kf(cluster)
        Track.last_id += 1
        self.id = Track.last_id
        self.last_cluster = cluster
        self.consecutive_matches = 1
        self.consecutive_unmatches = 0
        self.state = TrackState.INIT
        self.appearance = cluster.appearance
        self.idw = cluster.idw

    def _init_kf(self, cluster):
        # Define the Kalman filter model
        kf = KalmanFilter(dim_x=4, dim_z=2)

        # Define the state transition matrix
        dt = 1.0  # time step
        kf.F = np.array([[1, 0, dt, 0],
                        [0, 1, 0, dt],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])

        # Define the measurement function
        kf.H = np.array([[1, 0, 0, 0],
                        [0, 1, 0, 0]])

        # Define the measurement noise covariance matrix
        kf.R = np.array([[0.1, 0],
                        [0, 0.1]])

        # Define the process noise covariance matrix
        kf.Q = np.array([[0.01, 0, 0.01, 0],
                        [0, 0.01, 0, 0.01],
                        [0.01, 0, 0.01, 0],
                        [0, 0.01, 0, 0.01]])

        # Initialize the state vector and covariance matrix
        loc = cluster.center
        kf.x = np.array([loc[0], loc[1], 0, 0]) # x, y, vx, vy
        kf.P = np.eye(4) * 1000

        return kf

    def match_update(self, cluster):
        self.last_cluster = cluster
        self.kf.update(cluster.center)
        self.update_appearance(cluster)
        self.update_id_weight(cluster)
        self.consecutive_matches += 1
        self.consecutive_unmatches = 0
        if self.state == TrackState.INIT and self.consecutive_matches >= self.MIN_HITS:
            self.state = TrackState.ONLINE
        elif self.state == TrackState.ONLINE:
            pass
        elif self.state == TrackState.OFFLINE:
            self.state = TrackState.ONLINE
        elif self.state == TrackState.DELETE:
            raise Exception("Deleted track can not be updated")

    def unmatch_update(self):
        self.consecutive_matches = 0
        self.consecutive_unmatches += 1
        if self.state == TrackState.INIT:
            self.state = TrackState.DELETE
        elif self.state == TrackState.ONLINE:
            self.state = TrackState.OFFLINE
        elif self.state == TrackState.OFFLINE and self.consecutive_unmatches > self.MAX_UNMATCH:
            self.state = TrackState.DELETE
        elif self.state == TrackState.DELETE:
            raise Exception("Deleted track can not be updated")

    def predict(self):
        self.kf.predict()

    def get_loc(self):
        pred_center = self.kf.x[:2] # x, y
        return pred_center

    def update_appearance(self, cluster: Cluster):
        # get appearance inertia coefficient
        a = self.APPEARANCE_INERTIA
        # compute new appearance inertia by confident score of cluster
        c = cluster.max_conf
        m = 0.1 # detection threshold. 
        assert c >= m
        new_a = a + (1-a) * (1 - (c-m)/(1-m))
        # update appearance
        self.appearance = np.dot(new_a, self.appearance) + np.dot(1 - new_a, cluster.appearance)

    def update_id_weight(self, cluster: Cluster):
        a = self.ID_INERTIA
        new_idw = dict()
        id_space = set(self.idw.keys()).union(set(cluster.idw.keys()))
        for id in id_space:
            new_idw[id] = a * self.idw.get(id, 0) + (1 - a) * cluster.idw.get(id, 0)
        if abs(sum([w for id, w in new_idw.items()]) - 1) < 1e-5:
            self.idw = new_idw
        else:
            raise Exception("Sum weight must equal to 1")

    def __repr__(self) -> str:
        return f"Track ID {self.id}"