import sys
import os
sys.path.insert(0, os.getcwd())

import utils
from typing import List, Tuple
from collections import defaultdict
import pathlib
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import networkx as nx
from omegaconf import DictConfig

from cluster_tracking.cluster import Cluster
from cluster_tracking.track import Track, TrackState

class Tracker:
    def __init__(self, config) -> None:
        self.config = config

        # Parameters
        params = config.track_cluster.params
        self.MIN_HITS = params.MIN_HITS
        self.MAX_UNMATCH = params.MAX_UNMATCH
        self.APPEARANCE_INERTIA = params.APPEARANCE_INERTIA
        self.ID_INERTIA = params.ID_INERTIA
        self.match_coef1 = params.MATCH_COEF_1
        self.match_coef2 = params.MATCH_COEF_2
        self.match_thresh1 = params.MATCH_THRESH_1
        self.match_thresh2 = params.MATCH_THRESH_2

        # Global variables
        self.NAMES = config["cams"]
        self.N_FRAMES = config["frames"]

        # Init states
        Track.last_id = 0
        self.tracks : List[Track] = []
        self.frame_idx = 0  # frame_idx is used only for debugging

    def _embed_idws(self, idws1: List[dict], idws2: List[dict]):     
        # Get id space
        id_space = set()
        for idw in idws1:
            id_space = id_space.union(set(idw.keys()))
        for idw in idws2:
            id_space = id_space.union(set(idw.keys()))
        
        # Create embedding vector
        idws1_e, idws2_e = [], []
        for idw in idws1:
            e = []
            for id in id_space:
                e.append(idw.get(id, 0))
            idws1_e.append(e)
        for idw in idws2:
            e = []
            for id in id_space:
                e.append(idw.get(id, 0))
            idws2_e.append(e)
        
        return idws1_e, idws2_e

    def _match(self, clusters: List[Cluster], tracks: List[Track], coef, thresh) -> Tuple[List[Tuple[Cluster, Track]], List[Cluster], List[Track]]:
        """
        Match clusters to tracks

        Params:
        - clusters: list of clusters to match
        - tracks: list of tracks to match
        - coef: (id importance, location importance, appearance importance), example: (3,2,1), type float
        - thresh: max cost for a pair (id thresh, location thresh, appearance thresh), example: (0.9, 20, 0.5)
        Returns: matched, unmatched_clusters, unmatched_tracks
        - matched: [(cluster_i, track_j), ...]
        - unmatched_clusters: [cluster_i, ...]
        - unmatched_tracks: [track_i, ...]
        """

        if len(clusters) == 0 or len(tracks) == 0:
            return [], clusters, tracks

        # Get cluster attribute (id, loc, app)
        cluster_idws, cluster_locs, cluster_apps = [], [], []
        for cluster in clusters:
            cluster_idws.append(cluster.idw)
            cluster_locs.append(cluster.center)
            cluster_apps.append(cluster.appearance)

        # Get track attribute (id, loc, app)
        track_idws, track_locs, track_apps = [], [], []
        for track in tracks:
            track_idws.append(track.idw)
            track_locs.append(track.get_loc())
            track_apps.append(track.appearance)

        # Compute cost matrices
        cluster_idws_e, track_idws_e = self._embed_idws(cluster_idws, track_idws)
        idw_cost_matrix = cdist(cluster_idws_e, track_idws_e, metric="cosine")
        app_cost_matrix = cdist(cluster_apps, track_apps, metric="cosine")
        loc_cost_matrix = cdist(cluster_locs, track_locs, metric="euclidean")

        # Weighted sum three cost matricies
        # id cost and app cost are \in [0,1]
        # loc_cost can very large, so we devide it by 20
        cost_matrix = coef[0] * idw_cost_matrix + coef[1] * loc_cost_matrix / 20 + coef[2] * app_cost_matrix
        
        # Hungary matching
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Filter out matches with cost > threshold
        matched_indices = []
        for i, j in zip(row_ind, col_ind):
            if idw_cost_matrix[i, j] > thresh[0]:
                continue
            if loc_cost_matrix[i, j] > thresh[1]:
                continue
            if app_cost_matrix[i, j] > thresh[2]:
                continue
            matched_indices.append((i, j))

        # Return
        matched = []
        for i, j in matched_indices:
            matched.append((clusters[i], tracks[j]))
        unmatched_clusters = []
        for cluster in clusters:
            if cluster not in [m[0] for m in matched]:
                unmatched_clusters.append(cluster)
        unmatched_tracks = []
        for track in tracks:
            if track not in [m[1] for m in matched]:
                unmatched_tracks.append(track)
        
        return matched, unmatched_clusters, unmatched_tracks

    def update(self, clusters, frame_idx):
        self.frame_idx = frame_idx

        # Apply Kalman Filter to predict new location
        for track in self.tracks:
            track.predict()
        
        # Classify track by state
        online_tracks, offline_tracks, init_tracks = [], [], []
        for track in self.tracks:
            if track.state == TrackState.ONLINE:
                online_tracks.append(track)
            elif track.state == TrackState.OFFLINE:
                offline_tracks.append(track)
            elif track.state == TrackState.INIT:
                init_tracks.append(track)
            elif track.state == TrackState.DELETE:
                raise Exception("Track with state DELETE should be deleted before")
        
        # Match [detected clusters] vs [online + offline tracks]
        match_res1 = self._match(clusters, online_tracks + offline_tracks, coef=self.match_coef1, thresh=self.match_thresh1)
        matched1, remain_clusters, remain_on_off_tracks = match_res1

        # Match [remain clusters] vs [remain online/offline tracks + init tracks]
        match_res2 = self._match(remain_clusters, remain_on_off_tracks + init_tracks, coef=self.match_coef2, thresh=self.match_thresh2)
        matched2, unmatched_clusters, unmatched_tracks = match_res2

        # Update tracks
        for cluster, track in matched1 + matched2:
            track.match_update(cluster)
        for track in unmatched_tracks:
            track.unmatch_update()

        # Delete old unmatched tracks
        self.tracks = list(filter(lambda track: track.state != TrackState.DELETE, self.tracks))

        # Init new tracks
        for cluster in unmatched_clusters:
            new_track = Track(cluster, self.MIN_HITS, self.MAX_UNMATCH, self.APPEARANCE_INERTIA, self.ID_INERTIA)
            self.tracks.append(new_track)

        # Return        
        return self.tracks


    def build_graph(self, track_tables, matches, frame_idx):
        G = nx.Graph()

        # Create nodes
        for name in self.NAMES:
            ttf = track_tables[name]
            for row in ttf[ttf[:,0]==frame_idx]:
                id = int(row[1])
                gid = f"{name}.{id}"
                loc = row[[9, 10]]
                app = row[11:]
                conf = row[6]
                box = (row[2], row[3], row[4], row[5])
                G.add_node(gid, gid=gid, name=name, id=id, loc=loc, app=app, conf=conf, box=box)

        # Create edges
        for name1, name2 in matches:
            pairs = matches[(name1, name2)][frame_idx]
            for id1, id2 in pairs:
                gid1 = f"{name1}.{id1}"
                gid2 = f"{name2}.{id2}"
                loc1 = G.nodes[gid1]['loc']
                loc2 = G.nodes[gid2]['loc']
                dist = np.linalg.norm(loc1-loc2)
                G.add_edge(gid1, gid2, weight=1/dist)

        return G


    def run(self):
        config = self.config

        # Read input
        track_tables = utils.read_track_tables(config)
        matches = utils.read_refined_matches(config)

        # Output
        cluster_tracks = []
        box_tracks = defaultdict(list)

        # Run tracker
        print("Running tracker...")
        for frame_idx in range(self.N_FRAMES):
            # Get detected clusters
            G = self.build_graph(track_tables, matches, frame_idx)
            clusters = []
            for V in nx.connected_components(G):
                subG = G.subgraph(V)
                cluster = Cluster(list(subG.nodes.values()))
                clusters.append(cluster)
            
            # Update tracker
            tracks = self.update(clusters, frame_idx)
            
            # Record result
            for track in tracks:
                # Cluster track
                x, y = track.last_cluster.center
                cluster_tracks.append({
                    "frame": frame_idx,
                    "id": track.id,
                    "x": int(x),
                    "y": int(y),
                    "state": track.state
                })
                # Box track
                for node in track.last_cluster.V:
                    l, t, w, h = node['box']
                    box_tracks[node['name']].append({
                        "frame": frame_idx,
                        "id": track.id,
                        "old_id": node['id'],
                        "l": int(l),
                        "t": int(t),
                        "w": int(w),
                        "h": int(h),
                        "s": node['conf'],
                        "gpx": int(node['loc'][0]),
                        "gpy": int(node['loc'][1]),
                        "state": track.state
                    })
        
            if frame_idx % 200 == 199:
                print(f"Frame [{frame_idx + 1}/{self.N_FRAMES}]")

        # Create df
        cluster_track_df = pd.DataFrame(cluster_tracks)
        box_track_dfs = dict()
        for name in config['cams']:
            df = pd.DataFrame(box_tracks[name])
            box_track_dfs[name] = df
        
        ## Write results
        # Write cluster tracking result
        location_file = pathlib.Path(config["track_cluster"]["location_file"])
        location_file.parent.mkdir(exist_ok=True, parents=True)
        cluster_track_df.to_csv(location_file, index=False)
        print("Write track cluster result to", location_file)

        # Write box tracking result
        box_track_dir = pathlib.Path(config["track_cluster"]["track_dir"])
        box_track_dir.mkdir(exist_ok=True, parents=True)
        for name, df in box_track_dfs.items():
            filepath = box_track_dir / f"{name}.txt"
            df.to_csv(filepath, index=False)
        print("Write box track results to", box_track_dir)
            
        print("Done")

    
