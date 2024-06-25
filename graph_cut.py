import utils
import itertools
import os
from collections import defaultdict
import numpy as np
import pathlib
import json
import networkx as nx
from omegaconf import DictConfig
import utils
class GraphCutter:
    def __init__(self,config) -> None:
        self.config = config
        self.NAMES = utils.get_cams(config)
    def build_graph(self,track_data,matches,frame_idx):
        G = nx.Graph()

        # Create nodes
        
        for name in self.NAMES:
            info = track_data[name]
            for row in info[info[:,0] == frame_idx]:
                id = int(row[1])
                gid = f"{name}.{id}"
                loc = row[[10,11]]
                app = row[12:]
                conf = row[6]
                box = (row[2],row[3],row[4],row[5])
                G.add_node(gid,name = name,id = id,loc = loc,app = app,conf = conf,box = box)

        # Create edges

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
    def most_node_same_color(self, G: nx.Graph):
        n_points = defaultdict(lambda: 0)
        for node, data in G.nodes.items():
            n_points[data['name']] += 1
        return max(n_points.values())
    
    def cut_bridge(self, G, bridge):
        G = nx.Graph(G)
        G.remove_edge(*bridge)
        V1, V2 = list(nx.connected_components(G))
        return G.subgraph(V1), G.subgraph(V2)

    def find_cutting_edges(self, G: nx.Graph):
        cutting_edges = []

        # Find bridges
        bridges = list(nx.bridges(G))
        if len (bridges) > 0:
            # Find optimal bridge
            candidates = []
            for edge in bridges:
                G_copy = G.copy()
                g1, g2 = self.cut_bridge(G_copy, edge)
                candidates.append({
                    "l": max(self.most_node_same_color(g1), self.most_node_same_color(g2)),
                    "v": abs(len(g1.nodes) - len(g2.nodes)),
                    "bridge": edge,
                })
            candidates.sort(key=lambda x: (x["l"], x["v"]))
            optimal_bridge = candidates[0]["bridge"]        
            
            # Cut
            bridge = optimal_bridge
            cutting_edges.append(bridge)
            g1, g2 = self.cut_bridge(G, bridge)
            l1 = self.most_node_same_color(g1)
            l2 = self.most_node_same_color(g2)

            # Cut left/right if needed
            if l1 > 1:
                cutting_edges.extend(self.find_cutting_edges(g1))
            if l2 > 1:
                cutting_edges.extend(self.find_cutting_edges(g2))
        else:
            # Run Stoer Wagner algorithm
            cut_value, partition = nx.stoer_wagner(G)

            # Cut
            g1 = G.subgraph(partition[0])
            g2 = G.subgraph(partition[1])
            remove_edges = G.edges - g1.edges - g2.edges
            cutting_edges.extend(remove_edges)

            # Cut left/right if needed
            l1 = self.most_node_same_color(g1)
            l2 = self.most_node_same_color(g2)
            if l1 > 1:
                cutting_edges.extend(self.find_cutting_edges(g1))
            if l2 > 1:
                cutting_edges.extend(self.find_cutting_edges(g2))
            pass

        return cutting_edges
    
    def run(self):
        # Read track tables
        track_tables = utils.read_track_tables(self.config)
        # Read match pairs
        matches = utils.read_matches(self.config)

        # Find matches to remove
        print("Processing...")
        mark_removed = []
        for frame_idx in range(self.config["frames"]):
            G = self.build_graph(track_tables, matches, frame_idx)

            # For each connected component cc, if cc violate the color constraint, devide it.
            for V in nx.connected_components(G):
                subG = G.subgraph(V)
                k = self.most_node_same_color(subG)
                if k > 1:
                    cutting_edges = self.find_cutting_edges(subG)
                    for edge in cutting_edges:
                        node1 = G.nodes[edge[0]]
                        node2 = G.nodes[edge[1]]
                        mark_removed.append((frame_idx, node1["name"], node1["id"], node2["name"], node2["id"]))

            # Log
            if frame_idx % 100 == 99:
                print(f"Frame {frame_idx}/{self.config['frames']}")

        # Remove matches
        for frame_idx, name1, id1, name2, id2 in mark_removed:
            if name1 > name2:
                name1, name2 = name2, name1
                id1, id2 = id2, id1
            m = matches[(name1, name2)][frame_idx]
            if (id1, id2) in m:
                m.remove((id1, id2))
            else:
                print(f"frame={frame_idx}, match ({name1},{id1})-({name2},{id2})not exist")

        print(f"Remove {len(mark_removed)} edges")

        # Write result
        dir = self.config["match"]["refined_dir"]
        pathlib.Path(dir).mkdir(parents=True, exist_ok=True)
        for name1, name2 in itertools.combinations(self.config["cams"], r=2):
            # Write txt
            path = os.path.join(dir, f"{name1}-{name2}.txt")
            with open(path, 'wt') as f:
                f.write(str(matches[(name1, name2)]))
                print('Write', path)
            # Write json
            filepath = os.path.join(dir, f"{name1}-{name2}.json")
            with open(filepath, 'wt') as f:
                json.dump(matches[(name1, name2)], f)
                print("Write", filepath)
        
        print("Done")



