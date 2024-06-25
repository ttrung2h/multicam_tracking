import numpy as np
import os
import itertools
def get_cams(config):
    return list(config.cams.keys())

def read_track_tables(config):
    track_tables = []
    for name in get_cams(config):
        filepath = os.path.join(config["extracting_feature"]["output_path"], f"{name}.npy")
        ttf = np.load(filepath)
        track_tables.append(ttf)
        print("Load", filepath, "with shape", ttf.shape)
    return track_tables

def read_matches(config):
    return read_matches_from_dir(config['match']['out_dir'], get_cams(config))
    
def read_matches_from_dir(path, names):
    matches = {}
    for name1, name2 in itertools.combinations(names, r=2):
        filepath = os.path.join(path, f"{name1}_{name2}.txt")
        f = open(filepath, 'rt')
        matches[(name1, name2)] = eval(f.read())
        print("Read", filepath)
    return matches
