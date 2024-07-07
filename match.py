import gurobipy as gp
import itertools
import os
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from scipy.spatial.distance import cdist
from scipy.spatial import distance_matrix
from utils import get_cams
from loguru import logger
class Matcher:
    def __init__(self,config) -> None:
        self.config = config
        self.LOCATION_COEF = config.match.params.LOCATION_COEF
        self.APPEARANCE_COEF = 1 - self.LOCATION_COEF
        self.LOCATION_THRESH = config.match.params.LOCATION_THRESH
        self.APPEARANCE_THRESH = config.match.params.APPEARANCE_THRESH
        self.N_FRAMES = config.frames
        self.NAMES = config.cams
    
    def _match(self,location_cost,appearance_cost):
        cost_mat = self.LOCATION_COEF * location_cost + self.APPEARANCE_COEF * appearance_cost

        # Create a new model
        m = gp.Model("matrix1")

        # Disable log output
        m.setParam(gp.GRB.Param.OutputFlag, 0)

        # Create variables
        x = m.addMVar(shape=cost_mat.shape, vtype=gp.GRB.BINARY, name="x")

        # Set objective
        m.setObjective((cost_mat * x).sum() - 1000 * x.sum(), gp.GRB.MINIMIZE)

        # Add constraints
        one0 = np.ones(x.shape[0])
        one1 = np.ones(x.shape[1])
        m.addConstr(x @ one1 <= one0, name="r") # each row has at most one 1
        m.addConstr(x.T @ one0 <= one1, name="c") # each col has at most one 1
        m.addConstr(location_cost * x <= self.LOCATION_THRESH) # location threshold
        m.addConstr(appearance_cost * x <= self.APPEARANCE_THRESH) # appearance threshold

        # Optimize model
        m.optimize()
        matched_indices = np.argwhere(x.X == 1)
        return matched_indices.tolist()
        
    def match_two_camera(self,name1,name2,df1,df2):
        match_per_frame_dict = dict()
        for frame_idx in range(self.N_FRAMES):
            # Get list of points
            subdf1 = df1[df1[:,0]==frame_idx]
            subdf2 = df2[df2[:,0]==frame_idx]
            if len(subdf1) == 0 or len(subdf2) == 0:
                matches = []
            else:
                # List IDs
                ids1, ids2 = subdf1[:,1], subdf2[:,1]

                # Calculate distance matrix
                points1 = subdf1[:,[10,12]]
                points2 = subdf2[:,[10,12]]
                loc_cost_mat = distance_matrix(points1, points2)
                apps1 = subdf1[:,12:]
                apps2 = subdf2[:,12:]
                app_cost_mat = cdist(apps1, apps2, metric='cosine')
                # Match
                loc_cost_mat = loc_cost_mat.astype(int)
                app_cost_mat = (app_cost_mat * 100).astype(int)
                matched_pairs = self._match(loc_cost_mat, app_cost_mat)
                # Record matches
                match_per_frame_dict[frame_idx] = []
                for i, j in matched_pairs:
                    match_per_frame_dict[frame_idx].append((int(ids1[i]), int(ids2[j])))
                if self.config.log == True and frame_idx % 100 == 0:
                    print(f"Matched {len(match_per_frame_dict[frame_idx])} pairs in frame {frame_idx}/{self.N_FRAMES}")
        #Write to file
        self.out_dir = self.config.match.out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        if self.config.log == True:
            out_file = os.path.join(self.out_dir,f"{name1}_{name2}.txt")
            with open(out_file, 'wt') as f:
                f.write(str(match_per_frame_dict))
    def run(self):
        cams = get_cams(self.config)
        input_dir = self.config.extracting_feature.output_path
        
        for name1, name2 in itertools.combinations(cams, 2):
            cam1 = np.load(os.path.join(input_dir, f"{name1}.npy"))
            cam2 = np.load(os.path.join(input_dir, f"{name2}.npy"))
            self.match_two_camera(name1, name2, cam1, cam2)
        
            if self.config.log == True:
                logger.info(f"Matched {name1} and {name2}")

if __name__ == "__main__":
    config_file = "config/garden1.yaml"
    config = OmegaConf.load(config_file)
    matcher = Matcher(config)
    matcher.run()