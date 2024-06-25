import numpy as np
import pandas as pd
import os
import cv2
from pathlib import Path
from omegaconf import OmegaConf
from natsort import natsorted
from loguru import logger
from tqdm import tqdm
from typing import List
class Preprocessing:
    def __init__(self, config) -> None:
        self.config = config
        self.MAXIMUM_DISSAPPEARED_FRAME = config.preprocessing.maximum_disappeared_frame

    def interpolate_missing_box(self, data: pd.DataFrame,id, start, end):
        data = data.copy()
        left = start - 1
        right = end + 1        
        left_data = data[data["frame"] == left]
        right_data = data[data["frame"] == right]
        try:
            tt = right_data["x"].values[0]
        except:
            print(right_data['x'].values)
            print(start,end)
            print(id)

        # Get box in frame before start
        x_l, y_l, w_l, h_l, center_x_l, center_y_l, conf_l = (
            left_data["x"].values[0],
            left_data["y"].values[0],
            left_data["w"].values[0],
            left_data["h"].values[0],
            left_data["center_x"].values[0],
            left_data["center_y"].values[0],
            left_data["conf"].values[0],
        )
        # Get box in frame next end
        x_r, y_r, w_r, h_r, center_x_r, center_y_r, conf_r = (
            right_data["x"].values[0],
            right_data["y"].values[0],
            right_data["w"].values[0],
            right_data["h"].values[0],
            right_data["center_x"].values[0],
            right_data["center_y"].values[0],
            right_data["conf"].values[0],
        )

        number_missing_frame = end - start + 1
        # Create a dict to caculate
        extend_dict = {
            "frame" : [],
            "id" : [],
            "x" : [],
            "y" : [],
            "w" : [],
            "h" : [],
            "conf": [],
            "center_x":[],
            "center_y":[],
            "src":[]
        }
        
        for i in range(1,number_missing_frame):
            weight = i / number_missing_frame
            center_x = center_x_l + (center_x_r - center_x_l) * weight
            center_y = center_y_l + (center_y_r - center_y_l) * weight
            w = w_l + (w_r - w_l) * weight
            h = h_l + (h_r - h_l) * weight
            x = center_x - w / 2
            y = center_y - h / 2
            conf = conf_l + (conf_r - conf_l) * weight
            extend_dict['frame'].append(start + i - 1)
            extend_dict['id'].append(id)
            extend_dict['x'].append(x)
            extend_dict['y'].append(y)
            extend_dict['w'].append(w)
            extend_dict['h'].append(h)
            extend_dict['center_x'].append(center_x)
            extend_dict['center_y'].append(center_y)
            extend_dict['conf'].append(conf)
            extend_dict['src'].append(1)
        extend_df = pd.DataFrame(extend_dict)
        return extend_df
    
    def fill_missing_box(self, data: pd.DataFrame):
        data = data.copy()
        # Get unique id
        unique_ids = data["id"].unique()
        # List contain extend data after fill missing box
        extend_data = []
        for id in unique_ids:
            sub_data = data[data["id"] == id]
            # Sort the dataframe by frame
            sub_data = sub_data.sort_values(by="frame")
            
            # Get frames, start_frame and end_frame
            frames = sub_data["frame"].to_numpy()
            start_frame, end_frame = frames.min(), frames.max()
            num_frames = end_frame - start_frame + 1

            # Create a array with missing frame have value is 1 and existing frame have value is 0
            missing_frames = np.ones(num_frames)
            missing_frames[frames - start_frame] = 0
            missing_frames = missing_frames.astype(int).tolist()
            # find continuous missing chain
            missing_chain = self.find_continuous_chain(missing_frames)
            
            # loop through the missing chain
            for start_chain, end_chain in missing_chain:
                num_missing_frame = end_chain - start_chain + 1
                start_missing_frame = start_frame + start_chain
                end_missing_frame = start_frame + end_chain
                # check if the missing chain is less than the maximum dissapear frame
                if num_missing_frame < self.MAXIMUM_DISSAPPEARED_FRAME:
                    extend_df = self.interpolate_missing_box(
                        sub_data,id, start_missing_frame, end_missing_frame
                    )
                    extend_data.append(extend_df)
        
        # check if exist extented data
        if len(extend_data) > 0: 
            # Concat extend data to the original data
            extend_data = pd.concat(extend_data)
            data = pd.concat([data, extend_data])
        data = data.sort_values(by=["frame","id"])
        return data


    def find_continuous_chain(self,missing_frames: List):
        missing_frames = [0] + missing_frames + [0]
        missing_chain = []
        start = 0
        end = 0
        for i in range(1, len(missing_frames)):
            if missing_frames[i] == 1 and missing_frames[i - 1] == 0:
                start = i
            if missing_frames[i] == 0 and missing_frames[i - 1] == 1:
                end = i - 1
                missing_chain.append((start-1, end-1))
        return missing_chain
    
    def format_ouput(self,data):
        data = data.copy()
        data['id'] = data['id'].astype(int)
        data['frame'] = data['frame'].astype(int)
        data['src'] = data['src'].astype(int)
        return data
    
    def caculate_gp(self, data, file_name):
        data = data.copy()
        try:
            hormo_matrix = np.loadtxt(os.path.join(self.config.preprocessing.hormography, file_name))
        except Exception as e:
            logger.error(f"Error: {e}. Can not find hormography matrix for {file_name}")
            return None
        
        x = data['x'].values
        y = data['y'].values

        points = np.array([[x[i], y[i]] for i in range(len(x))], dtype=np.float32)
        transformed_points = cv2.perspectiveTransform(points.reshape(-1, 1, 2), hormo_matrix)
        gpx = transformed_points[:, 0, 0]
        gpy = transformed_points[:, 0, 1]

        data['gpx'] = gpx
        data['gpy'] = gpy
        return data     
        
    def run(self):
        singletrack_output_path = Path(self.config.preprocessing.singletrack_output)
        # Create folder to contain ouput file
        output_path = Path(self.config.preprocessing.output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        for singletrack_output_file in tqdm(natsorted(os.listdir(singletrack_output_path))):
            file_name = os.path.basename(singletrack_output_file)
            logger.info(f"Preprocessing {file_name} ...")
            # Read the output file to dataframe
            raw_data = pd.read_csv(
                os.path.join(singletrack_output_path, singletrack_output_file),
                names=[
                    "frame",
                    "id",
                    "x",
                    "y",
                    "w",
                    "h",
                    "conf",
                    "rm1",
                    "rm2",
                    "rm3",
                ],
            )
            # Remove collumn not used
            raw_data = raw_data.drop(["rm1", "rm2", "rm3"], axis=1)

            # Remove frame out of range
            raw_data = raw_data[raw_data['frame'] < self.config.frames]
            
            # Caculate the center of the box
            raw_data["center_x"] = raw_data["x"] + raw_data["w"] / 2
            raw_data["center_y"] = raw_data["y"] + raw_data["h"] / 2

            # Add coulumn for define filled box or raw box
            raw_data["src"] = 0
            processing_data = self.fill_missing_box(raw_data)
            processing_data = self.caculate_gp(processing_data,file_name)
            processing_data = self.format_ouput(processing_data)
            processing_data.to_csv(output_path / file_name, index=False)
            logger.info(f"Preprocessing {file_name} done save to {output_path / file_name}")

if __name__ == "__main__":
    config_file = "config/garden1.yaml"
    conf = OmegaConf.load(config_file)
    preprocessing = Preprocessing(conf)
    preprocessing.run()