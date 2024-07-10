import numpy as np
import cv2
from torchreid.utils import FeatureExtractor
import torch
import pandas as pd
from omegaconf import OmegaConf
import os
from pathlib import Path
from loguru import logger
class ReIDModel():
    def __init__(self,config):
        self.model_path = config['extracting_feature']['model_path']
        self.batch_size = config['extracting_feature']['batch_size']
        self.model_name = config['extracting_feature']['model_name']
        self.extractor = FeatureExtractor(
            model_name= self.model_name,
            model_path= self.model_path,
            device=config['extracting_feature']['device']
        )

    def batch(self, iterable, n=1):
        l = len(iterable)
        for ndx in range(0, l, n):
            yield iterable[ndx:min(ndx + n, l)]

    def extract(self, imgs):
        features = []
        for batch_imgs in self.batch(imgs, self.batch_size):
            features.append(self.extractor(batch_imgs))
        features = torch.cat(features, 0).cpu()
        return features

class Extractor:
    def __init__(self,config):
        self.config = config
    
    def run(self):
        reid_model = ReIDModel(self.config)
        
        # get cam information
        cams_info = self.config['cams']
        
        #num of frame
        num_frame = self.config['frames']

        # get files after preprocessing
        preprocessing_outdir = self.config['preprocessing']['output_path']

        # create folder for saving feature after extracting
        outdir = self.config['extracting_feature']['output_path']
        Path(outdir).mkdir(parents=True, exist_ok=True)

        # extractor
        for i,cam in enumerate(cams_info):
            cam_path = cams_info[cam]['path']
            cam_name = cams_info[cam]['name']
            width = cams_info[cam]['size'][1]
            height = cams_info[cam]['size'][0]

            logger.info("Extracting feature in "+ cam_name)
            cap = cv2.VideoCapture(cam_path)
            assert cap.isOpened() == True
            frame_idx = 0
            track_df = pd.read_csv(os.path.join(preprocessing_outdir,cam_name+'.txt')) 
            crops = []
            features = []
            while True:
                # check if frame more than limit_frame
                if frame_idx >= num_frame:
                    break
                ret,frame = cap.read()
                if ret == True:
                    for ridx, row in track_df[track_df['frame'] == frame_idx].iterrows():
                        id = row['id']
                        l = int(row['x'])
                        t = int(row['y'])
                        r = int(row['x']+row['w'])
                        b = int(row['y']+row['h'])
                        #if value is negative
                        l = int(max(l,0))
                        t = int(max(t,0))
                        r = int(min(width,r))
                        b = int(min(height,b))
                        crop = frame[t:b,l:r]
                        crops.append(crop)

                        #handle out of memory for crops list
                        if len(crops) == 500:
                            extracted_fear = reid_model.extract(crops)
                            features.append(extracted_fear)
                            crops.clear()
                    frame_idx +=1
                else:
                    assert "Can not read frame : "+ str(frame_idx)
                
                if frame_idx%100 == 0 and frame_idx > 0:
                    logger.info(f"Processed frame {frame_idx}")
            
            # extracting feature of imgs which are still in crops
            extracted_fear = reid_model.extract(crops)
            features.append(extracted_fear)
            crops.clear()

            features = np.concatenate(features,axis = 0)
            
            # concat preprocessing data and extracted feature
            track_data = np.concatenate([track_df.to_numpy(),features],axis =1)
            
            # Save
            filepath = os.path.join(outdir, f"{cam_name}.npy")
            np.save(filepath, track_data) 
            logger.info(f"Save extracted feature of {cam_name} and save in {filepath}.")


if __name__ == "__main__":
    config_file = "config/garden1.yaml"
    config = OmegaConf.load(config_file)
    extractor = Extractor(config=config)
    extractor._run()