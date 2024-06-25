import cv2
import pandas as pd
from omegaconf import OmegaConf
import os
import sys
from tqdm import tqdm
from loguru import logger
sys.path.append(".") # append root directory to sys.path


class Visualizer:
    def __init__(self,config) -> None:
        self.config = config
        assert len(self.config.input.videos) == len(self.config.input.annotations)
    
    def viz_preprocessing_output(self):
        # Draw annotation to video
        for i,video in tqdm(enumerate(self.config.input.videos)):
            annotation = pd.read_csv(self.config.input.annotations[i])
            annotation = annotation.copy()
            annotation = annotation.sort_values(by="frame")
            num_frames = annotation["frame"].unique()
            cap = cv2.VideoCapture(video)
            output_video_file = os.path.basename(video)
            #check output dir exist or not
            if not os.path.exists(self.config.output.output_dir):
                os.makedirs(self.config.output.output_dir)
            
            # Create video writer
            video_writer = cv2.VideoWriter(
                f"{self.config.output.output_dir}/{output_video_file}",
                cv2.VideoWriter_fourcc(*"mp4v"),
                self.config.output.fps,
                self.config.output.size,
            )
            frame_idx = 0
            # Loop through the video
            while True:
                if frame_idx > num_frames.max():
                    break
                ret,frame = cap.read()
                if not ret:
                    break
                # logger.info(annotation.head())
                sub_df = annotation[annotation["frame"] == frame_idx][['id','x','y','w','h','src']]
                values = sub_df.values
                # Loop through the annotation and draw the box with different color if the src is 0 or 1
                for id,x,y,w,h,src in values:
                    if src == 0:
                        color = (0,0,0)
                    else:
                        color = (0,0,255)
                    x,y,w,h = int(x),int(y),int(w),int(h)
                    cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)
                    cv2.putText(frame,str(id),(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
                video_writer.write(frame)
                if frame_idx % 100 == 0:
                    logger.info(f"Processing frame {frame_idx}")
                frame_idx += 1
            cap.release()
            video_writer.release()
            logger.info(f"Finish preprocessing {video} and save to {self.config.output.output_dir}/{output_video_file}")
    
    def run(self):
        task = self.config.input.task
        if task == 'preprocessing':
            self.viz_preprocessing_output()
        else:
            raise ValueError(f"Task {task} is not supported")

if __name__ == '__main__':
    config = OmegaConf.load('viz/viz_config.yaml')
    visualizer = Visualizer(config)
    visualizer.run()
    logger.info('Done!')