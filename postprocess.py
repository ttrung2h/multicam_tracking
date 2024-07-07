import os
from pathlib import Path

import pandas as pd
from omegaconf import DictConfig

from cluster_tracking.track import TrackState


class PostProcessor():
    def __init__(self, config) -> None:
        self.config = config

    def change_init_state_to_online_state(self):
        """Change state of beginning frames of online track from INIT to ONLINE"""
        df = self.location_df
        df.sort_values(by=["frame", "id"], inplace=True)
        for id in df['id'].unique():
            subdf = df[df['id'] == id]
            online = subdf[subdf['state'] == TrackState.ONLINE]
            init = subdf[subdf['state'] == TrackState.INIT]
            if len(online) > 0:
                df.loc[init.index, 'state'] = TrackState.ONLINE
                for bdf in self.track_dfs.values():
                    bdf.loc[(bdf['id'] == id) & (bdf['state'] ==
                                                 TrackState.INIT), 'state'] = TrackState.ONLINE

    def re_arrange_IDs(self):
        # Create a mapping of old ID to new ID
        mapping = {}
        last_id = 1
        df = self.location_df
        for id in df['id'].unique():
            mapping[id] = last_id
            last_id += 1

        # Replace old ID with new ID
        self.location_df['id'] = self.location_df['id'].map(mapping)
        for name, df in self.track_dfs.items():
            df['id'] = df['id'].map(mapping)


    def drop_not_online(self):
        """Drop tracks with state is different from ONLINE"""
        # Filter online
        self.location_df = self.location_df[self.location_df['state'] == TrackState.ONLINE]
        for name in self.track_dfs:
            df = self.track_dfs[name]
            self.track_dfs[name] = df[df['state'] == TrackState.ONLINE]

        # Drop state column
        self.location_df.drop(['state'], axis=1, inplace=True)
        for name, df in self.track_dfs.items():
            df.drop(['state'], axis=1, inplace=True)
        

    def run(self):
        config = self.config

        # Read output of tracker
        # Read location
        filepath = config.track_cluster.location_file
        self.location_df = pd.read_csv(filepath)
        print("Read", filepath)
        # Read track
        self.track_dfs = dict()
        for name in config.cams:
            filepath = os.path.join(
                config.track_cluster.track_dir, name + ".txt")
            self.track_dfs[name] = pd.read_csv(filepath)
            print("Read", filepath)

        # Do post process
        print("Processing...")
        print("Changing state...")
        self.change_init_state_to_online_state()
        print("Drop not-online track...")
        self.drop_not_online()
        print("Re-arrange IDs...")
        self.re_arrange_IDs()

        # Write output
        # Location
        filepath = config.postprocess.location_file
        Path(filepath).parent.mkdir(exist_ok=True, parents=True)
        self.location_df.to_csv(filepath, index=False, header=True)
        print("Write", filepath)
        # Track
        outdir = config.postprocess.track_dir
        Path(outdir).mkdir(exist_ok=True, parents=True)
        for name, df in self.track_dfs.items():
            filepath = os.path.join(outdir, name + ".txt")
            df.to_csv(filepath, index=False, header=True)
            print("Write", filepath)
