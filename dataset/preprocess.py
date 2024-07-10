import pandas as pd
import os
import argparse
import shutil

class DataSet:
    def __init__(self, data_dir):
        self.data = data_dir
    def convert_df(self):
        ...
    def save_dfs(self):
        pass
    def run(self):
        pass
    def load_dfs(self):
        pass   

class CAMPUS(DataSet):
    def __init__(self, data_dir):
        super().__init__(data_dir)
        self.data_dir = data_dir

    def convert_df(self,df):
        converted_df = pd.DataFrame()
        df = df.sort_values(by=['frame', 'id'])
        converted_df['frame'] = df['frame']
        converted_df['id'] = df['id']
        converted_df['l'] = df['l']
        converted_df['t'] = df['t']
        converted_df['w'] = df['r'] - df['l']
        converted_df['h'] = df['b'] - df['t']
        converted_df['x'] = df['x']
        converted_df['y'] = df['y']
        converted_df['z'] = df['z']
        return converted_df

    def save_df(self):
        pass
    
    def load_dfs(self,data_path,sep):
        data_dfs = {}
        columns = ['id','l','t','r','b','frame','x','y','z','class']
        #read all data files
        for data_file in os.listdir(data_path):
            if data_file.endswith(".txt"):
                data_dfs[data_file] = pd.read_csv(os.path.join(data_path, data_file), 
                                                  sep=" ", 
                                                  header=None, 
                                                  names=columns
                                                  )
        return data_dfs
          
    def run(self,sep = ","):

        # Check type of separator to create folders
        if sep == ",":
            save_folders = ["gt_csv"]
        elif sep == " ":
            save_folders = ["gt_np"]
        else:
            save_folders =  ["gt_csv", "gt_np"]
            sep = [',' , ' ']

        for i,save_folder in enumerate(save_folders):
            for folder in os.listdir(self.data_dir):
                if os.path.exists(os.path.join(self.data_dir, folder, save_folder)):
                    shutil.rmtree(os.path.join(self.data_dir, folder, save_folder))
                #create new gt folder
                os.makedirs(os.path.join(self.data_dir, folder, save_folder),exist_ok=True)
                
                #read data files
                data_path = os.path.join(self.data_dir, folder)
                # get all data frames and consider name file as key
                data_dfs = self.load_dfs(data_path,sep=sep[i])
                #covert and save data frames
                for key in data_dfs.keys():
                    data_dfs[key] = self.convert_df(data_dfs[key])
                    data_dfs[key].to_csv(os.path.join(self.data_dir, folder, save_folder, key), sep=sep[i], header=False, index=False)
                print(f"DONE: {folder} {save_folder}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process dataset.')
    parser.add_argument("--dataset", help="Name of dataset", default="name of dataset")
    parser.add_argument("--data_dir", help="Data directory", default="data directory")
    parser.add_argument("--sep", help="Separator", default=None)
    args = parser.parse_args()
    if args.dataset == "CAMPUS":
        data = CAMPUS(args.data_dir)
        data.run(args.sep)
    else:
        print("Dataset not supported")