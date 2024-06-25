from extract_feature import Extractor
from preprocessing import Preprocessing
from match import Matcher
from graph_cut import GraphCutter
from omegaconf import OmegaConf
import argparse

def parse_args():
    parser = argparse.ArgumentParser("Pipeline")
    parser.add_argument("--config_file", help="Experiment config file",
                        default="config/garden1.yaml")
    return parser.parse_args()
def main():
    args = parse_args()
    
    if args.config_file == "config/garden1.yaml":
        print("You are using default config file ",str(args.config_file))
    
    config = OmegaConf.load(args.config_file)

    # Preprocessing step
    preprocessing = Preprocessing(config=config)
    preprocessing.run()

    # Extracting feature
    extractor = Extractor(config=config)
    extractor.run()

    # Matching
    matcher = Matcher(config=config)
    matcher.run()

    #Graph cutter
    graph_cutter = GraphCutter(config=config)
    graph_cutter.run()
    

if __name__ == "__main__":
    main()