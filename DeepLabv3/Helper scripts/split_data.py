
import os
import argparse
import splitfolders 

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

if __name__ == "__main__":
    """ Arguments """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("input_dir", help="input data directory")
    parser.add_argument("output_dir", help="output data directory")
    args = parser.parse_args()

    scrDatasetPath = args.input_dir
    dstSplitDatasetPath = args.output_dir
    create_dir(dstSplitDatasetPath)
    splitfolders.ratio(scrDatasetPath, output=dstSplitDatasetPath, 
                    seed=42,  ratio=(.8, .2,), 
                    group_prefix=None)