#
# Created on Wed Jul 27 2022
# Author@ZHONG TAO
# Copyright (c) 2022 CUHK
#

from data_tools.utils.ark_run_tools import ArkRunTools

CONFIG = {
    # "original_pickle_file": "/project_bdda6/bdda/gnli/projects/LRS2_new/data/train_pretrain_32_rev1.pkl",
    # "original_pickle_file": "/project_bdda6/bdda/gnli/projects/LRS2_new/data/val_rev4.pkl",
    "original_pickle_file": "/project_bdda6/bdda/gnli/projects/LRS2_new/data/test_rev4.pkl",
    "max_time": 6,
    "output_dir": "/project_bdda7/bdda/tzhong/projects/Speech-Enhancement-Compression/data/filter_pickle",
}



def main():
    ArkRunTools.time_filter_pickle(original_pickle_file=CONFIG["original_pickle_file"], 
                                    max_time=CONFIG["max_time"], output_dir=CONFIG["output_dir"])


if __name__ == '__main__':
    main()