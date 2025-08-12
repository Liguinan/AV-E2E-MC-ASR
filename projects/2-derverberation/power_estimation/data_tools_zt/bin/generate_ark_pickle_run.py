#
# Created on Sun Jul 24 2022
# Author@ZHONG TAO
# Copyright (c) 2022 CUHK
#

from data_tools.utils.ark_run_tools import ArkRunTools

CONFIG = {
    "original_pickle_file": "/project_bdda5/bdda/jhxu/separation/choose_trptr_final.pkl",
    "ark_scp_file": "/project_bdda7/bdda/tzhong/projects/Speech-Enhancement-Compression/data/featdir/feats.scp",
    "output_dir": "/project_bdda7/bdda/tzhong/projects/Speech-Enhancement-Compression/data/ark_pickle",
}



def main():
    ArkRunTools.generate_ark_pickle(original_pickle_file=CONFIG["original_pickle_file"], 
            ark_scp_file=CONFIG["ark_scp_file"], output_dir=CONFIG["output_dir"])


if __name__ == '__main__':
    main()