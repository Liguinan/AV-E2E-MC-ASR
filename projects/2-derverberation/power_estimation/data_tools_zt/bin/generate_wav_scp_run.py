#
# Created on Sun Jul 24 2022
# Author@ZHONG TAO
# Copyright (c) 2022 CUHK
#

from data_tools.utils.ark_run_tools import ArkRunTools

CONFIG = {
    "pickle_path": "/project_bdda5/bdda/jhxu/separation/choose_trptr_final.pkl",
    "output_dir": "/project_bdda7/bdda/tzhong/projects/Speech-Enhancement-Compression/data/wav_scp"
}


def main():
    ArkRunTools.generate_wav_scp(pickle_path=CONFIG["pickle_path"], output_dir=CONFIG["output_dir"])


if __name__ == '__main__':
    main()

