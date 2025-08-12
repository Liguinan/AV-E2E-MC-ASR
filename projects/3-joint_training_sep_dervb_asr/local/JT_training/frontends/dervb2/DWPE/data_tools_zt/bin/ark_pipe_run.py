#
# Created on Sun Jul 24 2022
# Author@ZHONG TAO
# Copyright (c) 2022 CUHK
#


from data_tools.scripts.ark_pipe import ArkPipe
import socket


CONFIG = {
    "original_pickle_file": "/project_bdda7/bdda/tzhong/projects/Speech-Enhancement-Compression/data/filter_pickle/val_rev4_le6.pkl",
    "output_dir": "/project_bdda7/bdda/tzhong/projects/Speech-Enhancement-Compression/data/validation",
}



def main():
    # if socket.gethostname() != 'bdda1.itsc.cuhk.edu.hk':
        # raise Exception("Please use bdda1.itsc.cuhk.edu.hk to avoid disturbing other's work, since it needs a lot of cpus")
    ArkPipe(original_pickle_file=CONFIG["original_pickle_file"], output_dir=CONFIG["output_dir"]).run()


if __name__ == '__main__':
    main()