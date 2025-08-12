#
# Created on Sun Jul 24 2022
# Author@ZHONG TAO
# Copyright (c) 2022 CUHK
#


from data_tools.utils.ark_run_tools import ArkRunTools
import socket
import logging
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

CONFIG = {
    "wav_scp_path": "/project_bdda7/bdda/tzhong/projects/Speech-Enhancement-Compression/data/wav_scp/wav.scp",
    "featdir": "/project_bdda7/bdda/tzhong/projects/Speech-Enhancement-Compression/data/featdir"
}


def main():
    logging.info('Start')
    # if socket.gethostname() != 'bdda1.itsc.cuhk.edu.hk':
        # raise Exception("Please use bdda1.itsc.cuhk.edu.hk to avoid disturbing other's work, since it needs a lot of cpus")
    output = ArkRunTools.ark_run_for_python_wrapper(wav_scp_path=CONFIG["wav_scp_path"], featdir=CONFIG["featdir"])
    logging.info(output)
    logging.info("Done")


if __name__ == '__main__':
    main()