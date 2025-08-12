#
# Created on Mon Jul 25 2022
# Author@ZHONG TAO
# Copyright (c) 2022 CUHK
#

from data_tools.utils.ark_run_tools import ArkRunTools
import logging
import time
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)


class ArkPipe:
    def __init__(self, original_pickle_file, output_dir):
        self.original_pickle_file = original_pickle_file
        self.output_dir = output_dir

    def run(self):
        logging.info("start to generate wav_scp_path")
        wav_scp_dir = f"{self.output_dir}/wav_scp"
        ArkRunTools.generate_wav_scp(pickle_path=self.original_pickle_file, output_dir=wav_scp_dir)
        wav_scp_path = f"{wav_scp_dir}/wav.scp"
        logging.info("finish generating wav_scp_path=%s", wav_scp_path)
        time.sleep(30)

        logging.info("start to generate featdir")
        featdir = f"{self.output_dir}/featdir"
        output = ArkRunTools.ark_run_for_python_wrapper(wav_scp_path=wav_scp_path, featdir=featdir)
        ark_scp_file = f"{featdir}/feats.scp"
        logging.info(output)
        logging.info("finish generating feat_path=%s", ark_scp_file)
        time.sleep(60)

        logging.info("start to generate pickle_path")
        ark_pickle_dir = f"{self.output_dir}/ark_pickle"
        ArkRunTools.generate_ark_pickle(original_pickle_file=self.original_pickle_file, ark_scp_file=ark_scp_file, 
                                        output_dir=ark_pickle_dir)
        logging.info("finish generating pickle_dir=%s", ark_pickle_dir)

        logging.info("Done")

