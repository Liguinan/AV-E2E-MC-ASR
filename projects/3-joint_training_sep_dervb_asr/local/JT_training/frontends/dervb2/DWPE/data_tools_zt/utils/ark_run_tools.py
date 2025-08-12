#
# Created on Sun Jul 24 2022
# Author@ZHONG TAO
# Copyright (c) 2022 CUHK
#


from os import stat
import subprocess
from subprocess import Popen
import pathlib
import pickle
from data_tools_zt.utils.ark_read_scripts import load_kaldi

class ArkRunTools:
    C = 32768

    @staticmethod
    def ark_run_for_python_wrapper(wav_scp_path, featdir):
        path = f"{str(pathlib.Path(__file__).parent.resolve())}/shell_scripts"
        output = subprocess.check_output(['cd %s; ./ark_run_for_python_wrapper.sh  -w %s -f %s' % (path, wav_scp_path, featdir,)], shell=True)
        output = output.decode('utf8')
        return output

    @staticmethod
    def generate_wav_scp(pickle_path, output_dir):
        pathlib.Path(output_dir).mkdir(exist_ok=True, parents=True)
        output_path = f"{output_dir}/wav.scp"
        with open(pickle_path, "rb") as fp, open(output_path, "w") as w_fp:
            data = pickle.load(fp, encoding="utf-8")
            for key in data:
                wav_source_scp_path = data[key]['wav_path']
                # print(wav_source_scp_path)
                w_fp.write(f"{key} {wav_source_scp_path}\n")

    # WARNING: the correct data should be data.T / ArkRunTools.C. Nevertheless , the division costs too much time. We should put it in the network for GPU computation
    @staticmethod
    def ark_reader(ark_input):
        data = load_kaldi(ark_input)
        return data.T

    @staticmethod
    def generate_ark_pickle(original_pickle_file, ark_scp_file, output_dir):
        pathlib.Path(output_dir).mkdir(exist_ok=True, parents=True)
        output_path = f"{output_dir}/{pathlib.Path(original_pickle_file).stem}_ark.pkl"
        ark_scp_mapping = dict()
        with open(ark_scp_file) as fp:
            for line in fp:
                key, scp = line.strip().split(' ')
                assert key not in ark_scp_mapping
                ark_scp_mapping[key] = scp
        with open(original_pickle_file, "rb") as input_fp, open(output_path, "wb") as output_fp:
            data = pickle.load(input_fp, encoding="utf-8")
            for key in data:
                # replace wav_path
                assert 'wav_ark_path' not in data[key]
                data[key].update({
                    'wav_ark_path': ark_scp_mapping[key]
                })
            pickle.dump(data, output_fp)

    @staticmethod
    def time_filter_pickle(original_pickle_file, max_time, output_dir):
        pathlib.Path(output_dir).mkdir(exist_ok=True, parents=True)
        output_path = f"{output_dir}/{pathlib.Path(original_pickle_file).stem}_le{max_time}.pkl"
        filter_data = {}
        with open(original_pickle_file, "rb") as input_fp, open(output_path, "wb") as output_fp:
            data = pickle.load(input_fp, encoding="utf-8")
            for key in data:
                st, et = data[key]['time_idx'][0]
                sp = et - st
                if sp <= max_time:
                    filter_data.update({
                        key: data[key]
                    })
            pickle.dump(filter_data, output_fp)
            
            




