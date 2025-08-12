#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
import sys

# generate adding reverberation command for kaldi tool

new_wav_scp_dict = {}
wav_file_command = []


def main(argv):
    #wav_scp_path = "/project_bdda5/bdda/gnli/projects/lrs2/lrs2_kaldi_tdnn_lstm/data/train_multichannel_new1/wav_rvb_noisy.scp"
    wav_scp_path = argv[1]
    #wav_output_path = "/project_bdda3/bdda/gnli/data/LRS2_15C/train/rvb_noisy"
    wav_output_path = argv[2]
    run_cpu_core_num = 30
    with open(wav_scp_path, "r", encoding='utf-8') as fr:
        lines = fr.readlines()
        for line in lines:
            utter_id = line.split(" ", 1)[0]
            prefix = utter_id.split("-")[0]  # rev1 or rev2
            command = line.split(" ", 1)[1].strip()
            ori_wav_name = command.split(" ", 2)[1].strip()
            wav_dir = "{0}/{1}_{2}/{3}".format(wav_output_path, prefix, ori_wav_name.split("/")[-3], ori_wav_name.split("/")[-2])
            if not os.path.exists(wav_dir):
                os.makedirs(wav_dir)
            wav_file_path = wav_dir + "/" + ori_wav_name.split("/")[-1]
            command_list = command.split()
            del command_list[-1]
            del command_list[-1]
            new_command = "{0} {1}".format(" ".join(command_list), wav_file_path)
            wav_file_command.append(new_command)
            new_wav_scp_dict[utter_id] = wav_file_path
    with open(wav_output_path + "/wav.scp", 'w', encoding='utf-8') as fw1:
        for utter_id, wav_file_path in dict.items(new_wav_scp_dict):
            fw1.write("{0} {1}\n".format(utter_id, wav_file_path))

    split_data_dir = wav_output_path + "/splitdata"
    if not os.path.exists(split_data_dir):
        os.makedirs(split_data_dir)

    command_path = ""
    split_num = int(np.floor(len(wav_file_command) / run_cpu_core_num))
    with open(wav_output_path + "/split_num", 'w', encoding='utf-8') as fw:
      fw.write("{}\n".format(split_num))
      
    print("split num is {}".format(split_num))
    for idx, cmd in enumerate(wav_file_command):
        if idx % split_num == 0:
            command_path = "{0}{1}".format(split_data_dir, "/" + str(idx))
        with open(command_path + "-command.sh", 'a', encoding='utf-8') as fw2:
            fw2.write(cmd)
            fw2.write("\n")
            fw2.write("echo {0}\n".format(idx + 1))


if __name__ == '__main__':
    main(sys.argv)
