#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import librosa
import numpy as np
import soundfile
import sys


def wav_scp_2_dict(wav_scp_path):
    wav_scp_dict = {}
    with open(wav_scp_path, "r", encoding='utf-8') as fr1:
        lines = fr1.readlines()
        for line in lines:
            utter_id = line.split()[0].strip()
            value = line.split()[1].strip()
            wav_scp_dict[utter_id] = value
    return wav_scp_dict


def main(argv):
    clean_wav_scp_path = argv[1]
    rvb_only_tgt_wav_scp_path = argv[2]
    input_wav_scp_path = argv[3]
    output_wav_path = argv[4]

    if not os.path.exists(output_wav_path):
        os.makedirs(output_wav_path)

    clean_wav_scp_dict = wav_scp_2_dict(clean_wav_scp_path)
    early_rvb_tgt_wav_scp_dict = wav_scp_2_dict(rvb_only_tgt_wav_scp_path)
    input_wav_scp_dict = wav_scp_2_dict(input_wav_scp_path)

    wav_scp_mc_dict = {}
    cnt = 1
    for utter_id, wav_path in input_wav_scp_dict.items():
        print("This is the {0}-th utterance and utterance id is {1}".format(cnt, utter_id))
        wav_path_list = wav_path.split("/")
        y_rvb_early, sr_rvb_early = librosa.load(early_rvb_tgt_wav_scp_dict[utter_id], mono=False, sr=None)
        y_rvb_input, sr_rvb_input = librosa.load(input_wav_scp_dict[utter_id], mono=False, sr=None)
        clean_utter_id = utter_id.split("-", 1)[1]
        y_clean, sr_clean = librosa.load(clean_wav_scp_dict[clean_utter_id], mono=False, sr=None)
        
        assert y_rvb_input.shape[0] == 15
        assert y_rvb_input.shape == y_rvb_early.shape
        assert y_rvb_input.shape[1] == y_clean.shape[0]

        wav_data_mc = np.row_stack((y_rvb_input, y_clean, y_rvb_early[0]))
        #wav_data_mc = np.row_stack((y_rvb_noisy, y_clean, y_rvb_50))

        wav_file_dir = "{0}/{1}/{2}".format(output_wav_path, wav_path_list[-3], wav_path_list[-2])
        if not os.path.exists(wav_file_dir):
            os.makedirs(wav_file_dir)
        wav_file = wav_file_dir + "/" + wav_path_list[-1]
        soundfile.write(wav_file, np.transpose(wav_data_mc), sr_rvb_input)
        wav_scp_mc_dict[utter_id] = wav_file
        cnt = cnt + 1

    with open(output_wav_path + "/wav.scp", "w", encoding='utf-8') as fw:
        for utter_id, wav_path_file in wav_scp_mc_dict.items():
            fw.write("{0} {1}\n".format(utter_id, wav_path_file))


if __name__ == '__main__':
    main(sys.argv)
