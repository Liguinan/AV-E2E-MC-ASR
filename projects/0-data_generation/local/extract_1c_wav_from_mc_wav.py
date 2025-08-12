#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import pickle
import librosa
import soundfile
import os


def main(argv):
    # multichannel wav scp abs path
    mc_wav_scp_path = argv[1]  # /project_bdda4/bdda/gnli/data/LRS2_MC/test/rvb_only/wav.scp
    # save abs path for single channel wav
    sc_wav_path_prefix = argv[2]   # e.g. /project_bdda3/bdda/gnli/data/LRS2_1C/test/rvb_only
    new_scp = {}
    cnt = 0
    with open(mc_wav_scp_path, "r", encoding='utf-8') as fr:
        lines = fr.readlines()
        for line in lines:
            cnt = cnt + 1
            print(f"this is the {cnt}-th utterance") 
            line_lst = line.strip().split()
            utter_id = line_lst[0]
            #new_utter_id = line_lst[0].split("-", 1)[1]
            wav_path = line_lst[1]
            sc_wav_dir_suffix = "{}/{}".format(wav_path.split("/")[-3], wav_path.split("/")[-2])
            sc_wav_abs_dir = "{}/{}".format(sc_wav_path_prefix, sc_wav_dir_suffix)
            if not os.path.exists(sc_wav_abs_dir):
                os.makedirs(sc_wav_abs_dir)
            y_, sr_ = librosa.load(wav_path, mono=False, sr=None)
            sc_wav_abs_path = "{}/{}".format(sc_wav_abs_dir, wav_path.split("/")[-1])
            soundfile.write(sc_wav_abs_path, y_[0, :], sr_)
            new_scp[utter_id] = sc_wav_abs_path

    with open("{}/wav.scp".format(sc_wav_path_prefix), "w", encoding='utf-8') as fw:
        for key, value in new_scp.items():
            fw.write("{} {}\n".format(key, value))


if __name__ == '__main__':
    main(sys.argv)
