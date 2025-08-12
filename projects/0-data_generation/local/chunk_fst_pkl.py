#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import pickle
import sys


def save_dict(data, name):
    with open(name, 'wb') as f:
        pickle.dump(data, f)


def main(argv):
    input_pkl_file = argv[1]
    output_pkl_file = argv[2]
    utt2num_frames_file = argv[3]

    info = pickle.load(open(input_pkl_file, 'rb'), encoding='utf-8')
    
    
    utt2num_frames_dct = {}
    with open(utt2num_frames_file, "r", encoding='utf-8') as fr:
        lines = fr.readlines()
        for line in lines:
            line = line.strip()
            if line.split()[0].startswith("rev1"):
                utt2num_frames_dct[line.split()[0]] = int(line.split()[1])
    
    utt2num_frames_dct_ordered = sorted(utt2num_frames_dct.items(), key=lambda x: x[1], reverse=True)
    #import pdb; pdb.set_trace()
    idx = 0
    data = {}
    all_utter_num = len(utt2num_frames_dct_ordered)
    while (idx + 5) <= all_utter_num:
        current_utter_frame = utt2num_frames_dct_ordered[idx][1]
        five_th_utter_frame = utt2num_frames_dct_ordered[idx+4][1]
        frame_diff = current_utter_frame - five_th_utter_frame
        if frame_diff <= 2:
            for i in range(5):
                spk_utter_id = utt2num_frames_dct_ordered[idx + i][0]
                data[spk_utter_id] = info[spk_utter_id]
            idx = idx + 5
        else:
            #import pdb; pdb.set_trace()
            idx = idx + 1
    save_dict(data, output_pkl_file)


if __name__ == '__main__':
    main(sys.argv)
