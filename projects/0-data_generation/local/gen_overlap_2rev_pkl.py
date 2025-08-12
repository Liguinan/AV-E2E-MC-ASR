#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import pickle
import sys

def save_dict(data, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(data, f)

def main(argv):
    #input_dir = /project_bdda6/bdda/gnli/projects/LRS2/data_prepare/wav_data/17C/test/rvb_only
    input_dir = argv[1]
    # input_azimuth_dir = /project_bdda6/bdda/gnli/projects/LRS2/data_prepare/data/train_mc2
    input_azimuth_dir = argv[2]
    #lrs2_lip_embedding_dir = "/project_bdda3/bdda/jwyu/JianweiYuFrameWork/lrs3/visual_emb_oxford-ori/LRS2/pretrain/emb"
    lrs2_lip_emb_dir = argv[3]
    #output_dir = "/project_bdda6/bdda/gnli/projects/LRS2/data_prepare/pkl_data"
    output_dir = argv[4]
    # dataset = "test"
    dataset = argv[5]
    
    rt60 = argv[6]
    
    # emb 
    lrs2_lip_emb_dct = {}
    with open("{}/wav.scp".format(input_dir), "r", encoding='utf-8') as fr:
        lines = fr.readlines()
        for line in lines:
            if dataset == "test" and line.startswith("rev2"):
                continue
            spk_utter_id = line.split()[0]
            spk_id = spk_utter_id.split("-")[1]
            utter_id = spk_utter_id.split("-")[2].split("_")[0]  # "rev1-5542132598423013227-00023_PRE"
            lip_embedding_path = "{}/{}/{}.npy".format(lrs2_lip_emb_dir, spk_id, utter_id)
            if os.path.isfile(lip_embedding_path):
                lrs2_lip_emb_dct[spk_utter_id] = lip_embedding_path
    
    # input wav
    input_wav_scp_dct = {}
    with open("{}/wav.scp".format(input_dir), "r", encoding='utf-8') as fr:
        lines = fr.readlines()
        for line in lines:
            input_wav_scp_dct[line.split()[0]] = line.split()[1].strip()
    
    # azimuth 
    azimuth_dct = {}
    with open("{}/azimuth.txt".format(input_azimuth_dir), "r", encoding='utf-8') as fr:
        lines = fr.readlines()
        for line in lines:
            line_list = line.split()
            spk_utter_id = line_list[0]
            spk_interference_azimuth = [float(line_list[1]), float(line_list[2])]
            azimuth_dct[spk_utter_id] = spk_interference_azimuth

    # select the wav which has embedding (pretrain has a fewer emb than original wav scp)
    data = {}
    for spk_utter_id, emb in lrs2_lip_emb_dct.items():
        wav_path = input_wav_scp_dct[spk_utter_id]
        spk_interference_azimuth = azimuth_dct[spk_utter_id]
        spk_id = "{}-{}".format(spk_utter_id.split("-")[0], spk_utter_id.split("-")[1]) 
        utter_info = {"lip_path": [],
                      "lipemb_path": [emb],
                      "wav_path": wav_path,
                      "n_spk": 2,
                      "spk_id": [spk_id],
                      "spk_doa": spk_interference_azimuth,
                      "time_idx": [[], []]
                      }
        data[spk_utter_id] = utter_info
    
    save_dict(data, output_dir + "/" + dataset + "_overlap_" + rt60)


if __name__ == '__main__':
    main(sys.argv)
