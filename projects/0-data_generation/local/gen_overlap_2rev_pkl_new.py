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
    # input_utt_dur_dir = /project_bdda4/bdda/gnli/lrs2_data/ori_clean_data/train_mc2
    input_utt_dur_dir = argv[3]
    #lrs2_lip_embedding_dir = "/project_bdda3/bdda/jwyu/JianweiYuFrameWork/lrs3/visual_emb_oxford-ori/LRS2/pretrain/emb"
    lrs2_lip_emb_dir = argv[4]
    #output_dir = "/project_bdda6/bdda/gnli/projects/LRS2/data_prepare/pkl_data"
    output_dir = argv[5]
    # dataset = "test"
    dataset = argv[6]

    
    # tgt_interference_id_dct
    tgt_interference_id_dct = {}
    with open("{}/recording_id.txt".format(input_azimuth_dir), "r", encoding='utf-8') as fr:
        lines = fr.readlines()
        for line in lines:
            line_list = line.split()
            spk_utter_id = line_list[0]
            interference_spk_utter_id= line_list[1]
            tgt_interference_id_dct[spk_utter_id] = interference_spk_utter_id
    
    # emb 
    lrs2_lip_emb_dct = {}
    cnt = 0
    with open("{}/wav.scp".format(input_dir), "r", encoding='utf-8') as fr:
        lines = fr.readlines()
        # import pdb; pdb.set_trace()
        for line in lines:
            line = line.strip()
            spk_utter_id = line.split()[0]
            spk_id = spk_utter_id.split("-")[1]
            utter_id = spk_utter_id.split("-")[2].split("_")[0]  # "rev1-5542132598423013227-00023_PRE"
            
            interference_spk_utter_id = tgt_interference_id_dct[spk_utter_id] #"5542132598423013227-00023_PRE"
            interference_spk_id = interference_spk_utter_id.split("-")[0]
            interference_utter_id = interference_spk_utter_id.split("-")[1].split("_")[0]

            if dataset == 'train_pretrain'or dataset == 'train_pretrain_32':
                if spk_utter_id.endswith("_PRE"):
                    lip_embedding_path = "{}/pretrain/emb/{}/{}.npy".format(lrs2_lip_emb_dir, spk_id, utter_id)
                    if interference_spk_utter_id.endswith("_PRE"):
                        lip_embedding_path_i = "{}/pretrain/emb/{}/{}.npy".format(lrs2_lip_emb_dir, interference_spk_id, interference_utter_id)
                    else:
                        lip_embedding_path_i = "{}/train/emb/{}/{}.npy".format(lrs2_lip_emb_dir, interference_spk_id, interference_utter_id)
                else:
                    lip_embedding_path = "{}/train/emb/{}/{}.npy".format(lrs2_lip_emb_dir, spk_id, utter_id)
                    if interference_spk_utter_id.endswith("_PRE"):
                        lip_embedding_path_i = "{}/pretrain/emb/{}/{}.npy".format(lrs2_lip_emb_dir, interference_spk_id, interference_utter_id)
                    else:
                        lip_embedding_path_i = "{}/train/emb/{}/{}.npy".format(lrs2_lip_emb_dir, interference_spk_id, interference_utter_id)
                    
            elif dataset == 'train':
                lip_embedding_path = "{}/train/emb/{}/{}.npy".format(lrs2_lip_emb_dir, spk_id, utter_id)
                lip_embedding_path_i = "{}/train/emb/{}/{}.npy".format(lrs2_lip_emb_dir, interference_spk_id, interference_utter_id)
            elif dataset == 'val':
                lip_embedding_path = "{}/val/emb/{}/{}.npy".format(lrs2_lip_emb_dir, spk_id, utter_id)
                lip_embedding_path_i = "{}/val/emb/{}/{}.npy".format(lrs2_lip_emb_dir, interference_spk_id, interference_utter_id)
            else:
                lip_embedding_path = "{}/test/emb/{}/{}.npy".format(lrs2_lip_emb_dir, spk_id, utter_id)
                lip_embedding_path_i = "{}/test/emb/{}/{}.npy".format(lrs2_lip_emb_dir, interference_spk_id, interference_utter_id)

            # import pdb; pdb.set_trace()
            # lip_embedding_path = "{}/{}/{}.npy".format(lrs2_lip_emb_dir, spk_id, utter_id)
            # import pdb; pdb.set_trace()
            if not os.path.isfile(lip_embedding_path):
                print("{lip_embedding_path}")
                import pdb; pdb.set_trace()
            if not os.path.isfile(lip_embedding_path_i):
                print("{lip_embedding_path_i}")
                import pdb; pdb.set_trace()
            if os.path.isfile(lip_embedding_path) and os.path.isfile(lip_embedding_path_i):
                cnt += 1
                print(f"effective utterance num: {cnt}")
                lrs2_lip_emb_dct[spk_utter_id] = [lip_embedding_path, lip_embedding_path_i]
    
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
    
    # target utterance duration 
    dur_dct = {}
    with open("{}/utt2dur".format(input_utt_dur_dir), "r", encoding='utf-8') as fr:
        lines = fr.readlines()
        for line in lines:
            line_list = line.split()
            spk_utter_id = line_list[0] # no rev1, rev2, rev3, rev4
            spk_dur = [0.0, float(line_list[1])]
            dur_dct[spk_utter_id] = spk_dur
    
    # interference speaker time idx
    interfernece_time_idx_dct = {}
    with open("{}/interference_time_idx.scp".format(input_azimuth_dir), "r", encoding='utf-8') as fr:
        lines = fr.readlines()
        for line in lines:
            line = line.strip()
            line_list = line.split()
            interference_spk_utter_id = line_list[0] # rev1-xxx, rev2-xxx
            interfernece_time_idx_dct[interference_spk_utter_id] = [float(line_list[2]), float(line_list[3])]


    # rt60 
    rt60_dct = {}
    with open("{}/rt60.txt".format(input_azimuth_dir), "r", encoding='utf-8') as fr:
        lines = fr.readlines()
        for line in lines:
            line = line.strip()
            line_list = line.split()
            spk_utter_id = line_list[0]
            rt60_dct[spk_utter_id] = [float(line_list[1]), float(line_list[2])]

    # select the wav which has embedding (pretrain has a lower emb than original wav scp)
    # import pdb; pdb.set_trace()
    data = {}
    for spk_utter_id, emb in lrs2_lip_emb_dct.items():
        # import pdb; pdb.set_trace
        wav_path = input_wav_scp_dct[spk_utter_id]
        spk_interference_azimuth = azimuth_dct[spk_utter_id]
        spk_dur = dur_dct[spk_utter_id.split('-', 1)[-1]]
        # import pdb; pdb.set_trace()
        interfernece_time_idx = interfernece_time_idx_dct[spk_utter_id]
        rt60 = rt60_dct[spk_utter_id]
        spk_id = spk_utter_id.split("-")[1]
        interfernce_id = tgt_interference_id_dct[spk_utter_id].split("-")[0] #"5542132598423013227-00023_PRE"
        utter_info = {"lip_path": [],
                      "lipemb_path": emb,
                      "wav_path": wav_path,
                      "n_spk": 2,
                      "spk_id": [spk_id, interfernce_id],
                      "spk_doa": spk_interference_azimuth,
                      "time_idx": [spk_dur, interfernece_time_idx],
                      "spk_rt60": rt60
                      }
        data[spk_utter_id] = utter_info
    
    # import pdb; pdb.set_trace()
    if dataset == 'train' or dataset == 'train_pretrain' or dataset == 'train_pretrain_32': 
        suffix = "_rev1"
    else: 
        suffix = "_rev4"
    save_dict(data, output_dir + "/" + dataset + suffix)


if __name__ == '__main__':
    main(sys.argv)

