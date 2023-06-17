#
# Created on Fri Jul 22 2022
# Author@ZHONG TAO
# Copyright (c) 2022 CUHK
#

import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import pickle
import params
from data_tools_zt.utils.ark_run_tools import ArkRunTools
import numpy as np
import librosa


class TasnetDataset(Dataset):
    def __init__(self, annotations_file, max_nspk = 3):
        with open(annotations_file, 'rb') as fp:
            self.info = pickle.load(fp, encoding='utf-8')
        self.wav_list = list(self.info.keys())
        self.max_nspk = max_nspk


    def __len__(self):
        return len(self.wav_list)

    def __getitem__(self, idx):
        wav_file = self.wav_list[idx]
        # wav_path = self.info[wav_file]['wav_path']
        wav_path = self.info[wav_file]["wav_ark_path"]
        wav_info = self.info[wav_file]
        nspk_in_wav = wav_info['n_spk']
        spk_doa = wav_info['spk_doa']
        time_idx = wav_info['time_idx']
        directions = []
        for x in range(nspk_in_wav):
            directions.append(spk_doa[x] * np.pi / 180)
        for x in range(nspk_in_wav, self.max_nspk):
            directions.append(-1)
        directions = np.array(directions).astype(np.float32)

        # import pdb; pdb.set_trace()
        # wav, _ = librosa.load(wav_path, params.sampling_rate, mono=False)
        wav = ArkRunTools.ark_reader(wav_path)
        tgt_beg_idx, tgt_end_idx = int(time_idx[0][0]*params.sampling_rate), int(time_idx[0][1]*params.sampling_rate)
        wav = wav[:, tgt_beg_idx:tgt_end_idx]
        mix_wav = wav[:params.n_mic]
        # Pay Attention, The format is different between lr2 and lr3
        # tgt_reverb_wav = wav[n_mic + nspk_in_wav] # the data from Jianwei Yu
        tgt_reverb_wav = wav[params.n_mic + nspk_in_wav - 1] # rvb clean
        # tgt_reverb_wav = wav[params.n_mic] # dry clean

        lip_video = np.load(wav_info['lipemb_path'][0])
              
        # return mix_wav, [tgt_reverb_wav], directions, nspk_in_wav, lip_video, mix_wav.shape[1], lip_video.shape[0]  
        return np.array(mix_wav).astype(np.float32), np.array([tgt_reverb_wav]).astype(np.float32), directions, nspk_in_wav, lip_video, mix_wav.shape[1], lip_video.shape[0] 