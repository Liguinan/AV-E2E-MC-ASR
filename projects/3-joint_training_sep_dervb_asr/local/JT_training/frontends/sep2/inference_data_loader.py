#
# Created on Fri Jul 22 2022
# Author@ZHONG TAO
# Copyright (c) 2022 CUHK
#
import torch
import numpy as np
import params
from torch.nn.utils.rnn import pad_sequence
import time


def collate_fn(data):
    """
       data: is a list of tuples with (example, label, length)
             where 'example' is a tensor of arbitrary shape
             and label/length are scalars
    """
    mix_, ref_, doa, spk_num, lip_video_,  mix_lenghs, frames_lenghs, wav_files = zip(*data)
    batch_size = len(data)
    max_len = max(mix_lenghs)
    max_frames = max(frames_lenghs)

    # st = time.time()
    mixs = np.zeros([batch_size, params.n_mic, max_len], dtype=mix_[0].dtype)
    refs = np.zeros([batch_size, 1, max_len], dtype=mix_[0].dtype)
    lip_videos = np.zeros([batch_size, max_frames, 512], dtype=np.float32)

    for cnt in range(len(data)):
        mixs[cnt, :, :mix_[cnt].shape[1]] = mix_[cnt]
        refs[cnt, :, :mix_[cnt].shape[1]] = ref_[cnt]
        lip_videos[cnt, :lip_video_[cnt].shape[0]] = lip_video_[cnt]
    # print("copy time", time.time() - st)

    # we could also torch.nn.utils.rnn.pad_sequence to replace numpy copy. No difference
    # st = time.time()
    # mixs = pad_sequence([torch.from_numpy(_).transpose(0,1) for _ in mix_],  padding_value=0, batch_first=True)
    # refs = pad_sequence([torch.from_numpy(_).transpose(0,1) for _ in ref_],  padding_value=0, batch_first=True)
    # lip_videos = pad_sequence([torch.from_numpy(_) for _ in lip_video_] , batch_first=True)
    # print("copy time", time.time() - st)

    max_len = max_len // params.HOP_SIZE * params.HOP_SIZE
    mixs = mixs[:, :, :max_len]
    refs = refs[:, :, :max_len]
    lip_videos = lip_videos[:, :max_frames]

    data = {}
    # st = time.time()
    # print(mixs.dtype)
    data['mix'] = torch.from_numpy(mixs)
    data['ref'] = torch.from_numpy(refs)
    # data['mix'] = mixs
    # data['ref'] = refs
    data['src_doa'] = torch.Tensor(np.array(doa).astype(np.float32))
    data['spk_num'] = torch.Tensor(np.array(spk_num).astype(np.float32))
    data['lip_video'] = torch.from_numpy(lip_videos)
    data['seq_len'] = mix_lenghs
    data['wav_files'] = wav_files

    # data['lip_video'] = lip_videos
    # print("move time", time.time() - st)

    return data
