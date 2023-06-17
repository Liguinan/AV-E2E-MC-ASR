from __future__ import print_function
import numpy as np
import os
from params import *
import torch
import pickle
import librosa
import time
from scipy.spatial.distance import pdist, euclidean
from pt_trainer import get_logger

logger = get_logger(__name__)


class CHDataset(object):
    def __init__(self, stage, inference_job_num, job):
        self.stage = stage
        self.max_nspk = 3
        new_wav_scp_path = None
        if self.stage == 'train_pretrain':
            self.info = pickle.load(open(training_path, 'rb'), encoding='utf-8')
            # new_wav_scp_path = training_wavscp
        elif self.stage == 'val':
            self.info = pickle.load(open(validation_path, 'rb'), encoding='utf-8')
            # new_wav_scp_path = validation_wavscp
        elif self.stage == 'test':
            self.info = pickle.load(open(test_path, 'rb'), encoding='utf-8')
            # new_wav_scp_path = test_wavscp
        # self.load_data_dir_list()
        self.wav_list_all = list(self.info.keys())
        # self.wav_list_all = list(self.info.keys())[0:1243]
        logger.info(f"all_wav_num: {len(self.wav_list_all)}")
        # chunk for parallel generate wav
        chunk_num = len(self.wav_list_all) // inference_job_num
        if inference_job_num == job:
            self.wav_list = self.wav_list_all[chunk_num*(job-1):]
        else:
            self.wav_list = self.wav_list_all[chunk_num*(job-1):chunk_num*(job)]
        print(f"all_wav_num={len(self.wav_list_all)}, inference_job_num={inference_job_num}, job={job}, this_chunk_start_idx={chunk_num*(job-1)}")
        print(f"mode={self.stage}, {len(self.info)} utterance in pickle")

        # self.new_wav_scp = {}  # e.g. sep output
        # if new_wav_scp:
        #     with open(new_wav_scp_path, 'r', encoding='utf-8') as fr:
        #         lines = fr.readlines()
        #         for line in lines:
        #             line = line.strip()
        #             line_lst = line.split()
        #             self.new_wav_scp[line_lst[0]] = line_lst[1]
        # print(f"mode={self.stage}, {len(self.info)} utterance in pickle, new_wav_scp flag is: {new_wav_scp}, count {len(self.new_wav_scp)} in new wav scp")

    # def load_data_dir_list(self):
    #     if self.stage == 'tr':
    #         self.data_dir = os.path.join(data_dir, "train/mix")
    #     elif self.stage == 'cv':
    #         self.data_dir = os.path.join(data_dir, "cv/mix")
    #     elif self.stage == 'tt':
    #         self.data_dir = os.path.join(data_dir, "test/mix")
    #     self.wav_list = list(self.info.keys())
    #     print(len(self.wav_list))

    @staticmethod
    def angle_difference(a, b):
        # in [0, 180]
        return min(abs(a - b), 360 - abs(a - b))

    def get_closest_ad(self, spk_doas):
        tgt_doa = spk_doas[0]
        min_ad = 181
        for i in range(1, len(spk_doas)):
            ad = self.angle_difference(tgt_doa, spk_doas[i])
            if ad <= min_ad:
                min_ad = ad

        return min_ad

    def get_data(self, wav_file):
        '''
        audio data
        '''
        st = time.time()
        # wav_path = os.path.join(self.data_dir, wav_file)
        wav_path = self.info[wav_file]['wav_path']
        wav_info = self.info[wav_file]
        nspk_in_wav = wav_info['n_spk']
        spk_doa = wav_info['spk_doa']
        time_idx = wav_info['time_idx']
        spk_rt60 = wav_info['spk_rt60'][0]

        directions = []
        for x in range(nspk_in_wav):
            directions.append((float(spk_doa[x])) * np.pi / 180)
        for x in range(nspk_in_wav, self.max_nspk):
            directions.append(-1)
        directions = np.array(directions).astype(np.float32)

        wav, _ = librosa.load(wav_path, sr=sampling_rate, mono=False)
        mix_wav = wav[:n_mic]
        # mix_wav = wav[n_mic + 1: n_mic + 2]
        # if new_wav_scp:
        #     new_wav, _ = librosa.load(self.new_wav_scp[wav_file], sr=sampling_rate, mono=False)
        #     mix_wav = new_wav[None, :]

        # ref_clean = wav[n_mic]
        # tgt_50ms_rvb = wav[n_mic + 1] 
        tgt_clean_wav = wav[n_mic]
        # tgt_reverb_wav = wav[n_mic + nspk_in_wav]
        tgt_reverb_wav = wav[n_mic + nspk_in_wav - 1]

        # if self.stage != 'tt':
        #     min_len = wav.shape[-1] // HOP_SIZE * HOP_SIZE
        #     mix_wav = mix_wav[:, :min_len]
        #     tgt_clean = tgt_clean[:min_len]
        #     tgt_50ms_rvb = tgt_50ms_rvb[:min_len]
        # else:
        #     min_len = mix_wav.shape[-1]

        # wav_file_secs = min_len * 1.0 / sampling_rate

        if lip_fea == 'pixel':
            lip_path = wav_info['lip_pixel_path'][0]
            lip_video = np.load(lip_path)
            lip_video = np.transpose(lip_video / 255.0, (2, 0, 1))  # [t, h, w]
            lip_video = np.expand_dims(lip_video, axis=1)  # [t, 1, h, w]
        elif lip_fea == 'landmark':
            lip_path = wav_info['lip_landmark_path'][0]
            lip_video = np.load(lip_path)
            if lip_video.shape[1] != 68 or lip_video.shape[2] != 2:
                print("error:mouth landmark shape is not correct, should be T*68*2")
            lip_landmarks = lip_video[:, 48:68, :]
            norm_dist_face_width = euclidean(lip_video[5, 15, :], lip_video[5, 3, :])
            if (norm_dist_face_width < 1e-6):
                norm_dist_face_width = euclidean(lip_video[10, 15, :], lip_video[10, 3, :])
            mouth_dists = []
            for t in range(lip_landmarks.shape[0]):
                mouth_dists.append(pdist(lip_landmarks[t, :, :], metric="euclidean"))
            mouth_dists = np.array(mouth_dists) / norm_dist_face_width
            lip_video = mouth_dists
        elif lip_fea == "lipemb":
            lip_video = np.load(wav_info['lipemb_path'][0])
        else:
            raise NotImplementedError

        #if self.stage != 'test':
            #return mix_wav, tgt_clean, lip_video, wav_file, tgt_50ms_rvb
        #else:
        # return mix_wav, ref_clean, lip_video, wav_file, round(self.get_closest_ad(spk_doa),2), spk_rt60
        return mix_wav, [tgt_clean_wav, tgt_reverb_wav], directions, nspk_in_wav, lip_video, \
                   round(self.get_closest_ad(spk_doa),2), wav_file, spk_rt60


class CHDataLoader(object):
    def __init__(self, stage, batch_size):

        self.dataset = CHDataset(stage)
        self.stage = stage
        self.batch_size = batch_size

        self.reading_idx = 0
        self.reading_chunk_idx = 0
        self.epoch = 0

        self.data_keys = ['mix', 'tgt_clean', 'lip_video', 'tgt_50ms_rvb', 'ori_wav_len']
        self.seed = np.random.seed(seed)

    def init_batch_data(self):
        batch_data = dict()
        for i in self.data_keys:
            batch_data[i] = list()
        return batch_data

    def next_batch(self):
        n_begin = self.reading_idx
        n_end = self.reading_idx + self.batch_size
        new_epoch_flag = False

        if n_end >= len(self.dataset.wav_list):
            # rewire the index, shuffle the wav_list and drop the last few train samples
            # when it cannot be up to the batch_size
            self.epoch += 1
            self.reading_idx = 0
            n_begin = self.reading_idx
            n_end = self.reading_idx + self.batch_size
            new_epoch_flag = True
            if self.stage == 'tr':
                self.dataset.epoch += 1
                np.random.shuffle(list(self.dataset.wav_list))
            self.reading_idx -= self.batch_size  # otherwise, we lose the first batch in the new epoch

        self.reading_idx += self.batch_size

        mixs = np.zeros([self.batch_size, n_mic, 40 * sampling_rate], dtype=np.float32)
        tgt_cleans = np.zeros([self.batch_size, 1, 40 * sampling_rate], dtype=np.float32)
        tgt_50ms_rvbs = np.zeros([self.batch_size, 1, 40 * sampling_rate], dtype=np.float32)
        ori_wav_len = np.zeros([self.batch_size, 1], dtype=np.int32)
        lip_videos = None
        if lip_fea == 'pixel':
            lip_videos = np.zeros([self.batch_size, 28 * 20 + 2, 1, 112, 112], dtype=np.float32)
        elif lip_fea == 'landmark':
            lip_videos = np.zeros([self.batch_size, 28 * 20 + 2, 190], dtype=np.float32)
        elif lip_fea == 'lipemb':
            lip_videos = np.zeros([self.batch_size, 40 * 20 + 2, 512], dtype=np.float32)

        max_len = 0
        max_frames = 0
        cnt = 0
        for i in range(n_begin, n_end, 1):
            wav_file = list(self.dataset.wav_list)[i]
            mix, tgt_clean, lip_video, wav_file, tgt_50ms_rvb = self.dataset.get_data(wav_file)

            if mix.shape[1] > max_len:
                max_len = mix.shape[1]
            if lip_video.shape[0] > max_frames:
                max_frames = lip_video.shape[0]

            ori_wav_len[i % self.batch_size] = mix.shape[1]
            mixs[cnt, :, :mix.shape[1]] = mix
            tgt_cleans[cnt, :, :mix.shape[1]] = tgt_clean
            tgt_50ms_rvbs[cnt, :, :mix.shape[1]] = tgt_50ms_rvb
            lip_videos[cnt, :lip_video.shape[0]] = lip_video
            cnt += 1

        max_len = max_len // HOP_SIZE * HOP_SIZE
        mixs = mixs[:, :, :max_len]
        tgt_cleans = tgt_cleans[:, :, :max_len]
        tgt_50ms_rvbs = tgt_50ms_rvbs[:, :, :max_len]
        lip_videos = lip_videos[:, :max_frames]

        data = self.init_batch_data()
        data['mix'] = torch.Tensor(mixs)
        data['tgt_clean'] = torch.Tensor(tgt_cleans)
        data['lip_video'] = torch.Tensor(lip_videos)
        data['tgt_50ms_rvb'] = torch.Tensor(tgt_50ms_rvbs)
        data['ori_wav_len'] = torch.Tensor(ori_wav_len).long()

        return data, new_epoch_flag


if __name__ == '__main__':
    pass
