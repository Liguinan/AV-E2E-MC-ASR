from __future__ import print_function
import numpy as np
import os
from params import *
import torch
import pickle
import librosa
import time
from scipy.spatial.distance import pdist, euclidean
import random

from pt_trainer import get_logger

logger = get_logger(__name__)
class CHDataset(object):
    def __init__(self, stage, inference_job_num, job):
        self.stage = stage
        self.max_nspk = 3
        # if self.stage == 'train_pretrain':
        #     self.info = pickle.load(open(os.path.join(data_dir, "train_pretrain_32_rev1_le6_ark.pkl"), 'rb'), encoding='utf-8')
        # elif self.stage == 'val':
        #     self.info = pickle.load(open(os.path.join(data_dir, "val_rev4_le6_ark.pkl"), 'rb'), encoding='utf-8')
        if self.stage == 'replay':
            # self.info = pickle.load(open(os.path.join(data_dir, "test_rev4.pkl"), 'rb'), encoding='utf-8')
            self.info = pickle.load(open(replay_path, 'rb'), encoding='utf-8')
            self.info_ref = pickle.load(open(test_path, 'rb'), encoding='utf-8')
            new_wav_scp_path = replay_wav_scp
        else: 
            raise ValueError(f"Not support dataset type={self.stage} ")
        
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
        
        self.new_wav_scp = {} # e.g. sep output
        if new_wav_scp:
            with open(new_wav_scp_path, 'r', encoding='utf-8') as fr:
                lines = fr.readlines()
                for line in lines:
                    line = line.strip()
                    line_lst = line.split()
                    if line_lst[0] in self.wav_list: # only save this chunk new wav list
                        self.new_wav_scp[line_lst[0]] = line_lst[1]
        print(f"mode={self.stage}, {len(self.info)} utterance in pickle, new_wav_scp flag is: {new_wav_scp}, count {len(self.new_wav_scp)} in new wav scp")
        
        # self.load_data_dir_list()

        # self.epoch = 0
        # if self.stage == 'tt':
        #     self.wav_list = list(self.info.keys())
        #     self.wav_ref_list = list(self.info_ref.keys())
        #     print(len(self.wav_list))
        #     print(len(self.wav_ref_list))
        # else:
        #     self.load_data_dir_list()
        # self.wav_list = list(self.info.keys())
        # logger.info(f"{self.stage} for test, utterances: {len(self.wav_list)}")
        # print(f"{self.stage} for test, utterances: {len(self.wav_list)}")

    # def load_data_dir_list(self):
    #     # if self.stage == 'tr':
    #     #     self.data_dir = os.path.join(data_dir, "train/mix") 
    #     # elif self.stage == 'cv':
    #     #     self.data_dir = os.path.join(data_dir, "cv/mix") 
    #     # elif self.stage == 'tt':
    #     #     #self.data_dir = os.path.join(data_dir, "test/mix") 
    #     #     self.data_dir = os.path.join(data_dir, "test/mix") 
    #     # #self.wav_list = os.listdir(self.data_dir)
    #     length = {}
    #     # import pdb; pdb.set_trace()
    #     for item in self.info:
    #         # if item.endswith("_PRE"):
    #         #     continue
    #         #import pdb; pdb.set_trace()
    #         dur = self.info[item]['time_idx'][0][1] - self.info[item]['time_idx'][0][0]
    #         if dur < 6: # 32 all data with lip embed
    #             length[item] = (self.info[item]['time_idx'][0][1] - self.info[item]['time_idx'][0][0])
    #        # if length[item] 
    #     # import pdb; pdb.set_trace()
    #     out=sorted(length.items(), key = lambda item: item[1], reverse=False)
    #     #import pdb; pdb.set_trace()
    #     wavlist = [item[0] for item in out]
    #     self.utt_chunk = []
    #     for i in range(len(wavlist)//batch_size):
    #         self.utt_chunk.append(wavlist[i*batch_size:(i+1)*batch_size])
    #     #import pdb; pdb.set_trace()
    #     random.shuffle(self.utt_chunk)
    #     self.wav_list = sum(self.utt_chunk,[])
    #     #self.wav_list = [item[0] for item in out]
    #     #import pdb; pdb.set_trace()
    #     #self.wav_list = ['6365413404556770633-00013']
    #     print(len(self.wav_list))


    @staticmethod
    def angle_difference(a, b):
        # in [0, 180]
        return min(abs(a - b), 360 - abs(a - b))

    def get_closest_ad(self, spk_doas):
        tgt_doa = spk_doas[0]
        min_ad = 181
        for i in range(1, len(spk_doas)):
            ad = self.angle_difference(float(tgt_doa), float(spk_doas[i]))
            if ad <= min_ad:
                min_ad = ad

        return min_ad

    def get_data(self, wav_file):
        '''audio data'''
        st = time.time()
        #wav_path = os.path.join(self.data_dir, wav_file)
        wav_path = self.info[wav_file]['wav_path']
        wav_ref_path = self.info_ref[wav_file]['wav_path']
        wav_info = self.info[wav_file]
        nspk_in_wav = wav_info['n_spk']
        spk_doa = wav_info['spk_doa']
        time_idx = wav_info['time_idx']
        spk_rt60 = self.info_ref[wav_file]['spk_rt60'][0]
        #if nspk_in_wav >=2:
        #   swp = spk_doa[0]
        #   spk_doa[0] = spk_doa[1]
        #   spk_doa[1] = swp
        rate = 1
        target_sampling = random.random()  

        directions = []
        # import pdb; pdb.set_trace()
        for x in range(nspk_in_wav):
            directions.append((float(spk_doa[x])+180) * np.pi / 180)
        for x in range(nspk_in_wav, self.max_nspk):
            directions.append(-1)
        if nspk_in_wav >= 2 and target_sampling > rate:
            swp = directions[1]
            directions[1] = directions[0]
            directions[0] = swp
        directions = np.array(directions).astype(np.float32)

        wav, _ = librosa.load(wav_path, sr=sampling_rate, mono=False)
        wav_ref, _ = librosa.load(wav_ref_path, sr=sampling_rate, mono=False)
        if nspk_in_wav >= 2 and target_sampling > rate:
            tgt_beg_idx, tgt_end_idx = int(time_idx[1][0]*sampling_rate), int(time_idx[1][1]*sampling_rate)
        else:
            tgt_beg_idx, tgt_end_idx = int(time_idx[0][0]*sampling_rate), int(time_idx[0][1]*sampling_rate)
        wav = wav[:, tgt_beg_idx:tgt_end_idx]
        mix_wav = wav[:n_mic]
        if new_wav_scp:
            new_wav, _ = librosa.load(self.new_wav_scp[wav_file], sr=sampling_rate, mono=False)
            mix_wav = new_wav
            
        if nspk_in_wav >= 2 and target_sampling > rate:
            tgt_clean_wav = wav[n_mic+1]
            tgt_reverb_wav = wav[n_mic + nspk_in_wav+1]
        else:
            # tgt_clean_wav = wav[n_mic]
            tgt_clean_wav = wav_ref[n_mic]
            # tgt_reverb_wav = wav[n_mic + nspk_in_wav]
            tgt_reverb_wav = wav[n_mic + nspk_in_wav - 1]

        # if self.stage != 'tt':# and self.stage != 'cv':
        #     min_len = wav.shape[-1] // HOP_SIZE * HOP_SIZE
        #     mix_wav = mix_wav[:, :min_len]
        #     tgt_reverb_wav = tgt_reverb_wav[:min_len]
        # else:
        #     min_len = mix_wav.shape[-1]

        # wav_file_secs = min_len * 1.0 / sampling_rate
        
        if lip_fea == 'pixel':
            lip_path = wav_info['lip_pixel_path'][0]#.encode('utf-8')
            lip_video = np.load(lip_path)#os.path.join(lip_fea_dir, lip_path))  # with shape [height, width, time]
            lip_video = np.transpose(lip_video / 255.0, (2, 0, 1))  # [t, h, w]
            lip_video = np.expand_dims(lip_video, axis=1)  # [t, 1, h, w]
        elif lip_fea == 'landmark':
            lip_path = wav_info['lip_landmark_path'][0]
            lip_video = np.load(lip_path)
            if lip_video.shape[1] != 68 or lip_video.shape[2] != 2:
                print("error:mouth landmark shape is not correct, should be T*68*2")
            lip_landmarks = lip_video[:, 48:68, :]
            norm_dist_face_width = euclidean(lip_video[5, 15, :], lip_video[5,3,:])
            if (norm_dist_face_width < 1e-6):
                norm_dist_face_width = euclidean(lip_video[10, 15, :], lip_video[10,3,:])
            mouth_dists=[]
            for t in range(lip_landmarks.shape[0]):
                mouth_dists.append(pdist(lip_landmarks[t, :, :], metric="euclidean"))
            mouth_dists=np.array(mouth_dists)/norm_dist_face_width
            lip_video = mouth_dists
        elif lip_fea == "lipemb":
            if nspk_in_wav >= 2 and target_sampling > rate:
                lip_video = np.load(wav_info['lipemb_path'][1])
            else:
                lip_video = np.load(wav_info['lipemb_path'][0])
            #lip_video = np.expand_dims(lip_video, axis=1)
        else:
	          raise NotImplementedError


        # if self.stage != 'tt':
        #     return mix_wav, [tgt_reverb_wav], directions, nspk_in_wav, lip_video
        # else:
        #     return mix_wav, tgt_reverb_wav, directions, nspk_in_wav, lip_video, \
        #            self.get_closest_ad(spk_doa), wav_file
        return mix_wav, [tgt_clean_wav, tgt_clean_wav], directions, nspk_in_wav, lip_video, \
                   round(self.get_closest_ad(spk_doa),2), wav_file, spk_rt60


class CHDataLoader(object):
    def __init__(self, stage, batch_size):

        self.dataset = CHDataset(stage)
        self.stage = stage
        self.batch_size = batch_size

        self.reading_idx = 0
        self.reading_chunk_idx = 0
        self.epoch = 0

        self.data_keys = ['mix', 'ref', 'src_doa', 'spk_num', 'lip_video']
        self.seed = np.random.seed(seed)

    def init_batch_data(self):
        batch_data = dict()
        for i in self.data_keys:
            batch_data[i] = list()
        return batch_data

    def next_batch(self):
        st = time.time()
        n_begin = self.reading_idx
        n_end = self.reading_idx + self.batch_size
        new_epoch_flag = False

        if n_end >= len(self.dataset.wav_list) // batch_size * batch_size:
            # rewire the index
            self.epoch += 1
            self.reading_idx = 0
            n_begin = self.reading_idx
            n_end = self.reading_idx + self.batch_size
            new_epoch_flag = True
            if self.stage == 'tr':
                self.dataset.epoch += 1
                random.shuffle(self.dataset.utt_chunk)
                self.dataset.wav_list = sum(self.dataset.utt_chunk,[])
                #np.random.shuffle(list(self.dataset.wav_list))

        self.reading_idx += self.batch_size

        mixs = np.zeros([self.batch_size, n_mic, 40 * sampling_rate], dtype=np.float32)
        refs = np.zeros([self.batch_size, 1, 40 * sampling_rate], dtype=np.float32)
        if lip_fea == 'pixel':
            lip_videos = np.zeros([self.batch_size, 28 * 20 + 2, 1, 112, 112], dtype=np.float32)
        elif lip_fea == 'landmark':
            lip_videos = np.zeros([self.batch_size, 28 * 20 + 2, 190], dtype=np.float32)
        elif lip_fea == 'lipemb':
            lip_videos = np.zeros([self.batch_size, 40 * 20 + 2, 512], dtype=np.float32)


        max_len = 1000000000000
        max_frames = 10000000000000
        doas, spk_nums = [], []
        cnt = 0
        seq_lens =[]
        st = time.time()
        for i in range(n_begin, n_end, 1):
            wav_file = list(self.dataset.wav_list)[i] #lambo
            mix, ref, doa, spk_num, lip_video = self.dataset.get_data(wav_file)

            if mix.shape[1] < max_len: ## ATTENTION! Very dangerous operation
                max_len = mix.shape[1]
            if lip_video.shape[0] < max_frames:
                max_frames = lip_video.shape[0]

            mixs[cnt, :, :mix.shape[1]] = mix
            refs[cnt, :, :mix.shape[1]] = ref
            lip_videos[cnt, :lip_video.shape[0]] = lip_video
            spk_nums.append(spk_num)
            doas.append(doa)
            seq_lens.append(mix.shape[1])
            cnt += 1
        #print('sample time', time.time() - st)
        et = time.time()
        max_len = max_len // HOP_SIZE * HOP_SIZE
        mixs = mixs[:, :, :max_len]
        refs = refs[:, :, :max_len]
        #doas = doas[:, :, :max_len]
        lip_videos = lip_videos[:, :max_frames]

        data = self.init_batch_data()
        data['mix'] = torch.Tensor(mixs)
        data['ref'] = torch.Tensor(refs)
        data['src_doa'] = torch.Tensor(np.array(doas).astype(np.float32))
        data['spk_num'] = torch.Tensor(np.array(spk_nums).astype(np.float32))
        data['lip_video'] = torch.Tensor(lip_videos)
        data["seq_len"] = torch.tensor(seq_lens)
        #import pdb; pdb.set_trace()
        #print('padding+tensor:', time.time() - et)
        return data, new_epoch_flag


if __name__ == '__main__':
    data_loader = CHDataLoader(stage='tr', batch_size=batch_size)

    mix, ref, src_doa, spk_num, lip_video,  = data_loader.dataset.get_data(list(data_loader.dataset.wav_list)[255])
    print(np.shape(mix))
    print(np.shape(ref))
    print(np.shape(src_doa))
    print(np.shape(spk_num))
    print(np.shape(lip_video))
    import pdb; pdb.set_trace()
    import time

    st = time.time()
    for i in range(10):
        batch_data, flag = data_loader.next_batch()
    print('10 batch time:', time.time() - st)
    for key in batch_data.keys():
        print(key)
        print(batch_data[key].shape)
    if flag:
        exit()
