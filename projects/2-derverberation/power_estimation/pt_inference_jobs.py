import numpy as np
import soundfile
from params import *
from data_generator_jobs import CHDataset
import torch
import os
# from pt_dervb_net import DervbNet
from pt_dervb_net_single_channel import DervbNet
from pt_trainer import get_logger
import sys
import result_analysis as ra
from signalprocess import audiowrite, audioread

gpuid = int(gpu_id)
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2'

logger = get_logger(__name__)

logger.info(f"hop_size:{HOP_SIZE}, SpecM")

class NNetComputer(object):
    def __init__(self, save_model_dir=model_save_dir, gpuid=gpuid):
        print("model name is {}".format(ckpt))
        self.model_name = ckpt
        nnet = DervbNet()
        cpt_fname = os.path.join(save_model_dir, self.model_name)
        cpt = torch.load(cpt_fname, map_location="cpu")
        nnet.load_state_dict(cpt["model_state_dict"])
        logger.info("Load checkpoint from {}, epoch {:d}".format(cpt_fname, cpt["epoch"]))
        self.device = torch.device("cuda:{}".format(gpuid)) if gpuid >= 0 else torch.device("cpu")
        self.nnet = nnet.to(self.device) # if gpuid >= 0 else nnet
        self.nnet.eval()


    def compute(self, mix_wav, lip_video, tgt):
        """data["mix"],data["lip_video"],data["wpe_ref"], data["ori_wav_len"]"""
        wav_len = len(tgt)
        lens = int(wav_len // HOP_SIZE) - 1
        with torch.no_grad():
            mix_wav = torch.tensor(mix_wav, dtype=torch.float32, device=self.device)[None, :, :]
            lip_video = torch.tensor(lip_video, dtype=torch.float32, device=self.device)[None, :, :]
            tgt = torch.tensor(tgt, dtype=torch.float32, device=self.device)[None, None, :]
            ori_wav_len = torch.tensor([lens], dtype=torch.int, device=self.device)
            # import pdb; pdb.set_trace()
            wav_drvb, mag_est, _  = self.nnet([mix_wav, lip_video, tgt, ori_wav_len])
            wav = [np.squeeze(s.detach().cpu().numpy()) for s in wav_drvb]
            return wav


def run(eval_type):
    dataset = CHDataset(stage=eval_type, inference_job_num=int(sys.argv[1]), job=int(sys.argv[2]))
    computer = NNetComputer() 
    pc = ra.MultiChannelPerformance(name=computer.model_name)

    # unvalid_file_name = []
    # wav_scp_dct = {}
    # wav_dir = ''
    # abs_path = os.getcwd()
    for item_idx in range(len(dataset.wav_list)):
        # print("this is the {}-th wav".format(item_idx))
        wav_file = list(dataset.wav_list)[item_idx]

        mix_wav, s1, lip_video, wav_file, ad, spk_rt60 = dataset.get_data(wav_file)
      
        wav = computer.compute(mix_wav, lip_video, s1)
        
        # import pdb; pdb.set_trace()
        # wav = [mix_wav[0]]

        # norm = np.linalg.norm(mix_wav[0], np.inf)
        # norm = np.max(np.abs(mix_wav), axis=1, keepdims=True)
        est_s1 = wav[0]
        est_s1_chn0 = wav[0][0]
        # import pdb; pdb.set_trace()
        # est_s1 = est_s1 * norm / np.max(np.abs(est_s1), axis=1, keepdims=True)
        est_s1 = est_s1.transpose(1, 0)
        
        snr1 = ra.get_SI_SNR(est_s1_chn0, s1)
        pesq = ra.get_PESQ(est_s1_chn0, s1)
        stoi = ra.get_STOI(est_s1_chn0, s1)
        srmr = 0
        if all_metrics:  
            srmr = ra.get_SRMR(est_s1_chn0, s1)
        
        spk_num = 2
        logger.info("{}:{}/nspk:{},ad:{} snr1: {:.2f}, pesq: {:.2f}, stoi: {:.3f}, srmr: {:.2f}".format(wav_file, item_idx+1, spk_num, ad, snr1, pesq, stoi, srmr))
        # print("{}:{}/nspk:{},ad:{} snr1: {:.2f}, pesq: {:.2f}, stoi: {:.3f}, srmr: {:.2f}".format(wav_file, item_idx+1, spk_num, ad, snr1, pesq, stoi, srmr))

        pc.update_performance(wav_file, snr1, ad, spk_num, spk_rt60, metric='SISNR')
        pc.update_performance(wav_file, pesq, ad, spk_num, spk_rt60, metric='PESQ')
        pc.update_performance(wav_file, stoi, ad, spk_num, spk_rt60, metric='STOI')
        if all_metrics:
            pc.update_performance(wav_file, srmr, ad, spk_num, spk_rt60, metric='SRMR')

        if write_wav:
            # write_name = f'{model_save_dir}/{wav_dir_name}'
            # if not os.path.exists(write_name):
            #     os.makedirs(write_name)
          
            audiowrite(est_s1,f"{write_name}/{wav_file}.wav", sampling_rate)
        # if item_idx >= 1242:
        #     break
    pc.summary()
    # print(f"model_save_dir : {model_save_dir}")
            
        

if __name__ == "__main__":
    run(inference_dataset_name)
