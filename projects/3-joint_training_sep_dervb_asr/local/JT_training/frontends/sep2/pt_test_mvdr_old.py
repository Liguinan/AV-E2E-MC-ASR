import numpy as np
from params import *
from pt_avnet_mvdr import ConvTasNet
from data_generator_for_test import CHDataset, CHDataLoader
import os
import librosa
import result_analysis_new as ra
import torch
from signalprocess import audiowrite, audioread
import signal
import os
from pt_trainer import get_logger

logger = get_logger(__name__)


class NNetComputer(object):
    def __init__(self, save_model_dir=model_save_dir, gpuid=int(gpu_id)):
        nnet = ConvTasNet(norm=norm,
                          out_spk=out_spk,
                          non_linear=activation_function,
                          causal=causal,
                          cosIPD=True,
                          input_features=input_features,
                          spk_fea_dim=speaker_feature_dim, )
        self.model_name = model_name
        cpt_fname = os.path.join(save_model_dir, self.model_name)
        cpt = torch.load(cpt_fname, map_location="cpu")
        nnet.load_state_dict(cpt["model_state_dict"])
        logger.info("Load checkpoint from {}, epoch {:d}".format(
            cpt_fname, cpt["epoch"]))

        self.device = torch.device(
            "cuda:{}".format(gpuid)) if gpuid >= 0 else torch.device("cpu")
        #print(self.device)
        self.nnet = nnet.to(self.device) if gpuid >= 0 else nnet
        # set eval model
        self.nnet.eval()

    def compute(self, samples, directions, spk_num, lip_video):
        with torch.no_grad():
            raw = torch.tensor(samples, dtype=torch.float32, device=self.device)
            doa = torch.tensor(directions, dtype=torch.float32, device=self.device)
            spk_num = torch.tensor(spk_num, dtype=torch.float32, device=self.device)
            lip_video = torch.tensor(lip_video, dtype=torch.float32, device=self.device)
            separate_samples, _ = self.nnet([raw, doa, spk_num, lip_video, None])  # should have shape [#, NSPK, S]
            # separate_samples, _ = self.nnet([raw, doa, spk_num, lip_video, None])  # should have shape [#, NSPK, S]
            # import pdb; pdb.set_trace()
            separate_samples = [np.squeeze(s.detach().cpu().numpy()) for s in separate_samples]
            return separate_samples


def run(eval_type):
    # cnt = 0
    dataset = CHDataset(stage=eval_type)
    computer = NNetComputer()
    pc = ra.MultiChannelPerformance(name=computer.model_name)
# wav_replay_list
    for item_idx in range(len(dataset.wav_list)):
        wav_file = list(dataset.wav_list)[item_idx]
        mix_wav, s1, directions, spk_num, lip_video, ad, wav_name = dataset.get_data(wav_file)
        # import pdb; pdb.set_trace()
    
        separate_sample = computer.compute(mix_wav, directions, spk_num, lip_video)
        # import pdb; pdb.set_trace()
        # separate_sample = [mix_wav[0]]
        #import pdb; pdb.set_trace()
        norm = np.linalg.norm(mix_wav[0], np.inf)
        est_s1 = separate_sample[0]
        est_s1 = est_s1 * norm / np.max(np.abs(est_s1))
        
        snr1 = ra.get_SI_SNR(est_s1, s1)
        pesq = ra.get_PESQ(est_s1, s1)
        # pesq = 0
        
        logger.info("{}:{}/nspk:{},ad:{} snr1: {:.2f} pesq: {:.2f}".format(wav_file, item_idx+1, spk_num, ad, snr1, pesq))

        pc.update_performance(wav_file, snr1, ad, spk_num, metric='SI-SDRi')
        pc.update_performance(wav_file, pesq, ad, spk_num, metric='PESQ')


        write_name = f'{sum_dir}/val-76-oracle-inference'
        if not os.path.exists(write_name):
            os.makedirs(write_name)
        if write_wav:  # snr1 < 0 or snr2 < 0:
            # audiowrite(est_s1,writename.format(sum_dir, wav_file+".wav"), sampling_rate)
            audiowrite(est_s1,f"{write_name}/{wav_file}.wav", sampling_rate)

        
        if item_idx >= max_test_wav:
            break


    logger.info("Compute over {:d} utterances".format(max_test_wav))
    pc.summary()


if __name__ == "__main__":
    run(inference_dataset_name)
