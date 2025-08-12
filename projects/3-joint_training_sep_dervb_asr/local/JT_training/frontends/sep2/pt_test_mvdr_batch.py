import numpy as np
from joblib import Parallel, delayed

from params import *
from pt_avnet_mvdr import ConvTasNet
from data_generator import CHDataset, CHDataLoader
import os
import librosa
import result_analysis as ra
import torch
from signalprocess import audiowrite, audioread
from data_tools_zt.utils.ark_run_tools import ArkRunTools
import signal
import os

gpuid = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpuid)

from pt_trainer import get_logger, offload_data

logger = get_logger(__name__)


def angle_difference(a, b):
    # in [0, 180]
    return min(abs(a - b), 360 - abs(a - b))

def get_closest_ad(spk_doas, spk_num):
    tgt_doa = spk_doas[0] 
    min_ad = 181
    for i in range(1, spk_num):
        ad = angle_difference(float(tgt_doa) *  180 / np.pi, float(spk_doas[i]) * 180 / np.pi)
        if ad <= min_ad:
            min_ad = ad
    return min_ad

class NNetComputer(object):
    def __init__(self, save_model_dir=model_save_dir, gpuid=0):
        nnet = ConvTasNet(norm=norm,
                          out_spk=out_spk,
                          non_linear=activation_function,
                          causal=causal,
                          cosIPD=True,
                          input_features=input_features,
                          spk_fea_dim=speaker_feature_dim, )
        #self.model_name = "uTGT-LPS_IPD_AF-b128-BN-fd1-fixTrue-f0-cosTrue-sinFalse-best-82.pt.tar"
        #self.model_name = 'uTGT-LPS_IPD_AF-b128-BN-fd1-fixTrue-f0-cosTrue-sinFalse-best-89.pt.tar'
        # V
        # self.model_name = "uTGT-LPS_IPD-b36-BN-fd1-fixTrue-f0-cosTrue-sinFalse-best-47.pt.tar"
        # A
        # self.model_name = "uTGT-LPS_IPD_AF-b36-BN-fd1-fixTrue-f0-cosTrue-sinFalse-best-35.pt.tar"
        # AV
        self.model_name = "uTGT-LPS_IPD_AF-b12-BN-fd1-fixTrue-f0-cosTrue-sinFalse-best-76.pt.tar"
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

    def compute(self, data):
        with torch.no_grad():
            data = offload_data(data, self.device)
            # we need to divie ArkRunTools.C when using ark_scp
            data["mix"] = data["mix"] / ArkRunTools.C
            data["ref"] = data["ref"] / ArkRunTools.C
            # self.logger.info("use_ark_scp=%s", use_ark_scp
            ests, _ = self.nnet([data["mix"], data["src_doa"], data["spk_num"], data["lip_video"], None])
            separate_samples = [np.squeeze(s.detach().cpu().numpy()) for s in ests]
            return separate_samples


item_idx = 0
def run(eval_type):
    # cnt = 0
    # dataset = CHDataset(stage=eval_type)
    computer = NNetComputer()
    pc = ra.MultiChannelPerformance(name=computer.model_name)

    from inference_dataset import TasnetDataset
    from torch.utils.data import DataLoader
    from inference_data_loader import collate_fn
    # dataset_path = "/project_bdda7/bdda/tzhong/projects/Speech-Enhancement-Compression/data/train/ark_pickle/train_pretrain_32_rev1_le6_ark.pkl"
    dataset_path = "/project_bdda7/bdda/tzhong/projects/Speech-Enhancement-Compression/data/validation/ark_pickle/val_rev4_le6_ark.pkl"
    dataset = TasnetDataset(dataset_path)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            collate_fn=collate_fn,
                            num_workers=8,
                            prefetch_factor=4)

    def cal_metric_and_save_wav(mix_wav, ref_wav, separate_sample, wav_file, seq_len, directions, spk_num):
        global item_idx
        # import pdb; pdb.set_trace()
        item_idx += 1
        mix_wav = mix_wav[0, :seq_len] / ArkRunTools.C
        ref_wav = ref_wav[0, :seq_len] / ArkRunTools.C
        separate_sample = separate_sample[:seq_len]

        norm = np.linalg.norm(mix_wav, np.inf)
        est_s1 = separate_sample * norm / np.max(np.abs(separate_sample))

        snr1 = ra.get_SI_SNR(est_s1, ref_wav)
        pesq = ra.get_PESQ(est_s1, ref_wav)
        
        ad = get_closest_ad(directions, int(spk_num))

        # import pdb; pdb.set_trace()
        # import pdb; pdb.Pdb(nosigint=True).set_trace()

        logger.info(
            "{}:{}/nspk:{},ad:{} snr1: {:.2f} pesq: {:.2f}".format(wav_file, item_idx, spk_num, ad, snr1, pesq))
        print("{}:{}/nspk:{},ad:{} snr1: {:.2f} pesq: {:.2f}".format(wav_file, item_idx, spk_num, ad, snr1, pesq))

        pc.update_performance(wav_file, snr1, ad, spk_num, metric='SI-SDRi')
        pc.update_performance(wav_file, pesq, ad, spk_num, metric='PESQ')

        write_name = f'{sum_dir}/val-76'
        if not os.path.exists(write_name):
            os.makedirs(write_name)
        if write_wav:  # snr1 < 0 or snr2 < 0:
            # audiowrite(est_s1,writename.format(sum_dir, wav_file+".wav"), sampling_rate)
            audiowrite(est_s1, f"{write_name}/{wav_file}.wav", sampling_rate)
        
        return snr1, pesq


    cnt = 0
    cnt_batch = 0
    pesq = 0
    sisnr = 0
    for data in dataloader:
        cnt += 1
        cnt_batch += data['mix'].size(0)
        print(f"this is the {cnt}-th batch in {len(dataloader)}, all processed: {cnt_batch} utterances")
        separate_samples = computer.compute(data)

        # if cnt_batch >= 100:
        #     break

        metrics = Parallel(n_jobs=12)(
            delayed(cal_metric_and_save_wav)(mix_wav, ref_wav, separate_sample, wav_file, seq_len, ad, spk_num) for mix_wav, ref_wav, separate_sample, wav_file, seq_len, ad, spk_num in zip(list(data['mix']),
                                                                           list(data["ref"]),
                                                                           list(separate_samples[0]),
                                                                           data['wav_files'],
                                                                           list(data['seq_len']),
                                                                           list(data["src_doa"]),
                                                                           list(data["spk_num"]))
        )
        for metric in metrics:
            sisnr += metric[0]
            pesq += metric[1]


    print(f"sisnr is {sisnr/cnt_batch}, pesq is {pesq/cnt_batch}")
    # import pdb; pdb.set_trace()
    # pc.summary()



if __name__ == "__main__":
    run('tt')
