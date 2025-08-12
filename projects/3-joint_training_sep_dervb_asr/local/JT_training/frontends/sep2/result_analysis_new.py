'''
breif:  functions for results analysis
author: lorrygu
date:   2019-01-15
'''

import numpy as np
from pesq import pesq
from pypesq import pesq as nb_pesq
from pystoi.stoi import stoi
import speechmetrics as sm


class MultiChannelPerformance(object):
    def __init__(self, name):
        self.model_name = name
        self.metric = ['SI-SNR', 'PESQ', 'SRMR', 'STOI']

        self.init_metric()
#        self.load_info()
        self.cnt = 0

    def init_metric(self):
        self.metrics = dict()
        for metric in self.metric:
            self.metrics[metric] = {
                'all': [],
                'no mix':[],
		        '2-mix':[],
		        '3-mix':[],
                '0-15': [],
                '15-45': [],
                '45-90': [],
                '90-180': []
            }


    def update_performance(self, wav_file, metric_result, ad, spk_num, metric='SI-SDRi'):
        doa_str = ''
	      #ad = self.get_closest_ad(directions)

        if 0 <= ad <= 15:
            doa_str = '0-15'
        elif 15 < ad <= 45:
            doa_str = '15-45'
        elif 45 < ad <= 90:
            doa_str = '45-90'
        elif 90 < ad <= 180:
            doa_str = '90-180'
 #       else:
 #           doa_str = 'no mix'
        if spk_num == 1:
	          spk_str = 'no mix'
        elif spk_num == 2:
	          spk_str = '2-mix'
        elif spk_num == 3:
	          spk_str = '3-mix'

        # import pdb; pdb.set_trace()
        if isinstance(metric_result, list):
            for result in metric_result:
                # import pdb; pdb.set_trace()
                if doa_str != '':
                    self.metrics[metric][doa_str].append(result)
                self.metrics[metric]['all'].append(result)
                self.metrics[metric][spk_str].append(result)
                self.cnt += 1
        elif isinstance(metric_result, float):
            if doa_str != '':
                self.metrics[metric][doa_str].append(metric_result)
            self.metrics[metric]['all'].append(metric_result)
            self.metrics[metric][spk_str].append(metric_result)
            self.cnt += 1
        else:
            raise TypeError


    def get_SI_SNR(self, s1_est, s2_est, s1, s2):
        snr1 = get_SI_SNR(s1_est, s1)
        snr2 = get_SI_SNR(s2_est, s2)
        snr12 = get_SI_SNR(s1_est, s2)
        snr21 = get_SI_SNR(s2_est, s1)
        perm = False
        if snr1 + snr2 < (snr12 + snr21):
            snr1 = snr21
            snr2 = snr12
            perm = True
        return snr1, snr2, perm

    def get_SI_SNRi(self, s1_est, s2_est, s1, s2, mix):
        snr1 = get_SI_SNRi(s1_est, s1, mix)
        snr2 = get_SI_SNRi(s2_est, s2, mix)
        snr12 = get_SI_SNRi(s1_est, s2, mix)
        snr21 = get_SI_SNRi(s2_est, s1, mix)

        perm = False
        if snr1 + snr2 < (snr12 + snr21):
            snr1 = snr21
            snr2 = snr12
            perm = True

        return snr1, snr2, perm

    def summary(self):
        print('Performance of {}'.format(self.model_name))
        # import pdb; pdb.set_trace()
        for metric in self.metric:
            print(f"{metric}\n")
            for key in self.metrics[metric].keys():
                print('\t{}: {} ({}/{}, {}%)'.format(key, np.mean(self.metrics[metric][key]),
                                                     len(self.metrics[metric][key]),
                                                     self.cnt,
                                                     100.0 * len(self.metrics[metric][key]) / self.cnt))


    def get_closest_ad(self, spk_doas):
        tgt_doa = spk_doas[0]
        min_ad = 181
        for i in range(1, len(spk_doas)):
            ad = self.angle_difference(tgt_doa, spk_doas[i])
            if ad <= min_ad:
                min_ad = ad
 
        return min_ad
    
    @staticmethod
    def angle_difference(a, b):
        # in [0, 180]
        a = a /np.pi * 180
        b =b/np.pi * 180
        return min(abs(a - b), 360 - abs(a - b))

class PerformanceCheck():
    def __init__(self):
        self.n_clip = np.zeros((2), 'float')
        self.loss = np.zeros((2), 'float')

    def update(self, n_clip, loss, acc=np.zeros((2), 'float')):
        n_clip = np.array(n_clip)
        loss = np.array(loss)
        acc = np.array(acc)
        self.n_clip = self.n_clip + n_clip
        self.loss = self.loss + loss * n_clip

    def summarize(self):
        self.loss = self.loss / (self.n_clip + 1e-10)


def zero_mean(x, axis=0):
    x = np.array(x)
    return x - np.mean(x, axis)


def _complex_dot(a, b):
    # a / b shape [t]
    b_h = np.conjugate(b)
    return np.sum(a * b_h)


def get_SDR(s_spec_est, s_spec):
    min_t = min(s_spec_est.shape[0], s_spec.shape[0])  # , mix_mag.shape[0])
    s_spec_est = s_spec_est[:min_t]
    s_spec = s_spec[:min_t]
    s_pow = _complex_dot(s_spec, s_spec)
    s_target = s_spec * _complex_dot(s_spec_est, s_spec) / (s_pow + 1e-20)
    err = s_spec_est - s_target
    sdr = _complex_dot(s_target, s_target) / (_complex_dot(err, err) + 1e-20)
    sdr = 10. * np.log10(sdr + 1e-20)
    return sdr


def get_mag_SDR(s_mag_est, s_mag):
    min_t = min(s_mag_est.shape[0], s_mag.shape[0])  # , mix_mag.shape[0])
    s_mag_est = s_mag_est[:min_t, :]
    s_mag = s_mag[:min_t, :]
    # mix_mag = mix_mag[:min_t, :]
    F = s_mag.shape[1]
    s_target = np.tile(np.sum(s_mag_est * s_mag, axis=1, keepdims=True), np.stack([1, F]))
    s_target = (s_target * s_mag) / (np.tile(np.sum(s_mag * s_mag, axis=1, keepdims=True), np.stack([1, F])) + 1e-20)
    error = s_mag_est - s_target
    sdr = np.sum(np.square(s_target) + 1e-20) / np.sum(np.square(error) + 1e-20)
    sdr = 10 * np.log(sdr + 1e-20) / np.log(10.0)

    return sdr

def get_PESQ(est, ref, pesq_type='NB', sr=16000):
        if pesq_type=='NB':
            return nb_pesq(ref, est, sr)
        if pesq_type=='WB':
            return pesq(sr, ref, est, "wb")

def get_STOI(est, ref, sr=16000):
    return stoi(ref, est, sr, extended=False)

def get_SRMR(est, ref, sr=16000):
    srmr = sm.load('absolute.srmr')
    # import pdb; pdb.set_trace()
    return srmr(est, rate=sr)['srmr'].tolist()[0]

def get_SI_SNR(s_est, s):
    min_len = min(len(s), len(s_est))
    s = zero_mean(s[:min_len])
    s_est = zero_mean(s_est[:min_len])

    s_pow = np.dot(s, s)
    s_target = s * np.dot(s, s_est) / s_pow
    e_noise = s_est - s_target
    si_snr = 10 * np.log10(np.dot(s_target, s_target) / np.dot(e_noise, e_noise) + np.spacing(1))

    return si_snr


def get_SI_SNRi(s_est, s, mix):
    # zero-mean normalize
    min_len = min(len(s), len(s_est), len(mix))
    s = zero_mean(s[:min_len])
    s_est = zero_mean(s_est[:min_len])
    mix = zero_mean(mix[:min_len])

    s_pow = np.dot(s, s)
    mix_target = s * np.dot(s, mix) / s_pow
    mix_noise = mix - mix_target
    ori_si_snr = 10 * np.log10(np.dot(mix_target, mix_target) / np.dot(mix_noise, mix_noise) + np.spacing(1))

    s_target = s * np.dot(s, s_est) / s_pow
    e_noise = s_est - s_target
    si_snr = 10 * np.log10(np.dot(s_target, s_target) / np.dot(e_noise, e_noise) + np.spacing(1))

    return si_snr - ori_si_snr


if __name__ == "__main__":
    exit()
