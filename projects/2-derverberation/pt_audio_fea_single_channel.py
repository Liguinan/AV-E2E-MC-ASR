from params import *
import math
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np


def init_kernel(frame_len,
                frame_hop,
                num_fft=None,
                window="sqrt_hann"):
    if window != "sqrt_hann":
        raise RuntimeError("Now only support sqrt hanning window in order "
                           "to make signal perfectly reconstructed")
    if not num_fft:
        # FFT points
        fft_size = 2 ** math.ceil(math.log2(frame_len))
    else:
        fft_size = num_fft
    # window [window_length]
    window = torch.hann_window(frame_len) ** 0.5
    S_ = 0.5 * (fft_size * fft_size / frame_hop) ** 0.5
    # import pdb; pdb.set_trace()
    # window_length, F, 2 (real+imag)
    # import pdb; pdb.set_trace()
    # kernel = torch.rfft(torch.eye(fft_size) / S_, 1)[:frame_len]
    # print("torch.fft.rfft")
    kernel = torch.fft.rfft(torch.eye(fft_size) / S_, dim=-1)[:frame_len]
    kernel = torch.stack((kernel.real, kernel.imag), -1)
    # 2, F, window_length
    kernel = torch.transpose(kernel, 0, 2) * window
    # 2F, 1, window_length
    kernel = torch.reshape(kernel, (fft_size + 2, 1, frame_len))
    return kernel


class STFTBase(nn.Module):
    """
    Base layer for (i)STFT
    NOTE:
        1) Recommend sqrt_hann window with 2**N frame length, because it
           could achieve perfect reconstruction after overlap-add
        2) Now haven't consider padding problems yet
    """

    def __init__(self,
                 frame_len,
                 frame_hop,
                 window="sqrt_hann",
                 num_fft=None):
        super(STFTBase, self).__init__()
        K = init_kernel(
            frame_len,
            frame_hop,
            num_fft=num_fft,
            window=window)
        self.K = nn.Parameter(K, requires_grad=False)
        self.stride = frame_hop
        self.window = window

    def freeze(self):
        self.K.requires_grad = False

    def unfreeze(self):
        self.K.requires_grad = True

    def check_nan(self):
        num_nan = torch.sum(torch.isnan(self.K))
        if num_nan:
            raise RuntimeError(
                "detect nan in STFT kernels: {:d}".format(num_nan))

    def extra_repr(self):
        return "window={0}, stride={1}, requires_grad={2}, kernel_size={3[0]}x{3[2]}".format(
            self.window, self.stride, self.K.requires_grad, self.K.shape)


class STFT(STFTBase):
    """
    Short-time Fourier Transform as a Layer
    """

    def __init__(self, *args, **kwargs):
        super(STFT, self).__init__(*args, **kwargs)

    def forward(self, x):
        """
        Accept raw waveform and output magnitude and phase
        x: input signal, N x 1 x S or N x S
        m: magnitude, N x F x T
        p: phase, N x F x T
        """
        if x.dim() not in [2, 3]:
            raise RuntimeError("Expect 2D/3D tensor, but got {:d}D".format(
                x.dim()))
        self.check_nan()
        # if N x S, reshape N x 1 x S
        if x.dim() == 2:
            x = torch.unsqueeze(x, 1)
        # N x 2F x T
        c = F.conv1d(x, self.K, stride=self.stride, padding=0)
        # N x F x T
        r, i = torch.chunk(c, 2, dim=1)
        # import pdb; pdb.set_trace()
        m = (r ** 2 + i ** 2 + 1e-10) ** 0.5
        # m = (r ** 2 + i ** 2) ** 0.5
        # import pdb; pdb.set_trace()
        p = torch.atan2(i, r)
        return m, p


class iSTFT(STFTBase):
    """
    Inverse Short-time Fourier Transform as a Layer
    """

    def __init__(self, *args, **kwargs):
        super(iSTFT, self).__init__(*args, **kwargs)

    def forward(self, m, p, squeeze=False):
        """
        Accept phase & magnitude and output raw waveform
        m, p: N x F x T
        s: N x C x S
        """
        if p.dim() != m.dim() or p.dim() not in [2, 3]:
            raise RuntimeError("Expect 2D/3D tensor, but got {:d}D".format(
                p.dim()))
        self.check_nan()
        # if F x T, reshape 1 x F x T
        if p.dim() == 2:
            p = torch.unsqueeze(p, 0)
            m = torch.unsqueeze(m, 0)
        r = m * torch.cos(p)
        i = m * torch.sin(p)
        # N x 2F x T
        c = torch.cat([r, i], dim=1)
        # N x 2F x T
        s = F.conv_transpose1d(c, self.K, stride=self.stride, padding=0)
        if squeeze:
            s = torch.squeeze(s)
        return s


class ChannelWiseLayerNorm(nn.LayerNorm):
    """
    Channel wise layer normalization
    """

    def __init__(self, *args, **kwargs):
        super(ChannelWiseLayerNorm, self).__init__(*args, **kwargs)

    def forward(self, x):
        """
        x: BS x N x K
        """
        if x.dim() != 3:
            raise RuntimeError("{} accept 3D tensor as input".format(
                self.__name__))
        # BS x N x K => BS x K x N
        x = torch.transpose(x, 1, 2)
        x = super(ChannelWiseLayerNorm, self).forward(x)
        x = torch.transpose(x, 1, 2)
        return x


class DFComputer(nn.Module):
    def __init__(self,
                 frame_len=512,
                 frame_hop=256,
                 in_feature=['LPS'],
                 merge_mode='sum',
                 speaker_feature_dim=1,
                 cosIPD=True,
                 sinIPD=False):
        super(DFComputer, self).__init__()
        self.cosIPD = cosIPD
        self.sinIPD = sinIPD
        self.init_mic_pos()
        self.mic_pairs = np.array(mic_pairs)
        self.input_feature = in_feature
        self.spk_fea_dim = speaker_feature_dim
        self.spk_fea_merge_mode = merge_mode
        self.ipd_left = [t[0] for t in mic_pairs]
        self.ipd_right = [t[1] for t in mic_pairs]
        self.n_mic_pairs = self.mic_pairs.shape[0]
        self.num_bins = NEFF
        # print(self.num_bins)
        self.epsilon = 1e-8

        # calculate DF dimension
        self.df_dim = 0

        self.stft = STFT(frame_len=frame_len, frame_hop=frame_hop, num_fft=FFT_SIZE)

        if 'LPS' in self.input_feature:
            self.df_dim += self.num_bins
            self.ln_LPS = ChannelWiseLayerNorm(self.num_bins)

        if 'IPD' in self.input_feature:
            self.df_dim += self.num_bins * self.n_mic_pairs
            if self.sinIPD:
                self.df_dim += self.num_bins * self.n_mic_pairs
        if 'AF' in self.input_feature:
            self.df_dim += self.num_bins * speaker_feature_dim

        if 'DPR' in self.input_feature:
            self.ds_bf = DSBeamformer(sampling_rate)
            self.df_dim += self.num_bins * speaker_feature_dim

            self.stv_real = []
            self.stv_imag = []
            for degree in range(0, 360, self.ds_bf.spatial_resolution):
                stv = self.ds_bf.get_stv([degree * math.pi / 180], self.num_bins)
                self.stv_real.append(np.real(stv[0]))
                self.stv_imag.append(np.imag(stv[0]))
            # with shape [A, M, F]
            self.w_ds_real = nn.Parameter(torch.Tensor(np.array(self.stv_real) / self.n_mic_pairs), requires_grad=False)
            self.w_ds_imag = nn.Parameter(torch.Tensor(np.array(self.stv_imag) / self.n_mic_pairs), requires_grad=False)


    def init_mic_pos(self):
        self.radius = 0.10
        self.mic_position = np.array([-0.28, -0.21, -0.15, -0.10, -0.06, -0.03, -0.01, 0.0,
                                      0.01, 0.03, 0.06, 0.10, 0.15, 0.21, 0.28]) #lambo
        self.n_mic = self.mic_position.shape[0]
        distance = np.zeros([self.n_mic, self.n_mic])
        for i in range(self.n_mic):
            distance[i] = [abs(self.mic_position[i] - self.mic_position[j]) for j in range(self.n_mic)]
        self.mic_distance = distance

    def forward(self, all):
        """
        Compute directional features.
        :param all:
        [0] x - input mixture waveform, with shape [batch_size (B), n_channel (M), seq_len (S)]
        [1] directions - all speakers' directions with shape [batch_size (B), n_spk (C)]
        [2] spk_num - actual speaker number in current wav [batch_size (B)]
        :return: spatial features & directional features, with shape [batch size (B), ?, K]
        """
        # import pdb; pdb.set_trace()
        # analyzing directional features
        x = all[0]
        #directions = all[1]
        #nspk = all[2]

        batch_size, n_channel, S_ = x.shape
        # B, C, t -> BxC, t
        all_s = x.view(-1, S_)
        # BxC, F, T
        magnitude, phase = self.stft(all_s)
        _, F_, K_ = phase.shape
        # B, C, F, T
        phase = phase.view(batch_size, n_channel, F_, K_)
        magnitude = magnitude.view(batch_size, n_channel, F_, K_)

        df = []
        if 'LPS' in self.input_feature:
            lps = torch.log(magnitude[:, 0] ** 2 + self.epsilon)
            lps = self.ln_LPS(lps)
            df.append(lps)

        if 'IPD' in self.input_feature:
            # compute IPD
            # B, I, F, K
            cos_ipd, sin_ipd = self.compute_ipd(phase)
            # => B, IF, K
            cos_ipd, sin_ipd = cos_ipd.view(batch_size, -1, K_), sin_ipd.view(batch_size, -1, K_)
            df.append(cos_ipd)
            if sinIPD:
                df.append(sin_ipd)

        if 'AF' in self.input_feature:
            # compute ipd for AF
            ipd4AF = phase[:, self.ipd_left] - phase[:, self.ipd_right]
            ipd4AF_real = torch.cos(ipd4AF) / self.n_mic_pairs  # [B, I, F, K]
            ipd4AF_imag = torch.sin(ipd4AF) / self.n_mic_pairs

            # get steering vector [B, M, C, F]
            stv_real, stv_imag = self.get_stv(directions)
            AF = self.get_AF(stv_real, stv_imag, ipd4AF_real, ipd4AF_imag, nspk)
            df.append(AF)

        if 'DPR' in self.input_feature:
            real_part = magnitude * torch.cos(phase)
            imag_part = magnitude * torch.sin(phase)
            F1 = self.get_DPR(real_part, imag_part, directions, nspk)
            df.append(F1)

        df = torch.cat(df, dim=1)

        return df, magnitude, phase

    def compute_ipd(self, phase):
        '''phase [B, M, F, K], return IPD [B, I, F, K]'''
        cos_ipd = torch.cos(phase[:, self.ipd_left] - phase[:, self.ipd_right])
        sin_ipd = torch.sin(phase[:, self.ipd_left] - phase[:, self.ipd_right])
        return cos_ipd, sin_ipd

    def get_angle4pair(self, a, b):
        if a < b:  # a is on the left of b
            return math.pi  # 270 / 180 * math.pi
        else:  # a is on the right of b
            return 0  # 90 / 180 * math.pi

    def get_stv(self, angles, sndvelocity=340):
        '''
        :param angles: [Batch_size, max_nspk]
        :return: steering vector [Batch, n_channel, nspk, n_bins]
        '''
        dis = []
        for pair in self.mic_pairs:
            pair_d = self.mic_distance[pair[0], pair[1]]
            angle_between_face_d = angles - self.get_angle4pair(self.mic_position[pair[0]],
                                                                self.mic_position[pair[1]]) # angle
            dis_ = pair_d * torch.cos(angle_between_face_d)
            dis.append(dis_)
        # [BS, M, NSPK]
        distances = torch.stack(dis, dim=1)  # for ipd  [4, 5]
        deltas = distances / sndvelocity * sampling_rate
        stv_real = []
        stv_imag = []
        for f in range(self.num_bins):
            stv_real.append(torch.cos(deltas * math.pi * f / (self.num_bins - 1)))
            stv_imag.append(torch.sin(deltas * math.pi * f / (self.num_bins - 1)))
        stv_real = torch.stack(stv_real)
        stv_imag = torch.stack(stv_imag)
        stv_real /= math.sqrt(self.n_mic_pairs)
        stv_imag /= math.sqrt(self.n_mic_pairs)
        # [F, BS, M, C] => [BS, M, C, F]
        stv_real = stv_real.permute((1, 2, 3, 0))
        stv_imag = stv_imag.permute((1, 2, 3, 0))
        return stv_real, stv_imag

    def get_AF(self, stv_real, stv_imag, ipd4AF_real, ipd4AF_imag, spk_num):
        '''stv shape [B, M, C, F], ipd shape [B, M, F, K], spk_num [B]
        return AF: [B, K, XF]
        '''
        bs, m, f, k = ipd4AF_imag.shape
        #import pdb; pdb.set_trace()
        # compute cosine distance between stv and ipd
        rlt_rr_ein = torch.einsum('bmcf,bmfk->bcfk', (stv_real, ipd4AF_real))  # [n_freqs, n_frame]
        rlt_ri_ein = torch.einsum('bmcf,bmfk->bcfk', (stv_real, ipd4AF_imag))  # [n_freqs, n_frame]
        rlt_ir_ein = torch.einsum('bmcf,bmfk->bcfk', (stv_imag, ipd4AF_real))  # [n_freqs, n_frame]
        rlt_ii_ein = torch.einsum('bmcf,bmfk->bcfk', (stv_imag, ipd4AF_imag))  # [n_freqs, n_frame]
        # AF = torch.abs(torch.sqrt((rlt_rr_ein + rlt_ii_ein) ** 2 + (-rlt_ir_ein + rlt_ri_ein) ** 2))
        AF = rlt_rr_ein + rlt_ii_ein

        AFs = []
        for b in range(bs):
            nspk = spk_num[b]
            _AF_tgt = AF[b, 0]  # K, F
            for idx in range(1, nspk.int()):
                _AF_tgt[_AF_tgt < AF[b, idx]] = 0
            _AF_tgt = (_AF_tgt - torch.mean(_AF_tgt, dim=-1, keepdim=True)) \
                      / (torch.std(_AF_tgt, dim=-1, keepdim=True) + 1e-8)
            if self.spk_fea_dim == 2:
                if nspk == 1:
                    _AF_intf = torch.zeros_like(_AF_tgt)
                else:
                    _AF_intfs = []
                    for idx in range(1, nspk):
                        _AF_intf = AF[b, idx]
                        for j in range(1, nspk):
                            if idx != j:
                                _AF_intf[_AF_intf < AF[b, j]] = 0
                        _AF_intfs.append(_AF_intf)
                    if self.spk_fea_merge_mode == 'sum':
                        _AF_intf = torch.sum(torch.stack(_AF_intfs, dim=-1), dim=-1)
                    elif self.spk_fea_merge_mode == 'ave':
                        _AF_intf = torch.mean(torch.stack(_AF_intfs, dim=-1), dim=-1)
                    elif self.spk_fea_merge_mode == 'closest':
                        raise NotImplementedError  # need doa here
                    _AF_intf = (_AF_intf - torch.mean(_AF_intf, dim=-1, keepdim=True)) \
                               / (torch.std(_AF_intf, dim=-1, keepdim=True) + 1e-8)

                AFs.append(torch.cat((_AF_tgt, _AF_intf), dim=0))  # [XF, K]
            else:
                AFs.append(_AF_tgt)
        AF = torch.stack(AFs, dim=0)  # B, XF, K
        AF = AF.view(bs, self.num_bins * self.spk_fea_dim, k)
        # if self.ln_AF is not None:
        #     AF = self.ln_AF(AF)
        # AF = (AF - torch.mean(AF, dim=-1,keepdim=True)) / (torch.std(AF, dim=-1, keepdim=True) + 1e-8)
        return AF

    def get_DPR(self, X_real, X_imag, src_doa, spk_num):
        '''
        :param X_real/X_imag: [B, M, F, K]
        :param direction of arrival in rad with shape [BS, max_nspk]
        :param spk_num: [B]
        :return: F1 [B, K, XF]
        '''
        if X_real.shape[1] != self.n_mic:
            raise ValueError("The input channels of mixture should be the same with microphone array")
        rlt_rr_ein = torch.einsum('amf,bmfk->bafk', (self.w_ds_real, X_real))
        rlt_ii_ein = torch.einsum('amf,bmfk->bafk', (self.w_ds_imag, X_imag))
        rlt_ri_ein = torch.einsum('amf,bmfk->bafk', (self.w_ds_real, X_imag))
        rlt_ir_ein = torch.einsum('amf,bmfk->bafk', (self.w_ds_imag, X_real))
        azm_pow = (rlt_rr_ein + rlt_ii_ein) ** 2 + (rlt_ir_ein - rlt_ri_ein) ** 2
        # from [B, A, F, K] -> [B, K, F, A]
        azm_pow = azm_pow.permute((0, 3, 2, 1))  # [B, K, F, A]

        src_idxs = ((src_doa / math.pi * 180) // self.ds_bf.spatial_resolution).long()  # [B, C]
        bs, max_nspk = src_idxs.shape

        if self.spk_fea_dim == 1:
            b_idx = [b for b in range(bs)]
            DPR = azm_pow[b_idx, :, :, src_idxs[:, 0]] / (torch.sum(azm_pow, dim=3) + 1e-30)  # [B, K, F]
            DPR = (DPR - torch.mean(DPR, dim=1, keepdim=True)) / (torch.std(DPR, dim=1, keepdim=True) + self.epsilon)
        else:
            DPRs = []
            for b in range(bs):
                _DPR_tgt = azm_pow[b, :, :, src_idxs[b, 0]] / (torch.sum(azm_pow[b], dim=-1) + 1e-8)
                _DPR_tgt = (_DPR_tgt - torch.mean(_DPR_tgt, dim=1, keepdim=True)) / (
                    torch.std(_DPR_tgt, dim=1, keepdim=True) + self.epsilon)

                if spk_num[b] == 1:
                    _DPR_intf = torch.zeros_like(_DPR_tgt)
                else:
                    _DPR_intfs = []
                    for c in range(1, spk_num[b]):
                        _DPR_intf = azm_pow[b, :, :, src_idxs[b, c]] / (torch.sum(azm_pow[b], dim=-1) + 1e-8)
                        _DPR_intfs.append(_DPR_intf)
                    if self.spk_fea_merge_mode == 'sum':
                        _DPR_intf = torch.sum(torch.stack(_DPR_intfs, dim=-1), dim=-1)  # [B,K,F]
                    elif self.spk_fea_merge_mode == 'ave':
                        _DPR_intf = torch.mean(torch.stack(_DPR_intfs, dim=-1), dim=-1)
                    elif self.spk_fea_merge_mode == 'closest':
                        raise NotImplementedError  # need doa here
                    _DPR_intf = (_DPR_intf - torch.mean(_DPR_intf, dim=1, keepdim=True)) \
                                / (torch.std(_DPR_intf, dim=1, keepdim=True) + 1e-8)

                DPRs.append(torch.cat((_DPR_tgt, _DPR_intf), dim=1))

            DPR = torch.stack(DPRs, dim=0)  # B, K, XF

        DPR = DPR.permute(0, 2, 1)  # [B, XF, K]
        return DPR


class DSBeamformer(object):
    def __init__(self, sample_rate):
        """
        :param mic_positions: The positions of each microphone in the microphone
                                array of this beam-former. Each row should
                                represent a different mic, with the number of
                                columns indicating the dimensionality of the space
        :type mic_positions: ndarray
        """
        self.def_mic_idx = 4
        self._get_mic_pos()
        self._sample_rate = sample_rate
        self._sndv = 340  # m/s
        self.spatial_resolution = 10  # in degrees

    def _get_mic_pos(self):

        self.radius = 0.10
        self.mic_position = np.array(
            [-0.10, -0.06, -0.03, -0.01, 0.0, 0.01, 0.03, 0.06, 0.10]
        )
        self.n_mic = self.mic_position.shape[0]
        distance = np.zeros([self.n_mic, self.n_mic])
        for i in range(self.n_mic):
            distance[i] = [abs(self.mic_position[i] - self.mic_position[j]) for j in range(self.n_mic)]
        self.mic_distances = distance

    def get_stv(self, angles, freqs):
        """angles should be a list, every element denotes a DOA in rad.
        return: steering vector in shape [n_spks, n_channels, freqs]"""
        angles = np.array(angles)

        dist = np.stack(
            (self.mic_distances[i] * -np.cos(angles) for i in range(self.n_mic)), axis=0)
        delay = dist / self._sndv * self._sample_rate  # in samples

        steervecs = []
        for f in range(freqs):
            steervecs.append(np.exp(-1j * delay * math.pi * f / (freqs - 1)))
        steervecs = np.stack(steervecs)  # [F, n_channels, n_spks]
        steervecs = np.transpose(steervecs, [2, 1, 0])  # [n_spks, n_channels, F]

        return steervecs

    def get_nn_stv(self, angles, nfreqs):
        '''
        :param angles: [Batch_size, nspk], each elements denotes a DOA in rad.
        :param nfreqs: num_bins
        :return: steering vector for each spk in each frequency band, [Batch, n_channel, nspk, n_bins]
        '''
        dist_list = [self.mic_distances[i] * -torch.cos(angles) for i in range(self.n_mic)]
        dist = torch.stack(dist_list, dim=1)  # [BS, M, NSPK]

        delay = dist / self._sndv * self._sample_rate  # in samples

        stv_real = []
        stv_imag = []
        for f in range(nfreqs):
            stv_real.append(torch.cos(delay * math.pi * f / (nfreqs - 1)))
            stv_imag.append(-torch.sin(delay * math.pi * f / (nfreqs - 1)))
        stv_real = torch.stack(stv_real)
        stv_imag = torch.stack(stv_imag)

        # [F, BS, M, C] => [BS, C, M, F]
        stv_real = stv_real.permute((1, 3, 2, 0))
        stv_imag = stv_imag.permute((1, 3, 2, 0))

        return stv_real, stv_imag


if __name__ == '__main__':
    pass
    # df_computer = DFComputer(in_feature=['LPS', 'IPD', 'AF', 'DPR'],
    #                          frame_len=512, frame_hop=256, speaker_feature_dim=2,
    #                          merge_mode='sum', cosIPD=True, sinIPD=True)
    #
    # # import librosa
    # import scipy.io.wavfile as wf
    #
    # sr, wav = wf.read('C:/Users\Moplast\Desktop\Moplast\Codes/1905-work/time-tcn-ch/0-1.wav')
    # mix_wav = wav[:, :9].T
    # print(mix_wav.shape)
    #
    # x = [torch.Tensor(mix_wav).unsqueeze(0),
    #      torch.Tensor(np.array([70 * np.pi / 180, 122 * np.pi / 180, -1, -1])).unsqueeze(0),
    #      torch.LongTensor(np.array([2])).unsqueeze(0)]
    # df, mag, phase = df_computer(x)
    #
    # print(df.shape, mag.shape, phase.shape)
