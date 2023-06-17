import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# from params import *

########## relative import for asr #############
from .params import *


'''
    In torchaudio and kaldi the waveform is transfered into Frame
    N*1 -> [frame_number, frame_length]
    One can change Conv1d to DNN
    step1: dither (non-neural)
    step2: remove_dc_offset (normalization?)
    step3: raw_energy (non-neural)
    step4: preemphasis (neural)
    step5: window (elementproduct)
    step6: Padding window_size
    step7: energy
    step8: fft
    step9: spectrum

    Convert this to batch operation [N * max_length] : Not done

'''

# window types
HAMMING = 'hamming'
HANNING = 'hanning'
POVEY = 'povey'
RECTANGULAR = 'rectangular'
BLACKMAN = 'blackman'
WINDOWS = [HAMMING, HANNING, POVEY, RECTANGULAR, BLACKMAN]
EPSILON = torch.tensor(1e-8)  # torch.tensor(torch.finfo(torch.float).eps, dtype=torch.get_default_dtype())


class LFB(nn.Module):
    def __init__(self, blackman_coeff=0.42, channel=-1, dither=1.0,
                 energy_floor=0.0, min_duration=0.0, preemphasis_coefficient=0.97,
                 raw_energy=True, remove_dc_offset=True,
                 snip_edges=True, subtract_mean=False,
                 high_freq=0.0, htk_compat=False, low_freq=20.0,
                 num_mel_bins=23, use_energy=True, use_log_fbank=True,
                 use_power=True, vtln_high=-500.0, vtln_low=100.0, vtln_warp=1.0):
        super(LFB, self).__init__()

        self.blackman_coeff = blackman_coeff
        self.channel = channel
        self.dither = dither
        self.energy_floor = energy_floor
        self.min_duration = min_duration
        self.preemphasis_coefficient = preemphasis_coefficient
        self.raw_energy = raw_energy
        self.remove_dc_offset = remove_dc_offset
        self.sampling_rate = sampling_rate
        self.snip_edges = snip_edges
        self.subtract_mean = subtract_mean

        self.high_freq = high_freq
        self.htk_compat = htk_compat
        self.low_freq = low_freq
        self.num_mel_bins = num_mel_bins
        self.use_energy = use_energy
        self.use_log_fbank = use_log_fbank
        self.use_power = use_power
        self.vtln_high = vtln_high
        self.vtln_low = vtln_low
        self.vtln_warp = vtln_warp
        # Get mel banks
        mel_energies, _ = self.get_mel_banks(self.num_mel_bins,
                                             FFT_SIZE,
                                             sampling_rate,
                                             self.low_freq, self.high_freq,
                                             self.vtln_low, self.vtln_high,
                                             self.vtln_warp)
        # pad right column with zeros and add dimension, size (1, num_mel_bins, padded_window_size // 2 + 1)
        mel_energies = F.pad(mel_energies, (0, 1), mode='constant', value=0).unsqueeze(0)
        self.mel_energies = nn.Parameter(mel_energies, requires_grad=False)  # Make the parameters learnable
        #`print(self.mel_energies.shape)

    def _get_log_energy(self, strided_input, epsilon, energy_floor):
        r"""Returns the log energy of size (m) for a strided_input (m,*)
        """
        log_energy = (strided_input.pow(2).sum(2)).log()
        # log_energy = torch.max(strided_input.pow(2).sum(1), epsilon).log()  # size (m)
        if energy_floor == 0.0:
            return log_energy
        else:
            return torch.max(log_energy,
                             torch.tensor(math.log(energy_floor), dtype=torch.get_default_dtype()))

    def _cmvn(self, tensor, subtract_mean):
        # subtracts the column mean of the tensor size (N, m, n) if subtract_mean=True
        # it returns size (N, m, n)
        if subtract_mean:
            col_means = torch.mean(tensor, dim=1).unsqueeze(1)
            col_vars = torch.std(tensor, dim=1).unsqueeze(1)
            tensor = (tensor - col_means) / (col_vars + 1e-8)
        return tensor

    def inverse_mel_scale_scalar(self, mel_freq):
        # type: (float) -> float
        return 700.0 * (math.exp(mel_freq / 1127.0) - 1.0)

    def inverse_mel_scale(self, mel_freq):
        return 700.0 * ((mel_freq / 1127.0).exp() - 1.0)

    def mel_scale_scalar(self, freq):
        # type: (float) -> float
        return 1127.0 * math.log(1.0 + freq / 700.0)

    def mel_scale(self, freq):
        return 1127.0 * (1.0 + freq / 700.0).log()

    def vtln_warp_freq(self, vtln_low_cutoff, vtln_high_cutoff, low_freq, high_freq,
                       vtln_warp_factor, freq):
        r"""This computes a VTLN warping function that is not the same as HTK's one,
        but has similar inputs (this function has the advantage of never producing
        empty bins).

        This function computes a warp function F(freq), defined between low_freq
        and high_freq inclusive, with the following properties:
            F(low_freq) == low_freq
            F(high_freq) == high_freq
        The function is continuous and piecewise linear with two inflection
            points.
        The lower inflection point (measured in terms of the unwarped
            frequency) is at frequency l, determined as described below.
        The higher inflection point is at a frequency h, determined as
            described below.
        If l <= f <= h, then F(f) = f/vtln_warp_factor.
        If the higher inflection point (measured in terms of the unwarped
            frequency) is at h, then max(h, F(h)) == vtln_high_cutoff.
            Since (by the last point) F(h) == h/vtln_warp_factor, then
            max(h, h/vtln_warp_factor) == vtln_high_cutoff, so
            h = vtln_high_cutoff / max(1, 1/vtln_warp_factor).
              = vtln_high_cutoff * min(1, vtln_warp_factor).
        If the lower inflection point (measured in terms of the unwarped
            frequency) is at l, then min(l, F(l)) == vtln_low_cutoff
            This implies that l = vtln_low_cutoff / min(1, 1/vtln_warp_factor)
                                = vtln_low_cutoff * max(1, vtln_warp_factor)
        Args:
            vtln_low_cutoff (float): Lower frequency cutoffs for VTLN
            vtln_high_cutoff (float): Upper frequency cutoffs for VTLN
            low_freq (float): Lower frequency cutoffs in mel computation
            high_freq (float): Upper frequency cutoffs in mel computation
            vtln_warp_factor (float): Vtln warp factor
            freq (torch.Tensor): given frequency in Hz

        Returns:
            torch.Tensor: Freq after vtln warp
        """
        assert vtln_low_cutoff > low_freq, 'be sure to set the vtln_low option higher than low_freq'
        assert vtln_high_cutoff < high_freq, 'be sure to set the vtln_high option lower than high_freq [or negative]'
        l = vtln_low_cutoff * max(1.0, vtln_warp_factor)
        h = vtln_high_cutoff * min(1.0, vtln_warp_factor)
        scale = 1.0 / vtln_warp_factor
        Fl = scale * l  # F(l)
        Fh = scale * h  # F(h)
        assert l > low_freq and h < high_freq
        # slope of left part of the 3-piece linear function
        scale_left = (Fl - low_freq) / (l - low_freq)
        # [slope of center part is just "scale"]

        # slope of right part of the 3-piece linear function
        scale_right = (high_freq - Fh) / (high_freq - h)

        res = torch.empty_like(freq)

        outside_low_high_freq = torch.lt(freq, low_freq) | torch.gt(freq,
                                                                    high_freq)  # freq < low_freq || freq > high_freq
        before_l = torch.lt(freq, l)  # freq < l
        before_h = torch.lt(freq, h)  # freq < h
        after_h = torch.ge(freq, h)  # freq >= h

        # order of operations matter here (since there is overlapping frequency regions)
        res[after_h] = high_freq + scale_right * (freq[after_h] - high_freq)
        res[before_h] = scale * freq[before_h]
        res[before_l] = low_freq + scale_left * (freq[before_l] - low_freq)
        res[outside_low_high_freq] = freq[outside_low_high_freq]

        return res

    def vtln_warp_mel_freq(self, vtln_low_cutoff, vtln_high_cutoff, low_freq, high_freq,
                           vtln_warp_factor, mel_freq):
        r"""
        Args:
            vtln_low_cutoff (float): Lower frequency cutoffs for VTLN
            vtln_high_cutoff (float): Upper frequency cutoffs for VTLN
            low_freq (float): Lower frequency cutoffs in mel computation
            high_freq (float): Upper frequency cutoffs in mel computation
            vtln_warp_factor (float): Vtln warp factor
            mel_freq (torch.Tensor): Given frequency in Mel

        Returns:
            torch.Tensor: ``mel_freq`` after vtln warp
        """
        return self.mel_scale(self.vtln_warp_freq(vtln_low_cutoff, vtln_high_cutoff, low_freq, high_freq,
                                                  vtln_warp_factor, self.inverse_mel_scale(mel_freq)))

    def get_mel_banks(self, num_bins, window_length_padded, sample_freq,
                      low_freq, high_freq, vtln_low, vtln_high, vtln_warp_factor):
        # type: (int, int, float, float, float, float, float)
        """
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The tuple consists of ``bins`` (which is
            melbank of size (``num_bins``, ``num_fft_bins``)) and ``center_freqs`` (which is
            center frequencies of bins of size (``num_bins``)).
        """
        assert num_bins > 3, 'Must have at least 3 mel bins'
        assert window_length_padded % 2 == 0
        num_fft_bins = window_length_padded / 2
        nyquist = 0.5 * sample_freq

        if high_freq <= 0.0:
            high_freq += nyquist

        assert (0.0 <= low_freq < nyquist) and (0.0 < high_freq <= nyquist) and (low_freq < high_freq), \
            ('Bad values in options: low-freq %f and high-freq %f vs. nyquist %f' % (low_freq, high_freq, nyquist))

        # fft-bin width [think of it as Nyquist-freq / half-window-length]
        fft_bin_width = sample_freq / window_length_padded
        mel_low_freq = self.mel_scale_scalar(low_freq)
        mel_high_freq = self.mel_scale_scalar(high_freq)

        # divide by num_bins+1 in next line because of end-effects where the bins
        # spread out to the sides.
        mel_freq_delta = (mel_high_freq - mel_low_freq) / (num_bins + 1)

        if vtln_high < 0.0:
            vtln_high += nyquist

        assert vtln_warp_factor == 1.0 or ((low_freq < vtln_low < high_freq) and
                                           (0.0 < vtln_high < high_freq) and (vtln_low < vtln_high)), \
            ('Bad values in options: vtln-low %f and vtln-high %f, versus low-freq %f and high-freq %f' %
             (vtln_low, vtln_high, low_freq, high_freq))

        bin = torch.arange(num_bins, dtype=torch.get_default_dtype()).unsqueeze(1)
        left_mel = mel_low_freq + bin * mel_freq_delta  # size(num_bins, 1)
        center_mel = mel_low_freq + (bin + 1.0) * mel_freq_delta  # size(num_bins, 1)
        right_mel = mel_low_freq + (bin + 2.0) * mel_freq_delta  # size(num_bins, 1)

        if vtln_warp_factor != 1.0:
            left_mel = self.vtln_warp_mel_freq(vtln_low, vtln_high, low_freq, high_freq, vtln_warp_factor, left_mel)
            center_mel = self.vtln_warp_mel_freq(vtln_low, vtln_high, low_freq, high_freq, vtln_warp_factor, center_mel)
            right_mel = self.vtln_warp_mel_freq(vtln_low, vtln_high, low_freq, high_freq, vtln_warp_factor, right_mel)

        center_freqs = self.inverse_mel_scale(center_mel)  # size (num_bins)
        # size(1, num_fft_bins)
        mel = self.mel_scale(fft_bin_width * torch.arange(num_fft_bins, dtype=torch.get_default_dtype())).unsqueeze(0)

        # size (num_bins, num_fft_bins)
        up_slope = (mel - left_mel) / (center_mel - left_mel)
        down_slope = (right_mel - mel) / (right_mel - center_mel)

        if vtln_warp_factor == 1.0:
            # left_mel < center_mel < right_mel so we can min the two slopes and clamp negative values
            bins = torch.max(torch.zeros(1), torch.min(up_slope, down_slope))
        else:
            # warping can move the order of left_mel, center_mel, right_mel anywhere
            bins = torch.zeros_like(up_slope)
            up_idx = torch.gt(mel, left_mel) & torch.le(mel, center_mel)  # left_mel < mel <= center_mel
            down_idx = torch.gt(mel, center_mel) & torch.lt(mel, right_mel)  # center_mel < mel < right_mel
            bins[up_idx] = up_slope[up_idx]
            bins[down_idx] = down_slope[down_idx]

        return bins, center_freqs

    def _get_lifter_coeffs(self, num_ceps, cepstral_lifter):
        # returns size (num_ceps)
        # Compute liftering coefficients (scaling on cepstral coeffs)
        # coeffs are numbered slightly differently from HTK: the zeroth index is C0, which is not affected.
        i = torch.arange(num_ceps, dtype=torch.get_default_dtype())
        return 1.0 + 0.5 * cepstral_lifter * torch.sin(math.pi * i / cepstral_lifter)

    def forward(self, spectrum, epsilon, delta=True, delta_delta=True):
        """
        Args
        :param spectrum: [Batch size, T, F]
        :return:
        """

        if self.use_power:
            spectrum = spectrum.pow(2)

        mel_energies = F.linear(spectrum, self.mel_energies.squeeze())
        # mel_energies = (power_spectrum * self.mel_energies).sum(dim=2) # size (N, m, padded_window_size // 2 + 1)
        if self.use_log_fbank:
            mel_energies = torch.max(mel_energies, epsilon).log()  # avoid log of zero (which should be prevented anyway by dithering)
        # print(mel_energies.shape)
        mel_energies = self._cmvn(mel_energies, self.subtract_mean)  # [BS, T, mel_bins]

        if delta:
            t = mel_energies.size(1)
            mel_energies_1 = torch.cat([mel_energies[:, 0].unsqueeze(1), mel_energies],
                                       dim=1)  # [BS, T+1, mel_bins] // right shift
            mel_energies_2 = torch.cat([mel_energies, mel_energies[:, -1].unsqueeze(1), ],
                                       dim=1)  # [BS, T+1, mel_bins] // original
            delta_mel_energies = (mel_energies_2 - mel_energies_1)[:, :t]  # [BS, T, mel_bins]

            if delta_delta:
                delta_mel_energies_1 = torch.cat([delta_mel_energies[:, 0].unsqueeze(1), delta_mel_energies],
                                                 dim=1)  # [BS, T+1, mel_bins] // right shift
                delta_mel_energies_2 = torch.cat([delta_mel_energies, delta_mel_energies[:, -1].unsqueeze(1), ],
                                                 dim=1)  # [BS, T+1, mel_bins] // original
                delta_delta_mel_energies = (delta_mel_energies_2 - delta_mel_energies_1)[:, :t]  # [BS, T, mel_bins]
                return torch.cat([mel_energies, delta_mel_energies, delta_delta_mel_energies], dim=-1)
            else:
                return torch.cat([mel_energies, delta_mel_energies], dim=-1)
        else:
            return mel_energies


if __name__ == "__main__":
    # import librosa
    # wav, fs = librosa.load('s2.wav', sr=16000, mono=False)
    # wav = wav * (2 ** 15)
    from scipy.io.wavfile import read
    import numpy as np

    fs, wav = read('s2.wav')
    wav = wav.astype(np.float32) * 1.0  # / (2**15)
    print('wav', wav.shape)

    lfb = LFB()
    from pt_audio_fea1 import STFT

    stft = STFT(frame_len=512, frame_hop=256)
    wav = torch.tensor(wav).unsqueeze(0)
    mag, phase = stft(wav)
    print(mag.shape)
    print(mag)

    m = lfb(mag.permute((0, 2, 1)))
    print(m.size(), m[0, :, 1], m[0, :, 5])
