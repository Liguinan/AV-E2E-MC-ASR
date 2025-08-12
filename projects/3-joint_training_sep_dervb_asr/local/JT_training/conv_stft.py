# from .params import *
import math
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np


def init_kernel(frame_len,
                frame_hop,
                num_fft=None,
                window="sqrt_hann"):
    windows = ["sqrt_hann", "povey", "hann"]
    if window not in windows :
        raise RuntimeError(f"Now only support {windows} windows for fft")
    if not num_fft:
        # FFT points
        fft_size = 2 ** math.ceil(math.log2(frame_len))
    else:
        fft_size = num_fft
    # window [window_length]
    if window=='povey':
        window = torch.hann_window(frame_len) ** 0.85
    elif window == 'sqrt_hann':
        window = torch.hann_window(frame_len) ** 0.5
    elif window == 'hann':
        print(f"window is {window}")
        window = torch.hann_window(frame_len)
    else:
        NotImplementedError
        
    # import pdb; pdb.set_trace()
    left_pad = (num_fft - frame_len)//2
    right_pad = left_pad + (num_fft - frame_len) % 2
    window = F.pad(window, (left_pad, right_pad))
        
    # S_ = 0.5 * (fft_size * fft_size / frame_hop) ** 0.5
    # window_length, F, 2 (real+imag)
    # import pdb; pdb.set_trace()
    # kernel = torch.rfft(torch.eye(fft_size) / S_, 1)[:frame_len]
    # print("torch.fft.rfft")
    # kernel = torch.fft.rfft(torch.eye(fft_size) / S_, dim=-1)[:frame_len]
    # print("torch.fft.rfft(torch.eye(fft_size) / S_, dim=-1)[:frame_len]")
    # kernel = torch.fft.rfft(torch.eye(fft_size), dim=-1)[:frame_len]
    # print(f"torch.fft.rfft(torch.eye(fft_size), dim=-1)[:frame_len]")
    # kernal: (512, 257); fft_size: 512
    kernel = torch.fft.rfft(torch.eye(fft_size), dim=-1)
    # print(f"torch.fft.rfft(torch.eye(fft_size), dim=-1)")
    # -> (512, 257, 2)
    kernel = torch.stack((kernel.real, kernel.imag), -1)
    # (2, 257, 512)  *  window: (512,)  -> (2, 257, 512)
    kernel = torch.transpose(kernel, 0, 2) * window
    # -> (514, 1, 512)
    kernel = torch.reshape(kernel, (fft_size + 2, 1, fft_size))
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
        self.num_fft = num_fft
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
        self.pad_amount = self.num_fft // 2

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
        x = F.pad(x, (self.pad_amount, self.pad_amount), mode='reflect')
        # N x 2F x T
        c = F.conv1d(x, self.K, stride=self.stride, padding=0)
        # N x F x T
        r, i = torch.chunk(c, 2, dim=1)
        m = (r ** 2 + i ** 2) ** 0.5
        p = torch.atan2(i, r)
        return m, p


class iSTFT(STFTBase):
    """
    Inverse Short-time Fourier Transform as a Layer
    """

    def __init__(self, *args, **kwargs):
        super(iSTFT, self).__init__(*args, **kwargs)
        self.pad_amount = self.num_fft // 2

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
        s = s[..., self.pad_amount:]
        s = s[..., :self.pad_amount]
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

