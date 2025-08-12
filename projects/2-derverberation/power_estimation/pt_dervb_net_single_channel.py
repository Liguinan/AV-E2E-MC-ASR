#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
from torch.nn.utils.rnn import pad_sequence
#from torch_complex import ComplexTensor
from pt_audio_fea_single_channel import DFComputer, STFT, iSTFT
from params import *
import torch.nn as nn
from pt_video_fea import * 
from pt_log_fbank import LFB
from pt_fusion import FactorizedLayer
from sequence_model import SeqModel


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


class GlobalChannelLayerNorm(nn.Module):
    """
    Global channel layer normalization
    """

    def __init__(self, dim, eps=1e-05, elementwise_affine=True):
        super(GlobalChannelLayerNorm, self).__init__()
        self.eps = eps
        self.normalized_dim = dim
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.beta = nn.Parameter(torch.zeros(dim, 1))
            self.gamma = nn.Parameter(torch.ones(dim, 1))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x):
        """
        x: N x C x T
        """
        if x.dim() != 3:
            raise RuntimeError("{} accept 3D tensor as input".format(
                self.__name__))
        # N x 1 x 1
        mean = torch.mean(x, (1, 2), keepdim=True)
        var = torch.mean((x - mean) ** 2, (1, 2), keepdim=True)
        # N x T x C
        if self.elementwise_affine:
            x = self.gamma * (x - mean) / torch.sqrt(var + self.eps) + self.beta
        else:
            x = (x - mean) / torch.sqrt(var + self.eps)
        return x

    def extra_repr(self):
        return "{normalized_dim}, eps={eps}, " \
               "elementwise_affine={elementwise_affine}".format(**self.__dict__)


def build_norm(norm, dim):
    """
    Build normalize layer
    LN cost more memory than BN
    """
    if norm not in ["cLN", "gLN", "BN"]:
        raise RuntimeError("Unsupported normalize layer: {}".format(norm))
    if norm == "cLN":
        return ChannelWiseLayerNorm(dim, elementwise_affine=True)
    elif norm == "BN":
        return nn.BatchNorm1d(dim)
    else:
        return GlobalChannelLayerNorm(dim, elementwise_affine=True)


class Conv1D(nn.Conv1d):
    """
    1D conv in ConvTasNet
    """

    def __init__(self, *args, **kwargs):
        super(Conv1D, self).__init__(*args, **kwargs)

    def forward(self, x, squeeze=False):
        """
        x: N x L or N x C x L
        """
        if x.dim() not in [2, 3]:
            raise RuntimeError("{} accept 2/3D tensor as input".format(
                self.__name__))
        x = super(Conv1D, self).forward(x if x.dim() == 3 else torch.unsqueeze(x, 1))
        if squeeze:
            x = torch.squeeze(x)
        return x


class Conv1DBlock(nn.Module):
    """
    1D convolutional block:
        Conv1x1 - PReLU - Norm - DConv - PReLU - Norm - SConv
    """

    def __init__(self,
                 in_channels=B,
                 conv_channels=H,
                 kernel_size=P,
                 dilation=1,
                 norm=norm,
                 causal=False):
        super(Conv1DBlock, self).__init__()

        # <1> 1x1-conv
        self.conv1x1 = Conv1D(in_channels, conv_channels, 1)
        # <2> prelu
        self.prelu1 = nn.PReLU()
        # <3> normalization
        self.lnorm1 = build_norm(norm, conv_channels)
        # <4> D-conv
        dconv_pad = (dilation * (kernel_size - 1)) // 2 if not causal else (
            dilation * (kernel_size - 1))
        # depthwise conv
        self.dconv = nn.Conv1d(
            conv_channels,
            conv_channels,
            kernel_size,
            groups=conv_channels,
            padding=dconv_pad,
            dilation=dilation,
            bias=True)
        # <5> prelu
        self.prelu2 = nn.PReLU()
        # <6> normalization
        self.lnorm2 = build_norm(norm, conv_channels)
        # <7> 1x1-conv cross channel
        self.sconv = nn.Conv1d(conv_channels, in_channels, 1, bias=True)
        # different padding way
        self.causal = causal
        self.dconv_pad = dconv_pad

    def forward(self, x):
        y = self.conv1x1(x)
        y = self.lnorm1(self.prelu1(y))
        y = self.dconv(y)
        if self.causal:
            y = y[:, :, :-self.dconv_pad]
        y = self.lnorm2(self.prelu2(y))
        y = self.sconv(y)
        x = x + y  # identity connection
        return x


class DervbNet(nn.Module):
    def __init__(self):
        super(DervbNet, self).__init__()
        
        # import pdb; pdb.set_trace()
        # Feature extractor
        self.df_computer = DFComputer(frame_hop=HOP_SIZE, frame_len=FFT_SIZE, in_feature=['LPS'])

        # audio
        self.conv1x1_1 = Conv1D(self.df_computer.df_dim, B, 1)  # for audio dim
        # import pdb; pdb.set_trace()
        if TCN_for_lip:
            self.lip_blocks = self._build_repeats(
                num_repeats=5,
                num_blocks=X,
                in_channels=B,
                conv_channels=H,
                kernel_size=P,
                norm=norm,
                causal=causal)
        else:
            self.lip_blocks = OxfordLipNet(embedding_dim=V, conv_channels=V, num_blocks=lip_block_num)

        if visual_fusion_type == 'attention':
            self.av_fusion_layer = FactorizedLayer(factor=factor,
                                                   audio_features=256,
                                                   other_features=256,
                                                   out_features=256)

        self.audio_blocks = self._build_repeats(
            num_repeats=1,
            num_blocks=X,
            in_channels=B,
            conv_channels=H,
            kernel_size=P,
            norm=norm,
            causal=causal)
        if model_type == 'TCN':
            self.fusion_blocks = self._build_repeats(
                num_repeats=3,
                num_blocks=X,
                in_channels=B,
                conv_channels=H,
                kernel_size=P,
                norm=norm,
                causal=causal)
        elif model_type == 'LSTM':
            self.fusion_blocks = SeqModel(
                input_dim=256,
                num_bins=257,
                rnn='lstm',
                complex_mask=True,
                num_layers=1,
                hidden_size=722,  # 722
                non_linear='linear',
                bidirectional=True)
        else:
            raise ValueError(f"Not support model type={model_type}")
            
        self.conv1x1_2_real = Conv1D(B, out_spk * self.df_computer.num_bins, 1)
        self.conv1x1_2_imag = Conv1D(B, out_spk * self.df_computer.num_bins, 1)
        
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(512, 256)

        self.lfb = LFB(num_mel_bins=num_mel_bins)
        
        self.istft = iSTFT(frame_len=FFT_SIZE, frame_hop=HOP_SIZE, num_fft=FFT_SIZE)
        self.epsilon = nn.Parameter(torch.tensor(torch.finfo(torch.float).eps), requires_grad=False)
        

    def check_forward_args(self, all_x):
        x = all_x[0]  
        lip_video = all_x[1]  
        ref = all_x[2]  
        lens = all_x[3]  
        # todo check dimension
        return x, lip_video, ref, lens

    def forward(self, all_x):
        # import pdb; pdb.set_trace() 
        # x: (B, C, t), lip_fea:(B, V_T, V_F: 512), ref: (B, C, t), lens:(B,)
        x, video_fea, ref, lens = self.check_forward_args(all_x)

        # Feature extractor
        # audio_fea: (B, F, T), mag_rev: (B, C, F, T), phase_rev: (B, C, F, T)
        audio_fea, mag_rev, phase_rev = self.df_computer([x])
        # mag_ref: (B, C, F, T), phase_ref: (B, C, F, T)
        _, mag_ref, phase_ref = self.df_computer([ref])

        # (B, F, T) -> (B, F': 256, T)
        audio_fea = self.conv1x1_1(audio_fea)
        # (B, F': 256, T) -> (B, F', T)
        audio_fea = self.audio_blocks(audio_fea)

        # video
        # import pdb; pdb.set_trace()
        # (B, V_T, V_F: 512) -> (B, V_T/2, V_F: 512)
        # print(f"video frame of this batch : {video_fea.shape[1]}")
        if sampling:
            print(f"sampling: {sampling_frame_ratio}")
            video_fea = video_fea[:, 0::sampling_frame_ratio, :]
        # print(f"video frame of this batch after sampling: {video_fea.shape[1]}")
        
        
        # (B, V_T, V_F: 512) -> (B, V_T, F')
        video_fea = self.fc1(video_fea)
        if not add_visual:
            # (B, V_T, F') -> (B, V_T, F')
            video_fea = torch.zeros_like(video_fea)
        # (B, V_T, F') -> (B, F', V_T)
        
        if TCN_for_lip:
            # (B, V_T, F') -> (B, F', T)
            video_fea = video_fea.permute(0, 2, 1)
        video_fea = self.lip_blocks(video_fea)
        # import pdb; pdb.set_trace()
        # (B, F', V_T) -> (B, F', T)
        video_fea = nn.functional.interpolate(video_fea, size=audio_fea.size(2), mode=mode)
        

        # fusion
        if visual_fusion_type == 'attention':
            # import pdb; pdb.set_trace()
            av_fea = self.av_fusion_layer(audio_fea.permute(0, 2, 1), video_fea.permute(0, 2, 1))
        elif visual_fusion_type == 'concat':
            # (B, F', T) +  (B, F', T) -> (B, 2 * F', T) -> (B, T, 2 * F')
            av_fea = torch.cat((audio_fea, video_fea), dim=-2).permute(0, 2, 1)
            # (B, T, 2 * F') -> (B, T, F') -> (B, F', T)
            av_fea = self.fc2(av_fea).permute(0, 2, 1)
        else:
            raise ValueError(f"Not support visual fusion type = {visual_fusion_type}")

        if model_type == 'LSTM':
            # (B, F', T) -> (B, F, T), lens: (B,)
            e_real, e_imag = self.fusion_blocks(av_fea, lens)
        elif model_type == 'TCN':
            # (B, F', T) -> (B, F', T)
            y = self.fusion_blocks(av_fea)
            # (B, F', T) -> (B, F, T)
            e_real = self.conv1x1_2_real(y)
            e_imag = self.conv1x1_2_imag(y)
        else:
            raise ValueError(f"Not support model type = {model_type}")

        # Estimation
        # import pdb; pdb.set_trace()
        # (B, C, F, T) * (B, C, F, T) -> (B, C, F, T) 
        imag_ref = mag_ref * torch.sin(phase_ref)
        real_ref = mag_ref * torch.cos(phase_ref)
        # (B, C, F, T) * (B, C, F, T) -> (B, C, F, T) 
        imag_rev = mag_rev * torch.sin(phase_rev)
        real_rev = mag_rev * torch.cos(phase_rev)
        # e_real: (B, F, T) -> (B, C, F, T), (B, C, F, T) -> (B, C, F, T)
        real_est = e_real.unsqueeze(1) * real_rev - e_imag.unsqueeze(1) * imag_rev
        imag_est = e_real.unsqueeze(1) * imag_rev + e_imag.unsqueeze(1) * real_rev
        imag_est = imag_est + 1.0e-10


        # (B, C, F, T) -> (B, C, F, T)
        mag_est = (real_est ** 2 + imag_est ** 2) ** 0.5
        # (B, C, F, T) -> (B, C, F, T)
        pha_est = torch.atan2(imag_est, real_est)
        # (B, C, F, T) -> (B*C, F, T) -> (B*C, t)
        B, C, F, T = mag_est.shape
        ests = self.istft(mag_est.view(-1, F, T), pha_est.view(-1, F, T), squeeze=True)
        # (B*C, t) -> (B, C, t)
        ests = ests.view(B, C, -1)

        return [ests], [mag_est, mag_ref], [real_ref, imag_ref, real_est, imag_est]

    def _build_blocks(self, num_blocks, **block_kwargs):
        """
        Build Conv1D block
        """
        blocks = [Conv1DBlock(dilation=(2 ** b), **block_kwargs) for b in range(num_blocks)]
        return nn.Sequential(*blocks)

    def _build_repeats(self, num_repeats, num_blocks, **block_kwargs):
        """
        Build Conv1D block repeats
        """
        repeats = [
            self._build_blocks(num_blocks, **block_kwargs)
            for r in range(num_repeats)
            ]
        return nn.Sequential(*repeats)


if __name__ == '__main__':
    pass

