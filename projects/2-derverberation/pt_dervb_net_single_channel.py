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
from torch_complex import ComplexTensor
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from pytorch_wpe import wpe_one_iteration



SQRT2 = 1.41421356237
def complex_clamp(X: ComplexTensor, abs_min: float = None, abs_max: float = None):
    if abs_min is not None:
        thres = X.real.new_tensor(abs_min / SQRT2)
        X = ComplexTensor(
            torch.where(X.abs() >= abs_min, X.real, thres),
            torch.where(X.abs() >= abs_min, X.imag, thres),
        )
    if abs_max is not None:
        thres = X.real.new_tensor(abs_max / SQRT2)
        X = ComplexTensor(
            torch.where(X.abs() <= abs_max, X.real, thres),
            torch.where(X.abs() <= abs_max, X.imag, thres),
        )
    return X




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

        self.taps = taps
        self.delay = delay
        self.normalization = normalization
        self.inverse_power = True
        
        # import pdb; pdb.set_trace()
        # Feature extractor
        self.df_computer = DFComputer(frame_hop=HOP_SIZE, frame_len=FFT_SIZE, in_feature=['LPS'])

        # audio
        self.conv1x1_1 = Conv1D(self.df_computer.df_dim, B, 1)  # for audio dim
        # import pdb; pdb.set_trace()
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
        
        # if layer_norm_visual_1:
        #     self.ln_visual_512 = ChannelWiseLayerNorm(512)
        # if layer_norm_visual_2:
        #     self.ln_visual_256 = ChannelWiseLayerNorm(256)

    def check_forward_args(self, all_x):
        x = all_x[0]  
        lip_video = all_x[1]  
        ref = all_x[2]  
        lens = all_x[3]  
        # todo check dimension
        return x, lip_video, ref, lens

    def backward_hook(self, module, grad_input, grad_output):
        # print(f"module:{module}")
        import pdb; pdb.set_trace()
        print(module)  # 打印模块名，用于区分模块
        print('grad_output', grad_output)  # 打印该模块输出端的梯度
        print('grad_input', grad_input)    # 打印该模块输入端的梯度
        # self.total_grad_in.append(grad_input)   # 保存该模块输入端的梯度
        # self.total_grad_out.append(grad_output) # 保存该模块输出端的梯度
        return grad_input
    
    def forward(self, all_x):
        # import pdb; pdb.set_trace() 
        # x: (B, C, t), lip_fea:(B, V_T, V_F: 512), ref: (B, C, t), lens:(B,)
        x, video_fea, ref, lens = self.check_forward_args(all_x)

        # Feature extractor
        # audio_fea: (B, F, T), mag_rev: (B, C, F, T), phase_rev: (B, C, F, T)
        audio_fea, mag_rev, phase_rev = self.df_computer([x])
        # mag_ref: (B, C, F, T), phase_ref: (B, C, F, T)
        _, mag_ref, phase_ref = self.df_computer([ref])

        # (B, C, F, T) * (B, C, F, T) -> (B, C, F, T) 
        imag_rev = mag_rev * torch.sin(phase_rev)
        real_rev = mag_rev * torch.cos(phase_rev)
        enhanced = rev = ComplexTensor(real_rev, imag_rev)
        # (B, C, F, T) * (B, C, F, T) -> (B, C, F, T) 
        imag_ref = mag_ref * torch.sin(phase_ref)
        real_ref = mag_ref * torch.cos(phase_ref)
        ref = ComplexTensor(real_ref, imag_ref)

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
        
        # if layer_norm_visual_1:
        #     video_fea = self.ln_visual_512(video_fea.permute(0,2,1)).permute(0,2,1)
        
        # video
        # (B, V_T, V_F: 512) -> (B, V_T, F')
        video_fea = self.fc1(video_fea)
        if not add_visual:
            # (B, V_T, F') -> (B, V_T, F')
            video_fea = torch.zeros_like(video_fea)
        
        if TCN_for_lip:
            # (B, V_T, F') -> (B, F', T)
            video_fea = video_fea.permute(0, 2, 1)
            
        # (B, V_T, F') -> (B, F', V_T)
        video_fea = self.lip_blocks(video_fea)
        # (B, F', V_T) -> (B, F', T)
        video_fea = nn.functional.interpolate(video_fea, size=audio_fea.size(2), mode=mode)
        
        # if layer_norm_visual_2:
        #     video_fea = self.ln_visual_256(video_fea)
        
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

        # import pdb; pdb.set_trace()
        # (B, F, T) -> (B, F, T) 
        mask_real = e_real.masked_fill(make_pad_mask(lens, e_real, length_dim=2), 0)
        mask_imag = e_imag.masked_fill(make_pad_mask(lens, e_imag, length_dim=2), 0)

        mask = ComplexTensor(mask_real, mask_imag)
        # mask = ComplexTensor(mask_real, mask_imag).double()
        # mask = complex_clamp(mask, abs_min=1e-6).double()
        
        
        # # import pdb; pdb.set_trace()
        if self.normalization:
            # Normalize along T
            # denominator: (B, F, T) -> (B, F) -> (B, F, 1) for broadcast
            # mask: (B, F, T)
            # import pdb; pdb.set_trace()
            mask = mask / (mask.sum(dim=-1, keepdim=True) + 1e-10)
        #     mask_real = mask_real / (mask_real.sum(dim=-1, keepdim=True) + 1e-10)
        #     mask_real = mask_imag / (mask_imag.sum(dim=-1, keepdim=True) + 1e-10)
            
        if torch.any(torch.isnan(mask.real)):
            import pdb; pdb.set_trace()
        if torch.any(torch.isnan(mask.imag)):
            import pdb; pdb.set_trace()
        
        # {(B, C, F, T) -> (B, F, C, T)} * {(B, F, T)->(B, F, C, T)} -> (B, F, C, T)
        # enhanced = enhanced.permute(0, 2, 1, 3).double() * mask.unsqueeze(-2)
        enhanced = enhanced.permute(0, 2, 1, 3) * mask.unsqueeze(-2)
        enhanced.imag = enhanced.imag + 1e-10
        
    
        #### 1. transform to wav  2. scale by mixture  3. transform to stft 4.calculate power
        ## 1. transform to wav 
        # (B, C, F, T) -> (B, C, F, T)
        mag_est = enhanced.permute(0, 2, 1, 3).abs()
        # (B, C, F, T) -> (B, C, F, T)
        pha_est = enhanced.permute(0, 2, 1, 3).angle()
        # (B, C, F, T) -> (B*C, F, T) -> (B*C, t)
        B, C, F_1, T = mag_est.shape
        ests = self.istft(mag_est.view(-1, F_1, T), pha_est.view(-1, F_1, T), squeeze=True)
        # (B*C, t) -> (B, C, t)
        ests = ests.view(B, C, -1)
        
        ## 2. scale by mixture
        if scale_by_mixture:
            # import pdb; pdb.set_trace()
            norm_ests = torch.max(torch.abs(ests), dim=2, keepdim=True)[0]
            norm_mix = torch.max(torch.abs(x), dim=2, keepdim=True)[0]
            # norm_ests = torch.max(ests, dim=2, keepdim=True)[0]
            # norm_mix = torch.max(x, dim=2, keepdim=True)[0]
            ests_scale = ests * norm_mix / (norm_ests + 1e-8)
            
            ## 3. transform to stft
            batch_size, n_channel, S_ = ests_scale.shape
            #print(x.shape)
            # B, M, S -> BxM, S
            all_s = ests_scale.view(-1, S_)
            # BxM, F, K
            magnitude, phase = self.df_computer.stft(all_s)
            _, F_, K_ = phase.shape
            # B, C, F, T
            # phase = phase.view(batch_size, n_channel, F_, K_)
            magnitude = magnitude.view(batch_size, n_channel, F_, K_)
            
            
            ## 4.calculate power
            # import pdb; pdb.set_trace()
            #  -> (B, F, T)
            power = (magnitude ** 2).mean(dim=-3)
        else:
            # power: (B, F, C, T)
            # import pdb; pdb.set_trace()
            enhanced.imag = enhanced.imag + 1e-10
            power = enhanced.abs() ** 2

            # Averaging along the channel axis: (B, F, C, T) -> (B, F, T)
            power = power.mean(dim=-2)
            
            # import pdb; pdb.set_trace()

        ref_power = (ref.abs() ** 2).mean(dim=1)
        print(f"est_power max: {torch.max(power)}, est_power min: {torch.min(power)}")
        print(f"ref_power max: {torch.max(ref_power)}, ref_power min: {torch.min(ref_power)}")
        if est_power:
            # (B, C, F, T) -> (B, F, T)
            ref_power = (ref.abs() ** 2).mean(dim=1)
            return [power, ref_power]

        # import pdb; pdb.set_trace()
        # enhanced: (..., C, T) -> (..., C, T)
        enhanced = wpe_one_iteration(
            rev.permute(0, 2, 1, 3).contiguous(),
            power,
            taps=self.taps, 
            delay=self.delay,
            inverse_power=self.inverse_power,
            diag_loading_ratio=float(diag_loading_ratio),
            eps=float(power_flooring))

        # if torch.any(torch.isnan(enhanced.real)):
        #     import pdb; pdb.set_trace()
        # if torch.any(torch.isnan(enhanced.imag)):
        #     import pdb; pdb.set_trace()

        # (B, F, C, T) -> (B, F, C, T)
        enhanced.masked_fill_(make_pad_mask(lens, enhanced.real), 0)

        # (B, F, C, T) -> (B, C, F, T)
        enhanced = enhanced.permute(0, 2, 1, 3).float().contiguous()
        # enhanced = enhanced.permute(0, 2, 1, 3)
        # (B, C, F, T) -> (B*C, F, T) -> (B*C, t)
        B, C, F, T = enhanced.shape
        # import pdb; pdb.set_trace()
        # ests: (B, t)
        ests = self.istft(enhanced.abs().view(-1, F, T), enhanced.angle().view(-1, F, T), squeeze=True)
        # (B*C, t) -> (B, C, t)
        ests = ests.view(B, C, -1)
    

        return [ests], [enhanced.abs(), ref.abs()], [real_ref, imag_ref, enhanced.real, enhanced.imag]

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

