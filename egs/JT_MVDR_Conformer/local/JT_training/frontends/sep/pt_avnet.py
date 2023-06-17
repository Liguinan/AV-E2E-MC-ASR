import torch
import torch.nn as nn
from params import *
import torch.nn.functional as F
import math
#from pt_audio_fea_test import DFComputer, iSTFT
from pt_audio_fea import DFComputer,iSTFT
from pt_video_fea import LipReadingNet, OxfordLipNet
from pt_fusion import AudioVisualSpkEmbFusion, AudioVisualFusion,FactorizedLayer
from pt_log_fbank import LFB

def param(nnet, Mb=True):
    """
    Return number parameters(not bytes) in nnet
    """
    neles = sum([param.nelement() for param in nnet.parameters()])
    return neles * 1.0 / 10 ** 6 if Mb else neles


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


class ConvTrans1D(nn.ConvTranspose1d):
    """
    1D conv transpose in ConvTasNet
    """

    def __init__(self, *args, **kwargs):
        super(ConvTrans1D, self).__init__(*args, **kwargs)

    def forward(self, x, squeeze=False):
        """
        x: N x L or N x C x L
        """
        if x.dim() not in [2, 3]:
            raise RuntimeError("{} accept 2/3D tensor as input".format(
                self.__name__))
        x = super(ConvTrans1D, self).forward(x if x.dim() == 3 else torch.unsqueeze(x, 1))
        if squeeze:
            x = torch.squeeze(x)
        return x


# for each 1-D convolutional block (depth-wise separable convolution)
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


class ConvTasNet(nn.Module):
    def __init__(self,
                 norm=norm,
                 out_spk=out_spk,
                 non_linear=activation_function,
                 causal=causal,
                 input_features=input_features,
                 spk_fea_dim=speaker_feature_dim,
                 cosIPD=cosIPD,
                 sinIPD=sinIPD,
                 V=V,
                 av_fusion_idx=av_fusion_idx
                 ):
        super(ConvTasNet, self).__init__()

        supported_nonlinear = {
            "relu": F.relu,
            "sigmoid": torch.sigmoid,
            "softmax": F.softmax,
            "linear": None
        }
        if non_linear not in supported_nonlinear:
            raise RuntimeError("Unsupported non-linear function: {}",
                               format(non_linear))
        self.non_linear_type = non_linear  # string
        self.av_fusion_idx = av_fusion_idx
        self.lip_blocks = OxfordLipNet(embedding_dim=V, conv_channels=V)
        self.df_computer = DFComputer(frame_hop=HOP_SIZE,
                                      frame_len=FFT_SIZE,
                                      in_feature=input_features,
                                      merge_mode=merge_mode,
                                      cosIPD=cosIPD,
                                      sinIPD=sinIPD,
                                      speaker_feature_dim=speaker_feature_dim)
        #self.lip_reading_net = LipReadingNet(backend_dim=V)
        self.fc1 = nn.Linear(190, 512)
        self.fc2 = nn.Linear(512, V)

        #self.lip_blocks = OxfordLipNet(embedding_dim=V, conv_channels=V) # disable lipnet since we use lip embedding here

        dim_conv = self.df_computer.df_dim
        # 2> Module: Separator
        # <2.2> 1x1 conv
        # input: [BS, N, K] -> output: [BS, B, K]
        self.conv1x1_1 = Conv1D(dim_conv, B, 1)
        # <2.3> repeat blocks
        # input: [BS, B, K] -> output: [BS, B, K], the dimension remains
        self.audio_blocks = self._build_repeats(
            num_repeats=av_fusion_idx,
            num_blocks=X,
            in_channels=B,
            conv_channels=H,
            kernel_size=P,
            norm=norm,
            causal=causal)
        self.av_fusion_layer = FactorizedLayer(audio_features=B,
                                                        other_features=V,
#                                                        spk_features=U,
                                                        out_features=B)
        self.fusion_blocks = self._build_repeats(
            # num_repeats=R - av_fusion_idx,
            num_repeats=1,
            num_blocks=X,
            in_channels=B,
            conv_channels=H,
            kernel_size=P,
            norm=norm,
            causal=causal)

        # <2.4> output 1x1 Conv
        # input: [BS, B, K]
        # output: [BS, nspk*N, K]
        self.conv1x1_2_real = Conv1D(B, out_spk * self.df_computer.num_bins, 1)
        self.conv1x1_2_imag = Conv1D(B, out_spk * self.df_computer.num_bins, 1)
        # Adding noise output branch
        self.fusion_blocks_noise = self._build_repeats(
            # num_repeats=R - av_fusion_idx,
            num_repeats=1,
            num_blocks=X,
            in_channels=B,
            conv_channels=H,
            kernel_size=P,
            norm=norm,
            causal=causal)        
        self.conv1x1_2_real_noise = Conv1D(B, out_spk * self.df_computer.num_bins, 1)
        self.conv1x1_2_imag_noise = Conv1D(B, out_spk * self.df_computer.num_bins, 1)

        # <2.5> output nonlinear 
        # Jianwei Yu This should be disabled since there should be no limitation of the complex mask range
        self.non_linear = supported_nonlinear[non_linear]  # activation function instance

        # 3> Module: Decoder
        # input: [BS, nspk, F, T]
        self.istft = iSTFT(frame_len=FFT_SIZE, frame_hop=HOP_SIZE, num_fft=FFT_SIZE)
        # output: [BS, nspk, S]
        self.out_spk = out_spk
        self.lfb = LFB(num_mel_bins=num_mel_bins)
        self.epsilon = nn.Parameter(torch.tensor(torch.finfo(torch.float).eps), requires_grad=False)

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

    def check_forward_args(self, all_x):

        x = all_x[0]
        directions = all_x[1]
        spk_num = all_x[2]
        lip_video = all_x[3]
        seq_len = all_x[4]
#        spk_emb = all_x[4]

        if x.dim() >= 4:
            raise RuntimeError(
                "{} accept 2/3D tensor as input, but got {:d}".format(
                    self.__name__, x.dim()))

        if lip_video.dim() not in [2, 3, 4, 5]:#!= 3 and lip_video.dim() != 4:
            raise RuntimeError(
                "{} accept 3/4/5D tensor as video input, but got {:d}".format(
                    'Conv-Tavsnet', lip_video.dim()))

        # when inference, only one utt (BS = 1)
        if x.dim() == 2:
            x = torch.unsqueeze(x, 0)
            lip_video = torch.unsqueeze(lip_video, 0)
        if directions.dim() == 1:
            directions = torch.unsqueeze(directions, 0)
        if spk_num.dim() == 0:
            spk_num = torch.unsqueeze(spk_num, 0)
#        if lip_video.dim() == 4:
#            lip_video = torch.unsqueeze(lip_video, 0)
#        if spk_emb.dim() == 1:
#            spk_emb = torch.unsqueeze(spk_emb, 0)
    #print(x.shape, directions.shape, spk_num.shape, lip_video.shape)
        return x, directions, spk_num, lip_video, seq_len #, spk_emb

    def forward(self, all_x):
        '''
        :param all_x:
        [0] x - multi-channel mixture waveforms, [(batch size (B)), n_channel (M), seq_length (S)]
        [1] directions - all speakers' directions in mixture, [(batch size (B)), nspk (C)]
        [2] spk_num - actual speaker number in current wav [batch_size (B)]
        [3*1] lip_video - lip pixel [(batch size (BS), frames (FR), 1, height, width]
    [3*2] lip_video - lip landmark [(batch size (BS), frames (FR), embedding_dim]
        [4] spk_embedding - speaker embedding [(batch size (BS)), speaker_embedding_dimension (U)]
        :return: estimated s: single-channel separated waveforms, [batch size (B), out_spk (1/C), seq_length (S)]
        '''
        x, directions, spk_num, lip_video, seq_len = self.check_forward_args(all_x)
        #import pdb; pdb.set_trace()
        # Module 1.1: <Audio Encoder/STFT>
        audio_fea, mag, phase = self.df_computer([x, directions, spk_num])
        #print(audio_fea.shape)
        # Module 1.2: <Video Encoder>
        if lip_fea == 'pixel':
            video_fea = self.lip_reading_net(lip_video)
        elif lip_fea == 'landmark':
            video_fea = self.fc2(self.fc1(lip_video))
        elif lip_fea == 'lipemb':
            #import pdb; pdb.set_trace()
            video_fea = self.fc2(lip_video)
            if not add_visual:
                print("no add visual")
                video_fea = torch.zeros_like(video_fea)
                # print(f"{video_fea}")


        self.audio_fea, self.video_fea = audio_fea, video_fea

        # Module 2 <Separator>
        # 2.1 <audio blocks> [BS, B, K]
        audio_emb = self.conv1x1_1(audio_fea)
        audio_emb = self.audio_blocks(audio_emb)

        # 2.2 <video blocks> [BS, V, T]
        video_emb = self.lip_blocks(video_fea)

        # 2.3 <fusion layer>
        audio_emb = audio_emb.permute((0, 2, 1))# [BS, K ,B]
        video_emb = F.interpolate(video_emb, size=audio_emb.size(1)).permute((0,2,1))#[BS, K,V]
        av_emb = self.av_fusion_layer(audio_emb, video_emb)
        #avs_emb = self.avs_fusion_layer(audio_emb, spk_emb, video_emb)

        # 2.4 <fusion blocks>
        y = self.fusion_blocks(av_emb)
        y_noise = self.fusion_blocks_noise(av_emb)

        # 2.5 1x1 Conv
        # input: [BS, B, K] => output: [BS, NSPK * N * 2, K]
        e_real = self.conv1x1_2_real(y)
        e_imag = self.conv1x1_2_imag(y)
        e_real_noise = self.conv1x1_2_real_noise(y_noise) 
        e_imag_noise = self.conv1x1_2_imag_noise(y_noise)

        # 2.6 divide output dimension to NSPK speakers
        # input: [BS, NSPK * N] => output: NSPK x [BS, N * 2, K]
        e_real = torch.chunk(e_real, self.out_spk, 1)
        e_imag = torch.chunk(e_imag, self.out_spk, 1)
        e_real_noise = torch.chunk(e_real_noise, self.out_spk, 1)
        e_imag_noise = torch.chunk(e_imag_noise, self.out_spk, 1)

        # 2.7 nonlinear, Here we disable the nolinear activation function, since the complex mask value should be unbounded
        # input: NSPK x [BS, N, K] => output: NSPK x [BS, N, K]
        if self.non_linear_type == "softmax":
            m_real = self.non_linear(torch.stack(e_real, dim=0), dim=0)
            m_imag = self.non_linear(torch.stack(e_imag, dim=0), dim=0)
        elif self.non_linear_type == "linear":
            e_real = torch.stack(e_real, dim=0)
            e_imag = torch.stack(e_imag, dim=0)
            e_real_noise = torch.stack(e_real_noise, dim=0)
            e_imag_noise = torch.stack(e_imag_noise, dim=0)
        else:
            e_real = self.non_linear(torch.stack(e_real, dim=0))
            e_imag = self.non_linear(torch.stack(e_imag, dim=0))
            e_real_noise = self.non_linear(torch.stack(e_real_noise, dim=0))
            e_imag_noise = self.non_linear(torch.stack(e_imag_noise, dim=0))

        #for b in range(m_tgt_real.size(0)):
        #    frame_b = int((seq_len[b] // HOP_SIZE) + 1)
        #    #print(frame_b)
        #    e_real[b, :, frame_b:] = 0
        #    e_imag[b, :, frame_b:] = 0
        #    e_real_noise[b, :, frame_b:] = 0
        #    e_imag_noise[b, :, frame_b:] = 0
    
        # 2.8 mixture * mask
        imag = mag[:, 0] * torch.sin(phase[:, 0]) 
        real = mag[:, 0] * torch.cos(phase[:, 0])
        

        est_real_part = e_real[0] * real - e_imag[0] * imag
        est_imag_part = e_real[0] * imag + e_imag[0] * real
        est_imag_part = est_imag_part+1.0e-10
        est_mag = (est_real_part ** 2 + est_imag_part ** 2) ** 0.5
        est_pha = torch.atan2(est_imag_part, est_real_part)

        # This is the noise branch
        est_real_part_noise = e_real_noise[0] * real - e_imag_noise[0] * imag
        est_imag_part_noise = e_real_noise[0] * imag + e_imag_noise[0] * real
        est_imag_part_noise = est_imag_part_noise + 1.0e-10
        est_mag_noise = (est_real_part_noise ** 2 + est_imag_part_noise ** 2) ** 0.5
        est_pha_noise = torch.atan2(est_imag_part_noise, est_real_part_noise)        
        #est_pha—noise =  torch.atan2(est_imag_part_noise, est_real_part_noise)

        # output: NSPK x [BS, S]
        wav = [self.istft(est_mag, est_pha, squeeze=True)]
        wav_noise = [self.istft(est_mag_noise, est_pha_noise, squeeze=True)]

        return wav, wav_noise,  [est_mag]


def foo_conv1d_block():
    nnet = Conv1DBlock(256, 512, 3, 20)
    print(param(nnet))


def foo_layernorm():
    C, T = 256, 20
    nnet1 = nn.LayerNorm([C, T], elementwise_affine=True)
    print(param(nnet1, Mb=False))
    nnet2 = nn.LayerNorm([C, T], elementwise_affine=False)
    print(param(nnet2, Mb=False))


def foo_conv_tas_net():
    from data_generator import CHDataset
    import numpy as np
    dataset = CHDataset(stage='cv')
    wav_file = dataset.wav_list[15]
    mix, ref, spk_doa, spk_num, lip_video = dataset.get_data(wav_file)

    all_x = [torch.tensor(mix), torch.tensor(np.array(spk_doa)), torch.tensor(spk_num),
             torch.tensor(lip_video)]

    for i in range(len(all_x)):
        print(all_x[i].shape)
#        print(all_x[i])

    nnet = ConvTasNet(norm=norm,
                      causal=causal,
                      input_features=input_features,
                      spk_fea_dim=speaker_feature_dim,
                      cosIPD=cosIPD,
                      sinIPD=sinIPD,
                      av_fusion_idx=av_fusion_idx,
                      V=V)
    # print(nnet)
    print("ConvTasNet #param: {:.2f}".format(param(nnet)))
    x = nnet(all_x)
    print(nnet.audio_fea.shape, nnet.video_fea.shape)
    s1 = x[0]
    print(s1[0].shape)


if __name__ == "__main__":
    foo_conv_tas_net()
