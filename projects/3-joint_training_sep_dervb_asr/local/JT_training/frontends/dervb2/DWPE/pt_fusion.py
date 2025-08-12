from .params import *
import torch
from torch import nn
import torch.nn.functional as F


class FactorizedLayer(nn.Module):
    def __init__(self, factor=10, audio_features=256,
                 other_features=256, out_features=256):
        super(FactorizedLayer, self).__init__()
        self.factor = factor
        self.L = nn.Parameter(torch.rand(factor, audio_features, out_features), True)
        nn.init.xavier_normal(self.L)

        self.factor_layer = nn.Linear(in_features=other_features,
                                      out_features=factor)

    def forward(self, a, o):
        """
        a: audio features with shape     [Batch size (BS), K, B]
        o: other features with shape     [Batch size (BS), K, V]
        :return: h_out [BS, O, K]
        """
        # [BS, K, O]
        factorized_a = [F.relu(torch.matmul(a, self.L[i]))
                        for i in range(self.factor)]
        # [BS, K, F]
        factorized_o = F.softmax(self.factor_layer(o), dim=-1)

        h_out_list = [factorized_a[i] * factorized_o[:, :, i].unsqueeze(-1)
                      for i in range(self.factor)]
        h_out = h_out_list[0]
        for i in range(1, self.factor):
            h_out += h_out_list[i]
        h_out = h_out.permute((0, 2, 1))
        return h_out


class AudioVisualSpkEmbFusion(nn.Module):
    def __init__(self,
                 audio_features=256,
                 video_features=256,
                 spk_features=256,
                 out_features=256):
        super(AudioVisualSpkEmbFusion, self).__init__()
        self.audio_dim = audio_features
        self.spk_emb_dim = spk_features
        self.video_dim = video_features

        self.fc = nn.Linear(in_features=spk_features, out_features=spk_features)
        self.factor_layer = FactorizedLayer(factor=factor,
                                            audio_features=audio_features,
                                            other_features=spk_features,
                                            out_features=out_features)
        self.conv1d = nn.Conv1d(out_features + video_features, out_features, 1)

    def forward(self, a, s, v):
        """
        a: audio features with shape [Batch size (BS), B, K]
        v: video features with shape [Batch size (BS), V, T]
        s: speaker embedding with shape [Batch size (BS), U]
        """
        if a.size(1) != self.audio_dim or v.size(1) != self.video_dim or s.size(1) != self.spk_emb_dim:
            raise RuntimeError("Dimension mismatch for audio/video features, "
                               "{:d}/{:d}/{:d} vs {:d}/{:d}/{:d}".format(
                a.size(1), v.size(1), s.size(1), self.audio_dim, self.video_dim, self.spk_emb_dim))
        # upsample visual features, so T matches K => [BS, V, K]
        v = F.interpolate(v, size=a.size(-1))
        a = a.permute((0, 2, 1))    # [BS, K, b]
        # [BS, U] => [BS, U]
        s = F.sigmoid(self.fc(s))
        # repeat K times, [BS, U] => [BS, K, U]
        s = s.unsqueeze(-1).repeat(1, 1, a.size(1)).permute((0, 2, 1))

        # [BS, B, K]
        a_s = self.factor_layer(a, s)
        # [BS, B, K] concat [BS, V, K] => [BS, B+V, K]
        y = torch.cat((a_s, v), dim=1)
        # [BS, B+V, K] => [BS, B', K]
        y = self.conv1d(y)

        return y


class AudioVisualFusion(nn.Module):
    """
    Fusion layer: audio/visual features
    """

    def __init__(self,
                 audio_features=256,
                 video_features=256,
                 spk_features=256,
                 out_features=256):
        super(AudioVisualFusion, self).__init__()
        self.audio_dim = audio_features
        self.spk_emb_dim = spk_features
        self.video_dim = video_features * speaker_feature_dim
        self.conv1d = nn.Conv1d(audio_features + video_features + spk_features, out_features, 1)

    def forward(self, a, s, v):
        """
        a: audio features with shape [Batch size (BS), B, K]
        v: video features with shape [Batch size (BS), V, T]
        s: speaker embedding with shape [Batch size (BS), U]
        """
        if a.size(1) != self.audio_dim or v.size(1) != self.video_dim or s.size(1) != self.spk_emb_dim:
            raise RuntimeError("Dimention mismatch for audio/video features, "
                               "{:d}/{:d}/{:d} vs {:d}/{:d}/{:d}".format(
                a.size(1), v.size(1), s.size(1), self.audio_dim, self.video_dim, self.spk_emb_dim))
        # upsample visual features, so T matches K
        v = F.interpolate(v, size=a.size(-1))

        # repeat K times, [BS, U] => [BS, U, K]
        s = s.unsqueeze(-1).repeat(1, 1, a.size(-1))
        # [BS, B, K] concat [BS, U, K] concat [BS, V, K] => [BS, B+U+V, K]
        y = torch.cat((a, s, v), dim=1)
        # [BS, B+V, K] => [BS, B', K]
        y = self.conv1d(y)

        return y
