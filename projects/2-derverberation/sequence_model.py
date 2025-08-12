# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ sequence_model.py ]
#   Synopsis     [ the RNN model for speech separation ]
#   Source       [ The code is from https://github.com/funcwj/uPIT-for-speech-separation ]
"""*********************************************************************************************"""

import torch
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence, pack_padded_sequence



class SeqModel(torch.nn.Module):
    def __init__(self,
                 input_dim,
                 num_bins,
                 rnn="lstm",
                 complex_mask=True,
                 num_layers=3,
                 hidden_size=896,
                 dropout=0.0,
                 non_linear="linear",
                 bidirectional=True):
        super(SeqModel, self).__init__()
        if non_linear not in ["relu", "sigmoid", "tanh", "linear"]:
            raise ValueError(
                "Unsupported non-linear type:{}".format(non_linear))

        rnn = rnn.upper()
        if rnn not in ['RNN', 'LSTM', 'GRU']:
            raise ValueError("Unsupported rnn type: {}".format(rnn))
        self.rnn = getattr(torch.nn, rnn)(
            input_dim,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional)
        self.drops = torch.nn.Dropout(p=dropout)

        mask_num = 2  # default complex mask
        if not complex_mask:
            mask_num = 1

        self.linear = torch.nn.ModuleList([
            torch.nn.Linear(hidden_size * 2 if bidirectional
                            else hidden_size, num_bins)
            for _ in range(mask_num)
        ])
        self.non_linear = {
            "relu": torch.nn.functional.relu,
            "sigmoid": torch.nn.functional.sigmoid,
            "tanh": torch.nn.functional.tanh,
            "linear": None
        }[non_linear]
        self.num_bins = num_bins

    def forward(self, x, x_len):
        """
        x: (B, F, T)
        x_len: (B, )
        """
        # import pdb; pdb.set_trace()
        x = pack_padded_sequence(x.permute(0, 2, 1), x_len.cpu(), batch_first=True, enforce_sorted=False)
        x, _ = self.rnn(x)
        # using unpacked sequence
        # x [bs, seq_len, feat_dim]
        x, _ = pad_packed_sequence(x, batch_first=True)
        x = self.drops(x)
        mask = []
        for linear in self.linear:
            y = linear(x)
            # import pdb; pdb.set_trace()
            if self.non_linear:
                y = self.non_linear(y)
            # if not train:
            #     y = y.view(-1, self.num_bins)
            y = y.permute(0, 2, 1)
            mask.append(y)
        return mask
