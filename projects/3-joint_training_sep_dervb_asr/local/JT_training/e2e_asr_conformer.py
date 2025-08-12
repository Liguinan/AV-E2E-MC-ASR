# Copyright 2020 Johns Hopkins University (Shinji Watanabe)
#                Northwestern Polytechnical University (Pengcheng Guo)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""
Conformer speech recognition model (pytorch).

It is a fusion of `e2e_asr_transformer.py`
Refer to: https://arxiv.org/abs/2005.08100

"""

from espnet.nets.pytorch_backend.conformer.argument import (  # noqa: H301
    add_arguments_conformer_common,
    verify_rel_pos_type,
)
from espnet.nets.pytorch_backend.conformer.encoder import Encoder
# from espnet.nets.pytorch_backend.e2e_asr_transformer import E2E as E2ETransformer
from espnet.JT_training.e2e_asr_transformer import E2E as E2ETransformer



class E2E(E2ETransformer):
    """E2E module.

    :param int idim: dimension of inputs
    :param int odim: dimension of outputs
    :param Namespace args: argument Namespace containing options

    """

    @staticmethod
    def add_arguments(parser):
        """Add arguments."""
        E2ETransformer.add_arguments(parser)
        E2E.add_conformer_arguments(parser)
        return parser

    @staticmethod
    def add_conformer_arguments(parser):
        """Add arguments for conformer model."""
        group = parser.add_argument_group("conformer model specific setting")
        group = add_arguments_conformer_common(group)
        return parser

    def __init__(self, aidim, vidim, odim, args, ignore_id=-1):
        """Construct an E2E object.

        :param int idim: dimension of inputs
        :param int odim: dimension of outputs
        :param Namespace args: argument Namespace containing options
        """
        super().__init__(aidim, vidim, odim, args, ignore_id)
        if args.transformer_attn_dropout_rate is None:
            args.transformer_attn_dropout_rate = args.dropout_rate
        # import pdb; pdb.set_trace()
        # Check the relative positional encoding type
        args = verify_rel_pos_type(args)

        self.encoder = Encoder(
            idim=self.encoder_dim, # 83
            attention_dim=args.adim, # 256
            attention_heads=args.aheads, # 4
            linear_units=args.eunits, # 2048
            num_blocks=args.elayers, # 12
            input_layer=args.transformer_input_layer, # conv2d
            dropout_rate=args.dropout_rate, # 0.1
            positional_dropout_rate=args.dropout_rate, # 0.1
            attention_dropout_rate=args.transformer_attn_dropout_rate, # 0.0
            pos_enc_layer_type=args.transformer_encoder_pos_enc_layer_type, # legacy_rel_pos
            selfattention_layer_type=args.transformer_encoder_selfattn_layer_type, # legacy_rel_selfattn
            activation_type=args.transformer_encoder_activation_type, # 'swish'
            macaron_style=args.macaron_style, # 1
            use_cnn_module=args.use_cnn_module, # 1
            zero_triu=args.zero_triu, # False
            cnn_module_kernel=args.cnn_module_kernel, # 31
            stochastic_depth_rate=args.stochastic_depth_rate, # 0.0
            intermediate_layers=self.intermediate_ctc_layers, # None
            ctc_softmax=self.ctc.softmax if args.self_conditioning else None, # None
            conditioning_layer_dim=odim, # 502
        )
        self.reset_parameters(args)
