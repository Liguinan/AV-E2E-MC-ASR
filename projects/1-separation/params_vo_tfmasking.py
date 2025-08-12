gpu_id = '2'
# ================ Data processing =================
sampling_rate = 16000  # Hz
out_spk = 1
batch_size = 32
max_epoch = 100
diag_loading_ratio=1e-5
mvdr_vector_eps=1e-5
add_visual=True # or False

# ================= Network ===================== #
sample_duration = 0.0025  # s / 2.5ms
L = int(sampling_rate * sample_duration)  # length of the filters [20] samples
N = 256  # number of filters in encoder
B = 256  # number of channels in bottleneck 1x1-conv block
H = 512  # number of channels in convolutional blocks
P = 3  # kernel size in convolutional bolcks
X = 8  # number of convolutional blocks in each repeat
R = 4  # number of repeats
V = 256
U = 128
norm = "BN"  # /cLN/gLN/BN
causal = False
activation_function = 'linear'  # /sigmoid/softmax/linear/relu
model_type = 'uTGT'  # 'hard-encoder-tasnet/encoder-beam-tasnet/beam-tasnet/dir-informed-tgt-tasnet/dir-informed-pit-tasnet
fusion_idx = 0
av_fusion_idx = 1
fix_stft = True
cosIPD = True
sinIPD = False

# ================= DF settings ===================== #
debug_mode = 0
# input_features = ['LPS', 'IPD', 'AF']
input_features = ['LPS', 'IPD']
speaker_feature_dim = 1  # or 2
# 0.56, 0.42, 0.3, 0.28m, 0.2, 0.12,0.1,0.05,0.01
mic_pairs = [[0, 14], [1, 13], [2, 12], [0, 6], [11, 3], [10, 4], [11, 7], [6, 9], [7, 8]]
n_mic = 15
merge_mode = 'sum'
FFT_SIZE = 512
HOP_SIZE = 256
NEFF = FFT_SIZE // 2 + 1
factor = 10
lip_fea = 'lipemb' # or 'landmark'
# loss_type = ['SISNR', 'LFB', 'SISNR_NOISE'] #or 'SISNR'
loss_type = ['SISNR', 'SISNR_NOISE'] 

# ================= Training ===================== #
seed = 20220620
log_display_step = 100
lr = 1e-3
# lr = 5e-4
lr_decay = 1e-5
cuda = True  # True
max_error_deg = 5
s_max_keep_inteval = int(sampling_rate * 1.0)
s_min_keep_inteval = int(sampling_rate * 0.25)
num_mel_bins = 40

# ================= Evaluating ===================== #
write_wav = True
max_test_wav = 3000000000000000
inference_dataset_name = 'test' # val  train_pretrain
ckpt_epoch = 63
wav_dir_name = f"{inference_dataset_name}-{ckpt_epoch}"
model_name = "uTGT-LPS_IPD_AF-b12-BN-fd1-fixTrue-f0-cosTrue-sinFalse-best-76.pt.tar"


# ================= Directory ===================== #
data_dir = '/project_bdda7/bdda/gnli/Data/lrs2_new/pkl_data'
sum_dir = './summary_{}-{}-{}'.format(lip_fea, '+'.join(input_features), '+'.join(loss_type))
model_save_dir = sum_dir + '/model_tfmasking/'
log_dir = sum_dir + '/log/'
loss_dir = sum_dir + '/loss/'
model_name = '{}-{}-b{}-{}-fd{}-fix{}-f{}-cos{}-sin{}'.format(model_type,
                                                              '_'.join(input_features),
                                                              batch_size, norm, speaker_feature_dim, fix_stft,
                                                              fusion_idx, cosIPD, sinIPD)
