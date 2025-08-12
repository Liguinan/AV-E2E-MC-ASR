gpu_id = '2'
task = 'tf-masking'
# ================ Data processing =================
sampling_rate = 16000  # Hz
out_spk = 1
batch_size = 64
max_epoch = 100
diag_loading_ratio='1e-5'
add_visual=True # or False
AF_premasking=False

# ========================path ==========
# original mixture pickle path
training_path = "/project_bdda7/bdda/gnli/Data/lrs2_new/pkl_data/train_pretrain_32_rev1_le6_ark.pkl"
validation_path = "/project_bdda7/bdda/gnli/Data/lrs2_new/pkl_data/val_rev4_le6_ark.pkl"
test_path = "/project_bdda7/bdda/gnli/Data/lrs2_new/pkl_data/test_rev4.pkl"
replay_path = "/project_bdda7/bdda/gnli/Data/lrs2_new/pkl_data/test_replay.pkl"



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
input_features = ['LPS', 'IPD', 'AF']
# input_features = ['LPS', 'IPD']
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
seed = 0
log_display_step = 100
if task=='tf-masking':
    lr = 1e-3
else: # for mvdr
    lr = 5e-4
lr_decay = 1e-5
cuda = True  # True
max_error_deg = 5
s_max_keep_inteval = int(sampling_rate * 1.0)
s_min_keep_inteval = int(sampling_rate * 0.25)
num_mel_bins = 40
# =================Training =======================#
training_model_subpath="uTGT-LPS_IPD_AF-b32-BN-fd1-fixTrue-f0-cosTrue-sinFalse-best-50.pt.tar"


# ================= Evaluating ===================== #
replay=False
write_wav = True
all_metrics = False
inference_dataset_name = 'train_pretrain' # val train_pretrain test replay
ckpt_epoch = 50  # 48 50
# wav_dir_name = f"{inference_dataset_name}-{ckpt_epoch}"
wav_dir_name = f"{inference_dataset_name}_eps={ckpt_epoch}_dl={diag_loading_ratio}_hop_size={HOP_SIZE}"
ckpt = f"uTGT-LPS_IPD_AF-b32-BN-fd1-fixTrue-f0-cosTrue-sinFalse-best-{ckpt_epoch}.pt.tar"


# ================= Directory ===================== #
data_dir = '/project_bdda7/bdda/gnli/Data/lrs2_new/pkl_data'
sum_dir = './summary_{}-{}-{}'.format(lip_fea, '+'.join(input_features), '+'.join(loss_type))

model_save_dir = sum_dir + f'/model_tfmasking-AF_premasking={AF_premasking}-Batch_Size={batch_size}=Add_Visual={add_visual}/'





log_dir = model_save_dir + '/log/'
loss_dir = sum_dir + '/loss/'
model_name = '{}-{}-b{}-{}-fd{}-fix{}-f{}-cos{}-sin{}'.format(model_type,
                                                              '_'.join(input_features),
                                                              batch_size, norm, speaker_feature_dim, fix_stft,
                                                              fusion_idx, cosIPD, sinIPD)

# import os
if write_wav:
    write_name = f'{model_save_dir}/{wav_dir_name}'
#     if not os.path.exists(write_name):
#         os.makedirs(write_name)
