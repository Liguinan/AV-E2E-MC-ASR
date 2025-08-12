gpu_id = '0'
task = 'mvdr'
# ================ Data processing =================
sampling_rate = 16000  # Hz
out_spk = 1
batch_size = 32
max_epoch = 200
diag_loading_ratio='1e-5'
add_visual=True # or False
AF_premasking=False
resume=False
# taps=None
# delay=None
use_torch_solver=True

# ========================path ==========
# original mixture pickle path
training_path = "/project_bdda7/bdda/gnli/Data/lrs2_new/pkl_data/train_pretrain_32_rev1_le6_ark.pkl"
validation_path = "/project_bdda7/bdda/gnli/Data/lrs2_new/pkl_data/val_rev4_le6_ark.pkl"
test_path = "/project_bdda7/bdda/gnli/Data/lrs2_new/pkl_data/test_rev4.pkl"
replay_path = "/project_bdda7/bdda/gnli/Data/lrs2_new/pkl_data/test_replay.pkl"

# new mixture wav from upstream task
root_path="/users/bdda/gnli/projects/bdda8_projects/TASLP-22/Front-End/dereverberation/DNN-WPE-CM/summary_sys20_dervb_add_visual-True/model_dnn-wpe_pf-1e-10_scale_by_mixture-False_hop_size-128"

dl='1e-6'
taps=2 # 1 2 3 4 5 7 9 15 18 21 24 27
# print(f"tfmaksing eps=26 -> new wav mixture taps={taps}, dl={dl}")
training_wav_scp = f"{root_path}/train_pretrain_eps=26_dl={dl}_taps={taps}_delay=2_norm=True_sampling_frame_ratio=1/wav.scp"
validation_wav_scp = f"{root_path}/val_eps=26_dl={dl}_taps={taps}_delay=2_norm=True_sampling_frame_ratio=1/wav.scp"
test_wav_scp = f"{root_path}/test_eps=26_dl={dl}_taps={taps}_delay=2_norm=True_sampling_frame_ratio=1/wav.scp"
replay_wav_scp = f"{root_path}/replay_eps=26_dl={dl}_taps={taps}_delay=2_norm=True_sampling_frame_ratio=1/wav.scp"
new_wav_scp = True



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

if not AF_premasking:
    if resume:
        training_model_subpath="model_mvdr-AF_premasking=False/uTGT-LPS_IPD_AF-b12-BN-fd1-fixTrue-f0-cosTrue-sinFalse-last-xxx.pt.tar"
    else:
        # for tfmasking
        # training_model_subpath="uTGT-LPS_IPD_AF-b32-BN-fd1-fixTrue-f0-cosTrue-sinFalse-best-48.pt.tar"
        ep=55
        training_model_subpath=f"model_tfmasking-AF_premasking=False-Batch_Size=64=Add_Visual={add_visual}_dl={dl}_taps={taps}/uTGT-LPS_IPD_AF-b64-BN-fd1-fixTrue-f0-cosTrue-sinFalse-best-{ep}.pt.tar"
        # dl=1e-5_taps=6
        # training_model_subpath="model_tfmasking-AF_premasking=False-Batch_Size=64=Add_Visual=False_dl=1e-5_taps=6/uTGT-LPS_IPD_AF-b64-BN-fd1-fixTrue-f0-cosTrue-sinFalse-best-30.pt.tar"
        print(f"mvdr -> new wav mixture taps={taps}, dl={dl}, tfmaksing epoch: {ep}")

        
        
        

# ================= Evaluating ===================== #
replay=False
write_wav = True
all_metrics = True
inference_dataset_name = 'replay' # val train_pretrain test replay
ckpt_epoch = 65  # 76 78 73 37 54 34
# wav_dir_name = f"{inference_dataset_name}-{ckpt_epoch}"
wav_dir_name = f"{inference_dataset_name}_eps={ckpt_epoch}_dl={diag_loading_ratio}"
ckpt = f"uTGT-LPS_IPD_AF-b32-BN-fd1-fixTrue-f0-cosTrue-sinFalse-best-{ckpt_epoch}.pt.tar"


# ================= Directory ===================== #
#data_dir = '/data1/tomasyu/experiment/AVSR-2019-11/multi-channel/from_wubo/code_VM_simuData/jwyu/pretrain/scp/train_pretrain/''
data_dir = '/project_bdda7/bdda/gnli/Data/lrs2_new/pkl_data'
# sum_dir = './summary_0{}-{}-{}'.format(lip_fea, '+'.join(input_features), '+'.join(loss_type))
sum_dir = './summary_sys24_av_DWPE_av_{}-{}-{}'.format(lip_fea, '+'.join(input_features), '+'.join(loss_type))
model_save_dir = sum_dir + f'/model_mvdr-AF_premasking={AF_premasking}-BS={batch_size}-DL={diag_loading_ratio}_dl={dl}_taps={taps}/'
# model_save_dir = sum_dir + f'/model_mvdr-AF_premasking={AF_premasking}-BS={batch_size}-DL={diag_loading_ratio}_dl=1e-5_taps=6/'




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
