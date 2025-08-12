gpu_id = '2'
task = 'DNN-WPE-Offline'  # DNN-WPE-Offline,  DNN-WPE-Online for parameters setting
# =========================================
est_power = False
scratch = False
scale_by_mixture = False
sampling=False
sampling_frame_ratio=1
TCN_for_lip = False
layer_norm_visual_1 = False
layer_norm_visual_2 = True

## specM model for intialization
if not scratch:
    model_path = '/users/bdda/gnli/projects/bdda7_projects/TASLP-22/Front-End/dereverberation/SpecM/summary_sys14_sep_vo+dervb_add_visual-False/model_b=64_lr=0.001_vblk=5-vfusion=concat/b64-best-42.pt.tar'

visual_fusion_type = 'concat'  # or attention
add_visual = True 
lr = 1e-3
scale = 1
batch_size = 64
FFT_SIZE = 512
HOP_SIZE = 128
NEFF = FFT_SIZE // 2 + 1
diag_loading_ratio='1e-6'
power_flooring='1e-10'
taps = 2
delay = 2
normalization = True 
# use_torch_solver=True
# model_offline = f'model_b-{batch_size}_lr-{lr}-norm-{normalization}_scale-{scale}_scratch-{scratch}_scale_by_mixture-{scale_by_mixture}_hop_size-256'
# model_offline = 'model_dnn-wpe_offline'
model_offline = f'model_dnn-wpe_pf-1e-10_scale_by_mixture-False_hop_size-128'


if task == 'DNN-WPE-Online':
    lr = 5e-4
    HOP_SIZE = 128
    normalization = False
    model_online = f'model_b-{batch_size}_lr-{lr}-norm-{normalization}_scale-{scale}_scratch-{scratch}_taps-{taps}_delay-{delay}_dl-{diag_loading_ratio}_pf-{power_flooring}_scale_by_mixture-{scale_by_mixture}_hop_size-{HOP_SIZE}'
# print(f"lr: {lr}, HOP_SIZE: {HOP_SIZE}, normalization: {normalization}")


# ===============  data path ==============
# original mixture pickle path
training_path = "/project_bdda7/bdda/gnli/Data/lrs2_new/pkl_data/train_pretrain_32_rev1_le6_ark.pkl"
validation_path = "/project_bdda7/bdda/gnli/Data/lrs2_new/pkl_data/val_rev4_le6_ark.pkl"
test_path = "/project_bdda7/bdda/gnli/Data/lrs2_new/pkl_data/test_rev4.pkl"
replay_path = "/project_bdda7/bdda/gnli/Data/lrs2_new/pkl_data/test_replay.pkl"

# switch between original mixture and new wav scp
new_wav_scp = False

# NEW WAV SCP: audio only 
data_path_prefix = '/project_bdda7/bdda/gnli/projects/TASLP-22/Front-End'
# module_prefix =  f"separation/summary_0lipemb-LPS+IPD+AF-SISNR+SISNR_NOISE/model_mvdr-AF_premasking=False-Batch_Size=32-Dloading=1e-5"
# epoch_num = 79  
# training_wavscp = f"{data_path_prefix}/{module_prefix}/train_pretrain_eps={epoch_num}_dl=1e-5/wav.scp"
# validation_wavscp = f"{data_path_prefix}/{module_prefix}/val_eps={epoch_num}_dl=1e-5/wav.scp"
# test_wavscp = f"{data_path_prefix}/{module_prefix}/test_eps={epoch_num}_dl=1e-5/wav.scp"


# vo
# module_prefix =  f"separation/summary_lipemb-LPS+IPD-SISNR+SISNR_NOISE/model_mvdr-AF_premasking=False-Batch_Size=32-Dloading=1e-5-add_visual=True"
# epoch_num = 93
# training_wavscp = f"{data_path_prefix}/{module_prefix}/train_pretrain_eps={epoch_num}_dl=1e-5/wav.scp"
# validation_wavscp = f"{data_path_prefix}/{module_prefix}/val_eps={epoch_num}_dl=1e-5/wav.scp"
# test_wavscp = f"{data_path_prefix}/{module_prefix}/test_eps={epoch_num}_dl=1e-5/wav.scp"


# av
module_prefix =  f"separation/summary_lipemb-LPS+IPD+AF-SISNR+SISNR_NOISE/model_mvdr-AF_premasking=False-Batch_Size=32-Dloading=1e-5-add_visual=True"
epoch_num = 54
training_wavscp = f"{data_path_prefix}/{module_prefix}/train_pretrain_eps={epoch_num}_dl=1e-5/wav.scp"
validation_wavscp = f"{data_path_prefix}/{module_prefix}/val_eps={epoch_num}_dl=1e-5/wav.scp"
test_wavscp = f"{data_path_prefix}/{module_prefix}/test_eps={epoch_num}_dl=1e-5/wav.scp"
replay_wavscp = None

# ================ multichannel dervb ref wav scp (noise + overlap) ====================
mc_ref_wav_scp=True
training_ref_wavscp = '/project_bdda8/bdda/gnli/Data/lrs2_new/wav_data/15C/train_pretrain_32/mixture_direct_path/wav.scp'
validation_ref_wavscp = '/project_bdda8/bdda/gnli/Data/lrs2_new/wav_data/15C/val/mixture_direct_path/wav.scp'
test_ref_wavscp = '/project_bdda8/bdda/gnli/Data/lrs2_new/wav_data/15C/test/mixture_direct_path/wav.scp'





# ================ Data processing =================
sampling_rate = 16000  # Hz
out_spk = 1
max_epoch = 200
n_mic = 15
lip_fea = 'lipemb'
num_mel_bins = 40
lip_block_num = 5
factor = 10
# ==================Fixed =======================#
model_type = 'TCN'  # or 'LSTM', fixed
mode = 'nearest'  # basically fixed
weight_decay = 1e-5  # for regularization

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
# activation_function = 'relu'  # /sigmoid/softmax
# fusion_idx = 0
# av_fusion_idx = 1

# ================= Feature ========================= # 
input_features = ['LPS']
speaker_feature_dim = 1
fix_stft = True
cosIPD = True
sinIPD = False
merge_mode = 'sum'
mic_pairs = [[0, 14], [1, 13], [2, 12], [0, 6], [11, 3], [10, 4], [11, 7], [6, 9], [7, 8]]
n_mic = 15
# ================= Training ===================== #
seed = 3407
log_display_step = 100
cuda = True  # Trues
# ================= Evaluating ===================== #
write_wav = True
all_metrics = True
inference_dataset_name = 'replay' # val train_pretrain test
ckpt_epoch = 26
# wav_dir_name = f"{inference_dataset_name}_eps={ckpt_epoch}_dl={diag_loading_ratio}_delay={delay}_taps={taps}_hop_size={HOP_SIZE}_norm={normalization}_pf={power_flooring}_scale_by_mixture={scale_by_mixture}"
wav_dir_name = f"{inference_dataset_name}_eps={ckpt_epoch}_dl={diag_loading_ratio}_taps={taps}_delay={delay}_norm={normalization}_sampling_frame_ratio={sampling_frame_ratio}"
ckpt = f'b{batch_size}-best-{ckpt_epoch}.pt.tar'
# ================= Directory ===================== #
data_dir = '/project_bdda7/bdda/gnli/Data/lrs2_new/pkl_data'
sum_dir = f'./summary_sys20_dervb_add_visual-{add_visual}'


if task == 'DNN-WPE-Online':
    model_save_dir = f"{sum_dir}/{model_online}"
else:
    model_save_dir = f"{sum_dir}/{model_offline}"

log_dir = model_save_dir + '/log'
model_name = 'b{}'.format(batch_size)

############## print info ###############
print(f"model_save_dir: {model_save_dir}")
if new_wav_scp:
    print(f"training_wavscp: {training_wavscp}")
    print(f"validation_wavscp: {validation_wavscp}")
    print(f"test_wavscp: {test_wavscp}")

# import os
if write_wav:
    write_name = f'{model_save_dir}/{wav_dir_name}'
    # if not os.path.exists(write_name):
    #     os.makedirs(write_name)
