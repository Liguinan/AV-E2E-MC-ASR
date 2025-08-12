gpu_id = '0'
task = 'DNN-WPE-Offline'  # DNN-WPE-Offline,  DNN-WPE-Online for parameters setting
# =========================================
est_power = False
scratch = True
scale_by_mixture = False
sampling=False
sampling_frame_ratio=1
TCN_for_lip = False


## mask model for intialization 
if not scratch:
    model_path = '/users/bdda/gnli/projects/share_projects/DNN-WPE_for_yj/power_estimation/summary_dervb_add_visual-False_LRS2/b64-best-19.pt.tar'

visual_fusion_type = 'concat'  # or attention
add_visual = False 
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
model_offline = f'model_dnn-wpe_offline'

if task == 'DNN-WPE-Online':
    lr = 5e-4
    HOP_SIZE = 128
    normalization = False
    model_online = f'model_dnn-wpe_online'

# ===============  data path ==============
# original mixture pickle path
training_path = "/project_bdda7/bdda/gnli/Data/lrs2_new/pkl_data/train_pretrain_32_rev1_le6_ark.pkl"
validation_path = "/project_bdda7/bdda/gnli/Data/lrs2_new/pkl_data/val_rev4_le6_ark.pkl"
test_path = "/project_bdda7/bdda/gnli/Data/lrs2_new/pkl_data/test_rev4.pkl"
replay_path = "/project_bdda7/bdda/gnli/Data/lrs2_new/pkl_data/test_replay.pkl"
lrs3_test_path = "/users/bdda/gnli/data/data_bdda7/MC_MS_Simu/egs/lrs3/pkl_data/test_rev4.pkl"

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
seed = 0
log_display_step = 100
cuda = True  # Trues
# ================= Evaluating ===================== #
write_wav = True
all_metrics = False
inference_dataset_name = 'test' # val train_pretrain test
ckpt_epoch = 19
wav_dir_name = f"{inference_dataset_name}_taps={taps}_delay={delay}"
ckpt = f'b{batch_size}-best-{ckpt_epoch}.pt.tar'
# ================= Directory ===================== #
data_dir = '/project_bdda7/bdda/gnli/Data/lrs2_new/pkl_data'
sum_dir = f'./summary_dervb_add_visual-{add_visual}'


if task == 'DNN-WPE-Online':
    model_save_dir = f"{sum_dir}/{model_online}"
else:
    model_save_dir = f"{sum_dir}/{model_offline}"

log_dir = model_save_dir + '/log'
model_name = 'b{}'.format(batch_size)

############## print info ###############
print(f"model_save_dir: {model_save_dir}")

# import os
if write_wav:
    write_name = f'{model_save_dir}/{wav_dir_name}'
    # if not os.path.exists(write_name):
    #     os.makedirs(write_name)
