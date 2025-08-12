gpu_id = '2'
# =========================================
add_visual = False 
visual_fusion_type = 'concat'  # or attention
lr = 1e-3 
batch_size = 32
accum_grad = 1
loss_type = ['stft']

sampling=False
sampling_frame_ratio=1
TCN_for_lip = False



# ===============  data path ==============
training_path = "/project_bdda7/bdda/yjchen/aftcut_finetune_iemocap_DI/multi_channel_mixture_speech_simulation_4iemocap_cut7sec/pkl_ark_data/train/ark_pickle/train_ark.pkl"
validation_path = "/project_bdda7/bdda/yjchen/aftcut_finetune_iemocap_DI/multi_channel_mixture_speech_simulation_4iemocap_cut7sec/pkl_ark_data/dev/ark_pickle/dev_ark.pkl"
test_path = "/project_bdda7/bdda/yjchen/aftcut_finetune_iemocap_DI/multi_channel_mixture_speech_simulation_4iemocap_cut7sec/pkl_ark_data/test/ark_pickle/test_ark.pkl"

# ================ training target: multichannel dervb ref wav scp (noise + overlap) ====================
mc_ref_wav_scp=True
training_ref_wavscp = '/project_bdda8/bdda/yjchen/wav_data/15C/train/mixture_direct_path/wav.scp'
validation_ref_wavscp = '/project_bdda8/bdda/yjchen/wav_data/15C/dev/mixture_direct_path/wav.scp'
test_ref_wavscp = '/project_bdda8/bdda/yjchen/wav_data/15C/test/mixture_direct_path/wav.scp'

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
FFT_SIZE = 512
HOP_SIZE = 256
NEFF = FFT_SIZE // 2 + 1
# ================= Training ===================== #
seed = 0
log_display_step = 100
cuda = True  # Trues
# ================= Evaluating ===================== #
write_wav = True
all_metrics = False
inference_dataset_name = "test"
ckpt_epoch = 19
wav_dir_name = f"{inference_dataset_name}_eps={ckpt_epoch}"
ckpt = f'b{batch_size}-best-{ckpt_epoch}.pt.tar'
# ================= save Directory ===================== #
sum_dir = f'./summary_dervb_add_visual-{add_visual}'

model_save_dir = sum_dir + f"/model"

log_dir = model_save_dir + '/log'
model_name = 'b{}'.format(batch_size)

# import os
if write_wav:
    write_name = f'{model_save_dir}/{wav_dir_name}'
#     if not os.path.exists(write_name):
#         os.makedirs(write_name)
