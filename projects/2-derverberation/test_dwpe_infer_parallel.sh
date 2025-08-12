###
# check test.sh
# (1) inference_dataset_name
# (2) ckpt_epoch
# (3) dir 

# check params.py
# (1) gpu_id
# (2) diag_loading_ratio=1e-5
# (3) add_visual=False # or False
# (4) write_wav = True
# (5) all_metrics = True
# (6) inference_dataset_name = 'test' # val  train_pretrain
# (7) ckpt_epoch = 63
# (8) ckpt = "uTGT-LPS_IPD_AF-b12-BN-fd1-fixTrue-f0-cosTrue-sinFalse-best-63.pt.tar"
###

# task-dependent setting
# below is the same with params.py
inference_dataset_name='test' # val  train_pretrain test replay
ckpt_epoch=19 # 27
diag_loading_ratio="1e-6" # 
taps=2
delay=2
write_wav=1
# normalization=True
# hop_size=128  # 128 fixed
# power_flooring="1e-10"   # a little improve when pf=1e-5,can try
# scale_by_mixture=False  # False
# sampling_frame_ratio=1
task_name=${inference_dataset_name}-${ckpt_epoch}

# ao
base_1='summary_dervb_add_visual-False'
base_2='model_dnn-wpe_offline'

dir="$base_1/$base_2"

wav_dir_name="${inference_dataset_name}_taps=${taps}_delay=${delay}"
if [ $write_wav == 1 ] && [ ! -d ${dir}/${wav_dir_name} ]; then
  mkdir ${dir}/${wav_dir_name}
fi


currentTime=$(date "+%Y-%m-%d_%H:%M:%S")
log_dir=${dir}/log_${inference_dataset_name}_delay=${delay}_taps=${taps}_${currentTime}


mkdir "$log_dir"

# log file 
log_file_name=${log_dir}/${task_name}

# save params.py to log file
{
echo "#######################params.py##############"
cat params.py
echo "#######################params.py##############"
} >> "${log_file_name}.log"

# inference
inference_job_num=8
for job in $(seq $inference_job_num) 
  do 
    echo "inference_job_num: ${inference_job_num}, job:${job}" >> "${log_file_name}_job_${job}.log"
    nohup python pt_inference_jobs.py $inference_job_num $job >> "${log_file_name}_job_${job}.log" & 
  done
echo "generating..."
