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
inference_dataset_name='lrs3_test' # val  train_pretrain test replay
ckpt_epoch=19
write_wav=1
hop_size=256

task_name=${inference_dataset_name}-${ckpt_epoch}
# copy the dir of trained model log 
# ao
base_1='summary_sys26_dervb_add_visual-False'
# base_1='summary_sys26_dervb_add_visual-False-multi-channel'
# base_1='summary_sys27_dervb_add_visual-True'
# base_1='summary_sys27_dervb_add_visual-True-multi-channel'
base_2='model_b=64_lr=0.001_vblk=5-vfusion=concat-sampling=1-factor=10'
# base_2='model_b=8_lr=0.001_vblk=5-vfusion=concat-sampling=1-factor=10-accum_grad=8'
# base_2='model_b=64_lr=0.001_vblk=5-vfusion=concat-sampling=1-factor=10-layer-norm1-False-layer-norm2-True'
# base_2='model_b=8_lr=0.001_vblk=5-vfusion=concat-sampling=1-factor=10-accum_grad=8'


dir="$base_1/$base_2"

wav_dir_name="${inference_dataset_name}_eps=${ckpt_epoch}"
if [ $write_wav == 1 ] && [ ! -d ${dir}/${wav_dir_name} ]; then
  mkdir ${dir}/${wav_dir_name}
fi


currentTime=$(date "+%Y-%m-%d_%H:%M:%S")
log_dir=${dir}/log_${inference_dataset_name}_epoch-${ckpt_epoch}_${currentTime}
mkdir "$log_dir"

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
    nohup python pt_inference_jobs.py $inference_job_num $job >> "${log_file_name}_job_${job}.log" & 
  done
# echo "generating..."
