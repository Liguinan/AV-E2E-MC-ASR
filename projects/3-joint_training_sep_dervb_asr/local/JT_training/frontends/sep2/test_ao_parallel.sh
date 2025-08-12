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
inference_dataset_name='replay' # val  train_pretrain test replay
ckpt_epoch=46   #31 45 32 48 37 54 34
diag_loading_ratio="1e-5"
taps=
taps_wpe=2  #1 2 3 4 5 7 9 12 15 18 21 27
delay_wpe="1e-6"
delay=
write_wav=1
hop_size=256
task_name=${inference_dataset_name}-${ckpt_epoch}
# copy the dir of trained model log 

# ao
# base_1='summary_sys26_ao_SpecM_ao_lipemb-LPS+IPD+AF-SISNR+SISNR_NOISE'
# base_1='summary_sys27_av_SpecM_ao_lipemb-LPS+IPD+AF-SISNR+SISNR_NOISE'
# base_1='summary_sys19_ao_DWPE_ao_lipemb-LPS+IPD+AF-SISNR+SISNR_NOISE'
# base_1='summary_sys20_av_DWPE_ao_lipemb-LPS+IPD+AF-SISNR+SISNR_NOISE'

# vo
# base_1='summary_sys28_ao_SpecM_vo_lipemb-LPS+IPD-SISNR+SISNR_NOISE'
# base_1='summary_sys29_av_SpecM_vo_lipemb-LPS+IPD-SISNR+SISNR_NOISE'
# base_1='summary_sys21_ao_DWPE_vo_lipemb-LPS+IPD-SISNR+SISNR_NOISE'
# base_1='summary_sys22_av_DWPE_vo_lipemb-LPS+IPD-SISNR+SISNR_NOISE'

# av
# base_1='summary_sys30_ao_SpecM_av_lipemb-LPS+IPD+AF-SISNR+SISNR_NOISE'
base_1='summary_sys31_av_SpecM_av_lipemb-LPS+IPD+AF-SISNR+SISNR_NOISE'
# base_1='summary_sys23_ao_DWPE_av_lipemb-LPS+IPD+AF-SISNR+SISNR_NOISE'
# base_1='summary_sys24_av_DWPE_av_lipemb-LPS+IPD+AF-SISNR+SISNR_NOISE'

base_2='model_mvdr-AF_premasking=False-BS=32-DL=1e-5_new'
# base_2="model_mvdr-AF_premasking=False-BS=32-DL=1e-5_dl=${delay_wpe}_taps=${taps_wpe}"




dir="$base_1/$base_2"

# av
# dir='summary_lipemb-LPS+IPD+AF-SISNR+SISNR_NOISE/model_mvdr-AF_premasking=False-Batch_Size=32-Dloading=1e-5-add_visual=True'

# vo
# dir='summary_lipemb-LPS+IPD-SISNR+SISNR_NOISE/model_mvdr-AF_premasking=False-Batch_Size=32-Dloading=1e-5-add_visual=True'


wav_dir_name="${inference_dataset_name}_eps=${ckpt_epoch}_dl=${diag_loading_ratio}"
if [ $write_wav == 1 ] && [ ! -d ${dir}/${wav_dir_name} ]; then
  mkdir ${dir}/${wav_dir_name}
fi




currentTime=$(date "+%Y-%m-%d_%H:%M:%S")
log_dir=${dir}/${inference_dataset_name}-log_dl-${diag_loading_ratio}_taps-${taps}_delay-${delay}_epoch-${ckpt_epoch}_${currentTime}
mkdir "$log_dir"

# log file 
# log_file_name=${log_dir}/${task_name}_${currentTime}
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
    nohup python pt_mvdr_inference_jobs.py $inference_job_num $job >> "${log_file_name}_job_${job}.log" & 
    # nohup python pt_tfmasking_inference_jobs.py $inference_job_num $job >> "${log_file_name}_job_${job}.log" & 

  done
echo "generating..."
