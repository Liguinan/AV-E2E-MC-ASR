# This is file is used to generate the pickle file for front-end training. 
# Please run the following command stage by stage since there is an asynchronous execution when generate the overlapped speech by kaldi.
source /opt/share/etc/gcc-5.4.0.sh

set -e -o pipefail

stage=0
stop_stage=0
cmd="run.pl"
. utils/parse_options.sh

work_dir=`pwd`
dataset=test
rvb_affix=_mc_
num_data_reps=4
wav_data=wav_data
pkl_data=pkl_data

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
echo "this is stage 0, generate strategy of adding reverberation, noise and interference" 
  norvb_datadir=ori_clean_data/${dataset}
  output_datadir=data/${dataset}
  sample_rate=16000

  if [ ! -f ${norvb_datadir}${rvb_affix}${num_data_reps}_hires/wav.scp ]; then
    if [ ! -d "RIRS_NOISES/" ]; then
      ln -s /project_bdda5/bdda/gnli/data/Kaldi-Script-RIRs/rir/RIRS_NOISES .
    fi
    if [ ! -d "sim_15c_RIRs_Linear_Array/" ]; then
      ln -s /project_bdda7/bdda/gnli/Data/lrs2/sim_rir/sim_15c_RIRs_Linear_Array .
    fi
    
    rvb_opts=()
    rvb_opts+=(--rir-set-parameters "1.0, sim_15c_RIRs_Linear_Array/normal/rir_list")
    rvb_opts+=(--noise-set-parameters RIRS_NOISES/pointsource_noises/noise_list)

    if [ ${dataset} == "val" ] || [ ${dataset} == "test" ]; then
      python steps/data/rvb_noise_overlap_4val_4test.py \
      "${rvb_opts[@]}" \
      --prefix "rev" \
      --foreground-snrs "20:10:15:5:0" \
      --background-snrs "20:10:15:5:0" \
      --sirs "6:0:-6" \
      --speech-rvb-probability 1 \
      --pointsource-noise-addition-probability 1 \
      --isotropic-noise-addition-probability 1 \
      --num-replications ${num_data_reps} \
      --max-noises-per-minute 120 \
      --interference-ratio "0.6:1.0" \
      --source-sampling-rate $sample_rate \
      ${norvb_datadir} ${output_datadir}${rvb_affix}${num_data_reps}
    else
      python steps/data/rvb_noise_overlap.py \
      "${rvb_opts[@]}" \
      --prefix "rev" \
      --foreground-snrs "20:10:15:5:0" \
      --background-snrs "20:10:15:5:0" \
      --sirs "6:0:-6" \
      --speech-rvb-probability 1 \
      --pointsource-noise-addition-probability 1 \
      --isotropic-noise-addition-probability 1 \
      --num-replications ${num_data_reps} \
      --max-noises-per-minute 120 \
      --interference-ratio "0.6:1.0" \
      --source-sampling-rate $sample_rate \
      ${norvb_datadir} ${output_datadir}${rvb_affix}${num_data_reps}
    fi 
  fi
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "this is stage 1, gen mixture"
  datatype=rvb_noisy_overlap
  wav_scp=wav_rvb_noisy_overlapped.scp
  output_dir=${work_dir}/$wav_data/15C/$dataset/$datatype
  rm -rf sim_15c_RIRs_Linear_Array
  ln -s /project_bdda7/bdda/gnli/Data/lrs2/sim_rir/sim_15c_RIRs_Linear_Array .
  rm -rf $output_dir
  mkdir -p $output_dir
  python local/gen_pipe_2_wav_command.py ${work_dir}/data/${dataset}${rvb_affix}${num_data_reps}/$wav_scp $output_dir
  # run the command
  utter_num=`cat ${work_dir}/data/${dataset}${rvb_affix}${num_data_reps}/$wav_scp | wc -l`
  split_num=`cat $output_dir/split_num`
  for i in $(seq 0 $split_num $utter_num) 
  do 
    chmod 744 $output_dir/splitdata/${i}-command.sh
    nohup $output_dir/splitdata/${i}-command.sh >> $output_dir/splitdata/gen_${i}.log &
  done
  # check all wav is generated!
  # ll $output_dir/*main/* | grep "\.wav" | wc -l
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "this is stage 2, gen rvb-only"
  datatype=rvb_only
  wav_scp=wav_rvb_only.scp
  output_dir=${work_dir}/$wav_data/15C/$dataset/$datatype
  rm -rf sim_15c_RIRs_Linear_Array
  ln -s /project_bdda7/bdda/gnli/Data/lrs2/sim_rir/sim_15c_RIRs_Linear_Array .
  rm -rf $output_dir
  mkdir -p $output_dir
  python local/gen_pipe_2_wav_command.py ${work_dir}/data/${dataset}${rvb_affix}${num_data_reps}/$wav_scp $output_dir
  # run the command
  utter_num=`cat ${work_dir}/data/${dataset}${rvb_affix}${num_data_reps}/$wav_scp | wc -l`
  split_num=`cat $output_dir/split_num`
  for i in $(seq 0 $split_num $utter_num) 
  do 
    chmod 744 $output_dir/splitdata/${i}-command.sh
    nohup $output_dir/splitdata/${i}-command.sh >> $output_dir/splitdata/gen_${i}.log &
  done
  # check all wav is generated!
  # ll $output_dir/*main/* | grep "\.wav" | wc -l
fi

early_rvb_time=50ms
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "this is stage 3, generate 17C wav data for pickle file  1-15 input wav, 16 clean wav, 17 rvb-only wav"
  suffix=
  datatype=rvb_noisy_overlap
  clean_wav_scp_path=${work_dir}/ori_clean_data/${dataset}/wav.scp 
  rvb_only_tgt_wav_scp_path=${work_dir}/$wav_data/15C/${dataset}/rvb_only/wav.scp 
  input_wav_scp_path=${work_dir}/$wav_data/15C/${dataset}/${datatype}/wav.scp
  output_wav_path=${work_dir}/$wav_data/17C/$dataset/${datatype}${suffix}
  rm -rf $output_wav_path
  mkdir -p $output_wav_path
  python local/gen_mchn_wav_for_pickle.py $clean_wav_scp_path $rvb_only_tgt_wav_scp_path $input_wav_scp_path $output_wav_path
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  echo "this is stage 4, generate pickle file for front-end training."
  suffix=
  datatype=rvb_noisy_overlap
  input_dir=$work_dir/$wav_data/17C/$dataset/${datatype}${suffix}
  input_azimuth_dir=$work_dir/data/${dataset}${rvb_affix}${num_data_reps}
  input_utt_dur_dir=$work_dir/ori_clean_data/${dataset}
  # lrs2_lip_emb_dir=$work_dir/lip_emb_data/$dataset/emb
  lrs2_lip_emb_dir=$work_dir/lip_emb_data
  output_dir=$work_dir/${pkl_data}${suffix}
  python local/gen_overlap_2rev_pkl_new.py $input_dir $input_azimuth_dir $input_utt_dur_dir $lrs2_lip_emb_dir $output_dir $dataset
  # python local/gen_trprt_pkl.py $input_dir $input_azimuth_dir $input_utt_dur_dir $lrs2_lip_emb_dir $output_dir $dataset
fi
















