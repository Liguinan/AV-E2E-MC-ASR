#!/bin/bash

run_path=/project_bdda3/bdda/jwyu/JianweiYuFrameWork/lrs3/lrs3_pychain/LRS3-trptr-joint-jwyu
cd $run_path

. ./path.sh

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/share/src/gcc-5.4.0/stage1-x86_64-unknown-linux-gnu/libstdc++-v3/src/.libs

run_path=/project_bdda7/bdda/tzhong/projects/Speech-Enhancement-Compression/data_tools/utils/shell_scripts

cd $run_path

export PATH=$PATH:$run_path

. ./ark_run_for_python_wrapper_args.sh

nj=16
cmd="run.pl"
fs=16000

mkdir -p "${featdir}"


split_scps=""
for n in $(seq ${nj}); do
    split_scps="${split_scps} ${featdir}/wav.${n}.scp"
done
split_scp.pl "${wav_scp_path}" ${split_scps}

echo "[INFO] wav.*.scp --> feats.scp, feats.*.ark ..."
echo "[INFO] log in ${featdir}/logs/wavcopy.*.log"

"${cmd}" JOB=1:$nj ${featdir}/logs/wavcopy.JOB.log \
         wav-copy scp:"${featdir}/wav.JOB.scp" \
         ark,scp:"${featdir}"/feats.JOB.ark,"${featdir}"/feats.JOB.scp
for n in $(seq $nj); do
  cat "${featdir}"/feats.$n.scp || exit 1
done > "${featdir}"/feats.scp || exit 1

