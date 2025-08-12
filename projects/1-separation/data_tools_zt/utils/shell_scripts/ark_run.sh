#!/bin/bash


export PATH=$PATH:$PWD

nj=32
cmd="run.pl"
fs=16000
WAV_PATH=/project_bdda7/bdda/tzhong/projects/Speech-Enhancement-Compression/data/for_tianziwang/wav
# WAV_PATH=/project_bdda6/bdda/jwyu/simulation/lrs3/pretrain/wav
featdir=feat
datadir=data
mkdir -p "${featdir}"
mkdir -p "${datadir}"
# use integer for simplex uttid, pls use gn's wav.scp file here
find $WAV_PATH -type f -name '*.wav' | awk '{print NR " " $0}' > "${datadir}"/wav.scp

split_scps=""
for n in $(seq ${nj}); do
    split_scps="${split_scps} ${featdir}/wav.${n}.scp"
done
split_scp.pl "${datadir}/wav.scp" ${split_scps}

# if wav.scp is in pipe format,
# e.g. sw02054-A sph2pipe -f wav -p -c 1 /project_bdda6/bdda/jjdeng/espnet/egs/swbd/dataset/LDC/LDC97S62/swb1_d1/data/sw02054.sph \|
# sox -R -t wav - -t wav - rate 16000 dither |
# then pls uncomment following lines
#format_wav_scp.sh --nj "${nj}" --cmd "${cmd}" \
#                                --audio-format "wav" --fs "${fs}" \
#                                "${datadir}/wav.scp" "${featdir}"

echo "[INFO] wav.*.scp --> feats.scp, feats.*.ark ..."
echo "[INFO] log in ${featdir}/logs/wavcopy.*.log"

"${cmd}" JOB=1:$nj feat/logs/wavcopy.JOB.log \
         wav-copy scp:"${featdir}/wav.JOB.scp" \
         ark,scp:"${featdir}"/feats.JOB.ark,"${featdir}"/feats.JOB.scp
for n in $(seq $nj); do
  cat "${featdir}"/feats.$n.scp || exit 1
done > "${featdir}"/feats.scp || exit 1

echo "[INFO] loading .ark data ... "
python readark.py ${datadir}/wav.scp ${featdir}/feats.scp
