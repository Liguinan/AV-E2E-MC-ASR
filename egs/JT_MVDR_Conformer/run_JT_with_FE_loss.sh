#!/usr/bin/env bash

# Copyright 2021 Ruhr-University Bochum (Wentao Yu)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

source /opt/share/etc/gcc-5.4.0.sh
export CUDA_VISIBLE_DEVICES=0


# general configuration
backend=pytorch
### here, set stage=4 since each system shares the same dump data
stage=0    		# start from stage 0, stage -1 (Data Download has to be done by the user) 
stop_stage=100		# stage at which to stop
redumpprocessingstage=1
stop_redumpprocessingstage=1
ngpu=1         		# number of gpus ("0" uses cpu, otherwise use gpu)
nj=30
debugmode=1
dumpdir=dump   		# directory to dump full features
N=0            		# number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0      		# verbose option
resume=        		# Resume the training from snapshot
train_lm=false
ifmulticore=true

# feature configuration
do_delta=false

preprocess_config=
train_config=conf/train_conformer_lr5_patience5_with_FE_loss.yaml 
lm_config=conf/lm.yaml
decode_config=conf/decode.yaml

# rnnlm related
lm_resume= # specify a snapshot file to resume LM training
lmtag=     # tag for managing LMs

# decoding parameter
recog_model=model.acc.best  # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'
lang_model=rnnlm.model.best # set a language model to be used for decoding

# model average realted (only for transformer)
n_average=5                  # the number of ASR models to be averaged
use_valbest_average=true     # if true, the validation `n_average`-best ASR models will be averaged.
                             # if false, the last `n_average` ASR models will be averaged.
lm_n_average=0               # the number of languge models to be averaged
use_lm_valbest_average=false # if true, the validation `lm_n_average`-best language models will be averaged.
                             # if false, the last `lm_n_average` language models will be averaged.

# The LRS2 Corpus requires vertification. You have to download the 
# dataset and set your dataset dir here
datadir=data		     # The LRS2 dataset directory e.g. /home/foo/LRS2

pretrain=true		     # if use LRS2 pretrain set 
segment=true  		     # if do segmentation for pretrain set

# bpemode (unigram or bpe)
nbpe=500
bpemode=unigram

## train_lm=false, we have to download pretrained language model
function gdrive_download () {
  CONFIRM=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=$1" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')
  wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$CONFIRM&id=$1" -O $2
  rm -rf /tmp/cookies.txt
}

# exp tag
# tag for managing experiments.
tag="" 


. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

# define sets
if [ "$pretrain" = true ] ; then
	train_set="pretrain_Train"
else
	train_set="Train"
fi
train_dev="val"
other_recog_set="Replay"
recog_set="Val Test ${other_recog_set}"

# Stage -1: Data Download
if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    if [ -d "$datadir" ]; then
    	echo "Dataset already exists."
    else
    	echo "For downloading the data, please visit 'https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs2.html'."
    	echo "You will need to sign a Data Sharing agreement with BBC Research & Development before getting access."
    	echo "Please download the dataset by yourself and save the dataset directory in path.sh file"
    	echo "Thanks!"
    fi
fi

# Stage 0: Data preparation
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "stage 0: Data preparation"
    for part in Test Val Train; do
        local/data_preparation.sh $datadir $part $segment $nj || exit 1;
    done
    if [ "$pretrain" = true ] ; then
    	part=pretrain
    	local/data_preparation.sh $datadir $part $segment $nj || exit 1;
    fi
    for part in pretrain Test Val Train; do 
    	mv data/${part} data/${part}_org || exit 1;
    done
    echo "stage 0: Data preparation finished"

fi

feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
feat_test_dir=${dumpdir}/Test/delta${do_delta}; mkdir -p ${feat_test_dir}
feat_val_dir=${dumpdir}/Val/delta${do_delta}; mkdir -p ${feat_val_dir}
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    ### Task dependent. You have to design training and dev sets by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 1: Feature Generation"

    fbankdir=fbank
    # Generate the fbank features; by default 80-dimensional fbanks with pitch on each frame
    for x in ${train_set} Val Test; do
        utils/fix_data_dir.sh data/${x}
        steps/make_fbank.sh --cmd "$train_cmd" --nj ${nj} --write_utt2num_frames true \
            data/${x} exp/make_fbank/${x} ${fbankdir}
        utils/fix_data_dir.sh data/${x}
    done

    # if [ "$pretrain" = true ] ; then
	# remove_longshortdata.sh --maxframes 3000 --maxchars 400 data/pretrain_org data/pretrain
    #     steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj ${nj} --write_utt2num_frames true \
    #         data/pretrain exp/make_fbank/pretrain ${fbankdir}
    #     utils/fix_data_dir.sh data/pretrain
    # 	utils/combine_data.sh data/${train_set} \
	# 		      data/pretrain \
	# 		      data/Train
    # fi

    # compute global CMVN
    compute-cmvn-stats scp:data/${train_set}/feats.scp data/${train_set}/cmvn.ark

    # dump features for training
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta ${do_delta} \
        data/${train_set}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/train_set ${feat_tr_dir=}
    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}; mkdir -p ${feat_recog_dir}
        dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta ${do_delta} \
            data/${rtask}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/recog/${rtask} \
            ${feat_recog_dir}
    done
fi


dict=data/lang_char/${train_set}_${bpemode}${nbpe}_units.txt
bpemodel=data/lang_char/${train_set}_${bpemode}${nbpe}
echo "dictionary: ${dict}"
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Dictionary and Json Data Preparation"

    if [ "$train_lm" = true ] ; then
        mkdir -p data/lang_char/
        echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
        cut -f 2- -d" " data/${train_set}/text > data/lang_char/input.txt
        spm_train --input=data/lang_char/input.txt --vocab_size=${nbpe} --model_type=${bpemode} --model_prefix=${bpemodel} --input_sentence_size=100000000
        spm_encode --model=${bpemodel}.model --output_format=piece < data/lang_char/input.txt | tr ' ' '\n' | sort | uniq | awk '{print $0 " " NR+1}' >> ${dict}
        wc -l ${dict}

    else
	# gdrive_download '1ZXXCXSbbFS2PDlrs9kbJL9pE6-5nPPxi' 'model.v1.tar.gz'
	tar -xf model.v1.tar.gz
	# mv avsrlrs2_3/exp/train_rnnlm_pytorch_lm_unigram500 exp/pretrainedlm
	mv avsrlrs2_3/data/lang_char data/
    	mv data/lang_char/train_unigram500.model data/lang_char/${train_set}_unigram500.model
    	mv data/lang_char/train_unigram500.vocab data/lang_char/${train_set}_unigram500.vocab
    	mv data/lang_char/train_unigram500_units.txt data/lang_char/${train_set}_unigram500_units.txt
  	rm -rf avsrlrs2_3
	# rm -rf model.v1.tar.gz
	
	##### it is depands on your corpus, if the corpus text transcription is uppercase, use this to convert to lowercase
    	textfilenames1=data/${train_set}/text
   	    textfilenames2=data/Test/text	
    	textfilenames3=data/Val/text	
    	for textfilename in $textfilenames1 $textfilenames2 $textfilenames3
    	do
	    sed -r 's/([^ \t]+\s)(.*)/\1\L\2/' $textfilename > ${textfilename}1  || exit 1;
	    rm -rf $textfilename  || exit 1;
	    mv ${textfilename}1 $textfilename  || exit 1;
    	done
    fi

    # make json labels
    data2json.sh --nj ${nj} --feat ${feat_tr_dir}/feats.scp --bpecode ${bpemodel}.model \
        data/${train_set} ${dict} > ${feat_tr_dir}/data_${bpemode}${nbpe}.json

    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
        data2json.sh --nj ${nj} --feat ${feat_recog_dir}/feats.scp --bpecode ${bpemodel}.model \
            data/${rtask} ${dict} > ${feat_recog_dir}/data_${bpemode}${nbpe}.json
    done
fi


### Above audio data2json just a template for next redump data2json
############## visual embedding, audio in front-end pkl file ###############
declare -A path_dic
path_dic=()
path_dic[pretrain_Train_path]="/project_bdda7/bdda/gnli/Data/lrs2_new/pkl_data/train_pretrain_32_rev1_le6_ark.pkl"
path_dic[Val_path]="/project_bdda7/bdda/gnli/Data/lrs2_new/pkl_data/val_rev4_le6_ark.pkl"
# path_dic[Val_1082_path]="/project_bdda7/bdda/gnli/Data/lrs2_new/pkl_data/val_rev4.pkl"
path_dic[Test_path]="/project_bdda7/bdda/gnli/Data/lrs2_new/pkl_data/test_rev4.pkl"
path_dic[Replay_path]="/project_bdda7/bdda/gnli/Data/lrs2_new/pkl_data/test_replay.pkl"
##############
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 3: re dump to generate data2json file"
    if [ ${redumpprocessingstage} -le 1 ] && [ ${stop_redumpprocessingstage} -ge 1 ]; then
        echo "stage 3.1: 
        [1]. The key of 'feat' will be replaced by a real 'afeat' and shape by ashape
        [2]. Add 'afeat_wav'
        [3]. Add 'vfeat' key which a numpy format file.
        [4]. Add 'vshape'
        [5]. Add other front-end needed training info to new data2json file"

        for dataset in $recog_set $train_set; do
            echo "dumping data2json for ${dataset} set, pickle path is: ${path_dic[${dataset}_path]}"
            python3 local/dump/redump.py dump/av_JT_feat/${dataset} dump/${dataset} ${path_dic[${dataset}_path]} dump/video/${dataset} ${dataset} $ifmulticore || exit 1;
        done
    fi
fi

# You can skip this and remove --rnnlm option in the recognition (stage 5)
# if [ "$train_lm" = false ] ; then
#     lmexpname=pretrainedlm
#     lmexpdir=exp/${lmexpname}
# else
#     if [ -z ${lmtag} ]; then
#         lmtag=$(basename ${lm_config%.*})
#     fi
#     lmexpname=train_rnnlm_${backend}_${lmtag}_${bpemode}${nbpe}_ngpu${ngpu}
#     lmexpdir=exp/${lmexpname}
#     mkdir -p ${lmexpdir}
# fi
# if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
#     if [ "$train_lm" = false ] ; then
#         echo "stage 3: Use pretrained LM"
#     else
#         echo "stage 3: LM Preparation"
#         lmdatadir=data/local/lm_train_${bpemode}${nbpe}
#         # use external data
#         if [ ! -e data/local/lm_train/librispeech-lm-norm.txt.gz ]; then
#             wget http://www.openslr.org/resources/11/librispeech-lm-norm.txt.gz -P data/local/lm_train/
#         fi
#         if [ ! -e ${lmdatadir} ]; then
#             mkdir -p ${lmdatadir}
#             cut -f 2- -d" " data/${train_set}/text | gzip -c > data/local/lm_train/${train_set}_text.gz
#             # combine external text and transcriptions and shuffle them with seed 777
#             zcat data/local/lm_train/librispeech-lm-norm.txt.gz data/local/lm_train/${train_set}_text.gz |\
#                 spm_encode --model=${bpemodel}.model --output_format=piece > ${lmdatadir}/train.txt
#             cut -f 2- -d" " data/${train_dev}/text | spm_encode --model=${bpemodel}.model --output_format=piece \
#                                                             > ${lmdatadir}/valid.txt
#         fi
#         ${cuda_cmd} --gpu ${ngpu} ${lmexpdir}/train.log \
#             lm_train.py \
#             --config ${lm_config} \
#             --ngpu ${ngpu} \
#             --backend ${backend} \
#             --verbose 1 \
#             --outdir ${lmexpdir} \
#             --tensorboard-dir tensorboard/${lmexpname} \
#             --train-label ${lmdatadir}/train.txt \
#             --valid-label ${lmdatadir}/valid.txt \
#             --resume ${lm_resume} \
#             --dict ${dict} \
#             --dump-hdf5-path ${lmdatadir}
#     fi
# fi


if [ -z ${tag} ]; then
    expname=${train_set}_${backend}_$(basename ${train_config%.*})
    if ${do_delta}; then
        expname=${expname}_delta
    fi
    if [ -n "${preprocess_config}" ]; then
        expname=${expname}_$(basename ${preprocess_config%.*})
    fi
else
    expname=${train_set}_${backend}_${tag}
fi
expdir=exp/${expname}
mkdir -p ${expdir}

# ln customized espnet scripts
rm -rf /users/bdda/gnli/projects/bdda7_projects/espnet/espnet/JT_training
ln -s `pwd`/local/JT_training  /users/bdda/gnli/projects/bdda7_projects/espnet/espnet/JT_training
# ln and fixed the entry of our customized scripts
rm -rf /users/bdda/gnli/projects/bdda7_projects/espnet/espnet/bin/avsr_train_JT.py
rm -rf /users/bdda/gnli/projects/bdda7_projects/espnet/espnet/bin/avsr_recog_JT.py
ln -s `pwd`/local/JT_training/avsr_train_JT.py /users/bdda/gnli/projects/bdda7_projects/espnet/espnet/bin/avsr_train_JT.py
ln -s `pwd`/local/JT_training/avsr_recog_JT.py /users/bdda/gnli/projects/bdda7_projects/espnet/espnet/bin/avsr_recog_JT.py



if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: Network Training"
    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
        avsr_train_JT.py \
        --config ${train_config} \
        --preprocess-conf ${preprocess_config} \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --outdir ${expdir}/results \
        --tensorboard-dir tensorboard/${expname} \
        --debugmode ${debugmode} \
        --dict ${dict} \
        --debugdir ${expdir} \
        --minibatches ${N} \
        --verbose ${verbose} \
        --resume ${resume} \
        --train-json dump/av_JT_feat/pretrain_Train/deltafalse/data_${bpemode}${nbpe}.json \
        --valid-json dump/av_JT_feat/Val/deltafalse/data_${bpemode}${nbpe}.json
fi


# ln customized espnet scripts
rm -rf /users/bdda/gnli/projects/bdda7_projects/espnet/espnet/JT_training
ln -s `pwd`/local/JT_training  /users/bdda/gnli/projects/bdda7_projects/espnet/espnet/JT_training
# ln and fixed the entry of our customized scripts
rm -rf /users/bdda/gnli/projects/bdda7_projects/espnet/espnet/bin/avsr_train_JT.py
rm -rf /users/bdda/gnli/projects/bdda7_projects/espnet/espnet/bin/avsr_recog_JT.py
ln -s `pwd`/local/JT_training/avsr_train_JT.py /users/bdda/gnli/projects/bdda7_projects/espnet/espnet/bin/avsr_train_JT.py
ln -s `pwd`/local/JT_training/avsr_recog_JT.py /users/bdda/gnli/projects/bdda7_projects/espnet/espnet/bin/avsr_recog_JT.py



if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 5: Decoding"
    if [[ $(get_yaml.py ${train_config} model-module) = *transformer* ]] || \
           [[ $(get_yaml.py ${train_config} model-module) = *conformer* ]] || \
           [[ $(get_yaml.py ${train_config} etype) = custom ]] || \
           [[ $(get_yaml.py ${train_config} dtype) = custom ]]; then
        # Average ASR models
        if ${use_valbest_average}; then
            recog_model=model.val${n_average}.avg.best
            opt="--log ${expdir}/results/log"
        else
            recog_model=model.last${n_average}.avg.best
            opt="--log"
        fi
        average_checkpoints.py \
            ${opt} \
            --backend ${backend} \
            --snapshots ${expdir}/results/snapshot.ep.* \
            --out ${expdir}/results/${recog_model} \
            --num ${n_average}

        # # Average LM models
        # if [ ${lm_n_average} -eq 0 ]; then
        #     lang_model=rnnlm.model.best
        # else
        #     if ${use_lm_valbest_average}; then
        #         lang_model=rnnlm.val${lm_n_average}.avg.best
        #         opt="--log ${lmexpdir}/log"
        #     else
        #         lang_model=rnnlm.last${lm_n_average}.avg.best
        #         opt="--log"
        #     fi
        #     average_checkpoints.py \
        #         ${opt} \
        #         --backend ${backend} \
        #         --snapshots ${lmexpdir}/snapshot.ep.* \
        #         --out ${lmexpdir}/${lang_model} \
        #         --num ${lm_n_average}
        # fi
    fi

    pids=() # initialize pids
    for rtask in Test Replay; do
    (
        if [ "$rtask" == "Replay" ]; then
            nj=10
        fi 
        decode_dir=decode_${rtask}_${recog_model}_$(basename ${decode_config%.*})_${lmtag}
        # feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
        feat_recog_dir=${dumpdir}/av_JT_feat/${rtask}/delta${do_delta}

        # split data
        splitjson.py --parts ${nj} ${feat_recog_dir}/data_${bpemode}${nbpe}.json

        #### use CPU for decoding
        ngpu=0
        wav_dir=${expdir}/${decode_dir}/wav 
        mkdir -p $wav_dir
        # set batchsize 0 to disable batch decoding
        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            avsr_recog_JT.py \
            --config ${decode_config} \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --batchsize 0 \
            --recog-json ${feat_recog_dir}/split${nj}utt/data_${bpemode}${nbpe}.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/${recog_model}  \
            --write-wav true \
            --write-wav-dir $wav_dir
            # --rnnlm ${lmexpdir}/${lang_model} \
            # --api v2

        score_sclite.sh --bpe ${nbpe} --bpemodel ${bpemodel}.model --wer true ${expdir}/${decode_dir} ${dict}

    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished"
fi
           
exit 0
