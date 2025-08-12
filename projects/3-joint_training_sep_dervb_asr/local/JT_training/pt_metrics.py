import numpy as np
import os
import sys
import result_analysis as ra
import soundfile
import pickle


pc = ra.MultiChannelPerformance("Front-end metrics")

def run(wavscp, pkl, metric_res_dir, using_clean):
    wavscp_dct = {}
    with open(wavscp, "r") as fr:
        lines = fr.readlines()
        for line in lines:
            line = line.strip()
            line_lst = line.split()
            wavscp_dct[line_lst[0]] = line_lst[1]
    print(f"wavscp utter num: {len(wavscp_dct)}")
    
    # load pickle file
    pkldata = {}
    with open(pkl, 'rb') as fp:
        pkldata = pickle.load(fp, encoding='utf-8')
        print(f"utterance num in ${pkl} is:  {len(pkldata.keys())}.")

    sisnr_lst = []
    pesq_lst = []
    stoi_lst = []
    srmr_lst = []
    # import pdb;pdb.set_trace()
    for wav_file, est_wav_path in wavscp_dct.items():
        ref_path = pkldata[wav_file]["wav_path"]
        est, est_sr = soundfile.read(est_wav_path, dtype="float32")
        ref_mc, ref_sr = soundfile.read(ref_path, dtype="float32")
        if using_clean=="true":
            ref = ref_mc[:, 15] # 15 means clean ref, 16 means rvb-clean ref
        else:
            ref = ref_mc[:, 16] # 15 means clean ref, 16 means rvb-clean ref
            
        print(f"est sr:{est_sr}, ref sr: {ref_sr}")

        sisnr = ra.get_SI_SNR(est, ref)
        pesq = ra.get_PESQ(est, ref)
        stoi = ra.get_STOI(est, ref)
        srmr = ra.get_SRMR(est, ref)
        
        sisnr_lst.append(sisnr)
        pesq_lst.append(pesq)
        stoi_lst.append(stoi)
        srmr_lst.append(srmr)
    assert len(sisnr_lst) == len(pesq_lst) == len(stoi_lst) == len(srmr_lst)
    print(f"metrics len: {len(sisnr_lst)}")
    
    if using_clean=='true':
        print("Metrics saved in metrics_clean.log")
        # with open(f"{metric_res_dir}/pipeline_metrics_clean.log", 'w') as fw:
        with open(f"{metric_res_dir}/metrics_clean.log", 'w') as fw:
            fw.write(f"sisnr: {np.mean(np.array(sisnr_lst))}\n")
            fw.write(f"pesq: {np.mean(np.array(pesq_lst))}\n")
            fw.write(f"stoi: {np.mean(np.array(stoi_lst))}\n")
            fw.write(f"srmr: {np.mean(np.array(srmr_lst))}\n")
    else:
        print("Metrics saved in metrics_rvb_clean.log")
        # with open(f"{metric_res_dir}/pipeline_metrics_rvb_clean.log", 'w') as fw:
        with open(f"{metric_res_dir}/metrics_rvb_clean.log", 'w') as fw:
            fw.write(f"sisnr: {np.mean(np.array(sisnr_lst))}\n")
            fw.write(f"pesq: {np.mean(np.array(pesq_lst))}\n")
            fw.write(f"stoi: {np.mean(np.array(stoi_lst))}\n")
            fw.write(f"srmr: {np.mean(np.array(srmr_lst))}\n")

    print("finished calculating frontend metrics!")
        

    
        
        
"""
sys.argv[1]: est wav.scp
sys.argv[2]: ref pickle
sys.argv[3]: metric saved dir
sys.argv[4]: using rvb_clean or clean, default is true(clean)
"""

run(
    sys.argv[1],
    sys.argv[2],
    sys.argv[3],
    sys.argv[4],
)           
    
