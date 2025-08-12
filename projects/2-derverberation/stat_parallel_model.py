import sys

dir_name=sys.argv[1]
jobs=int(sys.argv[2])  # 8 or 10 or 15
epoch=int(sys.argv[3])
dataset=sys.argv[4]
res_dct = {'SISNR':[], 'PESQ':[], 'SRMR': [], 'STOI': [] }
for job_num in range(1, jobs+1):
    file_name = f'{dir_name}/{dataset}-{epoch}_job_{job_num}.log'
    with open(file_name, 'r') as fr:
        lines = fr.readlines()
        metric = None
        for line in lines:
            line = line.strip()
            if line.startswith('SISNR'):
                metric = 'SISNR'
            if line.startswith('PESQ'):
                metric = 'PESQ'
            if line.startswith('SRMR'):
                metric = 'SRMR'
            if line.startswith('STOI'):
                metric = 'STOI'
            if line.startswith("all:"):
                # import pdb; pdb.set_trace()
                line_lst = line.split()
                if line_lst[1] != 'nan':
                    res_dct[metric].append(round(float(line_lst[1]), 4)) 
                else:
                    res_dct[metric].append(0.0)

assert len(res_dct['SISNR']) == jobs and len(res_dct['PESQ']) == jobs and len(res_dct['SRMR']) == jobs and len(res_dct['STOI']) == jobs, f"len SISNR: {len(res_dct['SISNR'])}, len PESQ: {len(res_dct['PESQ'])}, len SRMR: {len(res_dct['SRMR'])}, len STOI: {len(res_dct['STOI'])}"
print(f"res_dct: {res_dct}")

import numpy as np
with open(f"{dir_name}/res.log", 'w') as fw:
    fw.write(f"{res_dct}\n")
    for k, v in res_dct.items():
        print(f"{k}: {np.mean(v)}")
        fw.write(f"{k}: {np.mean(v)}\n")

print("finished!")        
        
