import sys
import pickle
import os
trptr_pkl_path_file = sys.argv[1]  # leave lip emb
trptr_utt2dur_path_file = sys.argv[2]
output_dir = sys.argv[3]

trptr_utt2dur_dct = {}
with open(trptr_utt2dur_path_file, 'r', encoding='utf-8') as fr:
    lines = fr.readlines()
    for line in lines:
        line = line.strip()
        line_lst = line.split()
        trptr_utt2dur_dct[line_lst[0]] = line_lst[1]

trptr_pkl_lst = []
for k, _ in pickle.load(open(trptr_pkl_path_file, 'rb')).items():
    utter_id = k.split('-', 1)[-1]
    trptr_pkl_lst.append(utter_id)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

with open(f"{output_dir}/leave_utt2dur", 'w') as fw:
    # import pdb; pdb.set_trace()
    cnt = 1
    for k, v in trptr_utt2dur_dct.items():
        if k not in trptr_pkl_lst:
            print(f"cnt: {cnt}")
            fw.write(f"{k} {v}\n")
            cnt += 1







