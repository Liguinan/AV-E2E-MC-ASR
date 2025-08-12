import pickle
import sys

pkl_abs_path = sys.argv[1] # to calculate length by time idx as a filter
input_wavscp_path = sys.argv[2] # input all wavscp
output_wavscp_path = sys.argv[3] # output fixed duration wavscp
print(f"pkl_abs_path : {pkl_abs_path}")
print(f"input_wavscp_path: {input_wavscp_path}")
print(f"output_wavscp_path: {output_wavscp_path}")

input_wavscp = {}
with open(input_wavscp_path, 'r') as fr:
    lines = fr.readlines()
    for line in lines:
        line = line.strip()
        line_lst = line.split()
        input_wavscp[line_lst[0]] = line_lst[1]
       
# import pdb; pdb.set_trace()

info = pickle.load(open(pkl_abs_path, 'rb'), encoding='utf-8')

wav_scp = {}
duration = 6
for item in info:
    dur = info[item]['time_idx'][0][1] - info[item]['time_idx'][0][0]
    if dur < duration: # 32 all data with lip embed
        wav_scp[item] = input_wavscp[item]

print(f"all utterance lower than {duration} is {len(wav_scp)}")

with open(output_wavscp_path, 'w') as fw:
    for k, v in wav_scp.items():
        fw.write(f"{k} {v}\n")

print("finished!")    

