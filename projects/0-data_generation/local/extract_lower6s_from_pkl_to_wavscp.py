import pickle
import sys

pkl_abs_path = sys.argv[1]
output_wavscp_path = sys.argv[2]
print(f"pkl_abs_path : {pkl_abs_path}")
print(f"output_wavscp_path: {output_wavscp_path}")

info = pickle.load(open(pkl_abs_path, 'rb'), encoding='utf-8')

wav_scp = {}
duration = 6
for item in info:
    dur = info[item]['time_idx'][0][1] - info[item]['time_idx'][0][0]
    if dur < duration: # 32 all data with lip embed
        wav_scp[item] = info[item]['wav_path']

print(f"all utterance lower than {duration} is {len(wav_scp)}")

with open(output_wavscp_path, 'w') as fw:
    for k, v in wav_scp.items():
        fw.write(f"{k} {v}\n")

print("finished!")    

