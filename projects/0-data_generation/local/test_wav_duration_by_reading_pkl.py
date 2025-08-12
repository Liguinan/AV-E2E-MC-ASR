import pickle
import sys
import librosa

pkl_abs_path = sys.argv[1]
ref_pkl_abs_path = sys.argv[2]
output_dur_path = sys.argv[3]
print(f"pkl_abs_path : {pkl_abs_path}")
print(f"ref_pkl_abs_path : {ref_pkl_abs_path}")
print(f"output_dur_path: {output_dur_path}")

info = pickle.load(open(pkl_abs_path, 'rb'), encoding='utf-8')
info_ref = pickle.load(open(ref_pkl_abs_path, 'rb'), encoding='utf-8')

dur_ref_dct = {}
for item in info_ref:
    # import pdb; pdb.set_trace()
    dur = info_ref[item]['time_idx'][0][1] - info_ref[item]['time_idx'][0][0]
    dur_ref_dct[item] = dur

dur_scp = {}
for item in info:
    dur = info[item]['time_idx'][0][1] - info[item]['time_idx'][0][0]
    # import pdb; pdb.set_trace()
    dur_real = round(librosa.get_duration(librosa.load(info[item]['wav_path'], sr=16000, mono=False)[0], sr=16000), 3)
    dur_ref = dur_ref_dct[item]
    dur_scp[item] = [str(dur), str(dur_real), str(dur_ref)]

# print(f"all utterance lower than {duration} is {len(wav_scp)}")

with open(output_dur_path, 'w') as fw:
    for k, v in dur_scp.items():
        value = " ".join(v)
        fw.write(f"{k} {value}\n")

print("finished!")  
