import sys
path = sys.argv[1]
rt60=[]
with open(path, "r", encoding='utf-8') as fr:
    lines = fr.readlines()
    for line in lines:
        line = line.strip()
        rt60.append(float(line.split()[-1]))
print(f"average rt60: {round(sum(rt60)/len(rt60),2)}, min rt60:{min(rt60)}, max rt60:{max(rt60)} of all {len(rt60)} utterance")

# get start time and duration of interference from wav.scp and save spk_utt_id start_time end_time for pkl file
# get duration from utt2dur
# calculate the ratio and save them in ratio_post_cal.txt
# compare with the ratio.txt
# cal max ratio and cal min ratio and average ratio