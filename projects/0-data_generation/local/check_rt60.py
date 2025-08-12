import sys
path = sys.argv[1]
rt60=[]
with open(path, "r", encoding='utf-8') as fr:
    lines = fr.readlines()
    for line in lines:
        line = line.strip()
        rt60.append(float(line.split()[-1]))
print(f"average rt60: {round(sum(rt60)/len(rt60),2)}, min rt60:{min(rt60)}, max rt60:{max(rt60)} of all {len(rt60)} utterance")