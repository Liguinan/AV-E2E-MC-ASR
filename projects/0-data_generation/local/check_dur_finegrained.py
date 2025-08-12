import sys
import collections
def cal_dur_percent(path):
    cnt_dct = collections.defaultdict(lambda :0)
    with open(path, "r", encoding='utf-8') as fr:
        lines = fr.readlines()
        for line in lines:
            line = line.strip()
            dur = float(line.split()[-1])
            if dur >= 10: 
                continue
            cnt_dct[str(int(dur))] += 1 
    cnt_dct = sorted(cnt_dct.items(), key=lambda x:x[0])
    # import pdb; pdb.set_trace()
    all_cnt = 0
    for k, v in cnt_dct:
        all_cnt += v
    print(f"{all_cnt}")
    cnt_dct_percent = {}
    for k, v in cnt_dct:
        cnt_dct_percent[k] = round(v / all_cnt * 100, 2)
    return cnt_dct_percent, cnt_dct


path = sys.argv[1]
train_res = cal_dur_percent(path)

print(f"{train_res[0]}")
print(f"{train_res[1]}")
