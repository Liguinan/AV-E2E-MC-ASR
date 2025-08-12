import sys 
from pathlib import Path

rt60_abs_path_file=Path(sys.argv[1])
split_group = 3
out_path = rt60_abs_path_file.parent

rt60 = {}
with open(rt60_abs_path_file, 'r') as fr:
    lines = fr.readlines()
    for line in lines:
        line = line.strip()
        line_lst = line.split()
        rt60[line_lst[0]] = float(line_lst[1])

utter_num = len(rt60)
split_idx = utter_num // split_group
# sort as value
rt60_sort = sorted(rt60.items(), key=lambda x:x[1])
split_value_1 = float(rt60_sort[split_idx-1][1])
split_value_2 = float(rt60_sort[split_idx * 2 - 1][1])
# import pdb; pdb.set_trace()
print(f"split_value:{split_value_1}, {split_value_2}")
cnt1, cnt2, cnt3 = 0, 0, 0
with open(out_path/f'0-{split_value_1}.txt', 'w') as fw1, open(out_path/f'{split_value_1}-{split_value_2}.txt', 'w') as fw2, open(out_path/f'{split_value_2}-1.txt', 'w') as fw3:
    for k, v in rt60.items():
        # import pdb; pdb.set_trace()
        if 0<v<=split_value_1: 
            cnt1 += 1
            fw1.write(f"{k} {v}\n")
        elif split_value_1<v<=split_value_2: 
            cnt2 += 1
            fw2.write(f"{k} {v}\n")
        else: 
            cnt3 += 1
            fw3.write(f"{k} {v}\n")
print(f"group1: {cnt1}, group2: {cnt2}, group3: {cnt3}")










