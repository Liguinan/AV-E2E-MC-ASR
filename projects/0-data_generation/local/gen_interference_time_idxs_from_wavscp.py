import sys
path = sys.argv[1]
output_path = sys.argv[2]
new_time_idx_scp = []
with open(path, "r", encoding='utf-8') as fr:
    lines = fr.readlines()
    for line in lines:
        line = line.strip()
        line_lst = line.split()
        # print(len(line_lst))
        if len(line_lst) != 63:
            print(line_lst[0])
            break
        interference_lst = line_lst[37].split('.')[0].split("/")
        # import pdb; pdb.set_trace()
        if interference_lst[-3] == "main":
            interference_id = f"{interference_lst[-2]}-{interference_lst[-1]}"
        else:
            interference_id = f"{interference_lst[-2]}-{interference_lst[-1]}_PRE"
        
        target_id = line_lst[0]

        dur = float(line_lst[53].split('=')[1])
        start_time_idx = float(line_lst[58].split('=')[-1].split(',')[-1].split('\'')[0])
        end_time_idx = (start_time_idx * 1000 + dur * 1000) / 1000

        # import pdb; pdb.set_trace()
        new_time_idx_scp.append(f"{target_id} {interference_id} {start_time_idx} {end_time_idx}")
        # print("7 25 46")
with open(output_path, 'w') as fw:
    for item in new_time_idx_scp:
        fw.write(f"{item}\n")

print("finished!")