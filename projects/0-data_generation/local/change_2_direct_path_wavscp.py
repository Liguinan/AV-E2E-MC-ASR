import sys
path = sys.argv[1]
output_path = sys.argv[2]
new_wav_scp = []
with open(path, "r", encoding='utf-8') as fr:
    lines = fr.readlines()
    for line in lines:
        line = line.strip()
        line_lst = line.split()
        # print(len(line_lst))
        if len(line_lst) != 63:
            print(line_lst[0])
            break
        tgt_wav = line_lst[7].split('.')[0] + "_2.wav"
        interference_wav = line_lst[46].split('.')[0] + "_2.wav"
        line_lst[7] = tgt_wav
        line_lst[46] = interference_wav
        # import pdb; pdb.set_trace()
        new_wav_scp.append(" ".join(line_lst))
        # print("7 25 46")
with open(output_path, 'w') as fw:
    for item in new_wav_scp:
        fw.write(f"{item}\n")

print("finished!")