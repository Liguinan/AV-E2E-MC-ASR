import sys
path = sys.argv[1]
dur_0_1, dur_1_5, dur_5_10, dur_10_20, dur_20_32, dur_32_50, dur_over_50 = 0, 0, 0, 0, 0, 0, 0
with open(path, "r", encoding='utf-8') as fr:
    lines = fr.readlines()
    for line in lines:
        line = line.strip()
        dur = float(line.split()[-1])
        if 0 < dur < 1:
            dur_0_1 += 1
        elif 1 <= dur <= 5:
            dur_1_5 += 1
        elif 5 < dur <= 10:
            dur_5_10 += 1
        elif 10 < dur <= 20:
            dur_10_20 += 1
        elif 20 < dur < 32:
            dur_20_32 += 1
        elif 32 <= dur < 50:
            dur_32_50 += 1
        else:
            dur_over_50 += 1
print(f"dur_0_1:{dur_0_1}, dur_1_5: {dur_1_5}, dur_5_10:{dur_5_10}, dur_10_20:{dur_10_20}, dur_20_32:{dur_20_32}, dur_32_50:{dur_32_50}, dur_over_50:{dur_over_50}")
