import sys
path = sys.argv[1]
ad_0_15, ad_15_45, ad_45_90, ad_90_180, other = 0, 0, 0, 0, 0
with open(path, "r", encoding='utf-8') as fr:
    lines = fr.readlines()
    for line in lines:
        line = line.strip()
        # if line.split()[0].endswith("_PRE"):
        #     continue
        ad = float(line.split()[-1])
        if 0 <= ad <= 15:
            ad_0_15 += 1
        elif 15 < ad <= 45:
            ad_15_45 += 1
        elif 45 < ad <= 90:
            ad_45_90 += 1
        elif 90 < ad <= 180:
            ad_90_180 += 1
        else:
            other += 1
print(ad_0_15, ad_15_45, ad_45_90, ad_90_180, other)
