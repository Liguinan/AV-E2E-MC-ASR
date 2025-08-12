import sys
path_azimuth = sys.argv[1]
path_dur = sys.argv[2]
filter_dur = 6
all_dur = 0
dur_dct = {}
with open(path_dur, "r", encoding='utf-8') as fr:
    lines = fr.readlines()
    for line in lines:
        line = line.strip()
        line_lst = line.split()
        if float(line_lst[-1]) >= filter_dur:
            continue
        all_dur += float(line_lst[-1])
        dur_dct[line_lst[0]] = float(line_lst[-1])
    print(f"{len(list(dur_dct.items()))} utterances of dur below {filter_dur}, all duration under this dur filter is {all_dur/3600}")

dur_0_15, dur_15_45, dur_45_90, dur_90_180, other = 0, 0, 0, 0, 0
ad_0_15, ad_15_45, ad_45_90, ad_90_180, other = 0, 0, 0, 0, 0
with open(path_azimuth, "r", encoding='utf-8') as fr:
    lines = fr.readlines()
    for line in lines:
        line = line.strip()
        line_lst = line.split()
        utter_id = line_lst[0].split("-", 1)[-1]
        if utter_id not in dur_dct:
            continue
        # if line.split()[0].endswith("_PRE"):
        #     continue
        ad = float(line.split()[-1])
        if 0 <= ad <= 15:
            ad_0_15 += 1
            dur_0_15 += dur_dct[utter_id]
        elif 15 < ad <= 45:
            ad_15_45 += 1
            dur_15_45 += dur_dct[utter_id]
        elif 45 < ad <= 90:
            ad_45_90 += 1
            dur_45_90 += dur_dct[utter_id]
        elif 90 < ad <= 180:
            ad_90_180 += 1
            dur_90_180 += dur_dct[utter_id]
        else:
            other += 1
print(f"0-15: {ad_0_15}\t{dur_0_15/3600}\n 15-45: {ad_15_45}\t{dur_15_45/3600} \n 45-90:{ad_45_90}\t{dur_45_90/3600} \n 90-180:{ad_90_180}\t{dur_90_180/3600}")
