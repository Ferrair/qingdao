import os

file_list = os.listdir('/Users/bytedance/MODEL_ZS_5K_LD5_HS_08')
file_list = sorted(file_list)
rslt = []
for file in file_list:
    with open('/Users/bytedance/MODEL_ZS_5K_LD5_HS_08/' + file) as f:
        rslt.extend(f.readlines())

with open('/Users/bytedance/rslt.txt', 'w') as wf:
    wf.writelines(rslt)
