import json
import os
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

filename = '/Users/bytedance/Qihang/青岛/rslt.txt'
y_pred = []
y_true = []
flow = []
with open(filename) as f:
    lines = f.readlines()
    for line in lines:
        line = line[24:]
        line = eval(line)
        y_pred.append([line.get('tempRegion1'), line.get('tempRegion2')])
        y_true.append([line.get('debug_info').get('temp1'), line.get('debug_info').get('temp2')])
        flow.append(line.get('debug_info').get('flow'))

y_true = y_true
y_pred = y_pred
y_true = np.array(y_true)
y_pred = np.array(y_pred)

# plt.figure(figsize=(8, 8))
# plt.title('一区温度预测值和真实值比较')
# # plt.axis([130, 138, 130, 138])
# plt.xlabel('真实值')
# plt.ylabel('预测值')
# plt.scatter(y_true[:, 0], y_pred[:, 0], alpha=0.6)
# # plt.plot([135, 145], [120, 125], c='r')
# plt.show()
#
# plt.figure(figsize=(8, 8))
# plt.title('二区温度预测值和真实值比较')
# # plt.axis([130, 138, 130, 138])
# plt.xlabel('真实值')
# plt.ylabel('预测值')
# plt.scatter(y_true[:, 1], y_pred[:, 1], alpha=0.6)
# # plt.plot([130, 138], [130, 138], c='r')
# plt.show()


plt.figure(figsize=(16, 8))
plt.grid(True)
plt.plot(y_true[:, 0], 'b-', linewidth=1)
plt.plot(y_pred[:, 0], 'r-', linewidth=1)
plt.xlabel('Timeline', fontsize=18)
plt.xticks(fontsize=18)
plt.ylabel('Values', fontsize=18)
plt.yticks(fontsize=18)
plt.legend(['现有调控值', '模型预测值'], loc='best', fontsize=22)
plt.show()

plt.figure(figsize=(16, 8))
plt.grid(True)
plt.plot(y_true[:, 1], 'b-', linewidth=1)
plt.plot(y_pred[:, 1], 'r-', linewidth=1)
plt.xlabel('Timeline', fontsize=18)
plt.xticks(fontsize=18)
plt.ylabel('Values', fontsize=18)
plt.yticks(fontsize=18)
plt.legend(['现有调控值', '模型预测值'], loc='best', fontsize=22)
plt.show()

# plt.figure(figsize=(16, 8))
# plt.grid(True)
# plt.plot(flow, 'b-', linewidth=1)
# plt.xlabel('Timeline', fontsize=18)
# plt.xticks(fontsize=18)
# plt.ylabel('Values', fontsize=18)
# plt.yticks(fontsize=18)
# plt.legend(['流量'], loc='best', fontsize=22)
# plt.show()
