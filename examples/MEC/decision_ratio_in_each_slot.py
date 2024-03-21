import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(context='notebook', style='whitegrid', font='sans-serif', font_scale=1.0, color_codes=True, rc=None)

Local = np.loadtxt('LocalRatioInEachSlotFilePath10.cache')
Edge = np.loadtxt('EdgeRatioInEachSlotFilePath10.cache')
D2D = np.loadtxt('D2DRatioInEachSlotFilePath10.cache')

# 生成数据
# x = np.arange(0, 1000, 1)  # 横坐标数据为从0到10之间，步长为0.1的等差数组
x = np.arange(0, 1000, 5)  # 横坐标数据为从0到10之间，步长为0.1的等差数组

Local_temp = []
Edge_temp = []
D2D_temp = []

for i in range(0, 1000, 5):
    Local_temp.append(Local[i])
    Edge_temp.append(Edge[i])
    D2D_temp.append(D2D[i])

# 生成图形
# Local, = plt.plot(x, Local, 'r,-', linewidth=1)
# Edge, = plt.plot(x, Edge, 'y,-', linewidth=1)
# D2D, = plt.plot(x, D2D, 'b,-', linewidth=1)

Local_temp, = plt.plot(x, Local_temp, color='#FF6347', alpha=0.7, linestyle='-', linewidth=1.5)
Edge_temp, = plt.plot(x, Edge_temp, color='#009337', alpha=0.7, linestyle='-', linewidth=1.5)
D2D_temp, = plt.plot(x, D2D_temp, color='#0339f8', alpha=0.7, linestyle='-', linewidth=1.5)

# alpha 透明度 设置

plt.xlim(0, 1000)

plt.xlabel('Time slot (t)')
plt.ylabel('Ratio of different decision mode in each slot (s)')
# 设置legend
# plt.legend(handles=[Local, Edge, D2D], labels=['Local Ratio', 'Edge Ratio', 'D2D Ratio'], loc='best')

plt.legend(handles=[Local_temp, Edge_temp, D2D_temp], labels=['Local Ratio', 'Edge Ratio', 'D2D Ratio'], loc='best')

plt.savefig('./DecisionRatioInEachSlot.pdf', format='pdf')
# 显示图形

plt.show()
