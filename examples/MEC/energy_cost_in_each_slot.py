import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(context='notebook', style='whitegrid', palette='deep', font='sans-serif', font_scale=1.0, color_codes=True, rc=None)

Local = np.loadtxt('./result/LocalEnergyCostInEachSlotFilePath10.cache')
Nearest = np.loadtxt('./result/NearestEnergyCostInEachSlotFilePath10.cache')
Random = np.loadtxt('./result/RandomEnergyCostInEachSlotFilePath10.cache')
Game = np.loadtxt('./result/GameEnergyCostInEachSlotFilePath10.cache')

# 生成数据
x = np.arange(0, 1000, 5)  # 横坐标数据为从0到1000之间，步长为1的等差数组

Local_temp = []
Nearest_temp = []
Random_temp = []
Game_temp = []

for i in range(0, 1000, 5):
    Local_temp.append(Local[i])
    Nearest_temp.append(Nearest[i])
    Random_temp.append(Random[i])
    Game_temp.append(Game[i])

# 生成图形
Local, = plt.plot(x, Local_temp, color='#bb3f3f', alpha=0.7, linestyle='-', linewidth=2)
Nearest, = plt.plot(x, Nearest_temp, color='#fcb001', alpha=0.7, linestyle='-', linewidth=2)
Random, = plt.plot(x, Random_temp, color='#016795', alpha=0.7, linestyle='-', linewidth=2)
Game, = plt.plot(x, Game_temp, color='#ad03de', alpha=0.7, linestyle='-', linewidth=2)

# alpha：不透明度（值为“0”时为透明状态，默认为“1”）
# linestype：线条类型
# linewidth：线条宽度。
# color：颜色。

plt.xlim(0, 1000)


plt.xlabel('Time slot (t)')
plt.ylabel('Total energy cost in the system (MB)')
# 设置legend
plt.legend(handles=[Local, Nearest, Random, Game], labels=['Local', 'Nearest', 'Random', 'Game'], loc='best')

plt.savefig('./EnergyCostInEachSlot10.pdf', format='pdf')
# 显示图形
plt.show()