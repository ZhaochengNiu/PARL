import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(context='notebook', style='whitegrid', palette='deep', font='sans-serif', font_scale=1.0, color_codes=True, rc=None)

Local = np.loadtxt('./LocalIOInEachSlotFilePath20.cache')
Nearest = np.loadtxt('./NearestIOInEachSlotFilePath20.cache')
Random = np.loadtxt('./RandomIOInEachSlotFilePath20.cache')
Game = np.loadtxt('./GameIOInEachSlotFilePath20.cache')
# 生成数据
# x = np.arange(0, 1000, 1)  # 横坐标数据为从0到100之间，步长为1的等差数组

x = np.arange(0, 1000, 5)

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
# Local, = plt.plot(x, Local, 'r,-')
# Nearest, = plt.plot(x, Nearest, 'y,-')
# Random, = plt.plot(x, Random, 'b,-')
# Game, = plt.plot(x, Game, 'c,-')

Local_temp, = plt.plot(x, Local_temp, color='r', alpha=0.6, linestyle='-', linewidth=2)
Nearest_temp, = plt.plot(x, Nearest_temp, color='y', alpha=0.6, linestyle='-', linewidth=2)
Random_temp, = plt.plot(x, Random_temp, color='b', alpha=0.6, linestyle='-', linewidth=2)
Game_temp, = plt.plot(x, Game_temp, color='c', alpha=0.6, linestyle='-', linewidth=2)

plt.xlabel('Time slot (t)')
plt.ylabel('Throughput in each slot (s)')
# 设置legend
# plt.legend(handles=[Local, Nearest, Random, Game], labels=['Local', 'Nearest', 'Random', 'Game'], loc='best')
plt.legend(handles=[Local_temp, Nearest_temp, Random_temp, Game_temp], labels=['Local', 'Nearest', 'Random', 'Game'], loc='best')

plt.savefig('./IOInEachSlot.pdf', format='pdf')

# 显示图形
plt.show()