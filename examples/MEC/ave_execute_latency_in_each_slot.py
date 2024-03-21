import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(context='notebook', style='whitegrid', font='sans-serif', font_scale=1.0, color_codes=True, rc=None)

# styledict, or one of {darkgrid, whitegrid, dark, white, ticks}
# palette = "pastel" "husl" "Set2"

Local = np.loadtxt('./result/LocalAveExecuteLatencyInEachSlotFilePath10.cache')
Nearest = np.loadtxt('./result/NearestAveExecuteLatencyInEachSlotFilePath10.cache')
Random = np.loadtxt('./result/RandomAveExecuteLatencyInEachSlotFilePath10.cache')
Match = np.loadtxt('./result/MatchAveExecuteLatencyInEachSlotFilePath10.cache')
Proposed = np.loadtxt('./result/ProposedAveExecuteLatencyInEachSlotFilePath10.cache')

# 生成数据
x = np.arange(0, 1000, 5)  # 横坐标数据为从0到1000之间，步长为1的等差数组

Local_temp = []
Nearest_temp = []
Random_temp = []
Match_temp = []
# Proposed_temp = []

for i in range(0, 1000, 5):
    Local_temp.append(Local[i])
    Nearest_temp.append(Nearest[i])
    Random_temp.append(Random[i])
    Match_temp.append(Match[i])
    # Proposed_temp.append(Proposed[i])

# 生成图形
# Local, = plt.plot(x, Local, 'r,-', linewidth=1)
# Nearest, = plt.plot(x, Nearest, 'y,-', linewidth=1)
# Random, = plt.plot(x, Random, 'b,-', linewidth=1)
# Game, = plt.plot(x, Game, 'c,-', linewidth=1)

Local_temp, = plt.plot(x, Local_temp, color='#bb3f3f', alpha=0.8, linestyle='-', linewidth=1.2)
Nearest_temp, = plt.plot(x, Nearest_temp, color='#fcb001', alpha=0.8, linestyle='-', linewidth=1.2)
Random_temp, = plt.plot(x, Random_temp, color='#016795', alpha=0.8, linestyle='-', linewidth=1.2)
Match_temp, = plt.plot(x, Match_temp, color='#9999FF', alpha=0.8, linestyle='-', linewidth=1.2)
# Proposed_temp, = plt.plot(x, Proposed_temp, color='#ad03de', alpha=0.8, linestyle='-', linewidth=1.2)

# colors = ['#9999FF', '#58C9B9', '#CC33CC', '#D1B6E1', '#99FF99', '#FF6666']
plt.xlabel('Time slot (t)')
plt.ylabel('Average execute latency in each slot (s)')

plt.xlim(0, 1000)

# 设置legend
# plt.legend(handles=[Local, Nearest, Random, Game], labels=['Local', 'Nearest', 'Random', 'Game'], loc='best')
# plt.legend(handles=[Local_temp, Nearest_temp, Random_temp, Match_temp, Proposed_temp], labels=['Local', 'Nearest', 'Random', 'Match', 'Proposed'], loc='best')
# 显示图形
plt.legend(handles=[Local_temp, Nearest_temp, Random_temp, Match_temp], labels=['Local', 'Nearest', 'Random', 'Match'], loc='best')

plt.savefig('./AveExecuteLatencyInEachSlot10.pdf', format='pdf')

plt.show()
