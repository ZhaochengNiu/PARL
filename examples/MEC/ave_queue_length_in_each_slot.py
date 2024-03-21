import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(context='notebook', style='whitegrid', palette='deep', font='sans-serif', font_scale=1.0, color_codes=True, rc=None)

Local = np.loadtxt('./result/LocalAveQueueLengthInEachSlotFilePath10.cache')
Nearest = np.loadtxt('./result/NearestAveQueueLengthInEachSlotFilePath10.cache')
Random = np.loadtxt('./result/RandomAveQueueLengthInEachSlotFilePath10.cache')
Match = np.loadtxt('./result/MatchAveQueueLengthInEachSlotFilePath10.cache')
# Proposed = np.loadtxt('./result/ProposedAveQueueLengthInEachSlotFilePath50.cache')

# 生成数据
x = np.arange(0, 1000, 1)  # 横坐标数据为从0到1000之间，步长为1的等差数组

# 填充颜色
colors = ['#706fd3', '#34ace0', '#33d9b2', '#ffb142', '#c56cf0','#f78fb3', '#ff4d4d']

# 生成图形
Local, = plt.plot(x, Local, color=colors[0], alpha=0.7, linestyle='-', linewidth=2)
Nearest, = plt.plot(x, Nearest, color=colors[1], alpha=0.7, linestyle='-', linewidth=2)
Random, = plt.plot(x, Random, color=colors[2], alpha=0.7, linestyle='-', linewidth=2)
Match, = plt.plot(x, Match, color=colors[3], alpha=0.7, linestyle='-', linewidth=2)
# Proposed, = plt.plot(x, Proposed, color=colors[4], alpha=0.7, linestyle='-', linewidth=2)


# colors = ['#9999FF', '#58C9B9', '#CC33CC', '#D1B6E1', '#99FF99', '#FF6666']

# alpha：不透明度（值为“0”时为透明状态，默认为“1”）
# linestype：线条类型
# linewidth：线条宽度。
# color：颜色。

plt.xlim(0, 1000)


plt.xlabel('Time slot (t)')
plt.ylabel('Average backlog of queues in the system (MB)')
# 设置legend
# plt.legend(handles=[Local, Nearest, Random, Match, Proposed], labels=['Local', 'Nearest', 'Random', 'Match', 'Proposed'], loc='best')
plt.legend(handles=[Local, Nearest, Random, Match], labels=['Local', 'Nearest', 'Random', 'Match'], loc='best')
plt.savefig('./AveQueueLengthInEachSlot10.pdf', format='pdf')
# 显示图形
plt.show()
