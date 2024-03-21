import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_theme(context='notebook', style='whitegrid', palette='deep', font='sans-serif', font_scale=1.0, color_codes=True, rc=None)

Local10 = np.loadtxt('./result/LocalAveExecuteLatencyInEachSlotFilePath10.cache')
Nearest10 = np.loadtxt('./result/NearestAveExecuteLatencyInEachSlotFilePath10.cache')
Random10 = np.loadtxt('./result/RandomAveExecuteLatencyInEachSlotFilePath10.cache')
Proposed10 = np.loadtxt('./result/ProposedAveExecuteLatencyInEachSlotFilePath10.cache')

Local20 = np.loadtxt('./result/LocalAveExecuteLatencyInEachSlotFilePath20.cache')
Nearest20 = np.loadtxt('./result/NearestAveExecuteLatencyInEachSlotFilePath20.cache')
Random20 = np.loadtxt('./result/RandomAveExecuteLatencyInEachSlotFilePath20.cache')
Proposed20 = np.loadtxt('./result/ProposedAveExecuteLatencyInEachSlotFilePath20.cache')

Local30 = np.loadtxt('./result/LocalAveExecuteLatencyInEachSlotFilePath30.cache')
Nearest30 = np.loadtxt('./result/NearestAveExecuteLatencyInEachSlotFilePath30.cache')
Random30 = np.loadtxt('./result/RandomAveExecuteLatencyInEachSlotFilePath30.cache')
Proposed30 = np.loadtxt('./result/ProposedAveExecuteLatencyInEachSlotFilePath30.cache')

Local40 = np.loadtxt('./result/LocalAveExecuteLatencyInEachSlotFilePath40.cache')
Nearest40 = np.loadtxt('./result/NearestAveExecuteLatencyInEachSlotFilePath40.cache')
Random40 = np.loadtxt('./result/RandomAveExecuteLatencyInEachSlotFilePath40.cache')
Proposed40 = np.loadtxt('./result/ProposedAveExecuteLatencyInEachSlotFilePath40.cache')

Local50 = np.loadtxt('./result/LocalAveExecuteLatencyInEachSlotFilePath50.cache')
Nearest50 = np.loadtxt('./result/NearestAveExecuteLatencyInEachSlotFilePath50.cache')
Random50 = np.loadtxt('./result/RandomAveExecuteLatencyInEachSlotFilePath50.cache')
Match50 = np.loadtxt('./result/MatchAveExecuteLatencyInEachSlotFilePath50.cache')
Proposed50 = np.loadtxt('./result/ProposedAveExecuteLatencyInEachSlotFilePath50.cache')

Local10_utility = sum(Local10)
Nearest10_utility = sum(Nearest10)
Random10_utility = sum(Random10)
Match10_utility = sum(Match10)
Proposed10_utility = sum(Proposed10)

Local20_utility = sum(Local20)
Nearest20_utility = sum(Nearest20)
Random20_utility = sum(Random20)
Match20_utility = sum(Match20)
Proposed20_utility = sum(Proposed20)

Local30_utility = sum(Local30)
Nearest30_utility = sum(Nearest30)
Random30_utility = sum(Random30)
Match30_utility = sum(Match30)
Proposed30_utility = sum(Proposed30)

Local40_utility = sum(Local40)
Nearest40_utility = sum(Nearest40)
Random40_utility = sum(Random40)
Match40_utility = sum(Match40)
Proposed40_utility = sum(Proposed40)

Local50_utility = sum(Local50)
Nearest50_utility = sum(Nearest50)
Random50_utility = sum(Random50)
Match50_utility = sum(Match50)
Proposed50_utility = sum(Proposed50)

# print(Local20_utility)
# print(Nearest20_utility)
# print(Random20_utility)
# print(Game20_utility)
# print(Proposed20_utility)

labels = ['10', '20', '30', '40', '50']
Local = []
Nearest = []
Random = []
Match = []
Proposed = []

Local.append(Local10_utility)
Nearest.append(Nearest10_utility)
Random.append(Random10_utility)
Match.append(Match10_utility)
Proposed.append(Proposed10_utility)

Local.append(Local20_utility)
Nearest.append(Nearest20_utility)
Random.append(Random20_utility)
Match.append(Match20_utility)
Proposed.append(Proposed20_utility)

Local.append(Local30_utility)
Nearest.append(Nearest30_utility)
Random.append(Random30_utility)
Match.append(Match30_utility)
Proposed.append(Proposed30_utility)

Local.append(Local40_utility)
Nearest.append(Nearest40_utility)
Random.append(Random40_utility)
Match.append(Match40_utility)
Proposed.append(Proposed40_utility)

Local.append(Local50_utility)
Nearest.append(Nearest50_utility)
Random.append(Random50_utility)
Match.append(Match50_utility)
Proposed.append(Proposed50_utility)

x = np.arange(len(labels))  # the label locations
width = 0.1  # the width of the bars

# colors = ['#9999FF', '#58C9B9', '#CC33CC', '#D1B6E1', '#99FF99', '#FF6666']

# 填充颜色
colors = ['#706fd3', '#34ace0', '#33d9b2', '#ffb142', '#c56cf0','#f78fb3', '#ff4d4d']

fig, ax = plt.subplots()
Local = ax.bar(x - 2*width, Local, width=width, label='Local', edgecolor='black', color=colors[0], linewidth=.8, hatch='//')
Nearest = ax.bar(x - width, Nearest, width=width, label='Nearest', edgecolor='black', color=colors[1], linewidth=.8, hatch='//')
Random = ax.bar(x, Random, width=width, label='Random', edgecolor='black', color=colors[2], linewidth=.8, hatch='//')
Match = ax.bar(x + width, Match, width=width, label='Random', edgecolor='black', color=colors[3], linewidth=.8, hatch='//')
Proposed = ax.bar(x + 2*width, Proposed, width=width, label='Proposed', edgecolor='black', color=colors[4], linewidth=.8, hatch='//')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Number of devices')
ax.set_ylabel('Average execute latency')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
plt.savefig('./AveExecuteLatencyStackedBar.pdf', format='pdf')
plt.show()
