import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_theme(context='notebook', style='whitegrid', palette='deep', font='sans-serif', font_scale=1.0, color_codes=True, rc=None)

Local10 = np.loadtxt('./result/LocalIOInEachSlotFilePath10.cache')
Nearest10 = np.loadtxt('./result/NearestIOInEachSlotFilePath10.cache')
Random10 = np.loadtxt('./result/RandomIOInEachSlotFilePath10.cache')
Proposed10 = np.loadtxt('./result/ProposedIOInEachSlotFilePath10.cache')

Local20 = np.loadtxt('./result/LocalIOInEachSlotFilePath20.cache')
Nearest20 = np.loadtxt('./result/NearestIOInEachSlotFilePath20.cache')
Random20 = np.loadtxt('./result/RandomIOInEachSlotFilePath20.cache')
Proposed20 = np.loadtxt('./result/ProposedIOInEachSlotFilePath20.cache')

Local30 = np.loadtxt('./result/LocalIOInEachSlotFilePath30.cache')
Nearest30 = np.loadtxt('./result/NearestIOInEachSlotFilePath30.cache')
Random30 = np.loadtxt('./result/RandomIOInEachSlotFilePath30.cache')
Proposed30 = np.loadtxt('./result/ProposedIOInEachSlotFilePath30.cache')

Local40 = np.loadtxt('./result/LocalIOInEachSlotFilePath40.cache')
Nearest40 = np.loadtxt('./result/NearestIOInEachSlotFilePath40.cache')
Random40 = np.loadtxt('./result/RandomIOInEachSlotFilePath40.cache')
Proposed40 = np.loadtxt('./result/ProposedIOInEachSlotFilePath40.cache')

Local50 = np.loadtxt('./result/LocalIOInEachSlotFilePath50.cache')
Nearest50 = np.loadtxt('./result/NearestIOInEachSlotFilePath50.cache')
Random50 = np.loadtxt('./result/RandomIOInEachSlotFilePath50.cache')
Proposed50 = np.loadtxt('./result/ProposedIOInEachSlotFilePath50.cache')

Local10_io = sum(Local10)
Nearest10_io = sum(Nearest10)
Random10_io = sum(Random10)
Proposed10_io = sum(Proposed10)

Local20_io = sum(Local20)
Nearest20_io = sum(Nearest20)
Random20_io = sum(Random20)
Proposed20_io = sum(Proposed20)

Local30_io = sum(Local30)
Nearest30_io = sum(Nearest30)
Random30_io = sum(Random30)
Proposed30_io = sum(Proposed30)

Local40_io = sum(Local40)
Nearest40_io = sum(Nearest40)
Random40_io = sum(Random40)
Proposed40_io = sum(Proposed40)

Local50_io = sum(Local50)
Nearest50_io = sum(Nearest50)
Random50_io = sum(Random50)
Proposed50_io = sum(Proposed50)

labels = ['10', '20', '30', '40', '50']
Local = []
Nearest = []
Random = []
Match = []
Proposed = []

Local.append(Local10_io)
Nearest.append(Nearest10_io)
Random.append(Random10_io)
Proposed.append(Proposed10_io)

Local.append(Local20_io)
Nearest.append(Nearest20_io)
Random.append(Random20_io)
Proposed.append(Proposed20_io)

Local.append(Local30_io)
Nearest.append(Nearest30_io)
Random.append(Random30_io)
Proposed.append(Proposed30_io)

Local.append(Local40_io)
Nearest.append(Nearest40_io)
Random.append(Random40_io)
Proposed.append(Proposed40_io)

Local.append(Local50_io)
Nearest.append(Nearest50_io)
Random.append(Random50_io)
Proposed.append(Proposed50_io)

x = np.arange(len(labels))  # the label locations
width = 0.1  # the width of the bars

# 填充颜色
colors = ['#706fd3', '#34ace0', '#33d9b2', '#ffb142', '#c56cf0','#f78fb3', '#ff4d4d']

fig, ax = plt.subplots()
Local = ax.bar(x - 2*width, Local, width=width, label='Local', edgecolor='black', color=colors[0], linewidth=.8, hatch='//')
Nearest = ax.bar(x - width, Nearest, width=width, label='Nearest', edgecolor='black', color=colors[1], linewidth=.8, hatch='//')
Random = ax.bar(x, Random, width=width, label='Random', edgecolor='black', color=colors[2], linewidth=.8, hatch='//')
Match = ax.bar(x + width, Match, width=width, label='Match', edgecolor='black', color=colors[3], linewidth=.8, hatch='//')
Proposed = ax.bar(x + 2*width, Proposed, width=width, label='Proposed', edgecolor='black', color=colors[4], linewidth=.8, hatch='//')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Number of devices')
ax.set_ylabel('System throughput (MB)')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
plt.savefig('./IOTotalStackedBar.pdf', format='pdf')
plt.show()
