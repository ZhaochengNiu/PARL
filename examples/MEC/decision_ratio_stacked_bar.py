import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_theme(context='notebook', style='whitegrid', palette='deep', font='sans-serif', font_scale=1.0, color_codes=True, rc=None)

Local10 = np.loadtxt('./result/LocalRatioInEachSlotFilePath10.cache')
Edge10 = np.loadtxt('./result/EdgeRatioInEachSlotFilePath10.cache')
D2D10 = np.loadtxt('./result/D2DRatioInEachSlotFilePath10.cache')

Local20 = np.loadtxt('./result/LocalRatioInEachSlotFilePath20.cache')
Edge20 = np.loadtxt('./result/EdgeRatioInEachSlotFilePath20.cache')
D2D20 = np.loadtxt('./result/D2DRatioInEachSlotFilePath20.cache')

Local30 = np.loadtxt('./result/LocalRatioInEachSlotFilePath30.cache')
Edge30 = np.loadtxt('./result/EdgeRatioInEachSlotFilePath30.cache')
D2D30 = np.loadtxt('./result/D2DRatioInEachSlotFilePath30.cache')

Local40 = np.loadtxt('./result/LocalRatioInEachSlotFilePath40.cache')
Edge40 = np.loadtxt('./result/EdgeRatioInEachSlotFilePath40.cache')
D2D40= np.loadtxt('./result/D2DRatioInEachSlotFilePath40.cache')

Local50 = np.loadtxt('./result/LocalRatioInEachSlotFilePath50.cache')
Edge50 = np.loadtxt('./result/EdgeRatioInEachSlotFilePath50.cache')
D2D50= np.loadtxt('./result/D2DRatioInEachSlotFilePath50.cache')

Local10_temp = sum(Local10)/1000
Edge10_temp = sum(Edge10)/1000
D2D10_temp = sum(D2D10)/1000

Local20_temp = sum(Local20)/1000
Edge20_temp = sum(Edge20)/1000
D2D20_temp = sum(D2D20)/1000

Local30_temp = sum(Local30)/1000
Edge30_temp = sum(Edge30)/1000
D2D30_temp = sum(D2D30)/1000

Local40_temp = sum(Local40)/1000
Edge40_temp = sum(Edge40)/1000
D2D40_temp = sum(D2D40)/1000

Local50_temp = sum(Local50)/1000
Edge50_temp = sum(Edge50)/1000
D2D50_temp = sum(D2D50)/1000

labels = ['10', '20', '30', '40', '50']
Local = []
Edge = []
D2D = []

Local.append(Local10_temp)
Edge.append(Edge10_temp)
D2D.append(D2D10_temp)

Local.append(Local20_temp)
Edge.append(Edge20_temp)
D2D.append(D2D20_temp)

Local.append(Local30_temp)
Edge.append(Edge30_temp)
D2D.append(D2D30_temp)

Local.append(Local40_temp)
Edge.append(Edge40_temp)
D2D.append(D2D40_temp)

Local.append(Local50_temp)
Edge.append(Edge50_temp)
D2D.append(D2D50_temp)

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

# 填充颜色
colors = ['#706fd3', '#34ace0', '#33d9b2', '#ffb142', '#c56cf0','#f78fb3', '#ff4d4d']

fig, ax = plt.subplots()
Local = ax.bar(x - width/2, Local, width=width/2, label='Local', edgecolor='black', color=colors[0], linewidth=.8, hatch='//')
Edge = ax.bar(x, Edge, width=width/2, label='Edge', edgecolor='black', color=colors[2], linewidth=.8, hatch='//')
D2D = ax.bar(x + width/2, D2D, width=width/2, label='U2U', edgecolor='black', color=colors[5], linewidth=.8, hatch='//')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Number of devices')
ax.set_ylabel('Decision rate (%)')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
plt.savefig('./DecisionRatioStackedBar.pdf', format='pdf')
plt.show()
