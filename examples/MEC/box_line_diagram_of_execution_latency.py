import numpy as np
import matplotlib.pyplot as plt

Local = np.loadtxt('./result/LocalAveExecuteLatencyInEachSlotFilePath10.cache')
Nearest = np.loadtxt('./result/NearestAveExecuteLatencyInEachSlotFilePath10.cache')
Random = np.loadtxt('./result/RandomAveExecuteLatencyInEachSlotFilePath10.cache')
Match = np.loadtxt('./result/MatchAveExecuteLatencyInEachSlotFilePath10.cache')
Proposed = np.loadtxt('./result/ProposedAveExecuteLatencyInEachSlotFilePath10.cache')

all_data = [Local, Nearest, Random, Match, Proposed]
labels = ['Local', 'Nearest', 'Random', 'Match', 'Proposed']


# fig, axes = plt.subplots(figsize=(5, 5))

fig, axes = plt.subplots()

# 箱形图
bplot = axes.boxplot(all_data,
                         notch = False,
                         vert = True,  #是否需要将箱线图垂直摆放，默认垂直摆放；
                         patch_artist = True,   # 是否填充箱体的颜色；
                         showfliers = False,   # 是否显示异常值
                         showmeans = True,    # 是否显示均值，默认不显示；
                         meanline = False,   # meanline参数（bool值，是否用线的形式表示均值，默认值False用点来表示）
                         widths = [0.5, 0.5, 0.5, 0.5, 0.5],  # widths参数（float值，指定箱线图的宽度，默认值：0.5）
                         medianprops={'linestyle':'-','linewidth':1,'color':'k'},  # medianprops参数（设置中位数的属性，如线的类型、粗细等）
                         meanprops={'marker':'*', "markerfacecolor":'k', "markeredgecolor":'k',"markersize": 8},  # meanprops参数（设置均值的属性）
                         labels = labels)   # 为箱线图添加标签，类似于图例的作用；


# '-'       solid line style
# '--'      dashed line style
# '-.'      dash-dot line style
# ':'       dotted line style

# '.'       point marker
# ','       pixel marker
# 'o'       circle marker
# 'v'       triangle_down marker
# '^'       triangle_up marker
# '<'       triangle_left marker
# '>'       triangle_right marker
# '1'       tri_down marker
# '2'       tri_up marker
# '3'       tri_left marker
# '4'       tri_right marker
# 's'       square marker
# 'p'       pentagon marker
# '*'       star marker
# 'h'       hexagon1 marker
# 'H'       hexagon2 marker
# '+'       plus marker
# 'x'       x marker
# 'D'       diamond marker
# 'd'       thin_diamond marker
# '|'       vline marker
# '_'       hline marker

# 填充颜色
# colors = ['#706fd3', '#34ace0', '#33d9b2', '#ffb142', '#c56cf0','#f78fb3', '#ff4d4d']

# 填充颜色
colors = ['#706fd3', '#34ace0', '#33d9b2', '#ffb142', '#c56cf0']

for patch, color in zip(bplot['boxes'], colors):
    patch.set_facecolor(color)

# 添加水平网格线

axes.yaxis.grid(True)
axes.set_xlabel('Algorithms')
axes.set_ylabel('Execution latency (s)')

plt.savefig('./BoxLineDiagramOfExecutionLatency50.pdf', format='pdf')

plt.show()




