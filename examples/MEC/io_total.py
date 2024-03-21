import numpy as np
import matplotlib.pyplot as plt

Local = np.loadtxt('./LocalIOInEachSlotFilePath20.cache')
Nearest = np.loadtxt('./NearestIOInEachSlotFilePath20.cache')
Random = np.loadtxt('./RandomIOInEachSlotFilePath20.cache')
Proposed = np.loadtxt('./ProposedIOInEachSlotFilePath20.cache')


Local_io = sum(Local)
Nearest_io = sum(Nearest)
Random_io = sum(Random)
Proposed_io = sum(Proposed)

x_data = ['Local', 'Nearest', 'Random', 'Proposed']
y_data = [Local_io, Nearest_io, Random_io, Proposed_io]

colors = ['#9999FF', '#58C9B9', '#CC33CC', '#D1B6E1', '#99FF99', '#FF6666']

# hatch='//'
# hatch='xxx'

plt.figure(figsize=(5, 5))

for i in range(len(x_data)):
    plt.bar(x_data[i], y_data[i], edgecolor='black', color=colors[i], width=0.6, linewidth=.8, hatch='//')

# edgecolor：柱子边缘的颜色。颜色值或颜色值序列。
# linewidth：柱子边缘宽度。浮点数或类数组。如果为0，不绘制柱子边缘。
# width：柱子的宽度。浮点数或类数组结构。默认值为0.8。

plt.title('Total io of different algorithm')

plt.xlabel('Algorithms')

plt.ylabel('System throughput')

plt.savefig('./IOTotal.pdf', format='pdf')

plt.show()