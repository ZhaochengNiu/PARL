import numpy as np
import matplotlib.pyplot as plt


Random = np.loadtxt('./result/RandomTotalCostInEachSlotFilePath10.cache')
Proposed = np.loadtxt('./result/ProposedTotalCostInEachSlotFilePath10.cache')


Random_energy = sum(Random)
print(Random_energy)
Proposed_energy = sum(Proposed)

x_data = ['Random', 'proposed']
y_data = [Random_energy, Proposed_energy]

colors = ['#9999FF', '#58C9B9']

# hatch='//'
# hatch='xxx'

plt.figure(figsize=(5, 5))

for i in range(len(x_data)):
    plt.bar(x_data[i], y_data[i], edgecolor='black', color=colors[i], width=0.6, linewidth=.8, hatch='//')

# edgecolor：柱子边缘的颜色。颜色值或颜色值序列。
# linewidth：柱子边缘宽度。浮点数或类数组。如果为0，不绘制柱子边缘。
# width：柱子的宽度。浮点数或类数组结构。默认值为0.8。

plt.title('Total energy cost of different algorithm')

plt.xlabel('Algorithms')

plt.ylabel('System energy cost')

plt.savefig('./EnergyCostTotal10.pdf', format='pdf')

plt.show()

