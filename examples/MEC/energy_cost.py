import numpy as np
import matplotlib.pyplot as plt

Local = np.loadtxt('./result/LocalEnergyCostInEachSlotFilePath10.cache')
Nearest = np.loadtxt('./result/NearestEnergyCostInEachSlotFilePath10.cache')
Random = np.loadtxt('./result/RandomEnergyCostInEachSlotFilePath10.cache')
Match = np.loadtxt('./result/MatchEnergyCostInEachSlotFilePath10.cache')


Local_energy = sum(Local)
Nearest_energy = sum(Nearest)
Random_energy = sum(Random)
Match_energy = sum(Match)

x_data = ['Local', 'Nearest', 'Random', 'Match']
y_data = [Local_energy, Nearest_energy, Random_energy, Match_energy]

colors = ['#9999FF', '#58C9B9', '#CC33CC', '#D1B6E1', '#99FF99', '#FF6666']

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

