import math
import numpy as np
import random

# device_cpu_frequency = 10 * (10 ** 9)
# print(device_cpu_frequency)

# queue = []
# queue.insert(0, 5)
# print(queue)
# queue.insert(1, 6)
# print(math.exp(-0.8))

# test = [[250, 250], [750, 250], [250, 750], [750, 750]]
# for i in range(0, 4):
#     print(test[i][0])
#     print(test[i][1])

# x = 1221
# y = x % 10
# print(y)
# print(x)
# x = x / 10
# print(x)
# z = x % 10
# print(z)

# for i in range (0, 10):
#     n = random.uniform(0, 0.5)
#     print(n)
# list1 = np.random.uniform(low=0, high=5, size=5)
# print(list1)
# list1[0] = 10
# print(list1)

# print(list1[1])
# list = []
# for i in range(0, 5):
#     list.append(i)
# print(list.mean(axis=0))

# sla_in_each_slot = [0 for i in range(5)]
# print(sla_in_each_slot)


# edge_locations = [[250, 250], [750, 250], [250, 750], [750, 750], [500, 500]]
# print(len(edge_locations))
# for i in range(0, 100):
#     print(np.random.randint(-1, 4))
# for k in range(1, 6 + 1):
#     print(k)

import numpy as np
import matplotlib.pyplot as plt

# # 生成 x 值的范围
# x = np.linspace(-5, 5, 100)
#
# # 计算对应的 y 值，即反正切函数的值
# y = np.arctan(x)
#
# # 绘制曲线图
# plt.plot(x, y)
# plt.title('Plot of arctan(x)')
# plt.xlabel('x')
# plt.ylabel('arctan(x)')
# plt.grid(True)
# plt.show()

x = np.random.uniform(low=0, high=6, size=3)
y = np.random.uniform(low=0, high=6, size=3)
print(x)
print(y)
y[0]=x[0]
x[0]=0
print(y[0])