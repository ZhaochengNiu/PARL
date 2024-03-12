import random
import math


class MobileDevice:

    def __init__(self, nid, frequency, ncp_type, move_distance, total_number_of_edges,
                 min_x_location, max_x_location, min_y_location, max_y_location, generating_tasks_probability_of_device) -> None:
        super().__init__()
        self.id = nid       # 序号
        self.task_queue = 0
        self.frequency = frequency          # 当前计算能力
        self.ncp_type = ncp_type   # 节点类型
        # dBm 是功率单位，和 w 的换算如下：
        # 首先把功率换算成毫瓦(mW)，然后取对数，再乘以10。
        # 比如1W = 1000 mW = 10 * log(1000) = 10 * 3 = 30 dBm
        # 传输功率 5W 37dBm
        self.offload_trans_power = 5
        self.x = 0  # x 坐标
        self.y = 0  # y 坐标
        self.distance_BS = [0 for i in range(total_number_of_edges)]
        self.generating_tasks_probability_of_device = generating_tasks_probability_of_device
        self.MIN_X_LOCATION = min_x_location
        self.MAX_X_LOCATION = max_x_location
        self.MIN_Y_LOCATION = min_y_location
        self.MAX_Y_LOCATION = max_y_location
        self.MOVE_DISTANCE = move_distance

    def task_enqueue(self, cpu_frequency_demand):
        self.task_queue += cpu_frequency_demand

    def task_dequeue(self, cpu_frequency_demand):
        if self.task_queue >= cpu_frequency_demand:
            self.task_queue -= cpu_frequency_demand
        else:
            self.task_queue = 0.0

    def task_queue_length(self):
        return self.task_queue

    def init_position(self):
        self.x = round(random.uniform(self.MIN_X_LOCATION, self.MAX_X_LOCATION))
        self.y = round(random.uniform(self.MIN_Y_LOCATION, self.MAX_Y_LOCATION))
        return self.x, self.y

    def get_the_distance(self, total_number_of_edges, edges):
        x1 = self.x
        y1 = self.y
        for j in range(0, total_number_of_edges):
            x2 = edges[j].x
            y2 = edges[j].y
            distance = math.sqrt(math.pow((x2 - x1), 2) + math.pow((y2 - y1), 2))
            self.distance_BS[j] = distance

    def move(self):
        angle = random.random() * math.pi * 2
        self.x = self.x + math.ceil(random.uniform(0, 1) * math.cos(angle) * self.MOVE_DISTANCE)
        self.y = self.y + math.ceil(random.uniform(0, 1) * math.sin(angle) * self.MOVE_DISTANCE)
        if self.x > self.MAX_X_LOCATION:
            self.x = self.x - self.MAX_X_LOCATION
        if self.x < self.MIN_X_LOCATION:
            self.x = self.x + self.MAX_X_LOCATION
        if self.y > self.MAX_Y_LOCATION:
            self.y = self.y - self.MAX_Y_LOCATION
        if self.y < self.MIN_Y_LOCATION:
            self.y = self.y + self.MAX_Y_LOCATION


if __name__ == '__main__':
    print("test")
    # angle = random.random() * math.pi * 2
    # x = math.cos(angle)
    # y = math.sin(angle)
    # print(x)
    # print(y)
    # print(x**2+y**2)
    # distance = math.sqrt(math.pow((3 - 0), 2) + math.pow((4 - 0), 2))
    # print(distance)
    # task_queue = []
    # for i in range(0, 10):
    #     task_queue.insert(0, i)
    # print(task_queue)
    # for i in range(0, 5):
    #     print(task_queue.pop())
    # print(task_queue)
    # print(len(task_queue))
    # angle = random.random() * math.pi * 2
    # print(math.ceil(random.uniform(0, 1) * math.sin(angle) * 200))
