class Edge:

    def __init__(self, nid, frequency, ncp_type, x, y) -> None:
        super().__init__()
        self.queue = 0
        self.id = nid   # 序号
        self.frequency = frequency   # 当前计算能力
        self.origin_frequency = frequency # 总计算能力
        self.ncp_type = ncp_type # 节点类型
        self.cal_power_cost = 1 * (10 ** -5) # 计算成本
        self.d2d_trans_power = 0.04  # 计算成本
        self.trans_cost_cpu_frequency = 0  # 传输功率
        self.x = x  # x 坐标
        self.y = y  # y 坐标

    def push(self, size):
        self.queue = self.queue + size

    def poll(self, size):
        if size < self.queue:
            self.queue = self.queue -size
        else:
            size = self.queue
            self.queue = 0
        return size

    def queue_length(self):
        return self.queue

