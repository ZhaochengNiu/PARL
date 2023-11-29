
class MobileDevice:
    def __init__(self, nid, queue, queue_max, frequency, ncp_type, x, y) -> None:
        super().__init__()
        self.id = nid
        self.queue = queue
        self.queue_max = queue_max
        self.frequency = frequency
        self.origin_frequency = frequency
        self.ncp_type = ncp_type
        self.cal_power_cost = 1 * (10 ** -5)
        # 传输功率 1W 30 dbm
        self.trans_power = 1
        self.trans_cost_cpu_frequency = 0
        self.x = x
        self.y = y

    def push(self, size):
        self.queue = self.queue + size
        pass

    def poll(self, size):
        if size < self.queue:
            self.queue = self.queue - size
        else:
            size = self.queue
            self.queue = 0
        return size

    def queue_length(self):
        return self.queue
