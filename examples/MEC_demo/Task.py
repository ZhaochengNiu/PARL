class Task:
    def __init__(self, data_size, cpu_frequency_demand) -> None:
        super().__init__()
        self.data_size = data_size
        self.cpu_frequency_demand = cpu_frequency_demand

        # resolution ｜  Accuracy
        #    100     ｜   0.176
        #    200     ｜   0.570
        #    300     ｜   0.775
        #    400     ｜   0.882
        #    500     ｜   0.939
        #    600     ｜   0.968
