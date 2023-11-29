class Task:
    def __init__(self, data_size, cpu_frequency_demand) -> None:
        super().__init__()
        self.data_size = data_size
        self.cpu_frequency_demand = cpu_frequency_demand