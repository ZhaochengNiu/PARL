
class Config:

    def __init__(self) -> None:
        super().__init__()

        self.total_number_of_edges = 4

        self.total_number_of_devices = 5

        self.times = 1000

        self.time_slot_length = 0.1

        self.device_cpu_frequency = 10 ** 9

        self.edge_cpu_frequency = 10 ** 10

        self.uav_cpu_frequency = 5 * 10 ** 9

        self.SWITCHED_CAPACITANCE = 10 ** -27

        # [0.2, 5] MD (1 MB = 1024 KB = 1,048,576 Bytes = 8,388,608 bits)
        self.task_size = [1677721.6, 41943040]

        # 100 cycles/bit
        self.task_cpu_frequency_demand = 100

        # -100 dBm = 10 ** -13 W
        self.GAUSSIAN_WHITE_NOISE_POWER = 10 ** -13

        # 1 MHz
        self.RSU_BANDWIDTH = 1000000

        # 2MHz
        self.UAV_BANDWIDTH = 2000000

        # -50db
        self.CHANNEL_GAIN = 10 ** -5

        self.RSU_PATH_LOSS_EXPONENT = 4

        self.MIN_X_LOCATION = 0

        self.MAX_X_LOCATION = 1000

        self.MIN_Y_LOCATION = 0

        self.MAX_Y_LOCATION = 1000

        self.UAV_MOVE_DISTANCE = 10

        self.RSU_location = [[250, 250], [750, 250], [250, 750], [750, 750]]

        self.local_execute = 0

        self.offloading_to_rsu = 1

        self.experimental_algorithm = 1


