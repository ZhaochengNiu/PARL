# 10个用户
import gym
import math

class Config:
    '''配置类'''

    def __init__(self) -> None:
        super().__init__()

        # 边的数量
        self.total_number_of_edges = 4

        # 设备总数量
        self.total_number_of_devices = 10

        self.total_number_of_resolutions = 6


        # 信道增益，队列状态
        self.state_dim = self.total_number_of_edges * 2 + 1

        self.action_dim = self.total_number_of_edges * self.total_number_of_resolutions

        self.action_space = [gym.spaces.Discrete(self.total_number_of_edges * self.total_number_of_resolutions) for _ in range(0, self.total_number_of_devices)]

        self.act_shape_n = [self.action_dim for _ in range(0, self.total_number_of_devices)]

        # Box(low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
        self.observation_space = [gym.spaces.Box(-math.inf, math.inf, shape=(self.total_number_of_edges*2 + 1,1)) for _ in range(0, self.total_number_of_devices)]

        self.obs_shape_n = [self.state_dim for _ in range(0, self.total_number_of_devices)]

        self.model_dir ='./model_10'

        # 迭代次数
        self.times = 1000

        # 时隙长度 100ms
        self.time_slot_length = 0.1

        self.bits_per_pixel = 24

        # 1 GHZ  cycles/s = 1
        self.device_cpu_frequency = 3 * (10 ** 9)

        # 边缘服务器的算力
        self.edge_cpu_frequency = 8 * (10 ** 9)

        self.SWITCHED_CAPACITANCE = 10 ** -28

        # 100 cycles/bit
        self.task_cpu_frequency_demand = 100

        # -100 dBm = 10 ** -13 W
        self.GAUSSIAN_WHITE_NOISE_POWER = 10 ** -13

        # 300MHZ
        self.OFFLOAD_BANDWIDTH = 500000000

        # self.EDGE_CHANNEL_GAIN = 10 ** -4

        # - 50dB
        self.EDGE_CHANNEL_GAIN = 10 ** -4
        # self.EDGE_CHANNEL_GAIN = 0.1

        self.MIN_X_LOCATION = 0

        self.MAX_X_LOCATION = 500

        self.MIN_Y_LOCATION = 0

        self.MAX_Y_LOCATION = 500

        # 移动速度 10m/s
        self.MOVE_DISTANCE = 10

        self.edge_locations = [[125, 125], [125, 375], [375, 125], [375, 375]]

        # 延迟权重
        self.latency_weight = 1 / 3

        # 能量权重
        self.energy_weight = 1 / 3

        # 精度权重
        self.error_weight = 1 / 3

        # sca 参数设置
        self.pop_size = 60
        self.a = 2 # 感知概率
        self.max_iter = 30  # max iter

        # self.algorithm = 'local_algorithm'
        # self.algorithm = 'nearest_algorithm'
        # self.algorithm = 'random_algorithm'
        # self.algorithm = 'match_algorithm'
        self.algorithm = 'proposed_algorithm'
        # 'local_algorithm'  'nearest_algorithm' 'random_algorithm' 'proposed_algorithm'

        # 缓存设置
        self.cache = True
        # self.cache = False
        # True False
        self.devices_cache_file_path = './cache/Devices10.cache'
        self.edges_cache_file_path = './cache/Edges10.cache'

        self.local_ave_queue_length_in_each_slot_file_path = './result/LocalAveQueueLengthInEachSlotFilePath10.cache'
        self.nearest_ave_queue_length_in_each_slot_file_path = './result/NearestAveQueueLengthInEachSlotFilePath10.cache'
        self.random_ave_queue_length_in_each_slot_file_path = './result/RandomAveQueueLengthInEachSlotFilePath10.cache'
        self.match_ave_queue_length_in_each_slot_file_path = './result/MatchAveQueueLengthInEachSlotFilePath10.cache'
        self.proposed_ave_queue_length_in_each_slot_file_path = './result/ProposedAveQueueLengthInEachSlotFilePath10.cache'

        self.local_ave_execute_latency_in_each_slot_file_path = './result/LocalAveExecuteLatencyInEachSlotFilePath10.cache'
        self.nearest_ave_execute_latency_in_each_slot_file_path = './result/NearestAveExecuteLatencyInEachSlotFilePath10.cache'
        self.random_ave_execute_latency_in_each_slot_file_path = './result/RandomAveExecuteLatencyInEachSlotFilePath10.cache'
        self.match_ave_execute_latency_in_each_slot_file_path = './result/MatchAveExecuteLatencyInEachSlotFilePath10.cache'
        self.proposed_ave_execute_latency_in_each_slot_file_path = './result/ProposedAveExecuteLatencyInEachSlotFilePath10.cache'

        self.local_energy_cost_in_each_slot_file_path = './result/LocalEnergyCostInEachSlotFilePath10.cache'
        self.nearest_energy_cost_in_each_slot_file_path = './result/NearestEnergyCostInEachSlotFilePath10.cache'
        self.random_energy_cost_in_each_slot_file_path = './result/RandomEnergyCostInEachSlotFilePath10.cache'
        self.match_energy_cost_in_each_slot_file_path = './result/MatchEnergyCostInEachSlotFilePath10.cache'
        self.proposed_energy_cost_in_each_slot_file_path = './result/ProposedEnergyCostInEachSlotFilePath10.cache'

        self.local_latency_cost_in_each_slot_file_path = './result/LocalLatencyCostInEachSlotFilePath10.cache'
        self.nearest_latency_cost_in_each_slot_file_path = './result/NearestLatencyCostInEachSlotFilePath10.cache'
        self.random_latency_cost_in_each_slot_file_path = './result/RandomLatencyCostInEachSlotFilePath10.cache'
        self.match_latency_cost_in_each_slot_file_path = './result/MatchLatencyCostInEachSlotFilePath10.cache'
        self.proposed_latency_cost_in_each_slot_file_path = './result/ProposedLatencyCostInEachSlotFilePath10.cache'

        self.local_error_cost_in_each_slot_file_path = './result/LocalErrorCostInEachSlotFilePath10.cache'
        self.nearest_error_cost_in_each_slot_file_path = './result/NearestErrorCostInEachSlotFilePath10.cache'
        self.random_error_cost_in_each_slot_file_path = './result/RandomErrorCostInEachSlotFilePath10.cache'
        self.match_error_cost_in_each_slot_file_path = './result/MatchErrorCostInEachSlotFilePath10.cache'
        self.proposed_error_cost_in_each_slot_file_path = './result/ProposedErrorCostInEachSlotFilePath10.cache'

        self.local_total_cost_in_each_slot_file_path = './result/LocalTotalCostInEachSlotFilePath10.cache'
        self.nearest_total_cost_in_each_slot_file_path = './result/NearestTotalCostInEachSlotFilePath10.cache'
        self.random_total_cost_in_each_slot_file_path = './result/RandomTotalCostInEachSlotFilePath10.cache'
        self.match_total_cost_in_each_slot_file_path = './result/MatchTotalCostInEachSlotFilePath10.cache'
        self.proposed_total_cost_in_each_slot_file_path = './result/ProposedTotalCostInEachSlotFilePath10.cache'

        self.local_io_in_each_slot_file_path = './result/LocalIOInEachSlotFilePath10.cache'
        self.nearest_io_in_each_slot_file_path = './result/NearestIOInEachSlotFilePath10.cache'
        self.random_io_in_each_slot_file_path = './result/RandomIOInEachSlotFilePath10.cache'
        self.match_io_in_each_slot_file_path = './result/MatchIOInEachSlotFilePath10.cache'
        self.proposed_io_in_each_slot_file_path = './result/ProposedIOInEachSlotFilePath10.cache'

        self.local_ratio_in_each_slot_file_path = './result/LocalRatioInEachSlotFilePath10.cache'
        self.edge_ratio_in_each_slot_file_path = './result/EdgeRatioInEachSlotFilePath10.cache'

========================================================================================================================

# 10个用户
import gym
import math

class Config:
    '''配置类'''

    def __init__(self) -> None:
        super().__init__()

        # 边的数量
        self.total_number_of_edges = 4

        # 设备总数量
        self.total_number_of_devices = 15

        self.total_number_of_resolutions = 6


        # 信道增益，队列状态
        self.state_dim = self.total_number_of_edges * 2 + 1

        self.action_dim = self.total_number_of_edges * self.total_number_of_resolutions

        self.action_space = [gym.spaces.Discrete(self.total_number_of_edges * self.total_number_of_resolutions) for _ in range(0, self.total_number_of_devices)]

        self.act_shape_n = [self.action_dim for _ in range(0, self.total_number_of_devices)]

        # Box(low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
        self.observation_space = [gym.spaces.Box(-math.inf, math.inf, shape=(self.total_number_of_edges*2 + 1,1)) for _ in range(0, self.total_number_of_devices)]

        self.obs_shape_n = [self.state_dim for _ in range(0, self.total_number_of_devices)]

        self.model_dir ='./model_15'

        # 迭代次数
        self.times = 25

        # 时隙长度 100ms
        self.time_slot_length = 0.1

        self.bits_per_pixel = 24

        # 1 GHZ  cycles/s = 1
        self.device_cpu_frequency = 3 * (10 ** 9)

        # 边缘服务器的算力
        self.edge_cpu_frequency = 8 * (10 ** 9)

        self.SWITCHED_CAPACITANCE = 10 ** -28

        # 100 cycles/bit
        self.task_cpu_frequency_demand = 100

        # -100 dBm = 10 ** -13 W
        self.GAUSSIAN_WHITE_NOISE_POWER = 10 ** -13

        # 300MHZ
        self.OFFLOAD_BANDWIDTH = 500000000

        # self.EDGE_CHANNEL_GAIN = 10 ** -4

        # - 50dB
        self.EDGE_CHANNEL_GAIN = 10 ** -4
        # self.EDGE_CHANNEL_GAIN = 0.1

        self.MIN_X_LOCATION = 0

        self.MAX_X_LOCATION = 500

        self.MIN_Y_LOCATION = 0

        self.MAX_Y_LOCATION = 500

        # 移动速度 10m/s
        self.MOVE_DISTANCE = 10

        self.edge_locations = [[125, 125], [125, 375], [375, 125], [375, 375]]

        # 延迟权重
        self.latency_weight = 1 / 3

        # 能量权重
        self.energy_weight = 1 / 3

        # 精度权重
        self.error_weight = 1 / 3

        # sca 参数设置
        self.pop_size = 60
        self.a = 2 # 感知概率
        self.max_iter = 30  # max iter

        # self.algorithm = 'local_algorithm'
        # self.algorithm = 'nearest_algorithm'
        self.algorithm = 'random_algorithm'
        # self.algorithm = 'match_algorithm'
        # self.algorithm = 'proposed_algorithm'
        # 'local_algorithm'  'nearest_algorithm' 'random_algorithm' 'proposed_algorithm'

        # 缓存设置
        self.cache = True
        # self.cache = False
        # True False
        self.devices_cache_file_path = './cache/Devices15.cache'
        self.edges_cache_file_path = './cache/Edges15.cache'

        self.local_ave_queue_length_in_each_slot_file_path = './result/LocalAveQueueLengthInEachSlotFilePath15.cache'
        self.nearest_ave_queue_length_in_each_slot_file_path = './result/NearestAveQueueLengthInEachSlotFilePath15.cache'
        self.random_ave_queue_length_in_each_slot_file_path = './result/RandomAveQueueLengthInEachSlotFilePath15.cache'
        self.match_ave_queue_length_in_each_slot_file_path = './result/MatchAveQueueLengthInEachSlotFilePath15.cache'
        self.proposed_ave_queue_length_in_each_slot_file_path = './result/ProposedAveQueueLengthInEachSlotFilePath15.cache'

        self.local_ave_execute_latency_in_each_slot_file_path = './result/LocalAveExecuteLatencyInEachSlotFilePath15.cache'
        self.nearest_ave_execute_latency_in_each_slot_file_path = './result/NearestAveExecuteLatencyInEachSlotFilePath15.cache'
        self.random_ave_execute_latency_in_each_slot_file_path = './result/RandomAveExecuteLatencyInEachSlotFilePath15.cache'
        self.match_ave_execute_latency_in_each_slot_file_path = './result/MatchAveExecuteLatencyInEachSlotFilePath15.cache'
        self.proposed_ave_execute_latency_in_each_slot_file_path = './result/ProposedAveExecuteLatencyInEachSlotFilePath15.cache'

        self.local_energy_cost_in_each_slot_file_path = './result/LocalEnergyCostInEachSlotFilePath15.cache'
        self.nearest_energy_cost_in_each_slot_file_path = './result/NearestEnergyCostInEachSlotFilePath15.cache'
        self.random_energy_cost_in_each_slot_file_path = './result/RandomEnergyCostInEachSlotFilePath15.cache'
        self.match_energy_cost_in_each_slot_file_path = './result/MatchEnergyCostInEachSlotFilePath15.cache'
        self.proposed_energy_cost_in_each_slot_file_path = './result/ProposedEnergyCostInEachSlotFilePath15.cache'

        self.local_latency_cost_in_each_slot_file_path = './result/LocalLatencyCostInEachSlotFilePath15.cache'
        self.nearest_latency_cost_in_each_slot_file_path = './result/NearestLatencyCostInEachSlotFilePath15.cache'
        self.random_latency_cost_in_each_slot_file_path = './result/RandomLatencyCostInEachSlotFilePath15.cache'
        self.match_latency_cost_in_each_slot_file_path = './result/MatchLatencyCostInEachSlotFilePath15.cache'
        self.proposed_latency_cost_in_each_slot_file_path = './result/ProposedLatencyCostInEachSlotFilePath15.cache'

        self.local_error_cost_in_each_slot_file_path = './result/LocalErrorCostInEachSlotFilePath15.cache'
        self.nearest_error_cost_in_each_slot_file_path = './result/NearestErrorCostInEachSlotFilePath15.cache'
        self.random_error_cost_in_each_slot_file_path = './result/RandomErrorCostInEachSlotFilePath15.cache'
        self.match_error_cost_in_each_slot_file_path = './result/MatchErrorCostInEachSlotFilePath15.cache'
        self.proposed_error_cost_in_each_slot_file_path = './result/ProposedErrorCostInEachSlotFilePath15.cache'

        self.local_total_cost_in_each_slot_file_path = './result/LocalTotalCostInEachSlotFilePath15.cache'
        self.nearest_total_cost_in_each_slot_file_path = './result/NearestTotalCostInEachSlotFilePath15.cache'
        self.random_total_cost_in_each_slot_file_path = './result/RandomTotalCostInEachSlotFilePath15.cache'
        self.match_total_cost_in_each_slot_file_path = './result/MatchTotalCostInEachSlotFilePath15.cache'
        self.proposed_total_cost_in_each_slot_file_path = './result/ProposedTotalCostInEachSlotFilePath15.cache'

        self.local_io_in_each_slot_file_path = './result/LocalIOInEachSlotFilePath15.cache'
        self.nearest_io_in_each_slot_file_path = './result/NearestIOInEachSlotFilePath15.cache'
        self.random_io_in_each_slot_file_path = './result/RandomIOInEachSlotFilePath15.cache'
        self.match_io_in_each_slot_file_path = './result/MatchIOInEachSlotFilePath15.cache'
        self.proposed_io_in_each_slot_file_path = './result/ProposedIOInEachSlotFilePath15.cache'

        self.local_ratio_in_each_slot_file_path = './result/LocalRatioInEachSlotFilePath15.cache'
        self.edge_ratio_in_each_slot_file_path = './result/EdgeRatioInEachSlotFilePath15.cache'