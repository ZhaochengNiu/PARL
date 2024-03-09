import random
import gym
import numpy as np
import math
from config import Config
from mobiledevice import MobileDevice
from edge import Edge
from task import Task

target_reward = 8000

count_of_node_id = -1
count_of_edge_id = -1
config = Config()
devices = []
edges= []
uav_trans_rates = np.zeros((config.total_number_of_devices, config.total_number_of_edges), dtype=float)

def init_uav_position():
    x = round(random.uniform(config.MIN_X_LOCATION, config.MAX_X_LOCATION))
    y = round(random.uniform(config.MIN_Y_LOCATION, config.MAX_Y_LOCATION))
    return x, y

def uav_move_3d(device):
    angle = random.random() * math.pi * 2
    device.x = device.x + random.uniform(0,1) * math.ceil(math.cos(angle)) * config.UAV_MOVE_DISTANCE
    device.y = device.y + random.uniform(0,1) * math.ceil(math.sin(angle)) * config.UAV_MOVE_DISTANCE

    if device.x > config.MAX_X_LOCATION:
        device.x = device.x - config.MAX_X_LOCATION
    elif device.x < config.MIN_X_LOCATION:
        device.x = device.x + config.MAX_X_LOCATION
    elif device.y > config.MAX_Y_LOCATION:
        device.y = device.y - config.MAX_Y_LOCATION
    elif device.y < config.MIN_Y_LOCATION:
        device.y = device.y + config.MAX_Y_LOCATION


def init_devices():
    global devices

    for i in range(0, config.total_number_of_devices):
        frequency = config.device_cpu_frequency
        ncp_type = 'DEV'
        empty_queue = 0
        nid = i
        x, y = init_uav_position()
        device = MobileDevice(nid, empty_queue, 2** 20, frequency, ncp_type, x, y)
        devices.append(device)
    return devices


def init_device_queues(n_device):
    for device in n_device:
        device.queue = 0


def init_edge_queues(n_edge):
    for edge in n_edge:
        edge.queue = 0


def get_uav_upload_rate(device, edge):
    x1 = device.x
    y1 = device.y
    x2 = edge.x
    y2 = edge.y
    distance = math.sqrt(math.pow((x2-x1), 2) + math.pow((y2-y1), 2))
    distance2 = math.pow(distance, 2)
    power = device.trans_power
    noise2 = config.GAUSSIAN_WHITE_NOISE_POWER
    gain = config.CHANNEL_GAIN
    snr = (gain * power) / (noise2 * distance2)
    bandwidth = config.UAV_BANDWIDTH
    rate = bandwidth * (math.log(1 + snr, 2))
    return rate


def get_the_channel_gain(device, edge):
    x1 = device.x
    y1 = device.y
    x2 = edge.x
    y2 = edge.y
    distance= math.sqrt(math.pow((x2-x1), 2) + math.pow((y2-y1),2))
    distance2 = math.pow(distance,2)
    gain = config.CHANNEL_GAIN
    channel_gain = gain /distance2
    return channel_gain


def get_the_distance(device, edge):
    x1 = device.x
    y1 = device.y
    x2 = edge.x
    y2 = edge.y
    distance = math.sqrt(math.pow((x2-x1),2) + math.pow((y2-y1), 2))
    return distance


def update_system(n_device, n_edge):
    global uav_trans_rates

    for i in range(0, config.total_number_of_devices):
        uav_move_3d((n_device[i]))

    for i in range(0, config.total_number_of_devices):
        for j in range(0, config.total_number_of_edges):
            if n_device[i].ncp_type == 'DEV' and n_edge[j].ncp_type == 'EDGE':
                distance = get_the_distance(n_device[i], n_edge[j])
                upload_rate = get_uav_upload_rate(n_device[i], n_edge[j])
                uav_trans_rates[i][j] = upload_rate


def init_trans_states(n_device, n_edge):
    global uav_trans_rates

    for i in range(0, config.total_number_of_devices):
        for j in range(0, config.total_number_of_edges):
            if n_device[i].ncp_type == 'DEV' and n_edge[j].ncp_type == 'EDGE':
                upload_rate = get_uav_upload_rate(n_device[i], n_edge[j])
                uav_trans_rates[i][j] = upload_rate


def init_edges():
    global edges
    for i in range(0, config.total_number_of_edges):
        frequency = config.edge_cpu_frequency
        ncp_type = 'EDGE'
        nid = i
        edge = Edge(nid, frequency, ncp_type, config.RSU_location[i][0], config.RSU_location[i][1])
        edges.append(edge)
    return edges


def init_tasks():
    tasks= []
    for i in range(0, config.total_number_of_devices):
        data_size = np.random.uniform(config.task_size[0], config.task_size[1])
        cpu_frequency_demand = config.task_cpu_frequency_demand * data_size
        task = Task(data_size, cpu_frequency_demand)
        tasks.append(task)
    return tasks


class DispersedNetworkEnv(gym.Env):

    def __init__(self) -> None:
        super().__init__()

        self.total_number_of_devices = config.total_number_of_devices

        self.total_number_of_edges = config.total_number_of_edges

        self.state_dim = self.total_number_of_edges * 2 + 2

        self.action_space = [gym.spaces.Discrete(self.total_number_of_edges + 1) for i in range(0, self.total_number_of_devices)]

        self.act_shape_n = [self.total_number_of_edges + 1 for i in range(0, self.total_number_of_devices)]

        self.n = config.total_number_of_devices

        self.state_n = None
        # Box(low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
        self.observation_space = [gym.spaces.Box(-math.inf, math.inf, shape=(self.total_number_of_edges*2 + 2,1)) for i in range(0, self.total_number_of_devices)]
        self.obs_shape_n = [self.state_dim for i in range(0, self.total_number_of_devices)]

        self.env_name = 'multi_agent_mec'

        self.if_discrete = True

        self.action_dim = self.total_number_of_edges + 1

        self.target_reward = target_reward

        self.devices = init_devices()

        self.edges = init_edges()

        self.tasks = init_tasks()

        self.count = 0

    def step(self, action_n):
        reward_n = []
        done_n = []
        self.state_n = []

        for i in range(0, self.total_number_of_devices):
            total_cost = 0.0
            data_size = self.tasks[i].data_size
            cpu_frequency_demand = self.tasks[i].cpu_frequency_demand * data_size
            if action_n[i].argmax() == 0:
                devices[i].push(cpu_frequency_demand)
                local_queue_length = devices[i].queue_length()
                task_complete_time = local_queue_length / devices[i].frequency
                local_computing_power = config.SWITCHED_CAPACITANCE * math.pow(devices[i].frequency, 3)
                devices[i].poll(devices[i].frequency * config.time_slot_length)
                task_computing_time = cpu_frequency_demand / devices[i].frequency
                local_computing_cost = task_computing_time * local_computing_power
                # print('local_computing_cost', local_computing_cost)
                # print('task_computing_time', task_computing_time)
                total_cost += (local_computing_cost + task_computing_time)
                reward = - total_cost
                reward_n.append(reward)
            else:
                destination_edge = action_n[i].argmax()-1
                number_of_requires = 0
                for j in range(config.total_number_of_devices):
                    if action_n[j].argmax() == action_n[i].argmax():
                        number_of_requires += 1
                transmit_time = data_size / (uav_trans_rates[i][destination_edge] / number_of_requires)
                transmit_cost = transmit_time * devices[i].trans_power
                edge_computing_frequency = edges[destination_edge].frequency / number_of_requires
                compute_time = cpu_frequency_demand / edge_computing_frequency
                devices[i].poll(devices[i].frequency * config.time_slot_length)
                # print('transmit_cost', transmit_cost)
                # print('task_computing_time', transmit_time + compute_time)
                total_cost += (transmit_time + compute_time + transmit_cost)
                reward = - total_cost
                reward_n.append(reward)

        self.tasks = init_tasks()

        for i in range(0, self.total_number_of_devices):
            state = np.zeros(self.total_number_of_edges * 2 + 2, dtype=np.float32)
            for j in range(0, self.total_number_of_edges):
                state[j] = self.edges[j].queue_length()
                channel_gain = get_the_channel_gain(self.devices[i], self.edges[j])
                state[j + self.total_number_of_edges] = channel_gain
            state[self.total_number_of_edges * 2] = self.tasks[i].data_size
            state[self.total_number_of_edges * 2 + 1] = self.tasks[i].cpu_frequency_demand
            # 状态复位
            self.state_n.append(state)

        # 计算状态的reward

        self.count += 1
        done = self.count > 999

        for i in range(0, config.total_number_of_devices):
            done_n.append(done)

        update_system(self.devices, self.edges)

        return self.state_n, reward_n, done_n, {}


    def reset(self):
        self.state_n = []
        init_device_queues(self.devices)
        init_edge_queues(self.edges)
        self.tasks = init_tasks()
        for i in range(0, self.total_number_of_devices):
            state_empty = np.zeros(self.total_number_of_edges*2 + 2, dtype=np.float32)
            for j in range(0, self.total_number_of_edges):
                state_empty[j] = self.edges[j].queue_length()
                channel_gain = get_the_channel_gain(self.devices[i], self.edges[j])
                state_empty[j + self.total_number_of_edges] = channel_gain
            state_empty[self.total_number_of_edges*2] = self.tasks[i].data_size
            state_empty[self.total_number_of_edges*2+1] = self.tasks[i].cpu_frequency_demand
            # 状态复位
            self.state_n.append(state_empty)
        self.count = 0
        init_trans_states(self.devices, self.edges)
        return self.state_n








