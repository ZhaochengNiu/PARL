import random
from abc import ABC

import gym
import numpy as np
import math
from Config import Config
from MobileDevice import MobileDevice
from Edge import Edge
from Task import Task
from trans_rate import get_upload_gain, get_upload_rate

target_reward = 8000

count_of_node_id = -1
count_of_edge_id = -1
config = Config()
devices = []
edges = []


def init_devices():
    global devices

    for i in range(0, config.total_number_of_devices):
        frequency = config.activate_device_cpu_frequency
        ncp_type = 'DEV'
        nid = i
        device = MobileDevice(nid=nid, frequency=frequency,
                              ncp_type=ncp_type, move_distance=config.MOVE_DISTANCE,
                              total_number_of_edges=config.total_number_of_edges,
                              min_x_location=config.MIN_X_LOCATION, max_x_location=config.MAX_X_LOCATION,
                              min_y_location=config.MIN_Y_LOCATION, max_y_location=config.MAX_Y_LOCATION)
        device.init_position()
        device.get_the_distance(total_number_of_edges=config.total_number_of_edges, edges=edges)
        devices.append(device)
    return devices


def init_edges():
    global edges

    for i in range(0, config.total_number_of_edges):
        frequency = config.edge_cpu_frequency
        ncp_type = 'EDGE'
        nid = i
        edge = Edge(nid=nid, frequency=frequency, ncp_type=ncp_type, x=config.edge_locations[i][0],
                    y=config.edge_locations[i][1], total_number_of_devices=config.total_number_of_devices)
        edges.append(edge)
    return edges


def init_device_queues(device_vector):
    for device in device_vector:
        device.queue = 0


def init_edge_queues(edge_vector):
    for edge in edge_vector:
        edge.queue = 0


def edge_compute(edge):
    # 边缘计算
    slot_computing_data_size = config.time_slot_length * edge.frequency
    task_queue_length = edge.task_queue_length()
    if slot_computing_data_size <= task_queue_length:
        edge.task_dequeue(cpu_frequency_demand=slot_computing_data_size)
        edge_computing_size = slot_computing_data_size
    elif slot_computing_data_size > task_queue_length:
        edge.task_dequeue(cpu_frequency_demand=slot_computing_data_size)
        edge_computing_size = task_queue_length
    else:
        print('error_edge')
        edge_computing_size = 0
    return edge_computing_size


def device_compute(device):
    # 设备计算
    slot_computing_data_size = config.time_slot_length * device.frequency
    task_queue_length = device.task_queue_length()
    if slot_computing_data_size <= task_queue_length:
        device.task_dequeue(cpu_frequency_demand=slot_computing_data_size)
        local_computing_size = slot_computing_data_size
    elif slot_computing_data_size > task_queue_length:
        device.task_dequeue(cpu_frequency_demand=slot_computing_data_size)
        local_computing_size = task_queue_length
    else:
        print('error_device')
        local_computing_size = 0
    return local_computing_size


def update_system(devices_vector, edges_vector):
    # 更新系统
    # global uav_trans_rates
    for i in range(0, config.total_number_of_devices):
        devices_vector[i].move()
    for i in range(0, config.total_number_of_devices):
        devices_vector[i].get_the_distance(total_number_of_edges=config.total_number_of_edges, edges=edges_vector)


class MobileEdgeComputingEnv(gym.Env):

    def __init__(self) -> None:
        super().__init__()

        self.total_number_of_devices = config.total_number_of_devices

        self.total_number_of_edges = config.total_number_of_edges

        self.total_number_of_resolutions = config.total_number_of_resolutions

        # 信道增益，队列状态
        self.state_dim = self.total_number_of_edges * 2 + 1

        self.action_space = [gym.spaces.Discrete(self.total_number_of_edges * self.total_number_of_resolutions) for _ in range(0, self.total_number_of_devices)]

        self.act_shape_n = [self.total_number_of_edges + 1 for _ in range(0, self.total_number_of_devices)]

        self.n = config.total_number_of_devices

        self.state_n = None
        # Box(low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
        self.observation_space = [gym.spaces.Box(-math.inf, math.inf, shape=(self.total_number_of_edges*2 + 1,1)) for _ in range(0, self.total_number_of_devices)]

        self.obs_shape_n = [self.state_dim for _ in range(0, self.total_number_of_devices)]

        self.env_name = 'multi_agent_mec'

        self.if_discrete = True

        self.action_dim = self.total_number_of_edges

        self.target_reward = target_reward

        self.edges = init_edges()

        self.devices = init_devices()

        self.count = 0

    def step(self, action_n):
        reward_n = []
        done_n = []
        self.state_n = []

        if self.count > 0:
            for i in range(0, self.total_number_of_devices):
                device_compute(device=self.devices[i])
            for i in range(0, self.total_number_of_edges):
                edge_compute(edge=self.edges[i])

        for i in range(0, self.total_number_of_devices):
            total_cost = 0.0
            resolution_selection = action_n[i].argmax()//self.total_number_of_edges
            execute_edge_id = action_n[i].argmax() % self.total_number_of_edges
            # todo
            local_computing_portion = np.random.rand()
            execute_edge = self.edges[execute_edge_id]
            # 计算卸载数据大小
            offload_computing_portion = (1 - local_computing_portion)
            if resolution_selection == 0:
                resolution = 100
                accuracy = 0.176
            elif resolution_selection == 1:
                resolution = 200
                accuracy = 0.570
            elif resolution_selection == 2:
                resolution = 300
                accuracy = 0.775
            elif resolution_selection == 3:
                resolution = 400
                accuracy = 0.882
            elif resolution_selection == 4:
                resolution = 500
                accuracy = 0.939
            elif resolution_selection == 5:
                resolution = 600
                accuracy = 0.968
            else:
                print('error_resolution')
            data_size = config.bits_per_pixel * math.pow(resolution, 2)
            cpu_frequency_demand = data_size * config.task_cpu_frequency_demand
            task = Task(resolution=resolution, accuracy=accuracy, data_size=data_size,
                        cpu_frequency_demand=cpu_frequency_demand)
            if offload_computing_portion > 0:
                offload_data_size = task.data_size * offload_computing_portion
                # 计算传输速率
                interference = 0
                for j in range(0, config.total_number_of_devices):
                    execute_edge_id_temp = action_n[j].argmax() % self.total_number_of_edges
                    if j != i and execute_edge_id_temp == execute_edge_id:
                        gain = get_upload_gain(device=self.devices[j], edge=execute_edge)
                        interference += gain
                upload_rate = get_upload_rate(device=self.devices[i], edge=execute_edge, interference=interference)
                # print('upload_rate', upload_rate/8388608)
                # 传输时间
                edge_transmit_latency = offload_data_size / upload_rate
                edge_transmit_latency = round(edge_transmit_latency, 6)
                # print('edge_transmit_latency=', edge_transmit_latency)
                if edge_transmit_latency >= config.time_slot_length:
                    edge_transmit_latency = config.time_slot_length
                # print('real_edge_transmit_latency=', edge_transmit_latency)
                # 传输能耗
                trans_energy_cost = edge_transmit_latency * devices[i].offload_trans_power
            elif offload_computing_portion == 0:
                # 传输时间
                edge_transmit_latency = 0
                # 传输能耗
                trans_energy_cost = 0
            else:
                print('error_offload_computing_portion')
                # 传输时间
                edge_transmit_latency = 0
                # 传输能耗
                trans_energy_cost = 0

            # 本地计算时间
            if local_computing_portion > 0:
                if edge_transmit_latency == config.time_slot_length:
                    edge_transmit_data_size = upload_rate * config.time_slot_length
                    local_computing_data_size = task.data_size - edge_transmit_data_size
                    local_computing_cpu_demand = local_computing_data_size * config.task_cpu_frequency_demand
                else:
                    local_computing_cpu_demand = task.cpu_frequency_demand * local_computing_portion
                self.devices[i].task_enqueue(cpu_frequency_demand=local_computing_cpu_demand)
                # 队列长度
                queue_length = self.devices[i].task_queue_length()
                # 本地计算时间
                local_execute_latency = queue_length / self.devices[i].frequency
                # 本地计算能耗
                local_computing_power = config.SWITCHED_CAPACITANCE * math.pow(self.devices[i].frequency, 3)
                local_computing_latency = local_computing_cpu_demand / self.devices[i].frequency
                local_computing_energy_cost = local_computing_latency * local_computing_power
            elif local_computing_portion == 0:
                local_computing_cpu_demand = 0
                local_execute_latency = 0
                local_computing_energy_cost = 0
            else:
                print('error_local_computing_portion')
                local_computing_cpu_demand = 0
                local_execute_latency = 0
                local_computing_energy_cost = 0
            # 能量消耗
            energy_cost = local_computing_energy_cost + trans_energy_cost
            print('energy_cost', energy_cost)
             # print('energy_cost', energy_cost)
            # 边缘计算执行时间
            if offload_computing_portion > 0:
                offload_computing_cpu_demand = cpu_frequency_demand - local_computing_cpu_demand
                execute_edge.task_enqueue(cpu_frequency_demand=offload_computing_cpu_demand)
                edge_queue_length = execute_edge.task_queue_length()
                edge_compute_task_latency = edge_queue_length / execute_edge.frequency
            elif offload_computing_portion == 0:
                edge_compute_task_latency = 0
            else:
                print('error_offload_computing_portion')
                edge_compute_task_latency = 0
            edge_execute_latency = edge_transmit_latency + edge_compute_task_latency
            # 延迟效用
            # print('local_execute_latency', local_execute_latency)
            # print('edge_execute_latency', edge_execute_latency)
            latency_cost = max(local_execute_latency, edge_execute_latency)
            print('latency_cost', latency_cost)
            # print('latency_utility', latency_utility)
            # 精度消耗
            error_cost = 1 - task.accuracy
            print('error_cost', error_cost)
            # 总效用
            total_cost = config.latency_weight * latency_cost + config.energy_weight * energy_cost + config.error_weight * error_cost
            # 奖励设为消耗的负值
            reward = - total_cost
            # 奖励设为消耗分之一
            # reward = 1/total_cost
            # 奖励设为反切值
            # reward = -(math.atan(latency_cost) + math.atan(energy_cost) + math.atan(error_cost))
            reward_n.append(reward)
        for i in range(0, self.total_number_of_devices):
            state = np.zeros(self.state_dim, dtype=np.float32)
            for j in range(0, self.total_number_of_edges):
                state[j*2] = self.edges[j].task_queue_length()
                channel_gain = get_upload_gain(self.devices[i], self.edges[j])
                state[j*2+1] = channel_gain
            state[self.total_number_of_edges * 2] = devices[i].task_queue_length()
            # 状态复位
            self.state_n.append(state)

        # 计算状态的reward

        self.count += 1
        done = self.count > 999
        for i in range(0, config.total_number_of_devices):
            done_n.append(done)
        update_system(devices_vector=self.devices, edges_vector=self.edges)
        # # 归一化
        # max_reward = max(reward_n)
        # min_reward = min(reward_n)
        # for i in range(0, len(reward_n)):
        #     reward_n[i] = reward_n[i] - min_reward
        #     reward_n[i] = reward_n[i] / (max_reward - min_reward)
        return self.state_n, reward_n, done_n, {}

    def reset(self):
        self.state_n = []
        init_device_queues(self.devices)
        init_edge_queues(self.edges)
        for i in range(0, self.total_number_of_devices):
            state_empty = np.zeros(self.state_dim, dtype=np.float32)
            for j in range(0, self.total_number_of_edges):
                state_empty[j*2] = self.edges[j].task_queue_length()
                channel_gain = get_upload_gain(self.devices[i], self.edges[j])
                state_empty[j*2+1] = channel_gain
            state_empty[self.total_number_of_edges * 2] = devices[i].task_queue_length()
            # 状态复位
            self.state_n.append(state_empty)
        self.count = 0
        return self.state_n








