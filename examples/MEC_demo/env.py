import random
from abc import ABC

import gym
import numpy as np
import math
from Config import Config
from MobileDevice import MobileDevice
from Edge import Edge
from Task import Task
from copy import deepcopy
import pickle
from trans_rate import get_upload_gain, get_upload_rate

target_reward = 8000

count_of_node_id = -1
count_of_edge_id = -1
config = Config()


def init_task():
    # 模拟负载
    task_in_each_slot = [[0 for _ in range(config.total_number_of_devices)] for _ in range(config.times)]
    if config.cache is True:
        task_cache_file = open(config.task_cache_file_path, 'rb')
        task_in_each_slot = pickle.load(task_cache_file)
    else:
        for time_slot in range(0, config.times):
            for i in range(0, config.total_number_of_devices):
                data_size = np.random.uniform(config.task_size[0], config.task_size[1])
                cpu_frequency_demand = config.task_cpu_frequency_demand * data_size
                task = Task(data_size=data_size, cpu_frequency_demand=cpu_frequency_demand)
                task_in_each_slot[time_slot][i] = task
        task_cache_file = open(config.task_cache_file_path, 'wb')
        pickle.dump(task_in_each_slot, task_cache_file)
    return task_in_each_slot


def init_devices(edges_vector):
    if config.cache is True:
        devices_cache_file = open(config.devices_cache_file_path, 'rb')
        devices = pickle.load(devices_cache_file)
    else:
        devices = []
        for i in range(0, config.total_number_of_devices):
            frequency = config.device_cpu_frequency
            ncp_type = 'DEV'
            nid = i
            device = MobileDevice(nid=nid, frequency=frequency,
                                ncp_type=ncp_type, move_distance=config.MOVE_DISTANCE,
                                total_number_of_edges=config.total_number_of_edges,
                                min_x_location=config.MIN_X_LOCATION, max_x_location=config.MAX_X_LOCATION,
                                min_y_location=config.MIN_Y_LOCATION, max_y_location=config.MAX_Y_LOCATION)
            device.init_position()
            device.get_the_distance(total_number_of_edges=config.total_number_of_edges, edges=edges_vector)
            devices.append(device)
        # 保存到文件中
        devices_cache_file = open(config.devices_cache_file_path, 'wb')
        pickle.dump(devices, devices_cache_file)
    return devices


def init_edges():
    if config.cache is True:
        edges_cache_file = open(config.edges_cache_file_path, 'rb')
        edges = pickle.load(edges_cache_file)
    else:
        edges = []
        for i in range(0, config.total_number_of_edges):
            frequency = config.edge_cpu_frequency
            ncp_type = 'EDGE'
            nid = i
            edge = Edge(nid=nid, frequency=frequency, ncp_type=ncp_type, x=config.edge_locations[i][0],
                        y=config.edge_locations[i][1], total_number_of_devices=config.total_number_of_devices)
            edges.append(edge)
        # 保存到文件中
        edges_cache_file = open(config.edges_cache_file_path, 'wb')
        pickle.dump(edges, edges_cache_file)
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

        # 信道增益，队列状态
        self.state_dim = self.total_number_of_edges * 2 + 3

        self.action_dim = self.total_number_of_edges + 1

        self.action_space = [gym.spaces.Discrete(self.total_number_of_edges + 1) for _ in range(0, self.total_number_of_devices)]

        self.act_shape_n = [self.action_dim for _ in range(0, self.total_number_of_devices)]

        self.n = config.total_number_of_devices

        self.state_n = None
        # Box(low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
        self.observation_space = [gym.spaces.Box(-math.inf, math.inf, shape=(self.total_number_of_edges * 2 + 1 + 2,1)) for _ in range(0, self.total_number_of_devices)]

        self.obs_shape_n = [self.state_dim for _ in range(0, self.total_number_of_devices)]

        self.env_name = 'multi_agent_mec'

        self.if_discrete = True

        self.target_reward = target_reward

        self.edges = init_edges()

        self.devices = init_devices(self.edges)

        self.task_in_each_slot = init_task()

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
            offload_selection = action_n[i].argmax()
            data_size = self.task_in_each_slot[self.count][i].data_size
            cpu_frequency_demand = self.task_in_each_slot[self.count][i].cpu_frequency_demand
            if offload_selection == 0:
                self.devices[i].task_enqueue(cpu_frequency_demand=cpu_frequency_demand)
                # 队列长度
                queue_length = self.devices[i].task_queue_length()
                # 本地计算时间
                local_execute_latency = queue_length / self.devices[i].frequency
                # 本地计算能耗
                local_computing_power = config.SWITCHED_CAPACITANCE * math.pow(self.devices[i].frequency, 3)
                local_computing_latency = cpu_frequency_demand / self.devices[i].frequency
                local_computing_energy_cost = local_computing_latency * local_computing_power
                total_cost = config.latency_weight * local_execute_latency + config.energy_weight * local_computing_energy_cost
            else:
                execute_edge = self.edges[offload_selection-1]
                upload_rate = get_upload_rate(device=self.devices[i], edge=execute_edge, interference=0)
                edge_transmit_latency = data_size / upload_rate
                # 传输能耗
                trans_energy_cost = edge_transmit_latency * self.devices[i].offload_trans_power
                execute_edge.task_enqueue(cpu_frequency_demand=cpu_frequency_demand)
                edge_queue_length = execute_edge.task_queue_length()
                edge_compute_task_latency = edge_queue_length / execute_edge.frequency
                edge_execute_latency = edge_transmit_latency + edge_compute_task_latency
                edge_latency = edge_transmit_latency + edge_compute_task_latency
                total_cost = config.latency_weight * edge_latency + config.energy_weight * trans_energy_cost
            # 奖励设为消耗的负值
            reward = - total_cost
            # 奖励设为消耗分之一
            # reward = 1/total_cost
            # 奖励设为反切值
            reward_n.append(reward)
        # 计算状态的reward
        self.count += 1
        done = self.count >= 999
        for i in range(0, config.total_number_of_devices):
            done_n.append(done)
        update_system(devices_vector=self.devices, edges_vector=self.edges)
        for i in range(0, self.total_number_of_devices):
            state = np.zeros(self.state_dim, dtype=np.float32)
            for j in range(0, self.total_number_of_edges):
                state[j * 2] = self.edges[j].task_queue_length() / self.edges[j].frequency
                distance = self.devices[i].distance_BS[j] / 500
                state[j * 2 + 1] = distance
            state[self.total_number_of_edges * 2] = self.devices[i].task_queue_length()
            state[self.total_number_of_edges * 2 + 1] = self.task_in_each_slot[self.count][i].data_size
            state[self.total_number_of_edges * 2 + 2] = self.task_in_each_slot[self.count][i].cpu_frequency_demand
            # 状态复位
            self.state_n.append(state)
        return self.state_n, reward_n, done_n, {}

    def reset(self):
        self.state_n = []
        init_device_queues(self.devices)
        init_edge_queues(self.edges)
        for i in range(0, self.total_number_of_devices):
            state = np.zeros(self.state_dim, dtype=np.float32)
            for j in range(0, self.total_number_of_edges):
                state[j * 2] = self.edges[j].task_queue_length() / self.edges[j].frequency
                distance = self.devices[i].distance_BS[j] / 500
                state[j * 2 + 1] = distance
            state[self.total_number_of_edges * 2] = self.devices[i].task_queue_length()
            state[self.total_number_of_edges * 2 + 1] = self.task_in_each_slot[0][i].data_size
            state[self.total_number_of_edges * 2 + 2] = self.task_in_each_slot[0][i].cpu_frequency_demand
            # 状态复位
            self.state_n.append(state)
        self.count = 0
        return self.state_n








