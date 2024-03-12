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

    for i in range(0, config.total_number_of_active_devices):
        frequency = config.activate_device_cpu_frequency
        ncp_type = 'ACT DEV'
        nid = i
        device = MobileDevice(nid=nid, frequency=frequency,
                              ncp_type=ncp_type, move_distance=config.MOVE_DISTANCE,
                              total_number_of_edges=config.total_number_of_edges,
                              min_x_location=config.MIN_X_LOCATION, max_x_location=config.MAX_X_LOCATION,
                              min_y_location=config.MIN_Y_LOCATION, max_y_location=config.MAX_Y_LOCATION,
                              generating_tasks_probability_of_device=config.generating_tasks_probability_of_active_device)
        device.init_position()
        device.get_the_distance(total_number_of_edges=config.total_number_of_edges, edges=edges)
        devices.append(device)

    for i in range(config.total_number_of_active_devices, config.total_number_of_devices):
        frequency = config.passive_device_cpu_frequency
        ncp_type = 'PAS DEV'
        nid = i
        device = MobileDevice(nid=nid, frequency=frequency,
                              ncp_type=ncp_type, move_distance=config.MOVE_DISTANCE,
                              total_number_of_edges=config.total_number_of_edges,
                              min_x_location=config.MIN_X_LOCATION, max_x_location=config.MAX_X_LOCATION,
                              min_y_location=config.MIN_Y_LOCATION, max_y_location=config.MAX_Y_LOCATION,
                              generating_tasks_probability_of_device=config.generating_tasks_probability_of_active_device)
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
        self.state_dim = self.total_number_of_edges * 2 + 1

        self.action_space = [gym.spaces.Discrete(self.total_number_of_edges + 1) for _ in range(0, self.total_number_of_devices)]

        self.act_shape_n = [self.total_number_of_edges + 1 for _ in range(0, self.total_number_of_devices)]

        self.n = config.total_number_of_devices

        self.state_n = None
        # Box(low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
        self.observation_space = [gym.spaces.Box(-math.inf, math.inf, shape=(self.total_number_of_edges*2 + 2,1)) for _ in range(0, self.total_number_of_devices)]
        self.obs_shape_n = [self.state_dim for _ in range(0, self.total_number_of_devices)]

        self.env_name = 'multi_agent_mec'

        self.if_discrete = True

        self.action_dim = self.total_number_of_edges

        self.target_reward = target_reward

        self.devices = init_devices()

        self.edges = init_edges()

        self.count = 0

    def step(self, action_n):
        reward_n = []
        done_n = []
        self.state_n = []

        for i in range(0, self.total_number_of_devices):
            total_cost = 0.0
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
        for i in range(0, self.total_number_of_devices):
            state_empty = np.zeros(self.state_dim, dtype=np.float32)
            for j in range(0, self.total_number_of_edges):
                state_empty[j*2] = self.edges[j].queue_length()
                channel_gain = get_upload_gain(self.devices[i], self.edges[j])
                state_empty[j*2+1] = channel_gain
            state_empty[self.total_number_of_edges * 2] = devices[i].queue_length()
            # 状态复位
            self.state_n.append(state_empty)
        self.count = 0
        return self.state_n








