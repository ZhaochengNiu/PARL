import random
import matplotlib.pyplot as plt
import numpy as np
import math
from copy import deepcopy
from trans_rate import get_upload_gain, get_upload_rate
from simple_model import MAModel
from simple_agent import MAAgent
from parl.algorithms import MADDPG
import os

# CRITIC_LR = 0.0001  # learning rate for the critic model
# ACTOR_LR = 0.00001  # learning rate of the actor model
CRITIC_LR = 0.001  # learning rate for the critic model
ACTOR_LR = 0.0001  # learning rate of the actor model
GAMMA = 0.95  # reward discount factor
TAU = 0.01  # soft update
BATCH_SIZE = 1024
# MAX_STEP_PER_EPISODE = 25  # maximum step per episode
MAX_STEP_PER_EPISODE = 1000  # maximum step per episode
EVAL_EPISODES = 3


def local_algorithm(decisions, config):
    for j in range(0, config.total_number_of_devices):
        decisions.resolution_selection.append(5)
        decisions.execute_destination.append(-1)
        decisions.local_computing_portion.append(1)
    return decisions


def nearest_bs_algorithm(decisions, config, devices, edges):
    for j in range(0, config.total_number_of_devices):
        # 找最近的基站
        min_distance = 4000
        destination = -1
        for k in range(0, config.total_number_of_edges):
            distance = devices[j].distance_BS[k]
            if distance < min_distance:
                min_distance = distance
                destination = k
        decisions.resolution_selection.append(5)
        decisions.execute_destination.append(destination)
    for j in range(0, config.total_number_of_devices):
        resolution = 600
        accuracy = 0.968
        data_size = config.bits_per_pixel * math.pow(resolution, 2)
        cpu_frequency_demand = data_size * config.task_cpu_frequency_demand
        # 最大限度的卸载任务
        execute_destination = decisions.execute_destination[j]
        interference = 0
        for k in range(0, config.total_number_of_devices):
            if k != j and decisions.execute_destination[k] == execute_destination:
                gain = get_upload_gain(device=devices[k], edge=edges[execute_destination])
                interference += gain
        upload_rate = get_upload_rate(device=devices[j], edge=edges[execute_destination], interference=interference)
        slot_upload_data_size = upload_rate * config.time_slot_length
        upload_date_size_up_limit = min(slot_upload_data_size, data_size)
        offload_computing_portion = upload_date_size_up_limit / data_size
        local_computing_portion = 1.0 - offload_computing_portion
        # print('local_computing_portion=',local_computing_portion)
        decisions.local_computing_portion.append(local_computing_portion)
    return decisions


def random_algorithm(decisions, config, devices, edges):
    for j in range(0, config.total_number_of_devices):
        rand_offload = np.random.randint(0, config.total_number_of_edges+1)
        decisions.execute_destination.append(rand_offload)

    return decisions


def binary_match_game(decisions, config, devices, edges):
    for j in range(config.total_number_of_devices):
        # 遍历所有可以选择的决策，选择最优的决策。
        cost_flag = 1000000
        resolution_selection = -1
        execute_destination = -2
        # 遍历所有可能的分辨率
        for k in range(0, config.total_number_of_resolutions):
            if k == 0:
                resolution = 100
                accuracy = 0.176
            elif k == 1:
                resolution = 200
                accuracy = 0.570
            elif k == 2:
                resolution = 300
                accuracy = 0.775
            elif k == 3:
                resolution = 400
                accuracy = 0.882
            elif k == 4:
                resolution = 500
                accuracy = 0.939
            elif k == 5:
                resolution = 600
                accuracy = 0.968
            else:
                print('error')
            data_size = config.bits_per_pixel * math.pow(resolution, 2)
            cpu_frequency_demand = data_size * config.task_cpu_frequency_demand
            # 1. 本地计算决策
            # 本地计算大小
            local_compute_size = devices[j].task_queue_length() + cpu_frequency_demand
            # 本地执行延迟
            local_execute_latency = local_compute_size / devices[j].frequency
            local_latency_cost = local_execute_latency
            # 本地计算能耗
            local_computing_power = config.SWITCHED_CAPACITANCE * math.pow(devices[j].frequency, 3)
            local_computing_latency = cpu_frequency_demand / devices[j].frequency
            local_energy_cost = local_computing_power * local_computing_latency
            # 精度消耗
            error_cost = 1 - accuracy
            # 总效用
            # total_local_cost = config.latency_weight * local_latency_cost + config.energy_weight * local_energy_cost + config.error_weight * error_cost
            total_local_cost = config.latency_weight * math.atan(local_latency_cost) + config.energy_weight * math.atan(local_energy_cost) + config.error_weight * math.atan(error_cost)
            # 更新 flag
            if total_local_cost <= cost_flag:
                cost_flag = total_local_cost
                resolution_selection = k
                execute_destination = -1
            # 2. 边缘计算决策
            edges_execute_latency = [float("inf") for _ in range(config.total_number_of_edges)]
            edges_transmit_cost = [0 for _ in range(config.total_number_of_edges)]
            for m in range(0, config.total_number_of_edges):
                # 计算传输速率
                interference = 0
                upload_rate = get_upload_rate(devices[j], edges[m], interference)
                # 计算传输时间
                edge_transmit_latency = data_size / upload_rate
                # 计算传输能耗
                transmit_cost = edge_transmit_latency * devices[j].offload_trans_power
                edge_energy_cost = transmit_cost
                edges_transmit_cost[m] = transmit_cost
                # 边缘计算数据大小
                edge_compute_size = edges[m].task_queue_length() + cpu_frequency_demand
                # 边缘计算时间
                edge_compute_latency = edge_compute_size / edges[m].frequency
                # 边缘执行延迟
                edge_execute_latency = edge_transmit_latency + edge_compute_latency
                edges_execute_latency[m] = edge_execute_latency
                edge_latency_cost = edge_execute_latency
                # 精度消耗
                error_cost = 1 - accuracy
                # 总效用
                # total_edge_cost = config.latency_weight * edge_latency_cost + config.energy_weight * edge_energy_cost + config.error_weight * error_cost
                total_edge_cost = config.latency_weight * math.atan(edge_latency_cost) + config.energy_weight * math.atan(edge_energy_cost) + config.error_weight * math.atan(error_cost)
                # 更新 flag
                if total_edge_cost <= cost_flag:
                    cost_flag = total_edge_cost
                    resolution_selection = k
                    execute_destination = m
        decisions.resolution_selection.append(resolution_selection)
        decisions.execute_destination.append(execute_destination)
    # 确定卸载比率
    for j in range(0, config.total_number_of_devices):
        if decisions.execute_destination[j] == -1:
            decisions.local_computing_portion.append(1)
        else:
            if decisions.resolution_selection[j] == 1:
                resolution = 100
                accuracy = 0.176
            elif decisions.resolution_selection[j] == 2:
                resolution = 200
                accuracy = 0.570
            elif decisions.resolution_selection[j] == 3:
                resolution = 300
                accuracy = 0.775
            elif decisions.resolution_selection[j] == 4:
                resolution = 400
                accuracy = 0.882
            elif decisions.resolution_selection[j] == 5:
                resolution = 500
                accuracy = 0.939
            elif decisions.resolution_selection[j] == 6:
                resolution = 600
                accuracy = 0.968
            else:
                print('error')
            data_size = config.bits_per_pixel * math.pow(resolution, 2)
            cpu_frequency_demand = data_size * config.task_cpu_frequency_demand
            execute_destination = decisions.execute_destination[j]
            interference = 0
            for k in range(0, config.total_number_of_devices):
                if k != j and decisions.execute_destination[k] == execute_destination:
                    gain = get_upload_gain(device=devices[k], edge=edges[execute_destination])
                    interference += gain
            upload_rate = get_upload_rate(device=devices[j], edge=edges[execute_destination], interference=interference)
            slot_upload_data_size = upload_rate * config.time_slot_length
            upload_data_size_up_limit = min(slot_upload_data_size, data_size)
            offload_computing_portion_up_limit = upload_data_size_up_limit / data_size
            local_computing_portion = 1.0 - offload_computing_portion_up_limit
            decisions.local_computing_portion.append(local_computing_portion)
    return decisions


def proposed_algorithm(decisions, config, devices, edges, time_slot, task_in_each_slot):
    # build agents
    agents = []
    for i in range(config.total_number_of_devices):
        critic_in_dim = sum(config.obs_shape_n) + sum(config.act_shape_n)
        model = MAModel(config.obs_shape_n[i], config.act_shape_n[i], critic_in_dim)
        algorithm = MADDPG(
            model,
            agent_index=i,
            act_space=config.action_space,
            gamma=GAMMA,
            tau=TAU,
            critic_lr=CRITIC_LR,
            actor_lr=ACTOR_LR)
        agent = MAAgent(
            algorithm,
            agent_index=i,
            obs_dim_n=config.obs_shape_n,
            act_dim_n=config.act_shape_n,
            batch_size=BATCH_SIZE)
        agents.append(agent)
    for i in range(len(agents)):
        model_file = config.model_dir + '/agent_' + str(i)
        if not os.path.exists(model_file):
            raise Exception(
                'model file {} does not exits'.format(model_file))
        agents[i].restore(model_file)
    state_n = []
    for i in range(0, config.total_number_of_devices):
        state = np.zeros(config.state_dim, dtype=np.float32)
        for j in range(0, config.total_number_of_edges):
            state[j * 2] = edges[j].task_queue_length() / edges[j].frequency
            distance = devices[i].distance_BS[j] / 500
            state[j * 2 + 1] = distance
        state[config.total_number_of_edges * 2] = devices[i].task_queue_length()
        state[config.total_number_of_edges * 2 + 1] = task_in_each_slot[time_slot][i].data_size
        state[config.total_number_of_edges * 2 + 2] = task_in_each_slot[time_slot][i].cpu_frequency_demand
        # 状态复位
        state_n.append(state)
    action_n = [agent.predict(state) for agent, state in zip(agents, state_n)]
    for i in range(0, config.total_number_of_devices):
        offload_selection = action_n[i].argmax()
        decisions.execute_destination.append(offload_selection)

    return decisions

def proposed_algorithm_v2(decisions, config, devices, edges):
    # build agents
    agents = []
    for i in range(config.total_number_of_devices):
        critic_in_dim = sum(config.obs_shape_n) + sum(config.act_shape_n)
        model = MAModel(config.obs_shape_n[i], config.act_shape_n[i], critic_in_dim)
        algorithm = MADDPG(
            model,
            agent_index=i,
            act_space=config.action_space,
            gamma=GAMMA,
            tau=TAU,
            critic_lr=CRITIC_LR,
            actor_lr=ACTOR_LR)
        agent = MAAgent(
            algorithm,
            agent_index=i,
            obs_dim_n=config.obs_shape_n,
            act_dim_n=config.act_shape_n,
            batch_size=BATCH_SIZE)
        agents.append(agent)
    for i in range(len(agents)):
        model_file = config.model_dir + '/agent_' + str(i)
        if not os.path.exists(model_file):
            raise Exception(
                'model file {} does not exits'.format(model_file))
        agents[i].restore(model_file)
    state_n = []
    for i in range(0, config.total_number_of_devices):
        state = np.zeros(config.state_dim, dtype=np.float32)
        for j in range(0, config.total_number_of_edges):
            state[j * 2] = edges[j].task_queue_length()
            channel_gain = get_upload_gain(devices[i], edges[j])
            state[j * 2 + 1] = channel_gain
        state[config.total_number_of_edges * 2] = devices[i].task_queue_length()
        # 状态复位
        state_n.append(state)
    action_n = [agent.predict(state) for agent, state in zip(agents, state_n)]
    for i in range(0, config.total_number_of_devices):
        resolution_selection = action_n[i].argmax() // config.total_number_of_edges
        # print(resolution_selection)
        execute_edge_id = action_n[i].argmax() % config.total_number_of_edges
        # print(execute_edge_id)
        execute_edge = edges[execute_edge_id]
        decisions.resolution_selection.append(resolution_selection)
        decisions.execute_destination.append(execute_edge_id)
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
        interference = 0
        for j in range(0, config.total_number_of_devices):
            execute_edge_id_temp = action_n[j].argmax() % config.total_number_of_edges
            if j != i and execute_edge_id_temp == execute_edge_id:
                gain = get_upload_gain(device=devices[j], edge=execute_edge)
                interference += gain
        upload_rate = get_upload_rate(device=devices[i], edge=execute_edge, interference=interference)
        slot_upload_data_size = upload_rate * config.time_slot_length
        slot_upload_data_size = round(slot_upload_data_size, 6)
        offload_data_size_up_limit = min(slot_upload_data_size, data_size)
        offload_computing_portion_up_limit = offload_data_size_up_limit / data_size
        offload_computing_portion_up_limit = round(offload_computing_portion_up_limit, 6)
        X = np.random.uniform(low=0, high=offload_computing_portion_up_limit, size=config.pop_size)
        X[0] = 0
        X[1] = offload_computing_portion_up_limit
        Y = [0 for _ in range(len(X))]
        for j in range(len(X)):
            offload_computing_portion = X[j]
            if offload_computing_portion > 0:
                offload_data_size = data_size * offload_computing_portion
                # 传输时间
                edge_transmit_latency = offload_data_size / upload_rate
                edge_transmit_latency = round(edge_transmit_latency, 6)
                # print('edge_transmit_latency', edge_transmit_latency)
                # 传输能耗
                trans_energy_cost = edge_transmit_latency * devices[i].offload_trans_power
            elif offload_computing_portion == 0:
                # 传输时间
                edge_transmit_latency = 0
                # 传输能耗
                trans_energy_cost = 0
            else:
                print('error')
            # 本地计算时间
            local_computing_portion = 1 - offload_computing_portion
            if local_computing_portion > 0:
                local_computing_cpu_demand = cpu_frequency_demand * local_computing_portion
                local_computing_size = devices[i].task_queue_length() + local_computing_cpu_demand
                # 本地计算时间
                local_execute_latency = local_computing_size / devices[i].frequency
                # 本地计算能耗
                local_computing_power = config.SWITCHED_CAPACITANCE * math.pow(devices[i].frequency, 3)
                local_computing_latency = local_computing_cpu_demand / devices[i].frequency
                local_computing_energy_cost = local_computing_latency * local_computing_power
            elif local_computing_portion == 0:
                local_execute_latency = 0
                local_computing_energy_cost = 0
            else:
                print('error')
            # 能量消耗
            energy_cost = local_computing_energy_cost + trans_energy_cost
            # print(energy_cost)
            # 边缘计算执行时间
            if offload_computing_portion > 0:
                offload_computing_cpu_demand = cpu_frequency_demand * offload_computing_portion
                edge_queue_length = execute_edge.task_queue_length() + offload_computing_cpu_demand
                edge_computing_task_latency = edge_queue_length / execute_edge.frequency
            elif offload_computing_portion == 0:
                edge_computing_task_latency = 0
            else:
                print('error')
            edge_execute_latency = edge_transmit_latency + edge_computing_task_latency
            # 延迟效用
            latency_cost = max(local_execute_latency, edge_execute_latency)
            # 精度消耗
            error_cost = 1 - accuracy
            # 总效用
            # total_cost = config.latency_weight * latency_cost + config.energy_weight * energy_cost + config.error_weight * error_cost
            total_cost = config.latency_weight * math.atan(latency_cost) + config.energy_weight * math.atan(energy_cost) + config.error_weight * math.atan(error_cost)
            #
            Y[j] = total_cost
        pbest_x = X.copy()  # personal best location of every particle in history
        # self.pbest_x = self.X 表示地址传递,改变 X 值 pbest_x 也会变化
        pbest_y = [np.inf for _ in range(config.pop_size)]  # best image of every particle in history
        # self.gbest_x = self.pbest_x.mean(axis=0).reshape(1, -1)  # global best location for all particles
        gbest_x = pbest_x.mean(axis=0)
        gbest_y = np.inf  # global best y for all particles
        gbest_y_hist = []  # gbest_y of every iteration
        for j in range(config.max_iter):
            # # update
            # r1 = config.a - j * (config.a / config.max_iter)
            # 抛物线函数
            iter_period = j / config.max_iter
            inter_rest_phase = 1 - iter_period
            square = pow(inter_rest_phase, 2)
            r1 = config.a * square
            for k in range(config.pop_size):
                r2 = 2 * math.pi * random.uniform(0.0, 1.0)
                r3 = 2 * random.uniform(0.0, 1.0)
                r4 = random.uniform(0.0, 1.0)
                if r4 < 0.5:
                    X[k] = X[k] + (r1 * math.sin(r2) * abs(r3 * gbest_x - X[k]))
                else:
                    X[k] = X[k] + (r1 * math.cos(r2) * abs(r3 * gbest_x - X[k]))
            X = np.clip(a=X, a_min=0, a_max=offload_computing_portion_up_limit)
            for k in range(len(X)):
                offload_computing_portion = X[k]
                if offload_computing_portion > 0:
                    offload_data_size = data_size * offload_computing_portion
                    # 传输时间
                    edge_transmit_latency = offload_data_size / upload_rate
                    edge_transmit_latency = round(edge_transmit_latency, 6)
                    # print('edge_transmit_latency', edge_transmit_latency)
                    # 传输能耗
                    trans_energy_cost = edge_transmit_latency * devices[i].offload_trans_power
                elif offload_computing_portion == 0:
                    # 传输时间
                    edge_transmit_latency = 0
                    # 传输能耗
                    trans_energy_cost = 0
                else:
                    print('error')
                # 本地计算时间
                local_computing_portion = 1 - offload_computing_portion
                if local_computing_portion > 0:
                    local_computing_cpu_demand = cpu_frequency_demand * local_computing_portion
                    local_computing_size = devices[i].task_queue_length() + local_computing_cpu_demand
                    # 本地计算时间
                    local_execute_latency = local_computing_size / devices[i].frequency
                    # 本地计算能耗
                    local_computing_power = config.SWITCHED_CAPACITANCE * math.pow(devices[i].frequency, 3)
                    local_computing_latency = local_computing_cpu_demand / devices[i].frequency
                    local_computing_energy_cost = local_computing_latency * local_computing_power
                elif local_computing_portion == 0:
                    local_execute_latency = 0
                    local_computing_energy_cost = 0
                else:
                    print('error')
                # 能量消耗
                energy_cost = local_computing_energy_cost + trans_energy_cost
                # print(energy_cost)
                # 边缘计算执行时间
                if offload_computing_portion > 0:
                    offload_computing_cpu_demand = cpu_frequency_demand * offload_computing_portion
                    edge_queue_length = execute_edge.task_queue_length() + offload_computing_cpu_demand
                    edge_computing_task_latency = edge_queue_length / execute_edge.frequency
                elif offload_computing_portion == 0:
                    edge_computing_task_latency = 0
                else:
                    print('error')
                edge_execute_latency = edge_transmit_latency + edge_computing_task_latency
                # 延迟效用
                latency_cost = max(local_execute_latency, edge_execute_latency)
                # 精度消耗
                error_cost = 1 - accuracy
                # 总效用
                # total_cost = config.latency_weight * latency_cost + config.energy_weight * energy_cost + config.error_weight * error_cost
                total_cost = config.latency_weight * math.atan(latency_cost) + config.energy_weight * math.atan(energy_cost) + config.error_weight * math.atan(error_cost)
                #
                Y[k] = total_cost
            # update_pbest
            for k in range(len(Y)):
                if pbest_y[k] > Y[k]:
                    pbest_x[k] = X[k].copy()
                    # pbest_y[k] = Y[k].copy()
                    pbest_y[k] = Y[k]
            # update_gbest
            idx_min = pbest_y.index(min(pbest_y))
            if gbest_y > pbest_y[idx_min]:
                gbest_x = pbest_x[idx_min].copy()  # copy很重要！
                gbest_y = pbest_y[idx_min]
            gbest_y_hist.append(gbest_y)
        local_computing_portion = 1 - gbest_x
        decisions.local_computing_portion.append(local_computing_portion)
    return decisions

def proposed_algorithm_v2(decisions, config, devices, edges, time_slot, task_in_each_slot):
    decisions = random_algorithm(decisions, config, devices, edges, time_slot, task_in_each_slot)
    for i in range(0, config.iterations_number_of_game_theory):
        lucky_user = numpy.random.randint(0, config.total_number_of_devices)
        while decisions.execute_mode[lucky_user] == 'null':
            lucky_user = numpy.random.randint(0, config.total_number_of_devices)
        if decisions.execute_mode[lucky_user] != 'null':
            devices_temp = deepcopy(devices)
            edges_temp = deepcopy(edges)
            for j in range(0, lucky_user):
                if decisions.execute_mode[j] == 'local':
                    cpu_frequency_demand = task_in_each_slot[time_slot][j].cpu_frequency_demand
                    devices_temp[j].task_enqueue(cpu_frequency_demand)
                elif decisions.execute_mode[j] == 'edge':
                    execute_destination = decisions.execute_destination[j]
                    data_size = task_in_each_slot[time_slot][j].data_size
                    local_computing_portion = decisions.local_computing_portion[j]
                    edge_computing_portion = 1 - local_computing_portion
                    offload_data_size = data_size * edge_computing_portion
                    # 计算传输速率
                    interference = 0
                    for k in range(0, config.total_number_of_devices):
                        if k != j and decisions.execute_mode[k] == 'edge' and decisions.execute_destination[
                            k] == execute_destination:
                            gain = get_upload_gain(device=devices_temp[k], edge=edges_temp[execute_destination])
                            interference += gain
                    upload_rate = get_upload_rate(device=devices_temp[j], edge=edges_temp[execute_destination],
                                                  interference=interference)
                    # 传输时间
                    edge_transmit_latency = offload_data_size / upload_rate
                    edge_transmit_latency = round(edge_transmit_latency, 6)
                    if edge_transmit_latency >= config.time_slot_length:
                        edge_transmit_latency = config.time_slot_length
                    edge_transmit_data_size = edge_transmit_latency * upload_rate
                    edge_transmit_cpu_frequency_demand = edge_transmit_data_size * config.task_cpu_frequency_demand
                    edges_temp[execute_destination].task_enqueue(edge_transmit_cpu_frequency_demand)
                    local_computing_data_size = data_size - edge_transmit_data_size
                    if edge_transmit_latency == config.time_slot_length:
                        decisions.local_computing_portion[j] = local_computing_data_size / data_size
                    local_computing_cpu_frequency_demand = local_computing_data_size * config.task_cpu_frequency_demand
                    devices_temp[j].task_enqueue(local_computing_cpu_frequency_demand)
                elif decisions.execute_mode[j] == 'device':
                    execute_destination = decisions.execute_destination[j]
                    data_size = task_in_each_slot[time_slot][j].data_size
                    local_computing_portion = decisions.local_computing_portion[j]
                    d2d_computing_portion = 1 - local_computing_portion
                    d2d_data_size = data_size * d2d_computing_portion
                    # 计算传输速率
                    interference = 0
                    for k in range(0, config.total_number_of_devices):
                        if k != j and decisions.execute_mode[k] == 'device' and decisions.execute_destination[
                            k] == execute_destination:
                            gain = get_d2d_gain(device1=devices_temp[k], device2=devices_temp[execute_destination])
                            interference += gain
                    d2d_rate = get_d2d_rate(device1=devices_temp[j], device2=devices_temp[execute_destination],
                                            interference=interference)
                    # 传输时间
                    d2d_transmit_latency = d2d_data_size / d2d_rate
                    d2d_transmit_latency = round(d2d_transmit_latency, 6)
                    if d2d_transmit_latency >= config.time_slot_length:
                        d2d_transmit_latency = config.time_slot_length
                    d2d_transmit_data_size = d2d_transmit_latency * d2d_rate
                    d2d_transmit_cpu_frequency_demand = d2d_transmit_data_size * config.task_cpu_frequency_demand
                    devices_temp[execute_destination].task_enqueue(d2d_transmit_cpu_frequency_demand)
                    local_computing_data_size = data_size - d2d_transmit_data_size
                    if d2d_transmit_latency == config.time_slot_length:
                        decisions.local_computing_portion[j] = local_computing_data_size / data_size
                    local_computing_cpu_frequency_demand = local_computing_data_size * config.task_cpu_frequency_demand
                    devices_temp[j].task_enqueue(local_computing_cpu_frequency_demand)
            task = task_in_each_slot[time_slot][lucky_user]
            data_size = task_in_each_slot[time_slot][lucky_user].data_size
            cpu_frequency_demand = task_in_each_slot[time_slot][lucky_user].cpu_frequency_demand
            # 计算上一轮的效用
            last_total_utility = 0
            if decisions.execute_mode[lucky_user] == 'local':
                local_computing_size = devices_temp[lucky_user].task_queue_length() + cpu_frequency_demand
                local_execute_latency = local_computing_size / devices_temp[lucky_user].frequency
                latency_utility = local_execute_latency
                # 本地计算能耗
                local_computing_power = config.SWITCHED_CAPACITANCE * math.pow(devices_temp[lucky_user].frequency, 3)
                local_computing_latency = cpu_frequency_demand / devices_temp[lucky_user].frequency
                energy_cost = local_computing_latency * local_computing_power
                # 收集能量
                min_distance = 4000
                destination = -1
                for j in range(0, len(devices_temp[lucky_user].distance_BS)):
                    distance = devices_temp[lucky_user].distance_BS[j]
                    if distance < min_distance:
                        min_distance = distance
                        destination = j
                if devices_temp[lucky_user].locate_BS[destination] == 1:
                    energy_harvest_gain = config.ENERGY_CHANNEL_GAIN
                    energy_harvest = config.ENERGY_CONVERSION_EFFICIENCY * edges_temp[
                        destination].trans_power * energy_harvest_gain * config.time_slot_length
                else:
                    energy_harvest = 0
                # 能量效用
                energy_utility = energy_cost - energy_harvest
                # 获取权重
                latency_weight, energy_weight = devices_temp[lucky_user].get_weight()
                # 总效用
                total_local_utility = latency_weight * latency_utility + energy_weight * energy_utility
                last_total_utility = total_local_utility
            elif decisions.execute_mode[lucky_user] == 'edge':
                execute_edge_id = decisions.execute_destination[lucky_user]
                local_computing_portion = decisions.local_computing_portion[lucky_user]
                execute_edge = edges_temp[execute_edge_id]
                # 计算卸载数据大小
                offload_computing_portion = (1 - local_computing_portion)
                # print('offload_computing_portion', offload_computing_portion)
                if offload_computing_portion > 0:
                    offload_data_size = data_size * offload_computing_portion
                    # 计算传输速率
                    interference = 0
                    for j in range(0, config.total_number_of_devices):
                        if j != lucky_user and decisions.execute_mode[j] == 'edge' and decisions.execute_destination[
                            j] == execute_edge_id:
                            gain = get_upload_gain(device=devices_temp[j], edge=execute_edge)
                            interference += gain
                    upload_rate = get_upload_rate(device=devices_temp[lucky_user], edge=execute_edge,
                                                  interference=interference)
                    # 传输时间
                    edge_transmit_latency = offload_data_size / upload_rate
                    edge_transmit_latency = round(edge_transmit_latency, 6)
                    if edge_transmit_latency >= config.time_slot_length:
                        edge_transmit_latency = config.time_slot_length
                    # 传输能耗
                    trans_energy_cost = edge_transmit_latency * devices_temp[lucky_user].offload_trans_power
                elif offload_computing_portion == 0:
                    # 传输时间
                    edge_transmit_latency = 0
                    # 传输能耗
                    trans_energy_cost = 0
                else:
                    print('error')
                # 收集能量
                min_distance = 4000
                destination = -1
                for j in range(0, len(devices_temp[lucky_user].distance_BS)):
                    distance = devices_temp[lucky_user].distance_BS[j]
                    if distance < min_distance:
                        min_distance = distance
                        destination = j
                if devices_temp[lucky_user].locate_BS[destination] == 1:
                    energy_harvest_gain = config.ENERGY_CHANNEL_GAIN
                    energy_harvest_slot_length = config.time_slot_length - edge_transmit_latency
                    # print('energy harvest slot length =', energy_harvest_slot_length)
                    energy_harvest = config.ENERGY_CONVERSION_EFFICIENCY * edges_temp[
                        destination].trans_power * energy_harvest_gain * energy_harvest_slot_length
                    # print('energy harvest ', energy_harvest)
                else:
                    energy_harvest = 0
                # 本地计算时间
                if local_computing_portion > 0:
                    if edge_transmit_latency == config.time_slot_length:
                        edge_transmit_data_size = upload_rate * config.time_slot_length
                        local_computing_data_size = data_size - edge_transmit_data_size
                        decisions.local_computing_portion[lucky_user] = local_computing_data_size / data_size
                        local_computing_cpu_demand = local_computing_data_size * config.task_cpu_frequency_demand
                    else:
                        local_computing_cpu_demand = cpu_frequency_demand * local_computing_portion
                    local_computing_size = devices_temp[lucky_user].task_queue_length() + local_computing_cpu_demand
                    # 本地计算时间
                    local_execute_latency = local_computing_size / devices_temp[lucky_user].frequency
                    # 本地计算能耗
                    local_computing_power = config.SWITCHED_CAPACITANCE * math.pow(devices_temp[lucky_user].frequency,
                                                                                   3)
                    local_computing_latency = local_computing_cpu_demand / devices_temp[lucky_user].frequency
                    local_computing_energy_cost = local_computing_latency * local_computing_power
                elif local_computing_portion == 0:
                    local_computing_cpu_demand = 0
                    local_execute_latency = 0
                    local_computing_energy_cost = 0
                else:
                    print('error')
                # 能量消耗
                energy_cost = local_computing_energy_cost + trans_energy_cost
                # 能量效用
                energy_utility = energy_cost - energy_harvest
                # print(energy_utility)
                # 边缘计算执行时间
                if offload_computing_portion > 0:
                    offload_computing_cpu_demand = cpu_frequency_demand - local_computing_cpu_demand
                    edge_queue_length = execute_edge.task_queue_length() + offload_computing_cpu_demand
                    edge_computing_task_latency = edge_queue_length / execute_edge.frequency
                elif offload_computing_portion == 0:
                    edge_computing_task_latency = 0
                else:
                    print('error')
                edge_execute_latency = edge_transmit_latency + edge_computing_task_latency
                # 延迟效用
                latency_utility = max(local_execute_latency, edge_execute_latency)
                # print('energy queue', devices_temp[lucky_user].energy_queue)
                # 获取权重
                latency_weight, energy_weight = devices_temp[lucky_user].get_weight()
                # 总效用
                total_edge_utility = latency_weight * latency_utility + energy_weight * energy_utility
                last_total_utility = total_edge_utility
            elif decisions.execute_mode[lucky_user] == 'device':
                execute_device_id = decisions.execute_destination[lucky_user]
                local_computing_portion = decisions.local_computing_portion[lucky_user]
                execute_device = devices_temp[execute_device_id]
                # 计算卸载数据大小
                offload_computing_portion = (1 - local_computing_portion)
                if offload_computing_portion > 0:
                    offload_data_size = data_size * offload_computing_portion
                    # 计算传输速率
                    interference = 0
                    for j in range(0, config.total_number_of_devices):
                        if j != lucky_user and decisions.execute_mode[j] == 'device' and decisions.execute_destination[
                            j] == execute_device_id:
                            gain = get_d2d_gain(device1=devices_temp[j], device2=execute_device)
                            interference += gain
                    d2d_rate = get_d2d_rate(device1=devices_temp[lucky_user], device2=execute_device,
                                            interference=interference)
                    # 传输时间
                    d2d_transmit_latency = offload_data_size / d2d_rate
                    d2d_transmit_latency = round(d2d_transmit_latency, 6)
                    if d2d_transmit_latency >= config.time_slot_length:
                        d2d_transmit_latency = config.time_slot_length
                    # 传输能耗
                    trans_energy_cost = d2d_transmit_latency * devices_temp[lucky_user].d2d_trans_power
                elif offload_computing_portion == 0:
                    # 传输时间
                    d2d_transmit_latency = 0
                    # 传输能耗
                    trans_energy_cost = 0
                else:
                    print('error')
                # 收集能量
                min_distance = 4000
                destination = -1
                for j in range(0, len(devices_temp[lucky_user].distance_BS)):
                    distance = devices_temp[lucky_user].distance_BS[j]
                    if distance < min_distance:
                        min_distance = distance
                        destination = j
                if devices_temp[lucky_user].locate_BS[destination] == 1:
                    energy_harvest_gain = config.ENERGY_CHANNEL_GAIN
                    energy_harvest_slot_length = config.time_slot_length - d2d_transmit_latency
                    energy_harvest = config.ENERGY_CONVERSION_EFFICIENCY * edges_temp[
                        destination].trans_power * energy_harvest_gain * energy_harvest_slot_length
                else:
                    energy_harvest = 0
                # 本地计算时间
                if local_computing_portion > 0:
                    if d2d_transmit_latency == config.time_slot_length:
                        d2d_transmit_data_size = d2d_rate * config.time_slot_length
                        local_computing_data_size = data_size - d2d_transmit_data_size
                        decisions.local_computing_portion[lucky_user] = local_computing_data_size / data_size
                        local_computing_cpu_demand = local_computing_data_size * config.task_cpu_frequency_demand
                    else:
                        local_computing_cpu_demand = cpu_frequency_demand * local_computing_portion
                    local_computing_size = devices_temp[lucky_user].task_queue_length() + local_computing_cpu_demand
                    # 本地计算时间
                    local_execute_latency = local_computing_size / devices_temp[lucky_user].frequency
                    # 本地计算能耗
                    local_computing_power = config.SWITCHED_CAPACITANCE * math.pow(devices_temp[lucky_user].frequency,
                                                                                   3)
                    local_computing_latency = local_computing_cpu_demand / devices_temp[lucky_user].frequency
                    local_computing_energy_cost = local_computing_latency * local_computing_power
                elif local_computing_portion == 0:
                    local_computing_cpu_demand = 0
                    local_execute_latency = 0
                    local_computing_energy_cost = 0
                else:
                    print('error')
                # 能量消耗
                energy_cost = local_computing_energy_cost + trans_energy_cost
                # 能量效用
                energy_utility = energy_cost - energy_harvest
                # 边缘计算执行时间
                if offload_computing_portion > 0:
                    offload_computing_cpu_demand = cpu_frequency_demand - local_computing_cpu_demand
                    d2d_queue_length = execute_device.task_queue_length() + offload_computing_cpu_demand
                    d2d_compute_task_latency = d2d_queue_length / execute_device.frequency
                elif offload_computing_portion == 0:
                    d2d_compute_task_latency = 0
                else:
                    print('error')
                d2d_execute_latency = d2d_transmit_latency + d2d_compute_task_latency
                # 延迟效用
                latency_utility = max(local_execute_latency, d2d_execute_latency)
                # 获取权重
                latency_weight, energy_weight = devices_temp[lucky_user].get_weight()
                # 总效用
                total_d2d_utility = latency_weight * latency_utility + energy_weight * energy_utility
                last_total_utility = total_d2d_utility
            # this interation
            rand = numpy.random.randint(1, 4)
            if rand == 1:
                local_computing_size = devices_temp[lucky_user].task_queue_length() + cpu_frequency_demand
                local_execute_latency = local_computing_size / devices_temp[lucky_user].frequency
                latency_utility = local_execute_latency
                # 本地计算能耗
                local_computing_power = config.SWITCHED_CAPACITANCE * math.pow(devices_temp[lucky_user].frequency, 3)
                local_computing_latency = cpu_frequency_demand / devices_temp[lucky_user].frequency
                energy_cost = local_computing_latency * local_computing_power
                # 收集能量
                min_distance = 4000
                destination = -1
                for j in range(0, len(devices_temp[lucky_user].distance_BS)):
                    distance = devices_temp[lucky_user].distance_BS[j]
                    if distance < min_distance:
                        min_distance = distance
                        destination = j
                if devices_temp[lucky_user].locate_BS[destination] == 1:
                    energy_harvest_gain = config.ENERGY_CHANNEL_GAIN
                    energy_harvest = config.ENERGY_CONVERSION_EFFICIENCY * edges_temp[
                        destination].trans_power * energy_harvest_gain * config.time_slot_length
                    # print('energy_harvest', energy_harvest)
                else:
                    energy_harvest = 0
                # 能量效用
                energy_utility = energy_cost - energy_harvest
                # 获取权重
                latency_weight, energy_weight = devices_temp[lucky_user].get_weight()
                # 总效用
                total_local_utility = latency_weight * latency_utility + energy_weight * energy_utility
                this_total_utility = total_local_utility
                if this_total_utility <= last_total_utility:
                    decisions.execute_mode[lucky_user] = 'local'
                    decisions.execute_destination[lucky_user] = -1
                    decisions.local_computing_portion[lucky_user] = 1
            elif rand == 2:
                # 随机选动作
                choices_number = 0
                for j in range(0, config.total_number_of_edges):
                    if devices_temp[lucky_user].locate_BS[j] == 1:
                        choices_number = choices_number + 1
                if choices_number == 0:
                    execute_mode = 'local'
                    execute_destination = -1
                elif choices_number > 0:
                    lucky_number = numpy.random.randint(1, choices_number + 1)
                    for j in range(0, config.total_number_of_edges):
                        if devices_temp[lucky_user].locate_BS[j] == 1:
                            lucky_number = lucky_number - 1
                            if lucky_number == 0:
                                execute_mode = 'edge'
                                execute_destination = j
                # 计算效用
                if execute_mode == 'local':
                    local_computing_size = devices_temp[lucky_user].task_queue_length() + cpu_frequency_demand
                    local_execute_latency = local_computing_size / devices_temp[lucky_user].frequency
                    latency_utility = local_execute_latency
                    # 本地计算能耗
                    local_computing_power = config.SWITCHED_CAPACITANCE * math.pow(devices_temp[lucky_user].frequency,
                                                                                   3)
                    local_computing_latency = cpu_frequency_demand / devices_temp[lucky_user].frequency
                    energy_cost = local_computing_latency * local_computing_power
                    # 收集能量
                    min_distance = 4000
                    destination = -1
                    for j in range(0, len(devices_temp[lucky_user].distance_BS)):
                        distance = devices_temp[lucky_user].distance_BS[j]
                        if distance < min_distance:
                            min_distance = distance
                            destination = j
                    if devices_temp[lucky_user].locate_BS[destination] == 1:
                        energy_harvest_gain = config.ENERGY_CHANNEL_GAIN
                        energy_harvest = config.ENERGY_CONVERSION_EFFICIENCY * edges_temp[
                            destination].trans_power * energy_harvest_gain * config.time_slot_length
                        # print('energy_harvest', energy_harvest)
                    else:
                        energy_harvest = 0
                    # 能量效用
                    energy_utility = energy_cost - energy_harvest
                    # 获取权重
                    latency_weight, energy_weight = devices_temp[lucky_user].get_weight()
                    # 总效用
                    total_local_utility = latency_weight * latency_utility + energy_weight * energy_utility
                    this_total_utility = total_local_utility
                    if this_total_utility <= last_total_utility:
                        decisions.execute_mode[lucky_user] = 'local'
                        decisions.execute_destination[lucky_user] = -1
                        decisions.local_computing_portion[lucky_user] = 1
                elif execute_mode == 'edge':
                    execute_edge_id = execute_destination
                    execute_edge = edges_temp[execute_edge_id]
                    interference = 0
                    for j in range(0, config.total_number_of_devices):
                        if j != lucky_user and decisions.execute_mode[j] == 'edge' and decisions.execute_destination[
                            j] == execute_edge_id:
                            gain = get_upload_gain(device=devices_temp[j], edge=execute_edge)
                            interference += gain
                    upload_rate = get_upload_rate(device=devices_temp[lucky_user], edge=execute_edge,
                                                  interference=interference)
                    slot_upload_data_size = upload_rate * config.time_slot_length
                    slot_upload_data_size = round(slot_upload_data_size, 6)
                    offload_data_size_up_limit = min(slot_upload_data_size, data_size)
                    offload_computing_portion_up_limit = offload_data_size_up_limit / data_size
                    offload_computing_portion_up_limit = round(offload_computing_portion_up_limit, 6)
                    X = numpy.random.uniform(low=0, high=offload_computing_portion_up_limit, size=config.pop_size)
                    X[0] = 0
                    X[1] = offload_computing_portion_up_limit
                    Y = [0 for m in range(len(X))]
                    for j in range(len(X)):
                        offload_computing_portion = X[j]
                        if offload_computing_portion > 0:
                            offload_data_size = data_size * offload_computing_portion
                            # 传输时间
                            edge_transmit_latency = offload_data_size / upload_rate
                            edge_transmit_latency = round(edge_transmit_latency, 6)
                            # print('edge_transmit_latency', edge_transmit_latency)
                            # 传输能耗
                            trans_energy_cost = edge_transmit_latency * devices_temp[lucky_user].offload_trans_power
                        elif offload_computing_portion == 0:
                            # 传输时间
                            edge_transmit_latency = 0
                            # 传输能耗
                            trans_energy_cost = 0
                        else:
                            print('error')
                        # 收集能量
                        min_distance = 4000
                        destination = -1
                        for k in range(0, len(devices_temp[lucky_user].distance_BS)):
                            distance = devices_temp[lucky_user].distance_BS[k]
                            if distance < min_distance:
                                min_distance = distance
                                destination = k
                        if devices_temp[lucky_user].locate_BS[destination] == 1:
                            energy_harvest_gain = config.ENERGY_CHANNEL_GAIN
                            energy_harvest_slot_length = config.time_slot_length - edge_transmit_latency
                            # print('energy harvest slot length =', energy_harvest_slot_length)
                            energy_harvest = config.ENERGY_CONVERSION_EFFICIENCY * edges_temp[
                                destination].trans_power * energy_harvest_gain * energy_harvest_slot_length
                            # print('energy harvest =', energy_harvest)
                        else:
                            energy_harvest = 0
                        # 本地计算时间
                        local_computing_portion = 1 - offload_computing_portion
                        if local_computing_portion > 0:
                            local_computing_cpu_demand = cpu_frequency_demand * local_computing_portion
                            local_computing_size = devices_temp[
                                                       lucky_user].task_queue_length() + local_computing_cpu_demand
                            # 本地计算时间
                            local_execute_latency = local_computing_size / devices_temp[lucky_user].frequency
                            # 本地计算能耗
                            local_computing_power = config.SWITCHED_CAPACITANCE * math.pow(
                                devices_temp[lucky_user].frequency, 3)
                            local_computing_latency = local_computing_cpu_demand / devices_temp[lucky_user].frequency
                            local_computing_energy_cost = local_computing_latency * local_computing_power
                        elif local_computing_portion == 0:
                            local_execute_latency = 0
                            local_computing_energy_cost = 0
                        else:
                            print('error')
                        # 能量消耗
                        energy_cost = local_computing_energy_cost + trans_energy_cost
                        # 能量效用
                        energy_utility = energy_cost - energy_harvest
                        # print(energy_utility)
                        # 边缘计算执行时间
                        if offload_computing_portion > 0:
                            offload_computing_cpu_demand = cpu_frequency_demand * offload_computing_portion
                            edge_queue_length = execute_edge.task_queue_length() + offload_computing_cpu_demand
                            edge_computing_task_latency = edge_queue_length / execute_edge.frequency
                        elif offload_computing_portion == 0:
                            edge_computing_task_latency = 0
                        else:
                            print('error')
                        edge_execute_latency = edge_transmit_latency + edge_computing_task_latency
                        # 延迟效用
                        latency_utility = max(local_execute_latency, edge_execute_latency)
                        # 获取权重
                        latency_weight, energy_weight = devices_temp[lucky_user].get_weight()
                        # 总效用
                        total_edge_utility = latency_weight * latency_utility + energy_weight * energy_utility
                        #
                        Y[j] = total_edge_utility
                    pbest_x = X.copy()  # personal best location of every particle in history
                    # self.pbest_x = self.X 表示地址传递,改变 X 值 pbest_x 也会变化
                    pbest_y = [numpy.inf for m in range(config.pop_size)]  # best image of every particle in history
                    # self.gbest_x = self.pbest_x.mean(axis=0).reshape(1, -1)  # global best location for all particles
                    gbest_x = pbest_x.mean(axis=0)
                    gbest_y = numpy.inf  # global best y for all particles
                    gbest_y_hist = []  # gbest_y of every iteration
                    for j in range(config.max_iter):
                        # # update
                        # r1 = config.a - j * (config.a / config.max_iter)
                        # 抛物线函数
                        iter_period = j / config.max_iter
                        inter_rest_phase = 1 - iter_period
                        square = pow(inter_rest_phase, 2)
                        r1 = config.a * square
                        for k in range(config.pop_size):
                            r2 = 2 * math.pi * random.uniform(0.0, 1.0)
                            r3 = 2 * random.uniform(0.0, 1.0)
                            r4 = random.uniform(0.0, 1.0)
                            if r4 < 0.5:
                                X[k] = X[k] + (r1 * math.sin(r2) * abs(r3 * gbest_x - X[k]))
                            else:
                                X[k] = X[k] + (r1 * math.cos(r2) * abs(r3 * gbest_x - X[k]))
                        X = numpy.clip(a=X, a_min=0, a_max=offload_computing_portion_up_limit)
                        for k in range(len(X)):
                            offload_computing_portion = X[k]
                            if offload_computing_portion > 0:
                                offload_data_size = data_size * offload_computing_portion
                                # 传输时间
                                edge_transmit_latency = offload_data_size / upload_rate
                                edge_transmit_latency = round(edge_transmit_latency, 6)
                                # print('edge_transmit_latency', edge_transmit_latency)
                                # 传输能耗
                                trans_energy_cost = edge_transmit_latency * devices_temp[lucky_user].offload_trans_power
                            elif offload_computing_portion == 0:
                                # 传输时间
                                edge_transmit_latency = 0
                                # 传输能耗
                                trans_energy_cost = 0
                            else:
                                print('error')
                            # 收集能量
                            min_distance = 4000
                            destination = -1
                            for m in range(0, len(devices_temp[lucky_user].distance_BS)):
                                distance = devices_temp[lucky_user].distance_BS[m]
                                if distance < min_distance:
                                    min_distance = distance
                                    destination = m
                            if devices_temp[lucky_user].locate_BS[destination] == 1:
                                energy_harvest_gain = config.ENERGY_CHANNEL_GAIN
                                energy_harvest_slot_length = config.time_slot_length - edge_transmit_latency
                                # print('energy harvest slot length =', energy_harvest_slot_length)
                                energy_harvest = config.ENERGY_CONVERSION_EFFICIENCY * edges_temp[
                                    destination].trans_power * energy_harvest_gain * energy_harvest_slot_length
                                # print('energy harvest ', energy_harvest)
                            else:
                                energy_harvest = 0
                            # 本地计算时间
                            local_computing_portion = 1 - offload_computing_portion
                            if local_computing_portion > 0:
                                local_computing_cpu_demand = cpu_frequency_demand * local_computing_portion
                                local_computing_size = devices_temp[
                                                           lucky_user].task_queue_length() + local_computing_cpu_demand
                                # 本地计算时间
                                local_execute_latency = local_computing_size / devices_temp[lucky_user].frequency
                                # 本地计算能耗
                                local_computing_power = config.SWITCHED_CAPACITANCE * math.pow(
                                    devices_temp[lucky_user].frequency, 3)
                                local_computing_latency = local_computing_cpu_demand / devices_temp[
                                    lucky_user].frequency
                                local_computing_energy_cost = local_computing_latency * local_computing_power
                            elif local_computing_portion == 0:
                                local_execute_latency = 0
                                local_computing_energy_cost = 0
                            else:
                                print('error')
                            # 能量消耗
                            energy_cost = local_computing_energy_cost + trans_energy_cost
                            # 能量效用
                            energy_utility = energy_cost - energy_harvest
                            # print(energy_utility)
                            # 边缘计算执行时间
                            if offload_computing_portion > 0:
                                offload_computing_cpu_demand = cpu_frequency_demand * offload_computing_portion
                                edge_queue_length = execute_edge.task_queue_length() + offload_computing_cpu_demand
                                edge_computing_task_latency = edge_queue_length / execute_edge.frequency
                            elif offload_computing_portion == 0:
                                edge_computing_task_latency = 0
                            else:
                                print('error')
                            edge_execute_latency = edge_transmit_latency + edge_computing_task_latency
                            # 延迟效用
                            latency_utility = max(local_execute_latency, edge_execute_latency)
                            # 获取权重
                            latency_weight, energy_weight = devices_temp[lucky_user].get_weight()
                            # 总效用
                            total_edge_utility = latency_weight * latency_utility + energy_weight * energy_utility
                            #
                            Y[k] = total_edge_utility
                        # update_pbest
                        for k in range(len(Y)):
                            if pbest_y[k] > Y[k]:
                                pbest_x[k] = X[k].copy()
                                pbest_y[k] = Y[k].copy()
                        # update_gbest
                        idx_min = pbest_y.index(min(pbest_y))
                        if gbest_y > pbest_y[idx_min]:
                            gbest_x = pbest_x[idx_min].copy()  # copy很重要！
                            gbest_y = pbest_y[idx_min]
                        gbest_y_hist.append(gbest_y)
                    this_total_utility = gbest_y
                    # plt.plot(gbest_y_hist)
                    # plt.show()
                    if this_total_utility <= last_total_utility:
                        decisions.execute_mode[lucky_user] = 'edge'
                        decisions.execute_destination[lucky_user] = execute_edge_id
                        decisions.local_computing_portion[lucky_user] = 1 - gbest_x
            elif rand == 3:
                choices = [0 for m in range(0, config.total_number_of_devices)]
                for j in range(0, config.total_number_of_edges):
                    if devices_temp[lucky_user].locate_BS[j] == 1:
                        for k in range(0, config.total_number_of_devices):
                            if edges_temp[j].coverage_mobile_device[k] == 1:
                                choices[k] = 1
                choices_number = 0
                for j in range(0, config.total_number_of_devices):
                    if choices[j] == 1:
                        choices_number = choices_number + 1
                if choices_number == 0:
                    # 不会走
                    execute_mode = 'local'
                    execute_destination = -1
                elif choices_number > 0:
                    lucky_number = numpy.random.randint(1, choices_number + 1)
                    for j in range(0, config.total_number_of_devices):
                        if choices[j] == 1:
                            lucky_number = lucky_number - 1
                            if lucky_number == 0:
                                if lucky_user == j:
                                    execute_mode = 'local'
                                    execute_destination = -1
                                elif lucky_user != j:
                                    execute_mode = 'device'
                                    execute_destination = j
                                else:
                                    print('error')
                else:
                    print('error')
                # 计算效用
                if execute_mode == 'local':
                    local_computing_size = devices_temp[lucky_user].task_queue_length() + cpu_frequency_demand
                    local_execute_latency = local_computing_size / devices_temp[lucky_user].frequency
                    latency_utility = local_execute_latency
                    # 本地计算能耗
                    local_computing_power = config.SWITCHED_CAPACITANCE * math.pow(devices_temp[lucky_user].frequency,
                                                                                   3)
                    local_computing_latency = cpu_frequency_demand / devices_temp[lucky_user].frequency
                    energy_cost = local_computing_latency * local_computing_power
                    # 收集能量
                    min_distance = 4000
                    destination = -1
                    for j in range(0, len(devices_temp[lucky_user].distance_BS)):
                        distance = devices_temp[lucky_user].distance_BS[j]
                        if distance < min_distance:
                            min_distance = distance
                            destination = j
                    if devices_temp[lucky_user].locate_BS[destination] == 1:
                        energy_harvest_gain = config.ENERGY_CHANNEL_GAIN
                        energy_harvest = config.ENERGY_CONVERSION_EFFICIENCY * edges_temp[
                            destination].trans_power * energy_harvest_gain * config.time_slot_length
                        # print('energy_harvest', energy_harvest)
                    else:
                        energy_harvest = 0
                    # 能量效用
                    energy_utility = energy_cost - energy_harvest
                    # 获取权重
                    latency_weight, energy_weight = devices_temp[lucky_user].get_weight()
                    # 总效用
                    total_local_utility = latency_weight * latency_utility + energy_weight * energy_utility
                    this_total_utility = total_local_utility
                    if this_total_utility <= last_total_utility:
                        decisions.execute_mode[lucky_user] = 'local'
                        decisions.execute_destination[lucky_user] = -1
                        decisions.local_computing_portion[lucky_user] = 1
                elif execute_mode == 'device':
                    execute_device_id = execute_destination
                    execute_device = devices_temp[execute_device_id]
                    interference = 0
                    for j in range(0, config.total_number_of_devices):
                        if j != lucky_user and decisions.execute_mode[j] == 'device' and decisions.execute_destination[
                            j] == execute_device_id:
                            gain = get_d2d_gain(device1=devices_temp[j], device2=execute_device)
                            interference += gain
                    d2d_rate = get_d2d_rate(device1=devices_temp[lucky_user], device2=execute_device,
                                            interference=interference)
                    slot_d2d_data_size = d2d_rate * config.time_slot_length
                    d2d_data_size_up_limit = min(slot_d2d_data_size, data_size)
                    d2d_computing_portion_up_limit = d2d_data_size_up_limit / data_size
                    X = numpy.random.uniform(low=0, high=d2d_computing_portion_up_limit, size=config.pop_size)
                    X[0] = 0
                    X[1] = d2d_computing_portion_up_limit
                    Y = [0 for m in range(len(X))]
                    for j in range(len(X)):
                        d2d_computing_portion = X[j]
                        if d2d_computing_portion > 0:
                            d2d_data_size = data_size * d2d_computing_portion
                            # 传输时间
                            d2d_transmit_latency = d2d_data_size / d2d_rate
                            d2d_transmit_latency = round(d2d_transmit_latency, 6)
                            # print('d2d_transmit_latency', d2d_transmit_latency)
                            # 传输能耗
                            trans_energy_cost = d2d_transmit_latency * devices_temp[lucky_user].d2d_trans_power
                        elif d2d_computing_portion == 0:
                            # 传输时间
                            d2d_transmit_latency = 0
                            # 传输能耗
                            trans_energy_cost = 0
                        else:
                            print('error')
                        # 收集能量
                        min_distance = 4000
                        destination = -1
                        for k in range(0, len(devices_temp[lucky_user].distance_BS)):
                            distance = devices_temp[lucky_user].distance_BS[k]
                            if distance < min_distance:
                                min_distance = distance
                                destination = k
                        if devices_temp[lucky_user].locate_BS[destination] == 1:
                            energy_harvest_gain = config.ENERGY_CHANNEL_GAIN
                            energy_harvest_slot_length = config.time_slot_length - d2d_transmit_latency
                            # print('energy harvest slot length =', energy_harvest_slot_length)
                            energy_harvest = config.ENERGY_CONVERSION_EFFICIENCY * edges_temp[
                                destination].trans_power * energy_harvest_gain * energy_harvest_slot_length
                            # print('energy harvest ', energy_harvest)
                        else:
                            energy_harvest = 0
                        # 本地计算时间
                        local_computing_portion = 1 - d2d_computing_portion
                        if local_computing_portion > 0:
                            local_computing_cpu_demand = cpu_frequency_demand * local_computing_portion
                            local_computing_size = devices_temp[
                                                       lucky_user].task_queue_length() + local_computing_cpu_demand
                            # 本地计算时间
                            local_execute_latency = local_computing_size / devices_temp[lucky_user].frequency
                            # 本地计算能耗
                            local_computing_power = config.SWITCHED_CAPACITANCE * math.pow(
                                devices_temp[lucky_user].frequency, 3)
                            local_computing_latency = local_computing_cpu_demand / devices_temp[lucky_user].frequency
                            local_computing_energy_cost = local_computing_latency * local_computing_power
                        elif local_computing_portion == 0:
                            local_execute_latency = 0
                            local_computing_energy_cost = 0
                        else:
                            print('error')
                        # 能量消耗
                        energy_cost = local_computing_energy_cost + trans_energy_cost
                        # 能量效用
                        energy_utility = energy_cost - energy_harvest
                        # print(energy_utility)
                        # 边缘计算执行时间
                        if d2d_computing_portion > 0:
                            d2d_cpu_frequency_demand = cpu_frequency_demand * d2d_computing_portion
                            device_queue_length = execute_device.task_queue_length() + d2d_cpu_frequency_demand
                            d2d_computing_task_latency = device_queue_length / execute_device.frequency
                        elif d2d_computing_portion == 0:
                            d2d_computing_task_latency = 0
                        else:
                            print('error')
                        d2d_execute_latency = d2d_transmit_latency + d2d_computing_task_latency
                        # 延迟效用
                        latency_utility = max(local_execute_latency, d2d_execute_latency)
                        # 获取权重
                        latency_weight, energy_weight = devices_temp[lucky_user].get_weight()
                        # 总效用
                        total_d2d_utility = latency_weight * latency_utility + energy_weight * energy_utility
                        #
                        Y[j] = total_d2d_utility
                    pbest_x = X.copy()  # personal best location of every particle in history
                    # self.pbest_x = self.X 表示地址传递,改变 X 值 pbest_x 也会变化
                    pbest_y = [numpy.inf for m in range(config.pop_size)]  # best image of every particle in history
                    # self.gbest_x = self.pbest_x.mean(axis=0).reshape(1, -1)  # global best location for all particles
                    gbest_x = pbest_x.mean(axis=0)
                    gbest_y = numpy.inf  # global best y for all particles
                    gbest_y_hist = []  # gbest_y of every iteration
                    for j in range(config.max_iter):
                        # # update
                        # r1 = config.a - j * (config.a / config.max_iter)
                        # 抛物线函数
                        iter_period = j / config.max_iter
                        inter_rest_phase = 1 - iter_period
                        square = pow(inter_rest_phase, 2)
                        r1 = config.a * square
                        for k in range(config.pop_size):
                            r2 = 2 * math.pi * random.uniform(0.0, 1.0)
                            r3 = 2 * random.uniform(0.0, 1.0)
                            r4 = random.uniform(0.0, 1.0)
                            if r4 < 0.5:
                                X[k] = X[k] + (r1 * math.sin(r2) * abs(r3 * gbest_x - X[k]))
                            else:
                                X[k] = X[k] + (r1 * math.cos(r2) * abs(r3 * gbest_x - X[k]))
                        X = numpy.clip(a=X, a_min=0, a_max=d2d_computing_portion_up_limit)
                        for k in range(len(X)):
                            d2d_computing_portion = X[k]
                            if d2d_computing_portion > 0:
                                d2d_data_size = data_size * d2d_computing_portion
                                # 传输时间
                                d2d_transmit_latency = d2d_data_size / d2d_rate
                                d2d_transmit_latency = round(d2d_transmit_latency, 6)
                                # print('d2d_transmit_latency', d2d_transmit_latency)
                                # 传输能耗
                                trans_energy_cost = d2d_transmit_latency * devices_temp[lucky_user].d2d_trans_power
                            elif d2d_computing_portion == 0:
                                # 传输时间
                                d2d_transmit_latency = 0
                                # 传输能耗
                                trans_energy_cost = 0
                            else:
                                print('error')
                            # 收集能量
                            min_distance = 4000
                            destination = -1
                            for m in range(0, len(devices_temp[lucky_user].distance_BS)):
                                distance = devices_temp[lucky_user].distance_BS[m]
                                if distance < min_distance:
                                    min_distance = distance
                                    destination = m
                            if devices_temp[lucky_user].locate_BS[destination] == 1:
                                energy_harvest_gain = config.ENERGY_CHANNEL_GAIN
                                energy_harvest_slot_length = config.time_slot_length - d2d_transmit_latency
                                # print('energy harvest slot length =', energy_harvest_slot_length)
                                energy_harvest = config.ENERGY_CONVERSION_EFFICIENCY * edges_temp[
                                    destination].trans_power * energy_harvest_gain * energy_harvest_slot_length
                                # print('energy harvest ', energy_harvest)
                            else:
                                energy_harvest = 0
                            # 本地计算时间
                            local_computing_portion = 1 - d2d_computing_portion
                            if local_computing_portion > 0:
                                local_computing_cpu_demand = cpu_frequency_demand * local_computing_portion
                                local_computing_size = devices_temp[
                                                           lucky_user].task_queue_length() + local_computing_cpu_demand
                                # 本地计算时间
                                local_execute_latency = local_computing_size / devices_temp[lucky_user].frequency
                                # 本地计算能耗
                                local_computing_power = config.SWITCHED_CAPACITANCE * math.pow(
                                    devices_temp[lucky_user].frequency, 3)
                                local_computing_latency = local_computing_cpu_demand / devices_temp[
                                    lucky_user].frequency
                                local_computing_energy_cost = local_computing_latency * local_computing_power
                            elif local_computing_portion == 0:
                                local_execute_latency = 0
                                local_computing_energy_cost = 0
                            else:
                                print('error')
                            # 能量消耗
                            energy_cost = local_computing_energy_cost + trans_energy_cost
                            # 能量效用
                            energy_utility = energy_cost - energy_harvest
                            # print(energy_utility)
                            # 边缘计算执行时间
                            if d2d_computing_portion > 0:
                                d2d_cpu_frequency_demand = cpu_frequency_demand * d2d_computing_portion
                                device_queue_length = execute_device.task_queue_length() + d2d_cpu_frequency_demand
                                d2d_computing_task_latency = device_queue_length / execute_device.frequency
                            elif d2d_computing_portion == 0:
                                d2d_computing_task_latency = 0
                            else:
                                print('error')
                            d2d_execute_latency = d2d_transmit_latency + d2d_computing_task_latency
                            # 延迟效用
                            latency_utility = max(local_execute_latency, d2d_execute_latency)
                            # 获取权重
                            latency_weight, energy_weight = devices_temp[lucky_user].get_weight()
                            # 总效用
                            total_d2d_utility = latency_weight * latency_utility + energy_weight * energy_utility
                            #
                            Y[k] = total_d2d_utility
                        # update_pbest
                        for k in range(len(Y)):
                            if pbest_y[k] > Y[k]:
                                pbest_x[k] = X[k].copy()
                                pbest_y[k] = Y[k].copy()
                        # update_gbest
                        idx_min = pbest_y.index(min(pbest_y))
                        if gbest_y > pbest_y[idx_min]:
                            gbest_x = pbest_x[idx_min].copy()  # copy很重要！
                            gbest_y = pbest_y[idx_min].copy()
                        gbest_y_hist.append(gbest_y)
                    this_total_utility = gbest_y
                    # plt.plot(gbest_y_hist)
                    # plt.show()
                    if this_total_utility <= last_total_utility:
                        decisions.execute_mode[lucky_user] = 'device'
                        decisions.execute_destination[lucky_user] = execute_device_id
                        decisions.local_computing_portion[lucky_user] = 1 - gbest_x
        elif decisions.execute_mode[lucky_user] == 'null':
            pass
        else:
            pass
    return decisions


def proposed_algorithm_v2(decisions, config, devices, edges, time_slot, task_in_each_slot):
    decisions = random_algorithm(decisions, config, devices, edges, time_slot, task_in_each_slot)
    for i in range(0, config.iterations_number_of_game_theory):
        lucky_user = numpy.random.randint(0, config.total_number_of_devices)
        while decisions.execute_mode[lucky_user] == 'null':
            lucky_user = numpy.random.randint(0, config.total_number_of_devices)
        if decisions.execute_mode[lucky_user] != 'null':
            devices_temp = deepcopy(devices)
            edges_temp = deepcopy(edges)
            for j in range(0, lucky_user):
                if decisions.execute_mode[j] == 'local':
                    cpu_frequency_demand = task_in_each_slot[time_slot][j].cpu_frequency_demand
                    devices_temp[j].task_enqueue(cpu_frequency_demand)
                elif decisions.execute_mode[j] == 'edge':
                    execute_destination = decisions.execute_destination[j]
                    data_size = task_in_each_slot[time_slot][j].data_size
                    local_computing_portion = decisions.local_computing_portion[j]
                    edge_computing_portion = 1 - local_computing_portion
                    offload_data_size = data_size * edge_computing_portion
                    # 计算传输速率
                    interference = 0
                    for k in range(0, config.total_number_of_devices):
                        if k != j and decisions.execute_mode[k] == 'edge' and decisions.execute_destination[
                            k] == execute_destination:
                            gain = get_upload_gain(device=devices_temp[k], edge=edges_temp[execute_destination])
                            interference += gain
                    upload_rate = get_upload_rate(device=devices_temp[j], edge=edges_temp[execute_destination],
                                                  interference=interference)
                    # 传输时间
                    edge_transmit_latency = offload_data_size / upload_rate
                    edge_transmit_latency = round(edge_transmit_latency, 6)
                    if edge_transmit_latency >= config.time_slot_length:
                        edge_transmit_latency = config.time_slot_length
                    edge_transmit_data_size = edge_transmit_latency * upload_rate
                    edge_transmit_cpu_frequency_demand = edge_transmit_data_size * config.task_cpu_frequency_demand
                    edges_temp[execute_destination].task_enqueue(edge_transmit_cpu_frequency_demand)
                    local_computing_data_size = data_size - edge_transmit_data_size
                    if edge_transmit_latency == config.time_slot_length:
                        decisions.local_computing_portion[j] = local_computing_data_size / data_size
                    local_computing_cpu_frequency_demand = local_computing_data_size * config.task_cpu_frequency_demand
                    devices_temp[j].task_enqueue(local_computing_cpu_frequency_demand)
                elif decisions.execute_mode[j] == 'device':
                    execute_destination = decisions.execute_destination[j]
                    data_size = task_in_each_slot[time_slot][j].data_size
                    local_computing_portion = decisions.local_computing_portion[j]
                    d2d_computing_portion = 1 - local_computing_portion
                    d2d_data_size = data_size * d2d_computing_portion
                    # 计算传输速率
                    interference = 0
                    for k in range(0, config.total_number_of_devices):
                        if k != j and decisions.execute_mode[k] == 'device' and decisions.execute_destination[
                            k] == execute_destination:
                            gain = get_d2d_gain(device1=devices_temp[k], device2=devices_temp[execute_destination])
                            interference += gain
                    d2d_rate = get_d2d_rate(device1=devices_temp[j], device2=devices_temp[execute_destination],
                                            interference=interference)
                    # 传输时间
                    d2d_transmit_latency = d2d_data_size / d2d_rate
                    d2d_transmit_latency = round(d2d_transmit_latency, 6)
                    if d2d_transmit_latency >= config.time_slot_length:
                        d2d_transmit_latency = config.time_slot_length
                    d2d_transmit_data_size = d2d_transmit_latency * d2d_rate
                    d2d_transmit_cpu_frequency_demand = d2d_transmit_data_size * config.task_cpu_frequency_demand
                    devices_temp[execute_destination].task_enqueue(d2d_transmit_cpu_frequency_demand)
                    local_computing_data_size = data_size - d2d_transmit_data_size
                    if d2d_transmit_latency == config.time_slot_length:
                        decisions.local_computing_portion[j] = local_computing_data_size / data_size
                    local_computing_cpu_frequency_demand = local_computing_data_size * config.task_cpu_frequency_demand
                    devices_temp[j].task_enqueue(local_computing_cpu_frequency_demand)
            task = task_in_each_slot[time_slot][lucky_user]
            data_size = task_in_each_slot[time_slot][lucky_user].data_size
            cpu_frequency_demand = task_in_each_slot[time_slot][lucky_user].cpu_frequency_demand
            # 计算上一轮的效用
            last_total_utility = 0
            if decisions.execute_mode[lucky_user] == 'local':
                local_computing_size = devices_temp[lucky_user].task_queue_length() + cpu_frequency_demand
                local_execute_latency = local_computing_size / devices_temp[lucky_user].frequency
                latency_utility = local_execute_latency
                # 本地计算能耗
                local_computing_power = config.SWITCHED_CAPACITANCE * math.pow(devices_temp[lucky_user].frequency, 3)
                local_computing_latency = cpu_frequency_demand / devices_temp[lucky_user].frequency
                energy_cost = local_computing_latency * local_computing_power
                # 收集能量
                min_distance = 4000
                destination = -1
                for j in range(0, len(devices_temp[lucky_user].distance_BS)):
                    distance = devices_temp[lucky_user].distance_BS[j]
                    if distance < min_distance:
                        min_distance = distance
                        destination = j
                if devices_temp[lucky_user].locate_BS[destination] == 1:
                    energy_harvest_gain = config.ENERGY_CHANNEL_GAIN
                    energy_harvest = config.ENERGY_CONVERSION_EFFICIENCY * edges_temp[
                        destination].trans_power * energy_harvest_gain * config.time_slot_length
                else:
                    energy_harvest = 0
                # 能量效用
                energy_utility = energy_cost - energy_harvest
                # 获取权重
                latency_weight, energy_weight = devices_temp[lucky_user].get_weight()
                # 总效用
                total_local_utility = latency_weight * latency_utility + energy_weight * energy_utility
                lya = 0
                for j in range(0, config.total_number_of_devices):
                    # 计算lyapunov
                    cpu_frequency_demand_offload = 0
                    for k in range(0, config.total_number_of_devices):
                        if k != j and decisions.execute_mode[k] == 'device' and decisions.execute_destination[k] == j:
                            cpu_frequency_demand_lya = task_in_each_slot[time_slot][k].cpu_frequency_demand
                            local_computing_portion = decisions.local_computing_portion[k]
                            offload_computing_portion = (1 - local_computing_portion)
                            cpu_frequency_demand_offload = cpu_frequency_demand_offload + cpu_frequency_demand_lya * offload_computing_portion
                    if decisions.execute_mode[j] == 'local':
                        cpu_frequency_demand_offload = cpu_frequency_demand_offload + task_in_each_slot[time_slot][
                            j].cpu_frequency_demand
                    elif decisions.execute_mode[j] != 'null':
                        local_computing_portion = decisions.local_computing_portion[j]
                        cpu_frequency_demand_offload = cpu_frequency_demand_offload + task_in_each_slot[time_slot][
                            j].cpu_frequency_demand * local_computing_portion
                    cpu_frequency_demand_offload = cpu_frequency_demand_offload / config.task_cpu_frequency_demand
                    cpu_frequency_demand_offload = cpu_frequency_demand_offload / 8388608
                    queue_length = devices[j].task_queue_length() / config.task_cpu_frequency_demand
                    queue_length = queue_length / 8388608
                    lya = lya + pow(cpu_frequency_demand_offload, 2) + 2 * queue_length * cpu_frequency_demand_offload
                last_total_utility = total_local_utility + config.V * lya
            elif decisions.execute_mode[lucky_user] == 'edge':
                execute_edge_id = decisions.execute_destination[lucky_user]
                local_computing_portion = decisions.local_computing_portion[lucky_user]
                execute_edge = edges_temp[execute_edge_id]
                # 计算卸载数据大小
                offload_computing_portion = (1 - local_computing_portion)
                # print('offload_computing_portion', offload_computing_portion)
                if offload_computing_portion > 0:
                    offload_data_size = data_size * offload_computing_portion
                    # 计算传输速率
                    interference = 0
                    for j in range(0, config.total_number_of_devices):
                        if j != lucky_user and decisions.execute_mode[j] == 'edge' and decisions.execute_destination[
                            j] == execute_edge_id:
                            gain = get_upload_gain(device=devices_temp[j], edge=execute_edge)
                            interference += gain
                    upload_rate = get_upload_rate(device=devices_temp[lucky_user], edge=execute_edge,
                                                  interference=interference)
                    # 传输时间
                    edge_transmit_latency = offload_data_size / upload_rate
                    edge_transmit_latency = round(edge_transmit_latency, 6)
                    if edge_transmit_latency >= config.time_slot_length:
                        edge_transmit_latency = config.time_slot_length
                    # 传输能耗
                    trans_energy_cost = edge_transmit_latency * devices_temp[lucky_user].offload_trans_power
                elif offload_computing_portion == 0:
                    # 传输时间
                    edge_transmit_latency = 0
                    # 传输能耗
                    trans_energy_cost = 0
                else:
                    print('error')
                # 收集能量
                min_distance = 4000
                destination = -1
                for j in range(0, len(devices_temp[lucky_user].distance_BS)):
                    distance = devices_temp[lucky_user].distance_BS[j]
                    if distance < min_distance:
                        min_distance = distance
                        destination = j
                if devices_temp[lucky_user].locate_BS[destination] == 1:
                    energy_harvest_gain = config.ENERGY_CHANNEL_GAIN
                    energy_harvest_slot_length = config.time_slot_length - edge_transmit_latency
                    # print('energy harvest slot length =', energy_harvest_slot_length)
                    energy_harvest = config.ENERGY_CONVERSION_EFFICIENCY * edges_temp[
                        destination].trans_power * energy_harvest_gain * energy_harvest_slot_length
                    # print('energy harvest ', energy_harvest)
                else:
                    energy_harvest = 0
                # 本地计算时间
                if local_computing_portion > 0:
                    if edge_transmit_latency == config.time_slot_length:
                        edge_transmit_data_size = upload_rate * config.time_slot_length
                        local_computing_data_size = data_size - edge_transmit_data_size
                        decisions.local_computing_portion[lucky_user] = local_computing_data_size / data_size
                        local_computing_cpu_demand = local_computing_data_size * config.task_cpu_frequency_demand
                    else:
                        local_computing_cpu_demand = cpu_frequency_demand * local_computing_portion
                    local_computing_size = devices_temp[lucky_user].task_queue_length() + local_computing_cpu_demand
                    # 本地计算时间
                    local_execute_latency = local_computing_size / devices_temp[lucky_user].frequency
                    # 本地计算能耗
                    local_computing_power = config.SWITCHED_CAPACITANCE * math.pow(devices_temp[lucky_user].frequency,
                                                                                   3)
                    local_computing_latency = local_computing_cpu_demand / devices_temp[lucky_user].frequency
                    local_computing_energy_cost = local_computing_latency * local_computing_power
                elif local_computing_portion == 0:
                    local_computing_cpu_demand = 0
                    local_execute_latency = 0
                    local_computing_energy_cost = 0
                else:
                    print('error')
                # 能量消耗
                energy_cost = local_computing_energy_cost + trans_energy_cost
                # 能量效用
                energy_utility = energy_cost - energy_harvest
                # print(energy_utility)
                # 边缘计算执行时间
                if offload_computing_portion > 0:
                    offload_computing_cpu_demand = cpu_frequency_demand - local_computing_cpu_demand
                    edge_queue_length = execute_edge.task_queue_length() + offload_computing_cpu_demand
                    edge_computing_task_latency = edge_queue_length / execute_edge.frequency
                elif offload_computing_portion == 0:
                    edge_computing_task_latency = 0
                else:
                    print('error')
                edge_execute_latency = edge_transmit_latency + edge_computing_task_latency
                # 延迟效用
                latency_utility = max(local_execute_latency, edge_execute_latency)
                # print('energy queue', devices_temp[lucky_user].energy_queue)
                # 获取权重
                latency_weight, energy_weight = devices_temp[lucky_user].get_weight()
                # 总效用
                total_edge_utility = latency_weight * latency_utility + energy_weight * energy_utility
                lya = 0
                for j in range(0, config.total_number_of_devices):
                    # 计算lyapunov
                    cpu_frequency_demand_offload = 0
                    for k in range(0, config.total_number_of_devices):
                        if k != j and decisions.execute_mode[k] == 'device' and decisions.execute_destination[k] == j:
                            cpu_frequency_demand_lya = task_in_each_slot[time_slot][k].cpu_frequency_demand
                            local_computing_portion = decisions.local_computing_portion[k]
                            offload_computing_portion = (1 - local_computing_portion)
                            cpu_frequency_demand_offload = cpu_frequency_demand_offload + cpu_frequency_demand_lya * offload_computing_portion
                    if decisions.execute_mode[j] == 'local':
                        cpu_frequency_demand_offload = cpu_frequency_demand_offload + task_in_each_slot[time_slot][
                            j].cpu_frequency_demand
                    elif decisions.execute_mode[j] != 'null':
                        local_computing_portion = decisions.local_computing_portion[j]
                        cpu_frequency_demand_offload = cpu_frequency_demand_offload + task_in_each_slot[time_slot][
                            j].cpu_frequency_demand * local_computing_portion
                    cpu_frequency_demand_offload = cpu_frequency_demand_offload / config.task_cpu_frequency_demand
                    cpu_frequency_demand_offload = cpu_frequency_demand_offload / 8388608
                    queue_length = devices[j].task_queue_length() / config.task_cpu_frequency_demand
                    queue_length = queue_length / 8388608
                    lya = lya + pow(cpu_frequency_demand_offload, 2) + 2 * queue_length * cpu_frequency_demand_offload
                last_total_utility = total_edge_utility + config.V * lya
            elif decisions.execute_mode[lucky_user] == 'device':
                execute_device_id = decisions.execute_destination[lucky_user]
                local_computing_portion = decisions.local_computing_portion[lucky_user]
                execute_device = devices_temp[execute_device_id]
                # 计算卸载数据大小
                offload_computing_portion = (1 - local_computing_portion)
                if offload_computing_portion > 0:
                    offload_data_size = data_size * offload_computing_portion
                    # 计算传输速率
                    interference = 0
                    for j in range(0, config.total_number_of_devices):
                        if j != lucky_user and decisions.execute_mode[j] == 'device' and decisions.execute_destination[
                            j] == execute_device_id:
                            gain = get_d2d_gain(device1=devices_temp[j], device2=execute_device)
                            interference += gain
                    d2d_rate = get_d2d_rate(device1=devices_temp[lucky_user], device2=execute_device,
                                            interference=interference)
                    # 传输时间
                    d2d_transmit_latency = offload_data_size / d2d_rate
                    d2d_transmit_latency = round(d2d_transmit_latency, 6)
                    if d2d_transmit_latency >= config.time_slot_length:
                        d2d_transmit_latency = config.time_slot_length
                    # 传输能耗
                    trans_energy_cost = d2d_transmit_latency * devices_temp[lucky_user].d2d_trans_power
                elif offload_computing_portion == 0:
                    # 传输时间
                    d2d_transmit_latency = 0
                    # 传输能耗
                    trans_energy_cost = 0
                else:
                    print('error')
                # 收集能量
                min_distance = 4000
                destination = -1
                for j in range(0, len(devices_temp[lucky_user].distance_BS)):
                    distance = devices_temp[lucky_user].distance_BS[j]
                    if distance < min_distance:
                        min_distance = distance
                        destination = j
                if devices_temp[lucky_user].locate_BS[destination] == 1:
                    energy_harvest_gain = config.ENERGY_CHANNEL_GAIN
                    energy_harvest_slot_length = config.time_slot_length - d2d_transmit_latency
                    energy_harvest = config.ENERGY_CONVERSION_EFFICIENCY * edges_temp[
                        destination].trans_power * energy_harvest_gain * energy_harvest_slot_length
                else:
                    energy_harvest = 0
                # 本地计算时间
                if local_computing_portion > 0:
                    if d2d_transmit_latency == config.time_slot_length:
                        d2d_transmit_data_size = d2d_rate * config.time_slot_length
                        local_computing_data_size = data_size - d2d_transmit_data_size
                        decisions.local_computing_portion[lucky_user] = local_computing_data_size / data_size
                        local_computing_cpu_demand = local_computing_data_size * config.task_cpu_frequency_demand
                    else:
                        local_computing_cpu_demand = cpu_frequency_demand * local_computing_portion
                    local_computing_size = devices_temp[lucky_user].task_queue_length() + local_computing_cpu_demand
                    # 本地计算时间
                    local_execute_latency = local_computing_size / devices_temp[lucky_user].frequency
                    # 本地计算能耗
                    local_computing_power = config.SWITCHED_CAPACITANCE * math.pow(devices_temp[lucky_user].frequency,
                                                                                   3)
                    local_computing_latency = local_computing_cpu_demand / devices_temp[lucky_user].frequency
                    local_computing_energy_cost = local_computing_latency * local_computing_power
                elif local_computing_portion == 0:
                    local_computing_cpu_demand = 0
                    local_execute_latency = 0
                    local_computing_energy_cost = 0
                else:
                    print('error')
                # 能量消耗
                energy_cost = local_computing_energy_cost + trans_energy_cost
                # 能量效用
                energy_utility = energy_cost - energy_harvest
                # 边缘计算执行时间
                if offload_computing_portion > 0:
                    offload_computing_cpu_demand = cpu_frequency_demand - local_computing_cpu_demand
                    d2d_queue_length = execute_device.task_queue_length() + offload_computing_cpu_demand
                    d2d_compute_task_latency = d2d_queue_length / execute_device.frequency
                elif offload_computing_portion == 0:
                    d2d_compute_task_latency = 0
                else:
                    print('error')
                d2d_execute_latency = d2d_transmit_latency + d2d_compute_task_latency
                # 延迟效用
                latency_utility = max(local_execute_latency, d2d_execute_latency)
                # 获取权重
                latency_weight, energy_weight = devices_temp[lucky_user].get_weight()
                # 总效用
                total_d2d_utility = latency_weight * latency_utility + energy_weight * energy_utility
                lya = 0
                for j in range(0, config.total_number_of_devices):
                    # 计算lyapunov
                    cpu_frequency_demand_offload = 0
                    for k in range(0, config.total_number_of_devices):
                        if k != j and decisions.execute_mode[k] == 'device' and decisions.execute_destination[k] == j:
                            cpu_frequency_demand_lya = task_in_each_slot[time_slot][k].cpu_frequency_demand
                            local_computing_portion = decisions.local_computing_portion[k]
                            offload_computing_portion = (1 - local_computing_portion)
                            cpu_frequency_demand_offload = cpu_frequency_demand_offload + cpu_frequency_demand_lya * offload_computing_portion
                    if decisions.execute_mode[j] == 'local':
                        cpu_frequency_demand_offload = cpu_frequency_demand_offload + task_in_each_slot[time_slot][
                            j].cpu_frequency_demand
                    elif decisions.execute_mode[j] != 'null':
                        local_computing_portion = decisions.local_computing_portion[j]
                        cpu_frequency_demand_offload = cpu_frequency_demand_offload + task_in_each_slot[time_slot][
                            j].cpu_frequency_demand * local_computing_portion
                    cpu_frequency_demand_offload = cpu_frequency_demand_offload / config.task_cpu_frequency_demand
                    cpu_frequency_demand_offload = cpu_frequency_demand_offload / 8388608
                    queue_length = devices[j].task_queue_length() / config.task_cpu_frequency_demand
                    queue_length = queue_length / 8388608
                    lya = lya + pow(cpu_frequency_demand_offload, 2) + 2 * queue_length * cpu_frequency_demand_offload
                last_total_utility = total_d2d_utility + config.V * lya
            # this interation
            rand = numpy.random.randint(1, 4)
            if rand == 1:
                local_computing_size = devices_temp[lucky_user].task_queue_length() + cpu_frequency_demand
                local_execute_latency = local_computing_size / devices_temp[lucky_user].frequency
                latency_utility = local_execute_latency
                # 本地计算能耗
                local_computing_power = config.SWITCHED_CAPACITANCE * math.pow(devices_temp[lucky_user].frequency, 3)
                local_computing_latency = cpu_frequency_demand / devices_temp[lucky_user].frequency
                energy_cost = local_computing_latency * local_computing_power
                # 收集能量
                min_distance = 4000
                destination = -1
                for j in range(0, len(devices_temp[lucky_user].distance_BS)):
                    distance = devices_temp[lucky_user].distance_BS[j]
                    if distance < min_distance:
                        min_distance = distance
                        destination = j
                if devices_temp[lucky_user].locate_BS[destination] == 1:
                    energy_harvest_gain = config.ENERGY_CHANNEL_GAIN
                    energy_harvest = config.ENERGY_CONVERSION_EFFICIENCY * edges_temp[
                        destination].trans_power * energy_harvest_gain * config.time_slot_length
                    # print('energy_harvest', energy_harvest)
                else:
                    energy_harvest = 0
                # 能量效用
                energy_utility = energy_cost - energy_harvest
                # 获取权重
                latency_weight, energy_weight = devices_temp[lucky_user].get_weight()
                # 总效用
                total_local_utility = latency_weight * latency_utility + energy_weight * energy_utility
                lya = 0
                for j in range(0, config.total_number_of_devices):
                    # 计算lyapunov
                    cpu_frequency_demand_offload = 0
                    for k in range(0, config.total_number_of_devices):
                        if k != j and decisions.execute_mode[k] == 'device' and decisions.execute_destination[k] == j:
                            cpu_frequency_demand_lya = task_in_each_slot[time_slot][k].cpu_frequency_demand
                            local_computing_portion = decisions.local_computing_portion[k]
                            offload_computing_portion = (1 - local_computing_portion)
                            cpu_frequency_demand_offload = cpu_frequency_demand_offload + cpu_frequency_demand_lya * offload_computing_portion
                    if decisions.execute_mode[j] == 'local':
                        cpu_frequency_demand_offload = cpu_frequency_demand_offload + task_in_each_slot[time_slot][
                            j].cpu_frequency_demand
                    elif decisions.execute_mode[j] != 'null':
                        local_computing_portion = decisions.local_computing_portion[j]
                        cpu_frequency_demand_offload = cpu_frequency_demand_offload + task_in_each_slot[time_slot][
                            j].cpu_frequency_demand * local_computing_portion
                    cpu_frequency_demand_offload = cpu_frequency_demand_offload / config.task_cpu_frequency_demand
                    cpu_frequency_demand_offload = cpu_frequency_demand_offload / 8388608
                    queue_length = devices[j].task_queue_length() / config.task_cpu_frequency_demand
                    queue_length = queue_length / 8388608
                    lya = lya + pow(cpu_frequency_demand_offload, 2) + 2 * queue_length * cpu_frequency_demand_offload
                this_total_utility = total_local_utility + config.V * lya
                if this_total_utility <= last_total_utility:
                    decisions.execute_mode[lucky_user] = 'local'
                    decisions.execute_destination[lucky_user] = -1
                    decisions.local_computing_portion[lucky_user] = 1
            elif rand == 2:
                # 随机选动作
                choices_number = 0
                for j in range(0, config.total_number_of_edges):
                    if devices_temp[lucky_user].locate_BS[j] == 1:
                        choices_number = choices_number + 1
                if choices_number == 0:
                    execute_mode = 'local'
                    execute_destination = -1
                elif choices_number > 0:
                    lucky_number = numpy.random.randint(1, choices_number + 1)
                    for j in range(0, config.total_number_of_edges):
                        if devices_temp[lucky_user].locate_BS[j] == 1:
                            lucky_number = lucky_number - 1
                            if lucky_number == 0:
                                execute_mode = 'edge'
                                execute_destination = j
                # 计算效用
                if execute_mode == 'local':
                    local_computing_size = devices_temp[lucky_user].task_queue_length() + cpu_frequency_demand
                    local_execute_latency = local_computing_size / devices_temp[lucky_user].frequency
                    latency_utility = local_execute_latency
                    # 本地计算能耗
                    local_computing_power = config.SWITCHED_CAPACITANCE * math.pow(devices_temp[lucky_user].frequency,
                                                                                   3)
                    local_computing_latency = cpu_frequency_demand / devices_temp[lucky_user].frequency
                    energy_cost = local_computing_latency * local_computing_power
                    # 收集能量
                    min_distance = 4000
                    destination = -1
                    for j in range(0, len(devices_temp[lucky_user].distance_BS)):
                        distance = devices_temp[lucky_user].distance_BS[j]
                        if distance < min_distance:
                            min_distance = distance
                            destination = j
                    if devices_temp[lucky_user].locate_BS[destination] == 1:
                        energy_harvest_gain = config.ENERGY_CHANNEL_GAIN
                        energy_harvest = config.ENERGY_CONVERSION_EFFICIENCY * edges_temp[
                            destination].trans_power * energy_harvest_gain * config.time_slot_length
                        # print('energy_harvest', energy_harvest)
                    else:
                        energy_harvest = 0
                    # 能量效用
                    energy_utility = energy_cost - energy_harvest
                    # 获取权重
                    latency_weight, energy_weight = devices_temp[lucky_user].get_weight()
                    # 总效用
                    total_local_utility = latency_weight * latency_utility + energy_weight * energy_utility
                    lya = 0
                    for j in range(0, config.total_number_of_devices):
                        # 计算lyapunov
                        cpu_frequency_demand_offload = 0
                        for k in range(0, config.total_number_of_devices):
                            if k != j and decisions.execute_mode[k] == 'device' and decisions.execute_destination[
                                k] == j:
                                cpu_frequency_demand_lya = task_in_each_slot[time_slot][k].cpu_frequency_demand
                                local_computing_portion = decisions.local_computing_portion[k]
                                offload_computing_portion = (1 - local_computing_portion)
                                cpu_frequency_demand_offload = cpu_frequency_demand_offload + cpu_frequency_demand_lya * offload_computing_portion
                        if decisions.execute_mode[j] == 'local':
                            cpu_frequency_demand_offload = cpu_frequency_demand_offload + task_in_each_slot[time_slot][
                                j].cpu_frequency_demand
                        elif decisions.execute_mode[j] != 'null':
                            local_computing_portion = decisions.local_computing_portion[j]
                            cpu_frequency_demand_offload = cpu_frequency_demand_offload + task_in_each_slot[time_slot][
                                j].cpu_frequency_demand * local_computing_portion
                        cpu_frequency_demand_offload = cpu_frequency_demand_offload / config.task_cpu_frequency_demand
                        cpu_frequency_demand_offload = cpu_frequency_demand_offload / 8388608
                        queue_length = devices[j].task_queue_length() / config.task_cpu_frequency_demand
                        queue_length = queue_length / 8388608
                        lya = lya + pow(cpu_frequency_demand_offload,
                                        2) + 2 * queue_length * cpu_frequency_demand_offload
                    this_total_utility = total_local_utility + config.V * lya
                    if this_total_utility <= last_total_utility:
                        decisions.execute_mode[lucky_user] = 'local'
                        decisions.execute_destination[lucky_user] = -1
                        decisions.local_computing_portion[lucky_user] = 1
                elif execute_mode == 'edge':
                    execute_edge_id = execute_destination
                    execute_edge = edges_temp[execute_edge_id]
                    interference = 0
                    for j in range(0, config.total_number_of_devices):
                        if j != lucky_user and decisions.execute_mode[j] == 'edge' and decisions.execute_destination[
                            j] == execute_edge_id:
                            gain = get_upload_gain(device=devices_temp[j], edge=execute_edge)
                            interference += gain
                    upload_rate = get_upload_rate(device=devices_temp[lucky_user], edge=execute_edge,
                                                  interference=interference)
                    slot_upload_data_size = upload_rate * config.time_slot_length
                    slot_upload_data_size = round(slot_upload_data_size, 6)
                    offload_data_size_up_limit = min(slot_upload_data_size, data_size)
                    offload_computing_portion_up_limit = offload_data_size_up_limit / data_size
                    offload_computing_portion_up_limit = round(offload_computing_portion_up_limit, 6)
                    X = numpy.random.uniform(low=0, high=offload_computing_portion_up_limit, size=config.pop_size)
                    X[0] = 0
                    X[1] = offload_computing_portion_up_limit
                    Y = [0 for m in range(len(X))]
                    for j in range(len(X)):
                        offload_computing_portion = X[j]
                        if offload_computing_portion > 0:
                            offload_data_size = data_size * offload_computing_portion
                            # 传输时间
                            edge_transmit_latency = offload_data_size / upload_rate
                            edge_transmit_latency = round(edge_transmit_latency, 6)
                            # print('edge_transmit_latency', edge_transmit_latency)
                            # 传输能耗
                            trans_energy_cost = edge_transmit_latency * devices_temp[lucky_user].offload_trans_power
                        elif offload_computing_portion == 0:
                            # 传输时间
                            edge_transmit_latency = 0
                            # 传输能耗
                            trans_energy_cost = 0
                        else:
                            print('error')
                        # 收集能量
                        min_distance = 4000
                        destination = -1
                        for k in range(0, len(devices_temp[lucky_user].distance_BS)):
                            distance = devices_temp[lucky_user].distance_BS[k]
                            if distance < min_distance:
                                min_distance = distance
                                destination = k
                        if devices_temp[lucky_user].locate_BS[destination] == 1:
                            energy_harvest_gain = config.ENERGY_CHANNEL_GAIN
                            energy_harvest_slot_length = config.time_slot_length - edge_transmit_latency
                            # print('energy harvest slot length =', energy_harvest_slot_length)
                            energy_harvest = config.ENERGY_CONVERSION_EFFICIENCY * edges_temp[
                                destination].trans_power * energy_harvest_gain * energy_harvest_slot_length
                            # print('energy harvest =', energy_harvest)
                        else:
                            energy_harvest = 0
                        # 本地计算时间
                        local_computing_portion = 1 - offload_computing_portion
                        if local_computing_portion > 0:
                            local_computing_cpu_demand = cpu_frequency_demand * local_computing_portion
                            local_computing_size = devices_temp[
                                                       lucky_user].task_queue_length() + local_computing_cpu_demand
                            # 本地计算时间
                            local_execute_latency = local_computing_size / devices_temp[lucky_user].frequency
                            # 本地计算能耗
                            local_computing_power = config.SWITCHED_CAPACITANCE * math.pow(
                                devices_temp[lucky_user].frequency, 3)
                            local_computing_latency = local_computing_cpu_demand / devices_temp[lucky_user].frequency
                            local_computing_energy_cost = local_computing_latency * local_computing_power
                        elif local_computing_portion == 0:
                            local_computing_cpu_demand = 0
                            local_execute_latency = 0
                            local_computing_energy_cost = 0
                        else:
                            print('error')
                        # 能量消耗
                        energy_cost = local_computing_energy_cost + trans_energy_cost
                        # 能量效用
                        energy_utility = energy_cost - energy_harvest
                        # print(energy_utility)
                        # 边缘计算执行时间
                        if offload_computing_portion > 0:
                            offload_computing_cpu_demand = cpu_frequency_demand * offload_computing_portion
                            edge_queue_length = execute_edge.task_queue_length() + offload_computing_cpu_demand
                            edge_computing_task_latency = edge_queue_length / execute_edge.frequency
                        elif offload_computing_portion == 0:
                            edge_computing_task_latency = 0
                        else:
                            print('error')
                        edge_execute_latency = edge_transmit_latency + edge_computing_task_latency
                        # 延迟效用
                        latency_utility = max(local_execute_latency, edge_execute_latency)
                        # 获取权重
                        latency_weight, energy_weight = devices_temp[lucky_user].get_weight()
                        # 总效用
                        total_edge_utility = latency_weight * latency_utility + energy_weight * energy_utility
                        lya = 0
                        for m in range(0, config.total_number_of_devices):
                            # 计算lyapunov
                            cpu_frequency_demand_offload = 0
                            for k in range(0, config.total_number_of_devices):
                                if k != m and decisions.execute_mode[k] == 'device' and decisions.execute_destination[
                                    k] == m:
                                    cpu_frequency_demand_lya = task_in_each_slot[time_slot][k].cpu_frequency_demand
                                    local_computing_portion = decisions.local_computing_portion[k]
                                    offload_computing_portion = (1 - local_computing_portion)
                                    cpu_frequency_demand_offload = cpu_frequency_demand_offload + cpu_frequency_demand_lya * offload_computing_portion
                            if decisions.execute_mode[m] == 'local':
                                cpu_frequency_demand_offload = cpu_frequency_demand_offload + \
                                                               task_in_each_slot[time_slot][m].cpu_frequency_demand
                            elif decisions.execute_mode[m] != 'null':
                                local_computing_portion = decisions.local_computing_portion[m]
                                cpu_frequency_demand_offload = cpu_frequency_demand_offload + \
                                                               task_in_each_slot[time_slot][
                                                                   m].cpu_frequency_demand * local_computing_portion
                            cpu_frequency_demand_offload = cpu_frequency_demand_offload / config.task_cpu_frequency_demand
                            cpu_frequency_demand_offload = cpu_frequency_demand_offload / 8388608
                            queue_length = devices[m].task_queue_length() / config.task_cpu_frequency_demand
                            queue_length = queue_length / 8388608
                            lya = lya + pow(cpu_frequency_demand_offload,
                                            2) + 2 * queue_length * cpu_frequency_demand_offload
                        total_edge_utility = total_edge_utility + config.V * lya
                        #
                        Y[j] = total_edge_utility
                    pbest_x = X.copy()  # personal best location of every particle in history
                    # self.pbest_x = self.X 表示地址传递,改变 X 值 pbest_x 也会变化
                    pbest_y = [numpy.inf for m in range(config.pop_size)]  # best image of every particle in history
                    # self.gbest_x = self.pbest_x.mean(axis=0).reshape(1, -1)  # global best location for all particles
                    gbest_x = pbest_x.mean(axis=0)
                    gbest_y = numpy.inf  # global best y for all particles
                    gbest_y_hist = []  # gbest_y of every iteration
                    for j in range(config.max_iter):
                        # # update
                        # r1 = config.a - j * (config.a / config.max_iter)
                        # 抛物线函数
                        iter_period = j / config.max_iter
                        inter_rest_phase = 1 - iter_period
                        square = pow(inter_rest_phase, 2)
                        r1 = config.a * square
                        for k in range(config.pop_size):
                            r2 = 2 * math.pi * random.uniform(0.0, 1.0)
                            r3 = 2 * random.uniform(0.0, 1.0)
                            r4 = random.uniform(0.0, 1.0)
                            if r4 < 0.5:
                                X[k] = X[k] + (r1 * math.sin(r2) * abs(r3 * gbest_x - X[k]))
                            else:
                                X[k] = X[k] + (r1 * math.cos(r2) * abs(r3 * gbest_x - X[k]))
                        X = numpy.clip(a=X, a_min=0, a_max=offload_computing_portion_up_limit)
                        for k in range(len(X)):
                            offload_computing_portion = X[k]
                            if offload_computing_portion > 0:
                                offload_data_size = data_size * offload_computing_portion
                                # 传输时间
                                edge_transmit_latency = offload_data_size / upload_rate
                                edge_transmit_latency = round(edge_transmit_latency, 6)
                                # print('edge_transmit_latency', edge_transmit_latency)
                                # 传输能耗
                                trans_energy_cost = edge_transmit_latency * devices_temp[lucky_user].offload_trans_power
                            elif offload_computing_portion == 0:
                                # 传输时间
                                edge_transmit_latency = 0
                                # 传输能耗
                                trans_energy_cost = 0
                            else:
                                print('error')
                            # 收集能量
                            min_distance = 4000
                            destination = -1
                            for m in range(0, len(devices_temp[lucky_user].distance_BS)):
                                distance = devices_temp[lucky_user].distance_BS[m]
                                if distance < min_distance:
                                    min_distance = distance
                                    destination = m
                            if devices_temp[lucky_user].locate_BS[destination] == 1:
                                energy_harvest_gain = config.ENERGY_CHANNEL_GAIN
                                energy_harvest_slot_length = config.time_slot_length - edge_transmit_latency
                                # print('energy harvest slot length =', energy_harvest_slot_length)
                                energy_harvest = config.ENERGY_CONVERSION_EFFICIENCY * edges_temp[
                                    destination].trans_power * energy_harvest_gain * energy_harvest_slot_length
                                # print('energy harvest ', energy_harvest)
                            else:
                                energy_harvest = 0
                            # 本地计算时间
                            local_computing_portion = 1 - offload_computing_portion
                            if local_computing_portion > 0:
                                local_computing_cpu_demand = cpu_frequency_demand * local_computing_portion
                                local_computing_size = devices_temp[
                                                           lucky_user].task_queue_length() + local_computing_cpu_demand
                                # 本地计算时间
                                local_execute_latency = local_computing_size / devices_temp[lucky_user].frequency
                                # 本地计算能耗
                                local_computing_power = config.SWITCHED_CAPACITANCE * math.pow(
                                    devices_temp[lucky_user].frequency, 3)
                                local_computing_latency = local_computing_cpu_demand / devices_temp[
                                    lucky_user].frequency
                                local_computing_energy_cost = local_computing_latency * local_computing_power
                            elif local_computing_portion == 0:
                                local_computing_cpu_demand = 0
                                local_execute_latency = 0
                                local_computing_energy_cost = 0
                            else:
                                print('error')
                            # 能量消耗
                            energy_cost = local_computing_energy_cost + trans_energy_cost
                            # 能量效用
                            energy_utility = energy_cost - energy_harvest
                            # print(energy_utility)
                            # 边缘计算执行时间
                            if offload_computing_portion > 0:
                                offload_computing_cpu_demand = cpu_frequency_demand * offload_computing_portion
                                edge_queue_length = execute_edge.task_queue_length() + offload_computing_cpu_demand
                                edge_computing_task_latency = edge_queue_length / execute_edge.frequency
                            elif offload_computing_portion == 0:
                                edge_computing_task_latency = 0
                            else:
                                print('error')
                            edge_execute_latency = edge_transmit_latency + edge_computing_task_latency
                            # 延迟效用
                            latency_utility = max(local_execute_latency, edge_execute_latency)
                            # 获取权重
                            latency_weight, energy_weight = devices_temp[lucky_user].get_weight()
                            # 总效用
                            total_edge_utility = latency_weight * latency_utility + energy_weight * energy_utility
                            lya = 0
                            for m in range(0, config.total_number_of_devices):
                                # 计算lyapunov
                                cpu_frequency_demand_offload = 0
                                for n in range(0, config.total_number_of_devices):
                                    if n != m and decisions.execute_mode[n] == 'device' and \
                                            decisions.execute_destination[n] == m:
                                        cpu_frequency_demand_lya = task_in_each_slot[time_slot][n].cpu_frequency_demand
                                        local_computing_portion = decisions.local_computing_portion[n]
                                        offload_computing_portion = (1 - local_computing_portion)
                                        cpu_frequency_demand_offload = cpu_frequency_demand_offload + cpu_frequency_demand_lya * offload_computing_portion
                                if decisions.execute_mode[m] == 'local':
                                    cpu_frequency_demand_offload = cpu_frequency_demand_offload + \
                                                                   task_in_each_slot[time_slot][m].cpu_frequency_demand
                                elif decisions.execute_mode[m] != 'null':
                                    local_computing_portion = decisions.local_computing_portion[m]
                                    cpu_frequency_demand_offload = cpu_frequency_demand_offload + \
                                                                   task_in_each_slot[time_slot][
                                                                       m].cpu_frequency_demand * local_computing_portion
                                cpu_frequency_demand_offload = cpu_frequency_demand_offload / config.task_cpu_frequency_demand
                                cpu_frequency_demand_offload = cpu_frequency_demand_offload / 8388608
                                queue_length = devices[m].task_queue_length() / config.task_cpu_frequency_demand
                                queue_length = queue_length / 8388608
                                lya = lya + pow(cpu_frequency_demand_offload,
                                                2) + 2 * queue_length * cpu_frequency_demand_offload
                            total_edge_utility = total_edge_utility + config.V * lya
                            #
                            Y[k] = total_edge_utility
                        # update_pbest
                        for k in range(len(Y)):
                            if pbest_y[k] > Y[k]:
                                pbest_x[k] = X[k].copy()
                                pbest_y[k] = Y[k].copy()
                        # update_gbest
                        idx_min = pbest_y.index(min(pbest_y))
                        if gbest_y > pbest_y[idx_min]:
                            gbest_x = pbest_x[idx_min].copy()  # copy很重要！
                            gbest_y = pbest_y[idx_min]
                        gbest_y_hist.append(gbest_y)
                    this_total_utility = gbest_y
                    # plt.plot(gbest_y_hist)
                    # plt.show()
                    if this_total_utility <= last_total_utility:
                        decisions.execute_mode[lucky_user] = 'edge'
                        decisions.execute_destination[lucky_user] = execute_edge_id
                        decisions.local_computing_portion[lucky_user] = 1 - gbest_x
            elif rand == 3:
                choices = [0 for m in range(0, config.total_number_of_devices)]
                for j in range(0, config.total_number_of_edges):
                    if devices_temp[lucky_user].locate_BS[j] == 1:
                        for k in range(0, config.total_number_of_devices):
                            if edges_temp[j].coverage_mobile_device[k] == 1:
                                choices[k] = 1
                choices_number = 0
                for j in range(0, config.total_number_of_devices):
                    if choices[j] == 1:
                        choices_number = choices_number + 1
                if choices_number == 0:
                    # 不会走
                    execute_mode = 'local'
                    execute_destination = -1
                elif choices_number > 0:
                    lucky_number = numpy.random.randint(1, choices_number + 1)
                    for j in range(0, config.total_number_of_devices):
                        if choices[j] == 1:
                            lucky_number = lucky_number - 1
                            if lucky_number == 0:
                                if lucky_user == j:
                                    execute_mode = 'local'
                                    execute_destination = -1
                                elif lucky_user != j:
                                    execute_mode = 'device'
                                    execute_destination = j
                                else:
                                    print('error')
                else:
                    print('error')
                # 计算效用
                if execute_mode == 'local':
                    local_computing_size = devices_temp[lucky_user].task_queue_length() + cpu_frequency_demand
                    local_execute_latency = local_computing_size / devices_temp[lucky_user].frequency
                    latency_utility = local_execute_latency
                    # 本地计算能耗
                    local_computing_power = config.SWITCHED_CAPACITANCE * math.pow(devices_temp[lucky_user].frequency,
                                                                                   3)
                    local_computing_latency = cpu_frequency_demand / devices_temp[lucky_user].frequency
                    energy_cost = local_computing_latency * local_computing_power
                    # 收集能量
                    min_distance = 4000
                    destination = -1
                    for j in range(0, len(devices_temp[lucky_user].distance_BS)):
                        distance = devices_temp[lucky_user].distance_BS[j]
                        if distance < min_distance:
                            min_distance = distance
                            destination = j
                    if devices_temp[lucky_user].locate_BS[destination] == 1:
                        energy_harvest_gain = config.ENERGY_CHANNEL_GAIN
                        energy_harvest = config.ENERGY_CONVERSION_EFFICIENCY * edges_temp[
                            destination].trans_power * energy_harvest_gain * config.time_slot_length
                        # print('energy_harvest', energy_harvest)
                    else:
                        energy_harvest = 0
                    # 能量效用
                    energy_utility = energy_cost - energy_harvest
                    # 获取权重
                    latency_weight, energy_weight = devices_temp[lucky_user].get_weight()
                    # 总效用
                    total_local_utility = latency_weight * latency_utility + energy_weight * energy_utility
                    lya = 0
                    for j in range(0, config.total_number_of_devices):
                        # 计算lyapunov
                        cpu_frequency_demand_offload = 0
                        for k in range(0, config.total_number_of_devices):
                            if k != j and decisions.execute_mode[k] == 'device' and decisions.execute_destination[
                                k] == j:
                                cpu_frequency_demand_lya = task_in_each_slot[time_slot][k].cpu_frequency_demand
                                local_computing_portion = decisions.local_computing_portion[k]
                                offload_computing_portion = (1 - local_computing_portion)
                                cpu_frequency_demand_offload = cpu_frequency_demand_offload + cpu_frequency_demand_lya * offload_computing_portion
                        if decisions.execute_mode[j] == 'local':
                            cpu_frequency_demand_offload = cpu_frequency_demand_offload + task_in_each_slot[time_slot][
                                j].cpu_frequency_demand
                        elif decisions.execute_mode[j] != 'null':
                            local_computing_portion = decisions.local_computing_portion[j]
                            cpu_frequency_demand_offload = cpu_frequency_demand_offload + task_in_each_slot[time_slot][
                                j].cpu_frequency_demand * local_computing_portion
                        cpu_frequency_demand_offload = cpu_frequency_demand_offload / config.task_cpu_frequency_demand
                        cpu_frequency_demand_offload = cpu_frequency_demand_offload / 8388608
                        queue_length = devices[j].task_queue_length() / config.task_cpu_frequency_demand
                        queue_length = queue_length / 8388608
                        lya = lya + pow(cpu_frequency_demand_offload,
                                        2) + 2 * queue_length * cpu_frequency_demand_offload
                    this_total_utility = total_local_utility + config.V * lya
                    if this_total_utility <= last_total_utility:
                        decisions.execute_mode[lucky_user] = 'local'
                        decisions.execute_destination[lucky_user] = -1
                        decisions.local_computing_portion[lucky_user] = 1
                elif execute_mode == 'device':
                    execute_device_id = execute_destination
                    execute_device = devices_temp[execute_device_id]
                    interference = 0
                    for j in range(0, config.total_number_of_devices):
                        if j != lucky_user and decisions.execute_mode[j] == 'device' and decisions.execute_destination[
                            j] == execute_device_id:
                            gain = get_d2d_gain(device1=devices_temp[j], device2=execute_device)
                            interference += gain
                    d2d_rate = get_d2d_rate(device1=devices_temp[lucky_user], device2=execute_device,
                                            interference=interference)
                    slot_d2d_data_size = d2d_rate * config.time_slot_length
                    d2d_data_size_up_limit = min(slot_d2d_data_size, data_size)
                    d2d_computing_portion_up_limit = d2d_data_size_up_limit / data_size
                    X = numpy.random.uniform(low=0, high=d2d_computing_portion_up_limit, size=config.pop_size)
                    X[0] = 0
                    X[1] = d2d_computing_portion_up_limit
                    Y = [0 for m in range(len(X))]
                    for j in range(len(X)):
                        d2d_computing_portion = X[j]
                        if d2d_computing_portion > 0:
                            d2d_data_size = data_size * d2d_computing_portion
                            # 传输时间
                            d2d_transmit_latency = d2d_data_size / d2d_rate
                            d2d_transmit_latency = round(d2d_transmit_latency, 6)
                            # print('d2d_transmit_latency', d2d_transmit_latency)
                            # 传输能耗
                            trans_energy_cost = d2d_transmit_latency * devices_temp[lucky_user].d2d_trans_power
                        elif d2d_computing_portion == 0:
                            # 传输时间
                            d2d_transmit_latency = 0
                            # 传输能耗
                            trans_energy_cost = 0
                        else:
                            print('error')
                        # 收集能量
                        min_distance = 4000
                        destination = -1
                        for k in range(0, len(devices_temp[lucky_user].distance_BS)):
                            distance = devices_temp[lucky_user].distance_BS[k]
                            if distance < min_distance:
                                min_distance = distance
                                destination = k
                        if devices_temp[lucky_user].locate_BS[destination] == 1:
                            energy_harvest_gain = config.ENERGY_CHANNEL_GAIN
                            energy_harvest_slot_length = config.time_slot_length - d2d_transmit_latency
                            # print('energy harvest slot length =', energy_harvest_slot_length)
                            energy_harvest = config.ENERGY_CONVERSION_EFFICIENCY * edges_temp[
                                destination].trans_power * energy_harvest_gain * energy_harvest_slot_length
                            # print('energy harvest ', energy_harvest)
                        else:
                            energy_harvest = 0
                        # 本地计算时间
                        local_computing_portion = 1 - d2d_computing_portion
                        if local_computing_portion > 0:
                            local_computing_cpu_demand = cpu_frequency_demand * local_computing_portion
                            local_computing_size = devices_temp[
                                                       lucky_user].task_queue_length() + local_computing_cpu_demand
                            # 本地计算时间
                            local_execute_latency = local_computing_size / devices_temp[lucky_user].frequency
                            # 本地计算能耗
                            local_computing_power = config.SWITCHED_CAPACITANCE * math.pow(
                                devices_temp[lucky_user].frequency, 3)
                            local_computing_latency = local_computing_cpu_demand / devices_temp[lucky_user].frequency
                            local_computing_energy_cost = local_computing_latency * local_computing_power
                        elif local_computing_portion == 0:
                            local_computing_cpu_demand = 0
                            local_execute_latency = 0
                            local_computing_energy_cost = 0
                        else:
                            print('error')
                        # 能量消耗
                        energy_cost = local_computing_energy_cost + trans_energy_cost
                        # 能量效用
                        energy_utility = energy_cost - energy_harvest
                        # print(energy_utility)
                        # 边缘计算执行时间
                        if d2d_computing_portion > 0:
                            d2d_cpu_frequency_demand = cpu_frequency_demand * d2d_computing_portion
                            device_queue_length = execute_device.task_queue_length() + d2d_cpu_frequency_demand
                            d2d_computing_task_latency = device_queue_length / execute_device.frequency
                        elif d2d_computing_portion == 0:
                            d2d_computing_task_latency = 0
                        else:
                            print('error')
                        d2d_execute_latency = d2d_transmit_latency + d2d_computing_task_latency
                        # 延迟效用
                        latency_utility = max(local_execute_latency, d2d_execute_latency)
                        # 获取权重
                        latency_weight, energy_weight = devices_temp[lucky_user].get_weight()
                        # 总效用
                        total_d2d_utility = latency_weight * latency_utility + energy_weight * energy_utility
                        lya = 0
                        for m in range(0, config.total_number_of_devices):
                            # 计算lyapunov
                            cpu_frequency_demand_offload = 0
                            for k in range(0, config.total_number_of_devices):
                                if k != m and decisions.execute_mode[k] == 'device' and decisions.execute_destination[
                                    k] == m:
                                    cpu_frequency_demand_lya = task_in_each_slot[time_slot][k].cpu_frequency_demand
                                    local_computing_portion = decisions.local_computing_portion[k]
                                    offload_computing_portion = (1 - local_computing_portion)
                                    cpu_frequency_demand_offload = cpu_frequency_demand_offload + cpu_frequency_demand_lya * offload_computing_portion
                            if decisions.execute_mode[m] == 'local':
                                cpu_frequency_demand_offload = cpu_frequency_demand_offload + \
                                                               task_in_each_slot[time_slot][m].cpu_frequency_demand
                            elif decisions.execute_mode[m] != 'null':
                                local_computing_portion = decisions.local_computing_portion[m]
                                cpu_frequency_demand_offload = cpu_frequency_demand_offload + \
                                                               task_in_each_slot[time_slot][
                                                                   m].cpu_frequency_demand * local_computing_portion
                            cpu_frequency_demand_offload = cpu_frequency_demand_offload / config.task_cpu_frequency_demand
                            cpu_frequency_demand_offload = cpu_frequency_demand_offload / 8388608
                            queue_length = devices[m].task_queue_length() / config.task_cpu_frequency_demand
                            queue_length = queue_length / 8388608
                            lya = lya + pow(cpu_frequency_demand_offload,
                                            2) + 2 * queue_length * cpu_frequency_demand_offload
                        total_d2d_utility = total_d2d_utility + config.V * lya
                        #
                        Y[j] = total_d2d_utility
                    pbest_x = X.copy()  # personal best location of every particle in history
                    # self.pbest_x = self.X 表示地址传递,改变 X 值 pbest_x 也会变化
                    pbest_y = [numpy.inf for m in range(config.pop_size)]  # best image of every particle in history
                    # self.gbest_x = self.pbest_x.mean(axis=0).reshape(1, -1)  # global best location for all particles
                    gbest_x = pbest_x.mean(axis=0)
                    gbest_y = numpy.inf  # global best y for all particles
                    gbest_y_hist = []  # gbest_y of every iteration
                    for j in range(config.max_iter):
                        # # update
                        # r1 = config.a - j * (config.a / config.max_iter)
                        # 抛物线函数
                        iter_period = j / config.max_iter
                        inter_rest_phase = 1 - iter_period
                        square = pow(inter_rest_phase, 2)
                        r1 = config.a * square
                        for k in range(config.pop_size):
                            r2 = 2 * math.pi * random.uniform(0.0, 1.0)
                            r3 = 2 * random.uniform(0.0, 1.0)
                            r4 = random.uniform(0.0, 1.0)
                            if r4 < 0.5:
                                X[k] = X[k] + (r1 * math.sin(r2) * abs(r3 * gbest_x - X[k]))
                            else:
                                X[k] = X[k] + (r1 * math.cos(r2) * abs(r3 * gbest_x - X[k]))
                        X = numpy.clip(a=X, a_min=0, a_max=d2d_computing_portion_up_limit)
                        for k in range(len(X)):
                            d2d_computing_portion = X[k]
                            if d2d_computing_portion > 0:
                                d2d_data_size = data_size * d2d_computing_portion
                                # 传输时间
                                d2d_transmit_latency = d2d_data_size / d2d_rate
                                d2d_transmit_latency = round(d2d_transmit_latency, 6)
                                # print('d2d_transmit_latency', d2d_transmit_latency)
                                # 传输能耗
                                trans_energy_cost = d2d_transmit_latency * devices_temp[lucky_user].d2d_trans_power
                            elif d2d_computing_portion == 0:
                                # 传输时间
                                d2d_transmit_latency = 0
                                # 传输能耗
                                trans_energy_cost = 0
                            else:
                                print('error')
                            # 收集能量
                            min_distance = 4000
                            destination = -1
                            for m in range(0, len(devices_temp[lucky_user].distance_BS)):
                                distance = devices_temp[lucky_user].distance_BS[m]
                                if distance < min_distance:
                                    min_distance = distance
                                    destination = m
                            if devices_temp[lucky_user].locate_BS[destination] == 1:
                                energy_harvest_gain = config.ENERGY_CHANNEL_GAIN
                                energy_harvest_slot_length = config.time_slot_length - d2d_transmit_latency
                                # print('energy harvest slot length =', energy_harvest_slot_length)
                                energy_harvest = config.ENERGY_CONVERSION_EFFICIENCY * edges_temp[
                                    destination].trans_power * energy_harvest_gain * energy_harvest_slot_length
                                # print('energy harvest ', energy_harvest)
                            else:
                                energy_harvest = 0
                            # 本地计算时间
                            local_computing_portion = 1 - d2d_computing_portion
                            if local_computing_portion > 0:
                                local_computing_cpu_demand = cpu_frequency_demand * local_computing_portion
                                local_computing_size = devices_temp[
                                                           lucky_user].task_queue_length() + local_computing_cpu_demand
                                # 本地计算时间
                                local_execute_latency = local_computing_size / devices_temp[lucky_user].frequency
                                # 本地计算能耗
                                local_computing_power = config.SWITCHED_CAPACITANCE * math.pow(
                                    devices_temp[lucky_user].frequency, 3)
                                local_computing_latency = local_computing_cpu_demand / devices_temp[
                                    lucky_user].frequency
                                local_computing_energy_cost = local_computing_latency * local_computing_power
                            elif local_computing_portion == 0:
                                local_computing_cpu_demand = 0
                                local_execute_latency = 0
                                local_computing_energy_cost = 0
                            else:
                                print('error')
                            # 能量消耗
                            energy_cost = local_computing_energy_cost + trans_energy_cost
                            # 能量效用
                            energy_utility = energy_cost - energy_harvest
                            # print(energy_utility)
                            # 边缘计算执行时间
                            if d2d_computing_portion > 0:
                                d2d_cpu_frequency_demand = cpu_frequency_demand * d2d_computing_portion
                                device_queue_length = execute_device.task_queue_length() + d2d_cpu_frequency_demand
                                d2d_computing_task_latency = device_queue_length / execute_device.frequency
                            elif d2d_computing_portion == 0:
                                d2d_computing_task_latency = 0
                            else:
                                print('error')
                            d2d_execute_latency = d2d_transmit_latency + d2d_computing_task_latency
                            # 延迟效用
                            latency_utility = max(local_execute_latency, d2d_execute_latency)
                            # 获取权重
                            latency_weight, energy_weight = devices_temp[lucky_user].get_weight()
                            # 总效用
                            total_d2d_utility = latency_weight * latency_utility + energy_weight * energy_utility
                            lya = 0
                            for m in range(0, config.total_number_of_devices):
                                # 计算lyapunov
                                cpu_frequency_demand_offload = 0
                                for n in range(0, config.total_number_of_devices):
                                    if n != m and decisions.execute_mode[n] == 'device' and \
                                            decisions.execute_destination[n] == m:
                                        cpu_frequency_demand_lya = task_in_each_slot[time_slot][n].cpu_frequency_demand
                                        local_computing_portion = decisions.local_computing_portion[n]
                                        offload_computing_portion = (1 - local_computing_portion)
                                        cpu_frequency_demand_offload = cpu_frequency_demand_offload + cpu_frequency_demand_lya * offload_computing_portion
                                if decisions.execute_mode[m] == 'local':
                                    cpu_frequency_demand_offload = cpu_frequency_demand_offload + \
                                                                   task_in_each_slot[time_slot][m].cpu_frequency_demand
                                elif decisions.execute_mode[m] != 'null':
                                    local_computing_portion = decisions.local_computing_portion[m]
                                    cpu_frequency_demand_offload = cpu_frequency_demand_offload + \
                                                                   task_in_each_slot[time_slot][
                                                                       m].cpu_frequency_demand * local_computing_portion
                                cpu_frequency_demand_offload = cpu_frequency_demand_offload / config.task_cpu_frequency_demand
                                cpu_frequency_demand_offload = cpu_frequency_demand_offload / 8388608
                                queue_length = devices[m].task_queue_length() / config.task_cpu_frequency_demand
                                queue_length = queue_length / 8388608
                                lya = lya + pow(cpu_frequency_demand_offload,
                                                2) + 2 * queue_length * cpu_frequency_demand_offload
                            total_d2d_utility = total_d2d_utility + config.V * lya
                            #
                            Y[k] = total_d2d_utility
                        # update_pbest
                        for k in range(len(Y)):
                            if pbest_y[k] > Y[k]:
                                pbest_x[k] = X[k].copy()
                                pbest_y[k] = Y[k].copy()
                        # update_gbest
                        idx_min = pbest_y.index(min(pbest_y))
                        if gbest_y > pbest_y[idx_min]:
                            gbest_x = pbest_x[idx_min].copy()  # copy很重要！
                            gbest_y = pbest_y[idx_min].copy()
                        gbest_y_hist.append(gbest_y)
                    this_total_utility = gbest_y
                    # plt.plot(gbest_y_hist)
                    # plt.show()
                    if this_total_utility <= last_total_utility:
                        decisions.execute_mode[lucky_user] = 'device'
                        decisions.execute_destination[lucky_user] = execute_device_id
                        decisions.local_computing_portion[lucky_user] = 1 - gbest_x
        elif decisions.execute_mode[lucky_user] == 'null':
            pass
        else:
            pass
    return decisions


if __name__ == '__main__':
    for i in range(1000):
        print(numpy.random.randint(1, 4))
