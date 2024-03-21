import pickle
import logging
import math
import numpy as np

from Decision import Decision
from MobileDevice import MobileDevice
from Edge import Edge
from Config import Config
from Task import Task
from algorithm import local_algorithm, nearest_bs_algorithm, random_algorithm, proposed_algorithm, binary_match_game
from trans_rate import get_upload_gain, get_upload_rate

# 只考虑任务执行时间的版本

logging.basicConfig(level=logging.ERROR)

config = Config()
devices = []
edges = []

io_in_each_slot = []
ave_queue_length_in_each_slot = []
ave_execute_latency_in_each_slot = []
latency_cost_in_each_slot = []
energy_cost_in_each_slot = []
error_cost_in_each_slot = []
total_cost_in_each_slot = []
local_ratio_in_each_slot = [0 for _ in range(config.times)]
edge_ratio_in_each_slot = [0 for _ in range(config.times)]


def init_edge_device():
    # 初始化边缘和设备
    global devices
    global edges

    if config.cache is True:
        devices_cache_file = open(config.devices_cache_file_path, 'rb')
        devices = pickle.load(devices_cache_file)
        edges_cache_file = open(config.edges_cache_file_path, 'rb')
        edges = pickle.load(edges_cache_file)
    else:
        for i in range(0, config.total_number_of_edges):
            frequency = config.edge_cpu_frequency
            ncp_type = 'EDGE'
            nid = i
            edge = Edge(nid=nid, frequency=frequency, ncp_type=ncp_type, x=config.edge_locations[i][0],
                        y=config.edge_locations[i][1], total_number_of_devices=config.total_number_of_devices)
            edges.append(edge)

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
            device.get_the_distance(total_number_of_edges=config.total_number_of_edges, edges=edges)
            devices.append(device)

        # 保存到文件中
        devices_cache_file = open(config.devices_cache_file_path, 'wb')
        pickle.dump(devices, devices_cache_file)
        edges_cache_file = open(config.edges_cache_file_path, 'wb')
        pickle.dump(edges, edges_cache_file)


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
        print('error')
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
        print('error')
    return local_computing_size


def make_decision(time_slot):
    # 决策
    global local_ratio_in_each_slot
    global edge_ratio_in_each_slot

    decisions = Decision()
    if config.algorithm == 'local_algorithm':
        decisions = local_algorithm(decisions=decisions, config=config)
    elif config.algorithm == 'nearest_algorithm':
        decisions = nearest_bs_algorithm(decisions=decisions, config=config, devices=devices, edges=edges)
    elif config.algorithm == 'random_algorithm':
        decisions = random_algorithm(decisions=decisions, config=config, devices=devices, edges=edges)
    elif config.algorithm == 'match_algorithm':
        decisions = binary_match_game(decisions=decisions, config=config, devices=devices, edges=edges)
    elif config.algorithm == 'proposed_algorithm':
        decisions = proposed_algorithm(decisions=decisions, config=config, devices=devices, edges=edges,
                                       time_slot=time_slot)
    else:
        pass
    if config.algorithm == 'proposed_algorithm':
        for i in range(0, config.total_number_of_devices):
            if decisions.execute_mode[i] == 'local':
                local_ratio_in_each_slot[time_slot] = local_ratio_in_each_slot[time_slot] + 1
            elif decisions.execute_mode[i] == 'edge':
                edge_ratio_in_each_slot[time_slot] = edge_ratio_in_each_slot[time_slot] + 1
    return decisions


def execute_decision(device, edges, decisions):
    # 执行决策
    global io_in_each_slot
    resolution_selection = decisions.resolution_selection[device.id]
    print('resolution_selection',resolution_selection)
    if resolution_selection == 1:
        resolution = 100
        accuracy = 0.176
    elif resolution_selection == 2:
        resolution = 200
        accuracy = 0.570
    elif resolution_selection == 3:
        resolution = 300
        accuracy = 0.775
    elif resolution_selection == 4:
        resolution = 400
        accuracy = 0.882
    elif resolution_selection == 5:
        resolution = 500
        accuracy = 0.939
    elif resolution_selection == 6:
        resolution = 600
        accuracy = 0.968
    else:
        print('error')
    data_size = config.bits_per_pixel * math.pow(resolution, 2)
    cpu_frequency_demand = data_size * config.task_cpu_frequency_demand
    task = Task(resolution=resolution, accuracy=accuracy, data_size=data_size,
                cpu_frequency_demand=cpu_frequency_demand)
    print('execute_destination', decisions.execute_destination[device.id])
    if decisions.execute_destination[device.id] == -1:
        # print('local')
        # 本地计算时间
        device.task_enqueue(cpu_frequency_demand=task.cpu_frequency_demand)
        queue_length = device.task_queue_length()
        local_execute_latency = queue_length / device.frequency
        latency_cost = local_execute_latency
        print('latency_cost', latency_cost)
        # 本地计算能耗
        local_computing_power = config.SWITCHED_CAPACITANCE * math.pow(device.frequency, 3)
        local_computing_latency = task.cpu_frequency_demand / device.frequency
        energy_cost = local_computing_power * local_computing_latency
        print('energy_cost', energy_cost)
        # 精度消耗
        error_cost = 1 - task.accuracy
        print('error_cost', error_cost)
        # 总效用
        total_cost = config.latency_weight * latency_cost + config.energy_weight * energy_cost + config.error_weight * error_cost

    elif decisions.execute_destination[device.id] != -1:
        # print('edge')
        execute_edge_id = decisions.execute_destination[device.id]
        local_computing_portion = decisions.local_computing_portion[device.id]
        execute_edge = edges[execute_edge_id]
        # 计算卸载数据大小
        offload_computing_portion = (1 - local_computing_portion)
        # print('offload_computing_portion', offload_computing_portion)
        if offload_computing_portion > 0:
            offload_data_size = task.data_size * offload_computing_portion
            # 计算传输速率
            interference = 0
            for i in range(0, config.total_number_of_devices):
                if i != device.id and decisions.execute_destination[i] == execute_edge_id:
                    gain = get_upload_gain(device=devices[i], edge=execute_edge)
                    interference += gain
            upload_rate = get_upload_rate(device=device, edge=execute_edge, interference=interference)
            # print('upload_rate', upload_rate/8388608)
            # 传输时间
            edge_transmit_latency = offload_data_size / upload_rate
            edge_transmit_latency = round(edge_transmit_latency, 6)
            # print('edge_transmit_latency=', edge_transmit_latency)
            if edge_transmit_latency >= config.time_slot_length:
                edge_transmit_latency = config.time_slot_length
            print('real_edge_transmit_latency=', edge_transmit_latency)
            # 传输能耗
            trans_energy_cost = edge_transmit_latency * device.offload_trans_power
        elif offload_computing_portion == 0:
            # 传输时间
            edge_transmit_latency = 0
            # 传输能耗
            trans_energy_cost = 0
        else:
            print('error')
        # 本地计算时间
        if local_computing_portion > 0:
            if edge_transmit_latency == config.time_slot_length:
                edge_transmit_data_size = upload_rate * config.time_slot_length
                local_computing_data_size = task.data_size - edge_transmit_data_size
                local_computing_cpu_demand = local_computing_data_size * config.task_cpu_frequency_demand
            else:
                local_computing_cpu_demand = task.cpu_frequency_demand * local_computing_portion
            device.task_enqueue(cpu_frequency_demand=local_computing_cpu_demand)
            # 队列长度
            queue_length = device.task_queue_length()
            # 本地计算时间
            local_execute_latency = queue_length / device.frequency
            # 本地计算能耗
            local_computing_power = config.SWITCHED_CAPACITANCE * math.pow(device.frequency, 3)
            local_computing_latency = local_computing_cpu_demand / device.frequency
            local_computing_energy_cost = local_computing_latency * local_computing_power
        elif local_computing_portion == 0:
            local_computing_cpu_demand = 0
            local_execute_latency = 0
            local_computing_energy_cost = 0
        else:
            print('error')
        # 能量消耗
        energy_cost = local_computing_energy_cost + trans_energy_cost
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
            print('error')
            # edge_compute_task_latency = 0
        edge_execute_latency = edge_transmit_latency + edge_compute_task_latency
        # 延迟效用
        # print('local_execute_latency', local_execute_latency)
        # print('edge_execute_latency', edge_execute_latency)
        latency_cost = max(local_execute_latency, edge_execute_latency)
        # print('latency_utility', latency_utility)
        # 精度消耗
        error_cost = 1 - task.accuracy
        # 总效用
        total_cost = config.latency_weight * latency_cost + config.energy_weight * energy_cost + config.error_weight * error_cost
    else:
        print('error')
    return latency_cost, energy_cost, error_cost, total_cost


def update_system(devices_vector, edges_vector):
    # 更新系统
    # global uav_trans_rates
    for i in range(0, config.total_number_of_devices):
        devices_vector[i].move()
    for i in range(0, config.total_number_of_devices):
        devices_vector[i].get_the_distance(total_number_of_edges=config.total_number_of_edges, edges=edges_vector)


def do_simulation():
    # 系统仿真
    global devices
    global edges
    global io_in_each_slot
    global ave_queue_length_in_each_slot
    global ave_execute_latency_in_each_slot
    global latency_cost_in_each_slot
    global energy_cost_in_each_slot
    global error_cost_in_each_slot
    global total_cost_in_each_slot

    for time_slot in range(0, config.times):
        logging.error("%s / %s", time_slot, config.times)
        update_system(devices_vector=devices, edges_vector=edges)
        decisions = make_decision(time_slot=time_slot)
        device_compute_size = 0
        edge_compute_size = 0
        if time_slot > 0:
            for i in range(0, config.total_number_of_devices):
                device_compute_size = device_compute_size + device_compute(device=devices[i])
            for i in range(0, config.total_number_of_edges):
                edge_compute_size = edge_compute_size + edge_compute(edge=edges[i])
        io = device_compute_size + edge_compute_size
        io_in_each_slot.append(io)
        total_latency_cost = 0
        total_energy_cost = 0
        total_error_cost = 0
        total_cost = 0
        for j in range(0, config.total_number_of_devices):
            latency_cost, energy_cost, error_cost, cost = execute_decision(device=devices[j], edges=edges,
                                                                           decisions=decisions)
            total_latency_cost += latency_cost
            total_energy_cost += energy_cost
            total_error_cost += error_cost
            total_cost += cost
        latency_cost_in_each_slot.append(total_latency_cost)
        ave_execute_latency = total_latency_cost / config.total_number_of_devices
        ave_execute_latency_in_each_slot.append(ave_execute_latency)
        energy_cost_in_each_slot.append(total_energy_cost)
        error_cost_in_each_slot.append(total_error_cost)
        total_cost_in_each_slot.append(total_cost)
        sum_queue_length = 0
        for i in range(0, config.total_number_of_devices):
            sum_queue_length += devices[i].task_queue_length()
        # for i in range(0, config.total_number_of_edges):
        #     sum_queue_length += edges[i].task_queue_length()
        # ave_queue_length = sum_queue_length / (config.total_number_of_devices + config.total_number_of_edges)
        ave_queue_length = sum_queue_length / config.total_number_of_devices
        ave_queue_length_in_each_slot.append(ave_queue_length)


def print_result():
    # 输出结果
    global io_in_each_slot
    global ave_queue_length_in_each_slot
    global ave_execute_latency_in_each_slot
    global latency_cost_in_each_slot
    global energy_cost_in_each_slot
    global error_cost_in_each_slot
    global total_cost_in_each_slot
    global local_ratio_in_each_slot
    global edge_ratio_in_each_slot
    global d2d_ratio_in_each_slot

    # path = config.res_cache_path + str(config.local_algorithm)
    # file = open(path, 'w+')
    # for item in queue_len_of_each_slot:
    #     file.write(str(item) + " ")
    # file.close()
    # 输出配置
    # logging.error(config.__dict__)  # 打印init里面的所有配置
    # 输出总成本
    # logging.error("传输总成本：%s", sum(trans_cost_in_each_slot))
    # logging.error("总成本：%s", sum(total_cost_in_each_slot))

    if config.algorithm == 'proposed_algorithm':
        for i in range(0, config.times):
            local_ratio_in_each_slot[i] = local_ratio_in_each_slot[i] / config.total_number_of_devices
            edge_ratio_in_each_slot[i] = edge_ratio_in_each_slot[i] / config.total_number_of_devices
            d2d_ratio_in_each_slot[i] = d2d_ratio_in_each_slot[i] / config.total_number_of_devices

        local_ratio_in_each_slot_file_path = config.local_ratio_in_each_slot_file_path
        file = open(local_ratio_in_each_slot_file_path, 'w+')
        for item in local_ratio_in_each_slot:
            file.write(str(item) + " ")
        file.close()

        edge_ratio_in_each_slot_file_path = config.edge_ratio_in_each_slot_file_path
        file = open(edge_ratio_in_each_slot_file_path, 'w+')
        for item in edge_ratio_in_each_slot:
            file.write(str(item) + " ")
        file.close()

    # to do
    # ave_queue_length_in_each_slot_path = config.local_ave_queue_length_in_each_slot_file_path
    # ave_queue_length_in_each_slot_path = config.nearest_ave_queue_length_in_each_slot_file_path
    # ave_queue_length_in_each_slot_path = config.random_ave_queue_length_in_each_slot_file_path
    # ave_queue_length_in_each_slot_path = config.match_ave_queue_length_in_each_slot_file_path
    ave_queue_length_in_each_slot_path = config.proposed_ave_queue_length_in_each_slot_file_path
    file = open(ave_queue_length_in_each_slot_path, 'w+')
    for item in ave_queue_length_in_each_slot:
        file.write(str(item) + " ")
    file.close()

    # to do
    # ave_execute_latency_in_each_slot_file_path = config.local_ave_execute_latency_in_each_slot_file_path
    # ave_execute_latency_in_each_slot_file_path = config.nearest_ave_execute_latency_in_each_slot_file_path
    # ave_execute_latency_in_each_slot_file_path = config.random_ave_execute_latency_in_each_slot_file_path
    # ave_execute_latency_in_each_slot_file_path = config.match_ave_execute_latency_in_each_slot_file_path
    ave_execute_latency_in_each_slot_file_path = config.proposed_ave_execute_latency_in_each_slot_file_path
    file = open(ave_execute_latency_in_each_slot_file_path, 'w+')
    for item in ave_execute_latency_in_each_slot:
        file.write(str(item) + " ")
    file.close()

    # to do
    # energy_cost_in_each_slot_file_path = config.local_energy_cost_in_each_slot_file_path
    # energy_cost_in_each_slot_file_path = config.nearest_energy_cost_in_each_slot_file_path
    # energy_cost_in_each_slot_file_path = config.random_energy_cost_in_each_slot_file_path
    # energy_cost_in_each_slot_file_path = config.match_energy_cost_in_each_slot_file_path
    energy_cost_in_each_slot_file_path = config.proposed_energy_cost_in_each_slot_file_path
    file = open(energy_cost_in_each_slot_file_path, 'w+')
    for item in energy_cost_in_each_slot:
        file.write(str(item) + " ")
    file.close()

    # to do
    # latency_cost_in_each_slot_file_path = config.local_latency_cost_in_each_slot_file_path
    # latency_cost_in_each_slot_file_path = config.nearest_latency_cost_in_each_slot_file_path
    # latency_cost_in_each_slot_file_path = config.random_latency_cost_in_each_slot_file_path
    # latency_cost_in_each_slot_file_path = config.match_latency_cost_in_each_slot_file_path
    latency_cost_in_each_slot_file_path = config.proposed_latency_cost_in_each_slot_file_path
    file = open(latency_cost_in_each_slot_file_path, 'w+')
    for item in latency_cost_in_each_slot:
        file.write(str(item) + " ")
    file.close()

    # to do
    # error_cost_in_each_slot_file_path = config.local_error_cost_in_each_slot_file_path
    # error_cost_in_each_slot_file_path = config.nearest_error_cost_in_each_slot_file_path
    # error_cost_in_each_slot_file_path = config.random_error_cost_in_each_slot_file_path
    # error_cost_in_each_slot_file_path = config.match_error_cost_in_each_slot_file_path
    error_cost_in_each_slot_file_path = config.proposed_error_cost_in_each_slot_file_path
    file = open(error_cost_in_each_slot_file_path, 'w+')
    for item in error_cost_in_each_slot:
        file.write(str(item) + " ")
    file.close()

    # to do
    # total_cost_in_each_slot_file_path = config.local_total_cost_in_each_slot_file_path
    # total_cost_in_each_slot_file_path = config.nearest_total_cost_in_each_slot_file_path
    # total_cost_in_each_slot_file_path = config.random_total_cost_in_each_slot_file_path
    # total_cost_in_each_slot_file_path = config.match_total_cost_in_each_slot_file_path
    total_cost_in_each_slot_file_path = config.proposed_total_cost_in_each_slot_file_path
    file = open(total_cost_in_each_slot_file_path, 'w+')
    for item in total_cost_in_each_slot:
        file.write(str(item) + " ")
    file.close()

    # to do
    # io_in_each_slot_file_path = config.local_io_in_each_slot_file_path
    # io_in_each_slot_file_path = config.nearest_io_in_each_slot_file_path
    # io_in_each_slot_file_path = config.random_io_in_each_slot_file_path
    # io_in_each_slot_file_path = config.match_io_in_each_slot_file_path
    io_in_each_slot_file_path = config.proposed_io_in_each_slot_file_path
    file = open(io_in_each_slot_file_path, 'w+')
    for item in io_in_each_slot:
        file.write(str(item) + " ")
    file.close()

    # 保存任务
    # if config.is_task_cache is False:
    #     task_cache_file = open(config.task_cache_file_path, 'wb')
    #     pickle.dump(task_in_each_slot, task_cache_file)


def start():
    init_edge_device()
    do_simulation()
    print_result()


if __name__ == '__main__':
    start()
