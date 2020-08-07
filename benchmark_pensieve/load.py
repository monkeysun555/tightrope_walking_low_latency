import os
import numpy as np

DATA_DIR = '../bw_traces/'
TRACE_NAME = '../bw_traces/BKLYN_1.txt'

def loadBandwidth(data_dir=DATA_DIR):
    datas = os.listdir(data_dir)
    time_traces = []
    throughput_traces = []
    data_names = []
    for data in datas:
        if  '.DS' in data: continue
        file_path = data_dir + data
        time_trace = []
        throughput_trace = []
        time = 0.0
        with open(file_path, 'r') as f:
            for line in f:
                parse = line.strip('\n')
                time_trace.append(time)
                throughput_trace.append(float(parse))
                time += 1.0
        time_traces.append(time_trace)
        throughput_traces.append(throughput_trace)
        data_names.append(data)

    return time_traces, throughput_traces, data_names


def new_loadBandwidth(data_dir=DATA_DIR):
    datas = os.listdir(data_dir)
    time_traces = []
    throughput_traces = []
    data_names = []

    for data in datas:
        file_path = data_dir + data
        time_trace = []
        throughput_trace = []
        # time = 0.0
        # print(data)
        with open(file_path, 'r') as f:
            for line in f:
                # parse = line.split(',')
                parse = line.strip('\n').split()
                time_trace.append(float(parse[0]))
                # throughput_trace.append(float(parse[4]))
                throughput_trace.append(float(parse[1]))
                # time += 1.0
        # print(throughput_trace)
        time_traces.append(time_trace)
        throughput_traces.append(throughput_trace)
        data_names.append(data)
    # print throughput_traces
    return time_traces, throughput_traces, data_names

def load_single_trace(data_dir = TRACE_NAME):

    file_path = data_dir
    time_trace = []
    throughput_trace = []
    time = 0.0
    # print(data)
    with open(file_path, 'r') as f:
        for line in f:
            # parse = line.split(',')
            parse = line.strip('\n')
            # print(parse)
            time_trace.append(time)
            # throughput_trace.append(float(parse[4]))
            throughput_trace.append(float(parse))
            time += 1.0
    # print(throughput_trace)

    return time_trace, throughput_trace

def new_load_single_trace(data_dir = TRACE_NAME):

    file_path = data_dir
    time_trace = []
    throughput_trace = []
    # time = 0.0
    # print(data)
    with open(file_path, 'r') as f:
        for line in f:
            # parse = line.split(',')
            parse = line.strip('\n').split()
            # print(parse)
            time_trace.append(float(parse[0]))
            # throughput_trace.append(float(parse[4]))
            throughput_trace.append(float(parse[1]))
            # time += 1.0
    # print(throughput_trace)

    return time_trace, throughput_trace

if __name__ == '__main__':
    testing_trace = '../new_traces/train_sim_traces/'
    new_loadBandwidth(testing_trace)