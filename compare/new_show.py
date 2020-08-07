import numpy as np
import matplotlib.pyplot as plt
import os

FIGSHOW = True
SAVING_DIR = './metric/'
RES_DIR = './compare_results/'
# ALGOS = ['rate', 'lqr', 'ddqn', 'multi']
ALGOS = ['RATE', 'iLQR', 'DDQN', 'BDQ']

LINE_TYPES = ['+', '*', 'h', 'd', '.']
# COLORS = ['#1f77b4',  '#ff7f0e',  '#2ca02c',
#           '#d62728', '#9467bd',
#           '#8c564b', '#e377c2', '#7f7f7f',
#           '#bcbd22', '#17becf']
COLORS = ['#8c564b', '#1f77b4',  '#ff7f0e', 
          '#2ca02c', '#9467bd',
          '#e377c2', '#7f7f7f',
          '#bcbd22', '#17becf']

def name_translate(name):
    if name == 'iLQR':
        return 'dyn_mpc_normal.txt'
    elif name == 'DDQN':
        return 'naive_speed_normal.txt'
    elif name == 'RATE':
        return 'rate_adaption_normal.txt'
    elif name == 'BDQ':
        return 'multi_speed_normal.txt'

def main():
    if not os.path.exists(SAVING_DIR):
        os.makedirs(SAVING_DIR)
    all_info = []
    all_bw_info = dict()
    for algo in ALGOS:
        res_name = name_translate(algo)
        file_path = RES_DIR + res_name
        with open(file_path, 'r') as f:
            algo_info = dict()
            if algo == 'RATE':
                for line in f:
                    parse = line.strip('\n')
                    parse = parse.split('\t')
                    # name_record.append(parse[0])
                    qoe = float(parse[1])               #[0]
                    bit_rate = float(parse[2])/1000.0          #[1]
                    freeze = float(parse[3])            #[3]
                    change = float(parse[4])            #[4]
                    latency = float(parse[5])           #[5]
                    speed = 1.0                         # None
                    speed_change = 0.                   #[7]
                    algo_info[parse[0]] = [qoe, bit_rate, speed, freeze, change, latency, speed_change, 0.0]
            elif algo == 'DDQN':
                n_line = 0
                curr_name = ''
                for line in f:
                    if n_line%2 == 0:
                        parse = line.strip('\n')
                        parse = parse.split('\t')
                        # name_record.append(parse[0])      # BW_trace name
                        curr_name = parse[0]
                        qoe = float(parse[1])               #[1]
                        bit_rate = float(parse[2])/1000.0          #[2]
                        speed = float(parse[3])             #[3]
                        freeze = float(parse[4])            #[4]
                        change = float(parse[5])            #[5]
                        latency = float(parse[6])           #[6]
                        speed_change = float(parse[7]) #[7]
                        speed_var = float(parse[8])         #[7]
                        algo_info[curr_name] = [qoe, bit_rate, speed, freeze, change, latency, speed_change, 0.0]
                    else:
                        curr_bw = tuple(line.strip('\n').split('\t'))
                        # Use name get from QoE metrics
                        all_bw_info[curr_name] = curr_bw
                    n_line+=1
            else:
                for line in f:
                    parse = line.strip('\n')
                    parse = parse.split('\t')
                    # name_record.append(parse[0])
                    qoe = float(parse[1])               #[1]
                    bit_rate = float(parse[2])/1000.0          #[2]
                    speed = float(parse[3])             #[3]
                    freeze = float(parse[4])            #[4]
                    change = float(parse[5])            #[5]
                    latency = float(parse[6])           #[6]
                    speed_change = float(parse[7]) #[7]
                    speed_var = float(parse[8])      #[7]
                    n_fast = float(parse[9])
                    algo_info[parse[0]] = [qoe, bit_rate, speed, freeze, change, latency, speed_change, n_fast]
            all_info.append(algo_info)
    # print(algo_info)
    plot(all_info, all_bw_info)

def plot(all_info, all_bw_info):
    # Collect all results
    benchmark = all_info[1]     # Index of algorithm, which one is used to sort
    file_name_list = [k for k,v in sorted(benchmark.items(), key=lambda x:x[1][0])]
    coll_infos = []
    for i in range(len(ALGOS)):
        curr_name = ALGOS[i]  
        curr_res = []  
        for file in file_name_list:
            curr_res.append(all_info[i][file])
        coll_infos.append([curr_name, curr_res])

    # Plot
    data_len = len(coll_infos[0][1])
    plot_qoe(coll_infos, data_len)
    plot_rate(coll_infos, data_len)
    plot_speed(coll_infos, data_len)
    plot_freeze(coll_infos, data_len)
    plot_latency(coll_infos, data_len)
    plot_change(coll_infos, data_len)
    plot_speed_change(coll_infos, data_len)

    # Plot bw
    sort_bw = []
    max_bw = float('-inf')
    for file in file_name_list:
        list_bw = [float(i) for i in all_bw_info[file]]
        if max(list_bw) > max_bw:
            max_bw = max(list_bw)
        sort_bw.append(list_bw)
    min_len = min([len(x) for x in sort_bw])

    plot_bw([x[:min_len] for x in sort_bw], data_len, max_bw)

def traslate_markerwidth(name):
    if name == 'RATE':
        return 2
    else:
        return 1

def plot_bw(all_bw_info, data_len, max_bw):
    # p = plt.figure()
    # print(all_bw_info)
    fig, ax = plt.subplots(figsize=(20, 5))
    # for i in range(len(all_bw_info)):
    green_diamond = dict(markerfacecolor='g', marker='D', markersize=3, markeredgewidth=0.2)
    ax.boxplot(all_bw_info, flierprops=green_diamond)
    plt.xlabel('Environment Index', fontweight='bold', fontsize=26)
    plt.xticks(range(0, data_len+1, 30), range(0, data_len+1, 30), fontsize=22)
    plt.ylabel('Throughput (Mbps)', fontweight='bold', fontsize=22)
    plt.yticks(np.arange(0, max_bw//2*2+2, 3), fontsize=22)
    plt.axis([0, data_len+1, 0, max_bw+0.1])
    plt.subplots_adjust(top=0.99, right=0.99, bottom=0.15, left=0.15)
    fig.set_tight_layout(True)
    if FIGSHOW:
        fig.show()
        input()
    fig.savefig(SAVING_DIR + 'bw.eps', format='eps', dpi=1000, figsize=(20,5))

def plot_qoe(coll_infos, data_len):
    p = plt.figure(figsize=(10, 6))
    curr_min = float('inf')
    curr_max = float('-inf')
    for i in range(len(coll_infos)):
        curr_name = coll_infos[i][0]
        mw = traslate_markerwidth(curr_name)
        plot_data = [info[0] for info in coll_infos[i][1]]
        plt.plot(range(1,data_len+1), plot_data, LINE_TYPES[i], color=COLORS[i], label=curr_name, \
                markersize=6, markeredgewidth=mw)
        if np.amin(plot_data) < curr_min:
            curr_min = np.amin(plot_data)
        if np.amax(plot_data) > curr_max:
            curr_max = np.amax(plot_data)

    plt.legend(loc='upper left', fontsize = 22, ncol=2, frameon=True, handletextpad=0.1, columnspacing=0.2, markerscale=1.3)
    plt.xlabel('Environment Index', fontweight='bold', fontsize=26)
    plt.xticks(range(0, data_len+1, 30), fontsize=22)
    plt.ylabel('Total QoE ' + r'($\times 10^{2}$)', fontweight='bold', fontsize=22)
    plt.yticks(np.arange(curr_min//50*50, curr_max//50*50+50, 200), \
            [int(x/100) for x in np.arange(curr_min//50*50, curr_max//50*50+50, 200)],\
            fontsize=22)
    plt.axis([0, data_len+1, curr_min//50*50, curr_max//50*50+50])
    plt.subplots_adjust(top=0.99, right=0.99, bottom=0.15, left=0.15)
    p.set_tight_layout(True)
    
    if FIGSHOW:
        p.show()
        input()
    p.savefig(SAVING_DIR + 'qoe.eps', format='eps', dpi=1000, figsize=(10,6))

def plot_rate(coll_infos, data_len):
    p = plt.figure(figsize=(10, 6))
    curr_min = float('inf')
    curr_max = float('-inf')
    for i in range(len(coll_infos)):
        curr_name = coll_infos[i][0]
        mw = traslate_markerwidth(curr_name)
        plot_data = [info[1] for info in coll_infos[i][1]]
        plt.plot(range(1,data_len+1), plot_data, LINE_TYPES[i], color=COLORS[i], label=curr_name, \
            markersize=6, markeredgewidth=mw)
        if np.amin(plot_data) < curr_min:
            curr_min = np.amin(plot_data)
        if np.amax(plot_data) > curr_max:
            curr_max = np.amax(plot_data)
    plt.legend(loc='upper left', fontsize = 22, ncol=2, frameon=True, handletextpad=0.1, columnspacing=0.2, markerscale=1.3)
    plt.xlabel('Environment Index', fontweight='bold', fontsize=26)
    plt.xticks(range(0,data_len+1, 30), fontsize=22)
    plt.ylabel('Average Rate (Mbps)', fontweight='bold', fontsize=22)
    plt.yticks(np.arange(curr_min//0.5*0.5-0.5, curr_max//0.5*0.5+0.6, 2.0), fontsize=22)
    plt.axis([0, data_len+1, curr_min//0.5*0.5-0.5, curr_max//0.5*0.5+0.6])
    plt.subplots_adjust(top=0.99, right=0.99, bottom=0.15, left=0.15)
    p.set_tight_layout(True)
    if FIGSHOW:
        p.show()
        input()
    p.savefig(SAVING_DIR + 'rate.eps', format='eps', dpi=1000, figsize=(10,6))

def plot_speed(coll_infos, data_len):
    p = plt.figure(figsize=(10, 6))
    curr_min = float('inf')
    curr_max = float('-inf')
    for i in range(len(coll_infos)):
        curr_name = coll_infos[i][0]
        mw = traslate_markerwidth(curr_name)
        plot_data = [info[2]+info[7]*2/300.0 for info in coll_infos[i][1]]
        plt.plot(range(1,data_len+1), plot_data, LINE_TYPES[i], color=COLORS[i], label=curr_name, \
            markersize=6, markeredgewidth=mw)
        if np.amin(plot_data) < curr_min:
            curr_min = np.amin(plot_data)
        if np.amax(plot_data) > curr_max:
            curr_max = np.amax(plot_data)
    plt.legend(loc='upper right', fontsize = 22, ncol=2, frameon=True, handletextpad=0.1, columnspacing=0.2, markerscale=1.3)
    plt.xlabel('Environment Index', fontweight='bold', fontsize=26)
    plt.xticks(range(0,data_len+1, 30), fontsize=22)
    plt.ylabel('Average Speed', fontweight='bold', fontsize=22)
    plt.yticks(np.arange(0.98, 1.08, 0.02), fontsize=22)
    plt.axis([0, data_len+1, 0.995, 1.065])
    plt.subplots_adjust(top=0.99, right=0.99, bottom=0.15, left=0.15)
    p.set_tight_layout(True)
    if FIGSHOW:
        p.show()
        input()
    p.savefig(SAVING_DIR + 'speed.eps', format='eps', dpi=1000, figsize=(10,6))

def plot_freeze(coll_infos, data_len):
    curr_min = float('inf')
    curr_max = float('-inf')
    p = plt.figure(figsize=(10, 6))
    for i in range(len(coll_infos)):
        curr_name = coll_infos[i][0]
        mw = traslate_markerwidth(curr_name)
        plot_data = [info[3] for info in coll_infos[i][1]]
        plt.plot(range(1,data_len+1), plot_data, LINE_TYPES[i], color=COLORS[i], label=curr_name, \
            markersize=6, markeredgewidth=mw)
        if np.amin(plot_data) < curr_min:
            curr_min = np.amin(plot_data)
        if np.amax(plot_data) > curr_max:
            curr_max = np.amax(plot_data)
    plt.legend(loc='upper right', fontsize = 22, ncol=2, frameon=True, handletextpad=0.1, columnspacing=0.2, markerscale=1.3)
    plt.xlabel('Environment Index', fontweight='bold', fontsize=26)
    plt.xticks(range(0,data_len+1, 30), fontsize=22)
    plt.ylabel('Rebuffering (s)', fontweight='bold', fontsize=22)
    plt.yticks(np.arange(0, curr_max, 5000), [int(x/1000) for x in np.arange(0, curr_max+1, 5000)], fontsize=22)
    plt.subplots_adjust(top=0.99, right=0.99, bottom=0.15, left=0.15)
    plt.axis([0, data_len+1, 0, 11200])
    p.set_tight_layout(True)
    if FIGSHOW:
        p.show()
        input()
    p.savefig(SAVING_DIR + 'freeze.eps', format='eps', dpi=1000, figsize=(10,6))

def plot_change(coll_infos, data_len):
    p = plt.figure(figsize=(10, 6))
    curr_min = float('inf')
    curr_max = float('-inf')
    for i in range(len(coll_infos)):
        curr_name = coll_infos[i][0]
        mw = traslate_markerwidth(curr_name)
        plot_data = [info[4] for info in coll_infos[i][1]]
        plt.plot(range(1,data_len+1), plot_data, LINE_TYPES[i], color=COLORS[i], label=curr_name,\
            markersize=6, markeredgewidth=mw)
        if np.amin(plot_data) < curr_min:
            curr_min = np.amin(plot_data)
        if np.amax(plot_data) > curr_max:
            curr_max = np.amax(plot_data)
    plt.legend(loc='upper left', fontsize = 22, ncol=2, frameon=True, handletextpad=0.1, columnspacing=0.2, markerscale=1.3)
    plt.xlabel('Environment Index', fontweight='bold', fontsize=26)
    plt.xticks(range(0,data_len+1, 30), fontsize=22)
    plt.ylabel('Rate Fluctuation (Mbps)', fontweight='bold', fontsize=22)
    plt.yticks(np.arange(curr_min//100*100, curr_max//100*100+100, 300),\
     [x/1000 for x in np.arange(curr_min//100*100, curr_max//100*100+100, 300)], fontsize=22)
    plt.axis([0, data_len+1, 0, curr_max+10])
    plt.subplots_adjust(top=0.99, right=0.99, bottom=0.15, left=0.15)
    p.set_tight_layout(True)
    if FIGSHOW:
        p.show()
        input()
    p.savefig(SAVING_DIR + 'rate_change.eps', format='eps', dpi=1000, figsize=(10,6))

def plot_speed_change(coll_infos, data_len):
    p = plt.figure(figsize=(10, 6))
    curr_min = float('inf')
    curr_max = float('-inf')
    for i in range(len(coll_infos)):
        curr_name = coll_infos[i][0]
        mw = traslate_markerwidth(curr_name)
        plot_data = [info[6] for info in coll_infos[i][1]]
        plt.plot(range(1,data_len+1), plot_data, LINE_TYPES[i], color=COLORS[i], label=curr_name,\
            markersize=6, markeredgewidth=mw)
        if np.amin(plot_data) < curr_min:
            curr_min = np.amin(plot_data)
        if np.amax(plot_data) > curr_max:
            curr_max = np.amax(plot_data)
    plt.legend(loc='upper right', fontsize = 22, ncol=2, frameon=True, handletextpad=0.1, columnspacing=0.2, markerscale=1.3)
    plt.xlabel('Environment Index', fontweight='bold', fontsize=26)
    plt.xticks(range(0,data_len+1, 30), fontsize=22)
    plt.ylabel('Speed Change ' + r'$\times 10^{-3}$', fontweight='bold', fontsize=22)
    plt.yticks(np.arange(curr_min//0.005*0.005, curr_max//0.005*0.005+0.005, 0.005), \
        [int(1000*x) for x in np.arange(curr_min//0.005*0.005, curr_max//0.005*0.005+0.005, 0.005)], fontsize=22)
    plt.axis([0, data_len+1, -2*1e-3, curr_max+1e-3])
    plt.subplots_adjust(top=0.99, right=0.99, bottom=0.15, left=0.15)
    p.set_tight_layout(True)
    # plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    if FIGSHOW:
        p.show()
        input()
    p.savefig(SAVING_DIR + 'speed_change.eps', format='eps', dpi=1000, figsize=(10,6))


def plot_latency(coll_infos, data_len):
    p = plt.figure(figsize=(10, 6))
    curr_min = float('inf')
    curr_max = float('-inf')
    for i in range(len(coll_infos)):
        curr_name = coll_infos[i][0]
        mw = traslate_markerwidth(curr_name)
        plot_data = [info[5] for info in coll_infos[i][1]]
        plt.plot(range(1,data_len+1), plot_data, LINE_TYPES[i], color=COLORS[i], label=curr_name ,\
            markersize=6, markeredgewidth=mw)
        if np.amin(plot_data) < curr_min:
            curr_min = np.amin(plot_data)
        if np.amax(plot_data) > curr_max:
            curr_max = np.amax(plot_data)
    plt.legend(loc='lower center', fontsize = 22, ncol=4, frameon=True, handletextpad=0.1, columnspacing=0.2, markerscale=1.3)
    plt.xlabel('Environment Index', fontweight='bold', fontsize=26)
    plt.xticks(range(0,data_len+1, 30), fontsize=22)
    plt.ylabel('Average Latency (s)', fontweight='bold', fontsize=22)
    plt.yticks(np.arange(0, curr_max, 2000), [int(x/1000) for x in np.arange(0, curr_max, 2000)], fontsize=22)
    plt.subplots_adjust(top=0.99, right=0.99, bottom=0.15, left=0.15)
    plt.axis([0, data_len+1, 0, curr_max+100])
    p.set_tight_layout(True)
    if FIGSHOW:
        p.show()
        input()
    p.savefig(SAVING_DIR + 'latency.eps', format='eps', dpi=1000, figsize=(10,6))

if __name__ == '__main__':
    main()