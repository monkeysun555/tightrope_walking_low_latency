import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

FIGSHOW = True
SAVING_DIR = './metric_box/'
RES_DIR = './compare_results/'
# ALGOS = ['RATE', 'iLQR', 'DDQN', 'BDQ', 'iLQR*', 'iLQR**']
# ALGOS = ['Pensieve', 'iLQR', 'DDQN', 'BDQ', 'iLQR*', 'iLQR**']
ALGOS = ['Pensieve', 'STALLION', 'iLQR', 'DDQN', 'BDQ','iLQR*']

LINE_TYPES = ['+', '*', 'h', 'd', '.']
# COLORS = ['#1f77b4',  '#ff7f0e',  '#2ca02c',
#           '#d62728', '#9467bd',
#           '#8c564b', '#e377c2', '#7f7f7f',
#           '#bcbd22', '#17becf']
COLORS = ['#8c564b', '#1f77b4',  '#ff7f0e', 
          '#2ca02c', '#9467bd',
          '#e377c2', '#7f7f7f',
          '#bcbd22', '#17becf']
ORANGE_CIRCLE = dict(markerfacecolor='orange', marker='o', markersize=3, markeredgewidth=0.2)
MEDIANPROPS = dict(linestyle='-', linewidth=2, color='red')
BOXPROPS = dict(linestyle='-', linewidth=2, color='black')

def name_translate(name):
    if name == 'iLQR':
        return 'dyn_mpc_normal.txt'
    elif name == 'DDQN':
        return 'naive_speed_normal.txt'
    elif name == 'RATE':
        return 'rate_adaption_normal.txt'
    elif name == 'BDQ':
        return 'multi_speed_normal.txt'
    elif name == 'iLQR*':
        return 'dyn_mpc_opt.txt'
    elif name == 'iLQR**':
        return 'dyn_mpc_opt_enhanced.txt'
    elif name == 'Pensieve':
        return 'pensieve.txt'
    elif name == 'STALLION':
        return 'stallion.txt'

def main():
    if not os.path.exists(SAVING_DIR):
        os.makedirs(SAVING_DIR)
    all_info = []
    all_bw_info = dict()
    for algo in ALGOS:
        res_name = name_translate(algo)
        file_path = RES_DIR + res_name
        print(file_path)
        with open(file_path, 'r') as f:
            algo_info = dict()
            if algo == 'RATE' or algo == 'Pensieve':
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
    # plot_speed_change(coll_infos, data_len)

    # Plot bw
    # sort_bw = []
    # max_bw = float('-inf')
    # for file in file_name_list:
    #     list_bw = [float(i) for i in all_bw_info[file]]
    #     if max(list_bw) > max_bw:
    #         max_bw = max(list_bw)
    #     sort_bw.append(list_bw)
    # min_len = min([len(x) for x in sort_bw])
    # sort_bw = sorted(sort_bw, key = lambda x:np.median(x))
    # plot_bw([x[:min_len] for x in sort_bw], data_len, max_bw)

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
    plt.subplots_adjust(top=0.99, right=0.96, bottom=0.16, left=0.15)
    fig.set_tight_layout(True)
    if FIGSHOW:
        fig.show()
        input()
    fig.savefig(SAVING_DIR + 'bw.eps', format='eps', dpi=1000, figsize=(20,5), bbox_inches='tight')


def plot_change(coll_infos, data_len):
    fig, ax = plt.subplots(figsize=(4, 5))
    curr_min = float('inf')
    curr_max = float('-inf')

    bar_datas = []
    x_position = []
    name_list = []
    bar_width = 0.3
    gap_width = 0.3
    pre_location = 0
    bar_widths = [bar_width for i in range(len(coll_infos))]
    colors = ['pink', 'lightblue', 'lightgreen','gray', 'blueviolet', 'orange']
    for i in range(len(coll_infos)):
        curr_name = coll_infos[i][0]
        name_list.append(curr_name)
        mw = traslate_markerwidth(curr_name)
        plot_data = [info[4] for info in coll_infos[i][1]]
        bar_datas.append(plot_data)
        if np.amin(plot_data) < curr_min:
            curr_min = np.amin(plot_data)
        if np.amax(plot_data) > curr_max:
            curr_max = np.amax(plot_data)
        if i == 0:
            pre_location += 0.5*(gap_width + bar_width)
        else:
            pre_location += gap_width + bar_width
        x_position.append(pre_location)
    
    bplots = ax.boxplot(bar_datas, positions = x_position, widths = bar_widths, \
                flierprops=ORANGE_CIRCLE, patch_artist=True, medianprops=MEDIANPROPS, boxprops=BOXPROPS, notch=True)
    for patch, color in zip(bplots['boxes'], colors):
        patch.set_facecolor(color)
    x_position1 = [x+0.25*(gap_width + bar_width) for x in x_position]
    # x_position1[0] -= 0.41*(gap_width + bar_width)
    # x_position1[1] -= 0.41*(gap_width + bar_width)
    plt.xticks(x_position1, name_list, rotation=45, fontsize=22, verticalalignment='top', horizontalalignment='right')

    plt.ylabel('Rate Fluctuation (Mbps)', fontweight='bold', fontsize=20)
    plt.yticks(np.arange(curr_min//100*100, curr_max//100*100+100, 300),\
     [x/1000 for x in np.arange(curr_min//100*100, curr_max//100*100+100, 300)], fontsize=22)
    ax.tick_params(axis=u'x', length=0)
    plt.axis([0.05, x_position[-1]+0.5*bar_width+0.1, 0, curr_max+10])
    plt.subplots_adjust(top=0.99, right=0.96, bottom=0.25, left=0.25)
    # p.set_tight_layout(True)
    if FIGSHOW:
        fig.show()
        input()
    fig.savefig(SAVING_DIR + 'bar_rate_change.eps', format='eps', dpi=1000, figsize=(4,5))

def plot_latency(coll_infos, data_len):
    fig, ax = plt.subplots(figsize=(4, 5))
    curr_min = float('inf')
    curr_max = float('-inf')

    bar_datas = []
    x_position = []
    name_list = []
    bar_width = 0.3
    gap_width = 0.3
    pre_location = 0
    bar_widths = [bar_width for i in range(len(coll_infos))]
    colors = ['pink', 'lightblue', 'lightgreen','gray', 'blueviolet', 'orange']
    for i in range(len(coll_infos)):
        curr_name = coll_infos[i][0]
        name_list.append(curr_name)
        mw = traslate_markerwidth(curr_name)
        plot_data = [info[5] for info in coll_infos[i][1]]
        bar_datas.append(plot_data)
        if np.amin(plot_data) < curr_min:
            curr_min = np.amin(plot_data)
        if np.amax(plot_data) > curr_max:
            curr_max = np.amax(plot_data)
        if i == 0:
            pre_location += 0.5*(gap_width + bar_width)
        else:
            pre_location += gap_width + bar_width
        x_position.append(pre_location)
    bplots = ax.boxplot(bar_datas, positions = x_position, widths = bar_widths, \
                flierprops=ORANGE_CIRCLE, patch_artist=True, medianprops=MEDIANPROPS, boxprops=BOXPROPS, notch=True)
    for patch, color in zip(bplots['boxes'], colors):
        patch.set_facecolor(color)
    x_position1 = [x+0.25*(gap_width + bar_width) for x in x_position]
    # x_position1[0] -= 0.41*(gap_width + bar_width)
    # x_position1[1] -= 0.41*(gap_width + bar_width)
    plt.xticks(x_position1, name_list, rotation=45, fontsize=22, verticalalignment='top', horizontalalignment='right')
    plt.ylabel('Average Latency (s)', fontweight='bold', fontsize=22)

    plt.yticks(np.arange(1000, curr_max, 2000), [int(x/1000) for x in np.arange(1000, curr_max, 2000)], fontsize=22)
    ax.tick_params(axis=u'x', length=0)
    plt.subplots_adjust(top=0.99, right=0.96, bottom=0.25, left=0.21)
    plt.axis([0.05, x_position[-1]+0.5*bar_width+0.1, 1000, curr_max+100])      # in MS
    # p.set_tight_layout(True)
    if FIGSHOW:
        fig.show()
        input()
    fig.savefig(SAVING_DIR + 'bar_latency.eps', format='eps', dpi=1000, figsize=(4,5))


def plot_qoe(coll_infos, data_len):
    fig, ax = plt.subplots(figsize=(4, 5))
    curr_min = float('inf')
    curr_max = float('-inf')

    bar_datas = []
    x_position = []
    name_list = []
    bar_width = 0.3
    gap_width = 0.3
    pre_location = 0
    bar_widths = [bar_width for i in range(len(coll_infos))]
    colors = ['pink', 'lightblue', 'lightgreen','gray', 'blueviolet', 'orange']
    for i in range(len(coll_infos)):
        curr_name = coll_infos[i][0]
        name_list.append(curr_name)
        mw = traslate_markerwidth(curr_name)
        plot_data = [info[0] for info in coll_infos[i][1]]
        bar_datas.append(plot_data)
        if np.amin(plot_data) < curr_min:
            curr_min = np.amin(plot_data)
        if np.amax(plot_data) > curr_max:
            curr_max = np.amax(plot_data)
        if i == 0:
            pre_location += 0.5*(gap_width + bar_width)
        else:
            pre_location += gap_width + bar_width
        x_position.append(pre_location)
    bplots = ax.boxplot(bar_datas, positions = x_position, widths = bar_widths, \
                flierprops=ORANGE_CIRCLE, patch_artist=True, medianprops=MEDIANPROPS, boxprops=BOXPROPS, notch=True)
    for patch, color in zip(bplots['boxes'], colors):
        patch.set_facecolor(color)
    x_position1 = [x+0.25*(gap_width + bar_width) for x in x_position]
    # x_position1[0] -= 0.41*(gap_width + bar_width)
    # x_position1[1] -= 0.41*(gap_width + bar_width)
    plt.xticks(x_position1, name_list, rotation=45, fontsize=22, verticalalignment='top', horizontalalignment='right')
    # plt.xlabel('Algorithms', fontweight='bold', fontsize=22)
    plt.yticks(np.arange(curr_min//50*50, curr_max//50*50+50, 200), \
            [int(x/100) for x in np.arange(curr_min//50*50, curr_max//50*50+50, 200)],\
            fontsize=22)
    ax.tick_params(axis=u'x', length=0)
    plt.ylabel('QoE ' + r'($\times 10^{2}$)', fontweight='bold', fontsize=22)
    plt.axis([0.05, x_position[-1]+0.5*bar_width+0.1, curr_min//50*50, curr_max//50*50+50])
    plt.subplots_adjust(top=0.99, right=0.96, bottom=0.25, left=0.22)
    # p.set_tight_layout(True)
    
    if FIGSHOW:
        fig.show()
        input()
    fig.savefig(SAVING_DIR + 'bar_qoe.eps', format='eps', dpi=1000, figsize=(4,5))

def plot_rate(coll_infos, data_len):
    fig, ax = plt.subplots(figsize=(4, 5))
    curr_min = float('inf')
    curr_max = float('-inf')
    bar_datas = []
    x_position = []
    name_list = []
    bar_width = 0.3
    gap_width = 0.3
    pre_location = 0
    bar_widths = [bar_width for i in range(len(coll_infos))]
    colors = ['pink', 'lightblue', 'lightgreen','gray', 'blueviolet', 'orange']
    for i in range(len(coll_infos)):
        curr_name = coll_infos[i][0]
        name_list.append(curr_name)
        mw = traslate_markerwidth(curr_name)
        plot_data = [info[1] for info in coll_infos[i][1]]
        bar_datas.append(plot_data)
        if np.amin(plot_data) < curr_min:
            curr_min = np.amin(plot_data)
        if np.amax(plot_data) > curr_max:
            curr_max = np.amax(plot_data)
        if i == 0:
            pre_location += 0.5*(gap_width + bar_width)
        else:
            pre_location += gap_width + bar_width
        x_position.append(pre_location)

    bplots = ax.boxplot(bar_datas, positions = x_position, widths = bar_widths, \
                flierprops=ORANGE_CIRCLE, patch_artist=True, medianprops=MEDIANPROPS, boxprops=BOXPROPS, notch=True)
    for patch, color in zip(bplots['boxes'], colors):
        patch.set_facecolor(color)

    x_position1 = [x+0.25*(gap_width + bar_width) for x in x_position]
    # x_position1[0] -= 0.41*(gap_width + bar_width)
    # x_position1[1] -= 0.41*(gap_width + bar_width)
    plt.xticks(x_position1, name_list, rotation=45, fontsize=22, verticalalignment='top', horizontalalignment='right')
    # ax.set_xticklabels([])
    plt.ylabel('Average Rate (Mbps)', fontweight='bold', fontsize=22)
    plt.yticks(np.arange(curr_min//0.5*0.5, curr_max//0.5*0.5, 2.0), fontsize=22)
    ax.tick_params(axis=u'x', length=0)
    plt.axis([0.05, x_position[-1]+0.5*bar_width+0.1, curr_min//0.5*0.5-0.5, curr_max//0.5*0.5+0.6])
    plt.subplots_adjust(top=0.985, right=0.96, bottom=0.25, left=0.22)
    if FIGSHOW:
        fig.show()
        input()
    fig.savefig(SAVING_DIR + 'bar_rate.eps', format='eps', dpi=1000, figsize=(4,5))

def plot_speed(coll_infos, data_len):
    fig, ax = plt.subplots(figsize=(4, 5))
    curr_min = float('inf')
    curr_max = float('-inf')
    bar_datas = []
    x_position = []
    name_list = []
    bar_width = 0.3
    gap_width = 0.3
    pre_location = 0
    bar_widths = [bar_width for i in range(len(coll_infos))]
    colors = ['pink', 'lightblue', 'lightgreen','gray', 'blueviolet', 'orange']
    for i in range(len(coll_infos)):
        curr_name = coll_infos[i][0]
        name_list.append(curr_name)
        mw = traslate_markerwidth(curr_name)
        plot_data = [info[2]+info[7]*2/300.0 for info in coll_infos[i][1]]
        bar_datas.append(plot_data)
        if np.amin(plot_data) < curr_min:
            curr_min = np.amin(plot_data)
        if np.amax(plot_data) > curr_max:
            curr_max = np.amax(plot_data)
        if i == 0:
            pre_location += 0.5*(gap_width + bar_width)
        else:
            pre_location += gap_width + bar_width
        x_position.append(pre_location)

    bplots = ax.boxplot(bar_datas, positions = x_position, widths = bar_widths, \
                flierprops=ORANGE_CIRCLE, patch_artist=True, medianprops=MEDIANPROPS, boxprops=BOXPROPS, notch=True)
    for patch, color in zip(bplots['boxes'], colors):
        patch.set_facecolor(color)

    x_position1 = [x+0.25*(gap_width + bar_width) for x in x_position]
    # x_position1[0] -= 0.41*(gap_width + bar_width)
    # x_position1[1] -= 0.41*(gap_width + bar_width)
    plt.xticks(x_position1, name_list, rotation=45, fontsize=22, verticalalignment='top', horizontalalignment='right')
    plt.ylabel('Average Speed', fontweight='bold', fontsize=22)
    plt.yticks(np.arange(0.99, 1.04, 0.01), fontsize=22)
    ax.tick_params(axis=u'x', length=0)
    plt.axis([0.05, x_position[-1]+0.5*bar_width+0.1, 0.998, 1.035])
    plt.subplots_adjust(top=0.99, right=0.96, bottom=0.25, left=0.3)
    if FIGSHOW:
        fig.show()
        input()
    fig.savefig(SAVING_DIR + 'bar_speed.eps', format='eps', dpi=1000, figsize=(4,5))

def plot_freeze(coll_infos, data_len):
    fig, ax = plt.subplots(figsize=(4, 5))
    curr_min = float('inf')
    curr_max = float('-inf')
    bar_datas = []
    x_position = []
    name_list = []
    bar_width = 0.3
    gap_width = 0.3
    pre_location = 0
    bar_widths = [bar_width for i in range(len(coll_infos))]
    colors = ['pink', 'lightblue', 'lightgreen','gray', 'blueviolet', 'orange']
    for i in range(len(coll_infos)):
        curr_name = coll_infos[i][0]
        name_list.append(curr_name)
        mw = traslate_markerwidth(curr_name)
        plot_data = [info[3] for info in coll_infos[i][1]]
        bar_datas.append(plot_data)
        if np.amin(plot_data) < curr_min:
            curr_min = np.amin(plot_data)
        if np.amax(plot_data) > curr_max:
            curr_max = np.amax(plot_data)
        if i == 0:
            pre_location += 0.5*(gap_width + bar_width)
        else:
            pre_location += gap_width + bar_width
        x_position.append(pre_location)

    bplots = ax.boxplot(bar_datas, positions = x_position, widths = bar_widths, \
                flierprops=ORANGE_CIRCLE, patch_artist=True, medianprops=MEDIANPROPS, boxprops=BOXPROPS, notch=True)
    for patch, color in zip(bplots['boxes'], colors):
        patch.set_facecolor(color)

    x_position1 = [x+0.25*(gap_width + bar_width) for x in x_position]
    # x_position1[0] -= 0.41*(gap_width + bar_width)
    # x_position1[1] -= 0.41*(gap_width + bar_width)
    plt.xticks(x_position1, name_list, rotation=45, fontsize=22, verticalalignment='top', horizontalalignment='right')
    plt.ylabel('Freeze (s)', fontweight='bold', fontsize=22)
    plt.yticks(np.arange(0, 12000, 3000), [int(x/1000) for x in np.arange(0, 12000, 3000)], fontsize=22)
    ax.tick_params(axis=u'x', length=0)
    plt.subplots_adjust(top=0.975, right=0.96, bottom=0.25, left=0.23)
    plt.axis([0.05, x_position[-1]+0.5*bar_width+0.1, 0, 9000])
    # p.set_tight_layout(True)
    if FIGSHOW:
        fig.show()
        input()
    fig.savefig(SAVING_DIR + 'bar_freeze.eps', format='eps', dpi=1000, figsize=(4,5))

# def plot_speed_change(coll_infos, data_len):
#     fig, ax = plt.subplots(figsize=(4, 6))
#     curr_min = float('inf')
#     curr_max = float('-inf')

#     bar_datas = []
#     x_position = []
#     name_list = []
#     bar_width = 0.3
#     gap_width = 0.3
#     pre_location = 0
#     bar_widths = [bar_width for i in range(len(coll_infos))]
#     colors = ['pink', 'lightblue', 'lightgreen','gray']
#     for i in range(len(coll_infos)):
#         curr_name = coll_infos[i][0]
#         name_list.append(curr_name)
#         mw = traslate_markerwidth(curr_name)
#         plot_data = [info[6] for info in coll_infos[i][1]]
#         bar_datas.append(plot_data)
#         if np.amin(plot_data) < curr_min:
#             curr_min = np.amin(plot_data)
#         if np.amax(plot_data) > curr_max:
#             curr_max = np.amax(plot_data)
#         if i == 0:
#             pre_location += 0.5*(gap_width + bar_width)
#         else:
#             pre_location += gap_width + bar_width
#         x_position.append(pre_location)
#     orange_circle = dict(markerfacecolor='orange', marker='o', markersize=3, markeredgewidth=0.2)
#     bplots = ax.boxplot(bar_datas, positions = x_position, widths = bar_widths, flierprops=orange_circle, patch_artist=True)
#     for patch, color in zip(bplots['boxes'], colors):
#         patch.set_facecolor(color)
#     plt.xticks(x_position, name_list, rotation=45, fontsize=22)

#     plt.ylabel('Speed Change ' + r'$\times 10^{-3}$', fontweight='bold', fontsize=22)
#     plt.yticks(np.arange(curr_min//0.005*0.005, curr_max//0.005*0.005+0.005, 0.005), \
#         [int(1000*x) for x in np.arange(curr_min//0.005*0.005, curr_max//0.005*0.005+0.005, 0.005)], fontsize=22)
#     plt.axis([0.05, x_position[-1]+0.5*bar_width+0.1, -2*1e-3, curr_max+1e-3])
#     plt.subplots_adjust(top=0.99, right=0.99, bottom=0.16, left=0.22)
#     # p.set_tight_layout(True)
#     # plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
#     if FIGSHOW:
#         fig.show()
#         input()
#     fig.savefig(SAVING_DIR + 'bar_speed_change.eps', format='eps', dpi=1000, figsize=(10,6))


if __name__ == '__main__':
    main()