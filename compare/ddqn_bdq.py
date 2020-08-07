import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

SEED = 33
ALGOS = ['DDQN', 'BDQ']
FIG_SHOW = 1

COLORS = ['#8c564b', '#1f77b4',  '#ff7f0e', 
          '#2ca02c', '#9467bd',
          '#e377c2', '#7f7f7f',
          '#bcbd22', '#17becf']
MS_IN_S = 1000.0
PLT_BUFFER_A = 1e-5
KB_IN_MB = 1000.0
def main():
    np.random.seed(SEED)
    ddqn_path = '../dyn_naive_general(inuse)/all_results/model_0/latency_Nones/'
    ddqn_results = os.listdir(ddqn_path)
    trace_idx = np.random.randint(0, len(ddqn_results))
    trace_name = ddqn_results[trace_idx]

    # DDQN
    ddqn_info = []
    ddqn_file = ddqn_path + trace_name
    with open(ddqn_file, 'r') as f:
        for line in f:
            parse = line.strip('\n')
            parse = parse.split('\t')               
            ddqn_info.append(parse)

    # BDQ
    bdq_path = '../dyn_multispeed_torch/all_results/model_2/latency_Nones/'
    bdq_info = []
    bdq_file = bdq_path + trace_name
    with open(bdq_file, 'r') as f:
        for line in f:
            parse = line.strip('\n')
            parse = parse.split('\t')               
            bdq_info.append(parse)   

    plot_compare(ddqn_info, bdq_info)

def plot_compare(ddqn, bdq):
    assert ddqn[-1][0] == bdq[-1][0]
    starting_time = float(ddqn[-1][0])
    data_name = ddqn[0]
    tp_trace = np.array(ddqn[-3]).astype(np.float)
    new_time_traces = []
    ddqn_records = ddqn[1:-3]
    new_time_traces.append(np.array(ddqn[-2]).astype(np.float))

    bdq_records = bdq[1:-3]
    new_time_traces.append(np.array(bdq[-2]).astype(np.float))

    plt_buffer_rate([ddqn_records, bdq_records], starting_time, tp_trace, new_time_traces) 


def plt_buffer_rate(records, starting_time, tp_trace, new_time_traces):
    buffer_axis_upper = 6.5
    figure = plt.figure(figsize=(10, 5))
    ax1 = plt.subplot(212)
    ymax = starting_time/MS_IN_S*1.1
    buffer_lines = []
    for i in range(len(records)):
        buffer_trace = [float(info[2]) for info in records[i]]
        freezing_trace = [float(info[3]) for info in records[i]]
        server_wait_trace = [float(info[5]) for info in records[i]]
        latency_trace = [starting_time/MS_IN_S] + [float(info[6])/MS_IN_S for info in records[i]]
        state_trace = [float(info[7]) for info in records[i]]
        # speed = [float(info[9].split('(')[1].split(')')[0]) for info in records[i]]
        reward_trace = [float(info[-1]) for info in records[i]]
        real_time_trace = [float(info[0]) for info in records[i]]
        time_trace = [(r_time - starting_time) for r_time in real_time_trace]

        latency_time = [0.0] + [x/MS_IN_S for x in time_trace]
        time_trace = [0.0] + time_trace
        buffer_trace = [0.0] + buffer_trace
        state_trace = [0] + state_trace
        insert_buffer_trace = []
        insert_time_trace = []
        plot_state_left = 1
        # Process plot trace
        assert len(time_trace) == len(buffer_trace)
        for j in range(0, len(time_trace)):
            if state_trace[j] == 0:
                if j >= 1:
                    if state_trace[j-1] == 1:
                        insert_time = np.minimum(time_trace[i] - PLT_BUFFER_A, time_trace[j-1]+buffer_trace[j-1])
                        insert_buffer = np.maximum(0.0, buffer_trace[j-1] - (time_trace[j] - time_trace[j-1]))
                        insert_buffer_trace.append(insert_buffer)
                        insert_time_trace.append(insert_time)
                plot_state_left = 1
                continue
            else:
                if plot_state_left:
                    plot_state_left = 0
                    continue
                insert_buffer = np.maximum(0.0, buffer_trace[j-1] - (time_trace[j] - time_trace[j-1]))
                insert_time = np.minimum(time_trace[j] - PLT_BUFFER_A, time_trace[j-1]+buffer_trace[j-1])
                insert_buffer_trace.append(insert_buffer)
                insert_time_trace.append(insert_time)
                if insert_time < time_trace[j] - PLT_BUFFER_A:
                    assert insert_buffer == 0.0
                    insert_buffer_trace.append(0.0)
                    insert_time_trace.append(time_trace[j] - PLT_BUFFER_A)

        # Need to adjust about freezing
        # combine two buffer_traces
        plt_buffer_trace = []
        plt_time_trace = []
        # print(len(insert_time_trace), len(time_trace), len(latency_trace))
        # print(insert_time_trace[-1], time_trace[-1])
        # print(latency_trace)
        # print(len(insert_time_trace), len(insert_buffer_trace))
        for j in range(len(time_trace)):
            # if len(insert_time_trace) == 0:
            #   plt_time_trace.append(time_trace[i:])
            #   plt_buffer_trace.append(buffer_trace[i:])
            #   break
            # print(i, len(time_trace))
            if len(insert_time_trace) > 0:
                while insert_time_trace[0] < time_trace[j]:
                    plt_time_trace.append(insert_time_trace.pop(0)/MS_IN_S)
                    plt_buffer_trace.append(insert_buffer_trace.pop(0)/MS_IN_S)
                    # print(len(insert_time_trace), len(time_trace), i)
                    if len(insert_time_trace) == 0:
                        # plt_time_trace.extend(time_trace[i:])
                        # plt_buffer_trace.extend(buffer_trace[i:])
                        break
            plt_time_trace.append(time_trace[j]/MS_IN_S)
            plt_buffer_trace.append(buffer_trace[j]/MS_IN_S)

        ## Add ending phase
        curr_buff = plt_buffer_trace[-1]
        curr_time = plt_time_trace[-1]
        curr_latency_time = latency_time[-1]
        curr_latency = latency_trace[-1]
        while curr_buff >= 1:
            plt_buffer_trace.append(curr_buff-1)
            plt_time_trace.append(curr_time+1)
            curr_buff-=1
            curr_time+=1
            latency_time.append(curr_latency_time+1)
            latency_trace.append(curr_latency)
            curr_latency_time+=1
        plt_buffer_trace.append(0.0)
        plt_time_trace.append(plt_time_trace[-1] + plt_buffer_trace[-2])
        latency_time.append(300)
        latency_trace.append(curr_latency)

        if i == 0:
            color = COLORS[1]
        elif i == 1:
            color = 'chocolate'
        buffer_lines += plt.plot(plt_time_trace, plt_buffer_trace, color=color, linewidth=1.1)
        buffer_lines += plt.plot(latency_time, latency_trace, color=color, linestyle='--', linewidth=1.5)
    # plt.xlabel('Buffer & Latency', fontweight='bold', fontsize=14)
    ax1.legend(buffer_lines, ['DDQN Buffer','DDQN Latency', 'BDQ Buffer', 'BDQ Latency'], loc='best', \
            fontsize=14, ncol=2, columnspacing=0.7, handlelength= 1.5, labelspacing=0.2, frameon=True)
    plt.xticks(np.arange(0, 301, 50), fontsize=14)
    plt.xlabel('Time (s)', fontweight='bold', fontsize=14)
    plt.ylabel('Second', fontweight='bold', fontsize=14)
    plt.yticks(range(0,12,2), fontsize=14)
    plt.tick_params(labelsize=14)
    ax1.set_xlim([0,300])
    ax1.set_ylim([0,ymax])
    plt.subplots_adjust(top=0.95, right=0.98, bottom=0.12, left=0.04)

    bbox_props = dict(boxstyle="square,pad=0.4", fc="w", ec='r')
    kw = dict(arrowprops=dict(arrowstyle="->",color='r',
                             lw=2.0), bbox=bbox_props, zorder=3, va="center", ha='center')

    horizontalalignment = 'left'
    connectionstyle = "angle,angleA=115,angleB=0"
    kw["arrowprops"].update({"connectionstyle": connectionstyle})

    ax1.annotate('Skip', xy=(1, 5), xytext=(55, 4),
                    color='r', fontweight='bold', fontsize=14, horizontalalignment=horizontalalignment, **kw)

    horizontalalignment_2 = 'right'
    connectionstyle_2 = "angle,angleA=-115,angleB=0"
    kw["arrowprops"].update({"connectionstyle": connectionstyle_2, 'zorder':1})
    ax1.annotate('Skip', xy=(27, 2.7), xytext=(55, 4),
                    color='r', fontweight='bold', fontsize=14, horizontalalignment=horizontalalignment_2, **kw)

    #######################
    #######################
    #######################
    #######################
    #######################
    #######################
    # For rate
    #######################
    #######################
    #######################
    #######################
    #######################
    #######################

    ax2 = plt.subplot(211)
    rate_lines = []
    plot_tp = True
    ymax = -1
    for i in range(len(records)):
        bitrate_trace = [float(info[1])/KB_IN_MB for info in records[i]]
        if max(bitrate_trace)*1.2 > ymax:
            ymax = max(bitrate_trace)*1.2 
        init_time = new_time_traces[i][0]
        new_time_trace = [time-init_time for time in new_time_traces[i]]
        y_axis_upper = 10.0
        # For negative reward
        # y_axis_lower = np.floor(np.minimum(np.min(trace)*1.1,0.0))
        y_axis_lower = 0.0

        # For bitrate
        x_value = []
        y_value = []
        curr_x = 0.0
        for j in range(len(bitrate_trace)):
            x_value.append(curr_x)
            x_value.append(curr_x+1000.0/MS_IN_S-1e-5)  # Seg duration
            y_value.append(bitrate_trace[j])
            y_value.append(bitrate_trace[j])
            curr_x += 1000.0/MS_IN_S # Seg duration

        # For bw
        new_time = [i+0.5 for i in range(len(tp_trace))]
        if plot_tp:
            rate_lines += plt.plot(new_time, tp_trace, color=COLORS[6], label='Bandwidth', linewidth=1.5, zorder=0, alpha=1.)
            plot_tp = False
        if i == 0:
            color = COLORS[1]
            alpha = 0.99
        elif i == 1:
            color = 'chocolate'
            alpha = 0.99
        rate_lines += plt.plot(x_value, y_value, color=color, linewidth=1.8, alpha=1)
        # plt.plot(range(1,len(trace1)+1), trace1*Env_Config.kb_in_mb, color='blue', label=data_name + '_' + data_type1, linewidth=1.5,alpha=0.9)
        # print(len(new_time_trace))
        # print(len(trace1))

    leg_0 = ax2.legend(rate_lines,['Bandwidth', 'DDQN Rate','BDQ Rate'], loc='lower center', \
        fontsize=14, ncol=3, borderpad = 0.3, columnspacing=0.7, handlelength= 1.5, labelspacing=0.2, frameon=True)
    # plt.xlabel('Time (s)', fontweight='bold', fontsize=14)
    plt.xticks(np.arange(0, 301, 50), fontsize=14)
    plt.ylabel('Rate (Mbps)', fontweight='bold', fontsize=14)
    plt.yticks(np.arange(0, 10.01, 2), [int(x) for x in np.arange(0, 10.01, 2)], fontsize=22)
    plt.tick_params(labelsize=14)
    ax2.set_xlim([0,300])
    ax2.set_ylim([0,ymax])    
    plt.subplots_adjust(top=0.95, right=0.98, bottom=0.12, left=0.06)

    if FIG_SHOW:
        figure.show()
        input()
    fig_path = './ddqn_bdq/'
    if not os.path.isdir(fig_path):
        os.makedirs(fig_path)
    figure.savefig(fig_path + 'ddqn_bdq_compare.eps', format='eps', dpi=1000, figsize=(10, 5))
    plt.close()

if __name__ == '__main__':
    main()

