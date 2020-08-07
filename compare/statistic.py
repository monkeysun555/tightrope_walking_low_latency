# To plot speed pie

import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

ALGOS = ['RATE', 'iLQR', 'DDQN', 'BDQ', 'iLQR*', 'iLQR**']
ALGOS = ['Pensieve', 'iLQR', 'DDQN', 'BDQ', 'iLQR*', 'iLQR**']

SHOW = 1
n_compare = 8

COLORS_3 = ['C8', 'C1', 'C2']
COLORS_7 = ['C4', 'C0', 'C8', 'C1', 'C2', 'C9', 'lightpink']
COLORS_5 = ['C0', 'C8', 'C1', 'C2', 'C9']


def read_data():
    metric_path = './compare_results/'
    all_info = []
    all_bw_info = dict()
    name_list = []
    for algo in ALGOS:
        res_name = name_translate(algo)
        file_path = metric_path + res_name
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
                    name_list.append(parse[0])
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

    return all_info, all_bw_info

def name_translate(name):
    if name == 'iLQR':
        return 'dyn_mpc_normal_new.txt'
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

def get_director(name):
    if name == 'iLQR':
        return '../dyn_mpc_latency_normal/all_results/latency_Nones/' 
    elif name == 'DDQN':
        return '../dyn_naive_general(inuse)/all_results/model_0/latency_Nones/' 
    elif name == 'RATE':
        return '../rate_adaption_torch/all_results/model_0/latency_Nones/' 
    elif name == 'BDQ':
        return '../dyn_multispeed_torch/all_results/model_2/latency_Nones/' 
    elif name == 'iLQR*':
        return '../dyn_mpc_latency_opt_massive/all_results/latency_Nones/' 
    elif name == 'iLQR**':
        return '../dyn_mpc_latency_opt_massive_enhanced/all_results/latency_Nones/'
    elif name == 'Pensieve':
        return '../benchmark_pensieve/results/pensieve/'

def transform_to_index(speed):
    return int(speed.split('(')[1].split(')')[0])

def show_speeds(all_info, all_bw_info, selected_name):
    all_speeds = []
    print(selected_name)
    for i in range(len(selected_name)):
        trace_speeds = []
        for j in range(len(ALGOS)):
            if j == 0:
                trace_speeds.append([0,300,0])
                continue
            curr_name = ALGOS[j]  
            detailed_res = get_director(curr_name)
            file = detailed_res + selected_name[i]
            line_count = 0
            if j == 3:
                algo_speeds = [0] * 7
            elif j == 5:
                algo_speeds = [0] * 5
            else:
                algo_speeds = [0] * 3

            with open(file, 'r') as fr:
                for line in fr:
                    parse = line.strip().split('\t')
                    if float(parse[0]) < 100:
                        break
                    # print(parse)
                    if line_count == 0:
                        algo_speeds[len(algo_speeds)//2]+=2
                    else:
                        if j == 3:
                            index = transform_to_index(parse[9])
                        else:
                            index = int(parse[9])
                        algo_speeds[index] += 1
                    line_count += 1
            trace_speeds.append(algo_speeds)
        all_speeds.append(trace_speeds)

    # To plot
    print(all_speeds)
    current_n_plot = 1
    n_algo = len(ALGOS)
    n_trace = len(selected_name)
    position = (n_algo, n_trace)

    figure = plt.figure(figsize=(10, 5))
    x_ind = range(n_trace)
    y_ind = range(n_algo)
    # plt.xticks(x_ind, x_ind)
    # plt.xticks(y_ind, ALGOS)
    # plt.text(1, 0, "eggs")
    tags = None
    for i in range(len(selected_name)):
        for j in range(len(ALGOS)):
    # for i in range(1):
    #     for j in range(1):
            curr_ax = plt.subplot2grid(position, (j,i))
            current_n_plot += 1
            pos1 = curr_ax.get_position() # get the original position 
            speeds = all_speeds[i][j]
            if j == 0:
                distance = 0
            else:
                distance = 0.6
            total = sum(speeds)
            # props = ['{:.0f}%'.format(round(p/total)) for p in speeds if round(p/total) > 0]
            if j == 3:
                colors = COLORS_7
            elif j == 5:
                colors = COLORS_5
            else:
                colors = COLORS_3
            wedges, text = curr_ax.pie(speeds, colors=colors, radius=1.1)
            if j == 3 and i == 3:
                tags = wedges

            # wedges, text, _ = curr_ax.pie(speeds, \
            #             autopct=lambda p:'{:.0f}%'.format(round(p)) if round(p) > 0 else '',\
            #             radius=1.3, pctdistance=distance)
            # print(wedges)
            if j > 0:
                for k, p in enumerate(wedges):
                    # print(p)
                    gap = p.theta2 - p.theta1
                    if gap == 0:
                        continue
                    if (k==1 and j!=3 and j!=5) or (k==3 and j==3) or (j==5 and k==2):
                        if gap == 360:
                            curr_ax.text(-0.62, -0.2, '{:.0f}'.format(speeds[k]), {'fontweight': 'bold','fontsize': 12})                           
                        else:
                            curr_ax.text(-0.82, 0.09, '{:.0f}'.format(speeds[k]), {'fontweight': 'bold','fontsize': 12})
                    else:
                        ang = gap/2. + p.theta1
                        # if ang > 0 and ang < 30:
                        #     ang = 30.
                        # elif ang < 0 and ang < -30:
                        #     ang = -30.

                        y = np.sin(np.deg2rad(ang))
                        x = np.cos(np.deg2rad(ang))
                        x_position = 1.4*np.sign(x)
                        if j != 5:
                            if j == 3:
                                if k == 6:
                                    xy_ang = -45
                                    y_position = -0.5
                                elif k == 2:
                                    xy_ang = 45
                                    y_position = 0.5
                                else:
                                    xy_ang = ang
                            else:
                                if k == 0:
                                    xy_ang = 45
                                    y_position = 0.5
                                elif k == 1:
                                    xy_ang = ang
                                else:
                                    xy_ang = -45
                                    y_position = -0.5                            

                            bbox_props = dict(boxstyle="square,pad=0.3", fc=colors[k], ec=colors[k])
                            kw = dict(arrowprops=dict(arrowstyle="-"), bbox=bbox_props, zorder=1, va="center")
                            horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
                            connectionstyle = "angle,angleA=0,angleB={}".format(xy_ang)
                            kw["arrowprops"].update({"connectionstyle": connectionstyle})
                            # if y > 0:
                            #     y_position = max(y,0.3)
                            # else:
                            #     y_position = min(-0.2, 3)
                            # y_position = 2*y
                            # print(i, j)
                            # print(x_position)
                            # print(y_position)
                            # if y > 0:
                            #     y_position = max(0.5, y_position)
                            # elif y < 0:
                            #     y_position = min(-0.5, y_position)

                            
                        else:
                            bbox_props = dict(boxstyle="square,pad=0.3", fc=colors[k], ec=colors[k])
                            kw = dict(arrowprops=dict(arrowstyle="-"), bbox=bbox_props, zorder=1, va="center")
                            horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
                            if k == 0:
                                xy_ang = 45
                                y_position = 0.6
                                x_position = 2.8*np.sign(x)
                                connectionstyle = "angle,angleA={},angleB=0".format(xy_ang)
                            elif k == 1:
                                xy_ang = ang
                                y_position = 1.1
                                x_position = 1.1*np.sign(x)
                                connectionstyle = "angle,angleA=-90,angleB={}".format(xy_ang)
                            elif k == 3:
                                xy_ang = ang
                                y_position = -1.1
                                x_position = 1.1*np.sign(x)
                                connectionstyle = "angle,angleA=90,angleB={}".format(xy_ang)
                            elif k == 4:
                                xy_ang = -45
                                y_position = -0.6
                                x_position = 2.5*np.sign(x)

                                connectionstyle = "angle,angleA={},angleB=0".format(xy_ang)
                            
                            kw["arrowprops"].update({"connectionstyle": connectionstyle})

                        curr_ax.annotate('{:.0f}'.format(speeds[k]), fontsize=12, xy=(0.8*x, 0.8*y), xytext=(x_position, y_position),
                                        color='k', fontweight='bold', horizontalalignment=horizontalalignment, **kw)
            else:
                curr_ax.text(-0.82, -0.2, '{:.0f}'.format(300), {'fontweight': 'bold', 'fontsize': 12})  
            if i == 0:
                # plot algo
                if j == 3:
                    figure.text(0.053, pos1.y0-0.08, ALGOS[j] + ' (7)', {'fontsize': 16}, rotation=45)
                elif j == 5:
                    figure.text(0.053, pos1.y0-0.08, ALGOS[j] + ' (5)', {'fontsize': 16}, rotation=45)
                elif j == 0:
                    figure.text(0.028, pos1.y0-0.13, ALGOS[j] + ' (1)', {'fontsize': 16}, rotation=45)
                else:
                    figure.text(0.053, pos1.y0-0.08, ALGOS[j] + ' (3)', {'fontsize': 16}, rotation=45)
            if j == 0:
                # show trace id
                figure.text(pos1.x0+0.03, 0.06, str(i+1), {'fontsize': 18})
            # plt.tight_layout()
    # figure.text(, 0.04, 'common X', ha='center')
    # figure.text(0.04, 0.5, 'common Y', va='center', rotation='vertical')
    figure.legend(handles=tags, labels=['Repeat', '0.75', '0.9', '1', '1.1', '1.25', 'Skip'], ncol=7,\
                loc='upper center', fontsize=15, columnspacing=0.9, handletextpad= 0.5)
    figure.text(0.35, 0.01, 'Network Trace Index' , {'fontsize': 18, 'fontweight': 'bold'})
    # figure.set_tight_layout(True)
    if SHOW:
        figure.show()
        input()

    fig_path = './speeds_pie/'
    if not os.path.isdir(fig_path):
        os.makedirs(fig_path)
    figure.savefig(fig_path + 'speeds_pie.eps', format='eps', dpi=1000, figsize=(10, 5))

def find_initial_latncy(name):
    path = '../dyn_mpc_latency_normal/all_results/latency_Nones/' + name
    with open(path, 'r') as f:
        for line in f:
            pass
        initial_lat = np.round(float(line.strip('\n').split('\t')[0])/1000.0, 1)
    return initial_lat

def show_info(all_info, all_bw_info):
    # Select traces
    benchmark = all_info[1]     # To sort
    file_name_list = [k for k,v in sorted(benchmark.items(), key=lambda x:x[1][0])]
    select_unit_range = int(len(file_name_list)/n_compare)
    latex_str = ''
    selected_name = []
    for i in range(n_compare):
        t_idx = np.random.randint(i*select_unit_range, (i+1)*select_unit_range)
        t_name = file_name_list[t_idx]
        selected_name.append(t_name)
        print(t_name)
        #show info
        print(float(all_bw_info[t_name][0]))
        print('(' + str(np.round(np.mean([float(x) for x in all_bw_info[t_name]]),2)) + ',' + \
                    str(np.round(np.std([float(x) for x in all_bw_info[t_name]]),2)) + ')')
        latex_str += '(' + str(np.round(np.mean([float(x) for x in all_bw_info[t_name]]),1)) + ',' + \
                    str(np.round(np.std([float(x) for x in all_bw_info[t_name]]),1)) + ','

        # Add initial latency
        initial_lat = find_initial_latncy(t_name)
        latex_str += str(initial_lat) + ')\t \\arrvline &' 
        qoes = []
        rates = []
        changes = []
        latencys = []
        for j in range(len(ALGOS)):
            curr_name = ALGOS[j]  
            curr_res = all_info[j][t_name]
            print(curr_res)
            print(curr_name + '\n')
            print(np.round(curr_res[0], 2), ' ',\
                    np.round(curr_res[1], 2),' ',\
                    # np.round(curr_res[4]/1000, 2),' ',\
                    np.round(curr_res[3]/1000, 2),' ',\
                    np.round(curr_res[5]/1000, 2))

            qoes.append(process_num(np.round(curr_res[0], 1)))
            rates.append(process_num(np.round(curr_res[1], 2)))
            # changes.append(np.round(curr_res[4]/1000.0, 2))
            changes.append(process_num(np.round(curr_res[3]/1000.0, 2)))
            latencys.append(process_num(np.round(curr_res[5]/1000.0, 2)))

        # Change float to latex str, and add bold
        m_q = max(qoes)
        m_r = max(rates)
        m_c = min(changes)
        m_l = min(latencys)
        for k in range(len(ALGOS)):
            if qoes[k] == m_q:
                qoes[k] = '\\textbf{'+str(process_num(qoes[k])) + '}'
            else:
                qoes[k] = str(process_num(qoes[k]))

            if rates[k] == m_r:
                rates[k] = '\\textbf{'+str(process_num(rates[k])) + '}'
            else:
                rates[k] = str(process_num(rates[k]))

            if changes[k] == m_c:
                changes[k] = '\\textbf{'+str(process_num(changes[k])) + '}'
            else:
                changes[k] = str(process_num(changes[k]))

            if latencys[k] == m_l:
                latencys[k] = '\\textbf{'+str(process_num(latencys[k])) + '}'
            else:
                latencys[k] = str(process_num(latencys[k]))

        for k in range(len(ALGOS)):
            print(qoes[k])
            print(rates[k])
            print(changes[k])
            print(latencys[k])
            if k < len(ALGOS)-1:
                latex_str += '\\mbox{' + qoes[k] + '}&' + \
                        '\\mbox{' + rates[k] + '}&' + \
                        '\\mbox{' +changes[k] + '}&' + \
                        '\\mbox{' +latencys[k]  + '\\arrvline}&' 
            else:
                latex_str += '\\mbox{' + qoes[k] + '}&' + \
                        '\\mbox{' + rates[k] + '}&' + \
                        '\\mbox{' + changes[k] + '}&' + \
                        '\\mbox{' + latencys[k] + '}\\\\\n'

            # if j < len(ALGOS)-1:
            #     latex_str += str(np.round(curr_res[0], 1)) + '\t&\t' + \
            #         str(np.round(curr_res[1], 2)) + '\t&\t' + \
            #         str(np.round(curr_res[4]/1000.0, 2)) + '\t&\t' + \
            #         str(np.round(curr_res[5]/1000.0, 2)) + '\t&\t'
            # else:
            #     latex_str += str(np.round(curr_res[0], 1)) + '\t&\t' + \
            #         str(np.round(curr_res[1], 2)) + '\t&\t' + \
            #         str(np.round(curr_res[4]/1000.0, 2)) + '\t&\t' + \
            #         str(np.round(curr_res[5]/1000.0, 2)) + '\t\\\\\n'
        if i < n_compare-1:
            latex_str += '\\midrule\n'
        else:
            latex_str += '\\bottomrule\n'
    print("<=============>")
    print("latex table syntax")
    print("<=============>")
    print(latex_str)
    return selected_name

def process_num(num):
    if num >= 10:
        return int(np.round(num))
    elif num >= 1:
        return np.round(num, 1)
    elif num >= 0:
        return np.round(num, 2)
    elif num > -10:
        return np.round(num,1)
    else:
        return int(np.round(num))

def main():
    np.random.seed(6)       # user randon 6 for ACMMM20
    info, all_bw_info = read_data()
    names = show_info(info, all_bw_info)
    show_speeds(info, all_bw_info, names)

if __name__ == '__main__':
    main()