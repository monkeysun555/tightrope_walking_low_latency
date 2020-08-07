import os 
import numpy as np
import argparse
import env as Env
from config import Config
from utils import *
# import new_iLQR as iLQR
import new_iLQR_used as iLQR

# RLS

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--massive', dest='massive', help='massive testing',
                        default=False, action='store_true', required=False)
    parser.add_argument('-r', '--random', dest='random_latency', help='use randon latency',
                        default=False, action='store_true', required=False)
    parser.add_argument('-l', '--latency', dest='init_latency', help='initial latency',
                        default=2, type=int, required=False)
    parser.add_argument('-a', '--amplify', dest='bw_amplify', help='amplify bandwidth',
                        default=False, action='store_true')
    # parser.add_argument('-t', '--train', dest='train', help='train policy or not',
    #                     default=True, type=bool)
    args = parser.parse_args()
    return args
args = parse_args() 

def main():
    massive = args.massive
    init_latency = args.init_latency
    random_latency = args.random_latency
    bw_amplify = args.bw_amplify

    env = Env.Live_Streaming(init_latency, testing=True, massive=massive, random_latency=random_latency)
    _, action_dims = env.get_action_info()

    if massive:
        if bw_amplify:
            compare_path = Config.a_cdf_dir
            result_path = Config.a_massive_result_files + '/latency_Nones/'
        else:
            compare_path = Config.cdf_dir  
            result_path = Config.massive_result_files + '/latency_Nones/'
        if not os.path.exists(compare_path):
            os.makedirs(compare_path) 
        if not os.path.exists(result_path):
             os.makedirs(result_path) 
        if random_latency:
            compare_file = open(compare_path + 'dyn_mpc_opt_enhanced.txt' , 'w')
        else:
            compare_file = open(compare_path + 'dyn_mpc_opt' + str(int(init_latency)) +'s.txt' , 'w')
 
        iLQR_solver = iLQR.iLQR_solver()
        iLQR_solver.set_step() 
        while True:
            # Start testing
            env_end = env.reset(testing=True, bw_amplify=bw_amplify)
            iLQR_solver.reset()
            if env_end:
                break
            testing_start_time = env.get_server_time()
            print("Initial latency is: ", testing_start_time)
            tp_trace, time_trace, trace_name, starting_idx = env.get_player_trace_info()
            print("Trace name is: ", trace_name)
            
            # print(massive, episode, model_v)
            log_path = result_path + trace_name 
            log_file = open(log_path, 'w')
            ave_bw, reward = env.act(0, 2, massive=massive)   # Default
            # print("ave bandwidth", ave_bw)
            iLQR_solver.update_bw_record(ave_bw)
            total_reward = 0.0
            while not env.streaming_finish():
                if env.get_player_state() == 0:
                    action_1 = 0
                    action_2 = 1
                else:
                    tmp_buffer = env.get_buffer_length()
                    tmp_latency = env.get_latency()
                    tmp_pre_a_1, tmp_pre_a_2 = env.get_pre_actions()
                    iLQR_solver.set_bu(tmp_latency)
                    future_opt_trace = env.get_future_bw(iLQR_solver.get_step())
                    iLQR_solver.set_future_bw_rtt(future_opt_trace)
                    # print(np.round(tmp_buffer/Env_Config.ms_in_s, 2), np.round(tmp_latency/Env_Config.ms_in_s, 2), tmp_pre_a_1, tmp_pre_a_2)
                    iLQR_solver.set_x0(tmp_buffer, tmp_latency, tmp_pre_a_1, tmp_pre_a_2)
                    iLQR_solver.generate_initial_x()
                    action_1, action_2 = iLQR_solver.iterate_LQR()
                    if iLQR_solver.checking():
                        bit_rate = iLQR_solver.nan_index()
                        action_2 = env.get_pre_actions()[1]
                        print("got 1") 
                ave_bw, reward = env.act(action_1, action_2, log_file, massive=massive)
                iLQR_solver.update_bw_record(ave_bw)
                # print(reward)
                state_new = env.get_state()
                state = state_new
                total_reward += reward   
                # print(action_1, action_2, reward)
            print('File: ', trace_name, ' reward is: ', total_reward, ' init latency: ', testing_start_time) 
            # Get initial latency of player and how long time is used. and tp/time trace
            testing_duration = env.get_server_time() - testing_start_time
            tp_record, time_record = get_tp_time_trace_info(tp_trace, time_trace, starting_idx, testing_duration + env.player.get_buffer())
            log_file.write('\t'.join(str(tp) for tp in tp_record))
            log_file.write('\n')
            log_file.write('\t'.join(str(time) for time in time_record))
            # log_file.write('\n' + str(IF_NEW))
            log_file.write('\n' + str(testing_start_time))
            log_file.write('\n')
            log_file.close()
            env.massive_save(trace_name, compare_file)
        compare_file.close()
    else:
        assert 1 == 0
        # check results log path
        if bw_amplify:
            result_path = Config.a_regular_test_files + '/latency_' + str(init_latency) + 's/'
        else:
            result_path = Config.regular_test_files + '/latency_' + str(init_latency) + 's/'
        if not os.path.exists(result_path):
             os.makedirs(result_path) 
        # Start testing
        env_end = env.reset(testing=True, bw_amplify=bw_amplify)
        testing_start_time = env.get_server_time()
        print("Initial latency is: ", testing_start_time)
        tp_trace, time_trace, trace_name, starting_idx = env.get_player_trace_info()
        print("Trace name is: ", trace_name, starting_idx)
        iLQR_solver = iLQR.iLQR_solver()
        iLQR_solver.set_step()  
        iLQR_solver.reset()      
        # print(massive, episode, model_v)
        log_path = result_path + trace_name + '.txt'
        log_file = open(log_path, 'w')
        ave_bw, reward = env.act(0, 2, log_file)   # Default
        print("ave bandwidth", ave_bw)
        iLQR_solver.update_bw_record(ave_bw)
        state = env.get_state()
        total_reward = 0.0
        while not env.streaming_finish():
            if env.get_player_state() == 0:
                print("!!!!!!")
                action_1 = 0
                action_2 = 1
            else:
                tmp_buffer = env.get_buffer_length()
                tmp_latency = env.get_latency()
                tmp_pre_a_1, tmp_pre_a_2 = env.get_pre_actions()
                iLQR_solver.set_bu(tmp_latency)
                future_opt_trace = env.get_future_bw(iLQR_solver.get_step())
                iLQR_solver.set_future_bw_rtt(future_opt_trace)
                # print(np.round(tmp_buffer/Env_Config.ms_in_s, 2), np.round(tmp_latency/Env_Config.ms_in_s, 2), tmp_pre_a_1, tmp_pre_a_2)
                iLQR_solver.set_x0(tmp_buffer, tmp_latency, tmp_pre_a_1, tmp_pre_a_2)
                iLQR_solver.generate_initial_x()
                action_1, action_2 = iLQR_solver.iterate_LQR()
                if iLQR_solver.checking():
                    bit_rate = iLQR_solver.nan_index()
                    action_2 = env.get_pre_actions()[1]
                    print("got 1") 
            ave_bw, reward = env.act(action_1, action_2, log_file, massive=massive)
            print("download ave bw:" , ave_bw)
            iLQR_solver.update_bw_record(ave_bw)
            # print(reward)
            state_new = env.get_state()
            state = state_new
            # print(reward)
            total_reward += reward   
            # print(action_1, action_2, reward)
        print('File: ', trace_name, ' reward is: ', total_reward, ' init latency: ', testing_start_time) 
        # Get initial latency of player and how long time is used. and tp/time trace
        testing_duration = env.get_server_time() - testing_start_time
        tp_record, time_record = get_tp_time_trace_info(tp_trace, time_trace, starting_idx, testing_duration + env.player.get_buffer())
        log_file.write('\t'.join(str(tp) for tp in tp_record))
        log_file.write('\n')
        log_file.write('\t'.join(str(time) for time in time_record))
        # log_file.write('\n' + str(IF_NEW))
        log_file.write('\n' + str(testing_start_time))
        log_file.write('\n')
        log_file.close()

if __name__ == '__main__':
    main()
