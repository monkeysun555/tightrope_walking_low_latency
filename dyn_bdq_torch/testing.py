import os 
import numpy as np
import argparse
import torch
import env as Env
from config import Config
from reply_buffer import Reply_Buffer
from agent import Agent
from utils import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--massive', dest='massive', help='massive testing',
                        default=False, action='store_true')
    parser.add_argument('-l', '--latency', dest='init_latency', help='initial latency',
                        default=2, type=int, required=False)
    parser.add_argument('-e', '--episode', dest='episode', help='episode of checkpoint',
                        default=50000, type=int, required=False)
    parser.add_argument('-mv', '--model_version', dest='model_version', help='version of model',
                        default=2, type=int, required=False)
    parser.add_argument('-qv', '--q_version', dest='q_version', help='version of q_function',
                        default=2, type=int, required=False)
    parser.add_argument('-tv', '--target_version', dest='target_version', help='version of target',
                        default=2, type=int, required=False)
    parser.add_argument('-lv', '--loss_version', dest='loss_version', help='version of loss',
                        default=0, type=int, required=False)
    parser.add_argument('-r', '--random', dest='random_latency', help='use randon latency',
                        default=False, action='store_true', required=False)
    parser.add_argument('-a', '--amplify', dest='bw_amplify', help='amplify bandwidth',
                        default=False, action='store_true')
    # parser.add_argument('-t', '--train', dest='train', help='train policy or not',
    #                     default=True, type=bool)
    args = parser.parse_args()
    if args.model_version == 2:
        if args.q_version == -1:
            parser.error('Q version -qv is required if using dueling (mv equals to 2)')
    return args

args = parse_args() 

def main():
    massive = args.massive
    episode = args.episode
    model_v = args.model_version
    q_v = args.q_version
    target_v = args.target_version
    loss_v = args.loss_version
    init_latency = args.init_latency
    random_latency = args.random_latency
    bw_amplify = args.bw_amplify

    env = Env.Live_Streaming(init_latency, testing=True, massive=massive, random_latency=random_latency)
    _, action_dims = env.get_action_info()
    # reply_buffer = Reply_Buffer(Config.reply_buffer_size)
    agent = Agent(action_dims, model_v, q_v, target_v, loss_v)
    if model_v == 0 or model_v == 1:
        assert 0 == 1
        model_path = './models/logs_m_' + str(model_v) + '/t_' + str(target_v) + '/l_' + str(loss_v) + '/latency_' + str(init_latency) + 's/model-' + str(episode) + '.pth'
    elif model_v == 2:
        if bw_amplify:
            model_path = './models/logs_m_' + str(model_v) + '/q_' + str(q_v) + '/t_' + str(target_v) + '/l_' + str(loss_v) + '/latency_Nones_amplified/model-' + str(episode) + '.pth'
        else:
            model_path = './models/logs_m_' + str(model_v) + '/q_' + str(q_v) + '/t_' + str(target_v) + '/l_' + str(loss_v) + '/latency_Nones/model-' + str(episode) + '.pth'
    agent.restore(model_path)

    if massive:
        if bw_amplify:
            compare_path = Config.a_cdf_dir
            result_path = Config.a_massive_result_files + '/latency_Nones/'
        else:
            compare_path = Config.cdf_dir
            result_path = Config.massive_result_files + 'model_' + str(model_v) + '/latency_Nones/'
        if not os.path.exists(compare_path):
            os.makedirs(compare_path)
        if not os.path.exists(result_path):
             os.makedirs(result_path) 
        if random_latency:
            compare_file = open(compare_path + 'multi_speed_normal.txt' , 'w')
        else:
            compare_file = open(compare_path + 'multi_speed_' + str(int(init_latency)) +'s.txt' , 'w')
               
        while True:
            # Start testing
            env_end = env.reset(testing=True, bw_amplify=bw_amplify)
            if env_end:
                break
            testing_start_time = env.get_server_time()
            print("Initial latency is: ", testing_start_time)
            tp_trace, time_trace, trace_name, starting_idx = env.get_player_trace_info()
            print("Trace name is: ", trace_name)

            # print(massive, episode, model_v)
            log_path = result_path + trace_name 
            log_file = open(log_path, 'w')
            env.act(0, 3, massive=massive)   # Default
            state = env.get_state()
            total_reward = 0.0
            while not env.streaming_finish():
                if model_v == 0:
                    action = agent.testing_take_action(np.array([state]))
                    action_1 = int(action/action_dims[1])
                    action_2 = action%action_dims[1]
                    reward = env.act(action_1, action_2,log_file, massive=massive)
                    # print(reward)
                    state_new = env.get_state()
                    state = state_new
                    total_reward += reward   
                elif model_v == 1 or model_v == 2:
                    action_1, action_2 = agent.testing_take_action(np.array([state]))
                    # print(action_1, action_2)
                    reward = env.act(action_1, action_2,log_file, massive=massive)
                    state_new = env.get_state()
                    state = state_new
                    total_reward += reward
                    # print(action_1, action_2, reward)
            print('File: ', trace_name, ' reward is: ', total_reward) 
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
        if bw_amplify:
            result_path = Config.a_regular_test_files + 'model_' + str(model_v) + '/latency_' + str(init_latency) + 's/'
        else:
            result_path = Config.regular_test_files + 'model_' + str(model_v) + '/latency_' + str(init_latency) + 's/'
        if not os.path.exists(result_path):
             os.makedirs(result_path) 
        # Start testing
        env_end = env.reset(testing=True)
        testing_start_time = env.get_server_time()
        print("Initial latency is: ", testing_start_time)
        tp_trace, time_trace, trace_name, starting_idx = env.get_player_trace_info()
        print("Trace name is: ", trace_name, starting_idx)
        
        # print(massive, episode, model_v)
        log_path = result_path + trace_name + '.txt'
        log_file = open(log_path, 'w')
        env.act(0, 3, log_file)   # Default
        state = env.get_state()
        total_reward = 0.0
        while not env.streaming_finish():
            if model_v == 2:
                action_1, action_2 = agent.testing_take_action(np.array([state]))
                # action_1 = action//action_dims[1]
                # action_2 = action%action_dims[1]
                # print(action_1.item, action_2.item)
                action_1 = action_1.item()
                action_2 = action_2.item()
                reward = env.act(action_1, action_2, log_file)
                # print(reward)
                state_new = env.get_state()
                state = state_new
                # print(reward)
                total_reward += reward   
                # print(action_1, action_2, reward)
        print('File: ', trace_name, ' reward is: ', total_reward) 
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
