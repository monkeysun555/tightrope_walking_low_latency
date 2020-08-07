import os 
import random
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
    parser.add_argument('-l', '--latency', dest='init_latency', help='set initial latency',
                        default=None, type=int)
    parser.add_argument('-a', '--amplify', dest='bw_amplify', help='amplify bandwidth',
                        default=False, action='store_true')
    parser.add_argument('-r', '--restore', dest='restore', help='restore model',
                        default=False, action='store_true')

    args = parser.parse_args()
    return args
args = parse_args() 

def main():
    initial_latency = args.init_latency
    restore = args.restore
    bw_amplify = args.bw_amplify
    # Load env
    env = Env.Live_Streaming(initial_latency)
    action_dim = env.get_action_info()
    reply_buffer = Reply_Buffer(Config.reply_buffer_size)
    agent = Agent(action_dim)
    reward_logs = []
    loss_logs = []

    logs_path = Config.logs_path + '/'
    if bw_amplify:
        logs_path += 'latency_' + str(initial_latency) + 's_amplified/' 
    else:
        logs_path += 'latency_' + str(initial_latency) + 's'
    if not os.path.exists(logs_path):
         os.makedirs(logs_path) 
    
    starting_episode = 1
    # restore model
    if restore:
        starting_episode = agent.train_restore(logs_path)

    for episode in range(starting_episode, Config.total_episode+1):
        # reset env
        env_end = env.reset(bw_amplify=bw_amplify)
        env.act(0)   # Default
        state = env.get_state()
        total_reward = 0.0

        # Update epsilon
        agent.update_epsilon_by_epoch(episode)
        while not env.streaming_finish():
            if Config.model_version == 0:                
                action = agent.take_action(np.array([state]))
                reward = env.act(action)
                # print(reward)
                state_new = env.get_state()
                total_reward += reward
                action_onehot = np.zeros(action_dim)
                action_onehot[action] = 1
                # print(env.streaming_finish())
                reply_buffer.append((state, action_onehot, reward, state_new, env.streaming_finish()))
                state = state_new                

        # sample batch from reply buffer
        if episode < starting_episode + Config.observe_episode:
            continue

        # update target network
        if episode % Config.update_target_frequency == 0:
            agent.update_target_network()

        if Config.model_version == 0:
            batch_state, batch_actions, batch_reward, batch_state_new, batch_over = reply_buffer.sample()
            loss = agent.update_Q_network_v0(batch_state, batch_actions, batch_reward, batch_state_new, batch_over)
            
        loss_logs.extend([[episode, loss]])
        reward_logs.extend([[episode, total_reward]])

        # save model
        if episode % Config.save_logs_frequency == 0:
            print("episode:", episode)
            agent.save(episode, logs_path)
            # np.save(os.path.join(logs_path, 'loss.npy'), np.array(loss_logs))
            # np.save(os.path.join(logs_path, 'reward.npy'), np.array(reward_logs))

        # print reward and loss
        if episode % Config.show_loss_frequency == 0: 
            print('Episode: {} Reward: {:.3f} Loss: {:.3f}' .format(episode, total_reward, loss[0]))
        agent.update_epsilon_by_epoch(episode)

if __name__ == "__main__":
    main()



