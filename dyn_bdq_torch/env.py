# Define environment
import os
import logging
import numpy as np
import math
from config import Env_Config, Config
from player import *
from server import *
from utils import load_bandwidth, load_single_trace

class Live_Streaming(object):
    def __init__(self, initial_latency, testing=False, massive=False, random_latency=False, random_seed=Config.random_seed):
        np.random.seed(random_seed)
        if testing:
            self.time_traces, self.throughput_traces, self.name_traces = load_bandwidth(testing=True)
            if massive: 
                self.trace_idx = -1     # After first reset, it is 0
                self.a1_batch = []
                self.a2_batch = []
                self.c_batch = []
                self.l_batch = []
                self.f_batch = []
                self.r_batch = []
                self.sc_batch = []
                self.n_forward = 0
                self.n_backward = 0
            else:
                self.trace_idx = Config.trace_idx
        else:
            self.time_traces, self.throughput_traces, self.name_traces = load_bandwidth()
            self.trace_idx = np.random.randint(len(self.throughput_traces))
        # Initial server and player
        self.player = Live_Player(self.throughput_traces[self.trace_idx], self.time_traces[self.trace_idx], self.name_traces[self.trace_idx])
        self.server = Live_Server(initial_latency)
        self.buffer_ub = Env_Config.buffer_ub
        self.freezing_ub = self.player.get_freezing_tol()
        # Initial environment variables
        self.bitrates = Env_Config.bitrate
        self.speeds = Env_Config.speeds
        self.pre_action_1 = Env_Config.default_action_1
        self.pre_action_2 = Env_Config.default_action_2

        # Initial state
        self.state = np.zeros((Env_Config.s_info, Env_Config.s_len))
        self.video_length = 0
        self.ending_flag = 0
        self.random_latency = random_latency

    def act(self, action_1, action_2, log_file=None, massive=False):
        # Initial iteration variables
        action_reward = 0.0
        take_action = 1
        latency = self.server.get_time() - self.player.get_display_time()
        state = self.state
        transformed_action_2 = 0.0      # Will be updated 

        # Reward related variables
        log_bit_rate = 0.0
        pre_log_bit_rate = 0.0
        # Includes taking action and system evolve
        # Do pre processing in previous iteration
        # The first step in this function is to fetch()
        # But has to check if self.take_action
        action_freezing = 0.0
        action_wait = 0.0

        while True:
            # Initial reward
            smooth_p = 0.0
            unnormal_speed_p = 0.0
            speed_smooth_p = 0.0
            missing_p = 0.0
            repeat_p = 0.0

            # Inner iteraion variables
            display_duration = 0.0
            server_wait_time = 0.0
            skip_normal_repeat_flag = 1.0     # O repeat, 1 normal and 2 skip

            if take_action == 1:
                # Choose rate for a segment, get seg boundary info & boundary reward
                # 3rd reward, bitrate fluctuation
                log_bit_rate = np.log(self.bitrates[action_1] / self.bitrates[0])
                pre_log_bit_rate = np.log(self.bitrates[self.pre_action_1] / self.bitrates[0])
                smooth_p = self.get_smooth_penalty(log_bit_rate, pre_log_bit_rate)
                if massive:
                    self.c_batch.append(np.abs(self.bitrates[action_1] - self.bitrates[self.pre_action_1]))
                self.pre_action_1 = action_1

                # 6th reward, display speed fluctuation
                transformed_action_2 = self.translate_to_speed(action_2)        # Actual speed, e.g. 1.0, 1.25
                pre_transformed_action_2 = self.translate_to_speed(self.pre_action_2)            
                speed_smooth_p = self.get_speed_changing_penalty(transformed_action_2, pre_transformed_action_2)
                if massive:
                    self.sc_batch.append(np.abs(transformed_action_2 - pre_transformed_action_2))
                self.pre_action_2 = action_2

                # 7th reward, skip 
                if action_2 == len(self.speeds) - 1:
                    if massive:
                        self.n_forward += 1
                    skip_normal_repeat_flag = 2.0
                    if latency > Env_Config.skip_latency:
                        jump_time, server_buffer_head_time = self.server.skip()
                        self.player.skip_with_time(jump_time, server_buffer_head_time)
                    else:
                        pass
                    missing_p = self.get_skip_penaty()

                # 8th reward, repeat
                elif action_2 == 0:
                    if massive:
                        self.n_backward += 1
                    skip_normal_repeat_flag = 0.0
                    self.player.repeat()
                    # 8: QoE metric about repeat penalty
                    repeat_p = self.get_repeat_penalty()

            # Get next chunk from server
            self.server.generate_next_delivery()
            download_chunk_info = self.server.get_next_delivery()
            download_seg_idx = download_chunk_info[0]
            download_chunk_idx = download_chunk_info[1]
            download_chunk_end_idx = download_chunk_info[2]
            download_chunk_size = download_chunk_info[3]
            chunk_number = download_chunk_end_idx - download_chunk_idx + 1
            assert chunk_number == 1

            # Download chunk
            real_chunk_size, download_duration, freezing, time_out, player_state, rtt = self.player.fetch(action_1, 
                            download_chunk_size, download_seg_idx, download_chunk_idx, 
                            take_action, chunk_number, transformed_action_2)
            if not take_action:
                assert smooth_p == 0.0
                assert speed_smooth_p == 0.0
                assert missing_p == 0.0
                assert repeat_p == 0.0
            take_action = 0
            display_duration += (download_duration - freezing)
            # For debugging
            # print("Buffer length:", player.get_buffer())
            # print("Download time:", download_duration)
            # print("Freezing time:", freezing)
            # print("# chunks:", chunk_number)
            ##################
            server_time = self.server.update(download_duration)
            if not time_out:
                self.server.clean_next_delivery()
            else:
                assert self.player.get_state() == 0
                assert np.round(self.player.get_buffer(), 3) == 0.0
                index_gap = self.server.timeout_encoding_buffer()
                self.player.playing_time_back(index_gap)

            # chech whether need to wait, using number of available segs
            if self.server.check_chunks_empty():
                server_wait_time = self.server.wait()            # in ms
                assert server_wait_time > 0.0
                assert server_wait_time < Env_Config.chunk_duration
                wait_freezing = self.player.wait(server_wait_time, transformed_action_2)
                freezing += wait_freezing
                display_duration += (server_wait_time - wait_freezing)

            # Get state info after downloading & wait
            buffer_length = self.player.get_buffer()
            latency = self.server.get_time() - self.player.get_display_time()     # Affected by server wait if display speed changes
            player_state = self.player.get_state()

            # Calculate reward for each chunk
            # 1st reward, log_bit_rate
            quality_r = self.get_quality_reward(log_bit_rate, chunk_number)

            # 2nd reward, freezing
            rebuff_p = self.get_freeze_penalty(freezing/Env_Config.ms_in_s)

            # 4th reward, latency
            # delay_p = self.get_latency_penalty(latency/Env_Config.ms_in_s, chunk_number)
            delay_p = self.get_latency_penalty_new(latency/Env_Config.ms_in_s, chunk_number)
            
            # 5th reward, display speed
            # unnormal_speed_p = self.get_unnormal_speed_penalty(transformed_action_2, display_duration/Env_Config.ms_in_s)
            unnormal_speed_p = self.get_unnormal_speed_penalty(transformed_action_2, 0.2)
            # Sum of all metrics
            action_reward += quality_r - rebuff_p - smooth_p - delay_p - unnormal_speed_p - speed_smooth_p - missing_p - repeat_p

            # Update state
            # print(state)
            state = np.roll(state, -1, axis=1)
            state[0, -1] = real_chunk_size                                  # chunk delivered, in Kbits                    
            state[1, -1] = download_duration - rtt                          # download duration, in ms
            state[2, -1] = buffer_length                                    # buffer length, in ms
            state[3, -1] = server_wait_time                                 # waiting time, in ms
            state[4, -1] = freezing                                         # freezing time, in ms
            state[5, -1] = log_bit_rate                                     # log bitrate
            state[6, -1] = latency                                          # latency, in ms
            state[7, -1] = skip_normal_repeat_flag                          # flag of skip or repeat, 0, 1 or 2
            state[8, -1] = player_state                                     # player state, 0, 1, or 2
            state[9, -1] = transformed_action_2                             # playing speed, 0.75 to 1.25
            state[10, -1] = transformed_action_2                             # playing speed, 0.75 to 1.25
            
            state = self.normal(state)
            # print(state)
            action_freezing += freezing
            action_wait += server_wait_time
            # Check whether a segment is finished
            if self.server.check_take_action():
                if massive:
                    self.a1_batch.append(self.bitrates[action_1])
                    self.a2_batch.append(transformed_action_2)
                    self.f_batch.append(action_freezing)
                    self.l_batch.append(latency)
                    self.r_batch.append(action_reward)
                self.state = state
                self.video_length += 1
                if self.video_length >= Env_Config.video_terminal_length:
                    # A sequence is terminated, to reset
                    self.ending_flag = 1
                    # self.reset()          # Do reset in main.py
                if log_file:
                    log_file.write( str(self.server.get_time()) + '\t' +
                            str(self.bitrates[action_1]) + '\t' +
                            str(self.player.get_buffer()) + '\t' +
                            str(action_freezing) + '\t' +
                            str(time_out) + '\t' +
                            str(action_wait) + '\t' +
                            str(latency) + '\t' +
                            str(self.player.get_state()) + '\t' +
                            str(int(action_1/len(self.bitrates))) + '\t' +
                            str(action_2) + '\t' + 
                            str(action_reward) + '\n') 
                    log_file.flush()
                return action_reward

    def normal(self, state):
        state[0, -1] = state[0, -1]/(2.0*self.bitrates[-1]/Env_Config.chunk_in_seg)
        state[1, -1] = state[1, -1]/Env_Config.ms_in_s
        state[2, -1] = state[2, -1]/self.buffer_ub
        state[3, -1] = state[3, -1]/Env_Config.chunk_duration
        state[4, -1] = state[4, -1]/self.freezing_ub
        state[5, -1] = state[5, -1]/(np.log(self.bitrates[-1]/self.bitrates[0]))
        state[6, -1] = state[6, -1]/self.buffer_ub
        state[7, -1] = state[7, -1]/2.0
        state[8, -1] = state[8, -1]/2.0
        state[9, -1] = (state[9, -1]-self.speeds[1])/0.5
        state[10, -1] = np.abs(state[10, -1]-1.0)/0.25
        return state

    def get_server_time(self):
        return self.server.get_time()

    def get_player_trace_info(self):
        return self.player.get_tp_trace(), self.player.get_time_trace(), self.player.get_trace_name(), self.player.get_time_idx()

    def get_action_info(self):
        return Env_Config.a_num, Env_Config.a_dims

    def get_state(self):
        return self.state

    def streaming_finish(self):
        return self.ending_flag

    def reset(self, testing=False, bw_amplify=False):
        if testing:
            self.state = np.zeros((Env_Config.s_info, Env_Config.s_len))
            self.trace_idx += 1
            if self.trace_idx == len(self.throughput_traces):
                return 1
            self.player.reset(self.throughput_traces[self.trace_idx], \
                            self.time_traces[self.trace_idx], self.name_traces[self.trace_idx], \
                            testing=True, bw_amplify=bw_amplify)
            self.server.reset(testing=testing, random_latency=self.random_latency)
            self.ending_flag = 0
            self.video_length = 0
            self.a1_batch = []
            self.a2_batch = []
            self.c_batch = []
            self.l_batch = []
            self.f_batch = []
            self.r_batch = []
            self.sc_batch = []
            self.n_forward = 0
            self.n_backward = 0
            return 0
        else:
            self.state = np.zeros((Env_Config.s_info, Env_Config.s_len))
            self.trace_idx = np.random.randint(len(self.throughput_traces))
            self.player.reset(self.throughput_traces[self.trace_idx], \
                            self.time_traces[self.trace_idx], self.name_traces[self.trace_idx], \
                            testing=False, bw_amplify=bw_amplify)
            self.server.reset(random_latency=self.random_latency)
            self.ending_flag = 0
            self.video_length = 0
            return 0

    def massive_save(self, cooked_name, cdf_path=None):
        cdf_path.write(cooked_name + '\t')
        cdf_path.write(str(np.sum(self.r_batch)) + '\t')
        cdf_path.write(str(np.mean(self.a1_batch)) + '\t')
        cdf_path.write(str(np.mean(self.a2_batch)) + '\t')
        cdf_path.write(str(np.sum(self.f_batch)) + '\t')
        cdf_path.write(str(np.mean(self.c_batch)) + '\t')
        cdf_path.write(str(np.mean(self.l_batch)) + '\t')
        cdf_path.write(str(np.mean(self.sc_batch)) + '\t')

        cdf_path.write(str(np.var(self.a2_batch)) + '\t')
        cdf_path.write(str(self.n_forward) + '\t')
        cdf_path.write(str(self.n_backward) + '\t')
        cdf_path.write('\n')
        
    def translate_to_speed(self, action_2_index):
        # Translate to real speed
        if action_2_index == 0 or action_2_index == len(self.speeds) - 1:
            return 1.0
        return self.speeds[action_2_index]

    # 1st reward, log_bit_rate
    def get_quality_reward(self, log_bit_rate, chunk_number):
        return Env_Config.action_reward * log_bit_rate * chunk_number

    # 2nd reward, freezing
    def get_freeze_penalty(self, freezing):
        return Env_Config.rebuf_penalty * freezing

    # 3rd reward, bitrate fluctuation
    def get_smooth_penalty(self, log_bit_rate, pre_log_bit_rate):
        return Env_Config.smooth_penalty * np.abs(log_bit_rate - pre_log_bit_rate)

    # 4th reward, latency
    def get_latency_penalty(self, latency, chunk_number):
        return Env_Config.long_delay_penalty * (1.0/(1+math.exp(Env_Config.const-Env_Config.x_ratio*latency)) - 1.0/(1+math.exp(Env_Config.const))) * chunk_number

    def get_latency_penalty_new(self, latency, chunk_number):
        return Env_Config.long_delay_penalty_new * latency * chunk_number

    # 5th reward, display speed
    def get_unnormal_speed_penalty(self, speed, display_duration):
        speed_gap = np.abs(speed - 1.0)
        if speed_gap >= 0.1:
            return Env_Config.unnormal_playing_penalty * speed_gap * display_duration 
        return 0.0

    # 6th reward, display speed fluctuation
    def get_speed_changing_penalty(self, transformed_action_2, pre_transformed_action_2):
        # x is the index of speed
        # if the gap between two speed is greater than 0.1, then there is penalty
        return Env_Config.speed_smooth_penalty * np.abs(transformed_action_2 - pre_transformed_action_2)

    # 7th reward, skip 
    def get_skip_penaty(self):
        return Env_Config.skip_seg_penalty*Env_Config.skip_segs

    # 8th reward, repeat
    def get_repeat_penalty(self):
        return Env_Config.repeat_seg_penalty*Env_Config.repeat_segs

