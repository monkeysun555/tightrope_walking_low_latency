import os
import logging
import numpy as np
import multiprocessing as mp
os.environ['CUDA_VISIBLE_DEVICES']=''
import tensorflow as tf
import live_player
import live_server
import static_a3c_chunk as a3c
import load
import math

A_DIM = 6
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001

TEST_DURATION = 300             # Number of testing <===================== Change length here
BITRATE = [300.0, 500.0, 1000.0, 2000.0, 3000.0, 6000.0]

RANDOM_SEED = 10
RAND_RANGE = 1000
BITRATE_LOW_NOISE = 0.95
BITRATE_HIGH_NOISE = 1.05

MS_IN_S = 1000.0
KB_IN_MB = 1000.0   # in ms

SEG_DURATION = 1000.0
CHUNK_DURATION = 200.0
CHUNK_SEG_RATIO = CHUNK_DURATION/SEG_DURATION
CHUNK_IN_SEG = SEG_DURATION/CHUNK_DURATION

USER_START_UP_TH = 2000.0
USER_FREEZING_TOL = 3000.0                                  # Single time freezing time upper bound

DEFAULT_ACTION = 0          # lowest bitrate
ACTION_REWARD = 1.0 * CHUNK_SEG_RATIO   
REBUF_PENALTY = 6.0     # for second
SMOOTH_PENALTY = 1.0
MISSING_PENALTY = 6.0 * CHUNK_SEG_RATIO # not included
LONG_DELAY_PENALTY_NEW = 0.5 * CHUNK_SEG_RATIO  # For linear
# LONG_DELAY_PENALTY = 4.0 * CHUNK_SEG_RATIO  # For sigmoid
CONST = 6.0
X_RATIO = 1.0

RATIO_LOW_2 = 2.0               # This is the lowest ratio between first chunk and the sum of all others
RATIO_HIGH_2 = 10.0         # This is the highest ratio between first chunk and the sum of all others
RATIO_LOW_5 = 0.75              # This is the lowest ratio between first chunk and the sum of all others
RATIO_HIGH_5 = 1.0          # This is the highest ratio between first chunk and the sum of all others

NOR_BW = 10.0
NOR_CHUNK_SIZE = BITRATE[-1] / CHUNK_IN_SEG
NOR_BUFFER = 20000.0 / MS_IN_S
NOR_CHUNK = CHUNK_IN_SEG
NOR_FREEZING = USER_FREEZING_TOL / MS_IN_S
NOR_RATE = np.log(BITRATE[-1]/BITRATE[0])
NOR_WAIT = CHUNK_DURATION / MS_IN_S
NOR_STATE = 2.0 # 0, 1, 2

S_INFO = 8  
S_LEN = 15 
END_EPOCH =  90000              # <========================= CHANGE MODELS, 105000 is the best right now, for no sync mode, bw_traces
NN_MODEL = './models/nn_model_s_ep_' + str(END_EPOCH) + '.ckpt'

LOG_FILE_DIR = './results'
LOG_FILE = LOG_FILE_DIR + '/pensieve/'

COMPARE_PATH = '../compare/compare_results/'
COMPARE_FILE = COMPARE_PATH + 'pensieve.txt'
    
TEST_DATA_DIR = '../bw_traces_test/cooked_test_traces/'

def main():
    np.random.seed(RANDOM_SEED)
    if not os.path.exists(LOG_FILE):
        os.makedirs(LOG_FILE)
    if not os.path.exists(COMPARE_PATH):
        os.makedirs(COMPARE_PATH)
    all_testing_log = open(COMPARE_FILE, 'w')

    cooked_times, cooked_bws, cooked_names = load.loadBandwidth(TEST_DATA_DIR)

    with tf.Session() as sess:

        actor = a3c.ActorNetwork(sess, state_dim=[S_INFO, S_LEN], action_dim=A_DIM,
                                 learning_rate=ACTOR_LR_RATE)
        critic = a3c.CriticNetwork(sess, state_dim=[S_INFO, S_LEN],
                                   learning_rate=CRITIC_LR_RATE)

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()  # save neural net parameters

        # restore neural net parameters
        if NN_MODEL:  
            # NN_MODEL is the path to file
            print(NN_MODEL)
            saver.restore(sess, NN_MODEL)
            print("Testing model restored.")
        else: assert 0 == 1

        player = live_player.Live_Player(time_traces=cooked_times, throughput_traces=cooked_bws, name_traces = cooked_names,
                                            seg_duration=SEG_DURATION, chunk_duration=CHUNK_DURATION,
                                            start_up_th=USER_START_UP_TH, freezing_tol=USER_FREEZING_TOL,
                                            testing=True, randomSeed=RANDOM_SEED)
        server = live_server.Live_Server(seg_duration=SEG_DURATION, chunk_duration=CHUNK_DURATION)

        for i in range(len(cooked_times)):
            player.reset(testing=True)
            server.reset(testing=True)
            starting_time = server.get_time()
            tp_trace, time_trace, trace_name, starting_time_idx = player.get_player_trace_info()

            print("Initial latency is: ", starting_time)
            print("Trace name is: ", trace_name)
            log_path = LOG_FILE + trace_name
            log_file = open(log_path, 'w')

            action_num = DEFAULT_ACTION # 0
            last_bit_rate = action_num
            bit_rate = action_num

            state = np.zeros((S_INFO, S_LEN))
            r_batch = []
            f_batch = []
            a_batch = []
            c_batch = []
            l_batch = []
            action_reward = 0.0     # Total reward is for all chunks within on segment
            action_freezing = 0.0
            action_wait = 0.0
            take_action = 1
            latency = 0.0

            for i in range(TEST_DURATION):
                # print "Current index: ", i
                # print server.get_time()
                action_reward = 0.0 
                take_action = 1

                while True:  # serve video forever
                    download_chunk_info = server.get_next_delivery()
                    download_seg_idx = download_chunk_info[0]
                    download_chunk_idx = download_chunk_info[1]
                    download_chunk_end_idx = download_chunk_info[2]
                    download_chunk_size = download_chunk_info[3]
                    chunk_number = download_chunk_end_idx - download_chunk_idx+1
                    assert chunk_number == 1
                    # if download_seg_idx >= TEST_DURATION:
                    #     break
                    server_wait_time = 0.0  
                    sync = 0
                    missing_count = 0
                    real_chunk_size, download_duration, freezing, time_out, player_state, rtt = player.fetch(bit_rate, download_chunk_size, 
                                                                            download_seg_idx, download_chunk_idx, take_action, chunk_number)
                    take_action = 0
                    buffer_length = player.get_buffer()
                    server_time = server.update(download_duration)
                    action_freezing += freezing
                    if time_out:
                        assert player.get_state() == 0
                        assert np.round(player.get_buffer_length(), 3) == 0.0
                        # Pay attention here, how time out influence next reward, the smoothness
                        # Bit_rate will recalculated later, this is for reward calculation
                        bit_rate = 0
                        sync = 1
                    else:
                        server.clean_next_delivery()

                    # chech whether need to wait, using number of available segs
                    if server.check_chunks_empty():
                        server_wait_time = server.wait()
                        action_wait += server_wait_time
                        assert server_wait_time > 0.0
                        assert server_wait_time < CHUNK_DURATION
                        # print "Before wait, player: ", player.get_playing_time(), player.get_real_time()
                        player.wait(server_wait_time)
                        # print "After wait, player: ", player.get_playing_time(), player.get_real_time()
                        buffer_length = player.get_buffer()

                    # Calculate reward
                    latency = server.get_time() - player.get_display_time()
                    player_state = player.get_state()

                    log_bit_rate = np.log(BITRATE[bit_rate] / BITRATE[0])
                    log_last_bit_rate = np.log(BITRATE[last_bit_rate] / BITRATE[0])
                    last_bit_rate = bit_rate
                
                    reward = ACTION_REWARD * log_bit_rate * chunk_number \
                            - REBUF_PENALTY * freezing / MS_IN_S \
                            - SMOOTH_PENALTY * np.abs(log_bit_rate - log_last_bit_rate) \
                            - LONG_DELAY_PENALTY_NEW * lat_penalty_new(latency/MS_IN_S) * chunk_number \
                            - MISSING_PENALTY * missing_count
                            # - LONG_DELAY_PENALTY * lat_penalty(latency/MS_IN_S) * chunk_number \

                    action_reward += reward                    
                    server.generate_next_delivery()

                    # For state
                    state = np.roll(state, -1, axis=1)
                    
                    state[0, -1] = real_chunk_size / NOR_CHUNK_SIZE             # chunk size
                    state[1, -1] = (download_duration - rtt) / MS_IN_S          # downloading time
                    state[2, -1] = buffer_length / MS_IN_S / NOR_BUFFER         # buffer length
                    state[3, -1] = server_wait_time / MS_IN_S/ NOR_WAIT         # time of waiting for server
                    state[4, -1] = freezing / MS_IN_S / NOR_FREEZING            # current freezing time
                    state[5, -1] = log_bit_rate / NOR_RATE                      # video bitrate
                    state[6, -1] = latency / NOR_BUFFER                                  # whether there is resync
                    state[7, -1] = player_state / NOR_STATE

                    next_chunk_idx = server.get_next_delivery()[1]
                    if next_chunk_idx == 0 or sync:
                        take_action = 1
                        # print(action_reward)
                        r_batch.append(action_reward)
                        f_batch.append(action_freezing)
                        a_batch.append(BITRATE[bit_rate])
                        l_batch.append(latency)
                        action_prob = actor.predict(np.reshape(state, (1, S_INFO, S_LEN)))
                        
                        action_num = action_prob.argmax()
                        bit_rate = action_num
                        c_batch.append(np.abs(BITRATE[bit_rate] - BITRATE[last_bit_rate]))

                        log_file.write(str(server.time) + '\t' +
                                        str(BITRATE[last_bit_rate]) + '\t' +
                                        str(buffer_length) + '\t' +
                                        str(action_freezing) + '\t' +
                                        str(time_out) + '\t' +
                                        str(action_wait) + '\t' +
                                        str(latency) + '\t' +
                                        str(player.get_state()) + '\t' +
                                        str(int(action_num/len(BITRATE))) + '\t' +                          
                                        str(action_reward) + '\n')
                        log_file.flush()
                        action_reward = 0.0
                        action_freezing = 0.0
                        action_wait = 0.0
                        break
            
            time_duration = server.get_time() - starting_time

            # tp_record, time_record = record_tp(player.get_throughput_trace(), player.get_time_trace(), starting_time_idx, time_duration + buffer_length) 
            
            # print(starting_time_idx, cooked_name, len(player.get_throughput_trace()), player.get_time_idx(), len(tp_record), np.sum(r_batch))
            # log_file.write('\t'.join(str(tp) for tp in tp_record))
            # log_file.write('\n')

            # log_file.write('\t'.join(str(time) for time in time_record))
            # # log_file.write('\n' + str(IF_NEW))
            # log_file.write('\n' + str(starting_time))
            # log_file.write('\n')
            log_file.close()
        
            all_testing_log.write(trace_name + '\t')
            all_testing_log.write(str(np.sum(r_batch)) + '\t')
            all_testing_log.write(str(np.mean(a_batch)) + '\t')
            all_testing_log.write(str(np.sum(f_batch)) + '\t')
            all_testing_log.write(str(np.mean(c_batch)) + '\t')
            all_testing_log.write(str(np.mean(l_batch)) + '\t')
            print(np.sum(r_batch))

            all_testing_log.write('\n')
        all_testing_log.close()

# def main():

#   np.random.seed(RANDOM_SEED)
#   if not os.path.isdir(LOG_FILE_DIR):
#       os.makedirs(LOG_FILE_DIR)

#   assert len(BITRATE) == A_DIM

#   if not IF_NEW:
#       cooked_time, cooked_bw = load.load_single_trace(DATA_DIR + TRACE_NAME)      # For bw_traces
#   else:
#       cooked_time, cooked_bw = load.new_load_single_trace(DATA_DIR + TRACE_NAME)  # For new_traces

#   player = live_player.Live_Player(time_trace=cooked_time, throughput_trace=cooked_bw, 
#                                       seg_duration=SEG_DURATION, chunk_duration=CHUNK_DURATION,
#                                       start_up_th=USER_START_UP_TH, freezing_tol=USER_FREEZING_TOL, latency_tol = USER_LATENCY_TOL,
#                                       randomSeed=RANDOM_SEED)
#   server = live_server.Live_Server(seg_duration=SEG_DURATION, chunk_duration=CHUNK_DURATION, 
#                                       start_up_th=SERVER_START_UP_TH, randomSeed=RANDOM_SEED)
#   print(server.get_time())

#   log_path = LOG_FILE + '_' + TRACE_NAME
#   log_file = open(log_path, 'w')

#   with tf.Session() as sess:
#       actor = a3c.ActorNetwork(sess,
#                                state_dim=[S_INFO, S_LEN], action_dim=A_DIM,
#                                learning_rate=ACTOR_LR_RATE)
#       critic = a3c.CriticNetwork(sess,
#                                  state_dim=[S_INFO, S_LEN],
#                                  learning_rate=CRITIC_LR_RATE)

#       sess.run(tf.global_variables_initializer())
#       saver = tf.train.Saver()  # save neural net parameters

#       # restore neural net parameters
#       if NN_MODEL:  
#           # NN_MODEL is the path to file
#           print(NN_MODEL)
#           saver.restore(sess, NN_MODEL)
#           print("Testing model restored.")
#       else: assert 0 == 1

#       action_num = DEFAULT_ACTION # 0
#       last_bit_rate = action_num
#       bit_rate = action_num
#       state = np.zeros((S_INFO, S_LEN))
#       r_batch = []
#       action_reward = 0.0     # Total reward is for all chunks within on segment
#       action_freezing = 0.0
#       action_wait = 0.0
#       take_action = 1
#       latency = 0.0
#       starting_time = server.get_time()
#       starting_time_idx = player.get_time_idx()
#       for i in range(TEST_DURATION):
#           action_reward = 0.0 
#           take_action = 1

#           while True:  # serve video forever
#               download_chunk_info = server.get_next_delivery()
#               # print "chunk info is " + str(download_chunk_info)
#               download_seg_idx = download_chunk_info[0]
#               download_chunk_idx = download_chunk_info[1]
#               download_chunk_end_idx = download_chunk_info[2]
#               download_chunk_size = download_chunk_info[3][bit_rate]      # Might be several chunks
#               chunk_number = download_chunk_end_idx - download_chunk_idx + 1
#               assert chunk_number == 1
#               if download_seg_idx >= TEST_DURATION:
#                   break
#               server_wait_time = 0.0  
#               sync = 0
#               missing_count = 0
#               real_chunk_size, download_duration, freezing, time_out, player_state, rtt = player.fetch(download_chunk_size, 
#                                                                       download_seg_idx, download_chunk_idx, take_action, chunk_number)
#               take_action = 0
#               buffer_length = player.get_buffer_length()
#               server_time = server.update(download_duration)
#               action_freezing += freezing
#               if not time_out:
#                   # server.chunks.pop(0)
#                   server.clean_next_delivery()
#                   sync = player.check_resync(server_time)
#               else:
#                   assert player.get_state() == 0
#                   assert np.round(player.get_buffer_length(), 3) == 0.0
#                   # Pay attention here, how time out influence next reward, the smoothness
#                   # Bit_rate will recalculated later, this is for reward calculation
#                   bit_rate = 0
#                   sync = 1
#               # Disable sync for current situation
#               if sync:
#                   # if not IF_NEW:
#                   #   print "Should not happen"
#                   #   assert 0 == 1
#                   # To sync player, enter start up phase, buffer becomes zero
#                   sync_time, missing_count = server.sync_encoding_buffer()
#                   player.sync_playing(sync_time)
#                   buffer_length = player.get_buffer_length()

#               latency = server.get_time() - player.get_playing_time()
#               # print "latency is: ", latency/MS_IN_S
#               player_state = player.get_state()

#               log_bit_rate = np.log(BITRATE[bit_rate] / BITRATE[0])
#               log_last_bit_rate = np.log(BITRATE[last_bit_rate] / BITRATE[0])
#               last_bit_rate = bit_rate
#               # print(log_bit_rate, log_last_bit_rate)
#               # if not IF_NEW:
#               #   reward = ACTION_REWARD * log_bit_rate * chunk_number \
#               #           - REBUF_PENALTY * freezing / MS_IN_S \
#               #           - SMOOTH_PENALTY * np.abs(log_bit_rate - log_last_bit_rate) \
#               #           - LONG_DELAY_PENALTY*(LONG_DELAY_PENALTY_BASE**(ReLU(latency-TARGET_LATENCY)/ MS_IN_S)-1) * chunk_number
#                       # - UNNORMAL_PLAYING_PENALTY*(playing_speed-NORMAL_PLAYING)*download_duration/MS_IN_S
#                       # - MISSING_PENALTY * missing_count
#               reward = ACTION_REWARD * log_bit_rate * chunk_number \
#                       - REBUF_PENALTY * freezing / MS_IN_S \
#                       - SMOOTH_PENALTY * np.abs(log_bit_rate - log_last_bit_rate) \
#                       - LONG_DELAY_PENALTY*(lat_penalty(latency/MS_IN_S)) * chunk_number \
#                       - MISSING_PENALTY * missing_count
#                           # - UNNORMAL_PLAYING_PENALTY*(playing_speed-NORMAL_PLAYING)*download_duration/MS_IN_S
#               # print(reward)
#               action_reward += reward

#               # chech whether need to wait, using number of available segs
#               if server.check_chunks_empty():
#                   # print "Enter wait"
#                   server_wait_time = server.wait()
#                   action_wait += server_wait_time
#                   # print " Has to wait: ", server_wait_time
#                   assert server_wait_time > 0.0
#                   assert server_wait_time < CHUNK_DURATION
#                   # print "Before wait, player: ", player.get_playing_time(), player.get_real_time()
#                   player.wait(server_wait_time)
#                   # print "After wait, player: ", player.get_playing_time(), player.get_real_time()
#                   buffer_length = player.get_buffer_length()

#               state = np.roll(state, -1, axis=1)
#               # Establish state for next iteration
#               # if not IF_NEW:
#               #   # FOR BW_TRACES
#               #   state[0, -1] = real_chunk_size / KB_IN_MB       # chunk size
#               #   state[1, -1] = download_duration / MS_IN_S      # downloading time
#               #   state[2, -1] = buffer_length / MS_IN_S          # buffer length
#               #   # state[3, -1] = chunk_number
#               #   state[4, -1] = log_bit_rate                     # video bitrate
#               #   # state[4, -1] = latency / MS_IN_S              # accu latency from start up
#               #   state[5, -1] = sync                             # whether there is resync
#               #   # state[5, -1] = player_state                       # state of player
#               #   state[6, -1] = server_wait_time / MS_IN_S       # time of waiting for server
#               #   state[7, -1] = freezing / MS_IN_S               # current freezing time

#               # else:
#               # FOR NEW_TRACES
#               state[0, -1] = real_chunk_size / NOR_CHUNK_SIZE             # chunk size
#               state[1, -1] = (download_duration - rtt) / MS_IN_S          # downloading time
#               state[2, -1] = buffer_length / MS_IN_S / NOR_BUFFER         # buffer length
#               # state[2, -1] = chunk_number / NOR_CHUNK                   # number of chunk sent
#               state[3, -1] = log_bit_rate / NOR_RATE                      # video bitrate
#               # state[4, -1] = latency / MS_IN_S                          # accu latency from start up
#               state[4, -1] = sync                                         # whether there is resync
#               state[5, -1] = player_state / NOR_STATE                     # state of player
#               state[6, -1] = server_wait_time / MS_IN_S/ NOR_WAIT         # time of waiting for server
#               state[7, -1] = freezing / MS_IN_S / NOR_FREEZING            # current freezing time
#               # print "Current index is: ", i, " and state is: ", state
#               # generate next set of seg size
#               # if add this, this will return to environment
#               # next_chunk_size_info = server.chunks[0][2]    # not useful
#               # state[7, :A_DIM] = next_chunk_size_info       # not useful
#               # print(state)
#               if CHUNK_IN_SEG == 5:
#                   ratio = np.random.uniform(RATIO_LOW_5, RATIO_HIGH_5)
#               else:
#                   ratio = np.random.uniform(RATIO_LOW_2, RATIO_HIGH_2)
#               server.set_ratio(ratio)
#               server.generate_next_delivery()
#               next_chunk_idx = server.get_next_delivery()[1]

#               if next_chunk_idx == 0 or sync:
#                   # if sync and not IF_NEW:
#                   #   # Process sync
#                   #   print "Should not happen!"
#                   #   assert 0 == 1
#                   # else:
#                   take_action = 1
#                   # print(action_reward)
#                   r_batch.append(action_reward)
#                   # If sync, might go to medium of segment, and there is no estimated chunk size
#                   action_prob = actor.predict(np.reshape(state, (1, S_INFO, S_LEN)))
#                   # # Using random
#                   print(action_prob)
#                   # action_cumsum = np.cumsum(action_prob)
#                   # action_num = (action_cumsum > np.random.randint(1, RAND_RANGE) / float(RAND_RANGE)).argmax()

#                   # Use action with largest prob
#                   action_num = action_prob.argmax()
#                   bit_rate = action_num

#                   # if action_num >= len(BITRATE):
#                   #   playing_speed = FAST_PLAYING
#                   # else:
#                   #   playing_speed = NORMAL_PLAYING

#                   log_file.write( str(server.time) + '\t' +
#                                   str(BITRATE[last_bit_rate]) + '\t' +
#                                   str(buffer_length) + '\t' +
#                                   str(action_freezing) + '\t' +
#                                   str(time_out) + '\t' +
#                                   str(action_wait) + '\t' +
#                                   str(sync) + '\t' +
#                                   str(latency) + '\t' +
#                                   str(player.get_state()) + '\t' +
#                                   str(int(action_num/len(BITRATE))) + '\t' +                          
#                                   str(action_reward) + '\n')
#                   log_file.flush()
#                   action_reward = 0.0
#                   action_freezing = 0.0
#                   action_wait = 0.0
#                   break

#       # need to modify
#       time_duration = server.get_time() - starting_time
#       if not IF_NEW:
#           tp_record, time_record = record_tp(player.get_throughput_trace(), player.get_time_trace(), starting_time_idx, time_duration + buffer_length) 
#       else:
#           tp_record, time_record = new_record_tp(player.get_throughput_trace(), player.get_time_trace(), starting_time_idx, time_duration + buffer_length) 
#       print(starting_time_idx, TRACE_NAME, len(player.get_throughput_trace()), player.get_time_idx(), len(tp_record), np.sum(r_batch))
#       log_file.write('\t'.join(str(tp) for tp in tp_record))
#       log_file.write('\n')

#       log_file.write('\t'.join(str(time) for time in time_record))
#       # log_file.write('\n' + str(IF_NEW))
#       log_file.write('\n' + str(starting_time))
#       log_file.write('\n')
#       log_file.close()        
                  
def lat_penalty(x):
    return 1.0/(1+math.exp(CONST-X_RATIO*x)) - 1.0/(1+math.exp(CONST))

def lat_penalty_new(x):
    return x


def record_tp(tp_trace, time_trace, starting_time_idx, duration):
    tp_record = []
    time_record = []
    offset = 0
    time_offset = 0.0
    num_record = int(np.ceil(duration/SEG_DURATION))
    for i in range(num_record):
        if starting_time_idx + i + offset >= len(tp_trace):
            offset = -len(tp_trace)
            time_offset += time_trace[-1]
        tp_record.append(tp_trace[starting_time_idx + i + offset])
        time_record.append(time_trace[starting_time_idx + i + offset] + time_offset)
    return tp_record, time_record

def new_record_tp(tp_trace, time_trace, starting_time_idx, duration):
    # print starting_time_idx
    # print duration
    start_time = time_trace[starting_time_idx]
    tp_record = []
    time_record = []
    offset = 0
    time_offset = 0.0
    i = 0
    time_range = 0.0
    # num_record = int(np.ceil(duration/SEG_DURATION))
    while  time_range < duration/MS_IN_S:
        # print time_trace[starting_time_idx + i + offset]
        tp_record.append(tp_trace[starting_time_idx + i + offset])
        time_record.append(time_trace[starting_time_idx + i + offset] + time_offset)
        i += 1
        if starting_time_idx + i + offset >= len(tp_trace):
            offset -= len(tp_trace)
            time_offset += time_trace[-1]
        time_range = time_trace[starting_time_idx + i + offset] + time_offset - start_time
    return tp_record, time_record

if __name__ == '__main__':
    main()
