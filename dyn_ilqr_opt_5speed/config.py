# Configuration for all files

class Config(object):
    # there are 3 model versions,3 q_versions (1 local, 2 global) and 4 target vesions() 
    # For model v0 and v1, there is no difference for q_version and target and loss, if want to add, please modify q_update_v0/v1
    # For model v2(dueling), 3 q_versions and 3 target (1 indep, and 2 global)
    model_version = 0           #v0: single(6*7),   v1: two output  v2: dueling ddqn
    ############# DO NOT CHANGE #######################
    q_version = 0               #v0: indep: yd = r + Qd_(sd', argmax Qd(sd',ad')), multiple optimizers  v1: global y = r + max Qd_(sd', argmax Qd(sd',ad')) one optimize v2: global y = r + ave( Qd_(sd', argmax Qd(sd',ad')))), one optimizer
    target_version = 0          #v0: naive Qd = V + Ad , dueling or not    v1: Qd = V + (Ad - ave(Ad)), must dueling.  v2: Qd = V + (Ad - max(Ad)), must dueling
    loss_version = 0            #v0: global loss, v1: indep loss
    if model_version == 0:
        q_version = 0
        target_version = 0
        loss_version = 0        # for model v0(single output), loss version must be v0
    ############# DO NOT CHANGE #######################
    initial_epsilon = 1.0 
    epsilon_start = 1.0
    epsilon_final = 0.0001
    if model_version == 0:
        epsilon_decay = 5000.0          # less, focus faster
    else:
        epsilon_decay = 5000.0          # less, focus faster
    if model_version == 0 or model_version == 1:
        logs_path = './models/logs_m_' + str(model_version) + '/t_' + str(target_version) + '/l_' + str(loss_version)
    else:
        logs_path = './models/logs_m_' + str(model_version) + '/q_' + str(q_version) + '/t_' + str(target_version) + '/l_' + str(loss_version)
    reply_buffer_size = 3000
    total_episode = 70000
    discount_factor = 0.99
    save_logs_frequency = 1000
    lr = 1e-3
    momentum = 0.9
    # batch_size = 300
    observe_episode = 5
    sampling_batch_size = 300
    update_target_frequency = 50
    show_loss_frequency = 1000
    maximum_model = 5
    random_seed = 11
    massive_result_files = './all_results/'
    a_massive_result_files = './amplified_all_results/'
    regular_test_files = './debug/'
    a_regular_test_files = './amplified_debug/'
    cdf_dir = '../compare/compare_results/'
    a_cdf_dir = '../compare/amplified_compare_results/'
    trace_idx = 20

class Env_Config(object):
    # For environment, ms
    bw_env_version = 0              # O for LTE (NYC), 1 for 3G (Norway)
    if bw_env_version == 0:
        data_dir = '../bw_traces/'
        test_data_dir = '../bw_traces_test/cooked_test_traces/'
    elif bw_env_version == 1:
        data_dir = '../new_traces/train_sim_traces/'
        test_data_dir = '../new_traces/test_sim_traces/'
    s_info = 10
    s_len = 15
    a_num = 2
    a_dims = [6, 3] # 6 bitrates and 3 playing speed
    video_terminal_length = 300             # 200 for training, 300 testing

    packet_payload_portion = 0.973
    rtt_low = 30.0
    rtt_high = 40.0 
    range_low = 40
    range_high = 50
    chunk_random_ratio_low = 0.95
    chunk_random_ratio_high = 1.05

    bitrate = [300.0, 500.0, 1000.0, 2000.0, 3000.0, 6000.0]
    # speeds = [0.90, 1.0, 1.10]
    speeds = [0.75, 0.9, 1.0, 1.1, 1.25]
    ms_in_s = 1000.0
    kb_in_mb = 1000.0   # in ms
    seg_duration = 1000.0
    chunk_duration = 200.0
    chunk_in_seg = seg_duration/chunk_duration
    chunk_seg_ratio = chunk_duration/seg_duration
    server_init_lat_low = 2
    server_init_lat_high = 5
    start_up_ssh = 2000.0
    freezing_tol = 3000.0 
    buffer_ub = server_init_lat_high*seg_duration
    
    default_action_1 = 0
    default_action_2 = 2
    skip_segs = 3.0
    repeat_segs = 3.0

    # Server info
    bitrate_low_noise = 0.7
    bitrate_high_noise = 1.3
    ratio_low_2 = 2.0               # this is the lowest ratio between first chunk and the sum of all others
    ratio_high_2 = 10.0             # this is the highest ratio between first chunk and the sum of all others
    ratio_low_5 = 0.75              # this is the lowest ratio between first chunk and the sum of all others
    ratio_high_5 = 1.0              # this is the highest ratio between first chunk and the sum of all others
    est_low_noise = 0.98
    est_high_noise = 1.02

    # Reward metrics parameters
    action_reward = 1.0 * chunk_seg_ratio   
    rebuf_penalty = 6.0                         
    smooth_penalty = 1.0
    long_delay_penalty_new = 0.5 * chunk_seg_ratio
    # long_delay_penalty = 4.0 * chunk_seg_ratio
    const = 6.0
    x_ratio = 1.0 
    speed_smooth_penalty = 2.0
    unnormal_playing_penalty = 2.0              
    skip_seg_penalty = 2.0              
    repeat_seg_penalty = 2.0      
    skip_latency = skip_segs * seg_duration + chunk_duration 

class Plot_Config(object):
    result_dir = './debug/'
    figures_dir = './test_figures/'
    result_file = './test_figures/'
    plt_buffer_a = 1e-5
