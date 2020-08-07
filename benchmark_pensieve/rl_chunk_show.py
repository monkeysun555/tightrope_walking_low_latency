import numpy as np
import os
import matplotlib.pyplot as plt

SERVER_START_UP_TH = 2000.0			#<=== change this to get corresponding log files
MS_IN_S = 1000.0
RESULT_DIR = './test_results/'
FIGURES_DIR = './test_figures/'
RESULT_FILE = './test_figures/'
       
PLT_BUFFER_A = 1e-5		#ms
MS_IN_S = 1000.0
KB_IN_MB = 1000.0
SAVE = 1

SEG_DURATION = 1000.0
CHUNK_DURATION = 200.0
START_UP_TH = 2000.0

CHUNK_IN_SEG = int(SEG_DURATION/CHUNK_DURATION)		# 4

def plt_fig(trace, data_name, data_type):
	# print(tp_trace)
	y_axis_upper = np.ceil(np.max(trace)*1.35)

	# For negative reward
	# y_axis_lower = np.floor(np.minimum(np.min(trace)*1.1,0.0))
	y_axis_lower = 0.0

	p = plt.figure(figsize=(20,5))
	plt.plot(range(1,len(trace)+1), trace, color='chocolate', label=data_name + '_' + data_type, linewidth=1.5,alpha=0.9)
	plt.legend(loc='upper right',fontsize=30)
	plt.grid(linestyle='dashed', axis='y',linewidth=1.5, color='gray')
	plt.axis([0, len(trace), y_axis_lower, y_axis_upper])
	plt.xticks(np.arange(0, len(trace)+1, 50))
	plt.tick_params(labelsize=20)
	# plt.yticks(np.arange(200, 1200+1, 200))
	plt.close()
	return p

def plt_fig_full(trace, data_name, data_type):
	# print(tp_trace)
	if data_type == 'bitrate':
		# y_axis_upper = np.ceil(np.max(trace)*1.35)
		y_axis_upper = 8000.0
	elif data_type == 'reward':
		y_axis_upper = 1.0
	# For negative reward
	# y_axis_lower = np.floor(np.minimum(np.min(trace)*1.1,0.0))
	y_axis_lower = 0.0

	x_value = []
	curr_x = CHUNK_DURATION/MS_IN_S
	for i in range(len(trace)):
		x_value.append(curr_x)
		curr_x += SEG_DURATION/MS_IN_S
	p = plt.figure(figsize=(20,5))
	plt.plot(x_value, trace, color='chocolate', label=data_name + '_' + data_type, linewidth=1.5,alpha=0.9)
	plt.legend(loc='upper right',fontsize=30)
	plt.grid(linestyle='dashed', axis='y',linewidth=1.5, color='gray')
	plt.axis([0, int(len(trace)/SEG_DURATION*MS_IN_S), y_axis_lower, y_axis_upper])
	plt.xticks(np.arange(0, int(len(trace)/SEG_DURATION*MS_IN_S)+1, 50))
	# plt.yticks(np.arange(200, 1200+1, 200))
	plt.tick_params(labelsize=20)
	plt.close()
	return p

def plt_fig_mix_bw_action(trace1, trace2, new_time_trace, data_name, data_type1, data_type2):
	# type1: tp,  type2: bitrate
	init_time = new_time_trace[0]
	new_time_trace = [time-init_time for time in new_time_trace]
	y_axis_upper = 10000.0
	# For negative reward
	# y_axis_lower = np.floor(np.minimum(np.min(trace)*1.1,0.0))
	y_axis_lower = 0.0

	x_value = []
	curr_x = 0.0
	for i in range(len(trace2)):
		x_value.append(curr_x)
		curr_x += SEG_DURATION/MS_IN_S
	p = plt.figure(figsize=(20,5))
	plt.plot(x_value, trace2, color='chocolate', label=data_name + '_' + data_type2, linewidth=1.5,alpha=0.9)
	# plt.plot(range(1,len(trace1)+1), trace1*KB_IN_MB, color='blue', label=data_name + '_' + data_type1, linewidth=1.5,alpha=0.9)
	print len(new_time_trace)
	print len(trace1)
	plt.plot(new_time_trace, trace1*KB_IN_MB, color='blue', label=data_name + '_' + data_type1, linewidth=1.5,alpha=0.9)

	plt.legend(loc='upper right',fontsize=30)
	plt.grid(linestyle='dashed', axis='y',linewidth=1.5, color='gray')
	plt.axis([0, int(len(trace2)/SEG_DURATION*MS_IN_S), y_axis_lower, y_axis_upper])
	plt.xticks(np.arange(0, int(len(trace2)/SEG_DURATION*MS_IN_S)+1, 20))
	# plt.yticks(np.arange(200, 1200+1, 200))
	plt.tick_params(labelsize=20)
	plt.close()
	return p


def plt_buffer(time_trace, buffer_trace, state_trace, latency_trace, starting_time, data_name, data_type):
	# y_axis_upper = np.ceil(np.max(latency_trace)*1.6/MS_IN_S)
	y_axis_upper = 8
	latency_trace = [lat/MS_IN_S for lat in latency_trace]
	latency_time = [time/MS_IN_S for time in time_trace]
	time_trace = [0.0] + time_trace
	buffer_trace = [0.0] + buffer_trace
	state_trace = [0] + state_trace
	insert_buffer_trace = []
	insert_time_trace = []
	plot_state_left = 1
	# print(len(time_trace), len(state_trace))
	assert len(time_trace) == len(buffer_trace)
	for i in range(0, len(time_trace)):
		if state_trace[i] == 0:
			if i >= 1:
				if state_trace[i-1] == 1:
					insert_time = np.minimum(time_trace[i] - PLT_BUFFER_A, time_trace[i-1]+buffer_trace[i-1])
					insert_buffer = np.maximum(0.0, buffer_trace[i-1] - (time_trace[i] - time_trace[i-1]))
					insert_buffer_trace.append(insert_buffer)
					insert_time_trace.append(insert_time)
			plot_state_left = 1
			continue
		else:
			if not plot_state_left == 0:
				plot_state_left -= 1
				continue
			insert_buffer = np.maximum(0.0, buffer_trace[i-1] - (time_trace[i] - time_trace[i-1]))
			insert_time = np.minimum(time_trace[i] - PLT_BUFFER_A, time_trace[i-1]+buffer_trace[i-1])
			insert_buffer_trace.append(insert_buffer)
			insert_time_trace.append(insert_time)
			if insert_time < time_trace[i] - PLT_BUFFER_A:
				assert insert_buffer == 0.0
				insert_buffer_trace.append(0.0)
				insert_time_trace.append(time_trace[i] - PLT_BUFFER_A)

	# Need to adjust about freezing
	# combine two buffer_traces
	plt_buffer_trace = []
	plt_time_trace = []
	print(len(insert_time_trace), len(time_trace), len(latency_trace))
	print(insert_time_trace[-1], time_trace[-1])
	# print(latency_trace)
	# print(len(insert_time_trace), len(insert_buffer_trace))
	for i in range(len(time_trace)):
		# if len(insert_time_trace) == 0:
		# 	plt_time_trace.append(time_trace[i:])
		# 	plt_buffer_trace.append(buffer_trace[i:])
		# 	break
		# print(i, len(time_trace))
		if len(insert_time_trace) > 0:
			while insert_time_trace[0] < time_trace[i]:
				plt_time_trace.append(insert_time_trace.pop(0)/MS_IN_S)
				plt_buffer_trace.append(insert_buffer_trace.pop(0)/MS_IN_S)
				# print(len(insert_time_trace), len(time_trace), i)
				if len(insert_time_trace) == 0:
					# plt_time_trace.extend(time_trace[i:])
					# plt_buffer_trace.extend(buffer_trace[i:])
					break
		plt_time_trace.append(time_trace[i]/MS_IN_S)
		plt_buffer_trace.append(buffer_trace[i]/MS_IN_S)
		
	# print(plt_time_trace, plt_buffer_trace)
	# print(len(plt_time_trace), len(plt_buffer_trace))

	p = plt.figure(figsize=(20,5))
	plt.plot(plt_time_trace, plt_buffer_trace, color='chocolate', label=data_name + '_' + data_type, linewidth=1.5,alpha=0.9)
	plt.plot(latency_time, latency_trace, 'k', label=data_name + '_latency', linewidth=1.5,alpha=0.9)
	plt.plot([latency_time[0], latency_time[-1]], [starting_time/MS_IN_S]*2, 'r--', linewidth=1.5,alpha=0.9)
	plt.legend(loc='upper right',fontsize=30)
	plt.grid(linestyle='dashed', axis='y',linewidth=1.5, color='gray')
	plt.axis([0, plt_time_trace[-1], 0, y_axis_upper])
	plt.xticks(np.arange(0, plt_time_trace[-1]+1, 50))
	plt.tick_params(labelsize=20)
	
	# plt.yticks(np.arange(200, 1200+1, 200))
	plt.close()
	return p


def bar_freezing(time_trace, freezing_trace, data_name, data_type):
	bar_pos = []
	height = []
	for i in range(len(freezing_trace)):
		if not freezing_trace[i] == 0:
			bar_pos.append((time_trace[i] - freezing_trace[i]/2.0)/ MS_IN_S)
			height.append(freezing_trace[i] /MS_IN_S)

	# y_axis_upper = np.max(height) * 1.35
	y_axis_upper = 3.0
	p = plt.figure(figsize=(20,5))

	plt.bar(bar_pos, height, color='chocolate', label=data_name + '_' + data_type)
	plt.legend(loc='upper right',fontsize=30)
	plt.grid(linestyle='dashed', axis='y',linewidth=1.5, color='gray')
	plt.axis([0, time_trace[-1]/MS_IN_S + 1, 0, y_axis_upper])
	plt.xticks(np.arange(0, time_trace[-1]/MS_IN_S + 1, 50))
	plt.tick_params(labelsize=20)
	# plt.yticks(np.arange(200, 1200+1, 200))
	plt.close()
	return p

def bar_wait(time_trace, wait_trace, data_name, data_type):
	bar_pos = []
	height = []
	for i in range(len(wait_trace)):
		if not wait_trace[i] == 0:
			bar_pos.append((time_trace[i] - wait_trace[i]/2.0)/ MS_IN_S)
			height.append(wait_trace[i] /MS_IN_S)
	if not len(height):
		return 
	# y_axis_upper = np.max(height) * 1.35
	y_axis_upper = 1.0
	p = plt.figure(figsize=(20,5))
	plt.bar(bar_pos, height, color='chocolate', label=data_name + '_' + data_type)
	plt.legend(loc='upper right',fontsize=30)
	plt.grid(linestyle='dashed', axis='y',linewidth=1.5, color='gray')
	plt.axis([0, time_trace[-1]/MS_IN_S + 1, 0, y_axis_upper])
	plt.xticks(np.arange(0, time_trace[-1]/MS_IN_S + 1, 50))
	plt.tick_params(labelsize=20)
	# plt.yticks(np.arange(200, 1200+1, 200))
	plt.close()
	return p

# def bar_speed(time_trace, speed_trace, data_name, data_type):
# 	bar_pos = []
# 	height = []
# 	width = []
# 	for i in range(len(speed_trace)):
# 		if not speed_trace[i] == 0:
# 			bar_pos.append(time_trace[i]/ MS_IN_S)
# 			height.append(speed_trace[i])
# 			width.append(0.1)

# 	y_axis_upper = np.max(height) * 1.35
# 	p = plt.figure(figsize=(20,5))

# 	plt.bar(bar_pos, height, width, color='chocolate', label=data_name + '_' + data_type)
# 	plt.legend(loc='upper right',fontsize=30)
# 	plt.grid(linestyle='dashed', axis='y',linewidth=1.5, color='gray')
# 	plt.axis([0, time_trace[-1]/MS_IN_S + 1, 0, y_axis_upper])
# 	plt.xticks(np.arange(0, time_trace[-1]/MS_IN_S + 1, 50))
# 	plt.tick_params(labelsize=20)
# 	# plt.yticks(np.arange(200, 1200+1, 200))
# 	plt.close()
# 	return p

def bar_missing(time_trace, sync_trace, missing_trace, data_name, data_type):
	bar_pos = []
	height = []
	for i in range(len(sync_trace)):
		if sync_trace[i] == 1:
			assert not missing_trace[i] == 0
			bar_pos.append(time_trace[i]/ MS_IN_S)
			height.append(missing_trace[i])

	if len(height) >= 1:
		y_axis_upper = np.max(height) * 1.35

		p = plt.figure(figsize=(20,5))

		plt.bar(bar_pos, height, color='chocolate', label=data_name + '_' + data_type)
		plt.legend(loc='upper right',fontsize=30)
		plt.grid(linestyle='dashed', axis='y',linewidth=1.5, color='gray')
		plt.axis([0, time_trace[-1]/MS_IN_S + 1, 0, y_axis_upper])
		plt.xticks(np.arange(0, time_trace[-1]/MS_IN_S + 1, 50))
		plt.tick_params(labelsize=20)
		# plt.yticks(np.arange(200, 1200+1, 200))
		plt.close()
		return p
	else:
		return None

def main():
	if not os.path.isdir(FIGURES_DIR):
		os.makedirs(FIGURES_DIR)
	if not os.path.isdir(RESULT_FILE):
		os.makedirs(RESULT_FILE)

	results = os.listdir(RESULT_DIR)
	file_records = []
	for data in results:
		file_info = []
		file_path = RESULT_DIR + data
		file_info.append(data.split('.')[0])
		with open(file_path, 'rb') as f:
			for line in f:
				parse = line.strip('\n')
				parse = parse.split('\t')				
				file_info.append(parse)
		file_records.append(file_info)
	# print(file_records)

	# For figs
	tp_figs = []
	reward_figs = []
	bitrate_figs = []
	buffer_figs = []
	freezing_figs = []
	server_wait_figs = []
	# missing_figs = []
	speed_figs = []
	server_mix_figs = []

	# For numericals
	n_files = ['files']
	n_throughput = ['throughut']
	n_reward = ['reward']
	n_freezing = ['freezing']
	n_n_freezing = ['#freezing']
	n_latency = ['latency']
	n_switch = ['switch']
	n_sync = ['sync']
	n_rate = ['rate']
	n_starting_time  = ['starting_time']

	log_path = RESULT_FILE + 'table_rlchunk'
	log_file = open(log_path, 'wb')

	for i in range(len(file_records)):
		starting_time = float(file_records[i][-1][0])
		tp_trace = np.array(file_records[i][-3]).astype(np.float)
		new_time_trace = np.array(file_records[i][-2]).astype(np.float)
		records = file_records[i][1:-3]
		data_name = file_records[i][0]
		n_starting_time.append(starting_time)

		#For buffer
		bitrate_trace = [float(info[1]) for info in records]
		buffer_trace = [float(info[2]) for info in records]
		freezing_trace = [float(info[3]) for info in records]
		server_wait_trace = [float(info[5]) for info in records]
		sync_trace = [float(info[6]) for info in records]
		latency_trace = [float(info[7]) for info in records]
		state_trace = [float(info[8]) for info in records]
		# speed_trace = [float(info[9]) for info in records]
		reward_trace = [float(info[-1]) for info in records]


		# For numerical
		n_files.append(data_name)
		n_reward.append(np.round(np.sum(reward_trace)/len(reward_trace),3))
		n_freezing.append(np.round(np.sum(freezing_trace)/MS_IN_S,3))
		n_n_freezing.append(np.sum([ x>0 for x in freezing_trace]))
		n_latency.append(np.round(np.sum(latency_trace)/(len(latency_trace)*MS_IN_S),3))
		n_throughput.append(np.round(np.sum(tp_trace)/len(tp_trace),3))
		n_sync.append(np.sum(sync_trace))
		n_rate.append(np.round(np.sum(bitrate_trace)/len(bitrate_trace), 3))

		switch = 0	
		temp_rate = bitrate_trace[0]
		for i in range(len(bitrate_trace)):
			if not bitrate_trace[i] == temp_rate:
				switch += 1
				temp_rate = bitrate_trace[i]
		assert switch >= np.sum(sync_trace)
		n_switch.append(switch-np.sum(sync_trace))

		# Time
		real_time_trace = [float(info[0]) for info in records]
		plt_time_trace = [r_time - starting_time for r_time in real_time_trace]

		# For tp
		# print(len(tp_trace))
		tp_fig = plt_fig(tp_trace, data_name, 'tp')
		tp_figs.append([data_name, 'tp', tp_fig])
		
		buffer_fig = plt_buffer(plt_time_trace, buffer_trace, state_trace, latency_trace, starting_time, data_name, 'buffer')
		buffer_figs.append([data_name, 'buffer', buffer_fig])

		# For bitrate
		bitrate_fig = plt_fig_full(bitrate_trace, data_name, 'bitrate')
		bitrate_figs.append([data_name, 'bitrate', bitrate_fig])

		#For reward
		# print(reward)
		reward_fig = plt_fig_full(reward_trace, data_name, 'reward')
		reward_figs.append([data_name,'reward', reward_fig])

		# For freezing
		freezing_fig = bar_freezing(plt_time_trace, freezing_trace, data_name, 'freezing')
		freezing_figs.append([data_name, 'freezing', freezing_fig])


		# For server wait
		server_wait_fig = bar_wait(plt_time_trace, server_wait_trace, data_name, 'idle')
		server_wait_figs.append([data_name, 'idle', server_wait_fig])

		# For missing
		# missing_fig = bar_missing(plt_time_trace, sync_trace, missing_trace, data_name, 'missing')
		# if not missing_fig == None:
		# 	missing_figs.append([data_name, 'missing', missing_fig])

		# For speed 
		# speed_fig = bar_speed(plt_time_trace, speed_trace, data_name, 'speed')
		# speed_figs.append([data_name, 'speed', speed_fig])

		server_mix_fig = plt_fig_mix_bw_action(tp_trace, bitrate_trace, new_time_trace, data_name, 'tp', 'bitrate')
		server_mix_figs.append([data_name, 'mix', server_mix_fig])


	if SAVE:
		for p in tp_figs:
			p[2].savefig(FIGURES_DIR + p[0] + '_' + p[1] + '.eps', format='eps', dpi=1000, figsize=(30, 10))

		# for p in reward_figs:
		# 	p[2].savefig(FIGURES_DIR + p[0] + '_' + p[1] + '.eps', format='eps', dpi=1000, figsize=(30, 10))
		
		for p in bitrate_figs:
			p[2].savefig(FIGURES_DIR + p[0] + '_' + p[1] + '.eps', format='eps', dpi=1000, figsize=(30, 10))
		
		for p in buffer_figs:
			p[2].savefig(FIGURES_DIR + p[0] + '_' + p[1] + '.eps', format='eps', dpi=1000, figsize=(30, 10))
		
		for p in freezing_figs:
			p[2].savefig(FIGURES_DIR + p[0] + '_' + p[1] + '.eps', format='eps', dpi=1000, figsize=(30, 10))
		
		# for p in server_wait_figs:
		# 	p[2].savefig(FIGURES_DIR + p[0] + '_' + p[1] + '.eps', format='eps', dpi=1000, figsize=(30, 10))
		
		# for p in missing_figs:
		# 	p[2].savefig(FIGURES_DIR + p[0] + '_' + p[1] + '.eps', format='eps', dpi=1000, figsize=(30, 10))

		# for p in speed_figs:
		# 	p[2].savefig(FIGURES_DIR + p[0] + '_' + p[1] + '.eps', format='eps', dpi=1000, figsize=(30, 10))

		for p in server_mix_figs:
			p[2].savefig(FIGURES_DIR + p[0] + '_' + p[1] + '.eps', format='eps', dpi=1000, figsize=(30, 10))
	
		for i in range(len(n_files)):
			current_log = []
			current_log.append(n_files[i])
			current_log.append(n_throughput[i])
			current_log.append(n_rate[i])
			current_log.append(n_reward[i])
			current_log.append(n_freezing[i])
			current_log.append(n_n_freezing[i])
			current_log.append(n_latency[i])
			current_log.append(n_switch[i])
			current_log.append(n_sync[i])
			current_log.append(n_starting_time[i])
			log_file.write(str(current_log) + '\n')
			current_log = []

		log_file.flush()

		log_file.write('\n')

if __name__ == '__main__':
	main()
