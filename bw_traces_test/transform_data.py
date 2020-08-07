import csv
import os
import numpy as np

PATH = ['./Measurement_1/', './Measurement_2/', './Measurement_3/']

SAVE_DIR = './cooked_test_traces/'
TRACE_LEN = 150

def main():
	if not os.path.exists(SAVE_DIR):
		os.makedirs(SAVE_DIR)
	file_idx = 0
	for data_path in PATH:
		datas = os.listdir(data_path)
		for data in datas:
			if 'csv' not in data:
				continue

			file_path = data_path + data
			time_trace = []
			throughput_trace = []
			time = 0.0
			# print(data)
			with open(file_path) as csv_file:
				csv_reader = csv.reader(csv_file)
				line_count = 0
				for row in csv_reader:
					if row[1] == 'Mbits/sec':
						throughput_trace.append(float(row[0]))
					else:
						print row[1]
						assert 0 == 1

			# Generate random selection
			name_1 = data[:-4] + '_' + str(file_idx) + '_' +  str(1) + '.txt'
			name_2 = data[:-4] + '_' + str(file_idx) + '_' +  str(2) + '.txt'

			assert len(throughput_trace) > 450
			first_idx = np.random.randint(0, len(throughput_trace) - TRACE_LEN)
			# Get first trace
			trace_1 = throughput_trace[first_idx:first_idx+TRACE_LEN]

			if first_idx > len(throughput_trace) - first_idx - TRACE_LEN:
				assert first_idx > TRACE_LEN
				second_idx = np.random.randint(0, first_idx - TRACE_LEN)
			else:
				assert first_idx + TRACE_LEN < len(throughput_trace) - TRACE_LEN
				second_idx = np.random.randint(first_idx+TRACE_LEN, len(throughput_trace) - TRACE_LEN)
			assert second_idx + TRACE_LEN < len(throughput_trace)
			trace_2 = throughput_trace[second_idx:second_idx+TRACE_LEN]

			# Save
			save_path_1 = SAVE_DIR + name_1
			save_path_2 = SAVE_DIR + name_2

			end_log_file_1 = open(save_path_1, 'wb')
			for x in trace_1:
				end_log_file_1.write(str(x) + '\n')
			end_log_file_1.close()

			end_log_file_2 = open(save_path_2, 'wb')
			for x in trace_2:
				end_log_file_2.write(str(x) + '\n')
			end_log_file_2.close()

		file_idx += 1

if __name__ == '__main__':
	main()

