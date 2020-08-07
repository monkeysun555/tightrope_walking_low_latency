import numpy as np
import math 

LQR_DEBUG = 0
iLQR_SHOW = 0
RTT_LOW = 0.02
SEG_DURATION = 1.0
CHUNK_DURATION = 0.2
CHUNK_IN_SEG = SEG_DURATION/CHUNK_DURATION
BITRATE = [300.0, 500.0, 1000.0, 2000.0, 3000.0, 6000.0]
SPEED = [0.9, 1.0, 1.1]
MS_IN_S = 1000.0
KB_IN_MB = 1000.0
MIN_RATE = 10**-8
MAX_RATE = BITRATE[-1]/KB_IN_MB

class stallion_solver(object):
    def __init__(self):
        # For new traces
        self.tp_f = 1.0
        self.latency_f = 1.25
        self.n_step = 10
        self.target_latency = 1.0
        self.speed_buffer_tth = 0.6
        self.tp_history = []
        self.latency_history = []
        self.seg_duration = SEG_DURATION
        self.chunk_duration = CHUNK_DURATION

    def reset(self):
        self.tp_history = []
        self.latency_history = []

    def update_tp_latency(self, tp, latency):
        self.tp_history += [tp]
        self.latency_history += [latency]
        if len(self.tp_history) > self.n_step:
            self.tp_history.pop(0)
        if len(self.latency_history) > self.n_step:
            self.latency_history.pop(0)

    def choose_rate(self, tp):
        i = 0
        for i in reversed(range(len(BITRATE))):
            if BITRATE[i]/KB_IN_MB < tp:
                return i
        return i

    def solve(self, buffer_length, curr_latency):
        # First of all, get speed
        a1, a2 = None, None
        if curr_latency >= self.target_latency and buffer_length >= self.speed_buffer_tth:
            a2 = 2
        else:
            a2 = 1

        # Get rate
        mean_tp, mean_latency = np.mean(self.tp_history), np.mean(self.latency_history)
        std_tp, std_latency = np.std(self.tp_history), np.std(self.latency_history)
        predict_tp = mean_tp - self.tp_f*std_tp
        predict_latency = mean_latency + self.latency_f*std_latency
        overhead = max(predict_latency - self.target_latency, 0)
        # print(predict_tp, predict_latency, overhead)
        if overhead >= self.seg_duration:
            a1 = 0
        else:
            dead_time = self.seg_duration - overhead
            ratio = dead_time/self.seg_duration
            predict_tp *= ratio
            a1 = self.choose_rate(predict_tp)
        # print(a1,a2)
        return a1, a2

