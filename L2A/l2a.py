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

class l2a_solver(object):
    def __init__(self):
        # General
        self.n_step = 10
        self.tp_history = []
        self.latency_history = []
        self.seg_duration = SEG_DURATION
        self.chunk_duration = CHUNK_DURATION
        self.hm_pre_steps = 5

        # L2A
        self.lastQuality = 0
        self.currentPlaybackRate = 1.0
        self.prev_w = [0]*len(BITRATE)
        self.w = [0]*len(BITRATE)
        self.horizon = 4
        self.vl = math.pow(self.horizon, 0.99)
        self.alpha = max(math.pow(self.horizon, 1), self.vl * math.sqrt(self.horizon))
        self.Q = self.vl
        self.react = 2

        # For DASH playback rate
        self.LIVEDELAY = 1.
        self.MINPLAYBACKRATECHANGE = 0.02
        self.LIVECATCHUPPLAYBACKRATE = 0.1

    def reset(self):
        self.tp_history = []
        self.latency_history = []
        self.prev_w = [0]*len(BITRATE)
        self.w = [0]*len(BITRATE)
        self.currentPlaybackRate = 1.0
        self.vl = math.pow(self.horizon, 0.99)
        self.alpha = max(math.pow(self.horizon, 1), self.vl * math.sqrt(self.horizon))
        self.Q = self.vl

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

    def harmonic_prediction(self):
        if self.hm_pre_steps > len(self.tp_history):
            tmp = self.tp_history
        else:
            tmp = self.tp_history[-self.hm_pre_steps:]
        return len(tmp)/(np.sum([1/tp for tp in tmp]))

    def choose_speed(self, speed):
        min_abs, min_idx = float('inf'), None
        for i in range(len(SPEED)):
            if abs(SPEED[i] - speed) < min_abs:
                min_abs = abs(SPEED[i] - speed)
                min_idx = i
        return min_idx

    def adjust_rate(self):
        # lastthroughput = max(self.tp_history[-1], 0.001)
        lastthroughput = self.harmonic_prediction()
        self.lastQuality = self.choose_rate(lastthroughput)
        self.prev_w[self.lastQuality] = 1

    def solve(self, buffer_length, curr_latency, player_state):
        # First of all, get speed
        ## DASH default playbac rate adaption
        cpr = self.LIVECATCHUPPLAYBACKRATE
        deltaLatency = curr_latency - self.LIVEDELAY 
        # print(curr_latency, self.LIVEDELAY)
        d = deltaLatency * 5

        # Playback rate must be between (1 - cpr) - (1 + cpr)
        # ex: if cpr is 0.5, it can have values between 0.5 - 1.5
        s = (cpr * 2) / (1 + math.pow(np.e, -d))
        speed = (1 - cpr) + s
        # take into account situations in which there are buffer stalls,
        # in which increasing playbackRate to reach target latency will
        # just cause more and more stall situations
        if player_state == 0 or player_state == 2:
            # const bufferLevel = getBufferLevel();
            if buffer_length > self.LIVEDELAY / 2:
                pass
            elif deltaLatency > 0:
                speed = 1.0                

        # don't change playbackrate for small variations (don't overload element with playbackrate changes)
        if abs(self.currentPlaybackRate - speed) <= self.MINPLAYBACKRATECHANGE:
            speed = self.currentPlaybackRate

        speed = self.choose_speed(speed)

        self.currentPlaybackRate = speed

        # Then get rate
        diff1 = []
        lastthroughput = self.harmonic_prediction()
        # lastthroughput = max(self.tp_history[-1], 0.001) # To avoid division with 0 (avoid infinity) in case of an absolute network outage, throught is the effective average of all chunks of previous segment (without idles)
        # print(lastthroughput, self.currentPlaybackRate, self.lastQuality, curr_latency)
        lastSegmentDurationS = self.seg_duration/MS_IN_S   # Implement in segment version
        V = lastSegmentDurationS            
        sign = 1
        for i in range(len(BITRATE)):
            if self.currentPlaybackRate * BITRATE[i]/KB_IN_MB > lastthroughput :
                # In this case buffer would deplete, leading to a stall, which increases latency and thus the particular probability of selsection of bitrate[i] should be decreased.
                sign = -1
            # print(self.prev_w[i], sign, V, self.alpha)
            # print(self.Q, self.vl)
            # The objective of L2A is to minimize the overall latency=request-response time + buffer length after download+ potential stalling (if buffer less than chunk downlad time)
            self.w[i] = self.prev_w[i] + sign * (V / (2 * self.alpha)) * ((self.Q + self.vl) * (self.currentPlaybackRate*BITRATE[i]/KB_IN_MB/lastthroughput)) #Lagrangian descent
            # print(self.w[i])
            # print('lll')

        # print(self.w)
        temp = [0]*len(BITRATE)
        for i in range(len(BITRATE)):
            temp[i] = abs(BITRATE[i] - np.dot(self.w, BITRATE))

        # print('tmp:', temp)
        quality = temp.index(min(temp))
        
        # We employ a cautious -stepwise- ascent
        if quality > self.lastQuality:
            if BITRATE[self.lastQuality + 1]/KB_IN_MB <= lastthroughput:
                quality = self.lastQuality + 1
        # print('quality', quality)
        # Provision against bitrate over-estimation, by re-calibrating the Lagrangian multiplier Q, to be taken into account for the next chunk
        if BITRATE[quality]/KB_IN_MB >= lastthroughput:
            self.Q = self.react * max(self.vl, self.Q)
        self.lastQuality = quality
        ##### L2A Quality selection is completed
        
        # print(quality, speed)
        return quality, speed
