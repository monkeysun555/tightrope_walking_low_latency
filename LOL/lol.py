import numpy as np
import math 
from itertools import permutations
from config import *


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

class lol_solver(object):
    def __init__(self):
        # General
        self.n_step = 10
        self.hm_pre_steps = 5
        self.tp_history = []
        self.latency_history = []
        self.seg_duration = SEG_DURATION
        self.chunk_duration = CHUNK_DURATION
        self.eta = 0.9

        # Lol
        self.lookahead = 3
        self.options = self.getPermutations(BITRATE)
        self.lastrate = BITRATE[0]/KB_IN_MB
        self.lastspeed = 1.0

        # dash
        self.LIVEDELAY = 1.5
        self.MINPLAYBACKRATECHANGE = 0.05
        self.LIVECATCHUPPLAYBACKRATE = 0.1
        self.LIVECATCHUPMINDRIFT = 0.1

    def reset(self):
        self.tp_history = []
        self.latency_history = []
        self.lastrate = BITRATE[0]/KB_IN_MB
        self.lastspeed = 1.0

        # self.prev_w = [0]*len(BITRATE)
        # self.w = [0]*len(BITRATE)
        # self.currentPlaybackRate = 1.0
        # self.vl = math.pow(self.horizon, 0.99)
        # self.alpha = max(math.pow(self.horizon, 1), self.vl * math.sqrt(self.horizon))
        # self.Q = self.vl

    def update_tp_latency(self, tp, latency):
        self.tp_history += [tp]
        self.latency_history += [latency]
        if len(self.tp_history) > self.n_step:
            self.tp_history.pop(0)
        if len(self.latency_history) > self.n_step:
            self.latency_history.pop(0)

    def harmonic_prediction(self):
        if self.hm_pre_steps > len(self.tp_history):
            tmp = self.tp_history
        else:
            tmp = self.tp_history[-self.hm_pre_steps:]
        return len(tmp)/(np.sum([1/tp for tp in tmp]))

    def getPermutations(self, arr):
        res = [[]]
        for i in range(self.lookahead):
            new_res = []
            for pre in res:
                for j in range(len(arr)):
                    new_res.append(pre+[arr[j]])
            res = new_res
        return res

    def dash_playbackrate(self, curr_latency, buffer_length, player_state):
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
            # const bufferLevel = getBufferLevel()
            if buffer_length > self.LIVEDELAY / 2:
                pass
            elif deltaLatency > 0:
                speed = 1.0                

        # don't change playbackrate for small variations (don't overload element with playbackrate changes)
        if abs(self.lastspeed - speed) <= self.MINPLAYBACKRATECHANGE:
            speed = self.lastspeed

        # Change speed to index
        speed_idx = self.choose_speed(speed)
        return SPEED[speed_idx]

    def choose_speed(self, speed):
        min_abs, min_idx = float('inf'), None
        for i in range(len(SPEED)):
            if abs(SPEED[i] - speed) < min_abs:
                min_abs = abs(SPEED[i] - speed)
                min_idx = i
        return min_idx

    def solve(self, buffer_length, curr_latency, player_state):
        maxReward = float('-inf')
        bestOption = []
        bestQoeInfo = {}

        # Qoe stuff
        qoeEvaluatorTmp = QoeEvaluator()
        minBitrateMbps = BITRATE[0] / 1000.0    # in Mbps
        maxBitrateMbps = BITRATE[-1] / 1000.0   # in Mbps

        qualityList = []

        for i in range(len(self.options)):
            tmpLatency = curr_latency
            tmpBuffer = buffer_length
            curr_option = self.options[i]
            # print('curr_option: ', curr_option)
            qoeEvaluatorTmp.setupPerSegmentQoe(self.seg_duration, maxBitrateMbps, minBitrateMbps, self.lastrate, self.lastspeed)

            # tmpPlaybackSpeed = self.lastspeed   # initial to 1.0
            tmpSpeeds = []
            # Estimate futureBandwidth as harmonic mean of past X throughput values
            futureBandwidthMbps = self.eta * self.harmonic_prediction()

            # For each segment in lookahead window (window size: futureSegmentCount)
            for j in range(self.lookahead):
                segmentBitrateMbps = curr_option[j] / KB_IN_MB

                futureSegmentSizeMbits = self.seg_duration * segmentBitrateMbps
                downloadTime = futureSegmentSizeMbits / futureBandwidthMbps
                # print('downloadTime', downloadTime)
                # If buffer underflow
                if downloadTime > tmpBuffer: 
                    segmentRebufferTime = downloadTime - tmpBuffer
                    tmpBuffer = self.chunk_duration
                    tmpLatency += segmentRebufferTime
                    player_state = 0
                else:
                    segmentRebufferTime = 0
                    tmpBuffer -= downloadTime
                    tmpBuffer += self.seg_duration
                    player_state = 1


                liveCatchUpPlaybackRate = self.LIVECATCHUPPLAYBACKRATE
                liveCatchUpMinDrift = self.LIVECATCHUPMINDRIFT        
                playbackStalled = False

                # Check if need to catch up
                if abs(tmpLatency - self.LIVEDELAY) >= liveCatchUpMinDrift:
                    needToCatchUp = True
                else:
                    needToCatchUp = False

                # If need to catch up, calculate new playback rate (custom/default methods)
                if needToCatchUp:
                    newRate = self.dash_playbackrate(tmpLatency, tmpBuffer, player_state)
                    futurePlaybackSpeed = newRate
                else:
                    futurePlaybackSpeed = 1.0
                
                tmpSpeeds += [futurePlaybackSpeed]
                catchupDuration = self.seg_duration - (self.seg_duration / futurePlaybackSpeed)
                futureLatency = tmpLatency - catchupDuration

                qoeEvaluatorTmp.updateQoE(segmentBitrateMbps, segmentRebufferTime, futureLatency, futurePlaybackSpeed, segmentRebufferTime)

                tmpLatency = futureLatency
                # tmpPlaybackSpeed = futurePlaybackSpeed

            reward = qoeEvaluatorTmp.getPerSegmentQoe()

            if (reward > maxReward):
                maxReward = reward
                bestOption = curr_option
                # bestQoeInfo = qoeEvaluatorTmp.
                bestSpeed = tmpSpeeds

        # Take the first

        # print('Best:', bestOption, bestSpeed)
        self.lastrate = bestOption[0]
        self.lastspeed = bestSpeed[0]
        return BITRATE.index(bestOption[0]), SPEED.index(bestSpeed[0])


class QoeEvaluator():
    def __init__(self):
        self.voPerSegmentQoeInfo = None

    def getPerSegmentQoe(self):
        return self.voPerSegmentQoeInfo.getTotal()

    def updateQoE(self, segmentBitrateMbps, segmentRebufferTime, latency, playbackSpeed, rebufferTime):
        log_quality = np.log(segmentBitrateMbps/(BITRATE[0]/KB_IN_MB))

        self.voPerSegmentQoeInfo.bitrateWSum += self.voPerSegmentQoeInfo.weights['bitrateReward']*log_quality

        if self.voPerSegmentQoeInfo.lastBitrate:
            log_pre_quality = np.log(self.voPerSegmentQoeInfo.lastBitrate/(BITRATE[0]/KB_IN_MB))
            self.voPerSegmentQoeInfo.bitrateSwitchWSum += self.voPerSegmentQoeInfo.weights['bitrateSwitchPenalty'] * abs(log_quality - log_pre_quality)
        self.voPerSegmentQoeInfo.lastBitrate = segmentBitrateMbps

        self.voPerSegmentQoeInfo.rebufferWSum += self.voPerSegmentQoeInfo.weights['rebufferPenalty'] * rebufferTime

        self.voPerSegmentQoeInfo.latencyWSum += self.voPerSegmentQoeInfo.weights['latencyPenalty'] * latency

        self.voPerSegmentQoeInfo.playbackSpeedWSum += self.voPerSegmentQoeInfo.weights['playbackSpeedPenalty'] * abs(1 - playbackSpeed)

        if self.voPerSegmentQoeInfo.lastSpeed:
            lastSpeed = self.voPerSegmentQoeInfo.lastSpeed 
            self.voPerSegmentQoeInfo.speedChangeWSum += self.voPerSegmentQoeInfo.weights['speedChangePenalty'] * abs(playbackSpeed - lastSpeed)
        self.voPerSegmentQoeInfo.lastSpeed = playbackSpeed

        # Update: Total Qoe value
        self.updateTotal()

    def setupPerSegmentQoe(self, segmentDuration, maxBitrateMbps, minBitrateMbps, lastRate, lastSpeed):
        # Set up Per Segment QoeInfo
        self.voPerSegmentQoeInfo = self.createQoeInfo('segment', segmentDuration, maxBitrateMbps, minBitrateMbps)
        # self.setInitialLastBitrate(lastRate)
        # self.setInitialLastSpeed(lastSpeed)

    def setInitialLastBitrate(self, lastRate):
        self.voPerSegmentQoeInfo.lastBitrate = lastRate

    def setInitialLastSpeed(self, lastSpeed):
        self.voPerSegmentQoeInfo.lastSpeed = lastSpeed

    def updateTotal(self):
        return self.voPerSegmentQoeInfo.updateTotal()

    def createQoeInfo(self, fragmentType, fragmentDuration, maxBitrateMbps, minBitrateMbps):
        '''
         * [Weights][Source: Abdelhak Bentaleb, 2020 (last updated: 30 Mar 2020)]
         * bitrateReward:           segment duration, e.g. 0.5s
         * bitrateSwitchPenalty:    0.02s or 1s if the bitrate switch is too important
         * rebufferPenalty:         max encoding bitrate, e.g. 1 Mbps
         * latencyPenalty:          if L ≤ 1.1 seconds then = min encoding bitrate * 0.05, otherwise = max encoding bitrate * 0.1
         * playbackSpeedPenalty:    min encoding bitrate, e.g. 0.2 Mbps
        '''

        # Create new QoeInfo object
        qoeInfo = QoeInfo()
        qoeInfo.qoetype = fragmentType

        # Set weight: bitrateReward
        if not fragmentDuration:
            qoeInfo.weights['bitrateReward'] = 0.001
        else:
            qoeInfo.weights['bitrateReward'] = Env_Config.action_reward*Env_Config.chunk_in_seg       # The duration * 1

        # Set weight: bitrateSwitchPenalty
        # qoeInfo.weights.bitrateSwitchPenalty = 0.02
        qoeInfo.weights['bitrateSwitchPenalty'] = Env_Config.smooth_penalty 

        # Set weight: rebufferPenalty
        if not maxBitrateMbps:
            qoeInfo.weights['rebufferPenalty'] = Env_Config.rebuf_penalty
        else:
            qoeInfo.weights['rebufferPenalty'] = Env_Config.rebuf_penalty

        # Set weight: latencyPenalty
        qoeInfo.weights['latencyPenalty'] = Env_Config.long_delay_penalty_new*Env_Config.chunk_in_seg
        # qoeInfo.weights[latencyPenalty].append({threshold: 1.1, penalty: 0.5*long_delay_penalty_new*chunk_in_seg})
        # qoeInfo.weights[latencyPenalty].append({threshold: float('inf'), penalty: long_delay_penalty_new*chunk_in_seg})

        # Set weight: playbackSpeedPenalty
        if not minBitrateMbps:
            qoeInfo.weights['playbackSpeedPenalty'] = Env_Config.unnormal_playing_penalty
        else: 
            qoeInfo.weights['playbackSpeedPenalty'] = Env_Config.unnormal_playing_penalty

        qoeInfo.weights['speedChangePenalty'] = Env_Config.speed_smooth_penalty


        return qoeInfo


class QoeInfo():

    def __init__(self):
        self.qoetype = None
        self.lastBitrate = None
        self.lastSpeed = None

        self.weights = dict()
        self.weights['bitrateReward'] = None
        self.weights['bitrateSwitchPenalty'] = None
        self.weights['rebufferPenalty'] = None
        self.weights['latencyPenalty'] = None
        self.weights['playbackSpeedPenalty'] = None
        self.weights['speedChangePenalty'] = None

        self.bitrateWSum = 0
        self.bitrateSwitchWSum = 0     # kbps
        self.rebufferWSum = 0          # seconds
        self.latencyWSum = 0           # seconds
        self.playbackSpeedWSum = 0     # e.g. 0.95, 1.0, 1.05
        self.speedChangeWSum = 0
        
        self.totalQoe = 0
    def getTotal(self):
        return self.totalQoe

    def updateTotal(self):
        self.totalQoe = self.bitrateWSum - self.bitrateSwitchWSum - self.rebufferWSum - \
                        self.latencyWSum - self.playbackSpeedWSum - self.speedChangeWSum