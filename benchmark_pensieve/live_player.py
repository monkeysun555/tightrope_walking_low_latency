import numpy as np
from random import Random

RANDOM_SEED = 13
BITRATE = [300.0, 500.0, 1000.0, 2000.0, 3000.0, 6000.0]
# BITRATE = [500.0, 2000.0, 5000.0, 8000.0, 16000.0]    # 5 actions
PACKET_PAYLOAD_PORTION = 0.973  # 1460/1500

RTT_LOW = 30.0
RTT_HIGH = 40.0 
CHUNK_RANDOM_RATIO_LOW = 0.95
CHUNK_RANDOM_RATIO_HIGH = 1.05

# SEG_DURATION = 2000.0
# FRAG_DURATION = 1000.0
# CHUNK_DURATION = 500.0
# START_UP_TH = 1000.0
# FREEZING_TOL = 3000.0 # in ms
# CHUNK_IN_SEG = int(SEG_DURATION/CHUNK_DURATION)   # 4
# CHUNK_IN_FRAG = int(FRAG_DURATION/FRAG_DURATION)  # 2
# FRAG_IN_SEG = int(SEG_DURATION/FRAG_DURATION)     # 2 or 5

MS_IN_S = 1000.0    # in ms
KB_IN_MB = 1000.0   # in ms

class Live_Player(object):
    def __init__(self, time_traces, throughput_traces, name_traces, seg_duration, chunk_duration, start_up_th, freezing_tol, testing=False, randomSeed = RANDOM_SEED):
        self.myRandom = Random(randomSeed)

        self.time_traces = time_traces
        self.throughput_traces = throughput_traces
        self.name_traces = name_traces

        if testing:
            self.trace_idx = -1
        else:
            self.trace_idx = np.random.randint(len(self.throughput_traces))
        self.throughput_trace = self.throughput_traces[self.trace_idx]
        self.time_trace = self.time_traces[self.trace_idx]
        self.name_trace = self.name_traces[self.trace_idx]
        self.playing_time = 0.0
        self.time_idx = np.random.randint(1,len(self.time_trace)-1)
        self.last_trace_time = self.time_trace[self.time_idx-1] * MS_IN_S   # in ms

        self.seg_duration = seg_duration
        # self.frag_duration = frag_duration
        self.chunk_duration = chunk_duration

        # self.frag_in_seg = seg_duration/frag_duration
        # self.chunk_in_frag = frag_duration/chunk_duration
        # self.chunk_in_seg = seg_duration/chunk_duration

        self.buffer = 0.0   # ms
        self.state = 0  # 0: start up.  1: traceing. 2: rebuffering
        self.start_up_th = start_up_th
        self.freezing_tol = freezing_tol
        # self.latency_tol = latency_tol
        print('player initial finish')

    def fetch(self, quality, next_chunk_set, seg_idx, chunk_idx, take_action, num_chunk, playing_speed = 1.0):
        # Action initialization
        start_state = self.state
        chunk_size = next_chunk_set[quality] # in Kbits not KBytes
        chunk_start_time = seg_idx * self.seg_duration + chunk_idx * self.chunk_duration
        # as mpd is based on prediction, there is noise
        chunk_size = self.myRandom.uniform(CHUNK_RANDOM_RATIO_LOW*chunk_size, CHUNK_RANDOM_RATIO_HIGH*chunk_size)
        chunk_sent = 0.0    # in Kbits
        downloading_fraction = 0.0  # in ms
        freezing_fraction = 0.0 # in ms
        time_out = 0
        rtt = 0.0
        # Handle RTT 
        if take_action:
            rtt = self.myRandom.uniform(RTT_LOW, RTT_HIGH)  # in ms
            duration = self.time_trace[self.time_idx] * MS_IN_S - self.last_trace_time  # in ms
            if duration > rtt:
                self.last_trace_time += rtt
            else:
                temp_rtt = rtt
                while duration < temp_rtt:
                    self.last_trace_time = self.time_trace[self.time_idx] * MS_IN_S
                    self.time_idx += 1
                    if self.time_idx >= len(self.time_trace):
                        self.time_idx = 1
                        self.last_trace_time = 0.0
                    temp_rtt -= duration
                    duration = self.time_trace[self.time_idx] * MS_IN_S - self.last_trace_time
                self.last_trace_time += temp_rtt
                assert self.last_trace_time < self.time_trace[self.time_idx] * MS_IN_S
            downloading_fraction += rtt
            # Check whether during startup
            if self.state == 1:
                self.playing_time += np.minimum(self.buffer, playing_speed*rtt)         # modified based on playing speed, adjusted, * speed
                freezing_fraction += np.maximum(rtt - self.buffer/playing_speed, 0.0)   # modified based on playing speed, real time, /speed
                self.buffer = np.maximum(0.0, self.buffer - playing_speed*rtt)          # modified based on playing speed, adjusted, * speed
                # chech whether enter freezing
                if freezing_fraction > 0.0:
                    self.state = 2
            else:
                freezing_fraction += rtt    # in ms
        # Chunk downloading
        while True:
            throughput = self.throughput_trace[self.time_idx]   # in Mbps or Kbpms
            duration = self.time_trace[self.time_idx] * MS_IN_S - self.last_trace_time      # in ms
            deliverable_size = throughput * duration * PACKET_PAYLOAD_PORTION   # in Kbits      
            # Will also check whether freezing time exceeds the TOL
            if deliverable_size + chunk_sent > chunk_size:
                fraction = (chunk_size - chunk_sent) / (throughput * PACKET_PAYLOAD_PORTION)    # in ms, real time
                if self.state == 1:
                    assert freezing_fraction == 0.0
                    temp_freezing = np.maximum(fraction - self.buffer/playing_speed, 0.0)       # modified based on playing speed
                    if temp_freezing > self.freezing_tol:
                        # should not happen
                        time_out = 1
                        self.last_trace_time += self.buffer/playing_speed + self.freezing_tol
                        downloading_fraction += self.buffer/playing_speed + self.freezing_tol
                        self.playing_time += self.buffer
                        chunk_sent += (self.freezing_tol + self.buffer/playing_speed) * throughput * PACKET_PAYLOAD_PORTION # in Kbits  
                        self.state = 0
                        self.buffer = 0.0
                        assert chunk_sent < chunk_size
                        return chunk_sent, downloading_fraction, freezing_fraction, time_out, start_state, rtt

                    downloading_fraction += fraction
                    self.last_trace_time += fraction
                    freezing_fraction += np.maximum(fraction - self.buffer/playing_speed, 0.0)  # modified based on playing speed 
                    self.playing_time += np.minimum(self.buffer, playing_speed*fraction)        # modified based on playing speed 
                    self.buffer = np.maximum(self.buffer - playing_speed*fraction, 0.0)         # modified based on playing speed 
                    if np.round(self.playing_time + self.buffer, 2) == np.round(chunk_start_time, 2):
                        self.buffer += self.chunk_duration * num_chunk
                    else:
                        # Should not happen in normal case, this is constrain for training
                        self.buffer = self.chunk_duration * num_chunk
                        self.playing_time = chunk_start_time
                    break
                # Freezing
                elif self.state == 2:
                    assert self.buffer == 0.0
                    if freezing_fraction + fraction > self.freezing_tol:
                        time_out = 1
                        self.last_trace_time += self.freezing_tol - freezing_fraction
                        downloading_fraction += self.freezing_tol - freezing_fraction
                        chunk_sent += (self.freezing_tol - freezing_fraction) * throughput * PACKET_PAYLOAD_PORTION # in Kbits
                        freezing_fraction = self.freezing_tol
                        self.state = 0
                        assert chunk_sent < chunk_size
                        return chunk_sent, downloading_fraction, freezing_fraction, time_out, start_state, rtt
                    freezing_fraction += fraction
                    self.last_trace_time += fraction
                    downloading_fraction += fraction
                    self.buffer += self.chunk_duration * num_chunk
                    self.playing_time = chunk_start_time
                    self.state = 1
                    break

                else:
                    assert self.buffer < self.start_up_th
                    # if freezing_fraction + fraction > self.freezing_tol:
                    #   self.buffer = 0.0
                    #   time_out = 1
                    #   self.last_trace_time += self.freezing_tol - freezing_fraction   # in ms
                    #   downloading_fraction += self.freezing_tol - freezing_fraction
                    #   chunk_sent += (self.freezing_tol - freezing_fraction) * throughput * PACKET_PAYLOAD_PORTION # in Kbits
                    #   freezing_fraction = self.freezing_tol
                    #   # Download is not finished, chunk_size is not the entire chunk
                    #   # print()
                    #   assert chunk_sent < chunk_size
                    #   return chunk_sent, downloading_fraction, freezing_fraction, time_out, start_state
                    downloading_fraction += fraction
                    self.buffer += self.chunk_duration * num_chunk
                    freezing_fraction += fraction
                    self.last_trace_time += fraction
                    if self.buffer >= self.start_up_th:
                        # Because it might happen after one long freezing (not exceed freezing tol)
                        # And resync, enter initial phase
                        buffer_end_time = chunk_start_time + self.chunk_duration * num_chunk
                        self.playing_time = buffer_end_time - self.buffer
                        # print(buffer_end_time, self.buffer)
                        self.state = 1
                    break

            # One chunk downloading does not finish
            # traceing
            if self.state == 1:
                assert freezing_fraction == 0.0
                temp_freezing = np.maximum(duration - self.buffer/playing_speed, 0.0)       # modified based on playing speed
                self.playing_time += np.minimum(self.buffer, playing_speed*duration)        # modified based on playing speed
                # Freezing time exceeds tolerence
                if temp_freezing > self.freezing_tol:
                    # should not happen
                    time_out = 1
                    self.last_trace_time += self.freezing_tol + self.buffer/playing_speed
                    downloading_fraction += self.freezing_tol + self.buffer/playing_speed
                    freezing_fraction = self.freezing_tol
                    self.playing_time += self.buffer
                    self.buffer = 0.0
                    # exceed TOL, enter startup, freezing time equals TOL
                    self.state = 0
                    chunk_sent += (self.freezing_tol + self.buffer/playing_speed) * throughput * PACKET_PAYLOAD_PORTION # in Kbits
                    assert chunk_sent < chunk_size
                    return chunk_sent, downloading_fraction, freezing_fraction, time_out, start_state, rtt

                chunk_sent += duration * throughput * PACKET_PAYLOAD_PORTION    # in Kbits
                downloading_fraction += duration    # in ms
                self.last_trace_time = self.time_trace[self.time_idx] * MS_IN_S # in ms
                self.time_idx += 1
                if self.time_idx >= len(self.time_trace):
                    self.time_idx = 1
                    self.last_trace_time = 0.0  # in ms
                self.buffer = np.maximum(self.buffer - playing_speed*duration, 0.0)         # modified based on playing speed
                # update buffer and state
                if temp_freezing > 0:
                    # enter freezing
                    self.state = 2
                    assert self.buffer == 0.0
                    freezing_fraction += temp_freezing

            # Freezing during trace
            elif self.state == 2:
                assert self.buffer == 0.0
                if duration + freezing_fraction > self.freezing_tol:
                    time_out = 1
                    self.last_trace_time += self.freezing_tol - freezing_fraction   # in ms
                    self.state = 0
                    downloading_fraction += self.freezing_tol - freezing_fraction
                    chunk_sent += (self.freezing_tol - freezing_fraction) * throughput * PACKET_PAYLOAD_PORTION # in Kbits
                    freezing_fraction = self.freezing_tol
                    # Download is not finished, chunk_size is not the entire chunk
                    assert chunk_sent < chunk_size
                    return chunk_sent, downloading_fraction, freezing_fraction, time_out, start_state, rtt

                freezing_fraction += duration   # in ms
                chunk_sent += duration * throughput * PACKET_PAYLOAD_PORTION    # in kbits
                downloading_fraction += duration    # in ms
                self.last_trace_time = self.time_trace[self.time_idx] * MS_IN_S # in ms
                self.time_idx += 1
                if self.time_idx >= len(self.time_trace):
                    self.time_idx = 1
                    self.last_trace_time = 0.0  # in ms
            # Startup
            else:
                assert self.buffer < self.start_up_th
                chunk_sent += duration * throughput * PACKET_PAYLOAD_PORTION
                downloading_fraction += duration
                self.last_trace_time = self.time_trace[self.time_idx] * MS_IN_S # in ms
                self.time_idx += 1
                if self.time_idx >= len(self.time_trace):
                    self.time_idx = 1
                    self.last_trace_time = 0.0  # in ms
                freezing_fraction += duration

        return chunk_size, downloading_fraction, freezing_fraction, time_out, start_state, rtt

    def sync_playing(self, sync_time):
        self.buffer = 0
        self.state = 0
        self.playing_time = sync_time

    def playing_time_back(self, index_gap):
        assert self.buffer == 0.0
        self.playing_time -= index_gap*self.chunk_duration
        if not np.round(self.playing_time, 1)%self.seg_duration == 0.0:
            print(self.playing_time)
        assert np.round(self.playing_time, 1)%self.seg_duration == 0.0


    def adjust_start_up_th(self, new_start_up_th):
        self.start_up_th = new_start_up_th
        return

    def wait(self, wait_time):
        # If live server does not have any available chunks, need to wait
        assert self.buffer > wait_time
        self.buffer -= wait_time
        self.playing_time += wait_time
        past_wait_time = 0.0    # in ms
        while  True:
            duration = self.time_trace[self.time_idx] * MS_IN_S - self.last_trace_time
            if past_wait_time + duration > wait_time:
                self.last_trace_time += wait_time - past_wait_time
                break
            past_wait_time += duration
            self.last_trace_time += duration
            self.time_idx += 1
            if self.time_idx >= len(self.time_trace):
                self.time_idx = 1
                self.last_trace_time = 0.0
        return

    # def check_resync(self, server_time):
    #   sync = 0
    #   if server_time - self.playing_time > self.latency_tol:
    #       sync = 1
    #   return sync

    def reset(self, testing=False):
        self.playing_time = 0.0
        if testing:
            self.trace_idx += 1
            self.time_idx = 1
        else:
            self.trace_idx = np.random.randint(len(self.throughput_traces))
            self.time_idx = np.random.randint(1,len(self.time_trace))

        self.throughput_trace = self.throughput_traces[self.trace_idx]
        self.time_trace = self.time_traces[self.trace_idx]           
        self.name_trace = self.name_traces[self.trace_idx]
        self.last_trace_time = self.time_trace[self.time_idx-1] * MS_IN_S # in ms
        self.buffer = 0.0   # ms
        self.state = 0  # 0: start up.  1: traceing. 2: rebuffering

    def get_player_trace_info(self):
        return self.throughput_trace, self.time_trace, self.name_trace, self.time_idx

    def get_display_time(self):
        return self.playing_time

    def get_state(self):
        return self.state

    def get_buffer(self):
        return self.buffer

