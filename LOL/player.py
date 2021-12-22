import numpy as np
from config import Env_Config, Config
from random import Random

class Live_Player(object):
    def __init__(self, throughput_trace, time_trace, trace_name, random_seed=Config.random_seed):
        self.myRandom = Random(random_seed)
        self.ampRandom = Random(random_seed+1)
        self.throughput_trace = throughput_trace
        self.time_trace = time_trace
        self.trace_name = trace_name

        self.playing_time = 0.0
        self.time_idx = self.myRandom.randint(1, len(self.time_trace)-1)
        self.last_trace_time = self.time_trace[self.time_idx-1] * Env_Config.ms_in_s   # in ms

        self.seg_duration = Env_Config.seg_duration
        self.chunk_duration = Env_Config.chunk_duration

        self.buffer = 0.0   # ms
        self.state = 0  # 0: start up.  1: traceing. 2: rebuffering
        self.start_up_ssh = Env_Config.start_up_ssh
        self.freezing_tol = Env_Config.freezing_tol
        print('player initial finish')

    def fetch(self, quality, next_chunk_set, seg_idx, chunk_idx, take_action, num_chunk, playing_speed):
        # Action initialization
        start_state = self.state
        chunk_size = next_chunk_set[quality] # in Kbits not KBytes
        chunk_start_time = seg_idx * self.seg_duration + chunk_idx * self.chunk_duration
        # as mpd is based on prediction, there is noise
        chunk_size = self.myRandom.uniform(Env_Config.chunk_random_ratio_low*chunk_size, Env_Config.chunk_random_ratio_high*chunk_size)
        # print(chunk_size)
        chunk_sent = 0.0    # in Kbits
        downloading_fraction = 0.0  # in ms
        freezing_fraction = 0.0 # in ms
        time_out = 0
        rtt = 0.0
        # Handle RTT 
        if take_action:
            rtt = self.myRandom.uniform(Env_Config.rtt_low, Env_Config.rtt_high)  # in ms
            duration = self.time_trace[self.time_idx] * Env_Config.ms_in_s - self.last_trace_time  # in ms
            if duration > rtt:
                self.last_trace_time += rtt
            else:
                temp_rtt = rtt
                while duration < temp_rtt:
                    self.last_trace_time = self.time_trace[self.time_idx] * Env_Config.ms_in_s
                    self.time_idx += 1
                    if self.time_idx >= len(self.time_trace):
                        self.time_idx = 1
                        self.last_trace_time = 0.0
                    temp_rtt -= duration
                    duration = self.time_trace[self.time_idx] * Env_Config.ms_in_s - self.last_trace_time
                self.last_trace_time += temp_rtt
                assert self.last_trace_time < self.time_trace[self.time_idx] * Env_Config.ms_in_s
            downloading_fraction += rtt
            # assert self.state == 1 or self.state == 0
            # Check whether during startup
            if self.state == 1:
                self.playing_time += np.minimum(self.buffer, playing_speed*rtt)         # modified based on playing speed, adjusted, * speed
                freezing_fraction += np.maximum(rtt - self.buffer/playing_speed, 0.0)   # modified based on playing speed, real time, /speed
                self.buffer = np.maximum(0.0, self.buffer - playing_speed*rtt)          # modified based on playing speed, adjusted, * speed
                if freezing_fraction > 0.0:
                    self.state = 2
            elif self.state == 0:
                freezing_fraction += rtt    # in ms
            else:
                # It is possible to enter state 2 if player wait and make freeze
                freezing_fraction += rtt
        # Chunk downloading
        while True:
            throughput = self.throughput_trace[self.time_idx]   # in Mbps or Kbpms
            duration = self.time_trace[self.time_idx] * Env_Config.ms_in_s - self.last_trace_time      # in ms
            deliverable_size = throughput * duration * Env_Config.packet_payload_portion   # in Kbits      
            # Will also check whether freezing time exceeds the TOL
            if deliverable_size + chunk_sent > chunk_size:
                fraction = (chunk_size - chunk_sent) / (throughput * Env_Config.packet_payload_portion)    # in ms, real time
                if self.state == 1:
                    assert freezing_fraction == 0.0
                    temp_freezing = np.maximum(fraction - self.buffer/playing_speed, 0.0)       # modified based on playing speed
                    if temp_freezing > self.freezing_tol:
                        # should not happen
                        time_out = 1
                        self.last_trace_time += self.buffer/playing_speed + self.freezing_tol
                        downloading_fraction += self.buffer/playing_speed + self.freezing_tol
                        self.playing_time += self.buffer
                        chunk_sent += (self.freezing_tol + self.buffer/playing_speed) * throughput * Env_Config.packet_payload_portion   # in Kbits  
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
                        chunk_sent += (self.freezing_tol - freezing_fraction) * throughput * Env_Config.packet_payload_portion # in Kbits
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
                    assert self.buffer < self.start_up_ssh
                    downloading_fraction += fraction
                    self.buffer += self.chunk_duration * num_chunk
                    freezing_fraction += fraction
                    self.last_trace_time += fraction
                    if self.buffer >= self.start_up_ssh:
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
                    chunk_sent += (self.freezing_tol + self.buffer/playing_speed) * throughput * Env_Config.packet_payload_portion   # in Kbits
                    assert chunk_sent < chunk_size
                    return chunk_sent, downloading_fraction, freezing_fraction, time_out, start_state, rtt
                chunk_sent += duration * throughput * Env_Config.packet_payload_portion    # in Kbits
                downloading_fraction += duration    # in ms
                self.last_trace_time = self.time_trace[self.time_idx] * Env_Config.ms_in_s # in ms
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
                    chunk_sent += (self.freezing_tol - freezing_fraction) * throughput * Env_Config.packet_payload_portion # in Kbits
                    freezing_fraction = self.freezing_tol
                    # Download is not finished, chunk_size is not the entire chunk
                    assert chunk_sent < chunk_size
                    return chunk_sent, downloading_fraction, freezing_fraction, time_out, start_state, rtt

                freezing_fraction += duration   # in ms
                chunk_sent += duration * throughput * Env_Config.packet_payload_portion    # in kbits
                downloading_fraction += duration    # in ms
                self.last_trace_time = self.time_trace[self.time_idx] * Env_Config.ms_in_s # in ms
                self.time_idx += 1
                if self.time_idx >= len(self.time_trace):
                    self.time_idx = 1
                    self.last_trace_time = 0.0  # in ms
            # Startup
            else:
                assert self.buffer < self.start_up_ssh
                chunk_sent += duration * throughput * Env_Config.packet_payload_portion
                downloading_fraction += duration
                self.last_trace_time = self.time_trace[self.time_idx] * Env_Config.ms_in_s # in ms
                self.time_idx += 1
                if self.time_idx >= len(self.time_trace):
                    self.time_idx = 1
                    self.last_trace_time = 0.0  # in ms
                freezing_fraction += duration
        return chunk_size, downloading_fraction, freezing_fraction, time_out, start_state, rtt


    def sync_playing_timeout(self, sync_time):
        self.buffer = 0.0
        self.state = 0
        self.playing_time = sync_time

    def playing_time_back(self, index_gap):
        assert self.buffer == 0.0
        self.playing_time -= index_gap*self.chunk_duration
        if not np.round(self.playing_time, 1)%self.seg_duration == 0.0:
            print(self.playing_time)
        assert np.round(self.playing_time, 1)%self.seg_duration == 0.0

    def skip_with_time(self, jump_time, encoder_head_time):
        num_skip = Env_Config.skip_segs
        if np.round(self.buffer, 2) >= np.round(jump_time, 2):
            assert self.state == 1 or self.state == 0
            self.buffer -= np.round(jump_time, 1)
            self.playing_time += np.round(jump_time, 1) + num_skip * self.seg_duration
            assert np.round(self.playing_time + self.buffer, 1) == np.round(encoder_head_time, 1)
        else:
            print("in player: ", self.buffer, jump_time, self.playing_time)
            assert 0 == 1
        return

    def repeat(self):
        num_repeat = Env_Config.repeat_segs
        if self.playing_time <= num_repeat*self.seg_duration:
            return
        self.playing_time -= num_repeat*self.seg_duration
        self.buffer += num_repeat*self.seg_duration
        if self.buffer >= self.start_up_ssh:
            self.state = 1

    def adjust_start_up_ssh(self, new_start_up_ssh):
        self.start_up_ssh = new_start_up_ssh
        return

    def wait(self, wait_time, playing_speed = 1.0):
        freezing = np.maximum(wait_time * playing_speed - self.buffer, 0.0)
        self.playing_time += np.minimum(self.buffer, wait_time * playing_speed)
        self.buffer = np.maximum(0.0, self.buffer - wait_time * playing_speed)
        past_wait_time = 0.0    # in ms
        while  True:
            duration = self.time_trace[self.time_idx] * Env_Config.ms_in_s - self.last_trace_time
            if past_wait_time + duration > wait_time:
                self.last_trace_time += wait_time - past_wait_time
                break
            past_wait_time += duration
            self.last_trace_time += duration
            self.time_idx += 1
            if self.time_idx >= len(self.time_trace):
                self.time_idx = 1
                self.last_trace_time = 0.0
        if freezing > 0.0:
            self.state = 2
        return freezing

    def check_resync(self, server_time):
        # sync = 0
        # if server_time - self.playing_time > self.latency_tol:
        #   sync = 1
        # return sync
        pass

    def throughput_trace_amplifyer_mean(self, trace):
        s = self.ampRandom.randint(0,2)
        c_id = self.ampRandom.randint(Env_Config.range_low, Env_Config.range_high)
        # print(s, c_id)
        curr_mean = np.mean(trace[:c_id])
        new_trace = []
        for i in range(len(trace)):
            if i == c_id:
                s = self.ampRandom.randint(0, 2)
                p_id = c_id
                c_id += self.ampRandom.randint(Env_Config.range_low, Env_Config.range_high)
                curr_mean = np.mean(trace[p_id:c_id])
            if s == 0:
                r = 1
            elif s == 1:
                r = 0.2
            elif s == 2:
                r = 1.2
            new_trace.append((trace[i]-curr_mean)*r + curr_mean)
        # print(new_trace)
        return new_trace

    def reset(self, throughput_trace, time_trace, trace_name, testing=False, bw_amplify=False):
        self.playing_time = 0.0
        if bw_amplify:
            self.throughput_trace = self.throughput_trace_amplifyer_mean(throughput_trace)
        else:
            self.throughput_trace = throughput_trace
        self.time_trace = time_trace
        self.trace_name = trace_name
        if testing:
            self.time_idx = 1
        else:
            self.time_idx = self.myRandom.randint(1, len(self.time_trace)-1)
        self.last_trace_time = self.time_trace[self.time_idx-1] * Env_Config.ms_in_s # in ms
        self.buffer = 0.0   # ms
        self.state = 0  # 0: start up.  1: traceing. 2: rebuffering

    def get_time_idx(self):
        return self.time_idx

    def get_tp_trace(self):
        return self.throughput_trace

    def get_time_trace(self):
        return self.time_trace

    def get_trace_name(self):
        return self.trace_name

    def get_display_time(self):
        return self.playing_time

    def get_state(self):
        return self.state

    def get_buffer(self):
        return self.buffer

    def get_freezing_tol(self):
        return self.freezing_tol

