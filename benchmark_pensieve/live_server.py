import numpy as np
from random import Random

SEG_DURATION = 1000.0
# FRAG_DURATION = 1000.0
CHUNK_DURATION = 200.0
SERVER_START_UP_TH = 2000.0             # <========= TO BE MODIFIED. TEST WITH DIFFERENT VALUES

# CHUNK_IN_SEG = int(SEG_DURATION/CHUNK_DURATION)       # 4
# CHUNK_IN_FRAG = int(FRAG_DURATION/FRAG_DURATION)  # 2
# FRAG_IN_SEG = int(SEG_DURATION/FRAG_DURATION)     # 2 or 5
# ADD_DELAY = 3000.0

MS_IN_S = 1000.0
KB_IN_MB = 1000.0
# New bitrate setting, 6 actions, correspongding to 240p, 360p, 480p, 720p, 1080p and 1440p(2k)
BITRATE = [300.0, 500.0, 1000.0, 2000.0, 3000.0, 6000.0]
# BITRATE = [500.0, 2000.0, 5000.0, 8000.0, 16000.0]    # 5 actions

BITRATE_LOW_NOISE = 0.7
BITRATE_HIGH_NOISE = 1.3
RATIO_LOW_2 = 2.0               # This is the lowest ratio between first chunk and the sum of all others
RATIO_HIGH_2 = 10.0         # This is the highest ratio between first chunk and the sum of all others
RATIO_LOW_5 = 0.75              # This is the lowest ratio between first chunk and the sum of all others
RATIO_HIGH_5 = 1.0          # This is the highest ratio between first chunk and the sum of all others
EST_LOW_NOISE = 0.98
EST_HIGH_NOISE = 1.02

########## Added for benchmark MM pensieve 
SERVER_INIT_LAT_LOW = 2
SERVER_INIT_LAT_HIGH = 5
##########################################


class Live_Server(object):
    def __init__(self, seg_duration, chunk_duration, random_seed=11):
        self.myRandom = Random(random_seed)
        self.latency_random = Random(random_seed+1)
        self.seg_duration = seg_duration
        # self.frag_duration = frag_duration
        self.chunk_duration = chunk_duration
        # self.frag_in_seg = seg_duration/frag_duration
        # self.chunk_in_frag = frag_duration/chunk_duration
        self.chunk_in_seg = seg_duration/chunk_duration
        self.next_delivery = []
        
        # self.time = initial_time
        # self.current_seg_idx = -1 # For initial
        # self.current_chunk_idx = 0
        # self.chunks = []  # 1 for initial chunk, 0 for following chunks
        # self.current_seg_size = [[] for i in range(len(BITRATE))]
        # self.encoding_update(0.0, self.time)
        # self.generate_next_delivery()

    def generate_next_delivery(self):
        deliver_chunks = []
        deliver_chunks.append(self.chunks.pop(0))
        # Should still do chunk based streaming if there is a segment.
        # Otherwise there is no benefit from chunk streaming while buffer is low
        # deliver_end = 0
        # for i in range(len(self.chunks)):
        #   # Check how many chunks can be deliveryed together
        #   if not self.chunks[i][0] == deliver_chunks[-1][0]:
        #       break
        #   deliver_end += 1
        # deliver_chunks.extend(self.chunks[:deliver_end])
        # del self.chunks[:deliver_end]
        self.next_delivery.extend(deliver_chunks[0][:2])
        self.next_delivery.append(deliver_chunks[-1][1])
        delivery_sizes = []
        for i in range(len(BITRATE)):
            delivery_sizes.append(np.sum([chunk[2][i] for chunk in deliver_chunks]))
        self.next_delivery.append(delivery_sizes)
        
    def encoding_update(self, starting_time, end_time):
        temp_time = starting_time
        while True:
            next_time = (int(temp_time/self.chunk_duration) + 1) * self.chunk_duration
            if next_time > end_time:
                break
            # Generate chunks and insert to encoding buffer
            temp_time = next_time
            if next_time%self.seg_duration == self.chunk_duration:
            # If it is the first chunk in a seg
                self.current_seg_idx += 1
                self.current_chunk_idx = 0
                self.generate_chunk_size()
                self.chunks.append([self.current_seg_idx, self.current_chunk_idx, \
                                    [chunk_size[self.current_chunk_idx] for chunk_size in self.current_seg_size],\
                                    [np.sum(chunk_size) for chunk_size in self.current_seg_size]])  # for 2s segment
            else:
                self.current_chunk_idx += 1
                # print(self.current_chunk_idx, self.current_seg_size)
                self.chunks.append([self.current_seg_idx, self.current_chunk_idx, [chunk_size[self.current_chunk_idx] for chunk_size in self.current_seg_size]])

    def update(self, downloadig_time):
        # update time and encoding buffer
        # Has nothing to do with sync, migrate to player side
        # sync = 0  # sync play
        # missing_count = 0
        # new_heading_time = 0.0
        pre_time = self.time
        self.time += downloadig_time
        self.encoding_update(pre_time, self.time)
        # Generate new delivery for next
        # self.generate_next_delivery()
        return self.time

    

    # chunk size for next/current segment
    def generate_chunk_size(self):
        self.current_seg_size = [[] for i in range(len(BITRATE))]

        # Initial coef, all bitrate share the same coef 
        encoding_coef = np.random.uniform(BITRATE_LOW_NOISE, BITRATE_HIGH_NOISE)
        estimate_seg_size = [x * encoding_coef for x in BITRATE]
        # There is still noise for prediction, all bitrate cannot share the coef exactly same
        seg_size = [np.random.uniform(EST_LOW_NOISE*x, EST_HIGH_NOISE*x) for x in estimate_seg_size]

        if self.chunk_in_seg == 2:
        # Distribute size for chunks, currently, it should depend on chunk duration (200 or 500)
            ratio = np.random.uniform(RATIO_LOW_2, RATIO_HIGH_2)
            seg_ratio = [np.random.uniform(EST_LOW_NOISE*ratio, EST_HIGH_NOISE*ratio) for x in range(len(BITRATE))]
            for i in range(len(seg_ratio)):
                temp_ratio = seg_ratio[i]
                temp_aux_chunk_size = seg_size[i]/(1+temp_ratio)
                temp_ini_chunk_size = seg_size[i] - temp_aux_chunk_size
                self.current_seg_size[i].extend((temp_ini_chunk_size, temp_aux_chunk_size))
        # if 200ms, needs to be modified, not working
        elif self.chunk_in_seg == 5:
            # assert 1 == 0
            ratio = np.random.uniform(RATIO_LOW_5, RATIO_HIGH_5)
            seg_ratio = [np.random.uniform(EST_LOW_NOISE*ratio, EST_HIGH_NOISE*ratio) for x in range(len(BITRATE))]
            for i in range(len(seg_ratio)):
                temp_ratio = seg_ratio[i]
                temp_ini_chunk_size = seg_size[i] * temp_ratio / (1 + temp_ratio)
                temp_aux_chunk_size = (seg_size[i] - temp_ini_chunk_size) / (self.chunk_in_seg - 1)
                temp_chunks_size = [temp_ini_chunk_size]
                temp_chunks_size.extend([temp_aux_chunk_size for _ in range(int(self.chunk_in_seg) - 1)])
                self.current_seg_size[i].extend(temp_chunks_size)


    def wait(self):
        next_available_time = (int(self.time/self.chunk_duration) + 1) * self.chunk_duration
        self.encoding_update(self.time, next_available_time)
        assert len(self.chunks) == 1
        time_interval = next_available_time - self.time
        self.time = next_available_time
        return time_interval 

    def timeout_encoding_buffer(self):
        temp_seg_index = self.next_delivery[0]
        index_makeup = self.next_delivery[1]
        idx_timeout = index_makeup
        while index_makeup >= 0:
            if index_makeup == 0:
                self.chunks.insert(0, [temp_seg_index, index_makeup, \
                [chunk_size[index_makeup] for chunk_size in self.current_seg_size], \
                [np.sum(chunk_size) for chunk_size in self.current_seg_size]])  # trick: using current encoding setting
            else:
                self.chunks.insert(0, [temp_seg_index, index_makeup,
                [chunk_size[index_makeup] for chunk_size in self.current_seg_size]])  
            index_makeup -= 1
            # print("makeup")
            # print(self.chunks[0])
        return idx_timeout

    def check_take_action(self):
        assert len(self.chunks) >= 1
        if self.chunks[0][1] == 0:
            return True
        else:
            return False

    def reset(self, testing=False):
        if testing:
            self.time = (self.latency_random.randint(SERVER_INIT_LAT_LOW, \
                    SERVER_INIT_LAT_HIGH)+self.latency_random.random())\
                    *self.seg_duration 
        else:
            self.time = (self.myRandom.randint(SERVER_INIT_LAT_LOW, SERVER_INIT_LAT_HIGH)+np.random.random())*self.seg_duration 
        self.current_seg_idx = -1
        self.current_chunk_idx = 0
        self.chunks = []    # 1 for initial chunk, 0 for following chunks
        self.current_seg_size = [[] for i in range(len(BITRATE))]
        self.encoding_update(0.0, self.time)
        del self.next_delivery[:]
        self.generate_next_delivery()
        
    def get_time(self):
        return self.time

    def get_next_delivery(self):
        return self.next_delivery

    def check_chunks_empty(self):
        if len(self.chunks) == 0:
            return True
        else: return False

    def clean_next_delivery(self):
        del self.next_delivery[:]

def main():
    server = Live_Server(seg_duration=SEG_DURATION, chunk_duration=CHUNK_DURATION)
    print(server.chunks, server.time)
    print(server.next_delivery)


if __name__ == '__main__':
    main()