import numpy as np
import math 

LQR_DEBUG = 0
iLQR_SHOW = 0
RTT_LOW = 0.02
SEG_DURATION = 1.0
CHUNK_DURATION = 0.2
CHUNK_IN_SEG = SEG_DURATION/CHUNK_DURATION
DEF_N_STEP = 10
BITRATE = [300.0, 500.0, 1000.0, 2000.0, 3000.0, 6000.0]
SPEED = [0.75, 0.9, 1.0, 1.1, 1.25]
MS_IN_S = 1000.0
KB_IN_MB = 1000.0
MIN_RATE = 10**-8
MAX_RATE = BITRATE[-1]/KB_IN_MB

class iLQR_solver(object):
    def __init__(self):
        # For new traces
        self.w1 = 1
        self.w2 = 1
        self.w3 = 1             # Freeze
        self.w4 = 0.5           # For linear          
        self.w5 = 15           # Speed unnormal
        self.w6 = 15           # Speed change
        self.barrier_1 = 1
        self.barrier_2 = 1
        self.barrier_3 = 1
        self.barrier_4 = 1

        self.delta = 0.2  # 0.2s
        self.n_step = None
        self.predicted_bw = None
        # self.predicted_rtt = predicted_rtt
        self.predicted_rtt = None
        self.n_iteration = 10   # use 10 for opt_mass
        self.Bu = None
        self.b0 = None
        self.l0 = None
        self.pu0 = None
        self.ps0 = None
        self.med_latency = 6

        self.kt_step = 1.
        self.KT_step = 1.
        self.step_size = 0.15
        self.decay = 1.
        self.bw_ratio = 1.0 

    def set_step(self, step=DEF_N_STEP):
        self.n_step = step

    def set_future_bw_rtt(self, opt_trace):
        self.predicted_bw = [np.round(bw, 2) for bw in opt_trace]
        self.predicted_rtt = [RTT_LOW] * self.n_step

    def get_step(self):
        return self.n_step
        
    def reset(self):
        self.bw_record = [BITRATE[0]/KB_IN_MB for x in range(self.n_step)]

    def predict_bw(self):
        combined_tp = self.bw_record.copy().tolist()
        combined_tp.extend([0]*self.n_step)
        for i in range(self.n_step):
            combined_tp[self.n_step + i] = self.harmonic_prediction(combined_tp[i:i+self.n_step])
        return combined_tp[-self.n_step:]

    def harmonic_prediction(self, history):
        return len(history)/(np.sum([1/tp for tp in history]))

    def update_bw_record(self, new_tp):
        self.bw_record = np.roll(self.bw_record, -1, axis=0)
        self.bw_record[-1] = new_tp

    def set_x0(self, tmp_buffer, tmp_latency, tmp_pre_a_1, tmp_pre_a_2):
        self.b0 = np.round(tmp_buffer/MS_IN_S, 2)
        self.l0 = np.round(tmp_latency/MS_IN_S, 2)
        self.pu0 = BITRATE[tmp_pre_a_1]/KB_IN_MB
        self.ps0 = SPEED[tmp_pre_a_2]
        if iLQR_SHOW:
            print("Initial X0 is: ", self.b0, self.l0, self.pu0, self.ps0)
            input()

    def checking(self):
        if math.isnan(self.rates[0]) or math.isnan(self.speeds[0]):
            # input() 
            # self.reset()
            return True

    def nan_index(self):
        rate_idx = 0
        for j in reversed(range(len(BITRATE))):
            if BITRATE[j]/KB_IN_MB <= self.predicted_bw[0]:
                rate_idx = j
                break
        return rate_idx

    def set_predicted_bw_rtt(self):
        predicted_bw = self.predict_bw()
        assert len(self.bw_record) == self.n_step

        # print("predicted bw: ", predicted_bw)
        self.predicted_bw = [np.round(bw, 2) for bw in predicted_bw]
        self.predicted_rtt = [RTT_LOW] * self.n_step
        if iLQR_SHOW:
            print("iLQR p_bw: ", self.predicted_bw)
            print("iLQR p_rtt: ", self.predicted_rtt)

    def set_bu(self, bu):
        self.Bu = bu/MS_IN_S + SEG_DURATION
        if iLQR_SHOW:
            print("iLQR buffer upperbound is: ", self.Bu)

    def set_initial_rates(self):
        # self.rates = [max(min(self.predicted_bw[0], BITRATE[-1]/KB_IN_MB), BITRATE[0]/KB_IN_MB)] * self.n_step
        self.rates = [max(min(self.predicted_bw[0]*0.9, BITRATE[-1]/KB_IN_MB*0.9), BITRATE[0]/KB_IN_MB*1.1)] * self.n_step
        # self.rates = [self.pu0] * self.n_step
        # self.speeds = [self.ps0] * self.n_step
        self.speeds = [1] * self.n_step
        self.states = []
        self.states.append([self.b0, self.l0, self.pu0, self.ps0])

    def generate_initial_x(self):
        self.set_initial_rates()
        for r_idx in range(len(self.rates)):
            x = self.states[r_idx]
            u = self.rates[r_idx]
            s = self.speeds[r_idx]
            rtt = self.predicted_rtt[r_idx]
            bw = self.predicted_bw[r_idx]
            new_b, new_l = self.sim_fetch(x[0], x[1], u, s, rtt, bw)
            new_x = [new_b, new_l, u, s]
            # if r_idx < len(self.rates)-1:
            self.states.append(new_x)
        if iLQR_SHOW:
            print("initial iLQR rates are: ", self.rates)
            print("initial iLQR speed are: ", self.speeds)
            print("initial iLQR states are: ", self.states)

    def update_matrix(self, step_i):
        curr_state = self.states[step_i]
        bw = self.predicted_bw[step_i]
        rtt = self.predicted_rtt[step_i]
        b = curr_state[0]
        l = curr_state[1]
        u_p = curr_state[2]
        s_p = curr_state[3]
        u = self.rates[step_i]
        s = self.speeds[step_i]


        f_1 = b-s*(u/bw+rtt) + (CHUNK_IN_SEG-1)*self.delta        # \bar b
        f_2 = b-s*(u/bw+rtt) + CHUNK_IN_SEG*self.delta
        f_3 = b-s*(u/bw+rtt) + CHUNK_IN_SEG*self.delta-self.Bu

        f_5 = u-0.15
        f_6 = u-6.3

        f_7 = s-0.6
        f_8 = s-1.4
        
        if LQR_DEBUG:
            print("f1 is: ", f_1)
            print("f2 is: ", f_2)
            print("b: ", b)
            print("l: ", l)
            print("u: ", u)
            print("s: ", s)
            print("rtt: ", rtt)
            print("delta: ", self.delta)
            print("bu: ", self.Bu)
            print("f3 is: ", f_3)
            # input()

        approx_e_f1 = np.round(np.e**(50*f_1), 4)
        approx_e_f3 = np.round(np.e**(50*f_3), 4)
        approx_e_f5 = np.round(np.e**(-50*f_5), 4)
        approx_e_f6 = np.round(np.e**(50*f_6), 4)
        approx_e_f7 = np.round(np.e**(-50*f_7), 4)
        approx_e_f8 = np.round(np.e**(50*f_8), 4) 

        t_freeze = -(1/(approx_e_f1+1)*f_1/s)
        delta_tf_b = -(-50*approx_e_f1/(approx_e_f1+1)**2*f_1/s + 1/s/(approx_e_f1+1))
        delta_tf_u = delta_tf_b*(-s/bw)
        delta_tf_s = -(-50*-(u/bw+rtt)*approx_e_f1/(approx_e_f1+1)**2*f_1/s + \
                    (-(u/bw+rtt)*s-f_1)/s**2/(approx_e_f1+1))

        t_display = approx_e_f1/(approx_e_f1+1)*(u/bw+rtt) + 1/(approx_e_f1+1)*b/s
        delta_td_b = 50*approx_e_f1/(approx_e_f1+1)**2*(u/bw+rtt) + \
                     -50*approx_e_f1/(approx_e_f1+1)**2*b/s + \
                     1/s/(approx_e_f1+1)

        delta_td_u = (-s/bw)*50*approx_e_f1/(approx_e_f1+1)**2*(u/bw+rtt) + \
                     (1/bw)*approx_e_f1/(approx_e_f1+1) + \
                     -50*(-s/bw)*approx_e_f1/(approx_e_f1+1)**2*b/s

        delta_td_s = -(u/bw+rtt)*50*approx_e_f1/(approx_e_f1+1)**2*(u/bw+rtt) + \
                     -50*-(u/bw+rtt)*approx_e_f1/(approx_e_f1+1)**2*b/s + \
                     -b/s**2/(approx_e_f1+1) 

        delta_b_b = (50*approx_e_f1/(approx_e_f1+1)**2)*(self.Bu*approx_e_f3+f_2)/(approx_e_f3+1) + \
                    ((50*self.Bu*approx_e_f3+approx_e_f3+1-50*approx_e_f3*f_2)/(approx_e_f3+1)**2)*approx_e_f1/(approx_e_f1+1) +\
                    -50*self.delta*approx_e_f1/(approx_e_f1+1)**2
        delta_b_u = delta_b_b * -s/bw
        delta_b_s = delta_b_b * -(u/bw+rtt)
        
        delta_l_b = -(s-1)*delta_td_b + delta_tf_b

        delta_l_u = -(s-1)*delta_td_u + delta_tf_u

        delta_l_s = -(t_display + (s-1)*delta_td_s) + delta_tf_s

        # new_latency = l + t_freeze
        new_latency = l - (s-1)*t_display + t_freeze
        new_latency_power = self.med_latency - new_latency
        pre_latnecy_power = self.med_latency - l
        approx_new_latency = np.round(np.e**new_latency_power, 4)
        approx_pre_latency = np.round(np.e**pre_latnecy_power, 4)
        # Shape 2*3
        # (b', l', u', s') = f(b, l, p_u, p_s, u, s) So self.ft is 2*3
        self.ft = np.array([[delta_b_b, 0, 0, 0, delta_b_u, delta_b_s],
                            [delta_l_b, 1, 0, 0, delta_l_u, delta_l_s],
                            [0, 0, 0, 0, 1, 0],
                            [0, 0, 0, 0, 0, 1]])

        delta_c_b = self.w3*delta_tf_b + 2*self.w4*np.abs(new_latency)*delta_l_b     # Continue here
        delta_c_l = 2*self.w4*np.abs(new_latency)
        delta_c_p_u = self.w2*2*np.log(u_p/u)/u_p
        delta_c_p_s = self.w6*2*(s_p-s)

        delta_c_u = -self.w1/u + self.w2*2*np.log(u/u_p)/u + \
                    self.w3*delta_tf_u + \
                    2*self.w4*np.abs(new_latency)*delta_l_u - \
                    50*self.barrier_1*approx_e_f5 + \
                    50*self.barrier_2*approx_e_f6

        delta_c_s = self.w3*delta_tf_s + \
                    2*self.w4*np.abs(new_latency)*delta_l_s + \
                    self.w5*2*(s-1) + self.w6*2*(s-s_p) - \
                    50*self.barrier_3*approx_e_f7 +\
                    50*self.barrier_4*approx_e_f8

        self.ct = np.array([[delta_c_b, delta_c_l, delta_c_p_u, delta_c_p_s, delta_c_u, delta_c_s]]).T

        # delta_tf_b = -(-50*approx_e_f1/(approx_e_f1+1)**2*f_1/s + 1/s/(approx_e_f1+1))
        # delta_tf_u = delta_tf_b*(-s/bw)
        # delta_tf_s = -(-50*-(u/bw+rtt)*approx_e_f1/(approx_e_f1+1)**2*f_1/s + \
        #             (-(u/bw+rtt)*s-f_1)/s**2/(approx_e_f1+1))
        # Second order 
        delta_tf_b_b = 2500*approx_e_f1*(1-approx_e_f1)/(approx_e_f1+1)**3*f_1/s + \
                       50*approx_e_f1/s/(approx_e_f1+1)**2 + \
                       50*approx_e_f1/s/(approx_e_f1+1)**2
        delta_tf_b_u = delta_tf_b_b*(-s/bw)

        ## Modified
        delta_tf_b_s = 2500*(-(u/bw+rtt))*approx_e_f1*(1-approx_e_f1)/(approx_e_f1+1)**3*f_1/s + \
                        (-(u/bw+rtt)*s-f_1)/s**2*50*approx_e_f1/(approx_e_f1+1)**2 + \
                        1/s**2/(approx_e_f1+1) + \
                        50*(-(u/bw+rtt))*approx_e_f1/s/(approx_e_f1+1)**2
        delta_tf_u_u = delta_tf_b_b * (-s/bw)**2

        delta_tf_u_s = delta_tf_b_s*(-s/bw) - 1/bw*delta_tf_b

        # delta_tf_u_s = -1/bw*(50*approx_e_f1/(approx_e_f1+1)**2*f_1/s - 1/s/(approx_e_f1+1)) + \
        #               (-s/bw)*(2500*(-(u/bw+rtt))*approx_e_f1*(1-approx_e_f1)/(approx_e_f1+1)**3*f_1/s + \
        #               50*approx_e_f1/(approx_e_f1+1)**2*(-(u/bw+rtt)*s-f_1)/s**2 + \
        #               (1+approx_e_f1+s*50*-(u/bw+rtt)*approx_e_f1)/(s*(approx_e_f1+1))**2)
        delta_tf_s_s = 2500*(-(u/bw+rtt))**2*approx_e_f1*(1-approx_e_f1)/(approx_e_f1+1)**3*f_1/s + \
                        (-(u/bw+rtt)*s-f_1)/s**2*50*(-(u/bw+rtt))*approx_e_f1/(approx_e_f1+1)**2 + \
                        (-(u/bw+rtt)*s-f_1)/s**2*50*(-(u/bw+rtt))*approx_e_f1/(approx_e_f1+1)**2 + \
                        2*(-(u/bw+rtt)*s-f_1)/s**3/(approx_e_f1+1)
        # delta_tf_s = 50*-(u/bw+rtt)*approx_e_f1/(approx_e_f1+1)**2*f_1/s - \
                    # 1/(approx_e_f1+1)*(-(u/bw+rtt)*s-f_1)/s**2


        # t_display = approx_e_f1/(approx_e_f1+1)*(u/bw+rtt) + 1/(approx_e_f1+1)*b/s
        # delta_td_b = 50*approx_e_f1/(approx_e_f1+1)**2*(u/bw+rtt) + \
        #              -50*approx_e_f1/(approx_e_f1+1)**2*b/s + \
        #              1/s/(approx_e_f1+1)

        delta_td_b_b = 2500*approx_e_f1*(1-approx_e_f1)/(approx_e_f1+1)**3*(u/bw+rtt) - \
                       2500*approx_e_f1*(1-approx_e_f1)/(approx_e_f1+1)**3*b/s - \
                       50*approx_e_f1/s/(approx_e_f1+1)**2 - \
                       50*approx_e_f1/s/(approx_e_f1+1)**2
        delta_td_b_u = 2500*(-s/bw)*approx_e_f1*(1-approx_e_f1)/(approx_e_f1+1)**3*(u/bw+rtt) + \
                        1/bw*50*approx_e_f1/(approx_e_f1+1)**2 - \
                        2500*(-s/bw)*approx_e_f1*(1-approx_e_f1)/(approx_e_f1+1)**3*b/s - \
                        50*(-s/bw)*approx_e_f1/(approx_e_f1+1)**2/s

        delta_td_b_s = 2500*(-(u/bw+rtt))*approx_e_f1*(1-approx_e_f1)/(approx_e_f1+1)**3*(u/bw+rtt) - \
                       (2500*(-(u/bw+rtt))*approx_e_f1*(1-approx_e_f1)/(approx_e_f1+1)**3*(u/bw+rtt)*b/s - \
                       b/s**2*50*approx_e_f1/(approx_e_f1+1)**2) + \
                       (-1/s**2/(1+approx_e_f1)-50*(-(u/bw+rtt))*approx_e_f1/(1+approx_e_f1)**2/s)

        # delta_td_u = (-s/bw)*50*approx_e_f1/(approx_e_f1+1)**2*(u/bw+rtt) + \
        #              (1/bw)*approx_e_f1/(approx_e_f1+1) + \
        #              -50*(-s/bw)*approx_e_f1/(approx_e_f1+1)**2*b/s

        delta_td_u_u = 2500*(-s/bw)**2*approx_e_f1*(1-approx_e_f1)/(approx_e_f1+1)**3*(u/bw+rtt) + \
                        1/bw*50*(-s/bw)*approx_e_f1/(approx_e_f1+1)**2 + \
                        1/bw*50*(-s/bw)*approx_e_f1/(approx_e_f1+1)**2 - \
                        2500*(-s/bw)**2*approx_e_f1*(1-approx_e_f1)/(approx_e_f1+1)**3*(u/bw+rtt)*b/s

        delta_td_u_s = (-1/bw)*(50*approx_e_f1/(approx_e_f1+1)**2*(u/bw+rtt)) + \
                        (-s/bw)*(u/bw+rtt)*2500*-(u/bw+rtt)*approx_e_f1*(1-approx_e_f1)/(approx_e_f1+1)**3 + \
                        (1/bw)*50*-(u/bw+rtt)*approx_e_f1/(approx_e_f1+1)**2 + \
                        b/bw*2500*-(u/bw+rtt)*approx_e_f1*(1-approx_e_f1)/(approx_e_f1+1)**3

        # delta_td_s = -(u/bw+rtt)*50*approx_e_f1/(approx_e_f1+1)**2*(u/bw+rtt) + \
        #              -50*-(u/bw+rtt)*approx_e_f1/(approx_e_f1+1)**2*b/s + \
        #              -b/s**2/(approx_e_f1+1) 

        delta_td_s_s =  2500*(-(u/bw+rtt))**2*approx_e_f1*(1-approx_e_f1)/(approx_e_f1+1)**3*(u/bw+rtt) - \
                        2500*(-(u/bw+rtt))**2*approx_e_f1*(1-approx_e_f1)/(approx_e_f1+1)**3*b/s + \
                        b/s**2*50*(-(u/bw+rtt))*approx_e_f1/(approx_e_f1+1)**2 + \
                        2*b/s**3/(approx_e_f1+1) + \
                        b/s**2*50*-(u/bw+rtt)*approx_e_f1/(approx_e_f1+1)**2

        delta_l_b_b = -(s-1)*delta_td_b_b + delta_tf_b_b

        delta_l_b_u = -(s-1)*delta_td_b_u + delta_tf_b_u

        delta_l_b_s = -(delta_td_b + (s-1)*delta_td_b_s) + delta_tf_b_s
        delta_l_u_u = -(s-1)*delta_td_u_u + delta_tf_u_u
        delta_l_u_s = -(delta_td_u + (s-1)*delta_td_u_s) + delta_tf_u_s
        delta_l_s_s = -(delta_td_s + delta_td_s + (s-1)*delta_td_s_s) + delta_tf_s_s


        delta_c_b_b = self.w3*delta_tf_b_b + \
                      2*self.w4*delta_l_b*delta_l_b + 2*self.w4*np.abs(new_latency)*delta_l_b_b 
        delta_c_b_l = 2*self.w4*delta_l_b
        delta_c_b_pu = 0
        delta_c_b_ps = 0    
        delta_c_b_u = self.w3*delta_tf_b_u + \
                      2*self.w4*delta_l_u*delta_l_b + 2*self.w4*np.abs(new_latency)*delta_l_b_u

        delta_c_b_s = self.w3*delta_tf_b_s + \
                      2*self.w4*delta_l_s*delta_l_b + 2*self.w4*np.abs(new_latency)*delta_l_b_s

        # delta_c_l = self.w4*approx_new_latency/(approx_new_latency+1)**2
        delta_c_l_b = delta_c_b_l
        delta_c_l_l = 2*self.w4
        delta_c_l_pu = 0
        delta_c_l_ps = 0
        delta_c_l_u = 2*self.w4*delta_l_u
        delta_c_l_s = 2*self.w4*delta_l_s

        # delta_c_p_u = self.w2*2*np.log(u_p/u)/u_p
        delta_c_pu_b = 0
        delta_c_pu_l = 0
        delta_c_pu_pu = self.w2*2*(1-np.log(u_p/u))/u_p**2
        delta_c_pu_ps = 0
        delta_c_pu_u = self.w2*-2/u_p/u
        delta_c_pu_s = 0

        # delta_c_p_s = self.w6*2*(s_p-s)
        delta_c_ps_b = 0
        delta_c_ps_l = 0
        delta_c_ps_pu = 0
        delta_c_ps_ps = self.w6*2
        delta_c_ps_u = 0
        delta_c_ps_s = self.w6*-2

        # delta_c_u = -self.w1/u + self.w2*2*np.log(u/u_p)/u + \
        #             self.w3*delta_tf_u + \
        #             self.w4*delta_l_u*approx_new_latency/(approx_new_latency+1)**2 -\
        #             50*self.barrier_1*approx_e_f5 + \
        #             50*self.barrier_2*approx_e_f6
        delta_c_u_b = delta_c_b_u
        delta_c_u_l = delta_c_l_u
        delta_c_u_pu = delta_c_pu_u
        delta_c_u_ps = 0
        delta_c_u_u = self.w1/u**2 + self.w2*2*(1-np.log(u/u_p))/u**2 + \
                      self.w3* delta_tf_u_u + \
                      2*self.w4*delta_l_u*delta_l_u + 2*self.w4*np.abs(new_latency)*delta_l_u_u +\
                      2500.0*self.barrier_1*approx_e_f5 + \
                      2500.0*self.barrier_2*approx_e_f6 
        delta_c_u_s = self.w3*delta_tf_u_s + \
                      2*self.w4*delta_l_s*delta_l_u + 2*self.w4*np.abs(new_latency)*delta_l_u_s

        # delta_c_s = self.w3*delta_tf_s + \
        #             self.w4*delta_l_s*approx_new_latency/(approx_new_latency+1)**2 + \
        #             self.w5*2*(s-1) + self.w6*2*(s-s_p) - \
        #             50*self.barrier_3*approx_e_f7 +\
        #             50*self.barrier_4*approx_e_f8
        delta_c_s_b = delta_c_b_s
        delta_c_s_l = delta_c_l_s
        delta_c_s_pu = 0
        delta_c_s_ps = delta_c_ps_s
        delta_c_s_u = delta_c_u_s
        delta_c_s_s = self.w3*delta_tf_s_s + \
                      2*self.w4*delta_l_s*delta_l_s + 2*self.w4*np.abs(new_latency)*delta_l_s_s +\
                      self.w5*2 + self.w6*2 + \
                      2500.0*self.barrier_3*approx_e_f7 + \
                      2500.0*self.barrier_4*approx_e_f8 

        self.CT = np.array([[delta_c_b_b, delta_c_b_l, delta_c_b_pu, delta_c_b_ps, delta_c_b_u, delta_c_b_s],
                            [delta_c_l_b, delta_c_l_l, delta_c_l_pu, delta_c_l_ps, delta_c_l_u, delta_c_l_s],
                            [delta_c_pu_b, delta_c_pu_l, delta_c_pu_pu, delta_c_pu_ps, delta_c_pu_u, delta_c_pu_s],
                            [delta_c_ps_b, delta_c_ps_l, delta_c_ps_pu, delta_c_ps_ps, delta_c_ps_u, delta_c_ps_s],
                            [delta_c_u_b, delta_c_u_l, delta_c_u_pu, delta_c_u_ps, delta_c_u_u, delta_c_u_s],
                            [delta_c_s_b, delta_c_s_l, delta_c_s_pu, delta_c_s_ps, delta_c_s_u, delta_c_s_s]]).T
        if LQR_DEBUG:
            print("l_b: ", delta_l_b)
            print("l_s: ", delta_l_s)
            print("l_b_s: ", delta_l_b_s)
            print("td_b: ", delta_td_b)
            print("Update matrix in step: ", step_i)
            print("CT matrix: ", self.CT)
            print("ct matrix: ", self.ct)
            print("ft matrix: ", self.ft)

    def iterate_LQR(self):
        # Get first loop of state using initial_u
        VT = 0
        vt = 0
        for ite_i in range(self.n_iteration):
            converge = True
            KT_list = [0.0] * self.n_step
            kt_list = [0.0] * self.n_step
            VT_list = [0.0] * self.n_step
            vt_list = [0.0] * self.n_step
            pre_xt_list = [0.0] * self.n_step
            new_xt_list = [0.0] * self.n_step
            pre_ut_list  = [0.0] * self.n_step

            # Backward pass
            for step_i in reversed(range(self.n_step)):
                self.update_matrix(step_i)
                xt = np.array([[self.states[step_i][0]],[self.states[step_i][1]], [self.states[step_i][2]], [self.states[step_i][3]]])      #2*1
                ut = np.array([[self.rates[step_i]], [self.speeds[step_i]]]) 
                pre_xt_list[step_i] = xt
                pre_ut_list[step_i] = ut
                if step_i == self.n_step-1:
                    Qt = self.CT
                    qt = self.ct
                else:
                    # To be modified
                    Qt = self.CT + np.dot(np.dot(self.ft.T, VT), self.ft)    # 3*3
                    qt = self.ct + np.dot(self.ft.T, vt)                     # 3*1. self.ft is FT in equation, and ft in this equation is zeor (no constant)
                    if LQR_DEBUG:
                        print("vt: ", vt)
                        print("qt: ", qt)

                # Origin
                Q_xx = Qt[:4,:4]         #4*4
                Q_xu = Qt[:4,4:]         #4*2
                Q_ux = Qt[4:,:4]         #2*4
                Q_uu = Qt[4:,4:]         #2*2
                q_x = qt[:4]             #4*1
                q_u = qt[4:]             #2*1
                # print(q_x)
                # print(q_u)

                # print(Qt)
                # Q_xx = Qt[:4,:4]         #4*4
                # Q_xu = Qt[:4,4:]         #4*2
                # Q_ux = Qt[4:,:4]         #2*4
                # Q_uu = Qt[4:,4:]         #2*2
                # q_x = qt[:4]             #4*1
                # q_u = qt[4:]             #2*1


                KT = np.dot(-1, np.dot(np.linalg.inv(Q_uu), Q_ux))          
                kt = np.dot(-1, np.dot(np.linalg.inv(Q_uu), q_u))     

                if iLQR_SHOW:
                    print("Ct: ", self.CT)
                    print("   ")
                    print("self.ft.T: ", self.ft.T)
                    print("VT: ", VT)
                    print("self.ft: ", self.ft)
                    print("Q_ux: ", Q_ux)
                    print("Q_uu: ", Q_uu)
                    print("KT: ", KT)   
                    print("kt: ", kt)
                    print("Step: ", step_i)     
                    print("<======>")              

                VT = Q_xx + np.dot(Q_xu, KT) + np.dot(KT.T, Q_ux) + np.dot(np.dot(KT.T, Q_uu), KT)  #2*2
                vt = q_x + np.dot(Q_xu, kt) + np.dot(KT.T, q_u) + np.dot(np.dot(KT.T, Q_uu), kt)    #2*1
                KT_list[step_i] = KT
                kt_list[step_i] = kt
                VT_list[step_i] = self.decay*VT
                vt_list[step_i] = self.decay*vt
                
                if iLQR_SHOW:
                    print(VT)
                    print(",,,")
                    print(Q_xx)
                    print(",,,")
                    print(np.dot(Q_xu, KT))
                    print("...")
                    print(np.dot(KT.T, Q_ux))
                    print("last")
                    print(np.dot(np.dot(KT.T, Q_uu), KT))
                    print("end!!")

            if LQR_DEBUG:
                print("!!!!!! Backward done!!!!!!!!")
                print("pre xt: ", pre_xt_list)
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            # Forward pass
            new_xt_list[0] = pre_xt_list[0]
            for step_i in range(self.n_step):
                if LQR_DEBUG:
                    print("<=========================>")
                    print("forward pass, step: ", step_i)
                    print("new xt: ", new_xt_list[step_i])
                    print("pre xt: ", pre_xt_list[step_i])
                    print("kt matrix is: ", kt_list[step_i])
                d_x = new_xt_list[step_i] - pre_xt_list[step_i]
                k_t = self.kt_step*kt_list[step_i]
                K_T = self.KT_step*KT_list[step_i]

                d_u = np.dot(K_T, d_x) + k_t
                # new_u = pre_ut_list[step_i] + self.step_size*d_u       # New action
                new_u = [0,0]
                new_u[0] = max(0.75*pre_ut_list[step_i][0], min(1.1*pre_ut_list[step_i][0], pre_ut_list[step_i][0] + self.step_size*d_u[0]))
                new_u[1] = max(0.75*pre_ut_list[step_i][1], min(1.1*pre_ut_list[step_i][1], pre_ut_list[step_i][1] + self.step_size*d_u[1]))
                if iLQR_SHOW:
                    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
                    print("Step: ", step_i)
                    print("Dx is: ", d_x)
                    print("kt: ", k_t)
                    print("KT: ", K_T)
                    print("Du: ", d_u)
                    print("Ut: ", pre_ut_list[step_i])
                    print("New action: ", new_u)
                    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
                    if new_u[0] >= 2*self.predicted_bw[step_i]:
                        input()
                # n_rate = np.round(new_u[0][0], 2)
                n_rate =  min(max(np.round(new_u[0][0], 2), 0.3), 6.0)
                # n_speed = np.round(new_u[1][0], 2)
                n_speed = min(max(np.round(new_u[1][0], 2), 0.5), 1.5)
                # Check converge
                if converge and (np.round(n_rate,1) != np.round(self.rates[step_i],1) or np.round(n_speed,2) != np.round(self.speeds[step_i], 2)):
                    converge = False
                self.rates[step_i] = n_rate
                self.speeds[step_i] = n_speed
                new_x = new_xt_list[step_i]             # Get new state
                rtt = self.predicted_rtt[step_i]
                bw = self.predicted_bw[step_i]

                new_next_b, new_next_l = self.sim_fetch(new_x[0][0], new_x[1][0], n_rate, n_speed, rtt, bw)               # Simulate to get new next state
                if LQR_DEBUG:
                    print("new x: ", new_x)
                    print("new b: ", new_next_b)
                    print("bew l: ", new_next_l)
                if step_i < self.n_step - 1:
                    new_xt_list[step_i+1] = [[new_next_b], [new_next_l], [n_rate], [n_speed]]
                    self.states[step_i+1] = [np.round(new_next_b, 2), np.round(new_next_l, 2), self.rates[step_i], self.speeds[step_i]]
                else:
                    self.states[step_i+1] = [np.round(new_next_b, 2), np.round(new_next_l, 2), self.rates[step_i], self.speeds[step_i]]

            # Check converge
            if converge:
                break

            if LQR_DEBUG:
                print("New states: ", self.states)
                print("New actions: ", self.rates)
            
            # Check convergence
            if iLQR_SHOW:
                print("Iteration ", ite_i, ", previous rate: ", self.states[0][1])
                print("Iteration ", ite_i, ", buffer is: ", [x[0] for x in self.states])
                print("Iteration ", ite_i, ", latency is: ", [x[1] for x in self.states])
                print("Iteration ", ite_i, ", pre bw is: ", self.predicted_bw)
                print("Iteration ", ite_i, ", action is: ", self.rates)
                print("Iteration ", ite_i, ", action is: ", self.speeds)

                print("<===============================================>")

        r_idx = self.translate_to_rate_idx()
        # s_idx = self.translate_to_speed_idx()
        s_idx = self.translate_to_speed_idx_accu()
        # s_idx = self.translate_to_speed_idx_accu_new()
        return r_idx, s_idx

    def get_rates(self):
        return self.rates

    def translate_to_speed_idx(self):
        first_speed = self.speeds[0]
        # first_speed = np.mean(self.speeds)
        # print("speeds: ", self.speeds)
        # print("<-------------END------------>")
        distance = [np.abs(first_speed-s) for s in SPEED]
        speed_idx = distance.index(min(distance))
        return speed_idx

    def translate_to_speed_idx_accu(self):
        # first_speed = self.speeds[0]
        first_speed = np.mean(self.speeds[:3])
        # print("speeds: ", self.speeds)
        # print("<-------------END------------>")
        distance = [np.abs(first_speed-s) for s in SPEED]
        speed_idx = distance.index(min(distance))
        # if first_speed <= 0.96:
        #     return 0
        # elif first_speed >= 1.04:
        #     return 2
        # else:
        #     return 1
        return speed_idx

    def translate_to_speed_idx_accu_new(self):
        first_5_change = np.sum(self.speeds[:5])-5 
        if first_5_change >= 0.05:
            return 2
        elif first_5_change <= -0.05:
            return 0
        else:
            return 1

    def translate_to_rate_idx(self):
        first_action = self.rates[0]
        # distance = [np.abs(first_action-br/KB_IN_MB) for br in BITRATE]
        # rate_idx = distance.index(min(distance))
        # low quantize
        rate_idx = 0
        for j in reversed(range(len(BITRATE))):
            if BITRATE[j]/KB_IN_MB <= first_action:
                rate_idx = j
                break
        # print("Rate is: ", first_action)
        # print("Rate index: ", rate_idx)

            # input()
        return rate_idx

    def sim_fetch(self, buffer_len, latency, u, s, rtt, bw):
        # print("size: ", u)
        # print("speed: ", s)
        size = u
        speed = s
        # print('Seg size is: ', size)
        # Chunk downloading
        freezing = 0.0
        wait_time = 0.0
        current_reward = 0.0
        download_time = size/bw + rtt
        freezing = max(0.0, download_time - (buffer_len + (CHUNK_IN_SEG-1)*self.delta)/speed)
        display_time = min(download_time, buffer_len/speed)
        latency -= display_time*(speed-1)
        latency += freezing
        buffer_len = max(buffer_len + (CHUNK_IN_SEG-1)*self.delta - speed*download_time, 0.0)
        buffer_len += self.delta
        if freezing > 0.0:
            print
            if  buffer_len != self.delta:
                print("speed should be invalid: ", speed)
                print("freezing is: ", freezing)
        buffer_len = min(self.Bu, buffer_len)
        return buffer_len, latency

    # def LQR(self, step_i):
    #     xt = np.array([[self.states[step_i][0]],[self.states[step_i][1]]])  #2*1
    #     ut = np.array([[self.us[step_i]]])                                  #1*1

    #     if step_i == self.n_step-1:
    #         Qt = self.CT
    #         qt = self.ct
    #     else:
    #         # To be modified
    #         Qt = self.CT + self.ft.T


    #     Q_xx = Qt[:2,:2]        #2*2
    #     Q_xu = Qt[:2,2]         #2*1
    #     Q_ux = Qt[2,:2]         #1*2
    #     Q_uu = Qt[2,2]          #1*1
    #     q_x = qt[:2]            #2*1
    #     q_u = qt[2]             #1*1


    #     KT = np.dot(-1, np.dot(Q_uu**-1, Q_ux))         #1*2
    #     kt = np.dot(-1, np.dot(Q_uu**-1, q_u))          #1*1
    #     d_u = np.dot(KT, xt) + kt
    #     VT = Q_xx + np.dot(Q_xu, KT) + np.dot(KT.T, Q_ux) + np.dot(np.dot(KT.T, Q_uu), KT)  #2*2
    #     vt = q_x + np.dot(Q_xu, kt) + np.dot(KT.T, q_u) + np.dot(np.dot(KT.T, Q_uu), kt)

