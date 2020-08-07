import torch
from torch.nn.functional import mse_loss
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import glob
import os
import math
from config import Config
from models import Model

class Agent:
    def __init__(self, action_dims, model_version=Config.model_version,
                                    q_version=Config.q_version,
                                    target_version=Config.target_version,
                                    loss_version=Config.loss_version):
        # self.action_num = action_num
        self.action_dims = action_dims
        self.epsilon = Config.initial_epsilon
        self.epsilon_final = Config.epsilon_final
        self.epsilon_start = Config.epsilon_start
        self.epsilon_decay = Config.epsilon_decay
        self.model_version = model_version
        self.q_version = q_version
        self.target_version = target_version
        self.loss_version = loss_version
        self.build_network()

    def build_network(self):
        self.Q_network = Model(self.action_dims, self.model_version, self.loss_version)
        self.target_network = Model(self.action_dims, self.model_version, self.loss_version)
        # Change learning rate for commen net !!!! Start from here
        if self.loss_version == 0:
            self.optimizers = optim.Adam(self.Q_network.parameters(), lr=Config.lr)
        elif self.loss_version == 1:
            if self.model_version == 1:
                self.optimizers = [optim.Adam([
                    {'params': self.Q_network.multi_output_1.parameters(), 'lr':Config.lr},
                    {'params': self.Q_network.fc2.parameters()},
                    {'params': self.Q_network.fc1.parameters()},
                    {'params': self.Q_network.lstm1.parameters()}
                    ], lr=0.5*Config.lr), 
                    optim.Adam([
                    {'params': self.Q_network.multi_output_2.parameters(), 'lr':Config.lr},
                    {'params': self.Q_network.fc2.parameters()},
                    {'params': self.Q_network.fc1.parameters()},
                    {'params': self.Q_network.lstm1.parameters()}
                    ], lr=0.5*Config.lr)]
            elif self.model_version == 2:
                self.optimizers = [optim.Adam([
                    {'params': self.Q_network.multi_output_1.parameters(), 'lr':Config.lr},
                    {'params': self.Q_network.fc2.parameters()},
                    {'params': self.Q_network.fc1.parameters()},
                    {'params': self.Q_network.lstm1.parameters()},
                    {'params': self.Q_network.dueling.parameters()}
                    ], lr=0.5*Config.lr), 
                    optim.Adam([
                    {'params': self.Q_network.multi_output_2.parameters(), 'lr':Config.lr},
                    {'params': self.Q_network.fc2.parameters()},
                    {'params': self.Q_network.fc1.parameters()},
                    {'params': self.Q_network.lstm1.parameters()},
                    {'params': self.Q_network.dueling.parameters()}
                    ], lr=0.5*Config.lr)]

    def update_target_network(self):
        # copy current_network to target network
        self.target_network.load_state_dict(self.Q_network.state_dict())
    
    def update_Q_network_v0(self, state, action, reward, state_new, terminal):
        state = torch.from_numpy(state).float()
        action = torch.from_numpy(action).float()
        state_new = torch.from_numpy(state_new).float()
        terminal = torch.from_numpy(terminal).float()
        reward = torch.from_numpy(reward).float()
        state = Variable(state)
        action = Variable(action)                  # shape (batch, 6*7)
        state_new = Variable(state_new)
        terminal = Variable(terminal)
        reward = Variable(reward)
        self.Q_network.eval()
        self.target_network.eval()
        
        # use current network to evaluate action argmax_a' Q_current(s', a')_
        actions_new = self.Q_network.forward(state_new).max(dim=1)[1].cpu().data.view(-1, 1)
        actions_new_onehot = torch.zeros(Config.sampling_batch_size, self.action_dims[0]*self.action_dims[1]) 
        actions_new_onehot = Variable(actions_new_onehot.scatter_(1, actions_new, 1.0))
        
        # Different loss and object
        # use target network to evaluate value y = r + discount_factor * Q_tar(s', a')
        y = reward + torch.mul(((self.target_network.forward(state_new)*actions_new_onehot).sum(dim=1)*terminal),Config.discount_factor)
        self.Q_network.train()
        Q = (self.Q_network.forward(state)*action).sum(dim=1)
        losses = []
        # y = reward + torch.mul(((self.target_network.forward(state_new)[action_idx]*actions_new_onehot[action_idx]).sum(dim=1)*terminal), Config.discount_factor)
        # Q = (self.Q_network.forward(state)[action_idx]*actions[action_idx]).sum(dim=1)
        loss = mse_loss(input=Q, target=y.detach())
        self.optimizers.zero_grad()
        loss.backward()
        self.optimizers.step()
        losses.append(loss.item())
        # print(losses)
        return losses

    def update_Q_network_v1(self, state, action_1, action_2, reward, state_new, terminal):
        state = torch.from_numpy(state).float()
        action_1 = torch.from_numpy(action_1).float()
        action_2 = torch.from_numpy(action_2).float()
        state_new = torch.from_numpy(state_new).float()
        terminal = torch.from_numpy(terminal).float()
        reward = torch.from_numpy(reward).float()
        state = Variable(state)
        action_1 = Variable(action_1)                 
        action_2 = Variable(action_2)
        state_new = Variable(state_new)
        terminal = Variable(terminal)
        reward = Variable(reward)
        self.Q_network.eval()
        self.target_network.eval()
        
        # use current network to evaluate action argmax_a' Q_current(s', a')_
        new_q_values = self.Q_network.forward(state_new)
        actions_new = [torch.max(q_value, 1)[1].cpu().data.view(-1, 1) for q_value in new_q_values] 
        actions_new_onehot = [torch.zeros(Config.sampling_batch_size, action_dim) for action_dim in self.action_dims]
        actions_new_onehot = [Variable(actions_new_onehot[action_idx].scatter_(1, actions_new[action_idx], 1.0)) for action_idx in range(len(self.action_dims))]
        
        # Different loss and object
        # use target network to evaluate value y = r + discount_factor * Q_tar(s', a')
        actions = [action_1, action_2]
        raw_y = self.target_network.forward(state_new)
        y = []
        for new_q_idx in range(len(raw_y)):
            y.append(reward + torch.mul(((raw_y[new_q_idx]*actions_new_onehot[new_q_idx]).sum(dim=1)*terminal),Config.discount_factor))
        self.Q_network.train()
        raw_Q = self.Q_network.forward(state)
        Q = []
        for q_idx in range(len(raw_Q)):
            Q.append((raw_Q[q_idx]*actions[q_idx]).sum(dim=1))
        
        # Calculate loss
        losses = []
        if self.loss_version == 0:
            loss = 0.0
            for action_idx in range(len(self.action_dims)):
                loss += mse_loss(input=Q[action_idx], target=y[action_idx].detach())
            loss /= len(self.action_dims)
            self.optimizers.zero_grad()
            loss.backward()
            self.optimizers.step()
            losses.append(loss.item())

        elif self.loss_version == 1:
            for action_idx in range(len(self.action_dims)):
                loss = mse_loss(input=Q[action_idx], target=y[action_idx].detach())
                self.optimizers[action_idx].zero_grad()
                if action_idx < len(self.action_dims)-1:
                    loss.backward(retain_graph=True)
                else:
                    loss.backward()
                self.optimizers[action_idx].step()
                losses.append(loss.item())
        # print(losses)    
        return losses

    def update_Q_network_v2(self, state, action_1, action_2, reward, state_new, terminal):
        state = torch.from_numpy(state).float()
        action_1 = torch.from_numpy(action_1).float()
        action_2 = torch.from_numpy(action_2).float()
        state_new = torch.from_numpy(state_new).float()
        terminal = torch.from_numpy(terminal).float()
        reward = torch.from_numpy(reward).float()
        state = Variable(state)
        action_1 = Variable(action_1)                 
        action_2 = Variable(action_2)
        state_new = Variable(state_new)
        terminal = Variable(terminal)
        reward = Variable(reward)
        self.Q_network.eval()
        self.target_network.eval()
        
        # use current network to evaluate action argmax_a' Q_current(s', a')_
        # Get argmax action does not need aggregation
        new_q_values = self.Q_network.forward(state_new)
        # print(torch.max(new_q_values[0],1)[0], torch.max(new_q_values[0],1)[0].data)
        # print(new_q_values[0], new_q_values[0] + new_q_values[2], new_q_values[2])   
        actions_new = [torch.max(q_value, 1)[1].cpu().data.view(-1, 1) for q_value in new_q_values[:2]] 
        actions_new_onehot = [torch.zeros(Config.sampling_batch_size, action_dim) for action_dim in self.action_dims]
        actions_new_onehot = [Variable(actions_new_onehot[action_idx].scatter_(1, actions_new[action_idx], 1.0)) for action_idx in range(len(self.action_dims))]

        actions = [action_1, action_2]
        raw_y = self.target_network.forward(state_new)
        if self.q_version == 0:
            # Q = V + A
            # Get Q(s',a')
            aggregation = [raw_y[0]+ raw_y[2], raw_y[1]+raw_y[2]]
            # Get Q(s,a)
            self.Q_network.train()
            raw_Q = self.Q_network.forward(state)
            Q = []
            for q_idx in range(len(raw_Q[:2])):
                Q.append(((raw_Q[q_idx]+raw_Q[2])*actions[q_idx]).sum(dim=1))
        elif self.q_version == 1:
            # Q = V + Ad-max(Ad)
            # print(raw_y[0])
            # print(raw_y[1])
            # print(raw_y[2])
            # print(torch.max(raw_y[0],1, keepdim=True)[0])
            branch_0 = raw_y[0]-torch.max(raw_y[0], 1, keepdim=True)[0].data + raw_y[2] 
            branch_1 = raw_y[1]-torch.max(raw_y[1], 1, keepdim=True)[0].data + raw_y[2]
            aggregation = [branch_0, branch_1]
            print(aggregation)
            # Get Q(s,a)
            self.Q_network.train()
            raw_Q = self.Q_network.forward(state)
            Q = []
            for q_idx in range(len(raw_Q[:2])):
                Q.append(((raw_Q[q_idx]-torch.max(raw_y[0], 1, keepdim=True)[0].data+raw_Q[2])*actions[q_idx]).sum(dim=1))
        elif self.q_version == 2:
            # Q = V + Ad - mean(Ad)
            branch_0 = raw_y[0]-torch.mean(raw_y[0], 1, keepdim=True)[0].data + raw_y[2] 
            branch_1 = raw_y[1]-torch.mean(raw_y[1], 1, keepdim=True)[0].data + raw_y[2]
            aggregation = [branch_0, branch_1]
            # Get Q(s,a)
            self.Q_network.train()
            raw_Q = self.Q_network.forward(state)
            Q = []
            for q_idx in range(len(raw_Q[:2])):
                Q.append(((raw_Q[q_idx]-torch.mean(raw_y[0], 1, keepdim=True)[0].data+raw_Q[2])*actions[q_idx]).sum(dim=1))
       
        # Calculate target indep or global and then do loss (v0 or v1)
        losses = []
        if self.target_version == 0:
            # Independent target, 2 branches 
            y = []
            for new_q_idx in range(len(aggregation)):
                y.append(reward + torch.mul(((aggregation[new_q_idx]*actions_new_onehot[new_q_idx]).sum(dim=1)*terminal),Config.discount_factor))
            # print(y)

            # Calculate loss
            if self.loss_version == 0:
                loss = 0.0
                for action_idx in range(len(self.action_dims)):
                    loss += mse_loss(input=Q[action_idx], target=y[action_idx].detach())
                loss/=len(self.action_dims)                
                self.optimizers.zero_grad()
                loss.backward()
                self.optimizers.step()
                losses.append(loss.item())
            elif self.loss_version == 1:
                for action_idx in range(len(self.action_dims)):
                    loss = mse_loss(input=Q[action_idx], target=y[action_idx].detach())
                    self.optimizers[action_idx].zero_grad()
                    if action_idx < len(self.action_dims)-1:
                        loss.backward(retain_graph=True)
                    else:
                        loss.backward()
                    self.optimizers[action_idx].step()
                    losses.append(loss.item())
        elif self.target_version == 1:
            # Global target using max of branches, one y
            # Torch.max support find max over two 300*6 and 300*7 tensors
            local_max_0 = torch.mul(((aggregation[0]*actions_new_onehot[0]).sum(dim=1)*terminal),Config.discount_factor)
            local_max_1 = torch.mul(((aggregation[1]*actions_new_onehot[1]).sum(dim=1)*terminal),Config.discount_factor)
            # print(local_max_0, local_max_1)
            y = reward + torch.max(local_max_0, local_max_1)
            # print(y)
            # Calculate loss
            if self.loss_version == 0:
                loss = 0.0
                for action_idx in range(len(self.action_dims)):
                    loss += mse_loss(input=Q[action_idx], target=y.detach())
                loss/=len(self.action_dims)                
                self.optimizers.zero_grad()
                loss.backward()
                self.optimizers.step()
                losses.append(loss.item())
            elif self.loss_version == 1:
                for action_idx in range(len(self.action_dims)):
                    loss = mse_loss(input=Q[action_idx], target=y.detach())
                    self.optimizers[action_idx].zero_grad()
                    if action_idx < len(self.action_dims)-1:
                        loss.backward(retain_graph=True)
                    else:
                        loss.backward()
                    self.optimizers[action_idx].step()
                    losses.append(loss.item())

        # Global target using global average
        elif self.target_version == 2:
            local_max_0 = torch.mul(((aggregation[0]*actions_new_onehot[0]).sum(dim=1)*terminal),Config.discount_factor)
            local_max_1 = torch.mul(((aggregation[1]*actions_new_onehot[1]).sum(dim=1)*terminal),Config.discount_factor)
            # print(local_max_0)
            # print(local_max_1)
            # print(torch.mean(torch.stack((local_max_0, local_max_1)),0))
            y = reward + torch.mean(torch.stack((local_max_0, local_max_1)),dim=0)
            # Calculate loss
            if self.loss_version == 0:
                loss = 0.0
                for action_idx in range(len(self.action_dims)):
                    loss += mse_loss(input=Q[action_idx], target=y.detach())
                loss/=len(self.action_dims)                
                self.optimizers.zero_grad()
                loss.backward()
                self.optimizers.step()
                losses.append(loss.item())
            elif self.loss_version == 1:
                for action_idx in range(len(self.action_dims)):
                    loss = mse_loss(input=Q[action_idx], target=y.detach())
                    self.optimizers[action_idx].zero_grad()
                    if action_idx < len(self.action_dims)-1:
                        loss.backward(retain_graph=True)
                    else:
                        loss.backward()
                    self.optimizers[action_idx].step()
                    losses.append(loss.item())
        # print(losses)
        return losses

    def take_action(self, state):
        state = torch.from_numpy(state).float()
        state = Variable(state)
        self.Q_network.eval()

        if self.model_version == 0:
            estimate = torch.max(self.Q_network.forward(state), 1)[1].data[0]
            if np.random.random() < self.epsilon:
                return np.random.randint(0, self.action_dims[0]*self.action_dims[1])
            else:
                return estimate

        # elif self.model_version == 1:
        #     outputs = self.Q_network.forward(state)
        #     estimate = [torch.max(q_value, 1)[1].data[0] for q_value in outputs] 
        #     # with epsilon prob to choose random action else choose argmax Q estimate action
        #     if np.random.random() < self.epsilon:
        #         return [np.random.randint(0, self.action_dims[action_idx]-1) for action_idx in range(len(self.action_dims))]
        #     else:
        #         return estimate

        # elif self.model_version == 2:
        #     outputs = self.Q_network.forward(state)[:2]
        #     estimate = [torch.max(q_value, 1)[1].data[0] for q_value in outputs] 
        #     # with epsilon prob to choose random action else choose argmax Q estimate action
        #     if np.random.random() < self.epsilon:
        #         return [np.random.randint(0, self.action_dims[action_idx]-1) for action_idx in range(len(self.action_dims))]
        #     else:
        #         return estimate

    def testing_take_action(self, state):
        state = torch.from_numpy(state).float()
        state = Variable(state)
        self.Q_network.eval()
        if self.model_version == 0:
            estimate = torch.max(self.Q_network.forward(state), 1)[1].data[0]
            return estimate

        # elif self.model_version == 1:
        #     q_values = self.Q_network.forward(state)
        #     estimate = [torch.max(q_value, 1)[1].data[0] for q_value in q_values] 
        #     # with epsilon prob to choose random action else choose argmax Q estimate action
        #     return estimate

        # elif self.model_version == 2:
        #     q_values = self.Q_network.forward(state)[:2]
        #     estimate = [torch.max(q_value, 1)[1].data[0] for q_value in q_values] 
        #     # with epsilon prob to choose random action else choose argmax Q estimate action
        #     return estimate

    def update_epsilon_by_epoch(self, epoch):
        self.epsilon = self.epsilon_final+(self.epsilon_start - self.epsilon_final) * math.exp(-1.*epoch/self.epsilon_decay)   
    
    def save(self, step, logs_path):
        os.makedirs(logs_path, exist_ok=True)
        model_list =  glob.glob(os.path.join(logs_path, '*.pth'))
        if len(model_list) > Config.maximum_model - 1 :
            min_step = min([int(li.split('/')[-1][6:-4]) for li in model_list]) 
            os.remove(os.path.join(logs_path, 'model-{}.pth' .format(min_step)))
        logs_path = os.path.join(logs_path, 'model-{}.pth' .format(step))
        self.Q_network.save(logs_path, step=step, optimizers=self.optimizers)
        print('=> Save {}' .format(logs_path)) 
    
    def train_restore(self, logs_path):
        model_list =  glob.glob(os.path.join(logs_path, '*.pth'))
        max_step = max([int(li.split('/')[-1][6:-4]) for li in model_list]) 
        model_path = os.path.join(logs_path, 'model-{}.pth' .format(max_step))
        self.Q_network.load(model_path, self.optimizers)
        self.target_network.load(model_path, self.optimizers)
        print('=> Restore {}' .format(model_path))
        return int(max_step) + 1
        
    def restore(self, logs_path):
        if self.loss_version == 0:
            self.Q_network.load(logs_path, self.optimizers)
            self.target_network.load(logs_path, self.optimizers)
            print('=> Restore {}' .format(logs_path))
            # self.optimizers = optim.Adam(self.Q_network.parameters(), lr=Config.lr)
        elif self.loss_version == 1:
            self.Q_network.load(logs_path, self.optimizers)
            self.target_network.load(logs_path, self.optimizers)
            print('=> Restore {}' .format(logs_path)) 
            # if self.model_version == 0:
            #     self.optimizers = [optim.Adam([
            #         {'params': self.Q_network.multi_output_1.parameters(), 'lr':Config.lr},
            #         {'params': self.Q_network.fc2.parameters()},
            #         {'params': self.Q_network.fc1.parameters()},
            #         {'params': self.Q_network.lstm1.parameters()}
            #         ], lr=0.5*Config.lr), 
            #         optim.Adam([
            #         {'params': self.Q_network.multi_output_2.parameters(), 'lr':Config.lr},
            #         {'params': self.Q_network.fc2.parameters()},
            #         {'params': self.Q_network.fc1.parameters()},
            #         {'params': self.Q_network.lstm1.parameters()}
            #         ], lr=0.5*Config.lr)]
            # else:
            #     self.optimizers = [optim.Adam([
            #         {'params': self.Q_network.multi_output_1.parameters(), 'lr':Config.lr},
            #         {'params': self.Q_network.fc2.parameters()},
            #         {'params': self.Q_network.fc1.parameters()},
            #         {'params': self.Q_network.lstm1.parameters()},
            #         {'params': self.Q_network.dueling.parameters()}
            #         ], lr=0.5*Config.lr), 
            #         optim.Adam([
            #         {'params': self.Q_network.multi_output_2.parameters(), 'lr':Config.lr},
            #         {'params': self.Q_network.fc2.parameters()},
            #         {'params': self.Q_network.fc1.parameters()},
            #         {'params': self.Q_network.lstm1.parameters()},
            #         {'params': self.Q_network.dueling.parameters()}
            #         ], lr=0.5*Config.lr)]
