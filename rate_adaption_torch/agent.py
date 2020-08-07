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
    def __init__(self, action_dim, model_version=Config.model_version):
        # self.action_num = action_num
        self.action_dim = action_dim
        self.epsilon = Config.initial_epsilon
        self.epsilon_final = Config.epsilon_final
        self.epsilon_start = Config.epsilon_start
        self.epsilon_decay = Config.epsilon_decay
        self.model_version = model_version
        self.build_network()

    def build_network(self):
        self.Q_network = Model(self.action_dim, self.model_version)
        self.target_network = Model(self.action_dim, self.model_version)
        self.optimizer = optim.Adam(self.Q_network.parameters(), lr=Config.lr)        
    
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
        actions_new_onehot = torch.zeros(Config.sampling_batch_size, self.action_dim) 
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
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
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
                return np.random.randint(0, self.action_dim)
            else:
                return estimate

    def testing_take_action(self, state):
        state = torch.from_numpy(state).float()
        state = Variable(state)
        self.Q_network.eval()
        if self.model_version == 0:
            estimate = torch.max(self.Q_network.forward(state), 1)[1].data[0]
            return estimate

    def update_epsilon_by_epoch(self, epoch):
        self.epsilon = self.epsilon_final+(self.epsilon_start - self.epsilon_final) * math.exp(-1.*epoch/self.epsilon_decay)   
    
    def save(self, step, logs_path):
        os.makedirs(logs_path, exist_ok=True)
        model_list =  glob.glob(os.path.join(logs_path, '*.pth'))
        if len(model_list) > Config.maximum_model - 1 :
            min_step = min([int(li.split('/')[-1][6:-4]) for li in model_list]) 
            os.remove(os.path.join(logs_path, 'model-{}.pth' .format(min_step)))
        logs_path = os.path.join(logs_path, 'model-{}.pth' .format(step))
        self.Q_network.save(logs_path, step=step, optimizer=self.optimizer)
        print('=> Save {}' .format(logs_path)) 
    
    def restore(self, logs_path):
        self.Q_network.load(logs_path, self.optimizer)
        self.target_network.load(logs_path, self.optimizer)
        print('=> Restore {}' .format(logs_path))

    def train_restore(self, logs_path):
        model_list =  glob.glob(os.path.join(logs_path, '*.pth'))
        max_step = max([int(li.split('/')[-1][6:-4]) for li in model_list]) 
        model_path = os.path.join(logs_path, 'model-{}.pth' .format(max_step))
        self.Q_network.load(model_path, self.optimizer)
        self.target_network.load(model_path, self.optimizer)
        print('=> Restore {}' .format(model_path))
        return max_step + 1
