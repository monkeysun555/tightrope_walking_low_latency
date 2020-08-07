import torch
import torch.nn as nn
from config import Config

class Model(nn.Module):
    def __init__(self, action_dim, model_version):
        super().__init__()
        torch.manual_seed(7)
        self.model_version = model_version
        if self.model_version == 0:
            self.lstm1 = nn.LSTM(input_size=5, hidden_size=32, num_layers=2, dropout=0.1, batch_first=True, bidirectional=True)
            self.fc1 = nn.Sequential(
                nn.Linear(in_features=3, out_features=32),
                nn.ReLU())
            self.fc2 = nn.Sequential(
                nn.Linear(in_features=992, out_features=128),
                nn.ReLU())
            self.output = nn.Linear(in_features=128, out_features=action_dim)

    def forward(self, observation):
        # Shape of observation: (batch, 15, 10) (batch, seq, input_size)
        h0 = torch.randn(2*2, len(observation), 32)
        c0 = torch.randn(2*2, len(observation), 32)
        if self.model_version == 0:
            lstm1_out, (hn, cn) = self.lstm1(torch.transpose(observation[:, 0:5,:], 1, 2), (h0,c0))         # input: (5, 15) to (1,15,5) , output: (1,15, 2*32)
            fc1_out = self.fc1(observation[:, 5:, -1])
            fc2_out = self.fc2(torch.cat((torch.flatten(lstm1_out, start_dim=1), fc1_out), 1))                      # flatten: (1,15,2*32) to (1,-1) and cat with (1,32)
            advantage = self.output(fc2_out)
            # print(advantage)
            return advantage

    def save(self, path, step, optimizer):
        torch.save({
            'step': step,
            'state_dict': self.state_dict(),
            'optimizer': optimizer.state_dict()
        }, path)
            
    def load(self, checkpoint_path, optimizer=None):
        checkpoint = torch.load(checkpoint_path)
        step = checkpoint['step']
        self.load_state_dict(checkpoint['state_dict'])
        if not optimizer:
            optimizer.load_state_dict(checkpoint['optimizer'])
