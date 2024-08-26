import threading
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import Categorical
import math

def initialize_weights_he(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class BaseNetwork(nn.Module):
    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

class DQNBase(BaseNetwork):

    def __init__(self, num_channels):
        super(DQNBase, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            Flatten(),
        ).apply(initialize_weights_he)

    def forward(self, states):
        return self.net(states)

class CateoricalPolicy(nn.Module):

    def __init__(self, input_dim, num_actions, hidden_dim=256, num_layers=2, shared=False):
        super().__init__()
        self.shared = shared
        self.attention = Attention(hidden_dim)
        if not shared:
            self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
            self.layer_norm = nn.LayerNorm(hidden_dim)
            
        self.head = nn.Sequential(
            ResidualBlock(hidden_dim, hidden_dim, 512),  
            #nn.Linear(hidden_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
                       
            nn.Linear(512, num_actions)
        ).apply(self._initialize_weights)
        
        self.intensity_head = nn.Sequential(
            ResidualBlock(hidden_dim, hidden_dim, 512),
            #nn.Linear(hidden_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            
            nn.Linear(512, 1),
            nn.Sigmoid()  # Intensity between 0 and 1
    
        ).apply(self._initialize_weights)
        
        
    def _initialize_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    torch.nn.init.kaiming_uniform_(param.data, nonlinearity='relu')
                elif 'weight_hh' in name:
                    torch.nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0)
                
    def act(self, states):
        if not self.shared:
            states, _ = self.lstm(states)
            
        action_logits = self.head(states[:, -1, :])  # Use the output of the last time step
        greedy_actions = torch.argmax(action_logits, dim=1, keepdim=True)
        return greedy_actions

    def sample(self, states):
        if not self.shared:
            states = states.clone().detach().unsqueeze(1) if states.dim() == 2 else states
            states, _ = self.lstm(states)
            states = self.attention(states)
            states = self.layer_norm(states)
                   
        action_logits = self.head(states[:, -1, :])  # Use the output of the last time step

        #action_logits = torch.clamp(action_logits, 0, 1)       
        action_probs = F.softmax(action_logits, dim=1)
        action_dist = Categorical(action_probs)
        actions = action_dist.sample().view(-1, 1)

        intensity = self.intensity_head(states[:, -1, :])
        #intensity = torch.exp(intensity) - 1
        #print(f"Intensity: {intensity}")
        # Avoid numerical instability
        z = (action_probs == 0.0).float() * 1e-8
        log_action_probs = torch.log(action_probs + z)

        return actions, action_probs, log_action_probs, intensity
    
class ResidualBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ResidualBlock, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        if input_dim != output_dim:
            self.shortcut = nn.Linear(input_dim, output_dim)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out
    
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        attention_weights = self.softmax(torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(x.size(-1)))
        out = torch.matmul(attention_weights, v)
        return out

class LSTMQNetwork(nn.Module):
    def __init__(self, input_dim, num_actions, hidden_dim, num_layers, dueling_net=True):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.residual_block = ResidualBlock(hidden_dim, hidden_dim, hidden_dim)
        self.attention = Attention(hidden_dim)
        self.dueling_net = dueling_net

        if not dueling_net:
            self.head = nn.Sequential(
                nn.Linear(hidden_dim, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, num_actions)
            ).apply(self._initialize_weights)
        else:
            self.a_head = nn.Sequential(
                nn.Linear(hidden_dim, 512),
                nn.LayerNorm(512),
                nn.ReLU(inplace=True),

                # nn.Linear(128, 64),
                # nn.LayerNorm(64),
                # nn.ReLU(inplace=True),
                
                
                nn.Linear(512, num_actions)
            ).apply(self._initialize_weights)
            self.v_head = nn.Sequential(             
                nn.Linear(hidden_dim, 512),
                nn.LayerNorm(512),
                nn.ReLU(inplace=True),

                # nn.Linear(128, 64),
                # nn.LayerNorm(64),
                # nn.ReLU(inplace=True),
                
                
                nn.Linear(512, 1)
            ).apply(self._initialize_weights)

    def _initialize_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    torch.nn.init.kaiming_uniform_(param.data, nonlinearity='relu')
                elif 'weight_hh' in name:
                    torch.nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0)

    def forward(self, x):
        #with self.lock:
        x = x.unsqueeze(1).repeat(1, 20, 1) if x.dim() == 2 else x
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]  # Use the output of the last time step
        lstm_out = self.layer_norm(lstm_out)  # Apply layer normalization after LSTM
        lstm_out = self.residual_block(lstm_out)
        lstm_out = self.attention(lstm_out)
            
        if self.dueling_net:
            adv = self.a_head(lstm_out)
            val = self.v_head(lstm_out)
            q_values = val + adv - adv.mean(dim=1, keepdim=True)
            return q_values
        else:
            return self.head(lstm_out)



class TwinnedQNetwork(nn.Module):
    def __init__(self, input_dim, num_actions, hidden_dim=256, num_layers=2, dueling_net=True):
        super().__init__()
        self.Q1 = LSTMQNetwork(input_dim, num_actions, hidden_dim, num_layers, dueling_net)
        self.Q2 = LSTMQNetwork(input_dim, num_actions, hidden_dim, num_layers, dueling_net)

    def forward(self, states):
        q1 = self.Q1(states)
        q2 = self.Q2(states)
        return q1, q2
    
    
    """
                    nn.Linear(hidden_dim, 256),
                nn.LayerNorm(256),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.1),
                nn.Linear(256, 128),
                nn.LayerNorm(128),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.1),
                nn.Linear(128, 64),
                nn.LayerNorm(64),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.1),
                nn.Linear(64, 32),
                nn.LayerNorm(32),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.1),
    """