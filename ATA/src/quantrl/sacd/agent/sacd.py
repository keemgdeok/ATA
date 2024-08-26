import os
import numpy as np
import torch
from torch.optim import Adam

from sacd.agent.base import BaseSacd
from sacd.model import TwinnedQNetwork, CateoricalPolicy
from sacd.utils import disable_gradients

class Sacd(BaseSacd):

    def __init__(self, env, num_steps=100000, batch_size=64,
                 lr=1e-5, memory_size=1000000, gamma=0.99, multi_step=1,
                 target_entropy_ratio=0.98, start_steps=20000,
                 update_interval=4, target_update_interval=8000,
                 use_per=False, dueling_net=True, num_eval_steps=125000,
                 max_episode_steps=27000, log_interval=10, eval_interval=1000,
                 cuda=True, seed=0):
        super().__init__(
            env, num_steps, batch_size, memory_size, gamma,
            multi_step, target_entropy_ratio, start_steps, update_interval,
            target_update_interval, use_per, num_eval_steps, max_episode_steps,
            log_interval, eval_interval, cuda, seed)



        # Define networks.
        self.policy = CateoricalPolicy(
            57, env.action_space.n
        ).to(self.device)
        self.online_critic = TwinnedQNetwork(
            57, env.action_space.n,
            dueling_net=dueling_net).to(self.device)
        self.target_critic = TwinnedQNetwork(
            57, env.action_space.n,
            dueling_net=dueling_net).to(self.device).eval()

        # Copy parameters of the learning network to the target network.
        self.target_critic.load_state_dict(self.online_critic.state_dict())

        # Disable gradient calculations of the target network.
        disable_gradients(self.target_critic)

        self.policy_optim = Adam(self.policy.parameters(), lr=0.001)
        self.q1_optim = Adam(self.online_critic.Q1.parameters(), lr=0.001)
        self.q2_optim = Adam(self.online_critic.Q2.parameters(), lr=0.001)
        #print(env.action_space.n)
        # Target entropy is -log(1/|A|) * ratio (= maximum entropy * ratio).
        self.target_entropy = \
            -np.log(1.0 / env.action_space.n) * target_entropy_ratio
        #print(f"Target_Entropy_Ratio: {target_entropy_ratio}")
        
        # We optimize log(alpha), instead of alpha.
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp()
        self.alpha_optim = Adam([self.log_alpha], lr=0.001)
    
    def explore(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).unsqueeze(1).to(self.device)  # Adding batch and sequence dimensions
        with torch.no_grad():
            action, action_probs, _ , intensity = self.policy.sample(state)
        action = action.item()
        
        confidence = action_probs[0, action].item()  # Confidence based on the action probability
        return action, intensity # confidence

    def exploit(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).unsqueeze(1).to(self.device)  # Adding batch and sequence dimensions
        with torch.no_grad():
            action, action_probs, _, intensity= self.policy.sample(state)
        action = action.item()
        confidence = 1.0 
        return action, intensity # confidence

    def update_target(self):
        self.target_critic.load_state_dict(self.online_critic.state_dict())


    def calc_current_q(self, states, actions, rewards, next_states, dones):
        curr_q1, curr_q2 = self.online_critic.forward(states)
        #print(f"curr_q1 shape: {curr_q1.shape}")
        #print(f"actions shape: {actions.shape}")
            
        # Squeeze actions to shape [batch_size]
        actions = actions.squeeze(1)
        curr_q1 = curr_q1.gather(1, actions.unsqueeze(1)).squeeze(1)  
        curr_q2 = curr_q2.gather(1, actions.unsqueeze(1)).squeeze(1)  
        return curr_q1, curr_q2

    def calc_target_q(self, states, actions, rewards, next_states, dones):
        with torch.no_grad():
            _, action_probs, log_action_probs, _= self.policy.sample(next_states)
            next_q1, next_q2 = self.target_critic(next_states)
            next_q = (action_probs * (
                torch.min(next_q1, next_q2) - self.alpha * log_action_probs
                )).sum(dim=1, keepdim=True)
        
        assert rewards.shape == next_q.shape
        return rewards + (1.0 - dones) * self.gamma_n * next_q

    def calc_critic_loss(self, batch, weights):
        states = batch['states']
        actions = batch['actions']
        rewards = batch['rewards']
        next_states = batch['next_states'] 
        dones = batch['dones']
        
        curr_q1, curr_q2 = self.calc_current_q(states, actions, rewards, next_states, dones)
        target_q = self.calc_target_q(states, actions, rewards, next_states, dones)
        

        # TD errors for updating priority weights
        errors = torch.abs(curr_q1.detach() - target_q)

        # We log means of Q to monitor training.
        mean_q1 = curr_q1.detach().mean().item()
        mean_q2 = curr_q2.detach().mean().item()

        # Critic loss is mean squared TD errors with priority weights.
        q1_loss = torch.mean((curr_q1 - target_q).pow(2) * weights)
        q2_loss = torch.mean((curr_q2 - target_q).pow(2) * weights)

        return q1_loss, q2_loss, errors, mean_q1, mean_q2

    def calc_policy_loss(self, batch, weights):
        states = batch['states']

        # (Log of) probabilities to calculate expectations of Q and entropies.
        _, action_probs, log_action_probs, _ = self.policy.sample(states)
        with torch.no_grad():
            # Q for every actions to calculate expectations of Q.
            q1, q2 = self.online_critic(states)
            min_q = torch.min(q1, q2)

        # Expectations of entropies.
        entropies = -torch.sum(
            action_probs * log_action_probs, dim=1, keepdim=True)

        # Expectations of Q.
        q_expected = torch.sum(min_q * action_probs, dim=1, keepdim=True)
        
        # Policy objective is maximization of (Q + alpha * entropy) with
        # priority weights.
        policy_loss = (weights * (- q_expected - self.alpha * entropies)).mean()

        return policy_loss, entropies.detach()

    def calc_entropy_loss(self, entropies, weights):
        assert not entropies.requires_grad

        # Intuitively, we increse alpha when entropy is less than target
        # entropy, vice versa.
        entropy_loss = -torch.mean(
            self.log_alpha * (self.target_entropy - entropies)
            * weights)
        return entropy_loss

    def save_models(self, save_dir):
        super().save_models(save_dir)
        self.policy.save(os.path.join(save_dir, 'policy.pth'))
        self.online_critic.save(os.path.join(save_dir, 'online_critic.pth'))
        self.target_critic.save(os.path.join(save_dir, 'target_critic.pth'))


