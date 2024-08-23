### ------------------ SAC w/ Discrete Actions ------------------ ###
import os
import sys
import time
import json
import logging
import threading
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
from collections import deque

from quantrl.sacd.memory.base import LazyMultiStepMemory, LazyMemory
from quantrl.sacd.agent.sacd import Sacd
from quantrl.agent import Agent
from quantrl.environment import Environment
from quantrl.visualizer import Visualizer
from quantrl import data_manager
from quantrl import dataloader
from quantrl import utils
from quantrl import settings

import torch


logger = logging.getLogger(settings.LOGGER_NAME)


class SacdLearner(Sacd):
    lock = threading.Lock()
    def __init__(self, env, output_path, initial_balance, min_trading_price, max_trading_price, 
                 chart_data, training_data, num_steps=1, batch_size=60, memory_size=1000000, 
                 gamma=0.99, multi_step=1, target_entropy_ratio=0.99, start_steps=20000, update_interval=5, 
                 target_update_interval=500, use_per=False, dueling_net=True, num_eval_steps=125000, 
                 max_episode_steps=27000, log_interval=10, eval_interval=1000, cuda=True, seed=0, 
                 gen_output=True,
                 *args, **kwargs):
        super().__init__(env, output_path, num_steps, batch_size, memory_size, gamma, multi_step, 
                         target_entropy_ratio, start_steps, update_interval, target_update_interval, use_per, 
                         num_eval_steps, max_episode_steps, log_interval, eval_interval, cuda, seed)
        
        self.env = env # Environment(chart_data=chart_data)
        self.agent = Agent(self.env, initial_balance, min_trading_price, max_trading_price)
        self.chart_data = chart_data
        self.training_data = training_data
        self.memory = LazyMemory(memory_size, (57,), self.device)

        self.num_steps = num_steps
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.multi_setp = multi_step
        self.target_entropy_ratio = target_entropy_ratio
        self.start_steps = start_steps
        self.update_interval = update_interval
        self.target_update_interval = target_update_interval
        self.gen_output = gen_output
        self.output_path = output_path
        
        self.training_data_idx = -1
        self.num_buy = 0
        self.num_sell = 0
        self.num_hold = 0
        self.loss = 0.0
    

    def run(self, learning=True):
        info = (
            f'[{self.env}] RL:SACD '
        )
        with self.lock:
            logger.debug(info)

        if self.gen_output:
            self.epoch_summary_dir = os.path.join(self.output_path, f'epoch_summary')
            if not os.path.isdir(self.epoch_summary_dir):
                os.makedirs(self.epoch_summary_dir)
            else:
                for f in os.listdir(self.epoch_summary_dir):
                    os.remove(os.path.join(self.epoch_summary_dir, f))

        min_portfolio_value = 1e8
        max_portfolio_value = 0
        epoch_win_cnt = 0
        
        # rewards = []
        q1_losses = []
        q2_losses = []
        policy_losses = []
        entropy_losses = []
        profitloss = []
        actions = []
        avg_buy_prices = []
        self.reset()
        
        # Initialize total losses for the epoch
        q1_loss, q2_loss, policy_loss, entropy_loss = 0, 0, 0, 0
        
        for step in tqdm(range(len(self.training_data)), leave=False):
            state = self.build_sample()
            if state is None:
                print("next_sample is None")
                exit()

            action, confidence = self.explore(state)
            reward = self.agent.act(action, confidence)

            next_state = None
            if len(self.training_data) > self.training_data_idx + 1:
                next_state = self.training_data.iloc[self.training_data_idx+1].tolist()
            
            done = next_state is None
            if done:
                next_state = state[:]
            self.memory.append(state, action, reward, next_state, done)
            actions.append(action)
            avg_buy_prices.append(self.agent.avg_buy_price)
        
            if learning and (step+1) % self.update_interval == 0 and step > self.batch_size:
                q1_loss, q2_loss, policy_loss, entropy_loss = self.learn()
                tqdm.write(f"step {step+1}/{len(self.training_data)}, Q1 Loss: {q1_loss:.4f}, Q2 Loss: {q2_loss:.4f}, Policy Loss: {policy_loss:.4f}, Entropy Loss: {entropy_loss:.4f}")
        
                # Collect losses for plotting
                q1_losses.append(q1_loss)
                q2_losses.append(q2_loss)
                policy_losses.append(policy_loss)
                entropy_losses.append(entropy_loss)
            
            #print(f"self.target_update_interval: {self.target_update_interval}")
            if (step+1) % self.target_update_interval == 0:
                self.update_target()
        
        
            action_name = 'Hold'
            if action == 0:
                action_name = 'Buy '
            elif action == 1:
                action_name = 'Sell'
                
            logger.debug(f'[STEP {step}/{len(self.training_data)}] '
                f'Action:{action_name} Intensity:{confidence.item():.2f} '
                f'Stocks:{self.agent.num_stocks} Current Price:{self.env.get_price():,.0f} Average Buy Price:{int(self.agent.avg_buy_price):,.0f} '
                f'Balance:{self.agent.balance:,.0f} PV:{self.agent.portfolio_value:,.0f}')
            
            min_portfolio_value = min(min_portfolio_value, self.agent.portfolio_value)
            max_portfolio_value = max(max_portfolio_value, self.agent.portfolio_value)
            profitloss.append(self.agent.portfolio_value)
            
            if self.agent.portfolio_value > self.agent.initial_balance:
                epoch_win_cnt += 1

        logger.debug(f'[{self.env}]'
            f'Max PV:{max_portfolio_value:,.0f} Min PV:{min_portfolio_value:,.0f} #Win:{epoch_win_cnt}')
        tqdm.write(f"Max PV:{max_portfolio_value:,.0f} Min PV:{min_portfolio_value:,.0f} #Win:{epoch_win_cnt}")
        xx=0
        while xx <= self.batch_size:
            xx += self.update_interval 
        
        # Plot the losses
        x = np.arange(xx, len(self.training_data)+1, self.update_interval)
        fig, axs = plt.subplots(4, 1, figsize=(40, 30))
        # Q1 Losses
        axs[0].plot(x, q1_losses, label='Q1 Loss')
        axs[0].plot(x, q2_losses, label='Q2 Loss')
        axs[0].set_xlabel('Epochs')
        axs[0].set_ylabel('Loss')
        axs[0].legend()
        axs[0].set_title('Q1 and Q2 Losses over epochs')

        # Policy Losses
        axs[1].plot(x, policy_losses, label='Policy Loss', color='orange')
        axs[1].set_xlabel('Epochs')
        axs[1].set_ylabel('Loss')
        axs[1].legend()
        axs[1].set_title('Policy Loss over epochs')

        # Entropy Losses
        axs[2].plot(x, entropy_losses, label='Entropy Loss', color='red')
        axs[2].set_xlabel('Epochs')
        axs[2].set_ylabel('Loss')
        axs[2].legend()
        axs[2].set_title('Entropy Loss over epochs')
        
        step_profit = np.arange(0, len(self.training_data))
        normalized_profitloss = np.array(profitloss) / 1e8
        normalized_close = np.array(chart_data['Close'] / chart_data.loc[0, 'Close'])
        actions = np.array(actions)
        avg_buy_prices = np.array(avg_buy_prices)
        avg_buy_prices = avg_buy_prices / chart_data.loc[0, 'Close']
        axs[3].plot(step_profit, normalized_profitloss, label='profitloss', color='green')
        axs[3].plot(step_profit, normalized_close, label='close', color='black')
        axs[3].plot(step_profit, avg_buy_prices, label='avg_buy_prices', color='orange')
        axs[3].scatter(step_profit[actions == 0], normalized_close[actions == 0], marker='o', color='r', label='Buy', s=10)
        axs[3].scatter(step_profit[actions == 1], normalized_close[actions == 1], marker='o', color='b', label='Sell', s=10)
        axs[3].set_xlabel('Epochs')
        axs[3].set_ylabel('Profitloss vs Close')
        axs[3].legend(loc='upper left')
        axs[3].set_title('Reward over epochs')
        
        
        # 그래프 저장
        #plt.tight_layout()
        plt.savefig('/mnt/c/Users/keemg/dev/qd/output/test_scad/losses_over_epochs.png')

    # def stock_close(self):
    #     close = []
    #     for i in range(len(self.training_data)):
    #         close.append(chart_data['Close'])
    #         print(chart_data['Close'])
            
    #     return close
            

    def predict(self):
        self.agent.reset()
        q_sample = deque(maxlen=self.num_steps)
        result = []
        while True:
            next_sample = self.build_sample()
            if next_sample is None:
                break
            
            q_sample.append(next_sample)
            if len(q_sample) < self.num_steps:
                continue

            action, confidence = self.exploit(torch.tensor(q_sample, dtype=torch.float32))
            result.append((self.env.observation[0], action))

        if self.gen_output:
            with open(os.path.join(self.output_path, f'pred.json'), 'w') as f:
                print(json.dumps(result), file=f)

        return result
        
    def build_sample(self):
        self.env.observe()
        if len(self.training_data) > self.training_data_idx + 1:
            self.training_data_idx += 1
            self.sample = self.training_data.iloc[self.training_data_idx].tolist()
            #print(self.sample)
            #self.sample.extend(self.agent.get_states())
            return self.sample
        return None

    def save_models(self):
        super().save_models(self.output_path)
        self.policy.save(os.path.join(self.output_path, 'policy.pth'))
        self.online_critic.save(os.path.join(self.output_path, 'online_critic.pth'))
        self.target_critic.save(os.path.join(self.output_path, 'target_critic.pth'))

    def reset(self):
        self.sample = None
        self.training_data_idx = -1
        self.env.reset()
        self.agent.reset()
        self.itr_cnt = 0
        self.exploration_cnt = 0
        
        
if __name__ == "__main__":
    
    output_name = 'test_scad'
    output_path = os.path.join(settings.BASE_DIR, 'output', output_name)

    if not os.path.isdir(output_path):
        os.makedirs(output_path)
        
    log_path = os.path.join(output_path, f'{output_name}.log')
    if os.path.exists(log_path):
        os.remove(log_path)
    logging.basicConfig(format='%(message)s')
    logger = logging.getLogger(settings.LOGGER_NAME)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    file_handler = logging.FileHandler(filename=log_path, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
            
    chart_data, training_data = dataloader.load_data('000660') # 000660 # 005930
    env = Environment(chart_data=chart_data)

    learner = SacdLearner(env=env, output_path=output_path, 
                                    initial_balance=100000000, 
                                    min_trading_price=100000,
                                    max_trading_price=100000000, 
                                    chart_data=chart_data, training_data=training_data,
                                    use_per=False)
    
    learner.run(learning=True)
    
    # print(f'chart_data.loc[0, "Close"]: {chart_data.loc[0, "Close"]}')
    # normalized_data = list(chart_data['Close'] / chart_data.loc[0, 'Close'])
    # print(f'normalized_data: {normalized_data}')

   