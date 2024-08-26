import os
import sys
import time
import json
import logging
import threading
from tqdm import tqdm

from src.quantrl.sacd.memory.base import LazyMultiStepMemory, LazyMemory
from src.quantrl.sacd.agent.sacd import Sacd
from src.quantrl.agent import Agent
from src.quantrl.environment import Environment
from src.quantrl.visualizer import Visualizer
from src.quantrl import data_manager
from src.quantrl import dataloader
from src.quantrl import utils
from src.quantrl import settings


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