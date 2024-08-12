import os
import sys
import numpy as np

from quantrl import utils

class Agent:
    # 에이전트 상태가 구성하는 값 개수
    STATE_DIM = 3

    # 매매 수수료 및 세금
    TRADING_CHARGE = 0.00015  # 거래 수수료 0.015%
    TRADING_TAX = 0.002  # 거래세 0.2%

    # 행동
    ACTION_BUY = 0  # 매수
    ACTION_SELL = 1  # 매도
    ACTION_HOLD = 2  # 관망
    ACTIONS = [ACTION_BUY, ACTION_SELL, ACTION_HOLD]
    NUM_ACTIONS = len(ACTIONS)  # 인공 신경망에서 고려할 출력값의 개수

    def __init__(self, environment, initial_balance, min_trading_price, max_trading_price):
        self.environment = environment
        self.initial_balance = initial_balance  # 초기 자본금

        self.min_trading_price = min_trading_price
        self.max_trading_price = max_trading_price

        # Agent 클래스의 속성
        self.balance = initial_balance  # 현재 현금 잔고
        self.num_stocks = 0  # 보유 주식 수
        self.portfolio_value = initial_balance
        self.num_buy = 0  # 매수 횟수
        self.num_sell = 0  # 매도 횟수
        self.num_hold = 0  # 관망 횟수

        # Agent 클래스의 상태
        self.ratio_hold = 0  # 주식 보유 비율
        self.profitloss = 0  # 손익률
        self.avg_buy_price = 0  # 주당 매수 단가
        self.past_portfolio_values = []

    def reset(self):
        self.balance = self.initial_balance
        self.num_stocks = 0
        self.portfolio_value = self.initial_balance
        self.num_buy = 0
        self.num_sell = 0
        self.num_hold = 0
        self.ratio_hold = 0
        self.profitloss = 0
        self.avg_buy_price = 0
        self.past_portfolio_values = []

    def set_balance(self, balance):
        self.initial_balance = balance

    def get_states(self):
        self.ratio_hold = self.num_stocks * self.environment.get_price() / self.portfolio_value
        return (
            self.ratio_hold,
            self.profitloss,
            (self.environment.get_price() / self.avg_buy_price) - 1 if self.avg_buy_price > 0 else 0
        )

    def decide_action(self, pred_value, pred_policy, epsilon):
        confidence = 0.

        pred = pred_policy
        if pred is None:
            pred = pred_value

        if pred is None:
            # 예측 값이 없을 경우 탐험
            epsilon = 1
        else:
            # 값이 모두 같은 경우 탐험
            maxpred = np.max(pred)
            if (pred == maxpred).all():
                epsilon = 1

            # if pred_policy is not None:
            #     if np.max(pred_policy) - np.min(pred_policy) < 0.05:
            #         epsilon = 1

        # 탐험 결정
        if np.random.rand() < epsilon:
            exploration = True
            action = np.random.randint(self.NUM_ACTIONS)
        else:
            exploration = False
            action = np.argmax(pred)

        confidence = .5
        if pred_policy is not None:
            confidence = pred[action]
        elif pred_value is not None:
            confidence = utils.sigmoid(pred[action])

        return action, confidence, exploration

    def validate_action(self, action):
        if action == Agent.ACTION_BUY:
            if self.balance < self.environment.get_price() * (1 + self.TRADING_CHARGE):
                return False
        elif action == Agent.ACTION_SELL:
            if self.num_stocks <= 0:
                return False
        return True

    def decide_trading_unit(self, confidence):
        if np.isnan(confidence):
            return self.min_trading_price
        added_trading_price = max(min(
            int(confidence * (self.max_trading_price - self.min_trading_price)),
            self.max_trading_price - self.min_trading_price), 0)
        trading_price = self.min_trading_price + added_trading_price
        return max(int(trading_price / self.environment.get_price()), 1)

    def calculate_reward(self):
        # Calculate the reward based on the change in portfolio value
        current_portfolio_value = self.portfolio_value
        previous_portfolio_value = self.past_portfolio_values[-1] if self.past_portfolio_values else self.initial_balance

        # Reward is the change in portfolio value
        reward = current_portfolio_value - previous_portfolio_value

        # Update the past portfolio values list
        self.past_portfolio_values.append(current_portfolio_value)

        # Normalize the reward (optional)
        reward = np.clip(reward / self.initial_balance, -1, 1)

        return reward

    def act(self, action, confidence):
        if not self.validate_action(action):
            action = Agent.ACTION_HOLD

        # 환경에서 현재 가격 얻기
        curr_price = self.environment.get_price()

        # 매수
        if action == Agent.ACTION_BUY:
            trading_unit = self.decide_trading_unit(confidence)
            balance = (
                self.balance - curr_price *
                (1 + self.TRADING_CHARGE) * trading_unit
            )
            if balance < 0:
                trading_unit = min(
                    int(self.balance / (curr_price * (1 + self.TRADING_CHARGE))),
                    int(self.max_trading_price / curr_price)
                )
            invest_amount = curr_price * (1 + self.TRADING_CHARGE) * trading_unit
            if invest_amount > 0:
                self.avg_buy_price = \
                    (self.avg_buy_price * self.num_stocks + curr_price * trading_unit) \
                    / (self.num_stocks + trading_unit)
                self.balance -= invest_amount
                self.num_stocks += trading_unit
                self.num_buy += 1

        # 매도
        elif action == Agent.ACTION_SELL:
            trading_unit = self.decide_trading_unit(confidence)
            trading_unit = min(trading_unit, self.num_stocks)
            invest_amount = curr_price * (
                1 - (self.TRADING_TAX + self.TRADING_CHARGE)) * trading_unit
            if invest_amount > 0:
                self.avg_buy_price = \
                    (self.avg_buy_price * self.num_stocks - curr_price * trading_unit) \
                    / (self.num_stocks - trading_unit) \
                    if self.num_stocks > trading_unit else 0
                self.num_stocks -= trading_unit
                self.balance += invest_amount
                self.num_sell += 1

        # 관망
        elif action == Agent.ACTION_HOLD:
            self.num_hold += 1

        # 포트폴리오 가치 갱신
        self.portfolio_value = self.balance + curr_price * self.num_stocks
        self.profitloss = self.portfolio_value / self.initial_balance - 1

        # Calculate the reward based on FinRL structure
        reward = self.calculate_reward()
        
        return reward * 10