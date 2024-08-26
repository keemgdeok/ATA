class Environment:
    PRICE_IDX = 4  # 종가의 위치

    def __init__(self, chart_data=None):
        self.chart_data = chart_data
        self.observation = None
        self.idx = -1
        self.action_space = type('ActionSpace', (object,), {'n': 3})() 

    def reset(self):
        self.observation = None
        self.idx = -1

    def observe(self):
        if len(self.chart_data) > self.idx + 1:
            #print(f"self.idx: {self.idx}")
            self.idx += 1
            self.observation = self.chart_data.iloc[self.idx]
            return self.observation
        return None
    
    def step(self):
        if len(self.chart_data) > self.idx + 1:
            next_observation = self.chart_data.iloc[self.idx + 1]
            return next_observation.iloc[1:].tolist()
        return None
    
    def get_price(self):
        if self.observation is not None:
            return self.observation.iloc[self.PRICE_IDX]
        return None
    
    def get_past_prices(self, window_size=None):
        """Returns a list of past prices up to the current index.
        If window_size is provided, returns only the last `window_size` prices."""
        if self.idx == -1:
            return []
        if window_size is not None:
            start_idx = max(0, self.idx - window_size + 1)
        else:
            start_idx = 0
        return self.chart_data.iloc[start_idx:self.idx + 1, self.PRICE_IDX].tolist()
    
    @property
    def state_shape(self):
        return (len(self.chart_data.columns),)  # 상태 모양 반환
