from collections import deque
import numpy as np
import torch
import threading

class MultiStepBuff:

    def __init__(self, maxlen=3):
        super(MultiStepBuff, self).__init__()
        self.maxlen = int(maxlen)
        self.lock = threading.Lock()
        self.reset()

    def append(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def get(self, gamma=0.99):
        assert len(self.rewards) > 0
        state = self.states.popleft()
        action = self.actions.popleft()
        reward = self._nstep_return(gamma)
        return state, action, reward

    def _nstep_return(self, gamma):
        r = np.sum([r * (gamma ** i) for i, r in enumerate(self.rewards)])
        self.rewards.popleft()
        return r

    def reset(self):
        # Buffer to store n-step transitions.
        self.states = deque(maxlen=self.maxlen)
        self.actions = deque(maxlen=self.maxlen)
        self.rewards = deque(maxlen=self.maxlen)

    def is_empty(self):
        return len(self.rewards) == 0

    def is_full(self):
        return len(self.rewards) == self.maxlen

    def __len__(self):
        return len(self.rewards)


class LazyMemory(dict):

    def __init__(self, capacity, state_shape, device):
        super(LazyMemory, self).__init__()
        self.capacity = int(capacity)
        self.state_shape = state_shape
        self.device = device
        self.lock = threading.Lock()
        self._n = 0
        self._p = 0

    def reset(self):
        with self.lock:  # Ensure thread-safe access
            print("Resetting memory")
            self['state'] = []
            self['next_state'] = []
            self['action'] = np.empty((self.capacity, 1), dtype=np.int64)
            self['reward'] = np.empty((self.capacity, 1), dtype=np.float32)
            self['done'] = np.empty((self.capacity, 1), dtype=np.float32)

            self._n = 0
            self._p = 0

    def append(self, state, action, reward, next_state, done,
               episode_done=None):
        self._append(state, action, reward, next_state, done)

    def _append(self, state, action, reward, next_state, done):
        with self.lock:  # Ensure thread-safe access
        
            if 'state' not in self:
                self['state'] = []
            if 'next_state' not in self:
                self['next_state'] = []
            if 'action' not in self:
                self['action'] = np.empty((self.capacity, 1), dtype=np.int64)
            if 'reward' not in self:
                self['reward'] = np.empty((self.capacity, 1), dtype=np.float32)
            if 'done' not in self:
                self['done'] = np.empty((self.capacity, 1), dtype=np.float32)
            
            
            self['state'].append(state)
            self['next_state'].append(next_state)
            self['action'][self._p] = action
            self['reward'][self._p] = reward
            self['done'][self._p] = done
            
            #print(f"Full actions buffer up to index {self._p}: {self['action'][1]}")

            self._n = min(self._n + 1, self.capacity)
            self._p = (self._p + 1) % self.capacity

            self.truncate()

    def truncate(self):
        while len(self['state']) > self.capacity:
            del self['state'][0]
            del self['next_state'][0]
    
    def __len__(self):
        return self._n
    
    def sample(self, batch_size):
        
        if len(self) < batch_size:
            print("Not enough data to sample a full batch")
            return None

        # Ensure batch_size number of contiguous samples
        start_indices = np.arange(0, len(self) - batch_size + 1)
        selected_start = np.random.choice(start_indices, 1)[0]
        indices = np.arange(selected_start, selected_start + batch_size)
        return self._sample(indices, batch_size)

    def _sample(self, indices, batch_size):
        with self.lock:  # Ensure thread-safe access
            bias = -self._p if self._n == self.capacity else 0

            states = np.empty((batch_size, *self.state_shape), dtype=np.float32)
            next_states = np.empty((batch_size, *self.state_shape), dtype=np.float32)

            for i, index in enumerate(indices):
                _index = np.mod(index + bias, self.capacity)
                states[i, ...] = self['state'][_index]
                next_states[i, ...] = self['next_state'][_index]

            states = torch.tensor(states, dtype=torch.float32).to(self.device)
            next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
            actions = torch.tensor(self['action'][indices], dtype=torch.int64).to(self.device)
            rewards = torch.tensor(self['reward'][indices], dtype=torch.float32).to(self.device)
            dones = torch.tensor(self['done'][indices], dtype=torch.float32).to(self.device)

            #print(f"Batch Actions: {actions}")
            
            return {
                'states': states,
                'actions': actions,
                'rewards': rewards,
                'next_states': next_states,
                'dones': dones
            }
            

class LazyMultiStepMemory(LazyMemory):

    def __init__(self, capacity, state_shape, device, gamma=0.99,
                 multi_step=3):
        super(LazyMultiStepMemory, self).__init__(
            capacity, state_shape, device)

        self.gamma = gamma
        self.multi_step = int(multi_step)
        if self.multi_step != 1:
            self.buff = MultiStepBuff(maxlen=self.multi_step)

    def append(self, state, action, reward, next_state, done):
        if self.multi_step != 1:
            self.buff.append(state, action, reward)

            if self.buff.is_full():
                state, action, reward = self.buff.get(self.gamma)
                self._append(state, action, reward, next_state, done)

            if done:
                while not self.buff.is_empty():
                    state, action, reward = self.buff.get(self.gamma)
                    self._append(state, action, reward, next_state, done)
        else:
            self._append(state, action, reward, next_state, done)
