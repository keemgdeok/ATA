import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '/src'))

if os.environ.get('RLTRADER_BACKEND', 'pytorch') == 'pytorch':
    print('Enabling PyTorch...')
    from quantrl.networks.networks_pytorch import Network, DNN, LSTMNetwork, CNN
else:
    print('Enabling TensorFlow...')
    from quantrl.networks.networks_keras import Network, DNN, LSTMNetwork, CNN

__all__ = [
    'Network', 'DNN', 'LSTMNetwork', 'CNN'
]
