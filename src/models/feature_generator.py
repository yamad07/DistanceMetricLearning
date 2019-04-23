import torch.nn as nn
import torch.nn.functional as F
from .. import layers

class FeatureGenerator(nn.Module):

    def __init__(self):
        super(FeatureGenerator, self).__init__()
        self.l1 = layers.convolution(3, 32)
        self.l2= layers.convolution(32, 64)
        self.l3 = layers.convolution(64, 128)
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 128)

    def forward(self, inputs):
        '''
        inputs: [batch_size, n_pairs, ...image]
        '''
        batch_size = inputs.size(0)
        n_pairs = inputs.size(1)
        h = self.l1(inputs.view(batch_size * n_pairs))
        h = self.l2(h)
        h = self.l3(h)

        h = F.relu(self.fc1(h))
        h = self.fc2(h)
        return h.view(batch_size, n_pairs)
