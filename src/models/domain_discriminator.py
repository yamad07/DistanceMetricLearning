import torch.nn as nn
import torch.nn.functional as F

class DomainDiscriminator(nn.Module):

    def __init__(self):
        super(DomainDiscriminator, self).__init__()
        self.l1 = nn.Linear(256, 128)
        self.l2 = nn.Linear(128, 256)
        self.l2 = nn.Linear(256, 1)

    def forward(self, inputs):
        h = F.relu(self.l1(inputs))
        h = F.relu(self.l2(h))
        h = self.l3(h)
        return F.sigmoid(h)
