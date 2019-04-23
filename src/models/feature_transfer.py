import torch.nn as nn
import torch.nn.functional as F

class FeatureTransfer(nn.Module):

    def __init__(self):
        super(FeatureTransfer, self).__init__()
        self.fc1 = nn.Linear(128, 256)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)

    def forward(self, inputs):
        batch_size = inputs.view(0)
        n_pairs = inputs.view(1)

        h = F.relu(self.fc1(inputs.view(batch_size * n_pairs)))
        h = self.dropout(h)
        h = F.relu(self.fc2(inputs))
        h = F.relu(self.fc3(inputs))

        return (h + inputs).view(batch_size, n_pairs)

