import torch
import torch.utils.data as data
from sklearn import datasets
from torchvision import transforms

class DAMNIST(data.Dataset):

    def __init__(self):
        super(DAMNIST, self).__init__()
        mnist = datasets.fetch_mldata('MNIST original')
        x, y = mnist.data, mnist.target
        self.x_s = x[y < 5]
        self.x_t = x[y > 4]
        self.y_s = y[y < 5]

    def __getitem__(self, idx):
        x_s = torch.Tensor(self.x_s[idx].reshape(28, 28))
        x_t = torch.Tensor(self.x_t[idx].reshape(28, 28))
        y_s = torch.LongTensor([self.y_s[idx]])
        return x_s, x_t, y_s

    def __len__(self):
        return len(self.x_t)
