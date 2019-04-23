import torch
import torch.nn as nn
import torch.nn.functional as F
from .domain_discriminator import DomainDiscriminator
from .feature_generator import FeatureGenerator
from .feature_transfer import FeatureTransfer


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()


        self.feature_generator = FeatureGenerator()
        self.feature_transfer = FeatureTransfer()
        self.d1 = DomainDiscriminator()
        self.d2 = DomainDiscriminator()


    def forward(self, x_s, x_t, y_s):
        '''
        x_s: [batch_size, n_pair, n_channels, width, height]
        x_t: [batch_size, n_channels, width, height]
        y_s: [batch_size, n_pair]
        '''

        batch_size = x_s.size(0)
        n_pairs = x_s.size(1)

        f_s = self.feature_generator(x_s)
        f_t = self.feature_generator(x_t)

        g_s = self.feature_transfer(f_s)


        d1_s = self.d1(f_s.view(batch_size * n_pairs))
        d1_t = self.d1(f_t)
        d1_g = self.d1(g_s.view(batch_size * n_pairs))

        d2_g = self.d2(g_s)
        d2_t = self.d2(f_t)

        vrf_f = self.verify_loss(f_s)
        vrf_g = self.verify_loss(g_s)

        d1 = torch.log(d1_g) + torch.log(1 - d1_t)
        adv = torch.log(d1_s)
        sep = torch.log(d2_s) + 0.5 * (torch.log(1 - d2_g) + torch.log(1 - d2_f))
        self.f_loss = 0.5 * (vrf_f + vrf_g) + self.lambda_1 * adv + self.lambda2 * sep
        self.g_loss = vrf_g + self.lambda2 * torch.log(1 - d2_s)

        self.loss = self.f_loss + self.g_loss
        return f_s, f_t


    def verify_loss(self, f):
        p = 0
        n_pair = f.size(1)
        for i in range(n_pair):
            for t in range(n_pair):
                p += torch.dot(f[:, i], f[:, t]) / torch.dot(f[:, i], f)
        return p.mean()

