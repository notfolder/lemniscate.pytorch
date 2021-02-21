import torch
from torch import nn
from lib.normalize import Normalize

class FeatureDecorrelation(nn.Module):
    def __init__(self, low_dim, tau2):
        super(FeatureDecorrelation, self).__init__()
        self.low_dim = torch.tensor(low_dim)
        self.tau2 = torch.tensor(tau2)
        self.l2norm = Normalize(2)
        self.seq = torch.tensor(torch.range(0, low_dim-1), dtype=torch.long)
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, features):
        #Vt = torch.t(features)
        Vt = self.l2norm(torch.t(features))
        #first = -torch.sum(torch.pow(Vt, 2), 1)/self.tau2
        inner_product = torch.mm(Vt, features)/self.tau2
        #print(inner_product.shape)
        #second = torch.logsumexp(inner_product, 1)
        #Lf = torch.sum(first + second)
        #Lf = torch.mean(first + second)
        Lf = self.ce(inner_product, self.seq)
        #old = self.forward_old(features)
        #print(old)
        #print(Lf)
        return Lf

    def forward_old(self, features):
        #Vt = torch.t(features)
        Vt = self.l2norm(torch.t(features))
        #Lf = torch.tensor(0.0)
        Lf = 0.0
        for l in range(self.low_dim):
            fl_t = torch.t(Vt[l])
            first = -(torch.dot(fl_t, Vt[l]))/self.tau2
            #sum_second = torch.tensor(0.0)
            sum_second = 0.0
            for j in range(self.low_dim):
                fj_t = torch.t(Vt[j])
                sum_second += torch.exp(torch.dot(fj_t, Vt[l])/self.tau2)
            second = torch.log(sum_second)
            Lf += first + second
        #return Lf
        return Lf / self.low_dim
