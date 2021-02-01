import torch
from torch import nn

class FeatureDecorrelation(nn.Module):
    def __init__(self, low_dim, tau2):
        super(FeatureDecorrelation, self).__init__()
        self.low_dim = low_dim
        self.tau2 = tau2

    def forward(self, features):
        Vt = torch.t(features)
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
        return Lf
