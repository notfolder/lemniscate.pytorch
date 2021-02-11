import torch
from torch import nn

class InstanceDiscrimination(nn.Module):
    def __init__(self, tau):
        super(InstanceDiscrimination, self).__init__()
        self.tau = tau
        self.ce_loss = nn.CrossEntropyLoss(reduction='sum')

    def forward(self, features):
        #print(features.shape)
        #old = self.forward_old(features)
        v = features
        n = torch.tensor(features.shape[0]).to(features.device)
        #Li = 0.0

        #print(den.shape)
        #print(num.shape)
        #for i in range(n):
        #    num = torch.exp(torch.dot(torch.t(v[i]),v[i])/self.tau)
            #den_old = 0.0
            #for j in range(n):
            #    den_old += torch.exp(torch.dot(torch.t(v[j]),v[i])/self.tau)
            #print(den_old)
            #print(den[i])
        #    Li += torch.log(num/den[i])
        #print((den/num).shape)
        numerator = torch.exp(torch.sum(torch.pow(v,2), dim=1)/self.tau)
        denominator = torch.sum(torch.exp(torch.mm(v, torch.t(v))/self.tau), dim=0)
        Li = -torch.sum(torch.log(numerator/denominator))/n
        #Li = self.ce_loss(torch.mm(v, torch.t(v))/self.tau, torch.tensor(range(n)).to(self.device))
        #print(old)
        #print(Li)
        return Li

    def forward_old(self, features):
        #print(features.shape)
        v = features
        n = features.shape[0]
        Li = 0.0
        for i in range(n):
            num = torch.exp(torch.dot(torch.t(v[i]),v[i])/self.tau)
            den = 0.0
            for j in range(n):
                den += torch.exp(torch.dot(torch.t(v[j]),v[i])/self.tau)
            Li += torch.log(num/den)
        return -Li
