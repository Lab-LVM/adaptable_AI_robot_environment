import torch
from torch import nn
from torch.nn import functional as F


class arcFace(nn.Module):
    def __init__(self, nembed, nclass, s=64, m=0.5):
        super().__init__()
        self.weight = nn.Parameter(nn.init.xavier_uniform_(torch.rand(nclass, nembed)))
        self.s = s
        self.m = m

    def forward(self, x, label=None):
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))

        if label is None:
            return cosine

        index = torch.where(label != -1)[0]
        m_hot = torch.zeros(index.size()[0], cosine.size()[1], device=cosine.device)
        m_hot.scatter_(1, label[index, None], self.m)
        cosine.acos_()
        cosine[index] += m_hot
        cosine.cos_().mul_(self.s)
        return cosine