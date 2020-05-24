import torch
import torch.nn as nn
import torch.nn.functional as F
class APReLU(nn.Module):
    def __init__(self, in_channels):
        super(APReLU, self).__init__()
        self.in_channels=in_channels
        self.gap_min_branch=nn.AdaptiveAvgPool2d(1)
        self.gap_max_branch=nn.AdaptiveAvgPool2d(1)
        self.bn_squeeze=nn.BatchNorm2d(self.in_channels)
        self.bn_excitation=nn.BatchNorm2d(self.in_channels)
        self.fc_squeeze=nn.Linear(self.in_channels*2,self.in_channels)
        self.fc_excitation=nn.Linear(self.in_channels,self.in_channels)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.fc_squeeze.weight, mode='fan_in')
        nn.init.kaiming_normal_(self.fc_excitation.weight, mode='fan_in')

    def forward(self, x):
        N, C, H, W = x.size()
        x_min = x-x.abs()
        x_max = F.relu(x)
        x_min_gap = self.gap_min_branch(x_min)
        x_max_gap = self.gap_max_branch(x_max)
        x_concat = torch.cat((x_min_gap,x_max_gap),dim=1).view(N,C*2)
        x_squeeze = self.fc_squeeze(x_concat).view(N,C,1,1)
        x_squeeze = self.bn_squeeze(x_squeeze)
        x_squeeze = F.relu(x_squeeze).view(N,C)
        x_excitation = self.fc_excitation(x_squeeze).view(N,C,1,1)
        x_excitation = self.bn_excitation(x_excitation)
        sigma = F.sigmoid(x_excitation)
        output = F.relu(x)+0.5*sigma.expand_as(x)*(x-x.abs())
        return output
