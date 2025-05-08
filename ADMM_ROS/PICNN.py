import numpy as np
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import time
import matplotlib.pyplot as plt
import psutil
import sys
from * from ADMM_utils

class NonNegativeLinear(nn.Module):   #重みが非負の全結合
    def __init__(self, in_features, out_features):
        super(NonNegativeLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        # 重みを非負に制限
        with torch.no_grad():
            self.linear.weight.data.clamp_(min=0)
        return self.linear(x)
    

class NonNegativeOutputLinear(nn.Module):   #重みが非負の全結合
    def __init__(self, in_features, out_features):
        super(NonNegativeOutputLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        # 重みを非負に制限
        return nn.functional.relu(self.linear(x))
  

class myPICNN(nn.Module):
    def __init__(self, num_input, hidden_size_u, hidden_size_z, score_scaler = None, output_size = 1):
        super(myPICNN, self).__init__()

        self.x_u1 = nn.Linear(1, hidden_size_u)
        # self.w0zu = nn.Linear(1, num_input)
        self.w0zu = NonNegativeOutputLinear(1, num_input)
        self.w0yu = nn.Linear(1, num_input)
        self.w0z = NonNegativeLinear(num_input, hidden_size_z)
        self.w0y = nn.Linear(num_input, hidden_size_z)
        self.w0u =nn.Linear(1, hidden_size_z)

        self.u1_u2 = nn.Linear(hidden_size_u, hidden_size_u)
        # self.w1zu = nn.Linear(hidden_size_u, hidden_size_z)
        self.w1zu = NonNegativeOutputLinear(hidden_size_u, hidden_size_z)
        self.w1yu = nn.Linear(hidden_size_u, num_input)
        self.w1z = NonNegativeLinear(hidden_size_z, hidden_size_z)
        self.w1y = nn.Linear(num_input, hidden_size_z)
        self.w1u =nn.Linear(hidden_size_u, hidden_size_z)

        self.u2_u3 = nn.Linear(hidden_size_u, hidden_size_u)
        # self.w2zu = nn.Linear(hidden_size_u, hidden_size_z)
        self.w2zu = NonNegativeOutputLinear(hidden_size_u, hidden_size_z)
        self.w2yu = nn.Linear(hidden_size_u, num_input)
        self.w2z = NonNegativeLinear(hidden_size_z, hidden_size_z)
        self.w2y = nn.Linear(num_input, hidden_size_z)
        self.w2u =nn.Linear(hidden_size_u, hidden_size_z)

        # self.w3zu = nn.Linear(hidden_size_u, hidden_size_z)
        # self.w3yu = nn.Linear(hidden_size_u, num_input)
        # self.w3z = nn.Linear(hidden_size_z, hidden_size_z)
        # self.w3y = nn.Linear(num_input, hidden_size_z)
        # self.w3u =nn.Linear(hidden_size_u, hidden_size_z)

        self.output = NonNegativeLinear(hidden_size_z, 1)

        # self.activation = nn.LeakyReLU(negative_slope=0.1)
        self.activation = nn.ReLU()
        self.score_scaler = score_scaler

    def forward(self, x, y):

        # u1 = torch.relu(self.x_u1(x))
        u1 = self.activation(self.x_u1(x))
        # z1_1 = self.w0z( y * self.w0zu(x) )
        z1_2 = self.w0y( y * ( self.w0yu(x) ) )
        z1_3 = self.w0u(x)
        # z1 = torch.relu(z1_1 + z1_2 + z1_3)
        # z1 = self.activation(z1_1 + z1_2 + z1_3)
        z1 = self.activation(z1_2 + z1_3)

        # u2 = torch.relu(self.u1_u2(u1))
        u2 = self.activation(self.u1_u2(u1))
        # print(self.w1zu(u1))
        z2_1 = self.w1z( z1 * self.w1zu(u1) )
        z2_2 = self.w1y( y * ( self.w1yu(u1) ) )
        z2_3 = self.w1u(u1)
        # z2 = torch.relu(z2_1 + z2_2 + z2_3)
        z2 = self.activation(z2_1 + z2_2 + z2_3)

        # u3 = torch.relu(self.u2_u3(u2))
        u3 = self.activation(self.u2_u3(u2))
        z3_1 = self.w2z( z2 * self.w2zu(u2) )
        z3_2 = self.w2y( y * ( self.w2yu(u2) ) )
        z3_3 = self.w2u(u2)
        # z3 = torch.relu(z3_1 + z3_2 + z3_3)
        z3 = self.activation(z3_1 + z3_2 + z3_3)

        # output = self.output(z3)
        output = self.output(z2)

        if self.score_scaler is not None and not self.training:
            # 逆正規化を実行
            original_output = self.score_scaler.inverse_transform(output.detach().numpy())
            return torch.tensor(original_output)

        return output