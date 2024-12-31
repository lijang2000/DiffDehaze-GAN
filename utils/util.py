import random
import torch
import numpy as np
import torch.nn as nn

def synthesize_fog(J, t, A=None):
    """
    Synthesize hazy image base on optical model
    I = J * t + A * (1 - t)
    """

    if A is None:
        A = 1

    return J * t + A * (1 - t)

class TVLoss(nn.Module):
    '''
    Define Total Variance Loss for images
    which is used for smoothness regularization
    '''

    def __init__(self):
        super(TVLoss, self).__init__()

    def __call__(self, input):
        
        origin = input[:, :, :-1, :-1]
        right = input[:, :, :-1, 1:]
        down = input[:, :, 1:, :-1]

        tv = torch.mean(torch.abs(origin-right)) + torch.mean(torch.abs(origin-down))
        return tv * 0.5

class ReplayBuffer():
    def __init__(self, max_size=50):
        assert (max_size > 0)
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return torch.cat(to_return)



class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0)
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)
