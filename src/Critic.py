import torch
import torch.nn as nn

class Critic(nn.Module):
    def __init__(self, input_channels=1,alpha=0.1):
        super(Critic, self).__init__()
        self.alpha = alpha
        self.input_size = input_channels
        
        self.main = nn.Sequential(

        #input, output, kernel, size (taken from Doogesh's code I don't see thing described in the paper, same with padding
        nn.Conv3d(input_channels,8,kernel_size=(7,7,7),stride=(1,2,2,2,1),padding='valid'),
        nn.LeakyReLU(negative_slope=self.alpha),

        nn.Conv3d(8,16,kernel_size=(5,5,5),stride=(1,1,1,1,1),padding='valid'),
        nn.LeakyReLU(negative_slope=self.alpha),

        nn.Conv3d(16,32,kernel_size=(3,3,3),stride=(1,2,2,2,1),padding='valid'),
        nn.LeakyReLU(negative_slope=self.alpha),

        nn.Conv3d(32,64,kernel_size=(1,1,1),stride=(1,1,1,1,1),padding='valid'),
        nn.LeakyReLU(negative_slope=self.alpha)
        )

        def forward(self, x):
            out = self.main(x)
            return torch.flatten(out)
