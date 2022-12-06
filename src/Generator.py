import torch as torch
import torch.nn as nn

class Inception(nn.Module):
    '''single inception block for the generator'''

    def __init__(self, input_filter_size):
        super(Inception, self).__init__()
        #Branch A
        self.in_filter = input_filter_size
        self.branch_1x1x1_5x5x5 = nn.Conv3d(self.in_filter,10,1,padding = 'valid')
        self.branch_5x5x5       = nn.Conv3d(10,10,5,padding = 'valid')
        #Branch B
        self.branch_1x1x1_3x3x3 = nn.Conv3d(self.in_filter,10,1,padding = 'valid')
        self.branch_3x3x3       = nn.Conv3d(10,10,3,padding = 'valid')
        
        #Branch C
        self.brach_1x1x1        = nn.Conv3d(self.in_filter,10,1,padding = 'valid')

        self.l_relu = nn.LeakyReLU(negative_slope = 0.1)
    def forward(self,x):
        #x in shape [N x filters x D x H x W]
        branchA = self.branch_1x1x1_5x5x5(x)
        branchA = self.branch_5x5x5(branchA)

        branchB = self.branch_1x1x1_3x3x3(x)
        branchB = self.branch_3x3x3(branchB)
        branchB = branchB[:,:,1:(branchA.size(2)+1),1:(branchA.size(3)+1),1:(branchA.size(4)+1)]

        branchC =  self.brach_1x1x1(x)
        branchC = branchC[:,:,2:(branchA.size(2)+2),2:(branchA.size(3)+2),2:(branchA.size(4)+2)]

        outputs = [branchA, branchB, branchC]
        #outputs in shape [N x 30 x D' x H' x W']
        output  = torch.cat(outputs, dim = 1 )
        #element wise addition of previous layer
        # x must be resized for proper addition

        x_out = torch.stack([x[:,0,:,:,:] for i in range(output.size(1))]) #10xNxDxHxW
        x_out = torch.permute(x_out,[1,0,2,3,4] ) #Nx10xDxHxW
        x_out = x_out[:,:,2:(branchA.size(2)+2),2:(branchA.size(3)+2),2:(branchA.size(4)+2)]
        output = torch.add(output,x_out)
        
        return output

class Generator(nn.Module):
    def __init__(self,num_convs: int = 2, num_layers: int =4,initial_filter_num: int =1):
        super(Generator, self).__init__()
        self.num_convs = num_convs
        self.num_layers = num_layers

        self.inception1 = Inception(initial_filter_num)
        self.inception2 = Inception(30)
        self.conv1      = nn.Conv3d(30,10,1, padding = 'same')
        self.conv2      = nn.Conv3d(10,10,1, padding = 'same')
        self.conv3      = nn.Conv3d(10,1,1, padding = 'same')
        self.fc1         = nn.LeakyReLU(negative_slope = 0.1) 
        self.fc2        = nn.ReLU()

    def forward(self,x):
        x = self.inception1(x)
        x = self.fc1(x)

        for i in range(self.num_convs -1):
            x = self.inception2(x)
            x = self.fc1(x)

        x = self.conv1(x)
        x = self.fc1(x)
        for i in range(self.num_layers - 2):
        
            x = self.conv2(x)
            x = self.fc1(x)

        x = self.conv3(x)
        

        x_final = self.fc2(x)

        return x_final

    # Execute the backward propagation on the Generator. Warning : do not call forward method.
    # parameters :
    #   - inputs : input data for the generator
    #   - critic : the critic used 
    #   - optimizer : the optimizer we should use 
    def backprop(self, data, forwardCritic,optimizerGenerator,mone):
        self.zero_grad()
        generated = self.forward(data.x)
        train_val = forwardCritic(generated)
        train_val.backward(mone)
        optimizerGenerator.step()

        return train_val

    def prepareForBackprop(self,critic):
        #Allow no weight change of critic
        for p in critic.parameters():
            p.requires_grad = False
            #allows weight update of generator
        #for p in self.parameters():
            #p.requires_grad = True

        