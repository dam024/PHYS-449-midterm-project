import torch
import torch.nn as nn
import NeuralNetwork as NN
import gradient_penalty as GP


class Critic(nn.Module):
    def __init__(self, input_channels=1, alpha=0.1):
        super(Critic, self).__init__()
        self.alpha = alpha
        self.input_size = input_channels

        self.main = nn.Sequential(

            # input, output, kernel, size (taken from Doogesh's code I don't see thing described in the paper, same with padding
            nn.Conv3d(input_channels, 8, kernel_size=(7, 7, 7),
                      stride=(1, 2, 1), padding='valid'),
            nn.LeakyReLU(negative_slope=self.alpha),

            nn.Conv3d(8, 16, kernel_size=(5, 5, 5),
                      stride=(1, 1, 1), padding='valid'),
            nn.LeakyReLU(negative_slope=self.alpha),

            nn.Conv3d(16, 32, kernel_size=(3, 3, 3),
                      stride=(1, 2, 1), padding='valid'),
            nn.LeakyReLU(negative_slope=self.alpha),

            nn.Conv3d(32, 64, kernel_size=(1, 1, 1),
                      stride=(1, 1, 1), padding='valid'),
            nn.LeakyReLU(negative_slope=self.alpha)
        )

        # for backproping gradient
        self.one = torch.tensor(1, dtype=torch.float)
        self.mone = (self.one*-1).to()  # for backproping gradient
        self.one = self.one.to(NN.NeuralNetwork.device())
        self.mone = self.mone.to(NN.NeuralNetwork.device())

    def forward(self, x):
        out = self.main(x)
        return torch.flatten(out)

    def backprop(self, data, generated, forwardCritic, optimizerCritic, params):
        # Train on real image
        self.zero_grad()
        c_loss_real = forwardCritic(data.y)
        C_loss_real = c_loss_real.mean()
        c_loss_real.backward(self.mone)
        # train on generated image
        c_loss_fake = forwardCritic(generated)
        c_loss_fake = c_loss_fake.mean()
        c_loss_fake.backward(self.one)

        # train with gradient penalty
        # gradient_penalty = GP.gradient_penalty(data.y,data.x,self.forwardCritic,params['gp_weight']) Damien : wrong input ?? according to comments, you should give the output of the generator as 2nd parameter
        gradient_penalty = GP.gradient_penalty(data.y, generated, self.forward, params['gp_weight'])
        gradient_penalty.backward()

        train_val = c_loss_fake-c_loss_real + gradient_penalty
        Wasserstein_D = c_loss_real - c_loss_fake
        optimizerCritic.step()
        return train_val

    def prepareForBackprop(self, generator):
        # Allow no weight change of generator
        for p in generator.parameters():
            p.requires_grad = False
        # allows weight update of critic
        for p in self.parameters():
            p.requires_grad = True
