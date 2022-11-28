import torch
from torch.autograd import Variable
import torch.autograd

def gradient_penalty(Y_label, Y_predicted, discriminator, gp_weight):
        """
        Y_label: The expected results of the NN
        Y_predicted: The output of the NN
        discriminator: Determines which data is real
        gp_weight: reduces loss through weight clipping 
        """
        batch_size = Y_label.size()[0]

      
        alpha = torch.rand(batch_size, 1, 1, 1)
        alpha = alpha.expand_as(Y_label)
        interpolated = alpha * Y_label.data + (1 - alpha) * Y_predicted.data
        interpolated = Variable(interpolated, requires_grad=True)

        prob_interpolated = discriminator(interpolated)
        
        gradients = torch.autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(prob_interpolated.size()),
                               create_graph=True, retain_graph=True)[0]

      
        gradients = gradients.view(batch_size, -1)
        losses = []
        losses.append(gradients.norm(2, dim=1).mean().data[0])

        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

        return gp_weight * ((gradients_norm - 1) ** 2).mean()
