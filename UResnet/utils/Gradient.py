import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

    
def plot_grad_flow(named_parameters, save_path):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow
    
    https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/7?u=sangwook.kim 

    -------
    Example
    
        >>> loss = self.criterion(outputs, labels)
        >>> loss.backward()
        >>> plot_grad_flow(model.named_parameters())
    
    '''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().detach().cpu().mean())
            max_grads.append(p.grad.abs().detach().cpu().max())
    plt.figure(figsize=(30, 20))
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    

if __name__=="__main__":
    pass
