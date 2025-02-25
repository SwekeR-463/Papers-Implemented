import torch
import math

def gelu(x):
    # constants for the approximation
    sqrt_2_over_pi = math.sqrt(2 / math.pi)
    alpha = 0.044715

    # compute the GELU approximation
    return 0.5 * x * (1 + torch.tanh(sqrt_2_over_pi * (x + alpha * torch.pow(x, 3))))

# example input tensor
x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])

# apply custom GELU
output_custom = gelu(x)
print("GELU:", output_custom)
