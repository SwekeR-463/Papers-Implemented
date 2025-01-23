import torch
import torch.nn as nn
import torch.nn.functional as F

class LightningAttention:
    def __init__(self, block_size):
        self.block_size = block_size
    
    def forward(self, Q, K , V):
        N, d = Q.size()
        B = self.block_size
        T = N // B
        
        assert N % B == 0 
        
        Q_blocks = Q.view(T, B, d)
        K_blocks = K.view(T, B, d)
        V_blocks = V.view(T, B, d)
        
        KV = torch.zeros(d, d, device=Q.device)  # accumulated KV product
        O = torch.zeros(N, d, device=Q.device) # output
        
        for t in range(T):
            # load current blocks
            Qt = Q_blocks[t]
            Kt = K_blocks[t]
            Vt = V_blocks[t]

            # compute intra-block attention
            O_intra = F.softmax(Qt @ Kt.T, dim=-1) @ Vt

            # compute inter-block attention
            O_inter = Qt @ KV

            # update KV accumulator
            KV += Kt.T @ Vt

            # combine results and get output
            O[t * B: (t + 1) * B] = O_intra + O_inter

        return O
    
    
N, d = 64, 128  
B = 16

# random Inputs
Q = torch.randn(N, d)
K = torch.randn(N, d)
V = torch.randn(N, d)

lightning_attention = LightningAttention(block_size=B)

output = lightning_attention.forward(Q, K, V)

print(output.shape)