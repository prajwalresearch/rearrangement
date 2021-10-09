import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class DeepLagrangianNetwork(nn.Module):

    def __init__(self, q_dim, hidden_dim=64, device="cpu"):
        super().__init__()
        self.q_dim = q_dim
        self.q_dim_changed = int(0.5 * self.q_dim)
        # self.num_Lo = int(0.5 * (q_dim ** 2 - q_dim))
        self.num_Lo = int(0.5 * (self.q_dim_changed ** 2 - self.q_dim_changed))

        self.fc1 = nn.Linear(q_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        # Output layers
        # self.fc_G = nn.Linear(hidden_dim, self.q_dim_changed)


        self.fc_Ld = nn.Linear(hidden_dim, self.q_dim_changed)
        self.fc_Lo = nn.Linear(hidden_dim, self.num_Lo)
        
        self.drop_layer = nn.Dropout(p=0.3)
        self.act_fn = F.leaky_relu
        self.device = device

    
    def assemble_lower_triangular_matrix(self, Lo, Ld):

        assert (2 * Lo.shape[1] == (Ld.shape[1]**2 - Ld.shape[1]))

        diagonal_matrix = torch.diag_embed(Ld).cuda()
        L = torch.tril(torch.ones(*diagonal_matrix.shape, device=self.device)) - torch.eye(self.q_dim_changed).cuda() # gives lower triangle matrix. 

        # Set off diagonals
        L[L==1] = Lo.view(-1)
        # Add diagonals
        L = L + diagonal_matrix
        return L

    def forward(self, x):

        # q = torch.chunk(x, chunks=1, dim=1)
        q = x
        q = torch.reshape(q, (-1, 4))
        n, d = q.shape
        
        d_change = int(0.5 * ( d )) 
        hidden1 = self.act_fn(self.fc1(q))
        hidden2 = self.act_fn(self.fc2(hidden1))

        hidden2 = self.drop_layer(hidden2)
       
        hidden3 = self.fc_Ld(hidden2)
        Ld = F.softplus(hidden3).cuda()
        Lo = self.fc_Lo(hidden2).cuda()
        L = self.assemble_lower_triangular_matrix(Lo, Ld)
        H = L @ L.transpose(1, 2) + 1e-9 * torch.eye(d_change, device=self.device)
        return H

if __name__ == "__main__":

    network = DeepLagrangianNetwork(4, 64)

    test_input = torch.ones(1, 4)
    network(test_input)
