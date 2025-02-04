import torch
import torch.nn as nn
import torch.nn.functional as F


class PINNS_MLP(torch.nn.Module):
    """Simple MLP for the unsteady problem."""
    def __init__(self):
        super().__init__()
        self.inner = nn.Sequential(
            nn.Linear(3, 64),
            #tanh as there is secund derivative in the pde part of the loss
            nn.Tanh(),
            nn.Linear(64, 256),
            nn.Tanh(),
            nn.Linear(256, 64),
            nn.Tanh(),
            nn.Linear(64, 3)
        )

    def forward(self, X):
        return self.inner(X)
    

class PINNS_MLP_kovasznay(torch.nn.Module):
    """Simple MLP for the Kovasznay flow problem."""
    def __init__(self, num_hidden_layers=7, hidden_layer_size=50):
        super().__init__()
        layers = []
        
        #input layer
        layers.append(nn.Linear(2, hidden_layer_size))
        layers.append(nn.Tanh())
        
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(hidden_layer_size, hidden_layer_size))
            layers.append(nn.Tanh())
        
        #output layer
        layers.append(nn.Linear(hidden_layer_size, 3))
        
        self.inner = nn.Sequential(*layers)

    def forward(self, X):
        return self.inner(X)
    




class PINNS_MLP_ethz(nn.Module):
    """Novel architecture, please refer to the report for eplanation and references."""
    def __init__(self, input_dim = 2, hidden_dim = 256, output_dim = 3, L = 10):
        super().__init__()
        self.L = L  #number of layers

        #Path for U and V
        self.W1 = nn.Linear(input_dim, hidden_dim)  # W_1, b_1
        self.W2 = nn.Linear(input_dim, hidden_dim)  # W_2, b_2

        #Hidden layers
        self.Wz = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(L)])
        self.bz = nn.ParameterList([nn.Parameter(torch.zeros(hidden_dim)) for _ in range(L)])

        #output layer
        self.final_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Chemins pour U et V
        U = torch.tanh(self.W1(x))  # U = phi(XW1 + b1)
        V = torch.tanh(self.W2(x))  # V = phi(XW2 + b2)

        # Initialisation de H
        H = torch.zeros_like(U)

        #propagation
        for k in range(self.L):
            Zk = torch.tanh(F.linear(H, self.Wz[k].weight, self.bz[k]))  # Z^(k) = phi(H Wz + bz)
            H = (1 - Zk) * U + Zk * V  # H^(k+1) = (1 - Z^(k)) ⊙ U + Z^(k) ⊙ V

        #ouput
        out = self.final_layer(H)
        return out