import torch
import torch.nn as nn


class PINNS_MLP(torch.nn.Module):
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
    

"""# 2D and time independent: R² -> R³
class PINNS_MLP_kovasznay(torch.nn.Module):
    def __init__(self):
        super().__init__()
        print("Done Model New")
        self.inner = nn.Sequential(
            nn.Linear(2, 128, bias = True),
            # tanh as there is second derivative in the pde part of the loss
            nn.Tanh(),
            nn.Linear(128, 128, bias = True),
            nn.Tanh(),
            nn.Linear(128, 128, bias = True),
            nn.Tanh(),
            nn.Linear(128, 128, bias = True),
            nn.Tanh(),
            nn.Linear(128, 128, bias = True),
            nn.Tanh(),
            nn.Linear(64, 64, bias = True),
            nn.Tanh(),
            # nn.Linear(50, 50),
            # nn.Tanh(),
            # nn.Linear(50, 50),
            # nn.Tanh(),
            # nn.Linear(50, 50),
            # nn.Tanh(),
            nn.Linear(64, 3, bias = True)
        )

    def forward(self, X):
        return self.inner(X)
    """

import torch
import torch.nn as nn
import torch.nn.functional as F

class PINNS_MLP_kovasznay(nn.Module):
    def __init__(self, input_dim = 2, hidden_dim = 256, output_dim = 3, L = 10):
        super().__init__()
        self.L = L  # Nombre de couches

        # Chemin pour U et V
        self.W1 = nn.Linear(input_dim, hidden_dim)  # W_1, b_1
        self.W2 = nn.Linear(input_dim, hidden_dim)  # W_2, b_2

        # Couche Z pour chaque k dans 1,...,L
        self.Wz = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(L)])
        self.bz = nn.ParameterList([nn.Parameter(torch.zeros(hidden_dim)) for _ in range(L)])

        # Couche finale W, b pour f_theta(x)
        self.final_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Chemins pour U et V
        U = torch.tanh(self.W1(x))  # U = phi(XW1 + b1)
        V = torch.tanh(self.W2(x))  # V = phi(XW2 + b2)

        # Initialisation de H
        H = torch.zeros_like(U)

        # Propagation à travers les L couches
        for k in range(self.L):
            Zk = torch.tanh(F.linear(H, self.Wz[k].weight, self.bz[k]))  # Z^(k) = phi(H Wz + bz)
            H = (1 - Zk) * U + Zk * V  # H^(k+1) = (1 - Z^(k)) ⊙ U + Z^(k) ⊙ V

        # Dernière couche pour f_theta(x)
        out = self.final_layer(H)
        return out


