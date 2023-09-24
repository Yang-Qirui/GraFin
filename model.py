import torch
import torch.nn as nn
import math

class GraphSAGE(nn.Module):
    '''
        This is a GNN layer based on "Inductive representation learning on large graphs"
    '''
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.activate = nn.LeakyReLU()
        self.weight = nn.Parameter(torch.randn(input_dim, output_dim, dtype=torch.double))

    def forward(self, A, h):
        new_h = torch.zeros_like(h)
        for i in range(A.shape[0]):
            neighbors = torch.nonzero(A[i, :], as_tuple=True)[0]
            count_neighbors = len(neighbors)
            # Sum the features of the neighbors
            if count_neighbors > 0:
                temp_val = (torch.sum(h[neighbors], dim=0) + h[i]) / (count_neighbors + 1)
            else:
                temp_val = h[i]
            # Apply the weight and activation
            new_h[i] = self.activate(torch.mm(self.weight, torch.transpose(temp_val.unsqueeze(0), 0,))).squeeze()
        return new_h

class GraFinModel(nn.Module):
    
    def __init__(self, gnn_num, landmark_num):
        super().__init__()
        self.gnn_layers = nn.ModuleList([GraphSAGE(landmark_num, landmark_num) for _ in range(gnn_num)])
        self.mlp = nn.Sequential(nn.Linear(landmark_num, 400, dtype=torch.double), nn.LeakyReLU(), nn.Linear(400, 70, dtype=torch.double), nn.LeakyReLU()) # default setting

    def forward(self, adjacent_matrix, features):
        out = features
        for gnn_layer in self.gnn_layers:
            out = gnn_layer(adjacent_matrix, out)
        return self.mlp(out)

class GraFinLoss(nn.Module):

    def __init__(self, alpha) -> None:
        super().__init__()
        self.alpha = alpha

    def cal_Y_tMY(self, Y_t, matrix, Y):
        return torch.mm(torch.mm(Y_t, matrix), Y) 

    def _forward(self, Y, D, L):
        Y_t = torch.transpose(Y, 0, 1)
        Y_tLY = self.cal_Y_tMY(Y_t, L, Y)
        Y_tDY = self.cal_Y_tMY(Y_t, D, Y)
        loss_1 = torch.trace(Y_tLY) / torch.trace(Y_tDY) 
        loss_2 = torch.norm(Y_tDY / torch.norm(Y_tDY, p="fro") - torch.eye(Y_tDY.shape[0]) / math.sqrt(Y_tDY.shape[0]), p="fro")
        return loss_1 + loss_2

    def forward(self, Y, D_rp2rp, L_rp2rp, D_rp2ap, L_rp2ap):
        return self.alpha * self._forward(Y[:D_rp2rp.shape[0]], D_rp2rp, L_rp2rp) + (1 - self.alpha) * self._forward(Y, D_rp2ap, L_rp2ap)