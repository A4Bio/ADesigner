import torch
import torch.nn as nn
import torch.nn.functional as F


feat_dims = {
    'node': {
        'distance': 96,
    },
    'edge': {
        'distance': 256,
        'direction': 12,
    }
}


def _normalize(tensor, dim=-1):
    normed_tensor = torch.norm(tensor, dim=dim, keepdim=True)
    normed_tensor[normed_tensor == 0.0] = 1.0
    return torch.div(tensor, normed_tensor)


class Normalize(nn.Module):
    def __init__(self, features, epsilon=1e-6):
        super(Normalize, self).__init__()
        self.gain = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))
        self.epsilon = epsilon

    def forward(self, x, dim=-1):
        mu = x.mean(dim, keepdim=True)
        sigma = torch.sqrt(x.var(dim, keepdim=True) + self.epsilon)
        gain = self.gain
        bias = self.bias
        if dim != -1:
            shape = [1] * len(mu.size())
            shape[dim] = self.gain.size()[0]
            gain = gain.view(shape)
            bias = bias.view(shape)
        return gain * (x - mu) / (sigma + self.epsilon) + bias


class PIE(nn.Module):
    def __init__(self, edge_features, node_features, num_rbf=16):
        super(PIE, self).__init__()
        """Extract Protein Features"""
        self.num_rbf = num_rbf
        node_feat_types, edge_feat_types = ['distance'], ['distance', 'direction']
        node_in = sum([int(feat_dims['node'][feat]) for feat in node_feat_types])
        edge_in = sum([int(feat_dims['edge'][feat]) for feat in edge_feat_types])
        self.node_embedding = nn.Linear(node_in, node_features, bias=True)
        self.edge_embedding = nn.Linear(edge_in, edge_features, bias=True)
        self.norm_nodes = Normalize(node_features)
        self.norm_edges = Normalize(edge_features)

    def _rbf(self, D):
        D_min, D_max, D_count = 0., 20., self.num_rbf
        D_mu = torch.linspace(D_min, D_max, D_count, device=D.device).view([1,1,-1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)
        return torch.exp(-((D_expand - D_mu) / D_sigma)**2)

    def _get_rbf(self, A, B, E_idx=None):
        if E_idx is not None:
            src, dst = E_idx
            D_A_B_neighbors = torch.sqrt(torch.sum((A[src] - B[dst])**2,-1) + 1e-6).unsqueeze(-1)
            RBF_A_B = self._rbf(D_A_B_neighbors).squeeze(1)
        else:
            D_A_B = torch.sqrt(torch.sum((A[:,None,:] - B[:,None,:])**2,-1) + 1e-6)
            RBF_A_B = self._rbf(D_A_B)
        return RBF_A_B

    def _orientations_coarse(self, X, E_idx, eps=1e-6):
        src, dst = E_idx
        V = X.clone()
        X = X[:,:3,:].reshape(X.shape[0]*3, 3)
        dX = X[1:,:] - X[:-1,:] # CA-N, C-CA, N-C, CA-N...
        U = _normalize(dX, dim=-1)
        u_0, u_1 = U[:-2,:], U[1:-1,:]
        n_0 = _normalize(torch.cross(u_0, u_1), dim=-1)
        b_1 = _normalize(u_0 - u_1, dim=-1)
        
        n_0, b_1 = n_0[::3,:], b_1[::3,:]
        Q = torch.stack((b_1, n_0, torch.cross(b_1, n_0)), 2)
        Q = Q.view(list(Q.shape[:1]) + [9])
        Q = F.pad(Q, (0,0,0,1), 'constant', 0)

        Q_neighbors = Q[dst]
        list(map(lambda i: V[:, i, :][dst], [1, 0, 2, 3]))

        Q = Q.view(list(Q.shape[:1]) + [3,3]).unsqueeze(1)
        Q_neighbors = Q_neighbors.view(list(Q_neighbors.shape[:1]) + [3,3])

        dX = torch.stack(list(map(lambda i: V[:, i, :][dst], [1, 0, 2, 3])), dim=1) - V[src, 0].unsqueeze(1)
        dU = torch.matmul(Q[src], dX[...,None]).squeeze(-1)
        E_direct = _normalize(dU, dim=-1)
        E_direct = E_direct.reshape(E_direct.shape[0], -1)
        return E_direct

    def forward(self, X, E_in_idx=None, E_ex_idx=None):
        atom_N, atom_Ca, atom_C, atom_O = X.unbind(1)

        # node distance
        node_list = ['Ca-N', 'Ca-C', 'Ca-O', 'N-C', 'N-O', 'O-C']
        V_dist = []
        for pair in node_list:
            atom1, atom2 = pair.split('-')
            V_dist.append(self._get_rbf(vars()['atom_' + atom1], vars()['atom_' + atom2], None).squeeze())
        V_dist = torch.cat(tuple(V_dist), dim=-1).squeeze()
        # edge direction
        E_in_direct = self._orientations_coarse(X, E_in_idx)
        E_ex_direct = self._orientations_coarse(X, E_ex_idx)

        # edge distance
        edge_list = ['Ca-Ca', 'Ca-C', 'C-Ca', 'Ca-N', 'N-Ca', 'Ca-O', 'O-Ca', 'C-C', 'C-N', 'N-C', 'C-O', 'O-C', 'N-N', 'N-O', 'O-N', 'O-O']

        E_in_dist = [] 
        for pair in edge_list:
            atom1, atom2 = pair.split('-')
            E_in_dist.append(self._get_rbf(vars()['atom_' + atom1], vars()['atom_' + atom2], E_in_idx))
        E_in_dist = torch.cat(tuple(E_in_dist), dim=-1)

        E_ex_dist = [] 
        for pair in edge_list:
            atom1, atom2 = pair.split('-')
            E_ex_dist.append(self._get_rbf(vars()['atom_' + atom1], vars()['atom_' + atom2], E_ex_idx))
        E_ex_dist = torch.cat(tuple(E_ex_dist), dim=-1)

        h_V = self.norm_nodes(self.node_embedding(torch.cat([V_dist], dim=-1)))
        h_E_in = self.norm_edges(self.edge_embedding(torch.cat([E_in_dist, E_in_direct], dim=-1)))
        h_E_ex = self.norm_edges(self.edge_embedding(torch.cat([E_ex_dist, E_ex_direct], dim=-1)))
        return h_V, h_E_in, h_E_ex