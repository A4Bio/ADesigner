import torch
import torch.nn as nn
import torch.nn.functional as F


class CGMLP(nn.Module):
    def __init__(self, input_nf, output_nf, hidden_nf, n_channel, dropout=0.1):
        super(CGMLP, self).__init__()

        input_edge = input_nf * 2
        node_in = hidden_nf + input_nf
        edge_in = input_edge + hidden_nf * 2 + hidden_nf

        self.dropout = nn.Dropout(dropout)

        self.node_mlp = nn.Sequential(
            nn.Linear(node_in, hidden_nf),
            nn.SiLU(),
            nn.Linear(hidden_nf, output_nf))
    
        self.edge_row_mlp = nn.Sequential(
            nn.Linear(edge_in, hidden_nf),
            nn.SiLU(),
            nn.Linear(hidden_nf, hidden_nf))
        
        self.edge_col_mlp = nn.Sequential(
            nn.Linear(edge_in, hidden_nf),
            nn.SiLU(),
            nn.Linear(hidden_nf, hidden_nf))

        self.coord_pls = nn.Sequential(
                nn.Linear(input_edge + n_channel**2, hidden_nf),
                nn.SiLU(),
                nn.Linear(hidden_nf, hidden_nf))

        layer = nn.Linear(hidden_nf, n_channel, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_nf, hidden_nf),
            nn.SiLU(),
            layer
        )

    def node_model(self, x, edge_index, edge_attr):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))
        agg = torch.cat([x, agg], dim=1)
        out = self.node_mlp(agg)
        out = x + out
        return out

    def coord_model(self, coord, edge_index, coord_diff, edge_feat):
        row, col = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat).unsqueeze(-1)
        agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        coord = coord + agg
        return coord

    def forward(self, h, edge_index, coord, h_sv=None, h_se=None):
        row, col = edge_index
        radial, coord_diff = coord2radial(edge_index, coord)

        out = torch.cat([h[row], h[col], h_sv[row], h_sv[col], h_se], dim=1)
        row_out = self.edge_row_mlp(out)
        col_out = self.edge_col_mlp(out.clone())
        edge_feat = h[col] * torch.sigmoid(row_out) + h[row] * torch.sigmoid(col_out)

        radial = radial.reshape(radial.shape[0], -1)
        r_out = torch.cat([h[row], h[col], radial], dim=1)
        r_out = self.coord_pls(r_out)

        coord = self.coord_model(coord, edge_index, coord_diff, r_out)
        h = self.node_model(h, edge_index, edge_feat)
        return h, coord


class Model(nn.Module):
    def __init__(self, in_node_nf, hidden_nf, out_node_nf, n_channel,
                n_layers=3, dropout=0.1):
        super().__init__()
        '''
        :param in_node_nf: Number of features for 'h' at the input
        :param hidden_nf: Number of hidden features
        :param out_node_nf: Number of features for 'h' at the output
        :param n_channel: Number of channels of coordinates
        :param in_edge_nf: Number of features for the edge features
        :param act_fn: Non-linearity
        :param n_layers: Number of layer for the EGNN
        :param residual: Use residual connections, we recommend not changing this one
        :param dropout: probability of dropout
        '''
        self.hidden_nf = hidden_nf
        self.n_layers = n_layers

        self.dropout = nn.Dropout(dropout)

        self.linear_in = nn.Linear(in_node_nf, self.hidden_nf)
        self.linear_out = nn.Linear(self.hidden_nf, out_node_nf)

        self.norm = nn.BatchNorm1d(hidden_nf)

        for i in range(0, n_layers):
            self.add_module(f'cgmlp_intra_{i}', CGMLP(
                self.hidden_nf, self.hidden_nf, self.hidden_nf, n_channel,
            ))
            self.add_module(f'cgmlp_inter_{i}', CGMLP(
                self.hidden_nf, self.hidden_nf, self.hidden_nf, n_channel,
            ))
    
    def forward(self, h, x, ctx_edges, att_edges, H_sv=None, ctx_H_se=None, int_H_se=None):
        h = self.linear_in(h)
        h = self.norm(h + self.dropout(h))

        for i in range(0, self.n_layers):
            h, x = self._modules[f'cgmlp_intra_{i}'](h, ctx_edges, x, h_sv=H_sv, h_se=ctx_H_se)
            h, x = self._modules[f'cgmlp_inter_{i}'](h, att_edges, x, h_sv=H_sv, h_se=int_H_se)

        h = self.dropout(h)
        h = self.linear_out(h)
        return h, x


def coord2radial(edge_index, coord):
    row, col = edge_index
    coord_diff = coord[row] - coord[col]  # [n_edge, n_channel, d]
    radial = torch.bmm(coord_diff, coord_diff.transpose(-1, -2))  # [n_edge, n_channel, n_channel]
    # normalize radial
    radial = F.normalize(radial, dim=0)  # [n_edge, n_channel, n_channel]
    return radial, coord_diff


def unsorted_segment_sum(data, segment_ids, num_segments):
    '''
    :param data: [n_edge, *dimensions]
    :param segment_ids: [n_edge]
    :param num_segments: [bs * n_node]
    '''
    expand_dims = tuple(data.shape[1:])
    result_shape = (num_segments, ) + expand_dims
    for _ in expand_dims:
        segment_ids = segment_ids.unsqueeze(-1)
    segment_ids = segment_ids.expand(-1, *expand_dims)
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    result.scatter_add_(0, segment_ids, data)
    return result


def unsorted_segment_mean(data, segment_ids, num_segments):
    '''
    :param data: [n_edge, *dimensions]
    :param segment_ids: [n_edge]
    :param num_segments: [bs * n_node]
    '''
    expand_dims = tuple(data.shape[1:])
    result_shape = (num_segments, ) + expand_dims
    for _ in expand_dims:
        segment_ids = segment_ids.unsqueeze(-1)
    segment_ids = segment_ids.expand(-1, *expand_dims)
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)


def get_edges(n_nodes):
    rows, cols = [], []
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                rows.append(i)
                cols.append(j)
    edges = [rows, cols]
    return edges


def get_edges_batch(n_nodes, batch_size):
    edges = get_edges(n_nodes)
    edge_attr = torch.ones(len(edges[0]) * batch_size, 1)
    edges = [torch.LongTensor(edges[0]), torch.LongTensor(edges[1])]
    if batch_size == 1:
        return edges, edge_attr
    elif batch_size > 1:
        rows, cols = [], []
        for i in range(batch_size):
            rows.append(edges[0] + n_nodes * i)
            cols.append(edges[1] + n_nodes * i)
        edges = [torch.cat(rows), torch.cat(cols)]
    return edges, edge_attr