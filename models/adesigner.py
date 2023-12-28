import torch
from torch import nn
import torch.nn.functional as F
from torch_scatter import scatter_sum

from data import VOCAB
from .model import Model
from .pie import PIE


esm_all_toks = ['<cls>', '<pad>', '<eos>', '<unk>', 'L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D', 
'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C', 'X', 'B', 'U', 'Z', 'O', '.', '-', '<null_1>', '<mask>']
voc_toks = ['G', 'A', 'V', 'L', 'I', 'F', 'W', 'Y', 'D', 'H', 'N', 'E', 'K', 'Q', 'M', 'R', 'S', 'T', 'C', 'P', 'U']

esm2ind = [-1] * len(voc_toks)
for ii, key in enumerate(voc_toks):
    esm_pos = esm_all_toks.index(key)
    esm2ind[ii] = esm_pos


def diff_l1_loss(pred,
                     target,
                     weight=None,
                     eps=1e-8,
                     reduction='mean',
                     **kwargs):
    loss = torch.sqrt((pred - target) ** 2 + eps)
    if weight is not None:
        loss *= weight.expand_as(loss)
    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    return loss


def sequential_and(*tensors):
    res = tensors[0]
    for mat in tensors[1:]:
        res = torch.logical_and(res, mat)
    return res


def sequential_or(*tensors):
    res = tensors[0]
    for mat in tensors[1:]:
        res = torch.logical_or(res, mat)
    return res


class ProteinFeature(nn.Module):

    def __init__(self):
        super().__init__()
        # global nodes and mask nodes
        self.boa_idx = VOCAB.symbol_to_idx(VOCAB.BOA)
        self.boh_idx = VOCAB.symbol_to_idx(VOCAB.BOH)
        self.bol_idx = VOCAB.symbol_to_idx(VOCAB.BOL)

        # segment ids
        self.ag_seg_id, self.hc_seg_id, self.lc_seg_id = 1, 2, 3

    def _is_global(self, S):
        return sequential_or(S == self.boa_idx, S == self.boh_idx, S == self.bol_idx)  # [N]

    def _construct_segment_ids(self, S):
        # construct segment ids. 1/2/3 for antigen/heavy chain/light chain
        glbl_node_mask = self._is_global(S)
        glbl_nodes = S[glbl_node_mask]
        boa_mask, boh_mask, bol_mask = (glbl_nodes == self.boa_idx), (glbl_nodes == self.boh_idx), (glbl_nodes == self.bol_idx)
        glbl_nodes[boa_mask], glbl_nodes[boh_mask], glbl_nodes[bol_mask] = self.ag_seg_id, self.hc_seg_id, self.lc_seg_id
        segment_ids = torch.zeros_like(S)
        segment_ids[glbl_node_mask] = glbl_nodes - F.pad(glbl_nodes[:-1], (1, 0), value=0)
        segment_ids = torch.cumsum(segment_ids, dim=0)
        return segment_ids

    @torch.no_grad()
    def construct_edges(self, X, S, batch_id, segment_ids=None):
        '''
        Memory efficient with complexity of O(Nn) where n is the largest number of nodes in the batch
        '''
        # construct tensors to map between global / local node index
        lengths = scatter_sum(torch.ones_like(batch_id), batch_id)  # [bs]
        N, max_n = batch_id.shape[0], torch.max(lengths)
        offsets = F.pad(torch.cumsum(lengths, dim=0)[:-1], pad=(1, 0), value=0)  # [bs]
        # global node index to local index. lni2gni can be implemented as lni + offsets[batch_id]
        gni = torch.arange(N, device=batch_id.device)
        gni2lni = gni - offsets[batch_id]  # [N]

        # prepare inputs
        if segment_ids is None:
            segment_ids = self._construct_segment_ids(S)

        ctx_edges, inter_edges = [], []

        # all possible edges (within the same graph)
        # same bid (get rid of self-loop and none edges)
        same_bid = torch.zeros(N, max_n, device=batch_id.device)
        same_bid[(gni, lengths[batch_id] - 1)] = 1
        same_bid = 1 - torch.cumsum(same_bid, dim=-1)
        # shift right and pad 1 to the left
        same_bid = F.pad(same_bid[:, :-1], pad=(1, 0), value=1)
        same_bid[(gni, gni2lni)] = 0  # delete self loop
        row, col = torch.nonzero(same_bid).T  # [2, n_edge_all]
        col = col + offsets[batch_id[row]]  # mapping from local to global node index
        
        # not global edges 
        # CHECK if it is NOT in [antigen, heavy chain, light chain]
        is_global = sequential_or(S == self.boa_idx, S == self.boh_idx, S == self.bol_idx) # [N]
        row_global, col_global = is_global[row], is_global[col]
        not_global_edges = torch.logical_not(torch.logical_or(row_global, col_global))

        # all possible ctx edges: same seg, not global
        row_seg, col_seg = segment_ids[row], segment_ids[col]
        select_edges = torch.logical_and(row_seg == col_seg, not_global_edges) 
        # CHECK if it is in [antigen, heavy chain, light chain], and the two nodes belong to the same category
        ctx_all_row, ctx_all_col = row[select_edges], col[select_edges]
        # ctx edges
        ctx_edges = _radial_edges(X, torch.stack([ctx_all_row, ctx_all_col]).T, cutoff=8.0)

        # all possible inter edges: not same seg, not global 
        # CHECK the internal edges between antigen, heavy chain, and light chain
        select_edges = torch.logical_and(row_seg != col_seg, not_global_edges)
        inter_all_row, inter_all_col = row[select_edges], col[select_edges]
        inter_edges = _radial_edges(X, torch.stack([inter_all_row, inter_all_col]).T, cutoff=12.0)

        # edges between global and normal nodes
        select_edges = torch.logical_and(row_seg == col_seg, torch.logical_not(not_global_edges))
        global_normal = torch.stack([row[select_edges], col[select_edges]])  # [2, nE]
        # edges between global and global nodes
        select_edges = torch.logical_and(row_global, col_global) # self-loop has been deleted
        global_global = torch.stack([row[select_edges], col[select_edges]])  # [2, nE]

        # add additional edge to neighbors in 1D sequence (except epitope)
        select_edges = sequential_and(
            torch.logical_or((row - col) == 1, (row - col) == -1),  # adjacent in the graph
            not_global_edges,          # not global edges (also ensure the edges are in the same segment)
            row_seg != self.ag_seg_id  # not epitope
        )
        seq_adj = torch.stack([row[select_edges], col[select_edges]])  # [2, nE]

        ctx_edges = torch.cat([ctx_edges, global_normal, global_global, seq_adj], dim=1)  # [2, E]
        return ctx_edges, inter_edges
    
    def forward(self, X, S, offsets):
        batch_id = torch.zeros_like(S)
        batch_id[offsets[1:-1]] = 1
        batch_id.cumsum_(dim=0)
        return self.construct_edges(X, S, batch_id)


def _radial_edges(X, src_dst, cutoff):
    dist = X[:, 1][src_dst]  # [Ef, 2, 3], CA position
    dist = torch.norm(dist[:, 0] - dist[:, 1], dim=-1) # [Ef]
    src_dst = src_dst[dist <= cutoff]
    src_dst = src_dst.transpose(0, 1)  # [2, Ef]
    return src_dst


class ADesigner(nn.Module):
    def __init__(self, embed_size, hidden_size, n_channel, n_layers=3, cdr_type='3', alpha=0.1, dropout=0.1, args=None):
        super().__init__()

        self.num_aa_type = len(VOCAB)
        self.cdr_type = cdr_type
        self.mask_token_id = VOCAB.get_unk_idx()
        self.alpha = alpha

        self.aa_embedding = nn.Embedding(self.num_aa_type, embed_size)
        self.model = Model(embed_size, hidden_size, self.num_aa_type,
            n_channel, n_layers=n_layers, dropout=dropout)        
        self.protein_feature = ProteinFeature()
        self.pcie = PIE(hidden_size, hidden_size)

        self.args = args

    def seq_loss(self, _input, target):
        return F.cross_entropy(_input, target, reduction='none')

    def coord_loss(self, _input, target):
        return diff_l1_loss(_input, target, reduction='sum')

    def init_mask(self, X, S, cdr_range):
        '''
        set coordinates of masks following a unified distribution
        between the two ends
        '''
        X, S, cmask = X.clone(), S.clone(), torch.zeros_like(X, device=X.device)
        n_channel, n_dim = X.shape[1:]
        for start, end in cdr_range:
            S[start:end + 1] = self.mask_token_id
            l_coord, r_coord = X[start - 1], X[end + 1]  # [n_channel, 3]
            n_span = end - start + 2
            coord_offsets = (r_coord - l_coord).unsqueeze(0).expand(n_span - 1, n_channel, n_dim)  # [n_mask, n_channel, 3]
            coord_offsets = torch.cumsum(coord_offsets, dim=0)
            mask_coords = l_coord + coord_offsets / n_span
            X[start:end + 1] = mask_coords
            cmask[start:end + 1, ...] = 1
        return X, S, cmask

    def forward(self, X, S, L, offsets):
        '''
        :param X: [n_all_node, n_channel, 3]
        :param S: [n_all_node]
        :param L: list of cdr types
        :param offsets: [batch_size + 1]
        '''
        # prepare inputs
        cdr_range = torch.tensor(
            [(cdr.index(self.cdr_type), cdr.rindex(self.cdr_type)) for cdr in L],
            dtype=torch.long, device=X.device
        ) + offsets[:-1].unsqueeze(-1)

        # save ground truth
        true_X, true_S = X.clone(), S.clone()

        X, S, cmask = self.init_mask(X, S, cdr_range)  # [n_all_node, n_channel, 3]
        mask = cmask[:, 0, 0].bool()  # [n_all_node]
        aa_cnt = mask.sum()
        H_0 = self.aa_embedding(S)  # [n_all_node, embed_size]

        snll, closs = 0, 0
        with torch.no_grad():
            ctx_edges, inter_edges = self.protein_feature(X, S, offsets)

        H_sv, ctx_H_se, int_H_se = self.pcie(X, E_in_idx=ctx_edges, E_ex_idx=inter_edges)
        H, Z = self.model(H_0, X, ctx_edges, inter_edges, H_sv=H_sv, ctx_H_se=ctx_H_se, int_H_se=int_H_se)
        X[mask] = Z[mask] # update the coords
        logits = H[mask]
        snll = torch.sum(self.seq_loss(logits, true_S[mask])) / aa_cnt
        
        closs = self.coord_loss(Z[mask], true_X[mask]) / aa_cnt
        loss = snll + self.alpha * closs
        return loss, snll, closs  # only return the last snll

    def generate(self, X, S, L, offsets, greedy=True):
        '''
        :param X: [n_all_node, n_channel, 3]
        :param S: [n_all_node]
        :param L: list of cdr types
        :param offsets: [batch_size + 1]
        '''
        # prepare inputs
        cdr_range = torch.tensor(
            [(cdr.index(self.cdr_type), cdr.rindex(self.cdr_type)) for cdr in L],
            dtype=torch.long, device=X.device
        ) + offsets[:-1].unsqueeze(-1)

        # save ground truth
        true_X, true_S = X.clone(), S.clone()

        # init mask
        X, S, cmask = self.init_mask(X, S, cdr_range)
        mask = cmask[:, 0, 0].bool()  # [n_all_node]
        aa_cnt = mask.sum()

        special_mask = torch.tensor(VOCAB.get_special_mask(), device=S.device, dtype=torch.long)
        smask = special_mask.repeat(aa_cnt, 1).bool()
        H_0 = self.aa_embedding(S)  # [n_all_node, embed_size]
        
        with torch.no_grad():
            ctx_edges, inter_edges = self.protein_feature(X, S, offsets)
        
        H_sv, ctx_H_se, int_H_se = self.pcie(X, E_in_idx=ctx_edges, E_ex_idx=inter_edges)
        H, Z = self.model(H_0, X, ctx_edges, inter_edges, H_sv=H_sv, ctx_H_se=ctx_H_se, int_H_se=int_H_se)
        X[mask] = Z[mask] # update the coords
        logits = H[mask]  # [aa_cnt, vocab_size]
        logits = logits.masked_fill(smask, float('-inf'))  # mask special tokens

        if greedy:
            S[mask] = torch.argmax(logits, dim=-1)  # [n]
        else:
            prob = F.softmax(logits, dim=-1)
            S[mask] = torch.multinomial(prob, num_samples=1).squeeze()
        snll_all = self.seq_loss(logits, S[mask])
        return snll_all, S, X, true_X, cdr_range

    def infer(self, batch, device, greedy=True):
        X, S, L, offsets = batch['X'].to(device), batch['S'].to(device), batch['L'], batch['offsets'].to(device)
        snll_all, pred_S, pred_X, true_X, cdr_range = self.generate(
            X, S, L, offsets, greedy=greedy
        )
        pred_S, cdr_range = pred_S.tolist(), cdr_range.tolist()
        pred_X, true_X = pred_X.cpu().numpy(), true_X.cpu().numpy()
        # seqs, x, true_x
        seq, x, true_x = [], [], []
        for start, end in cdr_range:
            end = end + 1
            seq.append(''.join([VOCAB.idx_to_symbol(pred_S[i]) for i in range(start, end)]))
            x.append(pred_X[start:end])
            true_x.append(true_X[start:end])
        # ppl
        ppl = [0 for _ in range(len(cdr_range))]
        lens = [0 for _ in ppl]
        offset = 0
        for i, (start, end) in enumerate(cdr_range):
            length = end - start + 1
            for t in range(length):
                ppl[i] += snll_all[t + offset]
            offset += length
            lens[i] = length
        ppl = [p / n for p, n in zip(ppl, lens)]
        ppl = torch.exp(torch.tensor(ppl, device=device)).tolist()
        return ppl, seq, x, true_x, True