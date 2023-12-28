#!/usr/bin/python
# -*- coding:utf-8 -*-
from functools import partial
import json
import os
from copy import deepcopy
import numpy as np
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

import torch
from torch.utils.data import DataLoader

from data import AAComplex, EquiAACDataset
from evaluation.rmsd import kabsch
from utils import check_dir
from utils.logger import print_log
from evaluation import compute_rmsd, tm_score


def set_cdr(cplx, seq, x, cdr='H3'):
    cdr = cdr.upper()
    cplx: AAComplex = deepcopy(cplx)
    chains = cplx.peptides
    cdr_chain_key = cplx.heavy_chain if 'H' in cdr else cplx.light_chain
    refined_chain = chains[cdr_chain_key]
    start, end = cplx.get_cdr_pos(cdr)
    start_pos, end_pos = refined_chain.get_ca_pos(start), refined_chain.get_ca_pos(end)
    start_trans, end_trans = x[0][1] - start_pos, x[-1][1] - end_pos
    # left to start of cdr
    for i in range(0, start):
        refined_chain.set_residue_translation(i, start_trans)
    # end of cdr to right
    for i in range(end + 1, len(refined_chain)):
        refined_chain.set_residue_translation(i, end_trans)
    # cdr 
    for i, residue_x, symbol in zip(range(start, end + 1), x, seq):
        center = residue_x[4] if len(residue_x) > 4 else None
        refined_chain.set_residue(i, symbol,
            {
                'N': residue_x[0],
                'CA': residue_x[1],
                'C': residue_x[2],
                'O': residue_x[3]
            }, center, gen_side_chain=False
        )
    new_cplx = AAComplex(cplx.pdb_id, chains, cplx.heavy_chain, cplx.light_chain,
                         cplx.antigen_chains, numbering=None, cdr_pos=cplx.cdr_pos,
                         skip_cal_interface=True)
    return new_cplx


def eval_one(tup, out_dir, cdr='H3'):
    cplx, seq, x, true_x, aligned = tup
    summary = {
        'pdb': cplx.get_id(),
        'heavy_chain': cplx.heavy_chain,
        'light_chain': cplx.light_chain,
        'antigen_chains': cplx.antigen_chains
    }
    # kabsch
    if aligned:
        ca_aligned = x[:, 1, :]
    else:
        ca_aligned, rotation, t = kabsch(x[:, 1, :], true_x[:, 1, :])
        x = np.dot(x - np.mean(x, axis=0), rotation) + t
    summary['RMSD'] = compute_rmsd(ca_aligned, true_x[:, 1, :], aligned=True)
    # set cdr
    new_cplx = set_cdr(cplx, seq, x, cdr)
    pdb_path = os.path.join(out_dir, cplx.get_id() + '.pdb')
    new_cplx.to_pdb(pdb_path)
    summary['TMscore'] = tm_score(cplx.get_heavy_chain(), new_cplx.get_heavy_chain())
    # AAR
    origin_seq = cplx.get_cdr(cdr).get_seq()
    hit = 0
    for a, b in zip(origin_seq, seq):
        if a == b:
            hit += 1
    aar = hit * 1.0 / len(origin_seq)
    summary['AAR'] = aar
    return summary


def rabd_test(args, model, test_set, test_loader, out_dir, device):
    print_log('Doing RAbD test')
    args.rabd_topk = min(args.rabd_topk, args.rabd_sample)
    global_best_ppl = [[1e10 for _ in range(args.rabd_topk)] for _ in range(len(test_set))]
    global_best_results = [[None for _ in range(args.rabd_topk)] for _ in range(len(test_set))]
    k_ids = [k for k in range(args.rabd_topk)]
    with torch.no_grad():
        for _ in tqdm(range(args.rabd_sample)):
            results, ppl = [], []
            for batch in test_loader:
                ppls, seqs, xs, true_xs, aligned = model.infer(batch, device)
                ppl.extend(ppls)
                results.extend([(seqs[i], xs[i], true_xs[i], aligned) for i in range(len(seqs))])
            for i, p in enumerate(ppl):
                max_ppl_id = max(k_ids, key=lambda k: global_best_ppl[i][k])
                if p < global_best_ppl[i][max_ppl_id]:
                    global_best_ppl[i][max_ppl_id] = p
                    global_best_results[i][max_ppl_id] = results[i]
                        
    if out_dir is None:
        ckpt_dir = os.path.split(args.ckpt)[0]
        out_dir = os.path.join(ckpt_dir, 'results')
    out_dir = os.path.join(out_dir, 'rabd_test')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    print_log(f'dumped to {out_dir}')

    cdr_type = 'H' + model.cdr_type
    heads = ['PPL', 'RMSD', 'TMscore', 'AAR']
    eval_res = [[] for _ in heads]
    for k in range(args.rabd_topk):
        inputs = [(cplx, ) + item[k] for cplx, item in zip(test_set.data, global_best_results)]
        k_out_dir = os.path.join(out_dir, str(k))
        if not os.path.exists(k_out_dir):
            os.makedirs(k_out_dir)
        summaries = process_map(partial(eval_one, out_dir=k_out_dir, cdr=cdr_type), inputs, max_workers=args.num_workers, chunksize=10)

        summary_fout = open(os.path.join(k_out_dir, 'summary.json'), 'w')
        for i, summary in enumerate(summaries):
            summary['PPL'] = global_best_ppl[i][k]
            summary_fout.write(json.dumps(summary) + '\n')
        summary_fout.close()

        for i, h in enumerate(heads):
            eval_res[i].extend([summary[h] for summary in summaries])
    
    eval_res = np.array(eval_res)
    means = np.mean(eval_res, axis=1)
    stdvars = np.std(eval_res, axis=1)
    print_log(f'Results for top {args.rabd_topk} candidates:')
    report_res = {}
    for i, h in enumerate(heads):
        print_log(f'{h}: mean {means[i]}, std {stdvars[i]}')
        report_res['rabd_' + h + '_mean'] = means[i]
        report_res['rabd_' + h + '_std'] = stdvars[i]
    return report_res


def average_test(args, model, test_set, test_loader, out_dir, device):
    heads, eval_res = ['PPL', 'RMSD', 'TMscore', 'AAR'], []
    for _round in range(args.run):
        print_log(f'round {_round}')
        results, ppl = [], []
        with torch.no_grad():
            for batch in tqdm(test_loader):
                ppls, seqs, xs, true_xs, aligned = model.infer(batch, device)
                ppl.extend(ppls)
                results.extend([(seqs[i], xs[i], true_xs[i], aligned) for i in range(len(seqs))])

        assert len(test_set) == len(results)

        inputs = [(cplx, ) + item for cplx, item in zip(test_set.data, results)]  # cplx, seq, x 

        check_dir(out_dir)
        print_log(f'dumped to {out_dir}')
        
        cdr_type = 'H' + model.cdr_type
        summaries = process_map(partial(eval_one, out_dir=out_dir, cdr=cdr_type), inputs, max_workers=args.num_workers, chunksize=10)

        summary_fout = open(os.path.join(out_dir, 'summary.json'), 'w')
        for i, summary in enumerate(summaries):
            summary['PPL'] = ppl[i]
            summary_fout.write(json.dumps(summary) + '\n')
        summary_fout.close()

        rmsds = [summary['RMSD'] for summary in summaries]
        tm_scores = [summary['TMscore'] for summary in summaries]
        aars = [summary['AAR'] for summary in summaries]
        ppl, rmsd, tm, aar = np.mean(ppl), np.mean(rmsds), np.mean(tm_scores), np.mean(aars)
        print_log(f'ppl: {ppl}, rmsd: {rmsd}, TM score: {tm}, AAR: {aar}')
        eval_res.append([ppl, rmsd, tm, aar])

    eval_res = np.array(eval_res)
    means = np.mean(eval_res, axis=0)
    stdvars = np.std(eval_res, axis=0)
    print_log(f'Results after {args.run} runs:')
    report_means = {'PPL': [], 'RMSD': [], 'TMscore': [], 'AAR': []}
    for i, h in enumerate(heads):
        report_means[h] = means[i]
        print_log(f'{h}: mean {means[i]}, std {stdvars[i]}')
    return report_means


def main(args):
    print(str(args))
    model = torch.load(args.ckpt, map_location='cpu')
    device = torch.device('cpu' if args.gpu == -1 else f'cuda:{args.gpu}')
    # model_type = get_model_type(args.ckpt)
    # print(f'model type: {model_type}')
    test_set = EquiAACDataset(args.test_set)
    test_set.mode = args.mode
    test_loader = DataLoader(test_set, batch_size=args.batch_size,
                             num_workers=args.num_workers,
                             shuffle=False,
                             collate_fn=test_set.collate_fn)
    model.to(device)
    model.eval()
    if args.rabd_test:
        rabd_test(args, model, test_set, test_loader, device)
    else:
        average_test(args, model, test_set, test_loader, device)
    
    # writing original structures
    print_log(f'Writing original structures')
    out_dir = os.path.join(args.out_dir, 'original')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for cplx in tqdm(test_set.data):
        pdb_path = os.path.join(out_dir, cplx.get_id() + '.pdb')
        cplx.to_pdb(pdb_path)