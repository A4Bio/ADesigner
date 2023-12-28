import wandb
import torch
import pickle
import logging
import os
import os.path as osp
from parser import create_parser
import numpy as np

import warnings
warnings.filterwarnings('ignore')

from utils.main_utils import set_seed, output_namespace, check_dir
from utils.logger import print_log

from torch.utils.data import DataLoader
from models import ADesigner
from data import EquiAACDataset, ITAWrapper, AAComplex
from generate import average_test, rabd_test
from tqdm import tqdm
from ita_train import valid_check
from generate import set_cdr
from evaluation.rmsd import kabsch
from evaluation import pred_ddg

from trainer import TrainConfig, Trainer

import json
import time
import random


class Exp:
    def __init__(self, args):
        self.args = args
        self.config = args.__dict__
        print_log(output_namespace(self.args))

    def _get_data(self, train_path, valid_path, cdr_type):
        args = self.args
        ########## load your train / valid set ##########
        train_set = EquiAACDataset(train_path)
        train_set.mode = args.mode
        valid_set = EquiAACDataset(valid_path)
        valid_set.mode = args.mode

        ########## set your collate_fn ##########
        _collate_fn = EquiAACDataset.collate_fn

        # use a single gpu by default
        if len(args.gpus) > 1:
            torch.cuda.set_device(args.local_rank)
            torch.distributed.init_process_group(backend='nccl', world_size=len(args.gpus))
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, shuffle=args.shuffle, seed=args.seed)
            args.batch_size = int(args.batch_size / len(args.gpus))
            if args.local_rank == 0:
                print_log(f'Batch size on a single GPU: {args.batch_size}')
        else:
            train_sampler = None
        train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              num_workers=args.num_workers,
                              shuffle=(args.shuffle and train_sampler is None),
                              sampler=train_sampler,
                              collate_fn=_collate_fn)
        valid_loader = DataLoader(valid_set, batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                collate_fn=_collate_fn)
        
        n_channel = valid_set[0]['X'].shape[1]
        model = ADesigner(
            args.embed_size, args.hidden_size, n_channel, n_layers=args.n_layers, 
            cdr_type=cdr_type, alpha=args.alpha, dropout=args.dropout, args=args
        )
        return train_loader, valid_loader, model

    def _save(self, name=''):
        torch.save(self.trainer.model.state_dict(), osp.join(self.checkpoints_path, name + '.pth'))
        fw = open(osp.join(self.checkpoints_path, name + '.pkl'), 'wb')
        state = self.trainer.scheduler.state_dict()
        pickle.dump(state, fw)

    def _load(self, epoch):
        self.trainer.model.load_state_dict(torch.load(osp.join(self.checkpoints_path, str(epoch) + '.pth')))
        fw = open(osp.join(self.checkpoints_path, str(epoch) + '.pkl'), 'rb')
        state = pickle.load(fw)
        self.trainer.scheduler.load_state_dict(state)

    def _set_logfile(self, save_dir):
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(level=logging.INFO, filename=osp.join(save_dir, 'log.log'),
            filemode='a', format='%(asctime)s - %(message)s')

    def _generate(self, data_dir, save_dir, test_func):
        # generate
        model = torch.load(osp.join(save_dir, 'best.ckpt'))
        device = torch.device('cuda')
        test_set = EquiAACDataset(osp.join(data_dir, 'test.json'))
        test_set.mode = args.mode
        test_loader = DataLoader(test_set, batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            collate_fn=test_set.collate_fn)
        model.eval()

        out_dir = osp.join(save_dir, 'results', 'original')
        check_dir(out_dir)
        report_res = test_func(args, model, test_set, test_loader, out_dir, device)

        # writing original structures
        print_log(f'Writing original structures')
        for cplx in tqdm(test_set.data):
            pdb_path = osp.join(out_dir, cplx.get_id() + '.pdb')
            cplx.to_pdb(pdb_path)
        return report_res

    def k_fold_train(self):
        args = self.args
        # CDR 1, 2, 3
        for i in range(1, 4):
            print_log('CDR {}'.format(i))
            # fold 0, ..., 9
            for k in range(10):
                data_dir = osp.join(args.root_dir, 'cdrh{}'.format(i), 'fold_{}'.format(k))
                save_dir = osp.join(data_dir, 'ckpt', args.ex_name + '_CDR{}'.format(i) + '_{}'.format(args.mode))

                check_dir(save_dir)
                self._set_logfile(save_dir)
                train_loader, valid_loader, model = self._get_data(osp.join(data_dir, 'train.json'), osp.join(data_dir, 'valid.json'), cdr_type=str(i))
                print_log(f'parameters: {sum([p.numel() for p in model.parameters()])}') 

                config = TrainConfig(args, save_dir, args.lr, args.max_epoch, grad_clip=args.grad_clip, early_stop=args.early_stop, anneal_base=args.anneal_base) 
                trainer = Trainer(model, train_loader, valid_loader, config, cdr=i, fold=k, wandb=args.wandb)
                trainer.train(args.gpus, args.local_rank)

    def k_fold_eval(self):
        args = self.args
        # CDR 1, 2, 3
        sum_cdr_arr = 0.0

        for i in range(1, 4):
            print_log('CDR {}'.format(i))
            res_dict = {'PPL': [], 'RMSD': [], 'TMscore': [], 'AAR': []}
            # fold 0, ..., 9
            for k in range(10):
                data_dir = osp.join(args.root_dir, 'cdrh{}'.format(i), 'fold_{}'.format(k))
                save_dir = osp.join(data_dir, 'ckpt', args.ex_name + '_CDR{}'.format(i) + '_{}'.format(args.mode))
                report_res = self._generate(data_dir, save_dir, average_test)
                for key in res_dict.keys():
                    res_dict[key].append(report_res[key])

            write_buffer = {}
            for key in res_dict.keys():
                vals = res_dict[key]
                val_mean, val_std = np.mean(vals), np.std(vals)
                write_buffer['CDR{0}_'.format(i) + key + '_mean'] = val_mean
                write_buffer['CDR{0}_'.format(i) + key + '_std'] = val_std
                print_log(f'{key}: mean {val_mean}, std {val_std}')
            if args.wandb:
                wandb.log(write_buffer)
            sum_cdr_arr += write_buffer['CDR{0}_'.format(i) + 'AAR_mean']
        if args.wandb:
            wandb.log({'final': sum_cdr_arr})

    def cdr3_train(self, data_dir, save_dir):
        args = self.args
        check_dir(save_dir)
        self._set_logfile(save_dir)
        train_loader, valid_loader, model = self._get_data(osp.join(data_dir, 'train.json'), osp.join(data_dir, 'valid.json'), cdr_type='3')

        config = TrainConfig(args, save_dir, args.lr, args.max_epoch, grad_clip=args.grad_clip, early_stop=args.early_stop, anneal_base=args.anneal_base) 
        trainer = Trainer(model, train_loader, valid_loader, config, cdr=3, fold=0, wandb=args.wandb)
        trainer.train(args.gpus, args.local_rank)

    def rabd_eval(self, data_dir, save_dir):
        args = self.args
        report_res = self._generate(data_dir, save_dir, rabd_test)
        if args.wandb:
            wandb.log(report_res)

    def ita_train(self, data_dir, pretrain_dir, ita_save_dir):
        args = self.args
        check_dir(ita_save_dir)
        model = torch.load(osp.join(pretrain_dir, 'best.ckpt'))
        device = torch.device('cuda')

        dataset = EquiAACDataset(osp.join(data_dir, 'skempi_all.json'))
        dataset.mode = '111'
        itawrapper = ITAWrapper(dataset, args.n_samples)
        origin_cplx = [dataset.data[i] for i in dataset.idx_mapping]

        valid_loader = DataLoader(dataset, batch_size=args.ita_batch_size * args.update_freq,
                        num_workers=args.num_workers, shuffle=False, collate_fn=dataset.collate_fn)
        config = TrainConfig(args, ita_save_dir, args.lr, args.ita_epoch, grad_clip=args.grad_clip)
        with open(osp.join(ita_save_dir, 'train_config.json'), 'w') as fout:
            json.dump(config.__dict__, fout)

        def fake_log(*args, **kwargs):
            return

        origin_cplx_paths = []
        out_dir = osp.join(save_dir, 'original')
        check_dir(out_dir)

        print_log(f'Writing original structures to {out_dir}')
        for cplx in tqdm(origin_cplx):
            pdb_path = osp.join(out_dir, cplx.get_id() + '.pdb')
            cplx.to_pdb(pdb_path)
            origin_cplx_paths.append(osp.abspath(pdb_path))
        log = open(osp.join(save_dir, 'log.txt'), 'w')
        best_round, best_score = -1, 1e10

        for r in range(args.ita_n_iter):
            save_best = False
            res_dir = osp.join(save_dir, f'iter_{r}')
            check_dir(res_dir)
            # generate better samples
            print_log('Generating samples')
            model.eval()
            scores = []
            for i in tqdm(range(len(dataset))):
                origin_input = dataset[i]
                inputs = [origin_input for _ in range(args.n_tries)]
                candidates, results = [], []
                with torch.no_grad():
                    batch = dataset.collate_fn(inputs)
                    ppls, seqs, xs, true_xs, aligned = model.infer(batch, device, greedy=False)
                    results.extend([(ppls[i], seqs[i], xs[i], true_xs[i], aligned) for i in range(len(seqs))])
                recorded, candidate_pool = {}, []
                for n, (ppl, seq, x, true_x, aligned) in enumerate(results):
                    if seq in recorded:
                        continue
                    recorded[seq] = True
                    if ppl > 10:
                        continue
                    if not valid_check(seq):
                        continue
                    if not aligned:
                        ca_aligned, rotation, t = kabsch(x[:, 1, :], true_x[:, 1, :])
                        x = np.dot(x - np.mean(x, axis=0), rotation) + t
                    candidate_pool.append((ppl, seq, x, n))
                sorted_cand_idx = sorted([j for j in range(len(candidate_pool))], key=lambda j: candidate_pool[j][0])
                for j in sorted_cand_idx:
                    ppl, seq, x, n = candidate_pool[j]
                    new_cplx = set_cdr(origin_cplx[i], seq, x, cdr='H' + str(model.cdr_type))
                    pdb_path = osp.join(res_dir, new_cplx.get_id() + f'_{n}.pdb')
                    new_cplx.to_pdb(pdb_path)
                    new_cplx = AAComplex(
                        new_cplx.pdb_id, new_cplx.peptides,
                        new_cplx.heavy_chain, new_cplx.light_chain,
                        new_cplx.antigen_chains)
                    try:
                        score = pred_ddg(origin_cplx_paths[i], osp.abspath(pdb_path))
                    except Exception as e:
                        print_log(f'ddg prediction failed: {str(e)}', level='ERROR')
                        score = 0
                    if score < 0:
                        candidates.append((new_cplx, score))
                        scores.append(score)
                    if len(candidates) >= args.n_samples:
                        break
                while len(candidates) < args.n_samples:
                    candidates.append((origin_cplx[i], 0))
                    scores.append(0)
                itawrapper.update_candidates(i, candidates)

            itawrapper.finish_update()

            mean_score = np.mean(scores)
            if mean_score < best_score:
                best_round, best_score = r - 1, mean_score
                save_best = True
            log_line = f'model from iteration {r - 1}, ddg mean {mean_score}, std {np.std(scores)}, history best {best_score} at round {best_round}'
            print_log(log_line)
            log.write(log_line + '\n')
            if args.wandb:
                wandb.log({'ddg_mean': mean_score, 'ddg_std': np.std(scores)})

            # train
            print_log(f'Iteration {r}, result directory: {res_dir}')
            print_log(f'Start training')
            model.train()
            train_loader = DataLoader(itawrapper, batch_size=args.ita_batch_size,
                num_workers=args.num_workers, shuffle=False, collate_fn=itawrapper.collate_fn)
            trainer = Trainer(model, train_loader, valid_loader, config, cdr=3, fold=0, wandb=args.wandb)
            trainer.log = fake_log
            optimizer = trainer.get_optimizer()
            batch_idx = 0
            for e in range(args.ita_epoch):
                for batch in train_loader:
                    batch = trainer.to_device(batch, device)
                    loss = trainer.train_step(batch, batch_idx) / args.update_freq
                    loss.backward()
                    batch_idx += 1
                    if batch_idx % args.update_freq == 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                        optimizer.step()
                        optimizer.zero_grad()
            if batch_idx % args.update_freq != 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                optimizer.step()
                optimizer.zero_grad()
            # save model
            torch.save(model, osp.join(save_dir, f'iter_{r}.ckpt'))
            if save_best:
                torch.save(model, osp.join(save_dir, 'ita', 'best.ckpt'))
        log.close()

    def ita_eval(self, data_dir, ita_save_dir, ita_res_save_dir):
        check_dir(ita_res_save_dir)
        model = torch.load(osp.join(ita_save_dir, 'best.ckpt'))
        device = torch.device('cuda')

        dataset = EquiAACDataset(osp.join(data_dir, 'skempi_all.json'))
        dataset.mode = '111'
        n_samples = 100

        origin_cplx = [dataset.data[i] for i in dataset.idx_mapping]
        origin_cplx_paths = []
        out_dir = osp.join(ita_res_save_dir, 'original')
        check_dir(out_dir)
        print_log(f'Writing original structures to {out_dir}')
        for cplx in tqdm(origin_cplx):
            pdb_path = osp.join(out_dir, cplx.get_id() + '.pdb')
            cplx.to_pdb(pdb_path)
            origin_cplx_paths.append(osp.abspath(pdb_path))

        log = open(osp.join(ita_res_save_dir, 'log.txt'), 'w')
        res_dir = osp.join(ita_res_save_dir, 'optimized')
        check_dir(res_dir)

        scores = []
        for i in tqdm(range(len(dataset))):
            origin_input = dataset[i]
            inputs = [origin_input for _ in range(n_samples)]
            cur_scores, results = [], []
            with torch.no_grad():
                batch = dataset.collate_fn(inputs)
                ppls, seqs, xs, true_xs, aligned = model.infer(batch, device, greedy=False)
                results.extend([(seqs[i], xs[i], true_xs[i], aligned) for i in range(len(seqs))])
            for n, (seq, x, true_x, aligned) in enumerate(results):
                if not aligned:
                    ca_aligned, rotation, t = kabsch(x[:, 1, :], true_x[:, 1, :])
                    x = np.dot(x - np.mean(x, axis=0), rotation) + t
                new_cplx = set_cdr(origin_cplx[i], seq, x, cdr='H' + str(model.cdr_type))
                pdb_path = osp.join(res_dir, new_cplx.get_id() + f'_{n}.pdb')
                new_cplx.to_pdb(pdb_path)
                new_cplx = AAComplex(
                    new_cplx.pdb_id, new_cplx.peptides,
                    new_cplx.heavy_chain, new_cplx.light_chain,
                    new_cplx.antigen_chains)
                try:
                    score = pred_ddg(origin_cplx_paths[i], osp.abspath(pdb_path))
                except Exception as e:
                    print_log(str(e), level='ERROR')
                    score = 0
                cur_scores.append(score)
            mean_score = np.mean(cur_scores)
            best_score_idx = min([k for k in range(len(cur_scores))], key=lambda k: cur_scores[k])
            scores.append(cur_scores[best_score_idx])
            log.write(f'pdb {origin_cplx[i].get_id()}, mean ddg {mean_score}, best ddg {cur_scores[best_score_idx]}, sample {best_score_idx}\n')
            log.flush()
            
            if self.args.wandb:
                wandb.log({'ddg_mean': mean_score, 'ddg_best': cur_scores[best_score_idx]})

        mean_score = np.mean(scores)
        log_line = f'overall ddg mean {mean_score} WITHOUT sidechain packing'
        if self.args.wandb:
            wandb.log({'overall_ddg_mean': np.mean(scores)})
        print_log(log_line)
        log.write(log_line + '\n')
        log.close()


if __name__ == '__main__':
    args = create_parser()
    config = args.__dict__

    config['ex_name'] = time.strftime('%Y%m%d_%H%M%S', time.localtime()) + '{:05d}'.format(random.randint(0, 10000))
    set_seed(args.seed)
    
    # if args.wandb:
    #     os.environ["WANDB_API_KEY"] = ""
    #     wandb.init(project="Antibody", config=config, name=args.ex_name)

    exp = Exp(args)

    if args.task == 'kfold':
        exp.k_fold_train()
        exp.k_fold_eval()
    elif args.task == 'rabd':
        data_dir = osp.join(args.root_dir, 'cdrh3')
        save_dir = osp.join(data_dir, 'ckpt', args.ex_name + '_CDR3')
        exp.cdr3_train(data_dir, save_dir)
        exp.rabd_eval(data_dir, save_dir)
    elif args.task == 'ita':
        data_dir = args.root_dir
        save_dir = osp.join(data_dir, 'affopt', 'ckpt', args.ex_name + '_AffOpt')
        ita_save_dir = osp.join(save_dir, 'ita')
        ita_res_save_dir = osp.join(save_dir, 'ita_results')
        exp.cdr3_train(data_dir, save_dir)
        exp.ita_train(data_dir, save_dir, ita_save_dir)
        exp.ita_eval(data_dir, ita_save_dir, ita_res_save_dir)

    if args.wandb:
        wandb.finish()