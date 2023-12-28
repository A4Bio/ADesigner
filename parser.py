import argparse


def create_parser():
    parser = argparse.ArgumentParser(description='training')
    parser.add_argument('--ex_name', type=str, default='Debug')
    parser.add_argument('--root_dir', type=str, default='./summaries')

    # training related
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--max_epoch', type=int, default=20, help='max training epoch')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='clip gradients with too big norm')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--shuffle', action='store_true', help='shuffle data')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--mode', type=str, default='111', help='H/L/Antigen, 1 for include, 0 for exclude')
    parser.add_argument('--seed', type=int, default=42, help='Seed to use in training')
    parser.add_argument('--early_stop', action='store_true', help='Whether to use early stop')

    # device
    parser.add_argument('--gpus', type=int, default=[0], nargs='+', help='gpu to use, -1 for cpu')
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank. Necessary for using the torch.distributed.launch utility.")

    ## shared
    parser.add_argument('--cdr_type', type=str, default='3', help='type of cdr')
    parser.add_argument('--embed_size', type=int, default=64, help='embed size of amino acids')
    parser.add_argument('--hidden_size', type=int, default=128, help='hidden size')
    parser.add_argument('--n_layers', type=int, default=3, help='number of layers')
    parser.add_argument('--alpha', type=float, default=0.8, help='scale mse loss of coordinates')
    parser.add_argument('--anneal_base', type=float, default=0.95, help='Exponential lr decay, 1 for not decay')

    ## rabd
    parser.add_argument('--rabd_topk', type=int, default=100)
    parser.add_argument('--rabd_sample', type=int, default=1)

    ## ita 
    parser.add_argument('--ita_batch_size', type=int, default=4)
    parser.add_argument('--ita_n_iter', type=int, default=20)
    parser.add_argument('--ita_epoch', type=int, default=1)
    parser.add_argument('--update_freq', type=int, default=4)
    parser.add_argument('--n_samples', type=int, default=4, help='Number of samples each iteration')
    parser.add_argument('--n_tries', type=int, default=50, help='Number of tries each iteration')
    parser.add_argument('--task', default='rabd', choices=['kfold', 'rabd', 'ita'])
    parser.add_argument('--run', type=int, default=1, help='Number of runs for evaluation')
    parser.add_argument('--wandb', type=int, default=0)
    parser.add_argument('--dropout', default=0.1, type=float)
    return parser.parse_args()