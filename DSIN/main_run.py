import argparse
from utils.en_train import EnConfig, EnRun
from utils.ch_train import ChConfig, ChRun
from distutils.util import strtobool

def main(args):
    
    EnRun(EnConfig(batch_size=args.batch_size,learning_rate=args.lr,seed=args.seed, model=args.model, tasks = args.tasks,
                                cme_version=args.cme_version, dataset_name=args.dataset,num_hidden_layers=args.num_hidden_layers,
                                context=args.context, text_context_len=args.text_context_len, audio_context_len=args.audio_context_len))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('--lr', type=float, default=5e-6, help='learning rate, recommended: 5e-6 for mosi, mosei, 1e-5 for sims')
    parser.add_argument('--dataset', type=str, default='sims', help='dataset name: mosi, mosei, sims')
    parser.add_argument('--num_hidden_layers', type=int, default=3, help='number of hidden layers for cross-modality encoder')
    parser.add_argument('--tasks', type=str, default='MTAV', help='losses to train: M: multi-modal, T: text, A: audio, V: vision (defalut: MTA))')
    args = parser.parse_args()
    main(args)