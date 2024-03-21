from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import argparse

from glob import glob
from tqdm import tqdm
from multiprocessing import Pool
from toolkit.datasets import NUT2024_40LDataset, NUT2024_60LDataset, NUT2024_100LDataset
from toolkit.evaluation import OPEBenchmark

parser = argparse.ArgumentParser(description='tracking evaluation')
parser.add_argument('--tracker_path', '-p', type=str,default='./results', 
                    help='tracker result path')
parser.add_argument('--dataset', '-d', type=str,default='', # NUT2024-40L, NUT2024-60L, NUT2024-60L
                    help='dataset name')
parser.add_argument('--num', '-n', default=24, type=int,
                    help='number of thread to eval')
parser.add_argument('--tracker_prefix', '-t', default='',
                    type=str, help='tracker name')
parser.add_argument('--show_video_level', '-s', dest='show_video_level',
                    action='store_true')
parser.set_defaults(show_video_level=False)
args = parser.parse_args()

def main():
    tracker_dir = os.path.join(args.tracker_path, args.dataset)
    trackers = glob(os.path.join(args.tracker_path, args.dataset, args.tracker_prefix+'*'))

    trackers = [x.split('/')[-1] for x in trackers]

    assert len(trackers) > 0
    args.num = min(args.num, len(trackers))

    root = os.path.realpath(os.path.join(os.path.dirname(__file__),
                             '../test_dataset'))
    root = os.path.join(root, args.dataset)


    if '40L' in args.dataset:
        dataset = NUT2024_40LDataset(args.dataset, root)
    elif '60L' in args.dataset:
        dataset = NUT2024_60LDataset(args.dataset, root)
    elif '100L' in args.dataset:
        dataset = NUT2024_100LDataset(args.dataset, root)
    else:
        raise("unknown dataset")

    dataset.set_tracker(tracker_dir, trackers)
    benchmark = OPEBenchmark(dataset)
    # calculate success
    success_ret = {}
    with Pool(processes=args.num) as pool:
        for ret in tqdm(pool.imap_unordered(benchmark.eval_success,
                trackers), desc='eval success', total=len(trackers), ncols=100):
                success_ret.update(ret)
    # calculate precision
    precision_ret = {}
    with Pool(processes=args.num) as pool:
        for ret in tqdm(pool.imap_unordered(benchmark.eval_precision,
                trackers), desc='eval precision', total=len(trackers), ncols=100):
                precision_ret.update(ret)
    # calculate normalized precision
    norm_precision_ret = {}
    with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_norm_precision,
                trackers), desc='eval norm precision', total=len(trackers), ncols=100):
                norm_precision_ret.update(ret)
    # show results
    benchmark.show_result(success_ret, precision_ret, norm_precision_ret,
                show_video_level=args.show_video_level)

if __name__ == '__main__':
    main()
