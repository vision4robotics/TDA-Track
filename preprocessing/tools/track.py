import os
import os.path as osp
import argparse
import cv2
import torch
import random
import sys
import numpy as np
from glob import glob
from tqdm import tqdm
from loguru import logger
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '../../'))
from tracker.nettrack import NetTracker
from cotracker.predictor import CoTrackerPredictor

DEFAULT_DEVICE = ('cuda' if torch.cuda.is_available() else
                  'mps' if torch.backends.mps.is_available() else
                  'cpu')

def make_parser():
    parser = argparse.ArgumentParser("NetTrack")
    parser.add_argument('--exp_name', type=str, default='run NetTrack on NAT2021-train set', help='exp name')
    # input data
    parser.add_argument('--train_clip_dir', type=str, default='', help='path to NAT2021 train clip') 
    parser.add_argument('--seq', type=str, default='all', help='all or None ==> all the seqs') 
    parser.add_argument('--det_preds', type=str, default='./det-result', help='detection results')
    # point tracker config
    parser.add_argument('--point_tracker', type=str, default='cotracker', help='support cotracker, tapir, pips')
    parser.add_argument('--checkpoint', type=str, default='./weights/cotracker/cotracker_stride_4_wind_8.pth', help='initial directory')
    # tracker config
    parser.add_argument("--device", type=int, default=3, help='cuda device number')
    parser.add_argument("--track_thresh", type=float, default=0.4, help="tracking confidence threshold")
    parser.add_argument("--score_mode", type=str, default='quadratic', help="score mode for tracking")
    # output data
    parser.add_argument('--output_path', type=str, default='./track-result', help='output path')
    parser.add_argument("--save_result", type=bool, default=true, help='save result')
    parser.add_argument("--min_area", type=float, default=10, help='Minimum area of bboxes')
    parser.add_argument("--max_area", type=float, default=100000000, help='Minimum area of bboxes')
    parser.add_argument("--vis_box", type=bool, default=True, help='visualze boxes')
    parser.add_argument("--vis_points", type=bool, default=True, help='visualze points')
    return parser

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    ''' set up cuda and logger environment '''
    args = make_parser().parse_args()   
    torch.cuda.set_device(args.device)
    setup_seed(0)
    logger.add(f"{args.exp_name}.log", rotation="10 MB")


    ''' tracker model '''
    point_tracker = CoTrackerPredictor(checkpoint=args.checkpoint).to(args.device)
    tracker = NetTracker(args=args, point_tracker=point_tracker)
    logger.info('Tracker model loaded.')
    

    ''' check the detection and dataset '''
    seqs = os.listdir(args.train_clip_dir)
    annos = os.listdir(args.det_preds)
    if len(seqs)==len(annos):
        print("sequences underwent detection matched with the dataset")
    else:
        print(f"only {len(annos)} sequences underwent detection, while the dataset has {len(seqs)} sequences")
    # very portable data loader
    if args.seq in [None, 'all', 'ALL', 'All']:
        seqs.sort()
    else:
        seqs = [args.seq]
    
    ''' run the tracker model '''
    for seq in tqdm(seqs):
        res_file = os.path.join(args.output_path, f"{seq}.txt")
        
        # images of the sequence
        images_path = os.path.join(args.train_clip_dir, seq, '*.jpg')
        images = glob(images_path)
        if len(images) == 0:
            raise Exception(f'No images, please check {images_path}')
        images.sort()

        # detection results of the sequence
        det_boxes_path = os.path.join(args.det_preds, seq, '*.txt')
        det_boxes = glob(det_boxes_path)
        if len(det_boxes) == 0:
            raise Exception(f'No detection results, please check {det_boxes_path}')
        det_boxes.sort()
        det_boxes = det_boxes[:len(images)]

        ''' start tracking '''
        logger.info(f'Start tracking sequence: {seq}.')
        results = []
        seq_preds = []
        img_paths = []
        for frame_id, img_path in enumerate(images, 1):
            if frame_id == 1:
                img = cv2.imread(images[0])
                img_h, img_w = img.shape[:2]
            # get predictions from detection results
            preds = []
            with open(det_boxes[frame_id-1], 'r') as file:
                lines = file.readlines()
                for line in lines:
                    data = line.strip().split(',')
                    data = [float(x) for x in data]
                    pred = data[:5]
                    cls = int(data[5])
                    preds.append(pred)
            seq_preds.append(preds)
            img_paths.append(img_path)

        results = tracker.inference(seq_preds, 
                                    img_paths, 
                                    seq_name=seq
                                    )
        ''' saving results '''
        if args.save_result:
            os.makedirs(os.path.dirname(res_file), exist_ok=True)
            with open(res_file, 'w') as f:
                f.writelines(results)
            logger.info(f"Save results to {res_file}")
