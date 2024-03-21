

from __future__ import unicode_literals
import json
from os.path import join, exists
import os
import pandas as pd

dataset_path = 'path/to/NAT2021'
train_set = ['train_clip']
d_sets = {'videos_train':train_set}


def parse_and_sched(dl_dir='.'):
    js = {}
    for d_set in d_sets:
        for dataset in d_sets[d_set]:
            videos = os.listdir(os.path.join(dataset_path,dataset))
            for video in videos:
                if video == 'list.txt':
                    continue
                video = dataset+'/'+video
                gt_path = 'path/to/NetTrack/Pesudo/Label/' + f'{video}.txt'
                f = open(gt_path, 'r')
                groundtruth = f.readlines()
                f.close()
                for idx, gt_line in enumerate(groundtruth):
                    gt_image = gt_line.strip().split(',')
                    frame = '%06d' % (int(idx))
                    obj = '%02d' % (int(0))
                    bbox = [int(float(gt_image[0])), int(float(gt_image[1])),
                            int(float(gt_image[0])) + int(float(gt_image[2])),
                            int(float(gt_image[1])) + int(float(gt_image[3]))]  # xmin,ymin,xmax,ymax

                    if video not in js:
                        js[video] = {}
                    if obj not in js[video]:
                        js[video][obj] = {}
                    js[video][obj][frame] = bbox
       
        json.dump(js, open('train.json', 'w'), indent=4, sort_keys=True)
        js = {}

        print(d_set+': All videos downloaded' )


if __name__ == '__main__':
    parse_and_sched()
