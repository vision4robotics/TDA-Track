# 🏃‍♀️TDA-Track: Prompt-Driven Temporal Domain Adaptation for Nighttime UAV Tracking
Changhong Fu∗, Yiheng Wang, Liangliang Yao, Guangze Zheng, Haobo Zuo and Jia Pan
* Corresponding author.

## 📣 News
- [2024/03] 💻 Code will be released soon.

<!-- test, evaluation and train-->

<!-- Prerequisite-->
## :hammer_and_wrench: Installation
### 1. Test Prerequisite
This code has been tested on Ubuntu 18.04, Python 3.8.3, Pytorch 0.7.0/1.6.0, CUDA 10.2.  
Please install related libraries before running this code: 
```bash
pip install -r requirements.txt
```
### 2. Train Prerequisite
To train TDA-Track, more libraries are needed to obtain training samples from nighttime raw videos. More details can be found on [NetTrack](https://github.com/George-Zhuang/NetTrack).
  
## 🚀 Get started
### Quick test and evaluation
The tracking results of NUT2024-40L, NUT2024-60L, NUT2024-100L will be provided soon.  
If you want to evaluate the tracker, please put those results into  `results` directory.
```
python eval.py 	                          \
	--tracker_path ./results          \ # result path
	--dataset UAV10                   \ # dataset_name
	--tracker_prefix 'result'   # tracker_name
```

### Train TDA-Track
#### 1. Download training datasets
Download the daytime tracking datasets：
* [VID](http://image-net.org/challenges/LSVRC/2017/)
* [GOT-10K](http://got-10k.aitestunion.com/downloads)  
Note: train_dataset/dataset_name/readme.md has listed detailed operations about how to generate training datasets.

Download the nighttime tracking datasets：
* [NAT2021](https://vision4robotics.github.io/NAT2021)  
Note: NAT2021-train set is unannotated, the training samples are obtained with the prompt-driven object mining approach, as presented in [Preprocessing](#Preprocessing phase)
#### 2. Preprocessing phase

Preprocessing please refer to ... to be completed...

#### 3. Training phase
To train the model, run `train.py` with the desired configs:
to be completed...
  

## NUT2024-40L dataset
<!-- release the dataset demo-->
- 📊 Long-term Nighttime UAV Tracking (**NUT2024-40L**) Benchmark:
  - 🎬40 videos shot by UAV in various scenarios at night 
  - 🎯collected for artifical intelligence research
  
- 📥 Download **NUT2024-40L dataset**
  - To be released

<!-- release the tracking demos -->

## 🥰 Acknowledgement
The code is based on [NetTrack](https://github.com/George-Zhuang/NetTrack), [UDAT](https://github.com/vision4robotics/UDAT), and [TCTrack](https://github.com/vision4robotics/TCTrack). The authors appreciate the great work and the contributions they made.

<!-- release the cites -->
