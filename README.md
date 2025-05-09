# 🏃‍♀️TDA-Track: Prompt-Driven Temporal Domain Adaptation for Nighttime UAV Tracking
Changhong Fu∗, Yiheng Wang, Liangliang Yao, Guangze Zheng, Haobo Zuo, and Jia Pan
* Corresponding author.

## 📣 News
- [2024/10] 💻 NAT2024-1 benchmark and TDA-Track checkpoint have been released.
- [2024/03] 💻 Code has been released.
  
## Abstract
> Nighttime UAV tracking has achieved great progress by domain adaptation (DA) under low-illuminated scenarios. However, previous DA works are defcient in narrowing the discrepancy of temporal contexts for UAV trackers. To address the issue, this work proposes a prompt-driven temporal domain adaptation framework to fully utilize temporal contexts for challenging nighttime UAV tracking, i.e., TDA-Track. Specifically, the proposed framework aligns the distribution of temporal contexts from different domains by training the temporal feature generator against the discriminator. The temporal-consistent discriminator progressively extracts shared domain-specifc features to generate coherent domain discrimination results in the time series. Additionally, to obtain high-quality training samples, a prompt-driven object miner is employed to precisely locate objects in unannotated nighttime videos. Moreover, a new benchmark for nighttime UAV tracking is constructed. Exhaustive evaluations of TDA-Track demonstrate remarkable performance on both public and self-constructed benchmarks. Real-world tests also show its practicality. The code and demo videos are available here.

## 🎞️ Video Demo 
[![TDA-Track: Prompt-Driven Temporal Domain Adaptation for Nighttime UAV Tracking](https://res.cloudinary.com/marcomontalbano/image/upload/v1713340462/video_to_markdown/images/youtube--mmLlPr3iiv4-c05b58ac6eb4c4700831b2b3070cd403.jpg)](https://youtu.be/mmLlPr3iiv4 "TDA-Track: Prompt-Driven Temporal Domain Adaptation for Nighttime UAV Tracking")
<!-- test, evaluation and train-->

<!-- Prerequisite-->
## :hammer_and_wrench: Installation
### 1. Test Prerequisite
Please install related libraries: 
```bash
pip install -r requirement.txt
```
### 2. Train Prerequisite
To train TDA-Track, more libraries are needed to obtain training samples from nighttime raw videos. More details can be found on [NetTrack](https://github.com/George-Zhuang/NetTrack).
  
## 🚀 Get started
### 1. Quick test and evaluation
The checkpoint is available [here](https://pan.baidu.com/s/1gK2r_V1DMdJNKqGTfeMRew), password: djp8.

To test on NAT2024-1 and NUT_L benchmarks, you need to download them from the following links:

* [NAT2024-1](https://pan.baidu.com/s/1poS9XNVEnd7ObokJdRkJuQ), password: 6hpb

* [NUT_L](https://pan.baidu.com/share/init?surl=KWYp5UHflFuaPiWLFZvaKw&pwd=t4tr)
```
python test.py
```
You can find the tracking results of NAT2024-1, NUT_L in `results.zip` file.
If you want to evaluate the tracker, please put those results into  `results` directory.
```
python eval.py 	                          \
	--tracker_path ./results          \ # result path
	--dataset NAT2024                   \ # dataset_name
```

### 2. Train TDA-Track
#### - Download training datasets
Download the daytime tracking datasets：
* [VID](http://image-net.org/challenges/LSVRC/2017/)
* [GOT-10K](http://got-10k.aitestunion.com/downloads)  
Note: train_dataset/dataset_name/readme.md has listed detailed operations about how to generate training datasets.

Download the nighttime tracking datasets：
* [NAT2021](https://vision4robotics.github.io/NAT2021)  
Note: NAT2021-train set is unannotated, the training samples are obtained with the prompt-driven object mining approach, as presented in [Preprocessing](#Preprocessing phase)
#### - Preprocessing phase
```
cd preprocessing
python tools/detect.py
python tools/track.py
```

#### - Training phase
To train the model, run `train.py` with the desired configs:
```
cd TDA-Track
python tools/train_tctrack_temporalda.py
```
  

## NAT2024-1 dataset
<!-- release the dataset demo-->
- 📊 Long-term Nighttime UAV Tracking (**NAT2024-1**) Benchmark:
  - 🎬40 videos shot by UAV in various scenarios at night 
  - 🎯collected for artifical intelligence research
  
- 📥 Download **NAT2024-1 dataset**
  - [Baidu Disk](https://pan.baidu.com/s/1poS9XNVEnd7ObokJdRkJuQ (password = 6hpb))

<!-- release the tracking demos -->

## 🥰 Acknowledgement
The code is based on [NetTrack](https://github.com/George-Zhuang/NetTrack), [UDAT](https://github.com/vision4robotics/UDAT), and [TCTrack](https://github.com/vision4robotics/TCTrack). The authors appreciate the great work and the contributions they made.

<!-- release the cites -->
