# 🏃‍♀️TDA-Track: Prompt-Driven Temporal Domain Adaptation for Nighttime UAV Tracking
Changhong Fu∗, Yiheng Wang, Liangliang Yao, Guangze Zheng, Haobo Zuo and Jia Pan
* Corresponding author.

## 📣 News
- [2024/03] 💻 Code will be released soon.

<!-- test, evaluation and train-->
## 🚀 Get started
### Quick test and evaluation
The tracking results of NUT2024-40L, NUT2024-60L, NUT2024-100L will be provided soon. If you want to evaluate the tracker, please put those results into  `results` directory.
```
python eval.py 	                          \
	--tracker_path ./results          \ # result path
	--dataset UAV10                   \ # dataset_name
	--tracker_prefix 'result'   # tracker_name
```

### Train TDA-Track
#### Download training datasets

Download the datasets：
* [VID](http://image-net.org/challenges/LSVRC/2017/)
* [GOT-10K](http://got-10k.aitestunion.com/downloads)
  
#### Preprocessing phase

Preprocessing please refer to ... to be completed...

#### Training phase
To train the model, run `train.py` with the desired configs:
to be completed...
  
<!-- Prerequisite-->
## :hammer_and_wrench: Installation

- **Prerequisite**
  ```bash
  conda create -n nettrack python=3.10 # please use the default version
  pip3 install torch torchvision # --index-url https://download.pytorch.org/whl/cu121
  pip3 install -r requirements.txt
  pip3 install cython; pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
  pip3 install cython_bbox
  sudo apt update
  sudo apt install ffmpeg
  ```

  Install Grounding DINO and CoTracker:
  ```bash
  pip install git+https://github.com/IDEA-Research/GroundingDINO.git
  pip install git+https://github.com/facebookresearch/co-tracker.git@8d364031971f6b3efec945dd15c468a183e58212
  ```

- **Prepare weights:**
  Download the default pretrained Grouding DINO and CoTracker model:
  ```bash
  cd weights
  cd groundingdino
  wget https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swinb_cogcoor.pth
  cd ..
  mkdir cotracker && cd cotracker
  wget https://dl.fbaipublicfiles.com/cotracker/cotracker_stride_4_wind_8.pth
  cd ..
  ```

## :NUT2024-40L dataset
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
