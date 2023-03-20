# MMSegmentation video inference



## Features

- Single file inference
- Folder inference
- Seeking

## Installation

Tested on: python 3.9, torch 1.13, cuda 11.6, ffmpeg-python 0.2.0 (ffmpeg 4.2.2)

It assumes torch and mmsegmentation is installed.
If not, you can install by following link [mmsegmentation docs](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/get_started.md#installation) or commands below.

Install pre-requirement packages

```bash
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia

pip install openmin
mim install mmcv-full
pip install mmsegmentation

```

Install requirements.txt

```bash
pip install -r requirements.txt
```

## Usage

`config` : Path of config file

`checkpoint`  : Path of checkpoint file

`video_path` : Directory or Filename of video file

- If video path is directory, all videos in that path will be inference.

`--result_dir , -r` <Optional> : Inference result save path

`--interval, -i` <Optional> : Inference interval frame *(default=10)*

`--seek, -s` <Optional> :  Specify position of frames *(e.g. `-s 00:00:10 00:02:20`)*

- String format following ffmepg’s format. It can be a timestamp or frame number.

Example

```bash
python3 inference_video.py -s 00:00:00 00:00:10 <config> <ckpt> <video> -r .
```

## Issues

- [ ]  Inference module outputs low resolution result with specific ffmpeg version?