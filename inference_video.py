import os
import glob
import argparse
import traceback

from tqdm import tqdm
import numpy as np
import ffmpeg
from torch.utils.data import DataLoader

from mmseg_video_inference.inference_module import InferenceModule
from mmseg_video_inference.video_utils import open_ffmpeg_pipe, read_video


class VideoModule:
    def __init__(self, module, interval):
        self.module = module
        self.interval = interval
        
        
    def feed_data(self, video, result_dir, seek):
        process = open_ffmpeg_pipe(video, result_dir)
        self.process = process[0]
        self.data = read_video(video, seek)
    
    def write_frame(self, res):
        try:
            self.process.stdin.write(res.tobytes())
        except Exception as e:
            print(e)
            self.process.stdin.close()
            self.process.wait()
            self.process.terminate()
        

    def inference(self):
        try:
            for i in tqdm(range(len(self.data))):
                img = self.data[i:i+1, ...]

                if i % self.interval == 0:
                    res = self.module.inference(img)
                    if res is None:
                        continue
                else:
                    res = self.module.load_lastmask(img)

                self.write_frame(res)
    
        except Exception as e:
            print("Exception: ", e)
            print(traceback.print_exc())
        finally:
            self.process.stdin.close()
            self.process.wait()
            self.process.terminate()


def main(args):
    # argparse
    interval = args.interval
    config_dir = args.config
    ckpt_dir = args.checkpoint
    video_path = args.video_path
    result_dir = args.result_dir
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)
    
    seek = args.seek
    assert seek is None or len(seek) == 2, "Invalid seek value."
    
    # Init Inferece module. Change this module for your framework.
    module = InferenceModule(config_dir, ckpt_dir, result_dir)
    
    # Check video path is directory or file, and listing videos
    if os.path.isdir(video_path):
        file_list = glob.glob(os.path.join(video_path, "*"))
    elif os.path.isfile(video_path):
        file_list = [video_path]
    else:
        raise Exception("Invalid video path.")
    
    print("Start inference: ", file_list)
    

    
    # Inference all video via module.
    for video in file_list:
        print("\n\n###\nInference ", os.path.basename(video), "\n###\n\n")
        video_writer = VideoModule(module, interval, )
        video_writer.feed_data(video, result_dir, seek)
        video_writer.inference()
        
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help='test config file path')
    parser.add_argument("checkpoint", help='checkpoint file')
    parser.add_argument("video_path", help='Video folder or file for inference.')
    parser.add_argument("--result_dir", "-r", default="/mnt/nas2/project/sujawon/results", help="Save directory for result video.")
    parser.add_argument("--interval", "-i", type=int, default=10, help="Interval period for inference")
    parser.add_argument("--seek", "-s", type=str, nargs=2, help="Seek time for inference")
    args = parser.parse_args()
    main(args)