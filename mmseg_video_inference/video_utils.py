import os
import ffmpeg
import numpy as np

import cv2
import math
import torch

def parse_video_info(video_path):
    probe = ffmpeg.probe(video_path)
    video_streams = [stream for stream in probe["streams"] if stream["codec_type"] == "video"]
    
    w, h, fps = video_streams[0]["width"], video_streams[0]["height"], eval(video_streams[0]["r_frame_rate"])
    
    return w, h, fps

def read_video(video_path, seek=None):
    w, h, fps = parse_video_info(video_path)
    
    
    if seek:
        out, _ = (
            ffmpeg
            .input(video_path)
            .output('pipe:', ss=seek[0], to=seek[1], format='rawvideo', pix_fmt='rgb24')
            .run(capture_stdout=True)
        )
    else:
        out, _ = (
            ffmpeg
            .input(video_path)
            .output('pipe:', format='rawvideo', pix_fmt='rgb24')
            .run(capture_stdout=True)
        )
    
    
    video = (
        np
        .frombuffer(out, np.uint8)
        .reshape([-1, h, w, 3])
    )
    
    return video

def open_ffmpeg_pipe(video_path, result_dir):
    output_name = os.path.splitext(os.path.basename(video_path))[0]
    output_ext = ".mp4"                                   
    output_filename = os.path.join(result_dir, f"avle_{output_name}" + output_ext)
    
    w, h, fps = parse_video_info(video_path)
    
    print("\n\n" + "#"*30)
    print(f"Video info: width: {w} height: {h} fps: {fps}")
    print("#"*30 + "\n\n")
    
    process = (
        ffmpeg
        .input('pipe:', format='rawvideo', pix_fmt='bgr24', s=f'{w}x{h}', r=f'{fps}')
        .output(output_filename)
        .overwrite_output()
        .run_async(pipe_stdin=True, pipe_stderr=False, overwrite_output=True, quiet=False)
    )
    
    return process, output_filename


def img2tensor(imgs, bgr2rgb=True, float32=True):
    def _totensor(img, bgr2rgb, float32):
        if img.shape[2] == 3 and bgr2rgb:
            if img.dtype == 'float64':
                img = img.astype('float32')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        if float32:
            img = img.float()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    else:
        return _totensor(imgs, bgr2rgb, float32)


def tensor2img(tensor, rgb2bgr=True, out_type=np.uint8, min_max=(0, 1)):
    if not (torch.is_tensor(tensor) or (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError(f'tensor or list of tensors expected, got {type(tensor)}')

    if torch.is_tensor(tensor):
        tensor = [tensor]
    result = []
    for _tensor in tensor:
        _tensor = _tensor.squeeze(0).float().detach().cpu().clamp_(*min_max)
        _tensor = (_tensor - min_max[0]) / (min_max[1] - min_max[0])

        n_dim = _tensor.dim()
        if n_dim == 4:
            img_np = make_grid(_tensor, nrow=int(math.sqrt(_tensor.size(0))), normalize=False).numpy()
            img_np = img_np.transpose(1, 2, 0)
            if rgb2bgr:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 3:
            img_np = _tensor.numpy()
            img_np = img_np.transpose(1, 2, 0)
            if img_np.shape[2] == 1:  # gray image
                img_np = np.squeeze(img_np, axis=2)
            else:
                if rgb2bgr:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 2:
            img_np = _tensor.numpy()
        else:
            raise TypeError('Only support 4D, 3D or 2D tensor. ' f'But received with dimension: {n_dim}')
        if out_type == np.uint8:
            # Unlike MATLAB, numpy.unit8() WILL NOT round by default.
            img_np = (img_np * 255.0).round()
        img_np = img_np.astype(out_type)
        result.append(img_np)
    if len(result) == 1:
        result = result[0]
    return result
