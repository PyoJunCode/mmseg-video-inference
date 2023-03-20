import os
import datetime

import cv2
import mmcv
import torch
import numpy as np
from mmcv.cnn.utils import revert_sync_batchnorm
from mmcv.runner import load_checkpoint
from mmcv.image import tensor2imgs
from mmcv.parallel import collate, scatter

from mmseg.datasets.pipelines import Compose
from mmseg.models import build_segmentor
from mmseg.utils import build_dp, get_device

class InferenceModule:
    
    def __init__(self, config_dir, ckpt_dir, out_dir="./result_images", save_img=False):
        self.cfg = None
        self.test_pipeline = None
        self.model = None
        self.device = get_device()
        self.out_dir = out_dir
        self.save_img = save_img

        
        self.last_mask = None
        
        self.init_config(config_dir, ckpt_dir)

    def init_config(self, config_dir, ckpt_dir):
        if not os.path.isdir(self.out_dir):
            os.makedirs(self.out_dir)
                
        self.cfg = mmcv.Config.fromfile(config_dir)
        self.cfg.gpu_ids = [0]
        del self.cfg.data.test.pipeline[0]
        self.test_pipeline = Compose(self.cfg.data.test.pipeline)
        print(self.test_pipeline)
        
        self.cfg.model.train_cfg = None
        model = build_segmentor(self.cfg.model, test_cfg=self.cfg.get('test_cfg'))
        model = revert_sync_batchnorm(model)
        model = build_dp(model, self.device, device_ids=self.cfg.gpu_ids)
        
        self.model = model
        
        checkpoint = load_checkpoint(model, ckpt_dir, map_location='cpu')
        if 'CLASSES' in checkpoint.get('meta', {}):
            print("Class founded: ", checkpoint['meta']['CLASSES'])
            self.model.CLASSES = checkpoint['meta']['CLASSES']
            self.model.module.CLASSES = checkpoint['meta']['CLASSES']
        else:
            print('"CLASSES" not found in meta')
            raise Exception
        if 'PALETTE' in checkpoint.get('meta', {}):
            self.model.PALETTE = checkpoint['meta']['PALETTE']
        else:
            print('"PALETTE" not found in meta')
            raise Exception
        torch.cuda.empty_cache()
    
    def preprocessing(self, imgs):
        """
        input type: (b, c, h, w)tensor
        """

        datas = []
        for img in imgs:
            if torch.is_tensor(img):
                img_np = img.cpu().float().numpy()
            else:
                img_np = img
            # data
            data = dict(img=img_np,
                        img_shape=img_np.shape,
                        ori_shape=img_np.shape,
                        pad_shape=img_np.shape,
                        scale_factor=None,
                        img_prefix=None,
                        filename=None,
                        ori_filename=None,
                        img_fields=['img'])
            # build the data pipeline
            data = self.test_pipeline(data)
            datas.append(data)
        
        data = collate(datas, samples_per_gpu=len(imgs))
        # just get the actual data from DataContainer
        data['img_metas'] = [img_metas.data[0] for img_metas in data['img_metas']]
        data['img'] = [torch.unsqueeze(img.data[0], 0) for img in data['img']]
        if next(self.model.parameters()).is_cuda:
            # scatter to specified GPU
            data = scatter(data, [self.device])[0]
        
        return data
    
    def postprocessing(self, img, result, opacity=0.2):
        
        if isinstance(img, torch.Tensor):
            img = img.cpu().detach().numpy()
            img = np.squeeze(img).transpose(1, 2, 0)
        elif isinstance(img, np.ndarray):
            img = np.squeeze(img)
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = mmcv.imread(img)
        img = img.copy()
        seg = result[0]

        #palette = self.model.PALETTE
        palette = [[0,0,0], [0,0,255]]
        palette = np.array(palette)
        
        assert palette.shape[0] == len(self.model.CLASSES)
        assert palette.shape[1] == 3
        assert len(palette.shape) == 2
        assert 0 < opacity <= 1.0
        
        color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
        for label, color in enumerate(palette):
            if label == 0:
                continue
            color_seg[seg == label, :] = color
        # convert to BGR
        color_seg = color_seg[..., ::-1]

        img = img * (1 - opacity) + color_seg * opacity
        img = img.astype(np.uint8)
        
        if self.save_img:
            out_file = os.path.join(self.out_dir, datetime.datetime.now().strftime("%Y%m%d_%H%M%f") + ".png")
            mmcv.imwrite(img, out_file)

        return img

    def load_lastmask(self, img):
        mask = self.last_mask
        result = self.postprocessing(img, mask)
        return result

    def inference(self, data):
        try:
            """
            input type: (b, c, h, w)tensor
            """
            data = self.preprocessing(data)
            with torch.no_grad():
                results = self.model(return_loss=False, **data)
                self.last_mask = results
        
            img_tensor = data['img'][0]
            
            img_metas = data['img_metas'][0]
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas), f"{len(imgs)} and {len(img_metas)} are not same.}}"

            assert len(imgs) == 1 and len(img_metas) == 1, f"{len(imgs)} and {len(img_metas)} are not one.}}"
                
            for img, img_meta in zip(imgs, img_metas):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))
                
                result = self.postprocessing(img_show, results)

                return result
        except Exception as e:
            print(e)
            return None