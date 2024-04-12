# Copyright (c) OpenMMLab. All rights reserved.
import os
import glob
import asyncio
from argparse import ArgumentParser
import json
from mmdet.apis import (async_inference_detector, init_detector, show_result_pyplot)

from projects import *
from mmdet.core import DatasetEnum
import mmcv
import cv2

try:
    from sahi.models.mmdet import MmdetDetectionModel
    from sahi.predict_fisheye import get_sliced_prediction

except ImportError:
    raise ImportError('Please run "pip install -U sahi" '
                      'to install sahi first for large image inference.')


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('dir', help='images path')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--out-file', default=None, help='Path to output file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='random',
        choices=['coco', 'voc', 'citys', 'random'],
        help='Color palette used for visualization')
    parser.add_argument(
        '--dataset',
        default='fisheye8k',
        choices=['coco', 'lvis', 'fisheye8k', 'fisheye8klvis'],
        help='trained dataset')
    parser.add_argument(
        '--score-thr', type=float, default=0.5, help='bbox score threshold')
    parser.add_argument(
        '--postprocess-thr', type=float, default=0.5, help='postprocess threshold')
    parser.add_argument(
        '--async-test',
        action='store_true',
        help='whether to set async options for async inference.')
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='Batch size, must greater than or equal to 1')
    parser.add_argument(
        '--class-agnostic',
        type=bool,
        default=False)
    parser.add_argument(
        '--postprocess',
        type=str,
        default='NMS')
    # 'NMS', or 'GREEDYNMM'
    parser.add_argument(
        '--postprocess-metric',
        type=str,
        default='IOU')
    # 'IOS', or 'IOU'
    parser.add_argument(
        '--patch-size', type=int, default=1280, help='The size of patches')
    parser.add_argument(
        '--patch-overlap-ratio',
        type=float,
        default=0.25,
        help='Ratio of overlap between two patches')
    parser.add_argument(
        '--merge-nms-type',
        type=str,
        default='nms',
        help='NMS type for merging results')
    parser.add_argument(
        '--merge-iou-thr',
        type=float,
        default=0.5,
        help='IoU threshould for merging results')    
    parser.add_argument(
        '--use_super_resolution',
        type=bool,
        default=False,
        help='Whether to use super resolution')
    parser.add_argument(
        '--sr_scale',
        type=int,
        default=2,
        help='Super resolution scale')
    parser.add_argument(
        '--use_hist_equal',
        type=bool,
        default=False,
        help='Whether to use histogram equalization')
        
    args = parser.parse_args()
    return args

def get_image_Id(img_name):
  img_name = img_name.split('.png')[0]
  sceneList = ['M', 'A', 'E', 'N']
  cameraIndx = int(img_name.split('_')[0].split('camera')[1])
  sceneIndx = sceneList.index(img_name.split('_')[1])
  frameIndx = int(img_name.split('_')[2])
  imageId = int(str(cameraIndx)+str(sceneIndx)+str(frameIndx))
  return imageId

def xyxy2xywh(bbox):
        _bbox = bbox.tolist()
        return [
            _bbox[0],
            _bbox[1],
            _bbox[2] - _bbox[0],
            _bbox[3] - _bbox[1],
        ]

def xywh2xyxy(bbox):
        
        return [
            bbox[0],
            bbox[1],
            bbox[2] + bbox[0],
            bbox[3] + bbox[1],
        ]

def apply_histogram_equalization(image):
    ycrcb_img = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    ycrcb_img[:, :, 0] = cv2.equalizeHist(ycrcb_img[:, :, 0])
    image = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2BGR)
    return image

def main(args):
    # build the model from a config file and a checkpoint file
    if args.dataset == 'coco':
        model = init_detector(args.config, args.checkpoint, device=args.device)
    elif args.dataset == 'lvis':
        model = init_detector(args.config, args.checkpoint, device=args.device, dataset=DatasetEnum.LVIS)
    else:
        model = init_detector(args.config, args.checkpoint, device=args.device, dataset=DatasetEnum.FISHEYE8KLVIS)
    # test a single image
    sahi_model = MmdetDetectionModel(
        model_path=args.checkpoint,
        config_path=args.config,
        device=args.device,
        dataset=args.dataset
    )
    sahi_model.confidence_threshold = args.score_thr
    json_results = []
    input_images_path = glob.glob(os.path.join(args.dir, '*.png'))

    for file in mmcv.track_iter_progress(input_images_path):
        img_name = file.split('/')[-1]
        imgid = get_image_Id(img_name)
        img = mmcv.imread(file)
        if args.use_hist_equal:
            img = apply_histogram_equalization(img)


        # arrange slices
        height, width = img.shape[:2]
        if args.use_super_resolution:
            slice_height = args.patch_size
            slice_width = args.patch_size
        else:
            slice_height=int(height*2/3)
            slice_width=int(width*2/3)
        result = get_sliced_prediction(img, 
                              sahi_model, 
                              slice_height=slice_height,
                              slice_width=slice_width,
                              overlap_height_ratio=args.patch_overlap_ratio, 
                              overlap_width_ratio=args.patch_overlap_ratio,
                              postprocess_class_agnostic=args.class_agnostic,
                              postprocess_type=args.postprocess,
                              postprocess_match_metric=args.postprocess_metric,
                              postprocess_match_threshold=args.postprocess_thr
                            )
        if args.dataset == 'fisheye8k':
            mm_result = [[] for _ in range(5)]
        elif args.dataset == 'fisheye8klvis':
            mm_result = [[] for _ in range(326)]
        else:
            print('fisheye8k or fisheye8klvis only')
            return

        if args.use_super_resolution:
            for out in result.to_coco_annotations():
                out['image_id'] = imgid
                x, y, w, h = out['bbox']
                x = x / args.sr_scale
                y = y / args.sr_scale
                w = w / args.sr_scale
                h = h / args.sr_scale
                out['bbox'] = [x, y, w, h]
                mm_result[out['category_id']].append(xywh2xyxy(out['bbox'])+[out['score']])
                # out['bbox'] = out['bbox'][:4]
                json_results.append(out)            

            mm_result = [np.array(bbox_list) if len(bbox_list) > 0 else np.array([[0,0,0,0,0]]) for bbox_list in mm_result]

            test_dir = args.dir
            # test_dir = '/home/data/images'
            img_vis = mmcv.imread(os.path.join(test_dir, img_name))

            show_result_pyplot(
                model,
                img_vis,
                mm_result,
                palette=args.palette,
                score_thr=args.score_thr,
                out_file=os.path.join(args.out_file, img_name))                    
        elif args.use_super_resolution == False:
            for out in result.to_coco_annotations():
                out['image_id'] = imgid
                mm_result[out['category_id']].append(xywh2xyxy(out['bbox'])+[out['score']])
                # out['bbox'] = out['bbox'][:4]
                json_results.append(out)

            mm_result = [np.array(bbox_list) if len(bbox_list) > 0 else np.array([[0,0,0,0,0]]) for bbox_list in mm_result]

            show_result_pyplot(
                model,
                img,
                mm_result,
                palette=args.palette,
                score_thr=args.score_thr,
                out_file=os.path.join(args.out_file, img_name))
        
    with open(f'{args.out_file}/submit.json', 'w') as f:
        json.dump(json_results, f)

async def async_main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    tasks = asyncio.create_task(async_inference_detector(model, args.img))
    result = await asyncio.gather(tasks)
    # show the results
    show_result_pyplot(
        model,
        args.img,
        result[0],
        palette=args.palette,
        score_thr=args.score_thr,
        out_file=args.out_file)


if __name__ == '__main__':
    args = parse_args()
    if args.async_test:
        asyncio.run(async_main(args))
    else:
        main(args)
