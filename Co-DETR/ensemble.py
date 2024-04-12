from ensemble_boxes import weighted_boxes_fusion
import json
import cv2
import glob
import os
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--test_dataset_path', default='/data/')
    parser.add_argument('--target_json_dir', default='/data/ensemble_dir')
    parser.add_argument('--out_name', default='')
    parser.add_argument('--iou_thr', default=0.4)
    args = parser.parse_args()
    return args

def xywh2xyxy(bbox, shape):
        return [
            bbox[0] / shape[1],
            bbox[1] / shape[0],
            min(1., (bbox[2] + bbox[0]) / shape[1]),
            min(1., (bbox[3] + bbox[1]) / shape[0]),
        ]

def xyxy2xywh(bbox, shape):
        _bbox = bbox.tolist()
        return [
            _bbox[0] * shape[1],
            _bbox[1] * shape[0],
            (_bbox[2] - _bbox[0]) * shape[1],
            (_bbox[3] - _bbox[1]) * shape[0],
        ]
        
def main(args):
    map_id2name = {'0':'bus', '1':'bike', '2':'car', '3':'pedestrian', '4':'truck'}
    
    ann_by_img_s = []
    paths = glob.glob(os.path.join(args.target_json_dir, '*.json'))
    for path in paths:
        print(path)
        with open(path, 'r') as f:
            coco = json.load(f)
        ann_by_img = {}
        for annotation in coco:
            image_id = annotation['image_id']
            if image_id not in ann_by_img:
                ann_by_img[image_id] = []
            # sr conf thr 0.6
            if '54' in path:
                if annotation['score'] > 0.6:
                    ann_by_img[image_id].append(annotation)
            else:
                ann_by_img[image_id].append(annotation)
        ann_by_img_s.append(ann_by_img)

    image_list = list((set(list(map(lambda x: x['image_id'], coco)))))
    
    ensenble_anno = []
    for img_id in image_list:
        id = str(img_id)
        img_name = f'camera{id[:2]}_A_{id[3:]}.png' if id[2]=='1' else f'camera{id[:2]}_N_{id[3:]}.png'
        w,h,_ = cv2.imread(os.path.join(args.test_dataset_path, img_name)).shape
        bboxes_list = []
        scores_list = []
        labels_list = []
        weights = [1]*len(ann_by_img_s)
        for i, ann_by_img in enumerate(ann_by_img_s):
            if img_id in ann_by_img.keys():
                anns = ann_by_img[img_id]
                bboxes_list.append(list(map(lambda x: xywh2xyxy(x['bbox'], (w,h)), anns)))
                scores_list.append(list(map(lambda x: x['score'], anns)))
                labels_list.append(list(map(lambda x: x['category_id'], anns)))
            else:
                bboxes_list.append([])
                scores_list.append([])
                labels_list.append([])
        iou_thr = args.iou_thr
        skip_box_thr = 0.0001

        bboxes, scores, labels = weighted_boxes_fusion(bboxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
        bboxes = list(map(lambda x: xyxy2xywh(x, (w,h)), bboxes))

        for box,score,label in zip(bboxes, scores, labels):
            ensenble_anno.append({'image_id': img_id, 'bbox': box, 'score': score, 'category_id':int(label), 'category_name':map_id2name[str(int(label))]})
        
    with open(args.out_name, 'w') as f:
        json.dump(ensenble_anno, f)

if __name__=="__main__":
    args = parse_args()
    main(args)