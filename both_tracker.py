import torch
import numpy as np
import cv2
from YOLOX.yolox.data.datasets.coco_classes import COCO_CLASSES
from both_detector import BodyDetector
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
from utils.visualize import vis_track


class_names = COCO_CLASSES

class Tracker():
    def __init__(self, filter_class=["mask"], bodymodel='yolox-s', bodyckpt='/datav/shared/leon/YOLOX_DeepSort_stu/YOLOX/weights/crowdhuman_vbox_yolox_s_best_ckpt.pth'):
        self.bodydetector = BodyDetector(bodymodel, bodyckpt)
        cfg = get_config()
        cfg.merge_from_file("deep_sort/configs/deep_sort.yaml")
        self.deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                            max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                            nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                            max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                            use_cuda=True)
        self.filter_class = filter_class

        
        # self.headdetector = HeadDetector(headmodel, headckpt)
        
    

    def update(self, image):
        info = self.bodydetector.detect(image, visual=False)
        outputs = []
        if info['box_nums']>0:
            body_bbox_xywh = []
            body_scores = []

            head_bbox_xywh = []
            head_scores = []

            bbox_xywh = []
            scores = []
            #bbox_xywh = torch.zeros((info['box_nums'], 4))
            for (x1, y1, x2, y2), class_id, score  in zip(info['boxes'],info['class_ids'],info['scores']):
                if self.filter_class and class_names[int(class_id)] in self.filter_class:
                    continue

                if class_id == 0:
                    body_bbox_xywh.append([int((x1+x2)/2), int((y1+y2)/2), x2-x1, y2-y1])
                    body_scores.append(score)
                if class_id == 2:
                    head_bbox_xywh.append([int((x1+x2)/2), int((y1+y2)/2), x2-x1, y2-y1])
                    head_scores.append(score)
            scores = body_scores + head_scores
            bbox_xywh = body_bbox_xywh + head_bbox_xywh 
            bbox_xywh = torch.Tensor(bbox_xywh)
            outputs = self.deepsort.update(bbox_xywh, scores, image)
            vis_box = []

            for i in range(len(outputs)):
                for j in range(len(outputs)):
                    if i != j:
                        res = calculate_body_and_head(outputs[i][:4],outputs[j][:4])
                        if res>0.99:
                            outputs[i][4:5] = outputs[j][4:5]
                            vis_box.append(outputs[i])
            if len(vis_box) > 0:
                vis_box = np.stack(vis_box,axis=0)


            # image = vis_track(image, outputs,info['class_ids'],COCO_CLASSES)
            image = vis_track(image, vis_box,COCO_CLASSES)

        return image, outputs

def calculate_body_and_head(box1, box2):
        x1, y1, x2, y2 = box1
        xx1, yy1, xx2, yy2 = box2
    
        # 交
        w = max(0, min(x2, xx2) - max(x1, xx1))
        h = max(0, min(y2, yy2) - max(y1, yy1))
        interarea = w * h
    
        # 并
        union = (x2-x1)*(y2-y1) + (xx2-xx1)*(yy2-yy1) - interarea
    
        # 交并比
        # return interarea / union

        area1 = (x2 - x1) * (y2 -y1)
        # area2 = (xx2 - xx1) * (yy2 - yy1)
        res = interarea/area1
        return res