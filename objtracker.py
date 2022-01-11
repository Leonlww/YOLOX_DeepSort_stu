import torch
import cv2
from YOLOX.yolox.data.datasets.coco_classes import COCO_CLASSES
from objdetector import Detector
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
from utils.visualize import vis_track


class_names = COCO_CLASSES

class Tracker():
    def __init__(self, filter_class=["你需要过滤掉的类别"], model='yolox-m', ckpt='你自己训练的yolox检测模型'):
        self.detector = Detector(model, ckpt)
        cfg = get_config()
        cfg.merge_from_file("deep_sort/configs/deep_sort.yaml")
        self.deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                            max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                            nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                            max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                            use_cuda=True)
        self.filter_class = filter_class

    def update(self, image):
        info = self.detector.detect(image, visual=False)
        outputs = []
        if info['box_nums']>0:
            bbox_xywh = []
            scores = []
            cls_ids = []
            for (x1, y1, x2, y2), class_id, score  in zip(info['boxes'],info['class_ids'],info['scores']):
                if self.filter_class and class_names[int(class_id)] in self.filter_class:
                    continue
                bbox_xywh.append([int((x1+x2)/2), int((y1+y2)/2), x2-x1, y2-y1])
                scores.append(score)
                cls_ids.append(class_id)
            bbox_xywh = torch.Tensor(bbox_xywh)
            outputs = self.deepsort.update(bbox_xywh, scores, image, cls_ids)
          
            image = vis_track(image, outputs,COCO_CLASSES)

        return image, outputs
