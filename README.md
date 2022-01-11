## My human head tracked result demo
![](https://github.com/Leonlww/YOLOX_DeepSort_stu/result/road_attacked_track_result_demo2.gif)
<hr/>

## How to use this code to track using your own trained YOLOX detection model with DeepSort

 First, you have to have a trained detection model.

<hr/>

 ### So, how do you use YOLOX to train your own dataset on demand?
 

1.Download the latest version of YOLOX
```
    git clone https://github.com/Megvii-BaseDetection/YOLOX.git
```

2.Do some installation in the YOLOX directory
```
    pip install -r requirements.txt
    python setup.py develop
```

3.Install pycocotools
```
pip install cython
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python setup.py install --user
```

4.Download the pre-training weights file

I used the model yolox_m, so I downloaded the weight file yolox_m.pth and saved it under the weights directory for use.

5.Data Preparation

YOLOX can train data prepared in COCO format and VOC format. In my case, I organized my own dataset in COCO dataset format for training.Like this：
```
——datasets
————human_head
——————annotations     (Labeling information)
——————train2017       (Training set images)
——————val2017         (Validation set images)
```

6.Modify the relevant parameters under the file YOLOX-main\yolox\exp\yolox_base.py and YOLOX-main/exps/default/yolox_m.py according to the requirements

In particular, you need to pay attention to modifying the number of categories for training, and the path where your own dataset is located.
For mine:
```
        self.num_classes = 1
        self.name = "human_head"
        self.data_dir = ".../YOLOX/datasets/human_head"
        self.train_ann = ".../YOLOX/datasets/human_head/annotations/train.json"
        self.val_ann = ".../YOLOX/datasets/human_head/annotations/val.json"
```

7.Modify the name of the detection category according to your needs, the project I did was head detection, and the detection category was only head.

Since I trained using the COCO dataset format, I need to modify the file YOLOX-main/yolox/data/datasets/coco_classes.py.Like this:
```
COCO_CLASSES = (
    "head",
)
```
8.Training

There are two training methods.
(1) Modifying hyperparameters directly under the train.py file.
(2) Training in the terminal.
I choose the terminal way to train, under the YOLOX directory input command:
```
python tools/train.py \
        --exp_file exps/default/yolox_m.py \
        --device 4 \
        -b 64 \
        --fp16 \
        -o \
        -c weights/yolox_m.pth
```
The "best.pth" model file is obtained at the end of training.
<hr/>

### Write the path of your trained detection model to objdetector.py and objtracker.py
#### objdetector.py
```
class Detector():
    def __init__(self, model='yolox-m', ckpt='Path to your self-trained yolox detection model.pth'):
        super(Detector, self).__init__()
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.device = torch.device('cpu')
        self.exp = get_exp_by_name(model)
        self.test_size = self.exp.test_size 
        self.model = self.exp.get_model()
        self.model.to(self.device)
        self.model.eval()
        checkpoint = torch.load(ckpt)
        self.model.load_state_dict(checkpoint["model"])
```

#### objtracker.py

In addition to detecting model paths, you can optionally fill in "filter_class" to filter out unwanted classes.
For example： filter_clas = ["car","dog"]
```
from YOLOX.yolox.data.datasets.coco_classes import COCO_CLASSES
from objdetector import Detector
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
from utils.visualize import vis_track
class_names = COCO_CLASSES

class Tracker():
    def __init__(self, filter_class=["The categories you need to filter out"], model='yolox-m', ckpt='Path to your self-trained yolox detection model.pth'):
        self.detector = Detector(model, ckpt)
        cfg = get_config()
        cfg.merge_from_file("deep_sort/configs/deep_sort.yaml")
        self.deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                            max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                            nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                            max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                            use_cuda=True)
        self.filter_class = filter_class
```
### Finally, fill the path of the video you want to track into run.py, and configure the path to save the track results by entering the command under the YOLOX_DeepSort_stu/ path：
```
python run.py
```
