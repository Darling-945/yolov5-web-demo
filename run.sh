conda activate
rm -rf /project/datasets/banner/train.cache
rm -rf /project/datasets/banner/val.cache
python /project/yolov5/maketxt.py
python /project/yolov5/voc_label.py
python /project/yolov5/train.py






