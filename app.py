import argparse
from flask import Flask, render_template, request , send_file
from detect import main
import os
from math import floor
from PIL import Image
import numpy as np
import hashlib

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'uploads'
savefolder = './static'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/infer', methods=['POST'])
def success():
    global savefolder
    global saveinput
    if request.method == 'POST':
        f = request.files['file']
        saveLocation = f.filename
        saveinput = savefolder +'/img/'+ saveLocation
        f.save(saveinput)
        opt = parse_opt()
        main(opt)
        output_image = savefolder +'/img_out/exp/' + saveLocation
        
        return render_template('inference.html' , saveLocation=saveinput , output_image=output_image), saveinput
    
def parse_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument('--weights', nargs='+', type=str, default= 'best.pt', help='model path or triton URL')
    #parser.add_argument('--source', type=str, default=0, help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--source', type=str, default= saveinput, help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default= 'models/yolov5s.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--project', default= 'static/img_out', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    #print_args(vars(opt))
    args = parser.parse_args(args=[])
    print(args)
    return opt


if __name__ == '__main__':
    app.debug = True
    port = int(os.environ.get("PORT", 8001))
    app.run(host='localhost', port=port, debug=True)