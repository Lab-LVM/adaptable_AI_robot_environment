from flask import Flask, render_template, Response, request, jsonify, send_file, redirect
import cv2
import glob
import numpy as np
import yaml
import os
import signal
import sys
from pathlib import Path
import psutil
import subprocess
import time
import shutil
import flask_detect
from PIL import Image
from skimage.metrics import structural_similarity as ssim 
import shutil
import re
from flask_cors import CORS

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT)) 
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

from models.common import DetectMultiBackend
from utils.general import check_img_size

app = Flask(__name__, static_url_path='./static', static_folder='static')
CORS(app)

def unconvert(class_id, width, height, x, y, w, h):

    xmax = int((x*width) + (w * width)/2.0)
    xmin = int((x*width) - (w * width)/2.0)
    ymax = int((y*height) + (h * height)/2.0)
    ymin = int((y*height) - (h * height)/2.0)
    class_id = int(class_id)
    return (class_id, xmin, xmax, ymin, ymax)


@app.route('/get_detected_image/<image_filename>')
def get_detected_image(image_filename):
    DETECTED_IMAGES_PATH = './static/detection_save'
    image_path = os.path.join(DETECTED_IMAGES_PATH, image_filename)
    if os.path.exists(image_path):
        return send_file(image_path, mimetype='image/jpeg')
    else:
        return 'Image not found', 404

@app.route('/')
def index():
    app.logger.info('Accessed index page') 
    return render_template('index.html')

@app.route('/detect_objects', methods=['POST'])
def detect_objects_endpoint():

    model_dir = "./save_model"
    exist_model = 'yolov5s.pt'
    entries = os.listdir(model_dir)
    if len(entries) >= 1:
        directories = [entry for entry in entries if os.path.isdir(os.path.join(model_dir, entry))]
        sorted_directories = sorted(directories, key=lambda x: os.path.getctime(os.path.join(model_dir, x)), reverse=True)
        most_recent_directory = sorted_directories[0]
        pth_files_dir = most_recent_directory + '/' + 'weights'
        pth_files = glob.glob(os.path.join(pth_files_dir, '*.pt'))
        if len(pth_files) == 0:
            latest_model_file = False
            weights = exist_model
        else:
            latest_model_file = pth_files[0]
            weights = latest_model_file
    else:
        latest_model_file = False
        weights = exist_model

    yaml_dir = './save_yaml/'
    yaml_files = glob.glob(os.path.join(yaml_dir, '*.yaml'))
    if yaml_files:
        yaml_files.sort(key=os.path.getctime)
        latest_yaml_file = yaml_files[-1]
    else:
        latest_yaml_file = ROOT / 'data/coco.yaml'

    with open(latest_yaml_file, 'r') as yaml_file:
        yaml_data = yaml.load(yaml_file, Loader=yaml.SafeLoader)

    if latest_yaml_file and latest_model_file:
        data_yaml = latest_yaml_file
    else:
        data_yaml = ROOT / 'data/coco.yaml'

    unknown_start_index = None
    if 'names' in yaml_data:
        names = yaml_data['names']
        for index, class_name in names.items():
            if class_name == 'unknown':
                unknown_start_index = index
                break

    now = time
    filename = f"{now.localtime().tm_year}-{now.localtime().tm_mon}-{now.localtime().tm_mday}-{now.localtime().tm_hour}-{now.localtime().tm_min}-{now.localtime().tm_sec}"
    SAVE_PATH = './save'

    print('weights :', weights)
    model = DetectMultiBackend(weights, dnn=False, data=data_yaml, fp16=False)
    stride, names, pt = model.stride, model.names, model.pt
    for i in range(80,160): 
        names[i] = "unknown" 

    frame_data = request.files['frame'].read()
    frame_array = np.frombuffer(frame_data, np.uint8)
    frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)

    height, width, _ = frame.shape
    all_zeros = True
    for y in range(height):
        for x in range(width):
            pixel_value = frame[y, x]
            if any(pixel_value != 0):
                all_zeros = False
                break
        if not all_zeros:
            break

    if all_zeros == False:
        save_path = os.path.join(SAVE_PATH, filename+'.jpg')
        cv2.imwrite(save_path, frame)
        flask_detect.flask_detect(save_path+'/'+filename+'.jpg', model, stride, pt, names, filename, unknown_start_index)

        image_url = f'./static/detection_save/{filename}.jpg'

        return jsonify({'image_url': image_url})
    

@app.route('/generate_yaml_and_download', methods=['POST'])
def generate_yaml_and_download():

    model_dir = "./save_model"
    entries = os.listdir(model_dir)
    if len(entries) >= 1:
        directories = [entry for entry in entries if os.path.isdir(os.path.join(model_dir, entry))]
        sorted_directories = sorted(directories, key=lambda x: os.path.getctime(os.path.join(model_dir, x)), reverse=True)
        most_recent_directory = sorted_directories[0]
        pth_files_dir = most_recent_directory + '/' + 'weights'
        pth_files = glob.glob(os.path.join(pth_files_dir, '*.pth'))
        latest_model_file = pth_files
    else:
        latest_model_file = False

    yaml_dir = './save_yaml/'
    yaml_files = glob.glob(os.path.join(yaml_dir, '*.yaml'))
    if yaml_files:
        yaml_files.sort(key=os.path.getctime)
        latest_yaml_file = yaml_files[-1]
    else:
        latest_yaml_file = False

    if latest_yaml_file and latest_model_file:
        yaml_file_path = latest_yaml_file
    else:
        yaml_file_path = './data/coco.yaml'

    with open(yaml_file_path, 'r') as yaml_file:
        yaml_data = yaml.load(yaml_file, Loader=yaml.SafeLoader)

    
    if request.method == 'POST':
        word = request.form.get('word')
        print('word debug:', word)

        if word is None or word.strip() == '' :
            return jsonify({'message': 'No word provided'}), 400
        
        unknown_start_index = None
        if 'names' in yaml_data:
            names = yaml_data['names'] 
            for index, class_name in names.items():
                if class_name == 'unknown':
                    unknown_start_index = index
                    break

        yaml_data['names'][unknown_start_index] = word
        
        config_data = {
            'path': '../datasets/coco',
            'train': 'train2017.txt',
            'val': 'val2017.txt',
            'test': 'test-dev2017.txt',
            'names': yaml_data['names'],
            'download': """
                from utils.general import download, Path

                # Download labels
                segments = False  # segment or box labels
                dir = Path(yaml['path'])  # dataset root dir
                url = 'https://github.com/ultralytics/yolov5/releases/download/v1.0/'
                urls = [url + ('coco2017labels-segments.zip' if segments else 'coco2017labels.zip')]  # labels
                download(urls, dir=dir.parent)

                # Download data
                urls = ['http://images.cocodataset.org/zips/train2017.zip',  # 19G, 118k images
                        'http://images.cocodataset.org/zips/val2017.zip',  # 1G, 5k images
                        'http://images.cocodataset.org/zips/test2017.zip']  # 7G, 41k images (optional)
                download(urls, dir=dir / 'images', threads=3)
                """
            }

        now = time
        yamlname = f"{now.localtime().tm_year}-{now.localtime().tm_mon}-{now.localtime().tm_mday}-{now.localtime().tm_hour}-{now.localtime().tm_min}-{now.localtime().tm_sec}"
        yaml_filename = 'coco'+'_'+yamlname+'.yaml'

        new_yaml_file_path = os.path.join(yaml_dir, yaml_filename)
        with open(new_yaml_file_path, 'w') as yaml_file:
            yaml.dump(config_data, yaml_file, default_style='"')
        return send_file(new_yaml_file_path, as_attachment=True)

main_process = None
@app.route('/train', methods=['POST']) 
def train():
    print('trainButton is clicked!')
    global main_process
    if main_process is None or main_process.returncode is not None:
        print('\nmain_process pass!\n')
        main_process = subprocess.Popen(['python', 'train.py'], stdout=subprocess.PIPE, text=True)
        print("\n\nmain_process PID: ", main_process.pid, "\n")
        log = main_process.stdout
        return log


@app.route('/stop', methods=['POST'])
def stop_training():
    print('stopButton is clicked!')
    global main_process
    if main_process:
        process = psutil.Process(main_process.pid)
        for child in process.children(recursive=True):
            child.terminate()
        process.terminate()
        process.wait()
        main_process = None
        return 'Train이 중지되었습니다.'
    else:
        return 'Train이 실행 중이 아닙니다.'
    

@app.route('/restart', methods=['POST'])
def restart():
    os.execl(sys.executable, sys.executable, *sys.argv)
    return 'Flask 서버가 재시작되었습니다.'


def signal_handler(sig, frame):
    sys.exit(0)


@app.route('/align-dataset', methods=['GET', 'POST'])
def align_dataset():
    similarconf = 0.1
    if request.method == 'POST':
        if 'image' not in request.files:
            return jsonify({'error': 'No file part'})
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        try:
            temp_filepath = './unknown_label_image/' + file.filename
            file.save(temp_filepath)
            label_image = Image.open(file)
            label_image = label_image.convert('L')
            width, height = label_image.size

            now = time
            foldername = f"{now.localtime().tm_year}-{now.localtime().tm_mon}-{now.localtime().tm_mday}-{now.localtime().tm_hour}-{now.localtime().tm_min}-{now.localtime().tm_sec}"
            image_directory_train_path = f'./datasets/coco/images/train2017/{foldername}'
            label_directory_train_path = f'./datasets/coco/labels/train2017/{foldername}'
            image_directory_val_path = f'./datasets/coco/images/val2017/{foldername}'
            label_directory_val_path = f'./datasets/coco/labels/val2017/{foldername}'
            os.makedirs(image_directory_train_path)
            os.makedirs(label_directory_train_path)
            os.makedirs(image_directory_val_path)
            os.makedirs(label_directory_val_path)

            save_image_directory_path = './save'
            save_label_directory_path = './annosave'

            save_label = os.listdir(save_label_directory_path)
            vaild_save_label = []

            for i in save_label:
                count_label = 0
                unknown_label_number = float('-inf')
                with open(save_label_directory_path+'/'+i, 'r') as file:
                    lines = file.readlines()
                for line in lines:
                    elements = line.split()
                    for element in elements:
                        number = float(element)
                        if number >= 80 and number > unknown_label_number:
                            unknown_label_number = number
                    if elements and int(elements[0]) == unknown_label_number:
                        count_label += 1
                if count_label == 1:
                    vaild_save_label.append(i)

            unknown_save_image = []
            for i in vaild_save_label:
                filename = i.split('.txt')[0]+'.jpg'
                unknown_save_image.append(filename)

            similar_images = []
            for i in unknown_save_image:
                print('image :', i)
                img = cv2.imread(save_image_directory_path+'/'+i, cv2.COLOR_BGR2RGB)
                crop_height, crop_width, _ = img.shape
                txtfilename = i.split('.jpg')[0]+'.txt'
                with open(save_label_directory_path+'/'+txtfilename, 'r') as file:
                    lines = file.readlines()
                for line in lines:
                    parts = line.split()  
                    first_number = int(parts[0])  
                    if first_number >= 80:
                        _, xmin, xmax, ymin, ymax = unconvert(parts[0], crop_width, crop_height, float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
                        if xmin == -1:
                            xmin = 0
                        if xmax == -1:
                            xmax = 0
                        if ymin == -1:
                            ymin = 0
                        if ymax == -1:
                            ymax = 0
                        cropped_image = img[ymin:ymax, xmin:xmax]
                        cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2GRAY) 
                        resize_cropped_image = cv2.resize(cropped_image, dsize=(width, height), interpolation=cv2.INTER_AREA)
                        label_image = np.array(label_image)

                        (score, _) = ssim(label_image, resize_cropped_image, full=True)
                        if score > similarconf :
                            similar_images.append(i)

            similar_labels = []
            for i in similar_images:
                filename = i.split('.jpg')[0]+'.txt'
                similar_labels.append(filename)

            similar_labels.sort() 
            similar_images.sort() 

            split_train_val = round(int(len(similar_images)) * 0.7)
            save_label_train = similar_labels[:split_train_val]
            save_label_val = similar_labels[split_train_val:]
            save_image_train = similar_images[:split_train_val]
            save_image_val = similar_images[split_train_val:]

            for a, c in zip(save_label_train, save_image_train):
                label_file_train = os.path.join(save_label_directory_path, a)
                image_file_train = os.path.join(save_image_directory_path, c)
                shutil.move(label_file_train, label_directory_train_path)
                shutil.move(image_file_train, image_directory_train_path)

            for b, d in zip(save_label_val, save_image_val):
                label_file_val = os.path.join(save_label_directory_path, b)
                image_file_val = os.path.join(save_image_directory_path, d)
                shutil.move(label_file_val, label_directory_val_path)
                shutil.move(image_file_val, image_directory_val_path)

            image_directory_train_align_path = os.listdir(image_directory_train_path)
            image_directory_train_align_path = [image_directory_train_path + '/' + item + '\n' for item in image_directory_train_align_path]
            image_directory_val_align_path = os.listdir(image_directory_val_path)
            image_directory_val_align_path = [image_directory_val_path + '/' + item + '\n' for item in image_directory_val_align_path]

            with open('./datasets/coco/train2017.txt', 'r') as file:
                train_contents = file.readlines()
            train_contents.extend(image_directory_train_align_path)
            with open('./datasets/coco/train2017.txt', 'w') as file:
                file.writelines(train_contents)

            with open('./datasets/coco/val2017.txt', 'r') as file:
                val_contents = file.readlines()
            val_contents.extend(image_directory_val_align_path)
            with open('./datasets/coco/val2017.txt', 'w') as file:
                file.writelines(val_contents)

            return redirect('/')
        
        except:
                pass

if __name__ == '__main__':
    def signal_handler(sig, frame):
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)

    app.run(host='0.0.0.0',port='your port',debug=True)

