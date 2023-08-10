from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import socket
import os
import re
import time
import glob
import easyocr
import os
import dotenv
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import xml.etree.ElementTree as ET

import tensorflow as tf





print(tf.__version__)

import time
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from PIL import Image
# from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils


import pytesseract
from dotenv import load_dotenv
load_dotenv()
import platform

if platform.system=='windows':
# pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
 pytesseract.pytesseract.tesseract_cmd = os.getenv('TESSERACT_PATH')


    



app = Flask(__name__)
CORS(app, resources={r'/api/*': {'origins': '*'}})

# Load Saved Model
cwd = os.getcwd()
PATH_TO_SAVED_MODEL = os.path.join(cwd, "saved_model")
print('Loading model...', end='')

# Load saved model and build the detection function
detect_fn=tf.saved_model.load(PATH_TO_SAVED_MODEL)
print('Done!')

category_index = {
    1: {'id': 1, 'name': 'dob'},
    2: {'id': 2, 'name':  'father_name'},
    3: {'id': 3, 'name': 'name'},
    4: {'id': 4, 'name': 'pan_num'}
   
}

#Loading the label_map
# PATH_TO_LABELMAP = os.path.join(cwd, 'data', 'label_map.pbtxt')
# category_index=label_map_util.create_category_index_from_labelmap(PATH_TO_LABELMAP,use_display_name=True)

@app.route('/api/scan-pan', methods=['POST'])
@cross_origin()
def panscan():
    try:
        if request.method != 'POST':
            return jsonify({ "status": 405, "error": "Method not allowed."}), 405
            
        file = request.files['file']
        if file:
            allowed_files = ['jpg', 'png', 'JPEG', 'jpeg']
            if file.filename.split('.')[-1] not in allowed_files:
                return jsonify({ "status": 400, "error": "File type is not allowed. It must be in jpg, png, or JPEG format."}), 400
            
            im = Image.open(file)

            # Check if image is in portrait mode
            if im.size[0] < im.size[1]:
                # Rotate the image by 90 degrees to switch to landscape mode
                im = im.transpose(Image.ROTATE_90)

            # Save the image
            # filename = f'{time.time_ns()}_{file.filename}'
            # # file.save('uploads/' + filename)
            # im.save('uploads/' + filename)

            def load_image_into_numpy_array(path):
                return np.array(path)
                
            # image_path = os.path.join(cwd, "uploads", filename)

            image_np = load_image_into_numpy_array(im)

            # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
            input_tensor = tf.convert_to_tensor(image_np)

            # The model expects a batch of images, so add an axis with `tf.newaxis`.
            input_tensor = input_tensor[tf.newaxis, ...]

            detections = detect_fn(input_tensor)

            # All outputs are batches tensors.
            # Convert to numpy arrays, and take index [0] to remove the batch dimension.
            # We're only interested in the first num_detections.
            num_detections = int(detections.pop('num_detections'))
            print('num_detections==> ', num_detections)

            detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
            detections['num_detections'] = num_detections
            # detection_classes should be ints.
            detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

            image_np_with_detections = image_np.copy()
            # print('image with detection-> ', image_np_with_detections)

            viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                detections['detection_boxes'],
                detections['detection_classes'],
                detections['detection_scores'],
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=200,
                min_score_thresh=.3, # Adjust this value to set the minimum probability boxes to be classified as True
                agnostic_mode=False)

            # APPLY OCR TO DETECTION
            detection_threshold = 0.4
            image = image_np_with_detections
            scores = list(filter(lambda x: x> detection_threshold, detections['detection_scores']))
            boxes = detections['detection_boxes'][:len(scores)]
            classes = detections['detection_classes']
            width = image.shape[1]
            height = image.shape[0]

            results_arr = []

            for idx, (box, cls, score) in enumerate(zip(boxes, classes, scores)):
                # if category_index[cls]['name'] == 'cheque':
                #     continue
                roi = box*[height, width, height, width]

                region = image[int(roi[0]):int(roi[2]),int(roi[1]):int(roi[3])]

                if category_index[cls]['name'] == 'pan_num':
                    reader = easyocr.Reader(['en']) 
                    results = reader.readtext(region)[0][1]
                    print(results)


    

                    
                    # result = pytesseract.image_to_string(region, lang='eng')
                    # print(result)
                    # pan_card_number = re.search(r'[A-Z]{5}[0-9]{4}[A-Z]{1}', results)
                    # pan_pattern = re.search(r"\b\d{10}\b",result)
                    # pan_pattern = re.search(r'[A-Z]{5}[0-9]{4}[A-Z]{1}',results)
                    # matches=re.findall(pan_pattern,results)
                    
                        # if results:
                        #     isExists = any(info['label'] == 'pan_num' for info in results_arr)
                        #     if isExists:
                        #         continue
                        #     results_arr.append({
                        #         'label': 'pan_num',
                        #         'value': matches[0]
                        #     })
                        
                    # print(result)
                    # pan=result.group()

                    
                    # print(pan)
                    if results:
                        isExists = any(info['label'] == 'pan_num' for info in results_arr)
                        if isExists:
                            continue
                        results_arr.append({
                            'label': 'pan_num',
                            'value': results[:10]
                        })
             
                    continue
                
                """ Name """
                if category_index[cls]['name'] == 'name':
                    # reader = easyocr.Reader(['en']) 
                    # results = reader.readtext(region)[0][1]
                    # print(results)
                    results = pytesseract.image_to_string(region, lang='eng').strip()
                    print(results)
                    if results:
                        isExists = any(info['label'] == 'name' for info in results_arr)
                        if isExists:
                            continue
                        results_arr.append({
                            'label': 'name',
                            'value': results
                            
                        })
                    continue

                
                


                if category_index[cls]['name'] == 'dob':
                    reader = easyocr.Reader(['en']) 
                    results = reader.readtext(region)[0][1]
                    print(results)
                   
                    # txt = pytesseract.image_to_string(region, lang='eng')
                    # print(txt)
                    # dob_pattern = re.compile(r'\d{2}/\d{2}/\d{4}')
                    # txt=dob_pattern.search(txt).group()
                    
                    if results:
                        isExists = any(info['label'] == 'dob' for info in results_arr)
                        if isExists:
                            continue
                        results_arr.append({
                            'label': 'dob',
                            'value': results
                        })
                        # if accNo == '911010049001545':
                        #     results_arr.append({
                        #         'label': 'ifsc',
                        #         'value': 'UTIB0000426'
                        #     })
                    continue

                
                
                if category_index[cls]['name'] == 'father_name':
                    reader = easyocr.Reader(['en']) 
                    results = reader.readtext(region)[0][1]
                    print(results)
                    # result = pytesseract.image_to_string(region, lang='eng').split('\n')
                    # print(result)
                    

                    if results:
                        isExists = any(info['label'] == 'father_name' for info in results_arr)
                        if isExists:
                            continue
                        results_arr.append({
                            'label': 'father_name',
                            'value': results
                        })

                

            print(results_arr)
            return jsonify({ "status": 200, "data": results_arr}), 200

        else:
            return jsonify({ "status": 500, "error": "Failed to upload file. Please try again."}), 500

    except Exception as e:
        print("ERROR: ", e)
        return jsonify({ "status": 500, "error": "Unable to read the pan. Please upload clear image again."}), 500
    
if __name__ == '__main__':
    # Get the hostname of the current machine
    hostname = socket.gethostname()
    # Get the IP address of the current machine
    ip_address = socket.gethostbyname(hostname)
    app.run( host='0.0.0.0', port='4000', debug=True)
