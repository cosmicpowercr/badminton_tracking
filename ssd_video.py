import numpy as np
import os
import time
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import matplotlib
import cv2 as cv
import matplotlib.pyplot as plt
from collections import defaultdict
from io import StringIO
from PIL import Image
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# Matplotlib chooses Xwindows backend by default.
matplotlib.use('Agg')


##################### Download Model
# What model to download.
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS ='C:\\dataset\\VOC2007\\run\\data\\mscoco_label_map.pbtxt' #os.path.join('data', 'mscoco_label_map.pbtxt')
PATH_TO_CKPT='C:\\dataset\\VOC2007\\run\\data\\frozen_inference_graph.pb'
NUM_CLASSES = 90
print('1', PATH_TO_CKPT)
# Download model if not already downloaded
if not os.path.exists(PATH_TO_CKPT):
    #print('1',PATH_TO_CKPT)
    print('Downloading model... (This may take over 5 minutes)')
    opener = urllib.request.URLopener()
    opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
    print('Extracting...')
    tar_file = tarfile.open(MODEL_FILE)
    for file in tar_file.getmembers():
        file_name = os.path.basename(file.name)
        if 'frozen_inference_graph.pb' in file_name:
            tar_file.extract(file, os.getcwd())
else:
    print('Model already downloaded.')

##################### Load a (frozen) Tensorflow model into memory.
print('Loading model...')
detection_graph = tf.Graph()

with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


##################### Loading label map
print('Loading label map...')
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

##################### Helper code
def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

##################### Detection
# Path to test image
#path to video
VideoCaptrue = cv.VideoCapture('test16.MOV')

#get the fps and size
fps = VideoCaptrue.get(cv.CAP_PROP_FPS)
size = (int(VideoCaptrue.get(cv.CAP_PROP_FRAME_WIDTH)),
        int(VideoCaptrue.get(cv.CAP_PROP_FRAME_HEIGHT)))
numframes = VideoCaptrue.get(cv.CAP_PROP_FRAME_COUNT)
print('num of frames:'+ str(numframes))

#write
#VideoWriter=cv.VideoWriter('bad.avi', 0, fps, size)

#TEST_IMAGE_PATH = 'C:\\Users\\cr\\Desktop\\dataset\\VOC2007\\run\\test_images\\15.jpg'

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)


'''
    # 显示与保存图片
    print(TEST_IMAGE_PATH.split('.')[0]+'_labeled.jpg')
    plt.figure(figsize=IMAGE_SIZE, dpi=300)
    plt.imshow(image_np)
    plt.savefig(TEST_IMAGE_PATH.split('.')[0] + '_labeled.jpg')
'''
temp = np.repeat([0], 99)
##################### Detecting
def detect(image_np, image_tensor, boxes, scores, classes, num_detections, temp, i):
    # print('Detecting...')
    # print(TEST_IMAGE_PATH)
    # image = Image.open(TEST_IMAGE_PATH)
    # 将图片转入numpy中
    #image_np = load_image_into_numpy_array(image, size)
    #image_np = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    # 维数扩大
    image_np_expanded = np.expand_dims(image_np, axis=0)

    # 生成检测结构
    # get_tensor_by_name:返回给定名称的tensor
    #image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    #boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    #scores = detection_graph.get_tensor_by_name('detection_scores:0')
    #classes = detection_graph.get_tensor_by_name('detection_classes:0')
    #num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    # Actual detection.
    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})
    '''
    #limit
    if np.squeeze(scores)[0] < 0.6:
	    scores_h = 0.
    else:
        i = i+1
        scores_h = np.squeeze(scores)[0]

    scores = np.append(scores_h, temp)
    '''
    # Print the results of a detection.
    # print(scores)
    # print(classes)
    # print(category_index)
    # Visualization of the results of a detection.将结果显示在image上
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8)

    return image_np, i

################# Detecting
with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    #print(TEST_IMAGE_PATH)
    #image = Image.open(TEST_IMAGE_PATH)
    #image_np = load_image_into_numpy_array(image)
    #image_np_expanded = np.expand_dims(image_np, axis=0)
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    count = 0
    star  = time.clock()
    while(VideoCaptrue.isOpened()):
        ret, frame = VideoCaptrue.read()
        if ret == True:
            #cv.imshow('frame', frame)
            detected_image, count = detect(frame, image_tensor, boxes, scores, classes, num_detections, temp, count)
            #VideoWriter.write(detected_image)
            #detected_image = cv.cvtColor(detected_image, cv.COLOR_BGR2BGRA)
            cv.imshow('detected_image', detected_image)
            k = cv.waitKey(20)
            if (k & 0xff == ord('q')):
                break
        else:
            break
    end = time.clock()

    print('count:'+str(count))
    print('Runing time.%s Sec' % (end - star))
    VideoCaptrue.release()
    cv.destroyAllWindows()