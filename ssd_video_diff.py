import numpy as np
from numpy import *
import os
import time
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import matplotlib
import cv2
import matplotlib.pyplot as plt
from collections import defaultdict
from io import StringIO
from PIL import Image
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

def Tbar(int, *void):pass


gb_size = 4
kernel_size = 2
mb_size = 22
erode_times = 1
dilate_times = 4
threshold_diff = 50
number_frame = 0
pre_gray = 0
ret = True
kernel = np.ones((kernel_size * 2 + 1, kernel_size * 2 + 1), np.uint8)
pre_bin = 0
temp = np.repeat([0], 99)
box_cr = matrix([[], []])

##################### Window and Tbar
cv2.namedWindow("settings", 1)
cv2.createTrackbar("gb_size", "settings", gb_size, 20, Tbar)
cv2.createTrackbar("kernel_size", "settings", kernel_size, 20, Tbar)
cv2.createTrackbar("threshold_diff", "settings", threshold_diff, 255, Tbar)
#cv2.createTrackbar("erode_times1", "settings", erode_times, 20, Tbar)
#cv2.createTrackbar("dilate_times", "settings", dilate_times, 20, Tbar)
#cv2.createTrackbar("mb_size", "settings", mb_size, 100, Tbar)
##################### Window and Tbar

##################### VideoCapture
cap = cv2.VideoCapture('test8.mp4')
#cap = cv2.VideoCapture(0)
fps = cap.get(cv2.CAP_PROP_FPS)
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
numframes = cap.get(cv2.CAP_PROP_FRAME_COUNT)
##################### VideoCapture

#write
#VideoWriter = cv2.VideoWriter('cr.avi', 0, fps, size)

##################### Downloadandloading
def Downloadandloading():
    ##################### Download Model
    # What model to download.
    MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
    MODEL_FILE = MODEL_NAME + '.tar.gz'
    DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = 'C:\\Users\\cr\\Desktop\\dataset\\VOC2007\\run\\data\\mscoco_label_map.pbtxt'
    # os.path.join('data', 'mscoco_label_map.pbtxt')
    PATH_TO_CKPT = 'C:\\Users\\cr\\Desktop\\dataset\\VOC2007\\run\\ssd_mobilenet_v1_coco_11_06_2017\\frozen_inference_graph.pb'
    NUM_CLASSES = 90
    print('1', PATH_TO_CKPT)
    # Download model if not already downloaded
    if not os.path.exists(PATH_TO_CKPT):
        # print('1',PATH_TO_CKPT)
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
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                                use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    return category_index, detection_graph
##################### Downloadandloading

##################### Limit the box
def limit(boxes, scores):
    # '''
    # limit...
    global box_cr
    scores_h = 0.
    threshold = 9000
    temp_cr = eye(2, 1)
    decision = 0
    if (np.squeeze(scores)[0] < 0.6):
        scores_h = 0.
        scores = np.append(scores_h, temp)
        return scores
    else:
        # i = i + 1
        temp_cr[0, 0] = size[0]*(np.squeeze(boxes)[0, 1] + np.squeeze(boxes)[0, 3]) / 2
        temp_cr[1, 0] = size[1]*(np.squeeze(boxes)[0, 0] + np.squeeze(boxes)[0, 2]) / 2
    if (shape(box_cr)[1] == 3):
        A = matrix([[pow(box_cr[0, 0], 2), box_cr[0, 0], 1],
                    [pow(box_cr[0, 1], 2), box_cr[0, 1], 1],
                    [pow(box_cr[0, 2], 2), box_cr[0, 2], 1]])
        B = matrix([[box_cr[1, 0]], [box_cr[1, 1]], [box_cr[1, 2]]])
        belta = A.I * B
        print(belta)
        decision = pow(belta[0] * pow(temp_cr[0, 0], 2) + belta[1] * temp_cr[0, 0] + belta[2] - temp_cr[1, 0], 2)
        # draw the line...
        x = np.linspace(0, 50, 100)
        belta = belta.getA() # 不懂为什么要这样
        plt.figure(1)  # ❶ # 选择图表1
        plt.plot(x, pow(multiply(belta[0], x), 2) + belta[1] * x + belta[2])
        #plt.show()
        # ...draw the line
        if (decision > threshold):
            scores_h = 0.

        else:
            print('decision:', decision)
            scores_h = np.squeeze(scores)[0]
            temp_m = box_cr[:, 0:2]
            box_cr = hstack((temp_cr, temp_m))
    elif(shape(box_cr)[1] < 3):
        box_cr = hstack((temp_cr, box_cr))
    print('cr:', box_cr)
    scores = np.append(scores_h, temp)
        # ...limit
        # '''
    return scores


##################### Detecting
def detect(image_np, image_tensor, boxes, scores, classes, num_detections, temp, i):
    image_np_expanded = np.expand_dims(image_np, axis=0)

    # Actual detection...
    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})
    # ...Actual detection
    #print('1:', scores)

    # scores = limit(boxes, scores)
    # '''''
    if np.squeeze(scores)[0] < 0.6:
	    scores_h = 0.
    else:
        i = i+1
        scores_h = np.squeeze(scores)[0]

    scores = np.append(scores_h, temp)
    # '''''
    # Print the results of a detection.
    # print(scores)
    # print(classes)
    # print(category_index)
    # print('2:', scores)
    # Visualization of the results of a detection...
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        scores,#  scores
        category_index,
        use_normalized_coordinates=True,
        line_thickness=4)
    # ...Visualization of the results of a detection
    return image_np,  i
##################### Detecting

#####################
print('size:', size)

category_index, detection_graph = Downloadandloading()
with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        # print(TEST_IMAGE_PATH)
        # image = Image.open(TEST_IMAGE_PATH)
        # image_np = load_image_into_numpy_array(image)
        # image_np_expanded = np.expand_dims(image_np, axis=0)
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        scores = detection_graph.get_tensor_by_name('detection_scores:0')
        classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        count = 0
        #star = time.clock()
        print('Detecting...')
        while (cap.isOpened()):

            if (ret == True):

                ##################### Running Tbar
                gb_size = cv2.getTrackbarPos("gb_size", "settings")
                kernel_size = cv2.getTrackbarPos("kernel_size", "settings")
                threshold_diff = cv2.getTrackbarPos("threshold_diff", "settings")
                #erode_times = cv2.getTrackbarPos("erode_times1", "settings")
                #dilate_times = cv2.getTrackbarPos("dilate_times", "settings")
                #mb_size = cv2.getTrackbarPos("mb_size", "settings")
                ##################### Running Tbar

                number_frame += 1
                print(number_frame)
                if (number_frame == 1):
                    ret, frame = cap.read()
                    blur = cv2.GaussianBlur(frame, (gb_size * 2 + 1, gb_size * 2 + 1), 0)
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    pre_gray = gray.copy()
                    # cv2.imshow('video_src', frame)
                    continue
                else:
                    ret, frame = cap.read()
                    blur = cv2.GaussianBlur(frame, (gb_size * 2 + 1, gb_size * 2 + 1), 0)
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    # cv2.imshow('video_src', frame)

                    gray_diff = gray.copy()
                    cv2.subtract(pre_gray, gray, gray_diff)
                    # cv2.imshow('video_diff', gray_diff)

                    ret0, binary = cv2.threshold(abs(gray_diff), threshold_diff, 255, cv2.THRESH_BINARY)
                    # cv2.imshow("binary_src", binary)
                    binary = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

                    ##################### Detecting by ssd and diff...
                    # cv2.imshow('frame', frame)
                    # '''''
                    detected_image, count = detect(
                        binary, image_tensor, boxes, scores,
                        classes, num_detections, temp, count)
                    # '''''
                    #VideoWriter.write(binary)
                    # detected_image = cv2.cvtColor(detected_image, cv2.COLOR_BGR2BGRA)
                    cv2.imshow('detected_image', detected_image)
                    #VideoWriter.write(detected_image)
                    #end = time.clock()
                    ##################### ...Detecting by ssd and diff

                    pre_gray = gray.copy()

                if (number_frame > 544):
                        ret = False
                if cv2.waitKey(50) & 0xFF == ord('q'):
                        break
                        # if number_frame == 50:
                        #    break
            else:
                break
    cap.release()
    cv2.destroyAllWindows()





