import os, cv2, numpy

class MotionDetect():
    def __init__(self):
        self.background = None
        self.motionDetected = False
        self._kernel = numpy.ones((4,4),numpy.uint8)
    def contours(self, image):
        #        image = cv2.dilate(image, None, iterations=15);
        #        image = cv2.erode(image, None, iterations=9);
        image, contours, heirarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        movementLocations = []
        for contour in contours:
            rect = cv2.boundingRect(contour)
            movementLocations.append(rect)
        return movementLocations
    def detect(self, image, timestamp=None):
        currentImage = image
        # cv2.imshow("currentImage", currentImage)
        foreground = cv2.blur(currentImage, (7, 7))
        # cv2.imshow("foreground", foreground)
        if self.background is None:
            self.background = numpy.float32(foreground)
        # 如果model是当前背景模型，cur是当前帧去除背景后的图片，则新的模型为：
        #   modelnew = (1-a)*model+a*cur            ;a为学习率，随着时间的推移，之前的建模图片权重越来越小
        cv2.accumulateWeighted(foreground, self.background, 0)
        # foreground2 = cv2.erode(foreground, None, iterations=1)
        diffImg = cv2.absdiff(foreground,
                              cv2.convertScaleAbs(self.background))  # convertScaleAbs函数功能是将CV_16S型的输出图像转变成CV_8U型的图像。
        grayImage = cv2.cvtColor(diffImg, cv2.COLOR_BGR2GRAY)
        ret, bwImage = cv2.threshold(grayImage, 7, 255, cv2.THRESH_BINARY)
        # bwImage = cv2.morphologyEx(bwImage, cv2.MORPH_OPEN, self._kernel)  #影像型態open處理
        # bwImage = cv2.morphologyEx(bwImage, cv2.MORPH_CLOSE, self._kernel) #影像型態close處理
        # cv2.imshow("dilate_erode before", bwImage)
        bwImage = cv2.dilate(bwImage, None, iterations=17)
        bwImage = cv2.erode(bwImage, None, iterations=15)
        # cv2.imshow("dilate_erode after", bwImage)
        motionArea = bwImage
        height, width, channels = currentImage.shape
        motionPercent = 100.0 * cv2.countNonZero(motionArea) / (width * height)
        if motionPercent > 3.:
            print("change background")
            self.background = numpy.float32(foreground)
        else:
            print("no change")
        movementLocations = self.contours(motionArea)
        # Motion start stop events
        if self.motionDetected:
            if motionPercent <= 0.:
                self.motionDetected = False
        # Threshold to trigger motionStart
        elif motionPercent > 3. and motionPercent < 25.:
            self.motionDetected = True
        return currentImage, grayImage, bwImage, motionPercent, movementLocations


# Path to test image
#path to video
VideoCaptrue = cv2.VideoCapture(r'C:\Users\cr\Desktop\dataset\VOC2007\test16.MOV')
#get the fps and size
fps = VideoCaptrue.get(cv2.CAP_PROP_FPS)
size = (int(VideoCaptrue.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(VideoCaptrue.get(cv2.CAP_PROP_FRAME_HEIGHT)))
numframes = VideoCaptrue.get(cv2.CAP_PROP_FRAME_COUNT)
print('num of frames:'+ str(numframes))
# VideoWriter = cv2.VideoWriter('cr.avi', 0, fps, size)# ("/dev/shm/test1.mp4", cv.CV_FOURCC('D', 'I', 'V', 'X'), fps, frame_size, is_color )
motion = MotionDetect()
while(VideoCaptrue.isOpened()):
        ret, frame = VideoCaptrue.read()
        if ret == True:
            _, _, bwImage, _, _ = motion.detect(frame)
            bwImage = cv2.cvtColor(bwImage, cv2.COLOR_GRAY2BGR)
            cv2.imshow('bwImage', bwImage)
            # VideoWriter.write(bwImage)
            k = cv2.waitKey(20)
            if (k & 0xff == ord('q')):
                break
        else:
            break
VideoCaptrue.release()
cv2.destroyAllWindows()
