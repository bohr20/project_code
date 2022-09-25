# USAGE
# python align_faces.py --shape-predictor shape_predictor_68_face_landmarks.dat --image images/1.jpg

# import the necessary packages

# 人脸对齐：利用dlib的模型可以识别出图片中的人脸，为方便后续处理，
# 通常还需要把图片中的人脸截取出来并将倾斜的人脸处理成正常的姿态。
# imutils库中集成了一个人脸对齐的类 FaceAligner，我们直接使用它进行处理。

from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import argparse
import imutils
import dlib
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
#返回一个字典 键值是参数名称 值是输入的参数
#parse_args()解析添加的参数
args = vars(ap.parse_args())

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor and the face aligner
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])
fa = FaceAligner(predictor, desiredFaceWidth=256)

# load the input image, resize it, and convert it to grayscale
image = cv2.imread(args["image"])
image = imutils.resize(image, width=800)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# show the original input image and detect faces in the grayscale
# image
cv2.imshow("Input", image)
#返回人脸,(灰度图，采样次数)
rects = detector(gray, 2)

# loop over the face detections
for rect in rects:
	# extract the ROI of the *original* face, then align the face
	# using facial landmarks

# def rect_to_bb(rect):
	# # take a bounding predicted by dlib and convert it
	# # to the format (x, y, w, h) as we would normally do
	# # with OpenCV
	# x = rect.left()
	# y = rect.top()
	# w = rect.right() - x
	# h = rect.bottom() - y
	# return a tuple of (x, y, w, h)
	# return (x, y, w, h)
	(x, y, w, h) = rect_to_bb(rect)
	faceOrig = imutils.resize(image[y:y + h, x:x + w], width=256)
	faceAligned = fa.align(image, gray, rect)

	# import uuid
	# f = str(uuid.uuid4())
	# cv2.imwrite("foo/" + f + ".png", faceAligned)

	# display the output images
	cv2.imshow("Original", faceOrig)
	cv2.imshow("Aligned", faceAligned)
	cv2.waitKey(0)