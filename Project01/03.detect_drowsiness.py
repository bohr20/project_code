

# USAGE
# python detect_drowsiness.py --shape-predictor shape_predictor_68_face_landmarks.dat
# python detect_drowsiness.py --shape-predictor shape_predictor_68_face_landmarks.dat --alarm alarm.wav

# import the necessary packages
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import playsound
import argparse
import imutils
import time
import dlib
import cv2

def sound_alarm(path):
	# play an alarm sound
	playsound.playsound(path)


# 这个函数接受单一的参数，即给定的眼睛面部标志的（x，y）坐标  。
# A，B是计算两组垂直眼睛标志之间的距离，而C是计算水平眼睛标志之间的距离。
# 最后，将分子和分母相结合，得出最终的眼睛纵横比。然后将眼图长宽比返回给调用函数。
# 让我们继续解析我们的命令行参数：

def eye_aspect_ratio(eye):
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])

	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	C = dist.euclidean(eye[0], eye[3])

	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)

	# return the eye aspect ratio
	return ear
# detect_blinks.py脚本需要一个命令行参数，然后第二个是可选的参数：

# 1.--shape-predictor：这是dlib的预训练面部标志检测器的路径。

# 2.--video：它控制驻留在磁盘上的输入视频文件的路径。如果您想要使用实时视频流，则需在执行脚本时省略此开关。
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-a", "--alarm", type=str, default="",
	help="path alarm .WAV file")
ap.add_argument("-w", "--webcam", type=int, default=0,
	help="index of webcam on system")
args = vars(ap.parse_args())
 
# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold for to set off the
# alarm
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 5

# initialize the frame counter as well as a boolean used to
# indicate if the alarm is going off
COUNTER = 0
ALARM_ON = False

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor

# 当确定视频流中是否发生眨眼时，我们需要计算眼睛的长宽比。

# 如果眼睛长宽比低于一定的阈值，然后超过阈值，那么我们将记录一个“眨眼” -EYE_AR_THRESH是这个阈值，我们默认它的值为 0.3，您也可以为自己的应用程序调整它。另外，我们有一个重要的常量，EYE_AR_CONSEC_FRAME，这个值被设置为 3，表明眼睛长宽比小于3时，接着三个连续的帧一定发生眨眼动作。

# 同样，取决于视频的帧处理吞吐率，您可能需要提高或降低此数字以供您自己实施。

# 接着初始化两个计数器，COUNTER是眼图长宽比小于EYE_AR_THRESH的连续帧的总数，而 TOTAL则是脚本运行时发生的眨眼的总次数。

# 现在我们的输入，命令行参数和常量都已经写好了，接着可以初始化dlib的人脸检测器和面部标志检测器：


print("[INFO] loading facial landmark predictor...")
# 定义人脸检测器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
#为下面的左眼和右眼提取（x，y）坐标的起始和结束数组切片索引值
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# start the video stream thread
#决定是否使用基于文件的视频流或实时USB/网络摄像头/ Raspberry Pi摄像头视频流：
print("[INFO] starting video stream thread...")
vs = VideoStream(src=args["webcam"]).start()
time.sleep(1.0)

# Raspberry Pi相机模块，取消注释：# vs = VideoStream(usePiCamera=True).start()。

# 如果您未注释上述两个，你可以取消注释# fileStream = False以及以表明你是不是从磁盘读取视频文件。

# loop over frames from the video stream



# 在while处我们开始从视频流循环帧。

# 如果正在访问视频文件流，并且视频中没有剩余的帧，则从循环中断。

# 从我们的视频流中读取下一帧，然后调整大小并将其转换为灰度。然后，我们通过dlib内置的人脸检测器检测灰度帧中的人脸。

# 我们现在需要遍历帧中的每个面，然后对其中的每个面应用面部标志检测：
while True:
	# grab the frame from the threaded video file stream, resize
	# it, and convert it to grayscale
	# channels)
	frame = vs.read()
	frame = imutils.resize(frame, width=450)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# detect faces in the grayscale frame
	rects = detector(gray, 0)
	print('rects:', rects)

	# loop over the face detections
	for rect in rects:
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		# extract the left and right eye coordinates, then use the
		# coordinates to compute the eye aspect ratio for both eyes
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)

		# average the eye aspect ratio together for both eyes
		ear = (leftEAR + rightEAR) / 2.0
# shape确定面部区域的面部标志，接着将这些（x，y）坐标转换成NumPy阵列。
# 使用我们之前在这个脚本中的数组切片技术，我们可以分别为左眼left eye和右眼提取（x，y）坐标，然后我们计算每只眼睛的眼睛长宽比  。
# 下一个代码块简单地处理可视化眼部区域的面部标志：

		# compute the convex hull for the left and right eye, then
		# visualize each of the eyes
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

		# check to see if the eye aspect ratio is below the blink
		# threshold, and if so, increment the blink frame counter
# 我们已经计算了我们的（平均的）眼睛长宽比，但是我们并没有真正确定是否发生了眨眼，这在下一部分中将得到关注：
# 第一步检查眼睛纵横比是否低于我们的眨眼阈值，如果是，
# 我们递增指示正在发生眨眼的连续帧数。否则，我们将处理眼高宽比不低于眨眼阈值的情况，
# 我们对其进行检查，看看是否有足够数量的连续帧包含低于我们预先定义的阈值的眨眼率。
# 如果检查通过，我们增加总的闪烁次数。然后我们重新设置连续闪烁次数 COUNTER。


		if ear < EYE_AR_THRESH:
			COUNTER += 1
			# if the eyes were closed for a sufficient number of
			# then sound the alarm
			if COUNTER >= EYE_AR_CONSEC_FRAMES:
				# if the alarm is not on, turn it on
				if not ALARM_ON:
					ALARM_ON = True

					# check to see if an alarm file was supplied,
					# and if so, start a thread to have the alarm
					# sound played in the background
					if args["alarm"] != "":
						t = Thread(target=sound_alarm,
							args=(args["alarm"],))
						t.deamon = True
						t.start()
				# draw an alarm on the frame
				cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

		# otherwise, the eye aspect ratio is not below the blink
		# threshold, so reset the counter and alarm
		else:
			COUNTER = 0
			ALARM_ON = False

		# draw the computed eye aspect ratio on the frame to help
		# with debugging and setting the correct eye aspect ratio
		# thresholds and frame counters
		cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
 
	# show the frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()