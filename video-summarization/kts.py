import os
from collections import defaultdict
import numpy as np 
import cv2
from time import time
import math


def convert_video_to_frames(input_video,output_directory):
	#os.system("ffmpeg -i {0} -vf fps=1 output_directory/thumb%04d.jpg -hide_banner".format(input_video))

	cap = cv2.VideoCapture(input_video)
	print(cap)

	start = time()
	cnt = 0
	while(True):
		value,frame = cap.read()
		if value==False:
			break
		cv2.imwrite(output_directory + '/frame%04d.jpg' % cnt,frame)
		cnt = cnt + 1

	end = time()
	print("FPS: {}".format(120/(end-start)))
	cap.release()


infinity = 10000000000

class KTS:
	def __init__(self,C):
		self.frame_list = []
		self.frame_matrix = []
		self.gram_matrix = []
		self.segment_list = []
		self.C = C
		self.max_change_points = 0
		self.optimal_change_points = 0
		self.L = defaultdict(lambda:0)
		self.last_change_point = defaultdict(lambda:0)

	def read_frames_to_list(self,input_directory):
		cnt = 0
		for frame in os.listdir(input_directory):
			if cnt>0:
				self.frame_list.append(input_directory + "/" + frame)
			cnt = cnt + 1
		print self.frame_list
		self.max_change_points = len(self.frame_list) - 1

	def getAbsoluteValue(self, ax):
		return 1.0*math.sqrt(np.sum(np.square(ax)))

	def form_frame_matrix(self):
		for frame in self.frame_list:
			frame = cv2.imread(frame)
			#print(frame.shape)
			frame = np.array(frame,dtype = np.float64).reshape(1,-1)
			#print(frame[0])
			self.frame_matrix.append(frame[0]/self.getAbsoluteValue(frame[0]))
		self.frame_matrix = np.array(self.frame_matrix, dtype = np.float64)
		#print(self.frame_matrix)

	def compute_gram_matrix(self):
		self.gram_matrix = np.dot(self.frame_matrix,np.transpose(self.frame_matrix))
		print "Frame Matrix shape : " + str(self.frame_matrix.shape)
		print "Gram Matrix shape : " + str(self.gram_matrix.shape)

	def perform_kts(self):
		self.L[0] = defaultdict(lambda:0)
		self.last_change_point[0] = defaultdict(lambda:0)
		for i in range(len(self.frame_list)):
			self.L[0][i] = self.segment_variance(0,i)
			self.last_change_point[0][i] = 0
		
		for i in range(1,self.max_change_points+1):
			self.L[i] = defaultdict(lambda:0)
			self.last_change_point[i] = defaultdict(lambda:0)
			for j in range(len(self.frame_list)):
				self.L[i][j] = infinity
				for k in range(i,j+1):
					variance_in_segment = self.segment_variance(k,j)
					if self.L[i][j] > self.L[i-1][k-1] + variance_in_segment:
						self.L[i][j] = self.L[i-1][k-1] + variance_in_segment
						self.last_change_point[i][j] = k

	def segment_variance(self,segment_start,segment_end):
		variance = 0
		for frame1 in range(segment_start,segment_end+1):
			variance_with_frame1 = 0
			for frame2 in range(segment_start,segment_end+1):
				variance_with_frame1 = variance_with_frame1 + self.gram_matrix[frame1][frame2]
			variance_with_frame1 = variance_with_frame1/(segment_end - segment_start+1)
			variance = variance + (self.gram_matrix[frame1][frame1] - variance_with_frame1)
		return variance


	def g(self,m,n):
		return m*(math.log(n/m) + 1)

	def compute_optimal_change_points(self):
		min_variance = infinity
		for i in range(0,self.max_change_points+1):
			if min_variance > self.L[i][len(self.frame_list)-1] + self.C*self.g(i+1,len(self.frame_list)):
				min_variance = self.L[i][len(self.frame_list)-1] + self.C*self.g(i+1,len(self.frame_list))
				self.optimal_change_points = i


	def find_segments(self):
		current_segment_end  = len(self.frame_list) - 1
		current_change_point_index = self.optimal_change_points 
		while(current_change_point_index>=0):
			self.segment_list.append((self.last_change_point[current_change_point_index][current_segment_end],current_segment_end))
			current_segment_end = self.last_change_point[current_change_point_index][current_segment_end] - 1
			current_change_point_index = current_change_point_index - 1
		print "Segment List : ",
		print(self.segment_list)


if __name__=="__main__":
	#convert_video_to_frames("dataset/videos/video.mp4","dataset/video_frames")
	checkpoint = 0
	kts = KTS(3)
	print "checkpoint" + str(checkpoint)
	checkpoint = checkpoint + 1
	kts.read_frames_to_list("video_frames")
	print "checkpoint" + str(checkpoint)
	checkpoint = checkpoint + 1
	kts.form_frame_matrix()
	print "checkpoint" + str(checkpoint)
	checkpoint = checkpoint + 1
	kts.compute_gram_matrix()
	print "checkpoint" + str(checkpoint)
	checkpoint = checkpoint + 1
	kts.perform_kts()
	print "checkpoint" + str(checkpoint)
	checkpoint = checkpoint + 1
	kts.compute_optimal_change_points()
	print "checkpoint" + str(checkpoint)
	checkpoint = checkpoint + 1
	kts.find_segments()
	print "checkpoint" + str(checkpoint)
	checkpoint = checkpoint + 1
	print(kts.gram_matrix[0])



