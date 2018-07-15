import os
# from collections import defaultdict
# import numpy as np 
# import cv2
# from time import time
# import math
# import utils,vgg19_layer_generator
# from PIL import Image
import utils
import imageio

infinity = 10000000000

class KTS:
	def __init__(self,C):
		self.frame_list = defaultdict(lambda:0)
		self.frame_list['base'] = []
		self.frame_list['conv2_1'] = []
		self.frame_list['conv4_1'] = []

		self.frame_matrix = defaultdict(lambda:0)
		self.frame_matrix['base'] = []
		self.frame_matrix['conv2_1'] = []
		self.frame_matrix['conv4_1'] = []


		self.gram_matrix = defaultdict(lambda:0)
		self.gram_matrix['base'] = []
		self.gram_matrix['conv2_1'] = []
		self.gram_matrix['conv4_1'] = []

		self.vertical_segment_variance = defaultdict(lambda:0)

		self.segment_list = []
		self.C = C
		self.max_change_points = 0
		self.optimal_change_points = 0
		self.L = defaultdict(lambda:0)
		self.last_change_point = defaultdict(lambda:0)

		self.total_frames = 0

	def read_frames_to_list(self,input_directory,type):
		cnt = 0
		base_directory = input_directory
		input_directory = input_directory + str(type)
		for frame_dir in os.listdir(input_directory):
			if cnt>0:
				frame = np.load(input_directory + "/" + frame_dir)
				print frame_dir
				frame = np.array(frame['a'],dtype = np.float64).reshape(1,-1)
				#print(frame[0])
				if self.getAbsoluteValue(frame[0]) != 0 or type!='base':
					self.frame_list[type].append(input_directory + "/" + frame_dir)
				else:
					print "Black Frame Removed"
					os.system('rm -rf ' + base_directory + '/base/' + frame_dir)
					os.system('rm -rf ' + base_directory + '/conv2_1/' + frame_dir)
					os.system('rm -rf ' + base_directory + '/conv4_1/' + frame_dir)

			cnt = cnt + 1
		self.max_change_points = len(self.frame_list[type]) - 1
		self.total_frames = len(self.frame_list[type])

	def getAbsoluteValue(self, ax):
		return 1.0*math.sqrt(np.sum(np.square(ax)))

	def form_frame_matrix(self):
		for type in self.frame_list:
			for frame_dir in self.frame_list[type]:
				frame = np.load(frame_dir)
				frame = np.array(frame['a'],dtype = np.float64).reshape(1,-1)
				self.frame_matrix[type].append(frame[0]/self.getAbsoluteValue(frame[0]))
			self.frame_matrix[type] = np.array(self.frame_matrix[type], dtype = np.float64)

	def compute_gram_matrix(self):
		for type in self.frame_matrix:
			self.gram_matrix[type] = np.dot(self.frame_matrix[type],np.transpose(self.frame_matrix[type]))
			print "Frame Matrix shape : " + str(self.frame_matrix[type].shape)
			print "Gram Matrix shape : " + str(self.gram_matrix[type].shape)

	def perform_kts(self):
		self.L[0] = defaultdict(lambda:0)
		self.last_change_point[0] = defaultdict(lambda:0)
		for i in range(self.total_frames):
			self.L[0][i] = self.segment_variance(0,i)
			self.last_change_point[0][i] = 0
		
		kts_start = time()
		for i in range(1,self.max_change_points+1):
			self.L[i] = defaultdict(lambda:0)
			self.last_change_point[i] = defaultdict(lambda:0)
			i_start = time()
			for j in range(self.total_frames):
				self.L[i][j] = infinity
				#print "==========================================="
				#print "Looping over k for i,j :=  " + str(i),str(j)
				for k in range(i,j+1):
					variance_in_segment = self.segment_variance(k,j)
					#print self.L[i-1][k-1],variance_in_segment
					if self.L[i][j] > self.L[i-1][k-1] + variance_in_segment:
						self.L[i][j] = self.L[i-1][k-1] + variance_in_segment
						self.last_change_point[i][j] = k
				#print "==========================================="
				#print str(i),str(j),self.L[i][j]
			i_end = time()
		kts_end = time()
		print "Total time for KTS : " + str(kts_end-kts_start)
				

	def precompute_segment_variance(self):
		seg_start = time()
		variance = 0
		for index,type in enumerate(self.frame_list):
			for frame1 in range(0,self.total_frames):
				variance_with_frame1 = 0
				if index==0:
					self.vertical_segment_variance[frame1] = defaultdict(lambda:0)
				for frame2 in range(0,self.total_frames):
					variance_with_frame1 = variance_with_frame1 + self.gram_matrix[type][frame1][frame2]
					#Let variance_with_frame1 means varainace of frame frame1 with all frames upto frame frame2
					#vertical_segment_variance[j] means summation(horizontal_segment_variance[0][j] + horizontal_segment_variance[1][j] + .... + horizontal_segment_variance[j][j])
					if frame1==0 or index<2:
						print "Gram : " + str(frame1) + "," + str(frame2) + "  " + str(self.gram_matrix[type][frame1][frame2]) + "  " + str(variance_with_frame1)
						self.vertical_segment_variance[frame1][frame2] = self.vertical_segment_variance[frame1][frame2] + variance_with_frame1
					else:	
						self.vertical_segment_variance[frame1][frame2] = self.vertical_segment_variance[frame1][frame2] + self.vertical_segment_variance[frame1-1][frame2] + variance_with_frame1

		seg_end = time()
		print "precomputed segment variance in time : " + str(seg_end-seg_start)
		return variance


	def segment_variance(self,segment_start,segment_end):
		if segment_start == 0:
			variance = (self.vertical_segment_variance[segment_end][segment_end])/(segment_end+1)
			#print "Variance is : " +  str(variance) + " " + str(3*(segment_end+1))
			variance = 3*(segment_end+1) - variance
			if variance <= 1e-20:
				variance = 0.0	
			return variance
		#print self.vertical_segment_variance[segment_end][segment_end] 
		#print self.vertical_segment_variance[segment_start-1][segment_end]
		#print self.vertical_segment_variance[segment_end][segment_start-1] 
		#print self.vertical_segment_variance[segment_start-1][segment_start-1]
		variance = ((self.vertical_segment_variance[segment_end][segment_end] - self.vertical_segment_variance[segment_start-1][segment_end]) - (self.vertical_segment_variance[segment_end][segment_start-1] - self.vertical_segment_variance[segment_start-1][segment_start-1]))/(segment_end-segment_start+1)
		#print "Variance is : " +  str(variance) + " " + str(3*(segment_end-segment_start+1))
		variance = 3*(segment_end-segment_start+1) - variance
		if variance <= 1e-20:
			variance = 0.0
		return variance  

	def print_segment_variance(self):
		print "======================="
		print "SEGMENT VARIANCE MATRIX"
		print "======================="

		for i in range(self.total_frames):
			for j in range(self.total_frames):
				print self.segment_variance(min(i,j),max(i,j)),
			print '\n'

		print "======================="
		print "SAME VARIANCE MATRIX"
		print "======================="
		for i in range(0,self.total_frames):
			print "Segment_variance " + str(self.segment_variance(i,i))
		
	'''
	def segment_variance(self,type,segment_start,segment_end):
		seg_start = time()
		variance = 0
		for frame1 in range(segment_start,segment_end+1):
			variance_with_frame1 = 0
			for frame2 in range(segment_start,segment_end+1):
				variance_with_frame1 = variance_with_frame1 + self.gram_matrix[type][frame1][frame2]
			variance_with_frame1 = variance_with_frame1/(segment_end - segment_start+1)
			variance = variance + (self.gram_matrix[type][frame1][frame1] - variance_with_frame1)
		seg_end = time()
		print "segment variance time : " + str(seg_end-seg_start)
		return variance
	'''

	def g(self,m,n):
		return m*(math.log(n/m) + 1)

	def compute_optimal_change_points(self):
		min_variance = infinity
		print "=============================="
		print "compute_optimal_change_points"
		print "=============================="
		for i in range(0,self.max_change_points+1):
			print i,self.L[i][self.total_frames-1],self.C*self.g(i+1,self.total_frames)
			if min_variance > self.L[i][self.total_frames-1] + self.C*self.g(i+1,self.total_frames):
				min_variance = self.L[i][self.total_frames-1] + self.C*self.g(i+1,self.total_frames)
				self.optimal_change_points = i


	def find_segments(self):
		current_segment_end  = self.total_frames - 1
		current_change_point_index = self.optimal_change_points 
		while(current_change_point_index>=0):
			self.segment_list.append((self.last_change_point[current_change_point_index][current_segment_end],current_segment_end))
			current_segment_end = self.last_change_point[current_change_point_index][current_segment_end] - 1
			current_change_point_index = current_change_point_index - 1
		print "Segment List : ",
		print(self.segment_list)

	def save_segments(self,video):
		if not os.path.isdir('dataset/video_segments/' + video):
			os.system('mkdir dataset/video_segments/' + video)
		cnt = 0
		for i,frame in  enumerate(self.frame_list['base']):
			print frame,
		for segment in self.segment_list:
			seg_start = segment[0]
			seg_end = segment[1]
			print seg_start,seg_end
			if os.path.isdir('dataset/video_segments/' + video + '/segment_' + str(cnt)):
				os.system('rm -R dataset/video_segments/' + video + '/segment_' + str(cnt))
			os.system('mkdir dataset/video_segments/' + video + '/segment_' + str(cnt))
			for frame_number in range(seg_start,seg_end+1):
				image_dir = 'dataset/video_frames/' + video + '/' + self.frame_list['base'][frame_number].split('/')[-1]
				image_dir = image_dir.split('.')[0] + '.jpg'
				image_destination = 'dataset/video_segments/' + video + '/segment_' + str(cnt) + '/frame' + str(frame_number).zfill(6) + '.png'
				print image_dir + " to " + str(image_destination)
				image = cv2.imread(image_dir)
				cv2.imwrite(image_destination,image)
			cnt = cnt + 1

#video3 url : https://www.youtube.com/watch?v=5DkJDWxhu-g
if __name__=="__main__":
	#frame_importance = utils.get_frame_importance_vector("dataset/true_summaries/video4",367)
	#print frame_importance
	#os.system("mkdir -p dataset/tvsum_test_dataset/video_frames")
	for video in os.listdir("dataset/tvsum_test_dataset/videos"):
		#if('1' in video or '2' in video):continue
		#if(video[0]=='.'):continue
		if(".mp4" in video):
			utils.convert_video_to_frames("dataset/tvsum_test_dataset/videos/" + video,"dataset/tvsum_test_dataset/video_frames/" + video.split('.')[0])
		#for video_frame in os.listdir("vsumm_dataset/video_frames/" + video):
		#	utils.test("vsumm_dataset/video_frames/" + video + '/' + video_frame)
		


		'''
		#utils.extract_audio_from_video("dataset/videos/",video)
		vgg19_layer_generator.generateLayersForFramesInGivenVideo("dataset/video_frames/" + video.split('.')[0],"dataset/frame_numpy_arrays")
		
		checkpoint = 0
		kts = KTS(3)
		
		print "checkpoint" + str(checkpoint)
		checkpoint = checkpoint + 1
		kts.read_frames_to_list("dataset/frame_numpy_arrays/","base")
		kts.read_frames_to_list("dataset/frame_numpy_arrays/","conv2_1")
		kts.read_frames_to_list("dataset/frame_numpy_arrays/","conv4_1")
		print "checkpoint" + str(checkpoint)
		
		checkpoint = checkpoint + 1
		kts.form_frame_matrix()
		print "checkpoint" + str(checkpoint)
		checkpoint = checkpoint + 1
		kts.compute_gram_matrix()
		
		print "checkpoint" + str(checkpoint)
		checkpoint = checkpoint + 1
		kts.precompute_segment_variance()

		print kts.print_segment_variance()
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
		
		#video2 kts segment list
		#kts.segment_list = [(523, 603), (467, 522), (417, 466), (377, 416), (337, 376), (302, 336), (246, 301), (182, 245), (122, 181), (63, 121), (0, 62)]
		kts.segment_list.reverse()
		print kts.segment_list
		kts.save_segments(video.split('.')[0])	
		
		
		utils.convert_frames_to_video('dataset/video_segments/' + video.split('.')[0] + '/segment_0')	
		'''