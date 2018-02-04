import os
import sys
from utils import *
from PIL import Image
from scipy.misc import imresize
input_path = "dataset/Webscope_I4/ydata-tvsum50-v1_1/video"
FPS = 3
def generate_video_dataset():
	for video in os.listdir(input_path):
		if video[0]==".":
			continue
		print("working on video: " + str(video))
		video_dir = input_path + "/" + video
		if not os.path.isdir("dataset/video_frames/Webscope_I4/" + video.split('.')[0]):
			os.mkdir("dataset/video_frames/Webscope_I4/" + video.split('.')[0])
		convert_video_to_frames(video_dir, "dataset/video_frames/Webscope_I4/" + video.split('.')[0])
		print("Video done: " + str(video))

		for frame in os.listdir("dataset/video_frames/Webscope_I4/" + video.split('.')[0]):
			if frame[0] == '.':
				continue
			dirr = "dataset/video_frames/Webscope_I4/" + video.split('.')[0] + "/" + frame
			image = np.asarray(Image.open(dirr))
			image = imresize(image,(224,224,3))
			Image.fromarray(image).save("dataset/video_frames/Webscope_I4/" + video.split('.')[0] + "/" + frame)

		print("Resized all frames of video: " + str(video))

def get_frame_importance(file_dir):
  f = open(file_dir,"r")
  video_to_frame_importance = dict()
  for video_imp in f:
    tab_separated_values = video_imp.split('\t')
    scores = tab_separated_values[2].split(',')
    i=0
    final_scores=list()
    if not os.path.isdir("dataset/video_frames/Webscope_I4/"):
    	os.mkdir("dataset/video_frames/Webscope_I4/")
    frame_input_path = "dataset/video_frames/Webscope_I4/"
    for frame in os.listdir("dataset/video_frames/Webscope_all/" + tab_separated_values[0]):
		if frame[0] == '.':
			continue
		if index%10==0:
			dirr = "dataset/video_frames/Webscope_all/" + tab_separated_values[0] + "/" + frame
			image = np.asarray(Image.open(dirr))
			image = imresize(image,(224,224,3))
			Image.fromarray(image).save(frame_input_path + tab_separated_values[0] + "/" + frame)




    video_to_frame_importance[tab_separated_values[0]]=[int(score)-1 for score in scores[::10]]
    print "frame len: " + str(len(os.listdir(frame_input_path + tab_separated_values[0])))
    print "label len: " + str(len(video_to_frame_importance[tab_separated_values[0]]))
    print "video: " + str(tab_separated_values[0])
    print "\n\n#######################################################\n\n"
  print video_to_frame_importance
  return video_to_frame_importance

if __name__=='__main__':
	#get_frame_importance(("dataset/Webscope_I4/ydata-tvsum50-v1_1/data/ydata-tvsum50-anno.tsv"))
	generate_video_dataset()