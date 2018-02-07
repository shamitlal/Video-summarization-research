import os
import sys
from utils import *
from PIL import Image
from scipy.misc import imresize
from collections import defaultdict
import glob

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
  video_map = defaultdict(lambda:0)
  for video_imp in f:
    tab_separated_values = video_imp.split('\t')
    scores = tab_separated_values[2].split(',')
    i=0
    final_scores=list()
    if not os.path.isdir("dataset/video_frames/Webscope_I4/"):
        os.mkdir("dataset/video_frames/Webscope_I4/")
    frame_input_path = "dataset/video_frames/Webscope_I4/"
    
    if video_map[tab_separated_values[0]]==0:
        for index,frame in enumerate(os.listdir("dataset/video_frames/Webscope_all/" + tab_separated_values[0])):
            if frame[0] == '.':
                continue
            if index%10==0:
                dirr = "dataset/video_frames/Webscope_all/" + tab_separated_values[0] + "/" + frame
                image = np.asarray(Image.open(dirr))
                image = imresize(image,(224,224,3))
                if not os.path.isdir(frame_input_path + tab_separated_values[0]):
                    os.mkdir(frame_input_path + tab_separated_values[0])
                Image.fromarray(image).save(frame_input_path + tab_separated_values[0] + "/" + frame)
        video_map[tab_separated_values[0]] = 1

        #frame importance test
        video_to_frame_importance[tab_separated_values[0]]=[int(score)-1 for score in scores[::10]]
        print "frame len: " + str(len(os.listdir(frame_input_path + tab_separated_values[0])))
        print "label len: " + str(len(video_to_frame_importance[tab_separated_values[0]]))
        print "video: " + str(tab_separated_values[0])
        print "\n\n#######################################################\n\n"
  print video_to_frame_importance
  return video_to_frame_importance

def print_label_count(file_dir):
    fobj = open(file_dir, "r")
    score_map = dict()
    for line in fobj:
        tab_separated_values = line.split('\t')
        scores = tab_separated_values[2].split(',')
        scores = [int(score) for score in scores]
        #print scores
        for score in scores:
            if score in score_map:
                score_map[score] += 1
            else:
                score_map[score] = 1
    for i,j in score_map.iteritems():
        print "i=",str(i) + " " + "j=" + str(j)

def generate_mean_image(file_dir):
    img_cnt = 0
    images = glob.glob(file_dir + "*/*.jpg")
    mean_image = np.zeros((224,224,3), dtype=np.float32)
    for image in images:
        image_open = np.asarray(Image.open(image))
        mean_image += image_open
        print img_cnt
        img_cnt += 1
    mean_image /= len(images)
    print mean_image
    np.save("ConvolutionLSTM/mean_image.npy", mean_image)


if __name__=='__main__':
    #get_frame_importance(("dataset/Webscope_I4/ydata-tvsum50-v1_1/data/ydata-tvsum50-anno.tsv"))
    #get_frame_importance()
    #print_label_count("dataset/Webscope_I4/ydata-tvsum50-v1_1/data/ydata-tvsum50-anno.tsv")
    generate_mean_image("./dataset/video_frames/Webscope_I4/")