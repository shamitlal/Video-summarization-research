import skimage
import skimage.io
import skimage.transform
import numpy as np
import os,cv2
from time import time
from PIL import Image


# synset = [l.strip() for l in open('synset.txt').readlines()]


# returns image of shape [224, 224, 3]
# [height, width, depth]
def load_image(path):
    # load image
    img = skimage.io.imread(path)
    img = img / 255.0
    assert (0 <= img).all() and (img <= 1.0).all()
    # print "Original Image Shape: ", img.shape
    # we crop image from center

    #Uncomment below code to crop image from center
    '''
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    # resize to 224, 224
    resized_img = skimage.transform.resize(crop_img, (224, 224))
    '''
    resized_img = skimage.transform.resize(img, (224, 224))
    return resized_img


# returns the top1 string
def print_prob(prob, file_path):
    synset = [l.strip() for l in open(file_path).readlines()]

    # print prob
    pred = np.argsort(prob)[::-1]

    # Get top1 label
    top1 = synset[pred[0]]
    print(("Top1: ", top1, prob[pred[0]]))
    # Get top5 label
    top5 = [(synset[pred[i]], prob[pred[i]]) for i in range(5)]
    print(("Top5: ", top5))
    return top1


def load_image2(path, height=None, width=None):
    # load image
    img = skimage.io.imread(path)
    img = img / 255.0
    if height is not None and width is not None:
        ny = height
        nx = width
    elif height is not None:
        ny = height
        nx = img.shape[1] * ny / img.shape[0]
    elif width is not None:
        nx = width
        ny = img.shape[0] * nx / img.shape[1]
    else:
        ny = img.shape[0]
        nx = img.shape[1]
    return skimage.transform.resize(img, (ny, nx))


def convert_video_to_frames(input_video,output_directory):
    if not os.path.isdir(output_directory):
        os.mkdir(output_directory)
    os.system("ffmpeg -i {0} -vf fps=3 {1}/thumb%04d.jpg -hide_banner".format(input_video,output_directory))
    '''
    if not os.path.isdir(output_directory):
        os.system('mkdir ' + output_directory)
    cap = cv2.VideoCapture(input_video)
    print(cap)

    start = time()
    prev = time()
    cnt = 0
    while(True):
        curr = time()
        value,frame = cap.read()
        if value==False:
            break
        print "Working on frame number :  " + str(cnt) + "   " +  str(abs(curr-prev))
        if abs(curr-prev)>=0.05:
            print "Writing frame  : " + str(cnt)
            cv2.imwrite(output_directory + '/frame%04d.jpg' % cnt,frame)
            cnt = cnt + 1
            prev = curr

    end = time()
    print("FPS: {}".format(120/(end-start)))
    cap.release()

def convert_frames_to_video(input_directory):
    os.system('')
    image = Image.open(input_directory + '/' + os.listdir(input_directory)[2])
    print image.size
    video = cv2.VideoWriter(input_directory + '/video.avi',-1,1,(image.size),True)
    
    for frame in os.listdir(input_directory):
        print frame
        if 'video' in frame:
            continue
        if frame[0] =='.':
            continue
        image = Image.open(input_directory + '/' + frame)
        video.write(np.asarray(image))

    cv2.destroyAllWindows()
    video.release()
    '''

def extract_audio_from_video(video_base_dir,video_name):
    audio_base_dir = "dataset/audio/"
    audio_dir = audio_base_dir + video_name.split(".")[0] + ".wav"
    os.system("ffmpeg -i {0} -f wav -ab 192000 -vn {1}".format(video_base_dir+video_name,audio_dir))


def get_frame_importance_vector(input_directory,total_frames):
    frame_importance = np.zeros(total_frames)
    for frame in os.listdir(input_directory):
        if frame[0]=='.':
            continue
        frame_number = frame.split("thumb")[1].split('.')[0]
        frame_number = int(frame_number)
        frame_importance[frame_number] = 1

    return frame_importance



def test():
    img = skimage.io.imread("./test_data/starry_night.jpg")
    ny = 300
    nx = img.shape[1] * ny / img.shape[0]
    img = skimage.transform.resize(img, (ny, nx))
    skimage.io.imsave("./test_data/test/output.jpg", img)


if __name__ == "__main__":
    test()
