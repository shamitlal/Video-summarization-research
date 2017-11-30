import numpy as np
import tensorflow as tf
import os
import vgg19
import utils
from glob import glob

batch_size = 8

def generateLayersForFramesInAllVideos():
    with tf.device('/cpu:0'):
        with tf.Session() as sess:
            images = tf.placeholder("float", [batch_size, 224, 224, 3])

            vgg = vgg19.Vgg19()
            with tf.name_scope("content_vgg"):
                vgg.build(images)

            for video_frame_dir in os.listdir("./dataset/video_frames"):
                if video_frame_dir.startswith('.'):
                    continue
                print "processing frames in video directory:" + video_frame_dir
                data = glob('./dataset/video_frames/' + video_frame_dir +'/*.jpg')
                print "frames loaded:" + str(data)
                batch_idxs = len(data)//batch_size
                for idx in xrange(0, batch_idxs):
                    batch_files = data[idx*batch_size:(idx+1)*batch_size]
                    batch = [utils.load_image(batch_file) for batch_file in batch_files]
                    batch_images = np.array(batch).astype(np.float32)
                    print "Shape of current batch " + str(idx) + " loaded:" + str(batch_images.shape)
                
                    feed_dict = {images: batch_images}
                    conv2_1, conv4_1 = sess.run([vgg.conv2_1, vgg.conv4_1], feed_dict=feed_dict)
                    print "Shapes of conv2_1 and conv4_1 are:",conv2_1.shape, conv4_1.shape
                    folderToGenerate = "./dataset/frame_numpy_arrays/"+video_frame_dir

                    if not os.path.isdir(folderToGenerate):
                        os.system('mkdir ' + folderToGenerate)

                    print "Save npy files for base image, conv2_1 and conv4_1"
                    for file_name_idx in range(0, len(batch_files)):
                        np.savez_compressed(folderToGenerate+"/" + batch_files[file_name_idx].split('/')[-1].split('.')[0] + 
                            "_conv2_1", a=conv2_1[file_name_idx])

                        np.savez_compressed(folderToGenerate+"/" + batch_files[file_name_idx].split('/')[-1].split('.')[0] + 
                            "_conv4_1", a=conv4_1[file_name_idx])

                        np.savez_compressed(folderToGenerate+"/" + batch_files[file_name_idx].split('/')[-1].split('.')[0] + 
                            "_base", a=batch_images[file_name_idx])

                        # np.save(folderToGenerate+"/" + batch_files[file_name_idx].split('/')[-1].split('.')[0] + 
                        #     "_conv2_1", conv2_1[file_name_idx])

                        # print np.load(folderToGenerate+"/" + batch_files[file_name_idx].split('/')[-1].split('.')[0] + 
                        #     "_conv2_1.npz")['a']

def generateLayersForFramesInGivenVideo(video_frame_dir):
    with tf.device('/cpu:0'):
        with tf.Session() as sess:
            images = tf.placeholder("float", [batch_size, 224, 224, 3])

            vgg = vgg19.Vgg19()
            with tf.name_scope("content_vgg"):
                vgg.build(images)

            
            print "processing frames in video directory:" + video_frame_dir
            data = glob('./dataset/video_frames/' + video_frame_dir +'/*.jpg')
            print "frames loaded:" + str(data)
            batch_idxs = len(data)//batch_size
            for idx in xrange(0, batch_idxs):
                batch_files = data[idx*batch_size:(idx+1)*batch_size]
                batch = [utils.load_image(batch_file) for batch_file in batch_files]
                batch_images = np.array(batch).astype(np.float32)
                print "Shape of current batch " + str(idx) + " loaded:" + str(batch_images.shape)
            
                feed_dict = {images: batch_images}
                conv2_1, conv4_1 = sess.run([vgg.conv2_1, vgg.conv4_1], feed_dict=feed_dict)
                print "Shapes of conv2_1 and conv4_1 are:",conv2_1.shape, conv4_1.shape
                folderToGenerate = "./dataset/frame_numpy_arrays/"+video_frame_dir

                if not os.path.isdir(folderToGenerate):
                    os.system('mkdir ' + folderToGenerate)

                print "Save npy files for base image, conv2_1 and conv4_1"
                for file_name_idx in range(0, len(batch_files)):
                    np.savez_compressed(folderToGenerate+"/" + batch_files[file_name_idx].split('/')[-1].split('.')[0] + 
                        "_conv2_1", a=conv2_1[file_name_idx])

                    np.savez_compressed(folderToGenerate+"/" + batch_files[file_name_idx].split('/')[-1].split('.')[0] + 
                        "_conv4_1", a=conv4_1[file_name_idx])

                    np.savez_compressed(folderToGenerate+"/" + batch_files[file_name_idx].split('/')[-1].split('.')[0] + 
                        "_base", a=batch_images[file_name_idx])

if __name__ == '__main__':
    generateLayersForFramesInGivenVideo("video1")

