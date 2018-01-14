from PIL import Image
#return images,labels
import numpy as np
import os
import sys
import imp
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import glob
from scipy.misc import imread, imsave, imresize

def load_data(image_rows,image_columns,image_channels,data):
    images = [imresize(np.asarray(Image.open(element[0])), (image_rows,image_columns,image_channels)) for element in data]
    #images = [np.asarray(Image.open(element[0])) for element in data]
    images = np.array(images)
    print "images size: " + str(images.shape)
    labels = np.array([element[1] for element in data])

    return images,labels
    
def generate_and_split_spectograms_for_complete_data(data_base_dir):

    print os.listdir("../../dataset")
    
    #train
    generate_spectogram_and_save(output_dir='../../dataset/audio/music/train_spectogram',input_dir='../../dataset/audio/music/train')    
    generate_spectogram_and_save(output_dir='../../dataset/audio/speech/train_spectogram',input_dir='../../dataset/audio/speech/train')

    music_train = glob.glob("../../dataset/audio/music/train_spectogram/*.jpeg")
    speech_train = glob.glob("../../dataset/audio/speech/train_spectogram/*.jpeg")

    music_train_data = [(element,1) for element in music_train]
    speech_train_data = [(element,0) for element in speech_train]

    data_train = music_train_data + speech_train_data
    print "data_train length is: " + str(len(data_train))

    #validation
    generate_spectogram_and_save(output_dir='../dataset/audio/music/validation_spectogram',input_dir='../../dataset/audio/music/validation')    
    generate_spectogram_and_save(output_dir='../dataset/audio/speech/validation_spectogram',input_dir='../../dataset/audio/speech/validation')

    music_validation = glob.glob("../../dataset/audio/music/validation_spectogram/*.jpeg")
    speech_validation = glob.glob("../../dataset/audio/speech/validation_spectogram/*.jpeg")

    music_validation_data = [(element,1) for element in music_validation]
    speech_validation_data = [(element,0) for element in speech_validation]

    data_validation = music_validation_data + speech_validation_data
    print "data_validation length is: " + str(len(data_validation))
    return data_train,data_validation


def generate_spectogram_and_save(output_dir,input_dir):
    for element in os.listdir(input_dir):
        if element[0]=='.':
            continue
        print "element: " + element
        generate_spectogram_from_audio_file(output_dir + "/" + element.split(".")[0], input_dir+'/'+element)



def generate_spectogram_from_audio_file(output_spectogram_dir,input_audio_dir):
    yy, sr = librosa.load(input_audio_dir)
    print "length of signal: " + str(len(yy))
    #for viewing wavepliot
    #plt.figure(figsize=(12, 4))
    #librosa.display.waveplot(y, sr=sr)
    #plt.show()
    counter=0
    index=0
    while index+int(1.5*sr)<=len(yy):
        y = yy[index: index+int(1.5*sr)]
        print "length of splitted signal: " + str(len(y))
        index = index + int(1.5*sr)
        X = librosa.stft(y)
        powerX = X**2
        log_power_X = np.log(powerX + 0.000001)
        normalized_log_power_X = (log_power_X - np.min(log_power_X)) / (np.max(log_power_X) - np.min(log_power_X))

        shape = normalized_log_power_X.shape

        binary_X = [element>0.8 for element in normalized_log_power_X]
        binary_X = np.array(binary_X).reshape(shape)

        #spectogram_image = Image.fromarray(normalized_log_power_X.astype("uint8"))
        #spectogram_image.save(output_spectogram_dir, "JPEG")

        #for viewing binary spectogram
        #librosa.display.specshow(np.array(normalized_log_power_X), x_axis='time',y_axis='log')
        librosa.display.specshow(np.array(binary_X), y_axis='log')
        #plt.colorbar()
        #plt.show()
        plt.savefig(output_spectogram_dir + "_" + str(counter) + ".jpeg")
        counter += 1

def generate_melspectrogram_from_audio_file(output_spectogram_dir, input_audio_dir):
    y, sr = librosa.load(input_audio_dir, sr=16000)
    X = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=400, hop_length=160)
    X = (X - np.min(X)) / (np.max(X) - np.min(X) + 0.0000000001)

    librosa.display.specshow(X, sr=sr, hop_length=160, y_axis='log')
    plt.savefig(output_spectogram_dir)

