from PIL import Image
#return images,labels
import numpy as np
from ..spectogram_ectractor import generate_spectogram_from_audio_file



def load_data(image_rows,image_columns,image_channels,data):
	images = [Image.open(element[0]).reshape(image_rows,image_columns,image_channels) for element in data]
	images = np.array(images)

	labels = np.array([element[1] for element in data])

	return images,labels

def generate_and_split_spectograms_for_complete_data(data_base_dir):

	#train
	generate_spectogram_and_save(output_dir='"../../dataset/audio/music/train_spectogram',input_dir='../../dataset/audio/music/train')    
	generate_spectogram_and_save(output_dir='"../../dataset/audio/speech/train_spectogram',input_dir='../../dataset/audio/speech/train')

	music_train = glob.glob("../../dataset/audio/music/train_spectogram/*.jpeg")
	speech_train = glob.glob("../../dataset/audio/speech/train_spectogram/*.jpeg")

    music_train_data = [(element,1) for element in music_train]
    speech_train_data = [(element,0) for element in speech_train]

    data_train = music_train_data + speech_train_data

    #validation
    generate_spectogram_and_save(output_dir='"../../dataset/audio/music/validation_spectogram',input_dir='../../dataset/audio/music/validation')    
	generate_spectogram_and_save(output_dir='"../../dataset/audio/speech/validation_spectogram',input_dir='../../dataset/audio/speech/validation')

	music_validation = glob.glob("../../dataset/audio/music/validation_spectogram/*.jpeg")
	speech_validation = glob.glob("../../dataset/audio/speech/validation_spectogram/*.jpeg")

    music_validation_data = [(element,1) for element in music_validation]
    speech_validation_data = [(element,0) for element in speech_validation]

    data_validation = music_validation_data + speech_validation_data

    return data_train,data_validation


def generate_spectogram_and_save(output_dir,input_dir):
	for element in os.list_dir(input_dir):
		generate_spectogram_from_audio_file(output_dir+'/'+element,input_dir+'/'+element)

