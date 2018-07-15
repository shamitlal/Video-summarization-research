import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def generate_spectogram_from_audio_file(input_audio_dir):
	y, sr = librosa.load(input_audio_dir)

	#for viewing wavepliot
	#plt.figure(figsize=(12, 4))
	#librosa.display.waveplot(y, sr=sr)
	#plt.show()
	print sr

	X = librosa.stft(y)
	powerX = X**2
	log_power_X = np.log(powerX + 0.000001)

	shape = normalized_log_power_X.shape

	binary_X = [element>0.8 for element in normalized_log_power_X]
	binary_X = np.array(binary_X).reshape(shape)


	#spectogram_image = Image.fromarray(binary_X)
	#spectogram_image.save(output_spectogram_dir)

	#for viewing binary spectogram
	librosa.display.specshow(binary_X, x_axis='time',y_axis='log')
	plt.colorbar()
	plt.show()


generate_spectogram_from_audio_file('../dataset/audio/speech/train/test_speech.mp3')