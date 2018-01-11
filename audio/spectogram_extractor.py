import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


y, sr = librosa.load('../dataset/audio/test_speech.mp3')

plt.figure(figsize=(12, 4))
librosa.display.waveplot(y, sr=sr)
plt.show()

X = librosa.stft(y)
powerX = X**2
log_power_X = np.log(powerX + 0.000001)
normalized_log_power_X = (log_power_X - np.min(log_power_X)) / (np.max(log_power_X) - np.min(log_power_X))

shape = normalized_log_power_X.shape

binary_X = [element>0.8 for element in normalized_log_power_X]
binary_X = np.array(binary_X).reshape(shape)

librosa.display.specshow(binary_X, x_axis='time',y_axis='log')
plt.colorbar()
plt.show()
