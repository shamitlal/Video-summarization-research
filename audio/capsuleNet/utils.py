from PIL import Image
#return images,labels

def load_data(image_rows,image_columns,image_channels,data):
	data = data.eval()
	images = [Image.open(element[0]).reshape(image_rows,image_columns,image_channels) for element in data]
	images = np.array(images)

	labels = np.array([element[1] for element in data])

	return images,labels