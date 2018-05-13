#MNIST
from tensorflow.examples.tutorials.mnist import input_data
dataset = input_data.read_data_sets('/home/ubuntu/data_set', one_hot=True)

def getImageData():
	imageData = dataset.train.images
	return imageData

def getLabelData():
	labelData = dataset.train.labels
	return labelData

#CIFAR-10
from additionalinfo import get_data_set
train_x, train_y = get_data_set('train')
test_x, test_y = get_data_set('test')

def getImageData():
	imageData = train_x
	return imageData

def getLabelData():
	labelData = train_y
	return labelData